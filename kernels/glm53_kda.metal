// Kimi delta attention (KDA) single-token decode for GLM-5.3-Flash.
//
// Adapted from DwarfStar's kernel_glm53_kda_decode (metal/glm53_kda.metal at
// 0aaea5a238fb41a35106a551e73c8409dfb751ac), MIT; see
// docs/THIRD-PARTY-NOTICES.md. Changes: explicit
// buffer indices, and the decay consumes the GGUF's `ssm_a = -exp(A_log)`
// instead of `A_log`.
//
// Per head (d = 128), for one token:
//   q, k, v  = silu(causal depthwise conv, taps 0..2 history, tap 3 current)
//   q       *= rsqrt(sum q^2 + 1e-6) / sqrt(d);  k *= rsqrt(sum k^2 + 1e-6)
//   decay[c] = exp(lower_bound * sigmoid(-ssm_a[h] * (raw_gate[c] + dt_bias[c])))
//   beta     = sigmoid(raw_beta[h])
//   S[v][:] *= decay;  S[v][:] += k * beta * (v[v] - S[v] . k);  o[v] = S[v] . q
//   out      = o * rsqrt(mean(o^2) + norm_eps) * output_norm * sigmoid(output_gate)
// State is F32 [rows][heads][value][key]; conv state is F32
// [rows][q|k|v][3 history][heads * d], oldest first.

#include <metal_stdlib>
using namespace metal;

struct glm53_kda_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
};

// One threadgroup (128 threads, four simdgroups) owns one (row, head); each
// simdgroup updates value rows sg, sg + 4, ...; every lane owns four keys.
kernel void kernel_glm53_kda_decode(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * q_in [[buffer(1)]],
        device const float * k_in [[buffer(2)]],
        device const float * v_in [[buffer(3)]],
        device const float * raw_gate [[buffer(4)]],
        device const float * raw_beta [[buffer(5)]],
        device const float * output_gate [[buffer(6)]],
        device const float * q_conv [[buffer(7)]],
        device const float * k_conv [[buffer(8)]],
        device const float * v_conv [[buffer(9)]],
        device const float * neg_exp_a_log [[buffer(10)]],
        device const float * dt_bias [[buffer(11)]],
        device const float * output_norm [[buffer(12)]],
        device float * conv_state [[buffer(13)]],
        device float * state [[buffer(14)]],
        device float * out [[buffer(15)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    threadgroup float * sq = scratch;
    threadgroup float * sk = sq + D;
    threadgroup float * sd = sk + D;
    threadgroup float * sv = sd + D;
    threadgroup float * so = sv + D;
    threadgroup float * reduce_q = so + D;
    threadgroup float * reduce_k = reduce_q + 4u;
    threadgroup float * reduce_o = reduce_k + 4u;
    threadgroup float * beta_shared = reduce_o + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;
    const ulong conv_row_stride = 3ul * HISTORY * projection;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float * q_state = conv_state + (ulong)row * conv_row_stride;
        device float * k_state = q_state + HISTORY * projection;
        device float * v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel], q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel], k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel], v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        // sigmoid(-ssm_a * gate) with ssm_a = -exp(A_log).
        sd[tid] = exp(args.lower_bound * (1.0f / (1.0f + exp(neg_exp_a_log[head] * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] = 1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;  // 1/sqrt(128)
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head = ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 * hptr = (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate = 1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}

// Packed prefill in three dispatches, adapted from DwarfStar's
// kernel_glm53_kda_prefill_{prepare,recurrence,output} (same revision and
// notice as the decode kernel). Rows are consecutive tokens of one sequence;
// activations are [rows][heads * 128]. Changes: explicit buffer indices,
// `-ssm_a` in the decay, and decode's rounding order for the q scale and the
// output gate so prefill and decode agree.

// Stage 1, one threadgroup per head, serial over rows: causal conv + SiLU
// (advancing the conv tails), L2-normalized and scaled q/k, and the
// per-channel decay, all in place (raw_gate becomes the decay).
kernel void kernel_glm53_kda_prefill_prepare(
        constant glm53_kda_args & args [[buffer(0)]],
        device float * q [[buffer(1)]],
        device float * k [[buffer(2)]],
        device float * v [[buffer(3)]],
        device float * raw_gate [[buffer(4)]],
        device const float * q_conv [[buffer(5)]],
        device const float * k_conv [[buffer(6)]],
        device const float * v_conv [[buffer(7)]],
        device const float * neg_exp_a_log [[buffer(8)]],
        device const float * dt_bias [[buffer(9)]],
        device float * conv_state [[buffer(10)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint head [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    if (head >= args.n_heads) return;
    threadgroup float * sq = scratch;
    threadgroup float * sk = sq + D;
    threadgroup float * reduce_q = sk + D;
    threadgroup float * reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    device float * q_state = conv_state;
    device float * k_state = q_state + HISTORY * projection;
    device float * v_state = k_state + HISTORY * projection;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong index = (ulong)token * projection + channel;
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel], q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel], k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel], v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q[index];
        const float k_new = k[index];
        const float v_new = v[index];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);
        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        v[index] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[index] + dt_bias[channel];
        raw_gate[index] = exp(args.lower_bound * (1.0f / (1.0f + exp(neg_exp_a_log[head] * gate))));
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

        float q_sumsq = simd_sum(sq[tid] * sq[tid]);
        float k_sumsq = simd_sum(sk[tid] * sk[tid]);
        if (lane == 0u) {
            reduce_q[sg] = q_sumsq;
            reduce_k[sg] = k_sumsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
        float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
        q_total = simd_sum(q_total);
        k_total = simd_sum(k_total);
        const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
        const float k_scale = rsqrt(k_total + 1.0e-6f);
        q[index] = sq[tid] * q_scale;
        k[index] = sk[tid] * k_scale;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
}

// Parallel prefill preparation uses caller-provided dead scratch to preserve
// the projected q/k/v rows. This first pass snapshots them before any row is
// transformed, so every (row, head) group can evaluate its causal window
// independently without racing with neighboring rows.
kernel void kernel_glm53_kda_prefill_snapshot(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * q [[buffer(1)]],
        device const float * k [[buffer(2)]],
        device const float * v [[buffer(3)]],
        device float * raw_q [[buffer(4)]],
        device float * raw_k [[buffer(5)]],
        device float * raw_v [[buffer(6)]],
        uint index [[thread_position_in_grid]]) {
    const uint count = args.n_rows * args.n_heads * 128u;
    if (index < count) {
        raw_q[index] = q[index];
        raw_k[index] = k[index];
        raw_v[index] = v[index];
    }
}

// Parallel counterpart to kernel_glm53_kda_prefill_prepare. One group owns
// one (row, head), reads only the preserved projected rows plus the incoming
// three-row conv tail, and writes that row's prepared activations.
kernel void kernel_glm53_kda_prefill_prepare_parallel(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * raw_q [[buffer(1)]],
        device const float * raw_k [[buffer(2)]],
        device const float * raw_v [[buffer(3)]],
        device float * q [[buffer(4)]],
        device float * k [[buffer(5)]],
        device float * v [[buffer(6)]],
        device float * raw_gate [[buffer(7)]],
        device const float * q_conv [[buffer(8)]],
        device const float * k_conv [[buffer(9)]],
        device const float * v_conv [[buffer(10)]],
        device const float * neg_exp_a_log [[buffer(11)]],
        device const float * dt_bias [[buffer(12)]],
        device const float * conv_state [[buffer(13)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_rows || head >= args.n_heads) return;
    threadgroup float * sq = scratch;
    threadgroup float * sk = sq + D;
    threadgroup float * reduce_q = sk + D;
    threadgroup float * reduce_k = reduce_q + 4u;
    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong index = (ulong)token * projection + channel;
    const ulong q_state = 0ul;
    const ulong k_state = (ulong)HISTORY * projection;
    const ulong v_state = (ulong)(2u * HISTORY) * projection;

    float q_acc = 0.0f;
    float k_acc = 0.0f;
    float v_acc = 0.0f;
    for (uint w = 0; w < HISTORY; ++w) {
        const int source_token = (int)token + (int)w - (int)HISTORY;
        float q_hist;
        float k_hist;
        float v_hist;
        if (source_token < 0) {
            const ulong hist = (ulong)((int)token + (int)w) * projection + channel;
            q_hist = conv_state[q_state + hist];
            k_hist = conv_state[k_state + hist];
            v_hist = conv_state[v_state + hist];
        } else {
            const ulong source = (ulong)source_token * projection + channel;
            q_hist = raw_q[source];
            k_hist = raw_k[source];
            v_hist = raw_v[source];
        }
        q_acc = fma(q_hist, q_conv[(ulong)channel * 4u + w], q_acc);
        k_acc = fma(k_hist, k_conv[(ulong)channel * 4u + w], k_acc);
        v_acc = fma(v_hist, v_conv[(ulong)channel * 4u + w], v_acc);
    }
    const float q_new = raw_q[index];
    const float k_new = raw_k[index];
    const float v_new = raw_v[index];
    q_acc = fma(q_new, q_conv[(ulong)channel * 4u + HISTORY], q_acc);
    k_acc = fma(k_new, k_conv[(ulong)channel * 4u + HISTORY], k_acc);
    v_acc = fma(v_new, v_conv[(ulong)channel * 4u + HISTORY], v_acc);
    sq[tid] = q_acc / (1.0f + exp(-q_acc));
    sk[tid] = k_acc / (1.0f + exp(-k_acc));
    v[index] = v_acc / (1.0f + exp(-v_acc));
    const float gate = raw_gate[index] + dt_bias[channel];
    raw_gate[index] = exp(args.lower_bound * (1.0f / (1.0f + exp(neg_exp_a_log[head] * gate))));
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float q_sumsq = simd_sum(sq[tid] * sq[tid]);
    float k_sumsq = simd_sum(sk[tid] * sk[tid]);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    q[index] = sq[tid] * q_scale;
    k[index] = sk[tid] * k_scale;
}

// Commit the final three projected rows as the new convolution tail after
// every independent row has consumed the original tail.
kernel void kernel_glm53_kda_prefill_update_tail(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * raw_q [[buffer(1)]],
        device const float * raw_k [[buffer(2)]],
        device const float * raw_v [[buffer(3)]],
        device float * conv_state [[buffer(4)]],
        uint channel [[thread_position_in_grid]]) {
    const uint projection = args.n_heads * 128u;
    if (channel >= projection) return;
    const ulong section = (ulong)3u * projection;
    for (uint slot = 0; slot < 3u; ++slot) {
        const int source_token = (int)args.n_rows - 3 + (int)slot;
        float q_value;
        float k_value;
        float v_value;
        if (source_token < 0) {
            const ulong history = (ulong)((int)args.n_rows + (int)slot) * projection + channel;
            q_value = conv_state[history];
            k_value = conv_state[section + history];
            v_value = conv_state[2ul * section + history];
        } else {
            const ulong source = (ulong)source_token * projection + channel;
            q_value = raw_q[source];
            k_value = raw_k[source];
            v_value = raw_v[source];
        }
        conv_state[(ulong)slot * projection + channel] = q_value;
        conv_state[section + (ulong)slot * projection + channel] = k_value;
        conv_state[2ul * section + (ulong)slot * projection + channel] = v_value;
    }
}

// Stage 2, grid (heads, 32) x 128: each simdgroup owns one value row of the
// state in registers across all rows; lanes own four keys.
kernel void kernel_glm53_kda_prefill_recurrence(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * q [[buffer(1)]],
        device const float * k [[buffer(2)]],
        device const float * v [[buffer(3)]],
        device const float * decay [[buffer(4)]],
        device const float * raw_beta [[buffer(5)]],
        device float * state [[buffer(6)]],
        device float * out [[buffer(7)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint value = tgpig.y * 4u + sg;
    if (head >= args.n_heads || value >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    device float4 * state_ptr = (device float4 *)(state + ((ulong)head * D + value) * D + k0);
    float4 h = *state_ptr;

    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 = *((device const float4 *)(decay + base + k0));
        h *= decay4;
        const float hk = simd_sum(dot(h, k4));
        const float beta = 1.0f / (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
        const float delta_v = (v[base + value] - hk) * beta;
        h = fma(k4, float4(delta_v), h);
        const float result = simd_sum(dot(h, q4));
        if (lane == 0u) out[base + value] = result;
    }
    *state_ptr = h;
}

// Four independent value rows per SIMD group reuse each q/k/decay load.
// Each row retains the scalar kernel's float4 operations and SIMD reductions;
// only the independent rows are interleaved. Grid (heads, D/16) x 128.
kernel void kernel_glm53_kda_prefill_recurrence_value4(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * q [[buffer(1)]],
        device const float * k [[buffer(2)]],
        device const float * v [[buffer(3)]],
        device const float * decay [[buffer(4)]],
        device const float * raw_beta [[buffer(5)]],
        device float * state [[buffer(6)]],
        device float * out [[buffer(7)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = tgpig.x;
    const uint first = tgpig.y * 16u + sg * 4u;
    if (head >= args.n_heads || first >= D) return;
    const uint projection = args.n_heads * D;
    const uint k0 = lane * 4u;
    device float4 * state_ptr = (device float4 *)(state + ((ulong)head * D + first) * D + k0);
    float4 h[4];
#pragma clang loop unroll(full)
    for (uint j = 0; j < 4u; ++j) h[j] = state_ptr[j * (D / 4u)];
    for (uint token = 0; token < args.n_rows; token++) {
        const ulong base = (ulong)token * projection + head * D;
        const float4 q4 = *((device const float4 *)(q + base + k0));
        const float4 k4 = *((device const float4 *)(k + base + k0));
        const float4 decay4 = *((device const float4 *)(decay + base + k0));
        const float beta = 1.0f / (1.0f + exp(-raw_beta[(ulong)token * args.n_heads + head]));
#pragma clang loop unroll(full)
        for (uint j = 0; j < 4u; ++j) {
            h[j] *= decay4;
            const float hk = simd_sum(dot(h[j], k4));
            const float delta_v = (v[base + first + j] - hk) * beta;
            h[j] = fma(k4, float4(delta_v), h[j]);
            const float result = simd_sum(dot(h[j], q4));
            if (lane == 0u) out[base + first + j] = result;
        }
    }
#pragma clang loop unroll(full)
    for (uint j = 0; j < 4u; ++j) state_ptr[j * (D / 4u)] = h[j];
}

// Stage 3, grid (rows, heads) x 128: per-head RMSNorm, norm weight and
// sigmoid output gate, in place.
kernel void kernel_glm53_kda_prefill_output(
        constant glm53_kda_args & args [[buffer(0)]],
        device float * out [[buffer(1)]],
        device const float * output_gate [[buffer(2)]],
        device const float * output_norm [[buffer(3)]],
        threadgroup float * partial [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint token = tgpig.x;
    const uint head = tgpig.y;
    if (token >= args.n_rows || head >= args.n_heads) return;
    const uint projection = args.n_heads * D;
    const ulong base = (ulong)token * projection + head * D;
    const float raw = out[base + tid];
    float sumsq = simd_sum(raw * raw);
    if (lane == 0u) partial[sg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = lane < 4u ? partial[lane] : 0.0f;
    total = simd_sum(total);
    const float scale = rsqrt(total / (float)D + args.norm_eps);
    const float gate = 1.0f / (1.0f + exp(-output_gate[base + tid]));
    out[base + tid] = raw * scale * output_norm[tid] * gate;
}

// Independent members: identical per-head arithmetic, separate state pointers.
kernel void kernel_glm53_kda_decode_members(
        constant glm53_kda_args & args [[buffer(0)]],
        device const float * q_in [[buffer(1)]],
        device const float * k_in [[buffer(2)]],
        device const float * v_in [[buffer(3)]],
        device const float * raw_gate [[buffer(4)]],
        device const float * raw_beta [[buffer(5)]],
        device const float * output_gate [[buffer(6)]],
        device const float * q_conv [[buffer(7)]],
        device const float * k_conv [[buffer(8)]],
        device const float * v_conv [[buffer(9)]],
        device const float * neg_exp_a_log [[buffer(10)]],
        device const float * dt_bias [[buffer(11)]],
        device const float * output_norm [[buffer(12)]],
        device float * conv_state0 [[buffer(13)]],
        device float * state0 [[buffer(14)]],
        device float * out [[buffer(15)]],
        device float * conv_state1 [[buffer(16)]],
        device float * state1 [[buffer(17)]],
        device float * conv_state2 [[buffer(18)]],
        device float * state2 [[buffer(19)]],
        device float * conv_state3 [[buffer(20)]],
        device float * state3 [[buffer(21)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    constexpr uint HISTORY = 3u;
    const uint row = tgpig.x;
    const uint head = tgpig.y;
    if (row >= args.n_rows || head >= args.n_heads) return;

    device float * conv_state = row == 0u ? conv_state0 : row == 1u ? conv_state1 : row == 2u ? conv_state2 : conv_state3;
    device float * state = row == 0u ? state0 : row == 1u ? state1 : row == 2u ? state2 : state3;
    threadgroup float * sq = scratch;
    threadgroup float * sk = sq + D;
    threadgroup float * sd = sk + D;
    threadgroup float * sv = sd + D;
    threadgroup float * so = sv + D;
    threadgroup float * reduce_q = so + D;
    threadgroup float * reduce_k = reduce_q + 4u;
    threadgroup float * reduce_o = reduce_k + 4u;
    threadgroup float * beta_shared = reduce_o + 4u;

    const uint projection = args.n_heads * D;
    const uint channel = head * D + tid;
    const ulong input_base = (ulong)row * projection + head * D;

    if (tid < D) {
        float q_acc = 0.0f;
        float k_acc = 0.0f;
        float v_acc = 0.0f;
        device float * q_state = conv_state;
        device float * k_state = q_state + HISTORY * projection;
        device float * v_state = k_state + HISTORY * projection;
        for (uint w = 0; w < HISTORY; w++) {
            q_acc = fma(q_state[(ulong)w * projection + channel], q_conv[(ulong)channel * 4u + w], q_acc);
            k_acc = fma(k_state[(ulong)w * projection + channel], k_conv[(ulong)channel * 4u + w], k_acc);
            v_acc = fma(v_state[(ulong)w * projection + channel], v_conv[(ulong)channel * 4u + w], v_acc);
        }
        const float q_new = q_in[input_base + tid];
        const float k_new = k_in[input_base + tid];
        const float v_new = v_in[input_base + tid];
        q_acc = fma(q_new, q_conv[(ulong)channel * 4u + 3u], q_acc);
        k_acc = fma(k_new, k_conv[(ulong)channel * 4u + 3u], k_acc);
        v_acc = fma(v_new, v_conv[(ulong)channel * 4u + 3u], v_acc);

        q_state[channel] = q_state[projection + channel];
        q_state[projection + channel] = q_state[2ul * projection + channel];
        q_state[2ul * projection + channel] = q_new;
        k_state[channel] = k_state[projection + channel];
        k_state[projection + channel] = k_state[2ul * projection + channel];
        k_state[2ul * projection + channel] = k_new;
        v_state[channel] = v_state[projection + channel];
        v_state[projection + channel] = v_state[2ul * projection + channel];
        v_state[2ul * projection + channel] = v_new;

        sq[tid] = q_acc / (1.0f + exp(-q_acc));
        sk[tid] = k_acc / (1.0f + exp(-k_acc));
        sv[tid] = v_acc / (1.0f + exp(-v_acc));
        const float gate = raw_gate[input_base + tid] + dt_bias[channel];
        // sigmoid(-ssm_a * gate) with ssm_a = -exp(A_log).
        sd[tid] = exp(args.lower_bound * (1.0f / (1.0f + exp(neg_exp_a_log[head] * gate))));
    }
    if (tid == 0u) {
        beta_shared[0] = 1.0f / (1.0f + exp(-raw_beta[(ulong)row * args.n_heads + head]));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float q_sumsq = sq[tid] * sq[tid];
    float k_sumsq = sk[tid] * sk[tid];
    q_sumsq = simd_sum(q_sumsq);
    k_sumsq = simd_sum(k_sumsq);
    if (lane == 0u) {
        reduce_q[sg] = q_sumsq;
        reduce_k[sg] = k_sumsq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float q_total = lane < 4u ? reduce_q[lane] : 0.0f;
    float k_total = lane < 4u ? reduce_k[lane] : 0.0f;
    q_total = simd_sum(q_total);
    k_total = simd_sum(k_total);
    const float q_scale = rsqrt(q_total + 1.0e-6f) * 0x1.6a09e6p-4f;  // 1/sqrt(128)
    const float k_scale = rsqrt(k_total + 1.0e-6f);
    if (tid < D) {
        sq[tid] *= q_scale;
        sk[tid] *= k_scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head = (ulong)head * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 * hptr = (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr * decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        *hptr = h;
        float hq = simd_sum(dot(h, q4));
        if (lane == 0u) so[value] = hq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    float o_sumsq = so[tid] * so[tid];
    o_sumsq = simd_sum(o_sumsq);
    if (lane == 0u) reduce_o[sg] = o_sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float o_total = lane < 4u ? reduce_o[lane] : 0.0f;
    o_total = simd_sum(o_total);
    const float o_scale = rsqrt(o_total / (float)D + args.norm_eps);
    if (tid < D) {
        const ulong index = input_base + tid;
        const float gate = 1.0f / (1.0f + exp(-output_gate[index]));
        out[index] = so[tid] * o_scale * output_norm[tid] * gate;
    }
}
