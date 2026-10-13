// Research-only bounded FP32 ReplaySSM KDA, D=128, 128 threads/head.
// Experimental windows 1 and 8. Host owns residue, no device cursor.
// Checkpoint [head][value][key]; records [head][slot][channel], separate k/d/u.
// Historical u is cached at creation, never recomputed. Replay preserves the
// ordinary multiply then fma order for each state float4, chronologically.
// This adds record memory to an FP32 checkpoint; it is not memory compression.
// Arithmetic below is copied from the ordinary KDA decode; see attribution.
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

struct glm53_kda_replay_args {
    uint n_heads;
    uint n_rows;
    float lower_bound;
    float norm_eps;
    uint window;
    uint residue;
};

// One threadgroup (128 threads, four simdgroups) owns one (row, head); each
// simdgroup updates value rows sg, sg + 4, ...; every lane owns four keys.
kernel void kernel_research_glm53_kda_replay_f32(
        constant glm53_kda_replay_args & args [[buffer(0)]],
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
        device float * record_k [[buffer(16)]],
        device float * record_d [[buffer(17)]],
        device float * record_u [[buffer(18)]],
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

    // 656 ordinary scratch floats (2624 bytes), then 3*window*D history
    // floats (12288 bytes at L8). Every historical record is staged once/head.
    threadgroup float * hist_k = scratch + 656u;
    threadgroup float * hist_d = hist_k + args.window * D;
    threadgroup float * hist_u = hist_d + args.window * D;
    const ulong record_head = ulong(head) * args.window * D;
    for (uint j = 0; j < args.residue; ++j) {
        const uint i = j * D + tid;
        hist_k[i] = record_k[record_head + i];
        hist_d[i] = record_d[record_head + i];
        hist_u[i] = record_u[record_head + i];
    }
    const bool flush = args.residue + 1u == args.window;
    const ulong append = record_head + ulong(args.residue) * D;

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

    if (!flush) {
        record_k[append + tid] = sk[tid];
        record_d[append + tid] = sd[tid];
    }
    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head = ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        device float4 * hptr = (device float4 *)(state + state_head + (ulong)value * D + k0);
        float4 h = *hptr;
        for (uint j = 0; j < args.residue; ++j) {
            const float4 old_d = *((threadgroup float4 *)(hist_d + j * D + k0));
            const float4 old_k = *((threadgroup float4 *)(hist_k + j * D + k0));
            h *= old_d;
            h = fma(old_k, float4(hist_u[j * D + value]), h);
        }
        h *= decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        if (flush) *hptr = h;
        else if (lane == 0u) record_u[append + value] = delta_v;
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

// Read-only materialization at any driver-owned residue; output is a distinct
// FP32 state. No conv, checkpoint, record or cursor mutation.
kernel void kernel_research_glm53_kda_replay_materialize_f32(
        constant glm53_kda_replay_args & args [[buffer(0)]],
        device const float * checkpoint [[buffer(1)]],
        device const float * record_k [[buffer(2)]],
        device const float * record_d [[buffer(3)]],
        device const float * record_u [[buffer(4)]],
        device float * output [[buffer(5)]],
        threadgroup float * history [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = group.y;
    if (head >= args.n_heads) return;
    threadgroup float * hist_k = history;
    threadgroup float * hist_d = hist_k + args.window * D;
    threadgroup float * hist_u = hist_d + args.window * D;
    const ulong record_head = ulong(head) * args.window * D;
    for (uint j = 0; j < args.residue; ++j) {
        const uint i = j * D + tid;
        hist_k[i] = record_k[record_head + i];
        hist_d[i] = record_d[record_head + i];
        hist_u[i] = record_u[record_head + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k0 = lane * 4u;
    for (uint value = sg; value < D; value += 4u) {
        const ulong index = (ulong(head) * D + value) * D + k0;
        float4 h = *((device const float4 *)(checkpoint + index));
        for (uint j = 0; j < args.residue; ++j) {
            const float4 old_d = *((threadgroup float4 *)(hist_d + j * D + k0));
            const float4 old_k = *((threadgroup float4 *)(hist_k + j * D + k0));
            h *= old_d;
            h = fma(old_k, float4(hist_u[j * D + value]), h);
        }
        *((device float4 *)(output + index)) = h;
    }
}

// Physical-F16 read experiments, L8 only. ReplaySSM paper concept: separate
// approximate reads from authoritative replay/repair. Implementation adapted
// locally from the attributed ordinary KDA arithmetic above; no paper code.
// Nonflush uses cached LP u; flush stages raw post-conv/SiLU v in the same
// history plane and recomputes every delta from the anchor, never cached LP u.
// 656 + 3*8*128 + 8 floats = 14944 bytes, staged once/head; uniform barriers.
kernel void kernel_research_glm53_kda_replay_f16_separate(
        constant glm53_kda_replay_args & args [[buffer(0)]],
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
        device float * record_k [[buffer(16)]],
        device float * record_d [[buffer(17)]],
        device float * record_u [[buffer(18)]],
        device half * read_state [[buffer(19)]],
        device float * record_v [[buffer(20)]],
        device float * record_beta [[buffer(21)]],
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

    // Same three history planes; third plane changes meaning only at flush.
    threadgroup float * hist_k = scratch + 656u;
    threadgroup float * hist_d = hist_k + args.window * D;
    threadgroup float * hist_u = hist_d + args.window * D;
    threadgroup float * hist_beta = hist_u + args.window * D;
    const bool flush = args.residue + 1u == args.window;
    if (flush && tid < args.residue)
        hist_beta[tid] = record_beta[ulong(head) * args.window + tid];
    const ulong record_head = ulong(head) * args.window * D;
    for (uint j = 0; j < args.residue; ++j) {
        const uint i = j * D + tid;
        hist_k[i] = record_k[record_head + i];
        hist_d[i] = record_d[record_head + i];
        hist_u[i] = flush ? record_v[record_head + i] : record_u[record_head + i];
    }
    const ulong append = record_head + ulong(args.residue) * D;

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

    if (!flush) {
        record_k[append + tid] = sk[tid];
        record_d[append + tid] = sd[tid];
        record_v[append + tid] = sv[tid];
        if (tid == 0u) record_beta[ulong(head) * args.window + args.residue] = beta_shared[0];
    }
    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head = ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        const ulong index = state_head + (ulong)value * D + k0;
        // Load exactly one matrix: authoritative anchor on flush, physical
        // half read otherwise. No preliminary LP reconstruction on flush.
        float4 h;
        if (flush) h = float4(*((device const float4 *)(state + index)));
        else h = float4(*((device const half4 *)(read_state + index)));
        for (uint j = 0; j < args.residue; ++j) {
            const float4 old_d = *((threadgroup float4 *)(hist_d + j * D + k0));
            const float4 old_k = *((threadgroup float4 *)(hist_k + j * D + k0));
            h *= old_d;
            float old_u = hist_u[j * D + value];
            if (flush) {
                float old_hk = dot(h, old_k);
                old_hk = simd_sum(old_hk);
                old_u = (old_u - old_hk) * hist_beta[j];
            }
            h = fma(old_k, float4(old_u), h);
        }
        h *= decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        if (flush) {
            *((device float4 *)(state + index)) = h;
            *((device half4 *)(read_state + index)) = half4(h);
        }
        else if (lane == 0u) record_u[append + value] = delta_v;
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

kernel void kernel_research_glm53_kda_replay_f16_shared(
        constant glm53_kda_replay_args & args [[buffer(0)]],
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
        device half * state [[buffer(14)]],
        device float * out [[buffer(15)]],
        device float * record_k [[buffer(16)]],
        device float * record_d [[buffer(17)]],
        device float * record_u [[buffer(18)]],
        device half * read_state [[buffer(19)]],
        device float * record_v [[buffer(20)]],
        device float * record_beta [[buffer(21)]],
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

    // Same three history planes; third plane changes meaning only at flush.
    threadgroup float * hist_k = scratch + 656u;
    threadgroup float * hist_d = hist_k + args.window * D;
    threadgroup float * hist_u = hist_d + args.window * D;
    threadgroup float * hist_beta = hist_u + args.window * D;
    const bool flush = args.residue + 1u == args.window;
    if (flush && tid < args.residue)
        hist_beta[tid] = record_beta[ulong(head) * args.window + tid];
    const ulong record_head = ulong(head) * args.window * D;
    for (uint j = 0; j < args.residue; ++j) {
        const uint i = j * D + tid;
        hist_k[i] = record_k[record_head + i];
        hist_d[i] = record_d[record_head + i];
        hist_u[i] = flush ? record_v[record_head + i] : record_u[record_head + i];
    }
    const ulong append = record_head + ulong(args.residue) * D;

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

    if (!flush) {
        record_k[append + tid] = sk[tid];
        record_d[append + tid] = sd[tid];
        record_v[append + tid] = sv[tid];
        if (tid == 0u) record_beta[ulong(head) * args.window + args.residue] = beta_shared[0];
    }
    const uint k0 = lane * 4u;
    const float4 q4 = *((threadgroup float4 *)(sq + k0));
    const float4 k4 = *((threadgroup float4 *)(sk + k0));
    const float4 decay4 = *((threadgroup float4 *)(sd + k0));
    const ulong state_head = ((ulong)row * args.n_heads + head) * D * D;

    for (uint value = sg; value < D; value += 4u) {
        const ulong index = state_head + (ulong)value * D + k0;
        // Load exactly one matrix: authoritative anchor on flush, physical
        // half read otherwise. No preliminary LP reconstruction on flush.
        float4 h;
        if (flush) h = float4(*((device const half4 *)(state + index)));
        else h = float4(*((device const half4 *)(read_state + index)));
        for (uint j = 0; j < args.residue; ++j) {
            const float4 old_d = *((threadgroup float4 *)(hist_d + j * D + k0));
            const float4 old_k = *((threadgroup float4 *)(hist_k + j * D + k0));
            h *= old_d;
            float old_u = hist_u[j * D + value];
            if (flush) {
                float old_hk = dot(h, old_k);
                old_hk = simd_sum(old_hk);
                old_u = (old_u - old_hk) * hist_beta[j];
            }
            h = fma(old_k, float4(old_u), h);
        }
        h *= decay4;
        float hk = dot(h, k4);
        hk = simd_sum(hk);
        const float delta_v = (sv[value] - hk) * beta_shared[0];
        h = fma(k4, float4(delta_v), h);
        if (flush) {
            *((device half4 *)(state + index)) = half4(h); // read_state aliases state exactly
        }
        else if (lane == 0u) record_u[append + value] = delta_v;
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


// Read-only LP logical state, including the rounded read at residue zero.
kernel void kernel_research_glm53_kda_replay_materialize_f16(
        constant glm53_kda_replay_args & args [[buffer(0)]],
        device const half * checkpoint [[buffer(1)]],
        device const float * record_k [[buffer(2)]],
        device const float * record_d [[buffer(3)]],
        device const float * record_u [[buffer(4)]],
        device float * output [[buffer(5)]],
        threadgroup float * history [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    constexpr uint D = 128u;
    const uint head = group.y;
    if (head >= args.n_heads) return;
    threadgroup float * hist_k = history;
    threadgroup float * hist_d = hist_k + args.window * D;
    threadgroup float * hist_u = hist_d + args.window * D;
    const ulong record_head = ulong(head) * args.window * D;
    for (uint j = 0; j < args.residue; ++j) {
        const uint i = j * D + tid;
        hist_k[i] = record_k[record_head + i];
        hist_d[i] = record_d[record_head + i];
        hist_u[i] = record_u[record_head + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint k0 = lane * 4u;
    for (uint value = sg; value < D; value += 4u) {
        const ulong index = (ulong(head) * D + value) * D + k0;
        float4 h = float4(*((device const half4 *)(checkpoint + index)));
        for (uint j = 0; j < args.residue; ++j) {
            const float4 old_d = *((threadgroup float4 *)(hist_d + j * D + k0));
            const float4 old_k = *((threadgroup float4 *)(hist_k + j * D + k0));
            h *= old_d;
            h = fma(old_k, float4(hist_u[j * D + value]), h);
        }
        *((device float4 *)(output + index)) = h;
    }
}
