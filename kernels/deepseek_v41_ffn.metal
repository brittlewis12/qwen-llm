#include <metal_stdlib>
using namespace metal;

kernel void kernel_deepseek_v41_group_rows(
    constant uint3 &shape [[buffer(0)]], device const float *input [[buffer(1)]],
    device float *output [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    uint width = shape.x, groups = shape.y, rows = shape.z;
    if (i >= width * groups * rows) return;
    uint row = i / (width * groups), group = (i / width) % groups;
    output[(group * rows + row) * width + i % width] = input[i];
}

// Projection must precede RMS scaling; collapse uses a different, lagged pre.
kernel void kernel_deepseek_v41_projected_rms(
    constant uint &n [[buffer(0)]], constant float &eps [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *mix [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]], uint row [[threadgroup_position_in_grid]]) {
    x += row * n;
    mix += row * 24;
    threadgroup float sums[256];
    float sum = 0.0f;
    for (uint i = tid; i < n; i += 256) sum += x[i] * x[i];
    sums[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 128; stride; stride >>= 1) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid < 24) mix[tid] *= rsqrt(sums[0] / float(n) + eps);
}

// mHC pre-block for one decode row, as two dispatches instead of five.
//
// Projection: the 24 mixing logits are a 4*hidden -> 24 F32 GEMV. One
// simdgroup per output row (the generic F32 mat-vec) leaves 24 simdgroups
// on the GPU, each walking a 160-deep load chain; here one 1024-thread
// threadgroup owns each output and strides K, so the 2 MB of weights are in
// flight at once.
constant uint V41_HC_PROJECTION_THREADS = 1024;
kernel void kernel_deepseek_v41_hc_projection(
    constant uint &n_in [[buffer(0)]], constant uint &n_out [[buffer(1)]],
    device const float *weight [[buffer(2)]], device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[V41_HC_PROJECTION_THREADS / 32];
    if (row >= n_out) return;
    device const float4 *w4 = (device const float4 *)(weight + ulong(row) * n_in);
    device const float4 *x4 = (device const float4 *)x;
    const uint n4 = n_in / 4;
    float sum = 0.0f;
    for (uint i = tid; i < n4; i += V41_HC_PROJECTION_THREADS) {
        const float4 a = w4[i];
        const float4 b = x4[i];
        sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    for (uint i = n4 * 4 + tid; i < n_in; i += V41_HC_PROJECTION_THREADS) sum += weight[ulong(row) * n_in + i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0.0f;
        for (uint s = 0; s < V41_HC_PROJECTION_THREADS / 32; ++s) total += partial[s];
        y[row] = total;
    }
}

// Fused: projected RMS of the residual scales the 24 mixes; sigmoid pre/post
// and the Sinkhorn-normalized 4x4 combination (same arithmetic as
// kernel_deepseek_v4_hc_controls, in registers instead of device
// read-modify-write); collapse of the four streams with the *incoming*
// (lagged) pre; weighted RMS norm of the collapsed row. One 1024-thread
// threadgroup; reductions change F32 summation order relative to the
// separate kernels, nothing else.
struct HcPreArgs { uint hidden; float norm_eps; float hc_eps; };
constant uint V41_HC_PRE_THREADS = 1024;
constant uint V41_HC_CONNECTIONS = 4;
constant uint V41_HC_SINKHORN_ITERATIONS = 20;
inline float v41_hc_tg_sum(float v, threadgroup float *partial, uint lane, uint sg) {
    v = simd_sum(v);
    if (lane == 0) partial[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint s = 0; s < V41_HC_PRE_THREADS / 32; ++s) total += partial[s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}
kernel void kernel_deepseek_v41_hc_pre(
    constant HcPreArgs &a [[buffer(0)]],
    device const float *x [[buffer(1)]],            // residual [4*hidden]
    device float *mix [[buffer(2)]],                // [24] raw in, scaled out
    device const float *scale [[buffer(3)]],        // [3]
    device const float *base [[buffer(4)]],         // [24]
    device float *pre [[buffer(5)]],                // [4]
    device float *post [[buffer(6)]],               // [4]
    device float *combination [[buffer(7)]],        // [16]
    device const float *incoming_pre [[buffer(8)]], // [4] lagged
    device float *collapsed [[buffer(9)]],          // [hidden]
    device const float *norm_weight [[buffer(10)]], // [hidden]
    device float *normalized [[buffer(11)]],        // [hidden]
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[V41_HC_PRE_THREADS / 32];
    const uint n = V41_HC_CONNECTIONS * a.hidden;
    // A. projected RMS over the whole residual scales the mixes.
    float ss = 0.0f;
    for (uint i = tid; i < n; i += V41_HC_PRE_THREADS) ss += x[i] * x[i];
    const float inv = rsqrt(v41_hc_tg_sum(ss, partial, lane, sg) / float(n) + a.norm_eps);
    if (tid < 24) mix[tid] *= inv;
    threadgroup_barrier(mem_flags::mem_device);
    // B. controls (one thread; 24 inputs, 24 outputs).
    if (tid == 0) {
        const float eps = a.hc_eps;
        for (uint stream = 0; stream < V41_HC_CONNECTIONS; ++stream) {
            pre[stream] = 1.0f / (1.0f + exp(-(mix[stream] * scale[0] + base[stream]))) + eps;
            post[stream] = 2.0f / (1.0f + exp(-(mix[4 + stream] * scale[1] + base[4 + stream])));
        }
        float comb[16];
        for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) {
            float row_max = -INFINITY;
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                const uint index = source * V41_HC_CONNECTIONS + destination;
                const float value = mix[8 + index] * scale[2] + base[8 + index];
                comb[index] = value;
                row_max = max(row_max, value);
            }
            float sum = 0.0f;
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                const uint index = source * V41_HC_CONNECTIONS + destination;
                comb[index] = exp(comb[index] - row_max);
                sum += comb[index];
            }
            const float inverse = 1.0f / sum;
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                const uint index = source * V41_HC_CONNECTIONS + destination;
                comb[index] = comb[index] * inverse + eps;
            }
        }
        for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
            float sum = 0.0f;
            for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) sum += comb[source * V41_HC_CONNECTIONS + destination];
            const float inverse = 1.0f / (sum + eps);
            for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) comb[source * V41_HC_CONNECTIONS + destination] *= inverse;
        }
        for (uint iteration = 1; iteration < V41_HC_SINKHORN_ITERATIONS; ++iteration) {
            for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) {
                float sum = 0.0f;
                for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) sum += comb[source * V41_HC_CONNECTIONS + destination];
                const float inverse = 1.0f / (sum + eps);
                for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) comb[source * V41_HC_CONNECTIONS + destination] *= inverse;
            }
            for (uint destination = 0; destination < V41_HC_CONNECTIONS; ++destination) {
                float sum = 0.0f;
                for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) sum += comb[source * V41_HC_CONNECTIONS + destination];
                const float inverse = 1.0f / (sum + eps);
                for (uint source = 0; source < V41_HC_CONNECTIONS; ++source) comb[source * V41_HC_CONNECTIONS + destination] *= inverse;
            }
        }
        for (uint i = 0; i < 16; ++i) combination[i] = comb[i];
    }
    // C. collapse with the lagged incoming pre (independent of phase B).
    const float p0 = incoming_pre[0], p1 = incoming_pre[1], p2 = incoming_pre[2], p3 = incoming_pre[3];
    float ss2 = 0.0f;
    for (uint d = tid; d < a.hidden; d += V41_HC_PRE_THREADS) {
        float v = 0.0f;
        v += x[d] * p0;
        v += x[a.hidden + d] * p1;
        v += x[2 * a.hidden + d] * p2;
        v += x[3 * a.hidden + d] * p3;
        collapsed[d] = v;
        ss2 += v * v;
    }
    // D. weighted RMS norm of the collapsed row.
    const float inv2 = rsqrt(v41_hc_tg_sum(ss2, partial, lane, sg) / float(a.hidden) + a.norm_eps);
    for (uint d = tid; d < a.hidden; d += V41_HC_PRE_THREADS) {
        normalized[d] = (collapsed[d] * inv2) * norm_weight[d];
    }
}

struct RouteArgs { uint experts; uint topk; float temperature; float scale; };
inline bool finite_bits(float v) { return (as_type<uint>(v) & 0x7f800000u) != 0x7f800000u; }
// MSL has no log1p. Preserve tiny softplus tails instead of rounding 1+x to 1.
inline float log1p_positive(float x) {
    if (x < 0.01f) return x * (1.0f + x * (-0.5f + x * (1.0f / 3.0f - x * 0.25f)));
    return log(1.0f + x);
}
// One 256-thread threadgroup per row. Same per-expert arithmetic, status
// codes, lowest-ID tie-breaks, ascending ID order and sequential weight
// normalization as the serial reference below, so outputs are bitwise equal;
// only the work distribution changes (the serial form cost ~0.4 ms per layer
// on M5 Ultra, ~16 ms of a decode token).
#define ROUTE_THREADS 256u
#define ROUTE_MAX_TOPK 16u
#define ROUTE_NO_FAIL 0xffffffffu
kernel void kernel_deepseek_v41_route(
    constant RouteArgs &a [[buffer(0)]], device const float *logits [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *scores [[buffer(3)]],
    device int *ids [[buffer(4)]], device float *weights [[buffer(5)]],
    device int *status [[buffer(6)]], uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    logits += row * a.experts;
    scores += row * a.experts;
    ids += row * a.topk;
    weights += row * a.topk;
    status += row;
    threadgroup uint tg_fail[ROUTE_THREADS / 32];
    threadgroup float tg_value[ROUTE_THREADS / 32];
    threadgroup int tg_id[ROUTE_THREADS / 32];
    threadgroup int tg_ids[ROUTE_MAX_TOPK];
    if (tid == 0) {
        *status = -1;
        for (uint k = 0; k < a.topk; ++k) { ids[k] = -1; weights[k] = 0; }
    }
    // Phase 1: scores, with the serial kernel's first-failure precedence:
    // at index i, a non-finite logit (-1) precedes a non-finite bias (-2),
    // which precedes a non-finite score or score+bias (-1).
    uint fail = ROUTE_NO_FAIL;
    for (uint i = tid; i < a.experts; i += ROUTE_THREADS) {
        if (!finite_bits(logits[i])) { fail = min(fail, i * 4u); continue; }
        if (!finite_bits(bias[i])) { fail = min(fail, i * 4u + 1u); continue; }
        float z = logits[i] / a.temperature;
        scores[i] = sqrt(max(z, 0.0f) + log1p_positive(exp(-abs(z))));
        if (!finite_bits(scores[i]) || !finite_bits(scores[i] + bias[i])) fail = min(fail, i * 4u + 2u);
    }
    fail = simd_min(fail);
    if (lane == 0) tg_fail[sg] = fail;
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    fail = ROUTE_NO_FAIL;
    for (uint s = 0; s < ROUTE_THREADS / 32; ++s) fail = min(fail, tg_fail[s]);
    if (fail != ROUTE_NO_FAIL) {
        if (tid == 0 && (fail & 3u) == 1u) *status = -2;
        return;
    }
    // Phase 2: top-k by candidate value, lowest ID on ties, excluding chosen.
    for (uint k = 0; k < a.topk; ++k) {
        int best = -1;
        float value = -INFINITY;
        for (uint i = tid; i < a.experts; i += ROUTE_THREADS) {
            bool used = false;
            for (uint j = 0; j < k; ++j) used |= tg_ids[j] == int(i);
            float candidate = scores[i] + bias[i];
            if (!used && (best < 0 || candidate > value)) { best = int(i); value = candidate; }
        }
        float m = simd_max(value);
        int id = simd_min(best >= 0 && value == m ? best : INT_MAX);
        if (lane == 0) { tg_value[sg] = m; tg_id[sg] = id; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float best_value = -INFINITY;
        int best_id = INT_MAX;
        for (uint s = 0; s < ROUTE_THREADS / 32; ++s) {
            if (tg_id[s] == INT_MAX) continue;
            if (tg_value[s] > best_value || (tg_value[s] == best_value && tg_id[s] < best_id)) {
                best_value = tg_value[s];
                best_id = tg_id[s];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) tg_ids[k] = best_id == INT_MAX ? -1 : best_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    // Phase 3: ascending IDs, sequential normalization, serial order.
    if (tid == 0) {
        for (uint k = 0; k < a.topk; ++k) ids[k] = tg_ids[k];
        for (uint k = 1; k < a.topk; ++k) {
            int v = ids[k]; uint j = k;
            while (j && ids[j - 1] > v) { ids[j] = ids[j - 1]; --j; }
            ids[j] = v;
        }
        float denom = 1.0f;
        if (a.topk > 1) {
            denom = 0;
            for (uint k = 0; k < a.topk; ++k) denom += scores[ids[k]];
            denom += 1e-20f;
        }
        for (uint k = 0; k < a.topk; ++k) {
            weights[k] = scores[ids[k]] / denom * a.scale;
            if (!finite_bits(weights[k])) { *status = -6; return; }
        }
        *status = 1;
    }
}

// Serial reference (one thread per row): the parity oracle for the kernel above.
kernel void kernel_deepseek_v41_route_serial(
    constant RouteArgs &a [[buffer(0)]], device const float *logits [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *scores [[buffer(3)]],
    device int *ids [[buffer(4)]], device float *weights [[buffer(5)]],
    device int *status [[buffer(6)]], uint row [[threadgroup_position_in_grid]]) {
    logits += row * a.experts;
    scores += row * a.experts;
    ids += row * a.topk;
    weights += row * a.topk;
    status += row;
    *status = -1;
    for (uint k = 0; k < a.topk; ++k) { ids[k] = -1; weights[k] = 0; }
    for (uint i = 0; i < a.experts; ++i) {
        if (!finite_bits(logits[i])) return;
        if (!finite_bits(bias[i])) { *status = -2; return; }
        float z = logits[i] / a.temperature;
        scores[i] = sqrt(max(z, 0.0f) + log1p_positive(exp(-abs(z))));
        if (!finite_bits(scores[i]) || !finite_bits(scores[i] + bias[i])) return;
    }
    for (uint k = 0; k < a.topk; ++k) {
        int best = -1;
        float value = -INFINITY;
        for (uint i = 0; i < a.experts; ++i) {
            bool used = false;
            for (uint j = 0; j < k; ++j) used |= ids[j] == int(i);
            float candidate = scores[i] + bias[i];
            if (!used && (best < 0 || candidate > value)) { best = int(i); value = candidate; }
        }
        ids[k] = best;
    }
    for (uint k = 1; k < a.topk; ++k) {
        int v = ids[k]; uint j = k;
        while (j && ids[j - 1] > v) { ids[j] = ids[j - 1]; --j; }
        ids[j] = v;
    }
    float denom = 1.0f;
    if (a.topk > 1) {
        denom = 0;
        for (uint k = 0; k < a.topk; ++k) denom += scores[ids[k]];
        denom += 1e-20f;
    }
    for (uint k = 0; k < a.topk; ++k) {
        weights[k] = scores[ids[k]] / denom * a.scale;
        if (!finite_bits(weights[k])) { *status = -6; return; }
    }
    *status = 1;
}

kernel void kernel_deepseek_v41_weight(
    constant uint &n [[buffer(0)]], constant uint &width [[buffer(1)]],
    device const float *x [[buffer(2)]], device const float *weights [[buffer(3)]],
    device float *out [[buffer(4)]], device const int *status [[buffer(5)]],
    constant uint &topk [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = status[i / (width * topk)] == 1 ? x[i] * weights[i / width] : 0.0f;
}

kernel void kernel_deepseek_v41_sum(
    constant uint &hidden [[buffer(0)]], constant uint &topk [[buffer(1)]],
    device const float *slots [[buffer(2)]], device const float *shared [[buffer(3)]],
    device float *out [[buffer(4)]], device const int *status [[buffer(5)]],
    constant uint &rows [[buffer(6)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= hidden * rows) return;
    uint row = i / hidden, channel = i % hidden;
    float sum = 0;
    if (status[row] == 1) for (uint k = 0; k < topk; ++k) sum += slots[(row * topk + k) * hidden + channel];
    out[i] = sum + shared[i];
}
