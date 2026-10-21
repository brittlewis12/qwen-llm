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
