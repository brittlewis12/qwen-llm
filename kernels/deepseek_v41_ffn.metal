#include <metal_stdlib>
using namespace metal;

// Projection must precede RMS scaling; collapse uses a different, lagged pre.
kernel void kernel_deepseek_v41_projected_rms(
    constant uint &n [[buffer(0)]], constant float &eps [[buffer(1)]],
    device const float *x [[buffer(2)]], device float *mix [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]]) {
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
kernel void kernel_deepseek_v41_route(
    constant RouteArgs &a [[buffer(0)]], device const float *logits [[buffer(1)]],
    device const float *bias [[buffer(2)]], device float *scores [[buffer(3)]],
    device int *ids [[buffer(4)]], device float *weights [[buffer(5)]],
    device int *status [[buffer(6)]]) {
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
    uint i [[thread_position_in_grid]]) {
    if (i < n) out[i] = *status == 1 ? x[i] * weights[i / width] : 0.0f;
}

kernel void kernel_deepseek_v41_sum(
    constant uint &hidden [[buffer(0)]], constant uint &topk [[buffer(1)]],
    device const float *slots [[buffer(2)]], device const float *shared [[buffer(3)]],
    device float *out [[buffer(4)]], device const int *status [[buffer(5)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= hidden) return;
    float sum = 0;
    if (*status == 1) for (uint k = 0; k < topk; ++k) sum += slots[k * hidden + i];
    out[i] = sum + shared[i];
}
