#include <metal_stdlib>
using namespace metal;

// Preserve per-element multiplication grouping; reduction order is SIMD-parallel.
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)
kernel void kernel_deepseek_v41_engram(
    constant uint &d [[buffer(0)]], constant float &eps [[buffer(1)]],
    device float *x [[buffer(2)]], device const float *projected [[buffer(3)]],
    device const float *q [[buffer(4)]], device const float *k [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    uint stream = group % 4, row = group / 4;
    device float *h = x + group * d;
    device const float *key = projected + (row * 5 + stream) * d;
    device const float *value = projected + (row * 5 + 4) * d;
    float hh = 0.0f, kk = 0.0f, dot = -0.0f;
    bool negative_zero = true;
    for (uint j = tid; j < d; j += 32) {
        float hj = h[j], kj = key[j];
        float qj = q[stream * d + j], gj = k[stream * d + j];
        hh += hj * hj;
        kk += kj * kj;
        float term = hj * (qj * gj) * kj;
        // Fast-math builds may discard zero signs. Rust Sum starts at -0.
        uint sign = (as_type<uint>(hj) ^ as_type<uint>(qj)
            ^ as_type<uint>(gj) ^ as_type<uint>(kj)) & 0x80000000u;
        negative_zero &= term == 0.0f && sign != 0;
        dot += term;
    }
    hh = simd_sum(hh);
    kk = simd_sum(kk);
    dot = simd_sum(dot);
    // Inactive lanes are neutral for this predicate, including d < 32.
    negative_zero = simd_all(negative_zero);
    uint sign = negative_zero ? 0x80000000u : (as_type<uint>(dot) & 0x80000000u);
    float rstd = (1.0f / precise::sqrt(hh / float(d) + eps))
        * (1.0f / precise::sqrt(kk / float(d) + eps));
    float normalized = dot * rstd * (1.0f / precise::sqrt(float(d)));
    float magnitude = precise::sqrt(max(abs(normalized), 1e-6f));
    float signed_root = as_type<float>((as_type<uint>(magnitude) & 0x7fffffffu) | sign);
    float gate = 1.0f / (1.0f + precise::exp(-signed_root));
    for (uint j = tid; j < d; j += 32) h[j] += gate * value[j];
}
