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

// Same gate for few rows (decode): one 1024-thread threadgroup per stream
// instead of one simdgroup, so the 4*d loads are in flight at once (the
// 32-thread form walks a 128-deep load chain, ~170 us per decode call).
// Reductions add simdgroup partials in index order, from -0 like Rust Sum.
#define V41_ENGRAM_WIDE_THREADS 1024
kernel void kernel_deepseek_v41_engram_wide(
    constant uint &d [[buffer(0)]], constant float &eps [[buffer(1)]],
    device float *x [[buffer(2)]], device const float *projected [[buffer(3)]],
    device const float *q [[buffer(4)]], device const float *k [[buffer(5)]],
    uint group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[3][V41_ENGRAM_WIDE_THREADS / 32];
    threadgroup uint partial_zero[V41_ENGRAM_WIDE_THREADS / 32];
    uint stream = group % 4, row = group / 4;
    device float *h = x + group * d;
    device const float *key = projected + (row * 5 + stream) * d;
    device const float *value = projected + (row * 5 + 4) * d;
    float hh = 0.0f, kk = 0.0f, dot = -0.0f;
    bool negative_zero = true;
    for (uint j = tid; j < d; j += V41_ENGRAM_WIDE_THREADS) {
        float hj = h[j], kj = key[j];
        float qj = q[stream * d + j], gj = k[stream * d + j];
        hh += hj * hj;
        kk += kj * kj;
        float term = hj * (qj * gj) * kj;
        uint sign = (as_type<uint>(hj) ^ as_type<uint>(qj)
            ^ as_type<uint>(gj) ^ as_type<uint>(kj)) & 0x80000000u;
        negative_zero &= term == 0.0f && sign != 0;
        dot += term;
    }
    hh = simd_sum(hh);
    kk = simd_sum(kk);
    dot = simd_sum(dot);
    negative_zero = simd_all(negative_zero);
    if (lane == 0) {
        partial[0][sg] = hh;
        partial[1][sg] = kk;
        partial[2][sg] = dot;
        partial_zero[sg] = negative_zero ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hh = 0.0f;
    kk = 0.0f;
    dot = -0.0f;
    negative_zero = true;
    for (uint s = 0; s < V41_ENGRAM_WIDE_THREADS / 32; ++s) {
        hh += partial[0][s];
        kk += partial[1][s];
        dot += partial[2][s];
        negative_zero &= partial_zero[s] != 0u;
    }
    uint sign = negative_zero ? 0x80000000u : (as_type<uint>(dot) & 0x80000000u);
    float rstd = (1.0f / precise::sqrt(hh / float(d) + eps))
        * (1.0f / precise::sqrt(kk / float(d) + eps));
    float normalized = dot * rstd * (1.0f / precise::sqrt(float(d)));
    float magnitude = precise::sqrt(max(abs(normalized), 1e-6f));
    float signed_root = as_type<float>((as_type<uint>(magnitude) & 0x7fffffffu) | sign);
    float gate = 1.0f / (1.0f + precise::exp(-signed_root));
    for (uint j = tid; j < d; j += V41_ENGRAM_WIDE_THREADS) h[j] += gate * value[j];
}
