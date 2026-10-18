#include <metal_stdlib>
using namespace metal;

// Keep the baseline's left-to-right reductions and multiplication grouping.
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
    threadgroup float gate;
    if (tid == 0) {
        float hh = 0.0f, kk = 0.0f, dot = -0.0f;
        bool negative_zero = true;
        for (uint j = 0; j < d; ++j) {
            hh += h[j] * h[j];
            kk += key[j] * key[j];
            float term = h[j] * (q[stream * d + j] * k[stream * d + j]) * key[j];
            // Fast-math builds may discard zero signs. Rust Sum starts at -0.
            uint sign = (as_type<uint>(h[j]) ^ as_type<uint>(q[stream * d + j])
                ^ as_type<uint>(k[stream * d + j]) ^ as_type<uint>(key[j])) & 0x80000000u;
            negative_zero &= term == 0.0f && sign != 0;
            dot += term;
        }
        uint sign = negative_zero ? 0x80000000u : (as_type<uint>(dot) & 0x80000000u);
        float rstd = (1.0f / precise::sqrt(hh / float(d) + eps))
            * (1.0f / precise::sqrt(kk / float(d) + eps));
        float normalized = dot * rstd * (1.0f / precise::sqrt(float(d)));
        float magnitude = precise::sqrt(max(abs(normalized), 1e-6f));
        float signed_root = as_type<float>((as_type<uint>(magnitude) & 0x7fffffffu) | sign);
        gate = 1.0f / (1.0f + precise::exp(-signed_root));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint j = tid; j < d; j += 32) h[j] += gate * value[j];
}
