// Research-only re-grid of the Flash-Next BF16 HC up-mix kernel.
// Same per-hidden arithmetic and lane split as
// kernel_qwen4exp_hc_up_mix_bf16_k320; only the number of hidden rows per
// threadgroup changes (ROWS = threads / 32). Outputs are bitwise.
#include <metal_stdlib>
using namespace metal;

template <uint ROWS>
inline void hc_up_regrid(
        device const ushort4 * weight,
        device const float4 * low,
        device const float * normalized,
        device float * raw_gate,
        device float * mixed,
        uint group, ushort sg, ushort lane) {
    const uint hidden = group * ROWS + sg;
    const uint branch = lane / 8;
    const uint sublane = lane % 8;
    const uint row = branch * 2560 + hidden;
    device const ushort4 * w = weight + (ulong)row * 80;
    float acc = 0.0f;
    for (uint i = sublane; i < 80; i += 8) {
        const ushort4 u = w[i];
        const float4 a = float4(as_type<float>(uint(u.x) << 16), as_type<float>(uint(u.y) << 16),
                                as_type<float>(uint(u.z) << 16), as_type<float>(uint(u.w) << 16));
        const float4 b = low[i];
        acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    acc += simd_shuffle_xor(acc, 1);
    acc += simd_shuffle_xor(acc, 2);
    acc += simd_shuffle_xor(acc, 4);
    if (sublane == 0) raw_gate[row] = acc;
    const float value = (1.0f / (1.0f + exp(-acc))) * normalized[row];
    float sum = 0.0f;
    for (ushort b = 0; b < 4; ++b) {
        sum += simd_broadcast(value, ushort(b * 8));
    }
    if (lane == 0) mixed[hidden] = sum / 4.0f;
}

#define HC_UP_REGRID(NAME, ROWS) \
kernel void NAME( \
        device const ushort4 * weight [[buffer(0)]], \
        device const float4 * low [[buffer(1)]], \
        device const float * normalized [[buffer(2)]], \
        device float * raw_gate [[buffer(3)]], \
        device float * mixed [[buffer(4)]], \
        uint group [[threadgroup_position_in_grid]], \
        ushort sg [[simdgroup_index_in_threadgroup]], \
        ushort lane [[thread_index_in_simdgroup]]) { \
    hc_up_regrid<ROWS>(weight, low, normalized, raw_gate, mixed, group, sg, lane); \
}
HC_UP_REGRID(kernel_hc_up_regrid_r2, 2)
HC_UP_REGRID(kernel_hc_up_regrid_r4, 4)
HC_UP_REGRID(kernel_hc_up_regrid_r8, 8)
HC_UP_REGRID(kernel_hc_up_regrid_r16, 16)
