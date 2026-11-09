#include <metal_stdlib>
using namespace metal;

// Opt-in singleton HC: four branches, hidden 2560, rank 320, native Q8_0.
// Keep the existing activated low vector; do not repeat its division or SiLU.
kernel void kernel_qwen4exp_hc_up_mix_q8_k320(
        device const uchar * weight [[buffer(0)]],
        device const float * low [[buffer(1)]],
        device const float * normalized [[buffer(2)]],
        device float * raw_gate [[buffer(3)]],
        device float * mixed [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint hidden = group * 4 + sg;
    const uint branch = lane / 8;
    const uint sublane = lane % 8;
    const uint row = branch * 2560 + hidden;
    float acc = 0.0f;
    for (uint block = 0; block < 10; ++block) {
        device const uchar * bytes = weight + (ulong)row * 340 + block * 34;
        const float scale = float(*(device const half *)bytes);
        device const char * quants = (device const char *)(bytes + 2);
        for (uint i = sublane; i < 32; i += 8) {
            acc += (float(quants[i]) * scale) * low[block * 32 + i];
        }
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

// BF16 twin of kernel_qwen4exp_hc_up_mix_q8_k320 for artifacts that keep HC
// at BF16 (GSQ-RCO): same lane split (8 lanes per branch), reduction and
// gated-mean epilogue; 16-byte loads of 4 BF16 weights and 4 activations.
inline void fn_hc_up_mix_bf16(device const ushort4 * weight,
        device const float4 * low, device const float * normalized,
        device float * raw_gate, device float * mixed,
        uint group, ushort sg, ushort lane) {
    const uint hidden = group * 4 + sg;
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
kernel void kernel_qwen4exp_hc_up_mix_bf16_k320(
        device const ushort4 * weight [[buffer(0)]],
        device const float4 * low [[buffer(1)]],
        device const float * normalized [[buffer(2)]],
        device float * raw_gate [[buffer(3)]],
        device float * mixed [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    fn_hc_up_mix_bf16(weight, low, normalized, raw_gate, mixed, group, sg, lane);
}

// Independent sequence axis; identical per-row arithmetic and lane reduction.
kernel void kernel_qwen4exp_hc_up_mix_bf16_batch(
        device const ushort4 * weight [[buffer(0)]],
        device const float4 * low0 [[buffer(1)]],
        device const float4 * low1 [[buffer(2)]],
        device const float4 * low2 [[buffer(3)]],
        device const float4 * low3 [[buffer(4)]],
        device const float * normalized0 [[buffer(5)]],
        device const float * normalized1 [[buffer(6)]],
        device const float * normalized2 [[buffer(7)]],
        device const float * normalized3 [[buffer(8)]],
        device float * raw0 [[buffer(9)]],
        device float * raw1 [[buffer(10)]],
        device float * raw2 [[buffer(11)]],
        device float * raw3 [[buffer(12)]],
        device float * mixed0 [[buffer(13)]],
        device float * mixed1 [[buffer(14)]],
        device float * mixed2 [[buffer(15)]],
        device float * mixed3 [[buffer(16)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint seq = group.y;
    fn_hc_up_mix_bf16(weight, seq == 0 ? low0 : seq == 1 ? low1 : seq == 2 ? low2 : low3, seq == 0 ? normalized0 : seq == 1 ? normalized1 : seq == 2 ? normalized2 : normalized3, seq == 0 ? raw0 : seq == 1 ? raw1 : seq == 2 ? raw2 : raw3, seq == 0 ? mixed0 : seq == 1 ? mixed1 : seq == 2 ? mixed2 : mixed3, group.x, sg, lane);
}
