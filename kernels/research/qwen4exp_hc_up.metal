#include <metal_stdlib>
using namespace metal;

// Research-only singleton HC: four branches, hidden 2560, rank 320, native Q8_0.
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
