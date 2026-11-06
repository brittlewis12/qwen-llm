// Flash-Next BF16 HC down + SiLU, optionally preparing F32 injection early.
// Original BF16 mat-vec row arithmetic and 1024-lane injection sum order.
#include <metal_stdlib>
using namespace metal;

inline float fn_hc_dot8(uint4 u, float4 x0, float4 x1) {
    const float4 lo = float4(as_type<float>(u.x << 16), as_type<float>(u.x & 0xffff0000u),
                             as_type<float>(u.y << 16), as_type<float>(u.y & 0xffff0000u));
    const float4 hi = float4(as_type<float>(u.z << 16), as_type<float>(u.z & 0xffff0000u),
                             as_type<float>(u.w << 16), as_type<float>(u.w & 0xffff0000u));
    return dot(lo, x0) + dot(hi, x1);
}
inline void fn_hc_down_silu(
        device const uint4 * weights, device const float4 * normalized,
        device float * low, threadgroup float * partial,
        uint row, ushort tid, ushort sg, ushort lane) {
    device const uint4 * w = weights + ulong(row) * 1280;
    float s0 = 0.0f, s1 = 0.0f;
    for (uint c = tid; c < 1280; c += 256) {
        const uint d = c + 128;
        s0 += fn_hc_dot8(w[c], normalized[2*c], normalized[2*c+1]);
        s1 += fn_hc_dot8(w[d], normalized[2*d], normalized[2*d+1]);
    }
    const float sum = simd_sum(s0 + s1);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        volatile float raw = (partial[0] + partial[1]) + (partial[2] + partial[3]);
        const float value = raw * 0.25f;
        low[row] = value / (1.0f + exp(-value));
    }
}
kernel void kernel_qwen4exp_hc_down_silu_bf16(
        device const uint4 * weights [[buffer(0)]],
        device const float4 * normalized [[buffer(1)]],
        device float * low [[buffer(2)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[4];
    fn_hc_down_silu(weights, normalized, low, partial, row, tid, sg, lane);
}
kernel void kernel_qwen4exp_hc_down_silu_inject_bf16(
        device const uint4 * weights [[buffer(0)]],
        device const float4 * normalized [[buffer(1)]],
        device float * low [[buffer(2)]],
        device const float * inject [[buffer(3)]],
        device float * injection [[buffer(4)]],
        uint row [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float partial[32];
    if (row < 320) {
        fn_hc_down_silu(weights, normalized, low, partial, row, tid, sg, lane);
        return;
    }
    device const float * w = inject + (row - 320) * 10240;
    device const float * x = (device const float *)normalized;
    float sums[8] = {0.0f};
#pragma clang loop unroll(disable)
    for (uint i = 0; i < 10; ++i) {
#pragma clang loop unroll(full)
        for (uint v = 0; v < 8; ++v) {
            const uint k = uint(tid) + v * 128 + i * 1024;
            sums[v] += w[k] * x[k];
        }
    }
#pragma clang loop unroll(full)
    for (uint v = 0; v < 8; ++v) {
        const float sum = simd_sum(sums[v]);
        if (lane == 0) partial[v * 4 + sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        const float sum = simd_sum(partial[lane]);
        if (lane == 0) injection[row - 320] = sum;
    }
}
