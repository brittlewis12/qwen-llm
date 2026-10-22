// Raw Metal 4 tensor-ops ceiling probe (research-only, like
// kernels/research/mma_peak_probe.metal). Each threadgroup runs ITERS
// matmul2d steps on threadgroup-resident half operands (left [M x K],
// right [N x K] read transposed, the layout the tensor expert tile uses)
// into a float cooperative tensor. No device traffic inside the loop; the
// result is reduced and written once so the compiler keeps the work.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp;

template <int M, int N, int K, int SG>
kernel void kernel_tensor_peak(
        constant uint & iters [[buffer(0)]],
        device float * out [[buffer(1)]],
        uint tg [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half sa[M * K];
    threadgroup half sb[N * K];
    for (uint i = tid; i < M * K; i += SG * 32) sa[i] = half(float(i % 13) * 0.01f);
    for (uint i = tid; i < N * K; i += SG * 32) sb[i] = half(float(i % 7) * 0.02f);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        M, N, K, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<desc, execution_simdgroups<SG>> op;
    using left_t = tensor<threadgroup half, extents<int, K, M>, tensor_inline>;
    using right_t = tensor<threadgroup half, extents<int, K, N>, tensor_inline>;
    left_t left(sa, extents<int, K, M>());
    right_t right(sb, extents<int, K, N>());
    auto acc = op.template get_destination_cooperative_tensor<left_t, right_t, float>();
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        acc[i] = 0.0f;
    }
    for (uint it = 0; it < iters; ++it) {
        op.run(left, right, acc);
    }

    float s = 0.0f;
#pragma clang loop unroll(full)
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) s += acc[i];
    }
    s = simd_sum(s);
    threadgroup float red[SG];
    if (tid % 32 == 0) red[tid / 32] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float t = 0.0f;
        for (short i = 0; i < SG; ++i) t += red[i];
        out[tg] = t;
    }
}

#define TT_PEAK(M, N, K, SG) \
    template [[host_name("kernel_tensor_peak_m" #M "_n" #N "_k" #K "_sg" #SG)]] kernel void kernel_tensor_peak<M, N, K, SG>(constant uint &, device float *, uint, uint);

TT_PEAK(32, 64, 32, 4)
TT_PEAK(64, 32, 32, 4)
TT_PEAK(64, 64, 32, 4)
TT_PEAK(32, 64, 64, 4)
TT_PEAK(64, 64, 64, 4)
TT_PEAK(128, 64, 32, 4)
TT_PEAK(32, 32, 32, 1)
TT_PEAK(16, 64, 32, 4)
