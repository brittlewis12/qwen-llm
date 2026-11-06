// Qualified Flash-Next packed router: original F32 operands and F32 sums.
// No relaxed matmul, activation conversion, or additional device storage.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_router_f32_tensor(
        device const float *w [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device float *y [[buffer(2)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint m0 = group.x * 32u, n0 = group.y * 64u;
    threadgroup float ws[64 * 32];
    threadgroup float xs[32 * 32];
    constexpr auto d = tensor_ops::matmul2d_descriptor(
        32, 64, 32, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
    tensor_ops::matmul2d<d, execution_simdgroups<4>> mm;
    auto xt = tensor(xs, extents<int, 32, 32>());
    auto wt = tensor(ws, extents<int, 32, 64>());
    auto acc = mm.template get_destination_cooperative_tensor<decltype(xt), decltype(wt), float>();
    for (ushort i = 0; i < acc.get_capacity(); ++i) acc[i] = 0.0f;
    for (uint k0 = 0; k0 < 2560u; k0 += 32u) {
        for (uint cell = tid; cell < 64u * 32u; cell += 128u)
            ws[cell] = w[ulong(n0 + cell / 32u) * 2560u + k0 + cell % 32u];
        for (uint cell = tid; cell < 32u * 32u; cell += 128u)
            xs[cell] = x[ulong(m0 + cell / 32u) * 2560u + k0 + cell % 32u];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm.run(xt, wt, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    auto dst = tensor(y + ulong(m0) * 512u + n0,
                      dextents<int, 2>(64, 32), array<int, 2>({1, 512}));
    acc.store(dst);
}
