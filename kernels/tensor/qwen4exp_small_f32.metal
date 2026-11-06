// Original F32 operands, F32 accumulation, no relaxed precision.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
struct small_f32_args { uint k; uint n; uint m; };

template <int N>
inline void small_f32(constant small_f32_args &a,
        device const float *w, device const float *x, device float *y, uint group) {
#pragma METAL fp math_mode(safe)
    const uint m0=group*128u;
    constexpr auto d=tensor_ops::matmul2d_descriptor(
        128,N,static_cast<int>(dynamic_extent),false,true,false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<d,execution_simdgroups<4>> mm;
    auto xt=tensor((device float*)x+ulong(m0)*a.k,dextents<int,2>(a.k,128),array<int,2>({1,int(a.k)}));
    auto wt=tensor((device float*)w,dextents<int,2>(a.k,a.n),array<int,2>({1,int(a.k)}));
    auto acc=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    mm.run(xt,wt,acc);
    auto dst=tensor(y+ulong(m0)*a.n,dextents<int,2>(a.n,128),array<int,2>({1,int(a.n)}));
    acc.store(dst);
}
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_small_f32_n4(constant small_f32_args &a [[buffer(0)]],
        device const float *w [[buffer(1)]],device const float *x [[buffer(2)]],
        device float *y [[buffer(3)]],uint group [[threadgroup_position_in_grid]]) {
    small_f32<8>(a,w,x,y,group);
}
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_small_f32_n48(constant small_f32_args &a [[buffer(0)]],
        device const float *w [[buffer(1)]],device const float *x [[buffer(2)]],
        device float *y [[buffer(3)]],uint group [[threadgroup_position_in_grid]]) {
    small_f32<64>(a,w,x,y,group);
}
