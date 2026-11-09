// Original BF16 weights and incumbent BF16 activation values, F32 sums.
// Full device tensors let the primitive choose its K traversal and staging.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct hc_device_args { uint k; uint n; uint m; };

template<bool TAIL>
inline void hc_bf16_device(constant hc_device_args &a, device const bfloat *w,
        device const bfloat *x, device float *y, uint2 group) {
    const uint m0=group.x*128u, n0=group.y*64u;
    constexpr auto d=tensor_ops::matmul2d_descriptor(
        128,64,static_cast<int>(dynamic_extent),false,true,false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<d,execution_simdgroups<4>> mm;
    // The SDK takes unqualified element types; these operands are only read.
    const uint live=TAIL?min(128u,a.m-m0):128u;
    auto xt=tensor((device bfloat*)x+ulong(m0)*a.k,dextents<int,2>(a.k,live),array<int,2>({1,int(a.k)}));
    auto wt=tensor((device bfloat*)w+ulong(n0)*a.k,dextents<int,2>(a.k,min(64u,a.n-n0)),array<int,2>({1,int(a.k)}));
    auto acc=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    mm.run(xt,wt,acc);
    auto dst=tensor(y+ulong(m0)*a.n+n0,dextents<int,2>(min(64u,a.n-n0),live),array<int,2>({1,int(a.n)}));
    acc.store(dst);
}
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_hc_bf16_device(constant hc_device_args &a [[buffer(0)]],
        device const bfloat*w [[buffer(1)]],device const bfloat*x [[buffer(2)]],
        device float*y [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]]) {
    hc_bf16_device<false>(a,w,x,y,g);
}

// Four local K partitions improve both short-grid occupancy and summation.
// The BF16 products/rounding are unchanged. No partials leave the threadgroup.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_hc_bf16_device_split4(constant hc_device_args&a [[buffer(0)]],
        device const bfloat*w [[buffer(1)]],device const bfloat*x [[buffer(2)]],
        device float*y [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]]) {
#pragma METAL fp math_mode(safe)
    const uint m0=group.x*32u,n0=group.y*64u,live=min(32u,a.m-m0),part=a.k/4u;
    constexpr auto d=tensor_ops::matmul2d_descriptor(32,64,static_cast<int>(dynamic_extent),
        false,true,false,tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<d,execution_simdgroups<4>> mm;
    auto xt=tensor((device bfloat*)x+ulong(m0)*a.k,dextents<int,2>(part,live),array<int,2>({1,int(a.k)}));
    auto wt=tensor((device bfloat*)w+ulong(n0)*a.k,dextents<int,2>(part,min(64u,a.n-n0)),array<int,2>({1,int(a.k)}));
    auto s0=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    auto s1=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    mm.run(xt,wt,s0);
    auto x1=tensor((device bfloat*)x+ulong(m0)*a.k+part,dextents<int,2>(part,live),array<int,2>({1,int(a.k)}));
    auto w1=tensor((device bfloat*)w+ulong(n0)*a.k+part,dextents<int,2>(part,min(64u,a.n-n0)),array<int,2>({1,int(a.k)}));
    mm.run(x1,w1,s1);
    for(ushort i=0;i<s0.get_capacity();++i)s0[i]=s0[i]+s1[i];
    auto s2=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    auto s3=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    auto x2=tensor((device bfloat*)x+ulong(m0)*a.k+2u*part,dextents<int,2>(part,live),array<int,2>({1,int(a.k)}));
    auto w2=tensor((device bfloat*)w+ulong(n0)*a.k+2u*part,dextents<int,2>(part,min(64u,a.n-n0)),array<int,2>({1,int(a.k)}));
    auto x3=tensor((device bfloat*)x+ulong(m0)*a.k+3u*part,dextents<int,2>(part,live),array<int,2>({1,int(a.k)}));
    auto w3=tensor((device bfloat*)w+ulong(n0)*a.k+3u*part,dextents<int,2>(part,min(64u,a.n-n0)),array<int,2>({1,int(a.k)}));
    mm.run(x2,w2,s2);mm.run(x3,w3,s3);
    for(ushort i=0;i<s0.get_capacity();++i)s0[i]=s0[i]+(s2[i]+s3[i]);
    auto dst=tensor(y+ulong(m0)*a.n+n0,dextents<int,2>(min(64u,a.n-n0),live),array<int,2>({1,int(a.n)}));
    s0.store(dst);
}
