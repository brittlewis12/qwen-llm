// Original BF16 weights and incumbent BF16 activation values, F32 sums.
// Full device tensors let the primitive choose its K traversal and staging.
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct hc_device_args { uint k; uint n; uint m; };

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_hc_bf16_device(
        constant hc_device_args &a [[buffer(0)]],
        device const bfloat *w [[buffer(1)]],
        device const bfloat *x [[buffer(2)]],
        device float *y [[buffer(3)]],
        uint2 group [[threadgroup_position_in_grid]]) {
    const uint m0=group.x*128u, n0=group.y*64u;
    constexpr auto d=tensor_ops::matmul2d_descriptor(
        128,64,static_cast<int>(dynamic_extent),false,true,false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<d,execution_simdgroups<4>> mm;
    // The SDK takes unqualified element types; these operands are only read.
    auto xt=tensor((device bfloat*)x+ulong(m0)*a.k,dextents<int,2>(a.k,128),array<int,2>({1,int(a.k)}));
    auto wt=tensor((device bfloat*)w+ulong(n0)*a.k,dextents<int,2>(a.k,min(64u,a.n-n0)),array<int,2>({1,int(a.k)}));
    auto acc=mm.template get_destination_cooperative_tensor<decltype(xt),decltype(wt),float>();
    mm.run(xt,wt,acc);
    auto dst=tensor(y+ulong(m0)*a.n+n0,dextents<int,2>(min(64u,a.n-n0),128),array<int,2>({1,int(a.n)}));
    acc.store(dst);
}
