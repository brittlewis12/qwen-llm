#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

struct qsa_index_tensor_args { uint start_position; uint query_count; uint ratio; uint block_capacity; };

// One full 32-query band and 32 pooled keys. Original F32 queries, exact
// F16 -> F32 keys, strict F32 products/sums. No query narrowing or global scratch.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_qsa_index_tensor_f32(
        constant qsa_index_tensor_args &a [[buffer(0)]],
        device const float *queries [[buffer(1)]],device const half *keys [[buffer(2)]],
        device float *scores [[buffer(3)]],device int *visible_counts [[buffer(4)]],
        uint group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]]) {
#pragma METAL fp math_mode(safe)
    const uint maximum_visible=(a.start_position+a.query_count)/a.ratio;
    const uint b0=group*32u;
    if (group==0u && tid<a.query_count) {
        const uint visible=(a.start_position+uint(tid)+1u)/a.ratio;
        visible_counts[tid]=visible<=a.block_capacity ? int(visible) : -1;
    }
    threadgroup float key_tile[128*32];
    for (uint i=tid;i<128u*32u;i+=128u) {
        const uint block=b0+i/128u;
        key_tile[i]=block<maximum_visible && block<a.block_capacity ? float(keys[ulong(block)*128u+i%128u]) : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    constexpr auto d=tensor_ops::matmul2d_descriptor(32,32,128,false,true,false,tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<d,execution_simdgroups<4>> mm;
    auto x=tensor((device float*)queries,dextents<int,2>(128,32),array<int,2>({1,512}));
    auto w=tensor(key_tile,extents<int,128,32>());
    auto sum=mm.template get_destination_cooperative_tensor<decltype(x),decltype(w),float>();
    for (ushort i=0;i<sum.get_capacity();++i) sum[i]=0.0f;
    for (uint head=0;head<4u;++head) {
        auto q=tensor((device float*)queries+head*128u,dextents<int,2>(128,32),array<int,2>({1,512}));
        auto dot=mm.template get_destination_cooperative_tensor<decltype(x),decltype(w),float>();
        mm.run(q,w,dot);
        for (ushort i=0;i<sum.get_capacity();++i) sum[i]+=max(dot[i],0.0f);
    }
    for (ushort i=0;i<sum.get_capacity();++i) sum[i]*=0.08838834764831845f;
    // Future columns within this rectangle may be written. The unchanged
    // visible_counts contract excludes them from selection for each query.
    auto dst=tensor(scores+b0,dextents<int,2>(min(32u,maximum_visible-b0),32),array<int,2>({1,int(a.block_capacity)}));
    sum.store(dst);
}
