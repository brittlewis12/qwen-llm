#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;

// Query-group key staging follows this tree's glm_index_reuse.metal.
// Arithmetic is copied from the incumbent V4.1 index_scores_tensor:
// identical 32x64x128 Half tensor multiply and ascending 32-head F32 sum.
// Four independent key ranges preserve occupancy. Only key staging is shared. The query mirror is an exact Half cast of the
// same F32 operands the incumbent narrows on load. No upstream code copied.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_deepseek_v41_attention_index_scores_reuse_serial(
    constant uint4 &a [[buffer(0)]], constant uint4 &b [[buffer(1)]],
    constant uint4 &c [[buffer(2)]], device const half *q [[buffer(3)]],
    device const float *weights [[buffer(4)]], device const float *keys [[buffer(5)]],
    device float *scores [[buffer(6)]],
    uint2 group [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup half qblk[32 * 128];
    threadgroup half kblk[64 * 128];
    threadgroup float sblk[64 * 32];
    const uint first = group.y * c.z;
    if (first >= c.y) return;
    const uint last = min(first + c.z, c.y);
    const uint maximum = (a.x + last) / a.y;
    if (maximum <= a.z) return;
    constexpr auto desc = tensor_ops::matmul2d_descriptor(
        32, 64, 128, false, true, false,
        tensor_ops::matmul2d_descriptor::mode::multiply);
    tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    using q_t = tensor<threadgroup half, extents<int, 128, 32>, tensor_inline>;
    using k_t = tensor<threadgroup half, extents<int, 128, 64>, tensor_inline>;
    using s_t = tensor<threadgroup float, extents<int, 64, 32>, tensor_inline>;
    q_t qt(qblk, extents<int,128,32>(),array<int,2>{1,128});
    k_t kt(kblk, extents<int,128,64>());
    s_t st(sblk, extents<int,64,32>());
    auto acc = op.get_destination_cooperative_tensor<q_t,k_t,float>();
    const float scale = rsqrt(float(b.x)) * rsqrt(float(b.y));
    const uint panels = (maximum + 63) / 64;
    const uint begin = (panels * group.x / 4) * 64;
    const uint end = (panels * (group.x + 1) / 4) * 64;
    for (uint k0 = begin; k0 < end; k0 += 64) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e=tid;e<64*128/4;e+=128) {
            const uint key=e/(128/4), c4=e%(128/4), t=k0+key;
            float4 v=float4(0.0f);
            if(t<maximum)v=((device const float4*)(keys+(ulong)t*b.x))[c4];
            ((threadgroup half4*)kblk)[e]=half4(v);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint local=first;local<last;++local) {
            const uint length=(a.x+local+1)/a.y;
            if(length<=a.z || k0>=length)continue;
            device const half *qrow=q+(ulong)local*b.z;
            device const float *wrow=weights+(ulong)local*b.w;
            device float *srow=scores+(ulong)local*c.x;
            for(uint e=tid;e<32*128/4;e+=128) {
                const uint h=e/(128/4), c4=e%(128/4);
                ((threadgroup half4*)qblk)[e]=((device const half4*)(qrow+h*128))[c4];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            op.run(qt,kt,acc);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            acc.store(st);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint key=tid;key<64;key+=128) {
                const uint t=k0+key;
                if(t>=length)break;
                float score=0.0f;
                for(uint h=0;h<32;++h) {
                    const float dot=sblk[key+h*64];
                    score+=max(dot,0.0f)*(wrow[h]*scale);
                }
                srow[t]=score;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
