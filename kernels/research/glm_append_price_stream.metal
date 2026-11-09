#include <metal_stdlib>
using namespace metal;
// One architectural pass over each live expert bank, or a dense tensor.
kernel void kernel_glm_append_price_stream(constant uint4 &a [[buffer(0)]],
 device const uint *src [[buffer(1)]],device const int *counts [[buffer(2)]],
 device uint *sink [[buffer(3)]],uint2 g [[threadgroup_position_in_grid]],
 uint tid [[thread_index_in_threadgroup]]) {
 threadgroup uint sums[4];uint acc=0;
 if(!a.z||counts[g.y]>0){uint hi=min(a.x,(g.x+1)*16384);for(uint i=g.x*16384+tid;i<hi;i+=128)acc+=src[ulong(g.y)*a.x+i];}
 acc=simd_sum(acc);if((tid&31)==0)sums[tid/32]=acc;threadgroup_barrier(mem_flags::mem_threadgroup);
 if(tid==0)sink[g.y*a.y+g.x]=sums[0]+sums[1]+sums[2]+sums[3];
}
