#include <metal_stdlib>
using namespace metal;
struct cache_banks { device const uint4 *bank[32] [[id(0)]]; };
struct cache_args { uint count; uint total_blocks; uint vectors[32]; uint prefix[33]; };

// One group reads 64 KiB. Repeating the grid over a small working set keeps
// enough independent groups to separate capacity from small-grid occupancy.
kernel void cache_touch(device const cache_banks &banks [[buffer(0)]],
        constant cache_args &a [[buffer(1)]],device uint *sink [[buffer(2)]],
        uint group [[threadgroup_position_in_grid]],ushort tid [[thread_index_in_threadgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]],ushort lane [[thread_index_in_simdgroup]]) {
    const uint block=group%a.total_blocks;
    uint b=0;while(block>=a.prefix[b+1])++b;
    const uint base=(block-a.prefix[b])*4096;
    uint4 sum=0;
    for(uint j=0;j<16;++j){const uint v=base+uint(tid)+j*256;if(v<a.vectors[b])sum+=banks.bank[b][v];}
    const uint s=simd_sum(sum.x+sum.y+sum.z+sum.w);
    if(lane==0)sink[group*8+sg]=s;
}
kernel void cache_fill(constant uint &n [[buffer(0)]],device uint *x [[buffer(1)]],uint i [[thread_position_in_grid]]) {
    if(i<n)x[i]=1u;
}

kernel void roof4_noop(device uint*y[[buffer(0)]],constant uint2&g[[buffer(1)]],uint3 p[[threadgroup_position_in_grid]],uint t[[thread_index_in_threadgroup]]){if(t==0){uint k=p.x+g.x*(p.y+g.y*p.z);y[k]+=1u;}}
struct subst_args {uint n;uint sizes[8];uint offsets[8];};
kernel void roof4_substitute(constant subst_args&a[[buffer(0)]],device const uint*s[[buffer(1)]],device uint*d0[[buffer(2)]],device uint*d1[[buffer(3)]],device uint*d2[[buffer(4)]],device uint*d3[[buffer(5)]],device uint*d4[[buffer(6)]],device uint*d5[[buffer(7)]],device uint*d6[[buffer(8)]],device uint*d7[[buffer(9)]],uint t[[thread_position_in_grid]]){uint q=t;for(uint i=0;i<a.n;++i){if(q<a.sizes[i]){uint v=s[a.offsets[i]+q];switch(i){case 0:d0[q]=v;break;case 1:d1[q]=v;break;case 2:d2[q]=v;break;case 3:d3[q]=v;break;case 4:d4[q]=v;break;case 5:d5[q]=v;break;case 6:d6[q]=v;break;case 7:d7[q]=v;break;}return;}q-=a.sizes[i];}}
