#include <metal_stdlib>
using namespace metal;
kernel void roof4_read(device const float4*x[[buffer(0)]],device float*y[[buffer(1)]],uint g[[threadgroup_position_in_grid]],uint t[[thread_index_in_threadgroup]],ushort l[[thread_index_in_simdgroup]],ushort s[[simdgroup_index_in_threadgroup]]){float4 z=0.0f;for(uint i=0;i<16;++i)z+=x[ulong(g)*4096+t+i*256];float v=simd_sum(z.x+z.y+z.z+z.w);if(l==0)y[g*8+s]=v;}
kernel void roof4_noop(device uint*y[[buffer(0)]],constant uint2&g[[buffer(1)]],uint3 p[[threadgroup_position_in_grid]],uint t[[thread_index_in_threadgroup]]){if(t==0){uint k=p.x+g.x*(p.y+g.y*p.z);y[k]+=1u;}}
struct subst_args {uint n;uint sizes[16];uint offsets[16];};
kernel void roof4_substitute(constant subst_args&a[[buffer(0)]],device const uint*s[[buffer(1)]],
device uint*d0[[buffer(2)]],
device uint*d1[[buffer(3)]],
device uint*d2[[buffer(4)]],
device uint*d3[[buffer(5)]],
device uint*d4[[buffer(6)]],
device uint*d5[[buffer(7)]],
device uint*d6[[buffer(8)]],
device uint*d7[[buffer(9)]],
device uint*d8[[buffer(10)]],
device uint*d9[[buffer(11)]],
device uint*d10[[buffer(12)]],
device uint*d11[[buffer(13)]],
device uint*d12[[buffer(14)]],
device uint*d13[[buffer(15)]],
device uint*d14[[buffer(16)]],
device uint*d15[[buffer(17)]],uint t[[thread_position_in_grid]]) {
 uint q=t;for(uint i=0;i<a.n;++i){if(q<a.sizes[i]){uint v=s[a.offsets[i]+q];switch(i){
case 0:d0[q]=v;break;
case 1:d1[q]=v;break;
case 2:d2[q]=v;break;
case 3:d3[q]=v;break;
case 4:d4[q]=v;break;
case 5:d5[q]=v;break;
case 6:d6[q]=v;break;
case 7:d7[q]=v;break;
case 8:d8[q]=v;break;
case 9:d9[q]=v;break;
case 10:d10[q]=v;break;
case 11:d11[q]=v;break;
case 12:d12[q]=v;break;
case 13:d13[q]=v;break;
case 14:d14[q]=v;break;
case 15:d15[q]=v;break;
}return;}q-=a.sizes[i];}}
