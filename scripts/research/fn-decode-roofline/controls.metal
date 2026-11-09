#include <metal_stdlib>
using namespace metal;
kernel void roof4_read(device const float4*x[[buffer(0)]],device float*y[[buffer(1)]],uint g[[threadgroup_position_in_grid]],uint t[[thread_index_in_threadgroup]],ushort l[[thread_index_in_simdgroup]],ushort s[[simdgroup_index_in_threadgroup]]){float4 z=0.0f;for(uint i=0;i<16;++i)z+=x[ulong(g)*4096+t+i*256];float v=simd_sum(z.x+z.y+z.z+z.w);if(l==0)y[g*8+s]=v;}
kernel void roof4_noop(device uint*y[[buffer(0)]],constant uint2&g[[buffer(1)]],uint3 p[[threadgroup_position_in_grid]],uint t[[thread_index_in_threadgroup]]){if(t==0){uint k=p.x+g.x*(p.y+g.y*p.z);y[k]+=1u;}}
template<int A> kernel void fma_peak(constant uint&n[[buffer(0)]],device const float4*x[[buffer(1)]],device float*y[[buffer(2)]],uint t[[thread_position_in_grid]]){float4 a[A];float4 b=0.999f+0.00001f*x[t];for(uint i=0;i<A;++i)a[i]=float4(float(i+1)*0.01f)+x[t]*0.001f;
#pragma clang loop unroll(disable)
for(uint j=0;j<n;++j){
#pragma clang loop unroll(full)
for(uint i=0;i<A;++i)a[i]=fma(a[i],b,float4(0.0001f+float(i)*0.00001f));}
float s=0;for(uint i=0;i<A;++i)s+=a[i].x+a[i].y+a[i].z+a[i].w;y[t]=s;}
template [[host_name("roof4_fma4")]] kernel void fma_peak<4>(constant uint&,device const float4*,device float*,uint);
template [[host_name("roof4_fma8")]] kernel void fma_peak<8>(constant uint&,device const float4*,device float*,uint);
template [[host_name("roof4_fma16")]] kernel void fma_peak<16>(constant uint&,device const float4*,device float*,uint);
struct subst_args {uint n;uint sizes[8];uint offsets[8];};
kernel void roof4_substitute(constant subst_args&a[[buffer(0)]],device const uint*s[[buffer(1)]],device uint*d0[[buffer(2)]],device uint*d1[[buffer(3)]],device uint*d2[[buffer(4)]],device uint*d3[[buffer(5)]],device uint*d4[[buffer(6)]],device uint*d5[[buffer(7)]],device uint*d6[[buffer(8)]],device uint*d7[[buffer(9)]],uint t[[thread_position_in_grid]]){uint q=t;for(uint i=0;i<a.n;++i){if(q<a.sizes[i]){uint v=s[a.offsets[i]+q];switch(i){case 0:d0[q]=v;break;case 1:d1[q]=v;break;case 2:d2[q]=v;break;case 3:d3[q]=v;break;case 4:d4[q]=v;break;case 5:d5[q]=v;break;case 6:d6[q]=v;break;case 7:d7[q]=v;break;}return;}q-=a.sizes[i];}}
