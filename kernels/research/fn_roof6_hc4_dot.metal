#include <metal_stdlib>
using namespace metal;
struct fn_hc4_args {uint k,n,m;};
template<uint ROWS>
inline void fn_hc4_dot(constant fn_hc4_args&a,device const float*w,device const float*x,device float*y,uint group,ushort lane,ushort sg) {
 const uint row0=(group*4+sg)*ROWS;
 float4 acc[ROWS];for(uint r=0;r<ROWS;++r)acc[r]=0.f;
 for(uint j=lane;j<a.k;j+=32) {
  const float4 v=float4(w[j],w[a.k+j],w[2*a.k+j],w[3*a.k+j]);
  for(uint r=0;r<ROWS;++r)acc[r]=fma(float4(x[ulong(row0+r)*a.k+j]),v,acc[r]);
 }
 for(uint r=0;r<ROWS;++r){float4 v=acc[r];v=float4(simd_sum(v.x),simd_sum(v.y),simd_sum(v.z),simd_sum(v.w));if(lane==0)*reinterpret_cast<device float4*>(y+ulong(row0+r)*4)=v;}
}
#define FN_HC4(R) \
[[max_total_threads_per_threadgroup(128)]] kernel void kernel_fn_roof6_hc4_r##R(constant fn_hc4_args&a [[buffer(0)]],device const float*w [[buffer(1)]],device const float*x [[buffer(2)]],device float*y [[buffer(3)]],uint g [[threadgroup_position_in_grid]],ushort lane [[thread_index_in_simdgroup]],ushort sg [[simdgroup_index_in_threadgroup]]){fn_hc4_dot<R>(a,w,x,y,g,lane,sg);}
FN_HC4(1)
FN_HC4(2)
FN_HC4(4)
