// Cooperative F32 mat-vec arithmetic shared across independent members.
#include <metal_stdlib>
using namespace metal;
#define FOR_UNROLL _Pragma("clang loop unroll(full)") for
struct mat_vec_f32_members_args { uint n_in,n_out,n_cols; };
// Arithmetic of kernel_mat_vec_f32_f32_lcpp_r2, with weight loads and
// both barriers shared by independent member accumulators.
template<ushort B>
kernel void mat_vec_f32_members(
 constant mat_vec_f32_members_args & a [[buffer(0)]],
 device const float * weight [[buffer(1)]], device const float * x [[buffer(2)]],
 device float * y [[buffer(3)]], uint group [[threadgroup_position_in_grid]],
 ushort lane [[thread_index_in_simdgroup]], ushort sg [[simdgroup_index_in_threadgroup]]) {
 threadgroup float red[B][2][32];
 const uint first=group*2;
 if(first>=a.n_out)return;
 float sums[B][2];
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row)sums[b][row]=0;
 for(uint i=uint(sg)*32+lane;i<a.n_in/4;i+=128) {
  float4 xv[B];FOR_UNROLL(ushort b=0;b<B;++b)xv[b]=((device const float4 *)(x+ulong(b)*a.n_in))[i];
  FOR_UNROLL(ushort row=0;row<2;++row)if(first+row<a.n_out){
   const float4 wv=((device const float4 *)(weight+ulong(first+row)*a.n_in))[i];
   FOR_UNROLL(ushort b=0;b<B;++b){
    const float4 v=xv[b];sums[b][row]+=wv.x*v.x+wv.y*v.y+wv.z*v.z+wv.w*v.w;
   }
  }
 }
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row){
  if(sg==0)red[b][row][lane]=0;
  sums[b][row]=simd_sum(sums[b][row]);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row)if(lane==0)red[b][row][sg]=sums[b][row];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 FOR_UNROLL(ushort b=0;b<B;++b)FOR_UNROLL(ushort row=0;row<2;++row)if(sg==0&&first+row<a.n_out){
  const float total=simd_sum(red[b][row][lane]);if(lane==0)y[ulong(b)*a.n_out+first+row]=total;
 }
}
typedef decltype(mat_vec_f32_members<2>) mat_vec_f32_members_t;
#define R(B) template [[host_name("kernel_mat_vec_f32_members_b" #B)]] kernel mat_vec_f32_members_t mat_vec_f32_members<B>;
R(2) R(3) R(4)
