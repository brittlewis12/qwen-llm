#include <metal_stdlib>
using namespace metal;
struct fn_f32_mma_args { uint k; uint n; uint m; };

template<uint BK,uint SG>
inline void fn_f32_mma(constant fn_f32_mma_args &a,device const float*w,device const float*x,device float*y,threadgroup float*panel,uint2 group,uint tid,ushort sg) {
 const uint m0=group.x*(SG*8),n0=group.y*8;
 simdgroup_float8x8 acc=make_filled_simdgroup_matrix<float,8>(0.f);
 for(uint k0=0;k0<a.k;k0+=BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint i=tid;i<8*BK;i+=SG*32) {
   const uint row=i/BK,col=i%BK;
   panel[i]=n0+row<a.n?w[ulong(n0+row)*a.k+k0+col]:0.f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint ki=0;ki<BK;ki+=8) {
   simdgroup_float8x8 q,b;
   simdgroup_load(q,x+ulong(m0+sg*8)*a.k+k0+ki,a.k);
   simdgroup_load(b,panel+ki,BK,ulong2(0,0),true);
   simdgroup_multiply_accumulate(acc,q,b,acc);
  }
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 simdgroup_store(acc,panel+sg*64,8);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint i=tid;i<SG*64;i+=SG*32) {
  const uint row=i/8,col=i%8;
  if(n0+col<a.n)y[ulong(m0+row)*a.n+n0+col]=panel[i];
 }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_small_f32_mma_k128_s4(constant fn_f32_mma_args&a [[buffer(0)]],device const float*w [[buffer(1)]],device const float*x [[buffer(2)]],device float*y [[buffer(3)]],threadgroup float*p [[threadgroup(0)]],uint2 g [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],ushort sg [[simdgroup_index_in_threadgroup]]) {fn_f32_mma<128,4>(a,w,x,y,p,g,tid,sg);}
