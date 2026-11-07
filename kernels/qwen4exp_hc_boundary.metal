#include <metal_stdlib>
using namespace metal;
struct fn_hc_boundary_args { uint branches; uint hidden; float eps; };
// Preserve the incumbent 1024-lane normalization order after updating each
// branch's residual. The input coefficient was prepared by the HC read.
kernel void kernel_qwen4exp_hc_inject_norm_f32(
 constant fn_hc_boundary_args &a [[buffer(0)]],
 device const float * block [[buffer(1)]],
 device const float * raw [[buffer(2)]],
 device float * residual [[buffer(3)]],
 device const float * weight [[buffer(4)]],
 device float * normalized [[buffer(5)]],
 threadgroup float * partial [[threadgroup(0)]],
 uint branch [[threadgroup_position_in_grid]],
 uint tid [[thread_position_in_threadgroup]],
 ushort sg [[simdgroup_index_in_threadgroup]],
 ushort lane [[thread_index_in_simdgroup]],
 uint threads [[threads_per_threadgroup]]) {
 if(branch>=a.branches)return;
 const ulong base=(ulong)branch*a.hidden;
 const float injection=2.0f/(1.0f+exp(-raw[branch]/float(a.branches)));
 float ss=0.f;
 for(uint h=tid;h<a.hidden;h+=threads) {
  const float v=residual[base+h]+block[h]*injection;
  residual[base+h]=v;
  ss+=v*v;
 }
 ss=simd_sum(ss);
 if(lane==0)partial[sg]=ss;
 threadgroup_barrier(mem_flags::mem_threadgroup);
 ss=lane<(threads+31)/32?partial[lane]:0.f;
 ss=simd_sum(ss);
 const float scale=rsqrt(ss/float(a.hidden)+a.eps);
 for(uint h=tid;h<a.hidden;h+=threads)normalized[base+h]=residual[base+h]*scale*weight[base+h];
}
