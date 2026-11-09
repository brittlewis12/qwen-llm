#include <metal_stdlib>
using namespace metal;

struct fn_hc_inject_dot4_args { uint k; uint n; uint m; };

// Complete four-query tiles, K10240/N4, qualified by the host. Split K across
// lanes so query and weight reads are coalesced, reusing each query for all
// four outputs. Each SIMD group owns one row; the reduction stays in F32.
template<bool TAIL>
inline void hc_dot4(constant fn_hc_inject_dot4_args&a,device const float*w,
    device const float*x,device float4*y,uint group,ushort lane,ushort sg) {
    const uint row = group * 4 + sg;
    if (TAIL && row >= a.m) return; // Uniform within a SIMD group; no group barriers.
    float4 acc = 0.f;
    for (uint j = lane; j < a.k; j += 32) {
        const float4 weights = float4(w[j], w[a.k + j], w[2 * a.k + j], w[3 * a.k + j]);
        acc = fma(float4(x[ulong(row) * a.k + j]), weights, acc);
    }
    const float4 sum = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    if (lane == 0) y[row] = sum;
}
#define FN_DOT4(NAME,TAIL) \
[[max_total_threads_per_threadgroup(128)]] \
kernel void NAME(constant fn_hc_inject_dot4_args&a [[buffer(0)]],device const float*w [[buffer(1)]], \
device const float*x [[buffer(2)]],device float4*y [[buffer(3)]],uint g [[threadgroup_position_in_grid]], \
ushort l [[thread_index_in_simdgroup]],ushort s [[simdgroup_index_in_threadgroup]]) {hc_dot4<TAIL>(a,w,x,y,g,l,s);}
FN_DOT4(kernel_qwen4exp_hc_inject_dot4_f32,false)
FN_DOT4(kernel_qwen4exp_hc_inject_dot4_f32_tail,true)
#undef FN_DOT4
