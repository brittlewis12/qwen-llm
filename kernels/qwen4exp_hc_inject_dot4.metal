#include <metal_stdlib>
using namespace metal;

struct fn_hc_inject_dot4_args { uint k; uint n; uint m; };

// Complete four-query tiles, K10240/N4, qualified by the host. Split K across
// lanes so query and weight reads are coalesced, reusing each query for all
// four outputs. Each SIMD group owns one row; the reduction stays in F32.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_qwen4exp_hc_inject_dot4_f32(
    constant fn_hc_inject_dot4_args &a [[buffer(0)]],
    device const float *w [[buffer(1)]],
    device const float *x [[buffer(2)]],
    device float4 *y [[buffer(3)]],
    uint group [[threadgroup_position_in_grid]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = group * 4 + sg;
    float4 acc = 0.f;
    for (uint j = lane; j < a.k; j += 32) {
        const float4 weights = float4(w[j], w[a.k + j], w[2 * a.k + j], w[3 * a.k + j]);
        acc = fma(float4(x[ulong(row) * a.k + j]), weights, acc);
    }
    const float4 sum = float4(simd_sum(acc.x), simd_sum(acc.y), simd_sum(acc.z), simd_sum(acc.w));
    if (lane == 0) y[row] = sum;
}
