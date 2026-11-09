// Scalar FFN reduction, shared accumulation and hyper-connection injection.
// Each independent session binds its own outputs, routing weights and residual.
#include <metal_stdlib>
using namespace metal;
struct fn_ffn_epilogue_args { uint hidden; uint branches; uint topk; };
kernel void kernel_qwen4exp_ffn_epilogue_f32(
    constant fn_ffn_epilogue_args &a [[buffer(0)]],
    device const float *experts [[buffer(1)]],
    device const float *weights [[buffer(2)]],
    device const float *shared [[buffer(3)]],
    device const float *scale [[buffer(4)]],
    device const float *raw_inject [[buffer(5)]],
    device float *residual [[buffer(6)]],
    uint2 j [[thread_position_in_grid]]) {
    if (j.x >= a.hidden || j.y >= a.branches) return;
    float sum=0.f;
    for (uint e=0; e<a.topk; ++e) sum += weights[e]*experts[ulong(e)*a.hidden+j.x];
    sum += scale[0]*shared[j.x];
    const float injection=2.f/(1.f+exp(-raw_inject[j.y]/float(a.branches)));
    residual[ulong(j.y)*a.hidden+j.x] += sum*injection;
}
