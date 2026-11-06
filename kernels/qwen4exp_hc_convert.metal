#include <metal_stdlib>
using namespace metal;

// Same conversion as the incumbent BF16-activation matmul's staging cast.
kernel void kernel_qwen4exp_hc_convert_bf16(
        constant uint &n [[buffer(0)]], device const float *x [[buffer(1)]],
        device bfloat *y [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i<n) y[i]=bfloat(x[i]);
}
