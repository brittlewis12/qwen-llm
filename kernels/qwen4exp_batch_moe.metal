// Independent decode transport only. Native expert grids do the arithmetic.
#include <metal_stdlib>
using namespace metal;
struct fn_batch_moe_args {
    uint hidden;
    uint top_k;
    uint rows;
    uint pad;
};
kernel void
kernel_fn_batch_moe_gather(constant fn_batch_moe_args &a [[buffer(0)]], device const float *x0 [[buffer(1)]],
                           device const float *x1 [[buffer(2)]], device const float *x2 [[buffer(3)]],
                           device const float *x3 [[buffer(4)]], device const int *i0 [[buffer(5)]],
                           device const int *i1 [[buffer(6)]], device const int *i2 [[buffer(7)]],
                           device const int *i3 [[buffer(8)]], device const float *w0 [[buffer(9)]],
                           device const float *w1 [[buffer(10)]], device const float *w2 [[buffer(11)]],
                           device const float *w3 [[buffer(12)]], device const int *s0 [[buffer(13)]],
                           device const int *s1 [[buffer(14)]], device const int *s2 [[buffer(15)]],
                           device const int *s3 [[buffer(16)]], device float *x [[buffer(17)]],
                           device int *ids [[buffer(18)]], device float *weights [[buffer(19)]],
                           device int *status [[buffer(20)]], uint2 pos [[thread_position_in_grid]]) {
    const uint r = pos.y, c = pos.x;
    if (r >= a.rows)
        return;
    device const float *src = r == 0 ? x0 : r == 1 ? x1 : r == 2 ? x2 : x3;
    device const int *si = r == 0 ? i0 : r == 1 ? i1 : r == 2 ? i2 : i3;
    device const float *sw = r == 0 ? w0 : r == 1 ? w1 : r == 2 ? w2 : w3;
    device const int *ss = r == 0 ? s0 : r == 1 ? s1 : r == 2 ? s2 : s3;
    const int ready = ss[0];
    if (c < a.hidden)
        x[ulong(r) * a.hidden + c] = ready == 1 ? src[c] : 0.0f;
    if (c < a.top_k) {
        ids[ulong(r) * a.top_k + c] = ready == 1 ? si[c] : -1;
        weights[ulong(r) * a.top_k + c] = ready == 1 ? sw[c] : 0.0f;
    }
    if (c == 0)
        status[r] = ready;
}
kernel void
kernel_fn_batch_moe_scatter(constant fn_batch_moe_args &a [[buffer(0)]],
                            device const float *routed [[buffer(1)]], device const float *x0 [[buffer(2)]],
                            device const float *x1 [[buffer(3)]], device const float *x2 [[buffer(4)]],
                            device const float *x3 [[buffer(5)]], device const float *s0 [[buffer(6)]],
                            device const float *s1 [[buffer(7)]], device const float *s2 [[buffer(8)]],
                            device const float *s3 [[buffer(9)]], device float *y0 [[buffer(10)]],
                            device float *y1 [[buffer(11)]], device float *y2 [[buffer(12)]],
                            device float *y3 [[buffer(13)]], uint2 pos [[thread_position_in_grid]]) {
    const uint r = pos.y, c = pos.x;
    if (r >= a.rows || c >= a.hidden)
        return;
    device const float *x = r == 0 ? x0 : r == 1 ? x1 : r == 2 ? x2 : x3;
    device const float *scale = r == 0 ? s0 : r == 1 ? s1 : r == 2 ? s2 : s3;
    device float *y = r == 0 ? y0 : r == 1 ? y1 : r == 2 ? y2 : y3;
    y[c] = routed[ulong(r) * a.hidden + c] + scale[0] * x[c];
}
