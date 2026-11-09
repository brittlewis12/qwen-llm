// Exact independent decode projection transport. B=3's fourth gathered column
// is a duplicate of member zero for the incumbent NC4 projection, not a state.
#include <metal_stdlib>
using namespace metal;
struct fn_batch_copy_args {
    uint width;
    uint rows;
};
kernel void
kernel_fn_batch_gather_columns(constant fn_batch_copy_args &a [[buffer(0)]],
                               device const float *x0 [[buffer(1)]], device const float *x1 [[buffer(2)]],
                               device const float *x2 [[buffer(3)]], device const float *x3 [[buffer(4)]],
                               device float *matrix [[buffer(5)]], uint2 p [[thread_position_in_grid]]) {
    if (p.x >= a.width || p.y >= a.rows)
        return;
    device const float *x = p.y == 0 ? x0 : p.y == 1 ? x1 : p.y == 2 ? x2 : x3;
    matrix[ulong(p.y) * a.width + p.x] = x[p.x];
}
kernel void kernel_fn_batch_scatter_columns(constant fn_batch_copy_args &a [[buffer(0)]],
                                            device float *x0 [[buffer(1)]], device float *x1 [[buffer(2)]],
                                            device float *x2 [[buffer(3)]], device float *x3 [[buffer(4)]],
                                            device const float *matrix [[buffer(5)]],
                                            uint2 p [[thread_position_in_grid]]) {
    if (p.x >= a.width || p.y >= a.rows)
        return;
    device float *x = p.y == 0 ? x0 : p.y == 1 ? x1 : p.y == 2 ? x2 : x3;
    x[p.x] = matrix[ulong(p.y) * a.width + p.x];
}
