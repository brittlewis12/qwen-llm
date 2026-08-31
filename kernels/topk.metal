#include <metal_stdlib>
using namespace metal;

struct mask_row_indices_args {
    uint row_width;
    uint index_count;
};

kernel void kernel_mask_row_indices_f32(
        constant mask_row_indices_args & args [[buffer(0)]],
        device       float * values           [[buffer(1)]],
        device const uint  * indices          [[buffer(2)]],
        uint gid [[thread_position_in_grid]]) {
    const uint row = gid / args.index_count;
    const uint index = indices[gid];
    if (index < args.row_width) {
        values[(ulong)row * args.row_width + index] = -INFINITY;
    }
}
