#include <metal_stdlib>
using namespace metal;

struct vt_tiled_args {
    uint base_pos;
    uint n_rows;
    uint kv_dim;
    uint kv_stride;
    uint vt_stride;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void kernel_vt_transpose_u16_tiled(
        constant vt_tiled_args & args [[buffer(0)]],
        device const ushort * src [[buffer(1)]],
        device ushort * dst [[buffer(2)]],
        threadgroup ushort * tile [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    const uint col = tid % 32;
    const uint row = tid / 32;
    const uint component = group.x * 32 + col;
    for (uint j = 0; j < 32; j += 8) {
        const uint pos = group.y * 32 + row + j;
        if (component < args.kv_dim && pos < args.n_rows) {
            tile[(row + j) * 33 + col] =
                src[(ulong)(args.base_pos + pos) * args.kv_stride + component];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint pos = group.y * 32 + col;
    for (uint j = 0; j < 32; j += 8) {
        const uint output_component = group.x * 32 + row + j;
        if (output_component < args.kv_dim && pos < args.n_rows) {
            dst[(ulong)output_component * args.vt_stride + args.base_pos + pos] =
                tile[col * 33 + row + j];
        }
    }
}
