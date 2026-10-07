#include <metal_stdlib>
using namespace metal;

struct q2_0_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

inline float q2_0_dot(device const uchar * row, device const float * x,
                      uint n_in, ushort lane) {
    float sum = 0.0f;
    for (uint k = lane; k < n_in; k += 32) {
        device const uchar * block = row + (ulong)(k / 64) * 18;
        const uint j = k % 64;
        const int code = (block[2 + j / 4] >> (2 * (j % 4))) & 3;
        const float w = float(*((device const half *)block)) * float(code - 1);
        sum += w * x[k];
    }
    return simd_sum(sum);
}

kernel void kernel_mat_vec_q2_0_f32(
        constant q2_0_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const ulong row = (ulong)group.x * 4 + simd;
    if (row >= args.n_out) return;
    const ulong row_bytes = (ulong)(args.n_in / 64) * 18;
    const float sum = q2_0_dot(weight + row * row_bytes, x, args.n_in, lane);
    if (lane == 0) y[row] = sum;
}

kernel void kernel_moe_down_q2_0_f32(
        constant q2_0_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * inner [[buffer(2)]],
        device const int * ids [[buffer(3)]],
        device float * out [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort simd [[simdgroup_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const ulong row = (ulong)group.x * 4 + simd;
    const uint slot = group.y;
    if (row >= args.n_out || slot >= args.topk) return;
    const int expert = ids[slot];
    const ulong dst = (ulong)slot * args.n_out + row;
    if (expert < 0 || uint(expert) >= args.n_expert) {
        if (lane == 0) out[dst] = 0.0f;
        return;
    }
    const ulong row_bytes = (ulong)(args.n_in / 64) * 18;
    device const uchar * w = weight + ((ulong)expert * args.n_out + row) * row_bytes;
    const float sum = q2_0_dot(w, inner + (ulong)slot * args.n_in, args.n_in, lane);
    if (lane == 0) out[dst] = sum;
}
