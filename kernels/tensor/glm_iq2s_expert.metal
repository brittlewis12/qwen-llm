#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#include "../mat_mat_iq2_s.metal"

using namespace metal;
using namespace mpp;

struct glm_iq2s_tensor_args {
    uint hidden;
    uint ffn;
    uint rows;
    uint padded_rows;
    uint padded_ffn;
    uint expert;
    uint expert_stride_bytes;
    float clamp_value;
};

inline half glm_iq2s_scalar(device const uchar * block, uint index) {
    const half d_h = *((device const half *)block);
    device const uchar * qbase = block + 2;
    device const uchar * qh = qbase + IQ2S_QK / 4;
    device const uchar * scales = qh + IQ2S_QK / 32;
    const uint il = index / 16u;
    const uint lane = index % 16u;
    const uint ib32 = il >> 1u;
    const uint ih = il & 1u;
    device const uchar * qs = qbase + 4u * ib32 + 2u * ih;
    device const uchar * signs = qs + IQ2S_QK / 8;
    const uint qh_lane = uint(qh[ib32]) >> (4u * ih);
    const uint scale = (uint(scales[ib32]) >> (4u * ih)) & 0x0fu;
    const float dl = float(d_h) * (0.5f + float(scale)) * 0.25f;
    const uint pair = lane >> 3u;
    const uint i = lane & 7u;
    const uint grid_index = uint(qs[pair]) | ((qh_lane << (pair == 0u ? 8u : 6u)) & 0x300u);
    constant uchar * grid = (constant uchar *)(iq2s_grid_mm + grid_index);
    const float sign = (uint(signs[pair]) & uint(kmask_iq2s_mm[i])) != 0u ? -1.0f : 1.0f;
    return half(dl * float(grid[i]) * sign);
}

kernel void kernel_tensor_glm_iq2s_expert_partial(
        constant glm_iq2s_tensor_args & args [[buffer(0)]],
        device const uchar * gate [[buffer(1)]],
        device const uchar * up [[buffer(2)]],
        device const float * input [[buffer(3)]],
        device float * partial [[buffer(4)]],
        threadgroup half * staged [[threadgroup(0)]],
        uint tid [[thread_index_in_threadgroup]],
        uint3 group [[threadgroup_position_in_grid]]) {
    constexpr auto descriptor = tensor_ops::matmul2d_descriptor(16, 16, 32, false, true);
    tensor_ops::matmul2d<descriptor, execution_simdgroups<1>> operation;
    threadgroup half * a = staged;
    threadgroup half * b = staged + 512;
    const uint row0 = group.y * 16u;
    const uint col0 = group.x * 16u;
    const uint k0 = group.z * 32u;
    const uint tile_columns = args.padded_ffn / 16u;
    const ulong tile_index = ulong(group.y) * tile_columns + group.x;
    const ulong plane = ulong(args.padded_rows) * args.padded_ffn;
    const ulong tile_output_offset = tile_index * 256u;
    const uint row_bytes = (args.hidden / IQ2S_QK) * IQ2S_BYTES;
    device const uchar * gate_tile = gate + ulong(args.expert) * args.expert_stride_bytes;
    device const uchar * up_tile = up + ulong(args.expert) * args.expert_stride_bytes;
    for (uint i = tid; i < 512u; i += 32u) {
        const uint n = i / 32u;
        const uint k = i % 32u;
        const uint output_channel = col0 + n;
        const uint reduction = k0 + k;
        const uint block_index = reduction / IQ2S_QK;
        const uint block_offset = reduction % IQ2S_QK;
        const device uchar * gate_block = output_channel < args.ffn
            ? gate_tile + ulong(output_channel) * row_bytes + ulong(block_index) * IQ2S_BYTES
            : gate_tile;
        a[i] = output_channel < args.ffn ? glm_iq2s_scalar(gate_block, block_offset) : half(0.0h);
        b[i] = row0 + n < args.rows ? half(input[ulong(row0 + n) * args.hidden + reduction]) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> right(
        b, extents<int, 32, 16>());
    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> left_gate(
        a, extents<int, 32, 16>());
    tensor<device float, extents<int, 16, 16>, tensor_inline> out_gate(
        partial + ulong(group.z) * plane + tile_output_offset,
        extents<int, 16, 16>());
    auto gate_destination = operation.get_destination_cooperative_tensor<
        decltype(left_gate), decltype(right), float>();
    operation.run(left_gate, right, gate_destination);
    gate_destination.store(out_gate);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < 512u; i += 32u) {
        const uint n = i / 32u;
        const uint k = i % 32u;
        const uint output_channel = col0 + n;
        const uint reduction = k0 + k;
        const uint block_index = reduction / IQ2S_QK;
        const uint block_offset = reduction % IQ2S_QK;
        const device uchar * up_block = output_channel < args.ffn
            ? up_tile + ulong(output_channel) * row_bytes + ulong(block_index) * IQ2S_BYTES
            : up_tile;
        a[i] = output_channel < args.ffn ? glm_iq2s_scalar(up_block, block_offset) : half(0.0h);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tensor<threadgroup half, extents<int, 32, 16>, tensor_inline> left_up(
        a, extents<int, 32, 16>());
    tensor<device float, extents<int, 16, 16>, tensor_inline> out_up(
        partial + ulong(args.hidden / 32u + group.z) * plane + tile_output_offset,
        extents<int, 16, 16>());
    auto up_destination = operation.get_destination_cooperative_tensor<
        decltype(left_up), decltype(right), float>();
    operation.run(left_up, right, up_destination);
    up_destination.store(out_up);
}

kernel void kernel_tensor_glm_iq2s_expert_finish(
        constant glm_iq2s_tensor_args & args [[buffer(0)]],
        device const float * partial [[buffer(1)]],
        device float * output [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.rows * args.ffn;
    if (tid >= total) return;
    const uint row = tid / args.ffn;
    const uint channel = tid % args.ffn;
    const uint tiles = args.hidden / 32u;
    const uint tile_columns = args.padded_ffn / 16u;
    const uint tile_row = row / 16u;
    const uint tile_column = channel / 16u;
    const uint local_row = row % 16u;
    const uint local_column = channel % 16u;
    const ulong tile_index = ulong(tile_row) * tile_columns + tile_column;
    const ulong plane = ulong(args.padded_rows) * args.padded_ffn;
    const ulong local_offset = tile_index * 256u + local_column * 16u + local_row;
    float gate_total = 0.0f;
    float up_total = 0.0f;
    for (uint k = 0; k < tiles; ++k) {
        const ulong tile_offset = ulong(k) * plane + local_offset;
        gate_total += partial[tile_offset];
        up_total += partial[ulong(tiles + k) * plane + local_offset];
    }
    const float g = min(gate_total, args.clamp_value);
    const float u = clamp(up_total, -args.clamp_value, args.clamp_value);
    output[tid] = (g / (1.0f + exp(-g))) * u;
}
