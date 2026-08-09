#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant float mxfp4_mm_values[16] = {
    0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f,
    0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f,
};

static inline float mxfp4_mm_e8m0_scale(uchar e) {
    const uint bits = e == 0u ? 0x00200000u
        : (e == 1u ? 0x00400000u : (uint(e) - 1u) << 23u);
    return as_type<float>(bits);
}

struct mat_mat_mxfp4_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_mat_mat_mxfp4_f32_mm64x32(
        constant mat_mat_mxfp4_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device float * output [[buffer(3)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort simdgroup_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint tile_rows = 64u;
    constexpr uint tile_columns = 32u;
    constexpr uint tile_k = 32u;
    constexpr uint block_bytes = 17u;
    constexpr uint weight_elements = tile_rows * tile_k;

    const uint output_base = group.y * tile_rows;
    const uint column_base = group.x * tile_columns;
    if (output_base >= args.M || column_base >= args.N) {
        return;
    }

    const uint tid = uint(tid_u);
    const uint lane = uint(lane_u);
    const uint simdgroup = uint(simdgroup_u);
    const short output_rows = short(min(tile_rows, args.M - output_base));
    const short columns = short(min(tile_columns, args.N - column_base));
    const short output_lane = short(tid_u / 2u);
    const short source_output_row = min(output_lane, short(output_rows - 1));
    const short column_lane = short(tid_u / 4u);
    const short source_column = min(column_lane, short(columns - 1));
    const uint input_k_offset = 8u * (tid & 3u);
    const uint packed_half = tid & 1u;

    device const uchar * weight_block = weights
        + (ulong)(output_base + uint(source_output_row)) * args.nb01;
    device const float * input_row = input
        + (ulong)(column_base + uint(source_column)) * args.stride_b
        + input_k_offset;
    threadgroup float * weight_tile = scratch;
    threadgroup float * activation_tile = scratch + weight_elements;
    simdgroup_float8x8 accumulators[8];
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_base = 0u; k_base < args.K; k_base += tile_k) {
        const float scale = mxfp4_mm_e8m0_scale(weight_block[0]);
        float decoded[16];
        FOR_UNROLL (short i = 0; i < 16; ++i) {
            const uchar packed = weight_block[1u + uint(i)];
            const uint index = packed_half == 0u
                ? uint(packed & 0x0fu)
                : uint(packed >> 4u);
            decoded[i] = mxfp4_mm_values[index] * scale;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short i = 0; i < 16; ++i) {
            const short sx = short(2u * packed_half) + i / 8;
            const short sy = output_lane / 8;
            const short lx = output_lane % 8;
            const short ly = i % 8;
            const short block = 8 * sx + sy;
            weight_tile[64 * block + 8 * ly + lx] = decoded[i];
        }

        const short sx = short(tid_u % 4u);
        const short sy = column_lane / 8;
        const short ly = column_lane % 8;
        const short block = 4 * sx + sy;
        *(threadgroup float2x4 *)(activation_tile + 64 * block + 8 * ly) =
            *((device const float2x4 *)input_row);

        weight_block += block_bytes;
        input_row += tile_k;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * matrix_a =
            weight_tile + 4u * 64u * (simdgroup & 1u);
        threadgroup const float * matrix_b =
            activation_tile + 2u * 64u * (simdgroup >> 1u);
        FOR_UNROLL (short k_tile = 0; k_tile < 4; ++k_tile) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[2];
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(a[i], matrix_a + 64 * i, 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(b[i], matrix_b + 64 * i, 8);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(
                    accumulators[i], b[i / 4], a[i % 4], accumulators[i]);
            }
            matrix_a += 8u * 64u;
            matrix_b += 4u * 64u;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * tile_output = scratch;
    threadgroup float * simdgroup_output = tile_output
        + 32u * (simdgroup & 1u)
        + 16u * (simdgroup >> 1u) * tile_rows;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(
            accumulators[i],
            simdgroup_output + 8 * (i % 4) + 8 * tile_rows * (i / 4),
            tile_rows);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint column = simdgroup; column < uint(columns); column += 4u) {
        device float * destination = output
            + (ulong)(column_base + column) * args.M
            + output_base;
        threadgroup float * result = tile_output + column * tile_rows;
        for (uint row = lane; row < uint(output_rows); row += 32u) {
            destination[row] = result[row];
        }
    }
}
