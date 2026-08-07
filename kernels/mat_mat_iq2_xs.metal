// Exact mapped IQ2_XS matrix projection.
//
// One SIMD group computes 16 output rows by up to 16 expert-major routed rows
// with F32 matrix operands and accumulators. Promoted shapes retain the scalar
// IQ2_XS gate, up, and SwiGLU results bit for bit.

#include <metal_stdlib>
using namespace metal;

#define mv_iq2xs_grid mm_iq2xs_grid
#include "iq2_xs_grid.metalh"
#undef mv_iq2xs_grid

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr uint IQ2_XS_QK = 256;
constant constexpr uint IQ2_XS_BLOCK_BYTES = 74;
constant constexpr uint IQ2_XS_TILE_ROWS = 16;
constant constexpr uint IQ2_XS_TILE_ROUTES = 16;
constant constexpr uint IQ2_XS_TILE_K = 32;

constant uchar kmask_iq2_xs_mm[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

constant uchar ksigns_iq2_xs_mm[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

struct ds4_packed_grouped_iq2_projection_args {
    uint M;
    uint K;
    uint nb01;
    uint stride_b;
    uint n_expert;
    uint map_count;
    uint source_count;
    uint destination_count;
};

struct ds4_packed_iq2_expert_tile {
    uint expert;
    uint start;
    uint count;
};

static_assert(sizeof(ds4_packed_iq2_expert_tile) == 12);

inline void dequantize_iq2_xs_f32(device const uchar * block,
                                  short il,
                                  thread float4x4 & values) {
    const float d = float(((device const half *)block)[0]);
    const uint block32 = uint(il >> 1);
    const uint half32 = uint(il & 1);
    device const ushort * q2 = (device const ushort *)(block + 2)
        + 4u * block32;
    device const uchar * scales = block + 2 + IQ2_XS_QK / 4;
    const float dl = d
        * (0.5f + float((uint(scales[block32]) >> (4u * half32)) & 0x0fu))
        * 0.25f;

    uint packed = uint(q2[2u * half32]);
    constant uchar * grid =
        (constant uchar *)(mm_iq2xs_grid + (packed & 511u));
    uint signs = uint(ksigns_iq2_xs_mm[packed >> 9u]);
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        values[i / 4][i % 4] = dl * float(grid[i])
            * ((signs & uint(kmask_iq2_xs_mm[i])) != 0u ? -1.0f : 1.0f);
    }

    packed = uint(q2[2u * half32 + 1u]);
    grid = (constant uchar *)(mm_iq2xs_grid + (packed & 511u));
    signs = uint(ksigns_iq2_xs_mm[packed >> 9u]);
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        values[2 + i / 4][i % 4] = dl * float(grid[i])
            * ((signs & uint(kmask_iq2_xs_mm[i])) != 0u ? -1.0f : 1.0f);
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mma16(
        constant ds4_packed_grouped_iq2_projection_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * source_rows [[buffer(3)]],
        device const int * destination_slots [[buffer(4)]],
        constant ds4_packed_iq2_expert_tile * tiles [[buffer(5)]],
        device float * output [[buffer(6)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const ds4_packed_iq2_expert_tile tile = tiles[group.x];
    if (tile.expert >= args.n_expert || tile.count == 0u
            || tile.count > IQ2_XS_TILE_ROUTES
            || tile.start + tile.count > args.map_count) {
        return;
    }

    threadgroup uint * map_valid = (threadgroup uint *)scratch;
    if (lane == 0) {
        uint valid = 1u;
        for (uint row = 0; row < tile.count; ++row) {
            const int source = source_rows[tile.start + row];
            const int destination = destination_slots[tile.start + row];
            if (source < 0 || uint(source) >= args.source_count
                    || destination < 0
                    || uint(destination) >= args.destination_count) {
                valid = 0u;
            }
        }
        *map_valid = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*map_valid == 0u) {
        return;
    }

    threadgroup float * weight_tile = scratch;
    threadgroup float * activation_tile =
        scratch + IQ2_XS_TILE_ROWS * IQ2_XS_TILE_K;

    const uint output_base = group.y * IQ2_XS_TILE_ROWS;
    const uint output_rows = min(IQ2_XS_TILE_ROWS, args.M - output_base);
    const uint local_output_row = uint(lane) / 2u;
    short dequant_lane = short(uint(lane) & 1u);
    device const uchar * weight_block = weights
        + (ulong)tile.expert * (ulong)args.nb01 * args.M
        + (ulong)(output_base + min(local_output_row, output_rows - 1u))
            * args.nb01;

    simdgroup_float8x8 accumulators[2][2];
    FOR_UNROLL (short row = 0; row < 2; ++row) {
        FOR_UNROLL (short column = 0; column < 2; ++column) {
            accumulators[row][column] =
                make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    for (uint k_base = 0; k_base < args.K; k_base += IQ2_XS_TILE_K) {
        float4x4 dequantized;
        dequantize_iq2_xs_f32(weight_block, dequant_lane, dequantized);
        FOR_UNROLL (short element = 0; element < 16; ++element) {
            weight_tile[local_output_row * IQ2_XS_TILE_K
                        + (uint(lane) & 1u) * 16u + uint(element)] =
                dequantized[element / 4][element % 4];
        }

        for (uint element = uint(lane);
             element < IQ2_XS_TILE_ROUTES * IQ2_XS_TILE_K;
             element += 32u) {
            const uint route = element / IQ2_XS_TILE_K;
            const uint k = element % IQ2_XS_TILE_K;
            float value = 0.0f;
            if (route < tile.count) {
                const uint source = uint(source_rows[tile.start + route]);
                value = input[(ulong)source * args.stride_b + k_base + k];
            }
            activation_tile[k * IQ2_XS_TILE_ROUTES + route] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        FOR_UNROLL (short k_tile = 0; k_tile < 4; ++k_tile) {
            simdgroup_float8x8 weight_matrices[2];
            simdgroup_float8x8 activation_matrices[2];
            FOR_UNROLL (short row = 0; row < 2; ++row) {
                simdgroup_load(
                    weight_matrices[row],
                    weight_tile + uint(row) * 8u * IQ2_XS_TILE_K
                        + uint(k_tile) * 8u,
                    IQ2_XS_TILE_K);
            }
            FOR_UNROLL (short column = 0; column < 2; ++column) {
                simdgroup_load(
                    activation_matrices[column],
                    activation_tile + uint(k_tile) * 8u * IQ2_XS_TILE_ROUTES
                        + uint(column) * 8u,
                    IQ2_XS_TILE_ROUTES);
            }
            FOR_UNROLL (short row = 0; row < 2; ++row) {
                FOR_UNROLL (short column = 0; column < 2; ++column) {
                    simdgroup_multiply_accumulate(
                        accumulators[row][column],
                        weight_matrices[row],
                        activation_matrices[column],
                        accumulators[row][column]);
                }
            }
        }

        dequant_lane = (dequant_lane + 2 < 16)
            ? dequant_lane + 2
            : short(uint(dequant_lane) & 1u);
        if (dequant_lane < 2) {
            weight_block += IQ2_XS_BLOCK_BYTES;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    threadgroup float * result = scratch;
    FOR_UNROLL (short row = 0; row < 2; ++row) {
        FOR_UNROLL (short column = 0; column < 2; ++column) {
            simdgroup_store(
                accumulators[row][column],
                result + uint(row) * 8u * IQ2_XS_TILE_ROUTES
                    + uint(column) * 8u,
                IQ2_XS_TILE_ROUTES);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint element = uint(lane);
         element < output_rows * tile.count;
         element += 32u) {
        const uint row = element / tile.count;
        const uint route = element % tile.count;
        const uint destination = uint(destination_slots[tile.start + route]);
        output[(ulong)destination * args.M + output_base + row] =
            result[row * IQ2_XS_TILE_ROUTES + route];
    }
}

// The 64-output by 32-route work unit follows llama.cpp's indirect quantized
// matrix topology at revision 6a32c29a under MIT. See
// docs/THIRD-PARTY-NOTICES.md. IQ2_XS decoding, F32 matrix operands, stable
// route maps, and destination-slot ownership remain qwen-llm contracts.
[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_deepseek_v4_packed_grouped_mapped_iq2_xs_f32_mm64x32(
        constant ds4_packed_grouped_iq2_projection_args & args [[buffer(0)]],
        device const uchar * weights [[buffer(1)]],
        device const float * input [[buffer(2)]],
        device const int * source_rows [[buffer(3)]],
        device const int * destination_slots [[buffer(4)]],
        constant ds4_packed_iq2_expert_tile * tiles [[buffer(5)]],
        device float * output [[buffer(6)]],
        threadgroup float * scratch [[threadgroup(0)]],
        uint2 group [[threadgroup_position_in_grid]],
        ushort tid_u [[thread_index_in_threadgroup]],
        ushort lane_u [[thread_index_in_simdgroup]],
        ushort simdgroup_u [[simdgroup_index_in_threadgroup]]) {
    constexpr uint tile_rows = 64u;
    constexpr uint tile_routes = 32u;
    constexpr uint tile_k = 32u;
    constexpr uint weight_elements = tile_rows * tile_k;

    const uint tid = uint(tid_u);
    const uint lane = uint(lane_u);
    const uint simdgroup = uint(simdgroup_u);
    const ds4_packed_iq2_expert_tile tile = tiles[group.x];
    if (tile.expert >= args.n_expert || tile.count == 0u
            || tile.count > tile_routes
            || tile.start + tile.count > args.map_count) {
        return;
    }

    threadgroup uint * map_valid = (threadgroup uint *)scratch;
    if (tid == 0u) {
        uint valid = 1u;
        for (uint route = 0u; route < tile.count; ++route) {
            const int source = source_rows[tile.start + route];
            const int destination = destination_slots[tile.start + route];
            if (source < 0 || uint(source) >= args.source_count
                    || destination < 0
                    || uint(destination) >= args.destination_count) {
                valid = 0u;
            }
        }
        *map_valid = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*map_valid == 0u) {
        return;
    }

    const uint output_base = group.y * tile_rows;
    const short output_rows = short(min(tile_rows, args.M - output_base));
    const short output_lane = short(tid_u / 2u);
    const short source_output_row = min(output_lane, short(output_rows - 1));
    const short route_lane = short(tid_u / 4u);
    const short source_route = min(route_lane, short(tile.count - 1u));
    const short dequant_lane0 = short(tid_u & 1u);
    short dequant_lane = dequant_lane0;
    const uint input_k_offset = 8u * (tid & 3u);

    device const uchar * weight_block = weights
        + (ulong)tile.expert * (ulong)args.nb01 * args.M
        + (ulong)(output_base + uint(source_output_row)) * args.nb01;
    const uint source = uint(source_rows[tile.start + uint(source_route)]);
    device const float * input_row = input
        + (ulong)source * args.stride_b + input_k_offset;

    threadgroup float * weight_tile = scratch;
    threadgroup float * activation_tile = scratch + weight_elements;
    simdgroup_float8x8 accumulators[8];
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        accumulators[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k_base = 0u; k_base < args.K; k_base += tile_k) {
        float4x4 dequantized;
        dequantize_iq2_xs_f32(weight_block, dequant_lane, dequantized);

        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short i = 0; i < 16; ++i) {
            const short sx = 2 * dequant_lane0 + i / 8;
            const short sy = output_lane / 8;
            const short lx = output_lane % 8;
            const short ly = i % 8;
            const short block = 8 * sx + sy;
            weight_tile[64 * block + 8 * ly + lx] =
                dequantized[i / 4][i % 4];
        }

        const short sx = short(tid_u % 4u);
        const short sy = route_lane / 8;
        const short ly = route_lane % 8;
        const short block = 4 * sx + sy;
        *(threadgroup float2x4 *)(activation_tile + 64 * block + 8 * ly) =
            *((device const float2x4 *)input_row);

        dequant_lane = (dequant_lane + 2 < 16)
            ? dequant_lane + 2
            : short(uint(dequant_lane) & 1u);
        if (dequant_lane < 2) {
            weight_block += IQ2_XS_BLOCK_BYTES;
        }
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

    for (uint route = simdgroup; route < tile.count; route += 4u) {
        const uint destination = uint(destination_slots[tile.start + route]);
        device float * destination_row =
            output + (ulong)destination * args.M + output_base;
        threadgroup float * result_row = tile_output + route * tile_rows;
        device float4 * destination4 = (device float4 *)destination_row;
        threadgroup float4 * result4 = (threadgroup float4 *)result_row;
        for (uint row = lane; row < uint(output_rows) / 4u; row += 32u) {
            destination4[row] = result4[row];
        }
        for (uint row = 4u * (uint(output_rows) / 4u) + lane;
             row < uint(output_rows);
             row += 32u) {
            destination_row[row] = result_row[row];
        }
    }
}
