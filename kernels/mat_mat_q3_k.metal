// Q3_K mat-mat (W * X^T -> Y^T), using the shared 64x32x32 simdgroup_matrix
// tile from mat_mat_mm_tile.h; this file owns only the Q3_K dequant.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include "mat_mat_mm_tile.h"

constant constexpr int Q3K_QK       = 256;
constant constexpr int Q3K_BYTES    = 110;
constant constexpr int Q3K_NL       = Q3K_QK / 16;

struct mat_mat_q3k_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

inline int q3_k_scale_int_bytes(device const uchar * scales, uint sub) {
    const uint scale_2 = uint(scales[sub & 7u]);
    const uint scale_1 = uint(scales[8u + (sub & 3u)]);
    const uint quarter = sub >> 2;
    const uint kmask1 = quarter > 1u ? (quarter > 2u ? 192u : 48u) : (quarter > 0u ? 12u : 3u);
    const uint kmask2 = sub >= 8u ? 0xf0u : 0x0fu;
    uint raw;
    if ((quarter & 1u) != 0u) {
        raw = (scale_2 & kmask2) | ((scale_1 & kmask1) << 2);
    } else {
        raw = (scale_2 & kmask2) | ((scale_1 & kmask1) << 4);
    }
    if (sub >= 8u) {
        raw >>= 4;
    }
    return int(raw) - 32;
}

inline void dequantize_q3_K_half(device const uchar * blk_bytes,
                                 short il,
                                 thread half4x4 & reg) {
    device const uchar * hmask = blk_bytes;
    device const uchar * qs = blk_bytes + 32;
    device const uchar * scales = blk_bytes + 96;
    const half d_h = *((device const half *)(blk_bytes + 108));

    const uint sub = uint(il);
    const uint q_offset = 32u * (sub >> 3) + 16u * (sub & 1u);
    const uint h_offset = 16u * (sub & 1u);
    const uint shift = ((sub >> 1) & 3u) * 2u;
    const uint h_bit = 1u << (sub >> 1);
    const float dl = float(d_h) * float(q3_k_scale_int_bytes(scales, sub));

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint lane = uint(i);
        const int q_low = int((uint(qs[q_offset + lane]) >> shift) & 3u);
        const int q = q_low - ((uint(hmask[h_offset + lane]) & h_bit) != 0u ? 0 : 4);
        reg[i / 4][i % 4] = (half)(dl * float(q));
    }
}

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_q3_K_f32_mm,
                       mat_mat_q3k_args,
                       Q3K_BYTES,
                       Q3K_NL,
                       dequantize_q3_K_half)

struct ds4_packed_grouped_q3k_args {
    uint M;
    uint K;
    uint nb01;
    uint stride_b;
    uint n_expert;
    uint map_count;
    uint source_count;
    uint destination_count;
};

struct ds4_packed_q3k_expert_tile {
    uint expert;
    uint start;
    uint count;
};

static_assert(sizeof(ds4_packed_q3k_expert_tile) == 12);

kernel void kernel_deepseek_v4_packed_grouped_mapped_q3_K_f32_mm(
        constant ds4_packed_grouped_q3k_args & args [[buffer(0)]],
        device const uchar * srcA [[buffer(1)]],
        device const float * srcB [[buffer(2)]],
        device const int * source_rows [[buffer(3)]],
        device const int * destination_slots [[buffer(4)]],
        constant ds4_packed_q3k_expert_tile * tiles [[buffer(5)]],
        device float * dst [[buffer(6)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const ds4_packed_q3k_expert_tile tile = tiles[tgpig.x];
    if (tile.expert >= args.n_expert || tile.count == 0u || tile.count > 32u
            || tile.start > args.map_count
            || tile.count > args.map_count - tile.start) return;

    threadgroup uint * map_valid = (threadgroup uint *)shmem;
    if (tiitg == 0) {
        uint valid = 1u;
        for (uint j = 0; j < tile.count; ++j) {
            const int source = source_rows[tile.start + j];
            const int destination = destination_slots[tile.start + j];
            if (source < 0 || uint(source) >= args.source_count
                    || destination < 0 || uint(destination) >= args.destination_count) {
                valid = 0u;
            }
        }
        *map_valid = valid;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (*map_valid == 0u) return;

    const int r0 = tgpig.y * 64;
    const int nr0 = min(64, (int)args.M - r0);
    const short nr1 = (short)tile.count;

    const short lr0 = ((short)tiitg / 2) < nr0
                        ? ((short)tiitg / 2)
                        : (short)nr0 - 1;
    const short il0 = tiitg % 2;
    short il = il0;

    const short lr1 = ((short)tiitg / 4) < nr1
                        ? ((short)tiitg / 4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % 4);
    const int input_row = source_rows[tile.start + uint(lr1)];

    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + (ulong)tile.expert * expert_stride
        + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * input_row
        + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += 32) {
        {
            half4x4 temp_a;
            dequantize_q3_K_half(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / 2) / 8;
                const short lx = (tiitg / 2) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }
        {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q3K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + Q3K_BYTES : x_ptr;
        y_ptr += 32;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        FOR_UNROLL (short ik = 0; ik < 4; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * 64;
    FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                        64, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        threadgroup float * tile_output = (threadgroup float *)shmem;
        for (int j = tiitg; j < nr1; j += 32) {
            const int output_slot = destination_slots[tile.start + uint(j)];
            device float * D = dst + (ulong)output_slot * args.M + r0;
            threadgroup float * C = tile_output + j * 64;
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}
