// IQ3_S down, Flash M128/N16/K32 geometry with canonical half tile decode.
#include <metal_stdlib>
using namespace metal;
#include "quant_tiles.h"

// Same ABI as GenericMmArgs; nb01 is bytes, stride_b is F32 elements.
// Down only: b_div is always 1.
struct iq3_s_down_retile_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
    uint b_div;
    uint slot_limit;
};

kernel void kernel_moe_down_iq3_s_f32_grouped_slots_m128_n16_retile(
        constant iq3_s_down_retile_args & args           [[buffer(0)]],
        device const uchar * srcA       [[buffer(1)]],
        device const float              * srcB       [[buffer(2)]],
        device const int                * counts     [[buffer(3)]],
        device const int                * ids        [[buffer(4)]],
        device       float              * dst        [[buffer(5)]],
        threadgroup  uchar              * shmem      [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const ushort tiitg = ushort(tiitg_wide);
    const short TILE_M = 128;
    const short TILE_N = 16;
    const short TILE_K = 32;
    const short B_LOAD_THREADS = 64;
    const short NL1_MM = 4;

    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * TILE_M;
    const int r1 = tgpig.x * TILE_N;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count)
        || count > int(args.N) || r1 >= count || r0 >= int(args.M)) return;

    const short nr0 = ((int)args.M - r0 < TILE_M) ? (short)((int)args.M - r0) : TILE_M;
    const short nr1 = (count - r1 < TILE_N) ? (short)(count - r1) : TILE_N;

    const short lr0 = (short)tiitg < nr0 ? (short)tiitg : nr0 - 1;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im
                                                   + (ulong)args.nb01 * (r0 + lr0);

    const short b_thread = (short)(tiitg & (B_LOAD_THREADS - 1));
    const short lr1 = (b_thread / NL1_MM) < nr1 ? (b_thread / NL1_MM) : nr1 - 1;
    const short iy = 8 * (b_thread % NL1_MM);
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    const bool slot_valid = slot_id >= 0 && slot_id < int(args.slot_limit);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (slot_valid ? slot_id : 0) + (ulong)iy;
    short il = 0;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += TILE_K) {
        half4x4 temp_a0;
        half4x4 temp_a1;
        qt_dequantize_iq3_s(x_ptr, il, temp_a0);
        qt_dequantize_iq3_s(x_ptr, il + 1, temp_a1);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        const short sy = (short)tiitg / 8;
        const short lx = (short)tiitg % 8;
        for (short i = 0; i < 16; ++i) {
            const short sx = i / 8;
            const short ly = i % 8;
            const short ib = 16 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a0[i / 4][i % 4];
        }
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 + i / 8;
            const short ly = i % 8;
            const short ib = 16 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a1[i / 4][i % 4];
        }

        if ((short)tiitg < B_LOAD_THREADS) {
            const short sx = (short)tiitg % NL1_MM;
            const short sy_b = ((short)tiitg / NL1_MM) / 8;
            const short ly = ((short)tiitg / NL1_MM) % 8;
            const short ib = 2 * sx + sy_b;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        // Eight K32 steps consume one 256-element / 110-byte superblock.
        il += 2;
        if (il == QT_IQ3_S_NL) {
            il = 0;
            x_ptr += QT_IQ3_S_BYTES;
        }
        y_ptr += TILE_K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * sgitg;
        threadgroup const half * lsmb = sb;

        for (short ik = 0; ik < TILE_K / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 16 * 64;
            lsmb += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem) + 32 * sgitg;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * TILE_M * (i / 4),
                        TILE_M, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + 32 * sgitg + m_local;
    if (global_m < (int)args.M) {
        for (short c = 0; c < TILE_N; ++c) {
            const int global_n = r1 + c;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.N + global_n];
                if (slot < 0 || slot >= int(args.slot_limit)) continue;
                dst[global_m + (ulong)slot * args.M] =
                    temp_str[(8 * tile_i + mr) + c * TILE_M];
            }
        }
    }
}
