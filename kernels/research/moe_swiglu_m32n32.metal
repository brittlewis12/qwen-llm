// Experimental Half gate/up M32/N32/K32: halve the incumbent M64/N32
// per-SIMD-group accumulator state, without widening the token tile. Each
// SIMD group owns 16 rows x 16 tokens (four gate plus four up accumulators).
// The native dequantizer, activation half conversion, per-output K/MMA order,
// and clamped SwiGLU expression match moe_grouped_n32.metal. Equality is
// qualification-tested, not assumed. 128 threads, 8 KiB dynamic shared memory;
// research-library only, selected by a cfg(test) scoped host override.
#include <metal_stdlib>
using namespace metal;
#include "../quant_tiles.h"

constant constexpr short M32N32_NR0 = 32; // outputs per tile
constant constexpr short M32N32_NR1 = 32; // tokens per tile
constant constexpr short M32N32_NK = 32;  // K per step
constant constexpr short M32N32_NL0 = 2;  // 16-element dequant calls per row per step
constant constexpr short M32N32_NL1 = 4;  // 8-wide activation chunks per token per step

// Layout of moe.metal's moe_group_q4k_args.
struct moe_group_m32n32_args {
    uint ffn;
    uint hidden;
    uint n_expert;
    uint topk;
    uint n_tokens;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
};

// moe.metal's moe_silu_f.
inline float m32n32_silu(float x) {
    return x / (1.0f + exp(-x));
}

template <int BYTES, short NL, void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_moe_swiglu_clamped_grouped_slots_m32n32(
        constant moe_group_m32n32_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        constant     float & limit         [[buffer(7)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const ushort tiitg = ushort(tiitg_wide);
    const ushort weight_thread = tiitg % 64;
    threadgroup half * sa_g = (threadgroup half *)(shmem);         // 32 x 32
    threadgroup half * sa_u = (threadgroup half *)(shmem + 2048);  // 32 x 32
    threadgroup half * sb   = (threadgroup half *)(shmem + 4096);  // 32 x 32

    const int im = tgpig.z;
    const int r0 = tgpig.y * M32N32_NR0;
    const int r1 = tgpig.x * M32N32_NR1;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < M32N32_NR0) ? (short)((int)args.ffn - r0) : M32N32_NR0;
    const short nr1 = (count - r1 < M32N32_NR1) ? (short)(count - r1) : M32N32_NR1;

    const short lr0 = ((short)weight_thread / M32N32_NL0) < nr0
                        ? ((short)weight_thread / M32N32_NL0)
                        : nr0 - 1;
    const short il0 = (tiitg % M32N32_NL0);
    short il = il0;

    const short lr1 = ((short)tiitg / M32N32_NL1) < nr1
                        ? ((short)tiitg / M32N32_NL1)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % M32N32_NL1);

    const short offset1 = il0 / NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * BYTES;
    const int slot_limit = int(args.n_tokens * args.topk);
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const bool slot_valid = slot_id >= 0 && slot_id < slot_limit;
    const int token = (slot_valid ? slot_id : 0) / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[2];
    simdgroup_half8x8 ma_u[2];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += M32N32_NK) {
        // All SIMD groups finish consuming the previous input tile before reuse.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < 64) {
            half4x4 temp_a;
            DEQ(x_ptr_g, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / M32N32_NL0) / 8;
                const short lx = (tiitg / M32N32_NL0) % 8;
                const short ly = i % 8;
                const short ib = 4 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        if (tiitg < 64) {
            half4x4 temp_a;
            DEQ(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / M32N32_NL0) / 8;
                const short lx = (tiitg / M32N32_NL0) % 8;
                const short ly = i % 8;
                const short ib = 4 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            // All 128 threads: token tiitg / 4, 8-wide K chunk tiitg % 4.
            const short sx = (tiitg % M32N32_NL1);
            const short sy = (tiitg / M32N32_NL1) / 8;
            const short ib = 4 * sx + sy;
            const short ly = (tiitg / M32N32_NL1) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + BYTES * ((2 + NL - 1) / NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + BYTES * ((2 + NL - 1) / NL)
                    : x_ptr_u;
        y_ptr += M32N32_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 2 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 2 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < M32N32_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 2], ma_g[i % 2], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 2], ma_u[i % 2], mc_u[i]);
            }
            lsma_g += 4 * 64;
            lsma_u += 4 * 64;
            lsmb   += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate and up results: 32 rows x 32 tokens F32 each (4 KiB apiece).
    // The 8 KiB output overlay also covers the 6 KiB input staging.
    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 16 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * M32N32_NR0;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 16 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * M32N32_NR0;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 2) + 8 * M32N32_NR0 * (i / 2), M32N32_NR0, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 2) + 8 * M32N32_NR0 * (i / 2), M32N32_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 16 * (sgitg & 1);
    const short n_off = 16 * (sgitg >> 1);
    const short lane = (short)tiitg & 31;
    const short m_local = lane & 15;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 16; ++c) {
        const int global_n = r1 + n_off + c;
        if (lane < 16 && m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * M32N32_NR0];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * M32N32_NR0];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            if (slot >= 0 && slot < slot_limit) {
                dst[global_m + (ulong)slot * args.ffn] =
                    m32n32_silu(min(g_val, limit)) * clamp(u_val, -limit, limit);
            }
        }
    }
}

typedef decltype(kernel_moe_swiglu_clamped_grouped_slots_m32n32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, qt_dequantize_iq2_s>) m32n32_swiglu_t;

template [[host_name("kernel_moe_swiglu_clamped_iq2_s_f32_grouped_slots_m32n32")]] kernel m32n32_swiglu_t kernel_moe_swiglu_clamped_grouped_slots_m32n32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, qt_dequantize_iq2_s>;
template [[host_name("kernel_moe_swiglu_clamped_iq3_s_f32_grouped_slots_m32n32")]] kernel m32n32_swiglu_t kernel_moe_swiglu_clamped_grouped_slots_m32n32<QT_IQ3_S_BYTES, QT_IQ3_S_NL, qt_dequantize_iq3_s>;
