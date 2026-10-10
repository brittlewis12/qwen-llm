// F32-operand grouped-slot expert tiles (map #12 accuracy lane). Copies of
// moe.metal's kernel_moe_grouped_slots_mm_generic (EPI 0: the down
// projection) and moe_swiglu_grouped_slots_n16_body (clamped SwiGLU) with F32
// threadgroup tiles and simdgroup_float8x8 operands, and F32 dequantizers for
// the GLM-5.3 expert types: the arithmetic of quant_tiles.h's
// qt_dequantize_{iq2_s,iq3_s,iq4_xs} (ggml's order) without the half
// narrowing. The half-staged product kernels are untouched.
//
// Each output row's K reduction runs in a fixed order over its own weight row
// and its own slot's activation row; rows and slots past a tile's edge are
// clamped (never read past the bucket), and only in-range outputs of valid
// slots are stored. A token's outputs therefore depend neither on how many
// tokens share its expert's bucket nor on its place in the bucket.
#include <metal_stdlib>
using namespace metal;
#include "quant_tiles.h"

#define F32X_FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr short F32X_NR0 = 64;        // outputs per tile
constant constexpr short F32X_NK = 32;         // K per step
constant constexpr short F32X_NL0 = 2;         // 16-element dequant calls per row per step
constant constexpr short F32X_NL1 = 4;         // 8-wide activation chunks per slot per step
constant constexpr short F32X_NR1_DOWN = 32;   // slots per down tile
constant constexpr short F32X_NR1_SWIGLU = 16; // slots per gate/up tile

// ---------------------------------------------------------------------------
// F32 tile dequantizers (16 consecutive elements of sub-block il).

inline void f32x_dequantize_iq2_s(device const uchar * blk_bytes,
                                  short il,
                                  thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qbase = blk_bytes + 2;
    device const uchar * qh = qbase + QT_QK_K / 4;
    device const uchar * scales = qh + QT_QK_K / 32;

    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * qs = qbase + 4u * ib32 + 2u * ih;
    device const uchar * signs = qs + QT_QK_K / 8;

    const uint qh_lane = uint(qh[ib32]) >> (4u * ih);
    const uint scale = (uint(scales[ib32]) >> (4u * ih)) & 0x0fu;
    const float dl = float(d_h) * (0.5f + float(scale)) * 0.25f;

    constant uchar * grid1 = (constant uchar *)(qt_iq2s_grid +
        (uint(qs[0]) | ((qh_lane << 8u) & 0x300u)));
    constant uchar * grid2 = (constant uchar *)(qt_iq2s_grid +
        (uint(qs[1]) | ((qh_lane << 6u) & 0x300u)));

    F32X_FOR_UNROLL (int i = 0; i < 8; ++i) {
        const float s1 = ((uint(signs[0]) & uint(qt_kmask_iq2xs[i])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[1]) & uint(qt_kmask_iq2xs[i])) != 0u) ? -1.0f : 1.0f;
        reg[i / 4 + 0][i % 4] = dl * float(grid1[i]) * s1;
        reg[i / 4 + 2][i % 4] = dl * float(grid2[i]) * s2;
    }
}

inline void f32x_dequantize_iq3_s(device const uchar * blk_bytes,
                                  short il,
                                  thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs_base = blk_bytes + 2;
    device const uchar * qh = qs_base + QT_QK_K / 4;
    device const uchar * signs_base = qh + QT_QK_K / 32;
    device const uchar * scales = signs_base + QT_QK_K / 8;

    const uint ib32 = uint(il >> 1);
    const uint ih = uint(il & 1);
    device const uchar * qs = qs_base + 8u * ib32;
    device const uchar * signs = signs_base + 4u * ib32 + 2u * ih;
    const uint qh_lane = uint(qh[ib32]) >> (4u * ih);
    const uint scale = (uint(scales[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const float dl = float(d_h) * (1.0f + 2.0f * float(scale));

    constant uchar * grid1 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 0u]) | ((qh_lane << 8u) & 256u)));
    constant uchar * grid2 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 1u]) | ((qh_lane << 7u) & 256u)));
    F32X_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((uint(signs[0]) & uint(qt_kmask_iq2xs[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[0]) & uint(qt_kmask_iq2xs[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[0][i] = dl * float(grid1[i]) * s1;
        reg[1][i] = dl * float(grid2[i]) * s2;
    }

    grid1 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 2u]) | ((qh_lane << 6u) & 256u)));
    grid2 = (constant uchar *)(qt_iq3s_grid +
        (uint(qs[4u * ih + 3u]) | ((qh_lane << 5u) & 256u)));
    F32X_FOR_UNROLL (int i = 0; i < 4; ++i) {
        const float s1 = ((uint(signs[1]) & uint(qt_kmask_iq2xs[i + 0])) != 0u) ? -1.0f : 1.0f;
        const float s2 = ((uint(signs[1]) & uint(qt_kmask_iq2xs[i + 4])) != 0u) ? -1.0f : 1.0f;
        reg[2][i] = dl * float(grid1[i]) * s1;
        reg[3][i] = dl * float(grid2[i]) * s2;
    }
}

inline void f32x_dequantize_iq4_xs(device const uchar * blk_bytes,
                                   short il,
                                   thread float4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    const ushort scales_h = *((device const ushort *)(blk_bytes + 2));
    device const uchar * scales_l = blk_bytes + 4;
    device const uchar * qs = blk_bytes + 8;

    const uint ib32 = uint(il >> 1);
    const uint scale_l = (uint(scales_l[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const uint scale_h = (uint(scales_h) >> (2u * ib32)) & 3u;
    const float d = float(d_h) * float(int(scale_l | (scale_h << 4)) - 32);
    const bool hi = (il & 1) != 0;
    device const uchar * q = qs + ib32 * 16u;

    F32X_FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint idx = hi ? uint(q[i] >> 4) : uint(q[i] & 0x0f);
        reg[i / 4][i % 4] = d * qt_iq4nl_values[idx];
    }
}

inline float f32x_silu(float x) {
    return x / (1.0f + exp(-x));
}

// ---------------------------------------------------------------------------
// Down: dst[slot, :] = W_e * srcB[slot / b_div, :] for every slot of expert e.
// Same arguments, buffers and grid as kernel_moe_grouped_slots_mm_generic;
// 12 KiB of threadgroup memory.

struct f32x_moe_mm_args {
    uint M;          // output rows per slot
    uint N;          // ids row stride per expert (== n_tokens)
    uint K;          // inner dim
    uint nb01;       // weight row stride in bytes
    uint stride_b;   // activation row stride in f32 elements
    uint min_count;
    uint max_count;
    uint b_div;      // activation row = slot / b_div
    uint slot_limit; // number of valid slots (output rows)
};

template <int BYTES, short NL, void (*DEQ)(device const uchar *, short, thread float4x4 &)>
kernel void kernel_moe_down_grouped_slots_f32x(
        constant f32x_moe_mm_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  float * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup float * sa = shmem;                  // 64 rows x 32 K
    threadgroup float * sb = shmem + F32X_NR0 * F32X_NK; // 32 slots x 32 K

    const int im = tgpig.z;
    const int r0 = tgpig.y * F32X_NR0;
    const int r1 = tgpig.x * F32X_NR1_DOWN;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = (short)min((int)F32X_NR0, (int)args.M - r0);
    const short nr1 = (short)min((int)F32X_NR1_DOWN, count - r1);

    const short lr0 = min((short)(tiitg / F32X_NL0), (short)(nr0 - 1));
    const short il0 = tiitg % F32X_NL0;
    short il = il0;
    const short lr1 = min((short)(tiitg / F32X_NL1), (short)(nr1 - 1));
    const short iy = 8 * (tiitg % F32X_NL1);

    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (ulong)(r0 + lr0);
    const int slot_limit = int(args.slot_limit);
    const int slot_id = ids[(ulong)im * args.N + (ulong)(r1 + lr1)];
    const bool slot_valid = slot_id >= 0 && slot_id < slot_limit;
    const int b_row = (slot_valid ? slot_id : 0) / int(args.b_div);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (ulong)b_row + (ulong)iy;

    simdgroup_float8x8 ma[4];
    simdgroup_float8x8 mb[2];
    simdgroup_float8x8 mc[8];
    F32X_FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += F32X_NK) {
        float4x4 temp_a;
        DEQ(x_ptr, il, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        F32X_FOR_UNROLL (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / F32X_NL0) / 8;
            const short lx = (tiitg / F32X_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        {
            const short sx = tiitg % F32X_NL1;
            const short sy = (tiitg / F32X_NL1) / 8;
            const short ly = (tiitg / F32X_NL1) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup float2x4 *)(sb + 64 * ib + 8 * ly) =
                *((device const float2x4 *)y_ptr);
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        y_ptr += F32X_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb = sb + 2 * 64 * (sgitg / 2);
        F32X_FOR_UNROLL (short ik = 0; ik < F32X_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            F32X_FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            F32X_FOR_UNROLL (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            F32X_FOR_UNROLL (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    // Stage the 64 x 32 result (slot-major) over the weight tile, then store
    // the in-range outputs of valid slots.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = shmem + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * F32X_NR0;
    F32X_FOR_UNROLL (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * F32X_NR0 * (i / 4), F32X_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short j = (short)sgitg; j < nr1; j += 4) {
        const int slot = ids[(ulong)im * args.N + (ulong)(r1 + j)];
        if (slot < 0 || slot >= slot_limit) {
            continue;
        }
        device float * D = dst + (ulong)slot * args.M + (ulong)r0;
        threadgroup const float * C = shmem + j * F32X_NR0;
        for (short i = (short)tiisg; i < nr0; i += 32) {
            D[i] = C[i];
        }
    }
}

// ---------------------------------------------------------------------------
// Fused gate/up with the clamped SwiGLU epilogue (DeepSeek V4 / GLM-5.3):
//   dst[slot, :] = silu(min(g, limit)) * clamp(u, -limit, limit),
//   g = W_gate_e x[slot / topk], u = W_up_e x[slot / topk].
// Same arguments, buffers and grid as the clamped
// kernel_moe_swiglu_clamped_grouped_slots_n16_generic; 18 KiB of threadgroup
// memory.

struct f32x_moe_swiglu_args {
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

template <int BYTES, short NL, void (*DEQ)(device const uchar *, short, thread float4x4 &)>
kernel void kernel_moe_swiglu_clamped_grouped_slots_f32x(
        constant f32x_moe_swiglu_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        constant     float & limit         [[buffer(7)]],
        threadgroup  float * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup float * sa_g = shmem;                            // 64 rows x 32 K
    threadgroup float * sa_u = shmem + F32X_NR0 * F32X_NK;       // 64 rows x 32 K
    threadgroup float * sb   = shmem + 2 * F32X_NR0 * F32X_NK;   // 16 slots x 32 K

    const int im = tgpig.z;
    const int r0 = tgpig.y * F32X_NR0;
    const int r1 = tgpig.x * F32X_NR1_SWIGLU;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = (short)min((int)F32X_NR0, (int)args.ffn - r0);
    const short nr1 = (short)min((int)F32X_NR1_SWIGLU, count - r1);

    const short lr0 = min((short)(tiitg / F32X_NL0), (short)(nr0 - 1));
    const short il0 = tiitg % F32X_NL0;
    short il = il0;
    const short lr1 = min((short)(tiitg / F32X_NL1), (short)(nr1 - 1));
    const short iy = 8 * (tiitg % F32X_NL1);

    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (ulong)(r0 + lr0);
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (ulong)(r0 + lr0);
    const int slot_limit = int(args.n_tokens * args.topk);
    const int slot_id = ids[(ulong)im * args.n_tokens + (ulong)(r1 + lr1)];
    const bool slot_valid = slot_id >= 0 && slot_id < slot_limit;
    const int token = (slot_valid ? slot_id : 0) / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (ulong)token + (ulong)iy;

    simdgroup_float8x8 ma_g[4];
    simdgroup_float8x8 ma_u[4];
    simdgroup_float8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];
    F32X_FOR_UNROLL (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += F32X_NK) {
        float4x4 temp_g;
        float4x4 temp_u;
        DEQ(x_ptr_g, il, temp_g);
        DEQ(x_ptr_u, il, temp_u);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        F32X_FOR_UNROLL (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / F32X_NL0) / 8;
            const short lx = (tiitg / F32X_NL0) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa_g[64 * ib + 8 * ly + lx] = temp_g[i / 4][i % 4];
            sa_u[64 * ib + 8 * ly + lx] = temp_u[i / 4][i % 4];
        }
        if (tiitg < F32X_NR1_SWIGLU * F32X_NL1) {
            const short sx = tiitg % F32X_NL1;
            const short sy = (tiitg / F32X_NL1) / 8;
            const short ly = (tiitg / F32X_NL1) % 8;
            const short ib = 2 * sx + sy;
            *(threadgroup float2x4 *)(sb + 64 * ib + 8 * ly) =
                *((device const float2x4 *)y_ptr);
        }

        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2) ? x_ptr_g + BYTES * ((2 + NL - 1) / NL) : x_ptr_g;
        x_ptr_u = (il < 2) ? x_ptr_u + BYTES * ((2 + NL - 1) / NL) : x_ptr_u;
        y_ptr += F32X_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const float * lsma_g = sa_g + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsma_u = sa_u + 4 * 64 * (sgitg % 2);
        threadgroup const float * lsmb   = sb   + 1 * 64 * (sgitg / 2);
        F32X_FOR_UNROLL (short ik = 0; ik < F32X_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            F32X_FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            F32X_FOR_UNROLL (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    // Stage gate and up results (slot-major, 64 x 16 each) over the gate
    // tile, then apply the epilogue to in-range outputs of valid slots.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str_g = shmem
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * F32X_NR0;
    threadgroup float * temp_str_u = shmem + F32X_NR0 * F32X_NR1_SWIGLU
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * F32X_NR0;
    F32X_FOR_UNROLL (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, F32X_NR0, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, F32X_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = global_m < (int)args.ffn;
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * F32X_NR0];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * F32X_NR0];
            const int slot = ids[(ulong)im * args.n_tokens + (ulong)global_n];
            if (slot >= 0 && slot < slot_limit) {
                dst[(ulong)global_m + (ulong)slot * args.ffn] =
                    f32x_silu(min(g_val, limit)) * clamp(u_val, -limit, limit);
            }
        }
    }
}

typedef decltype(kernel_moe_down_grouped_slots_f32x<QT_IQ3_S_BYTES, QT_IQ3_S_NL, f32x_dequantize_iq3_s>) f32x_down_t;
typedef decltype(kernel_moe_swiglu_clamped_grouped_slots_f32x<QT_IQ2_S_BYTES, QT_IQ2_S_NL, f32x_dequantize_iq2_s>) f32x_swiglu_t;

template [[host_name("kernel_moe_down_iq3_s_f32x_grouped_slots")]] kernel f32x_down_t kernel_moe_down_grouped_slots_f32x<QT_IQ3_S_BYTES, QT_IQ3_S_NL, f32x_dequantize_iq3_s>;
template [[host_name("kernel_moe_down_iq4_xs_f32x_grouped_slots")]] kernel f32x_down_t kernel_moe_down_grouped_slots_f32x<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, f32x_dequantize_iq4_xs>;
template [[host_name("kernel_moe_swiglu_clamped_iq2_s_f32x_grouped_slots")]] kernel f32x_swiglu_t kernel_moe_swiglu_clamped_grouped_slots_f32x<QT_IQ2_S_BYTES, QT_IQ2_S_NL, f32x_dequantize_iq2_s>;
template [[host_name("kernel_moe_swiglu_clamped_iq3_s_f32x_grouped_slots")]] kernel f32x_swiglu_t kernel_moe_swiglu_clamped_grouped_slots_f32x<QT_IQ3_S_BYTES, QT_IQ3_S_NL, f32x_dequantize_iq3_s>;
