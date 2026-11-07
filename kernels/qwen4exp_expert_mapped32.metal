// Flash-Next mapped 64-output x 32-slot gate/up tile, derived from the
// existing clamped N32 tile. Preserve half operands and F32 per-output K
// order. Skip upper-half MMA work when a descriptor has <=16 valid slots.
#include <metal_stdlib>
using namespace metal;
#include "quant_tiles.h"

#define FN32_FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr short FN32_NR0 = 64; // outputs per tile
constant constexpr short FN32_NR1 = 32; // tokens per tile
constant constexpr short FN32_NK = 32;  // K per step
constant constexpr short FN32_NL0 = 2;  // 16-element dequant calls per row per step
constant constexpr short FN32_NL1 = 4;  // 8-wide activation chunks per token per step

// Layout of moe.metal's moe_group_q4k_args.
struct fn_mapped32_args {
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

// A single bounded scan builds expert/token tile coordinates. The family
// caller owns unique route slots whose total population is tokens * topk.
template<ushort ROWS>
inline void fn_expert_build_map(
        constant fn_mapped32_args &args,
        device const int *counts, device int2 *tiles,
        threadgroup uint *prefix, uint tid) {
    const uint bound = (args.n_tokens * args.topk + ROWS - 1) / ROWS + args.n_expert;
    for (uint i = tid; i < bound; i += 512) tiles[i] = int2(-1, 0);
    const int c = counts[tid];
    const uint n = c > 0 && uint(c) <= args.n_tokens ? (uint(c) + ROWS - 1) / ROWS : 0;
    prefix[tid] = n;
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    for (uint step = 1; step < 512; step <<= 1) {
        const uint add = tid >= step ? prefix[tid - step] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[tid] += add;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint first = prefix[tid] - n;
    for (uint j = 0; j < n && first + j < bound; ++j)
        tiles[first + j] = int2(int(tid), int(j));
}
#define FN_EXPERT_MAP(ROWS) \
[[max_total_threads_per_threadgroup(512)]] kernel void kernel_qwen4exp_expert_tile_map##ROWS( \
 constant fn_mapped32_args &args [[buffer(0)]], device const int *counts [[buffer(1)]], \
 device int2 *tiles [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) { \
 threadgroup uint prefix[512]; fn_expert_build_map<ROWS>(args,counts,tiles,prefix,tid); }
FN_EXPERT_MAP(32)

// moe.metal's moe_silu_f.
inline float fn32_silu(float x) {
    return x / (1.0f + exp(-x));
}

template <int BYTES, short NL, void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_qwen4exp_expert_mapped32(
        constant fn_mapped32_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        device const int2 * tiles           [[buffer(7)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  raw_group [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const int2 desc = tiles[raw_group.x];
    if (desc.x < 0 || uint(desc.x) >= args.n_expert || desc.y < 0) return;
    const uint3 tgpig = uint3(uint(desc.y), raw_group.y, uint(desc.x));
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup half * sa_g = (threadgroup half *)(shmem);         // 64 x 32
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);  // 64 x 32
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);  // 32 x 32

    const int im = tgpig.z;
    const int r0 = tgpig.y * FN32_NR0;
    const int r1 = tgpig.x * FN32_NR1;

    const int count = counts[im];
    if (count < 0 || uint(count) > args.n_tokens
        || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < FN32_NR0) ? (short)((int)args.ffn - r0) : FN32_NR0;
    const short nr1 = (count - r1 < FN32_NR1) ? (short)(count - r1) : FN32_NR1;

    const short lr0 = ((short)tiitg / FN32_NL0) < nr0
                        ? ((short)tiitg / FN32_NL0)
                        : nr0 - 1;
    const short il0 = (tiitg % FN32_NL0);
    short il = il0;

    const short lr1 = ((short)tiitg / FN32_NL1) < nr1
                        ? ((short)tiitg / FN32_NL1)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % FN32_NL1);

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

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_g[8];
    simdgroup_float8x8 mc_u[8];

    for (short i = 0; i < 8; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += FN32_NK) {
        {
            half4x4 temp_a;
            DEQ(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / FN32_NL0) / 8;
                const short lx = (tiitg / FN32_NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            DEQ(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / FN32_NL0) / 8;
                const short lx = (tiitg / FN32_NL0) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            // All 128 threads: token tiitg / 4, 8-wide K chunk tiitg % 4.
            const short sx = (tiitg % FN32_NL1);
            const short sy = (tiitg / FN32_NL1) / 8;
            const short ib = 4 * sx + sy;
            const short ly = (tiitg / FN32_NL1) % 8;
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
        y_ptr += FN32_NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 2 * 64 * (sgitg / 2));

        if (nr1 > 16 || sgitg < 2) {
        for (short ik = 0; ik < FN32_NK / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 4], ma_g[i % 4], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 4], ma_u[i % 4], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 4 * 64;
        }
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Gate and up results (token-major, 64 rows x 32 tokens F32 each: 8 KiB
    // apiece) over the whole 16 KiB.
    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * FN32_NR0;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 8192))
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * FN32_NR0;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 4) + 8 * FN32_NR0 * (i / 4), FN32_NR0, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 4) + 8 * FN32_NR0 * (i / 4), FN32_NR0, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 16 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 16; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * FN32_NR0];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * FN32_NR0];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            if (slot >= 0 && slot < slot_limit) {
                dst[global_m + (ulong)slot * args.ffn] =
                    fn32_silu(g_val) * u_val;
            }
        }
    }
}

typedef decltype(kernel_qwen4exp_expert_mapped32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, qt_dequantize_iq2_s>) fn32_swiglu_t;

template [[host_name("kernel_qwen4exp_expert_iq2_s_mapped32")]] kernel fn32_swiglu_t kernel_qwen4exp_expert_mapped32<QT_IQ2_S_BYTES, QT_IQ2_S_NL, qt_dequantize_iq2_s>;
template [[host_name("kernel_qwen4exp_expert_iq3_s_mapped32")]] kernel fn32_swiglu_t kernel_qwen4exp_expert_mapped32<QT_IQ3_S_BYTES, QT_IQ3_S_NL, qt_dequantize_iq3_s>;

template [[host_name("kernel_qwen4exp_expert_iq3_xxs_mapped32")]] kernel fn32_swiglu_t kernel_qwen4exp_expert_mapped32<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, qt_dequantize_iq3_xxs>;
