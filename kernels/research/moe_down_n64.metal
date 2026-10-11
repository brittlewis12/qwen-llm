// Experimental half-operand grouped DOWN: M64/N64/K32, 128 threads.
// Two incumbent N32 panels reuse one decoded weight tile. Each output keeps
// moe_grouped_slots_mm_generic_body<EPI=0>'s half conversion and K/MMA order.
// Staging: 4096 B weights + 4096 B activations. Drain each N32 F32 result
// panel through the same 8192 B overlay before storing the next panel.
// No persistent scratch or repacked weights.
#include <metal_stdlib>
using namespace metal;
#include "../quant_tiles.h"

struct moe_down_n64_args {
    uint M;
    uint N; // actual ids stride, independent of the launch grid
    uint K;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
    uint b_div;
    uint slot_limit;
};

template <int BYTES, short NL,
          void (*DEQ)(device const uchar *, short, thread half4x4 &)>
kernel void kernel_moe_down_grouped_slots_n64(
        constant moe_down_n64_args & args [[buffer(0)]],
        device const uchar * srcA [[buffer(1)]],
        device const float * srcB [[buffer(2)]],
        device const int * counts [[buffer(3)]],
        device const int * ids [[buffer(4)]],
        device float * dst [[buffer(5)]],
        threadgroup uchar * shmem [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint tiitg_wide [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    const ushort tiitg = ushort(tiitg_wide);
    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);
    const int im = tgpig.z;
    const int r0 = tgpig.y * 64;
    const int r1 = tgpig.x * 64;
    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;
    const short nr0 = short(min(64, int(args.M) - r0));
    const short nr1 = short(min(64, count - r1));
    const short lr0 = min(short(tiitg / 2), short(nr0 - 1));
    const short il0 = tiitg % 2;
    short il = il0;
    const short iy = 8 * (tiitg % 4);
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im
        + (ulong)args.nb01 * (r0 + lr0) + (ulong)(il0 / NL) * BYTES;
    const int slot_limit = int(args.slot_limit);
    device const float * y_ptr[2];
    for (short p = 0; p < 2; ++p) {
        const short lr1 = min(short(tiitg / 4 + 32 * p), short(nr1 - 1));
        const int slot = ids[(ulong)im * args.N + r1 + lr1];
        const bool valid = slot >= 0 && slot < slot_limit;
        const int b_row = (valid ? slot : 0) / int(args.b_div);
        y_ptr[p] = srcB + (ulong)args.stride_b * b_row + (ulong)iy;
    }

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[4];
    simdgroup_float8x8 mc[16];
    for (short i = 0; i < 16; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }
    for (uint loop_k = 0; loop_k < args.K; loop_k += 32) {
        half4x4 temp_a;
        DEQ(x_ptr, il, temp_a);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 16; ++i) {
            const short sx = 2 * il0 + i / 8;
            const short sy = (tiitg / 2) / 8;
            const short lx = (tiitg / 2) % 8;
            const short ly = i % 8;
            const short ib = 8 * sx + sy;
            sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
        }
        for (short p = 0; p < 2; ++p) {
            const short sx = tiitg % 4;
            const short sy = (tiitg / 4) / 8;
            const short ly = (tiitg / 4) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 1024 * p + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr[p]));
            y_ptr[p] += 32;
        }
        il = (il + 2 < NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + BYTES * ((2 + NL - 1) / NL) : x_ptr;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);
        for (short ik = 0; ik < 32 / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short p = 0; p < 2; ++p) {
                for (short i = 0; i < 2; ++i) {
                    simdgroup_load(mb[2 * p + i], lsmb + 1024 * p + 64 * i, 8, 0, false);
                }
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short p = 0; p < 2; ++p) {
                for (short i = 0; i < 8; ++i) {
                    simdgroup_multiply_accumulate(mc[8 * p + i], mb[2 * p + i / 4], ma[i % 4], mc[8 * p + i]);
                }
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }
    threadgroup float * temp_str = (threadgroup float *)shmem
        + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * 64;
    for (short p = 0; p < 2; ++p) {
        // All threads finish input reads (p=0), or the previous panel's
        // scatter (p=1), before any SIMD group overwrites the shared overlay.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[8 * p + i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),
                            64, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < 32 && 32 * p + j < nr1; j += 32) {
                const int slot = ids[(ulong)im * args.N + r1 + 32 * p + j];
                if (slot < 0 || slot >= slot_limit) continue;
                device float * D = dst + (ulong)slot * args.M + r0;
                threadgroup float * C = temp_str + j * 64;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
        // No barrier is conditional on a slot, row tail, or SIMD group. The
        // next iteration's leading barrier protects reuse even for an empty
        // second panel; after the last scatter the overlay is no longer used.
    }
}

typedef decltype(kernel_moe_down_grouped_slots_n64<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>) moe_down_n64_t;
template [[host_name("kernel_moe_down_q2_0_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q2_0_BYTES, QT_Q2_0_NL, qt_dequantize_q2_0>;
template [[host_name("kernel_moe_down_q2_K_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q2_K_BYTES, QT_Q2_K_NL, qt_dequantize_q2_K>;
template [[host_name("kernel_moe_down_q3_K_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q3_K_BYTES, QT_Q3_K_NL, qt_dequantize_q3_K>;
template [[host_name("kernel_moe_down_q4_K_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q4_K_BYTES, QT_Q4_K_NL, qt_dequantize_q4_K>;
template [[host_name("kernel_moe_down_q5_K_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q5_K_BYTES, QT_Q5_K_NL, qt_dequantize_q5_K>;
template [[host_name("kernel_moe_down_q6_K_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q6_K_BYTES, QT_Q6_K_NL, qt_dequantize_q6_K>;
template [[host_name("kernel_moe_down_q8_0_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q8_0_BYTES, QT_Q8_0_NL, qt_dequantize_q8_0>;
template [[host_name("kernel_moe_down_q4_0_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q4_0_BYTES, QT_Q4_0_NL, qt_dequantize_q4_0>;
template [[host_name("kernel_moe_down_q4_1_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_Q4_1_BYTES, QT_Q4_1_NL, qt_dequantize_q4_1>;
template [[host_name("kernel_moe_down_iq2_s_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_IQ2_S_BYTES, QT_IQ2_S_NL, qt_dequantize_iq2_s>;
template [[host_name("kernel_moe_down_iq3_xxs_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_IQ3_XXS_BYTES, QT_IQ3_XXS_NL, qt_dequantize_iq3_xxs>;
template [[host_name("kernel_moe_down_iq3_s_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_IQ3_S_BYTES, QT_IQ3_S_NL, qt_dequantize_iq3_s>;
template [[host_name("kernel_moe_down_iq4_nl_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_IQ4_NL_BYTES, QT_IQ4_NL_NL, qt_dequantize_iq4_nl>;
template [[host_name("kernel_moe_down_iq4_xs_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_IQ4_XS_BYTES, QT_IQ4_XS_NL, qt_dequantize_iq4_xs>;
template [[host_name("kernel_moe_down_f32_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_F32_BYTES, QT_F32_NL, qt_dequantize_f32>;
template [[host_name("kernel_moe_down_f16_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_F16_BYTES, QT_F16_NL, qt_dequantize_f16>;
template [[host_name("kernel_moe_down_bf16_f32_grouped_slots_n64")]] kernel moe_down_n64_t kernel_moe_down_grouped_slots_n64<QT_BF16_BYTES, QT_BF16_NL, qt_dequantize_bf16>;
