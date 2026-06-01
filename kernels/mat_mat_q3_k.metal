// Q3_K mat-mat (W * X^T -> Y^T), using the 64x32x32 simdgroup_matrix tile.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int Q3K_QK       = 256;
constant constexpr int Q3K_BYTES    = 110;
constant constexpr int Q3K_NL       = Q3K_QK / 16;

constant constexpr int NR0_Q3K      = 64;
constant constexpr int NR1_Q3K      = 32;
constant constexpr int NK_Q3K       = 32;
constant constexpr int NL0_Q3K      = NK_Q3K / 16;
constant constexpr int NL1_Q3K      = NK_Q3K / 8;
constant constexpr int NW_Q3K       = 32;
constant constexpr int NSG_Q3K      = 4;

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

kernel void kernel_mat_mat_q3_K_f32_mm(
        constant mat_mat_q3k_args & args [[buffer(0)]],
        device const uchar        * srcA [[buffer(1)]],
        device const float        * srcB [[buffer(2)]],
        device       float        * dst  [[buffer(3)]],
        threadgroup  uchar        * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_Q3K;
    const int r1 = tgpig.x * NR1_Q3K;
    const short nr0 = ((int)args.M - r0 < NR0_Q3K) ? (short)((int)args.M - r0) : NR0_Q3K;
    const short nr1 = ((int)args.N - r1 < NR1_Q3K) ? (short)((int)args.N - r1) : NR1_Q3K;

    const short lr0 = ((short)tiitg / NL0_Q3K) < nr0
                        ? ((short)tiitg / NL0_Q3K)
                        : nr0 - 1;
    const short il0 = tiitg % NL0_Q3K;
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_Q3K) < nr1
                        ? ((short)tiitg / NL1_Q3K)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_Q3K);

    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_Q3K) {
        {
            half4x4 temp_a;
            dequantize_q3_K_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_Q3K) / 8;
                const short lx = (tiitg / NL0_Q3K) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_Q3K;
            const short sy = (tiitg / NL1_Q3K) / 8;
            const short ly = (tiitg / NL1_Q3K) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q3K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + Q3K_BYTES : x_ptr;
        y_ptr += NK_Q3K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_Q3K / 8; ++ik) {
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

    if (r0 + NR0_Q3K <= (int)args.M && r1 + NR1_Q3K <= (int)args.N) {
        device float * C = dst + (r0 + 32 * (sgitg & 1))
                               + (r1 + 16 * (sgitg >> 1)) * args.M;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),
                            args.M, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *)shmem)
                                       + 32 * (sgitg & 1)
                                       + (16 * (sgitg >> 1)) * NR0_Q3K;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_Q3K * (i / 4),
                            NR0_Q3K, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1_Q3K) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + j * NR0_Q3K;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
