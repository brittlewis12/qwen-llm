// Q2_K mat-mat (W * X^T -> Y^T), using the 64x32x32 simdgroup_matrix tile.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int Q2K_QK       = 256;
constant constexpr int Q2K_BYTES    = 84;
constant constexpr int Q2K_NL       = Q2K_QK / 16;

constant constexpr int NR0_Q2K      = 64;
constant constexpr int NR1_Q2K      = 32;
constant constexpr int NK_Q2K       = 32;
constant constexpr int NL0_Q2K      = NK_Q2K / 16;
constant constexpr int NL1_Q2K      = NK_Q2K / 8;
constant constexpr int NW_Q2K       = 32;
constant constexpr int NSG_Q2K      = 4;

struct mat_mat_q2k_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

inline void dequantize_q2_K_half(device const uchar * blk_bytes,
                                 short il,
                                 thread half4x4 & reg) {
    device const uchar * scales = blk_bytes;
    device const uchar * qs = blk_bytes + 16;
    const half d_h = *((device const half *)(blk_bytes + 80));
    const half dmin_h = *((device const half *)(blk_bytes + 82));

    const uint sub = uint(il);
    const uint q_offset = 32u * (sub >> 3) + 16u * (sub & 1u);
    const uint shift = ((sub >> 1) & 3u) * 2u;
    const uint sc = uint(scales[sub]);
    const float dl = float(d_h) * float(sc & 0x0fu);
    const float ml = float(dmin_h) * float(sc >> 4);

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint q = (uint(qs[q_offset + uint(i)]) >> shift) & 3u;
        reg[i / 4][i % 4] = (half)(dl * float(q) - ml);
    }
}

kernel void kernel_mat_mat_q2_K_f32_mm(
        constant mat_mat_q2k_args & args [[buffer(0)]],
        device const uchar        * srcA [[buffer(1)]],
        device const float        * srcB [[buffer(2)]],
        device       float        * dst  [[buffer(3)]],
        threadgroup  uchar        * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_Q2K;
    const int r1 = tgpig.x * NR1_Q2K;
    const short nr0 = ((int)args.M - r0 < NR0_Q2K) ? (short)((int)args.M - r0) : NR0_Q2K;
    const short nr1 = ((int)args.N - r1 < NR1_Q2K) ? (short)((int)args.N - r1) : NR1_Q2K;

    const short lr0 = ((short)tiitg / NL0_Q2K) < nr0
                        ? ((short)tiitg / NL0_Q2K)
                        : nr0 - 1;
    const short il0 = tiitg % NL0_Q2K;
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_Q2K) < nr1
                        ? ((short)tiitg / NL1_Q2K)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_Q2K);

    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_Q2K) {
        {
            half4x4 temp_a;
            dequantize_q2_K_half(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_Q2K) / 8;
                const short lx = (tiitg / NL0_Q2K) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_Q2K;
            const short sy = (tiitg / NL1_Q2K) / 8;
            const short ly = (tiitg / NL1_Q2K) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q2K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2) ? x_ptr + Q2K_BYTES : x_ptr;
        y_ptr += NK_Q2K;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_Q2K / 8; ++ik) {
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

    if (r0 + NR0_Q2K <= (int)args.M && r1 + NR1_Q2K <= (int)args.N) {
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
                                       + (16 * (sgitg >> 1)) * NR0_Q2K;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_Q2K * (i / 4),
                            NR0_Q2K, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1_Q2K) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + j * NR0_Q2K;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
