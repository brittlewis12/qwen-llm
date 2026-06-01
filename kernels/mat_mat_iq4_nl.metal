// IQ4_NL mat-mat (W * X^T -> Y^T).
//
// This is the same 64x32x32 simdgroup_matrix tile shape as the Q4_K/Q5_K/Q6_K
// prompt mat-mat kernels, specialized for GGML IQ4_NL blocks. It replaces the
// coverage-first scalar/threadgroup-row kernel for prompt prefill, where reusing
// one dequantized 64-row weight tile across 32 query rows is the whole win.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int IQ4NL_QK       = 32;
constant constexpr int IQ4NL_BYTES    = 18;
constant constexpr int IQ4NL_NL       = IQ4NL_QK / 16;

constant constexpr int NR0_IQ4NL      = 64;
constant constexpr int NR1_IQ4NL      = 32;
constant constexpr int NK_IQ4NL       = 32;
constant constexpr int NL0_IQ4NL      = NK_IQ4NL / 16;
constant constexpr int NL1_IQ4NL      = NK_IQ4NL / 8;
constant constexpr int NW_IQ4NL       = 32;
constant constexpr int NSG_IQ4NL      = 4;
constant constexpr int NTH_IQ4NL      = NW_IQ4NL * NSG_IQ4NL;

constant float iq4nl_values_mm[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f,
};

struct mat_mat_iq4nl_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

inline void dequantize_iq4_nl_half(device const uchar * blk_bytes,
                                   short il,
                                   thread half4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    device const uchar * qs = blk_bytes + 2;
    const bool hi = (il & 1) != 0;

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uchar q = qs[i];
        const uint idx = hi ? uint(q >> 4) : uint(q & 0x0f);
        reg[i / 4][i % 4] = (half)(float(d_h) * iq4nl_values_mm[idx]);
    }
}

kernel void kernel_mat_mat_iq4_nl_f32_mm(
        constant mat_mat_iq4nl_args & args [[buffer(0)]],
        device const uchar          * srcA [[buffer(1)]],
        device const float          * srcB [[buffer(2)]],
        device       float          * dst  [[buffer(3)]],
        threadgroup  uchar          * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int r0 = tgpig.y * NR0_IQ4NL;
    const int r1 = tgpig.x * NR1_IQ4NL;
    const short nr0 = ((int)args.M - r0 < NR0_IQ4NL) ? (short)((int)args.M - r0) : NR0_IQ4NL;
    const short nr1 = ((int)args.N - r1 < NR1_IQ4NL) ? (short)((int)args.N - r1) : NR1_IQ4NL;

    const short lr0 = ((short)tiitg / NL0_IQ4NL) < nr0
                        ? ((short)tiitg / NL0_IQ4NL)
                        : nr0 - 1;
    const short il0 = tiitg % NL0_IQ4NL;

    const short lr1 = ((short)tiitg / NL1_IQ4NL) < nr1
                        ? ((short)tiitg / NL1_IQ4NL)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_IQ4NL);

    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)
                                       + (ulong)iy;

    simdgroup_half8x8  ma[4];
    simdgroup_half8x8  mb[2];
    simdgroup_float8x8 mc[8];

    FOR_UNROLL (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_IQ4NL) {
        {
            half4x4 temp_a;
            dequantize_iq4_nl_half(x_ptr, il0, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_IQ4NL) / 8;
                const short lx = (tiitg / NL0_IQ4NL) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = tiitg % NL1_IQ4NL;
            const short sy = (tiitg / NL1_IQ4NL) / 8;
            const short ly = (tiitg / NL1_IQ4NL) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        x_ptr += IQ4NL_BYTES;
        y_ptr += NK_IQ4NL;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);

        FOR_UNROLL (short ik = 0; ik < NK_IQ4NL / 8; ++ik) {
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

    if (r0 + NR0_IQ4NL <= (int)args.M && r1 + NR1_IQ4NL <= (int)args.N) {
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
                                       + (16 * (sgitg >> 1)) * NR0_IQ4NL;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_IQ4NL * (i / 4),
                            NR0_IQ4NL, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1_IQ4NL) {
                device float * D = dst + r0 + (r1 + j) * args.M;
                threadgroup float * C = temp_str + j * NR0_IQ4NL;
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}
