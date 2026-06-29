// Legacy Q4_0/Q4_1 mat-mat (W * X^T -> Y).
//
// The older block32 kernels stage one weight row and thirty-two query rows per
// threadgroup. This file mirrors the 64x32x32 simdgroup-matrix shape used by
// the K-quant and Q8_0 prompt kernels, specialized for QK=32 legacy Q4 blocks.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

constant constexpr int QK4_0_BYTES = 18;
constant constexpr int QK4_1_BYTES = 20;
constant constexpr int NR0_MM = 64;
constant constexpr int NR1_MM = 32;
constant constexpr int NK_MM = 32;
constant constexpr int NL0_MM = 2;
constant constexpr int NL1_MM = 4;

struct mat_mat_q4_legacy_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

inline void dequantize_q4_0_half(device const uchar * blk,
                                 short il,
                                 thread half4x4 & reg) {
    const half d_h = *((device const half *)blk);
    const float d = float(d_h);
    device const uchar * qs = blk + 2;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uchar packed = qs[i];
        const int q = (il == 0) ? int(packed & 0x0f) : int(packed >> 4);
        reg[i / 4][i % 4] = half(d * float(q - 8));
    }
}

inline void dequantize_q4_1_half(device const uchar * blk,
                                 short il,
                                 thread half4x4 & reg) {
    const half d_h = *((device const half *)blk);
    const half m_h = *((device const half *)(blk + 2));
    const float d = float(d_h);
    const float m = float(m_h);
    device const uchar * qs = blk + 4;
    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uchar packed = qs[i];
        const int q = (il == 0) ? int(packed & 0x0f) : int(packed >> 4);
        reg[i / 4][i % 4] = half(d * float(q) + m);
    }
}

#define Q4_LEGACY_MATMUL_KERNEL(NAME, BLOCK_BYTES, DEQUANT_FN)                  \
kernel void NAME(                                                               \
        constant mat_mat_q4_legacy_args & args [[buffer(0)]],                  \
        device const uchar * srcA [[buffer(1)]],                               \
        device const float * srcB [[buffer(2)]],                               \
        device float * dst [[buffer(3)]],                                      \
        threadgroup uchar * shmem [[threadgroup(0)]],                          \
        uint3 tgpig [[threadgroup_position_in_grid]],                          \
        ushort tiitg [[thread_index_in_threadgroup]],                          \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                     \
    threadgroup half * sa = (threadgroup half *)(shmem);                       \
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);                \
                                                                                 \
    const int r0 = tgpig.y * NR0_MM;                                           \
    const int r1 = tgpig.x * NR1_MM;                                           \
    const short nr0 = ((int)args.M - r0 < NR0_MM)                              \
        ? (short)((int)args.M - r0)                                            \
        : NR0_MM;                                                              \
    const short nr1 = ((int)args.N - r1 < NR1_MM)                              \
        ? (short)((int)args.N - r1)                                            \
        : NR1_MM;                                                              \
                                                                                 \
    const short lr0 = ((short)tiitg / NL0_MM) < nr0                            \
        ? ((short)tiitg / NL0_MM)                                              \
        : nr0 - 1;                                                             \
    const short il0 = tiitg % NL0_MM;                                          \
    short il = il0;                                                            \
    const short lr1 = ((short)tiitg / NL1_MM) < nr1                            \
        ? ((short)tiitg / NL1_MM)                                              \
        : nr1 - 1;                                                             \
    const short iy = 8 * (tiitg % NL1_MM);                                     \
                                                                                 \
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);         \
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)      \
                                      + (ulong)iy;                             \
                                                                                 \
    simdgroup_half8x8 ma[4];                                                   \
    simdgroup_half8x8 mb[2];                                                   \
    simdgroup_float8x8 mc[8];                                                  \
    FOR_UNROLL (short i = 0; i < 8; ++i) {                                     \
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);                  \
    }                                                                          \
                                                                                 \
    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {                  \
        {                                                                      \
            half4x4 temp_a;                                                    \
            DEQUANT_FN(x_ptr, il, temp_a);                                     \
            threadgroup_barrier(mem_flags::mem_threadgroup);                   \
            FOR_UNROLL (short i = 0; i < 16; ++i) {                            \
                const short sx = 2 * il0 + i / 8;                              \
                const short sy = (tiitg / NL0_MM) / 8;                         \
                const short lx = (tiitg / NL0_MM) % 8;                         \
                const short ly = i % 8;                                        \
                const short ib = 8 * sx + sy;                                  \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];          \
            }                                                                  \
        }                                                                      \
        {                                                                      \
            const short sx = tiitg % NL1_MM;                                   \
            const short sy = (tiitg / NL1_MM) / 8;                             \
            const short ly = (tiitg / NL1_MM) % 8;                             \
            const short ib = 4 * sx + sy;                                      \
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =                  \
                (half2x4)(*((device const float2x4 *)y_ptr));                  \
        }                                                                      \
                                                                                 \
        il = il % 2;                                                           \
        x_ptr += BLOCK_BYTES;                                                  \
        y_ptr += NK_MM;                                                        \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
                                                                                 \
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);             \
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);             \
        FOR_UNROLL (short ik = 0; ik < NK_MM / 8; ++ik) {                      \
            simdgroup_barrier(mem_flags::mem_none);                            \
            FOR_UNROLL (short i = 0; i < 4; ++i) {                             \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            FOR_UNROLL (short i = 0; i < 2; ++i) {                             \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);             \
            }                                                                  \
            simdgroup_barrier(mem_flags::mem_none);                            \
            FOR_UNROLL (short i = 0; i < 8; ++i) {                             \
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]); \
            }                                                                  \
            lsma += 8 * 64;                                                    \
            lsmb += 4 * 64;                                                    \
        }                                                                      \
    }                                                                          \
                                                                                 \
    if (r0 + NR0_MM <= (int)args.M && r1 + NR1_MM <= (int)args.N) {            \
        device float * C = dst + (r0 + 32 * (sgitg & 1))                       \
            + (r1 + 16 * (sgitg >> 1)) * args.M;                               \
        FOR_UNROLL (short i = 0; i < 8; ++i) {                                 \
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),     \
                            args.M, 0, false);                                 \
        }                                                                      \
    } else {                                                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        threadgroup float * temp_str = ((threadgroup float *)shmem)            \
            + 32 * (sgitg & 1) + (16 * (sgitg >> 1)) * NR0_MM;                 \
        FOR_UNROLL (short i = 0; i < 8; ++i) {                                 \
            simdgroup_store(mc[i], temp_str + 8 * (i % 4)                      \
                + 8 * NR0_MM * (i / 4), NR0_MM, 0, false);                     \
        }                                                                      \
        threadgroup_barrier(mem_flags::mem_threadgroup);                       \
        if (sgitg == 0) {                                                       \
            for (int j = tiitg; j < nr1; j += 32) {                            \
                device float * D = dst + (ulong)(r1 + j) * args.M + r0;        \
                threadgroup float * C = temp_str + j * NR0_MM;                 \
                for (int i = 0; i < nr0; ++i) {                                \
                    D[i] = C[i];                                               \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

Q4_LEGACY_MATMUL_KERNEL(kernel_mat_mat_q4_0_f32_mm, QK4_0_BYTES, dequantize_q4_0_half)
Q4_LEGACY_MATMUL_KERNEL(kernel_mat_mat_q4_1_f32_mm, QK4_1_BYTES, dequantize_q4_1_half)
