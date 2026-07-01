// Shared 64x32x32 simdgroup_matrix mat-mat tile (W * X^T -> Y col-major).
//
// One definition of the GEMM skeleton that was previously copy-pasted across
// mat_mat_q2_k / mat_mat_q3_k / mat_mat_iq4_nl / mat_mat_iq4_xs /
// mat_mat_q4_legacy. Each format file keeps its own dequant function and
// constant tables and instantiates the kernel via MAT_MAT_MM_TILE_KERNEL.
//
// Tile geometry (lifted from llama.cpp kernel_mul_mm, documented in detail
// in mat_mat_q4_k.metal):
//   NR0 = 64  output rows per threadgroup (M tile)
//   NR1 = 32  output cols per threadgroup (N tile)
//   NK  = 32  K elements per outer-loop step
//   4 simdgroups x 32 threads; sa = 4 KiB dequant'd weight tile (half),
//   sb = 4 KiB activation tile (half); fp32 simdgroup accumulators.
//
// Parameters:
//   NAME         kernel entry-point name
//   ARGS_T       args struct type: { uint M, N, K, nb01, stride_b; }
//   BLOCK_BYTES  bytes per quant block (super-block for QK=256 formats)
//   NL           dequant calls per block = QK/16 (16 for QK=256, 2 for QK=32)
//   DEQUANT_FN   inline void fn(device const uchar *, short il, thread half4x4 &)
//
// The A-pointer advance rule is uniform across block sizes:
//   il    = (il + 2 < NL) ? il + 2 : il % 2;
//   x_ptr = (il < 2) ? x_ptr + BLOCK_BYTES * ((2 + NL - 1) / NL) : x_ptr;
// For NL=16 (QK=256) this advances one super-block when il wraps; for NL=2
// (QK=32) it degenerates to `il = il0; x_ptr += BLOCK_BYTES` every K-step —
// exactly the behavior the per-file copies hand-coded. Do not "simplify"
// either branch per-format; the uniform spelling is what lets one macro
// serve both families.
//
// This header is textual macro expansion only (no functions, no state); it
// must produce token-for-token the same kernel bodies the per-file copies
// had. Any tile-logic change here applies to every instantiation — that is
// the point, and also the risk: re-run the dense low-bit prompt oracles
// after touching anything below.

#pragma once

#define MAT_MAT_MM_TILE_KERNEL(NAME, ARGS_T, BLOCK_BYTES, NL, DEQUANT_FN)       \
kernel void NAME(                                                               \
        constant ARGS_T           & args  [[buffer(0)]],                        \
        device const uchar        * srcA  [[buffer(1)]],                        \
        device const float        * srcB  [[buffer(2)]],                        \
        device       float        * dst   [[buffer(3)]],                        \
        threadgroup  uchar        * shmem [[threadgroup(0)]],                   \
        uint3  tgpig [[threadgroup_position_in_grid]],                          \
        ushort tiitg [[thread_index_in_threadgroup]],                           \
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {                      \
    threadgroup half * sa = (threadgroup half *)(shmem);                        \
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);                 \
                                                                                \
    const int r0 = tgpig.y * 64;                                                \
    const int r1 = tgpig.x * 32;                                                \
    const short nr0 = ((int)args.M - r0 < 64) ? (short)((int)args.M - r0) : 64; \
    const short nr1 = ((int)args.N - r1 < 32) ? (short)((int)args.N - r1) : 32; \
                                                                                \
    const short lr0 = ((short)tiitg / 2) < nr0 ? ((short)tiitg / 2) : nr0 - 1;  \
    const short il0 = tiitg % 2;                                                \
    short il = il0;                                                             \
                                                                                \
    const short lr1 = ((short)tiitg / 4) < nr1 ? ((short)tiitg / 4) : nr1 - 1;  \
    const short iy = 8 * (tiitg % 4);                                           \
                                                                                \
    device const uchar * x_ptr = srcA + (ulong)args.nb01 * (r0 + lr0);          \
    device const float * y_ptr = srcB + (ulong)args.stride_b * (r1 + lr1)       \
                                       + (ulong)iy;                             \
                                                                                \
    simdgroup_half8x8  ma[4];                                                   \
    simdgroup_half8x8  mb[2];                                                   \
    simdgroup_float8x8 mc[8];                                                   \
                                                                                \
    FOR_UNROLL (short i = 0; i < 8; ++i) {                                      \
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);                   \
    }                                                                           \
                                                                                \
    for (uint loop_k = 0; loop_k < args.K; loop_k += 32) {                      \
        {                                                                       \
            half4x4 temp_a;                                                     \
            DEQUANT_FN(x_ptr, il, temp_a);                                      \
                                                                                \
            threadgroup_barrier(mem_flags::mem_threadgroup);                    \
                                                                                \
            FOR_UNROLL (short i = 0; i < 16; ++i) {                             \
                const short sx = 2 * il0 + i / 8;                               \
                const short sy = (tiitg / 2) / 8;                               \
                const short lx = (tiitg / 2) % 8;                               \
                const short ly = i % 8;                                         \
                const short ib = 8 * sx + sy;                                   \
                *(sa + 64 * ib + 8 * ly + lx) = temp_a[i / 4][i % 4];           \
            }                                                                   \
        }                                                                       \
                                                                                \
        {                                                                       \
            const short sx = tiitg % 4;                                         \
            const short sy = (tiitg / 4) / 8;                                   \
            const short ly = (tiitg / 4) % 8;                                   \
            const short ib = 4 * sx + sy;                                       \
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =                   \
                (half2x4)(*((device const float2x4 *)y_ptr));                   \
        }                                                                       \
                                                                                \
        il = (il + 2 < (NL)) ? il + 2 : il % 2;                                 \
        x_ptr = (il < 2)                                                        \
                  ? x_ptr + (BLOCK_BYTES) * ((2 + (NL) - 1) / (NL))             \
                  : x_ptr;                                                      \
        y_ptr += 32;                                                            \
                                                                                \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
                                                                                \
        threadgroup const half * lsma = sa + 4 * 64 * (sgitg % 2);              \
        threadgroup const half * lsmb = sb + 2 * 64 * (sgitg / 2);              \
                                                                                \
        FOR_UNROLL (short ik = 0; ik < 32 / 8; ++ik) {                          \
            simdgroup_barrier(mem_flags::mem_none);                             \
                                                                                \
            FOR_UNROLL (short i = 0; i < 4; ++i) {                              \
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);              \
            }                                                                   \
                                                                                \
            simdgroup_barrier(mem_flags::mem_none);                             \
            FOR_UNROLL (short i = 0; i < 2; ++i) {                              \
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);              \
            }                                                                   \
                                                                                \
            simdgroup_barrier(mem_flags::mem_none);                             \
            FOR_UNROLL (short i = 0; i < 8; ++i) {                              \
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]); \
            }                                                                   \
                                                                                \
            lsma += 8 * 64;                                                     \
            lsmb += 4 * 64;                                                     \
        }                                                                       \
    }                                                                           \
                                                                                \
    if (r0 + 64 <= (int)args.M && r1 + 32 <= (int)args.N) {                     \
        device float * C = dst + (r0 + 32 * (sgitg & 1))                        \
                               + (r1 + 16 * (sgitg >> 1)) * args.M;             \
        FOR_UNROLL (short i = 0; i < 8; ++i) {                                  \
            simdgroup_store(mc[i], C + 8 * (i % 4) + 8 * args.M * (i / 4),      \
                            args.M, 0, false);                                  \
        }                                                                       \
    } else {                                                                    \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
        threadgroup float * temp_str = ((threadgroup float *)shmem)             \
                                       + 32 * (sgitg & 1)                       \
                                       + (16 * (sgitg >> 1)) * 64;              \
        FOR_UNROLL (short i = 0; i < 8; ++i) {                                  \
            simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * 64 * (i / 4),   \
                            64, 0, false);                                      \
        }                                                                       \
        threadgroup_barrier(mem_flags::mem_threadgroup);                        \
        if (sgitg == 0) {                                                       \
            for (int j = tiitg; j < nr1; j += 32) {                             \
                device float * D = dst + r0 + (ulong)(r1 + j) * args.M;         \
                threadgroup float * C = temp_str + j * 64;                      \
                for (int i = 0; i < nr0; ++i) {                                 \
                    D[i] = C[i];                                                \
                }                                                               \
            }                                                                   \
        }                                                                       \
    }                                                                           \
}
