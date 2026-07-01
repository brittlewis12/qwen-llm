// Legacy Q4_0/Q4_1 mat-mat (W * X^T -> Y).
//
// The older block32 kernels stage one weight row and thirty-two query rows per
// threadgroup. These entry points use the shared 64x32x32 simdgroup-matrix
// tile in mat_mat_mm_tile.h (this file originally carried its own macro copy
// of the tile — it was the proof-of-pattern that the shared header
// generalizes; both QK=32 formats have NL=2, where the uniform advance rule
// degenerates to "advance one block per K-step").

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include "mat_mat_mm_tile.h"

constant constexpr int QK4_0_BYTES = 18;
constant constexpr int QK4_1_BYTES = 20;
constant constexpr int Q4_LEGACY_NL = 2; // QK=32 / 16

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

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_q4_0_f32_mm,
                       mat_mat_q4_legacy_args,
                       QK4_0_BYTES,
                       Q4_LEGACY_NL,
                       dequantize_q4_0_half)

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_q4_1_f32_mm,
                       mat_mat_q4_legacy_args,
                       QK4_1_BYTES,
                       Q4_LEGACY_NL,
                       dequantize_q4_1_half)
