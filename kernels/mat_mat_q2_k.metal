// Q2_K mat-mat (W * X^T -> Y^T), using the shared 64x32x32 simdgroup_matrix
// tile from mat_mat_mm_tile.h; this file owns only the Q2_K dequant.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include "mat_mat_mm_tile.h"

constant constexpr int Q2K_QK       = 256;
constant constexpr int Q2K_BYTES    = 84;
constant constexpr int Q2K_NL       = Q2K_QK / 16;

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

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_q2_K_f32_mm,
                       mat_mat_q2k_args,
                       Q2K_BYTES,
                       Q2K_NL,
                       dequantize_q2_K_half)
