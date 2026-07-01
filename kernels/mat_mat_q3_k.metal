// Q3_K mat-mat (W * X^T -> Y^T), using the shared 64x32x32 simdgroup_matrix
// tile from mat_mat_mm_tile.h; this file owns only the Q3_K dequant.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include "mat_mat_mm_tile.h"

constant constexpr int Q3K_QK       = 256;
constant constexpr int Q3K_BYTES    = 110;
constant constexpr int Q3K_NL       = Q3K_QK / 16;

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

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_q3_K_f32_mm,
                       mat_mat_q3k_args,
                       Q3K_BYTES,
                       Q3K_NL,
                       dequantize_q3_K_half)
