// IQ4_NL mat-mat (W * X^T -> Y^T), using the shared 64x32x32 simdgroup_matrix
// tile from mat_mat_mm_tile.h; this file owns only the IQ4_NL dequant.
//
// QK=32 / NL=2: the shared tile's uniform advance rule degenerates to
// "il stays il0; advance one 18-byte block per K-step", which is exactly
// what the original per-file copy hand-coded.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include "mat_mat_mm_tile.h"

constant constexpr int IQ4NL_QK       = 32;
constant constexpr int IQ4NL_BYTES    = 18;
constant constexpr int IQ4NL_NL       = IQ4NL_QK / 16;

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

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_iq4_nl_f32_mm,
                       mat_mat_iq4nl_args,
                       IQ4NL_BYTES,
                       IQ4NL_NL,
                       dequantize_iq4_nl_half)
