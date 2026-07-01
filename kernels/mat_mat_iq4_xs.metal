// IQ4_XS mat-mat (W * X^T -> Y^T), using the shared 64x32x32 simdgroup_matrix
// tile from mat_mat_mm_tile.h; this file owns only the IQ4_XS dequant.

#include <metal_stdlib>
using namespace metal;

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") for (x)

#include "mat_mat_mm_tile.h"

constant constexpr int IQ4XS_QK       = 256;
constant constexpr int IQ4XS_BYTES    = 136;
constant constexpr int IQ4XS_NL       = IQ4XS_QK / 16;

constant float iq4xs_values_mm[16] = {
    -127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f,
       1.0f,   13.0f,  25.0f,  38.0f,  53.0f,  69.0f,  89.0f, 113.0f,
};

struct mat_mat_iq4xs_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

inline void dequantize_iq4_xs_half(device const uchar * blk_bytes,
                                   short il,
                                   thread half4x4 & reg) {
    const half d_h = *((device const half *)blk_bytes);
    const ushort scales_h = *((device const ushort *)(blk_bytes + 2));
    device const uchar * scales_l = blk_bytes + 4;
    device const uchar * qs = blk_bytes + 8;

    const uint ib32 = uint(il >> 1);
    const uint scale_l = (uint(scales_l[ib32 >> 1]) >> (4u * (ib32 & 1u))) & 0x0fu;
    const uint scale_h = (uint(scales_h) >> (2u * ib32)) & 3u;
    const float d = float(d_h) * float(int(scale_l | (scale_h << 4)) - 32);
    const bool hi = (il & 1) != 0;
    device const uchar * q = qs + ib32 * 16u;

    FOR_UNROLL (int i = 0; i < 16; ++i) {
        const uint idx = hi ? uint(q[i] >> 4) : uint(q[i] & 0x0f);
        reg[i / 4][i % 4] = (half)(d * iq4xs_values_mm[idx]);
    }
}

MAT_MAT_MM_TILE_KERNEL(kernel_mat_mat_iq4_xs_f32_mm,
                       mat_mat_iq4xs_args,
                       IQ4XS_BYTES,
                       IQ4XS_NL,
                       dequantize_iq4_xs_half)
