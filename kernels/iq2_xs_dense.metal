// Dense IQ2_XS F32 register MMA, adapted from llama.cpp ggml-quants.c
// and mul_mv_mma.metal (NT=2, RT=1, NSG=2), via the IQ2_XXS primitive.
/*
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#include <metal_stdlib>
using namespace metal;
#define mv_iq2xs_grid dense_iq2xs_grid
#include "iq2_xs_grid.metalh"
#undef mv_iq2xs_grid

#define IQ2_XS_UNROLL _Pragma("clang loop unroll(full)") for

struct iq2_xs_dense_args {
    uint K;
    uint M;
    uint N;
    uint row_bytes;
};

constant uchar dense_iq2xs_signs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

// Each 74-byte block is only ushort-aligned. A 16-value group has one
// scale nibble and two ushort grid/sign codes; dequantize entirely in F32.
inline void iq2_xs_dequant16(device const uchar * block, uint il, thread float4x4 & w) {
    device const ushort * q = (device const ushort *)(block + 2) + 2u * il;
    const uint scale = (uint(block[66u + il / 2u]) >> (4u * (il % 2u))) & 15u;
    const float dl = float(((device const half *)block)[0]) * (0.5f + float(scale)) * 0.25f;
    IQ2_XS_UNROLL (uint l = 0; l < 2; ++l) {
        const uint packed = uint(q[l]);
        constant uchar * grid = (constant uchar *)(dense_iq2xs_grid + (packed & 511u));
        const uint signs = dense_iq2xs_signs[packed >> 9u];
        IQ2_XS_UNROLL (uint j = 0; j < 8; ++j) {
            w[2u * l + j / 4u][j % 4u] = dl * float(grid[j]) * ((signs & (1u << j)) ? -1.0f : 1.0f);
        }
    }
}

inline uint iq2_xs_fm(uint lane) { return ((lane / 4u) & 4u) + ((lane / 2u) % 4u); }
inline uint iq2_xs_fn(uint lane) { return ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u; }

// Upstream mul_mv_mma_gen with NT=2, RT=1, NSG=2: 16 output rows x 8
// tokens. F32 dequantization, operands and accumulation; no half staging.
kernel void kernel_mat_mat_iq2_xs_f32_mma(
        constant iq2_xs_dense_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        threadgroup float * red [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint fm = iq2_xs_fm(uint(lane));
    const uint fn = iq2_xs_fn(uint(lane));
    device const uchar * rows[2];
    IQ2_XS_UNROLL (uint t = 0; t < 2; ++t) {
        const uint row = min(tgpig.x * 16u + 8u * t + fm, args.M - 1u);
        rows[t] = weight + (ulong)row * args.row_bytes;
    }
    device const float4 * inputs[2];
    IQ2_XS_UNROLL (uint e = 0; e < 2; ++e) {
        const uint token = min(tgpig.y * 8u + fn + e, args.N - 1u);
        inputs[e] = (device const float4 *)(x + (ulong)token * args.K) + 4u * (fm / 2u) + 2u * (fm % 2u);
    }
    simdgroup_float8x8 accum[2];
    IQ2_XS_UNROLL (uint t = 0; t < 2; ++t) accum[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint g = uint(sg); g < args.K / 64u; g += 2u) {
        const float4 a0 = inputs[0][16u * g];
        const float4 a1 = inputs[0][16u * g + 1u];
        const float4 b0 = inputs[1][16u * g];
        const float4 b1 = inputs[1][16u * g + 1u];
        simdgroup_float8x8 mb[8];
        IQ2_XS_UNROLL (uint s = 0; s < 4; ++s) {
            mb[s].thread_elements()[0] = a0[s];
            mb[s].thread_elements()[1] = b0[s];
            mb[s + 4].thread_elements()[0] = a1[s];
            mb[s + 4].thread_elements()[1] = b1[s];
        }
        const uint ci = 4u * g + fn / 2u;
        IQ2_XS_UNROLL (uint t = 0; t < 2; ++t) {
            float4x4 w;
            iq2_xs_dequant16(rows[t] + (ulong)(ci / 16u) * 74u, ci % 16u, w);
            IQ2_XS_UNROLL (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s / 4u][s % 4u];
                ma.thread_elements()[1] = w[s / 4u + 2u][s % 4u];
                simdgroup_multiply_accumulate(accum[t], ma, mb[s], accum[t]);
            }
        }
    }
    IQ2_XS_UNROLL (uint t = 0; t < 2; ++t) {
        red[uint(sg) * 128u + t * 64u + 2u * uint(lane)] = accum[t].thread_elements()[0];
        red[uint(sg) * 128u + t * 64u + 2u * uint(lane) + 1u] = accum[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = uint(sg) * 32u + uint(lane); idx < 128u; idx += 64u) {
        float sum = 0.0f;
        IQ2_XS_UNROLL (uint s = 0; s < 2; ++s) sum += red[s * 128u + idx];
        const uint l = (idx % 64u) / 2u;
        const uint row = tgpig.x * 16u + 8u * (idx / 64u) + iq2_xs_fm(l);
        const uint token = tgpig.y * 8u + iq2_xs_fn(l) + idx % 2u;
        if (row < args.M && token < args.N) y[(ulong)token * args.M + row] = sum;
    }
}
