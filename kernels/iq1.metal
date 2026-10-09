// Dense IQ1. Canonical dequantization and the F32 register-MMA mapping
// are adapted from llama.cpp ggml-quants.c / dequantize.h / mul_mv_mma.metal.
// Copyright (c) 2023-2026 The ggml authors. MIT license in iq1_grid.metalh.
#include <metal_stdlib>
using namespace metal;
#include "iq1_grid.metalh"

#define IQ1_UNROLL _Pragma("clang loop unroll(full)") for

struct iq1_args {
    uint K;
    uint M;
    uint N;
    uint row_bytes;
    uint is_m;
};

// Only ushort loads: IQ1_S blocks have a 50-byte stride. IQ1_M's half
// scale is distributed over the high nibbles of four ushort scale words.
inline constant char * iq1_grid8(device const uchar * block, uint group8,
        bool is_m, thread float & dl, thread float & delta) {
    uint code;
    if (is_m) {
        device const ushort * sc = (device const ushort *)(block + 48);
        const ushort bits = ushort((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u)
            | ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u));
        const uint qh = uint(block[32u + group8 / 2u]) >> (4u * (group8 % 2u));
        code = uint(block[group8]) | ((qh & 7u) << 8);
        const uint scale = (uint(sc[group8 / 8u]) >> (3u * ((group8 / 2u) % 4u))) & 7u;
        dl = float(as_type<half>(bits)) * float(2u * scale + 1u);
        delta = (qh & 8u) ? -0.125f : 0.125f;
    } else {
        const uint qh = ((device const ushort *)(block + 34))[group8 / 4u];
        code = uint(block[2u + group8]) | (((qh >> (3u * (group8 % 4u))) & 7u) << 8);
        dl = float(((device const half *)block)[0]) * float(2u * ((qh >> 12) & 7u) + 1u);
        delta = (qh & 0x8000u) ? -0.125f : 0.125f;
    }
    return (constant char *)(iq1_grid + code);
}

inline float iq1_value(device const uchar * block, uint i, bool is_m) {
    float dl, delta;
    constant char * grid = iq1_grid8(block, i / 8u, is_m, dl, delta);
    return dl * (float(grid[i % 8u]) + delta);
}

inline void iq1_dequant16(device const uchar * block, uint il, bool is_m, thread float4x4 & w) {
    IQ1_UNROLL (uint l = 0; l < 2; ++l) {
        float dl, delta;
        constant char * grid = iq1_grid8(block, 2u * il + l, is_m, dl, delta);
        IQ1_UNROLL (uint j = 0; j < 8; ++j) {
            w[2u * l + j / 4u][j % 4u] = dl * (float(grid[j]) + delta);
        }
    }
}

kernel void kernel_mat_vec_iq1_f32(
        constant iq1_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint first_row = (tgpig * 2u + uint(sg)) * 4u;
    if (first_row >= args.M) return;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint ib32 = uint(lane); ib32 < args.K / 32u; ib32 += 32u) {
        float values[32];
        IQ1_UNROLL (uint j = 0; j < 32; ++j) values[j] = x[(ulong)ib32 * 32u + j];
        IQ1_UNROLL (uint r = 0; r < 4; ++r) {
            if (first_row + r < args.M) {
                device const uchar * block = weight + (ulong)(first_row + r) * args.row_bytes + (ulong)(ib32 / 8u) * (args.is_m ? 56u : 50u);
                IQ1_UNROLL (uint h = 0; h < 2; ++h) {
                    float4x4 w;
                    iq1_dequant16(block, 2u * (ib32 % 8u) + h, args.is_m, w);
                    IQ1_UNROLL (uint j = 0; j < 16; ++j) sums[r] += w[j / 4u][j % 4u] * values[16u * h + j];
                }
            }
        }
    }
    IQ1_UNROLL (uint r = 0; r < 4; ++r) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0 && first_row + r < args.M) y[first_row + r] = sum;
    }
}

// One output row by 32 tokens. Inactive token lanes still stage weights and
// participate in both barriers; only the dot and output store are masked.
kernel void kernel_mat_mat_iq1_f32_scalar(
        constant iq1_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        threadgroup float * tile [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint row = tgpig.x;
    const uint token = tgpig.y * 32u + uint(lane);
    float sum = 0.0f;
    for (uint k0 = 0; k0 < args.K; k0 += 32u) {
        device const uchar * block = weight + (ulong)row * args.row_bytes + (ulong)(k0 / 256u) * (args.is_m ? 56u : 50u);
        tile[lane] = iq1_value(block, k0 % 256u + uint(lane), args.is_m);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < args.N) {
            device const float * input = x + (ulong)token * args.K + k0;
            for (uint j = 0; j < 32; ++j) sum += input[j] * tile[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < args.N) y[(ulong)token * args.M + row] = sum;
}

inline uint iq1_fm(uint lane) { return ((lane / 4u) & 4u) + ((lane / 2u) % 4u); }
inline uint iq1_fn(uint lane) { return ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u; }

// Upstream mul_mv_mma_gen with NT=2, RT=1, NSG=2: 16 output rows x 8
// tokens. F32 dequantization, operands and accumulation; no half staging.
kernel void kernel_mat_mat_iq1_f32_mma(
        constant iq1_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        threadgroup float * red [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint fm = iq1_fm(uint(lane));
    const uint fn = iq1_fn(uint(lane));
    device const uchar * rows[2];
    IQ1_UNROLL (uint t = 0; t < 2; ++t) {
        const uint row = min(tgpig.x * 16u + 8u * t + fm, args.M - 1u);
        rows[t] = weight + (ulong)row * args.row_bytes;
    }
    device const float4 * inputs[2];
    IQ1_UNROLL (uint e = 0; e < 2; ++e) {
        const uint token = min(tgpig.y * 8u + fn + e, args.N - 1u);
        inputs[e] = (device const float4 *)(x + (ulong)token * args.K) + 4u * (fm / 2u) + 2u * (fm % 2u);
    }
    simdgroup_float8x8 accum[2];
    IQ1_UNROLL (uint t = 0; t < 2; ++t) accum[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint g = uint(sg); g < args.K / 64u; g += 2u) {
        const float4 a0 = inputs[0][16u * g];
        const float4 a1 = inputs[0][16u * g + 1u];
        const float4 b0 = inputs[1][16u * g];
        const float4 b1 = inputs[1][16u * g + 1u];
        simdgroup_float8x8 mb[8];
        IQ1_UNROLL (uint s = 0; s < 4; ++s) {
            mb[s].thread_elements()[0] = a0[s];
            mb[s].thread_elements()[1] = b0[s];
            mb[s + 4].thread_elements()[0] = a1[s];
            mb[s + 4].thread_elements()[1] = b1[s];
        }
        const uint ci = 4u * g + fn / 2u;
        IQ1_UNROLL (uint t = 0; t < 2; ++t) {
            float4x4 w;
            iq1_dequant16(rows[t] + (ulong)(ci / 16u) * (args.is_m ? 56u : 50u), ci % 16u, args.is_m, w);
            IQ1_UNROLL (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s / 4u][s % 4u];
                ma.thread_elements()[1] = w[s / 4u + 2u][s % 4u];
                simdgroup_multiply_accumulate(accum[t], ma, mb[s], accum[t]);
            }
        }
    }
    IQ1_UNROLL (uint t = 0; t < 2; ++t) {
        red[uint(sg) * 128u + t * 64u + 2u * uint(lane)] = accum[t].thread_elements()[0];
        red[uint(sg) * 128u + t * 64u + 2u * uint(lane) + 1u] = accum[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = uint(sg) * 32u + uint(lane); idx < 128u; idx += 64u) {
        float sum = 0.0f;
        IQ1_UNROLL (uint s = 0; s < 2; ++s) sum += red[s * 128u + idx];
        const uint l = (idx % 64u) / 2u;
        const uint row = tgpig.x * 16u + 8u * (idx / 64u) + iq1_fm(l);
        const uint token = tgpig.y * 8u + iq1_fn(l) + idx % 2u;
        if (row < args.M && token < args.N) y[(ulong)token * args.M + row] = sum;
    }
}

// IDs are signed I32. Invalid IDs produce zero without forming a bank address.
kernel void kernel_get_rows_iq1_m_f32(
        constant iq1_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const int * ids [[buffer(2)]],
        device float * y [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= args.K || gid.y >= args.N) return;
    const int row = ids[gid.y];
    const ulong output = (ulong)gid.y * args.K + gid.x;
    if (row < 0 || uint(row) >= args.M) {
        y[output] = 0.0f;
        return;
    }
    device const uchar * block = weight + (ulong)uint(row) * args.row_bytes + (ulong)(gid.x / 256u) * 56u;
    y[output] = iq1_value(block, gid.x % 256u, true);
}
