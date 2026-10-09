// Dense IQ2_XXS. Canonical dequantization and the F32 register-MMA mapping
// are adapted from llama.cpp ggml-quants.c / dequantize.h / mul_mv_mma.metal.
// Copyright (c) 2023-2026 The ggml authors. MIT license in iq2_xxs_grid.metalh.
#include <metal_stdlib>
using namespace metal;
#include "iq2_xxs_grid.metalh"

#define IQ2_XXS_UNROLL _Pragma("clang loop unroll(full)") for

struct iq2_xxs_args {
    uint K;
    uint M;
    uint N;
    uint row_bytes;
};

// Blocks are 66 bytes: metadata is only ushort-aligned, never uint-aligned.
inline uint iq2_xxs_metadata(device const ushort * q) {
    return uint(q[2]) | (uint(q[3]) << 16);
}

inline float iq2_xxs_value(device const uchar * block, uint i) {
    device const ushort * q = (device const ushort *)(block + 2) + 4u * (i / 32u);
    const uint meta = iq2_xxs_metadata(q);
    const uint l = (i % 32u) / 8u;
    const uint j = i % 8u;
    const uint code = ((device const uchar *)q)[l];
    constant uchar * grid = (constant uchar *)(iq2_xxs_grid + code);
    const uint signs = iq2_xxs_signs[(meta >> (7u * l)) & 127u];
    const float dl = float(((device const half *)block)[0]) * (0.5f + float(meta >> 28)) * 0.25f;
    return dl * float(grid[j]) * ((signs & (1u << j)) ? -1.0f : 1.0f);
}

inline void iq2_xxs_dequant16(device const uchar * block, uint il, thread float4x4 & w) {
    device const ushort * q = (device const ushort *)(block + 2) + 4u * (il / 2u);
    const uint meta = iq2_xxs_metadata(q);
    const float dl = float(((device const half *)block)[0]) * (0.5f + float(meta >> 28)) * 0.25f;
    IQ2_XXS_UNROLL (uint l = 0; l < 2; ++l) {
        const uint part = 2u * (il % 2u) + l;
        constant uchar * grid = (constant uchar *)(iq2_xxs_grid + ((device const uchar *)q)[part]);
        const uint signs = iq2_xxs_signs[(meta >> (7u * part)) & 127u];
        IQ2_XXS_UNROLL (uint j = 0; j < 8; ++j) {
            w[2u * l + j / 4u][j % 4u] = dl * float(grid[j]) * ((signs & (1u << j)) ? -1.0f : 1.0f);
        }
    }
}

kernel void kernel_mat_vec_iq2_xxs_f32(
        constant iq2_xxs_args & args [[buffer(0)]],
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
        IQ2_XXS_UNROLL (uint j = 0; j < 32; ++j) values[j] = x[(ulong)ib32 * 32u + j];
        IQ2_XXS_UNROLL (uint r = 0; r < 4; ++r) {
            if (first_row + r < args.M) {
                device const uchar * block = weight + (ulong)(first_row + r) * args.row_bytes + (ulong)(ib32 / 8u) * 66u;
                IQ2_XXS_UNROLL (uint h = 0; h < 2; ++h) {
                    float4x4 w;
                    iq2_xxs_dequant16(block, 2u * (ib32 % 8u) + h, w);
                    IQ2_XXS_UNROLL (uint j = 0; j < 16; ++j) sums[r] += w[j / 4u][j % 4u] * values[16u * h + j];
                }
            }
        }
    }
    IQ2_XXS_UNROLL (uint r = 0; r < 4; ++r) {
        const float sum = simd_sum(sums[r]);
        if (lane == 0 && first_row + r < args.M) y[first_row + r] = sum;
    }
}

// One output row by 32 tokens. Inactive token lanes still stage weights and
// participate in both barriers; only the dot and output store are masked.
kernel void kernel_mat_mat_iq2_xxs_f32_scalar(
        constant iq2_xxs_args & args [[buffer(0)]],
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
        device const uchar * block = weight + (ulong)row * args.row_bytes + (ulong)(k0 / 256u) * 66u;
        tile[lane] = iq2_xxs_value(block, k0 % 256u + uint(lane));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (token < args.N) {
            device const float * input = x + (ulong)token * args.K + k0;
            for (uint j = 0; j < 32; ++j) sum += input[j] * tile[j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (token < args.N) y[(ulong)token * args.M + row] = sum;
}

inline uint iq2_xxs_fm(uint lane) { return ((lane / 4u) & 4u) + ((lane / 2u) % 4u); }
inline uint iq2_xxs_fn(uint lane) { return ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u; }

// Upstream mul_mv_mma_gen with NT=2, RT=1, NSG=2: 16 output rows x 8
// tokens. F32 dequantization, operands and accumulation; no half staging.
kernel void kernel_mat_mat_iq2_xxs_f32_mma(
        constant iq2_xxs_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * x [[buffer(2)]],
        device float * y [[buffer(3)]],
        threadgroup float * red [[threadgroup(0)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint fm = iq2_xxs_fm(uint(lane));
    const uint fn = iq2_xxs_fn(uint(lane));
    device const uchar * rows[2];
    IQ2_XXS_UNROLL (uint t = 0; t < 2; ++t) {
        const uint row = min(tgpig.x * 16u + 8u * t + fm, args.M - 1u);
        rows[t] = weight + (ulong)row * args.row_bytes;
    }
    device const float4 * inputs[2];
    IQ2_XXS_UNROLL (uint e = 0; e < 2; ++e) {
        const uint token = min(tgpig.y * 8u + fn + e, args.N - 1u);
        inputs[e] = (device const float4 *)(x + (ulong)token * args.K) + 4u * (fm / 2u) + 2u * (fm % 2u);
    }
    simdgroup_float8x8 accum[2];
    IQ2_XXS_UNROLL (uint t = 0; t < 2; ++t) accum[t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    for (uint g = uint(sg); g < args.K / 64u; g += 2u) {
        const float4 a0 = inputs[0][16u * g];
        const float4 a1 = inputs[0][16u * g + 1u];
        const float4 b0 = inputs[1][16u * g];
        const float4 b1 = inputs[1][16u * g + 1u];
        simdgroup_float8x8 mb[8];
        IQ2_XXS_UNROLL (uint s = 0; s < 4; ++s) {
            mb[s].thread_elements()[0] = a0[s];
            mb[s].thread_elements()[1] = b0[s];
            mb[s + 4].thread_elements()[0] = a1[s];
            mb[s + 4].thread_elements()[1] = b1[s];
        }
        const uint ci = 4u * g + fn / 2u;
        IQ2_XXS_UNROLL (uint t = 0; t < 2; ++t) {
            float4x4 w;
            iq2_xxs_dequant16(rows[t] + (ulong)(ci / 16u) * 66u, ci % 16u, w);
            IQ2_XXS_UNROLL (uint s = 0; s < 8; ++s) {
                simdgroup_float8x8 ma;
                ma.thread_elements()[0] = w[s / 4u][s % 4u];
                ma.thread_elements()[1] = w[s / 4u + 2u][s % 4u];
                simdgroup_multiply_accumulate(accum[t], ma, mb[s], accum[t]);
            }
        }
    }
    IQ2_XXS_UNROLL (uint t = 0; t < 2; ++t) {
        red[uint(sg) * 128u + t * 64u + 2u * uint(lane)] = accum[t].thread_elements()[0];
        red[uint(sg) * 128u + t * 64u + 2u * uint(lane) + 1u] = accum[t].thread_elements()[1];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = uint(sg) * 32u + uint(lane); idx < 128u; idx += 64u) {
        float sum = 0.0f;
        IQ2_XXS_UNROLL (uint s = 0; s < 2; ++s) sum += red[s * 128u + idx];
        const uint l = (idx % 64u) / 2u;
        const uint row = tgpig.x * 16u + 8u * (idx / 64u) + iq2_xxs_fm(l);
        const uint token = tgpig.y * 8u + iq2_xxs_fn(l) + idx % 2u;
        if (row < args.M && token < args.N) y[(ulong)token * args.M + row] = sum;
    }
}
