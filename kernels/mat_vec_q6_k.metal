// Q6_K mat-vec — bartowski's UD-quant ships output_proj, attn_v, ffn_down
// at Q6_K for higher precision in the model's most sensitive matmul-output
// positions. We need this kernel for any 27B-Q4_K_M end-to-end forward.
//
// Q6_K block layout (block_q6_K, 210 bytes / 256 elements):
//     u8   ql[128]           // low 4 bits of every quant
//     u8   qh[64]            // high 2 bits of every quant (4 quants per byte)
//     i8   scales[16]        // per-16-element scales
//     half d                 // super-block scale
//
// Element value: x[k] = d * scales[k/16] * (q6[k] - 32)  where
//   q6[k] = (ql[k_lo] & 0xF) | ((qh[k_qh] >> shift) & 0x3) << 4
// with packing details lifted from llama.cpp's kernel below.
//
// Lifted from `kernel_mul_mv_q6_K_f32_impl` at
// ggml/src/ggml-metal/ggml-metal.metal:7968. Same NSG=2, NR0=2 design as
// our Q4_K kernel.

#include <metal_stdlib>
using namespace metal;

constant constexpr int   QK_K      = 256;
constant constexpr int   Q6K_BYTES = 210;

struct mat_vec_q6k_args {
    uint n_in;
    uint n_out;
};

#define NR0_Q6K 2
#define NSG_Q6K 2

kernel void kernel_mat_vec_q6_K_f32(
        constant mat_vec_q6k_args & args   [[buffer(0)]],
        device const uchar        * weight [[buffer(1)]],
        device const float        * x      [[buffer(2)]],
        device       float        * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig * NSG_Q6K + sgitg) * NR0_Q6K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    // Lane mapping (matches llama.cpp's Q6_K kernel):
    // tid = tiisg/2  ∈ [0, 16)  picks one of 16 (ip, il) pairs
    // ix  = tiisg%2  ∈ [0, 2)   picks 1 of 2 super-blocks per iter
    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;        // 0 or 1: which 128-element half
    const ushort il  = tid % 8;        // 0..7: 4-element chunk inside that half
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float sumf[NR0_Q6K] = {0.f, 0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const float * y_blk = x + (ulong)i * QK_K + y_offset;

        // Pre-load 16 y values that this lane needs (4 chunks of 4).
        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y_blk[l +  0];
            yl[4*l + 1] = y_blk[l + 32];
            yl[4*l + 2] = y_blk[l + 64];
            yl[4*l + 3] = y_blk[l + 96];
        }

        for (short row = 0; row < NR0_Q6K; ++row) {
            if (first_row + row >= args.n_out) break;

            device const uchar * blk = row0 + row * row_stride_bytes
                                      + (ulong)i * Q6K_BYTES;
            // Layout in block: ql[128], qh[64], scales[16], d (half).
            device const uchar * q1 = blk + q_offset_l;        // ql start
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = blk + 128 + q_offset_h;
            device const int8_t * sc = (device const int8_t *)(blk + 128 + 64) + is;
            device const half   * dh = (device const half *)(blk + 128 + 64 + 16);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += (float)dh[0] * (
                  sums[0] * (float)sc[0]
                + sums[1] * (float)sc[2]
                + sums[2] * (float)sc[4]
                + sums[3] * (float)sc[6]
            );
        }
    }

    for (short row = 0; row < NR0_Q6K; ++row) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

// Token-axis form of the exact singleton body above. Grid Y selects an
// independent activation/output row; every lane mapping and accumulation step
// within a row remains expression-identical to kernel_mat_vec_q6_K_f32.
kernel void kernel_mat_vec_q6_K_f32_batch(
        constant mat_vec_q6k_args & args   [[buffer(0)]],
        device const uchar        * weight [[buffer(1)]],
        device const float        * x      [[buffer(2)]],
        device       float        * y      [[buffer(3)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q6K + sgitg) * NR0_Q6K;
    if (first_row >= args.n_out) return;

    x += (ulong)tgpig.y * args.n_in;
    y += (ulong)tgpig.y * args.n_out;
    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;
    const ushort il  = tid % 8;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float sumf[NR0_Q6K] = {0.f, 0.f};
    float yl[16];

    for (uint i = ix; i < nb; i += 2) {
        device const float * y_blk = x + (ulong)i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y_blk[l +  0];
            yl[4*l + 1] = y_blk[l + 32];
            yl[4*l + 2] = y_blk[l + 64];
            yl[4*l + 3] = y_blk[l + 96];
        }

        for (short row = 0; row < NR0_Q6K; ++row) {
            if (first_row + row >= args.n_out) break;

            device const uchar * blk = row0 + row * row_stride_bytes
                                      + (ulong)i * Q6K_BYTES;
            device const uchar * q1 = blk + q_offset_l;
            device const uchar * q2 = q1 + 32;
            device const uchar * qh = blk + 128 + q_offset_h;
            device const int8_t * sc = (device const int8_t *)(blk + 128 + 64) + is;
            device const half   * dh = (device const half *)(blk + 128 + 64 + 16);

            float4 sums = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }
            sumf[row] += (float)dh[0] * (
                  sums[0] * (float)sc[0]
                + sums[1] * (float)sc[2]
                + sums[2] * (float)sc[4]
                + sums[3] * (float)sc[6]
            );
        }
    }

    for (short row = 0; row < NR0_Q6K; ++row) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}
