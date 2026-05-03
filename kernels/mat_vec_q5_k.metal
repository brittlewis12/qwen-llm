// Q5_K mat-vec.
//
// Block layout (block_q5_K, 176 bytes / 256 elements):
//   half  d                  (super-block scale)
//   half  dmin               (super-block min)
//   u8    scales[12]         (8 sub-block scale+min, 6-bit packed,
//                             same packing as Q4_K)
//   u8    qh[32]             (high bit of each 5-bit quant: 256 bits)
//   u8    qs[128]            (low 4 bits of each quant)
//
// Element value:
//   q5[k] = (qs[k_lo] & 0xF) | ((qh[k_byte] >> bit) & 1) << 4
//   x[k]  = d * sc[k/32] * q5[k] - dmin * min[k/32]
//
// Lifted from llama.cpp's `kernel_mul_mv_q5_K_f32_impl` at
// ggml/src/ggml-metal/ggml-metal.metal:7837. Same lane mapping as
// Q4_K's fast kernel (32-lane simdgroup, packed accumulation, scale/
// min folded into one per-row correction at the end). Difference vs
// Q4_K: we also accumulate `acc2` for the high-bit contribution, which
// adds 16x the y-value when the corresponding qh bit is set.
//
// llama.cpp uses NR0_Q5_K=1 / NSG_Q5_K=2 (one row per simdgroup). We
// match that.

#include <metal_stdlib>
using namespace metal;

constant constexpr int  QK_K      = 256;
constant constexpr int  Q5K_BYTES = 176;

struct mat_vec_q5k_args {
    uint n_in;
    uint n_out;
};

#define NR0_Q5K 1
#define NSG_Q5K 2

kernel void kernel_mat_vec_q5_K_f32(
        constant mat_vec_q5k_args & args   [[buffer(0)]],
        device const uchar        * weight [[buffer(1)]],
        device const float        * x      [[buffer(2)]],
        device       float        * y      [[buffer(3)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // Lane mapping (matches llama.cpp Q5_K kernel):
    //   tid = tiisg/4 in [0, 8)   — picks 1 of 8 (iq, ir) pairs
    //   ix  = tiisg%4 in [0, 4)   — picks 1 of 4 super-blocks per iter
    //   iq  = tid/4   in [0, 2)   — half-block (0=low 128, 1=high 128)
    //   ir  = tid%4   in [0, 4)   — chunk-of-8 within the half
    const ushort tid = tiisg / 4;
    const ushort ix  = tiisg % 4;
    const ushort iq  = tid / 4;
    const ushort ir  = tid % 4;

    const ushort l0 = 8u * ir;
    const ushort q_offset = 32u * iq + l0;
    const ushort y_offset = 64u * iq + l0;

    const uchar hm1 = 1u << (2u * iq);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;

    float yl[16];
    float yh[16];
    float sumf = 0.0f;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = x + ix * QK_K + y_offset;

    for (uint i = ix; i < nb; i += 4) {
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        // For NR0=1 we just have the one row this simdgroup is responsible for.
        device const uchar * blk = weight + (ulong)first_row * row_stride_bytes
                                          + (ulong)i * Q5K_BYTES;
        // Layout: half d, half dmin, u8 scales[12], u8 qh[32], u8 qs[128]
        device const half     * dh = (device const half *) blk;
        device const uint16_t * a  = (device const uint16_t *)(blk + 4) + iq;
        device const uchar    * qh = (blk + 4 + 12) + l0;
        device const uchar    * q1 = (blk + 4 + 12 + 32) + q_offset;
        device const uchar    * q2 = q1 + 64;

        sc16[0] =  a[0]                & kmask1;
        sc16[1] =  a[2]                & kmask1;
        sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
        sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

        float4 acc1 = {0.f, 0.f, 0.f, 0.f};
        float4 acc2 = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            const uchar h = qh[l];
            acc1[0] += yl[l+0] * (q1[l] & 0x0F);
            acc1[1] += yl[l+8] * (q1[l] & 0xF0);
            acc1[2] += yh[l+0] * (q2[l] & 0x0F);
            acc1[3] += yh[l+8] * (q2[l] & 0xF0);
            acc2[0] += (h & hm1) ? yl[l+0] : 0.f;
            acc2[1] += (h & hm2) ? yl[l+8] : 0.f;
            acc2[2] += (h & hm3) ? yh[l+0] : 0.f;
            acc2[3] += (h & hm4) ? yh[l+8] : 0.f;
        }

        sumf += (float)dh[0] * (
              sc8[0] * (acc1[0]        + 16.f * acc2[0])
            + sc8[1] * (acc1[1] / 16.f + 16.f * acc2[1])
            + sc8[4] * (acc1[2]        + 16.f * acc2[2])
            + sc8[5] * (acc1[3] / 16.f + 16.f * acc2[3])
        ) - (float)dh[1] * (
              sumy[0] * sc8[2]
            + sumy[1] * sc8[3]
            + sumy[2] * sc8[6]
            + sumy[3] * sc8[7]
        );

        y1 += 4 * QK_K;
    }

    const float tot = simd_sum(sumf);
    if (tiisg == 0 && first_row < args.n_out) {
        y[first_row] = tot;
    }
}
