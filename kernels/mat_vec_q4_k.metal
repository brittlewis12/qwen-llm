// Q4_K mat-vec.
//
// The CPU reference in `forward.rs` is the correctness anchor.
//   * `kernel_mat_vec_q4_K_f32`: lifted in spirit from llama.cpp's
//     `kernel_mul_mv_q4_K_f32_impl` (ggml/src/ggml-metal/ggml-metal.metal:7716).
//     32 lanes cooperate per super-block, 8 lanes per row-pair, packed
//     uint16 nibble loads, scale/min folded into one per-row correction
//     at the end. NSG=2, NR0=2 — i.e. each threadgroup is two simdgroups,
//     each simdgroup produces two output rows simultaneously, sharing the
//     loaded `y` slice across both rows.
//
// Q4_K block layout (block_q4_K, 144 bytes / 256 elements):
//     half  d
//     half  dmin
//     u8    scales[12]    // 6-bit packed (sc, min) for 8 sub-blocks
//     u8    qs[128]       // 4-bit nibbles, paired sub-blocks per byte
//
// Element value: x[k] = (d * sc[j]) * q4[k] - (dmin * min[j]),
//                where j = k/32 ∈ [0, 8).

#include <metal_stdlib>
using namespace metal;

constant constexpr int   QK_K      = 256;
constant constexpr int   Q4K_BYTES = 144;

struct mat_vec_q4k_args {
    uint n_in;
    uint n_out;
};

struct ds4_all_slots_q4k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint top_k;
    float clamp;
};

// ---------------------------------------------------------------------------
// Helper: decode (sc, min) for sub-block j from the 12-byte scales array.
// (kept for the naive kernel's clarity.)
inline void get_scale_min_q4k(int j, device const uchar* scales,
                              thread uchar& sc, thread uchar& m) {
    if (j < 4) {
        sc = scales[j] & 63;
        m  = scales[j + 4] & 63;
    } else {
        sc = (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4);
        m  = (scales[j + 4] >> 4)   | ((scales[j - 0] >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------
// Fast kernel — lift of llama.cpp's `kernel_mul_mv_q4_K_f32_impl`.
//
// 32-lane simdgroup mapping:
//   ix = tiisg / 8   in [0, 4)   — picks 1 of 4 super-blocks per outer iter
//   it = tiisg % 8   in [0, 8)   — picks 32 elements within the super-block
//   iq = it / 4      in [0, 2)   — half-block (0=low 64, 1=high 64)
//   ir = it % 4      in [0, 4)   — chunk-of-8 within the half
//
// Each simdgroup computes NR0=2 output rows. Multiple simdgroups per
// threadgroup; we use NSG=2 so each threadgroup produces 4 output rows.
//
// The dequant is fused: yl/yh hold y values; the loop accumulates
// y * (raw 4-bit weight) into 4-vec acc1/acc2. After the inner loop,
// scale and min are applied ONCE per row, exploiting the fact that
// (d*sc)*x*q - (dmin*min)*x = (d*sc)*(x*q) - (dmin*min)*sumy  per sub-block.

#define NR0_Q4K 2
#define NSG_Q4K 2

kernel void kernel_mat_vec_q4_K_f32(
        constant mat_vec_q4k_args & args   [[buffer(0)]],
        device const uchar        * weight [[buffer(1)]], // raw block_q4_K bytes
        device const float        * x      [[buffer(2)]], // [n_in]
        device       float        * y      [[buffer(3)]], // [n_out]
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K;
    // First output row this simdgroup is responsible for.
    const uint first_row = (tgpig * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;

    // Per-row weight stride (in bytes). nblocks_per_row * Q4K_BYTES.
    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;

    // Pointer to the first row's super-block array.
    device const uchar * row0 = weight + first_row * row_stride_bytes;

    // y starts at the iq-th half (offset 64*iq within super-block) plus
    // 8*ir elements into that half.
    device const float * y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        // Load 32 y values per lane (two halves of 16, four 8-element strides).
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // For each output row in NR0, do its scale/quant work for this
        // super-block.
        for (short row = 0; row < NR0_Q4K; row++) {
            if (first_row + row >= args.n_out) break;

            device const uchar * blk = row0 + row * row_stride_bytes
                                      + (ulong)ib * Q4K_BYTES;
            device const half     * dh = (device const half *) blk;
            device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
            device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
            device const uint16_t * q2 = q1 + 32;

            // Decode the four 6-bit (sc, min) pairs that this lane needs.
            sc16[0] =  sc[0]                & kmask1;
            sc16[1] =  sc[2]                & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            // Unrolled accumulate: pair y with raw 4-bit nibbles in 4 packed lanes.
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += (float)dh[0] * (
                  (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
            ) - (float)dh[1] * (
                  sumy[0] * sc8[2] + sumy[1] * sc8[3]
                + sumy[2] * sc8[6] + sumy[3] * sc8[7]
            );
        }

        y4 += 4 * QK_K;
    }

    // Final reduction across the 32 lanes for each row.
    for (short row = 0; row < NR0_Q4K; row++) {
        float total = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            y[first_row + row] = total;
        }
    }
}

kernel void kernel_deepseek_v4_all_slots_mat_vec_q4_K_f32_fast(
        constant ds4_all_slots_q4k_args & args [[buffer(0)]],
        device const uchar * weight [[buffer(1)]],
        device const float * inner [[buffer(2)]],
        device const int * expert_ids [[buffer(3)]],
        device const int * route_status [[buffer(4)]],
        device float * y [[buffer(5)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;
    const uint slot = tgpig.y;
    const uint first_row = (tgpig.x * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;
    const int expert = slot < args.top_k ? expert_ids[slot] : -1;
    if (slot >= args.top_k || route_status[0] != 1
            || expert < 0 || uint(expert) >= args.n_expert) {
        if (tiisg == 0) {
            for (short row = 0; row < NR0_Q4K; ++row) {
                const uint out_row = first_row + uint(row);
                if (out_row < args.n_out) {
                    y[(ulong)slot * args.n_out + out_row] = 0.0f;
                }
            }
        }
        return;
    }

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;
    const uint nb = args.n_in / QK_K;
    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * row0 = weight + (ulong)expert * expert_stride_bytes
        + first_row * row_stride_bytes;
    device const float * x = inner + (ulong)slot * args.n_in;
    device const float * y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf[NR0_Q4K] = {0.0f, 0.0f};
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.0f, 0.0f, 0.0f, 0.0f};
        for (short i = 0; i < 8; ++i) {
            yl[i + 0] = y4[i +   0]; sumy[0] += yl[i + 0];
            yl[i + 8] = y4[i +  32]; sumy[1] += yl[i + 8];
            yh[i + 0] = y4[i + 128]; sumy[2] += yh[i + 0];
            yh[i + 8] = y4[i + 160]; sumy[3] += yh[i + 8];
        }

        for (short row = 0; row < NR0_Q4K; ++row) {
            if (first_row + uint(row) >= args.n_out) break;
            device const uchar * blk = row0 + row * row_stride_bytes
                + (ulong)ib * Q4K_BYTES;
            device const half * dh = (device const half *)blk;
            device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
            device const uint16_t * q1 =
                (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
            device const uint16_t * q2 = q1 + 32;

            sc16[0] = sc[0] & kmask1;
            sc16[1] = sc[2] & kmask1;
            sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
            sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

            float4 acc1 = {0.0f, 0.0f, 0.0f, 0.0f};
            float4 acc2 = {0.0f, 0.0f, 0.0f, 0.0f};
            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2 * i + 0] * (q1[i] & 0x000F);
                acc1[1] += yl[2 * i + 1] * (q1[i] & 0x0F00);
                acc1[2] += yl[2 * i + 8] * (q1[i] & 0x00F0);
                acc1[3] += yl[2 * i + 9] * (q1[i] & 0xF000);
                acc2[0] += yh[2 * i + 0] * (q2[i] & 0x000F);
                acc2[1] += yh[2 * i + 1] * (q2[i] & 0x0F00);
                acc2[2] += yh[2 * i + 8] * (q2[i] & 0x00F0);
                acc2[3] += yh[2 * i + 9] * (q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * (
                  (acc1[0] + (1.0f / 256.0f) * acc1[1]) * sc8[0]
                + (acc1[2] + (1.0f / 256.0f) * acc1[3]) * sc8[1] * (1.0f / 16.0f)
                + (acc2[0] + (1.0f / 256.0f) * acc2[1]) * sc8[4]
                + (acc2[2] + (1.0f / 256.0f) * acc2[3]) * sc8[5] * (1.0f / 16.0f)
            ) - float(dh[1]) * (
                  sumy[0] * sc8[2] + sumy[1] * sc8[3]
                + sumy[2] * sc8[6] + sumy[3] * sc8[7]
            );
        }
        y4 += 4 * QK_K;
    }

    for (short row = 0; row < NR0_Q4K; ++row) {
        const uint out_row = first_row + uint(row);
        if (out_row >= args.n_out) continue;
        const float total = simd_sum(sumf[row]);
        if (tiisg == 0) y[(ulong)slot * args.n_out + out_row] = total;
    }
}
