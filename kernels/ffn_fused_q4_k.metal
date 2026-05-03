// Fused SwiGLU FFN kernel for Q4_K weights — replaces 3 dispatches:
//   gate = mat_vec_q4_K(W_gate, x)
//   up   = mat_vec_q4_K(W_up, x)
//   inner[i] = silu(gate[i]) * up[i]
//
// Shape: x ∈ R^{n_in}, W_gate, W_up ∈ R^{n_out × n_in} (Q4_K),
//        inner ∈ R^{n_out}.
//
// For each output row, we compute BOTH gate and up dot products in the
// same simdgroup, sharing the input load yl/yh and the sumy reduction.
// This halves the input memory bandwidth vs. dispatching the two
// mat-vecs separately, and eliminates the materialization of the
// `gate` and `up` intermediate buffers (each n_out floats × per layer
// × 48 layers = ~3.3 MB/token of writes+reads avoided).
//
// Per Jeff & Sanjay (Bulk APIs / amortize boundary crossings) and their
// "avoid materializing intermediates" advice — this is the GPU expression
// of both principles in one kernel.
//
// Lifted structure from kernel_mat_vec_q4_K_f32 in mat_vec_q4_k.metal —
// kept the same simdgroup mapping, NR0_Q4K=2 rows per simdgroup,
// NSG_Q4K=2 simdgroups per threadgroup.

#include <metal_stdlib>
using namespace metal;

// Match constants from mat_vec_q4_k.metal so the dequant math is identical.
#define QK_K 256
#define Q4K_BYTES 144  // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) per super-block

#define NR0_Q4K 2
#define NSG_Q4K 2

struct ffn_fused_q4k_args {
    uint n_in;   // input dim (= hidden_size = 5120 for 27B)
    uint n_out;  // intermediate_size (= 17408 for 27B)
};

// Stable SiLU: x / (1 + exp(-x)) using Metal's exp().
inline float silu_f(float x) {
    return x / (1.0f + exp(-x));
}

kernel void kernel_ffn_swiglu_q4_K_f32(
        constant ffn_fused_q4k_args & args   [[buffer(0)]],
        device const uchar          * w_gate [[buffer(1)]], // Q4_K bytes, row-major
        device const uchar          * w_up   [[buffer(2)]], // Q4_K bytes, row-major
        device const float          * x      [[buffer(3)]], // [n_in]
        device       float          * inner  [[buffer(4)]], // [n_out] = silu(W_g·x) * (W_u·x)
        uint   tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;     // 0..3 — picks 1 of 4 super-blocks
    const ushort it = tiisg % 8;     // 0..7 — picks 32 elements within
    const ushort iq = it / 4;        // 0..1 — half-block
    const ushort ir = it % 4;        // 0..3 — chunk-of-8 within half

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;

    // Per-row pointers for both weight matrices.
    device const uchar * row0_g = w_gate + first_row * row_stride_bytes;
    device const uchar * row0_u = w_up   + first_row * row_stride_bytes;

    device const float * y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf_g[NR0_Q4K] = {0.f, 0.f};
    float sumf_u[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        // Load 32 input values per lane — SHARED across both gate/up paths.
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0_Q4K; row++) {
            if (first_row + row >= args.n_out) break;

            // ===== Gate path =====
            {
                device const uchar * blk = row0_g + row * row_stride_bytes
                                          + (ulong)ib * Q4K_BYTES;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
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
                sumf_g[row] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }

            // ===== Up path (identical math, different weight buffer) =====
            {
                device const uchar * blk = row0_u + row * row_stride_bytes
                                          + (ulong)ib * Q4K_BYTES;
                device const half     * dh = (device const half *) blk;
                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq;
                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq + 4 * ir;
                device const uint16_t * q2 = q1 + 32;

                sc16[0] =  sc[0]                & kmask1;
                sc16[1] =  sc[2]                & kmask1;
                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
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
                sumf_u[row] += (float)dh[0] * (
                      (acc1[0] + 1.f/256.f * acc1[1]) * sc8[0]
                    + (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f
                    + (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4]
                    + (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f
                ) - (float)dh[1] * (
                      sumy[0] * sc8[2] + sumy[1] * sc8[3]
                    + sumy[2] * sc8[6] + sumy[3] * sc8[7]
                );
            }
        }

        y4 += 4 * QK_K;
    }

    // Reduce across the 32 lanes for each row, then fuse silu(gate) * up.
    for (short row = 0; row < NR0_Q4K; row++) {
        float total_g = simd_sum(sumf_g[row]);
        float total_u = simd_sum(sumf_u[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            inner[first_row + row] = silu_f(total_g) * total_u;
        }
    }
}
