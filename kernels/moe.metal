// MoE expert-bank kernels for single-token decode.
//
// The qwen35moe A3B FFN selects top-k experts per token. The dense path can
// bind one 2D weight tensor per mat-vec, but MoE needs the expert id to choose
// a 2D slice from a 3D expert bank while staying GPU-resident.

#include <metal_stdlib>
using namespace metal;

#define QK_K 256
#define Q4K_BYTES 144
#define Q5K_BYTES 176
#define Q6K_BYTES 210

#define NR0_Q4K 2
#define NSG_Q4K 2
#define NR0_Q5K 1
#define NSG_Q5K 2
#define NR0_Q6K 2
#define NSG_Q6K 2

struct moe_q4k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_q5k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_q6k_args {
    uint n_in;
    uint n_out;
    uint n_expert;
    uint topk;
};

struct moe_sum_args {
    uint n_out;
    uint topk;
};

struct axpy_scalar_args {
    uint n;
};

struct topk_logits_args {
    uint n;
    uint k;
};

struct dot_sigmoid_args {
    uint n;
};

struct topk_dot_sigmoid_args {
    uint n_expert;
    uint topk;
    uint hidden;
};

inline float moe_silu_f(float x) {
    return x / (1.0f + exp(-x));
}

kernel void kernel_topk_logits_softmax_f32(
        constant topk_logits_args & args [[buffer(0)]],
        device const float       * logits [[buffer(1)]],
        device       int         * out_idx [[buffer(2)]],
        device       float       * out_w   [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    const uint MAX_K = 16;
    if (args.k == 0 || args.k > MAX_K) return;

    int top_idx[MAX_K];
    float top_val[MAX_K];
    for (uint i = 0; i < args.k; ++i) {
        top_idx[i] = -1;
        top_val[i] = -INFINITY;
    }

    for (uint i = 0; i < args.n; ++i) {
        const float v = logits[i];
        for (uint j = 0; j < args.k; ++j) {
            const bool better = (v > top_val[j]) || (v == top_val[j] && (top_idx[j] < 0 || int(i) < top_idx[j]));
            if (better) {
                for (uint m = args.k - 1; m > j; --m) {
                    top_val[m] = top_val[m - 1];
                    top_idx[m] = top_idx[m - 1];
                }
                top_val[j] = v;
                top_idx[j] = int(i);
                break;
            }
        }
    }

    const float max_top = top_val[0];
    float sum = 0.0f;
    float exp_val[MAX_K];
    for (uint i = 0; i < args.k; ++i) {
        exp_val[i] = top_idx[i] >= 0 ? exp(top_val[i] - max_top) : 0.0f;
        sum += exp_val[i];
    }
    sum = max(sum, 6.103515625e-5f);
    for (uint i = 0; i < args.k; ++i) {
        out_idx[i] = max(top_idx[i], 0);
        out_w[i] = exp_val[i] / sum;
    }
}

kernel void kernel_dot_sigmoid_f32(
        constant dot_sigmoid_args & args [[buffer(0)]],
        device const float       * weight [[buffer(1)]],
        device const float       * x      [[buffer(2)]],
        device       float       * out    [[buffer(3)]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    float sum = 0.0f;
    for (uint i = tiisg; i < args.n; i += 32) {
        sum += weight[i] * x[i];
    }
    sum = simd_sum(sum);
    if (tiisg == 0) {
        out[0] = 1.0f / (1.0f + exp(-sum));
    }
}

inline bool moe_better_pair(float cand_v, int cand_i, float best_v, int best_i) {
    return cand_i >= 0 && (best_i < 0 || cand_v > best_v || (cand_v == best_v && cand_i < best_i));
}

kernel void kernel_topk_logits_softmax_dot_sigmoid_f32(
        constant topk_dot_sigmoid_args & args          [[buffer(0)]],
        device const float             * logits        [[buffer(1)]],
        device const float             * shared_weight [[buffer(2)]],
        device const float             * x             [[buffer(3)]],
        device       int               * out_idx       [[buffer(4)]],
        device       float             * out_w         [[buffer(5)]],
        device       float             * shared_out    [[buffer(6)]],
        threadgroup  float             * sh_score      [[threadgroup(0)]],
        threadgroup  float             * red_val       [[threadgroup(1)]],
        threadgroup  int               * red_idx       [[threadgroup(2)]],
        uint  tid [[thread_position_in_threadgroup]],
        uint  ntg [[threads_per_threadgroup]]) {
    const uint MAX_K = 16;
    if (args.n_expert > ntg || args.topk == 0 || args.topk > MAX_K) return;

    float shared_sum = 0.0f;
    for (uint i = tid; i < args.hidden; i += ntg) {
        shared_sum += shared_weight[i] * x[i];
    }
    red_val[tid] = shared_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_val[tid] += red_val[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_out[0] = 1.0f / (1.0f + exp(-red_val[0]));

    sh_score[tid] = tid < args.n_expert ? logits[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < args.topk; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = tid < args.n_expert ? int(tid) : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float cand_v = red_val[tid + stride];
                const int cand_i = red_idx[tid + stride];
                if (moe_better_pair(cand_v, cand_i, red_val[tid], red_idx[tid])) {
                    red_val[tid] = cand_v;
                    red_idx[tid] = cand_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            out_idx[slot] = max(red_idx[0], 0);
            out_w[slot] = red_val[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) == red_idx[0]) sh_score[tid] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float max_top = out_w[0];
        float sum = 0.0f;
        float exp_val[MAX_K];
        for (uint i = 0; i < args.topk; ++i) {
            exp_val[i] = exp(out_w[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.topk; ++i) {
            out_w[i] = exp_val[i] / sum;
        }
    }
}

kernel void kernel_topk_logits_softmax_dot_sigmoid_packed_f32(
        constant topk_dot_sigmoid_args & args          [[buffer(0)]],
        device const float             * logits        [[buffer(1)]],
        device const float             * shared_weight [[buffer(2)]],
        device const float             * x             [[buffer(3)]],
        device       int               * out_idx       [[buffer(4)]],
        device       float             * out_w         [[buffer(5)]],
        device       float             * shared_out    [[buffer(6)]],
        threadgroup  float             * sh_score      [[threadgroup(0)]],
        threadgroup  float             * red_val       [[threadgroup(1)]],
        threadgroup  int               * red_idx       [[threadgroup(2)]],
        uint2 tgpig [[threadgroup_position_in_grid]],
        uint2 tid2 [[thread_position_in_threadgroup]],
        uint2 ntg2 [[threads_per_threadgroup]]) {
    const uint token = tgpig.y;
    const uint tid = tid2.x;
    const uint ntg = ntg2.x;
    const uint MAX_K = 16;
    if (args.n_expert > ntg || args.topk == 0 || args.topk > MAX_K) return;

    device const float * logits_t = logits + (ulong)token * args.n_expert;
    device const float * x_t = x + (ulong)token * args.hidden;
    device int * out_idx_t = out_idx + (ulong)token * args.topk;
    device float * out_w_t = out_w + (ulong)token * args.topk;

    float shared_sum = 0.0f;
    for (uint i = tid; i < args.hidden; i += ntg) {
        shared_sum += shared_weight[i] * x_t[i];
    }
    red_val[tid] = shared_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) red_val[tid] += red_val[tid + stride];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) shared_out[token] = 1.0f / (1.0f + exp(-red_val[0]));

    sh_score[tid] = tid < args.n_expert ? logits_t[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < args.topk; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = tid < args.n_expert ? int(tid) : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = ntg >> 1; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float cand_v = red_val[tid + stride];
                const int cand_i = red_idx[tid + stride];
                if (moe_better_pair(cand_v, cand_i, red_val[tid], red_idx[tid])) {
                    red_val[tid] = cand_v;
                    red_idx[tid] = cand_i;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tid == 0) {
            out_idx_t[slot] = max(red_idx[0], 0);
            out_w_t[slot] = red_val[0];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (int(tid) == red_idx[0]) sh_score[tid] = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0) {
        const float max_top = out_w_t[0];
        float sum = 0.0f;
        float exp_val[MAX_K];
        for (uint i = 0; i < args.topk; ++i) {
            exp_val[i] = exp(out_w_t[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.topk; ++i) {
            out_w_t[i] = exp_val[i] / sum;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32(
        constant moe_q4k_args & args    [[buffer(0)]],
        device const uchar    * w_gate  [[buffer(1)]],
        device const uchar    * w_up    [[buffer(2)]],
        device const float    * x       [[buffer(3)]],
        device const int      * top_idx [[buffer(4)]],
        device       float    * inner   [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_gate = w_gate + (ulong)expert_i * expert_stride_bytes;
    device const uchar * expert_up   = w_up   + (ulong)expert_i * expert_stride_bytes;

    device const uchar * row0_g = expert_gate + (ulong)first_row * row_stride_bytes;
    device const uchar * row0_u = expert_up   + (ulong)first_row * row_stride_bytes;

    device const float * y4 = x + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf_g[NR0_Q4K] = {0.f, 0.f};
    float sumf_u[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0_Q4K; row++) {
            if (first_row + row >= args.n_out) break;

            {
                device const uchar * blk = row0_g + row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
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

            {
                device const uchar * blk = row0_u + row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
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

    for (short row = 0; row < NR0_Q4K; row++) {
        float total_g = simd_sum(sumf_g[row]);
        float total_u = simd_sum(sumf_u[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            inner[(ulong)slot * args.n_out + first_row + row] = moe_silu_f(total_g) * total_u;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_packed_slots(
        constant moe_q4k_args & args    [[buffer(0)]],
        device const uchar    * w_gate  [[buffer(1)]],
        device const uchar    * w_up    [[buffer(2)]],
        device const float    * x_pack  [[buffer(3)]],
        device const int      * top_idx [[buffer(4)]],
        device       float    * inner   [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix = tiisg / 8;
    const ushort it = tiisg % 8;
    const ushort iq = it / 4;
    const ushort ir = it % 4;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q4K + sgitg) * NR0_Q4K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q4K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_gate = w_gate + (ulong)expert_i * expert_stride_bytes;
    device const uchar * expert_up   = w_up   + (ulong)expert_i * expert_stride_bytes;

    device const uchar * row0_g = expert_gate + (ulong)first_row * row_stride_bytes;
    device const uchar * row0_u = expert_up   + (ulong)first_row * row_stride_bytes;
    const uint token = slot / args.topk;
    device const float * y4 = x_pack + (ulong)token * args.n_in + ix * QK_K + 64u * iq + 8u * ir;

    float yl[16];
    float yh[16];
    float sumf_g[NR0_Q4K] = {0.f, 0.f};
    float sumf_u[NR0_Q4K] = {0.f, 0.f};

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint ib = ix; ib < nb; ib += 4) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        for (short row = 0; row < NR0_Q4K; ++row) {
            if (first_row + row >= args.n_out) break;

            {
                device const uchar * blk = row0_g + (ulong)row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
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

            {
                device const uchar * blk = row0_u + (ulong)row * row_stride_bytes + (ulong)ib * Q4K_BYTES;
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

    for (short row = 0; row < NR0_Q4K; row++) {
        float total_g = simd_sum(sumf_g[row]);
        float total_u = simd_sum(sumf_u[row]);
        if (tiisg == 0 && first_row + row < args.n_out) {
            inner[(ulong)slot * args.n_out + first_row + row] = moe_silu_f(total_g) * total_u;
        }
    }
}

kernel void kernel_moe_down_q5_K_f32(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
    device const float * x = inner + (ulong)slot * args.n_in;

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

        device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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
        out[(ulong)slot * args.n_out + first_row] = tot;
    }
}

kernel void kernel_moe_down_weighted_sum_q5_K_f32_packed_slots(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device const float    * top_w   [[buffer(4)]],
        device       float    * out     [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint token = tgpig.y;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    const ulong base_slot = (ulong)token * args.topk;

    float acc = 0.0f;
    float yl[16];
    float yh[16];

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (uint slot_k = 0; slot_k < args.topk; ++slot_k) {
        const ulong slot = base_slot + slot_k;
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;

        device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
        device const float * x = inner + slot * args.n_in;
        device const float * y1 = x + ix * QK_K + y_offset;
        float sumf = 0.0f;

        for (uint i = ix; i < nb; i += 4) {
            device const float * y2 = y1 + 128;
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            for (short l = 0; l < 8; ++l) {
                yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
                yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
                yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
                yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
            }

            device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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

        const float total = simd_sum(sumf);
        if (tiisg == 0) {
            acc += top_w[slot] * total;
        }
    }

    if (tiisg == 0) {
        out[(ulong)token * args.n_out + first_row] = acc;
    }
}

kernel void kernel_moe_mat_vec_q5_K_f32(
        constant moe_q5k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * x       [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

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
    const uint first_row = (tgpig.x * NSG_Q5K + sgitg) * NR0_Q5K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q5K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;

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

        device const uchar * blk = expert_w + (ulong)first_row * row_stride_bytes + (ulong)i * Q5K_BYTES;
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
        out[(ulong)slot * args.n_out + first_row] = tot;
    }
}

kernel void kernel_moe_down_q6_K_f32(
        constant moe_q6k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device       float    * out     [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    if (slot >= args.topk) return;

    const int expert_i = top_idx[slot];
    if (expert_i < 0 || expert_i >= int(args.n_expert)) return;

    constexpr uchar kmask1 = 0x03;
    constexpr uchar kmask2 = 0x0C;
    constexpr uchar kmask3 = 0x30;
    constexpr uchar kmask4 = 0xC0;

    const uint nb = args.n_in / QK_K;
    const uint first_row = (tgpig.x * NSG_Q6K + sgitg) * NR0_Q6K;
    if (first_row >= args.n_out) return;

    const ulong row_stride_bytes = (ulong)nb * Q6K_BYTES;
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;
    device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
    device const float * x = inner + (ulong)slot * args.n_in;

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

            device const uchar * blk = expert_w + (ulong)(first_row + row) * row_stride_bytes
                                      + (ulong)i * Q6K_BYTES;
            device const uchar  * q1 = blk + q_offset_l;
            device const uchar  * q2 = q1 + 32;
            device const uchar  * qh = blk + 128 + q_offset_h;
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
            out[(ulong)slot * args.n_out + first_row + row] = total;
        }
    }
}

kernel void kernel_moe_down_weighted_sum_q6_K_f32(
        constant moe_q6k_args & args    [[buffer(0)]],
        device const uchar    * weight  [[buffer(1)]],
        device const float    * inner   [[buffer(2)]],
        device const int      * top_idx [[buffer(3)]],
        device const float    * top_w   [[buffer(4)]],
        device       float    * out     [[buffer(5)]],
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
    const ulong expert_stride_bytes = (ulong)args.n_out * row_stride_bytes;

    const ushort tid = tiisg / 2;
    const ushort ix  = tiisg % 2;
    const ushort ip  = tid / 8;
    const ushort il  = tid % 8;
    const ushort l0  = 4u * il;
    const ushort is  = 8u * ip + l0 / 16u;
    const ushort y_offset   = 128u * ip + l0;
    const ushort q_offset_l =  64u * ip + l0;
    const ushort q_offset_h =  32u * ip + l0;

    float acc[NR0_Q6K] = {0.f, 0.f};
    float yl[16];

    for (uint slot = 0; slot < args.topk; ++slot) {
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;

        device const uchar * expert_w = weight + (ulong)expert_i * expert_stride_bytes;
        device const float * x = inner + (ulong)slot * args.n_in;
        float sumf[NR0_Q6K] = {0.f, 0.f};

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

                device const uchar * blk = expert_w + (ulong)(first_row + row) * row_stride_bytes
                                          + (ulong)i * Q6K_BYTES;
                device const uchar  * q1 = blk + q_offset_l;
                device const uchar  * q2 = q1 + 32;
                device const uchar  * qh = blk + 128 + q_offset_h;
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
            const float total = simd_sum(sumf[row]);
            if (tiisg == 0 && first_row + row < args.n_out) {
                acc[row] += top_w[slot] * total;
            }
        }
    }

    if (tiisg == 0) {
        for (short row = 0; row < NR0_Q6K; ++row) {
            if (first_row + row < args.n_out) out[first_row + row] = acc[row];
        }
    }
}

kernel void kernel_moe_weighted_sum_f32(
        constant moe_sum_args & args [[buffer(0)]],
        device const float   * expert_out [[buffer(1)]],
        device const float   * weights    [[buffer(2)]],
        device       float   * out        [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n_out) return;
    float sum = 0.0f;
    for (uint e = 0; e < args.topk; ++e) {
        sum += weights[e] * expert_out[(ulong)e * args.n_out + tid];
    }
    out[tid] = sum;
}

kernel void kernel_axpy_scalar_f32(
        constant axpy_scalar_args & args [[buffer(0)]],
        device const float       * x     [[buffer(1)]],
        device const float       * scale [[buffer(2)]],
        device       float       * accum [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    accum[tid] += scale[0] * x[tid];
}

struct axpy_rowwise_args {
    uint n_cols;
    uint n_rows;
};

kernel void kernel_axpy_rowwise_f32(
        constant axpy_rowwise_args & args [[buffer(0)]],
        device const float        * x     [[buffer(1)]],
        device const float        * scale [[buffer(2)]],
        device       float        * accum [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_cols * args.n_rows;
    if (tid >= total) return;
    const uint row = tid / args.n_cols;
    accum[tid] += scale[row] * x[tid];
}
