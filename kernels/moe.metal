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
#define Q4K_NL (QK_K / 16)

#define NR0_Q4K 2
#define NSG_Q4K 2
#define NR0_Q5K 1
#define NSG_Q5K 2
#define NR0_Q6K 2
#define NSG_Q6K 2

#define Q5K_NL (QK_K / 16)
#define Q6K_NL (QK_K / 16)
#define NR0_MM 64
#define NR1_MM 32
#define NK_MM 32
#define NL0_MM (NK_MM / 16)
#define NL1_MM (NK_MM / 8)

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
    uint n_tokens;
};

struct moe_route_bucket_args {
    uint n_expert;
    uint n_tokens;
    uint topk;
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

kernel void kernel_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
        constant topk_dot_sigmoid_args & args          [[buffer(0)]],
        device const float             * logits        [[buffer(1)]],
        device const float             * shared_weight [[buffer(2)]],
        device const float             * x             [[buffer(3)]],
        device       int               * out_idx       [[buffer(4)]],
        device       float             * out_w         [[buffer(5)]],
        device       float             * shared_out    [[buffer(6)]],
        device atomic_int              * counts        [[buffer(7)]],
        device int                     * ids           [[buffer(8)]],
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
        const uint base = token * args.topk;
        for (uint i = 0; i < args.topk; ++i) {
            exp_val[i] = exp(out_w_t[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.topk; ++i) {
            out_w_t[i] = exp_val[i] / sum;
            const int expert = out_idx_t[i];
            if (expert >= 0 && expert < int(args.n_expert)) {
                const int dst = atomic_fetch_add_explicit(&counts[expert], 1, memory_order_relaxed);
                ids[(ulong)expert * args.n_tokens + dst] = int(base + i);
            }
        }
    }
}

kernel void kernel_moe_route_bucket_slots_f32(
        constant moe_route_bucket_args & args [[buffer(0)]],
        device const int              * top_idx [[buffer(1)]],
        device       int              * counts  [[buffer(2)]],
        device       int              * ids     [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n_expert) return;
    int count = 0;
    for (uint token = 0; token < args.n_tokens; ++token) {
        const uint base = token * args.topk;
        for (uint k = 0; k < args.topk; ++k) {
            const int expert_i = top_idx[base + k];
            if (expert_i == int(tid)) {
                ids[(ulong)tid * args.n_tokens + count] = int(base + k);
                count += 1;
            }
        }
    }
    counts[tid] = count;
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

#define NR1_GROUP_Q4 16
#define NL1_GROUP_Q4 (NK_MM / 8)
#define B_LOAD_THREADS_GROUP_Q4 (NR1_GROUP_Q4 * NL1_GROUP_Q4)
#define NR1_GROUP_Q5 16
#define NL1_GROUP_Q5 (NK_MM / 8)
#define B_LOAD_THREADS_GROUP_Q5 (NR1_GROUP_Q5 * NL1_GROUP_Q5)

struct moe_group_q4k_args {
    uint ffn;
    uint hidden;
    uint n_expert;
    uint topk;
    uint n_tokens;
    uint nb01;
    uint stride_b;
    uint min_count;
    uint max_count;
};

inline void dequantize_q4_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qs     = blk_bytes + 4 + 12;

    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }

    qs = qs + (il / 4) * 32 + 16 * (il & 1);
    short il_inner = il & 3;
    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;

    for (int i = 0; i < 16; ++i) {
        reg[i / 4][i % 4] = (half)(dl * (float)(qs[i] & mask) - ml);
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q4;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q4) ? (short)(count - r1) : NR1_GROUP_Q4;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    // Clamp lr1 to nr1-1 so partial-tile threads don't read stale ids[].
    // Mirrors the Q5 grouped kernel's lr1 clamp (line ~1359).
    const short lr1 = ((short)tiitg / NL1_GROUP_Q4) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q4);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q4) {
            const short sx = (tiitg % NL1_GROUP_Q4);
            const short sy = (tiitg / NL1_GROUP_Q4) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q4) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

inline void dequantize_q5_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg);

kernel void kernel_moe_swiglu_q5_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q5;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q5) ? (short)(count - r1) : NR1_GROUP_Q5;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q5) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q5)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q5);

    const short offset1 = il0 / Q5K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q5K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q5K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q5_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q5_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q5) {
            const short sx = (tiitg % NL1_GROUP_Q5);
            const short sy = (tiitg / NL1_GROUP_Q5) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q5) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_fused    [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q4;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q4) ? (short)(count - r1) : NR1_GROUP_Q4;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q4) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q4);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn * 2;
    device const uchar * x_ptr = srcA_fused + expert_stride * (ulong)im
                                           + (ulong)(args.nb01 * 2) * (r0 + lr0)
                                           + (ulong)offset1 * (2 * Q4K_BYTES);
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc_g[4];
    simdgroup_float8x8 mc_u[4];

    for (short i = 0; i < 4; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr + Q4K_BYTES, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q4) {
            const short sx = (tiitg % NL1_GROUP_Q4);
            const short sy = (tiitg / NL1_GROUP_Q4) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q4) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                    ? x_ptr + (2 * Q4K_BYTES) * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb, ma_g[i], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb, ma_u[i], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb   += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 4096))
                                     + 32 * (sgitg & 1)
                                     + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * i, NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const float g_val = temp_str_g[(8 * tile_i + mr) + c * NR0_MM];
            const float u_val = temp_str_u[(8 * tile_i + mr) + c * NR0_MM];
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = moe_silu_f(g_val) * u_val;
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_n32(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_gate     [[buffer(1)]],
        device const uchar * srcA_up       [[buffer(2)]],
        device const float * srcB          [[buffer(3)]],
        device const int   * counts        [[buffer(4)]],
        device const int   * ids           [[buffer(5)]],
        device       float * dst           [[buffer(6)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr_g = srcA_gate + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    device const uchar * x_ptr_u = srcA_up   + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                           + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_g[8];
    simdgroup_float8x8 mc_u[8];

    for (short i = 0; i < 8; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_g, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr_u, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr_g = (il < 2)
                    ? x_ptr_g + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_g;
        x_ptr_u = (il < 2)
                    ? x_ptr_u + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr_u;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 4], ma_g[i % 4], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 4], ma_u[i % 4], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 8192))
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int global_n = r1 + j;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.n_tokens + global_n];
                device float * D = dst + (ulong)slot * args.ffn + r0;
                threadgroup float * Cg = temp_str_g + (j * NR0_MM);
                threadgroup float * Cu = temp_str_u + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = moe_silu_f(Cg[i]) * Cu[i];
                }
            }
        }
    }
}

kernel void kernel_moe_swiglu_q4_K_f32_grouped_slots_fused_n32(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA_fused    [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa_g = (threadgroup half *)(shmem);
    threadgroup half * sa_u = (threadgroup half *)(shmem + 4096);
    threadgroup half * sb   = (threadgroup half *)(shmem + 8192);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn * 2;
    device const uchar * x_ptr = srcA_fused + expert_stride * (ulong)im
                                           + (ulong)(args.nb01 * 2) * (r0 + lr0)
                                           + (ulong)offset1 * (2 * Q4K_BYTES);
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma_g[4];
    simdgroup_half8x8 ma_u[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc_g[8];
    simdgroup_float8x8 mc_u[8];

    for (short i = 0; i < 8; ++i) {
        mc_g[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        mc_u[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_g[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr + Q4K_BYTES, il, temp_a);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa_u[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                    ? x_ptr + (2 * Q4K_BYTES) * ((2 + Q4K_NL - 1) / Q4K_NL)
                    : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma_g = (sa_g + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsma_u = (sa_u + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb   = (sb   + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma_g[i], lsma_g + 64 * i, 8, 0, false);
                simdgroup_load(ma_u[i], lsma_u + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc_g[i], mb[i / 4], ma_g[i % 4], mc_g[i]);
                simdgroup_multiply_accumulate(mc_u[i], mb[i / 4], ma_u[i % 4], mc_u[i]);
            }
            lsma_g += 8 * 64;
            lsma_u += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str_g = ((threadgroup float *)shmem)
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    threadgroup float * temp_str_u = ((threadgroup float *)(shmem + 8192))
                                     + 32 * (sgitg & 1)
                                     + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc_g[i], temp_str_g + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
        simdgroup_store(mc_u[i], temp_str_u + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int global_n = r1 + j;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.n_tokens + global_n];
                device float * D = dst + (ulong)slot * args.ffn + r0;
                threadgroup float * Cg = temp_str_g + (j * NR0_MM);
                threadgroup float * Cu = temp_str_u + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = moe_silu_f(Cg[i]) * Cu[i];
                }
            }
        }
    }
}

kernel void kernel_moe_matmul_q4_K_f32_grouped_slots_n16(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA          [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_GROUP_Q4;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_GROUP_Q4) ? (short)(count - r1) : NR1_GROUP_Q4;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_GROUP_Q4) < nr1
                        ? ((short)tiitg / NL1_GROUP_Q4)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_GROUP_Q4);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                          + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb;
    simdgroup_float8x8 mc[4];

    for (short i = 0; i < 4; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        if (tiitg < B_LOAD_THREADS_GROUP_Q4) {
            const short sx = (tiitg % NL1_GROUP_Q4);
            const short sy = (tiitg / NL1_GROUP_Q4) / 8;
            const short ib = 2 * sx + sy;
            const short ly = (tiitg / NL1_GROUP_Q4) % 8;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 1 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            simdgroup_load(mb, lsmb, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb, ma[i], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 2 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (8 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 4; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * i, NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const short m_off = 32 * (sgitg & 1);
    const short n_off = 8 * (sgitg >> 1);
    const short m_local = (short)tiitg & 31;
    const short tile_i = m_local >> 3;
    const short mr = m_local & 7;
    const int global_m = r0 + m_off + m_local;
    const bool m_in = (global_m < (int)args.ffn);
    for (short c = 0; c < 8; ++c) {
        const int global_n = r1 + n_off + c;
        if (m_in && global_n < count) {
            const int slot = ids[(ulong)im * args.n_tokens + global_n];
            dst[global_m + (ulong)slot * args.ffn] = temp_str[(8 * tile_i + mr) + c * NR0_MM];
        }
    }
}

kernel void kernel_moe_matmul_q4_K_f32_grouped_slots_n32(
        constant moe_group_q4k_args & args [[buffer(0)]],
        device const uchar * srcA          [[buffer(1)]],
        device const float * srcB          [[buffer(2)]],
        device const int   * counts        [[buffer(3)]],
        device const int   * ids           [[buffer(4)]],
        device       float * dst           [[buffer(5)]],
        threadgroup  uchar * shmem         [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;

    const short nr0 = ((int)args.ffn - r0 < NR0_MM) ? (short)((int)args.ffn - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q4K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.ffn;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                          + (ulong)offset1 * Q4K_BYTES;
    const int slot_id = ids[(ulong)im * args.n_tokens + r1 + lr1];
    const int token = slot_id / int(args.topk);
    device const float * y_ptr = srcB + (ulong)args.stride_b * token + (ulong)iy;

    simdgroup_half8x8 ma[4];
    simdgroup_half8x8 mb[2];
    simdgroup_float8x8 mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.hidden; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q4_K_half_grouped(x_ptr, il, temp_a);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q4K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q4K_BYTES * ((2 + Q4K_NL - 1) / Q4K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int global_n = r1 + j;
            if (global_n < count) {
                const int slot = ids[(ulong)im * args.n_tokens + global_n];
                device float * D = dst + (ulong)slot * args.ffn + r0;
                threadgroup float * C = temp_str + (j * NR0_MM);
                for (int i = 0; i < nr0; ++i) {
                    D[i] = C[i];
                }
            }
        }
    }
}

struct moe_fused_q4q5_args {
    uint hidden;
    uint ffn;
    uint n_expert;
    uint topk;
};

kernel void kernel_moe_fused_routed_q4q5_token_f32(
        constant moe_fused_q4q5_args & args [[buffer(0)]],
        device const uchar * w_gate        [[buffer(1)]],
        device const uchar * w_up          [[buffer(2)]],
        device const uchar * w_down        [[buffer(3)]],
        device const float * x_pack        [[buffer(4)]],
        device const int   * top_idx       [[buffer(5)]],
        device const float * top_w         [[buffer(6)]],
        device       float * out_pack      [[buffer(7)]],
        threadgroup  float * inner         [[threadgroup(0)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint token = tgpig.y;
    device const float * x = x_pack + (ulong)token * args.hidden;
    device float * out = out_pack + (ulong)token * args.hidden;
    const ulong base_slot = (ulong)token * args.topk;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const ushort ix_q4 = tiisg / 8;
    const ushort it_q4 = tiisg % 8;
    const ushort iq_q4 = it_q4 / 4;
    const ushort ir_q4 = it_q4 % 4;

    const ushort tid_q5 = tiisg / 4;
    const ushort ix_q5  = tiisg % 4;
    const ushort iq_q5  = tid_q5 / 4;
    const ushort ir_q5  = tid_q5 % 4;

    const ushort l0_q5 = 8u * ir_q5;
    const ushort q_offset_q5 = 32u * iq_q5 + l0_q5;
    const ushort y_offset_q5 = 64u * iq_q5 + l0_q5;

    const uchar hm1 = 1u << (2u * iq_q5);
    const uchar hm2 = hm1 << 1;
    const uchar hm3 = hm1 << 4;
    const uchar hm4 = hm2 << 4;

    const uint nb_h = args.hidden / QK_K;
    const uint nb_f = args.ffn / QK_K;
    const ulong gate_row_stride_bytes = (ulong)nb_h * Q4K_BYTES;
    const ulong gate_expert_stride_bytes = (ulong)args.ffn * gate_row_stride_bytes;
    const ulong down_row_stride_bytes = (ulong)nb_f * Q5K_BYTES;
    const ulong down_expert_stride_bytes = (ulong)args.hidden * down_row_stride_bytes;

    float yl_q4[16];
    float yh_q4[16];
    float yl_q5[16];
    float yh_q5[16];
    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *) sc16;

    if (tiisg == 0) {
        for (uint i = sgitg; i < args.hidden; i += 8) {
            out[i] = 0.0f;
        }
    }
    // mem_device is required because `out[]` is device memory and is
    // both written above and accumulated into (`out[first_row] += ...`)
    // by other threads of this threadgroup further down. A plain
    // mem_threadgroup barrier only orders threadgroup memory; device
    // writes can still be reordered across it.
    threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

    for (uint slot_k = 0; slot_k < args.topk; ++slot_k) {
        const ulong slot = base_slot + slot_k;
        const int expert_i = top_idx[slot];
        if (expert_i < 0 || expert_i >= int(args.n_expert)) continue;
        const float slot_w = top_w[slot];
        if (slot_w == 0.0f) continue;

        device const uchar * expert_gate = w_gate + (ulong)expert_i * gate_expert_stride_bytes;
        device const uchar * expert_up   = w_up   + (ulong)expert_i * gate_expert_stride_bytes;
        device const uchar * expert_down = w_down + (ulong)expert_i * down_expert_stride_bytes;

        for (uint f_base = 0; f_base < args.ffn; f_base += 16) {
            if (sgitg < 8) {
                const uint first_row = f_base + (uint)sgitg * NR0_Q4K;
                if (first_row < args.ffn) {
                    device const uchar * row0_g = expert_gate + (ulong)first_row * gate_row_stride_bytes;
                    device const uchar * row0_u = expert_up   + (ulong)first_row * gate_row_stride_bytes;
                    device const float * y4 = x + ix_q4 * QK_K + 64u * iq_q4 + 8u * ir_q4;

                    float sumf_g[NR0_Q4K] = {0.f, 0.f};
                    float sumf_u[NR0_Q4K] = {0.f, 0.f};

                    for (uint ib = ix_q4; ib < nb_h; ib += 4) {
                        float4 sumy = {0.f, 0.f, 0.f, 0.f};
                        for (short i = 0; i < 8; ++i) {
                            yl_q4[i+0] = y4[i+  0]; sumy[0] += yl_q4[i+0];
                            yl_q4[i+8] = y4[i+ 32]; sumy[1] += yl_q4[i+8];
                            yh_q4[i+0] = y4[i+128]; sumy[2] += yh_q4[i+0];
                            yh_q4[i+8] = y4[i+160]; sumy[3] += yh_q4[i+8];
                        }

                        for (short row = 0; row < NR0_Q4K; ++row) {
                            if (first_row + row >= args.ffn) break;

                            {
                                device const uchar * blk = row0_g + (ulong)row * gate_row_stride_bytes + (ulong)ib * Q4K_BYTES;
                                device const half     * dh = (device const half *) blk;
                                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq_q4;
                                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq_q4 + 4 * ir_q4;
                                device const uint16_t * q2 = q1 + 32;

                                sc16[0] =  sc[0]                & kmask1;
                                sc16[1] =  sc[2]                & kmask1;
                                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                                for (short i = 0; i < 4; ++i) {
                                    acc1[0] += yl_q4[2*i + 0] * (q1[i] & 0x000F);
                                    acc1[1] += yl_q4[2*i + 1] * (q1[i] & 0x0F00);
                                    acc1[2] += yl_q4[2*i + 8] * (q1[i] & 0x00F0);
                                    acc1[3] += yl_q4[2*i + 9] * (q1[i] & 0xF000);
                                    acc2[0] += yh_q4[2*i + 0] * (q2[i] & 0x000F);
                                    acc2[1] += yh_q4[2*i + 1] * (q2[i] & 0x0F00);
                                    acc2[2] += yh_q4[2*i + 8] * (q2[i] & 0x00F0);
                                    acc2[3] += yh_q4[2*i + 9] * (q2[i] & 0xF000);
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
                                device const uchar * blk = row0_u + (ulong)row * gate_row_stride_bytes + (ulong)ib * Q4K_BYTES;
                                device const half     * dh = (device const half *) blk;
                                device const uint16_t * sc = (device const uint16_t *)(blk + 4) + iq_q4;
                                device const uint16_t * q1 = (device const uint16_t *)(blk + 4 + 12) + 16 * iq_q4 + 4 * ir_q4;
                                device const uint16_t * q2 = q1 + 32;

                                sc16[0] =  sc[0]                & kmask1;
                                sc16[1] =  sc[2]                & kmask1;
                                sc16[2] = ((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2);
                                sc16[3] = ((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2);

                                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                                float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                                for (short i = 0; i < 4; ++i) {
                                    acc1[0] += yl_q4[2*i + 0] * (q1[i] & 0x000F);
                                    acc1[1] += yl_q4[2*i + 1] * (q1[i] & 0x0F00);
                                    acc1[2] += yl_q4[2*i + 8] * (q1[i] & 0x00F0);
                                    acc1[3] += yl_q4[2*i + 9] * (q1[i] & 0xF000);
                                    acc2[0] += yh_q4[2*i + 0] * (q2[i] & 0x000F);
                                    acc2[1] += yh_q4[2*i + 1] * (q2[i] & 0x0F00);
                                    acc2[2] += yh_q4[2*i + 8] * (q2[i] & 0x00F0);
                                    acc2[3] += yh_q4[2*i + 9] * (q2[i] & 0xF000);
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

                    for (short row = 0; row < NR0_Q4K; ++row) {
                        float total_g = simd_sum(sumf_g[row]);
                        float total_u = simd_sum(sumf_u[row]);
                        if (tiisg == 0 && first_row + row < args.ffn) {
                            inner[first_row + row] = moe_silu_f(total_g) * total_u;
                        }
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint row_base = 0; row_base < args.hidden; row_base += 16) {
            const uint first_row = row_base + sgitg;
            if (first_row < args.hidden) {
                float sumf = 0.0f;
                threadgroup const float * y1 = inner + ix_q5 * QK_K + y_offset_q5;

                for (uint i = ix_q5; i < nb_f; i += 4) {
                    threadgroup const float * y2 = y1 + 128;
                    float4 sumy = {0.f, 0.f, 0.f, 0.f};
                    for (short l = 0; l < 8; ++l) {
                        yl_q5[l+0] = y1[l+ 0]; sumy[0] += yl_q5[l+0];
                        yl_q5[l+8] = y1[l+32]; sumy[1] += yl_q5[l+8];
                        yh_q5[l+0] = y2[l+ 0]; sumy[2] += yh_q5[l+0];
                        yh_q5[l+8] = y2[l+32]; sumy[3] += yh_q5[l+8];
                    }

                    device const uchar * blk = expert_down + (ulong)first_row * down_row_stride_bytes + (ulong)i * Q5K_BYTES;
                    device const half     * dh = (device const half *) blk;
                    device const uint16_t * a  = (device const uint16_t *)(blk + 4) + iq_q5;
                    device const uchar    * qh = (blk + 4 + 12) + l0_q5;
                    device const uchar    * q1 = (blk + 4 + 12 + 32) + q_offset_q5;
                    device const uchar    * q2 = q1 + 64;

                    sc16[0] =  a[0]                & kmask1;
                    sc16[1] =  a[2]                & kmask1;
                    sc16[2] = ((a[4] >> 0) & kmask2) | ((a[0] & kmask3) >> 2);
                    sc16[3] = ((a[4] >> 4) & kmask2) | ((a[2] & kmask3) >> 2);

                    float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                    float4 acc2 = {0.f, 0.f, 0.f, 0.f};
                    for (short l = 0; l < 8; ++l) {
                        const uchar hq = qh[l];
                        acc1[0] += yl_q5[l+0] * (q1[l] & 0x0F);
                        acc1[1] += yl_q5[l+8] * (q1[l] & 0xF0);
                        acc1[2] += yh_q5[l+0] * (q2[l] & 0x0F);
                        acc1[3] += yh_q5[l+8] * (q2[l] & 0xF0);
                        acc2[0] += (hq & hm1) ? yl_q5[l+0] : 0.f;
                        acc2[1] += (hq & hm2) ? yl_q5[l+8] : 0.f;
                        acc2[2] += (hq & hm3) ? yh_q5[l+0] : 0.f;
                        acc2[3] += (hq & hm4) ? yh_q5[l+8] : 0.f;
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
                    out[first_row] += slot_w * total;
                }
            }
        }
        // mem_device for the same reason as the initial-zero barrier:
        // the next expert iteration accumulates into `out[first_row]`,
        // so writes by other simdgroups in this iteration must be
        // visible across the boundary.
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
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

kernel void kernel_moe_down_q5_K_f32_grouped_rows(
        constant moe_q5k_args & args       [[buffer(0)]],
        device const uchar    * weight     [[buffer(1)]],
        device const float    * inner      [[buffer(2)]],
        device const int      * expert_idx [[buffer(3)]],
        device       float    * out        [[buffer(4)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint slot = tgpig.y;
    const int expert_i = expert_idx[slot];
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

struct moe_group_q5k_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

struct moe_group_q6k_args {
    uint M;
    uint N;
    uint K;
    uint nb01;
    uint stride_b;
};

inline void dequantize_q5_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    const half d_h    = ((device const half *)blk_bytes)[0];
    const half dmin_h = ((device const half *)blk_bytes)[1];
    device const uchar * scales = blk_bytes + 4;
    device const uchar * qh     = blk_bytes + 4 + 12;
    device const uchar * qs     = blk_bytes + 4 + 12 + 32;

    const short is  = (il / 4) * 2;
    const short k01 = (il / 2) & 1;
    uchar sc_u, m_u;
    if (is < 4) {
        sc_u = scales[is + k01] & 63;
        m_u  = scales[is + k01 + 4] & 63;
    } else {
        sc_u = (scales[is + k01 + 4] & 0x0F) | ((scales[is + k01 - 4] >> 6) << 4);
        m_u  = (scales[is + k01 + 4] >>   4) | ((scales[is + k01    ] >> 6) << 4);
    }

    qs = qs + 32 * (il / 4) + 16 * (il & 1);
    qh = qh + 16 * (il & 1);
    const uchar ul = 1u << (il / 2);
    short il_inner = il & 3;

    const float d   = il_inner < 2 ? (float)d_h : (float)d_h / 16.0f;
    const float dmin = (float)dmin_h;
    const float dl  = d   * (float)sc_u;
    const float ml  = dmin * (float)m_u;
    const ushort mask = il_inner < 2 ? 0x0F : 0xF0;
    const float qh_val = il_inner < 2 ? 16.0f : 256.0f;

    for (int i = 0; i < 16; ++i) {
        const float q_low  = (float)(qs[i] & mask);
        const float q_high = (qh[i] & ul) ? qh_val : 0.0f;
        reg[i / 4][i % 4] = (half)(dl * (q_low + q_high) - ml);
    }
}

kernel void kernel_moe_down_q5_K_f32_grouped_slots(
        constant moe_group_q5k_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q5K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q5K_BYTES;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q5_K_half_grouped(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q5K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q5K_BYTES * ((2 + Q5K_NL - 1) / Q5K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
    }
}

inline void dequantize_q6_K_half_grouped(device const uchar * blk_bytes,
                                         short il,
                                         thread half4x4 & reg) {
    device const uint16_t * ql = (device const uint16_t *)(blk_bytes + 0);
    device const uint16_t * qh = (device const uint16_t *)(blk_bytes + 128);
    device const int8_t   * scales = (device const int8_t *)(blk_bytes + 128 + 64);
    const half d_all = ((device const half *)(blk_bytes + 128 + 64 + 16))[0];

    ql = ql + 32 * (il / 8) + 16 * ((il / 2) & 1) + 8 * (il & 1);
    qh = qh + 16 * (il / 8) + 8 * (il & 1);
    const float sc = scales[(il % 2) + 2 * (il / 2)];
    const short il_inner = (il / 2) & 3;

    const uint32_t kmask1 = il_inner > 1
        ? (il_inner > 2 ? 0xC0C0C0C0 : 0x30303030)
        : (il_inner > 0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il_inner > 1 ? 0xF0F0F0F0 : 0x0F0F0F0F;
    const float ml  = (float)d_all * sc * 32.0f;
    const float dl0 = (float)d_all * sc;
    const float dl1 = dl0 / 256.0f;
    const float dl2 = dl0 / (256.0f * 256.0f);
    const float dl3 = dl0 / (256.0f * 256.0f * 256.0f);
    const uint8_t shr_h = il_inner > 2 ? 2 : 0;
    const uint8_t shl_h = il_inner > 1 ? 0 : (il_inner > 0 ? 2 : 4);
    const uint8_t shr_l = il_inner > 1 ? 4 : 0;

    for (int i = 0; i < 4; ++i) {
        const uint32_t low  = (ql[2 * i] | (uint32_t)(ql[2 * i + 1] << 16)) & kmask2;
        const uint32_t high = (qh[2 * i] | (uint32_t)(qh[2 * i + 1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (half)(dl0 * ((float)(q & 0xFF))         - ml);
        reg[i][1] = (half)(dl1 * ((float)(q & 0xFF00))       - ml);
        reg[i][2] = (half)(dl2 * ((float)(q & 0xFF0000))     - ml);
        reg[i][3] = (half)(dl3 * ((float)(q & 0xFF000000))   - ml);
    }
}

kernel void kernel_moe_down_q6_K_f32_grouped_slots(
        constant moe_group_q6k_args & args [[buffer(0)]],
        device const uchar * srcA         [[buffer(1)]],
        device const float * srcB         [[buffer(2)]],
        device const int   * counts       [[buffer(3)]],
        device const int   * ids          [[buffer(4)]],
        device       float * dst          [[buffer(5)]],
        threadgroup  uchar * shmem        [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    threadgroup half * sa = (threadgroup half *)(shmem);
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    const int im = tgpig.z;
    const int r0 = tgpig.y * NR0_MM;
    const int r1 = tgpig.x * NR1_MM;

    const int count = counts[im];
    if (r1 >= count) return;

    const short nr0 = ((int)args.M - r0 < NR0_MM) ? (short)((int)args.M - r0) : NR0_MM;
    const short nr1 = (count - r1 < NR1_MM) ? (short)(count - r1) : NR1_MM;

    const short lr0 = ((short)tiitg / NL0_MM) < nr0
                        ? ((short)tiitg / NL0_MM)
                        : nr0 - 1;
    const short il0 = (tiitg % NL0_MM);
    short il = il0;

    const short lr1 = ((short)tiitg / NL1_MM) < nr1
                        ? ((short)tiitg / NL1_MM)
                        : nr1 - 1;
    const short iy = 8 * (tiitg % NL1_MM);

    const short offset1 = il0 / Q6K_NL;
    const ulong expert_stride = (ulong)args.nb01 * args.M;
    device const uchar * x_ptr = srcA + expert_stride * (ulong)im + (ulong)args.nb01 * (r0 + lr0)
                                       + (ulong)offset1 * Q6K_BYTES;
    const int slot_id = ids[(ulong)im * args.N + (r1 + lr1)];
    device const float * y_ptr = srcB + (ulong)args.stride_b * slot_id + (ulong)iy;

    simdgroup_half8x8   ma[4];
    simdgroup_half8x8   mb[2];
    simdgroup_float8x8  mc[8];

    for (short i = 0; i < 8; ++i) {
        mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint loop_k = 0; loop_k < args.K; loop_k += NK_MM) {
        {
            half4x4 temp_a;
            dequantize_q6_K_half_grouped(x_ptr, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short i = 0; i < 16; ++i) {
                const short sx = 2 * il0 + i / 8;
                const short sy = (tiitg / NL0_MM) / 8;
                const short lx = (tiitg / NL0_MM) % 8;
                const short ly = i % 8;
                const short ib = 8 * sx + sy;
                sa[64 * ib + 8 * ly + lx] = temp_a[i / 4][i % 4];
            }
        }

        {
            const short sx = (tiitg % NL1_MM);
            const short sy = (tiitg / NL1_MM) / 8;
            const short ly = (tiitg / NL1_MM) % 8;
            const short ib = 4 * sx + sy;
            *(threadgroup half2x4 *)(sb + 64 * ib + 8 * ly) =
                (half2x4)(*((device const float2x4 *)y_ptr));
        }

        il = (il + 2 < Q6K_NL) ? il + 2 : il % 2;
        x_ptr = (il < 2)
                  ? x_ptr + Q6K_BYTES * ((2 + Q6K_NL - 1) / Q6K_NL)
                  : x_ptr;
        y_ptr += NK_MM;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = (sa + 4 * 64 * (sgitg % 2));
        threadgroup const half * lsmb = (sb + 2 * 64 * (sgitg / 2));

        for (short ik = 0; ik < NK_MM / 8; ++ik) {
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 4; ++i) {
                simdgroup_load(ma[i], lsma + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; ++i) {
                simdgroup_load(mb[i], lsmb + 64 * i, 8, 0, false);
            }
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; ++i) {
                simdgroup_multiply_accumulate(mc[i], mb[i / 4], ma[i % 4], mc[i]);
            }
            lsma += 8 * 64;
            lsmb += 4 * 64;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * temp_str = ((threadgroup float *)shmem)
                                   + 32 * (sgitg & 1)
                                   + (16 * (sgitg >> 1)) * NR0_MM;
    for (short i = 0; i < 8; ++i) {
        simdgroup_store(mc[i], temp_str + 8 * (i % 4) + 8 * NR0_MM * (i / 4),
                        NR0_MM, 0, false);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sgitg == 0) {
        for (int j = tiitg; j < nr1; j += NR1_MM) {
            const int slot = ids[(ulong)im * args.N + r1 + j];
            device float * D = dst + (ulong)slot * args.M + r0;
            threadgroup float * C = temp_str + (j * NR0_MM);
            for (int i = 0; i < nr0; ++i) {
                D[i] = C[i];
            }
        }
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

kernel void kernel_moe_weighted_sum_packed_f32(
        constant moe_sum_args & args [[buffer(0)]],
        device const float   * expert_out [[buffer(1)]],
        device const float   * weights    [[buffer(2)]],
        device       float   * out        [[buffer(3)]],
        uint2 tid2 [[thread_position_in_grid]]) {
    const uint tid = tid2.x;
    const uint token = tid2.y;
    if (tid >= args.n_out) return;
    float sum = 0.0f;
    const ulong base = (ulong)token * args.topk;
    for (uint e = 0; e < args.topk; ++e) {
        const ulong slot = base + e;
        sum += weights[slot] * expert_out[slot * args.n_out + tid];
    }
    out[(ulong)token * args.n_out + tid] = sum;
}

kernel void kernel_moe_grouped_finalizer_f32(
        constant moe_sum_args & args [[buffer(0)]],
        device const float   * expert_out  [[buffer(1)]],
        device const float   * topk_w      [[buffer(2)]],
        device const float   * shared_gate [[buffer(3)]],
        device const float   * shared_out  [[buffer(4)]],
        device       float   * x_pack      [[buffer(5)]],
        uint2 tid2 [[thread_position_in_grid]]) {
    const uint col = tid2.x;
    const uint token = tid2.y;
    if (col >= args.n_out) return;
    float routed = 0.0f;
    const ulong base = (ulong)token * args.topk;
    for (uint e = 0; e < args.topk; ++e) {
        const ulong slot = base + e;
        routed += topk_w[slot] * expert_out[slot * args.n_out + col];
    }
    const ulong out_idx = (ulong)token * args.n_out + col;
    x_pack[out_idx] += routed + shared_gate[token] * shared_out[out_idx];
}

struct scatter_axpy_rows_args {
    uint n_cols;
    uint n_rows;
    uint out_rows;
};

kernel void kernel_scatter_rows_f32_unique(
        constant scatter_axpy_rows_args & args [[buffer(0)]],
        device const float             * x     [[buffer(1)]],
        device const int               * rows  [[buffer(2)]],
        device       float             * out   [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_cols * args.n_rows;
    if (tid >= total) return;
    const uint src_row = tid / args.n_cols;
    const uint col = tid % args.n_cols;
    const int dst_row = rows[src_row];
    if (dst_row < 0 || dst_row >= int(args.out_rows)) return;
    out[(ulong)dst_row * args.n_cols + col] = x[tid];
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

kernel void kernel_scatter_axpy_rows_unique_f32(
        constant scatter_axpy_rows_args & args [[buffer(0)]],
        device const float             * x     [[buffer(1)]],
        device const int               * rows  [[buffer(2)]],
        device const float             * scale [[buffer(3)]],
        device       float             * accum [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.n_cols * args.n_rows;
    if (tid >= total) return;
    const uint src_row = tid / args.n_cols;
    const uint col = tid % args.n_cols;
    const int dst_row = rows[src_row];
    if (dst_row < 0 || dst_row >= int(args.out_rows)) return;
    accum[(ulong)dst_row * args.n_cols + col] += scale[src_row] * x[tid];
}
