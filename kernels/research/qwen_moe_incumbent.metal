// Reference copies of the pre-guard Qwen/Flash-Next top-k and bucket kernels
// (main at 5c3da609) for bitwise parity tests. Test-only: not on a product
// dispatch path. Kept in research/ so parity tests can load it beside the
// product metallib.

#include <metal_stdlib>
using namespace metal;
struct topk_logits_args {
    uint n;
    uint k;
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

inline bool moe_better_pair(float cand_v, int cand_i, float best_v, int best_i) {
    return cand_i >= 0 && (best_i < 0 || cand_v > best_v || (cand_v == best_v && cand_i < best_i));
}

kernel void kernel_incumbent_topk_logits_softmax_f32(
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

kernel void kernel_incumbent_topk_logits_softmax_parallel_f32(
        constant topk_logits_args & args [[buffer(0)]],
        device const float       * logits [[buffer(1)]],
        device       int         * out_idx [[buffer(2)]],
        device       float       * out_w   [[buffer(3)]],
        threadgroup  float       * sh_score [[threadgroup(0)]],
        threadgroup  float       * red_val  [[threadgroup(1)]],
        threadgroup  int         * red_idx  [[threadgroup(2)]],
        uint  tid [[thread_position_in_threadgroup]],
        uint  ntg [[threads_per_threadgroup]]) {
    const uint MAX_K = 16;
    if (args.n > ntg || args.k == 0 || args.k > MAX_K) return;

    sh_score[tid] = tid < args.n ? logits[tid] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint slot = 0; slot < args.k; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = tid < args.n ? int(tid) : -1;
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
        for (uint i = 0; i < args.k; ++i) {
            exp_val[i] = exp(out_w[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.k; ++i) {
            out_w[i] = exp_val[i] / sum;
        }
    }
}

kernel void kernel_incumbent_qwen4exp_topk_guarded_f32(
        constant topk_logits_args & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device int * out_idx [[buffer(2)]],
        device float * out_w [[buffer(3)]],
        threadgroup float * sh_score [[threadgroup(0)]],
        threadgroup float * red_val [[threadgroup(1)]],
        threadgroup int * red_idx [[threadgroup(2)]],
        uint tid [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]]) {
    // The checked host fixes N512/K10/TG512. Keep fallback loop bounds dynamic.
    if (args.k == 0 || args.k > 16) return;
    const device uint * bits = reinterpret_cast<const device uint *>(logits);
    const bool nonfinite = simd_any((bits[tid] & 0x7f800000u) == 0x7f800000u);
    if (lane == 0) red_idx[tid / 32] = nonfinite ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int any = 0;
        for (uint i = 0; i < 16; ++i) any |= red_idx[i];
        red_idx[0] = any;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool serial = red_idx[0] != 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (serial) {
        if (tid == 0) {
            const uint MAX_K = 16;
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
        return;
    }
    sh_score[tid] = logits[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint slot = 0; slot < args.k; ++slot) {
        red_val[tid] = sh_score[tid];
        red_idx[tid] = int(tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = 256; stride > 0; stride >>= 1) {
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
        float exp_val[16];
        for (uint i = 0; i < args.k; ++i) {
            exp_val[i] = exp(out_w[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.k; ++i) out_w[i] = exp_val[i] / sum;
    }
}

kernel void kernel_incumbent_qwen4exp_topk_guarded_simd_f32(
        constant topk_logits_args & args [[buffer(0)]],
        device const float * logits [[buffer(1)]],
        device int * out_idx [[buffer(2)]],
        device float * out_w [[buffer(3)]],
        threadgroup float * sh_score [[threadgroup(0)]],
        threadgroup float * red_val [[threadgroup(1)]],
        threadgroup int * red_idx [[threadgroup(2)]],
        uint tid [[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort simd [[simdgroup_index_in_threadgroup]]) {
    if (args.k == 0 || args.k > 16) return;
    const device uint * bits = reinterpret_cast<const device uint *>(logits);
    const bool nonfinite = simd_any((bits[tid] & 0x7f800000u) == 0x7f800000u);
    if (lane == 0) red_idx[tid / 32] = nonfinite ? 1 : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        int any = 0;
        for (uint i = 0; i < 16; ++i) any |= red_idx[i];
        red_idx[0] = any;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const bool serial = red_idx[0] != 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (serial) {
        if (tid == 0) {
            const uint MAX_K = 16;
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
        return;
    }
    float v = logits[tid];
    for (uint slot = 0; slot < args.k; ++slot) {
        const float m = simd_max(v);
        const int lowest = simd_min(v == m ? int(tid) : INT_MAX);
        if (lane == 0) {
            red_val[simd] = m;
            red_idx[simd] = lowest;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd == 0) {
            const float cv = lane < 16 ? red_val[lane] : -INFINITY;
            const int ci = lane < 16 ? red_idx[lane] : INT_MAX;
            const float m2 = simd_max(cv);
            const int winner = simd_min(cv == m2 ? ci : INT_MAX);
            if (lane == 0) {
                out_idx[slot] = max(winner, 0);
                out_w[slot] = m2;
                red_idx[16] = winner;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (int(tid) == red_idx[16]) v = -INFINITY;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) {
        const float max_top = out_w[0];
        float sum = 0.0f;
        float exp_val[16];
        for (uint i = 0; i < args.k; ++i) {
            exp_val[i] = exp(out_w[i] - max_top);
            sum += exp_val[i];
        }
        sum = max(sum, 6.103515625e-5f);
        for (uint i = 0; i < args.k; ++i) out_w[i] = exp_val[i] / sum;
    }
}

kernel void kernel_incumbent_topk_logits_softmax_dot_sigmoid_f32(
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

kernel void kernel_incumbent_topk_logits_softmax_dot_sigmoid_packed_f32(
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

kernel void kernel_incumbent_topk_bucket_logits_softmax_dot_sigmoid_packed_f32(
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

kernel void kernel_incumbent_moe_route_bucket_slots_f32(
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

kernel void kernel_incumbent_moe_route_bucket_slots_parallel_f32(
        constant moe_route_bucket_args & args [[buffer(0)]],
        device const int              * top_idx [[buffer(1)]],
        device       int              * counts  [[buffer(2)]],
        device       int              * ids     [[buffer(3)]],
        uint expert [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup int prefix[256];
    if (expert >= args.n_expert) return;
    const uint total = args.n_tokens * args.topk;
    const uint per = (total + 255u) / 256u;
    const uint first = min(tid * per, total);
    const uint last = min(first + per, total);
    int local = 0;
    for (uint s = first; s < last; ++s) local += top_idx[s] == int(expert) ? 1 : 0;
    prefix[tid] = local;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint offset = 1; offset < 256u; offset <<= 1) {
        const int v = tid >= offset ? prefix[tid - offset] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix[tid] += v;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    int pos = prefix[tid] - local;
    device int * out = ids + (ulong)expert * args.n_tokens;
    for (uint s = first; s < last; ++s) {
        if (top_idx[s] == int(expert)) out[pos++] = int(s);
    }
    if (tid == 255u) counts[expert] = prefix[255];
}
