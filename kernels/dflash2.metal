// DFlash 2 kernels: two-tap dynamic depthwise convolution + vocab top-16.
//
// Semantics match llama.cpp PR 27342 (`build_dflash2_conv` /
// `build_post_sampling` in src/models/dflash.cpp) and the DFlash 2 blog
// post (inco.ai/blog/dflash2).
//
// Conv: for each noise-block token t and channel c,
//   y[t][c] = sum_tap (base[side][tap][c] + dyn[t][side][tap][group(c)]) * x[t - tap][c]
// with x[t - tap] treated as zeros when t < tap (block position 0 is the
// carry/anchor token, so the first DRAFT position's tap-1 reads the last
// verified token — exactly the blog's "first position reads the last
// verified token's representation").
//
// Top-16: per-row top-k(16) over the vocab-sized logit rows, values +
// indices, sorted descending, ties broken toward the lower index (same
// policy as kernel_argmax_f32). Compiled with -ffast-math like the rest
// of the library; logits are finite so NaN ordering is not a concern
// (mirrors kernel_argmax_f32's stance, not the _greedy bit-pattern one).

#include <metal_stdlib>
using namespace metal;

struct dflash2_conv_args {
    uint h;           // hidden size (channels)
    uint n_tokens;    // noise block length N
    uint kernel_size; // taps (2 for released drafters)
    uint group_size;  // channels per dynamic-coefficient group (16)
    uint n_groups;    // h / group_size
    uint side;        // 0 = pre-sublayer conv, 1 = post-sublayer conv
    uint dyn_stride;  // per-token dyn row stride = 2 * kernel_size * n_groups
};

// x:    [n_tokens, h] F32 — sublayer input (side 0: post-norm hidden;
//       side 1: sublayer output pre-residual)
// dyn:  [n_tokens, dyn_stride] F32 — dynamic coefficients from the conv
//       projection, laid out (group, tap, side) fastest-to-slowest per
//       token (ggml reshape [n_groups, kernel, 2, n_tokens])
// base: [h, kernel_size, 2] F32 — static base kernel, (channel, tap, side)
// y:    [n_tokens, h] F32 — MUST NOT alias x (taps read neighbor rows)
kernel void kernel_dflash2_conv_f32(
        constant dflash2_conv_args & args [[buffer(0)]],
        device const float * x    [[buffer(1)]],
        device const float * dyn  [[buffer(2)]],
        device const float * base [[buffer(3)]],
        device       float * y    [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint total = args.n_tokens * args.h;
    if (gid >= total) {
        return;
    }
    const uint t = gid / args.h;
    const uint c = gid % args.h;
    const uint g = c / args.group_size;

    const ulong base_side = (ulong)args.side * args.kernel_size * args.h;
    const ulong dyn_row   = (ulong)t * args.dyn_stride
                          + (ulong)args.side * args.kernel_size * args.n_groups;

    float acc = 0.0f;
    for (uint tap = 0; tap < args.kernel_size; ++tap) {
        if (t < tap) {
            break; // taps reach before the block start: zero padding
        }
        const float b = base[base_side + (ulong)tap * args.h + c];
        const float d = dyn[dyn_row + (ulong)tap * args.n_groups + g];
        acc = fma(b + d, x[(ulong)(t - tap) * args.h + c], acc);
    }
    y[gid] = acc;
}

constant constexpr ushort DFLASH2_TOP_K = 16;

inline int f32_total_order_key(float value) {
    int bits = as_type<int>(value);
    return bits ^ int(uint(bits >> 31) >> 1);
}

inline bool topk16_better(float value, uint index, float current, uint current_index) {
    const int value_key = f32_total_order_key(value);
    const int current_key = f32_total_order_key(current);
    return value_key > current_key ||
           (value_key == current_key && index < current_index);
}

struct topk16_args {
    uint n;        // row length (vocab size)
    uint stride_x; // row stride in elements
};

// One threadgroup per row. Pass 1: each thread keeps a sorted top-16 of
// its strided slice in registers. Pass 2: lists spill to threadgroup
// memory ([ntg * 16] value/index pairs) and thread 0 does a serial
// insertion merge — ntg*16 comparisons against the current 16th-best,
// insertions are rare. ntg is capped by the encoder so the threadgroup
// allocation stays within the 32 KB Apple GPU limit.
kernel void kernel_topk16_f32(
        constant topk16_args & args [[buffer(0)]],
        device const float * x       [[buffer(1)]],
        device       int   * out_idx [[buffer(2)]], // [n_rows, 16]
        device       float * out_val [[buffer(3)]], // [n_rows, 16]
        threadgroup  float * sh_val  [[threadgroup(0)]],
        threadgroup  uint  * sh_idx  [[threadgroup(1)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tpitg [[thread_position_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    const uint row = tgpig;
    device const float * x_row = x + (ulong)row * args.stride_x;

    float best_val[DFLASH2_TOP_K];
    uint  best_idx[DFLASH2_TOP_K];
    for (ushort j = 0; j < DFLASH2_TOP_K; ++j) {
        best_val[j] = -INFINITY;
        best_idx[j] = 0xFFFFFFFFu;
    }

    for (uint i = tpitg; i < args.n; i += ntg) {
        const float v = x_row[i];
        const ushort last = DFLASH2_TOP_K - 1;
        if (topk16_better(v, i, best_val[last], best_idx[last])) {
            short j = last;
            while (j > 0 && topk16_better(v, i, best_val[j - 1], best_idx[j - 1])) {
                best_val[j] = best_val[j - 1];
                best_idx[j] = best_idx[j - 1];
                --j;
            }
            best_val[j] = v;
            best_idx[j] = i;
        }
    }

    for (ushort j = 0; j < DFLASH2_TOP_K; ++j) {
        sh_val[(uint)tpitg * DFLASH2_TOP_K + j] = best_val[j];
        sh_idx[(uint)tpitg * DFLASH2_TOP_K + j] = best_idx[j];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tpitg == 0) {
        float top_val[DFLASH2_TOP_K];
        uint  top_idx[DFLASH2_TOP_K];
        for (ushort j = 0; j < DFLASH2_TOP_K; ++j) {
            top_val[j] = -INFINITY;
            top_idx[j] = 0xFFFFFFFFu;
        }
        const uint total = ntg * DFLASH2_TOP_K;
        const ushort last = DFLASH2_TOP_K - 1;
        for (uint e = 0; e < total; ++e) {
            const float v = sh_val[e];
            const uint  i = sh_idx[e];
            if (i == 0xFFFFFFFFu) {
                continue; // unfilled sentinel (n < ntg * 16 edge)
            }
            if (topk16_better(v, i, top_val[last], top_idx[last])) {
                short j = last;
                while (j > 0 && topk16_better(v, i, top_val[j - 1], top_idx[j - 1])) {
                    top_val[j] = top_val[j - 1];
                    top_idx[j] = top_idx[j - 1];
                    --j;
                }
                top_val[j] = v;
                top_idx[j] = i;
            }
        }
        device int   * dst_idx = out_idx + (ulong)row * DFLASH2_TOP_K;
        device float * dst_val = out_val + (ulong)row * DFLASH2_TOP_K;
        for (ushort j = 0; j < DFLASH2_TOP_K; ++j) {
            dst_idx[j] = (int)top_idx[j];
            dst_val[j] = top_val[j];
        }
    }
}
