// Elementwise + small ops needed for the inference forward pass.
// These are the "boring" kernels: trivial math, trivial parallelism,
// single-row activation in / single-row activation out. Batch=1
// single-token decode is the v1 target.
//
// Production form: every kernel takes its `n` from a small Pod arg
// passed via setBytes, not via a buffer. Keeps dispatch overhead minimal.
//
// CPU oracle: the corresponding free fns in crate::forward (silu,
// sigmoid, softplus, etc.).

#include <metal_stdlib>
using namespace metal;

struct n_args {
    uint n;
};

// y[i] = x[i] / (1 + exp(-x[i]))
kernel void kernel_silu_f32(
        constant n_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = x[tid];
    y[tid] = v / (1.0f + exp(-v));
}

// y[i] = 1 / (1 + exp(-x[i]))
kernel void kernel_sigmoid_f32(
        constant n_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[tid] = 1.0f / (1.0f + exp(-x[tid]));
}

// y[i] = log(1 + exp(x[i]))   (numerically-stable form)
kernel void kernel_softplus_f32(
        constant n_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = x[tid];
    if (v > 20.0f) {
        y[tid] = v;
    } else if (v < -20.0f) {
        y[tid] = exp(v);
    } else {
        y[tid] = log(1.0f + exp(v));
    }
}

// out[i] = a[i] + b[i]
kernel void kernel_add_f32(
        constant n_args & args [[buffer(0)]],
        device const float * a [[buffer(1)]],
        device const float * b [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    out[tid] = a[tid] + b[tid];
}

// In-place residual: x[i] += y[i]. Used for the two residual adds per
// transformer block. Saves the read-modify-write traffic vs an out-of-
// place add.
kernel void kernel_add_inplace_f32(
        constant n_args & args [[buffer(0)]],
        device       float * x [[buffer(1)]],
        device const float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    x[tid] += y[tid];
}

// out[i] = a[i] * b[i]
kernel void kernel_mul_f32(
        constant n_args & args [[buffer(0)]],
        device const float * a [[buffer(1)]],
        device const float * b [[buffer(2)]],
        device       float * out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    out[tid] = a[tid] * b[tid];
}

// SwiGLU FFN inner: ffn[i] = silu(gate[i]) * up[i].
// Two-input fused op, saves a separate silu kernel + intermediate buffer
// in the FFN path. (down(silu(gate(x)) * up(x)) is the full SwiGLU; this
// is the middle product, executed before the final down projection.)
kernel void kernel_silu_mul_f32(
        constant n_args & args [[buffer(0)]],
        device const float * gate [[buffer(1)]],
        device const float * up   [[buffer(2)]],
        device       float * out  [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float g = gate[tid];
    const float silu_g = g / (1.0f + exp(-g));
    out[tid] = silu_g * up[tid];
}

// Softmax along the (only) dimension. One threadgroup; up to 1024 threads.
// Used for attention scores (n = context_len, typically ≤ 4-256K). For
// large n we'd want a 2-pass shared-memory reduce; this version does one
// simdgroup-then-shared-mem reduce.
kernel void kernel_softmax_f32(
        constant n_args & args [[buffer(0)]],
        device       float * x  [[buffer(1)]],
        threadgroup  float * shmem [[threadgroup(0)]],
        uint  tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint  ntg   [[threads_per_threadgroup]]) {
    // Pass 1: max.
    float m = -INFINITY;
    for (uint i = tpitg; i < args.n; i += ntg) {
        m = max(m, x[i]);
    }
    m = simd_max(m);
    if (tiisg == 0) shmem[sgitg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : -INFINITY;
    m = simd_max(m);

    // Pass 2: sum of exp(x - m).
    float s = 0.0f;
    for (uint i = tpitg; i < args.n; i += ntg) {
        s += exp(x[i] - m);
    }
    s = simd_sum(s);
    if (tiisg == 0) shmem[sgitg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    s = simd_sum(s);

    const float inv = 1.0f / s;

    // Pass 3: write normalized.
    for (uint i = tpitg; i < args.n; i += ntg) {
        x[i] = exp(x[i] - m) * inv;
    }
}

// L2 norm: y = x / max(||x||, eps). Per-vector; single threadgroup. Used
// for Q/K l2-norm inside the GDN per-head path. CPU oracle:
// crate::forward::l2_norm_in_place. ggml semantics: the "max with eps"
// form, NOT 1/sqrt(sum + eps).
struct l2_norm_args {
    uint n_dim;
    float eps;
};
kernel void kernel_l2_norm_f32(
        constant l2_norm_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        threadgroup  float * shmem [[threadgroup(0)]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        const float v = x[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float scale = 1.0f / max(sqrt(sumsq), args.eps);
    for (uint i = tpitg; i < args.n_dim; i += ntg) {
        y[i] = x[i] * scale;
    }
}

// Batched per-head L2 norm: y[h, :] = x[h, :] / max(||x[h, :]||, eps),
// for h in 0..n_heads. One threadgroup per head. Used in the GDN
// front-end where Q and K are l2-normed per K-head before the
// recurrence kernel. Replaces n_heads separate dispatches with one.
struct l2_norm_batched_args {
    uint n_heads;
    uint head_dim;
    float eps;
};
kernel void kernel_l2_norm_batched_f32(
        constant l2_norm_batched_args & args [[buffer(0)]],
        device const float * x      [[buffer(1)]], // [n_heads, head_dim]
        device       float * y      [[buffer(2)]], // [n_heads, head_dim]
        threadgroup  float * shmem  [[threadgroup(0)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    const uint hi = tgpig;
    if (hi >= args.n_heads) return;

    device const float * x_h = x + (ulong)hi * args.head_dim;
    device       float * y_h = y + (ulong)hi * args.head_dim;

    float sumsq = 0.0f;
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        const float v = x_h[i];
        sumsq += v * v;
    }
    sumsq = simd_sum(sumsq);
    if (tiisg == 0) shmem[sgitg] = sumsq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumsq = (tiisg < (ntg + 31) / 32) ? shmem[tiisg] : 0.0f;
    sumsq = simd_sum(sumsq);

    const float scale = 1.0f / max(sqrt(sumsq), args.eps);
    for (uint i = tpitg; i < args.head_dim; i += ntg) {
        y_h[i] = x_h[i] * scale;
    }
}

// Copy with offset: y[i] = x[src_off + i]  for i in 0..n.
// Used to slice the GDN post-conv qkv buffer into per-role Q/K/V views
// without doing a CPU round-trip. v2 will replace this with kernels
// that read the slice via offset directly.
struct copy_offset_args {
    uint n;
    uint src_off;
};
kernel void kernel_copy_offset_f32(
        constant copy_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[tid] = x[args.src_off + tid];
}

// Scatter with destination offset: y[dst_off + i] = x[i]  for i in 0..n.
// Used for KV cache append (write current K/V at slot `position`).
struct scatter_offset_args {
    uint n;
    uint dst_off;
};
kernel void kernel_scatter_offset_f32(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       float * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[args.dst_off + tid] = x[tid];
}

// Same but writes F16 destination from F32 source. Used for F16 KV
// cache append. Precision: x is f32 in the model's residual stream,
// half() conversion is fine for K/V (llama.cpp also defaults to f16
// KV). The half value is what attn_decode_f16 will read back.
kernel void kernel_scatter_offset_f32_to_f16(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * x [[buffer(1)]],
        device       half  * y [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    y[args.dst_off + tid] = (half)x[tid];
}

// Fused K+V scatter: writes both K and V into their respective F16 caches
// in a single dispatch. K and V always share the same `n` (= kv_dim) and
// the same `dst_off` (= position * kv_dim) at decode time, so we amortize
// dispatch overhead across both writes.
//
// Per Jeff & Sanjay (Bulk APIs): "amortize boundary crossings". Saves
// 1 dispatch per attn layer × 16 attn layers = 16 dispatches/token.
kernel void kernel_scatter_offset_f32_to_f16_kv(
        constant scatter_offset_args & args [[buffer(0)]],
        device const float * k_src [[buffer(1)]],
        device const float * v_src [[buffer(2)]],
        device       half  * k_dst [[buffer(3)]],
        device       half  * v_dst [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint i = args.dst_off + tid;
    k_dst[i] = (half)k_src[tid];
    v_dst[i] = (half)v_src[tid];
}

// get_rows: y[r * n_cols + i] = embed[ids[r] * n_cols + i].
// For embedding lookup at decode (n_rows=1) and prefill (n_rows=batch).
struct get_rows_args {
    uint n_rows;
    uint n_cols;
};
kernel void kernel_get_rows_f32(
        constant get_rows_args & args [[buffer(0)]],
        device const float * embed [[buffer(1)]],
        device const int   * ids   [[buffer(2)]],
        device       float * y     [[buffer(3)]],
        uint2 gid [[thread_position_in_grid]]) {
    const uint r = gid.y;
    const uint i = gid.x;
    if (r >= args.n_rows || i >= args.n_cols) return;
    const int row = ids[r];
    y[r * args.n_cols + i] = embed[(uint)row * args.n_cols + i];
}

// GDN α-chain fusion (replaces 3 dispatches: add_inplace + softplus + mul).
//
//   out[i] = softplus(a[i] + dt_bias[i]) * a_log[i]
//
// Per the GDN forward pass:
//   gdn_alpha = softplus(gdn_a + dt_bias) * a_log
//
// Saves 2 dispatches per layer × 32 GDN layers = 64 dispatches/token.
// Per the v0.25 intra-profiler the alpha chain measured 0.043 ms/layer
// (4 dispatches), so this should shave ~1.5–2.5 ms/token. Keeps the same
// numerical-stability branch (v > 20: identity; v < -20: exp(v); else log(1+exp)).
//
// Inputs: a (mutable, will be added with dt_bias and softplus'd in-place),
// dt_bias (per-channel), a_log (per-channel scale).
// Output: written to `out` (out and a may alias).
kernel void kernel_gdn_alpha_chain_f32(
        constant n_args & args   [[buffer(0)]],
        device const float * a       [[buffer(1)]],
        device const float * dt_bias [[buffer(2)]],
        device const float * a_log   [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const float v = a[tid] + dt_bias[tid];
    float sp;
    if (v > 20.0f) {
        sp = v;
    } else if (v < -20.0f) {
        sp = exp(v);
    } else {
        sp = log(1.0f + exp(v));
    }
    out[tid] = sp * a_log[tid];
}

// Batched GDN α-chain (v0.73a layer-major batching).
//
//   out[r, c] = softplus(a[r, c] + dt_bias[c]) * a_log[c]
//
// Same numerics as kernel_gdn_alpha_chain_f32 (bit-identical when N=1)
// but broadcasts dt_bias and a_log across N rows so it can run once per
// GDN layer instead of N times. dt_bias and a_log are each `[n_v]`;
// a and out are each `[N, n_v]` row-major.
//
// Dispatch: 1D grid of size N * n_v; each thread handles one (r, c)
// element. r = tid / n_v ; c = tid % n_v.
struct gdn_alpha_chain_batched_args {
    uint n;       // total elements = N * n_v
    uint n_cols;  // n_v
};

kernel void kernel_gdn_alpha_chain_batched_f32(
        constant gdn_alpha_chain_batched_args & args [[buffer(0)]],
        device const float * a       [[buffer(1)]],
        device const float * dt_bias [[buffer(2)]],
        device const float * a_log   [[buffer(3)]],
        device       float * out     [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.n) return;
    const uint c = tid % args.n_cols;
    const float v = a[tid] + dt_bias[c];
    float sp;
    if (v > 20.0f) {
        sp = v;
    } else if (v < -20.0f) {
        sp = exp(v);
    } else {
        sp = log(1.0f + exp(v));
    }
    out[tid] = sp * a_log[c];
}

// =============================================================================
// kernel_argmax_f32 — GPU-side argmax for one row of length `n`.
//
// One threadgroup processes one row. Up to 1024 threads per threadgroup,
// reduced via simd_max + a small shmem step. Tie policy: LOWEST INDEX wins
// (matches numpy / torch semantics; tested explicitly in
// `metal::tests::argmax_matches_cpu_with_tie_to_lowest_index`).
//
// Used by H5.3a `packed_verify` to produce `verify_argmax: [N] i32`
// without a `[N, V]` CPU readback (saves 15.9 MB per outer step at
// V=248320, N=16). For packed-N argmax over `[N, V] -> [N] i32`, dispatch
// with `grid.width = N`; each TG handles row `tgpig.x`.
//
// Layout:
//   x         [n_rows, n] (row-major)        — input logits
//   out_idx   [n_rows]                       — output argmax index per row (i32)
//   stride_x  = n (row stride in elements)
//
// Reduction strategy:
//   Pass 1 — each lane scans strided slice; tracks (max_val, min_idx_at_max).
//   Pass 2 — simdgroup-wide reduce: pick max over lanes, breaking ties to
//            lowest index. Use simd_max for the value, then simd_min over
//            lanes whose value matches lane_max.
//   Pass 3 — across simdgroups via shmem; one thread writes out_idx.
//
// Edge-case contract (per codex H5.3a review + tests 9 / 10 in
// `argmax_matches_cpu_with_tie_to_lowest_index`):
//   * All-zero row     → returns 0 (every position ties; lowest idx wins).
//   * All −INFINITY row → returns 0 (every position ties at −inf;
//                          lowest idx wins, NOT -1).
//   * All-NaN row       → returns -1 (UINT_MAX cast). IEEE comparisons
//                          fail for any NaN operand, so no lane ever
//                          claims a candidate idx; simd_min(UINT_MAX) =
//                          UINT_MAX. Production lm_head should never
//                          produce a NaN row; if it does, the -1
//                          surfaces it.
//
// Bound: this kernel assumes `tg_threads ≤ 1024` (Metal hardware spec
// → n_simdgroups ≤ 32). The cross-simdgroup reduce in Pass 3 only runs
// in simdgroup 0; lanes 0..n_simdgroups-1 read shmem entries. If
// n_simdgroups > 32 we'd silently lose data — but Metal will reject
// the dispatch first.
// =============================================================================
struct argmax_args {
    uint n;          // length of each row
    uint stride_x;   // row stride in elements (= n unless caller asks for padding)
};

kernel void kernel_argmax_f32(
        constant argmax_args & args [[buffer(0)]],
        device const float   * x        [[buffer(1)]],
        device       int     * out_idx  [[buffer(2)]],
        threadgroup  float   * sh_val   [[threadgroup(0)]],
        threadgroup  uint    * sh_idx   [[threadgroup(1)]],
        uint   tgpig [[threadgroup_position_in_grid]],
        uint   tpitg [[thread_position_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]],
        ushort tiisg [[thread_index_in_simdgroup]],
        uint   ntg   [[threads_per_threadgroup]]) {
    const uint row = tgpig;
    device const float * x_row = x + (ulong)row * args.stride_x;

    // Pass 1: per-lane scan, tracking (best_val, best_idx). Tie-break to
    // lower idx within a single lane's traversal.
    float best_val = -INFINITY;
    uint  best_idx = UINT_MAX;
    for (uint i = tpitg; i < args.n; i += ntg) {
        const float v = x_row[i];
        if (v > best_val || (v == best_val && i < best_idx)) {
            best_val = v;
            best_idx = i;
        }
    }

    // Pass 2: simdgroup reduce. simd_max gives us the value; then we
    // broadcast each lane's idx and pick the smallest idx among lanes
    // whose val equals the max.
    const float lane_max = simd_max(best_val);
    // Lanes whose val < lane_max get idx = UINT_MAX (will lose the
    // simd_min tiebreak); lanes that match get their actual idx.
    uint lane_idx = (best_val == lane_max) ? best_idx : UINT_MAX;
    const uint sg_idx = simd_min(lane_idx);

    if (tiisg == 0) {
        sh_val[sgitg] = lane_max;
        sh_idx[sgitg] = sg_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Pass 3: cross-simdgroup reduce (simdgroup 0 only).
    const uint n_simdgroups = (ntg + 31) / 32;
    if (sgitg == 0) {
        const float sgv = (tiisg < n_simdgroups) ? sh_val[tiisg] : -INFINITY;
        const uint  sgi = (tiisg < n_simdgroups) ? sh_idx[tiisg] : UINT_MAX;
        const float global_max = simd_max(sgv);
        const uint  match_idx = (sgv == global_max) ? sgi : UINT_MAX;
        const uint  global_idx = simd_min(match_idx);
        if (tiisg == 0) {
            out_idx[row] = (int)global_idx;
        }
    }
}
