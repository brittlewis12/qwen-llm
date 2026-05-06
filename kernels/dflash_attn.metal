// Drafter attention kernel for DFlash speculative decode.
//
// Distinct from `attn_v4.metal` and `attn.metal` (target's flash-attn-v4
// + naive single-Q-decode kernels) because the drafter has a different
// attention shape:
//
//   * Q is [N, n_q · head_dim]  (N=16 noise rows, NOT 1)
//   * K, V are [n_kv_total, n_kv · head_dim] where
//     n_kv_total = ctx_len + N  (concat of cross-context + noise rows)
//   * Cross-context K, V come from the per-layer K/V projection of
//     `target_ctx` (the ctx_len target hidden columns); noise K, V from
//     the noise rows themselves.
//   * Mask is per-layer (sliding_window_pattern[layer]):
//     * SWA layers (Qwen3.6: layers 0..3): allow ctx key k iff
//       `k_pos <= q_pos` AND `q_pos - k_pos <= swa_window`. Noise key
//       k iff `k_pos == k - ctx_len <= q_idx` (block-causal over noise).
//     * Full-attn layer (Qwen3.6: layer 4): allow ctx key k iff
//       `k_pos <= q_pos` (NO sliding window). Noise key block-causal
//       same as SWA.
//   * Q/K-norm + RoPE are applied OUTSIDE this kernel (already Metal
//     since v0.51). This kernel just does scoring + softmax + V agg.
//
// Codex's "mask defensiveness" flag (v0.72-design): require
// `k_pos <= q_pos` for ctx keys explicitly. The current CPU
// fallback uses `q_pos.saturating_sub(k_pos) <= swa_window` which
// would allow future ctx positions (k_pos > q_pos) if an invariant
// slipped. This kernel's mask is `(k_pos <= q_pos) AND (q_pos - k_pos
// <= swa_window OR full_attn)` — both clauses required.
//
// Tile/thread layout (codex Q2 design B — custom small-N fused):
//
//   threadgroup grid = (n_q_heads, N) → for Qwen3.6 drafter that's
//     (n_q=32, N=16) = 512 threadgroups per layer. Each TG produces
//     ONE Q-row × Q-head output of length head_dim=128.
//
//   threads per TG = 32 (one simdgroup). Each lane handles
//     head_dim / 32 = 4 output dims in registers. simd_sum reductions
//     for Q·K dot products.
//
// Algorithm (per (q_idx, q_head)):
//
//   kv_head = q_head / group
//   load Q[q_idx, q_head, :] into 4 registers per lane (head_dim=128
//     across 32 lanes)
//
//   PASS 1 — running max:
//     For each k in 0..n_kv_total:
//       if mask deny: skip
//       compute s = simd_sum( Q[d_per_lane] * K[k, kv_head, d_per_lane] ) * scale
//       update m_running = max(m_running, s)
//
//   PASS 2 — running sum (stable softmax):
//     l_sum = 0
//     For each k in 0..n_kv_total:
//       if mask deny: skip
//       compute s = simd_sum(Q · K[k, kv_head, :]) * scale  (recompute)
//       l_sum += exp(s - m_running)
//
//   PASS 3 — V aggregate:
//     For each k in 0..n_kv_total:
//       if mask deny: skip
//       compute s = simd_sum(Q · K[k, kv_head, :]) * scale  (recompute)
//       w = exp(s - m_running) / l_sum
//       o[d_per_lane] += w * V[k, kv_head, d_per_lane]
//     write o[d_per_lane] to attn_o[q_idx, q_head, d_per_lane]
//
// 3-pass over k (recomputing Q·K each pass) is wasteful in compute but
// avoids materializing scores in threadgroup memory (which would fail
// at long ctx; n_kv_total × 4 B exceeds the 32 KB threadgroup mem
// limit at ctx > ~7K). For drafter perf this is fine: the pass-1
// alternative (online softmax with rescaling) saves 2 K/V loads but
// adds rescaling complexity. v1 simplicity wins; v2 can fuse if
// profile shows attention BW-bound.
//
// Buffers:
//   q          [N, n_q · head_dim]      F32, RoPE'd Q noise rows
//   k          [n_kv_total, n_kv · head_dim]   F32, RoPE'd K (ctx + noise)
//   v          [n_kv_total, n_kv · head_dim]   F32, V (ctx + noise)
//   pos_k      [n_kv_total]              i32, absolute K positions
//                                         (ctx pos for k < ctx_len,
//                                          drafter_pos + (k - ctx_len)
//                                          for k >= ctx_len)
//   o          [N, n_q · head_dim]       F32, output
//
// Args:
//   n_q_heads, n_kv_heads, head_dim, n_kv_total, ctx_len,
//   noise_start_pos, swa_window (0 = full-attn layer), scale = 1/sqrt(head_dim)

#include <metal_stdlib>
using namespace metal;

constant constexpr ushort NW_DFLASH = 32;            // simdgroup width

struct dflash_attn_args {
    uint  n_q_heads;
    uint  n_kv_heads;
    uint  head_dim;
    uint  n_kv_total;
    uint  ctx_len;
    uint  noise_start_pos;
    uint  swa_window;            // 0 means full-attn layer (no SWA)
    float scale;                 // 1/sqrt(head_dim)
};

kernel void kernel_dflash_attn_f32(
        constant dflash_attn_args & args [[buffer(0)]],
        device const float * q       [[buffer(1)]],
        device const float * k       [[buffer(2)]],
        device const float * v       [[buffer(3)]],
        device const int   * pos_k   [[buffer(4)]],
        device       float * o       [[buffer(5)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint q_head = tgpig.x;
    const uint q_idx  = tgpig.y;
    if (q_head >= args.n_q_heads || q_idx >= args.n_q_heads * 16) return; // belt-and-suspenders
    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kv_head = q_head / group;
    const uint head_dim = args.head_dim;
    const ushort dk_per_lane = head_dim / NW_DFLASH;   // 4 for head_dim=128
    const uint q_pos = args.noise_start_pos + q_idx;
    const bool full_attn = (args.swa_window == 0);

    // ---- Load Q vector for this (q_idx, q_head) into per-lane registers ----
    float q_reg[8];   // up to head_dim=256/32 = 8; for 128 we use 4
    {
        device const float * q_base =
            q + ((ulong)q_idx * args.n_q_heads + (ulong)q_head) * head_dim;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            q_reg[j] = q_base[j * NW_DFLASH + tiisg];
        }
    }

    // K/V row stride (in floats).
    const ulong k_stride = (ulong)args.n_kv_heads * head_dim;

    // Helper: compute s = (Q · K[k_global, kv_head]) * scale, with
    // simd_sum reduction across lanes. Returns the same scalar to all
    // lanes in the simdgroup.
    //
    // Mask check: returns false (and -inf score) if denied; mask
    // logic is inlined per pass to avoid a separate branch barrier.

    // ---- PASS 1: running max ----
    float m_run = -INFINITY;
    for (uint kk = 0; kk < args.n_kv_total; ++kk) {
        // Mask
        bool allowed;
        if (kk < args.ctx_len) {
            const int k_pos = pos_k[kk];
            const bool causal = ((uint)k_pos <= q_pos);
            const bool windowed =
                full_attn || (q_pos - (uint)k_pos <= args.swa_window);
            allowed = causal && windowed;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (noise_idx <= q_idx);
        }
        if (!allowed) continue;
        device const float * k_row = k + (ulong)kk * k_stride
                                       + (ulong)kv_head * head_dim;
        float partial = 0.0f;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            partial += q_reg[j] * k_row[j * NW_DFLASH + tiisg];
        }
        const float s = simd_sum(partial) * args.scale;
        m_run = max(m_run, s);
    }

    // ---- PASS 2: running sum (stable softmax) ----
    float l_sum = 0.0f;
    for (uint kk = 0; kk < args.n_kv_total; ++kk) {
        bool allowed;
        if (kk < args.ctx_len) {
            const int k_pos = pos_k[kk];
            const bool causal = ((uint)k_pos <= q_pos);
            const bool windowed =
                full_attn || (q_pos - (uint)k_pos <= args.swa_window);
            allowed = causal && windowed;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (noise_idx <= q_idx);
        }
        if (!allowed) continue;
        device const float * k_row = k + (ulong)kk * k_stride
                                       + (ulong)kv_head * head_dim;
        float partial = 0.0f;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            partial += q_reg[j] * k_row[j * NW_DFLASH + tiisg];
        }
        const float s = simd_sum(partial) * args.scale;
        l_sum += exp(s - m_run);
    }
    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;

    // ---- PASS 3: V aggregate ----
    float o_acc[8];
    for (ushort j = 0; j < dk_per_lane; ++j) o_acc[j] = 0.0f;

    for (uint kk = 0; kk < args.n_kv_total; ++kk) {
        bool allowed;
        if (kk < args.ctx_len) {
            const int k_pos = pos_k[kk];
            const bool causal = ((uint)k_pos <= q_pos);
            const bool windowed =
                full_attn || (q_pos - (uint)k_pos <= args.swa_window);
            allowed = causal && windowed;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (noise_idx <= q_idx);
        }
        if (!allowed) continue;
        device const float * k_row = k + (ulong)kk * k_stride
                                       + (ulong)kv_head * head_dim;
        device const float * v_row = v + (ulong)kk * k_stride
                                       + (ulong)kv_head * head_dim;
        // Recompute s.
        float partial = 0.0f;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            partial += q_reg[j] * k_row[j * NW_DFLASH + tiisg];
        }
        const float s = simd_sum(partial) * args.scale;
        const float w = exp(s - m_run) * inv_l;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            o_acc[j] += w * v_row[j * NW_DFLASH + tiisg];
        }
    }

    // ---- Write output ----
    device float * o_base =
        o + ((ulong)q_idx * args.n_q_heads + (ulong)q_head) * head_dim;
    for (ushort j = 0; j < dk_per_lane; ++j) {
        o_base[j * NW_DFLASH + tiisg] = o_acc[j];
    }
}
