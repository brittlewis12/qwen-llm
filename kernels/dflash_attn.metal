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
//     * Full-attn layer (Qwen3.6: layer 4): allow every ctx key. Noise
//       keys remain block-causal as in the SWA layers.
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
    uint  n_rows;                // N noise rows (= block_size). Codex code-review
                                 // catch: previous version had a bogus
                                 // `q_idx >= n_q_heads * 16` guard (no n_rows
                                 // arg). Real bound is q_idx < n_rows.
    uint  noise_start_pos;
    uint  swa_window;            // 0 means full-attn layer (no SWA over ctx);
                                 // codex flag: full-attn ALSO drops the causal
                                 // restriction over ctx, matching the CPU
                                 // oracle in `forward.rs::dflash_draft`.
    uint  ctx_scan_start;        // first context row worth scanning for SWA;
                                 // ignored for full-attn layers.
    float scale;                 // 1/sqrt(head_dim)
    uint  noncausal_noise;       // A/B experiment (QWEN_DFLASH_NONCAUSAL_NOISE):
                                 // nonzero drops the block-causal restriction
                                 // over noise keys (llama.cpp sets
                                 // `causal_attn=false` for the DFlash drafter;
                                 // this engine has historically masked
                                 // noise_idx > q_idx).
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
    // Codex code-review v0.72.2: real bounds use `n_rows`, not the
    // bogus `n_q_heads * 16` placeholder. A dispatch-shape bug or
    // different block_size would silently OOB read/write q/o without
    // this fix.
    if (q_head >= args.n_q_heads || q_idx >= args.n_rows) return;
    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kv_head = q_head / group;
    const uint head_dim = args.head_dim;
    const ushort dk_per_lane = head_dim / NW_DFLASH;   // 4 for head_dim=128
    const uint q_pos = args.noise_start_pos + q_idx;
    const bool full_attn = (args.swa_window == 0);

    // ---- Load Q vector for this (q_idx, q_head) into per-lane registers ----
    // Sized for head_dim ∈ [32, 256]. Host wrapper rejects head_dim
    // > 256. Codex code-review v0.72.2: previous version sized [8] with
    // a comment but no host check; head_dim=320 would stack-OOB.
    float q_reg[8];
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
    //
    // Mask semantics (codex code-review v0.72.2 — match CPU oracle in
    // forward.rs::dflash_draft):
    //   * full-attn ctx key: ALWAYS allowed (no causal restriction).
    //     CPU oracle treats ctx as committed past + the dflash recipe
    //     allows the full-attn layer to peek.
    //   * SWA ctx key: allowed iff causal (k_pos <= q_pos) AND
    //                  windowed (q_pos - k_pos <= swa_window).
    //     causal short-circuits FIRST so the unsigned subtraction
    //     never wraps.
    //   * Noise key: block-causal (noise_idx <= q_idx).
    float m_run = -INFINITY;
    const uint kk_start = full_attn ? 0 : min(args.ctx_scan_start, args.ctx_len);
    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_k[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
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
    // Same mask as PASS 1 — see PASS 1 comment.
    float l_sum = 0.0f;
    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_k[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
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
    // Same mask as PASS 1 / PASS 2.
    float o_acc[8];
    for (ushort j = 0; j < dk_per_lane; ++j) o_acc[j] = 0.0f;

    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_k[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
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

// Same math and mask contract as `kernel_dflash_attn_f32`, but reads the
// committed-context K/V cache and the N noise rows from two separate ranges.
// This avoids materializing `k_full/v_full = ctx || noise` before every
// drafter attention layer. It deliberately keeps the 3-pass softmax shape so
// the dataflow change can be isolated before the online-softmax rewrite.
kernel void kernel_dflash_attn_two_range_f32(
        constant dflash_attn_args & args [[buffer(0)]],
        device const float * q          [[buffer(1)]],
        device const float * k_ctx      [[buffer(2)]],
        device const float * v_ctx      [[buffer(3)]],
        device const float * k_noise    [[buffer(4)]],
        device const float * v_noise    [[buffer(5)]],
        device const int   * pos_ctx    [[buffer(6)]],
        device       float * o          [[buffer(7)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint q_head = tgpig.x;
    const uint q_idx  = tgpig.y;
    if (q_head >= args.n_q_heads || q_idx >= args.n_rows) return;
    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kv_head = q_head / group;
    const uint head_dim = args.head_dim;
    const ushort dk_per_lane = head_dim / NW_DFLASH;
    const uint q_pos = args.noise_start_pos + q_idx;
    const bool full_attn = (args.swa_window == 0);
    const ulong k_stride = (ulong)args.n_kv_heads * head_dim;
    const uint kk_start = full_attn ? 0 : min(args.ctx_scan_start, args.ctx_len);

    float q_reg[8];
    {
        device const float * q_base =
            q + ((ulong)q_idx * args.n_q_heads + (ulong)q_head) * head_dim;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            q_reg[j] = q_base[j * NW_DFLASH + tiisg];
        }
    }

    float m_run = -INFINITY;
    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        device const float * k_row;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_ctx[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
            k_row = k_ctx + (ulong)kk * k_stride + (ulong)kv_head * head_dim;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
            k_row = k_noise + (ulong)noise_idx * k_stride + (ulong)kv_head * head_dim;
        }
        if (!allowed) continue;
        float partial = 0.0f;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            partial += q_reg[j] * k_row[j * NW_DFLASH + tiisg];
        }
        const float s = simd_sum(partial) * args.scale;
        m_run = max(m_run, s);
    }

    float l_sum = 0.0f;
    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        device const float * k_row;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_ctx[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
            k_row = k_ctx + (ulong)kk * k_stride + (ulong)kv_head * head_dim;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
            k_row = k_noise + (ulong)noise_idx * k_stride + (ulong)kv_head * head_dim;
        }
        if (!allowed) continue;
        float partial = 0.0f;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            partial += q_reg[j] * k_row[j * NW_DFLASH + tiisg];
        }
        const float s = simd_sum(partial) * args.scale;
        l_sum += exp(s - m_run);
    }
    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;

    float o_acc[8];
    for (ushort j = 0; j < dk_per_lane; ++j) o_acc[j] = 0.0f;

    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        device const float * k_row;
        device const float * v_row;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_ctx[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
            k_row = k_ctx + (ulong)kk * k_stride + (ulong)kv_head * head_dim;
            v_row = v_ctx + (ulong)kk * k_stride + (ulong)kv_head * head_dim;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
            k_row = k_noise + (ulong)noise_idx * k_stride + (ulong)kv_head * head_dim;
            v_row = v_noise + (ulong)noise_idx * k_stride + (ulong)kv_head * head_dim;
        }
        if (!allowed) continue;
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

    device float * o_base =
        o + ((ulong)q_idx * args.n_q_heads + (ulong)q_head) * head_dim;
    for (ushort j = 0; j < dk_per_lane; ++j) {
        o_base[j * NW_DFLASH + tiisg] = o_acc[j];
    }
}

// Online-softmax form of the two-range DFlash attention kernel. This traverses
// ctx+noise once, keeping the running max, normalization denominator, and output
// accumulator in registers. It removes the 3x QK recompute in the legacy kernel
// while preserving the exact mask semantics.
kernel void kernel_dflash_attn_online_two_range_f32(
        constant dflash_attn_args & args [[buffer(0)]],
        device const float * q          [[buffer(1)]],
        device const float * k_ctx      [[buffer(2)]],
        device const float * v_ctx      [[buffer(3)]],
        device const float * k_noise    [[buffer(4)]],
        device const float * v_noise    [[buffer(5)]],
        device const int   * pos_ctx    [[buffer(6)]],
        device       float * o          [[buffer(7)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint q_head = tgpig.x;
    const uint q_idx  = tgpig.y;
    if (q_head >= args.n_q_heads || q_idx >= args.n_rows) return;
    const uint group = args.n_q_heads / args.n_kv_heads;
    const uint kv_head = q_head / group;
    const uint head_dim = args.head_dim;
    const ushort dk_per_lane = head_dim / NW_DFLASH;
    const uint q_pos = args.noise_start_pos + q_idx;
    const bool full_attn = (args.swa_window == 0);
    const ulong k_stride = (ulong)args.n_kv_heads * head_dim;
    const uint kk_start = full_attn ? 0 : min(args.ctx_scan_start, args.ctx_len);

    float q_reg[8];
    {
        device const float * q_base =
            q + ((ulong)q_idx * args.n_q_heads + (ulong)q_head) * head_dim;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            q_reg[j] = q_base[j * NW_DFLASH + tiisg];
        }
    }

    float m_run = -INFINITY;
    float l_sum = 0.0f;
    float o_acc[8];
    for (ushort j = 0; j < dk_per_lane; ++j) o_acc[j] = 0.0f;

    for (uint kk = kk_start; kk < args.n_kv_total; ++kk) {
        bool allowed;
        device const float * k_row;
        device const float * v_row;
        if (kk < args.ctx_len) {
            if (full_attn) {
                allowed = true;
            } else {
                const uint k_pos = (uint)pos_ctx[kk];
                const bool causal = (k_pos <= q_pos);
                allowed = causal && ((q_pos - k_pos) <= args.swa_window);
            }
            k_row = k_ctx + (ulong)kk * k_stride + (ulong)kv_head * head_dim;
            v_row = v_ctx + (ulong)kk * k_stride + (ulong)kv_head * head_dim;
        } else {
            const uint noise_idx = kk - args.ctx_len;
            allowed = (args.noncausal_noise != 0) || (noise_idx <= q_idx);
            k_row = k_noise + (ulong)noise_idx * k_stride + (ulong)kv_head * head_dim;
            v_row = v_noise + (ulong)noise_idx * k_stride + (ulong)kv_head * head_dim;
        }
        if (!allowed) continue;

        float partial = 0.0f;
        for (ushort j = 0; j < dk_per_lane; ++j) {
            partial += q_reg[j] * k_row[j * NW_DFLASH + tiisg];
        }
        const float s = simd_sum(partial) * args.scale;
        const float m_new = max(m_run, s);
        const float old_scale = exp(m_run - m_new);
        const float new_scale = exp(s - m_new);
        for (ushort j = 0; j < dk_per_lane; ++j) {
            o_acc[j] = o_acc[j] * old_scale + new_scale * v_row[j * NW_DFLASH + tiisg];
        }
        l_sum = l_sum * old_scale + new_scale;
        m_run = m_new;
    }

    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    device float * o_base =
        o + ((ulong)q_idx * args.n_q_heads + (ulong)q_head) * head_dim;
    for (ushort j = 0; j < dk_per_lane; ++j) {
        o_base[j * NW_DFLASH + tiisg] = o_acc[j] * inv_l;
    }
}

// Fixed Qwen3.6 DFlash full-attention path. One simdgroup handles the four
// sibling Q heads for a KV head, so each K/V value is loaded once instead of
// once per sibling. Four context partitions preserve the baseline's 512 main
// threadgroups. The final partition also consumes the causal noise prefix.
[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_dflash_attn_full_gqa_split4_main_f32(
        constant dflash_attn_args & args [[buffer(0)]],
        device const float * q          [[buffer(1)]],
        device const float * k_ctx      [[buffer(2)]],
        device const float * v_ctx      [[buffer(3)]],
        device const float * k_noise    [[buffer(4)]],
        device const float * v_noise    [[buffer(5)]],
        device       float * o_partial  [[buffer(6)]],
        device       float * ml_partial [[buffer(7)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint NKV = 8;
    constexpr uint GROUP = 4;
    constexpr uint HD = 128;
    constexpr uint SPLIT = 4;
    const uint kv_head = tgpig.x;
    const uint q_idx = tgpig.y;
    const uint part = tgpig.z;
    if (kv_head >= NKV || q_idx >= 16 || part >= SPLIT) return;

    const ulong q_row = (ulong)q_idx * 32 * HD + (ulong)kv_head * GROUP * HD;
    const uint d0 = tiisg;
    const uint d1 = d0 + 32;
    const uint d2 = d1 + 32;
    const uint d3 = d2 + 32;
    const float4 q0 = float4(q[q_row + d0], q[q_row + d1],
                             q[q_row + d2], q[q_row + d3]);
    const float4 q1 = float4(q[q_row + HD + d0], q[q_row + HD + d1],
                             q[q_row + HD + d2], q[q_row + HD + d3]);
    const float4 q2 = float4(q[q_row + 2 * HD + d0], q[q_row + 2 * HD + d1],
                             q[q_row + 2 * HD + d2], q[q_row + 2 * HD + d3]);
    const float4 q3 = float4(q[q_row + 3 * HD + d0], q[q_row + 3 * HD + d1],
                             q[q_row + 3 * HD + d2], q[q_row + 3 * HD + d3]);

    float4 m_run = float4(-INFINITY);
    float4 l_sum = float4(0.0f);
    float4 o0 = float4(0.0f);
    float4 o1 = float4(0.0f);
    float4 o2 = float4(0.0f);
    float4 o3 = float4(0.0f);
    const uint ctx_begin = uint(((ulong)args.ctx_len * part) / SPLIT);
    const uint ctx_end = uint(((ulong)args.ctx_len * (part + 1)) / SPLIT);
    const uint scan_end = ctx_end + ((part == SPLIT - 1) ? q_idx + 1 : 0);
    const ulong kv_stride = NKV * HD;

    for (uint kk = ctx_begin; kk < scan_end; ++kk) {
        device const float * k_row;
        device const float * v_row;
        if (kk < ctx_end) {
            k_row = k_ctx + (ulong)kk * kv_stride + (ulong)kv_head * HD;
            v_row = v_ctx + (ulong)kk * kv_stride + (ulong)kv_head * HD;
        } else {
            const uint noise_idx = kk - ctx_end;
            k_row = k_noise + (ulong)noise_idx * kv_stride + (ulong)kv_head * HD;
            v_row = v_noise + (ulong)noise_idx * kv_stride + (ulong)kv_head * HD;
        }
        const float4 k4 = float4(k_row[d0], k_row[d1], k_row[d2], k_row[d3]);
        const float4 s = float4(simd_sum(dot(q0, k4)), simd_sum(dot(q1, k4)),
                                simd_sum(dot(q2, k4)), simd_sum(dot(q3, k4))) * args.scale;
        const float4 m_new = max(m_run, s);
        const float4 old_scale = exp(m_run - m_new);
        const float4 new_scale = exp(s - m_new);
        const float4 v4 = float4(v_row[d0], v_row[d1], v_row[d2], v_row[d3]);
        o0 = o0 * old_scale.x + v4 * new_scale.x;
        o1 = o1 * old_scale.y + v4 * new_scale.y;
        o2 = o2 * old_scale.z + v4 * new_scale.z;
        o3 = o3 * old_scale.w + v4 * new_scale.w;
        l_sum = l_sum * old_scale + new_scale;
        m_run = m_new;
    }

    const ulong group_base = (((ulong)q_idx * NKV + kv_head) * SPLIT + part) * GROUP;
    device float * po = o_partial + group_base * HD;
    po[d0] = o0.x; po[d1] = o0.y; po[d2] = o0.z; po[d3] = o0.w;
    po[HD + d0] = o1.x; po[HD + d1] = o1.y;
    po[HD + d2] = o1.z; po[HD + d3] = o1.w;
    po[2 * HD + d0] = o2.x; po[2 * HD + d1] = o2.y;
    po[2 * HD + d2] = o2.z; po[2 * HD + d3] = o2.w;
    po[3 * HD + d0] = o3.x; po[3 * HD + d1] = o3.y;
    po[3 * HD + d2] = o3.z; po[3 * HD + d3] = o3.w;
    if (tiisg == 0) {
        device float * ml = ml_partial + group_base * 2;
        ml[0] = m_run.x; ml[1] = l_sum.x;
        ml[2] = m_run.y; ml[3] = l_sum.y;
        ml[4] = m_run.z; ml[5] = l_sum.z;
        ml[6] = m_run.w; ml[7] = l_sum.w;
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_dflash_attn_full_gqa_split4_reduce_f32(
        constant dflash_attn_args & args [[buffer(0)]],
        device const float * o_partial  [[buffer(1)]],
        device const float * ml_partial [[buffer(2)]],
        device       float * o          [[buffer(3)]],
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    constexpr uint NKV = 8;
    constexpr uint GROUP = 4;
    constexpr uint HD = 128;
    constexpr uint SPLIT = 4;
    const uint q_head = tgpig.x;
    const uint q_idx = tgpig.y;
    if (q_head >= 32 || q_idx >= 16) return;
    const uint kv_head = q_head / GROUP;
    const uint g = q_head % GROUP;
    const ulong pair_base = (ulong)q_idx * NKV + kv_head;

    float4 m;
    float4 l;
    for (uint part = 0; part < SPLIT; ++part) {
        const ulong group_base = (pair_base * SPLIT + part) * GROUP + g;
        m[part] = ml_partial[group_base * 2];
        l[part] = ml_partial[group_base * 2 + 1];
    }
    float m_global = -INFINITY;
    for (uint part = 0; part < SPLIT; ++part) {
        if (l[part] > 0.0f) m_global = max(m_global, m[part]);
    }
    float4 factor = float4(0.0f);
    for (uint part = 0; part < SPLIT; ++part) {
        if (l[part] > 0.0f) factor[part] = exp(m[part] - m_global);
    }
    const float denom = dot(l, factor);

    const uint d0 = tiisg;
    const uint d1 = d0 + 32;
    const uint d2 = d1 + 32;
    const uint d3 = d2 + 32;
    float4 acc = float4(0.0f);
    for (uint part = 0; part < SPLIT; ++part) {
        const ulong group_base = (pair_base * SPLIT + part) * GROUP + g;
        device const float * po = o_partial + group_base * HD;
        acc += float4(po[d0], po[d1], po[d2], po[d3]) * factor[part];
    }
    const float inv_denom = (denom > 0.0f) ? 1.0f / denom : 0.0f;
    device float * out = o + ((ulong)q_idx * 32 + q_head) * HD;
    out[d0] = acc.x * inv_denom;
    out[d1] = acc.y * inv_denom;
    out[d2] = acc.z * inv_denom;
    out[d3] = acc.w * inv_denom;
}
