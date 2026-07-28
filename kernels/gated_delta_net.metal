// Gated DeltaNet recurrence step (single-token decode).
//
// One step of the per-V-head delta-rule recurrence:
//
//     S ← exp(g) · S + β · (v − exp(g) · (S · k)) ⊗ k
//     o = S · q
//
// The usual 1/√head_dim factor is folded into the following RMSNormGated
// epsilon, so this kernel deliberately emits the unscaled S·q value.
//
// where S ∈ R^{head_dim × head_dim} is the per-head SSM state, indexed
// as S[dv, dk]; k, v, q are length-head_dim vectors per head; g and β
// are scalars per head; and the recurrence runs once per V-head.
//
// State is stored in device memory: [n_v_heads, head_dim, head_dim]
// row-major as `state[hi * Hk*Hv + dv * Hk + dk]`. The kernel reads the
// row `S[dv, :]` for its (hi, dv) threadgroup, mutates it in registers,
// writes back at the end. For multi-token prefill we'd loop the recurrence
// inside the kernel and amortize the state I/O across timesteps; v1
// targets single-token decode where T=1 and that optimization saves
// nothing.
//
// Threadgroup mapping:
//   tgpig.x = dv  (V-row index ∈ [0, head_dim))
//   tgpig.y = hi  (V-head index ∈ [0, n_v_heads))
//   threads per group = 32 (one simdgroup), each lane handles
//     head_dim/32 = 4 dk positions of S[dv, :] in registers.
//
// Hardcoded for head_dim=128 (Qwen3.5/3.6 GDN). When that changes, the
// `dks_per_lane = head_dim / 32` constant needs templating.
//
// CPU oracle: the inner loop in `crate::forward::gdn_step` (the per-V-head
// `for hi { sk = …; delta = …; ssm += …; o = … }` block).

#include <metal_stdlib>
using namespace metal;

constant constexpr ushort HEAD_DIM = 128;
constant constexpr ushort SIMD_LANES = 32;
constant constexpr ushort DKS_PER_LANE = HEAD_DIM / SIMD_LANES; // 4

struct gdn_step_args {
    uint n_v_heads;
    uint n_k_heads;  // q/k have this many heads; we map V-head hi -> K-head (hi % n_k_heads)
};

struct gdn_step_packed_args {
    uint n_tokens;
    uint n_v_heads;
    uint n_k_heads;
};

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_gdn_step_f32(
        constant gdn_step_args & args  [[buffer(0)]],
        device const float     * q     [[buffer(1)]], // [n_k_heads, head_dim]
        device const float     * k     [[buffer(2)]], // [n_k_heads, head_dim]
        device const float     * v     [[buffer(3)]], // [n_v_heads, head_dim]
        device const float     * g     [[buffer(4)]], // [n_v_heads]   per-head decay (already exp-arg)
        device const float     * beta  [[buffer(5)]], // [n_v_heads]   per-head, already sigmoid'd
        device       float     * state [[buffer(6)]], // [n_v_heads, head_dim, head_dim]
        device       float     * out   [[buffer(7)]], // [n_v_heads, head_dim]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv = tgpig.x;
    const uint hi = tgpig.y;
    if (hi >= args.n_v_heads || dv >= HEAD_DIM) return;

    // Q/K head index: GGML-style repeat tiling, hi % n_k_heads.
    // For 0.8B (n_v == n_k) this is identity; for 27B (n_v=48, n_k=16) this
    // gives [h0,h1,...,h15, h0,h1,...,h15, h0,h1,...,h15] — matching
    // ggml_repeat_4d semantics in qwen35.cpp.
    const uint hk = hi % args.n_k_heads;

    // Pointer to S[hi, dv, :] — one row of head_dim elements.
    device float * s_row = state + (ulong)hi * HEAD_DIM * HEAD_DIM
                                  + (ulong)dv * HEAD_DIM;

    // Per-head vectors.
    device const float * q_h = q + (ulong)hk * HEAD_DIM;
    device const float * k_h = k + (ulong)hk * HEAD_DIM;
    const float v_dv = v[(ulong)hi * HEAD_DIM + dv];
    const float g_exp = exp(g[hi]);
    const float beta_h = beta[hi];

    // Each lane holds 4 contiguous dk positions of S[dv, :] in registers.
    // Lane `tiisg` owns dk ∈ [tiisg*DKS_PER_LANE, (tiisg+1)*DKS_PER_LANE).
    float s_reg[DKS_PER_LANE];
    float k_reg[DKS_PER_LANE];
    float q_reg[DKS_PER_LANE];

    const ushort dk_base = tiisg * DKS_PER_LANE;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] = s_row[dk_base + j];
        k_reg[j] = k_h[dk_base + j];
        q_reg[j] = q_h[dk_base + j];
    }

    // (1) Decay: S[dv, :] *= exp(g)
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] *= g_exp;
    }

    // (2) Compute s_k = sum_dk S[dv, dk] * k[dk]   (post-decay)
    float sk_partial = 0.0f;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        sk_partial += s_reg[j] * k_reg[j];
    }
    const float sk = simd_sum(sk_partial);

    // (3) delta = (v[dv] - sk) * beta
    const float delta = (v_dv - sk) * beta_h;

    // (4) Update: S[dv, dk] += delta * k[dk]
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] += delta * k_reg[j];
    }

    // (5) o[dv] = sum_dk S[dv, dk] * q[dk]   (post-update)
    float o_partial = 0.0f;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        o_partial += s_reg[j] * q_reg[j];
    }
    const float o = simd_sum(o_partial);

    // (6) Write the (one) output element + state row back.
    if (tiisg == 0) {
        out[(ulong)hi * HEAD_DIM + dv] = o;
    }
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_row[dk_base + j] = s_reg[j];
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_gdn_step_decay_f32(
        constant gdn_step_args & args    [[buffer(0)]],
        device const float     * q       [[buffer(1)]], // [n_k_heads, head_dim]
        device const float     * k       [[buffer(2)]], // [n_k_heads, head_dim]
        device const float     * v       [[buffer(3)]], // [n_v_heads, head_dim]
        device const float     * decay   [[buffer(4)]], // [n_v_heads] exp(g)
        device const float     * beta    [[buffer(5)]], // [n_v_heads]
        device       float     * state   [[buffer(6)]], // [n_v_heads, head_dim, head_dim]
        device       float     * out     [[buffer(7)]], // [n_v_heads, head_dim]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv = tgpig.x;
    const uint hi = tgpig.y;
    if (hi >= args.n_v_heads || dv >= HEAD_DIM) return;

    const uint hk = hi % args.n_k_heads;
    device float * s_row = state + (ulong)hi * HEAD_DIM * HEAD_DIM
                                  + (ulong)dv * HEAD_DIM;

    device const float * q_h = q + (ulong)hk * HEAD_DIM;
    device const float * k_h = k + (ulong)hk * HEAD_DIM;
    const float v_dv = v[(ulong)hi * HEAD_DIM + dv];
    const float g_exp = decay[hi];
    const float beta_h = beta[hi];

    float s_reg[DKS_PER_LANE];
    float k_reg[DKS_PER_LANE];
    float q_reg[DKS_PER_LANE];

    const ushort dk_base = tiisg * DKS_PER_LANE;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] = s_row[dk_base + j];
        k_reg[j] = k_h[dk_base + j];
        q_reg[j] = q_h[dk_base + j];
    }

    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] *= g_exp;
    }

    float sk_partial = 0.0f;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        sk_partial += s_reg[j] * k_reg[j];
    }
    const float sk = simd_sum(sk_partial);

    const float delta = (v_dv - sk) * beta_h;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] += delta * k_reg[j];
    }

    float o_partial = 0.0f;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        o_partial += s_reg[j] * q_reg[j];
    }
    const float o = simd_sum(o_partial);

    if (tiisg == 0) {
        out[(ulong)hi * HEAD_DIM + dv] = o;
    }
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_row[dk_base + j] = s_reg[j];
    }
}

[[max_total_threads_per_threadgroup(32)]]
kernel void kernel_gdn_step_decay_packed_f32(
        constant gdn_step_packed_args & args [[buffer(0)]],
        device const float * q_pack   [[buffer(1)]], // [n_tokens, n_k_heads, head_dim]
        device const float * k_pack   [[buffer(2)]], // [n_tokens, n_k_heads, head_dim]
        device const float * v_pack   [[buffer(3)]], // [n_tokens, n_v_heads, head_dim]
        device const float * decay    [[buffer(4)]], // [n_tokens, n_v_heads]
        device const float * beta     [[buffer(5)]], // [n_tokens, n_v_heads]
        device float       * state    [[buffer(6)]], // [n_v_heads, head_dim, head_dim]
        device float       * out_pack [[buffer(7)]], // [n_tokens, n_v_heads, head_dim]
        uint2  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]]) {
    const uint dv = tgpig.x;
    const uint hi = tgpig.y;
    if (hi >= args.n_v_heads || dv >= HEAD_DIM) return;

    const uint hk = hi % args.n_k_heads;
    device float * s_row = state + (ulong)hi * HEAD_DIM * HEAD_DIM + (ulong)dv * HEAD_DIM;

    float s_reg[DKS_PER_LANE];
    const ushort dk_base = tiisg * DKS_PER_LANE;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] = s_row[dk_base + j];
    }

    const ulong qk_stride = (ulong)args.n_k_heads * HEAD_DIM;
    const ulong v_stride = (ulong)args.n_v_heads * HEAD_DIM;
    const ulong head_stride = (ulong)args.n_v_heads;
    const ulong hv_off = (ulong)hi * HEAD_DIM + dv;

    device const float * q_lane = q_pack + (ulong)hk * HEAD_DIM + dk_base;
    device const float * k_lane = k_pack + (ulong)hk * HEAD_DIM + dk_base;
    device const float * v_t = v_pack + hv_off;
    device const float * decay_t = decay + hi;
    device const float * beta_t = beta + hi;
    device float * out_t = out_pack + hv_off;

    for (uint t = 0; t < args.n_tokens; ++t) {
        const float v_dv = *v_t;
        const float g_exp = *decay_t;
        const float beta_h = *beta_t;

        float k_reg[DKS_PER_LANE];
        float q_reg[DKS_PER_LANE];
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            k_reg[j] = k_lane[j];
            q_reg[j] = q_lane[j];
            s_reg[j] *= g_exp;
        }

        float sk_partial = 0.0f;
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            sk_partial += s_reg[j] * k_reg[j];
        }
        const float sk = simd_sum(sk_partial);

        const float delta = (v_dv - sk) * beta_h;
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            s_reg[j] += delta * k_reg[j];
        }

        float o_partial = 0.0f;
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            o_partial += s_reg[j] * q_reg[j];
        }
        const float o = simd_sum(o_partial);
        if (tiisg == 0) {
            *out_t = o;
        }

        q_lane += qk_stride;
        k_lane += qk_stride;
        v_t += v_stride;
        decay_t += head_stride;
        beta_t += head_stride;
        out_t += v_stride;
    }

    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_row[dk_base + j] = s_reg[j];
    }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void kernel_gdn_step_decay_packed_nsg4_f32(
        constant gdn_step_packed_args & args [[buffer(0)]],
        device const float * q_pack   [[buffer(1)]], // [n_tokens, n_k_heads, head_dim]
        device const float * k_pack   [[buffer(2)]], // [n_tokens, n_k_heads, head_dim]
        device const float * v_pack   [[buffer(3)]], // [n_tokens, n_v_heads, head_dim]
        device const float * decay    [[buffer(4)]], // [n_tokens, n_v_heads]
        device const float * beta     [[buffer(5)]], // [n_tokens, n_v_heads]
        device float       * state    [[buffer(6)]], // [n_v_heads, head_dim, head_dim]
        device float       * out_pack [[buffer(7)]], // [n_tokens, n_v_heads, head_dim]
        uint3  tgpig [[threadgroup_position_in_grid]],
        uint3  tpitg [[thread_position_in_threadgroup]]) {
    const uint dv = tgpig.x * 4u + tpitg.y;
    const uint hi = tgpig.y;
    if (hi >= args.n_v_heads || dv >= HEAD_DIM) return;

    const ushort lane = (ushort)tpitg.x;
    const uint hk = hi % args.n_k_heads;
    device float * s_row = state + (ulong)hi * HEAD_DIM * HEAD_DIM + (ulong)dv * HEAD_DIM;

    float s_reg[DKS_PER_LANE];
    const ushort dk_base = lane * DKS_PER_LANE;
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_reg[j] = s_row[dk_base + j];
    }

    const ulong qk_stride = (ulong)args.n_k_heads * HEAD_DIM;
    const ulong v_stride = (ulong)args.n_v_heads * HEAD_DIM;
    const ulong head_stride = (ulong)args.n_v_heads;
    const ulong hv_off = (ulong)hi * HEAD_DIM + dv;

    device const float * q_lane = q_pack + (ulong)hk * HEAD_DIM + dk_base;
    device const float * k_lane = k_pack + (ulong)hk * HEAD_DIM + dk_base;
    device const float * v_t = v_pack + hv_off;
    device const float * decay_t = decay + hi;
    device const float * beta_t = beta + hi;
    device float * out_t = out_pack + hv_off;

    for (uint t = 0; t < args.n_tokens; ++t) {
        const float v_dv = *v_t;
        const float g_exp = *decay_t;
        const float beta_h = *beta_t;

        float k_reg[DKS_PER_LANE];
        float q_reg[DKS_PER_LANE];
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            k_reg[j] = k_lane[j];
            q_reg[j] = q_lane[j];
            s_reg[j] *= g_exp;
        }

        float sk_partial = 0.0f;
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            sk_partial += s_reg[j] * k_reg[j];
        }
        const float sk = simd_sum(sk_partial);

        const float delta = (v_dv - sk) * beta_h;
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            s_reg[j] += delta * k_reg[j];
        }

        float o_partial = 0.0f;
        for (ushort j = 0; j < DKS_PER_LANE; ++j) {
            o_partial += s_reg[j] * q_reg[j];
        }
        const float o = simd_sum(o_partial);
        if (lane == 0) {
            *out_t = o;
        }

        q_lane += qk_stride;
        k_lane += qk_stride;
        v_t += v_stride;
        decay_t += head_stride;
        beta_t += head_stride;
        out_t += v_stride;
    }

    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_row[dk_base + j] = s_reg[j];
    }
}
