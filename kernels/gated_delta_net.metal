// Gated DeltaNet recurrence step (single-token decode).
//
// One step of the per-V-head delta-rule recurrence:
//
//     S ← exp(g) · S + β · (v − exp(g) · (S · k)) ⊗ k
//     o = S · q · (1/√head_dim)
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
};

kernel void kernel_gdn_step_f32(
        constant gdn_step_args & args  [[buffer(0)]],
        device const float     * q     [[buffer(1)]], // [n_v_heads, head_dim]
        device const float     * k     [[buffer(2)]], // [n_v_heads, head_dim]
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

    // Pointer to S[hi, dv, :] — one row of head_dim elements.
    device float * s_row = state + (ulong)hi * HEAD_DIM * HEAD_DIM
                                  + (ulong)dv * HEAD_DIM;

    // Per-head vectors.
    device const float * q_h = q + (ulong)hi * HEAD_DIM;
    device const float * k_h = k + (ulong)hi * HEAD_DIM;
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
        out[(ulong)hi * HEAD_DIM + dv] = o * (1.0f / sqrt((float)HEAD_DIM));
    }
    for (ushort j = 0; j < DKS_PER_LANE; ++j) {
        s_row[dk_base + j] = s_reg[j];
    }
}
