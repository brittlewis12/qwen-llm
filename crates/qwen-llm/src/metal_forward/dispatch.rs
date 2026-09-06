//! GEMV/GEMM dispatch tables across quant formats.

use super::*;

pub fn with_matmat_bf16_bfloat_act_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    let previous = MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| {
        let previous = slot.get();
        slot.set(Some(enabled));
        previous
    });
    let out = f();
    MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.set(previous));
    out
}

pub(super) fn matmat_bf16_bfloat_act_enabled() -> bool {
    if let Some(enabled) = MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on("QWEN_MATMAT_BF16_BFLOAT_ACT"))
}

/// Dispatch the right `encode_mat_vec_*` based on `weight.dtype`. This
/// is the single seam that lets the same MetalForward driver run on
/// F32, Q4_K_M, Q6_K, etc. weights. New quant types plug in here.
pub fn encode_mat_vec_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), MfError> {
    // Debug-only concurrent-pass hazard tracking (no-op on serial encoders
    // and in release builds). This is the chokepoint for the concurrent
    // GDN/attention front-projection encoders; a future edit that makes one
    // projection consume another's output inside the same Concurrent pass
    // will panic here instead of racing on the GPU.
    enc.note_read(weight);
    enc.note_read(x);
    enc.note_write(y);
    match weight.dtype {
        GgmlType::F32 => Ok(encode_mat_vec_f32(ctx, enc, weight, x, y, n_in, n_out)?),
        GgmlType::F16 => Ok(crate::metal::encode_mat_vec_f16_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::BF16 => Ok(crate::metal::encode_mat_vec_bf16_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q2_K => Ok(crate::metal::encode_mat_vec_q2_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q3_K => Ok(crate::metal::encode_mat_vec_q3_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ2_XS => Ok(crate::metal::encode_mat_vec_iq2_xs_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ2_S => Ok(crate::metal::encode_mat_vec_iq2_s_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ3_XXS => Ok(crate::metal::encode_mat_vec_iq3_xxs_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ3_S => Ok(crate::metal::encode_mat_vec_iq3_s_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q4_0 => Ok(crate::metal::encode_mat_vec_q4_0_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q4_1 => Ok(crate::metal::encode_mat_vec_q4_1_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q4_K => Ok(encode_mat_vec_q4_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q5_K => Ok(encode_mat_vec_q5_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q6_K => Ok(encode_mat_vec_q6_k_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::MXFP4 => Ok(crate::metal::encode_mat_vec_mxfp4_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::Q8_0 => Ok(crate::metal::encode_mat_vec_q8_0_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ4_NL => Ok(crate::metal::encode_mat_vec_iq4_nl_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        GgmlType::IQ4_XS => Ok(crate::metal::encode_mat_vec_iq4_xs_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?),
        other => Err(MfError::UnsupportedDtype {
            name: "(weight at mat_vec dispatch)".to_string(),
            dtype: other,
        }),
    }
}

#[cfg(test)]
pub(crate) fn matmat_smalln_table_enabled_for_test() -> bool {
    matmat_smalln_table_enabled()
}

pub(crate) fn validate_f32_q8_mat_mat_addressing(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MfError> {
    if !matches!(dtype, GgmlType::F32 | GgmlType::Q8_0) {
        return Ok(());
    }
    for (name, value) in [("n_in", n_in), ("n_out", n_out), ("n_query", n_query)] {
        if value == 0 || u32::try_from(value).is_err() {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_dispatch",
                detail: format!("{name}={value} must fit nonzero u32 shader addressing"),
            }
            .into());
        }
    }
    for (name, elements) in [
        ("weight", n_in.checked_mul(n_out)),
        ("input", n_in.checked_mul(n_query)),
        ("output", n_out.checked_mul(n_query)),
    ] {
        let elements = elements.ok_or_else(|| MetalError::BadShape {
            kernel: "mat_mat_dispatch",
            detail: format!("{name} element count overflow"),
        })?;
        if u32::try_from(elements).is_err() {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_dispatch",
                detail: format!("{name} element count {elements} exceeds u32 shader addressing"),
            }
            .into());
        }
    }
    if dtype == GgmlType::Q8_0 {
        let row_bytes = n_in
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(34))
            .ok_or_else(|| MetalError::BadShape {
                kernel: "mat_mat_dispatch",
                detail: "Q8_0 row-byte stride overflow".into(),
            })?;
        if !n_in.is_multiple_of(32) || u32::try_from(row_bytes).is_err() {
            return Err(MetalError::BadShape {
                kernel: "mat_mat_dispatch",
                detail: format!(
                    "Q8_0 n_in={n_in} must be block-aligned with u32 row-byte stride, got {row_bytes}"
                ),
            }
            .into());
        }
    }
    Ok(())
}

/// Mat-mat dispatch routing for the H5.3b layer-major path. Picks the
/// right `kernel_mul_mm_*` lift based on weight dtype. Output is
/// row-major `[n_query, n_out]` (codex H5.3b mid-impl review verified
/// the col-major framing in the lifted llama kernels is bit-identical
/// to row-major storage at this stride).
///
/// Production 27B Q4_K_M reaches several weight dtypes via mat-mat:
///   * F32/F16/BF16 (full-precision and mixed GGUF variants)
///   * Q2_K/Q3_K (low-bit K-quant compatibility)
///   * IQ2_S/IQ3_S (Ridge low-bit FFNs)
///   * Q4_0/Q4_1 (legacy quant compatibility)
///   * Q4_K (ffn_gate, ffn_up, attn projections)
///   * Q5_K (GDN out_proj — added by v0.73a.0)
///   * Q6_K (ffn_down, lm_head)
///   * Q8_0 (DFlash drafter projections, lm_head — added by v0.73b.0)
///   * IQ4_NL/IQ4_XS (IQ quant compatibility)
pub fn encode_mat_mat_dispatch(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MfError> {
    encode_mat_mat_dispatch_with_policy(ctx, enc, weight, x, y, n_in, n_out, n_query, true)
}

/// Prompt GEMM dispatch with an explicit choice about the Qwen-tuned
/// `n_query == 1` mat-vec shortcut. Families that pin a bitwise matrix
/// lineage at N=1 (DeepSeek V4 packed prefill) pass `false` so a Qwen
/// routing decision cannot change their arithmetic.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_dispatch_with_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    allow_n1_mat_vec: bool,
) -> Result<(), MfError> {
    validate_f32_q8_mat_mat_addressing(weight.dtype, n_in, n_out, n_query)?;
    // v0.77: n_query == 1 is exactly the mat-vec contract (x = [n_in],
    // y = [n_out]) — route to the production single-token kernels (c=1).
    // Before this arm, n=1 fell through the small-N table to the GENERIC
    // 32-wide tile (c 5.1-8.3): the 2026-08-19 interleaved verify
    // microbench measured packed_verify(n_eff=1) at 202 ms vs a 39 ms
    // single_token forward — a 5.2× pure kernel-selection artifact hit by
    // every n_eff_override=1 caller (adaptive-N tails, MTP packet tails).
    // Exactness: mat-vec is E0, tighter than the tile it replaces.
    // Rollback: QWEN_MATMAT_N1_MATVEC=0.
    if allow_n1_mat_vec && matmat_n1_matvec_enabled() && n_query == 1 {
        return encode_mat_vec_dispatch(ctx, enc, weight, x, y, n_in, n_out);
    }

    // The generic Q5_K matrix kernel owns a physical 32-column tile. At N=2
    // it computes that tile to retain two columns and is substantially slower
    // than two mature Q5_K mat-vec dispatches. Keep the packed row contract,
    // but compose exact row views until a true dequant-once NC2 body exists.
    // v0.77: extended from n_query == 2 to 2..=4 — the 2026-08-19 verify
    // microbench showed Q5_K n≥3 leaking to the generic tile (c 5.1-8.0);
    // sequential mat-vec is c≈n, a clear win through n=4 (the 48 GDN
    // out_proj dispatches in packed verify are the production victim).
    // At n≥8 c≈n ≈ generic; left on the generic tile pending a re-sweep.
    // Rollback: QWEN_MATMAT_Q5_K_N2_SEQ=0.
    if matmat_q5_k_n2_seq_enabled() && weight.dtype == GgmlType::Q5_K && (2..=4).contains(&n_query)
    {
        for row in 0..n_query {
            let x_row = x.view_subrange((row * n_in) as u64, vec![n_in as u64]);
            let y_row = y.view_subrange((row * n_out) as u64, vec![n_out as u64]);
            encode_mat_vec_q5_k_f32(ctx, enc, weight, &x_row, &y_row, n_in, n_out)?;
        }
        return Ok(());
    }

    let dense_27b_ffn_shape = matches!((n_in, n_out), (5120, 17408) | (17408, 5120));
    if matmat_iq2_s_n2_nc2_enabled()
        && weight.dtype == GgmlType::IQ2_S
        && n_query == 2
        && dense_27b_ffn_shape
    {
        return Ok(crate::metal::encode_mat_vec_iq2_s_nc2_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?);
    }

    if matmat_iq3_s_n2_nc2_enabled()
        && weight.dtype == GgmlType::IQ3_S
        && n_query == 2
        && dense_27b_ffn_shape
    {
        return Ok(crate::metal::encode_mat_vec_iq3_s_nc2_f32(
            ctx, enc, weight, x, y, n_in, n_out,
        )?);
    }

    // v0.501: small-N best-kernel table from the v0.500 selection sweep
    // (PERF-LOG v0.500; the pre-v0.501 selection fell through to the
    // GENERIC 32-wide tile at every N < 16 and lost 2-5x). Only fires
    // where the sweep measured a clear win AND the buffer contract is
    // drop-in (x = [n_query, n_in], y = [n_query, n_out], exact).
    // Exactness: nc is E0 per column vs mat-vec (tighter than the E1
    // half-staged tile it replaces); mma8v is E1 with F32 activations
    // (cos 1.000000 at rel_rms ~2e-4 on all swept shapes). End-to-end
    // greedy equivalence re-gated at v0.501. Rollback:
    // QWEN_MATMAT_SMALLN_TABLE=0.
    if matmat_smalln_table_enabled()
        && matches!(
            weight.dtype,
            GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::Q5_K | GgmlType::Q8_0
        )
        && n_in.is_multiple_of(256)
    {
        match n_query {
            2 if weight.dtype == GgmlType::Q4_K => {
                // nc2rp4: c(2) 1.53-1.72 vs generic 5.1-8.0.
                return Ok(crate::metal::encode_mat_vec_q4_k_nc2_rp4_f32(
                    ctx, enc, weight, x, y, n_in, n_out,
                )?);
            }
            // nc kernels exist only for Q4_K/Q6_K (Q5_K n=2..4 already
            // routed to sequential mat-vec above; Q8_0 n=2/4 falls
            // through to the generic tile — unswept).
            2 | 4 if matches!(weight.dtype, GgmlType::Q4_K | GgmlType::Q6_K) => {
                // nc2 (Q6_K) / nc4: c 1.6-3.3 vs generic 5.1-8.3.
                return Ok(crate::metal::encode_mat_vec_nc_dispatch(
                    ctx, enc, weight, x, y, n_in, n_out, n_query,
                )?);
            }
            8 if weight.dtype == GgmlType::Q6_K && n_out.is_multiple_of(8) => {
                // r1c1k128: flat c ~1.6-1.8 across N on Q6_K shapes.
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx, enc, weight, x, y, n_in, n_out, "r1c1k128",
                )?);
            }
            // v0.77 sweep (matmat-smalln-micro, production shapes): Q5_K
            // and Q8_0 had NO tuned N=8 arm and paid the generic tile.
            // r1c1k128 won every swept shape for both dtypes:
            //   Q5_K [6144,5120]: 0.336 -> 0.105 ms/dispatch (the 48 GDN
            //     out_proj dispatches in packed verify: ~-11 ms/pass)
            //   Q8_0 drafter shapes: -49% to -75% (DFlash 2 draft_block
            //     phases 2/3: ~-8 ms/draft)
            8 if matches!(weight.dtype, GgmlType::Q5_K | GgmlType::Q8_0)
                && n_out.is_multiple_of(8) =>
            {
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx, enc, weight, x, y, n_in, n_out, "r1c1k128",
                )?);
            }
            // v0.77 sweep: Q4_K down-projections (n_in > n_out) prefer
            // r1c1k128 over the sg2 all-rounder — [6144,5120] 0.103 ->
            // 0.094, [17408,5120] 0.311 -> 0.268 ms. Up/square shapes
            // keep sg2 ([5120,6144] 0.099 vs 0.123, [5120,12288] 0.186
            // vs 0.194, [5120,17408] 0.247 vs 0.263).
            8 if weight.dtype == GgmlType::Q4_K && n_in > n_out && n_out.is_multiple_of(8) => {
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx,
                    enc,
                    weight,
                    x,
                    y,
                    n_in,
                    n_out,
                    if matmat_q4_vec4_enabled() {
                        "r1c1k128_vec4"
                    } else {
                        "r1c1k128"
                    },
                )?);
            }
            8 if weight.dtype == GgmlType::Q4_K && n_out.is_multiple_of(16) => {
                // r1c1k64_sg2: the Q4_K N=8 all-rounder (2.25-2.78,
                // never worst) vs generic 5.1-8.3.
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx,
                    enc,
                    weight,
                    x,
                    y,
                    n_in,
                    n_out,
                    if matmat_q4_vec4_enabled() {
                        if n_in.checked_mul(2) == Some(n_out) {
                            "r2c1k64_vec4"
                        } else {
                            "r1c1k64_sg2_vec4"
                        }
                    } else {
                        "r1c1k64_sg2"
                    },
                )?);
            }
            // ct=2 variants (16 columns) exist only for Q4_K/Q6_K — the
            // v1 (N=16) drafter's Q8_0 mat-mats must NOT land here.
            16 if matches!(weight.dtype, GgmlType::Q4_K | GgmlType::Q6_K)
                && n_out.is_multiple_of(16)
                && n_out < 100_000 =>
            {
                // r2c2k64 beats n16 by 8-35% on ffn/gdn/attn shapes;
                // n16 retained for lm_head-class (n_out >= 100k) where
                // it still wins (2.35 vs 2.81).
                return Ok(crate::metal::encode_mat_mat_mma8_variant(
                    ctx, enc, weight, x, y, n_in, n_out, "r2c2k64",
                )?);
            }
            _ => {}
        }
    }
    match weight.dtype {
        GgmlType::Q4_K => Ok(crate::metal::encode_mat_mat_q4_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::F32 => Ok(crate::metal::encode_mat_mat_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::F16 => Ok(crate::metal::encode_mat_mat_f16_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::BF16
            if matmat_bf16_bfloat_act_enabled() && n_in.is_multiple_of(32) && n_query >= 16 =>
        {
            Ok(crate::metal::encode_mat_mat_bf16_bfloat_act_f32(
                ctx, enc, weight, x, y, n_in, n_out, n_query,
            )?)
        }
        GgmlType::BF16 => Ok(crate::metal::encode_mat_mat_bf16_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q2_K => Ok(crate::metal::encode_mat_mat_q2_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q3_K => Ok(crate::metal::encode_mat_mat_q3_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ2_XS => Ok(crate::metal::encode_mat_mat_iq2_xs_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ2_S => Ok(crate::metal::encode_mat_mat_iq2_s_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ3_XXS => Ok(crate::metal::encode_mat_mat_iq3_xxs_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ3_S => Ok(crate::metal::encode_mat_mat_iq3_s_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q4_0 => Ok(crate::metal::encode_mat_mat_q4_0_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q4_1 => Ok(crate::metal::encode_mat_mat_q4_1_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q5_K => Ok(crate::metal::encode_mat_mat_q5_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q6_K => Ok(crate::metal::encode_mat_mat_q6_k_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::Q8_0 => Ok(crate::metal::encode_mat_mat_q8_0_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ4_NL => Ok(crate::metal::encode_mat_mat_iq4_nl_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        GgmlType::IQ4_XS => Ok(crate::metal::encode_mat_mat_iq4_xs_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?),
        other => Err(MfError::UnsupportedDtype {
            name: "(weight at mat_mat dispatch)".to_string(),
            dtype: other,
        }),
    }
}
