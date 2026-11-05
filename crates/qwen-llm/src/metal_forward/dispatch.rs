//! GEMV/GEMM dispatch tables across quant formats.

use super::*;

type MatVecKernel = fn(
    &MetalContext,
    &KernelEncoder,
    &MetalTensor,
    &MetalTensor,
    &MetalTensor,
    usize,
    usize,
) -> Result<(), MetalError>;

fn mat_vec_kernel(dtype: GgmlType) -> Option<MatVecKernel> {
    Some(match dtype {
        GgmlType::IQ1_S | GgmlType::IQ1_M => crate::metal::encode_mat_vec_iq1_f32,
        GgmlType::IQ2_XXS => crate::metal::encode_mat_vec_iq2_xxs_f32,
        GgmlType::F32 => encode_mat_vec_f32,
        GgmlType::F16 => crate::metal::encode_mat_vec_f16_f32,
        GgmlType::BF16 => crate::metal::encode_mat_vec_bf16_f32,
        GgmlType::Q2_K => crate::metal::encode_mat_vec_q2_k_f32,
        GgmlType::Q3_K => crate::metal::encode_mat_vec_q3_k_f32,
        GgmlType::IQ2_XS => crate::metal::encode_mat_vec_iq2_xs_f32,
        GgmlType::IQ2_S => crate::metal::encode_mat_vec_iq2_s_f32,
        GgmlType::IQ3_XXS => crate::metal::encode_mat_vec_iq3_xxs_f32,
        GgmlType::IQ3_S => crate::metal::encode_mat_vec_iq3_s_f32,
        GgmlType::Q4_0 => crate::metal::encode_mat_vec_q4_0_f32,
        GgmlType::Q4_1 => crate::metal::encode_mat_vec_q4_1_f32,
        GgmlType::Q4_K => encode_mat_vec_q4_k_f32,
        GgmlType::Q5_K => encode_mat_vec_q5_k_f32,
        GgmlType::Q6_K => encode_mat_vec_q6_k_f32,
        GgmlType::MXFP4 => crate::metal::encode_mat_vec_mxfp4_f32,
        GgmlType::Q8_0 => crate::metal::encode_mat_vec_q8_0_f32,
        GgmlType::IQ4_NL => crate::metal::encode_mat_vec_iq4_nl_f32,
        GgmlType::IQ4_XS => crate::metal::encode_mat_vec_iq4_xs_f32,
        _ => return None,
    })
}

/// Whether generic GEMV dispatch has a native weight path.
/// Geometry and buffer validation remain the encoder's responsibility.
pub fn mat_vec_dispatch_supported(dtype: GgmlType) -> bool {
    mat_vec_kernel(dtype).is_some()
}

type MatMatKernel = fn(
    &MetalContext,
    &KernelEncoder,
    &MetalTensor,
    &MetalTensor,
    &MetalTensor,
    usize,
    usize,
    usize,
) -> Result<(), MetalError>;

fn mat_mat_kernel(dtype: GgmlType) -> Option<MatMatKernel> {
    Some(match dtype {
        GgmlType::IQ1_S | GgmlType::IQ1_M => crate::metal::encode_mat_mat_iq1_f32,
        GgmlType::IQ2_XXS => crate::metal::encode_mat_mat_iq2_xxs_f32,
        GgmlType::Q4_K => crate::metal::encode_mat_mat_q4_k_f32,
        GgmlType::F32 => crate::metal::encode_mat_mat_f32,
        GgmlType::F16 => crate::metal::encode_mat_mat_f16_f32,
        GgmlType::BF16 => crate::metal::encode_mat_mat_bf16_f32,
        GgmlType::Q2_K => crate::metal::encode_mat_mat_q2_k_f32,
        GgmlType::Q3_K => crate::metal::encode_mat_mat_q3_k_f32,
        GgmlType::IQ2_XS => crate::metal::encode_mat_mat_iq2_xs_f32,
        GgmlType::IQ2_S => crate::metal::encode_mat_mat_iq2_s_f32,
        GgmlType::IQ3_XXS => crate::metal::encode_mat_mat_iq3_xxs_f32,
        GgmlType::IQ3_S => crate::metal::encode_mat_mat_iq3_s_f32,
        GgmlType::Q4_0 => crate::metal::encode_mat_mat_q4_0_f32,
        GgmlType::Q4_1 => crate::metal::encode_mat_mat_q4_1_f32,
        GgmlType::Q5_K => crate::metal::encode_mat_mat_q5_k_f32,
        GgmlType::Q6_K => crate::metal::encode_mat_mat_q6_k_f32,
        GgmlType::Q8_0 => crate::metal::encode_mat_mat_q8_0_f32,
        GgmlType::IQ4_NL => crate::metal::encode_mat_mat_iq4_nl_f32,
        GgmlType::IQ4_XS => crate::metal::encode_mat_mat_iq4_xs_f32,
        _ => return None,
    })
}

/// Whether generic multi-row GEMM dispatch has a native weight path.
/// Geometry and buffer validation remain the encoder's responsibility.
pub fn mat_mat_dispatch_supported(dtype: GgmlType) -> bool {
    mat_mat_kernel(dtype).is_some()
}

pub fn with_matmat_bf16_bfloat_act_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }
    let _restore =
        Restore(MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.replace(Some(enabled))));
    f()
}

#[test]
fn bf16_bfloat_act_override_restores_nested_unwind_and_thread() {
    let initial = MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.get());
    with_matmat_bf16_bfloat_act_override(true, || {
        assert!(matmat_bf16_bfloat_act_enabled());
        with_matmat_bf16_bfloat_act_override(false, || {
            assert!(!matmat_bf16_bfloat_act_enabled());
        });
        assert!(matmat_bf16_bfloat_act_enabled());
        assert!(
            std::panic::catch_unwind(|| {
                with_matmat_bf16_bfloat_act_override(false, || panic!("override unwind probe"))
            })
            .is_err()
        );
        assert!(matmat_bf16_bfloat_act_enabled());
        assert!(
            std::thread::spawn(|| MATMAT_BF16_BFLOAT_ACT_OVERRIDE
                .with(|slot| slot.get())
                .is_none())
            .join()
            .unwrap()
        );
    });
    assert_eq!(
        MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.get()),
        initial
    );
    assert!(
        std::panic::catch_unwind(|| {
            with_matmat_bf16_bfloat_act_override(false, || panic!("outer unwind probe"))
        })
        .is_err()
    );
    assert_eq!(
        MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.get()),
        initial
    );
}

pub(super) fn matmat_bf16_bfloat_act_enabled() -> bool {
    if let Some(enabled) = MATMAT_BF16_BFLOAT_ACT_OVERRIDE.with(|slot| slot.get()) {
        return enabled;
    }
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| crate::env_flag::read_default_on("QWEN_MATMAT_BF16_BFLOAT_ACT"))
}

/// IQ1/IQ2_XXS primitives deliberately require exact vector/matrix ranks. Packed
/// callers may supply flat scratch views (including unused trailing capacity).
/// Check the source's physical extent and logical prefix before view_subrange,
/// which asserts rather than returning an error. No activation/weight copy.
pub(super) fn native_iq_dispatch_views(
    x: &MetalTensor,
    y: &MetalTensor,
    k: usize,
    m: usize,
    n: usize,
    matrix: bool,
) -> Result<(MetalTensor, MetalTensor), MfError> {
    let bad = |detail: String| MetalError::BadShape {
        kernel: "native_iq_dispatch",
        detail,
    };
    for (name, dim) in [("K", k), ("M", m), ("N", n)] {
        if dim == 0 || i32::try_from(dim).is_err() {
            return Err(bad(format!("{name}={dim} must fit positive i32")).into());
        }
    }
    if !k.is_multiple_of(256) || (!matrix && n != 1) {
        return Err(bad("K must be block-aligned and GEMV requires N=1".into()).into());
    }
    let view = |name: &str, src: &MetalTensor, width: usize, writable: bool| {
        let needed = width
            .checked_mul(n)
            .ok_or_else(|| bad(format!("{name} prefix overflow")))?;
        if src.dtype != GgmlType::F32
            || !src.offset.is_multiple_of(4)
            || (writable && !src.is_writable())
        {
            return Err(bad(format!(
                "{name} must be aligned F32{}",
                if writable { " and writable" } else { "" }
            )));
        }
        let (elements, bytes) = crate::metal::checked_ggml_shape_bytes(&src.shape, src.dtype)?;
        if elements < needed
            || src
                .offset
                .checked_add(bytes as u64)
                .is_none_or(|end| end > src.buffer.length() as u64)
        {
            return Err(bad(format!(
                "{name} source does not contain a physical {needed}-element prefix"
            )));
        }
        Ok(src.view_subrange(
            0,
            if matrix {
                vec![width as u64, n as u64]
            } else {
                vec![width as u64]
            },
        ))
    };
    Ok((view("input", x, k, false)?, view("output", y, m, true)?))
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
    if matches!(
        weight.dtype,
        GgmlType::IQ2_XXS | GgmlType::IQ1_S | GgmlType::IQ1_M
    ) {
        // The native IQ primitive records the normalized ranges itself. Do not also
        // run generic notes: duplicate writes would trigger concurrent hazards.
        let (x, y) = native_iq_dispatch_views(x, y, n_in, n_out, 1, false)?;
        let kernel = mat_vec_kernel(weight.dtype).expect("native IQ GEMV selector");
        return Ok(kernel(ctx, enc, weight, &x, &y, n_in, n_out)?);
    }
    // Debug-only concurrent-pass hazard tracking (no-op on serial encoders
    // and in release builds). This is the chokepoint for the concurrent
    // GDN/attention front-projection encoders; a future edit that makes one
    // projection consume another's output inside the same Concurrent pass
    // will panic here instead of racing on the GPU.
    enc.note_read(weight);
    enc.note_read(x);
    enc.note_write(y);
    let kernel = mat_vec_kernel(weight.dtype).ok_or_else(|| MfError::UnsupportedDtype {
        name: "(weight at mat_vec dispatch)".to_string(),
        dtype: weight.dtype,
    })?;
    Ok(kernel(ctx, enc, weight, x, y, n_in, n_out)?)
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
    encode_mat_mat_dispatch_routed(
        ctx, enc, weight, x, y, n_in, n_out, n_query, true, true, false,
    )
}

/// Qwen DFlash verify projections: the only callers that may take the
/// generic few-row arm. Every other dispatch entry point keeps
/// `allow_fewrow_gen = false`, so enabling `QWEN_MATMAT_FEWROW_GEN` cannot
/// change Flash-Next, K2, GLM, DeepSeek, drafter or ordinary Qwen paths.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_dispatch_dflash_verify(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MfError> {
    encode_mat_mat_dispatch_routed(
        ctx, enc, weight, x, y, n_in, n_out, n_query, true, true, true,
    )
}

/// [`encode_mat_mat_dispatch`] without the Q8_0 few-row arm (2..=8 rows):
/// exactly the routes before `c640a113`. For families whose small-N
/// routes have not been qualified on that arm (Qwen3.8-Flash-Next, K2
/// Horizon); a family moves to [`encode_mat_mat_dispatch`] only after its
/// route, numerical, continuation and performance checks pass (lane audit
/// 2026-10-06, B1.5).
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_dispatch_without_fewrow(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
) -> Result<(), MfError> {
    encode_mat_mat_dispatch_routed(
        ctx, enc, weight, x, y, n_in, n_out, n_query, true, false, false,
    )
}

/// Prompt GEMM dispatch with an explicit choice about the Qwen-tuned
/// `n_query == 1` mat-vec shortcut. Families that pin a bitwise matrix
/// lineage at N=1 (DeepSeek V4 packed prefill) pass `false` so a Qwen
/// routing decision cannot change their arithmetic. Explicit-policy callers
/// (DeepSeek V4, GLM5-Next) also never take the Q8_0 few-row arm; it reaches
/// only callers of [`encode_mat_mat_dispatch`].
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
    encode_mat_mat_dispatch_routed(
        ctx,
        enc,
        weight,
        x,
        y,
        n_in,
        n_out,
        n_query,
        allow_n1_mat_vec,
        false,
        false,
    )
}

fn dense_27b_ffn_shape(n_in: usize, n_out: usize) -> bool {
    matches!((n_in, n_out), (5120, 17408) | (17408, 5120))
}

/// Whether the caller may take the generic few-row arm, after the routes
/// that keep precedence. `allow` is the caller policy: only Qwen DFlash
/// verify passes `true`; prompt lookup and packed MTP pass `false` even
/// when the knob is on. `enabled` is the explicit `QWEN_MATMAT_FEWROW_GEN`
/// opt-in (there is no automatic default). Q5_K at 2..=4 rows keeps its
/// sequential mat-vec route, and dense 27B IQ2_S/IQ3_S at 2 rows keep the
/// nc2 routes. Q8_0 is limited to 9..=16 rows so its tuned 1..=8 arm is
/// unchanged.
pub(crate) fn fewrow_gen_route_selected_impl(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    allow: bool,
    enabled: bool,
    q5_k_seq: bool,
    iq2_s_nc2: bool,
    iq3_s_nc2: bool,
) -> bool {
    if !allow || !enabled || !(2..=16).contains(&n_query) {
        return false;
    }
    if q5_k_seq && dtype == GgmlType::Q5_K && (2..=4).contains(&n_query) {
        return false;
    }
    let dense_ffn = dense_27b_ffn_shape(n_in, n_out);
    if iq2_s_nc2 && dtype == GgmlType::IQ2_S && n_query == 2 && dense_ffn {
        return false;
    }
    if iq3_s_nc2 && dtype == GgmlType::IQ3_S && n_query == 2 && dense_ffn {
        return false;
    }
    match dtype {
        GgmlType::Q4_K
        | GgmlType::Q5_K
        | GgmlType::Q6_K
        | GgmlType::IQ2_XXS
        | GgmlType::IQ2_XS
        | GgmlType::IQ2_S
        | GgmlType::IQ3_XXS
        | GgmlType::IQ3_S
        | GgmlType::IQ4_XS => n_in.is_multiple_of(256),
        GgmlType::Q8_0 => n_query > 8 && n_in.is_multiple_of(64),
        GgmlType::IQ4_NL => n_in.is_multiple_of(64),
        _ => false,
    }
}

fn fewrow_gen_route_selected(
    dtype: GgmlType,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    allow: bool,
) -> bool {
    fewrow_gen_route_selected_impl(
        dtype,
        n_in,
        n_out,
        n_query,
        allow,
        matmat_fewrow_gen_enabled(),
        matmat_q5_k_n2_seq_enabled(),
        matmat_iq2_s_n2_nc2_enabled(),
        matmat_iq3_s_n2_nc2_enabled(),
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_mat_mat_dispatch_routed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    n_query: usize,
    allow_n1_mat_vec: bool,
    allow_q8_fewrow: bool,
    allow_fewrow_gen: bool,
) -> Result<(), MfError> {
    validate_f32_q8_mat_mat_addressing(weight.dtype, n_in, n_out, n_query)?;
    if weight.dtype == GgmlType::IQ2_XS {
        crate::metal::validate_iq2_xs_mat_mat_addressing(n_in, n_out, n_query)?;
    }
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

    // Generic few-row MMA (llama.cpp `kernel_mul_mv_mma_gen`): one weight
    // read per 8-column tile, K split across simdgroups. The caller policy
    // and `fewrow_gen_route_selected` gate it; only the Qwen DFlash verify
    // entry point passes `allow_fewrow_gen = true`. If the selected
    // pipeline cannot run on this device the arm declines here and the
    // incumbent routes below stay in charge. Rollback:
    // QWEN_MATMAT_FEWROW_GEN=0 (default).
    if fewrow_gen_route_selected(weight.dtype, n_in, n_out, n_query, allow_fewrow_gen) {
        match crate::metal::try_encode_mat_mat_fewrow_gen_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        ) {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) => return Err(error.into()),
        }
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
            // Few-row Q8_0 MMA (adapted from llama.cpp PR #29869): K split
            // across simdgroups, register-filled fragments, any n_out, and
            // 2..=8 rows, so N < 8 no longer falls to the generic tile.
            // Production Q8_0 verify shapes run 1.34-1.50x faster than
            // r1c1k128 (420-472 GB/s); the top1-top2 margin error against a
            // serial reference fell from 0.0012 to 0.0003 median.
            // Rollback: QWEN_MATMAT_Q8_FEWROW=0.
            2..=8
                if allow_q8_fewrow
                    && weight.dtype == GgmlType::Q8_0
                    && matmat_q8_fewrow_enabled() =>
            {
                return Ok(crate::metal::encode_mat_mat_q8_0_fewrow_f32(
                    ctx, enc, weight, x, y, n_in, n_out, n_query,
                )?);
            }
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
    if weight.dtype == GgmlType::BF16
        && matmat_bf16_bfloat_act_enabled()
        && n_in.is_multiple_of(32)
        && n_query >= 16
    {
        return Ok(crate::metal::encode_mat_mat_bf16_bfloat_act_f32(
            ctx, enc, weight, x, y, n_in, n_out, n_query,
        )?);
    }
    let kernel = mat_mat_kernel(weight.dtype).ok_or_else(|| MfError::UnsupportedDtype {
        name: "(weight at mat_mat dispatch)".to_string(),
        dtype: weight.dtype,
    })?;
    if matches!(
        weight.dtype,
        GgmlType::IQ1_S | GgmlType::IQ1_M | GgmlType::IQ2_XXS
    ) {
        let (x, y) = native_iq_dispatch_views(x, y, n_in, n_out, n_query, true)?;
        return Ok(kernel(ctx, enc, weight, &x, &y, n_in, n_out, n_query)?);
    }
    Ok(kernel(ctx, enc, weight, x, y, n_in, n_out, n_query)?)
}

#[test]
fn dense_capabilities_keep_partial_and_unimplemented_codecs_honest() {
    use GgmlType::*;
    for dtype in [
        F32, F16, BF16, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS, Q2_K,
        Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, Q8_0,
    ] {
        assert!(mat_vec_dispatch_supported(dtype), "{dtype:?}");
        assert!(mat_mat_dispatch_supported(dtype), "{dtype:?}");
    }
    assert!(mat_vec_dispatch_supported(MXFP4));
    assert!(!mat_mat_dispatch_supported(MXFP4));
    for dtype in [
        NVFP4, TQ1_0, TQ2_0, Q1_0, Q2_0, Q5_0, Q5_1, Q8_1, Q8_K, I32, Unknown,
    ] {
        assert!(!mat_vec_dispatch_supported(dtype), "{dtype:?}");
        assert!(!mat_mat_dispatch_supported(dtype), "{dtype:?}");
    }
}

#[test]
fn fewrow_gen_route_selection_respects_policy_and_precedence() {
    use GgmlType::*;
    let sel = |dtype, n_in, n_out, n_query, allow, enabled, q5, iq2, iq3| {
        fewrow_gen_route_selected_impl(dtype, n_in, n_out, n_query, allow, enabled, q5, iq2, iq3)
    };

    // The opt-in is chip-independent now that there is no automatic
    // default; the loop documents the Apple10/other result equivalence.
    for _chip in ["apple10", "other"] {
        // DFlash verify caller: unset/off is false, =1 is true.
        assert!(!sel(Q4_K, 5120, 17408, 8, true, false, true, true, true));
        assert!(sel(Q4_K, 5120, 17408, 8, true, true, true, true, true));
        assert!(sel(Q6_K, 17408, 5120, 8, true, true, true, true, true));
        // Prompt lookup and packed MTP callers: incumbent in every mode.
        assert!(!sel(Q4_K, 5120, 17408, 8, false, true, true, true, true));
        assert!(!sel(Q4_K, 5120, 17408, 8, false, false, true, true, true));
        assert!(!sel(Q6_K, 17408, 5120, 8, false, true, true, true, true));
    }

    // Explicit opt-in widens the chip and model cells for DFlash verify.
    assert!(sel(Q4_K, 2560, 12288, 8, true, true, true, true, true));
    assert!(sel(IQ3_S, 5120, 17408, 8, true, true, true, true, true));
    assert!(sel(Q8_0, 5120, 1024, 9, true, true, true, true, true));

    // Existing tuned routes keep precedence under the opt-in.
    assert!(!sel(Q5_K, 6144, 5120, 2, true, true, true, true, true));
    assert!(!sel(Q5_K, 6144, 5120, 4, true, true, true, true, true));
    assert!(sel(Q5_K, 6144, 5120, 5, true, true, true, true, true));
    assert!(!sel(IQ2_S, 5120, 17408, 2, true, true, true, true, true));
    assert!(!sel(IQ3_S, 17408, 5120, 2, true, true, true, true, true));
    assert!(sel(IQ2_S, 5120, 17408, 2, true, true, true, false, false));
    assert!(sel(IQ3_S, 17408, 5120, 2, true, true, true, false, false));

    // Q8_0 1..=8 remains the existing few-row arm; generic is 9..=16 only.
    assert!(!sel(Q8_0, 5120, 1024, 8, true, true, true, true, true));
    assert!(sel(Q8_0, 5120, 1024, 9, true, true, true, true, true));
    assert!(sel(Q8_0, 5120, 1024, 16, true, true, true, true, true));

    // Block alignment and row-count bounds.
    assert!(!sel(Q4_K, 640, 17408, 8, true, true, true, true, true));
    assert!(sel(IQ4_NL, 512, 1024, 8, true, true, true, true, true));
    assert!(sel(IQ4_NL, 64, 1024, 8, true, true, true, true, true));
    assert!(!sel(Q4_K, 5120, 17408, 1, true, true, true, true, true));
    assert!(!sel(Q4_K, 5120, 17408, 17, true, true, true, true, true));
}

#[test]
fn fewrow_gen_caller_matrix_respects_knob_and_chip() {
    use crate::metal_dflash::PackedVerifyCaller::{DFlash, Mtp, PromptLookup};
    use GgmlType::Q4_K;

    // Each caller x {unset/off, =1} x Apple10/other. The opt-in is
    // chip-independent by design; the loop records that equivalence.
    for caller in [DFlash, PromptLookup, Mtp] {
        let allow = caller == DFlash;
        for enabled in [false, true] {
            for _chip in ["apple10", "other"] {
                let selected = fewrow_gen_route_selected_impl(
                    Q4_K, 5120, 17408, 8, allow, enabled, true, true, true,
                );
                assert_eq!(selected, allow && enabled, "caller={caller:?}");
            }
        }
    }
}
