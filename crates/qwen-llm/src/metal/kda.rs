//! Kimi delta attention (KDA) single-token decode, GLM-5.3-Flash's linear
//! attention: fused conv + SiLU, q/k L2 norm, per-channel decay along the key
//! axis, delta-rule update, readout, and sigmoid-gated per-head RMSNorm.
//!
//! CPU contract: [`crate::glm5_next::oracle::kda_decode_step`]. Kernel:
//! `kernel_glm53_kda_decode` (adapted from DwarfStar).

use super::checks::{check_alignment, check_disjoint, check_tensor, require_serial, to_u32};
use super::*;

pub const KDA_HEAD_DIM: usize = 128;
const THREADS: usize = 128;
/// q, k, decay, v, o rows plus three 4-entry reductions and beta.
const SCRATCH_FLOATS: usize = 5 * KDA_HEAD_DIM + 12 + 4;

/// Buffers for one token of one KDA layer with `heads` heads (`width = heads *
/// 128`). Activations, gates and `out` are F32 `[width]`; `raw_beta` and
/// `neg_exp_a_log` (GGUF `ssm_a = -exp(A_log)`) are `[heads]`; conv taps are
/// GGUF `[4, 1, width]`; `output_norm` is `[128]`. `conv_state` is `[width, 3,
/// 3]` (memory `[q|k|v][history][width]`, oldest first) and `state` is
/// `[128, 128, heads]` (memory `[head][value][key]`); both update in place.
pub struct KdaDecode<'a> {
    pub q: &'a MetalTensor,
    pub k: &'a MetalTensor,
    pub v: &'a MetalTensor,
    pub raw_gate: &'a MetalTensor,
    pub raw_beta: &'a MetalTensor,
    pub output_gate: &'a MetalTensor,
    pub q_conv: &'a MetalTensor,
    pub k_conv: &'a MetalTensor,
    pub v_conv: &'a MetalTensor,
    pub neg_exp_a_log: &'a MetalTensor,
    pub dt_bias: &'a MetalTensor,
    pub output_norm: &'a MetalTensor,
    pub conv_state: &'a MetalTensor,
    pub state: &'a MetalTensor,
    pub out: &'a MetalTensor,
}

pub fn encode_kda_decode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    b: &KdaDecode<'_>,
    lower_bound: f32,
    norm_eps: f32,
) -> Result<(), MetalError> {
    const K: &str = "kda_decode";
    require_serial(K, enc)?;
    if heads == 0 {
        return Err(checks::bad_shape(K, "head count must be positive"));
    }
    if !lower_bound.is_finite() || lower_bound >= 0.0 || !norm_eps.is_finite() || norm_eps <= 0.0 {
        return Err(checks::bad_shape(
            K,
            "lower bound must be negative and epsilon positive",
        ));
    }
    let width = (heads * KDA_HEAD_DIM) as u64;
    let f32_ = GgmlType::F32;
    for (tensor, name) in [
        (b.q, "q"),
        (b.k, "k"),
        (b.v, "v"),
        (b.raw_gate, "raw gate"),
        (b.output_gate, "output gate"),
        (b.dt_bias, "dt bias"),
    ] {
        check_tensor(K, tensor, f32_, &[width], false, name)?;
    }
    check_tensor(K, b.raw_beta, f32_, &[heads as u64], false, "raw beta")?;
    check_tensor(K, b.neg_exp_a_log, f32_, &[heads as u64], false, "ssm_a")?;
    for (tensor, name) in [
        (b.q_conv, "q conv"),
        (b.k_conv, "k conv"),
        (b.v_conv, "v conv"),
    ] {
        check_tensor(K, tensor, f32_, &[4, 1, width], false, name)?;
    }
    check_tensor(
        K,
        b.output_norm,
        f32_,
        &[KDA_HEAD_DIM as u64],
        false,
        "output norm",
    )?;
    check_tensor(K, b.conv_state, f32_, &[width, 3, 3], true, "conv state")?;
    let d = KDA_HEAD_DIM as u64;
    check_tensor(K, b.state, f32_, &[d, d, heads as u64], true, "state")?;
    // The recurrence reads and writes state rows as float4.
    check_alignment(K, b.state, 16, "state")?;
    check_tensor(K, b.out, f32_, &[width], true, "output")?;
    let inputs = [
        (b.q, "q"),
        (b.k, "k"),
        (b.v, "v"),
        (b.raw_gate, "raw gate"),
        (b.raw_beta, "raw beta"),
        (b.output_gate, "output gate"),
        (b.q_conv, "q conv"),
        (b.k_conv, "k conv"),
        (b.v_conv, "v conv"),
        (b.neg_exp_a_log, "ssm_a"),
        (b.dt_bias, "dt bias"),
        (b.output_norm, "output norm"),
    ];
    for (written, name) in [
        (b.conv_state, "conv state"),
        (b.state, "state"),
        (b.out, "output"),
    ] {
        check_disjoint(K, written, &inputs)
            .map_err(|e| checks::bad_shape(K, format!("{name}: {e}")))?;
    }
    check_disjoint(
        K,
        b.out,
        &[(b.conv_state, "conv state"), (b.state, "state")],
    )?;
    check_disjoint(K, b.state, &[(b.conv_state, "conv state")])?;
    let pso = ctx.pipeline("kernel_glm53_kda_decode")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < THREADS {
        return Err(checks::bad_shape(
            K,
            "needs 32-lane simdgroups and 128 threads",
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        n_rows: u32,
        lower_bound: f32,
        norm_eps: f32,
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_heads: to_u32(K, heads, "heads")?,
            n_rows: 1,
            lower_bound,
            norm_eps,
        },
    );
    for (index, tensor) in [
        b.q,
        b.k,
        b.v,
        b.raw_gate,
        b.raw_beta,
        b.output_gate,
        b.q_conv,
        b.k_conv,
        b.v_conv,
        b.neg_exp_a_log,
        b.dt_bias,
        b.output_norm,
        b.conv_state,
        b.state,
        b.out,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(index + 1, tensor);
    }
    enc.set_threadgroup_memory(0, SCRATCH_FLOATS * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: 1,
            height: heads,
            depth: 1,
        },
        MTLSize {
            width: THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
thread_local! {
    static KDA_PREPARE_REGISTER_HISTORY: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Thread-local, nestable and unwind-safe history-storage experiment. False
/// selects the actual production prepare; true retains nine history floats
/// per channel in registers. Decode and recurrence/output are unchanged.
#[cfg(test)]
pub fn with_kda_prepare_register_history_override<R>(enabled: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            KDA_PREPARE_REGISTER_HISTORY.with(|v| v.set(self.0));
        }
    }
    let _restore = Restore(KDA_PREPARE_REGISTER_HISTORY.with(|v| v.replace(Some(enabled))));
    f()
}

fn kda_prepare_pipeline_name() -> &'static str {
    #[cfg(test)]
    if let Some(enabled) = KDA_PREPARE_REGISTER_HISTORY.with(std::cell::Cell::get) {
        return if enabled {
            "kernel_glm53_kda_prefill_prepare_register_history"
        } else {
            "kernel_glm53_kda_prefill_prepare"
        };
    }
    // Keep the forced baseline above if a future result justifies changing
    // this production default; both sides of the diagnostic remain usable.
    "kernel_glm53_kda_prefill_prepare"
}

/// Encode only packed prefill's prepare stage for diagnostics, honoring
/// [`with_kda_prepare_register_history_override`]. Uses the existing full
/// [`KdaDecode`] packed bindings and validation: even unused recurrence/output
/// bindings must be valid. Mutates only q/k/v, raw_gate and conv_state; no
/// scratch allocations, recurrence or output dispatches.
#[cfg(test)]
pub fn encode_kda_prefill_prepare(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    rows: usize,
    b: &KdaDecode<'_>,
    lower_bound: f32,
    norm_eps: f32,
) -> Result<(), MetalError> {
    encode_kda_prefill_impl(ctx, enc, heads, rows, b, lower_bound, norm_eps, true)
}

/// Packed prefill of `rows` consecutive tokens. Same bindings as
/// [`encode_kda_decode`] with per-row activations: `q`, `k`, `v`, `raw_gate`,
/// `output_gate` and `out` are `[width, rows]` and `raw_beta` `[heads, rows]`.
/// Stage 1 rewrites `q`, `k`, `v` and `raw_gate` in place (normalized
/// activations and decay), so those must be writable scratch. Equivalent to
/// `rows` decode steps: state and conv tails continue across calls and
/// interleave with decode.
pub fn encode_kda_prefill(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    rows: usize,
    b: &KdaDecode<'_>,
    lower_bound: f32,
    norm_eps: f32,
) -> Result<(), MetalError> {
    encode_kda_prefill_impl(ctx, enc, heads, rows, b, lower_bound, norm_eps, false)
}

#[allow(clippy::too_many_arguments)]
fn encode_kda_prefill_impl(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    rows: usize,
    b: &KdaDecode<'_>,
    lower_bound: f32,
    norm_eps: f32,
    prepare_only: bool,
) -> Result<(), MetalError> {
    const K: &str = "kda_prefill";
    require_serial(K, enc)?;
    if heads == 0 || rows == 0 {
        return Err(checks::bad_shape(K, "heads and rows must be positive"));
    }
    if !lower_bound.is_finite() || lower_bound >= 0.0 || !norm_eps.is_finite() || norm_eps <= 0.0 {
        return Err(checks::bad_shape(
            K,
            "lower bound must be negative and epsilon positive",
        ));
    }
    let width = (heads * KDA_HEAD_DIM) as u64;
    let (r, h, d) = (rows as u64, heads as u64, KDA_HEAD_DIM as u64);
    let f32_ = GgmlType::F32;
    for (tensor, writable, name) in [
        (b.q, true, "q"),
        (b.k, true, "k"),
        (b.v, true, "v"),
        (b.raw_gate, true, "raw gate"),
        (b.output_gate, false, "output gate"),
        (b.out, true, "output"),
    ] {
        check_tensor(K, tensor, f32_, &[width, r], writable, name)?;
        check_alignment(K, tensor, 16, name)?;
    }
    check_tensor(K, b.raw_beta, f32_, &[h, r], false, "raw beta")?;
    check_tensor(K, b.neg_exp_a_log, f32_, &[h], false, "ssm_a")?;
    check_tensor(K, b.dt_bias, f32_, &[width], false, "dt bias")?;
    for (tensor, name) in [
        (b.q_conv, "q conv"),
        (b.k_conv, "k conv"),
        (b.v_conv, "v conv"),
    ] {
        check_tensor(K, tensor, f32_, &[4, 1, width], false, name)?;
    }
    check_tensor(K, b.output_norm, f32_, &[d], false, "output norm")?;
    check_tensor(K, b.conv_state, f32_, &[width, 3, 3], true, "conv state")?;
    check_tensor(K, b.state, f32_, &[d, d, h], true, "state")?;
    check_alignment(K, b.state, 16, "state")?;
    let all = [
        (b.q, "q"),
        (b.k, "k"),
        (b.v, "v"),
        (b.raw_gate, "raw gate"),
        (b.raw_beta, "raw beta"),
        (b.output_gate, "output gate"),
        (b.q_conv, "q conv"),
        (b.k_conv, "k conv"),
        (b.v_conv, "v conv"),
        (b.neg_exp_a_log, "ssm_a"),
        (b.dt_bias, "dt bias"),
        (b.output_norm, "output norm"),
        (b.conv_state, "conv state"),
        (b.state, "state"),
        (b.out, "output"),
    ];
    for (i, (written, name)) in all.iter().enumerate() {
        if !matches!(
            *name,
            "q" | "k" | "v" | "raw gate" | "conv state" | "state" | "output"
        ) {
            continue;
        }
        for (j, (other, other_name)) in all.iter().enumerate() {
            if i != j && checks::overlaps(written, other) {
                return Err(checks::bad_shape(K, format!("{name} aliases {other_name}")));
            }
        }
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        n_rows: u32,
        lower_bound: f32,
        norm_eps: f32,
    }
    let args = Args {
        n_heads: to_u32(K, heads, "heads")?,
        n_rows: to_u32(K, rows, "rows")?,
        lower_bound,
        norm_eps,
    };
    let threads = MTLSize {
        width: THREADS,
        height: 1,
        depth: 1,
    };
    let pipeline = |name: &str| -> Result<Pipeline, MetalError> {
        let pso = ctx.pipeline(name)?;
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < THREADS {
            return Err(checks::bad_shape(
                K,
                "needs 32-lane simdgroups and 128 threads",
            ));
        }
        Ok(pso)
    };

    let prepare = pipeline(kda_prepare_pipeline_name())?;
    enc.set_pipeline(&prepare);
    enc.set_bytes(0, &args);
    for (index, tensor) in [
        b.q,
        b.k,
        b.v,
        b.raw_gate,
        b.q_conv,
        b.k_conv,
        b.v_conv,
        b.neg_exp_a_log,
        b.dt_bias,
        b.conv_state,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(index + 1, tensor);
    }
    enc.set_threadgroup_memory(0, (2 * KDA_HEAD_DIM + 8) * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: heads,
            height: 1,
            depth: 1,
        },
        threads,
    );

    if prepare_only {
        return Ok(());
    }

    let recurrence = pipeline("kernel_glm53_kda_prefill_recurrence")?;
    enc.set_pipeline(&recurrence);
    enc.set_bytes(0, &args);
    for (index, tensor) in [b.q, b.k, b.v, b.raw_gate, b.raw_beta, b.state, b.out]
        .into_iter()
        .enumerate()
    {
        enc.set_tensor(index + 1, tensor);
    }
    enc.dispatch(
        MTLSize {
            width: heads,
            height: KDA_HEAD_DIM / 4,
            depth: 1,
        },
        threads,
    );

    let output = pipeline("kernel_glm53_kda_prefill_output")?;
    enc.set_pipeline(&output);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, b.out);
    enc.set_tensor(2, b.output_gate);
    enc.set_tensor(3, b.output_norm);
    enc.set_threadgroup_memory(0, 4 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: rows,
            height: heads,
            depth: 1,
        },
        threads,
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{offset_tensor, tensor_f32_at_offset};
    use super::*;
    use crate::glm5_next::oracle::{KdaStepInput, KdaWeights, kda_decode_step};

    fn val(i: usize, salt: usize, scale: f64) -> f32 {
        ((((i * 7919 + salt * 104_729) % 2003) as f64 / 2003.0 - 0.5) * scale) as f32
    }

    fn series(n: usize, salt: usize, scale: f64) -> Vec<f32> {
        (0..n).map(|i| val(i, salt, scale)).collect()
    }

    fn tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(
            ctx,
            16,
            bytemuck::cast_slice(values),
            20,
            shape,
            GgmlType::F32,
        )
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], relative: f32) {
        let scale = expected
            .iter()
            .fold(0.0f32, |m, v| m.max(v.abs()))
            .max(1e-6);
        let worst = super::super::test_support::max_abs_diff_finite(label, actual, expected);
        assert!(
            worst / scale <= relative,
            "{label}: max abs {worst} vs scale {scale}"
        );
    }

    /// 64 heads (the release width), five tokens, nonzero nonsymmetric state
    /// and nonuniform per-channel decay: output, next state and conv state
    /// against the CPU contract after every token.
    #[test]
    fn kda_decode_matches_cpu_contract_over_steps() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const HEADS: usize = 64;
        const STEPS: usize = 5;
        let width = HEADS * KDA_HEAD_DIM;
        let q_conv = series(width * 4, 7, 1.0);
        let k_conv = series(width * 4, 8, 1.0);
        let v_conv = series(width * 4, 9, 1.0);
        let neg_exp_a_log: Vec<f32> = (0..HEADS).map(|h| -(0.75 + val(h, 10, 1.0))).collect();
        let dt_bias = series(width, 11, 2.0);
        let output_norm: Vec<f32> = (0..KDA_HEAD_DIM).map(|i| 1.0 + val(i, 12, 0.5)).collect();
        let mut conv_ref = series(9 * width, 13, 1.0);
        let mut state_ref = series(HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM, 14, 0.2);
        let weights = KdaWeights {
            q_conv: &q_conv,
            k_conv: &k_conv,
            v_conv: &v_conv,
            neg_exp_a_log: &neg_exp_a_log,
            dt_bias: &dt_bias,
            output_norm: &output_norm,
            lower_bound: -5.0,
            norm_eps: 1e-5,
        };
        let w = width as u64;
        let d = KDA_HEAD_DIM as u64;
        let q_conv_t = tensor(&ctx, &q_conv, vec![4, 1, w]);
        let k_conv_t = tensor(&ctx, &k_conv, vec![4, 1, w]);
        let v_conv_t = tensor(&ctx, &v_conv, vec![4, 1, w]);
        let a_t = tensor(&ctx, &neg_exp_a_log, vec![HEADS as u64]);
        let dt_t = tensor(&ctx, &dt_bias, vec![w]);
        let norm_t = tensor(&ctx, &output_norm, vec![d]);
        let conv_t = tensor(&ctx, &conv_ref, vec![w, 3, 3]);
        let state_t = tensor(&ctx, &state_ref, vec![d, d, HEADS as u64]);
        for step in 0..STEPS {
            let o = step * width;
            let q = series(width, 1 + 20 * step, 2.0);
            let k = series(width, 2 + 20 * step, 2.0);
            let v = series(width, 3 + 20 * step, 2.0);
            let raw_gate = series(width, 4 + 20 * step, 6.0);
            let raw_beta: Vec<f32> = (0..HEADS).map(|h| val(h + o, 5, 4.0)).collect();
            let output_gate = series(width, 6 + 20 * step, 4.0);
            let expected = kda_decode_step(
                HEADS,
                &KdaStepInput {
                    q: &q,
                    k: &k,
                    v: &v,
                    raw_gate: &raw_gate,
                    raw_beta: &raw_beta,
                    output_gate: &output_gate,
                },
                &weights,
                &mut conv_ref,
                &mut state_ref,
            );
            let out_t = tensor(&ctx, &vec![0.0; width], vec![w]);
            let bindings = KdaDecode {
                q: &tensor(&ctx, &q, vec![w]),
                k: &tensor(&ctx, &k, vec![w]),
                v: &tensor(&ctx, &v, vec![w]),
                raw_gate: &tensor(&ctx, &raw_gate, vec![w]),
                raw_beta: &tensor(&ctx, &raw_beta, vec![HEADS as u64]),
                output_gate: &tensor(&ctx, &output_gate, vec![w]),
                q_conv: &q_conv_t,
                k_conv: &k_conv_t,
                v_conv: &v_conv_t,
                neg_exp_a_log: &a_t,
                dt_bias: &dt_t,
                output_norm: &norm_t,
                conv_state: &conv_t,
                state: &state_t,
                out: &out_t,
            };
            let command = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&command);
            encode_kda_decode(&ctx, &enc, HEADS, &bindings, -5.0, 1e-5).unwrap();
            enc.end();
            command.commit();
            wait_completed(&command).expect("KDA command");
            assert_close(
                &format!("step {step} output"),
                &tensor_f32_at_offset(&out_t),
                &expected,
                2e-5,
            );
            assert_close(
                &format!("step {step} state"),
                &tensor_f32_at_offset(&state_t),
                &state_ref,
                2e-6,
            );
            assert_eq!(
                tensor_f32_at_offset(&conv_t),
                conv_ref,
                "step {step} conv state"
            );
        }
    }

    #[test]
    fn kda_decode_refuses_misaligned_state_and_aliasing() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let w = KDA_HEAD_DIM as u64;
        let d = KDA_HEAD_DIM as u64;
        let act = tensor(&ctx, &vec![0.1; KDA_HEAD_DIM], vec![w]);
        let one = tensor(&ctx, &[0.0], vec![1]);
        let conv = tensor(&ctx, &vec![0.1; 4 * KDA_HEAD_DIM], vec![4, 1, w]);
        let norm = tensor(&ctx, &vec![1.0; KDA_HEAD_DIM], vec![d]);
        let conv_state = tensor(&ctx, &vec![0.0; 9 * KDA_HEAD_DIM], vec![w, 3, 3]);
        let state_values = vec![0.0f32; KDA_HEAD_DIM * KDA_HEAD_DIM];
        let misaligned = offset_tensor(
            &ctx,
            20,
            bytemuck::cast_slice(&state_values),
            20,
            vec![d, d, 1],
            GgmlType::F32,
        );
        let state = tensor(&ctx, &state_values, vec![d, d, 1]);
        let out = tensor(&ctx, &vec![0.0; KDA_HEAD_DIM], vec![w]);
        let bindings = |state, out| KdaDecode {
            q: &act,
            k: &act,
            v: &act,
            raw_gate: &act,
            raw_beta: &one,
            output_gate: &act,
            q_conv: &conv,
            k_conv: &conv,
            v_conv: &conv,
            neg_exp_a_log: &one,
            dt_bias: &act,
            output_norm: &norm,
            conv_state: &conv_state,
            state,
            out,
        };
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let err =
            encode_kda_decode(&ctx, &enc, 1, &bindings(&misaligned, &out), -5.0, 1e-5).unwrap_err();
        assert!(err.to_string().contains("16-byte aligned"), "{err}");
        let err =
            encode_kda_decode(&ctx, &enc, 1, &bindings(&state, &act), -5.0, 1e-5).unwrap_err();
        assert!(err.to_string().contains("aliases"), "{err}");
    }

    /// Packed prefill over two chunks (37 then 20 rows, so state and conv
    /// tails continue across calls), then one decode step, against the CPU
    /// contract iterated token by token: outputs, state and conv tails.
    #[test]
    fn kda_prefill_matches_stepwise_contract_across_chunks_and_decode() {
        with_kda_prepare_register_history_override(false, kda_prefill_stepwise_contract);
    }

    #[test]
    fn kda_prepare_register_history_matches_stepwise_contract_across_chunks_and_decode() {
        with_kda_prepare_register_history_override(true, kda_prefill_stepwise_contract);
    }

    fn kda_prefill_stepwise_contract() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const HEADS: usize = 64;
        let width = HEADS * KDA_HEAD_DIM;
        let (w, d, h) = (width as u64, KDA_HEAD_DIM as u64, HEADS as u64);
        let q_conv = series(width * 4, 7, 1.0);
        let k_conv = series(width * 4, 8, 1.0);
        let v_conv = series(width * 4, 9, 1.0);
        let neg_exp_a_log: Vec<f32> = (0..HEADS).map(|i| -(0.75 + val(i, 10, 1.0))).collect();
        let dt_bias = series(width, 11, 2.0);
        let output_norm: Vec<f32> = (0..KDA_HEAD_DIM).map(|i| 1.0 + val(i, 12, 0.5)).collect();
        let mut conv_ref = series(9 * width, 13, 1.0);
        let mut state_ref = series(HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM, 14, 0.2);
        let weights = KdaWeights {
            q_conv: &q_conv,
            k_conv: &k_conv,
            v_conv: &v_conv,
            neg_exp_a_log: &neg_exp_a_log,
            dt_bias: &dt_bias,
            output_norm: &output_norm,
            lower_bound: -5.0,
            norm_eps: 1e-5,
        };
        let q_conv_t = tensor(&ctx, &q_conv, vec![4, 1, w]);
        let k_conv_t = tensor(&ctx, &k_conv, vec![4, 1, w]);
        let v_conv_t = tensor(&ctx, &v_conv, vec![4, 1, w]);
        let a_t = tensor(&ctx, &neg_exp_a_log, vec![h]);
        let dt_t = tensor(&ctx, &dt_bias, vec![w]);
        let norm_t = tensor(&ctx, &output_norm, vec![d]);
        let conv_t = tensor(&ctx, &conv_ref, vec![w, 3, 3]);
        let state_t = tensor(&ctx, &state_ref, vec![d, d, h]);
        let token = |t: usize| {
            (
                series(width, 1 + 31 * t, 2.0),
                series(width, 2 + 31 * t, 2.0),
                series(width, 3 + 31 * t, 2.0),
                series(width, 4 + 31 * t, 6.0),
                (0..HEADS)
                    .map(|i| val(i + t * HEADS, 5, 4.0))
                    .collect::<Vec<_>>(),
                series(width, 6 + 31 * t, 4.0),
            )
        };
        let mut position = 0;
        for rows in [37usize, 20, 1] {
            let mut cols: [Vec<f32>; 6] = Default::default();
            let mut expected = Vec::with_capacity(rows * width);
            for t in position..position + rows {
                let (q, k, v, g, b, og) = token(t);
                expected.extend(kda_decode_step(
                    HEADS,
                    &KdaStepInput {
                        q: &q,
                        k: &k,
                        v: &v,
                        raw_gate: &g,
                        raw_beta: &b,
                        output_gate: &og,
                    },
                    &weights,
                    &mut conv_ref,
                    &mut state_ref,
                ));
                for (col, values) in cols.iter_mut().zip([q, k, v, g, b, og]) {
                    col.extend(values);
                }
            }
            let r = rows as u64;
            let out_t = tensor(&ctx, &vec![0.0; rows * width], vec![w, r]);
            let q_t = tensor(&ctx, &cols[0], vec![w, r]);
            let k_t = tensor(&ctx, &cols[1], vec![w, r]);
            let v_t = tensor(&ctx, &cols[2], vec![w, r]);
            let g_t = tensor(&ctx, &cols[3], vec![w, r]);
            let b_t = tensor(&ctx, &cols[4], vec![h, r]);
            let og_t = tensor(&ctx, &cols[5], vec![w, r]);
            let command = ctx.queue.commandBuffer().expect("command buffer");
            let enc = KernelEncoder::begin(&command);
            if rows == 1 {
                let flat = |t: &MetalTensor, n: u64| MetalTensor {
                    shape: vec![n],
                    ..t.clone()
                };
                let bindings = KdaDecode {
                    q: &flat(&q_t, w),
                    k: &flat(&k_t, w),
                    v: &flat(&v_t, w),
                    raw_gate: &flat(&g_t, w),
                    raw_beta: &flat(&b_t, h),
                    output_gate: &flat(&og_t, w),
                    q_conv: &q_conv_t,
                    k_conv: &k_conv_t,
                    v_conv: &v_conv_t,
                    neg_exp_a_log: &a_t,
                    dt_bias: &dt_t,
                    output_norm: &norm_t,
                    conv_state: &conv_t,
                    state: &state_t,
                    out: &flat(&out_t, w),
                };
                encode_kda_decode(&ctx, &enc, HEADS, &bindings, -5.0, 1e-5).unwrap();
            } else {
                let bindings = KdaDecode {
                    q: &q_t,
                    k: &k_t,
                    v: &v_t,
                    raw_gate: &g_t,
                    raw_beta: &b_t,
                    output_gate: &og_t,
                    q_conv: &q_conv_t,
                    k_conv: &k_conv_t,
                    v_conv: &v_conv_t,
                    neg_exp_a_log: &a_t,
                    dt_bias: &dt_t,
                    output_norm: &norm_t,
                    conv_state: &conv_t,
                    state: &state_t,
                    out: &out_t,
                };
                encode_kda_prefill(&ctx, &enc, HEADS, rows, &bindings, -5.0, 1e-5).unwrap();
            }
            enc.end();
            command.commit();
            wait_completed(&command).expect("KDA prefill command");
            assert_close(
                &format!("rows {rows} output"),
                &tensor_f32_at_offset(&out_t),
                &expected,
                3e-5,
            );
            assert_close(
                &format!("rows {rows} state"),
                &tensor_f32_at_offset(&state_t),
                &state_ref,
                3e-6,
            );
            assert_eq!(
                tensor_f32_at_offset(&conv_t),
                conv_ref,
                "rows {rows} conv state"
            );
            position += rows;
        }
    }

    #[test]
    fn kda_prepare_register_history_override_restores_scope() {
        const BASE: &str = "kernel_glm53_kda_prefill_prepare";
        const CANDIDATE: &str = "kernel_glm53_kda_prefill_prepare_register_history";
        assert_eq!(kda_prepare_pipeline_name(), BASE);
        with_kda_prepare_register_history_override(true, || {
            assert_eq!(kda_prepare_pipeline_name(), CANDIDATE);
            with_kda_prepare_register_history_override(false, || {
                assert_eq!(kda_prepare_pipeline_name(), BASE);
            });
            let _ = std::panic::catch_unwind(|| {
                with_kda_prepare_register_history_override(false, || panic!("restore"));
            });
            assert_eq!(kda_prepare_pipeline_name(), CANDIDATE);
            std::thread::spawn(|| assert_eq!(kda_prepare_pipeline_name(), BASE))
                .join()
                .unwrap();
        });
        assert_eq!(kda_prepare_pipeline_name(), BASE);
    }

    /// Only the history storage differs: exact prepared q/k/v/decay and final
    /// tails for short and long chunks, irregular heads, and nonzero offsets.
    #[test]
    fn kda_prepare_register_history_matches_bits_and_raw_history() {
        use super::super::test_support::{assert_offset_guards, tensor_backing_bytes};
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        for (heads, rows) in [(1, 1), (3, 2), (5, 3), (3, 512)] {
            let width = heads * KDA_HEAD_DIM;
            let (w, r, h, d) = (width as u64, rows as u64, heads as u64, KDA_HEAD_DIM as u64);
            let raw: [Vec<f32>; 4] =
                std::array::from_fn(|i| series(width * rows, i + 1, 2.0 + i as f64));
            let initial_history = series(9 * width, 13, 1.0);
            let mut expected_history = initial_history.clone();
            for token in 0..rows {
                for bank in 0..3 {
                    for c in 0..width {
                        let i = bank * 3 * width + c;
                        expected_history[i] = expected_history[i + width];
                        expected_history[i + width] = expected_history[i + 2 * width];
                        expected_history[i + 2 * width] = raw[bank][token * width + c];
                    }
                }
            }
            let q_conv = tensor(&ctx, &series(width * 4, 7, 1.0), vec![4, 1, w]);
            let k_conv = tensor(&ctx, &series(width * 4, 8, 1.0), vec![4, 1, w]);
            let v_conv = tensor(&ctx, &series(width * 4, 9, 1.0), vec![4, 1, w]);
            let a = tensor(
                &ctx,
                &(0..heads)
                    .map(|i| -(0.75 + val(i, 10, 1.0)))
                    .collect::<Vec<_>>(),
                vec![h],
            );
            let dt = tensor(&ctx, &series(width, 11, 2.0), vec![w]);
            // Full packed bindings are reused, but prepare must leave these
            // recurrence/output inputs and destinations untouched.
            let beta = tensor(&ctx, &series(heads * rows, 5, 4.0), vec![h, r]);
            let output_gate = tensor(&ctx, &series(width * rows, 6, 4.0), vec![w, r]);
            let norm = tensor(&ctx, &vec![1.0; KDA_HEAD_DIM], vec![d]);
            let state = tensor(
                &ctx,
                &series(heads * KDA_HEAD_DIM * KDA_HEAD_DIM, 14, 0.2),
                vec![d, d, h],
            );
            let out = tensor(&ctx, &vec![-777.0; width * rows], vec![w, r]);
            let unchanged = [
                &q_conv,
                &k_conv,
                &v_conv,
                &a,
                &dt,
                &beta,
                &output_gate,
                &norm,
                &state,
                &out,
            ];
            let before: Vec<_> = unchanged.iter().map(|t| tensor_backing_bytes(t)).collect();
            let mut baseline: Vec<Vec<f32>> = Vec::new();
            for enabled in [false, true] {
                let acts: Vec<_> = raw
                    .iter()
                    .enumerate()
                    .map(|(i, values)| {
                        offset_tensor(
                            &ctx,
                            16 * (i + 1),
                            bytemuck::cast_slice(values),
                            20,
                            vec![w, r],
                            GgmlType::F32,
                        )
                    })
                    .collect();
                let history = offset_tensor(
                    &ctx,
                    20,
                    bytemuck::cast_slice(&initial_history),
                    20,
                    vec![w, 3, 3],
                    GgmlType::F32,
                );
                let b = KdaDecode {
                    q: &acts[0],
                    k: &acts[1],
                    v: &acts[2],
                    raw_gate: &acts[3],
                    raw_beta: &beta,
                    output_gate: &output_gate,
                    q_conv: &q_conv,
                    k_conv: &k_conv,
                    v_conv: &v_conv,
                    neg_exp_a_log: &a,
                    dt_bias: &dt,
                    output_norm: &norm,
                    conv_state: &history,
                    state: &state,
                    out: &out,
                };
                if rows == 1 {
                    // The diagnostic leaf uses the same validation as full
                    // prefill, before encoding even its single dispatch.
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    let mut bad_q = acts[0].clone();
                    bad_q.offset += 4;
                    let bad = KdaDecode { q: &bad_q, ..b };
                    dispatch_census_begin();
                    assert!(
                        with_kda_prepare_register_history_override(enabled, || {
                            encode_kda_prefill_prepare(&ctx, &enc, heads, rows, &bad, -5.0, 1e-5)
                        })
                        .is_err()
                    );
                    assert!(dispatch_census_take().is_empty());
                    // Wrong shape and aliased writable input.
                    let bad = KdaDecode {
                        conv_state: &acts[0],
                        ..b
                    };
                    dispatch_census_begin();
                    assert!(
                        encode_kda_prefill_prepare(&ctx, &enc, heads, rows, &bad, -5.0, 1e-5)
                            .is_err()
                    );
                    assert!(dispatch_census_take().is_empty());
                    enc.end();
                }
                dispatch_census_begin();
                one_shot(&ctx, |enc| {
                    with_kda_prepare_register_history_override(enabled, || {
                        encode_kda_prefill_prepare(&ctx, enc, heads, rows, &b, -5.0, 1e-5)
                    })
                })
                .unwrap();
                let census = dispatch_census_take();
                assert_eq!(
                    census.len(),
                    1,
                    "prepare leaf must encode no recurrence/output"
                );
                assert_eq!(
                    census[0].kernel,
                    with_kda_prepare_register_history_override(enabled, kda_prepare_pipeline_name)
                );
                assert_eq!(census[0].grid_width, heads as u64);
                assert_eq!(census[0].tg_threads, THREADS as u64);
                let snapshots: Vec<Vec<f32>> = acts
                    .iter()
                    .chain(std::iter::once(&history))
                    .map(tensor_f32_at_offset)
                    .collect();
                if enabled {
                    for ((label, actual), expected) in ["q", "k", "v", "gate", "history"]
                        .into_iter()
                        .zip(&snapshots)
                        .zip(&baseline)
                    {
                        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "heads={heads} rows={rows} {label}[{i}]: candidate={a:e} baseline={b:e}"
                            );
                        }
                    }
                } else {
                    baseline = snapshots;
                }
                assert_eq!(
                    tensor_f32_at_offset(&history)
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    expected_history
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "heads={heads} rows={rows}: final raw history, oldest first"
                );
                for (i, t) in acts.iter().enumerate() {
                    assert_offset_guards(t, 16 * (i + 1), 20);
                }
                assert_offset_guards(&history, 20, 20);
                for (t, bytes) in unchanged.iter().zip(&before) {
                    assert_eq!(&tensor_backing_bytes(t), bytes);
                }
            }
        }
    }

    /// Timing screen (not qualification): GPU time of one GLM-5.3 KDA decode
    /// block (hidden 4096, 64 heads; Q6_K q/k/v [4096 -> 8192] and output
    /// [8192 -> 4096]; Q8_0 f_a/g_a [4096 -> 128], beta [4096 -> 64],
    /// f_b/g_b [128 -> 8192]; the recurrence) as a chain of 34 dependent
    /// blocks per command, cycling 8 distinct weight and state sets so the
    /// weights stream from DRAM, whole and per dispatch group. Proxy rows
    /// time one dispatch over the concatenated rows of a group: an upper
    /// bound for a multi-matrix dispatch with the same per-row kernel.
    /// Synthetic weights; refuses MTL_DEBUG_LAYER.
    #[test]
    #[ignore = "timing screen; run without MTL_DEBUG_LAYER"]
    fn kda_block_dispatch_costs() {
        assert!(
            std::env::var_os("MTL_DEBUG_LAYER").is_none(),
            "timing runs must not enable MTL_DEBUG_LAYER"
        );
        let _lease = crate::metal::acquire_metal_benchmark_lease().expect("GPU lease");
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const H: usize = 4096;
        const HEADS: usize = 64;
        const RANK: usize = KDA_HEAD_DIM;
        const SETS: usize = 8;
        const BLOCKS: usize = 34;
        let width = HEADS * KDA_HEAD_DIM;
        let q6 = |n_in: usize, n_out: usize, seed: usize| -> MetalTensor {
            let per_row = n_in / 256;
            let mut bytes = Vec::with_capacity(n_out * per_row * 210);
            let blocks: Vec<[u8; 210]> = (0..16)
                .map(|i| super::super::test_support::encode_q6_k_block(0.01, seed * 31 + i).0)
                .collect();
            for i in 0..n_out * per_row {
                bytes.extend_from_slice(&blocks[i % blocks.len()]);
            }
            offset_tensor(
                &ctx,
                0,
                &bytes,
                0,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q6_K,
            )
        };
        let q8 = |n_in: usize, n_out: usize| -> MetalTensor {
            let bytes = super::super::test_support::synthetic_q8_0_bytes(n_in, n_out);
            offset_tensor(
                &ctx,
                0,
                &bytes,
                0,
                vec![n_in as u64, n_out as u64],
                GgmlType::Q8_0,
            )
        };
        struct Set {
            q: MetalTensor,
            k: MetalTensor,
            v: MetalTensor,
            output: MetalTensor,
            f_a: MetalTensor,
            g_a: MetalTensor,
            beta: MetalTensor,
            f_b: MetalTensor,
            g_b: MetalTensor,
            conv: MetalTensor,
            state: MetalTensor,
        }
        let w = width as u64;
        let d = KDA_HEAD_DIM as u64;
        let sets: Vec<Set> = (0..SETS)
            .map(|i| Set {
                q: q6(H, width, 3 * i),
                k: q6(H, width, 3 * i + 1),
                v: q6(H, width, 3 * i + 2),
                output: q6(width, H, 100 + i),
                f_a: q8(H, RANK),
                g_a: q8(H, RANK),
                beta: q8(H, HEADS),
                f_b: q8(RANK, width),
                g_b: q8(RANK, width),
                conv: tensor(&ctx, &series(9 * width, 13 + i, 1.0), vec![w, 3, 3]),
                state: tensor(
                    &ctx,
                    &series(HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM, 14 + i, 0.2),
                    vec![d, d, HEADS as u64],
                ),
            })
            .collect();
        // Proxies: one dispatch over the concatenated rows of a group.
        let qkv_proxy: Vec<MetalTensor> = (0..SETS).map(|i| q6(H, 3 * width, 200 + i)).collect();
        let a_proxy: Vec<MetalTensor> = (0..SETS).map(|_| q8(H, 2 * RANK + HEADS)).collect();
        let b_proxy: Vec<MetalTensor> = (0..SETS).map(|_| q8(RANK, 2 * width)).collect();
        let f32_zeros = |n: usize| tensor(&ctx, &vec![0.0; n], vec![n as u64]);
        let normed = tensor(&ctx, &series(H, 1, 2.0), vec![H as u64]);
        let (q, k, v) = (f32_zeros(width), f32_zeros(width), f32_zeros(width));
        let (rank_a, rank_b) = (f32_zeros(RANK), f32_zeros(RANK));
        let (raw_gate, output_gate) = (f32_zeros(width), f32_zeros(width));
        let raw_beta = f32_zeros(HEADS);
        let kda_out = f32_zeros(width);
        let block_out = f32_zeros(H);
        let qkv_out = f32_zeros(3 * width);
        let a_out = f32_zeros(2 * RANK + HEADS);
        let b_out = f32_zeros(2 * width);
        let q_conv = tensor(&ctx, &series(width * 4, 7, 1.0), vec![4, 1, w]);
        let k_conv = tensor(&ctx, &series(width * 4, 8, 1.0), vec![4, 1, w]);
        let v_conv = tensor(&ctx, &series(width * 4, 9, 1.0), vec![4, 1, w]);
        let neg_exp_a_log: Vec<f32> = (0..HEADS).map(|h| -(0.75 + val(h, 10, 1.0))).collect();
        let a_t = tensor(&ctx, &neg_exp_a_log, vec![HEADS as u64]);
        let dt_t = tensor(&ctx, &series(width, 11, 2.0), vec![w]);
        let norm_t = tensor(&ctx, &vec![1.0; KDA_HEAD_DIM], vec![d]);
        let mv = |enc: &KernelEncoder, wt: &MetalTensor, x: &MetalTensor, y: &MetalTensor| {
            let (n_in, n_out) = (wt.shape[0] as usize, wt.shape[1] as usize);
            crate::metal_forward::encode_mat_vec_dispatch(&ctx, enc, wt, x, y, n_in, n_out)
                .unwrap();
        };
        let qkv = |enc: &KernelEncoder, s: &Set| {
            mv(enc, &s.q, &normed, &q);
            mv(enc, &s.k, &normed, &k);
            mv(enc, &s.v, &normed, &v);
        };
        let small = |enc: &KernelEncoder, s: &Set| {
            mv(enc, &s.f_a, &normed, &rank_a);
            mv(enc, &s.f_b, &rank_a, &raw_gate);
            mv(enc, &s.beta, &normed, &raw_beta);
            mv(enc, &s.g_a, &normed, &rank_b);
            mv(enc, &s.g_b, &rank_b, &output_gate);
        };
        let short = |enc: &KernelEncoder, wt: &MetalTensor, x: &MetalTensor, y: &MetalTensor| {
            let (n_in, n_out) = (wt.shape[0] as usize, wt.shape[1] as usize);
            crate::metal::encode_mat_vec_q8_0_short_k_f32(&ctx, enc, wt, x, y, n_in, n_out, 1)
                .unwrap();
        };
        let small_short = |enc: &KernelEncoder, s: &Set| {
            mv(enc, &s.f_a, &normed, &rank_a);
            short(enc, &s.f_b, &rank_a, &raw_gate);
            mv(enc, &s.beta, &normed, &raw_beta);
            mv(enc, &s.g_a, &normed, &rank_b);
            short(enc, &s.g_b, &rank_b, &output_gate);
        };
        let recurrence = |enc: &KernelEncoder, s: &Set| {
            encode_kda_decode(
                &ctx,
                enc,
                HEADS,
                &KdaDecode {
                    q: &q,
                    k: &k,
                    v: &v,
                    raw_gate: &raw_gate,
                    raw_beta: &raw_beta,
                    output_gate: &output_gate,
                    q_conv: &q_conv,
                    k_conv: &k_conv,
                    v_conv: &v_conv,
                    neg_exp_a_log: &a_t,
                    dt_bias: &dt_t,
                    output_norm: &norm_t,
                    conv_state: &s.conv,
                    state: &s.state,
                    out: &kda_out,
                },
                -5.0,
                1e-5,
            )
            .unwrap();
        };
        let output = |enc: &KernelEncoder, s: &Set| mv(enc, &s.output, &kda_out, &block_out);
        let time = |encode: &dyn Fn(&KernelEncoder, usize)| -> f64 {
            let mut samples: Vec<f64> = (0..6)
                .map(|_| {
                    let command = ctx.queue.commandBuffer().expect("command buffer");
                    let enc = KernelEncoder::begin(&command);
                    for block in 0..BLOCKS {
                        encode(&enc, block % SETS);
                    }
                    enc.end();
                    command.commit();
                    wait_completed(&command).expect("command buffer failed");
                    (command.GPUEndTime() - command.GPUStartTime()) * 1e6 / BLOCKS as f64
                })
                .skip(1)
                .collect();
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        let whole = |enc: &KernelEncoder, i: usize| {
            let s = &sets[i];
            qkv(enc, s);
            small(enc, s);
            recurrence(enc, s);
            output(enc, s);
        };
        let warm = std::time::Instant::now();
        while warm.elapsed() < std::time::Duration::from_secs(2) {
            time(&whole);
        }
        let q6_bytes = |n_in: usize, n_out: usize| (n_in / 256 * n_out * 210) as f64;
        let q8_bytes = |n_in: usize, n_out: usize| (n_in / 32 * n_out * 34) as f64;
        let small_bytes =
            2.0 * q8_bytes(H, RANK) + q8_bytes(H, HEADS) + 2.0 * q8_bytes(RANK, width);
        let qkv_bytes = 3.0 * q6_bytes(H, width);
        let state_bytes = 2.0 * (HEADS * KDA_HEAD_DIM * KDA_HEAD_DIM * 4) as f64;
        type Encode<'a> = &'a dyn Fn(&KernelEncoder, usize);
        let rows: [(&str, Encode<'_>, f64); 13] = [
            (
                "whole block, short-K b projections",
                &|e: &KernelEncoder, i: usize| {
                    let s = &sets[i];
                    qkv(e, s);
                    small_short(e, s);
                    recurrence(e, s);
                    output(e, s);
                },
                qkv_bytes + small_bytes + q6_bytes(width, H) + state_bytes,
            ),
            (
                "small q8_0, short-K b projections",
                &|e: &KernelEncoder, i: usize| small_short(e, &sets[i]),
                small_bytes,
            ),
            (
                "q8_0 128->8192 short-K",
                &|e: &KernelEncoder, i: usize| short(e, &sets[i].f_b, &rank_a, &raw_gate),
                q8_bytes(RANK, width),
            ),
            (
                "whole block (9 matvecs + recurrence)",
                &whole,
                qkv_bytes + small_bytes + q6_bytes(width, H) + state_bytes,
            ),
            (
                "q, k, v (3 x q6_k 4096->8192)",
                &|e: &KernelEncoder, i: usize| qkv(e, &sets[i]),
                qkv_bytes,
            ),
            (
                "proxy: one q6_k 4096->24576",
                &|e: &KernelEncoder, i: usize| mv(e, &qkv_proxy[i], &normed, &qkv_out),
                qkv_bytes,
            ),
            (
                "small q8_0 (5 dispatches)",
                &|e: &KernelEncoder, i: usize| small(e, &sets[i]),
                small_bytes,
            ),
            (
                "proxy: q8_0 4096->320 + 128->16384",
                &|e: &KernelEncoder, i: usize| {
                    mv(e, &a_proxy[i], &normed, &a_out);
                    mv(
                        e,
                        &b_proxy[i],
                        &a_out.view_subrange(0, vec![RANK as u64]),
                        &b_out,
                    );
                },
                small_bytes,
            ),
            (
                "recurrence",
                &|e: &KernelEncoder, i: usize| recurrence(e, &sets[i]),
                state_bytes,
            ),
            (
                "output (q6_k 8192->4096)",
                &|e: &KernelEncoder, i: usize| output(e, &sets[i]),
                q6_bytes(width, H),
            ),
            (
                "q6_k 4096->8192 alone",
                &|e: &KernelEncoder, i: usize| mv(e, &sets[i].q, &normed, &q),
                q6_bytes(H, width),
            ),
            (
                "q8_0 128->8192 generic",
                &|e: &KernelEncoder, i: usize| mv(e, &sets[i].f_b, &rank_a, &raw_gate),
                q8_bytes(RANK, width),
            ),
            (
                "q8_0 4096->128 alone",
                &|e: &KernelEncoder, i: usize| mv(e, &sets[i].f_a, &normed, &rank_a),
                q8_bytes(H, RANK),
            ),
        ];
        for (label, encode, bytes) in rows {
            let us = time(encode);
            eprintln!(
                "kda block: {label:<40} {us:8.2} us/block {:7.1} GB/s",
                bytes / us / 1e3
            );
        }
    }
}
