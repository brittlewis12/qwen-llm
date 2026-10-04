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
        let worst = actual
            .iter()
            .zip(expected)
            .map(|(a, e)| (a - e).abs())
            .fold(0.0f32, f32::max);
        assert!(
            actual.iter().all(|v| v.is_finite()) && worst / scale <= relative,
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
}
