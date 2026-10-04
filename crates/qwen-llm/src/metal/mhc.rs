//! Four-stream manifold-constrained hyper-connection (mHC) primitives and the
//! clamped SwiGLU activation, shared by DeepSeek V4 and GLM-5.3-Flash.
//!
//! Allocation-free: callers own every buffer. Layouts: residuals are
//! stream-major `[hidden, 4]` (stream `s` occupies `[s * hidden, (s + 1) *
//! hidden)`); the 24 mixes are `pre[4] | post[4] | comb[16]`; the combination
//! matrix is source-major, `comb[src * 4 + dst]`. The mix RMSNorm and
//! projection stay with the caller; [`encode_mhc4_controls`] consumes projected
//! F32 mixes. Single-token forms only; packed forms land with their first
//! shared caller.
//!
//! CPU contract: `crate::deepseek_v4_oracle::{hyper_connection_pre,
//! hyper_connection_post}`.

use super::*;

/// Residual streams.
pub const MHC_STREAMS: usize = 4;
/// `pre[4] + post[4] + comb[4 * 4]`.
pub const MHC_MIXES: usize = 24;
/// Sinkhorn rounds baked into the controls kernel (the first row step is a
/// softmax, with epsilon added after it).
pub const MHC_SINKHORN_ITERATIONS: usize = 20;

use super::checks::{
    bad_shape as bad, check_disjoint, check_tensor, overlaps, require_serial, same_range,
};

fn check_f32(
    kernel: &'static str,
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
    name: &str,
) -> Result<(), MetalError> {
    check_tensor(kernel, tensor, GgmlType::F32, shape, writable, name)
}

/// `hidden` as u32 and the flattened residual length, both nonzero and u32.
fn residual_geometry(kernel: &'static str, hidden: usize) -> Result<(u32, usize), MetalError> {
    let len = hidden
        .checked_mul(MHC_STREAMS)
        .filter(|&len| hidden > 0 && u32::try_from(len).is_ok())
        .ok_or_else(|| bad(kernel, format!("hidden size {hidden} is zero or too large")))?;
    Ok((hidden as u32, len))
}

fn dispatch_linear(enc: &KernelEncoder, threads: usize) {
    enc.dispatch(
        MTLSize {
            width: threads.div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
}

/// Repeat one `[hidden]` row into all four residual streams.
pub fn encode_mhc4_repeat(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    input: &MetalTensor,
    residual: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_repeat";
    require_serial(K, enc)?;
    let (h, len) = residual_geometry(K, hidden)?;
    check_f32(K, input, &[hidden as u64], false, "input")?;
    check_f32(K, residual, &[hidden as u64, 4], true, "residual")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_repeat")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &h);
    enc.set_tensor(1, input);
    enc.set_tensor(2, residual);
    dispatch_linear(enc, len);
    Ok(())
}

/// Controls from projected mixes: `pre = sigmoid(m * scale[0] + base) + eps`,
/// `post = 2 sigmoid(m * scale[1] + base)`, and the Sinkhorn-normalized
/// source-major combination from `m * scale[2] + base`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mhc4_controls(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    eps: f32,
    mixes: &MetalTensor,
    scale: &MetalTensor,
    base: &MetalTensor,
    pre: &MetalTensor,
    post: &MetalTensor,
    comb: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_controls";
    require_serial(K, enc)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(bad(
            K,
            format!("epsilon must be finite and positive, got {eps}"),
        ));
    }
    check_f32(K, mixes, &[MHC_MIXES as u64], false, "mixes")?;
    check_f32(K, scale, &[3], false, "scale")?;
    check_f32(K, base, &[MHC_MIXES as u64], false, "base")?;
    check_f32(K, pre, &[4], true, "pre")?;
    check_f32(K, post, &[4], true, "post")?;
    check_f32(K, comb, &[4, 4], true, "combination")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_controls")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &eps);
    enc.set_tensor(1, mixes);
    enc.set_tensor(2, scale);
    enc.set_tensor(3, base);
    enc.set_tensor(4, pre);
    enc.set_tensor(5, post);
    enc.set_tensor(6, comb);
    let one = MTLSize {
        width: 1,
        height: 1,
        depth: 1,
    };
    enc.dispatch(one, one);
    Ok(())
}

/// Weighted stream sum `output[d] = sum_s residual[s, d] * pre[s]`. With
/// `pre = 1/4` this is the unweighted mean used before an output head.
pub fn encode_mhc4_collapse(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    residual: &MetalTensor,
    pre: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_collapse";
    require_serial(K, enc)?;
    let (h, _) = residual_geometry(K, hidden)?;
    check_f32(K, residual, &[hidden as u64, 4], false, "residual")?;
    check_f32(K, pre, &[4], false, "pre")?;
    check_f32(K, output, &[hidden as u64], true, "output")?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_collapse")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &h);
    enc.set_tensor(1, residual);
    enc.set_tensor(2, pre);
    enc.set_tensor(3, output);
    dispatch_linear(enc, hidden);
    Ok(())
}

/// `output[dst, d] = block[d] * post[dst] + sum_src comb[src, dst] *
/// residual[src, d]`. `output` must not alias `residual`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mhc4_post(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    block_output: &MetalTensor,
    residual: &MetalTensor,
    post: &MetalTensor,
    comb: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_post";
    require_serial(K, enc)?;
    let (h, len) = residual_geometry(K, hidden)?;
    check_f32(K, block_output, &[hidden as u64], false, "block output")?;
    check_f32(K, residual, &[hidden as u64, 4], false, "residual")?;
    check_f32(K, post, &[4], false, "post")?;
    check_f32(K, comb, &[4, 4], false, "combination")?;
    check_f32(K, output, &[hidden as u64, 4], true, "output")?;
    // Streams read across the residual while others are written.
    check_disjoint(
        K,
        output,
        &[
            (block_output, "block output"),
            (residual, "residual"),
            (post, "post"),
            (comb, "combination"),
        ],
    )?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_post")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &h);
    enc.set_tensor(1, block_output);
    enc.set_tensor(2, residual);
    enc.set_tensor(3, post);
    enc.set_tensor(4, comb);
    enc.set_tensor(5, output);
    dispatch_linear(enc, len);
    Ok(())
}

/// `output = silu(min(gate, clamp)) * clamp(up, -clamp, clamp)` elementwise.
/// `output` may be exactly `gate` or exactly `up` (same bytes); any other
/// overlap is refused.
pub fn encode_clamped_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    output: &MetalTensor,
    clamp: f32,
) -> Result<(), MetalError> {
    const K: &str = "clamped_swiglu";
    require_serial(K, enc)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return Err(bad(
            K,
            format!("clamp must be finite and positive, got {clamp}"),
        ));
    }
    let n = gate.n_elements();
    check_f32(K, gate, &[n], false, "gate")?;
    check_f32(K, up, &[n], false, "up")?;
    check_f32(K, output, &[n], true, "output")?;
    for (input, name) in [(gate, "gate"), (up, "up")] {
        if overlaps(output, input) && !same_range(output, input) {
            return Err(bad(K, format!("output partially overlaps {name}")));
        }
    }
    let n = u32::try_from(n).map_err(|_| bad(K, "width exceeds u32"))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        clamp: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_clamped_swiglu")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n, clamp });
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, output);
    dispatch_linear(enc, n as usize);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{offset_tensor, tensor_f32_at_offset};
    use super::*;
    use crate::deepseek_v4_oracle::{clamped_swiglu, hyper_connection_post, hyper_connection_pre};

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

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() <= tolerance * (1.0 + e.abs()),
                "{label}[{i}]: {a} vs {e}"
            );
        }
    }

    fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        wait_completed(&command).expect("command buffer failed");
    }

    /// GLM-5.3 width with asymmetric streams and offset bindings: controls from
    /// oracle mixes, weighted and mean collapse, and post, against the CPU contract.
    #[test]
    fn mhc4_primitives_match_cpu_contract_at_glm_width() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const H: usize = 4096;
        let n = H * MHC_STREAMS;
        let residual: Vec<f32> = (0..n)
            .map(|i| {
                let (s, d) = (i / H, i % H);
                (s as f32 - 1.3) * 0.61 + ((d * 7 + s * 3) % 29) as f32 * 0.013 - 0.17
            })
            .collect();
        let function: Vec<f32> = (0..n * MHC_MIXES)
            .map(|i| ((i * 17 + i / 13) % 37) as f32 * 0.0011 - 0.019)
            .collect();
        let scale = [0.8f32, -0.45, 1.1];
        let base: Vec<f32> = (0..MHC_MIXES)
            .map(|i| ((i * 5 + 2) % 17) as f32 * 0.04 - 0.3)
            .collect();
        let block: Vec<f32> = (0..H).map(|d| ((d % 23) as f32 - 11.0) * 0.07).collect();
        let (rms_eps, hc_eps) = (1e-5, 1e-6);
        let pre = hyper_connection_pre(
            &residual,
            H,
            MHC_STREAMS,
            &function,
            &scale,
            &base,
            rms_eps,
            MHC_SINKHORN_ITERATIONS,
            hc_eps,
        )
        .unwrap();
        let post = hyper_connection_post(&block, &residual, &pre.controls, H, MHC_STREAMS).unwrap();
        let mean: Vec<f32> = (0..H)
            .map(|d| (0..4).map(|s| residual[s * H + d]).sum::<f32>() * 0.25)
            .collect();

        let residual_t = tensor(&ctx, &residual, vec![H as u64, 4]);
        let mixes_t = tensor(&ctx, &pre.mixes, vec![MHC_MIXES as u64]);
        let scale_t = tensor(&ctx, &scale, vec![3]);
        let base_t = tensor(&ctx, &base, vec![MHC_MIXES as u64]);
        let pre_t = tensor(&ctx, &[0.0; 4], vec![4]);
        let post_t = tensor(&ctx, &[0.0; 4], vec![4]);
        let comb_t = tensor(&ctx, &[0.0; 16], vec![4, 4]);
        let collapsed_t = tensor(&ctx, &vec![0.0; H], vec![H as u64]);
        let quarter_t = tensor(&ctx, &[0.25; 4], vec![4]);
        let mean_t = tensor(&ctx, &vec![0.0; H], vec![H as u64]);
        let block_t = tensor(&ctx, &block, vec![H as u64]);
        let out_t = tensor(&ctx, &vec![0.0; n], vec![H as u64, 4]);
        let embed_t = tensor(&ctx, &block, vec![H as u64]);
        let repeated_t = tensor(&ctx, &vec![0.0; n], vec![H as u64, 4]);
        run(&ctx, |enc| {
            encode_mhc4_controls(
                &ctx, enc, hc_eps, &mixes_t, &scale_t, &base_t, &pre_t, &post_t, &comb_t,
            )
            .unwrap();
            encode_mhc4_collapse(&ctx, enc, H, &residual_t, &pre_t, &collapsed_t).unwrap();
            encode_mhc4_collapse(&ctx, enc, H, &residual_t, &quarter_t, &mean_t).unwrap();
            encode_mhc4_post(
                &ctx,
                enc,
                H,
                &block_t,
                &residual_t,
                &post_t,
                &comb_t,
                &out_t,
            )
            .unwrap();
            encode_mhc4_repeat(&ctx, enc, H, &embed_t, &repeated_t).unwrap();
        });
        assert_close(
            "pre",
            &tensor_f32_at_offset(&pre_t),
            &pre.controls.pre,
            1e-5,
        );
        assert_close(
            "post",
            &tensor_f32_at_offset(&post_t),
            &pre.controls.post,
            1e-5,
        );
        assert_close(
            "combination",
            &tensor_f32_at_offset(&comb_t),
            &pre.controls.combination,
            1e-5,
        );
        assert_close(
            "collapse",
            &tensor_f32_at_offset(&collapsed_t),
            &pre.input,
            1e-5,
        );
        assert_close("mean collapse", &tensor_f32_at_offset(&mean_t), &mean, 1e-6);
        assert_close("post residual", &tensor_f32_at_offset(&out_t), &post, 1e-5);
        let repeated = tensor_f32_at_offset(&repeated_t);
        for s in 0..4 {
            assert_eq!(&repeated[s * H..(s + 1) * H], &block[..], "stream {s}");
        }
    }

    #[test]
    fn clamped_swiglu_matches_contract_at_boundaries_and_in_place() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let gate = [
            -20.0f32,
            -10.0,
            -0.5,
            0.0,
            3.0,
            9.99,
            10.0,
            10.01,
            50.0,
            f32::MIN_POSITIVE,
        ];
        let up = [
            20.0f32, -10.0, 0.25, 7.0, -10.5, 10.0, 11.0, -50.0, 2.0, -1.0,
        ];
        let expected = clamped_swiglu(&gate, &up, 10.0).unwrap();
        let shape = vec![gate.len() as u64];
        let gate_t = tensor(&ctx, &gate, shape.clone());
        let up_t = tensor(&ctx, &up, shape.clone());
        let out_t = tensor(&ctx, &[0.0; 10], shape.clone());
        let in_place_t = tensor(&ctx, &gate, shape);
        run(&ctx, |enc| {
            encode_clamped_swiglu(&ctx, enc, &gate_t, &up_t, &out_t, 10.0).unwrap();
            encode_clamped_swiglu(&ctx, enc, &in_place_t, &up_t, &in_place_t, 10.0).unwrap();
        });
        assert_close(
            "clamped swiglu",
            &tensor_f32_at_offset(&out_t),
            &expected,
            1e-6,
        );
        assert_close(
            "in place",
            &tensor_f32_at_offset(&in_place_t),
            &expected,
            1e-6,
        );
        assert!(
            encode_clamped_swiglu(
                &ctx,
                &KernelEncoder::begin(&ctx.queue.commandBuffer().unwrap()),
                &gate_t,
                &up_t,
                &out_t,
                0.0
            )
            .is_err()
        );
    }

    #[test]
    fn post_and_swiglu_refuse_unsafe_aliasing() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const H: usize = 8;
        let residual = tensor(&ctx, &[0.5; H * 4], vec![H as u64, 4]);
        let block = tensor(&ctx, &[0.25; H], vec![H as u64]);
        let post = tensor(&ctx, &[1.0; 4], vec![4]);
        let comb = tensor(&ctx, &[0.25; 16], vec![4, 4]);
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let err = encode_mhc4_post(&ctx, &enc, H, &block, &residual, &post, &comb, &residual)
            .unwrap_err();
        assert!(err.to_string().contains("aliases residual"), "{err}");

        let gate = tensor(&ctx, &[1.0; 8], vec![8]);
        let up = tensor(&ctx, &[2.0; 8], vec![8]);
        let shifted = MetalTensor {
            offset: gate.offset + 4,
            ..gate.clone()
        };
        let err = encode_clamped_swiglu(&ctx, &enc, &gate, &up, &shifted, 10.0).unwrap_err();
        assert!(err.to_string().contains("partially overlaps gate"), "{err}");
        encode_clamped_swiglu(&ctx, &enc, &gate, &up, &gate, 10.0).unwrap();
    }
}
