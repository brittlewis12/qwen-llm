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

/// Control outputs share no byte with the inputs or with each other.
fn check_controls_disjoint(
    kernel: &'static str,
    mixes: &MetalTensor,
    scale: &MetalTensor,
    base: &MetalTensor,
    pre: &MetalTensor,
    post: &MetalTensor,
    comb: &MetalTensor,
) -> Result<(), MetalError> {
    let inputs = [(mixes, "mixes"), (scale, "scale"), (base, "base")];
    check_disjoint(kernel, pre, &inputs)?;
    check_disjoint(kernel, post, &inputs)?;
    check_disjoint(kernel, comb, &inputs)?;
    check_disjoint(kernel, pre, &[(post, "post"), (comb, "combination")])?;
    check_disjoint(kernel, post, &[(comb, "combination")])
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
    check_controls_disjoint(K, mixes, scale, base, pre, post, comb)?;
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
    check_disjoint(K, output, &[(residual, "residual"), (pre, "pre")])?;
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

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RowsArgs {
    hidden_size: u32,
    n_tokens: u32,
}

fn rows_u32(kernel: &'static str, rows: usize) -> Result<u32, MetalError> {
    u32::try_from(rows)
        .ok()
        .filter(|&r| r > 0)
        .ok_or_else(|| bad(kernel, format!("rows {rows} must be positive and fit u32")))
}

/// [`encode_mhc4_repeat`] for `rows` tokens: `input` `[hidden, rows]` into
/// `residual` `[hidden, 4, rows]`.
pub fn encode_mhc4_repeat_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    rows: usize,
    input: &MetalTensor,
    residual: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_repeat_rows";
    require_serial(K, enc)?;
    let (h, len) = residual_geometry(K, hidden)?;
    let n = rows_u32(K, rows)?;
    let r = rows as u64;
    check_f32(K, input, &[hidden as u64, r], false, "input")?;
    check_f32(K, residual, &[hidden as u64, 4, r], true, "residual")?;
    check_disjoint(K, residual, &[(input, "input")])?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_repeat_batch")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &RowsArgs {
            hidden_size: h,
            n_tokens: n,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, residual);
    dispatch_linear(enc, len * rows);
    Ok(())
}

/// [`encode_mhc4_controls`] for `rows` tokens: mixes `[24, rows]` into pre and
/// post `[4, rows]` and combinations `[4, 4, rows]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mhc4_controls_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    rows: usize,
    eps: f32,
    mixes: &MetalTensor,
    scale: &MetalTensor,
    base: &MetalTensor,
    pre: &MetalTensor,
    post: &MetalTensor,
    comb: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_controls_rows";
    require_serial(K, enc)?;
    if !eps.is_finite() || eps <= 0.0 {
        return Err(bad(
            K,
            format!("epsilon must be finite and positive, got {eps}"),
        ));
    }
    let n = rows_u32(K, rows)?;
    let r = rows as u64;
    check_f32(K, mixes, &[MHC_MIXES as u64, r], false, "mixes")?;
    check_f32(K, scale, &[3], false, "scale")?;
    check_f32(K, base, &[MHC_MIXES as u64], false, "base")?;
    check_f32(K, pre, &[4, r], true, "pre")?;
    check_f32(K, post, &[4, r], true, "post")?;
    check_f32(K, comb, &[4, 4, r], true, "combination")?;
    check_controls_disjoint(K, mixes, scale, base, pre, post, comb)?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_tokens: u32,
        eps: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_controls_batch")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n_tokens: n, eps });
    enc.set_tensor(1, mixes);
    enc.set_tensor(2, scale);
    enc.set_tensor(3, base);
    enc.set_tensor(4, pre);
    enc.set_tensor(5, post);
    enc.set_tensor(6, comb);
    enc.dispatch(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// [`encode_mhc4_collapse`] for `rows` tokens: residual `[hidden, 4, rows]`
/// and pre `[4, rows]` into output `[hidden, rows]`.
pub fn encode_mhc4_collapse_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    rows: usize,
    residual: &MetalTensor,
    pre: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_collapse_rows";
    require_serial(K, enc)?;
    let (h, _) = residual_geometry(K, hidden)?;
    let n = rows_u32(K, rows)?;
    let r = rows as u64;
    check_f32(K, residual, &[hidden as u64, 4, r], false, "residual")?;
    check_f32(K, pre, &[4, r], false, "pre")?;
    check_f32(K, output, &[hidden as u64, r], true, "output")?;
    check_disjoint(K, output, &[(residual, "residual"), (pre, "pre")])?;
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_collapse_batch")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &RowsArgs {
            hidden_size: h,
            n_tokens: n,
        },
    );
    enc.set_tensor(1, residual);
    enc.set_tensor(2, pre);
    enc.set_tensor(3, output);
    dispatch_linear(enc, hidden * rows);
    Ok(())
}

/// [`encode_mhc4_post`] for `rows` tokens; `output` must not alias `residual`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mhc4_post_rows(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    rows: usize,
    block_output: &MetalTensor,
    residual: &MetalTensor,
    post: &MetalTensor,
    comb: &MetalTensor,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_post_rows";
    require_serial(K, enc)?;
    let (h, len) = residual_geometry(K, hidden)?;
    let n = rows_u32(K, rows)?;
    let r = rows as u64;
    check_f32(K, block_output, &[hidden as u64, r], false, "block output")?;
    check_f32(K, residual, &[hidden as u64, 4, r], false, "residual")?;
    check_f32(K, post, &[4, r], false, "post")?;
    check_f32(K, comb, &[4, 4, r], false, "combination")?;
    check_f32(K, output, &[hidden as u64, 4, r], true, "output")?;
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
    let pso = ctx.pipeline("kernel_deepseek_v4_hc_post_batch")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &RowsArgs {
            hidden_size: h,
            n_tokens: n,
        },
    );
    enc.set_tensor(1, block_output);
    enc.set_tensor(2, residual);
    enc.set_tensor(3, post);
    enc.set_tensor(4, comb);
    enc.set_tensor(5, output);
    dispatch_linear(enc, len * rows);
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
    fn mhc_and_swiglu_refuse_unsafe_aliasing() {
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

        // Controls: outputs share no byte with the inputs or each other.
        let mixes = tensor(&ctx, &[0.1; 24], vec![24]);
        let scale = tensor(&ctx, &[1.0; 3], vec![3]);
        let base = tensor(&ctx, &[0.0; 24], vec![24]);
        let pre = tensor(&ctx, &[0.0; 4], vec![4]);
        let in_mixes = mixes.view_subrange(4, vec![4]);
        let controls = |pre: &MetalTensor, post: &MetalTensor, comb: &MetalTensor| {
            encode_mhc4_controls(&ctx, &enc, 1e-6, &mixes, &scale, &base, pre, post, comb)
                .map(|()| String::new())
                .unwrap_or_else(|e| e.to_string())
        };
        let refused = |message: String, name: &str| {
            assert!(
                message.contains(&format!("output aliases {name}")),
                "{message}"
            );
        };
        refused(controls(&in_mixes, &post, &comb), "mixes");
        refused(controls(&pre, &pre, &comb), "post");
        let shared = tensor(&ctx, &[0.0; 20], vec![20]);
        let (post_s, comb_s) = (
            shared.view_subrange(0, vec![4]),
            shared.view_subrange(2, vec![4, 4]),
        );
        refused(controls(&pre, &post_s, &comb_s), "combination");
        assert_eq!(controls(&pre, &post, &comb), "");
        let rows = |pre: &MetalTensor, post: &MetalTensor| {
            let mixes = mixes.view_subrange(0, vec![24, 1]);
            encode_mhc4_controls_rows(
                &ctx,
                &enc,
                1,
                1e-6,
                &mixes,
                &scale,
                &base,
                pre,
                post,
                &comb.view_subrange(0, vec![4, 4, 1]),
            )
            .map(|()| String::new())
            .unwrap_or_else(|e| e.to_string())
        };
        let (pre_rows, post_rows) = (
            pre.view_subrange(0, vec![4, 1]),
            post.view_subrange(0, vec![4, 1]),
        );
        refused(rows(&pre_rows, &pre_rows), "post");
        assert_eq!(rows(&pre_rows, &post_rows), "");

        // Collapse: the output is disjoint from the residual and pre.
        let first_stream = residual.view_subrange(0, vec![H as u64]);
        let err = encode_mhc4_collapse(&ctx, &enc, H, &residual, &pre, &first_stream).unwrap_err();
        refused(err.to_string(), "residual");
        let err = encode_mhc4_collapse_rows(
            &ctx,
            &enc,
            H,
            1,
            &residual.view_subrange(0, vec![H as u64, 4, 1]),
            &pre_rows,
            &first_stream.view_subrange(0, vec![H as u64, 1]),
        )
        .unwrap_err();
        refused(err.to_string(), "residual");
    }

    /// Rows variants equal the single-row encoders row by row (5 rows, GLM
    /// width, asymmetric streams).
    #[test]
    fn mhc4_rows_match_single_row_encoders() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const H: usize = 4096;
        const R: usize = 5;
        let n = H * MHC_STREAMS;
        let residual: Vec<f32> = (0..n * R)
            .map(|i| ((i * 13 + i / 4096 * 7) % 61) as f32 * 0.021 - 0.6)
            .collect();
        let mixes: Vec<f32> = (0..MHC_MIXES * R)
            .map(|i| ((i * 11) % 23) as f32 * 0.05 - 0.5)
            .collect();
        let scale = [0.8f32, -0.45, 1.1];
        let base: Vec<f32> = (0..MHC_MIXES)
            .map(|i| ((i * 5 + 2) % 17) as f32 * 0.04 - 0.3)
            .collect();
        let block: Vec<f32> = (0..H * R)
            .map(|d| ((d % 23) as f32 - 11.0) * 0.07)
            .collect();
        let r = R as u64;
        let h = H as u64;
        let residual_t = tensor(&ctx, &residual, vec![h, 4, r]);
        let mixes_t = tensor(&ctx, &mixes, vec![24, r]);
        let scale_t = tensor(&ctx, &scale, vec![3]);
        let base_t = tensor(&ctx, &base, vec![24]);
        let pre_t = tensor(&ctx, &[0.0; 4 * R], vec![4, r]);
        let post_t = tensor(&ctx, &[0.0; 4 * R], vec![4, r]);
        let comb_t = tensor(&ctx, &vec![0.0; 16 * R], vec![4, 4, r]);
        let collapsed_t = tensor(&ctx, &vec![0.0; H * R], vec![h, r]);
        let block_t = tensor(&ctx, &block, vec![h, r]);
        let out_t = tensor(&ctx, &vec![0.0; n * R], vec![h, 4, r]);
        let repeated_t = tensor(&ctx, &vec![0.0; n * R], vec![h, 4, r]);
        run(&ctx, |enc| {
            encode_mhc4_controls_rows(
                &ctx, enc, R, 1e-6, &mixes_t, &scale_t, &base_t, &pre_t, &post_t, &comb_t,
            )
            .unwrap();
            encode_mhc4_collapse_rows(&ctx, enc, H, R, &residual_t, &pre_t, &collapsed_t).unwrap();
            encode_mhc4_post_rows(
                &ctx,
                enc,
                H,
                R,
                &block_t,
                &residual_t,
                &post_t,
                &comb_t,
                &out_t,
            )
            .unwrap();
            encode_mhc4_repeat_rows(&ctx, enc, H, R, &block_t, &repeated_t).unwrap();
        });
        let (pre, post, comb) = (
            tensor_f32_at_offset(&pre_t),
            tensor_f32_at_offset(&post_t),
            tensor_f32_at_offset(&comb_t),
        );
        let (collapsed, out, repeated) = (
            tensor_f32_at_offset(&collapsed_t),
            tensor_f32_at_offset(&out_t),
            tensor_f32_at_offset(&repeated_t),
        );
        for row in 0..R {
            let one =
                |values: &[f32], width: usize| values[row * width..(row + 1) * width].to_vec();
            let res_t = tensor(&ctx, &one(&residual, n), vec![h, 4]);
            let mix_t = tensor(&ctx, &one(&mixes, 24), vec![24]);
            let p_t = tensor(&ctx, &[0.0; 4], vec![4]);
            let q_t = tensor(&ctx, &[0.0; 4], vec![4]);
            let c_t = tensor(&ctx, &[0.0; 16], vec![4, 4]);
            let col_t = tensor(&ctx, &vec![0.0; H], vec![h]);
            let blk_t = tensor(&ctx, &one(&block, H), vec![h]);
            let o_t = tensor(&ctx, &vec![0.0; n], vec![h, 4]);
            let rep_t = tensor(&ctx, &vec![0.0; n], vec![h, 4]);
            run(&ctx, |enc| {
                encode_mhc4_controls(&ctx, enc, 1e-6, &mix_t, &scale_t, &base_t, &p_t, &q_t, &c_t)
                    .unwrap();
                encode_mhc4_collapse(&ctx, enc, H, &res_t, &p_t, &col_t).unwrap();
                encode_mhc4_post(&ctx, enc, H, &blk_t, &res_t, &q_t, &c_t, &o_t).unwrap();
                encode_mhc4_repeat(&ctx, enc, H, &blk_t, &rep_t).unwrap();
            });
            assert_eq!(one(&pre, 4), tensor_f32_at_offset(&p_t), "row {row} pre");
            assert_eq!(one(&post, 4), tensor_f32_at_offset(&q_t), "row {row} post");
            assert_eq!(one(&comb, 16), tensor_f32_at_offset(&c_t), "row {row} comb");
            assert_eq!(
                one(&collapsed, H),
                tensor_f32_at_offset(&col_t),
                "row {row} collapse"
            );
            assert_eq!(
                one(&out, n),
                tensor_f32_at_offset(&o_t),
                "row {row} post residual"
            );
            assert_eq!(
                one(&repeated, n),
                tensor_f32_at_offset(&rep_t),
                "row {row} repeat"
            );
        }
    }

    /// Timing screen (not qualification): GPU time of the single-token mHC
    /// pre sequence at GLM-5.3 width (hidden 4096, Q8_0 mix [16384 -> 24]),
    /// as a chain of 90 dependent repetitions per command (one decode step's
    /// worth of pre sub-blocks), whole and per dispatch. Synthetic weights.
    #[test]
    #[ignore = "timing screen; run without MTL_DEBUG_LAYER"]
    fn mhc4_pre_dispatch_costs() {
        assert!(
            std::env::var_os("MTL_DEBUG_LAYER").is_none(),
            "timing runs must not enable MTL_DEBUG_LAYER"
        );
        let _lease = crate::metal::acquire_metal_benchmark_lease().expect("GPU lease");
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const H: usize = 4096;
        const REPS: usize = 90;
        let n = H * MHC_STREAMS;
        let residual: Vec<f32> = (0..n)
            .map(|i| ((i * 7 + 3) % 29) as f32 * 0.013 - 0.17)
            .collect();
        let (mix_bytes, _) = super::super::test_support::synthetic_q8_0_bank(n, MHC_MIXES);
        let mix_t = offset_tensor(
            &ctx,
            0,
            &mix_bytes,
            0,
            vec![n as u64, MHC_MIXES as u64],
            GgmlType::Q8_0,
        );
        let residual_t = tensor(&ctx, &residual, vec![H as u64, 4]);
        let ones_t = tensor(&ctx, &vec![1.0; n], vec![n as u64]);
        let normalized_t = tensor(&ctx, &vec![0.0; n], vec![n as u64]);
        let mixes_t = tensor(&ctx, &[0.0; MHC_MIXES], vec![MHC_MIXES as u64]);
        let scale_t = tensor(&ctx, &[0.8, -0.45, 1.1], vec![3]);
        let base_t = tensor(&ctx, &[0.1; MHC_MIXES], vec![MHC_MIXES as u64]);
        let pre_t = tensor(&ctx, &[0.0; 4], vec![4]);
        let post_t = tensor(&ctx, &[0.0; 4], vec![4]);
        let comb_t = tensor(&ctx, &[0.0; 16], vec![4, 4]);
        let collapsed_t = tensor(&ctx, &vec![0.0; H], vec![H as u64]);
        let norm_w_t = tensor(&ctx, &vec![1.0; H], vec![H as u64]);
        let normed_t = tensor(&ctx, &vec![0.0; H], vec![H as u64]);
        let (rms_eps, hc_eps) = (1e-5f32, 1e-6f32);
        let rms = |enc: &KernelEncoder| {
            crate::metal::encode_rms_norm_mul_f32(
                &ctx,
                enc,
                &residual_t,
                &ones_t,
                &normalized_t,
                rms_eps,
            )
            .unwrap()
        };
        let mix = |enc: &KernelEncoder| {
            crate::metal::encode_mat_vec_q8_0_f32(
                &ctx,
                enc,
                &mix_t,
                &normalized_t,
                &mixes_t,
                n,
                MHC_MIXES,
            )
            .unwrap()
        };
        let controls = |enc: &KernelEncoder| {
            encode_mhc4_controls(
                &ctx, enc, hc_eps, &mixes_t, &scale_t, &base_t, &pre_t, &post_t, &comb_t,
            )
            .unwrap()
        };
        let collapse = |enc: &KernelEncoder| {
            encode_mhc4_collapse(&ctx, enc, H, &residual_t, &pre_t, &collapsed_t).unwrap()
        };
        let block_norm = |enc: &KernelEncoder| {
            crate::metal::encode_rms_norm_mul_f32(
                &ctx,
                enc,
                &collapsed_t,
                &norm_w_t,
                &normed_t,
                rms_eps,
            )
            .unwrap()
        };
        let time = |encode: &dyn Fn(&KernelEncoder)| -> f64 {
            let mut samples: Vec<f64> = (0..8)
                .map(|_| {
                    let command = ctx.queue.commandBuffer().expect("command buffer");
                    let enc = KernelEncoder::begin(&command);
                    for _ in 0..REPS {
                        encode(&enc);
                    }
                    enc.end();
                    command.commit();
                    wait_completed(&command).expect("command buffer failed");
                    (command.GPUEndTime() - command.GPUStartTime()) * 1e6 / REPS as f64
                })
                .skip(1)
                .collect();
            samples.sort_by(f64::total_cmp);
            samples[samples.len() / 2]
        };
        type Encode<'a> = &'a dyn Fn(&KernelEncoder);
        let rows: [(&str, Encode<'_>); 6] = [
            ("whole pre (5 dispatches)", &|enc: &KernelEncoder| {
                rms(enc);
                mix(enc);
                controls(enc);
                collapse(enc);
                block_norm(enc);
            }),
            ("rms 16384", &rms),
            ("mix q8_0 16384->24", &mix),
            ("controls (1 thread)", &controls),
            ("collapse 4096", &collapse),
            ("block rms 4096", &block_norm),
        ];
        for (label, encode) in rows {
            eprintln!("mhc4 pre: {label:<26} {:7.2} us/rep", time(encode));
        }
    }
}
