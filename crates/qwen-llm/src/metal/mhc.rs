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
    bad_shape as bad, check_alignment, check_disjoint, check_tensor, overlaps, require_serial,
    same_range,
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

/// Flattened residual values per partial of [`encode_mhc4_pre_q8_0`].
pub const MHC4_PRE_CHUNK: usize = 256;

/// Partials per row of [`encode_mhc4_pre_q8_0`] at `hidden`.
/// Saturates on absurd widths; [`encode_mhc4_pre_q8_0`] validates the
/// geometry (width a multiple of [`MHC4_PRE_CHUNK`], 32-bit offsets).
pub fn mhc4_pre_chunks(hidden: usize) -> usize {
    hidden.saturating_mul(MHC_STREAMS).div_ceil(MHC4_PRE_CHUNK)
}

/// Scratch of [`encode_mhc4_pre_q8_0`]: F32 `[24, chunks, rows]` partial
/// dots and F32 `[chunks, rows]` partial sums of squares.
pub struct Mhc4PrePartials<'a> {
    pub dots: &'a MetalTensor,
    pub sumsq: &'a MetalTensor,
}

/// Outputs of [`encode_mhc4_pre_q8_0`], all F32 per row: mixes `[24]`, pre
/// and post `[4]`, combination `[4, 4]`, collapsed and normed `[hidden]`.
pub struct Mhc4PreOutputs<'a> {
    pub mixes: &'a MetalTensor,
    pub pre: &'a MetalTensor,
    pub post: &'a MetalTensor,
    pub comb: &'a MetalTensor,
    pub collapsed: &'a MetalTensor,
    pub normed: &'a MetalTensor,
}

/// Inputs of [`encode_mhc4_pre_q8_0`]: residual `[hidden, 4, rows]`, Q8_0
/// mix `[4 * hidden, 24]`, F32 scale `[3]` and base `[24]`, and the block's
/// F32 norm weight `[hidden]`.
pub struct Mhc4PreInputs<'a> {
    pub residual: &'a MetalTensor,
    pub mix: &'a MetalTensor,
    pub scale: &'a MetalTensor,
    pub base: &'a MetalTensor,
    pub norm_weight: &'a MetalTensor,
}

/// Fused single-token-or-rows mHC pre with a Q8_0 mix projection: per row,
/// `mixes = (mix . residual) * rsqrt(mean(residual^2) + hc_rms_eps)`, the
/// controls of [`encode_mhc4_controls`], the collapse of
/// [`encode_mhc4_collapse`] and the block RMSNorm with `norm_weight`. Two
/// dispatches: split-K partials per (256-value chunk, row), then one
/// 1024-thread threadgroup per row. Rows are independent, so a row's result
/// does not depend on how many rows share the dispatch (decode is rows = 1).
/// The RMS scale multiplies the reduced dots instead of the residual, so
/// the mixes differ from rms_norm + mat_vec numerically, not bitwise. The
/// controls repeat [`encode_mhc4_controls`]' expressions in registers,
/// which under fast math is numerical too (up to 35 ulps in the
/// combination). Given the same gates, the collapse and block norm equal
/// [`encode_mhc4_collapse`] and a 1024-thread `encode_rms_norm_mul_f32`
/// bitwise.
#[allow(clippy::too_many_arguments)]
pub fn encode_mhc4_pre_q8_0(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hidden: usize,
    rows: usize,
    eps: Mhc4PreEps,
    inputs: &Mhc4PreInputs<'_>,
    partials: &Mhc4PrePartials<'_>,
    outputs: &Mhc4PreOutputs<'_>,
) -> Result<(), MetalError> {
    const K: &str = "mhc4_pre_q8_0";
    require_serial(K, enc)?;
    for (value, name) in [
        (eps.hc_rms, "hyper-connection RMS epsilon"),
        (eps.hc, "hyper-connection epsilon"),
        (eps.norm, "norm epsilon"),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(bad(
                K,
                format!("{name} must be finite and positive, got {value}"),
            ));
        }
    }
    let (h, width) = residual_geometry(K, hidden)?;
    if !width.is_multiple_of(MHC4_PRE_CHUNK) {
        return Err(bad(
            K,
            format!("flattened width {width} is not a multiple of {MHC4_PRE_CHUNK}"),
        ));
    }
    let n = rows_u32(K, rows)?;
    let chunks = mhc4_pre_chunks(hidden);
    let r = rows as u64;
    if (chunks as u64)
        .checked_mul(r)
        .and_then(|u| u.checked_mul(MHC_MIXES as u64))
        .is_none_or(|v| u32::try_from(v).is_err())
    {
        return Err(bad(K, "partials exceed 32-bit shader offsets"));
    }
    if (width as u64)
        .checked_mul(r)
        .is_none_or(|v| u32::try_from(v).is_err())
    {
        return Err(bad(K, "residual rows exceed 32-bit shader offsets"));
    }
    check_f32(
        K,
        inputs.residual,
        &[hidden as u64, 4, r],
        false,
        "residual",
    )?;
    check_tensor(
        K,
        inputs.mix,
        GgmlType::Q8_0,
        &[width as u64, MHC_MIXES as u64],
        false,
        "mix",
    )?;
    check_f32(K, inputs.scale, &[3], false, "scale")?;
    check_f32(K, inputs.base, &[MHC_MIXES as u64], false, "base")?;
    check_f32(
        K,
        inputs.norm_weight,
        &[hidden as u64],
        false,
        "norm weight",
    )?;
    check_f32(
        K,
        partials.dots,
        &[MHC_MIXES as u64, chunks as u64, r],
        true,
        "partial dots",
    )?;
    check_f32(K, partials.sumsq, &[chunks as u64, r], true, "partial sums")?;
    check_f32(K, outputs.mixes, &[MHC_MIXES as u64, r], true, "mixes")?;
    check_f32(K, outputs.pre, &[4, r], true, "pre")?;
    check_f32(K, outputs.post, &[4, r], true, "post")?;
    check_f32(K, outputs.comb, &[4, 4, r], true, "combination")?;
    check_f32(K, outputs.collapsed, &[hidden as u64, r], true, "collapsed")?;
    check_f32(K, outputs.normed, &[hidden as u64, r], true, "normed")?;
    check_alignment(K, inputs.residual, 16, "residual")?;
    let written = [
        (partials.dots, "partial dots"),
        (partials.sumsq, "partial sums"),
        (outputs.mixes, "mixes"),
        (outputs.pre, "pre"),
        (outputs.post, "post"),
        (outputs.comb, "combination"),
        (outputs.collapsed, "collapsed"),
        (outputs.normed, "normed"),
    ];
    let read = [
        (inputs.residual, "residual"),
        (inputs.mix, "mix"),
        (inputs.scale, "scale"),
        (inputs.base, "base"),
        (inputs.norm_weight, "norm weight"),
    ];
    for (i, &(tensor, name)) in written.iter().enumerate() {
        check_disjoint(K, tensor, &read).map_err(|e| bad(K, format!("{name}: {e}")))?;
        check_disjoint(K, tensor, &written[i + 1..]).map_err(|e| bad(K, format!("{name}: {e}")))?;
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        hidden: u32,
        rows: u32,
        chunks: u32,
        hc_rms_eps: f32,
        hc_eps: f32,
        norm_eps: f32,
    }
    let args = Args {
        hidden: h,
        rows: n,
        chunks: u32::try_from(chunks).map_err(|_| bad(K, "chunks exceed u32"))?,
        hc_rms_eps: eps.hc_rms,
        hc_eps: eps.hc,
        norm_eps: eps.norm,
    };
    let partial = ctx.pipeline("kernel_mhc4_pre_mix_partial_q8_0")?;
    let finish = ctx.pipeline("kernel_mhc4_pre_finish")?;
    if partial.threadExecutionWidth() != 32 || partial.maxTotalThreadsPerThreadgroup() < 256 {
        return Err(bad(K, "the partial pass needs 8 simdgroups of 32 lanes"));
    }
    if finish.threadExecutionWidth() != 32 || finish.maxTotalThreadsPerThreadgroup() < 1024 {
        return Err(bad(K, "the finish pass needs 32 simdgroups of 32 lanes"));
    }
    enc.set_pipeline(&partial);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, inputs.residual);
    enc.set_tensor(2, inputs.mix);
    enc.set_tensor(3, partials.dots);
    enc.set_tensor(4, partials.sumsq);
    enc.dispatch(
        MTLSize {
            width: chunks,
            height: rows,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    enc.set_pipeline(&finish);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, inputs.residual);
    enc.set_tensor(2, partials.dots);
    enc.set_tensor(3, partials.sumsq);
    enc.set_tensor(4, inputs.scale);
    enc.set_tensor(5, inputs.base);
    enc.set_tensor(6, inputs.norm_weight);
    enc.set_tensor(7, outputs.mixes);
    enc.set_tensor(8, outputs.pre);
    enc.set_tensor(9, outputs.post);
    enc.set_tensor(10, outputs.comb);
    enc.set_tensor(11, outputs.collapsed);
    enc.set_tensor(12, outputs.normed);
    enc.set_threadgroup_memory(0, 96 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: rows,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 1024,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Epsilons of [`encode_mhc4_pre_q8_0`]: the flattened residual's RMS, the
/// controls (Sinkhorn) and the block RMSNorm.
#[derive(Clone, Copy, Debug)]
pub struct Mhc4PreEps {
    pub hc_rms: f32,
    pub hc: f32,
    pub norm: f32,
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

    /// The single-token and per-row controls kernels agree bitwise over 512
    /// rows of wide random mixes, scales and biases: decode and packed
    /// prefill share the Sinkhorn controls lineage (Exact packed rows equal
    /// serial decode).
    #[test]
    fn mhc4_controls_single_and_rows_agree_bitwise() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 512;
        let mut state = 0x6c8e_9cf5u32;
        let mut noise = |scale: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
        };
        let mixes: Vec<f32> = (0..ROWS * MHC_MIXES).map(|_| noise(6.0)).collect();
        let scale = [noise(2.0), noise(2.0), noise(2.0)];
        let base: Vec<f32> = (0..MHC_MIXES).map(|_| noise(3.0)).collect();
        let eps = 1e-6;
        let mixes_t = tensor(&ctx, &mixes, vec![MHC_MIXES as u64, ROWS as u64]);
        let scale_t = tensor(&ctx, &scale, vec![3]);
        let base_t = tensor(&ctx, &base, vec![MHC_MIXES as u64]);
        let outputs = || {
            (
                tensor(&ctx, &vec![7.0; 4 * ROWS], vec![4, ROWS as u64]),
                tensor(&ctx, &vec![7.0; 4 * ROWS], vec![4, ROWS as u64]),
                tensor(&ctx, &vec![7.0; 16 * ROWS], vec![4, 4, ROWS as u64]),
            )
        };
        let (rows_pre, rows_post, rows_comb) = outputs();
        let (one_pre, one_post, one_comb) = outputs();
        run(&ctx, |enc| {
            encode_mhc4_controls_rows(
                &ctx, enc, ROWS, eps, &mixes_t, &scale_t, &base_t, &rows_pre, &rows_post,
                &rows_comb,
            )
            .unwrap();
            for row in 0..ROWS as u64 {
                let view = |t: &MetalTensor, width: u64, shape: Vec<u64>| {
                    t.view_subrange(row * width, shape)
                };
                encode_mhc4_controls(
                    &ctx,
                    enc,
                    eps,
                    &view(&mixes_t, MHC_MIXES as u64, vec![MHC_MIXES as u64]),
                    &scale_t,
                    &base_t,
                    &view(&one_pre, 4, vec![4]),
                    &view(&one_post, 4, vec![4]),
                    &view(&one_comb, 16, vec![4, 4]),
                )
                .unwrap();
            }
        });
        let bits = |t: &MetalTensor| -> Vec<u32> {
            tensor_f32_at_offset(t)
                .iter()
                .map(|v| v.to_bits())
                .collect()
        };
        for (label, rows, one) in [
            ("pre", &rows_pre, &one_pre),
            ("post", &rows_post, &one_post),
            ("combination", &rows_comb, &one_comb),
        ] {
            let (rows, one) = (bits(rows), bits(one));
            assert!(
                rows.iter().all(|&b| f32::from_bits(b).is_finite()),
                "{label}: non-finite"
            );
            assert_eq!(rows, one, "{label}: rows vs single");
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

    /// Fused Q8_0 pre at GLM width over three distinct rows, against the
    /// five-dispatch path (rms_norm, Q8_0 mat-vec, controls, collapse, block
    /// norm) and the CPU contract. Rows are dispatch-independent: each row
    /// of the 3-row dispatch equals its single-row dispatch bitwise. Given
    /// the unfused path's controls, the fused collapse and block norm keep
    /// their kernels' arithmetic, so those outputs match bitwise when the
    /// pre gates do.
    #[test]
    fn mhc4_pre_q8_0_matches_the_unfused_path_and_is_row_independent() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const H: usize = 4096;
        const ROWS: usize = 3;
        let n = H * MHC_STREAMS;
        let residual: Vec<f32> = (0..n * ROWS)
            .map(|i| {
                let (row, s, d) = (i / n, (i % n) / H, i % H);
                (s as f32 - 1.3) * (0.4 + 0.3 * row as f32)
                    + ((d * 7 + s * 3 + row * 11) % 29) as f32 * 0.013
                    - 0.17
            })
            .collect();
        let (mix_bytes, function) = super::super::test_support::synthetic_q8_0_bank(n, MHC_MIXES);
        let scale = [0.8f32, -0.45, 1.1];
        let base: Vec<f32> = (0..MHC_MIXES)
            .map(|i| ((i * 5 + 2) % 17) as f32 * 0.04 - 0.3)
            .collect();
        let norm_weight: Vec<f32> = (0..H).map(|d| 0.5 + (d % 13) as f32 * 0.07).collect();
        let eps = Mhc4PreEps {
            hc_rms: 1e-5,
            hc: 1e-6,
            norm: 1e-5,
        };
        let chunks = mhc4_pre_chunks(H);
        let mix_t = offset_tensor(
            &ctx,
            32,
            &mix_bytes,
            32,
            vec![n as u64, MHC_MIXES as u64],
            GgmlType::Q8_0,
        );
        let residual_t = tensor(&ctx, &residual, vec![H as u64, 4, ROWS as u64]);
        let scale_t = tensor(&ctx, &scale, vec![3]);
        let base_t = tensor(&ctx, &base, vec![MHC_MIXES as u64]);
        let weight_t = tensor(&ctx, &norm_weight, vec![H as u64]);
        let zeros = |shape: Vec<u64>| {
            let len = shape.iter().product::<u64>() as usize;
            tensor(&ctx, &vec![f32::NAN; len], shape)
        };
        struct Out {
            mixes: MetalTensor,
            pre: MetalTensor,
            post: MetalTensor,
            comb: MetalTensor,
            collapsed: MetalTensor,
            normed: MetalTensor,
        }
        let outputs = |rows: u64| Out {
            mixes: zeros(vec![MHC_MIXES as u64, rows]),
            pre: zeros(vec![4, rows]),
            post: zeros(vec![4, rows]),
            comb: zeros(vec![4, 4, rows]),
            collapsed: zeros(vec![H as u64, rows]),
            normed: zeros(vec![H as u64, rows]),
        };
        let dots_t = zeros(vec![MHC_MIXES as u64, chunks as u64, ROWS as u64]);
        let sums_t = zeros(vec![chunks as u64, ROWS as u64]);
        let fused = |out: &Out, residual: &MetalTensor, rows: usize, enc: &KernelEncoder| {
            let r = rows as u64;
            encode_mhc4_pre_q8_0(
                &ctx,
                enc,
                H,
                rows,
                eps,
                &Mhc4PreInputs {
                    residual,
                    mix: &mix_t,
                    scale: &scale_t,
                    base: &base_t,
                    norm_weight: &weight_t,
                },
                &Mhc4PrePartials {
                    dots: &dots_t.view_subrange(0, vec![MHC_MIXES as u64, chunks as u64, r]),
                    sumsq: &sums_t.view_subrange(0, vec![chunks as u64, r]),
                },
                &Mhc4PreOutputs {
                    mixes: &out.mixes,
                    pre: &out.pre,
                    post: &out.post,
                    comb: &out.comb,
                    collapsed: &out.collapsed,
                    normed: &out.normed,
                },
            )
            .unwrap();
        };
        let all = outputs(ROWS as u64);
        let singles: Vec<Out> = (0..ROWS).map(|_| outputs(1)).collect();
        run(&ctx, |enc| {
            fused(&all, &residual_t, ROWS, enc);
            for (row, out) in singles.iter().enumerate() {
                let view = residual_t.view_subrange((row * n) as u64, vec![H as u64, 4, 1]);
                fused(out, &view, 1, enc);
            }
        });
        let read = |t: &MetalTensor| tensor_f32_at_offset(t);
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for (row, single) in singles.iter().enumerate() {
            for (label, whole, one, width) in [
                ("mixes", &all.mixes, &single.mixes, MHC_MIXES),
                ("pre", &all.pre, &single.pre, 4),
                ("post", &all.post, &single.post, 4),
                ("combination", &all.comb, &single.comb, 16),
                ("collapsed", &all.collapsed, &single.collapsed, H),
                ("normed", &all.normed, &single.normed, H),
            ] {
                assert_eq!(
                    bits(&read(whole)[row * width..(row + 1) * width]),
                    bits(&read(one)),
                    "row {row} {label}: 3-row dispatch vs single"
                );
            }
        }

        for row in 0..ROWS {
            let r = &residual[row * n..(row + 1) * n];
            let cpu = crate::deepseek_v4_oracle::hyper_connection_pre(
                r,
                H,
                MHC_STREAMS,
                &function,
                &scale,
                &base,
                eps.hc_rms,
                MHC_SINKHORN_ITERATIONS,
                eps.hc,
            )
            .unwrap();
            let mean = cpu.input.iter().map(|v| (v * v) as f64).sum::<f64>() / H as f64;
            let inverse = 1.0 / (mean + eps.norm as f64).sqrt();
            let normed: Vec<f32> = cpu
                .input
                .iter()
                .zip(&norm_weight)
                .map(|(v, w)| (*v as f64 * inverse * *w as f64) as f32)
                .collect();
            let at =
                |t: &MetalTensor, width: usize| read(t)[row * width..(row + 1) * width].to_vec();
            // The CPU oracle accumulates 16,384 F32 products sequentially; both
            // GPU paths sit ~2e-5 (scaled) from it on the mixes.
            assert_close("fused mixes", &at(&all.mixes, MHC_MIXES), &cpu.mixes, 5e-5);
            assert_close("fused pre", &at(&all.pre, 4), &cpu.controls.pre, 1e-5);
            assert_close("fused post", &at(&all.post, 4), &cpu.controls.post, 1e-5);
            assert_close(
                "fused combination",
                &at(&all.comb, 16),
                &cpu.controls.combination,
                1e-5,
            );
            assert_close("fused collapsed", &at(&all.collapsed, H), &cpu.input, 1e-5);
            assert_close("fused normed", &at(&all.normed, H), &normed, 1e-4);
        }

        // The unfused path on row 1, then its block norm from the same gates.
        let row = 1;
        let view = residual_t.view_subrange((row * n) as u64, vec![H as u64, 4]);
        let ones_t = tensor(&ctx, &vec![1.0; n], vec![n as u64]);
        let normalized_t = zeros(vec![n as u64]);
        let unfused = outputs(1);
        let flat = |t: &MetalTensor, len: u64| t.view_subrange(0, vec![len]);
        run(&ctx, |enc| {
            crate::metal::encode_rms_norm_mul_f32(
                &ctx,
                enc,
                &view.view_subrange(0, vec![n as u64]),
                &ones_t,
                &normalized_t,
                eps.hc_rms,
            )
            .unwrap();
            crate::metal::encode_mat_vec_q8_0_f32(
                &ctx,
                enc,
                &mix_t,
                &normalized_t,
                &flat(&unfused.mixes, MHC_MIXES as u64),
                n,
                MHC_MIXES,
            )
            .unwrap();
            encode_mhc4_controls(
                &ctx,
                enc,
                eps.hc,
                &flat(&unfused.mixes, MHC_MIXES as u64),
                &scale_t,
                &base_t,
                &flat(&unfused.pre, 4),
                &flat(&unfused.post, 4),
                &unfused.comb.view_subrange(0, vec![4, 4]),
            )
            .unwrap();
            encode_mhc4_collapse(
                &ctx,
                enc,
                H,
                &view,
                &flat(&unfused.pre, 4),
                &flat(&unfused.collapsed, H as u64),
            )
            .unwrap();
            crate::metal::encode_rms_norm_mul_f32(
                &ctx,
                enc,
                &flat(&unfused.collapsed, H as u64),
                &weight_t,
                &flat(&unfused.normed, H as u64),
                eps.norm,
            )
            .unwrap();
        });
        let at = |t: &MetalTensor, width: usize| read(t)[row * width..(row + 1) * width].to_vec();
        for (label, fused_v, unfused_v, tolerance) in [
            (
                "mixes",
                at(&all.mixes, MHC_MIXES),
                read(&unfused.mixes),
                1e-5,
            ),
            ("pre", at(&all.pre, 4), read(&unfused.pre), 2e-6),
            ("post", at(&all.post, 4), read(&unfused.post), 2e-6),
            ("combination", at(&all.comb, 16), read(&unfused.comb), 2e-6),
            (
                "collapsed",
                at(&all.collapsed, H),
                read(&unfused.collapsed),
                2e-6,
            ),
            ("normed", at(&all.normed, H), read(&unfused.normed), 2e-6),
        ] {
            assert_close(
                &format!("fused vs unfused {label}"),
                &fused_v,
                &unfused_v,
                tolerance,
            );
        }

        // Given the fused gates, the standalone collapse and 1024-thread block
        // norm reproduce the fused collapse and norm bitwise.
        let gates_t = tensor(&ctx, &at(&all.pre, 4), vec![4]);
        let collapsed_t = zeros(vec![H as u64]);
        let normed_t = zeros(vec![H as u64]);
        let rms_threads = ctx
            .pipeline("kernel_rms_norm_mul_f32")
            .unwrap()
            .maxTotalThreadsPerThreadgroup()
            .min(1024);
        assert_eq!(
            rms_threads, 1024,
            "the bitwise claim is scoped to 1024 threads"
        );
        run(&ctx, |enc| {
            encode_mhc4_collapse(&ctx, enc, H, &view, &gates_t, &collapsed_t).unwrap();
            crate::metal::encode_rms_norm_mul_f32(
                &ctx,
                enc,
                &collapsed_t,
                &weight_t,
                &normed_t,
                eps.norm,
            )
            .unwrap();
        });
        assert_eq!(
            bits(&read(&collapsed_t)),
            bits(&at(&all.collapsed, H)),
            "collapse from the fused gates"
        );
        assert_eq!(
            bits(&read(&normed_t)),
            bits(&at(&all.normed, H)),
            "block norm from the fused collapse"
        );
    }

    /// Directed numerics of the fused Q8_0 pre against an independent f64
    /// reference (mix dots and RMS in f64, the CPU Sinkhorn on the reference
    /// mixes, collapse and block norm in f64 from the GPU gates). The Q8_0
    /// bank spans the full quant range (-128..=127) with signed scales.
    /// Widths of 1, 31, 32 and 33 chunks (hidden 64, 1984, 2048, 2112), and
    /// at hidden 2112 six rows in one dispatch: zero, near-epsilon,
    /// alternating (cancelling), wide-spread with spikes, random, and
    /// random again under saturating control scales and biases. Every
    /// output is finite.
    #[test]
    fn mhc4_pre_q8_0_directed_numerics_and_widths() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let mut state = 0x51ed_2701u32;
        let mut noise = move |scale: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
        };
        let eps = Mhc4PreEps {
            hc_rms: 1e-5,
            hc: 1e-6,
            norm: 1e-5,
        };
        struct Case {
            name: &'static str,
            hidden: usize,
            residuals: Vec<(&'static str, Vec<f32>)>,
            scale: [f32; 3],
            base: Vec<f32>,
        }
        let mut cases = Vec::new();
        for hidden in [64usize, 1984, 2048, 2112] {
            let n = hidden * MHC_STREAMS;
            let mut residuals = vec![("random", (0..n).map(|_| noise(2.0)).collect::<Vec<f32>>())];
            if hidden == 2112 {
                residuals.push(("zero", vec![0.0; n]));
                residuals.push(("near epsilon", (0..n).map(|_| noise(1e-4)).collect()));
                residuals.push((
                    "alternating",
                    (0..n)
                        .map(|i| if i % 2 == 0 { 3.0 } else { -3.0 })
                        .collect(),
                ));
                residuals.push((
                    "spread with spikes",
                    (0..n)
                        .map(|i| if i % 997 == 0 { 1.0e3 } else { noise(1e-3) })
                        .collect(),
                ));
            }
            let base: Vec<f32> = (0..MHC_MIXES).map(|_| noise(0.5)).collect();
            cases.push(Case {
                name: "plain",
                hidden,
                residuals,
                scale: [0.8, -0.45, 1.1],
                base,
            });
        }
        let n = 2112 * MHC_STREAMS;
        cases.push(Case {
            name: "saturating controls",
            hidden: 2112,
            residuals: vec![("random", (0..n).map(|_| noise(2.0)).collect())],
            scale: [4.0, -4.0, 3.0],
            base: (0..MHC_MIXES)
                .map(|i| if i % 2 == 0 { 6.0 } else { -6.0 })
                .collect(),
        });

        for case in &cases {
            let (h, rows) = (case.hidden, case.residuals.len());
            let n = h * MHC_STREAMS;
            let chunks = mhc4_pre_chunks(h);
            // Full-range Q8_0 bank: quants cycle through -128..=127, scales
            // are signed and vary per block.
            let blocks = n / 32;
            let mut bytes = Vec::with_capacity(MHC_MIXES * blocks * 34);
            let mut weights = vec![0.0f64; MHC_MIXES * n];
            for m in 0..MHC_MIXES {
                for b in 0..blocks {
                    let d = half::f16::from_f32(noise(2e-3));
                    bytes.extend_from_slice(&d.to_bits().to_le_bytes());
                    for l in 0..32 {
                        let q = (((m * 131 + b * 37 + l * 11) % 256) as i32 - 128) as i8;
                        bytes.push(q as u8);
                        weights[m * n + b * 32 + l] = d.to_f64() * f64::from(q);
                    }
                }
            }
            let residual: Vec<f32> = case.residuals.iter().flat_map(|(_, r)| r.clone()).collect();
            let norm_weight: Vec<f32> = (0..h).map(|_| 0.5 + noise(0.4).abs()).collect();
            let mix_t = offset_tensor(
                &ctx,
                36,
                &bytes,
                6,
                vec![n as u64, MHC_MIXES as u64],
                GgmlType::Q8_0,
            );
            let r = rows as u64;
            let residual_t = tensor(&ctx, &residual, vec![h as u64, 4, r]);
            let scale_t = tensor(&ctx, &case.scale, vec![3]);
            let base_t = tensor(&ctx, &case.base, vec![MHC_MIXES as u64]);
            let weight_t = tensor(&ctx, &norm_weight, vec![h as u64]);
            let nan = |shape: Vec<u64>| {
                let len = shape.iter().product::<u64>() as usize;
                tensor(&ctx, &vec![f32::NAN; len], shape)
            };
            let dots_t = nan(vec![MHC_MIXES as u64, chunks as u64, r]);
            let sums_t = nan(vec![chunks as u64, r]);
            let mixes_t = nan(vec![MHC_MIXES as u64, r]);
            let pre_t = nan(vec![4, r]);
            let post_t = nan(vec![4, r]);
            let comb_t = nan(vec![4, 4, r]);
            let collapsed_t = nan(vec![h as u64, r]);
            let normed_t = nan(vec![h as u64, r]);
            run(&ctx, |enc| {
                encode_mhc4_pre_q8_0(
                    &ctx,
                    enc,
                    h,
                    rows,
                    eps,
                    &Mhc4PreInputs {
                        residual: &residual_t,
                        mix: &mix_t,
                        scale: &scale_t,
                        base: &base_t,
                        norm_weight: &weight_t,
                    },
                    &Mhc4PrePartials {
                        dots: &dots_t,
                        sumsq: &sums_t,
                    },
                    &Mhc4PreOutputs {
                        mixes: &mixes_t,
                        pre: &pre_t,
                        post: &post_t,
                        comb: &comb_t,
                        collapsed: &collapsed_t,
                        normed: &normed_t,
                    },
                )
                .unwrap();
            });
            let (mixes, pre, post, comb, collapsed, normed) = (
                tensor_f32_at_offset(&mixes_t),
                tensor_f32_at_offset(&pre_t),
                tensor_f32_at_offset(&post_t),
                tensor_f32_at_offset(&comb_t),
                tensor_f32_at_offset(&collapsed_t),
                tensor_f32_at_offset(&normed_t),
            );
            for (label, values) in [
                ("mixes", &mixes),
                ("pre", &pre),
                ("post", &post),
                ("combination", &comb),
                ("collapsed", &collapsed),
                ("normed", &normed),
            ] {
                assert!(
                    values.iter().all(|v| v.is_finite()),
                    "{} hidden {h}: non-finite {label}",
                    case.name
                );
            }
            for (row, (kind, x)) in case.residuals.iter().enumerate() {
                let label = format!("{} hidden {h} {kind}", case.name);
                let sumsq: f64 = x.iter().map(|v| f64::from(*v).powi(2)).sum();
                let inverse = 1.0 / (sumsq / n as f64 + f64::from(eps.hc_rms)).sqrt();
                let mut reference = vec![0.0f32; MHC_MIXES];
                for (m, out) in reference.iter_mut().enumerate() {
                    let w = &weights[m * n..(m + 1) * n];
                    let dot: f64 = w.iter().zip(x).map(|(w, x)| w * f64::from(*x)).sum();
                    let magnitude: f64 = w
                        .iter()
                        .zip(x)
                        .map(|(w, x)| (w * f64::from(*x)).abs())
                        .sum();
                    *out = (dot * inverse) as f32;
                    let got = f64::from(mixes[row * MHC_MIXES + m]);
                    let bound = 2e-6 * magnitude * inverse + 1e-6;
                    assert!(
                        (got - dot * inverse).abs() <= bound,
                        "{label}: mix {m} {got} vs {} (bound {bound:e})",
                        dot * inverse
                    );
                }
                let controls = crate::deepseek_v4_oracle::split_sinkhorn(
                    &reference,
                    &case.scale,
                    &case.base,
                    MHC_STREAMS,
                    MHC_SINKHORN_ITERATIONS,
                    eps.hc,
                )
                .unwrap();
                for (what, got, want) in [
                    ("pre", &pre[row * 4..(row + 1) * 4], &controls.pre[..]),
                    ("post", &post[row * 4..(row + 1) * 4], &controls.post[..]),
                    (
                        "combination",
                        &comb[row * 16..(row + 1) * 16],
                        &controls.combination[..],
                    ),
                ] {
                    for (i, (g, w)) in got.iter().zip(want).enumerate() {
                        assert!((g - w).abs() <= 1e-4, "{label}: {what}[{i}] {g} vs {w}");
                    }
                }
                // Collapse and block norm in f64 from the GPU gates.
                let gates = &pre[row * 4..(row + 1) * 4];
                let want: Vec<f64> = (0..h)
                    .map(|d| {
                        (0..MHC_STREAMS)
                            .map(|s| f64::from(x[s * h + d]) * f64::from(gates[s]))
                            .sum()
                    })
                    .collect();
                let scale_c: f64 = (0..h)
                    .map(|d| {
                        (0..MHC_STREAMS)
                            .map(|s| f64::from(x[s * h + d]).abs())
                            .sum::<f64>()
                    })
                    .fold(0.0, f64::max)
                    * 2.0;
                for (d, w) in want.iter().enumerate() {
                    let g = f64::from(collapsed[row * h + d]);
                    assert!(
                        (g - w).abs() <= 1e-6 * scale_c + 1e-30,
                        "{label}: collapsed[{d}] {g} vs {w}"
                    );
                }
                let mean = want.iter().map(|v| v * v).sum::<f64>() / h as f64;
                let norm_inverse = 1.0 / (mean + f64::from(eps.norm)).sqrt();
                for (d, w) in want.iter().enumerate() {
                    let expected = w * norm_inverse * f64::from(norm_weight[d]);
                    let g = f64::from(normed[row * h + d]);
                    let bound = 1e-5 * (scale_c * norm_inverse) + 1e-30;
                    assert!(
                        (g - expected).abs() <= bound,
                        "{label}: normed[{d}] {g} vs {expected} (bound {bound:e})"
                    );
                }
            }
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
        let chunks = mhc4_pre_chunks(H) as u64;
        let dots_t = tensor(
            &ctx,
            &vec![0.0; MHC_MIXES * chunks as usize],
            vec![MHC_MIXES as u64, chunks, 1],
        );
        let sums_t = tensor(&ctx, &vec![0.0; chunks as usize], vec![chunks, 1]);
        let fused = |enc: &KernelEncoder| {
            encode_mhc4_pre_q8_0(
                &ctx,
                enc,
                H,
                1,
                Mhc4PreEps {
                    hc_rms: rms_eps,
                    hc: hc_eps,
                    norm: rms_eps,
                },
                &Mhc4PreInputs {
                    residual: &residual_t.view_subrange(0, vec![H as u64, 4, 1]),
                    mix: &mix_t,
                    scale: &scale_t,
                    base: &base_t,
                    norm_weight: &norm_w_t,
                },
                &Mhc4PrePartials {
                    dots: &dots_t,
                    sumsq: &sums_t,
                },
                &Mhc4PreOutputs {
                    mixes: &mixes_t.view_subrange(0, vec![MHC_MIXES as u64, 1]),
                    pre: &pre_t.view_subrange(0, vec![4, 1]),
                    post: &post_t.view_subrange(0, vec![4, 1]),
                    comb: &comb_t.view_subrange(0, vec![4, 4, 1]),
                    collapsed: &collapsed_t.view_subrange(0, vec![H as u64, 1]),
                    normed: &normed_t.view_subrange(0, vec![H as u64, 1]),
                },
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
        // Latency-bound chains read the GPU clock state: ramp it first.
        let warm = std::time::Instant::now();
        while warm.elapsed() < std::time::Duration::from_secs(2) {
            time(&|enc: &KernelEncoder| {
                rms(enc);
                mix(enc);
                controls(enc);
                collapse(enc);
                block_norm(enc);
            });
        }
        type Encode<'a> = &'a dyn Fn(&KernelEncoder);
        let rows: [(&str, Encode<'_>); 7] = [
            ("fused pre (2 dispatches)", &fused),
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
