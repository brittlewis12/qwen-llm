//! Test-only bounded FP32 ReplaySSM leaf. No runtime/session admission or knobs.
//! The experiment admits L=1 (control) and L=8 only, not a product policy.
//! Checkpoint and materialized matrices both use ordinary KDA's F32 layout.
//! Historical updates cache the computed u; replay never recomputes old dots.

use super::*;
use crate::metal::checks::bad_shape as bad;

pub(crate) const REPLAY_KERNEL: &str = "kernel_research_glm53_kda_replay_f32";
pub(crate) const MATERIALIZE_KERNEL: &str = "kernel_research_glm53_kda_replay_materialize_f32";

/// Separate F32 `[128, window, heads]` arrays, memory `[head][slot][channel]`.
/// k is normalized key, d is per-key decay, u is the computed value update.
/// Only slots `0..residue` are live; inactive slots need no initialization.
/// At L8 this adds 12 KiB/head to the existing 64 KiB/head FP32 checkpoint.
pub(crate) struct ReplayRecords<'a> {
    pub k: &'a MetalTensor,
    pub d: &'a MetalTensor,
    pub u: &'a MetalTensor,
}

/// Explicit driver-owned state: pending update count before this encode.
/// After a successful decode submission use `(residue + 1) % window`.
/// A flush at `residue == window-1` writes the checkpoint and leaves records
/// untouched; otherwise only slot `residue` is appended. Never advance this
/// after failed validation or treat it as a device-maintained cursor.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReplayPosition {
    pub window: usize,
    pub residue: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    n_heads: u32,
    n_rows: u32,
    lower_bound: f32,
    norm_eps: f32,
    window: u32,
    residue: u32,
}

fn geometry(
    kernel: &'static str,
    heads: usize,
    p: ReplayPosition,
) -> Result<(u64, Args), MetalError> {
    if heads == 0 || !matches!(p.window, 1 | 8) || p.residue >= p.window {
        return Err(bad(
            kernel,
            "positive heads, experimental window 1 or 8, residue < window required",
        ));
    }
    let width = heads
        .checked_mul(KDA_HEAD_DIM)
        .filter(|&n| u32::try_from(n).is_ok())
        .ok_or_else(|| bad(kernel, "projection width exceeds u32"))?;
    // Check byte extents before check_tensor (which assumes valid shapes).
    for factors in [
        KDA_HEAD_DIM * KDA_HEAD_DIM,
        9 * KDA_HEAD_DIM,
        4 * KDA_HEAD_DIM,
        p.window * KDA_HEAD_DIM,
    ] {
        heads
            .checked_mul(factors)
            .and_then(|n| n.checked_mul(4))
            .ok_or_else(|| bad(kernel, "binding byte extent overflow"))?;
    }
    Ok((
        width as u64,
        Args {
            n_heads: heads as u32,
            n_rows: 1,
            lower_bound: 0.0,
            norm_eps: 0.0,
            window: p.window as u32,
            residue: p.residue as u32,
        },
    ))
}

fn binding(
    kernel: &'static str,
    ctx: &MetalContext,
    tensor: &MetalTensor,
    shape: &[u64],
    writable: bool,
    alignment: u64,
    name: &str,
) -> Result<(), MetalError> {
    check_tensor(kernel, tensor, GgmlType::F32, shape, writable, name)?;
    check_alignment(kernel, tensor, alignment, name)?;
    if tensor.buffer.device().registryID() != ctx.device.registryID() {
        return Err(bad(kernel, format!("{name} belongs to a different device")));
    }
    Ok(())
}

fn encoder(
    kernel: &'static str,
    ctx: &MetalContext,
    enc: &KernelEncoder,
) -> Result<(), MetalError> {
    require_serial(kernel, enc)?;
    if enc.parent_command_buffer().device().registryID() != ctx.device.registryID() {
        return Err(bad(kernel, "encoder belongs to a different device"));
    }
    Ok(())
}

fn records(
    kernel: &'static str,
    ctx: &MetalContext,
    heads: usize,
    p: ReplayPosition,
    r: &ReplayRecords<'_>,
    writable: bool,
) -> Result<(), MetalError> {
    for (t, name) in [(r.k, "record k"), (r.d, "record d"), (r.u, "record u")] {
        binding(
            kernel,
            ctx,
            t,
            &[128, p.window as u64, heads as u64],
            writable,
            16,
            name,
        )?;
    }
    check_disjoint(kernel, r.k, &[(r.d, "record d"), (r.u, "record u")])?;
    check_disjoint(kernel, r.d, &[(r.u, "record u")])
}

fn pipeline(ctx: &MetalContext, name: &'static str, tgm: usize) -> Result<Pipeline, MetalError> {
    // ctx.pipeline resolves product first, then the lazy research metallib.
    // There is no separate public research_pipeline API.
    let pso = ctx.pipeline(name)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < THREADS {
        return Err(bad(name, "requires four 32-lane simdgroups"));
    }
    let needed = pso
        .staticThreadgroupMemoryLength()
        .checked_add(tgm)
        .ok_or_else(|| bad(name, "threadgroup capacity overflow"))?;
    if needed > ctx.device.maxThreadgroupMemoryLength() {
        return Err(bad(
            name,
            format!("requires {needed} bytes of threadgroup memory"),
        ));
    }
    Ok(pso)
}

fn launch(enc: &KernelEncoder, heads: usize, tgm: usize) {
    enc.set_threadgroup_memory(0, tgm);
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
}

/// One token, no allocations or submission. `b.state` is the FP32 checkpoint;
/// all other KdaDecode shapes/arithmetic are ordinary single-token KDA. Conv
/// state always updates in place. Checkpoint only updates on flush. Outputs
/// and cached u expose the actual recurrent result, not a reconstructed dot.
/// TGM = 2624 + 3*window*128*4 bytes (14912 at L8).
/// Every check, including pipeline capacity, precedes any encoded dispatch.
pub(crate) fn encode_kda_replay_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    p: ReplayPosition,
    b: &KdaDecode<'_>,
    r: &ReplayRecords<'_>,
    lower_bound: f32,
    norm_eps: f32,
) -> Result<(), MetalError> {
    const K: &str = REPLAY_KERNEL;
    encoder(K, ctx, enc)?;
    let (width, mut args) = geometry(K, heads, p)?;
    if !lower_bound.is_finite() || lower_bound >= 0.0 || !norm_eps.is_finite() || norm_eps <= 0.0 {
        return Err(bad(
            K,
            "lower bound must be finite negative; norm epsilon finite positive",
        ));
    }
    args.lower_bound = lower_bound;
    args.norm_eps = norm_eps;
    for (t, name) in [
        (b.q, "q"),
        (b.k, "k"),
        (b.v, "v"),
        (b.raw_gate, "raw gate"),
        (b.output_gate, "output gate"),
        (b.dt_bias, "dt bias"),
    ] {
        binding(K, ctx, t, &[width], false, 4, name)?;
    }
    for (t, name) in [(b.raw_beta, "raw beta"), (b.neg_exp_a_log, "ssm_a")] {
        binding(K, ctx, t, &[heads as u64], false, 4, name)?;
    }
    for (t, name) in [
        (b.q_conv, "q conv"),
        (b.k_conv, "k conv"),
        (b.v_conv, "v conv"),
    ] {
        binding(K, ctx, t, &[4, 1, width], false, 4, name)?;
    }
    binding(K, ctx, b.output_norm, &[128], false, 4, "output norm")?;
    binding(K, ctx, b.conv_state, &[width, 3, 3], true, 4, "conv state")?;
    binding(
        K,
        ctx,
        b.state,
        &[128, 128, heads as u64],
        true,
        16,
        "checkpoint",
    )?;
    binding(K, ctx, b.out, &[width], true, 4, "output")?;
    records(K, ctx, heads, p, r, true)?;
    let read = [
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
    let written = [
        (b.conv_state, "conv state"),
        (b.state, "checkpoint"),
        (b.out, "output"),
        (r.k, "record k"),
        (r.d, "record d"),
        (r.u, "record u"),
    ];
    for (i, &(t, name)) in written.iter().enumerate() {
        check_disjoint(K, t, &read).map_err(|e| bad(K, format!("{name}: {e}")))?;
        check_disjoint(K, t, &written[i + 1..]).map_err(|e| bad(K, format!("{name}: {e}")))?;
    }
    let tgm = (SCRATCH_FLOATS + 3 * p.window * KDA_HEAD_DIM) * 4;
    let pso = pipeline(ctx, K, tgm)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    for (i, t) in [
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
        r.k,
        r.d,
        r.u,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(i + 1, t);
    }
    launch(enc, heads, tgm);
    Ok(())
}

/// Read-only chronological replay to separate F32 `[128,128,heads]` output.
/// `residue=0` copies checkpoint. Active record slots must have been produced
/// by successful steps since that checkpoint. Does not touch conv state.
pub(crate) fn encode_kda_replay_materialize_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    p: ReplayPosition,
    checkpoint: &MetalTensor,
    r: &ReplayRecords<'_>,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = MATERIALIZE_KERNEL;
    encoder(K, ctx, enc)?;
    let (_, args) = geometry(K, heads, p)?;
    let shape = [128, 128, heads as u64];
    binding(K, ctx, checkpoint, &shape, false, 16, "checkpoint")?;
    binding(K, ctx, output, &shape, true, 16, "materialized state")?;
    records(K, ctx, heads, p, r, false)?;
    check_disjoint(
        K,
        checkpoint,
        &[(r.k, "record k"), (r.d, "record d"), (r.u, "record u")],
    )?;
    check_disjoint(
        K,
        output,
        &[
            (checkpoint, "checkpoint"),
            (r.k, "record k"),
            (r.d, "record d"),
            (r.u, "record u"),
        ],
    )?;
    let tgm = 3 * p.window * KDA_HEAD_DIM * 4;
    let pso = pipeline(ctx, K, tgm)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    for (i, t) in [checkpoint, r.k, r.d, r.u, output].into_iter().enumerate() {
        enc.set_tensor(i + 1, t);
    }
    launch(enc, heads, tgm);
    Ok(())
}

#[cfg(test)]
mod tests;
