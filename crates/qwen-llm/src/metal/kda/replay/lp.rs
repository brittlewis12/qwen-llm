//! Test-only physical F16 reads with either an F32 or a shared F16 anchor.
//! No runtime knobs, allocations, cursor, or product admission policy.
use super::*;

pub(crate) const LP_TGM_BYTES: usize = (SCRATCH_FLOATS + 3 * 8 * 128 + 8) * 4;
pub(crate) const LP_MATERIALIZE_KERNEL: &str = "kernel_research_glm53_kda_replay_materialize_f16";
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReplayLpMode {
    SeparateF32Anchor,
    SharedF16Anchor,
}
impl ReplayLpMode {
    pub(crate) fn kernel(self) -> &'static str {
        match self {
            Self::SeparateF32Anchor => "kernel_research_glm53_kda_replay_f16_separate",
            Self::SharedF16Anchor => "kernel_research_glm53_kda_replay_f16_shared",
        }
    }
    fn anchor_dtype(self) -> GgmlType {
        match self {
            Self::SeparateF32Anchor => GgmlType::F32,
            Self::SharedF16Anchor => GgmlType::F16,
        }
    }
}
/// Four F32 [128,8,heads] arrays; beta is F32 [8,heads]. All head-major.
/// v is raw post-convolution/SiLU, beta is sigmoid(raw_beta). Inactive slots
/// are uninitialized; only nonflush appends. Flush ignores cached LP u.
pub(crate) struct ReplayLpRecords<'a> {
    pub k: &'a MetalTensor,
    pub d: &'a MetalTensor,
    pub u: &'a MetalTensor,
    pub v: &'a MetalTensor,
    pub beta: &'a MetalTensor,
}
impl ReplayLpRecords<'_> {
    pub(crate) fn cached(&self) -> ReplayRecords<'_> {
        ReplayRecords {
            k: self.k,
            d: self.d,
            u: self.u,
        }
    }
}
fn typed_binding(
    k: &'static str,
    ctx: &MetalContext,
    t: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
    alignment: u64,
    name: &str,
) -> Result<(), MetalError> {
    check_tensor(k, t, dtype, shape, writable, name)?;
    check_alignment(k, t, alignment, name)?;
    if t.buffer.device().registryID() != ctx.device.registryID() {
        return Err(bad(k, format!("{name} belongs to a different device")));
    }
    Ok(())
}
/// One token, L8 only; driver advances residue after successful submission.
/// b.state = authoritative [128,128,heads] anchor: F32 separate, F16 shared.
/// read_state = physical F16 matrix of the same shape, initialized by caller
/// by rounding the anchor. Shared mode requires exactly the anchor's range;
/// all other writable ranges must be disjoint. Both matrices stay untouched
/// on nonflush. Flush outputs use F32 work before rounding, and independently
/// store anchor/read. Conv remains ordinary in-place F32. TGM = 14944 bytes.
/// Every fallible check, including device pipeline capacity, precedes encoding.
pub(crate) fn encode_kda_replay_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    p: ReplayPosition,
    b: &KdaDecode<'_>,
    mode: ReplayLpMode,
    read_state: &MetalTensor,
    r: &ReplayLpRecords<'_>,
    lower_bound: f32,
    norm_eps: f32,
) -> Result<(), MetalError> {
    let k = mode.kernel();
    encoder(k, ctx, enc)?;
    if p.window != 8 {
        return Err(bad(k, "physical-F16 experiment requires L8"));
    }
    let (width, mut args) = geometry(k, heads, p)?;
    if !lower_bound.is_finite() || lower_bound >= 0.0 || !norm_eps.is_finite() || norm_eps <= 0.0 {
        return Err(bad(
            k,
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
        binding(k, ctx, t, &[width], false, 4, name)?;
    }
    for (t, name) in [(b.raw_beta, "raw beta"), (b.neg_exp_a_log, "ssm_a")] {
        binding(k, ctx, t, &[heads as u64], false, 4, name)?;
    }
    for (t, name) in [
        (b.q_conv, "q conv"),
        (b.k_conv, "k conv"),
        (b.v_conv, "v conv"),
    ] {
        binding(k, ctx, t, &[4, 1, width], false, 4, name)?;
    }
    binding(k, ctx, b.output_norm, &[128], false, 4, "output norm")?;
    binding(k, ctx, b.conv_state, &[width, 3, 3], true, 4, "conv state")?;
    typed_binding(
        k,
        ctx,
        b.state,
        mode.anchor_dtype(),
        &[128, 128, heads as u64],
        true,
        16,
        "checkpoint",
    )?;
    binding(k, ctx, b.out, &[width], true, 4, "output")?;
    records(k, ctx, heads, p, &r.cached(), true)?;
    binding(k, ctx, r.v, &[128, 8, heads as u64], true, 16, "record v")?;
    binding(k, ctx, r.beta, &[8, heads as u64], true, 16, "record beta")?;
    typed_binding(
        k,
        ctx,
        read_state,
        GgmlType::F16,
        &[128, 128, heads as u64],
        true,
        16,
        "physical read",
    )?;
    if mode == ReplayLpMode::SharedF16Anchor
        && !crate::metal::checks::same_range(b.state, read_state)
    {
        return Err(bad(
            k,
            "shared mode requires exactly the same read/anchor range",
        ));
    }
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
        (r.v, "record v"),
        (r.beta, "record beta"),
        (read_state, "physical read"),
    ];
    let written = &written[..if mode == ReplayLpMode::SeparateF32Anchor {
        9
    } else {
        8
    }];
    for (i, &(t, name)) in written.iter().enumerate() {
        check_disjoint(k, t, &read).map_err(|e| bad(k, format!("{name}: {e}")))?;
        check_disjoint(k, t, &written[i + 1..]).map_err(|e| bad(k, format!("{name}: {e}")))?;
    }
    let tgm = LP_TGM_BYTES;
    let pso = pipeline(ctx, k, tgm)?;
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
        read_state,
        r.v,
        r.beta,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(i + 1, t);
    }
    launch(enc, heads, tgm);
    Ok(())
}

/// Read-only LP logical state to distinct F32 output at any residue.
pub(crate) fn encode_kda_replay_materialize_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    heads: usize,
    p: ReplayPosition,
    checkpoint: &MetalTensor,
    r: &ReplayRecords<'_>,
    output: &MetalTensor,
) -> Result<(), MetalError> {
    const K: &str = LP_MATERIALIZE_KERNEL;
    if p.window != 8 {
        return Err(bad(K, "physical-F16 experiment requires L8"));
    }
    encoder(K, ctx, enc)?;
    let (_, args) = geometry(K, heads, p)?;
    let shape = [128, 128, heads as u64];
    typed_binding(
        K,
        ctx,
        checkpoint,
        GgmlType::F16,
        &shape,
        false,
        16,
        "physical read",
    )?;
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
