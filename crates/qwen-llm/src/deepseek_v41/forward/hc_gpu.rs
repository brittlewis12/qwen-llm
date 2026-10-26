//! Decode-row mHC pre-block: the 4*hidden -> 24 projection, projected RMS,
//! sigmoid/Sinkhorn controls, collapse with the lagged incoming pre, and the
//! weighted RMS norm, as two dispatches. The five separate kernels cost
//! ~0.08 ms per layer on M5 Ultra (24 simdgroups walking a 160-deep load
//! chain for the projection, then four small launches), ~6 ms of a decode
//! token across both HC sites of 40 layers.
//!
//! Eligible sites (`wide`) instead run one dispatch: 24 projection
//! threadgroups that fold the projected RMS beside one collapse + norm
//! threadgroup, with the controls computed in the site's post (their first
//! reader). The one-threadgroup pre had to stream the 4*hidden residual
//! through one core: GSQ tg64 64.25 -> 66.62 tok/s at d0, 56.73 -> 58.68 at
//! d2048. The FFN site's combine also folds the routing weights and routed
//! sum (`encode_ffn_post`).
use super::*;
use objc2_metal::MTLSize;

const HC_MIXES: u64 = 24;
const HC_CONNECTIONS: u64 = 4;
const PROJECTION_THREADS: usize = 1024;
const PRE_THREADS: usize = 1024;

/// `two` keeps the two-dispatch pre-block (projection, then the
/// one-threadgroup pre with the controls) and the plain mHC post; otherwise
/// eligible sites run `kernel_deepseek_v41_hc_pre_wide` and compute their
/// controls in `kernel_deepseek_v41_hc_post_controls` (A/B lever).
const HC_PRE_ENV: &str = "QWEN_V41_HC_PRE";

#[cfg(test)]
thread_local! {
    pub(super) static HC_WIDE_OVERRIDE: std::cell::Cell<Option<bool>> =
        const { std::cell::Cell::new(None) };
}

fn wide_enabled() -> bool {
    #[cfg(test)]
    if let Some(wide) = HC_WIDE_OVERRIDE.with(|o| o.get()) {
        return wide;
    }
    static WIDE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *WIDE.get_or_init(|| std::env::var(HC_PRE_ENV).as_deref() != Ok("two"))
}

/// Whether this site's pre-block runs as one dispatch with its controls
/// deferred to the post. The pre and post sides of a site must agree, so both
/// call this with the same views: the residual, collapsed, normalized and
/// norm-weight views (and the projection) need float4 alignment.
pub(super) fn wide(
    hc: &Hc<'_>,
    hidden: usize,
    x: &MetalTensor,
    collapsed: &MetalTensor,
    normalized: &MetalTensor,
    norm_weight: &MetalTensor,
) -> bool {
    let hidden = hidden as u64;
    wide_enabled()
        && fused_supported(hc)
        && hc.projection.input as u64 == HC_CONNECTIONS * hidden
        && hidden.is_multiple_of(4)
        && hidden / 4 <= (4 * PRE_THREADS) as u64
        && [x, collapsed, normalized, norm_weight, &hc.projection.tensor]
            .iter()
            .all(|t| t.offset.is_multiple_of(16))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcPreArgs {
    hidden: u32,
    norm_eps: f32,
    hc_eps: f32,
}

/// Buffers of one mHC pre-block site. `incoming_pre` is the lagged pre the
/// collapse consumes (previous FFN's, or the initial unit pre); `pre`,
/// `post` and `combination` receive this site's controls.
pub(super) struct HcPreSite<'t> {
    pub mix: &'t MetalTensor,
    pub pre: &'t MetalTensor,
    pub post: &'t MetalTensor,
    pub combination: &'t MetalTensor,
    pub incoming_pre: &'t MetalTensor,
    pub collapsed: &'t MetalTensor,
    pub normalized: &'t MetalTensor,
}

/// Whether the fused decode path serves this site's projection storage.
pub(super) fn fused_supported(hc: &Hc<'_>) -> bool {
    hc.projection.tensor.dtype == GgmlType::F32
        && hc.projection.output == HC_MIXES as usize
        && hc.projection.input.is_multiple_of(4)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_pre(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    hc: &Hc<'_>,
    constants: &[MetalTensor],
    x: &MetalTensor,
    site: HcPreSite<'_>,
) -> std::result::Result<(), metal::MetalError> {
    const K: &str = "v41_hc_pre";
    let hidden = c.hidden as u64;
    let n_in = hc.projection.input;
    let bad = |detail: &str| metal::MetalError::BadShape {
        kernel: K,
        detail: detail.into(),
    };
    if !fused_supported(hc) || n_in as u64 != HC_CONNECTIONS * hidden {
        return Err(bad("fused mHC pre needs an F32 [4*hidden, 24] projection"));
    }
    // Residual and mix views may be 2-D ([hidden, 4], [24, 1]); only the
    // flat extents matter to the kernels.
    let f32_elems = |t: &MetalTensor, n: u64, name: &str| {
        if t.dtype != GgmlType::F32 || t.n_elements() != n {
            return Err(bad(&format!(
                "{name}: expected {n} F32 elements, found {:?} {:?}",
                t.dtype, t.shape
            )));
        }
        Ok(())
    };
    f32_elems(x, HC_CONNECTIONS * hidden, "residual")?;
    f32_elems(site.mix, HC_MIXES, "mix")?;
    f32_elems(&constants[0], 3, "scale")?;
    f32_elems(&constants[1], HC_MIXES, "base")?;
    f32_elems(&constants[2], hidden, "norm weight")?;
    f32_elems(site.pre, HC_CONNECTIONS, "pre")?;
    f32_elems(site.post, HC_CONNECTIONS, "post")?;
    f32_elems(site.incoming_pre, HC_CONNECTIONS, "incoming pre")?;
    f32_elems(site.collapsed, hidden, "collapsed")?;
    f32_elems(site.normalized, hidden, "normalized")?;
    f32_elems(site.combination, 16, "combination")?;
    for (t, name) in [
        (site.mix, "mix"),
        (site.pre, "pre"),
        (site.post, "post"),
        (site.combination, "combination"),
        (site.collapsed, "collapsed"),
        (site.normalized, "normalized"),
    ] {
        if !t.is_writable() {
            return Err(bad(&format!("{name} must be writable")));
        }
    }

    if wide(
        hc,
        c.hidden,
        x,
        site.collapsed,
        site.normalized,
        &constants[2],
    ) {
        // One dispatch: 24 projection threadgroups (each folding the projected
        // RMS) beside one collapse + norm threadgroup. Controls: encode_post.
        let pso = ctx.pipeline("kernel_deepseek_v41_hc_pre_wide")?;
        enc.set_pipeline(&pso);
        enc.set_bytes(
            0,
            &HcPreArgs {
                hidden: c.hidden as u32,
                norm_eps: c.norm_eps,
                hc_eps: c.hc_eps,
            },
        );
        enc.set_tensor(1, x);
        enc.set_tensor(2, &hc.projection.tensor);
        enc.set_tensor(3, site.mix);
        enc.set_tensor(4, site.incoming_pre);
        enc.set_tensor(5, site.collapsed);
        enc.set_tensor(6, &constants[2]);
        enc.set_tensor(7, site.normalized);
        enc.dispatch(
            MTLSize {
                width: HC_MIXES as usize + 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: PRE_THREADS,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }

    // 1. Projection: one threadgroup per mixing logit.
    let pso = ctx.pipeline("kernel_deepseek_v41_hc_projection")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &(n_in as u32));
    enc.set_bytes(1, &(HC_MIXES as u32));
    enc.set_tensor(2, &hc.projection.tensor);
    enc.set_tensor(3, x);
    enc.set_tensor(4, site.mix);
    enc.dispatch(
        MTLSize {
            width: HC_MIXES as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: PROJECTION_THREADS,
            height: 1,
            depth: 1,
        },
    );

    // 2. Projected RMS, controls, collapse, RMS norm in one threadgroup. The
    // vector kernel needs float4 rows (V41_HC_PRE_MAX_D4 per thread) and
    // 16-byte aligned residual/collapsed/normalized/norm-weight views.
    let vector = hidden.is_multiple_of(4)
        && hidden / 4 <= (4 * PRE_THREADS) as u64
        && [x, site.collapsed, site.normalized, &constants[2]]
            .iter()
            .all(|t| t.offset.is_multiple_of(16));
    let pso = ctx.pipeline(if vector {
        "kernel_deepseek_v41_hc_pre"
    } else {
        "kernel_deepseek_v41_hc_pre_scalar"
    })?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &HcPreArgs {
            hidden: c.hidden as u32,
            norm_eps: c.norm_eps,
            hc_eps: c.hc_eps,
        },
    );
    enc.set_tensor(1, x);
    enc.set_tensor(2, site.mix);
    enc.set_tensor(3, &constants[0]);
    enc.set_tensor(4, &constants[1]);
    enc.set_tensor(5, site.pre);
    enc.set_tensor(6, site.post);
    enc.set_tensor(7, site.combination);
    enc.set_tensor(8, site.incoming_pre);
    enc.set_tensor(9, site.collapsed);
    enc.set_tensor(10, &constants[2]);
    enc.set_tensor(11, site.normalized);
    enc.dispatch(
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: PRE_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcPostControlsArgs {
    hidden: u32,
    hc_eps: f32,
}

/// Buffers of one mHC post site. `mix`, `pre`, `post` and `combination` are
/// the same views the site's pre-block used.
pub(super) struct HcPostSite<'t> {
    pub block: &'t MetalTensor,
    pub residual: &'t MetalTensor,
    pub mix: &'t MetalTensor,
    pub pre: &'t MetalTensor,
    pub post: &'t MetalTensor,
    pub combination: &'t MetalTensor,
    pub output: &'t MetalTensor,
}

/// mHC post for a site. After a wide pre-block (`wide` true for the same
/// views) the controls are computed here, in the post's prologue, and stored
/// for the next site; otherwise this is `metal::encode_mhc4_post`.
pub(super) fn encode_post(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    wide: bool,
    constants: &[MetalTensor],
    site: HcPostSite<'_>,
) -> std::result::Result<(), metal::MetalError> {
    if !wide {
        return metal::encode_mhc4_post(
            ctx,
            enc,
            c.hidden,
            site.block,
            site.residual,
            site.post,
            site.combination,
            site.output,
        );
    }
    const K: &str = "v41_hc_post_controls";
    let hidden = c.hidden as u64;
    let bad = |detail: String| metal::MetalError::BadShape { kernel: K, detail };
    for (t, n, name, writable) in [
        (site.block, hidden, "block", false),
        (site.residual, HC_CONNECTIONS * hidden, "residual", false),
        (site.mix, HC_MIXES, "mix", false),
        (&constants[0], 3, "scale", false),
        (&constants[1], HC_MIXES, "base", false),
        (site.pre, HC_CONNECTIONS, "pre", true),
        (site.post, HC_CONNECTIONS, "post", true),
        (site.combination, 16, "combination", true),
        (site.output, HC_CONNECTIONS * hidden, "output", true),
    ] {
        if t.dtype != GgmlType::F32 || t.n_elements() != n || (writable && !t.is_writable()) {
            return Err(bad(format!(
                "{name}: expected {n} {}F32 elements, found {:?} {:?}",
                if writable { "writable " } else { "" },
                t.dtype,
                t.shape
            )));
        }
    }
    let pso = ctx.pipeline("kernel_deepseek_v41_hc_post_controls")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &HcPostControlsArgs {
            hidden: c.hidden as u32,
            hc_eps: c.hc_eps,
        },
    );
    enc.set_tensor(1, site.block);
    enc.set_tensor(2, site.residual);
    enc.set_tensor(3, site.mix);
    enc.set_tensor(4, &constants[0]);
    enc.set_tensor(5, &constants[1]);
    enc.set_tensor(6, site.pre);
    enc.set_tensor(7, site.post);
    enc.set_tensor(8, site.combination);
    enc.set_tensor(9, site.output);
    enc.dispatch(
        MTLSize {
            width: (HC_CONNECTIONS * hidden).div_ceil(256) as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct FfnPostControlsArgs {
    hidden: u32,
    hc_eps: f32,
    topk: u32,
}

/// Decode FFN combine of a wide site in one dispatch: routing weights applied
/// to the down outputs, routed sum plus shared expert, then the mHC post with
/// the site's controls (`kernel_deepseek_v41_ffn_post_controls`).
pub(super) struct FfnPostSite<'t> {
    pub slots: &'t MetalTensor,
    pub weights: &'t MetalTensor,
    pub shared: &'t MetalTensor,
    pub status: &'t MetalTensor,
    pub residual: &'t MetalTensor,
    pub mix: &'t MetalTensor,
    pub pre: &'t MetalTensor,
    pub post: &'t MetalTensor,
    pub combination: &'t MetalTensor,
    pub output: &'t MetalTensor,
}

pub(super) fn encode_ffn_post(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    c: &Config,
    constants: &[MetalTensor],
    site: FfnPostSite<'_>,
) -> std::result::Result<(), metal::MetalError> {
    const K: &str = "v41_ffn_post_controls";
    let hidden = c.hidden as u64;
    let topk = c.topk as u64;
    let bad = |detail: String| metal::MetalError::BadShape { kernel: K, detail };
    for (t, n, name, writable) in [
        (site.slots, topk * hidden, "slots", false),
        (site.weights, topk, "weights", false),
        (site.shared, hidden, "shared", false),
        (site.residual, HC_CONNECTIONS * hidden, "residual", false),
        (site.mix, HC_MIXES, "mix", false),
        (&constants[0], 3, "scale", false),
        (&constants[1], HC_MIXES, "base", false),
        (site.pre, HC_CONNECTIONS, "pre", true),
        (site.post, HC_CONNECTIONS, "post", true),
        (site.combination, 16, "combination", true),
        (site.output, HC_CONNECTIONS * hidden, "output", true),
    ] {
        if t.dtype != GgmlType::F32 || t.n_elements() < n || (writable && !t.is_writable()) {
            return Err(bad(format!(
                "{name}: expected {n} {}F32 elements, found {:?} {:?}",
                if writable { "writable " } else { "" },
                t.dtype,
                t.shape
            )));
        }
    }
    let pso = ctx.pipeline("kernel_deepseek_v41_ffn_post_controls")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &FfnPostControlsArgs {
            hidden: c.hidden as u32,
            hc_eps: c.hc_eps,
            topk: c.topk as u32,
        },
    );
    for (slot, tensor) in [
        site.slots,
        site.weights,
        site.shared,
        site.status,
        site.residual,
        site.mix,
        &constants[0],
        &constants[1],
        site.pre,
        site.post,
        site.combination,
        site.output,
    ]
    .into_iter()
    .enumerate()
    {
        enc.set_tensor(slot + 1, tensor);
    }
    enc.dispatch(
        MTLSize {
            width: (HC_CONNECTIONS * hidden).div_ceil(256) as usize,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
