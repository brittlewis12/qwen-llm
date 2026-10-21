//! Decode-row mHC pre-block: the 4*hidden -> 24 projection, projected RMS,
//! sigmoid/Sinkhorn controls, collapse with the lagged incoming pre, and the
//! weighted RMS norm, as two dispatches. The five separate kernels cost
//! ~0.08 ms per layer on M5 Ultra (24 simdgroups walking a 160-deep load
//! chain for the projection, then four small launches), ~6 ms of a decode
//! token across both HC sites of 40 layers.
use super::*;
use objc2_metal::MTLSize;

const HC_MIXES: u64 = 24;
const HC_CONNECTIONS: u64 = 4;
const PROJECTION_THREADS: usize = 1024;
const PRE_THREADS: usize = 1024;

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

    // 2. Projected RMS, controls, collapse, RMS norm in one threadgroup.
    let pso = ctx.pipeline("kernel_deepseek_v41_hc_pre")?;
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
