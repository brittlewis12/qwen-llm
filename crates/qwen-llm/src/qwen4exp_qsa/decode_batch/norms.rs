//! Independent positions and output buffers; one SIMD computes each head RMS.
use super::*;
crate::env_flag!(
    /// Native M5 independent-decode QSA norms and RoPE in member grids with SIMD RMS reduction. 0 restores ordinary private norms. Cache types, positions and scalar/prefill paths are unchanged.
    default_on flag, "QWEN_FN_BATCH_QSA_NORMS"
);
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    rope: RopeArgs,
    gated: u32,
}
pub(super) fn eligible(ctx: &MetalContext, g: QwenSparseAttentionMetalGeometry, b: usize) -> bool {
    #[cfg(feature = "perf-ablations")]
    let enabled = crate::metal::fn_batch_probe::axis("qsa", flag());
    #[cfg(not(feature = "perf-ablations"))]
    let enabled = flag();
    enabled
        && ctx.tensor_ops_status().is_available()
        && (2..=4).contains(&b)
        && g.head_dim == 256
        && g.index_head_dim == 128
}
pub(super) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: QwenSparseAttentionMetalWeights<'_>,
    members: &[&mut QwenSparseAttentionMetalWorkspace],
    controls: &[(usize, usize)],
) -> Result<(), Qwen4ExpQsaError> {
    #[cfg(feature = "perf-ablations")]
    crate::metal::fn_batch_probe::observe("qsa_norm_grids", 3);
    let g = weights.geometry;
    for kind in 0..3 {
        let (heads, dim, norm) = match kind {
            0 => (
                g.index_query_heads,
                g.index_head_dim,
                weights.index_query_norm,
            ),
            1 => (g.kv_heads, g.head_dim, weights.key_norm),
            _ => (g.query_heads, g.head_dim, weights.query_norm),
        };
        let args: Vec<_> = controls
            .iter()
            .map(|&(p, _)| Args {
                rope: RopeArgs {
                    head_count: heads as u32,
                    head_dim: dim as u32,
                    rotary_dim: g.rotary_dim as u32,
                    position: p as u32,
                    theta: g.theta,
                    eps: g.eps,
                },
                gated: u32::from(kind == 2),
            })
            .collect();
        let inputs: Vec<_> = members
            .iter()
            .map(|w| match kind {
                0 => &w.index_query_raw,
                1 => &w.key_raw,
                _ => &w.query_gate_projection,
            })
            .collect();
        let outputs: Vec<_> = members
            .iter()
            .map(|w| match kind {
                0 => &w.index_query,
                1 => &w.key,
                _ => &w.query,
            })
            .collect();
        let gates: Vec<_> = members.iter().map(|w| &w.raw_gate).collect();
        encode_grid(ctx, enc, norm, &inputs, &outputs, &gates, &args)?;
    }
    Ok(())
}
fn encode_grid(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    norm: &MetalTensor,
    inputs: &[&MetalTensor],
    outputs: &[&MetalTensor],
    gates: &[&MetalTensor],
    args: &[Args],
) -> Result<(), MetalError> {
    enc.set_pipeline(&ctx.pipeline("kernel_fn_batch_qsa_prepare_norm_f32")?);
    enc.set_bytes_slice(0, args);
    enc.set_tensor(1, norm);
    for i in 0..4 {
        let j = i.min(inputs.len() - 1);
        enc.set_tensor(2 + i, inputs[j]);
        enc.set_tensor(6 + i, outputs[j]);
        enc.set_tensor(10 + i, gates[j]);
    }
    enc.dispatch(
        MTLSize {
            width: args[0].rope.head_count as usize,
            height: inputs.len(),
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests;
