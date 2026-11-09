use super::*;
use crate::qwen4exp_metal::decode_projection::Scratch;
pub(crate) fn encode<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    inputs: &[&MetalTensor],
    weights: GatedDeltaNetMetalWeights<'_>,
    mut members: Vec<&'a mut GatedDeltaNetMetalWorkspace>,
    scratch: &Scratch,
) -> Result<Vec<GatedDeltaNetMetalRead<'a>>, Qwen4ExpGdnError> {
    for (w, x) in members.iter().zip(inputs) {
        validate_decode(ctx, enc, x, weights, w)?;
    }
    for w in &mut members {
        reserve_command(w, enc)?;
    }
    let qkv: Vec<_> = members.iter().map(|w| &w.qkv).collect();
    scratch.project(ctx, enc, weights.qkv, inputs, &qkv)?;
    let gates: Vec<_> = members.iter().map(|w| &w.gate).collect();
    scratch.project(ctx, enc, weights.gate, inputs, &gates)?;
    for (w, x) in members.iter().zip(inputs) {
        encode_recurrence(ctx, enc, x, weights, w)?;
    }
    let normalized: Vec<_> = members.iter().map(|w| &w.normalized).collect();
    let outputs: Vec<_> = members.iter().map(|w| &w.output).collect();
    scratch.project(ctx, enc, weights.output, &normalized, &outputs)?;
    Ok(members
        .into_iter()
        .map(|workspace| GatedDeltaNetMetalRead { workspace })
        .collect())
}
