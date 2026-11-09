//! Layer-major independent owners with private causal state and grouped kernels.
use super::*;
use crate::qwen4exp_metal::decode_batch::{self as hc, Member};
pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    positions: &[usize],
    inputs: &[&MetalTensor],
    weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
    members: &mut [&mut Qwen4ExpPostPleBlockMetalWorkspace],
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    moe_scratch: Option<&crate::qwen4exp_moe::Qwen4ExpMoePackedMotorScratch>,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    validate_encoder(ctx, enc)?;
    for ((w, &p), &x) in members.iter().zip(positions).zip(inputs) {
        validate_and_preflight(ctx, p, x, weights, w)?;
    }
    for w in members.iter_mut() {
        reserve_command(w, enc)?;
        w.encode_failed = true;
    }
    let mut hc_members = Vec::with_capacity(members.len());
    let mut rest = Vec::with_capacity(members.len());
    for (w, &x) in members.iter_mut().zip(inputs) {
        let Qwen4ExpPostPleBlockMetalWorkspace {
            residual,
            mixer,
            mixer_output,
            ..
        } = &mut **w;
        hc_members.push(Member {
            input: x,
            output: mixer_output,
            scratch: residual,
        });
        rest.push(mixer);
    }
    let reads = hc::encode(
        ctx,
        enc,
        weights.geometry.eps,
        weights.attention_residual.read,
        weights.attention_residual.inject,
        hc_members,
    )?;
    let mixed: Vec<_> = reads.iter().map(|r| r.mixed()).collect();
    match weights.mixer {
        Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(w) => {
            let states = rest
                .into_iter()
                .map(|m| match m {
                    Qwen4ExpPostPleMixerMetalWorkspace::GatedDeltaNet(s) => Ok(s.as_mut()),
                    _ => invalid("native mixer variant mismatch"),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let results =
                crate::qwen4exp_gdn::decode_batch::encode(ctx, enc, &mixed, w, states, projection)?;
            for (result, read) in results.into_iter().zip(&reads) {
                result
                    .output()
                    .encode_copy_to(ctx, enc, read.block_output())?;
            }
        }
        Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(w) => {
            let states = rest
                .into_iter()
                .map(|m| match m {
                    Qwen4ExpPostPleMixerMetalWorkspace::QwenSparseAttention(s) => Ok(s.as_mut()),
                    _ => invalid("native mixer variant mismatch"),
                })
                .collect::<Result<Vec<_>, _>>()?;
            let results =
                crate::qwen4exp_qsa::decode_batch::encode(ctx, enc, &mixed, w, states, projection)?;
            for (result, read) in results.into_iter().zip(&reads) {
                result
                    .output()
                    .encode_copy_to(ctx, enc, read.block_output())?;
            }
        }
    }
    drop(mixed);
    hc::combine(reads)?;
    let mut hc_members = Vec::with_capacity(members.len());
    let mut rest = Vec::with_capacity(members.len());
    for (w, &x) in members.iter_mut().zip(inputs) {
        let Qwen4ExpPostPleBlockMetalWorkspace {
            residual,
            moe,
            moe_output,
            ..
        } = &mut **w;
        hc_members.push(Member {
            input: x,
            output: moe_output,
            scratch: residual,
        });
        rest.push(moe);
    }
    let reads = hc::encode(
        ctx,
        enc,
        weights.geometry.eps,
        weights.ffn_residual.read,
        weights.ffn_residual.inject,
        hc_members,
    )?;
    let mixed: Vec<_> = reads.iter().map(|r| r.mixed()).collect();
    let results = crate::qwen4exp_moe::decode_batch::encode(
        ctx,
        enc,
        &mixed,
        weights.moe,
        rest,
        moe_scratch,
    )?;
    for (result, read) in results.into_iter().zip(&reads) {
        result
            .output()
            .encode_copy_to(ctx, enc, read.block_output())?;
    }
    drop(mixed);
    hc::combine(reads)?;
    for w in members {
        w.encode_failed = false;
    }
    Ok(())
}
