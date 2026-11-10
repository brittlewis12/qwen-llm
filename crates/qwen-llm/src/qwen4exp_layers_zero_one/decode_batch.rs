//! Member-axis bootstrap decode. HC reads and injection span independent
//! owners; each member keeps private GDN/MoE/PLE causal state and buffers.
use super::*;
use crate::metal::encode_get_rows_f32;
use crate::qwen4exp_metal::decode_batch as hc;

crate::env_flag!(
    /// Batch the layers-zero-one HC norm/down/up reads and injection across independent members. 0 keeps one bootstrap pass per member. Arithmetic, causal state and buffers are unchanged.
    default_on enabled,
    "QWEN_FN_ZERO_ONE_HC_BATCH"
);

fn selected() -> bool {
    #[cfg(feature = "perf-ablations")]
    {
        return crate::metal::fn_batch_probe::axis("zero_one_hc", enabled());
    }
    #[cfg(not(feature = "perf-ablations"))]
    {
        enabled()
    }
}

/// Whether the parent should route the bootstrap stage through the member axis.
pub(crate) fn selected_for_batch() -> bool {
    selected()
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    histories: &[PleHistory],
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    members: &mut [&mut Qwen4ExpLayersZeroOneMetalWorkspace],
    destinations: &[&MetalTensor],
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    let b = members.len();
    if !(2..=4).contains(&b)
        || token_ids.len() != b
        || histories.len() != b
        || destinations.len() != b
    {
        return invalid("member-axis bootstrap requires 2..=4 aligned members");
    }
    let g = members[0].geometry;
    if members.iter().any(|w| w.geometry != g) {
        return invalid("member-axis bootstrap geometries differ");
    }
    if token_ids.iter().any(|&t| t as usize >= g.vocab_size()) {
        return invalid("member-axis bootstrap token ID is outside the vocabulary");
    }
    let hidden = g.hidden_size();
    let hyper = g.hyper_width();
    let branches = g.branch_count();
    let encoded = (|| {
        // Embeddings: one gather and one per-branch copy per member.
        {
            let mut tokens = Vec::with_capacity(b);
            let mut embeddings = Vec::with_capacity(b);
            let mut hypers = Vec::with_capacity(b);
            for (i, w) in members.iter_mut().enumerate() {
                let ws = &mut **w;
                ws.begin_batch(enc, histories[i].clone())?;
                ws.layer_zero.stage_token(enc, token_ids[i])?;
                let z = ws.layer_zero.batch_parts();
                tokens.push(z.token_id);
                embeddings.push(z.embedding);
                hypers.push(z.hyper_residual);
            }
            for i in 0..b {
                encode_get_rows_f32(
                    ctx,
                    enc,
                    weights.layer_zero.token_embedding,
                    tokens[i],
                    embeddings[i],
                    1,
                    hidden,
                )?;
                for branch in 0..branches {
                    let destination =
                        hypers[i].view_subrange((branch * hidden) as u64, vec![hidden as u64]);
                    encode_copy_offset_f32(ctx, enc, embeddings[i], 0, &destination, hidden)?;
                }
            }
        }
        // Layer-zero attention residual.
        {
            let mut inputs = Vec::with_capacity(b);
            let mut outputs = Vec::with_capacity(b);
            let mut residuals = Vec::with_capacity(b);
            let mut gdns = Vec::with_capacity(b);
            for w in members.iter_mut() {
                let z = w.layer_zero.batch_parts();
                inputs.push(z.hyper_residual);
                outputs.push(z.attention_output);
                residuals.push(z.residual);
                gdns.push(z.gdn);
            }
            let reads = hc::encode(
                ctx,
                enc,
                weights.layer_zero.geometry.eps(),
                weights.layer_zero.attention_residual.read,
                weights.layer_zero.attention_residual.inject,
                inputs
                    .into_iter()
                    .zip(outputs)
                    .zip(residuals)
                    .map(|((input, output), scratch)| hc::Member {
                        input,
                        output,
                        scratch,
                    })
                    .collect(),
            )?;
            for (read, gdn) in reads.iter().zip(gdns) {
                let gdn =
                    encode_gated_delta_net(ctx, enc, read.mixed(), weights.layer_zero.gdn, gdn)?;
                gdn.output().encode_copy_to(ctx, enc, read.block_output())?;
            }
            hc::combine(reads)?;
        }
        // Layer-zero FFN residual, then its output becomes the layer-one input.
        {
            let mut inputs = Vec::with_capacity(b);
            let mut outputs = Vec::with_capacity(b);
            let mut residuals = Vec::with_capacity(b);
            let mut moes = Vec::with_capacity(b);
            let mut sources = Vec::with_capacity(b);
            let mut copies = Vec::with_capacity(b);
            for w in members.iter_mut() {
                let ws = &mut **w;
                let z = ws.layer_zero.batch_parts();
                inputs.push(z.hyper_residual);
                outputs.push(z.ffn_output);
                residuals.push(z.residual);
                moes.push(z.moe);
                sources.push(z.hyper_residual);
                copies.push(&ws.hyper_residual);
            }
            let reads = hc::encode(
                ctx,
                enc,
                weights.layer_zero.geometry.eps(),
                weights.layer_zero.ffn_residual.read,
                weights.layer_zero.ffn_residual.inject,
                inputs
                    .into_iter()
                    .zip(outputs)
                    .zip(residuals)
                    .map(|((input, output), scratch)| hc::Member {
                        input,
                        output,
                        scratch,
                    })
                    .collect(),
            )?;
            for (read, moe) in reads.iter().zip(moes) {
                let moe = encode_qwen4exp_moe(ctx, enc, read.mixed(), weights.layer_zero.moe, moe)?;
                moe.output().encode_copy_to(ctx, enc, read.block_output())?;
            }
            hc::combine(reads)?;
            for i in 0..b {
                encode_copy_offset_f32(ctx, enc, sources[i], 0, copies[i], hyper)?;
            }
        }
        // PLE owns per-sequence rows; keep it on the member route.
        {
            let mut hypers = Vec::with_capacity(b);
            let mut ples = Vec::with_capacity(b);
            for w in members.iter_mut() {
                let ws = &mut **w;
                hypers.push(&ws.hyper_residual);
                ples.push(&mut ws.ple);
            }
            for (hyper_input, ple) in hypers.iter().zip(ples) {
                let ple = encode_qwen4exp_ple(ctx, enc, hyper_input, weights.ple, ple)?;
                ple.output().encode_copy_to(ctx, enc, hyper_input)?;
            }
        }
        // Layer-one attention residual.
        {
            let mut inputs = Vec::with_capacity(b);
            let mut outputs = Vec::with_capacity(b);
            let mut residuals = Vec::with_capacity(b);
            let mut gdns = Vec::with_capacity(b);
            for w in members.iter_mut() {
                let ws = &mut **w;
                inputs.push(&ws.hyper_residual);
                outputs.push(&ws.mixer_output);
                residuals.push(&mut ws.residual);
                gdns.push(&mut ws.layer_one_gdn);
            }
            let reads = hc::encode(
                ctx,
                enc,
                g.eps(),
                weights.layer_one_attention_residual.read,
                weights.layer_one_attention_residual.inject,
                inputs
                    .into_iter()
                    .zip(outputs)
                    .zip(residuals)
                    .map(|((input, output), scratch)| hc::Member {
                        input,
                        output,
                        scratch,
                    })
                    .collect(),
            )?;
            for (read, gdn) in reads.iter().zip(gdns) {
                let gdn =
                    encode_gated_delta_net(ctx, enc, read.mixed(), weights.layer_one_gdn, gdn)?;
                gdn.output().encode_copy_to(ctx, enc, read.block_output())?;
            }
            hc::combine(reads)?;
        }
        // Layer-one FFN residual.
        {
            let mut inputs = Vec::with_capacity(b);
            let mut outputs = Vec::with_capacity(b);
            let mut residuals = Vec::with_capacity(b);
            let mut moes = Vec::with_capacity(b);
            for w in members.iter_mut() {
                let ws = &mut **w;
                inputs.push(&ws.hyper_residual);
                outputs.push(&ws.moe_output);
                residuals.push(&mut ws.residual);
                moes.push(&mut ws.layer_one_moe);
            }
            let reads = hc::encode(
                ctx,
                enc,
                g.eps(),
                weights.layer_one_ffn_residual.read,
                weights.layer_one_ffn_residual.inject,
                inputs
                    .into_iter()
                    .zip(outputs)
                    .zip(residuals)
                    .map(|((input, output), scratch)| hc::Member {
                        input,
                        output,
                        scratch,
                    })
                    .collect(),
            )?;
            for (read, moe) in reads.iter().zip(moes) {
                let moe = encode_qwen4exp_moe(ctx, enc, read.mixed(), weights.layer_one_moe, moe)?;
                moe.output().encode_copy_to(ctx, enc, read.block_output())?;
            }
            hc::combine(reads)?;
        }
        for (w, destination) in members.iter().zip(destinations) {
            encode_copy_offset_f32(ctx, enc, w.output_tensor(), 0, destination, hyper)?;
        }
        #[cfg(feature = "perf-ablations")]
        crate::metal::fn_batch_probe::observe("zero_one_hc", b);
        Ok(())
    })();
    if encoded.is_err() {
        for w in members.iter_mut() {
            w.mark_batch_failed();
        }
    }
    encoded
}
