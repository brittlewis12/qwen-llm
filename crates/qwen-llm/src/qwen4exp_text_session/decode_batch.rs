//! Independent owners sharing dense/head projections and causal/expert grids.
use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    tokens: &[u32],
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    inputs: &MetalTensor,
    logits: &MetalTensor,
    started: &mut bool,
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
) -> Result<(), Qwen4ExpTextSessionError> {
    validate_encoder(ctx, enc)?;
    let h = weights.geometry.hidden_size();
    let v = weights.geometry.vocab_size();
    let b = members.len();
    let columns = b.next_power_of_two();
    require_tensor(
        "batch head inputs",
        inputs,
        GgmlType::F32,
        &[h as u64, columns as u64],
        true,
    )?;
    require_tensor(
        "batch logits",
        logits,
        GgmlType::F32,
        &[v as u64, columns as u64],
        true,
    )?;
    require_disjoint(&[("batch inputs", inputs), ("batch logits", logits)])?;
    for (&t, s) in tokens.iter().zip(members.iter()) {
        if s.geometry != weights.geometry {
            return invalid("batch member geometry differs");
        }
        validate_and_preflight(ctx, enc, t, s.committed_length, weights, s)?;
    }
    *started = true;
    let positions: Vec<_> = members.iter().map(|s| s.committed_length).collect();
    let mut histories = Vec::with_capacity(b);
    for ((&t, s), &p) in tokens.iter().zip(members.iter_mut()).zip(&positions) {
        histories.push(prepare_qwen4exp_text_token(
            ctx, enc, t, p, table, weights, s,
        )?);
    }
    for (i, s) in members.iter_mut().enumerate() {
        encode_zero_one_stage(
            ctx,
            enc,
            tokens[i],
            positions[i],
            histories[i].clone(),
            weights,
            s,
        )?;
    }
    {
        let mut layers = Vec::with_capacity(b);
        let mut residuals = Vec::with_capacity(b);
        let mut moe_scratch = None;
        for (i, s) in members.iter_mut().enumerate() {
            let Qwen4ExpTextSessionMetalWorkspace {
                post_ple,
                hyper_residual,
                packed,
                ..
            } = &mut **s;
            layers.push(post_ple);
            residuals.push(&*hyper_residual);
            if i == 0 {
                moe_scratch = packed.as_ref().map(|p| &p.moe);
            }
        }
        for layer in 0..weights.post_ple.len() {
            let mut blocks: Vec<_> = layers.iter_mut().map(|p| &mut p[layer]).collect();
            crate::qwen4exp_post_ple_block::decode_batch::encode(
                ctx,
                enc,
                &positions,
                &residuals,
                weights.post_ple[layer],
                &mut blocks,
                projection,
                moe_scratch,
            )?;
        }
    }
    for (i, s) in members.iter_mut().enumerate() {
        let read = crate::qwen4exp_metal::encode_final_gated_residual_mix_with_policy(
            ctx,
            enc,
            &s.hyper_residual,
            s.geometry.eps(),
            weights.final_read,
            &mut s.final_read,
            true,
        )?;
        encode_copy_offset_f32(
            ctx,
            enc,
            read.mixed(),
            0,
            &inputs.view_subrange((i * h) as u64, vec![h as u64]),
            h,
        )?;
    }
    if b == 3 && !crate::qwen4exp_metal::decode_projection::mma_enabled() {
        encode_copy_offset_f32(
            ctx,
            enc,
            &inputs.view_subrange(0, vec![h as u64]),
            0,
            &inputs.view_subrange((3 * h) as u64, vec![h as u64]),
            h,
        )?;
    }
    if crate::qwen4exp_metal::decode_projection::mma_enabled() {
        crate::metal::encode_mat_mat_fewrow_gen_f32_weights(
            ctx,
            enc,
            weights.output,
            &inputs.view_subrange(0, vec![h as u64, b as u64]),
            &logits.view_subrange(0, vec![v as u64, b as u64]),
            h,
            v,
            b,
        )?;
    } else {
        crate::metal::encode_mat_vec_nc_dispatch(
            ctx,
            enc,
            weights.output,
            inputs,
            logits,
            h,
            v,
            columns,
        )?;
    }
    for (i, s) in members.iter_mut().enumerate() {
        encode_copy_offset_f32(
            ctx,
            enc,
            &logits.view_subrange((i * v) as u64, vec![v as u64]),
            0,
            &s.logits,
            v,
        )?;
    }
    Ok(())
}

pub(crate) fn poison(members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace]) {
    for s in members {
        s.state_poisoned = true;
        s.encode_failed = true;
        s.logits_ready = false;
    }
}
