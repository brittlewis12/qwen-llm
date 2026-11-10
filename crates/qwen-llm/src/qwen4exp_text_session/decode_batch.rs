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
    encode_post_zero_one(
        ctx, enc, weights, members, &positions, inputs, logits, projection, true,
    )
}

/// GPU-feedback batch forward. Each member's token is read from a device tensor
/// by `GpuInputs`, so no host token is staged and the next command can be
/// encoded while this one executes. `positions[i]` is the position that token is
/// consumed at, not the workspace's published `committed_length` (which lags the
/// pipeline by one frame).
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gpu(
    ctx: &MetalContext,
    enc: &mut KernelEncoder,
    feedback: &[&MetalTensor],
    positions: &[usize],
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    inputs: &MetalTensor,
    logits: &MetalTensor,
    gpu_inputs: &[&super::gpu_inputs::GpuInputs],
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    validate: bool,
) -> Result<(), Qwen4ExpTextSessionError> {
    validate_encoder(ctx, enc)?;
    let h = weights.geometry.hidden_size();
    let v = weights.geometry.vocab_size();
    let b = members.len();
    let columns = b.next_power_of_two();
    if feedback.len() != b || positions.len() != b || gpu_inputs.len() != b {
        return invalid("GPU batch forward member extents differ");
    }
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
    for s in members.iter() {
        if s.geometry != weights.geometry {
            return invalid("batch member geometry differs");
        }
        s.require_idle()?;
    }
    for (&p, s) in positions.iter().zip(members.iter()) {
        if p >= s.geometry.capacity() {
            return invalid("GPU batch position exceeds session capacity");
        }
    }
    // Device input prefix: PLE rows, control scalars, PLE history and the token
    // embedding selection all read the sampled token from GPU storage. Argument
    // pointees need explicit fences before the model wave records scopes.
    for ((&p, &feedback), gpu_inputs) in
        positions.iter().zip(feedback.iter()).zip(gpu_inputs.iter())
    {
        gpu_inputs.encode(ctx, enc, feedback, p)?;
    }
    // SAFETY: the token loop owns every member exclusively until it drains each
    // committed command; all token-dependent staging is on the device.
    unsafe {
        enc.delegate_session_owner(positions[0]);
    }
    for (i, s) in members.iter_mut().enumerate() {
        let history = s.zero_one.gpu_decode_history();
        encode_zero_one_stage(ctx, enc, 0, positions[i], history, weights, s)?;
    }
    encode_post_zero_one(
        ctx, enc, weights, members, positions, inputs, logits, projection, validate,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_post_zero_one(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    positions: &[usize],
    inputs: &MetalTensor,
    logits: &MetalTensor,
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    validate: bool,
) -> Result<(), Qwen4ExpTextSessionError> {
    let h = weights.geometry.hidden_size();
    let v = weights.geometry.vocab_size();
    let b = members.len();
    let columns = b.next_power_of_two();
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
                positions,
                &residuals,
                weights.post_ple[layer],
                &mut blocks,
                projection,
                moe_scratch,
                validate,
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
