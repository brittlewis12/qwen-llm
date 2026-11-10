//! Host-sampled batched decode with device-owned inputs.
use super::gpu_inputs::GpuInputs;
use super::*;
use crate::metal::wait_completed;
use objc2_metal::MTLCommandQueue;
use std::time::Instant;

fn command_alloc_error() -> Qwen4ExpTextSessionError {
    Qwen4ExpTextSessionError::Invalid("GPU-input batch command allocation failed".into())
}

/// The completed logits row of the current frame. Callers may only read this
/// after the owning command has completed.
fn logits_row(member: &Qwen4ExpTextSessionMetalWorkspace, vocab: usize) -> &[f32] {
    unsafe {
        std::slice::from_raw_parts(
            member
                .logits
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(member.logits.offset as usize)
                .cast::<f32>(),
            vocab,
        )
    }
}

fn write_token(token: &MetalTensor, value: i32) {
    unsafe {
        *token
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(token.offset as usize)
            .cast::<i32>() = value;
    }
}

/// One host-sampled batch forward with device-owned inputs. Unlike `generate`,
/// this commits and waits one command, so the host can sample between rounds.
/// PLE rows, embedding selection and control scalars are still built on the GPU,
/// which removes the CPU PLE gather from every batched round.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_gpu_inputs(
    ctx: &MetalContext,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    batch_inputs: &MetalTensor,
    batch_logits: &MetalTensor,
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    tokens: &[u32],
    device_tokens: &[MetalTensor],
    diagnostic: &mut [f64; 3],
) -> Result<Vec<Vec<f32>>, Qwen4ExpTextSessionError> {
    let started = Instant::now();
    let b = members.len();
    if b < 2 || tokens.len() != b || device_tokens.len() < b {
        return invalid("invalid GPU-input batch extents");
    }
    let config = weights.zero_one.ple_config;
    let vocab = weights.geometry.vocab_size();
    let positions: Vec<usize> = members.iter().map(|m| m.committed_length).collect();
    for (&p, m) in positions.iter().zip(members.iter()) {
        if p >= m.geometry.capacity() {
            return invalid("GPU-input batch position exceeds capacity");
        }
    }
    let mut gpu_inputs = Vec::with_capacity(b);
    for m in members.iter_mut() {
        let inputs = match m.gpu_inputs.take() {
            Some(inputs) => inputs,
            None => GpuInputs::new(ctx, table, config, m)?,
        };
        inputs.reset_history(m);
        gpu_inputs.push(inputs);
    }
    for (i, &token) in tokens.iter().enumerate() {
        write_token(&device_tokens[i], token as i32);
    }
    let result = (|| -> Result<Vec<Vec<f32>>, Qwen4ExpTextSessionError> {
        let command = ctx.queue.commandBuffer().ok_or_else(command_alloc_error)?;
        let encode_started = Instant::now();
        let mut enc = KernelEncoder::begin(&command);
        let feedback: Vec<&MetalTensor> = device_tokens[..b].iter().collect();
        let input_refs: Vec<&GpuInputs> = gpu_inputs.iter().collect();
        crate::qwen4exp_text_session::decode_batch::encode_gpu(
            ctx,
            &mut enc,
            &feedback,
            &positions,
            weights,
            members,
            batch_inputs,
            batch_logits,
            &input_refs,
            projection,
            true,
        )?;
        enc.end();
        let encode_ms = encode_started.elapsed().as_secs_f64() * 1e3;
        command.commit();
        wait_completed(&command)?;
        let gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        let mut out = Vec::with_capacity(b);
        for i in 0..b {
            members[i].zero_one.validate_gpu_decode_status()?;
            for block in &members[i].post_ple {
                block.validate_gpu_decode_status(positions[i])?;
            }
            let mut history = members[i].zero_one.gpu_decode_history();
            history = history.advanced(config, tokens[i], positions[i] as u64)?.0;
            members[i].zero_one.publish_gpu_decode_history(history);
            for block in &mut members[i].post_ple {
                block.publish_gpu_decode_length(positions[i] + 1);
            }
            members[i].committed_length = positions[i] + 1;
            members[i].logits_ready = true;
            members[i].state_poisoned = false;
            members[i].checkpoint_lineage.record_append()?;
            out.push(logits_row(members[i], vocab).to_vec());
        }
        diagnostic[0] = encode_ms;
        diagnostic[1] = if gpu_ms.is_finite() { gpu_ms } else { 0. };
        diagnostic[2] = started.elapsed().as_secs_f64() * 1e3;
        Ok(out)
    })();
    if result.is_err() {
        for m in members.iter_mut() {
            m.state_poisoned = true;
            m.logits_ready = false;
        }
    }
    for (m, inputs) in members.iter_mut().zip(gpu_inputs) {
        m.gpu_inputs = Some(inputs);
    }
    result
}
