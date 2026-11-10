//! Batched independent decode with device-owned token feedback.
use super::gpu_inputs::GpuInputs;
use super::*;
use crate::metal::{GpuTokenSampler, wait_completed};
use crate::sampling::Sampler;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
use std::time::Instant;

/// Batched token-loop counters. `timeline` is captured only under
/// `perf-ablations` so a probe can place each command's encode, commit and wait
/// on the GPU clock.
#[derive(Clone, Debug, Default)]
pub struct BatchGpuDecodeStats {
    pub sampled_tokens: usize,
    pub forwarded_tokens: usize,
    pub emitted_tokens: usize,
    pub encode_cpu_ms: f64,
    pub completion_wait_ms: f64,
    pub gpu_ms: f64,
    pub wall_ms: f64,
    pub stopped_on_eos: usize,
    pub terminal_tokens: Vec<i32>,
    /// Per-member wall time from burst start to its first selected token.
    pub first_token_ms: Vec<Option<f64>>,
    #[cfg(feature = "perf-ablations")]
    pub timeline: Vec<crate::metal::TokenFrameTimeline>,
}

struct Prepared {
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    tokens: Vec<MetalTensor>,
    /// Position each member's feedback token is consumed at, when this command
    /// runs a forward. `None` for the initial sampler-only frame.
    forward: Option<Vec<usize>>,
    /// Feedback tokens this forward consumes, known once the previous frame's
    /// outputs have been received.
    inputs: Option<Vec<i32>>,
}

struct InFlight(Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>);
impl Drop for InFlight {
    fn drop(&mut self) {
        if let Some(command) = self.0.take()
            && command.status() != objc2_metal::MTLCommandBufferStatus::NotEnqueued
        {
            crate::metal::wait_unchecked(&command);
        }
    }
}

fn prepare_sampler_only(
    ctx: &MetalContext,
    members: &[&mut Qwen4ExpTextSessionMetalWorkspace],
    samplers: &[GpuTokenSampler],
    uniforms: &[f32],
    output: &[MetalTensor],
) -> Result<Prepared, Qwen4ExpTextSessionError> {
    let command = ctx.queue.commandBuffer().ok_or_else(command_alloc_error)?;
    let enc = KernelEncoder::begin(&command);
    let tokens = output.to_vec();
    for (i, member) in members.iter().enumerate() {
        samplers[i].encode(ctx, &enc, &member.logits, &tokens[i], uniforms[i])?;
    }
    enc.end();
    Ok(Prepared {
        command,
        tokens,
        forward: None,
        inputs: None,
    })
}

#[allow(clippy::too_many_arguments)]
fn prepare_forward(
    ctx: &MetalContext,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    batch_inputs: &MetalTensor,
    batch_logits: &MetalTensor,
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    gpu_inputs: &[GpuInputs],
    samplers: &[GpuTokenSampler],
    feedback: &[MetalTensor],
    positions: &[usize],
    uniforms: &[f32],
    output: &[MetalTensor],
    validate: bool,
) -> Result<Prepared, Qwen4ExpTextSessionError> {
    let command = ctx.queue.commandBuffer().ok_or_else(command_alloc_error)?;
    let mut enc = KernelEncoder::begin(&command);
    let feedback_refs: Vec<&MetalTensor> = feedback.iter().collect();
    let gpu_input_refs: Vec<&GpuInputs> = gpu_inputs.iter().collect();
    crate::qwen4exp_text_session::decode_batch::encode_gpu(
        ctx,
        &mut enc,
        &feedback_refs,
        positions,
        weights,
        members,
        batch_inputs,
        batch_logits,
        &gpu_input_refs,
        projection,
        validate,
    )?;
    let tokens = output.to_vec();
    for (i, member) in members.iter().enumerate() {
        samplers[i].encode(ctx, &enc, &member.logits, &tokens[i], uniforms[i])?;
    }
    enc.end();
    Ok(Prepared {
        command,
        tokens,
        forward: Some(positions.to_vec()),
        inputs: None,
    })
}

/// Sample a bounded burst for a fixed member set. The first token comes from
/// each member's completed logits; every later token comes from one pipelined
/// forward. Any member stop (EOS, budget, or `emit=false`) ends the burst before
/// the prepared frame is committed, so no member's state advances past a token
/// the caller has not accepted.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate(
    ctx: &MetalContext,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    batch_inputs: &MetalTensor,
    batch_logits: &MetalTensor,
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    samplers: &mut [&mut Sampler],
    max_tokens: &[usize],
    eos: &[&[i32]],
    mut emit: impl FnMut(usize, i32, &[f32]) -> bool,
) -> Result<BatchGpuDecodeStats, Qwen4ExpTextSessionError> {
    let started = Instant::now();
    let b = members.len();
    if !(2..=4).contains(&b) {
        return invalid("GPU batch loop requires 2..=4 members");
    }
    if samplers.len() != b || max_tokens.len() != b || eos.len() != b {
        return invalid("GPU batch loop member extents differ");
    }
    if members.iter().zip(max_tokens).any(|(m, &n)| {
        m.geometry != weights.geometry
            || m.active_command.is_some()
            || !m.logits_ready
            || m.state_poisoned
            || n == 0
            || n - 1 > m.geometry.capacity().saturating_sub(m.committed_length)
    }) {
        return invalid("GPU batch member is not ready for a token loop");
    }
    let vocab = weights.geometry.vocab_size();
    let config = weights.zero_one.ple_config;

    // Take each member's immutable GPU PLE banks and seed the device history
    // from the state the last completed forward published.
    let mut gpu_inputs = Vec::with_capacity(b);
    for member in members.iter_mut() {
        let inputs = match member.gpu_inputs.take() {
            Some(inputs) => inputs,
            None => GpuInputs::new(ctx, table, config, member)?,
        };
        inputs.reset_history(member);
        gpu_inputs.push(inputs);
    }

    let mut gpu_samplers = Vec::with_capacity(b);
    for sampler in samplers.iter() {
        gpu_samplers.push(GpuTokenSampler::new(ctx, vocab, sampler.config())?);
    }

    let starts: Vec<usize> = members.iter().map(|m| m.committed_length).collect();
    let mut histories: Vec<PleHistory> = members
        .iter()
        .map(|m| m.zero_one.gpu_decode_history())
        .collect();
    for member in members.iter_mut() {
        member.state_poisoned = true;
        member.logits_ready = false;
    }

    // Two token-tensor sets alternate so a prepared frame never writes the
    // feedback set the executing frame reads. Reusing buffers also keeps Metal
    // from allocating (and possibly waiting on) a buffer every round.
    let mut token_sets = Vec::with_capacity(2);
    for _ in 0..2 {
        let mut set = Vec::with_capacity(b);
        for _ in 0..b {
            set.push(MetalTensor::zeros_i32(ctx, vec![1])?);
        }
        token_sets.push(set);
    }
    let mut current_slot = 0usize;
    let initial_uniforms: Vec<f32> = samplers.iter().map(|s| s.gpu_uniform(0)).collect();
    #[cfg(feature = "perf-ablations")]
    let mut current_encode_start = crate::metal::host_now_s();
    let mut current = prepare_sampler_only(
        ctx,
        members,
        &gpu_samplers,
        &initial_uniforms,
        &token_sets[0],
    )?;
    #[cfg(feature = "perf-ablations")]
    let mut current_encode_end = crate::metal::host_now_s();
    let mut running = InFlight(Some(current.command.clone()));
    let mut forwarded = 0usize;
    let mut sampled = vec![0usize; b];
    let mut stats = BatchGpuDecodeStats::default();

    let encode_started = Instant::now();
    current.command.commit();
    stats.encode_cpu_ms += encode_started.elapsed().as_secs_f64() * 1e3;
    #[cfg(feature = "perf-ablations")]
    let mut current_commit = crate::metal::host_now_s();

    loop {
        let positions: Vec<usize> = (0..b).map(|i| starts[i] + forwarded).collect();
        let needs_forward = (0..b).any(|i| sampled[i] + 1 < max_tokens[i]);
        let uniforms: Vec<f32> = samplers.iter().map(|s| s.gpu_uniform(1)).collect();
        let prepare_started = Instant::now();
        #[cfg(feature = "perf-ablations")]
        let next_encode_start = crate::metal::host_now_s();
        let next = if needs_forward {
            Some(prepare_forward(
                ctx,
                weights,
                batch_inputs,
                batch_logits,
                projection,
                members,
                &gpu_inputs,
                &gpu_samplers,
                &current.tokens,
                &positions,
                &uniforms,
                &token_sets[1 - current_slot],
                forwarded == 0,
            )?)
        } else {
            None
        };
        #[cfg(feature = "perf-ablations")]
        let next_encode_end = crate::metal::host_now_s();
        stats.encode_cpu_ms += prepare_started.elapsed().as_secs_f64() * 1e3;

        let wait_started = Instant::now();
        #[cfg(feature = "perf-ablations")]
        let wait_start_s = crate::metal::host_now_s();
        // A direct blocking wait plus a shared-memory read avoids the Metal
        // completion-handler dispatch latency, which dominates the round when
        // the host CPU is busy. The next command is already encoded, so the GPU
        // still overlaps that encode.
        #[cfg(feature = "perf-ablations")]
        if crate::metal::spin_wait_enabled() {
            crate::metal::spin_wait_until_completed(&current.command);
        }
        wait_completed(&current.command)?;
        #[cfg(feature = "perf-ablations")]
        let wait_end_s = crate::metal::host_now_s();
        let tokens: Vec<i32> = current.tokens.iter().map(read_token).collect();
        stats.completion_wait_ms += wait_started.elapsed().as_secs_f64() * 1e3;
        let elapsed = (current.command.GPUEndTime() - current.command.GPUStartTime()) * 1e3;
        if elapsed.is_finite() && elapsed > 0. {
            stats.gpu_ms += elapsed;
        }
        #[cfg(feature = "perf-ablations")]
        stats.timeline.push(crate::metal::TokenFrameTimeline {
            frame: stats.sampled_tokens,
            forward: current.forward.is_some(),
            encode_start_s: current_encode_start,
            encode_end_s: current_encode_end,
            commit_s: current_commit,
            wait_start_s,
            wait_end_s,
            gpu_start_s: current.command.GPUStartTime(),
            gpu_end_s: current.command.GPUEndTime(),
        });

        if let Some(positions) = current.forward.clone() {
            let inputs = current.inputs.as_ref().expect("forward frame has inputs");
            for i in 0..b {
                members[i].zero_one.validate_gpu_decode_status()?;
                for block in &members[i].post_ple {
                    block.validate_gpu_decode_status(positions[i])?;
                }
                let token = inputs[i];
                // The CPU history mirrors the device history for publication;
                // it advances by the token the forward consumed at its position.
                histories[i] = histories[i]
                    .advanced(config, token as u32, positions[i] as u64)?
                    .0;
            }
        }
        running.0 = None;

        let mut stop = false;
        for sampler in samplers.iter_mut() {
            sampler.commit_gpu_draw();
        }
        for i in 0..b {
            let token = tokens[i];
            if token < 0 || token as usize >= vocab {
                return invalid(format!(
                    "GPU batch loop selected invalid token {token} for vocabulary {vocab}"
                ));
            }
            sampled[i] += 1;
            stats.sampled_tokens += 1;
            if stats.terminal_tokens.len() <= i {
                stats.terminal_tokens.resize(i + 1, -1);
                stats.first_token_ms.resize(i + 1, None);
            }
            stats.terminal_tokens[i] = token;
            stats.first_token_ms[i].get_or_insert(started.elapsed().as_secs_f64() * 1e3);
            if eos[i].contains(&token) {
                stats.stopped_on_eos += 1;
                stop = true;
            } else {
                stats.emitted_tokens += 1;
                if !emit(i, token, logits_row(members[i], vocab)) {
                    stop = true;
                }
            }
            if sampled[i] >= max_tokens[i] {
                stop = true;
            }
        }
        if stop {
            // `next` is uncommitted: drop it, leaving every member's state at
            // the last accepted token.
            break;
        }
        let mut next = next.expect("a continuing burst prepared a forward frame");
        next.inputs = Some(tokens);
        running.0 = Some(next.command.clone());
        let commit_started = Instant::now();
        next.command.commit();
        stats.encode_cpu_ms += commit_started.elapsed().as_secs_f64() * 1e3;
        #[cfg(feature = "perf-ablations")]
        {
            current_encode_start = next_encode_start;
            current_encode_end = next_encode_end;
            current_commit = crate::metal::host_now_s();
        }
        if next.forward.is_some() {
            forwarded += 1;
        }
        current_slot = 1 - current_slot;
        current = next;
    }

    stats.forwarded_tokens = forwarded * b;
    let ends: Vec<usize> = starts.iter().map(|start| start + forwarded).collect();
    for i in 0..b {
        for _ in 0..forwarded {
            members[i].checkpoint_lineage.record_append()?;
        }
        members[i]
            .zero_one
            .publish_gpu_decode_history(histories[i].clone());
        for block in &mut members[i].post_ple {
            block.publish_gpu_decode_length(ends[i]);
        }
        members[i].committed_length = ends[i];
        members[i].logits_ready = true;
        members[i].state_poisoned = false;
    }
    for (member, inputs) in members.iter_mut().zip(gpu_inputs) {
        member.gpu_inputs = Some(inputs);
    }
    stats.wall_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(stats)
}

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

fn read_token(token: &MetalTensor) -> i32 {
    unsafe {
        *token
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(token.offset as usize)
            .cast::<i32>()
    }
}

/// One host-sampled batch forward that also samples every member on the GPU at
/// the end of the command. The host writes the pending token, the device builds
/// the inputs and forward, the per-member `GpuTokenSampler` selects the next
/// token, and only the tokens are read back.
#[allow(clippy::too_many_arguments)]
pub(crate) fn step_gpu_tokens(
    ctx: &MetalContext,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    batch_inputs: &MetalTensor,
    batch_logits: &MetalTensor,
    projection: &crate::qwen4exp_metal::decode_projection::Scratch,
    members: &mut [&mut Qwen4ExpTextSessionMetalWorkspace],
    tokens: &[u32],
    device_tokens: &[MetalTensor],
    gpu_samplers: &[&GpuTokenSampler],
    uniforms: &[f32],
    diagnostic: &mut [f64; 3],
) -> Result<Vec<u32>, Qwen4ExpTextSessionError> {
    let started = Instant::now();
    let b = members.len();
    if b < 2
        || tokens.len() != b
        || device_tokens.len() < b
        || gpu_samplers.len() < b
        || uniforms.len() < b
    {
        return invalid("invalid GPU-token batch extents");
    }
    let config = weights.zero_one.ple_config;
    let vocab = weights.geometry.vocab_size();
    let positions: Vec<usize> = members.iter().map(|m| m.committed_length).collect();
    for (&p, m) in positions.iter().zip(members.iter()) {
        if p >= m.geometry.capacity() {
            return invalid("GPU-token batch position exceeds capacity");
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
    let result = (|| -> Result<Vec<u32>, Qwen4ExpTextSessionError> {
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
        for i in 0..b {
            gpu_samplers[i].encode(
                ctx,
                &enc,
                &members[i].logits,
                &device_tokens[i],
                uniforms[i],
            )?;
        }
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
            let token = read_token(&device_tokens[i]);
            if token < 0 || token as usize >= vocab {
                return invalid(format!(
                    "GPU sampler selected invalid token {token} for vocabulary {vocab}"
                ));
            }
            out.push(token as u32);
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
