//! Flash-Next adapter for the reusable pre-encoded token driver. The root stays
//! exclusively borrowed and poisoned until every committed frame is validated.
use super::gpu_inputs::GpuInputs;
use super::*;
pub use crate::metal::GpuTokenLoopStats as Qwen4ExpGpuDecodeStats;
use crate::metal::{GpuTokenLoopDriver, GpuTokenSampler, PreparedTokenStep, run_preencoded_tokens};
use crate::sampling::Sampler;
use objc2_metal::MTLCommandQueue;

struct Driver<'w, 'ctx, 'weights, 'tensors> {
    workspace: &'w mut Qwen4ExpTextSessionMetalWorkspace,
    ctx: &'ctx MetalContext,
    weights: &'weights Qwen4ExpTextSessionMetalWeights<'tensors>,
    inputs: Option<GpuInputs>,
    sampling: GpuTokenSampler,
    history: PleHistory,
    start: usize,
}
impl Drop for Driver<'_, '_, '_, '_> {
    fn drop(&mut self) {
        // run_preencoded_tokens drains on error/unwind before this adapter can
        // return its resources. Root poison is cleared only by publish below.
        self.workspace.gpu_inputs = self.inputs.take();
    }
}
impl Driver<'_, '_, '_, '_> {
    fn prepare(
        &mut self,
        uniform: f32,
        input: Option<(&MetalTensor, usize)>,
    ) -> Result<PreparedTokenStep<Option<usize>>, Qwen4ExpTextSessionError> {
        let command = self.ctx.queue.commandBuffer().ok_or_else(|| {
            Qwen4ExpTextSessionError::Invalid("GPU loop command allocation failed".into())
        })?;
        let position = input.map(|(_, p)| p);
        let mut body = if input.is_some() && self.workspace.causal_executor {
            KernelEncoder::begin_causal(&command)
        } else {
            KernelEncoder::begin(&command)
        };
        if let Some((token, position)) = input {
            // The indirect control writes are ordered explicitly; reflection
            // cannot see argument-buffer pointees. Keep this prefix outside
            // the recorder and fence it before the first model wave.
            if body.causal_mode {
                body.cancel_causal_scope();
            }
            self.inputs
                .as_ref()
                .expect("forward frame has GPU inputs")
                .encode(self.ctx, &body, token, position)?;
            if body.causal_mode {
                body.begin_causal_scope();
            }
            // The adapter owns the full workspace until the reusable driver's
            // in-flight guard drains. Inputs/control writes are GPU dispatches;
            // preparing another command writes no in-flight tensor on the CPU.
            unsafe {
                body.delegate_session_owner(position);
            }
            encode_step(
                self.ctx,
                &body,
                0,
                position,
                self.history.clone(),
                self.weights,
                self.workspace,
                None,
                None,
                None,
            )?;
            if body.causal_mode {
                body.finish_causal_scope(self.ctx, false)?;
            }
        }
        let token = MetalTensor::zeros_i32(self.ctx, vec![1])?;
        if body.causal_mode {
            body.begin_causal_scope();
        }
        self.sampling
            .encode(self.ctx, &body, &self.workspace.logits, &token, uniform)?;
        if body.causal_mode {
            body.finish_causal_scope(self.ctx, false)?;
        }
        body.end();
        Ok(PreparedTokenStep::new(command, token, position)?)
    }
    fn logits(&self) -> &[f32] {
        // Called only after completion and before the prepared next command is
        // committed. The callback cannot retain the borrow or access the owner.
        unsafe {
            std::slice::from_raw_parts(
                self.workspace
                    .logits
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(self.workspace.logits.offset as usize)
                    .cast::<f32>(),
                self.workspace.geometry.vocab_size(),
            )
        }
    }
}
// SAFETY: root ownership is exclusive, all child CPU staging is bypassed, and
// the generic driver orders commits and drains on every return/unwind path.
unsafe impl GpuTokenLoopDriver for Driver<'_, '_, '_, '_> {
    type Metadata = Option<usize>;
    type Error = Qwen4ExpTextSessionError;
    fn vocab_size(&self) -> usize {
        self.workspace.geometry.vocab_size()
    }
    fn prepare_first(
        &mut self,
        uniform: f32,
    ) -> Result<PreparedTokenStep<Self::Metadata>, Self::Error> {
        self.prepare(uniform, None)
    }
    fn prepare_next(
        &mut self,
        input: &MetalTensor,
        ordinal: usize,
        uniform: f32,
    ) -> Result<PreparedTokenStep<Self::Metadata>, Self::Error> {
        self.prepare(uniform, Some((input, self.start + ordinal)))
    }
    fn started(&mut self) {
        self.workspace.state_poisoned = true;
        self.workspace.logits_ready = false;
    }
    fn completed(
        &mut self,
        position: &Option<usize>,
        input: Option<i32>,
    ) -> Result<(), Self::Error> {
        if let Some(position) = *position {
            self.workspace.zero_one.validate_gpu_decode_status()?;
            for block in &self.workspace.post_ple {
                block.validate_gpu_decode_status(position)?;
            }
            self.history = self
                .history
                .advanced(
                    self.weights.zero_one.ple_config,
                    input.expect("forward frame has input") as u32,
                    position as u64,
                )?
                .0;
        }
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn generate(
    ctx: &MetalContext,
    table: PleIq4NlTable<'_>,
    weights: &Qwen4ExpTextSessionMetalWeights<'_>,
    workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
    sampler: &mut Sampler,
    max_tokens: usize,
    eos: &[i32],
    mut emit: impl FnMut(i32, &[f32]) -> bool,
) -> Result<Qwen4ExpGpuDecodeStats, Qwen4ExpTextSessionError> {
    let started = Instant::now();
    if crate::metal::dispatch_profile_records() {
        // Per-dispatch profiling rebuilds encoders and intentionally removes
        // concurrency. Its replay cannot represent indirect control writes.
        return invalid(
            "GPU feedback requires ordinary dispatches; use token forwards while profiling",
        );
    }
    workspace.require_idle()?;
    if workspace.is_poisoned() || !workspace.logits_ready {
        return invalid("GPU generation requires healthy completed logits");
    }
    if max_tokens == 0 {
        return Ok(Qwen4ExpGpuDecodeStats::default());
    }
    let start = workspace.committed_length;
    if max_tokens - 1 > workspace.geometry.capacity().saturating_sub(start) {
        return invalid("GPU generation exceeds session capacity");
    }
    let check = ctx.queue.commandBuffer().ok_or_else(|| {
        Qwen4ExpTextSessionError::Invalid("GPU preflight command allocation failed".into())
    })?;
    let check_enc = KernelEncoder::begin(&check);
    if max_tokens > 1 {
        validate_and_preflight(ctx, &check_enc, 0, start, weights, workspace)?;
    }
    check_enc.end();
    drop(check);
    let inputs = match workspace.gpu_inputs.take() {
        Some(inputs) => Some(inputs),
        None if max_tokens > 1 => Some(GpuInputs::new(
            ctx,
            table,
            weights.zero_one.ple_config,
            workspace,
        )?),
        None => None,
    };
    if let Some(inputs) = &inputs {
        inputs.reset_history(workspace);
    }
    let sampling = GpuTokenSampler::new(ctx, workspace.geometry.vocab_size(), sampler.config())?;
    let history = workspace.zero_one.gpu_decode_history();
    let mut driver = Driver {
        workspace,
        ctx,
        weights,
        inputs,
        sampling,
        history,
        start,
    };
    let setup_ms = started.elapsed().as_secs_f64() * 1e3;
    let mut stats =
        run_preencoded_tokens(&mut driver, sampler, max_tokens, eos, |driver, token| {
            emit(token, driver.logits())
        })?;
    let end = start + stats.forwarded_tokens;
    for _ in 0..stats.forwarded_tokens {
        driver.workspace.checkpoint_lineage.record_append()?;
    }
    driver
        .workspace
        .zero_one
        .publish_gpu_decode_history(driver.history.clone());
    for block in &mut driver.workspace.post_ple {
        block.publish_gpu_decode_length(end);
    }
    driver.workspace.committed_length = end;
    driver.workspace.logits_ready = true;
    driver.workspace.state_poisoned = false;
    stats.first_token_ready_ms = stats.first_token_ready_ms.map(|v| v + setup_ms);
    stats.first_token_callback_ms = stats.first_token_callback_ms.map(|v| v + setup_ms);
    stats.wall_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(stats)
}
