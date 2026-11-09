//! Adapter to the shared preencoded token loop. All mutable storage belongs to
//! this exclusively borrowed session; prepared work never writes on the CPU.
use super::*;
use crate::metal::{
    GpuTokenLoopDriver, GpuTokenLoopStats, GpuTokenSampler, PreparedTokenStep,
    run_preencoded_tokens,
};
use crate::sampling::Sampler;
use objc2_metal::MTLSize;
use std::time::Instant;
crate::env_flag!(
    /// Default-on M5 GLM single-stream GPU token feedback with asynchronous streaming and native-batch handoff. Set 0 to restore CPU sampling and token-at-a-time submission. Scalar model arithmetic and native B>1 rounds are unchanged.
    default_on gpu_token_loop_requested,
    "QWEN_GLM53_GPU_TOKEN_LOOP"
);
/// Frontend policy for the measured M5 token-feedback path. Explicit generation
/// APIs remain available for qualification independently of this rollback.
pub fn gpu_token_loop_enabled(ctx: &MetalContext) -> bool {
    gpu_token_loop_requested() && ctx.device_facts().gpu_families.apple10
}
struct Driver<'a, 'w> {
    ctx: &'a MetalContext,
    session: &'a mut Glm5NextSession<'w>,
    initial: MetalTensor,
    sampling: GpuTokenSampler,
    start: usize,
    forwarded: bool,
}
impl Driver<'_, '_> {
    fn prepare(
        &mut self,
        uniform: f32,
        input: Option<(&MetalTensor, usize)>,
    ) -> Result<PreparedTokenStep<Option<usize>>> {
        let mut command = self
            .ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no GPU loop command".into()))?;
        let mut enc = KernelEncoder::begin(&command);
        if let Some((input, position)) = input {
            let c = &self.session.weights.config;
            if position + 1 >= c.sparse_frontier() as usize {
                let s = self.session.sparse.as_ref().ok_or_else(|| {
                    Glm5NextMetalError::Invalid("GPU loop sparse scratch missing".into())
                })?;
                let args = [
                    (position + 1) as u32,
                    c.indexer_pool,
                    c.block_count(MixerKind::Mla) as u32,
                ];
                enc.set_pipeline(&self.ctx.pipeline("kernel_glm_gpu_controls")?);
                enc.set_bytes(0, &args);
                enc.set_tensor(1, &s.visible_pools);
                enc.set_tensor(2, &s.visible_rows);
                enc.set_tensor(3, &s.select_status);
                enc.dispatch(
                    MTLSize {
                        width: (args[2] as usize).max(1).div_ceil(64),
                        height: 1,
                        depth: 1,
                    },
                    MTLSize {
                        width: 64,
                        height: 1,
                        depth: 1,
                    },
                );
            }
            let mut discard = |_: Glm5NextSiteCapture, _: &[f32]| {};
            let mut hooks = interventions::Hooks {
                module: &[],
                captures: &[],
                observer: &mut discard,
            };
            self.session.encode_token_body(
                self.ctx,
                &mut command,
                &mut enc,
                input,
                position,
                true,
                &[],
                &mut |_, _, _| {},
                &mut hooks,
                None,
            )?;
        }
        let output = MetalTensor::zeros_i32(self.ctx, vec![1])?;
        let logits = if input.is_some() {
            &self.session.s.logits
        } else {
            &self.initial
        };
        self.sampling
            .encode(self.ctx, &enc, logits, &output, uniform)?;
        enc.end();
        Ok(PreparedTokenStep::new(
            command,
            output,
            input.map(|(_, p)| p),
        )?)
    }
    fn validate(&self, position: usize) -> Result<()> {
        for (layer, route) in self.session.routes.iter().enumerate() {
            if let Some(route) = route {
                let status = read_i32(&route.status)?[0];
                if status != ROUTE_STATUS_READY {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "route",
                        block: layer,
                        row: None,
                        status,
                    });
                }
            }
        }
        let c = &self.session.weights.config;
        if position + 1 >= c.sparse_frontier() as usize {
            for (index, &status) in read_i32(&self.session.sparse.as_ref().unwrap().select_status)?
                .iter()
                .enumerate()
            {
                if status != crate::metal::SELECT_STATUS_OK {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "sparse selection",
                        block: mla_block(c, index),
                        row: None,
                        status,
                    });
                }
            }
        }
        Ok(())
    }
    fn logits(&self) -> &[f32] {
        let t = if self.forwarded {
            &self.session.s.logits
        } else {
            &self.initial
        };
        // Current command completed and the prepared next command is uncommitted.
        unsafe {
            std::slice::from_raw_parts(
                t.buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(t.offset as usize)
                    .cast::<f32>(),
                self.session.weights.config.vocab_size as usize,
            )
        }
    }
}
// SAFETY: the session is exclusively borrowed, no encoding writes host data,
// queue order is causal and the shared driver drains on error and unwind.
unsafe impl GpuTokenLoopDriver for Driver<'_, '_> {
    type Metadata = Option<usize>;
    type Error = Glm5NextMetalError;
    fn vocab_size(&self) -> usize {
        self.session.weights.config.vocab_size as usize
    }
    fn prepare_first(&mut self, u: f32) -> Result<PreparedTokenStep<Self::Metadata>> {
        self.prepare(u, None)
    }
    fn prepare_next(
        &mut self,
        t: &MetalTensor,
        ordinal: usize,
        u: f32,
    ) -> Result<PreparedTokenStep<Self::Metadata>> {
        self.prepare(u, Some((t, self.start + ordinal)))
    }
    fn started(&mut self) {
        self.session.poisoned = true;
    }
    fn completed(&mut self, p: &Self::Metadata, _: Option<i32>) -> Result<()> {
        if let Some(p) = p {
            self.validate(*p)?;
            self.session.position = p + 1;
            self.forwarded = true;
        }
        Ok(())
    }
}
impl Glm5NextSession<'_> {
    /// Generate from the completed current-position head supplied by the caller.
    /// The final selection remains unconsumed, as in ordinary sampled decode.
    pub fn generate_gpu_tokens(
        &mut self,
        ctx: &MetalContext,
        initial_logits: &[f32],
        sampler: &mut Sampler,
        max_tokens: usize,
        eos: &[i32],
        mut emit: impl FnMut(i32) -> bool,
    ) -> Result<GpuTokenLoopStats> {
        self.generate_gpu_tokens_observed(
            ctx,
            initial_logits,
            sampler,
            max_tokens,
            eos,
            |token, _| emit(token),
        )
    }
    /// Like `generate_gpu_tokens`, borrowing completed heads for diagnostics.
    pub fn generate_gpu_tokens_observed(
        &mut self,
        ctx: &MetalContext,
        initial_logits: &[f32],
        sampler: &mut Sampler,
        max_tokens: usize,
        eos: &[i32],
        mut emit: impl FnMut(i32, &[f32]) -> bool,
    ) -> Result<GpuTokenLoopStats> {
        let started = Instant::now();
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        if max_tokens == 0 {
            return Ok(GpuTokenLoopStats::default());
        }
        if initial_logits.len() != self.weights.config.vocab_size as usize
            || !initial_logits.iter().all(|v| v.is_finite())
        {
            return invalid("GPU loop requires a finite full current-position head");
        }
        if max_tokens - 1 > self.capacity.saturating_sub(self.position) {
            return invalid("GPU loop exceeds capacity");
        }
        if self.stage_recorder.borrow().is_some()
            || self.decode_timing.get().is_some()
            || crate::metal::dispatch_profile_records()
        {
            return invalid("GPU loop requires ordinary unsampled dispatches");
        }
        self.prepare_decode(ctx)?;
        let initial = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(initial_logits),
            vec![initial_logits.len() as u64],
            GgmlType::F32,
        )?;
        let sampling = GpuTokenSampler::new(ctx, initial_logits.len(), sampler.config())?;
        let start = self.position;
        let mut driver = Driver {
            ctx,
            session: self,
            initial,
            sampling,
            start,
            forwarded: false,
        };
        let setup_ms = started.elapsed().as_secs_f64() * 1e3;
        let mut stats =
            run_preencoded_tokens(&mut driver, sampler, max_tokens, eos, |driver, token| {
                emit(token, driver.logits())
            })?;
        driver.session.poisoned = false;
        stats.first_token_ready_ms = stats.first_token_ready_ms.map(|v| v + setup_ms);
        stats.first_token_callback_ms = stats.first_token_callback_ms.map(|v| v + setup_ms);
        stats.wall_ms = started.elapsed().as_secs_f64() * 1e3;
        Ok(stats)
    }
}
