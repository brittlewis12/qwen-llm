//! Reusable one-command-ahead host driver. Model adapters own inputs, causal
//! state and completion validation; this module owns submission and stop order.
use super::*;
use crate::sampling::{GreedySelection, Sampler};
use std::time::Instant;

#[derive(Clone, Debug, Default)]
pub struct GpuTokenLoopStats {
    pub sampled_tokens: usize,
    pub emitted_tokens: usize,
    pub forwarded_tokens: usize,
    pub encode_cpu_ms: f64,
    pub completion_wait_ms: f64,
    pub gpu_ms: f64,
    pub wall_ms: f64,
    pub terminal_token: Option<i32>,
    pub stopped_on_eos: bool,
    pub first_token_ready_ms: Option<f64>,
    pub first_token_callback_ms: Option<f64>,
    pub first_token_selection_ms: f64,
    pub transition_ms: f64,
    pub first_transition_ms: Option<f64>,
    /// Per-command host/GPU timeline, captured only under `perf-ablations`.
    #[cfg(feature = "perf-ablations")]
    pub timeline: Vec<TokenFrameTimeline>,
}

pub struct PreparedTokenStep<M> {
    pub command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    pub token: MetalTensor,
    pub metadata: M,
    readback: AsyncTokenReadback,
}
impl<M> PreparedTokenStep<M> {
    pub fn new(
        command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        token: MetalTensor,
        metadata: M,
    ) -> Result<Self, MetalError> {
        if command.status() != objc2_metal::MTLCommandBufferStatus::NotEnqueued {
            return Err(MetalError::Counter(
                "prepared token command was already submitted".into(),
            ));
        }
        let readback = AsyncTokenReadback::attach(&command, &token);
        Ok(Self {
            command,
            token,
            metadata,
            readback,
        })
    }
}

/// # Safety
/// Adapters must own model state exclusively, encode no CPU writes/reads of
/// in-flight GPU storage, and leave prepared commands uncommitted. Commands
/// execute in one causal queue. State stays unpublished until this driver has
/// drained and validated every committed command, including failure/unwind.
pub unsafe trait GpuTokenLoopDriver {
    type Metadata;
    type Error: From<MetalError>;
    fn vocab_size(&self) -> usize;
    fn prepare_first(
        &mut self,
        uniform: f32,
    ) -> Result<PreparedTokenStep<Self::Metadata>, Self::Error>;
    fn prepare_next(
        &mut self,
        input: &MetalTensor,
        ordinal: usize,
        uniform: f32,
    ) -> Result<PreparedTokenStep<Self::Metadata>, Self::Error>;
    fn started(&mut self);
    fn completed(
        &mut self,
        metadata: &Self::Metadata,
        input: Option<i32>,
    ) -> Result<(), Self::Error>;
}

struct InFlight(Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>);
impl Drop for InFlight {
    fn drop(&mut self) {
        if let Some(command) = self.0.take() {
            if command.status() != objc2_metal::MTLCommandBufferStatus::NotEnqueued {
                wait_unchecked(&command);
            }
        }
    }
}

/// The next command is encoded during GPU execution, then submitted only after
/// the current token's stop/stream callback accepts it. EOS and callback stop
/// never execute speculative model work. Completion blocks read dedicated slots.
pub fn run_preencoded_tokens<D: GpuTokenLoopDriver>(
    driver: &mut D,
    sampler: &mut Sampler,
    max_tokens: usize,
    eos: &[i32],
    mut emit: impl FnMut(&mut D, i32) -> bool,
) -> Result<GpuTokenLoopStats, D::Error> {
    let started = Instant::now();
    let mut stats = GpuTokenLoopStats::default();
    if max_tokens == 0 {
        return Ok(stats);
    }
    let mut running = InFlight(None);
    #[cfg(feature = "perf-ablations")]
    let mut current_encode_start = host_now_s();
    let mut current = driver.prepare_first(sampler.gpu_uniform(0))?;
    #[cfg(feature = "perf-ablations")]
    let mut current_encode_end = host_now_s();
    let mut input_token = None;
    driver.started();
    running.0 = Some(current.command.clone());
    let mut submitted = Instant::now();
    current.command.commit();
    #[cfg(feature = "perf-ablations")]
    let mut current_commit = host_now_s();
    loop {
        let encode_started = Instant::now();
        #[cfg(feature = "perf-ablations")]
        let next_encode_start = host_now_s();
        let next = if stats.sampled_tokens + 1 < max_tokens {
            Some(driver.prepare_next(
                &current.token,
                stats.forwarded_tokens + usize::from(input_token.is_some()),
                sampler.gpu_uniform(1),
            )?)
        } else {
            None
        };
        #[cfg(feature = "perf-ablations")]
        let next_encode_end = host_now_s();
        stats.encode_cpu_ms += encode_started.elapsed().as_secs_f64() * 1e3;
        let wait_started = Instant::now();
        #[cfg(feature = "perf-ablations")]
        let wait_start_s = host_now_s();
        #[cfg(feature = "perf-ablations")]
        if spin_wait_enabled() {
            spin_wait_until_completed(&current.command);
        }
        let selection = current.readback.receive();
        wait_completed(&current.command)?;
        let token = selection?;
        #[cfg(feature = "perf-ablations")]
        let wait_end_s = host_now_s();
        stats.completion_wait_ms += wait_started.elapsed().as_secs_f64() * 1e3;
        #[cfg(feature = "perf-ablations")]
        stats.timeline.push(TokenFrameTimeline {
            frame: stats.sampled_tokens,
            forward: input_token.is_some(),
            encode_start_s: current_encode_start,
            encode_end_s: current_encode_end,
            commit_s: current_commit,
            wait_start_s,
            wait_end_s,
            gpu_start_s: current.command.GPUStartTime(),
            gpu_end_s: current.command.GPUEndTime(),
        });
        running.0 = None;
        if token < 0 || token as usize >= driver.vocab_size() {
            return Err(MetalError::BadShape {
                kernel: "gpu_sampler",
                detail: if token == i32::MIN {
                    "all logits are negative infinity".into()
                } else {
                    format!(
                        "invalid selection {:?} for vocabulary {}",
                        GreedySelection::from_encoded(token),
                        driver.vocab_size()
                    )
                },
            }
            .into());
        }
        driver.completed(&current.metadata, input_token)?;
        if input_token.is_some() {
            stats.forwarded_tokens += 1;
            let elapsed = submitted.elapsed().as_secs_f64() * 1e3;
            stats.transition_ms += elapsed;
            stats.first_transition_ms.get_or_insert(elapsed);
        }
        let elapsed = (current.command.GPUEndTime() - current.command.GPUStartTime()) * 1e3;
        if elapsed.is_finite() && elapsed > 0. {
            stats.gpu_ms += elapsed;
        }
        if stats.sampled_tokens == 0 {
            stats.first_token_selection_ms = elapsed.max(0.);
        }
        sampler.commit_gpu_draw();
        stats.sampled_tokens += 1;
        stats.terminal_token = Some(token);
        stats
            .first_token_ready_ms
            .get_or_insert_with(|| started.elapsed().as_secs_f64() * 1e3);
        if eos.contains(&token) {
            stats.stopped_on_eos = true;
            break;
        }
        stats.emitted_tokens += 1;
        let keep_going = emit(driver, token);
        stats
            .first_token_callback_ms
            .get_or_insert_with(|| started.elapsed().as_secs_f64() * 1e3);
        if !keep_going {
            break;
        }
        let Some(next) = next else { break };
        input_token = Some(token);
        running.0 = Some(next.command.clone());
        submitted = Instant::now();
        next.command.commit();
        #[cfg(feature = "perf-ablations")]
        {
            current_encode_start = next_encode_start;
            current_encode_end = next_encode_end;
            current_commit = host_now_s();
        }
        current = next;
    }
    stats.wall_ms = started.elapsed().as_secs_f64() * 1e3;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sampling::SamplingConfig;
    struct Driver<'a> {
        ctx: &'a MetalContext,
        marker: MetalTensor,
        logits: MetalTensor,
        sampling: GpuTokenSampler,
        prepared: usize,
        fail_next: bool,
        first: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    }
    impl Driver<'_> {
        fn prepare(
            &mut self,
            ordinal: Option<usize>,
            uniform: f32,
        ) -> Result<PreparedTokenStep<Option<usize>>, MetalError> {
            let command = self.ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            if let Some(n) = ordinal {
                encode_fill_f32(self.ctx, &enc, &self.marker, (n + 1) as f32)?;
            }
            let token = MetalTensor::zeros_i32(self.ctx, vec![1])?;
            self.sampling
                .encode(self.ctx, &enc, &self.logits, &token, uniform)?;
            enc.end();
            if ordinal.is_none() {
                self.first = Some(command.clone());
            }
            PreparedTokenStep::new(command, token, ordinal)
        }
        fn marker(&self) -> f32 {
            unsafe { *self.marker.buffer.contents().as_ptr().cast::<f32>() }
        }
    }
    // Each frame owns a shared output and uses one serial GPU queue. The test
    // model changes its marker only when a model-step command is committed.
    unsafe impl GpuTokenLoopDriver for Driver<'_> {
        type Metadata = Option<usize>;
        type Error = MetalError;
        fn vocab_size(&self) -> usize {
            3
        }
        fn prepare_first(
            &mut self,
            u: f32,
        ) -> Result<PreparedTokenStep<Self::Metadata>, MetalError> {
            self.prepare(None, u)
        }
        fn prepare_next(
            &mut self,
            _: &MetalTensor,
            n: usize,
            u: f32,
        ) -> Result<PreparedTokenStep<Self::Metadata>, MetalError> {
            self.prepared += 1;
            if self.fail_next {
                return Err(MetalError::Counter("injected pre-encode failure".into()));
            }
            self.prepare(Some(n), u)
        }
        fn started(&mut self) {}
        fn completed(&mut self, n: &Option<usize>, input: Option<i32>) -> Result<(), MetalError> {
            assert_eq!(n.is_some(), input.is_some());
            Ok(())
        }
    }
    #[test]
    fn stop_eos_and_limit_discard_prepared_model_work_without_extra_draws() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        let config = SamplingConfig {
            temperature: 1.,
            top_k: 1,
            ..Default::default()
        };
        for (limit, eos, accept, expected_forwarded, expected_prepared) in [
            (8, vec![], false, 0, 1),
            (8, vec![1], true, 0, 1),
            (1, vec![], true, 0, 0),
            (3, vec![], true, 2, 2),
        ] {
            let mut driver = Driver {
                ctx: &ctx,
                marker: MetalTensor::zeros_f32(&ctx, vec![1]).unwrap(),
                logits: MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&[0f32, 2., 1.]),
                    vec![3],
                    GgmlType::F32,
                )
                .unwrap(),
                sampling: GpuTokenSampler::new(&ctx, 3, config).unwrap(),
                prepared: 0,
                fail_next: false,
                first: None,
            };
            let mut sampler = Sampler::new(config).unwrap();
            let mut published = Vec::new();
            let result =
                run_preencoded_tokens(&mut driver, &mut sampler, limit, &eos, |_, token| {
                    published.push(token);
                    accept
                })
                .unwrap();
            assert_eq!(result.forwarded_tokens, expected_forwarded);
            assert_eq!(driver.prepared, expected_prepared);
            assert_eq!(driver.marker(), expected_forwarded as f32);
            assert_eq!(sampler.draws(), result.sampled_tokens);
            assert_eq!(published.len(), result.emitted_tokens);
        }
    }
    #[test]
    fn failed_preencoding_drains_the_committed_command() {
        let Some(ctx) = test_support::metal_test_context() else {
            return;
        };
        let config = SamplingConfig::default();
        let mut driver = Driver {
            ctx: &ctx,
            marker: MetalTensor::zeros_f32(&ctx, vec![1]).unwrap(),
            logits: MetalTensor::from_bytes(
                &ctx,
                bytemuck::cast_slice(&[0f32, 2., 1.]),
                vec![3],
                GgmlType::F32,
            )
            .unwrap(),
            sampling: GpuTokenSampler::new(&ctx, 3, config).unwrap(),
            prepared: 0,
            fail_next: true,
            first: None,
        };
        let mut sampler = Sampler::new(config).unwrap();
        assert!(run_preencoded_tokens(&mut driver, &mut sampler, 3, &[], |_, _| true).is_err());
        assert_eq!(
            driver.first.as_ref().unwrap().status(),
            objc2_metal::MTLCommandBufferStatus::Completed
        );
        assert_eq!(sampler.draws(), 0);
    }
}
