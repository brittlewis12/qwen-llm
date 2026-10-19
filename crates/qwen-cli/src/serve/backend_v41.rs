//! Exact-prefix GGUF HTTP backend. Forward borrows the caller's resident
//! weights/context; neither they nor the backend move to an HTTP worker.
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink, Headroom};
use super::items::{ServeError, ServeRequest};
#[cfg(test)]
use super::output_partition::GenerationEnd;
use super::request_profile::{OutputLimits, RequestProfile, sampling_with_defaults};
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::forward::{Forward, RequestMemoryEstimate, StateError};
use qwen_llm::metal::{MetalContext, MetalMemorySignals};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use qwen_llm::tokenizer::NativeTokenizer;

pub(super) struct Prepared {
    settings: Settings,
    snapshot_mib: Option<u64>,
    snapshot_policy: qwen_llm::snapshot_policy::SnapshotPolicyConfig,
}
impl Prepared {
    pub(super) fn new(
        gguf: &qwen_llm::gguf::GgufFile,
        invocation: &crate::cli::ServeInvocation,
    ) -> Result<Self> {
        let plan = qwen_llm::deepseek_v41::gguf::plan(gguf)?;
        ensure!(
            invocation.lens_data_dir.is_none() && invocation.web_root.is_none(),
            "DeepSeek V4.1 has no Lens backend"
        );
        let limits = super::resolve_serve_limits(
            super::profile(qwen_llm::model_family::ModelFamily::DeepSeekV41),
            Some(usize::try_from(plan.engram_hash.max_position)?),
            invocation.max_context_tokens,
            invocation.max_tokens,
        )?
        .context("V4.1 requires fixed request limits")?;
        let tokenizer = NativeTokenizer::from_gguf(gguf)?;
        ensure!(
            tokenizer.n_vocab() as usize == plan.config.vocab,
            "V4.1 tokenizer/model vocabulary mismatch"
        );
        let eos = tokenizer.eos().context("V4.1 GGUF has no EOS")? as u32;
        Ok(Self {
            snapshot_mib: invocation.snapshot_cache_mib,
            snapshot_policy: invocation.snapshot_policy,
            settings: Settings::new(
                tokenizer,
                limits.context_tokens,
                limits.max_tokens,
                crate::deepseek_v41::default_sampling(crate::release_sampling::fresh_seed()),
                eos,
                String::new(),
            )?,
        })
    }
}

pub(super) fn start(
    prepared: Prepared,
    gguf: qwen_llm::gguf::GgufFile,
    listening: super::Listening,
) -> Result<()> {
    crate::shutdown::checkpoint()?;
    let ctx = MetalContext::new()?;
    let started = Instant::now();
    let weights = crate::deepseek_v41::load(&ctx, std::sync::Arc::new(gguf))?;
    let forward =
        Forward::from_resident(&ctx, &weights, weights.token_map.clone(), weights.raw_pad)
            .map_err(anyhow::Error::msg)?;
    let plan = super::SnapshotCachePlan::resolve(
        prepared.snapshot_mib,
        prepared.snapshot_policy,
        ctx.memory_signals(),
    )?;
    tracing::info!("serve: V4.1 snapshot cache: {plan}");
    let mut settings = prepared.settings;
    settings.model_id = listening.model_id.clone();
    let mut backend = Backend::new(&ctx, forward, settings);
    backend.reuse = reuse::Store::new(plan.bytes, plan.policy);
    listening.serve(started.elapsed().as_secs_f64() * 1e3, &mut backend)
}

mod execution;
#[path = "backend_v41/io.rs"]
mod memory_io;
mod reuse;

pub(crate) struct Backend<'a> {
    ctx: &'a MetalContext,
    forward: Forward<'a>,
    settings: Settings,
    reuse: reuse::Store,
}

/// Caller-retained logits and sampler candidates/weights/sort storage, separate
/// from the forward's request estimate. CPU-only bytes never consume Metal headroom.
pub(crate) fn caller_cpu_reserve(output_peak: u64, transport: u64, vocab: usize) -> Option<u64> {
    output_peak
        .checked_add(transport)?
        .checked_add(u64::try_from(vocab).ok()?.checked_mul(64)?)
}

pub(crate) fn admit_request_memory(
    estimate: RequestMemoryEstimate,
    extra_cpu: u64,
    signals: MetalMemorySignals,
) -> std::result::Result<(), qwen_llm::metal::MemoryAdmissionDenied> {
    let cpu = estimate.cpu_bytes.checked_add(extra_cpu).ok_or_else(|| {
        qwen_llm::metal::MemoryAdmissionDenied::new(
            qwen_llm::metal::MetalMemoryAdmissionReason::RequiredBytesOverflow,
            None,
            signals,
            signals
                .recommended_max_bytes
                .checked_sub(signals.current_allocated_bytes),
        )
        .expect("required byte overflow refuses admission")
    })?;
    let decision = qwen_llm::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        estimate.metal_bytes,
        cpu,
        0,
        signals,
        true,
    );
    match decision.refusal() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

pub(crate) struct Settings {
    tokenizer: NativeTokenizer,
    context: usize,
    limits: OutputLimits,
    sampling: SamplingConfig,
    eos: u32,
    model_id: String,
    headroom: Headroom,
}

impl Settings {
    pub(crate) fn new(
        tokenizer: NativeTokenizer,
        context: usize,
        default_max: usize,
        sampling: SamplingConfig,
        eos: u32,
        model_id: String,
    ) -> Result<Self> {
        ensure!(
            context > 0 && default_max > 0 && default_max <= context,
            "V4.1 default output limit must fit positive context"
        );
        sampling.validate()?;
        ensure!(
            tokenizer.encode(deepseek_recipe_encoding::v4::EOS_TOKEN, false)? == [eos as i32],
            "V4.1 tokenizer/config EOS mismatch"
        );
        ensure!(eos <= i32::MAX as u32, "EOS ID exceeds server token range");
        let max_piece_bytes = tokenizer.max_decoded_piece_bytes();
        Ok(Self {
            tokenizer,
            context,
            limits: OutputLimits {
                default_max_tokens: default_max,
                max_piece_bytes,
            },
            sampling,
            eos,
            model_id,
            headroom: qwen_llm::metal::MetalContext::process_limit_bytes_remaining,
        })
    }

    fn profile(&self) -> RequestProfile {
        RequestProfile::DeepSeekV41 {
            limits: self.limits,
            sampling: Some(self.sampling),
        }
    }
}

impl<'a> Backend<'a> {
    pub(crate) fn new(ctx: &'a MetalContext, forward: Forward<'a>, settings: Settings) -> Self {
        Self {
            ctx,
            forward,
            settings,
            reuse: reuse::Store::new(0, Default::default()),
        }
    }
}

impl GenerationBackend for Backend<'_> {
    fn model_id(&self) -> &str {
        &self.settings.model_id
    }
    fn request_profile(&self) -> RequestProfile {
        self.settings.profile()
    }
    fn idle(&mut self) {
        self.reuse.sweep();
    }
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        self.generate_prompt(request, prompt, None, sink)
    }

    fn generate_prepared(
        &mut self,
        prepared: std::sync::Arc<super::http::PreparedResponse>,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        self.generate_prompt(
            prepared.request(),
            &prepared.prompt,
            prepared.boundaries,
            sink,
        )
    }
}

impl Backend<'_> {
    fn generate_prompt(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        boundaries: Option<super::http::PromptBoundaries>,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let epoch = Instant::now();
        self.reuse.sweep();
        let mut observations = request.echo_stats.then(memory_io::Observations::default);
        let mut prefill_observations = request
            .echo_stats
            .then(memory_io::ChunkObservations::default);
        let mut outcome = self.settings.execute(
            request,
            prompt,
            boundaries,
            sink,
            |positions, reserve| {
                let estimate = self
                    .forward
                    .estimate_request_memory_with_gpu_step(positions)
                    .map_err(|e| StateError::Invalid(format!("V4.1 request estimate: {e}")))?;
                admit_request_memory(estimate, reserve, self.ctx.memory_signals())
                    .map_err(StateError::MemoryAdmissionDenied)
            },
            |capacity| {
                self.forward
                    .new_state_with_gpu_step(capacity)
                    .map_err(state_error)
            },
            &mut reuse::Session {
                forward: &self.forward,
                store: &mut self.reuse,
                ctx: self.ctx,
            },
            |state, tokens, cancel, with_logits| {
                memory_io::observe_chunk(
                    prefill_observations.as_mut(),
                    state.position(),
                    tokens.len(),
                    || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
                    || {
                        if with_logits {
                            self.forward.prefill(state, tokens, Some(cancel))
                        } else {
                            self.forward
                                .advance_prefill(state, tokens, Some(cancel))
                                .map(|()| Vec::new())
                        }
                    },
                )
                .map_err(state_error)
            },
            |state, token, cancel| {
                let position = state.position();
                memory_io::observe(
                    observations.as_mut(),
                    position,
                    || qwen_llm::pid_metrics::PidSnapshot::now().ok(),
                    || self.forward.step(state, token, Some(cancel)),
                )
            },
            || epoch.elapsed(),
        )?;
        if let Some(observations) = observations {
            let phases = outcome
                .stats
                .as_mut()
                .expect("shared stats")
                .phases
                .as_mut()
                .expect("V4.1 phases");
            phases["process_token_io"] = observations.json();
            phases["process_prefill_io"] = prefill_observations.expect("stats enabled").json();
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod reuse_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod tokenizer_fixture;

fn state_error(error: qwen_llm::deepseek_v41::forward::StateError) -> ServeError {
    match error.memory_admission() {
        Some(denied) => super::transport_memory::memory_refusal("V4.1 state", denied),
        None => ServeError::server_error(format!("V4.1 state: {error}")),
    }
}
