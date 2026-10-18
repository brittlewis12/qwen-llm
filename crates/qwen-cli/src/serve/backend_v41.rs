//! Stateless GGUF HTTP backend. Forward borrows the caller's resident
//! weights/context; neither they nor the backend move to an HTTP worker.
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink, Headroom};
use super::items::{ServeError, ServeRequest};
use super::output_partition::GenerationEnd;
use super::request_profile::{OutputLimits, RequestProfile, sampling_with_defaults};
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::forward::{Forward, RequestMemoryEstimate};
use qwen_llm::metal::{MetalContext, MetalMemorySignals};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

use qwen_llm::tokenizer::NativeTokenizer;

pub(crate) enum RequestTokenizer {
    Native(NativeTokenizer),
    #[cfg(test)]
    Fixture(Tokenizer),
}
impl From<NativeTokenizer> for RequestTokenizer {
    fn from(value: NativeTokenizer) -> Self {
        Self::Native(value)
    }
}
#[cfg(test)]
impl From<Tokenizer> for RequestTokenizer {
    fn from(value: Tokenizer) -> Self {
        Self::Fixture(value)
    }
}
impl RequestTokenizer {
    #[cfg(test)]
    fn hf_mut(&mut self) -> &mut Tokenizer {
        match self {
            Self::Fixture(t) => t,
            Self::Native(_) => panic!("expected test tokenizer"),
        }
    }
    fn hf(&self) -> Option<&Tokenizer> {
        match self {
            Self::Native(_) => None,
            #[cfg(test)]
            Self::Fixture(t) => Some(t),
        }
    }
    fn vocab(&self) -> usize {
        match self {
            Self::Native(t) => t.n_vocab() as usize,
            #[cfg(test)]
            Self::Fixture(t) => t.get_vocab_size(true),
        }
    }
    fn encode(&self, prompt: &str) -> Result<Vec<u32>> {
        match self {
            Self::Native(t) => Ok(t
                .encode(prompt, false)?
                .into_iter()
                .map(|id| id as u32)
                .collect()),
            #[cfg(test)]
            Self::Fixture(t) => t
                .encode(prompt, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| anyhow::anyhow!(e.to_string())),
        }
    }
    fn decode(&self, ids: &[u32]) -> Result<String> {
        match self {
            Self::Native(t) => {
                Ok(t.try_decode(&ids.iter().map(|&id| id as i32).collect::<Vec<_>>())?)
            }
            #[cfg(test)]
            Self::Fixture(t) => t
                .decode(ids, false)
                .map_err(|e| anyhow::anyhow!(e.to_string())),
        }
    }
}

pub(super) struct Prepared {
    settings: Settings,
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
    let mut settings = prepared.settings;
    settings.model_id = listening.model_id.clone();
    let mut backend = Backend::new(&ctx, forward, settings);
    listening.serve(started.elapsed().as_secs_f64() * 1e3, &mut backend)
}

#[path = "backend_v41/io.rs"]
mod memory_io;

pub(crate) struct Backend<'a> {
    ctx: &'a MetalContext,
    forward: Forward<'a>,
    settings: Settings,
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
    let cpu = estimate.cpu_bytes.checked_add(extra_cpu).ok_or(
        qwen_llm::metal::MemoryAdmissionDenied {
            reason: qwen_llm::metal::MetalMemoryAdmissionReason::RequiredBytesOverflow,
            required_bytes: None,
            signals,
            working_set_headroom_bytes: signals
                .recommended_max_bytes
                .checked_sub(signals.current_allocated_bytes),
        },
    )?;
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
    tokenizer: RequestTokenizer,
    context: usize,
    limits: OutputLimits,
    sampling: SamplingConfig,
    eos: u32,
    model_id: String,
    headroom: Headroom,
}

impl Settings {
    pub(crate) fn new(
        tokenizer: impl Into<RequestTokenizer>,
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
        let mut tokenizer = tokenizer.into();
        ensure!(
            tokenizer.encode(deepseek_recipe_encoding::v4::EOS_TOKEN)? == [eos],
            "V4.1 tokenizer/config EOS mismatch"
        );
        ensure!(eos <= i32::MAX as u32, "EOS ID exceeds server token range");
        // ByteLevel emits at most one byte per encoded scalar (or the literal
        // token's bytes for added tokens). The shared ceiling adds 3x for U+FFFD.
        let max_piece_bytes = match &mut tokenizer {
            RequestTokenizer::Native(t) => t.max_decoded_piece_bytes(),
            #[cfg(test)]
            RequestTokenizer::Fixture(tokenizer) => {
                let decoder = serde_json::to_value(tokenizer.get_decoder())?;
                ensure!(
                    decoder["type"] == "ByteLevel",
                    "V4.1 serving requires the official ByteLevel decoder for byte budgeting"
                );
                tokenizer
                    .with_truncation(None)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?;
                tokenizer.with_padding(None);
                tokenizer
                    .get_vocab(true)
                    .keys()
                    .map(String::len)
                    .max()
                    .filter(|n| *n > 0)
                    .context("empty tokenizer vocabulary")?
            }
        };
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

    fn execute<S>(
        &self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
        mut admit: impl FnMut(usize, u64) -> std::result::Result<(), ServeError>,
        new_state: impl FnOnce(usize) -> std::result::Result<S, ServeError>,
        mut step: impl FnMut(&mut S, u32, &AtomicBool) -> std::result::Result<Vec<f32>, String>,
        mut now: impl FnMut() -> Duration,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let maximum = self.limits.max_tokens(request);
        // Check the output-only lower bound before tokenization or state allocation.
        super::decode_loop::required_forwards("V4.1", 1, maximum, self.context)?;
        let ids = self.tokenizer.encode(prompt).map_err(|e| {
            ServeError::invalid_request(Some("input"), format!("V4.1 tokenization: {e}"))
        })?;
        let vocab = self.tokenizer.vocab();
        if ids.is_empty() || ids.iter().any(|&id| id as usize >= vocab) {
            return Err(ServeError::invalid_request(
                Some("input"),
                "V4.1 prompt has no tokens or an out-of-vocabulary ID",
            )
            .into());
        }
        let required =
            super::decode_loop::required_forwards("V4.1", ids.len(), maximum, self.context)?;
        let ceiling = super::output_memory::tool_byte_ceiling(maximum, self.limits.max_piece_bytes)
            .ok_or_else(|| {
                ServeError::invalid_request(
                    Some("max_output_tokens"),
                    "V4.1 output byte budget overflow",
                )
            })?;
        let peak = qwen_llm::glm5_next_chat::checked_tool_block_peak_bytes(ceiling)
            .ok_or_else(|| super::output_memory::size_overflow("V4.1 output"))?;
        let reserve = u64::try_from(peak)
            .ok()
            .and_then(|n| caller_cpu_reserve(n, sink.transport_reserve_bytes(), vocab))
            .ok_or_else(|| super::output_memory::size_overflow("V4.1 transport"))?;
        admit(required, reserve)?;
        let mut sampler = Sampler::new(sampling_with_defaults(request, self.sampling))
            .map_err(|e| ServeError::invalid_request(None, format!("V4.1 sampler: {e}")))?;
        let flag = sink.cancellation_flag();
        let cancel = flag
            .as_deref()
            .unwrap_or_else(|| crate::shutdown::cancellation_flag());
        let check = |sink: &mut dyn GenerationSink| -> std::result::Result<(), BackendFailure> {
            sink.tick().map_err(BackendFailure::Aborted)?;
            if cancel.load(Ordering::Acquire)
                || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
            {
                return Err(BackendFailure::Aborted(io::Error::new(
                    io::ErrorKind::Interrupted,
                    "V4.1 request cancelled",
                )));
            }
            Ok(())
        };
        let forward_error = |message: String| {
            if cancel.load(Ordering::Acquire)
                || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
            {
                BackendFailure::Aborted(io::Error::new(io::ErrorKind::Interrupted, message))
            } else {
                BackendFailure::Serve(ServeError::server_error(format!("V4.1 forward: {message}")))
            }
        };
        check(sink)?;
        let mut state = new_state(required).map_err(|error| {
            if cancel.load(Ordering::Acquire)
                || crate::shutdown::cancellation_flag().load(Ordering::Acquire)
            {
                BackendFailure::Aborted(io::Error::new(io::ErrorKind::Interrupted, error.message))
            } else {
                BackendFailure::Serve(error)
            }
        })?;
        let prefill_start = now();
        let mut logits = Vec::new();
        for &token in &ids {
            check(sink)?;
            // Conservative re-admission against current signals before every
            // token. Never credit an upper-bound cache estimate as measured live
            // ownership: that could hide unrelated process growth.
            admit(required, reserve)?;
            logits = step(&mut state, token, cancel).map_err(&forward_error)?;
        }
        check(sink)?;
        let prefill_ms = (now() - prefill_start).as_secs_f64() * 1e3;
        let decode_start = now();
        let mut generated = Vec::new();
        let mut emitted = String::new();
        let mut decoder = self.tokenizer.hf().map(|t| t.decode_stream(false));
        let mut utf8 = super::utf8::Utf8Assembler::new();
        let mut end = GenerationEnd::TokenLimit;
        let mut transitions = 0;
        let mut transition_ms = 0.0;
        let mut first_transition_ms = None;
        let mut first_token_selection_ms = None;
        let mut first_token_ready_ms = None;
        let mut first_token_callback_ms = None;
        for index in 0..maximum {
            check(sink)?;
            super::output_memory::admit_growth(reserve, emitted.len() as u64, (self.headroom)())?;
            if logits.len() != vocab || !logits.iter().all(|v| v.is_finite()) {
                return Err(ServeError::server_error("V4.1 returned malformed logits").into());
            }
            let selection_start = now();
            let token = sampler
                .sample(&logits)
                .map_err(|e| ServeError::server_error(format!("V4.1 sampling: {e}")))?
                .token;
            first_token_selection_ms
                .get_or_insert_with(|| (now() - selection_start).as_secs_f64() * 1e3);
            first_token_ready_ms.get_or_insert_with(|| (now() - decode_start).as_secs_f64() * 1e3);
            generated.push(token);
            if token == self.eos as i32 {
                end = GenerationEnd::StopToken(token);
                break;
            }
            let piece =
                if let Some(decoder) = &mut decoder {
                    decoder
                        .step(token as u32)
                        .map_err(|e| ServeError::server_error(format!("V4.1 decode: {e}")))?
                } else {
                    match &self.tokenizer {
                        RequestTokenizer::Native(t) => Some(utf8.push(
                            t.try_decode_piece_bytes_exact(token).map_err(|e| {
                                ServeError::server_error(format!("V4.1 decode: {e}"))
                            })?,
                        )),
                        #[cfg(test)]
                        RequestTokenizer::Fixture(_) => unreachable!(),
                    }
                };
            if let Some(piece) = piece {
                if piece.len() > ceiling.saturating_sub(emitted.len()) {
                    return Err(super::output_memory::ceiling_exceeded(
                        "V4.1 decoded output",
                        ceiling,
                    )
                    .into());
                }
                super::output_memory::admit_growth(
                    reserve,
                    emitted.len() as u64,
                    (self.headroom)(),
                )?;
                sink.piece(piece.as_bytes())
                    .map_err(BackendFailure::Aborted)?;
                emitted.push_str(&piece);
            }
            check(sink)?;
            first_token_callback_ms
                .get_or_insert_with(|| (now() - decode_start).as_secs_f64() * 1e3);
            if index + 1 < maximum {
                let transition_start = now();
                admit(required, reserve)?;
                logits = step(&mut state, token as u32, cancel).map_err(&forward_error)?;
                check(sink)?;
                let elapsed = (now() - transition_start).as_secs_f64() * 1e3;
                transitions += 1;
                transition_ms += elapsed;
                first_transition_ms.get_or_insert(elapsed);
            }
        }
        check(sink)?;
        super::output_memory::admit_growth(reserve, emitted.len() as u64, (self.headroom)())?;
        let text_tokens: Vec<u32> = generated
            [..generated.len() - usize::from(matches!(end, GenerationEnd::StopToken(_)))]
            .iter()
            .map(|&id| id as u32)
            .collect();
        let complete = self
            .tokenizer
            .decode(&text_tokens)
            .map_err(|e| ServeError::server_error(format!("V4.1 decode tail: {e}")))?;
        if complete.len() > ceiling || !complete.starts_with(&emitted) {
            return Err(ServeError::server_error(
                "V4.1 incremental decoder violated output budget/prefix",
            )
            .into());
        }
        if complete.len() > emitted.len() {
            sink.piece(complete[emitted.len()..].as_bytes())
                .map_err(BackendFailure::Aborted)?;
        }
        check(sink)?;
        let generation = crate::GenerationResult {
            tokens: generated,
            wall_ms: (now() - decode_start).as_secs_f64() * 1e3,
            first_token_selection_ms: first_token_selection_ms.unwrap_or(0.),
            first_token_ready_ms,
            first_token_callback_ms,
            transitions,
            transition_ms,
            first_transition_ms,
            stop_reason: if matches!(end, GenerationEnd::StopToken(_)) {
                crate::StopReason::Eos
            } else {
                crate::StopReason::TokenLimit
            },
        };
        tracing::info!(target: "qwen_diag",
            "serve phases: family=deepseek_v41 prefill_ms={prefill_ms:.3} reused_tokens=0 reuse_source=none restore_ms=0.0 prefill_tokens={} prefill_mode=single_token_forward prefill_rows=1 decode_ms={:.3} required_forwards={required} capacity={} transitions={transitions} transition_ms={transition_ms:.3} ced=false",
            ids.len(), generation.wall_ms, self.context);
        let phases = serde_json::json!({
            "family":"deepseek_v41", "prefill_ms":prefill_ms, "prefill_tokens":ids.len(),
            "prefill_mode":"single_token_forward", "prefill_rows":1,
            "decode_ms":generation.wall_ms, "generated_tokens":generation.tokens.len(),
            "transitions":transitions, "transition_ms":transition_ms,
            "required_forwards":required, "capacity":self.context, "reused_tokens":0, "ced":false,
            "cache_policy":"fresh_request",
        });
        let mut outcome = super::outcome::finish_generation(ids.len(), &generation, 0, 0.);
        outcome.stats.as_mut().expect("shared stats").phases = Some(phases);
        Ok(outcome)
    }
}

impl<'a> Backend<'a> {
    pub(crate) fn new(ctx: &'a MetalContext, forward: Forward<'a>, settings: Settings) -> Self {
        Self {
            ctx,
            forward,
            settings,
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
    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let epoch = Instant::now();
        let mut observations = request.echo_stats.then(memory_io::Observations::default);
        let mut outcome = self.settings.execute(
            request,
            prompt,
            sink,
            |positions, reserve| {
                let estimate = self
                    .forward
                    .estimate_request_memory_with_gpu_step(positions)
                    .map_err(|e| ServeError::server_error(format!("V4.1 request estimate: {e}")))?;
                admit_request_memory(estimate, reserve, self.ctx.memory_signals())
                    .map_err(|e| super::transport_memory::memory_refusal("V4.1 request", &e))
            },
            |capacity| {
                self.forward
                    .new_state_with_gpu_step(capacity)
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
        }
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests;

fn state_error(error: qwen_llm::deepseek_v41::forward::StateError) -> ServeError {
    match error.memory_admission() {
        Some(denied) => super::transport_memory::memory_refusal("V4.1 state", denied),
        None => ServeError::server_error(format!("V4.1 state: {error}")),
    }
}
