//! Stateless safetensors HTTP backend. Forward borrows the caller's resident
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

pub(crate) struct Backend<'a> {
    ctx: &'a MetalContext,
    forward: Forward<'a>,
    settings: Settings,
    gpu_step: bool,
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
    tokenizer: Tokenizer,
    context: usize,
    limits: OutputLimits,
    sampling: SamplingConfig,
    eos: u32,
    model_id: String,
    headroom: Headroom,
}

impl Settings {
    pub(crate) fn new(
        mut tokenizer: Tokenizer,
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
            tokenizer.token_to_id(deepseek_recipe_encoding::v4::EOS_TOKEN) == Some(eos),
            "V4.1 tokenizer/config EOS mismatch"
        );
        ensure!(eos <= i32::MAX as u32, "EOS ID exceeds server token range");
        // ByteLevel emits at most one byte per encoded scalar (or the literal
        // token's bytes for added tokens). The shared ceiling adds 3x for U+FFFD.
        let decoder = serde_json::to_value(tokenizer.get_decoder())?;
        ensure!(
            decoder["type"] == "ByteLevel",
            "V4.1 serving requires the official ByteLevel decoder for byte budgeting"
        );
        tokenizer
            .with_truncation(None)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        tokenizer.with_padding(None);
        let max_piece_bytes = tokenizer
            .get_vocab(true)
            .keys()
            .map(String::len)
            .max()
            .filter(|n| *n > 0)
            .context("empty tokenizer vocabulary")?;
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
        new_state: impl FnOnce(usize) -> std::result::Result<S, String>,
        mut step: impl FnMut(&mut S, u32, &AtomicBool) -> std::result::Result<Vec<f32>, String>,
        mut now: impl FnMut() -> Duration,
    ) -> std::result::Result<GenerationOutcome, BackendFailure> {
        let maximum = self.limits.max_tokens(request);
        // Check the output-only lower bound before tokenization or state allocation.
        super::decode_loop::required_forwards("V4.1", 1, maximum, self.context)?;
        let ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| {
                ServeError::invalid_request(Some("input"), format!("V4.1 tokenization: {e}"))
            })?
            .get_ids()
            .to_vec();
        let vocab = self.tokenizer.get_vocab_size(true);
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
        let mut state = new_state(required).map_err(&forward_error)?;
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
        let mut decoder = self.tokenizer.decode_stream(false);
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
            if let Some(piece) = decoder
                .step(token as u32)
                .map_err(|e| ServeError::server_error(format!("V4.1 decode: {e}")))?
            {
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
            .decode(&text_tokens, false)
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
            gpu_step: false,
        }
    }

    pub(crate) fn with_gpu_step(mut self, enabled: bool) -> Self {
        self.gpu_step = enabled;
        self
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
        self.settings.execute(
            request,
            prompt,
            sink,
            |positions, reserve| {
                let estimate = if self.gpu_step {
                    self.forward
                        .estimate_request_memory_with_gpu_step(positions)
                } else {
                    self.forward.estimate_request_memory(positions)
                }
                .map_err(|e| ServeError::server_error(format!("V4.1 request estimate: {e}")))?;
                admit_request_memory(estimate, reserve, self.ctx.memory_signals())
                    .map_err(|e| super::transport_memory::memory_refusal("V4.1 request", &e))
            },
            |capacity| {
                if self.gpu_step {
                    self.forward.new_state_with_gpu_step(capacity)
                } else {
                    self.forward.new_state(capacity)
                }
            },
            |state, token, cancel| self.forward.step(state, token, Some(cancel)),
            || epoch.elapsed(),
        )
    }
}

#[cfg(test)]
mod tests;
