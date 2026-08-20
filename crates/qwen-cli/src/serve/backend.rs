//! Engine-backed [`GenerationBackend`]: resident model, RAM prefix cache
//! with dual-boundary capture, chunked prefill with cancellation ticks,
//! and the serial decode loop streaming pieces into the transport sink.
//!
//! Residency is the TTFT mechanism (S0 F2); dual capture — prompt boundary
//! and completed turn — makes the server's next-turn path independent of
//! client echo policy (S0 F1, SERVE.md review R2: RAM always, durable
//! stays on the existing CLI shadowing policy and is out of this slice).

use super::events::{ServeStats, StopReason, Usage};
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{QwenTemplate, ServeError, ServeRequest};
use super::utf8::Utf8Assembler;
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{Model, open_dflash_drafter};
use qwen_llm::metal::{MetalTensor, evaluate_metal_memory_admission};
use qwen_llm::metal_dflash::{
    MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession, MetalDFlashVerifyScratch,
    PrefillScratchConfig, plan_prefill_scratch_with_matrix_max_pos_configured,
};
use qwen_llm::metal_forward::MetalForward;
use qwen_llm::runtime::{LoadedModel, Sequence, SequenceConfig};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

const DFLASH_FIXED_SCRATCH_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
const SERIAL_TAIL_THRESHOLD: usize = 48;
type DflashPromptCapture = (MetalTensor, usize, usize, usize);

impl ServeError {
    pub(crate) fn server_error(message: impl Into<String>) -> Self {
        Self {
            status: 500,
            error_type: "server_error",
            code: None,
            param: None,
            message: message.into(),
        }
    }
}

pub(crate) struct EngineBackend {
    loaded: LoadedModel,
    tokenizer: Tokenizer,
    model_id: String,
    default_max_tokens: usize,
    max_context_tokens: Option<usize>,
    template: super::items::QwenTemplate,
    no_thinking_supported: bool,
    /// DFlash drafter (v0.77 speculative decode). Speculation requires
    /// captured target hidden states for every context position, which
    /// restored checkpoints do not carry — so a request uses the drafter
    /// only when it cold-prefills the whole prompt. Output is identical
    /// either way (greedy accept-prefix over an exact target verify).
    dflash_head: Option<MetalDFlashHead>,
}

impl EngineBackend {
    pub(crate) fn new(
        loaded: LoadedModel,
        model_id: String,
        default_max_tokens: usize,
        max_context_tokens: Option<usize>,
        drafter: Option<&std::path::Path>,
        template: super::items::QwenTemplate,
        no_thinking_supported: bool,
    ) -> anyhow::Result<Self> {
        let tokenizer = loaded.tokenizer().context("initialize serve tokenizer")?;
        let dflash_head = match drafter {
            Some(path) => {
                let t0 = Instant::now();
                let drafter_gguf = GgufFile::open(path)
                    .with_context(|| format!("open drafter {}", path.display()))?;
                qwen_llm::runtime::prefetch_opened_gguf(
                    &drafter_gguf,
                    &qwen_llm::runtime::LoadedModelConfig::default(),
                );
                let target_model = Model::from_gguf(loaded.gguf())
                    .context("parse target arch for drafter binding")?;
                let bound = open_dflash_drafter(&drafter_gguf, &target_model)
                    .with_context(|| format!("bind drafter {}", path.display()))?;
                let head = MetalDFlashHead::load(loaded.context(), &drafter_gguf, &bound)
                    .context("metal-load drafter")?;
                tracing::info!(
                    target: "qwen_diag",
                    "serve: dflash drafter loaded path={} block_size={} dflash2={} load_ms={:.1}",
                    path.display(),
                    head.config.block_size,
                    head.config.selector_top_k > 0,
                    t0.elapsed().as_secs_f64() * 1e3,
                );
                Some(head)
            }
            None => None,
        };
        Ok(Self {
            loaded,
            tokenizer,
            model_id,
            default_max_tokens,
            max_context_tokens,
            template,
            no_thinking_supported,
            dflash_head,
        })
    }
}

fn preopens(template: QwenTemplate, request: &ServeRequest) -> bool {
    template == QwenTemplate::Qwen38
        && !request.no_thinking
        && request.reasoning_effort.as_deref() != Some("none")
}

fn request_sampler(request: &ServeRequest) -> Result<Sampler, ServeError> {
    Sampler::new(SamplingConfig {
        temperature: request.temperature.unwrap_or(0.0),
        top_k: request.top_k.unwrap_or(200),
        top_p: request.top_p.unwrap_or(1.0),
        min_p: request.min_p.unwrap_or(0.05),
        seed: request.seed.unwrap_or(42),
    })
    .map_err(|error| ServeError::invalid_request(None, format!("sampling: {error}")))
}

fn complete_dflash_capture(start: usize, captured: usize, prompt_len: usize) -> bool {
    start == 0 && captured == prompt_len
}

fn use_serial_tail(
    remaining: usize,
    threshold: usize,
    speculate: bool,
    capture_supported: bool,
) -> bool {
    remaining <= threshold && (!speculate || capture_supported)
}

fn should_plan_dflash(has_head: bool, matched_tokens: usize, temperature: f32) -> bool {
    has_head && matched_tokens == 0 && temperature == 0.0
}

fn request_capacity(
    prompt_tokens: usize,
    generation_tokens: usize,
    configured_limit: Option<usize>,
) -> Result<usize, &'static str> {
    let required = prompt_tokens
        .checked_add(generation_tokens)
        .ok_or("prompt plus generation token count overflow")?;
    let limit = configured_limit.unwrap_or(super::DEFAULT_SERVE_MAX_CONTEXT_TOKENS);
    if required > limit {
        return Err("request exceeds max context token limit");
    }
    if configured_limit.is_some() {
        Ok(limit)
    } else {
        Ok(required.saturating_add(16).min(limit))
    }
}

fn dflash_capture_elements(prompt_tokens: usize, features: usize) -> Result<usize, ServeError> {
    prompt_tokens.checked_mul(features).ok_or_else(|| {
        ServeError::invalid_request(
            Some("input"),
            "DFlash prompt capture element count overflow",
        )
    })
}

#[allow(clippy::too_many_arguments)]
fn prefill_remaining(
    loaded: &LoadedModel,
    forward: &MetalForward<'_>,
    dflash_head: Option<&MetalDFlashHead>,
    speculate: bool,
    prompt_ids: &[i32],
    chunk: usize,
    sequence: &mut Sequence,
    scratch: &mut Option<MetalDFlashLayerMajorScratch>,
    dflash_capture: &mut Option<DflashPromptCapture>,
    sink: &mut dyn GenerationSink,
) -> Result<Option<Vec<f32>>, BackendFailure> {
    let mut prompt_logits = None;
    while sequence.position() < prompt_ids.len() {
        sink.tick().map_err(BackendFailure::Aborted)?;
        let start = sequence.position();
        let remaining = prompt_ids.len() - start;
        // The per-token multi-hidden API is currently dense-only. MoE
        // speculative requests keep the packed capture path even for a short
        // prompt rather than entering DFlash with missing context.
        let serial_capture_supported = loaded.arch().kind == qwen_llm::model::ArchKind::Dense;
        if use_serial_tail(
            remaining,
            SERIAL_TAIL_THRESHOLD,
            speculate,
            serial_capture_supported,
        ) {
            for (offset, &token) in prompt_ids[start..].iter().enumerate() {
                let position = start + offset;
                let position_u32 = u32::try_from(position)
                    .map_err(|_| ServeError::server_error("position overflow"))?;
                let logits = match (dflash_head, dflash_capture.as_mut()) {
                    (Some(head), Some((dst, capture_start, captured, n_features))) => {
                        let capture_offset =
                            position.checked_sub(*capture_start).ok_or_else(|| {
                                ServeError::server_error("drafter capture position underflow")
                            })?;
                        let view = dst.view_subrange(
                            (capture_offset * *n_features) as u64,
                            vec![*n_features as u64],
                        );
                        let logits = forward
                            .single_token_with_multi_hidden(
                                token,
                                position_u32,
                                unsafe { sequence.metal_session_mut() },
                                &head.target_layer_ids,
                                &view,
                            )
                            .map_err(|error| {
                                ServeError::server_error(format!(
                                    "serial tail capture prefill: {error:#}"
                                ))
                            })?;
                        *captured += 1;
                        logits
                    }
                    _ => forward
                        .single_token(token, position_u32, unsafe { sequence.metal_session_mut() })
                        .map_err(|error| {
                            ServeError::server_error(format!("serial tail prefill: {error:#}"))
                        })?,
                };
                sequence
                    .advance_by(1)
                    .map_err(|error| ServeError::server_error(format!("advance: {error:#}")))?;
                prompt_logits = Some(logits);
            }
            break;
        }
        let end = prompt_ids.len().min(start + chunk);
        let (logits, _span_ms) = match dflash_head.filter(|_| speculate) {
            Some(head) => {
                let span = prompt_ids.len() - start;
                let scratch = scratch.as_mut().ok_or_else(|| {
                    ServeError::server_error("uncached prefill has no scratch allocation")
                })?;
                let (dst, capture_start, captured, _) =
                    dflash_capture.as_mut().ok_or_else(|| {
                        ServeError::server_error("speculative prefill has no capture buffer")
                    })?;
                let out = crate::prefill_span_with_capture(
                    forward,
                    sequence,
                    scratch,
                    &prompt_ids[start..],
                    start,
                    &head.target_layer_ids,
                    dst,
                )
                .map_err(|error| ServeError::server_error(format!("capture prefill: {error:#}")))?;
                *capture_start = start;
                *captured = span;
                out
            }
            None => {
                let scratch = scratch.as_mut().ok_or_else(|| {
                    ServeError::server_error("uncached prefill has no scratch allocation")
                })?;
                crate::prefill_span(forward, sequence, scratch, &prompt_ids[start..end], start)
                    .map_err(|error| ServeError::server_error(format!("prefill: {error:#}")))?
            }
        };
        prompt_logits = Some(logits);
    }
    Ok(prompt_logits)
}

impl GenerationBackend for EngineBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn preopens_reasoning(&self, request: &ServeRequest) -> bool {
        preopens(self.template, request)
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        let mut request = request.clone();
        request.template = self.template;
        if request.no_thinking && !self.no_thinking_supported {
            return Err(ServeError::invalid_request(
                Some("x_qwen.no_thinking"),
                "x_qwen.no_thinking is not validated for the loaded model identity",
            ));
        }
        if self.template != QwenTemplate::Qwen38 && request.reasoning_effort.is_some() {
            return Err(ServeError::invalid_request(
                Some("reasoning.effort"),
                "reasoning.effort is only supported for validated Qwen3.8 identities",
            ));
        }
        if self.template == QwenTemplate::Qwen38 {
            match (request.no_thinking, request.reasoning_effort.as_deref()) {
                (true, Some(_)) => {
                    return Err(ServeError::invalid_request(
                        Some("reasoning.effort"),
                        "reasoning.effort cannot be combined with x_qwen.no_thinking",
                    ));
                }
                (_, None | Some("none" | "low" | "medium" | "xhigh")) => {}
                (_, Some(other)) => {
                    return Err(ServeError::invalid_request(
                        Some("reasoning.effort"),
                        format!(
                            "Qwen3.8 supports reasoning.effort none|low|medium|xhigh; got {other:?}"
                        ),
                    ));
                }
            }
        }
        Ok(super::render::render_qwen_serve_prompt(&request))
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let max_tokens = request.max_output_tokens.unwrap_or(self.default_max_tokens);
        if max_tokens == 0 {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                "max_output_tokens must be >= 1",
            )
            .into());
        }
        // Validate all user-controlled sampling values before tokenization,
        // request-state allocation, prefix restore, or model execution.
        let mut sampler = request_sampler(request)?;
        // Rendered ChatML carries its own special-token markers; matches the
        // legacy messages path (prompt_add_special_tokens = false).
        let tokenize_t0 = Instant::now();
        let prompt_ids = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|error| ServeError::server_error(format!("tokenize prompt: {error}")))?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        if prompt_ids.is_empty() {
            return Err(ServeError::invalid_request(
                Some("input"),
                "prompt tokenized to zero tokens",
            )
            .into());
        }

        // Context admission fails closed (S0 F3 == spec truncation:"disabled").
        let capacity = request_capacity(prompt_ids.len(), max_tokens, self.max_context_tokens)
            .map_err(|message| {
                ServeError::invalid_request(
                    Some("max_output_tokens"),
                    format!(
                        "{message}: max context {} is smaller than prompt {} + generation {max_tokens}",
                        self.max_context_tokens
                            .unwrap_or(super::DEFAULT_SERVE_MAX_CONTEXT_TOKENS),
                        prompt_ids.len(),
                    ),
                )
            })?;

        let restore_t0 = Instant::now();
        let cached_lookup = self.loaded.lookup_cached_prefix(&prompt_ids);
        let exact_cached = cached_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.is_exact_with_final_logits());
        // Exact-final-logits hits allocate only a destination session. Every
        // other request admits the maximum production prefill topology,
        // including deferred MoE fallback packs.
        let prefill_scratch_upper_bytes = if exact_cached {
            0
        } else {
            let legacy_chunk = crate::baseline_prefill_chunk(prompt_ids.len());
            let legacy_plan = plan_prefill_scratch_with_matrix_max_pos_configured(
                self.loaded.metal_model(),
                u32::try_from(legacy_chunk)
                    .map_err(|_| ServeError::server_error("prefill chunk does not fit u32"))?,
                prompt_ids.len().max(legacy_chunk),
                PrefillScratchConfig::default(),
            )
            .map_err(|error| ServeError::server_error(format!("plan prefill memory: {error}")))?;
            legacy_plan
                .priced_upper_bound(|logical_bytes| {
                    Ok(self
                        .loaded
                        .context()
                        .shared_buffer_size_and_align(logical_bytes)?
                        .size)
                })
                .map_err(|error| {
                    ServeError::server_error(format!("price prefill memory: {error}"))
                })?
        };
        let admission = self
            .loaded
            .qwen_execution_memory_admission(1, capacity, prefill_scratch_upper_bytes, 0)
            .map_err(|error| {
                ServeError::server_error(format!("price request memory: {error:#}"))
            })?;
        if !admission.admitted {
            return Err(ServeError {
                status: 503,
                error_type: "server_busy",
                code: Some("memory_admission_denied"),
                param: None,
                message: format!(
                    "request memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                    admission.reason.as_str(),
                    admission.required_bytes,
                    admission.working_set_headroom_bytes,
                    admission.signals.process_limit_remaining_bytes,
                ),
            }
            .into());
        }

        let alloc_t0 = Instant::now();
        let (mut chunk, mut scratch, mut sequence) = if exact_cached {
            let sequence = self
                .loaded
                .create_sequence(SequenceConfig::new(capacity))
                .map_err(|error| {
                    ServeError::server_error(format!("allocate request sequence: {error:#}"))
                })?;
            (
                crate::baseline_prefill_chunk(prompt_ids.len()),
                None,
                sequence,
            )
        } else {
            let allocated = crate::allocate_prefill_request_state(
                &self.loaded,
                crate::PrefillChunkArg::Auto,
                prompt_ids.len(),
                capacity,
                true,
            )
            .map_err(|error| {
                ServeError::server_error(format!("allocate request state: {error:#}"))
            })?;
            (allocated.chunk, Some(allocated.scratch), allocated.sequence)
        };
        let mut alloc_ms = alloc_t0.elapsed().as_secs_f64() * 1e3;
        let forward = self.loaded.forward();

        // RAM prefix cache restore (dual-boundary entries from prior turns).
        let restore = cached_lookup
            .map(|lookup| {
                self.loaded
                    .restore_prepared_cached_prefix(lookup, &mut sequence, &prompt_ids)
            })
            .transpose()
            .map_err(|error| ServeError::server_error(format!("prefix restore: {error:#}")))?;
        let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        let matched_tokens = restore
            .as_ref()
            .map_or(0, |restore| restore.matched_prefix_len);
        let mut prompt_logits = restore
            .as_ref()
            .filter(|restore| restore.exact)
            .and_then(|restore| restore.exact_final_logits.clone());

        // Chunked prefill of whatever the restore left; tick between chunks
        // carries heartbeats and surfaces client disconnects (cancellation).
        // Tails at or below SERIAL_TAIL_THRESHOLD decode token-by-token.
        // Speculation needs captured hiddens for the whole context; a
        // restore leaves earlier positions uncaptured, so the drafter only
        // runs on cold-prefilled requests (CLI parity: --drafter excludes
        // --durable-prefix-cache for the same reason).
        // Greedy-only (CLI parity: T>0 needs the maximal-coupling
        // rejection sampler), and cold-prefill-only. Decided before the
        // capture buffer is allocated so sampled requests pay nothing.
        let speculate_candidate = should_plan_dflash(
            self.dflash_head.is_some(),
            matched_tokens,
            sampler.config().temperature,
        );
        // Optional DFlash state is admitted only after restore establishes
        // that this is a cold request. Denial disables speculation rather than
        // rejecting an otherwise viable serial request.
        let mut dflash_plan = match self.dflash_head.as_ref().filter(|_| speculate_candidate) {
            Some(head) => (|| -> anyhow::Result<(usize, usize)> {
                let n_features = head
                    .target_layer_ids
                    .len()
                    .checked_mul(self.loaded.arch().hidden_size as usize)
                    .context("DFlash feature count overflow")?;
                let capture_elements = prompt_ids
                    .len()
                    .checked_mul(n_features)
                    .context("DFlash prompt capture element count overflow")?;
                let capture_logical_bytes = u64::try_from(capture_elements)
                    .ok()
                    .and_then(|elements| elements.checked_mul(size_of::<f32>() as u64))
                    .context("DFlash capture byte count overflow")?;
                let capture_bytes = self
                    .loaded
                    .context()
                    .shared_buffer_size_and_align(capture_logical_bytes)
                    .context("price DFlash capture allocation")?
                    .size;
                let session_capacity = prompt_ids
                    .len()
                    .checked_add(max_tokens)
                    .and_then(|value| value.checked_add(16))
                    .context("DFlash session capacity overflow")?;
                let session_bytes = MetalDFlashSession::priced_bytes(
                    self.loaded.context(),
                    head,
                    self.loaded.arch().hidden_size as u64,
                    self.loaded.arch().vocab_size as u64,
                    session_capacity,
                )
                .context("price DFlash session")?;
                let target_layers = u32::try_from(head.target_layer_ids.len())
                    .context("DFlash target layer count overflow")?;
                let verify_scratch_bytes = MetalDFlashVerifyScratch::priced_bytes(
                    self.loaded.context(),
                    self.loaded.metal_model(),
                    head.config.block_size,
                    target_layers,
                )
                .context("price DFlash verify scratch")?;
                let layer_scratch_bytes = MetalDFlashLayerMajorScratch::speculative_priced_bytes(
                    self.loaded.context(),
                    self.loaded.metal_model(),
                    head.config.block_size,
                )
                .context("price DFlash layer scratch")?;
                let optional_bytes = capture_bytes
                    .checked_add(session_bytes)
                    .and_then(|bytes| bytes.checked_add(verify_scratch_bytes))
                    .and_then(|bytes| bytes.checked_add(layer_scratch_bytes))
                    .context("DFlash optional byte total overflow")?;
                let admission = evaluate_metal_memory_admission(
                    optional_bytes,
                    DFLASH_FIXED_SCRATCH_RESERVE_BYTES,
                    self.loaded.context().memory_signals(),
                    true,
                );
                anyhow::ensure!(
                    admission.admitted,
                    "reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                    admission.reason.as_str(),
                    admission.required_bytes,
                    admission.working_set_headroom_bytes,
                    admission.signals.process_limit_remaining_bytes,
                );
                Ok((n_features, session_capacity))
            })()
            .map_err(|error| {
                tracing::warn!(
                    "serve: optional DFlash admission denied; falling back to serial decode: {error:#}"
                );
            })
            .ok(),
            None => None,
        };
        let mut speculate = dflash_plan.is_some();
        // (buffer, start position, captured positions, features per position)
        let mut dflash_capture: Option<DflashPromptCapture> = match self
            .dflash_head
            .as_ref()
            .filter(|_| speculate)
        {
            Some(_head) => {
                let (n_features, _) =
                    dflash_plan.expect("speculative requests have an admitted DFlash plan");
                let capture_elements = dflash_capture_elements(prompt_ids.len(), n_features)?;
                match MetalTensor::zeros_f32(self.loaded.context(), vec![capture_elements as u64]) {
                    Ok(dst) => Some((dst, sequence.position(), 0, n_features)),
                    Err(error) => {
                        tracing::warn!(
                            "serve: optional DFlash capture allocation failed; falling back to serial decode: {error:#}"
                        );
                        dflash_plan = None;
                        speculate = false;
                        None
                    }
                }
            }
            None => None,
        };
        let prefill_t0 = Instant::now();
        let prefill_result = prefill_remaining(
            &self.loaded,
            &forward,
            self.dflash_head.as_ref(),
            speculate,
            &prompt_ids,
            chunk,
            &mut sequence,
            &mut scratch,
            &mut dflash_capture,
            sink,
        );
        match prefill_result {
            Ok(logits) => {
                if logits.is_some() {
                    prompt_logits = logits;
                }
            }
            Err(BackendFailure::Serve(error)) if speculate && matched_tokens == 0 => {
                let capture_error = error.message;
                tracing::warn!(
                    "serve: optional DFlash capture prefill failed; restarting serial prefill: {}",
                    capture_error
                );
                dflash_plan = None;
                speculate = false;
                drop(dflash_capture.take());
                drop(scratch.take());
                drop(sequence);
                let retry_alloc_t0 = Instant::now();
                let allocated = crate::allocate_prefill_request_state(
                    &self.loaded,
                    crate::PrefillChunkArg::Auto,
                    prompt_ids.len(),
                    capacity,
                    true,
                )
                .map_err(|retry_error| {
                    ServeError::server_error(format!(
                        "allocate serial fallback request state after DFlash capture failure ({capture_error}): {retry_error:#}"
                    ))
                })?;
                alloc_ms += retry_alloc_t0.elapsed().as_secs_f64() * 1e3;
                chunk = allocated.chunk;
                scratch = Some(allocated.scratch);
                sequence = allocated.sequence;
                prompt_logits = prefill_remaining(
                    &self.loaded,
                    &forward,
                    self.dflash_head.as_ref(),
                    false,
                    &prompt_ids,
                    chunk,
                    &mut sequence,
                    &mut scratch,
                    &mut dflash_capture,
                    sink,
                )?;
            }
            Err(error) => return Err(error),
        }
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
        let logits = prompt_logits
            .ok_or_else(|| ServeError::server_error("prefill produced no prompt logits"))?;

        // Prompt-boundary capture into the RAM cache (skip when this exact
        // prompt was already an exact hit).
        let capture_t0 = Instant::now();
        if !restore.as_ref().is_some_and(|restore| restore.exact) {
            self.try_cache_boundary(&sequence, &prompt_ids, None, Some(&logits), "prompt");
        }

        let prompt_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
        let stop_tokens = self
            .loaded
            .gguf()
            .stop_token_ids()
            .map_err(|error| ServeError::server_error(format!("stop tokens: {error}")))?;

        let mut abort: Option<io::Error> = None;
        let mut assembler = Utf8Assembler::new();
        let tokenizer = &self.tokenizer;
        // Speculative path: seed the drafter cross-context from the captured
        // prompt hiddens, then greedy accept-prefix over an exact target
        // verify — emitted tokens are identical to serial greedy.
        if let Some(head) = self.dflash_head.as_ref().filter(|_| speculate) {
            let (dst, capture_start, captured, n_features) =
                dflash_capture.as_ref().ok_or_else(|| {
                    ServeError::server_error("speculative decode has no prompt hidden capture")
                })?;
            if !complete_dflash_capture(*capture_start, *captured, prompt_ids.len())
                || sequence.position() != prompt_ids.len()
            {
                return Err(ServeError::server_error(format!(
                    "refusing speculative decode with incomplete prompt capture: start={} captured={} prompt={} sequence={}",
                    capture_start,
                    captured,
                    prompt_ids.len(),
                    sequence.position(),
                ))
                .into());
            }
            let (_, capacity) =
                dflash_plan.expect("speculative requests have an admitted DFlash plan");
            let dsess = MetalDFlashSession::fresh(
                self.loaded.context(),
                head,
                self.loaded.arch().hidden_size as u64,
                self.loaded.arch().vocab_size as u64,
                capacity,
            );
            let mut dsess = match dsess {
                Ok(dsess) => Some(dsess),
                Err(error) => {
                    tracing::warn!(
                        "serve: optional DFlash session allocation failed; falling back to serial decode: {error:#}"
                    );
                    None
                }
            };
            if let Some(session) = dsess.as_mut()
                && let Err(error) = session.append_target_ctx_columns_contiguous_now(
                    self.loaded.context(),
                    dst,
                    *capture_start as u32,
                    *captured,
                    *n_features,
                )
            {
                tracing::warn!(
                    "serve: optional DFlash context seeding failed; falling back to serial decode: {error:#}"
                );
                dsess = None;
            }
            let dflash_scratch = if dsess.is_some() {
                match crate::allocate_dflash_decode_scratch(&self.loaded, head) {
                    Ok(scratch) => Some(scratch),
                    Err(error) => {
                        tracing::warn!(
                            "serve: optional DFlash decode scratch allocation failed; falling back to serial decode: {error:#}"
                        );
                        None
                    }
                }
            } else {
                None
            };
            if let (Some(dsess), Some(dflash_scratch)) = (dsess, dflash_scratch) {
                let result = {
                    let abort = &mut abort;
                    crate::generate_dflash(
                        &self.loaded,
                        &forward,
                        head,
                        dsess,
                        dflash_scratch,
                        sequence,
                        logits,
                        max_tokens,
                        &stop_tokens,
                        |token| {
                            let bytes = tokenizer
                                .try_decode_piece_bytes_exact(token)
                                .with_context(|| format!("decode token {token}"))?;
                            let piece = assembler.push(bytes);
                            if piece.is_empty() {
                                return Ok(());
                            }
                            sink.piece(&piece).map_err(|error| {
                                *abort = Some(error);
                                anyhow::anyhow!("client disconnected during decode")
                            })
                        },
                    )
                };
                let result = match result {
                    Ok(result) => result,
                    Err(error) => {
                        return Err(match abort {
                            Some(io_error) => BackendFailure::Aborted(io_error),
                            None => {
                                ServeError::server_error(format!("dflash decode: {error:#}")).into()
                            }
                        });
                    }
                };
                let generation = result.generation;
                let sequence = result.sequence;
                let stats = result.stats;
                let tail = assembler.finish();
                if !tail.is_empty() {
                    sink.piece(&tail).map_err(BackendFailure::Aborted)?;
                }
                return self.finish_generation(
                    generation,
                    Some(stats),
                    sequence,
                    prompt_ids,
                    matched_tokens,
                    restore_ms,
                    format!(
                        "tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1} decode_path=dflash"
                    ),
                );
            }
        }
        let generation = {
            let abort = &mut abort;
            crate::generate_serial(
                logits,
                max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    // Exact bytes + incremental UTF-8 assembly: per-token
                    // lossy decode corrupts multibyte characters split
                    // across tokens (k3 R1.6).
                    let bytes = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode token {token}"))?;
                    let piece = assembler.push(bytes);
                    if piece.is_empty() {
                        return Ok(());
                    }
                    sink.piece(&piece).map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
                },
                |token| {
                    let position = sequence.position();
                    let next = forward
                        .single_token(
                            token,
                            u32::try_from(position).context("position does not fit u32")?,
                            unsafe { sequence.metal_session_mut() },
                        )
                        .context("decode token")?;
                    sequence.advance_by(1)?;
                    Ok(next)
                },
            )
        };
        let generation = match generation {
            Ok(generation) => generation,
            Err(error) => {
                return Err(match abort {
                    Some(io_error) => BackendFailure::Aborted(io_error),
                    None => ServeError::server_error(format!("decode: {error:#}")).into(),
                });
            }
        };
        let tail = assembler.finish();
        if !tail.is_empty() {
            sink.piece(&tail).map_err(BackendFailure::Aborted)?;
        }

        self.finish_generation(
            generation,
            None,
            sequence,
            prompt_ids,
            matched_tokens,
            restore_ms,
            format!("tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1} decode_path=serial"),
        )
    }
}

impl EngineBackend {
    fn try_cache_boundary(
        &self,
        sequence: &qwen_llm::runtime::Sequence,
        prefix_tokens: &[i32],
        pending_token: Option<i32>,
        final_logits: Option<&[f32]>,
        boundary: &'static str,
    ) {
        let estimate = match self.loaded.estimate_checkpoint_boundary_sizes(
            sequence,
            prefix_tokens.len(),
            pending_token.is_some(),
            final_logits.is_some(),
        ) {
            Ok(estimate) => estimate.snapshot_bytes,
            Err(error) => {
                tracing::warn!(
                    "serve: {boundary} snapshot estimate failed; skipping cache capture: {error}"
                );
                return;
            }
        };
        if !self.loaded.prefix_cache_strict_eligible(estimate) {
            tracing::warn!(
                "serve: {boundary} snapshot denied by cache budget; snapshot_bytes={estimate} cache_budget_bytes={}",
                self.loaded.prefix_cache_stats().max_indexed_bytes,
            );
            return;
        }
        let signals = self.loaded.context().memory_signals();
        if let Err(reason) = super::snapshot_capture_admission(estimate, signals) {
            tracing::warn!(
                "serve: {boundary} snapshot denied by memory headroom; reason={reason:?} snapshot_bytes={estimate} metal_current_bytes={} metal_recommended_bytes={} process_remaining_bytes={:?}",
                signals.current_allocated_bytes,
                signals.recommended_max_bytes,
                signals.process_limit_remaining_bytes,
            );
            return;
        }
        match self.loaded.prepare_checkpoint_boundary(
            sequence,
            prefix_tokens.to_vec(),
            pending_token,
            final_logits.map(<[f32]>::to_vec),
        ) {
            Ok(prepared) => match self.loaded.cache_prepared_checkpoint_strict(&prepared) {
                Ok(Some(_)) => {}
                Ok(None) => tracing::warn!(
                    "serve: {boundary} snapshot rejected at strict cache insertion; request continues"
                ),
                Err(error) => tracing::warn!(
                    "serve: {boundary} strict cache insertion failed; request continues: {error}"
                ),
            },
            Err(error) => tracing::warn!(
                "serve: {boundary} snapshot capture failed; request continues: {error}"
            ),
        }
    }

    /// Shared completion for both decode paths: completed-turn capture,
    /// phase/stats lines, and the outcome.
    #[allow(clippy::too_many_arguments)]
    fn finish_generation(
        &self,
        generation: crate::GenerationResult,
        dflash: Option<crate::DflashDecodeStats>,
        sequence: qwen_llm::runtime::Sequence,
        prompt_ids: Vec<i32>,
        matched_tokens: usize,
        restore_ms: f64,
        phases: String,
    ) -> Result<GenerationOutcome, BackendFailure> {
        match crate::derive_completed_checkpoint_boundary(
            prompt_ids.len(),
            &generation.tokens,
            generation.transitions,
            sequence.position(),
        ) {
            Ok(boundary) => {
                let pending_token = boundary.pending_token;
                let consumed = boundary.consumed_tokens(&prompt_ids, &generation.tokens);
                self.try_cache_boundary(
                    &sequence,
                    &consumed,
                    Some(pending_token),
                    None,
                    "completed",
                );
            }
            Err(error) => tracing::warn!("serve: completed boundary derivation failed: {error}"),
        }

        let stop_reason = match generation.stop_reason {
            crate::StopReason::Eos => StopReason::Eos,
            crate::StopReason::TokenLimit => StopReason::TokenLimit,
        };
        let decode_tps = if generation.wall_ms > 0.0 {
            generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
        } else {
            0.0
        };
        tracing::info!(target: "qwen_diag", "serve phases: {phases}");
        tracing::info!(
            target: "qwen_diag",
            "serve stats: version=serve_stats_v1 prompt_tokens={} generated_tokens={} stop_reason={} matched_tokens={} restore_ms={:.1} decode_tps={:.2}",
            prompt_ids.len(),
            generation.tokens.len(),
            match stop_reason {
                StopReason::Eos => "eos",
                StopReason::TokenLimit => "token_limit",
            },
            matched_tokens,
            restore_ms,
            decode_tps,
        );
        if let Some(stats) = dflash {
            tracing::info!(
                target: "qwen_diag",
                "serve dflash: spec_steps={} off_steps={} accepted={}/{} alpha_backoff={} draft_ms={:.1} verify_ms={:.1}",
                stats.spec_steps,
                stats.off_steps,
                stats.accepted_drafts,
                stats.drafts_scored,
                stats.alpha_backoff,
                stats.draft_ms,
                stats.verify_ms,
            );
        }
        Ok(GenerationOutcome {
            stop_reason,
            usage: Usage {
                input_tokens: prompt_ids.len(),
                output_tokens: generation.tokens.len(),
                cached_tokens: matched_tokens,
            },
            stats: Some(ServeStats {
                matched_tokens,
                restore_ms,
                prompt_tokens: prompt_ids.len(),
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen38_thinking_prompt_is_headless() {
        let request = ServeRequest::default();
        assert!(preopens(QwenTemplate::Qwen38, &request));
        assert!(!preopens(
            QwenTemplate::Qwen38,
            &ServeRequest {
                no_thinking: true,
                ..request
            }
        ));
        assert!(!preopens(
            QwenTemplate::Qwen38,
            &ServeRequest {
                reasoning_effort: Some("none".into()),
                ..ServeRequest::default()
            }
        ));
        assert!(!preopens(QwenTemplate::Generic, &ServeRequest::default()));
    }

    #[test]
    fn dflash_requires_capture_from_zero_through_entire_prompt() {
        assert!(complete_dflash_capture(0, 1, 1));
        assert!(complete_dflash_capture(0, 48, 48));
        assert!(!complete_dflash_capture(1, 47, 48));
        assert!(!complete_dflash_capture(0, 47, 48));
        assert!(use_serial_tail(48, 48, true, true));
        assert!(!use_serial_tail(48, 48, true, false));
        assert!(use_serial_tail(48, 48, false, false));
        assert!(should_plan_dflash(true, 0, 0.0));
        assert!(!should_plan_dflash(true, 1, 0.0));
        assert!(!should_plan_dflash(true, 0, 0.1));
    }

    #[test]
    fn invalid_sampling_is_rejected_by_request_sampler() {
        let request = ServeRequest {
            top_p: Some(0.0),
            ..ServeRequest::default()
        };
        assert!(request_sampler(&request).is_err());
    }

    #[test]
    fn dflash_capture_size_is_checked() {
        assert_eq!(dflash_capture_elements(48, 32), Ok(1536));
        assert!(dflash_capture_elements(usize::MAX, 2).is_err());
    }

    #[test]
    fn default_context_cap_is_finite_but_allocates_per_request() {
        assert_eq!(request_capacity(133_000, 1_000, None), Ok(134_016));
        assert_eq!(request_capacity(100, 20, None), Ok(136));
        assert!(request_capacity(super::super::DEFAULT_SERVE_MAX_CONTEXT_TOKENS, 1, None).is_err());
        assert_eq!(request_capacity(100, 20, Some(1024)), Ok(1024));
    }
}
