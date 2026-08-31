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
use super::output_partition::{GenerationEnd, OutputProtocol};
use anyhow::Context as _;
use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::{Model, open_dflash_drafter};
use qwen_llm::metal::{MetalTensor, evaluate_metal_memory_admission};
use qwen_llm::metal_dflash::{
    MetalDFlashHead, MetalDFlashLayerMajorScratch, MetalDFlashSession, MetalDFlashVerifyScratch,
    PrefillScratchConfig, dflash_capture_window_complete, dflash_capture_window_limit,
    dflash_capture_window_span, plan_prefill_scratch_with_matrix_max_pos_configured,
};
use qwen_llm::metal_forward::MetalForward;
use qwen_llm::runtime::{LoadedModel, Sequence, SequenceConfig};
use qwen_llm::sampling::{SAMPLER_ALGORITHM_VERSION, Sampler, SamplingConfig};
use qwen_llm::tokenizer::{Tokenizer, token_ids_sha256_i32le};
use std::collections::VecDeque;
use std::io;
use std::time::Instant;

const DFLASH_FIXED_SCRATCH_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
const SERIAL_TAIL_THRESHOLD: usize = 48;
const DFLASH_PREFIX_REPLAY_ENV: &str = "QWEN_DFLASH_PREFIX_REPLAY";
const DFLASH_PREFIX_REPLAY_MAX_ENTRIES: usize = 4;
const DFLASH_PREFIX_REPLAY_MAX_TOKENS: usize = 4096;
const DFLASH_PREFIX_REPLAY_MIN_SAMPLED_ENTRIES: usize = 2;
const DFLASH_PREFIX_REPLAY_MIN_SAMPLED_TOKENS: usize = 32;
type DflashPromptCapture = (MetalTensor, usize, usize, usize, Option<usize>);

#[derive(Clone, Debug, Eq, PartialEq)]
struct DflashPrefixReplayKey {
    prompt_sha256: String,
    sampler_version: u32,
    temperature_bits: u32,
    top_k: usize,
    top_p_bits: u32,
    min_p_bits: u32,
}

// Seed is intentionally excluded so repeated sampling runs can share history;
// sampled admission separately requires consensus from distinct seeds.
fn dflash_prefix_replay_key(prompt_ids: &[i32], sampling: SamplingConfig) -> DflashPrefixReplayKey {
    DflashPrefixReplayKey {
        prompt_sha256: token_ids_sha256_i32le(prompt_ids),
        sampler_version: SAMPLER_ALGORITHM_VERSION,
        temperature_bits: sampling.temperature.to_bits(),
        top_k: sampling.top_k,
        top_p_bits: sampling.top_p.to_bits(),
        min_p_bits: sampling.min_p.to_bits(),
    }
}

#[derive(Default)]
struct DflashPrefixReplayCache {
    enabled: bool,
    entry: Option<(DflashPrefixReplayKey, VecDeque<DflashPrefixReplayHistory>)>,
}

struct DflashPrefixReplayHistory {
    seed: u64,
    tokens: Box<[i32]>,
}

impl DflashPrefixReplayCache {
    fn from_env() -> Self {
        let enabled = match std::env::var(DFLASH_PREFIX_REPLAY_ENV) {
            Err(std::env::VarError::NotPresent) => false,
            Ok(value) if matches!(value.as_str(), "0" | "false" | "off") => false,
            Ok(value) if matches!(value.as_str(), "1" | "true" | "on") => true,
            Err(std::env::VarError::NotUnicode(_)) | Ok(_) => {
                tracing::warn!(
                    "serve: {DFLASH_PREFIX_REPLAY_ENV} must be 0/1, false/true, or off/on; disabling"
                );
                false
            }
        };
        Self {
            enabled,
            entry: None,
        }
    }

    fn lookup(&self, key: &DflashPrefixReplayKey) -> Option<Vec<i32>> {
        let (_, histories) = self
            .enabled
            .then_some(())
            .and_then(|()| self.entry.as_ref())
            .filter(|(cached_key, _)| cached_key == key)?;
        let sampled = key.temperature_bits != 0.0f32.to_bits();
        let distinct_seeds = histories
            .iter()
            .enumerate()
            .filter(|(index, history)| {
                !histories
                    .iter()
                    .take(*index)
                    .any(|prior| prior.seed == history.seed)
            })
            .count();
        if sampled && distinct_seeds < DFLASH_PREFIX_REPLAY_MIN_SAMPLED_ENTRIES {
            return None;
        }
        let first = histories.front()?;
        let mut common = first.tokens.len();
        for history in histories.iter().skip(1) {
            common = common.min(history.tokens.len()).min(
                first
                    .tokens
                    .iter()
                    .zip(history.tokens.iter())
                    .take_while(|(a, b)| a == b)
                    .count(),
            );
        }
        let minimum = if sampled {
            DFLASH_PREFIX_REPLAY_MIN_SAMPLED_TOKENS
        } else {
            qwen_llm::prompt_lookup::DRAFT_TOKENS + 1
        };
        (common >= minimum).then(|| first.tokens[..common].to_vec())
    }

    fn insert(&mut self, key: DflashPrefixReplayKey, seed: u64, tokens: &[i32]) {
        if self.enabled
            && (qwen_llm::prompt_lookup::DRAFT_TOKENS + 1..=DFLASH_PREFIX_REPLAY_MAX_TOKENS)
                .contains(&tokens.len())
        {
            match self.entry.as_mut() {
                Some((cached_key, histories)) if *cached_key == key => {
                    if let Some(index) = histories.iter().position(|history| history.seed == seed) {
                        histories.remove(index);
                    }
                    if histories.len() == DFLASH_PREFIX_REPLAY_MAX_ENTRIES {
                        histories.pop_front();
                    }
                    histories.push_back(DflashPrefixReplayHistory {
                        seed,
                        tokens: tokens.into(),
                    });
                }
                _ => {
                    self.entry = Some((
                        key,
                        VecDeque::from([DflashPrefixReplayHistory {
                            seed,
                            tokens: tokens.into(),
                        }]),
                    ));
                }
            }
        }
    }

    fn enabled(&self) -> bool {
        self.enabled
    }
}

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
    /// the drafter's relevant target-hidden window. Restored requests use the
    /// drafter only when their cache entry carries a compatible capture tail.
    /// Sampled requests use
    /// rejection sampling against the packed target forward, whose numerics
    /// can differ slightly from serial token-major decoding.
    dflash_head: Option<MetalDFlashHead>,
    dflash_prefix_replay: DflashPrefixReplayCache,
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
            dflash_prefix_replay: DflashPrefixReplayCache::from_env(),
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

fn complete_dflash_capture(
    start: usize,
    captured: usize,
    prompt_len: usize,
    window_limit: usize,
) -> bool {
    dflash_capture_window_complete(prompt_len, start, captured, window_limit)
}

fn use_serial_tail(
    remaining: usize,
    threshold: usize,
    speculate: bool,
    capture_supported: bool,
) -> bool {
    remaining <= threshold && (!speculate || capture_supported)
}

fn should_plan_dflash(
    has_head: bool,
    matched_tokens: usize,
    restore_capture_complete: bool,
    dense: bool,
) -> bool {
    has_head && dense && (matched_tokens == 0 || restore_capture_complete)
}

fn restored_dflash_capture_complete(
    matched_tokens: usize,
    capture_start: usize,
    captured: usize,
) -> bool {
    captured == matched_tokens.saturating_sub(capture_start)
}

fn dflash_ring_offset(position: usize, capture_start: usize, ring_window: usize) -> Option<usize> {
    if ring_window == 0 {
        None
    } else {
        position
            .checked_sub(capture_start)
            .map(|offset| offset % ring_window)
    }
}

fn dflash_prompt_capture_offset(position: usize, capture_start: usize) -> Option<usize> {
    position.checked_sub(capture_start)
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

fn dflash_capture_elements(
    prompt_tokens: usize,
    features: usize,
    window_limit: usize,
) -> Result<usize, ServeError> {
    let (_, window) = dflash_capture_window_span(prompt_tokens, window_limit);
    window.checked_mul(features).ok_or_else(|| {
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
    _speculate: bool,
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
            dflash_capture.is_some(),
            serial_capture_supported,
        ) {
            for (offset, &token) in prompt_ids[start..].iter().enumerate() {
                let position = start + offset;
                let position_u32 = u32::try_from(position)
                    .map_err(|_| ServeError::server_error("position overflow"))?;
                let logits = match (dflash_head, dflash_capture.as_mut()) {
                    (Some(head), Some((dst, capture_start, captured, n_features, _ring)))
                        if position >= *capture_start =>
                    {
                        let capture_offset = dflash_prompt_capture_offset(position, *capture_start)
                            .expect("guarded prompt capture position");
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
        let (logits, _span_ms) = match dflash_capture.as_mut() {
            Some((dst, capture_start, captured, n_features, _ring)) => {
                let head = dflash_head.expect("capture window buffer implies a drafter head");
                let window_limit = dflash_capture_window_limit(head);
                let (wstart, _window) = dflash_capture_window_span(prompt_ids.len(), window_limit);
                let scratch = scratch.as_mut().ok_or_else(|| {
                    ServeError::server_error("uncached prefill has no scratch allocation")
                })?;
                if end <= wstart {
                    crate::prefill_span(forward, sequence, scratch, &prompt_ids[start..end], start)
                        .map_err(|error| ServeError::server_error(format!("prefill: {error:#}")))?
                } else {
                    let cstart = start.max(wstart);
                    if cstart > start {
                        crate::prefill_span(
                            forward,
                            sequence,
                            scratch,
                            &prompt_ids[start..cstart],
                            start,
                        )
                        .map_err(|error| ServeError::server_error(format!("prefill: {error:#}")))?;
                    }
                    let view = dst.view_subrange(
                        ((cstart - wstart) * *n_features) as u64,
                        vec![((end - cstart) * *n_features) as u64],
                    );
                    *capture_start = wstart;
                    let out = crate::prefill_span_with_capture(
                        forward,
                        sequence,
                        scratch,
                        &prompt_ids[cstart..end],
                        cstart,
                        &head.target_layer_ids,
                        &view,
                    )
                    .map_err(|error| {
                        ServeError::server_error(format!("capture prefill: {error:#}"))
                    })?;
                    *captured += end - cstart;
                    out
                }
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

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: preopens(self.template, request),
            parse_tools: true,
        }
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        let request = crate::open_responses::bind_qwen_request(
            request,
            self.template,
            self.no_thinking_supported,
        )?;
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
        let replay_sampling = sampler.config();
        let dflash_prefix_replay_key = (self.dflash_prefix_replay.enabled()
            && self.dflash_head.is_some())
        .then(|| dflash_prefix_replay_key(&prompt_ids, replay_sampling));
        let dflash_prefix_replay = dflash_prefix_replay_key
            .as_ref()
            .and_then(|key| self.dflash_prefix_replay.lookup(key));

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

        // Windowed capture window for dense requests with a drafter head.
        // Doubles as (a) the drafter seed source on speculative requests and
        // (b) the rolling capture ring that keeps completed-boundary tails
        // fresh during serial decode. Allocated before the speculate decision
        // so restored requests can publish a tail for the NEXT turn even when
        // this one stays serial.
        let dense = self.loaded.arch().kind == qwen_llm::model::ArchKind::Dense;
        let restore_tail = restore
            .as_ref()
            .and_then(|restore| restore.capture_tail.clone());
        let restored_prefix_len = restore
            .as_ref()
            .map_or(0, |restore| restore.restored_prefix_len);
        let mut dflash_capture: Option<DflashPromptCapture> = match self
            .dflash_head
            .as_ref()
            .filter(|_| dense)
        {
            Some(head) => {
                let n_features = head
                    .target_layer_ids
                    .len()
                    .checked_mul(self.loaded.arch().hidden_size as usize)
                    .ok_or_else(|| ServeError::server_error("DFlash feature count overflow"))?;
                let window_limit = dflash_capture_window_limit(head);
                let (wstart, window) = dflash_capture_window_span(prompt_ids.len(), window_limit);
                // Ring capacity: a fixed window for all-SWA heads (the ring
                // wraps through decode, so it must hold a full window even
                // for short prompts); legacy full-attn heads keep a
                // prompt-sized buffer and no ring.
                let ring_columns = if window_limit == usize::MAX {
                    None
                } else {
                    Some(window_limit)
                };
                let capture_columns = ring_columns.unwrap_or(prompt_ids.len());
                let capture_elements =
                    capture_columns.checked_mul(n_features).ok_or_else(|| {
                        ServeError::server_error("DFlash capture element count overflow")
                    })?;
                match MetalTensor::zeros_f32(self.loaded.context(), vec![capture_elements as u64]) {
                    Ok(dst) => {
                        // Seed the leading part of the window from the
                        // checkpoint's capture tail (restored-request
                        // speculation): columns [seed_start, seed_end) are
                        // already captured by the previous turn.
                        let mut seeded = 0usize;
                        if let Some(tail) = restore_tail.as_ref() {
                            let tail_src_cols = restored_prefix_len.min(window);
                            // Fail closed on a wrong-length tail: the drafter
                            // is not part of the snapshot identity, so a
                            // durable store shared across drafter revisions
                            // can serve a tail whose feature width differs.
                            // Treat it as absent rather than slicing/panicking.
                            if tail.len() != tail_src_cols * n_features {
                                tracing::warn!(
                                    "serve: checkpoint capture tail length {} != {} features x {} columns; ignoring tail",
                                    tail.len(),
                                    n_features,
                                    tail_src_cols,
                                );
                            } else {
                                let tail_wstart = restored_prefix_len - tail_src_cols;
                                let seed_start = wstart.max(tail_wstart);
                                let seed_end = matched_tokens.min(restored_prefix_len);
                                if seed_end > seed_start {
                                    let skip = seed_start - tail_wstart;
                                    let count = seed_end - seed_start;
                                    let dst_off = seed_start - wstart;
                                    let src = &tail[skip * n_features..(skip + count) * n_features];
                                    unsafe {
                                        let dst_ptr = dst.buffer.contents().as_ptr() as *mut f32;
                                        std::ptr::copy_nonoverlapping(
                                            src.as_ptr(),
                                            dst_ptr.add(dst_off * n_features),
                                            count * n_features,
                                        );
                                    }
                                    seeded = count;
                                }
                            }
                        }
                        Some((dst, wstart, seeded, n_features, ring_columns))
                    }
                    Err(error) => {
                        tracing::warn!(
                            "serve: optional DFlash capture allocation failed; continuing without capture: {error:#}"
                        );
                        None
                    }
                }
            }
            None => None,
        };

        // Speculation needs captured hiddens covering the whole window: cold
        // requests capture via prefill; restored requests need the capture
        // tail from the checkpoint they restored. Decided after the capture
        // buffer so restored requests pay nothing extra.
        let restore_capture_complete =
            dflash_capture
                .as_ref()
                .is_some_and(|(_, capture_start, captured, _, _)| {
                    restored_dflash_capture_complete(matched_tokens, *capture_start, *captured)
                });
        let speculate_candidate = should_plan_dflash(
            self.dflash_head.is_some(),
            matched_tokens,
            restore_capture_complete,
            dense,
        ) && dflash_capture.is_some();
        // Optional DFlash state is admitted only after restore establishes an
        // eligible capture window. Denial disables speculation rather than
        // rejecting an otherwise viable serial request.
        let mut dflash_plan = match self.dflash_head.as_ref().filter(|_| speculate_candidate) {
            Some(head) => (|| -> anyhow::Result<(usize, usize)> {
                let window_limit = dflash_capture_window_limit(head);
                let (_, capture_window) =
                    dflash_capture_window_span(prompt_ids.len(), window_limit);
                let capture_columns = if window_limit == usize::MAX {
                    prompt_ids.len()
                } else {
                    window_limit
                };
                let n_features = head
                    .target_layer_ids
                    .len()
                    .checked_mul(self.loaded.arch().hidden_size as usize)
                    .context("DFlash feature count overflow")?;
                let capture_elements = capture_columns
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
                let session_capacity = capture_window
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
                let sampled_logits_bytes = if sampler.config().temperature > 0.0 {
                    let elements = u64::from(head.config.block_size)
                        .checked_mul(u64::from(self.loaded.arch().vocab_size))
                        .context("DFlash sampled logits element count overflow")?;
                    let logical_bytes = elements
                        .checked_mul(size_of::<f32>() as u64)
                        .context("DFlash sampled logits byte count overflow")?;
                    self.loaded
                        .context()
                        .shared_buffer_size_and_align(logical_bytes)
                        .context("price DFlash sampled logits allocation")?
                        .size
                } else {
                    0
                };
                let optional_bytes = capture_bytes
                    .checked_add(session_bytes)
                    .and_then(|bytes| bytes.checked_add(verify_scratch_bytes))
                    .and_then(|bytes| bytes.checked_add(layer_scratch_bytes))
                    .and_then(|bytes| bytes.checked_add(sampled_logits_bytes))
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
        if speculate
            && !dflash_capture
                .as_ref()
                .is_some_and(|(_, capture_start, captured, _, _)| {
                    complete_dflash_capture(
                        *capture_start,
                        *captured,
                        prompt_ids.len(),
                        self.dflash_head
                            .as_ref()
                            .map(dflash_capture_window_limit)
                            .unwrap_or(usize::MAX),
                    )
                })
        {
            tracing::warn!(
                "serve: DFlash capture remained incomplete after prefill; falling back to serial decode"
            );
            dflash_plan = None;
            speculate = false;
        }

        // Prompt-boundary capture into the RAM cache (skip when this exact
        // prompt was already an exact hit). The capture window buffer holds
        // the prompt's trailing columns; publish them as the drafter tail.
        let capture_t0 = Instant::now();
        if !restore.as_ref().is_some_and(|restore| restore.exact) {
            let (tail, features) = match dflash_capture.as_ref() {
                Some((dst, wstart, _, n_features, ring)) => (
                    ring.map(|window| {
                        read_window_tail(dst, *n_features, prompt_ids.len(), *wstart, window)
                    }),
                    *n_features,
                ),
                None => (None, 0),
            };
            self.try_cache_boundary(
                &sequence,
                &prompt_ids,
                None,
                Some(&logits),
                tail,
                features,
                "prompt",
            );
        }

        let prompt_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
        let stop_tokens = self
            .loaded
            .gguf()
            .stop_token_ids()
            .map_err(|error| ServeError::server_error(format!("stop tokens: {error}")))?;

        let mut abort: Option<io::Error> = None;
        let tokenizer = &self.tokenizer;
        // Speculative path: seed the drafter cross-context from the captured
        // prompt hiddens, then verify greedy or sampled proposals with the
        // packed target forward.
        if let Some(head) = self.dflash_head.as_ref().filter(|_| speculate) {
            let (dst, capture_start, captured, n_features, _ring) =
                dflash_capture.as_ref().ok_or_else(|| {
                    ServeError::server_error("speculative decode has no prompt hidden capture")
                })?;
            if !complete_dflash_capture(
                *capture_start,
                *captured,
                prompt_ids.len(),
                dflash_capture_window_limit(head),
            ) || sequence.position() != prompt_ids.len()
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
                match crate::allocate_dflash_decode_scratch(
                    &self.loaded,
                    head,
                    sampler.config().temperature > 0.0,
                ) {
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
                        &mut sampler,
                        max_tokens,
                        &stop_tokens,
                        dflash_capture
                            .as_ref()
                            .and_then(|(dst, wstart, _, n_features, ring)| {
                                ring.map(|window| (dst.clone(), *wstart, *n_features, window))
                            }),
                        dflash_prefix_replay.as_deref(),
                        None,
                        |token| {
                            let bytes = tokenizer
                                .try_decode_piece_bytes_exact(token)
                                .with_context(|| format!("decode token {token}"))?;
                            sink.piece(bytes).map_err(|error| {
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
                return self.finish_generation(
                    generation,
                    Some(stats),
                    sequence,
                    prompt_ids,
                    matched_tokens,
                    restore_ms,
                    dflash_capture
                        .as_ref()
                        .and_then(|(dst, wstart, _, n_features, ring)| {
                            ring.map(|window| (dst.clone(), *wstart, *n_features, window))
                        }),
                    dflash_prefix_replay_key.as_ref(),
                    replay_sampling.seed,
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
                    let bytes = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode token {token}"))?;
                    sink.piece(bytes).map_err(|error| {
                        *abort = Some(error);
                        anyhow::anyhow!("client disconnected during decode")
                    })
                },
                |token| {
                    let position = sequence.position();
                    let next = match dflash_capture.as_ref() {
                        Some((dst, wstart, _, n_features, Some(ring_window)))
                            if position >= *wstart =>
                        {
                            let offset = dflash_ring_offset(position, *wstart, *ring_window)
                                .expect("validated capture ring position and window");
                            let view = dst.view_subrange(
                                (offset * *n_features) as u64,
                                vec![*n_features as u64],
                            );
                            forward
                                .single_token_with_multi_hidden(
                                    token,
                                    u32::try_from(position).context("position does not fit u32")?,
                                    unsafe { sequence.metal_session_mut() },
                                    &self
                                        .dflash_head
                                        .as_ref()
                                        .expect("capture window buffer implies a drafter head")
                                        .target_layer_ids,
                                    &view,
                                )
                                .context("decode token")?
                        }
                        _ => forward
                            .single_token(
                                token,
                                u32::try_from(position).context("position does not fit u32")?,
                                unsafe { sequence.metal_session_mut() },
                            )
                            .context("decode token")?,
                    };
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
        self.finish_generation(
            generation,
            None,
            sequence,
            prompt_ids,
            matched_tokens,
            restore_ms,
            dflash_capture
                .as_ref()
                .and_then(|(dst, wstart, _, n_features, ring)| {
                    ring.map(|window| (dst.clone(), *wstart, *n_features, window))
                }),
            dflash_prefix_replay_key.as_ref(),
            replay_sampling.seed,
            format!("tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1} decode_path=serial"),
        )
    }
}

/// Read the trailing capture window for `consumed` committed positions from
/// the windowed ring: columns `[consumed - window, consumed)` in order,
/// wrapped at the ring boundary.
fn read_window_tail(
    dst: &MetalTensor,
    features: usize,
    consumed: usize,
    wstart: usize,
    window: usize,
) -> Vec<f32> {
    let start = consumed.saturating_sub(window);
    let count = consumed - start;
    let mut tail: Vec<f32> = Vec::with_capacity(count * features);
    unsafe {
        let src = dst.buffer.contents().as_ptr() as *const f32;
        for p in start..consumed {
            let offset = (p - wstart) % window;
            let column = src.add(offset * features);
            let column_dst = tail.as_mut_ptr().add((p - start) * features);
            std::ptr::copy_nonoverlapping(column, column_dst, features);
        }
        tail.set_len(count * features);
    }
    tail
}

impl EngineBackend {
    fn try_cache_boundary(
        &self,
        sequence: &qwen_llm::runtime::Sequence,
        prefix_tokens: &[i32],
        pending_token: Option<i32>,
        final_logits: Option<&[f32]>,
        capture_tail: Option<Vec<f32>>,
        capture_tail_features: usize,
        boundary: &'static str,
    ) {
        let estimate = match self.loaded.estimate_checkpoint_boundary_sizes(
            sequence,
            prefix_tokens.len(),
            pending_token.is_some(),
            final_logits.is_some(),
            capture_tail_features,
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
            capture_tail,
            capture_tail_features,
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
        &mut self,
        generation: crate::GenerationResult,
        dflash: Option<crate::DflashDecodeStats>,
        sequence: qwen_llm::runtime::Sequence,
        prompt_ids: Vec<i32>,
        matched_tokens: usize,
        restore_ms: f64,
        capture_ring: Option<(MetalTensor, usize, usize, usize)>,
        dflash_prefix_replay_key: Option<&DflashPrefixReplayKey>,
        dflash_prefix_replay_seed: u64,
        phases: String,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let completed_boundary_valid = match crate::derive_completed_checkpoint_boundary(
            prompt_ids.len(),
            &generation.tokens,
            generation.transitions,
            sequence.position(),
        ) {
            Ok(boundary) => {
                let pending_token = boundary.pending_token;
                let consumed = boundary.consumed_tokens(&prompt_ids, &generation.tokens);
                let (tail, features) = match capture_ring.as_ref() {
                    Some((dst, wstart, n_features, ring_window)) => (
                        Some(read_window_tail(
                            dst,
                            *n_features,
                            consumed.len(),
                            *wstart,
                            *ring_window,
                        )),
                        *n_features,
                    ),
                    None => (None, 0),
                };
                self.try_cache_boundary(
                    &sequence,
                    &consumed,
                    Some(pending_token),
                    None,
                    tail,
                    features,
                    "completed",
                );
                true
            }
            Err(error) => {
                tracing::warn!("serve: completed boundary derivation failed: {error}");
                false
            }
        };
        if let Some(key) = dflash_prefix_replay_key.filter(|_| completed_boundary_valid) {
            self.dflash_prefix_replay.insert(
                key.clone(),
                dflash_prefix_replay_seed,
                &generation.tokens,
            );
        }

        let (stop_reason, end) = match generation.stop_reason {
            crate::StopReason::Eos => (
                StopReason::Eos,
                GenerationEnd::StopToken(
                    *generation
                        .tokens
                        .last()
                        .expect("EOS generation includes its terminal token"),
                ),
            ),
            crate::StopReason::TokenLimit => (StopReason::TokenLimit, GenerationEnd::TokenLimit),
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
                "serve dflash: off_ctx={} spec_steps={} off_steps={} accepted={}/{} prefix_replay={}/{}/{}/{} drafter_calls={} alpha_backoff={} reason={} backoff_probes={} probe_ms={:.1} fallback={}/{:.1}ms draft_first_ms={:.1} draft_steady_ms={:.1} verify_ms={:.1}",
                stats.off_ctx,
                stats.spec_steps,
                stats.off_steps,
                stats.accepted_drafts,
                stats.drafts_scored,
                stats.prefix_replay_steps,
                stats.prefix_replay_accepted_drafts,
                stats.prefix_replay_drafts_scored,
                stats.prefix_replay_mismatches,
                stats.drafter_calls,
                stats.alpha_backoff,
                stats.backoff_reason.map_or("none", crate::DflashBackoffReason::as_str),
                stats.backoff_probe_steps,
                stats.backoff_probe_ms,
                stats.fallback_calls,
                stats.fallback_ms,
                stats.draft_first_call_ms,
                stats.draft_ms,
                stats.verify_ms,
            );
        }
        Ok(GenerationOutcome {
            end,
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

    fn replay_cache() -> DflashPrefixReplayCache {
        DflashPrefixReplayCache {
            enabled: true,
            entry: None,
        }
    }

    #[test]
    fn dflash_prefix_replay_key_allows_new_seeds_but_not_new_sampling_shapes() {
        let first = dflash_prefix_replay_key(
            &[1, 2, 3],
            SamplingConfig {
                temperature: 0.7,
                top_k: 200,
                top_p: 1.0,
                min_p: 0.05,
                seed: 101,
            },
        );
        let mut changed_seed = SamplingConfig {
            temperature: 0.7,
            top_k: 200,
            top_p: 1.0,
            min_p: 0.05,
            seed: 102,
        };
        assert_eq!(first, dflash_prefix_replay_key(&[1, 2, 3], changed_seed));
        changed_seed.min_p = 0.1;
        assert_ne!(first, dflash_prefix_replay_key(&[1, 2, 3], changed_seed));
    }

    #[test]
    fn dflash_prefix_replay_cache_is_single_entry_and_bounded() {
        let mut cache = replay_cache();
        let config = SamplingConfig::default();
        let first = dflash_prefix_replay_key(&[1], config);
        let second = dflash_prefix_replay_key(&[2], config);
        cache.insert(first.clone(), 1, &[7; 8]);
        assert_eq!(cache.lookup(&first), Some(vec![7; 8]));

        cache.insert(second.clone(), 1, &[8; 9]);
        assert_eq!(cache.lookup(&first), None);
        assert_eq!(cache.lookup(&second), Some(vec![8; 9]));

        cache.insert(first.clone(), 1, &[9; 7]);
        assert_eq!(cache.lookup(&first), None);
        cache.insert(
            first.clone(),
            1,
            &vec![9; DFLASH_PREFIX_REPLAY_MAX_TOKENS + 1],
        );
        assert_eq!(cache.lookup(&first), None);
        assert_eq!(cache.lookup(&second), Some(vec![8; 9]));
    }

    #[test]
    fn dflash_prefix_replay_uses_only_prior_consensus() {
        let mut cache = replay_cache();
        let key = dflash_prefix_replay_key(&[1], SamplingConfig::default());
        cache.insert(key.clone(), 1, &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        cache.insert(key.clone(), 2, &[1, 2, 3, 4, 5, 6, 7, 8, 20, 21]);
        assert_eq!(cache.lookup(&key), Some(vec![1, 2, 3, 4, 5, 6, 7, 8]));

        cache.insert(key.clone(), 3, &[1, 2, 3, 4, 5, 6, 30, 31, 32]);
        assert_eq!(cache.lookup(&key), None);
    }

    #[test]
    fn dflash_prefix_replay_sampled_requires_two_long_consistent_histories() {
        let mut cache = replay_cache();
        let key = dflash_prefix_replay_key(
            &[1],
            SamplingConfig {
                temperature: 0.7,
                ..SamplingConfig::default()
            },
        );
        cache.insert(key.clone(), 1, &[1; 64]);
        assert_eq!(cache.lookup(&key), None);
        cache.insert(key.clone(), 1, &[1; 64]);
        assert_eq!(cache.lookup(&key), None);

        let mut second = vec![1; 64];
        second[31] = 2;
        cache.insert(key.clone(), 2, &second);
        assert_eq!(cache.lookup(&key), None);

        let third = vec![1; 64];
        cache.insert(key.clone(), 3, &third);
        assert_eq!(cache.lookup(&key), None);

        let mut consistent = replay_cache();
        consistent.insert(key.clone(), 1, &[1; 64]);
        consistent.insert(key.clone(), 2, &[1; 64]);
        assert_eq!(consistent.lookup(&key), Some(vec![1; 64]));
    }

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
    fn dflash_requires_capture_window_through_entire_prompt() {
        let lim = usize::MAX;
        assert!(complete_dflash_capture(0, 1, 1, lim));
        assert!(complete_dflash_capture(0, 48, 48, lim));
        assert!(!complete_dflash_capture(1, 47, 48, lim));
        assert!(!complete_dflash_capture(0, 47, 48, lim));
        let w = qwen_llm::metal_dflash::DFLASH_CAPTURE_WINDOW;
        assert!(complete_dflash_capture(0, 100, 100, w));
        assert!(complete_dflash_capture(1028, 2048, 3076, w));
        assert!(!complete_dflash_capture(0, 3076, 3076, w));
        assert!(!complete_dflash_capture(1028, 2047, 3076, w));
        assert!(!complete_dflash_capture(1029, 2048, 3076, w));
        assert!(use_serial_tail(48, 48, true, true));
        assert!(!use_serial_tail(48, 48, true, false));
        assert!(use_serial_tail(48, 48, false, false));
        assert!(should_plan_dflash(true, 0, false, true));
        assert!(!should_plan_dflash(true, 0, false, false));
        assert!(!should_plan_dflash(true, 1, false, true));
        assert!(should_plan_dflash(true, 5, true, true));
        assert!(!should_plan_dflash(true, 5, true, false));
        assert!(!should_plan_dflash(true, 5, false, true));
        assert!(!should_plan_dflash(false, 0, true, true));
        assert!(restored_dflash_capture_complete(5, 0, 5));
        assert!(!restored_dflash_capture_complete(5, 0, 0));
        assert!(restored_dflash_capture_complete(5, 10, 0));
        assert!(!restored_dflash_capture_complete(5, 0, 6));
        assert_eq!(dflash_ring_offset(10, 8, 3), Some(2));
        assert_eq!(dflash_ring_offset(11, 8, 3), Some(0));
        assert_eq!(dflash_ring_offset(7, 8, 3), None);
        assert_eq!(dflash_ring_offset(8, 8, 0), None);
        let restored_extension_offsets: Vec<_> = (2..6)
            .map(|position| dflash_prompt_capture_offset(position, 3))
            .collect();
        assert_eq!(
            restored_extension_offsets,
            [None, Some(0), Some(1), Some(2)]
        );
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
        assert_eq!(dflash_capture_elements(48, 32, usize::MAX), Ok(1536));
        let w = qwen_llm::metal_dflash::DFLASH_CAPTURE_WINDOW;
        assert_eq!(dflash_capture_elements(3076, 32, w), Ok(2048 * 32));
        assert!(dflash_capture_elements(usize::MAX, 2, usize::MAX).is_err());
    }

    #[test]
    fn default_context_cap_is_finite_but_allocates_per_request() {
        assert_eq!(request_capacity(133_000, 1_000, None), Ok(134_016));
        assert_eq!(request_capacity(100, 20, None), Ok(136));
        assert!(request_capacity(super::super::DEFAULT_SERVE_MAX_CONTEXT_TOKENS, 1, None).is_err());
        assert_eq!(request_capacity(100, 20, Some(1024)), Ok(1024));
    }
}
