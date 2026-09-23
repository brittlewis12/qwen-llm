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
use super::output_partition::{GenerationEnd, OutputProtocol, ToolGrammar};
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
use qwen_llm::snapshot_policy::EntryId;
use qwen_llm::tokenizer::{Tokenizer, token_ids_sha256_i32le};
use std::collections::VecDeque;
use std::io;
use std::time::Instant;

const DFLASH_FIXED_SCRATCH_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
const RESTORED_TAIL_SCRATCH_LIMIT: u64 = 128 * 1024 * 1024;
/// Longest generation header (`<|im_start|>assistant\n` plus a preopened or
/// preclosed think block) separating a rendered transcript from its
/// generation suffix.
const TRANSCRIPT_HEADER_MAX_TOKENS: usize = 16;
pub(super) const IM_START_MARKER: &str = "<|im_start|>";

#[cfg(test)]
#[path = "restored_tail_pilot.rs"]
mod restored_tail_pilot;
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
    /// Effective admission ceiling: the explicit `--max-context-tokens`, else
    /// the smaller of the hard default and the model's declared context.
    context_ceiling: usize,
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
        context_ceiling: usize,
        drafter: Option<&std::path::Path>,
        template: super::items::QwenTemplate,
        no_thinking_supported: bool,
    ) -> anyhow::Result<Self> {
        let tokenizer = loaded.tokenizer().context("initialize serve tokenizer")?;
        anyhow::ensure!(
            drafter.is_none() || loaded.arch().kind == qwen_llm::model::ArchKind::Dense,
            "serve DFlash speculation currently supports dense targets only; this MoE model would run serially, so omit --drafter"
        );
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
            context_ceiling,
            template,
            no_thinking_supported,
            dflash_head,
            dflash_prefix_replay: DflashPrefixReplayCache::from_env(),
        })
    }
}

fn preopens(template: QwenTemplate, request: &ServeRequest) -> bool {
    let mut bound = request.clone();
    bound.template = template;
    crate::open_responses::render::qwen_generation(&bound)
        == crate::open_responses::render::QwenGeneration::PreOpen
}

/// Sampler for a Qwen or DeepSeek V4 serve request with the greedy-leaning
/// serve defaults (temperature 0, top_k 200, min_p 0.05).
pub(crate) fn request_sampler(request: &ServeRequest) -> Result<Sampler, ServeError> {
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

/// Token index of the generation header's `<|im_start|>`: the end of the
/// rendered transcript. Qwen templates re-render a prior assistant turn
/// differently from the generation suffix that preceded it (`<think>\n`
/// versus `<think>\n\n</think>` when history reasoning is dropped), so
/// prompt-end and completed snapshots stop prefixing the next request one
/// token after `<think>`. The boundary before the header survives that
/// re-render. A hybrid's recurrent state cannot be truncated, so the
/// snapshot has to be taken here rather than recovered later.
pub(super) fn transcript_boundary<T: PartialEq>(prompt_ids: &[T], im_start: T) -> Option<usize> {
    let boundary = prompt_ids.iter().rposition(|id| *id == im_start)?;
    let header = prompt_ids.len() - boundary;
    (boundary > 0 && (2..=TRANSCRIPT_HEADER_MAX_TOKENS).contains(&header)).then_some(boundary)
}

/// Largest prompt remainder prefilled one token at a time instead of as a
/// chunk. Both paths are correct for every model; this only picks the
/// cheaper one. Measured 2026-09-23 on M4 Max with restored remainders at
/// 3.2K and 29K context: a serial token costs ~41-48 ms on dense 27B and
/// ~10-11 ms on 35B-A3B, while a small chunked prefill has a floor of
/// ~390-640 ms dense and ~170-300 ms MoE, so serial wins only below ~10-13
/// tokens dense and ~17-25 MoE. The previous fixed 48 made 20-47 token
/// remainders 2-3x slower than chunked.
fn serial_tail_max(kind: qwen_llm::model::ArchKind) -> usize {
    match kind {
        qwen_llm::model::ArchKind::Dense => 12,
        _ => 20,
    }
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

/// Whether the request prefills a chunk (and so needs prefill scratch):
/// anything but an exact hit or a remainder small enough for the serial tail.
fn needs_prefill_scratch(
    serial_tail_max: usize,
    prompt_tokens: usize,
    restored_tokens: usize,
    exact_with_logits: bool,
) -> bool {
    !exact_with_logits
        && !prompt_tokens
            .checked_sub(restored_tokens)
            .is_some_and(|remaining| (1..=serial_tail_max).contains(&remaining))
}

/// Geometry the single-chunk packed kernels and their scratch pricing are
/// sized for (the 27B dense shape shared by Qwen3.5/3.6/3.8). A
/// specialization: anything else takes the general chunked path.
fn bounded_packed_dense_arch(arch: &qwen_llm::model::Arch) -> bool {
    arch.kind == qwen_llm::model::ArchKind::Dense
        && arch.n_layer == 64
        && arch.hidden_size == 5120
        && arch.n_q_heads == 24
        && arch.n_kv_heads == 4
        && arch.attn_head_dim == 256
}

fn fresh_packed_arch(
    arch: &qwen_llm::model::Arch,
    template: QwenTemplate,
    lm_head_dtype: qwen_llm::tensor::GgmlType,
) -> bool {
    bounded_packed_dense_arch(arch)
        && template == QwenTemplate::Qwen38
        && lm_head_dtype == qwen_llm::tensor::GgmlType::Q8_0
}

fn fresh_packed_enabled(value: Option<&std::ffi::OsStr>) -> bool {
    value.is_some_and(|value| value == "1")
}

fn fresh_packed_width(
    enabled: bool,
    qualified: bool,
    has_drafter: bool,
    greedy: bool,
    has_cached_prefix: bool,
    prompt: usize,
) -> Option<usize> {
    (enabled
        && qualified
        && !has_drafter
        && greedy
        && !has_cached_prefix
        && (19..=48).contains(&prompt))
    .then_some(prompt)
}

/// Restored 7-32 token remainders on the 27B geometry prefill in one packed
/// block (a specialization over the chunked path, which is the fallback if
/// the packed plan is not admitted). Not limited by template, weight or
/// head dtype, or greedy sampling: packed and chunked prefill are the same
/// numerical class as a fresh prefill.
fn restored_packed_tail_width(
    qualified_dense: bool,
    has_drafter: bool,
    prompt: usize,
    restored: usize,
    exact: bool,
) -> Option<usize> {
    let remaining = prompt.checked_sub(restored)?;
    (qualified_dense && !has_drafter && !exact && restored > 0 && (7..=32).contains(&remaining))
        .then_some(remaining)
}

fn allocate_single_chunk_request_state(
    loaded: &LoadedModel,
    capacity: usize,
    plan: qwen_llm::metal_dflash::PrefillScratchPlan,
) -> anyhow::Result<(usize, Option<MetalDFlashLayerMajorScratch>, Sequence)> {
    let chunk = plan.block_size() as usize;
    let scratch = MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(
        loaded.context(),
        loaded.metal_model(),
        plan,
    )?;
    let sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    Ok((chunk, Some(scratch), sequence))
}

fn admit_optional_tail<P, E>(
    mut candidate: Option<(P, u64)>,
    baseline_price: u64,
    mut admit: impl FnMut(u64) -> Result<qwen_llm::metal::MetalMemoryAdmission, E>,
) -> Result<(Option<P>, qwen_llm::metal::MetalMemoryAdmission), E> {
    let mut admission = admit(
        candidate
            .as_ref()
            .map_or(baseline_price, |(_, price)| *price),
    )?;
    if !admission.admitted && candidate.take().is_some() {
        admission = admit(baseline_price)?;
    }
    Ok((candidate.map(|(plan, _)| plan), admission))
}

fn allocate_optional_tail<P, T, E>(
    candidate: Option<P>,
    allocate: impl FnOnce(P) -> Result<T, E>,
    fallback: impl FnOnce() -> Result<T, E>,
) -> Result<T, E> {
    match candidate {
        Some(plan) => allocate(plan).or_else(|_| fallback()),
        None => fallback(),
    }
}

fn allocate_serve_request_state(
    loaded: &LoadedModel,
    prompt_tokens: usize,
    capacity: usize,
    needs_scratch: bool,
) -> anyhow::Result<(usize, Option<MetalDFlashLayerMajorScratch>, Sequence)> {
    if needs_scratch {
        let allocated = crate::allocate_prefill_request_state(
            loaded,
            crate::PrefillChunkArg::Auto,
            prompt_tokens,
            capacity,
            true,
        )?;
        Ok((
            allocated.chunk,
            Some(allocated.scratch.into_inner()),
            allocated.sequence,
        ))
    } else {
        Ok((
            crate::baseline_prefill_chunk(prompt_tokens),
            None,
            loaded.create_sequence(SequenceConfig::new(capacity))?,
        ))
    }
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
    context_ceiling: usize,
) -> Result<usize, &'static str> {
    let required = prompt_tokens
        .checked_add(generation_tokens)
        .ok_or("prompt plus generation token count overflow")?;
    let limit = configured_limit.unwrap_or(context_ceiling);
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
        let single_chunk = scratch
            .as_ref()
            .is_some_and(|s| s.prefill_scratch_plan().is_single_chunk());
        if !single_chunk
            && use_serial_tail(
                remaining,
                serial_tail_max(loaded.arch().kind),
                dflash_capture.is_some(),
                serial_capture_supported,
            )
        {
            for (offset, &token) in prompt_ids[start..].iter().enumerate() {
                let position = start + offset;
                let position_u32 = u32::try_from(position)
                    .map_err(|_| ServeError::server_error("position overflow"))?;
                let skip_tail = serial_capture_supported && position + 1 < prompt_ids.len();
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
                        let logits = if skip_tail {
                            forward
                                .single_token_with_multi_hidden_no_tail(
                                    token,
                                    position_u32,
                                    unsafe { sequence.metal_session_mut() },
                                    &head.target_layer_ids,
                                    &view,
                                )
                                .map(|()| None)
                        } else {
                            forward
                                .single_token_with_multi_hidden(
                                    token,
                                    position_u32,
                                    unsafe { sequence.metal_session_mut() },
                                    &head.target_layer_ids,
                                    &view,
                                )
                                .map(Some)
                        }
                        .map_err(|error| {
                            ServeError::server_error(format!(
                                "serial tail capture prefill: {error:#}"
                            ))
                        })?;
                        sequence.advance_by(1).map_err(|error| {
                            ServeError::server_error(format!("advance: {error:#}"))
                        })?;
                        *captured += 1;
                        logits
                    }
                    _ if skip_tail => loaded
                        .prefill_token_prompt_only(sequence, token)
                        .map(|()| None)
                        .map_err(|error| {
                            ServeError::server_error(format!("serial no-tail prefill: {error:#}"))
                        })?,
                    _ => loaded
                        .decode_token(sequence, token)
                        .map(Some)
                        .map_err(|error| {
                            ServeError::server_error(format!("serial tail prefill: {error:#}"))
                        })?,
                };
                prompt_logits = logits;
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

    fn idle(&mut self) {
        super::log_expired_snapshots("qwen", &self.loaded.sweep_prefix_cache());
    }

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: preopens(self.template, request),
            parse_tools: true,
            tool_grammar: ToolGrammar::QwenXml,
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
        let capacity = request_capacity(
            prompt_ids.len(),
            max_tokens,
            self.max_context_tokens,
            self.context_ceiling,
        )
        .map_err(|message| {
            ServeError::invalid_request(
                Some("max_output_tokens"),
                format!(
                    "{message}: max context {} is smaller than prompt {} + generation {max_tokens}",
                    self.context_ceiling,
                    prompt_ids.len(),
                ),
            )
        })?;

        let restore_t0 = Instant::now();
        let cached_lookup = self.loaded.lookup_cached_prefix(&prompt_ids);
        let exact_cached = cached_lookup
            .as_ref()
            .is_some_and(|lookup| lookup.is_exact_with_final_logits());
        let dense = self.loaded.arch().kind == qwen_llm::model::ArchKind::Dense;
        let serial_tail_max = serial_tail_max(self.loaded.arch().kind);
        let needs_scratch = needs_prefill_scratch(
            serial_tail_max,
            prompt_ids.len(),
            cached_lookup
                .as_ref()
                .map_or(0, |lookup| lookup.restored_prefix_len()),
            exact_cached,
        );
        // The retained lookup pins the restore boundary through allocation.
        // Packed execution still admits its complete fallback topology.
        let prefill_scratch_upper_bytes = if !needs_scratch {
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
        let packed_width = fresh_packed_width(
            fresh_packed_enabled(std::env::var_os("QWEN_SERVE_FRESH_PACKED").as_deref()),
            fresh_packed_arch(
                &self.loaded.arch(),
                self.template,
                self.loaded.metal_model().lm_head.dtype,
            ),
            self.dflash_head.is_some(),
            sampler.config().temperature == 0.0,
            cached_lookup.is_some(),
            prompt_ids.len(),
        )
        .or_else(|| {
            restored_packed_tail_width(
                bounded_packed_dense_arch(&self.loaded.arch()),
                self.dflash_head.is_some(),
                prompt_ids.len(),
                cached_lookup
                    .as_ref()
                    .map_or(0, |lookup| lookup.restored_prefix_len()),
                exact_cached,
            )
        });
        let packed_tail_plan = packed_width.and_then(|width| {
            let plan = qwen_llm::metal_dflash::plan_single_chunk_prefill_scratch(
                self.loaded.metal_model(),
                width as u32,
                prompt_ids.len(),
                PrefillScratchConfig::default(),
            )
            .ok()?;
            if plan.matrix_max_pos() < prompt_ids.len() as u64 {
                return None;
            }
            let price = plan
                .priced_upper_bound(|bytes| {
                    Ok(self
                        .loaded
                        .context()
                        .shared_buffer_size_and_align(bytes)?
                        .size)
                })
                .ok()?;
            (price <= RESTORED_TAIL_SCRATCH_LIMIT).then_some((plan, price))
        });
        let (packed_tail_plan, admission) =
            admit_optional_tail(packed_tail_plan, prefill_scratch_upper_bytes, |price| {
                self.loaded
                    .qwen_execution_memory_admission(1, capacity, price, 0)
            })
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
        let allocation = allocate_optional_tail(
            packed_tail_plan,
            |plan| {
                allocate_single_chunk_request_state(&self.loaded, capacity, plan)
                .inspect_err(|error| {
                    tracing::warn!("serve: single-chunk tail allocation failed; retaining serial prefill: {error:#}");
                })
            },
            || {
                allocate_serve_request_state(
                    &self.loaded,
                    prompt_ids.len(),
                    capacity,
                    needs_scratch,
                )
            },
        );
        let (mut chunk, mut scratch, mut sequence) = allocation.map_err(|error| {
            ServeError::server_error(format!("allocate request state: {error:#}"))
        })?;
        // Which prefill path the remainder takes, for the phases line: a slow
        // path is never silent.
        let prefill_path = if exact_cached {
            "exact"
        } else if scratch
            .as_ref()
            .is_some_and(|s| s.prefill_scratch_plan().is_single_chunk())
        {
            "single_chunk"
        } else if scratch.is_none() {
            "serial_tail"
        } else {
            "chunked"
        };
        if let Some(selected_scratch) = scratch
            .as_ref()
            .filter(|s| s.prefill_scratch_plan().is_single_chunk())
        {
            if cached_lookup.is_none() {
                let plan = selected_scratch.prefill_scratch_plan();
                tracing::info!(target: "qwen_diag", "serve prefill: fresh_packed rows={chunk} query_rows={} matrix_max_pos={}", plan.matrix_query_rows(), plan.matrix_max_pos());
            } else {
                tracing::info!(target: "qwen_diag", "serve prefill: restored_packed_tail rows={chunk}");
            }
        }
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
        // Transcript-boundary capture: stop prefill before the generation
        // header, snapshot, then finish the header. Drafter capture windows
        // are sized to the whole prompt, so speculative requests keep the
        // single-pass prefill.
        let transcript_split = self
            .tokenizer
            .encode(IM_START_MARKER, false)
            .ok()
            .and_then(|ids| (ids.len() == 1).then(|| ids[0]))
            .and_then(|im_start| transcript_boundary(&prompt_ids, im_start))
            .filter(|&boundary| {
                boundary > sequence.position() && !speculate && dflash_capture.is_none()
            });
        let prefill_t0 = Instant::now();
        let mut transcript_capture_ms = 0.0;
        let mut transcript_entry = None;
        if let Some(boundary) = transcript_split {
            prefill_remaining(
                &self.loaded,
                &forward,
                self.dflash_head.as_ref(),
                false,
                &prompt_ids[..boundary],
                chunk,
                &mut sequence,
                &mut scratch,
                &mut dflash_capture,
                sink,
            )?;
            let capture_t0 = Instant::now();
            transcript_entry = self.try_cache_boundary(
                &sequence,
                &prompt_ids[..boundary],
                None,
                None,
                None,
                0,
                "transcript",
            );
            transcript_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
        }
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
                let allocated = allocate_serve_request_state(
                    &self.loaded,
                    prompt_ids.len(),
                    capacity,
                    needs_scratch,
                )
                .map_err(|retry_error| {
                    ServeError::server_error(format!(
                        "allocate serial fallback request state after DFlash capture failure ({capture_error}): {retry_error:#}"
                    ))
                })?;
                alloc_ms += retry_alloc_t0.elapsed().as_secs_f64() * 1e3;
                (chunk, scratch, sequence) = allocated;
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
        let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3 - transcript_capture_ms;
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
        // prompt was already an exact hit, or when the transcript boundary a
        // few header tokens earlier already holds this state's reusable
        // prefix). The capture window buffer holds the prompt's trailing
        // columns; publish them as the drafter tail.
        let capture_t0 = Instant::now();
        if !restore.as_ref().is_some_and(|restore| restore.exact) && transcript_entry.is_none() {
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

        let prompt_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3 + transcript_capture_ms;
        let transcript_phase = match transcript_split {
            Some(boundary) => format!(
                " transcript_boundary={boundary} transcript_captured={}",
                transcript_entry.is_some()
            ),
            None => String::new(),
        };
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
                    transcript_entry,
                    format!(
                        "tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1}{transcript_phase} prefill_path={prefill_path} decode_path=dflash"
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
                            let next = forward
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
                                .context("decode token")?;
                            sequence.advance_by(1)?;
                            next
                        }
                        _ => self
                            .loaded
                            .decode_token(&mut sequence, token)
                            .context("decode token")?,
                    };
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
            transcript_entry,
            format!("tokenize_ms={tokenize_ms:.1} alloc_ms={alloc_ms:.1} restore_ms={restore_ms:.1} prefill_ms={prefill_ms:.1} prompt_capture_ms={prompt_capture_ms:.1}{transcript_phase} prefill_path={prefill_path} decode_path=serial"),
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
    ) -> Option<EntryId> {
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
                return None;
            }
        };
        // Pinned entries (this request's transcript boundary) count against
        // the budget: they are never evicted to admit this one.
        if !self.loaded.prefix_cache_strict_eligible(estimate) {
            tracing::warn!(
                "serve: {boundary} snapshot denied by cache budget; snapshot_bytes={estimate} pinned_bytes={} cache_budget_bytes={}",
                self.loaded.prefix_cache_pinned_bytes(),
                self.loaded.prefix_cache_stats().max_indexed_bytes,
            );
            return None;
        }
        if let Err((reason, signals)) = super::admit_snapshot_capture(
            estimate,
            || self.loaded.context().memory_signals(),
            |bytes| self.loaded.evict_prefix_cache_for(bytes),
        ) {
            tracing::warn!(
                "serve: {boundary} snapshot denied by memory headroom; reason={reason:?} snapshot_bytes={estimate} metal_current_bytes={} metal_recommended_bytes={} process_remaining_bytes={:?}",
                signals.current_allocated_bytes,
                signals.recommended_max_bytes,
                signals.process_limit_remaining_bytes,
            );
            return None;
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
                Ok(Some(inserted)) => Some(inserted.entry),
                Ok(None) => {
                    tracing::warn!(
                        "serve: {boundary} snapshot rejected at strict cache insertion; request continues"
                    );
                    None
                }
                Err(error) => {
                    tracing::warn!(
                        "serve: {boundary} strict cache insertion failed; request continues: {error}"
                    );
                    None
                }
            },
            Err(error) => {
                tracing::warn!(
                    "serve: {boundary} snapshot capture failed; request continues: {error}"
                );
                None
            }
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
        transcript_entry: Option<EntryId>,
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
                // The transcript boundary is what the next turn reuses when
                // the template re-renders this turn; never evict it to admit
                // the completed boundary.
                let pinned =
                    transcript_entry.filter(|&entry| self.loaded.pin_prefix_cache_entry(entry));
                self.try_cache_boundary(
                    &sequence,
                    &consumed,
                    Some(pending_token),
                    None,
                    tail,
                    features,
                    "completed",
                );
                if let Some(entry) = pinned {
                    self.loaded.unpin_prefix_cache_entry(entry);
                }
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
        tracing::info!(target: "qwen_diag", "serve phases: family=qwen {phases}");
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
        assert!(use_serial_tail(12, 12, true, true));
        assert!(!use_serial_tail(12, 12, true, false));
        assert!(use_serial_tail(12, 12, false, false));
        assert!(!use_serial_tail(13, 12, false, true));
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
    fn transcript_boundary_is_the_generation_header_start() {
        const IM: i32 = 7;
        // system, user, then `<|im_start|>assistant\n<think>\n`.
        let prompt = [IM, 1, 2, 3, IM, 4, 5, 6, IM, 8, 9, 10, 11];
        assert_eq!(transcript_boundary(&prompt, IM), Some(8));
        // Preclosed headers are longer but still bounded.
        let mut preclosed = vec![IM, 1, 2, IM];
        preclosed.extend([8, 9, 10, 12, 13, 14]);
        assert_eq!(transcript_boundary(&preclosed, IM), Some(3));
        // A trailing marker with no header, a marker at position zero, a
        // tail longer than any header, or no marker at all: no boundary.
        assert_eq!(transcript_boundary(&[1, 2, IM], IM), None);
        assert_eq!(transcript_boundary(&[IM, 1, 2, 3], IM), None);
        let mut long_tail = vec![1, IM];
        long_tail.extend(std::iter::repeat_n(3, TRANSCRIPT_HEADER_MAX_TOKENS));
        assert_eq!(transcript_boundary(&long_tail, IM), None);
        assert_eq!(transcript_boundary(&[1, 2, 3], IM), None);
    }

    #[test]
    fn serial_tail_scratch_plan_uses_consumed_not_matched_tokens() {
        use qwen_llm::model::ArchKind;
        let (dense, moe) = (
            serial_tail_max(ArchKind::Dense),
            serial_tail_max(ArchKind::Moe),
        );
        assert_eq!((dense, moe), (12, 20));
        for max in [dense, moe] {
            for restored in [0, 8192, 32768] {
                for tail in [1, 2, max] {
                    assert!(!needs_prefill_scratch(
                        max,
                        restored + tail,
                        restored,
                        false
                    ));
                }
                assert!(needs_prefill_scratch(
                    max,
                    restored + max + 1,
                    restored,
                    false
                ));
                assert!(needs_prefill_scratch(max, restored + 48, restored, false));
                assert!(needs_prefill_scratch(max, restored, restored, false));
                assert!(!needs_prefill_scratch(max, restored, restored, true));
            }
            // The matched pending token is still one of the rows to execute.
            assert!(needs_prefill_scratch(max, 8193 + max, 8192, false));
            assert!(!needs_prefill_scratch(max, 8193 + max - 1, 8192, false));
            assert!(needs_prefill_scratch(max, 8, 9, false));
        }
    }

    #[test]
    #[ignore = "loads a local model and prices real Metal request allocations"]
    fn serial_tail_scratch_allocation_inventory() {
        let model = std::env::var("QWEN_NO_TAIL_TEST_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
        let runtime = qwen_llm::runtime::Runtime::metal().expect("Metal runtime");
        let loaded = runtime.load_model(&model).expect("explicit local fixture");
        for prompt_tokens in [32, 8192 + 16, 32768 + 16] {
            let mut allocated_bytes = Vec::new();
            for needs_scratch in [true, false] {
                let before = loaded.context().current_allocated_size();
                let state = allocate_serve_request_state(
                    &loaded,
                    prompt_tokens,
                    prompt_tokens + 128,
                    needs_scratch,
                )
                .unwrap();
                assert_eq!(state.1.is_some(), needs_scratch);
                let allocated = loaded
                    .context()
                    .current_allocated_size()
                    .checked_sub(before)
                    .unwrap();
                allocated_bytes.push(allocated);
                drop(state);
                assert_eq!(loaded.context().current_allocated_size(), before);
            }
            assert!(allocated_bytes[0] > allocated_bytes[1]);
            eprintln!(
                "serial-tail-allocation prompt={prompt_tokens} baseline={} candidate={} removed={}",
                allocated_bytes[0],
                allocated_bytes[1],
                allocated_bytes[0] - allocated_bytes[1]
            );
        }
    }

    #[test]
    #[ignore = "loads a local model and checks Metal scratch teardown"]
    fn released_prefill_scratch_preserves_snapshot_and_continuation() {
        struct Sink;
        impl GenerationSink for Sink {
            fn piece(&mut self, _: &[u8]) -> io::Result<()> {
                panic!("prefill must not emit")
            }
            fn tick(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let model = std::env::var("QWEN_NO_TAIL_TEST_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
        let runtime = qwen_llm::runtime::Runtime::metal().unwrap();
        let loaded = runtime.load_model(&model).unwrap();
        let tokens = loaded
            .tokenizer()
            .unwrap()
            .encode("The quick brown fox jumps over the lazy dog.", false)
            .unwrap();
        let prompt: Vec<_> = tokens.iter().copied().cycle().take(65).collect();
        let forward = loaded.forward();
        let mut states = Vec::new();
        let mut retained_scratch = None;
        let mut logits = Vec::new();
        for release in [false, true] {
            let (chunk, mut scratch, mut sequence) =
                allocate_serve_request_state(&loaded, prompt.len(), 80, true).unwrap();
            logits.push(
                prefill_remaining(
                    &loaded,
                    &forward,
                    None,
                    false,
                    &prompt,
                    chunk,
                    &mut sequence,
                    &mut scratch,
                    &mut None,
                    &mut Sink,
                )
                .unwrap_or_else(|_| panic!("packed prefill"))
                .unwrap(),
            );
            if release {
                let before = loaded.context().current_allocated_size();
                drop(scratch);
                assert!(loaded.context().current_allocated_size() < before);
            } else {
                retained_scratch = scratch;
            }
            states.push(sequence);
        }
        assert!(
            logits[0]
                .iter()
                .zip(&logits[1])
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        let pending = prompt[0];
        for consumed in [false, true] {
            let mut snapshots = Vec::new();
            let mut continuation_logits = Vec::new();
            for sequence in &mut states {
                let mut history = prompt.clone();
                if consumed {
                    continuation_logits.push(
                        forward
                            .single_token(pending, sequence.position() as u32, unsafe {
                                sequence.metal_session_mut()
                            })
                            .unwrap(),
                    );
                    sequence.advance_by(1).unwrap();
                    history.push(pending);
                }
                snapshots.push(
                    sequence
                        .metal_session()
                        .snapshot(loaded.snapshot_identity(sequence).unwrap(), history, None)
                        .unwrap(),
                );
            }
            let (a, b) = (&snapshots[0], &snapshots[1]);
            assert_eq!(a.kv_n_pos, b.kv_n_pos);
            assert!(a.kv_k_arena == b.kv_k_arena);
            assert!(a.kv_v_arena == b.kv_v_arena);
            assert!(a.gdn_conv_arena == b.gdn_conv_arena);
            assert!(a.gdn_state_arena == b.gdn_state_arena);
            if consumed {
                assert!(
                    continuation_logits[0]
                        .iter()
                        .zip(&continuation_logits[1])
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
            }
        }
        drop(retained_scratch);
    }

    #[test]
    #[ignore = "loads a local model and runs serial Metal work"]
    fn serial_prefill_tail_keeps_final_logits_and_persistent_state() {
        struct Sink;
        impl GenerationSink for Sink {
            fn piece(&mut self, _: &[u8]) -> io::Result<()> {
                panic!("prefill must not emit")
            }
            fn tick(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let model = std::env::var("QWEN_NO_TAIL_TEST_MODEL")
            .unwrap_or_else(|_| "/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf".into());
        let runtime = qwen_llm::runtime::Runtime::metal().expect("Metal runtime");
        let loaded = runtime.load_model(&model).expect("explicit local fixture");
        let tokens = loaded
            .tokenizer()
            .unwrap()
            .encode("The quick brown fox jumps over the lazy dog.", false)
            .unwrap();
        let forward = loaded.forward();
        for prefix in [0, 8] {
            for tail in [1, 2, 12] {
                let prompt: Vec<_> = tokens.iter().copied().cycle().take(prefix + tail).collect();
                let mut reference = loaded
                    .create_sequence(SequenceConfig::new(prompt.len() + 2))
                    .unwrap();
                let (chunk, mut scratch, mut candidate) = allocate_serve_request_state(
                    &loaded,
                    prompt.len(),
                    prompt.len() + 2,
                    needs_prefill_scratch(
                        serial_tail_max(qwen_llm::model::ArchKind::Dense),
                        prompt.len(),
                        prefix,
                        false,
                    ),
                )
                .unwrap();
                assert!(scratch.is_none());
                let mut expected = Vec::new();
                for (position, &token) in prompt.iter().enumerate() {
                    expected = forward
                        .single_token(token, position as u32, unsafe {
                            reference.metal_session_mut()
                        })
                        .unwrap();
                    reference.advance_by(1).unwrap();
                    if position < prefix {
                        forward
                            .single_token(token, position as u32, unsafe {
                                candidate.metal_session_mut()
                            })
                            .unwrap();
                        candidate.advance_by(1).unwrap();
                    }
                }
                let actual = prefill_remaining(
                    &loaded,
                    &forward,
                    None,
                    false,
                    &prompt,
                    chunk,
                    &mut candidate,
                    &mut scratch,
                    &mut None,
                    &mut Sink,
                )
                .unwrap_or_else(|_| panic!("prefill prefix={prefix} tail={tail}"))
                .expect("final prompt row must produce logits");
                assert_eq!(candidate.position(), prompt.len());
                assert_eq!(actual.len(), expected.len());
                assert!(
                    actual
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.to_bits() == b.to_bits())
                );
                let identity = loaded.snapshot_identity(&reference).unwrap();
                let a = reference
                    .metal_session()
                    .snapshot(identity.clone(), prompt.clone(), None)
                    .unwrap();
                let b = candidate
                    .metal_session()
                    .snapshot(identity, prompt, None)
                    .unwrap();
                assert_eq!(a.kv_n_pos, b.kv_n_pos);
                assert!(a.kv_k_arena == b.kv_k_arena);
                assert!(a.kv_v_arena == b.kv_v_arena);
                assert!(a.gdn_conv_arena == b.gdn_conv_arena);
                assert!(a.gdn_state_arena == b.gdn_state_arena);
            }
        }
        let prompt: Vec<_> = tokens.iter().copied().cycle().take(57).collect();
        let mut source = loaded.create_sequence(SequenceConfig::new(64)).unwrap();
        for (position, &token) in prompt[..8].iter().enumerate() {
            forward
                .single_token(token, position as u32, unsafe {
                    source.metal_session_mut()
                })
                .unwrap();
            source.advance_by(1).unwrap();
        }
        let checkpoint = loaded
            .prepare_checkpoint_boundary(
                &source,
                prompt[..8].to_vec(),
                Some(prompt[8]),
                None,
                None,
                0,
            )
            .unwrap();
        loaded
            .cache_prepared_checkpoint_strict(&checkpoint)
            .unwrap()
            .expect("cache insertion");
        // Restored at 8: remainders 1 and 12 stay serial, 13 is chunked.
        for length in [9, 20, 21] {
            let lookup = loaded
                .lookup_cached_prefix(&prompt[..length])
                .expect("pending-token lookup");
            assert_eq!(lookup.restored_prefix_len(), 8);
            assert!(!lookup.is_exact_with_final_logits());
            let needs_scratch = needs_prefill_scratch(
                serial_tail_max(qwen_llm::model::ArchKind::Dense),
                length,
                lookup.restored_prefix_len(),
                lookup.is_exact_with_final_logits(),
            );
            assert_eq!(needs_scratch, length == 21);
            let (_, scratch, mut restored) =
                allocate_serve_request_state(&loaded, length, 64, needs_scratch).unwrap();
            assert_eq!(scratch.is_some(), length == 21);
            let report = loaded
                .restore_prepared_cached_prefix(lookup, &mut restored, &prompt[..length])
                .unwrap();
            assert_eq!(report.matched_prefix_len, 9);
            assert_eq!(restored.position(), 8);
        }
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
    #[ignore = "serial Metal, real 0.8B fresh/exact-hit serial backend parity"]
    fn owned_serial_backend_preserves_emission_and_completed_checkpoint() {
        #[derive(Default)]
        struct Sink(Vec<u8>);
        impl GenerationSink for Sink {
            fn piece(&mut self, bytes: &[u8]) -> io::Result<()> {
                self.0.extend_from_slice(bytes);
                Ok(())
            }
            fn tick(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        let runtime = qwen_llm::runtime::Runtime::metal().unwrap();
        let loaded = runtime
            .load_model("/Users/tito/models/Qwen3.5-0.8B-Q4_K_M.gguf")
            .unwrap();
        let family = qwen_llm::model_family::ModelFamily::detect(loaded.gguf()).unwrap();
        let template = crate::prompt_template::serve_qwen_template(family, loaded.gguf()).unwrap();
        let no_thinking = crate::supports_qwen_no_thinking_prompt(family, loaded.gguf());
        let mut backend = EngineBackend::new(
            loaded,
            "owned-serial-test".into(),
            4,
            Some(128),
            128,
            None,
            template,
            no_thinking,
        )
        .unwrap();
        let request = crate::open_responses::items::parse_request(&serde_json::json!({
            "model":"owned-serial-test", "input":"Hi",
            "temperature":0.0, "max_output_tokens":4,
        }))
        .unwrap();
        let prompt = backend.render_prompt(&request).unwrap();
        let prompt_ids = backend.tokenizer.encode(&prompt, false).unwrap();
        // Serve prefills a fresh prompt in two pieces around the transcript
        // boundary; both must fit the serial tail for the single-token
        // reference below to be bit-identical.
        let im_start = backend.tokenizer.encode(IM_START_MARKER, false).unwrap();
        let boundary = transcript_boundary(&prompt_ids, im_start[0]).unwrap();
        let serial_max = serial_tail_max(qwen_llm::model::ArchKind::Dense);
        assert!(boundary <= serial_max && prompt_ids.len() - boundary <= serial_max);
        let mut reference = backend
            .loaded
            .create_sequence(SequenceConfig::new(128))
            .unwrap();
        let mut logits = Vec::new();
        for (position, &token) in prompt_ids.iter().enumerate() {
            logits = backend
                .loaded
                .forward()
                .single_token(token, position as u32, unsafe {
                    reference.metal_session_mut()
                })
                .unwrap();
            reference.advance_by(1).unwrap();
        }
        let mut expected = Sink::default();
        let generation = crate::generate_serial(
            logits,
            4,
            &backend.loaded.gguf().stop_token_ids().unwrap(),
            &mut request_sampler(&request).unwrap(),
            |token| {
                expected
                    .0
                    .extend(backend.tokenizer.try_decode_piece_bytes_exact(token)?);
                Ok(())
            },
            |token| {
                let position = reference.position();
                let logits =
                    backend
                        .loaded
                        .forward()
                        .single_token(token, position as u32, unsafe {
                            reference.metal_session_mut()
                        })?;
                reference.advance_by(1)?;
                Ok(logits)
            },
        )
        .unwrap();
        assert_eq!(generation.tokens.len(), 4);
        assert_eq!(generation.transitions, 3);
        let mut completed = prompt_ids.clone();
        completed.extend_from_slice(&generation.tokens);
        assert_eq!(reference.position(), completed.len() - 1);
        let identity = backend.loaded.snapshot_identity(&reference).unwrap();
        let expected_state = reference
            .metal_session()
            .snapshot(
                identity.clone(),
                completed[..completed.len() - 1].to_vec(),
                None,
            )
            .unwrap();
        for cached in [false, true] {
            let mut actual = Sink::default();
            let outcome = backend
                .generate(&request, &prompt, &mut actual)
                .unwrap_or_else(|_| panic!("serial backend failed cached={cached}"));
            assert_eq!(actual.0, expected.0);
            assert_eq!(outcome.usage.output_tokens, 4);
            // An exact resend restores the transcript boundary and prefills
            // only the generation header.
            assert_eq!(
                outcome.usage.cached_tokens,
                if cached { boundary } else { 0 }
            );
            let lookup = backend.loaded.lookup_cached_prefix(&completed).unwrap();
            let mut restored = backend
                .loaded
                .create_sequence(SequenceConfig::new(128))
                .unwrap();
            let report = backend
                .loaded
                .restore_prepared_cached_prefix(lookup, &mut restored, &completed)
                .unwrap();
            assert!(report.exact);
            assert_eq!(restored.position(), completed.len() - 1);
            let actual_state = restored
                .metal_session()
                .snapshot(
                    identity.clone(),
                    completed[..completed.len() - 1].to_vec(),
                    None,
                )
                .unwrap();
            assert_eq!(actual_state.kv_k_arena, expected_state.kv_k_arena);
            assert_eq!(actual_state.kv_v_arena, expected_state.kv_v_arena);
            assert_eq!(actual_state.gdn_conv_arena, expected_state.gdn_conv_arena);
            assert_eq!(actual_state.gdn_state_arena, expected_state.gdn_state_arena);
        }
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
        const HARD: usize = super::super::DEFAULT_SERVE_MAX_CONTEXT_TOKENS;
        assert_eq!(request_capacity(133_000, 1_000, None, HARD), Ok(134_016));
        assert_eq!(request_capacity(100, 20, None, HARD), Ok(136));
        assert!(request_capacity(HARD, 1, None, HARD).is_err());
        assert_eq!(request_capacity(100, 20, Some(1024), HARD), Ok(1024));
        // A model declaring a shorter context lowers the default ceiling.
        assert!(request_capacity(40_000, 1, None, 32_768).is_err());
        assert_eq!(request_capacity(100, 20, None, 32_768), Ok(136));
    }
}
