//! Resident Qwen3.8-Flash-Next [`GenerationBackend`].
//!
//! The text session's workspace is allocated once at load for a fixed
//! forward limit. Healthy ordinary turns retain their exact consumed token key
//! and workspace; compatible continuations avoid reset and restore. Branches use
//! immutable RAM checkpoints, with completed live state copied on departure.
//! Short reused appends preserve header arithmetic and cache the actual prompt
//! endpoint plus logits. DFlash remains cold and request-local.

use super::backend::{IM_START_MARKER, transcript_boundary};
use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{ServeError, ServeRequest};
use super::snapshot_cache::SnapshotCache;
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::metal_dflash::MetalDFlashHead;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpLoadedModel, Qwen4ExpRuntimeError, Qwen4ExpSessionCapacity, Qwen4ExpTextRunner,
};
use qwen_llm::qwen4exp_text_session::{Qwen4ExpTextSessionPlan, Qwen4ExpTextSnapshot};
use qwen_llm::sampling::Sampler;
use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

const FAMILY: &str = "Qwen3.8-Flash-Next";

mod concurrent;
mod dflash;
mod reuse;
use reuse::*;

pub(super) struct Prepared {
    context_limit: usize,
    checkpoint_ladder: super::checkpoint_ladder::Config,
    default_max_tokens: usize,
    release_sampling: crate::release_sampling::SamplingDefaults,
    tokenizer: Tokenizer,
    stop_tokens: Vec<i32>,
    im_start: Option<u32>,
    capacity: Qwen4ExpSessionCapacity,
    decode_options: qwen_llm::qwen4exp_runtime::Qwen4ExpDecodeOptions,
    drafter: Option<crate::drafter_policy::PreparedDrafter>,
    batching: concurrent::Config,
}

impl Prepared {
    /// Checks the prompt contract, capacity, tokenizer stops, and decode
    /// options from GGUF metadata without binding or initializing Metal.
    pub(super) fn new(
        gguf: &GgufFile,
        invocation: &crate::cli::ServeInvocation,
    ) -> anyhow::Result<Self> {
        if let Some(failure) = crate::qwen4exp_prompt_capability_failure(
            qwen_llm::model_family::ModelFamily::Qwen4Exp,
            gguf,
        ) {
            anyhow::bail!(
                "Qwen3.8-Flash-Next serve does not support the declared {}",
                failure.as_str()
            );
        }
        let config =
            Qwen4ExpConfig::from_gguf(gguf).context("bind Qwen3.8-Flash-Next serve geometry")?;
        let limits = super::resolve_serve_limits(
            super::profile(qwen_llm::model_family::ModelFamily::Qwen4Exp),
            Some(config.context_length as usize),
            invocation.max_context_tokens,
            invocation.max_tokens,
        )?
        .context("Qwen3.8-Flash-Next profile has no fixed serve capacity")?;
        let (context_limit, default_max_tokens) = (limits.context_tokens, limits.max_tokens);
        let tokenizer = Tokenizer::from_gguf(gguf).context("load Qwen3.8-Flash-Next tokenizer")?;
        config
            .validate_artifact()
            .context("validate Qwen4 artifact contract")?;
        let vocab_size = tokenizer.n_vocab();
        anyhow::ensure!(
            vocab_size == config.vocab_size,
            "Qwen3.8-Flash-Next tokenizer vocabulary {vocab_size} differs from model vocabulary {}",
            config.vocab_size
        );
        let stop_tokens = gguf
            .stop_token_ids()
            .context("load producer-declared Qwen3.8-Flash-Next stop tokens")?;
        crate::qwen4exp::validate_qwen4exp_stop_tokens(&stop_tokens, vocab_size)?;
        let im_start = tokenizer
            .encode(IM_START_MARKER, false)
            .ok()
            .and_then(|ids| (ids.len() == 1).then(|| u32::try_from(ids[0]).ok())?);
        let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, context_limit)
            .context("derive Qwen3.8-Flash-Next resident session capacity")?;
        let decode_options = crate::family_options::qwen4exp_decode_options_from_env()?;
        let drafter = crate::drafter_policy::PreparedDrafter::prepare_for_target(
            invocation.drafter.as_deref(),
            qwen_llm::loader::DFlashTargetShape {
                hidden_size: config.hidden_size,
                vocab_size: config.vocab_size,
                n_layer: config.layer_count,
            },
            Some(qwen_llm::model_family::ModelFamily::Qwen4Exp),
            crate::drafter_policy::Lane::Serve,
        )?;
        Ok(Self {
            checkpoint_ladder: super::checkpoint_ladder::Config::from_env()?,
            release_sampling: qwen4_sampling_defaults(gguf),
            context_limit,
            default_max_tokens,
            tokenizer,
            stop_tokens,
            im_start,
            capacity,
            decode_options,
            drafter,
            batching: concurrent::Config::from_env()?,
        })
    }
}

fn qwen4_sampling_defaults(gguf: &GgufFile) -> crate::release_sampling::SamplingDefaults {
    crate::release_sampling::gguf_sampling_defaults(
        crate::release_identity::ReleaseIdentity::detect(
            qwen_llm::model_family::ModelFamily::Qwen4Exp,
            gguf,
        ),
        gguf,
    )
}

/// Loads the prepared Flash-Next session after bind, then serves.
pub(super) fn start(
    prepared: Prepared,
    gguf: GgufFile,
    invocation: &crate::cli::ServeInvocation,
    listening: super::Listening,
    idle_window: std::time::Duration,
) -> anyhow::Result<()> {
    crate::shutdown::checkpoint()?;
    let ctx = MetalContext::new().context("initialize Metal context")?;
    let load_started = Instant::now();
    let context_limit = prepared.context_limit;
    let default_max_tokens = prepared.default_max_tokens;
    let mut backend = FlashNextBackend::new(
        ctx,
        Box::leak(Box::new(gguf)),
        listening.model_id.clone(),
        prepared,
        invocation.snapshot_cache_mib,
        invocation.snapshot_policy,
    )?;
    backend.template_style = invocation.template_style;
    backend.set_idle_residency(idle_window);
    let load_ms = load_started.elapsed().as_secs_f64() * 1e3;
    tracing::info!(target: "qwen_diag", "serve limits: family=qwen4exp max_context_tokens={context_limit} default_max_tokens={default_max_tokens} {}", backend.snapshot_cache_plan);
    crate::shutdown::checkpoint()?;
    let warm_up_ms = backend.warm_up()?;
    tracing::info!(target: "qwen_diag", "serve startup: family=qwen4exp load_ms={load_ms:.1} warm_up_ms={warm_up_ms:.1}");
    crate::shutdown::checkpoint()?;
    if backend.batching.active > 1
        && backend.dflash_head.is_none()
        && backend.ctx.device_facts().gpu_families.apple10
    {
        let mut owner = concurrent::Owner::new(&mut backend)?;
        listening.serve(load_ms, &mut owner)
    } else {
        listening.serve(load_ms, &mut backend)
    }
}

pub(crate) struct FlashNextBackend {
    ctx: MetalContext,
    tokenizer: Tokenizer,
    loaded: Qwen4ExpLoadedModel<'static>,
    dflash_head: Option<MetalDFlashHead>,
    model_id: String,
    default_max_tokens: usize,
    release_sampling: crate::release_sampling::SamplingDefaults,
    /// Longest decoded piece of one token (output ceilings).
    max_piece_bytes: usize,
    forward_limit: usize,
    vocab_size: u32,
    stop_tokens: Vec<i32>,
    im_start: Option<u32>,
    /// Owned by this backend and its one loaded model; restore additionally
    /// refuses a snapshot whose session geometry differs.
    cache: SnapshotCache<Checkpoint>,
    turns: TurnConfig,
    live: Option<Live>,
    batching: concurrent::Config,
    pub(super) snapshot_cache_plan: super::SnapshotCachePlan,
    /// Deployment default for `x_qwen.template_style` (`--template-style`).
    pub(super) template_style: super::items::TemplateStyle,
    control_cpu_reserve: u64,
    /// Off until [`FlashNextBackend::set_idle_residency`].
    idle_residency: super::idle_residency::IdleResidency,
}

#[cfg(test)]
mod tests;

impl FlashNextBackend {
    /// Keep the no-copy Metal weights wired for `window` after each request
    /// (the CPU-read PLE table stays in the page cache, outside the pulse).
    pub(super) fn set_idle_residency(&mut self, window: std::time::Duration) {
        self.idle_residency = super::idle_residency::IdleResidency::new("qwen4exp", window);
    }

    /// One short packed prefill on a throwaway runner before the listener
    /// accepts, so the first request is not charged the weights' first GPU
    /// use. Every layer's dense projections and expert banks are bound by any
    /// prompt; the row count only needs to take the packed path.
    pub(super) fn warm_up(&mut self) -> anyhow::Result<f64> {
        let started = Instant::now();
        let tokens = self
            .tokenizer
            .encode(
                "Warm-up prompt: the server binds every weight it will need before the first request arrives.",
                false,
            )
            .context("warm-up tokens")?
            .into_iter()
            .map(|id| u32::try_from(id).context("warm-up token id"))
            .collect::<anyhow::Result<Vec<_>>>()?;
        let mut runner = self
            .loaded
            .create_runner(&self.ctx)
            .context("bind Qwen3.8-Flash-Next warm-up runner")?;
        let outcome = runner
            .prefill(&tokens)
            .map(|_| ())
            .context("Qwen3.8-Flash-Next serve warm-up");
        // The workspace goes back to the loaded model whatever the prefill
        // did; a workspace that cannot be reset is a startup failure.
        let workspace = runner.into_workspace();
        self.loaded
            .restore_workspace(workspace)
            .context("return Qwen3.8-Flash-Next warm-up workspace")?;
        outcome?;
        // The server is about to accept: open the idle-residency window.
        self.idle_residency.note_activity();
        Ok(started.elapsed().as_secs_f64() * 1e3)
    }

    pub(super) fn new(
        ctx: MetalContext,
        gguf: &'static GgufFile,
        model_id: String,
        prepared: Prepared,
        snapshot_cache_mib: Option<u64>,
        snapshot_policy: SnapshotPolicyConfig,
    ) -> anyhow::Result<Self> {
        let vocab_size = prepared.tokenizer.n_vocab();
        // Head residency must be visible when adaptive target scratch is chosen.
        let dflash_head = prepared
            .drafter
            .as_ref()
            .map(|prepared| dflash::load_head(&ctx, prepared))
            .transpose()?;
        drop(prepared.drafter);
        // The context determines selected-QSA needs, not packed row count.
        // Narrow reusable scratch under pressure before scalarizing prefill.
        let loaded = if let Some(head) = &dflash_head {
            let weights = Qwen4ExpMetalWeightPlan::for_native(&ctx, gguf)?;
            let plan = if prepared.context_limit > 1 {
                Qwen4ExpTextSessionPlan::for_config_with_packed_prefill_tokens(
                    &ctx,
                    weights.config(),
                    prepared.capacity.qsa_physical_capacity(),
                    weights.memory_plan(),
                    prepared.context_limit,
                )?
            } else {
                Qwen4ExpTextSessionPlan::for_config(
                    &ctx,
                    weights.config(),
                    prepared.capacity.qsa_physical_capacity(),
                    weights.memory_plan(),
                )?
            };
            let width = (head.config.block_size as usize)
                .min(prepared.capacity.forward_limit().saturating_sub(1));
            Qwen4ExpLoadedModel::load_with_adaptive_prefill_plan(
                &ctx,
                gguf,
                prepared.capacity,
                plan,
                prepared.decode_options,
                if width > 1 { width } else { 0 },
            )
        } else {
            Qwen4ExpLoadedModel::load_with_adaptive_prefill(
                &ctx,
                gguf,
                prepared.capacity,
                prepared.context_limit,
                prepared.decode_options,
            )
        }
        .context("load Qwen3.8-Flash-Next serve session")?;
        if let Some(head) = &dflash_head {
            let width = (head.config.block_size as usize)
                .min(prepared.capacity.forward_limit().saturating_sub(1));
            anyhow::ensure!(
                width <= 1 || loaded.packed_prefill_capacity().is_some_and(|n| n >= width),
                "Flash-Next DFlash requires packed verification width {width}; resident workspace has {:?}",
                loaded.packed_prefill_capacity()
            );
        }
        tracing::info!(
            target: "qwen_diag",
            "serve: qwen4exp resident forward_limit={} qsa_physical_capacity={} packed_prefill_capacity={:?} packed_fallback={:?} guarded_topk={} hc_up_mix={}",
            prepared.capacity.forward_limit(),
            prepared.capacity.qsa_physical_capacity(),
            loaded.packed_prefill_capacity(),
            loaded.packed_prefill_capacity().is_none(),
            loaded.guarded_topk_enabled(),
            loaded.hc_up_mix_enabled(),
        );
        // Sized after load so auto budgets see the resident model.
        let snapshot_cache_plan = super::SnapshotCachePlan::resolve(
            snapshot_cache_mib,
            snapshot_policy,
            ctx.memory_signals(),
        )?;
        Ok(Self {
            release_sampling: prepared.release_sampling,
            ctx,
            max_piece_bytes: prepared.tokenizer.max_decoded_piece_bytes(),
            tokenizer: prepared.tokenizer,
            loaded,
            dflash_head,
            model_id,
            default_max_tokens: prepared.default_max_tokens,
            forward_limit: prepared.capacity.forward_limit(),
            vocab_size,
            stop_tokens: prepared.stop_tokens,
            im_start: prepared.im_start,
            turns: TurnConfig::from_env(),
            live: None,
            batching: prepared.batching,
            cache: SnapshotCache::new(snapshot_cache_plan.bytes, snapshot_cache_plan.policy)
                .with_checkpoint_ladder(prepared.checkpoint_ladder),
            snapshot_cache_plan,
            template_style: super::items::TemplateStyle::House,
            control_cpu_reserve: 0,
            idle_residency: super::idle_residency::IdleResidency::new(
                "qwen4exp",
                std::time::Duration::ZERO,
            ),
        })
    }
}

impl FlashNextBackend {
    /// The limits ceilings and generation both resolve against.
    fn output_limits(&self) -> crate::serve::request_profile::OutputLimits {
        crate::serve::request_profile::OutputLimits {
            default_max_tokens: self.default_max_tokens,
            max_piece_bytes: self.max_piece_bytes,
        }
    }
}

impl GenerationBackend for FlashNextBackend {
    fn set_control_memory_reserve(&mut self, bytes: u64) {
        self.control_cpu_reserve = bytes;
    }
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn idle(&mut self) {
        super::log_expired_snapshots("qwen4exp", &self.cache.sweep());
        let loaded = &self.loaded;
        self.idle_residency
            .on_idle(&self.ctx, || loaded.retained_buffers());
    }

    fn request_finished(&mut self) {
        self.idle_residency.request_finished();
    }
    fn request_failed_on_server(&mut self) {
        self.idle_residency.request_failed_on_server();
    }

    fn shutdown(&mut self) {
        self.idle_residency.shutdown();
    }

    fn request_profile(&self) -> super::request_profile::RequestProfile {
        super::request_profile::RequestProfile::FlashNext {
            sampling: self.release_sampling,
            style: self.template_style,
            limits: self.output_limits(),
        }
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let conversation = super::checkpoint_ladder::conversation(request);
        self.idle_residency.before_request();
        // One resolution, shared with the output ceiling.
        let max_tokens = self.output_limits().max_tokens(request);
        let mut sampler = Sampler::new(super::request_profile::sampling_with_defaults(
            request,
            self.release_sampling,
        ))
        .map_err(|error| ServeError::invalid_request(None, format!("sampling: {error}")))?;
        let tokenize_t0 = Instant::now();
        let prompt_ids = decode_loop::encode_checked(
            &self.tokenizer,
            prompt,
            false,
            self.vocab_size,
            crate::family_profile::profile(qwen_llm::model_family::ModelFamily::Qwen4Exp).display,
        )?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        let required = decode_loop::required_forwards(
            crate::family_profile::profile(qwen_llm::model_family::ModelFamily::Qwen4Exp).display,
            prompt_ids.len(),
            max_tokens,
            self.forward_limit,
        )?;
        if self.dflash_head.is_none() {
            super::transport_memory::admit_resident_transport(
                sink.transport_reserve_bytes(),
                qwen_llm::metal::effective_host_memory_remaining_bytes(),
            )?;
        }
        let mut runner = self
            .loaded
            .create_runner(&self.ctx)
            .map_err(|error| ServeError::server_error(format!("bind {FAMILY} runner: {error}")))?;

        let previous_live = self.live.take();
        let mut next_live = None;
        let mut run = || -> Result<GenerationOutcome, BackendFailure> {
            if let Some(head) = &self.dflash_head {
                sink.tick().map_err(BackendFailure::Aborted)?;
                let reserve = sink.transport_reserve_bytes();
                let admit = || {
                    qwen_llm::qwen4exp_runtime::Qwen4ExpDFlashAdapter::admit_request(
                        &runner,
                        head,
                        prompt_ids.len(),
                        max_tokens,
                        reserve,
                    )
                };
                super::memory_relief::retry(
                    FAMILY,
                    admit(),
                    dflash::relief_request,
                    |deficit| self.cache.evict_for(deficit),
                    admit,
                )
                .map_err(dflash::map_error)?;
                let (generation, stats) = dflash::generate_cold(
                    &mut runner,
                    head,
                    &prompt_ids,
                    sampler.clone(),
                    max_tokens,
                    &self.stop_tokens,
                    &self.tokenizer,
                    sink,
                    reserve,
                )?;
                tracing::info!(target: "qwen_diag",
                    "serve phases: family=qwen4exp mode=dflash_cold tokenize_ms={tokenize_ms:.1} matched_tokens=0 restore_ms=0 prefill_tokens={} prefill_ms={:.1} decode_ms={:.1} transitions={} required_forwards={required} forward_limit={}",
                    prompt_ids.len(), stats.prefill.as_secs_f64() * 1e3,
                    generation.wall_ms, generation.transitions, self.forward_limit);
                dflash::log_stats(&stats);
                return Ok(super::outcome::finish_generation(
                    prompt_ids.len(),
                    &generation,
                    0,
                    0.0,
                ));
            }
            let restore_t0 = Instant::now();
            let ready = !runner.is_poisoned() && runner.logits().is_ok();
            let live_match = previous_live
                .as_ref()
                .filter(|_| self.turns.live)
                .and_then(|live| {
                    live_prefix_len(&live.tokens, &prompt_ids, runner.next_position(), ready)
                });
            let mut restored_logits = None;
            let mut departing_capture_ms = 0.0;
            let (matched_tokens, reuse_source) = if let Some(matched) = live_match {
                (matched, "live_session")
            } else {
                let hit = checkpoint_candidate(
                    &mut self.cache,
                    &prompt_ids,
                    self.turns.live || self.turns.endpoint_checkpoint,
                    |checkpoint| {
                        checkpoint.logits.len() == self.vocab_size as usize
                            && checkpoint.logits.iter().all(|x| x.is_finite())
                    },
                );
                // Publish a departing live branch only when it is actually
                // needed. Protect the selected restore during admission.
                let pinned = hit
                    .as_ref()
                    .map(|hit| hit.id)
                    .filter(|&id| self.cache.pin(id));
                if self.turns.live
                    && ready
                    && let Some(live) = previous_live.as_ref()
                    && live.tokens.len() == runner.next_position()
                {
                    let started = Instant::now();
                    let head = runner
                        .logits()
                        .map_err(|e| ServeError::server_error(e.to_string()))?
                        .to_vec();
                    capture_with_logits(
                        &mut self.cache,
                        &self.ctx,
                        &runner,
                        &live.tokens,
                        "departing_live",
                        self.control_cpu_reserve,
                        &head,
                    );
                    self.cache.record_tip(live.conversation, &(), &live.tokens);
                    departing_capture_ms = started.elapsed().as_secs_f64() * 1e3;
                }
                if let Some(id) = pinned {
                    self.cache.unpin(id);
                }
                let restored = restore_prefix_hit(
                    &mut self.cache,
                    hit,
                    &mut runner,
                    |runner, checkpoint: &Checkpoint| {
                        if checkpoint.logits.is_empty() {
                            runner.restore_snapshot(&checkpoint.snapshot)?;
                        } else {
                            runner.restore_snapshot_with_logits(
                                &checkpoint.snapshot,
                                &checkpoint.logits,
                            )?;
                        }
                        restored_logits =
                            (!checkpoint.logits.is_empty()).then(|| checkpoint.logits.clone());
                        Ok(runner.next_position())
                    },
                    |runner| runner.reset(),
                );
                let matched = restored.map_err(|error| {
                    ServeError::server_error(format!("reset {FAMILY} session: {error}"))
                })?;
                if matched == 0 && (runner.next_position() != 0 || runner.is_poisoned()) {
                    runner.reset().map_err(|e| {
                        ServeError::server_error(format!("reset {FAMILY} session: {e}"))
                    })?;
                }
                (matched, if matched == 0 { "cold" } else { "snapshot_ram" })
            };
            let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;

            // Transcript-boundary capture: stop prefill before the generation
            // header, snapshot, then finish the header.
            let transcript_at = self
                .im_start
                .and_then(|im_start| transcript_boundary(&prompt_ids, im_start));
            let endpoint_checkpoint = endpoint_checkpoint(
                self.turns.endpoint_checkpoint,
                matched_tokens,
                prompt_ids.len(),
                transcript_at,
                runner.packed_prefill_capacity(),
            );
            let transcript_split = transcript_split(transcript_at, matched_tokens);
            // A restore ending exactly on the transcript boundary (a resent or
            // regenerated turn) already holds the transcript snapshot; it is
            // protected like a fresh capture below.
            let mut transcript_entry = transcript_at
                .filter(|&boundary| boundary == matched_tokens)
                .and_then(|boundary| self.cache.entry_for(&prompt_ids[..boundary]));
            let transcript_restored = transcript_entry.is_some();
            let prefill_t0 = Instant::now();
            let mut capture_ms = 0.0;
            let mut prefill_calls = 0;
            if let Some(boundary) = transcript_split {
                prefill_calls += 1;
                capture_ms += prefill_ladder_segment(
                    &mut runner,
                    &mut self.cache,
                    &self.ctx,
                    &prompt_ids,
                    matched_tokens..boundary,
                    sink,
                    conversation,
                    self.control_cpu_reserve,
                )?;
                if !endpoint_checkpoint {
                    let capture_t0 = Instant::now();
                    if capture_snapshot(
                        &mut self.cache,
                        &self.ctx,
                        &runner,
                        &prompt_ids[..boundary],
                        "transcript",
                        self.control_cpu_reserve,
                    ) {
                        transcript_entry = self.cache.entry_for(&prompt_ids[..boundary]);
                    }
                    capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
                }
            }
            let start = transcript_split.unwrap_or(matched_tokens);
            if start < prompt_ids.len() {
                prefill_calls += 1;
                capture_ms += prefill_ladder_segment(
                    &mut runner,
                    &mut self.cache,
                    &self.ctx,
                    &prompt_ids,
                    start..prompt_ids.len(),
                    sink,
                    conversation,
                    self.control_cpu_reserve,
                )?;
            }
            let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3 - capture_ms;
            let logits = if matched_tokens == prompt_ids.len() && restored_logits.is_some() {
                restored_logits.take().expect("exact checkpoint logits")
            } else {
                runner
                    .logits()
                    .map_err(|error| ServeError::server_error(format!("{FAMILY} logits: {error}")))?
                    .to_vec()
            };
            // The transcript boundary a few header tokens earlier already
            // holds this prompt's reusable prefix.
            if transcript_entry.is_none() {
                let capture_t0 = Instant::now();
                capture_with_logits(
                    &mut self.cache,
                    &self.ctx,
                    &runner,
                    &prompt_ids,
                    "prompt",
                    self.control_cpu_reserve,
                    if endpoint_checkpoint { &logits } else { &[] },
                );
                capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
            }

            let tip_end = if transcript_entry.is_some() {
                transcript_at.expect("transcript entry boundary")
            } else {
                prompt_ids.len()
            };
            let mut retention_ms = 0.0;
            let mut completed_capture_ms = 0.0;
            let mut transport_abort = None;
            let gpu = crate::qwen4exp::try_generate_qwen4exp_gpu(
                &mut runner,
                max_tokens,
                &self.stop_tokens,
                &mut sampler,
                |token| {
                    let bytes = self
                        .tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode {FAMILY} token {token}"))?;
                    sink.piece(&bytes)
                        .and_then(|()| sink.tick())
                        .map_err(|error| {
                            transport_abort = Some(error);
                            anyhow::anyhow!("client disconnected during GPU decode")
                        })
                },
            )
            .map_err(|error| match transport_abort {
                Some(error) => BackendFailure::Aborted(error),
                None => ServeError::server_error(format!("{FAMILY} GPU decode: {error:#}")).into(),
            })?;
            let generation = if let Some((generation, _)) = gpu {
                generation
            } else {
                decode_loop::decode_serial(
                    decode_loop::DecodeRequest {
                        family: FAMILY,
                        logits,
                        max_tokens,
                        stop_tokens: &self.stop_tokens,
                        vocab_size: self.vocab_size,
                    },
                    &mut sampler,
                    &self.tokenizer,
                    sink,
                    |token| Ok(runner.forward_token(token)?.to_vec()),
                )?
            };

            // Keep exactly the consumed prefix; the final sampled token has
            // not been forwarded. Completed state stays live until a branch
            // departure needs a RAM copy. Rollback retains eager capture.
            match crate::derive_completed_checkpoint_boundary(
                prompt_ids.len(),
                &generation.tokens,
                generation.transitions,
                runner.next_position(),
            )
            .and_then(|_| {
                let mut consumed = prompt_ids.clone();
                for &token in &generation.tokens[..generation.transitions] {
                    consumed.push(crate::checked_token_id(token, self.vocab_size, "consumed")?);
                }
                Ok(consumed)
            }) {
                Ok(consumed) => {
                    let pinned = self
                        .cache
                        .entry_for(&prompt_ids[..tip_end])
                        .filter(|&entry| self.cache.pin(entry));
                    let retain =
                        self.turns.live && !runner.is_poisoned() && runner.logits().is_ok();
                    if !retain {
                        let started = Instant::now();
                        capture_snapshot(
                            &mut self.cache,
                            &self.ctx,
                            &runner,
                            &consumed,
                            "completed",
                            self.control_cpu_reserve,
                        );
                        completed_capture_ms = started.elapsed().as_secs_f64() * 1e3;
                    }
                    let retention_start = Instant::now();
                    self.cache
                        .record_tip(conversation, &(), &prompt_ids[..tip_end]);
                    self.cache.record_tip(conversation, &(), &consumed);
                    retention_ms += retention_start.elapsed().as_secs_f64() * 1e3;
                    if let Some(entry) = pinned {
                        self.cache.unpin(entry);
                    }
                    if retain {
                        next_live = Some(Live {
                            tokens: consumed,
                            conversation,
                        });
                    }
                }
                Err(error) => {
                    tracing::warn!("serve: qwen4exp completed boundary derivation failed: {error}")
                }
            }

            let transcript_phase = match (transcript_split, transcript_at) {
                (Some(boundary), _) => format!(
                    " transcript_boundary={boundary} transcript_captured={}",
                    transcript_entry.is_some()
                ),
                (None, Some(boundary)) if transcript_restored => {
                    format!(" transcript_boundary={boundary} transcript_restored=true")
                }
                _ => String::new(),
            };
            tracing::info!(
                target: "qwen_diag",
                "serve phases: family=qwen4exp tokenize_ms={tokenize_ms:.1} restore_ms={restore_ms:.1} reuse_source={reuse_source} departing_capture_ms={departing_capture_ms:.1} endpoint_checkpoint={endpoint_checkpoint} prefill_calls={prefill_calls} completed_capture_ms={completed_capture_ms:.1} matched_tokens={matched_tokens} prefill_ms={prefill_ms:.1} prefill_tokens={} snapshot_capture_ms={capture_ms:.1}{transcript_phase} decode_ms={:.1} required_forwards={required} forward_limit={} transitions={} snapshot_cache_bytes={} ladder={:?} retention_ms={retention_ms:.3}",
                prompt_ids.len() - matched_tokens,
                generation.wall_ms,
                self.forward_limit,
                generation.transitions,
                self.cache.indexed_bytes(),
                self.cache.ladder_status(),
            );
            Ok(super::outcome::finish_generation(
                prompt_ids.len(),
                &generation,
                matched_tokens,
                restore_ms,
            ))
        };
        let outcome = run();
        let retain = outcome.is_ok() && next_live.is_some();
        let position = runner.next_position();
        let workspace = runner.into_workspace();
        let returned_at = Instant::now();
        let outcome = with_request_workspace(
            outcome,
            workspace,
            |workspace| {
                if retain {
                    self.loaded.retain_workspace(workspace)
                } else {
                    self.loaded.restore_workspace(workspace)
                }
            },
            |error| {
                tracing::error!(
                    target: "qwen_diag",
                    "serve: qwen4exp workspace could not be returned ({error}); aborting"
                );
                std::process::abort();
            },
        );
        self.live = if retain { next_live } else { None };
        tracing::info!(target: "qwen_diag", "serve workspace: family=qwen4exp retained={retain} position={position} return_ms={:.3}", returned_at.elapsed().as_secs_f64() * 1e3);
        outcome
    }
}

/// Prefill `tokens` after the runner's committed length, heartbeating the
/// transport between commands.
fn prefill_segment(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    sink: &mut dyn GenerationSink,
) -> Result<(), BackendFailure> {
    let mut checkpoint_abort: Option<io::Error> = None;
    let result = runner
        .prefill_continuation_with_command_checkpoint(tokens, || {
            sink.tick().map_err(|error| {
                checkpoint_abort = Some(error);
                Qwen4ExpRuntimeError::Checkpoint(format!(
                    "transport aborted during {FAMILY} prefill"
                ))
            })
        })
        .map(|_| ());
    map_prefill_result(result, checkpoint_abort)
}

/// Observe one existing full packed command near the end of a long segment.
/// Short appends take the previous API directly and make no extra state copy.
fn prefill_ladder_segment(
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    cache: &mut SnapshotCache<Checkpoint>,
    ctx: &MetalContext,
    tokens: &[u32],
    range: std::ops::Range<usize>,
    sink: &mut dyn GenerationSink,
    conversation: Option<super::checkpoint_ladder::Conversation>,
    reserve: u64,
) -> Result<f64, BackendFailure> {
    let rows = runner.packed_prefill_capacity().unwrap_or(0);
    if !cache.ladder_enabled() || conversation.is_none() || range.len() <= rows || rows == 0 {
        prefill_segment(runner, &tokens[range], sink)?;
        return Ok(0.0);
    }
    let mut elapsed = 0.0;
    let mut checkpoint_abort = None;
    let endpoint = range.end;
    let result=runner.prefill_continuation_with_observer(&tokens[range], || sink.tick().map_err(|error| {
        checkpoint_abort=Some(error);
        Qwen4ExpRuntimeError::Checkpoint(format!("transport aborted during {FAMILY} prefill"))
    }), &mut |runner, chunk| {
        if chunk.len()==rows && chunk.end<endpoint && endpoint-chunk.end<=rows {
            let started=Instant::now();
            let retained=capture_snapshot_impl(cache,ctx,runner,&tokens[..chunk.end],"ladder",reserve,conversation,&[]);
            elapsed+=started.elapsed().as_secs_f64()*1e3;
            tracing::info!(target:"qwen_diag", "serve checkpoint: family=qwen4exp position={} retained={retained} ladder={:?}",chunk.end,cache.ladder_status());
        }
    }).map(|_|());
    map_prefill_result(result, checkpoint_abort)?;
    Ok(elapsed)
}

fn map_prefill_result<T>(
    result: Result<T, Qwen4ExpRuntimeError>,
    checkpoint_abort: Option<io::Error>,
) -> Result<T, BackendFailure> {
    result.map_err(|error| match checkpoint_abort {
        Some(error) => BackendFailure::Aborted(error),
        None => ServeError::server_error(format!("prefill {FAMILY} prompt: {error}")).into(),
    })
}
