//! Resident Qwen3.8-Flash-Next [`GenerationBackend`].
//!
//! The text session's workspace is allocated once at load for a fixed
//! forward limit; each request takes it as a runner and hands it back reset
//! (`Qwen4ExpLoadedModel::restore_workspace`), so no per-request allocation.
//! History carries across requests only through a serve-owned RAM snapshot
//! cache keyed by exact token prefix: a request restores the longest cached
//! strictly-shorter prefix, then prefills the rest. As on the Qwen backend,
//! prefill stops before the generation header to snapshot the transcript
//! boundary — the only point a thinking turn's next request still extends,
//! since the template re-renders prior reasoning and GDN state cannot be
//! rewound.

use super::backend::{IM_START_MARKER, transcript_boundary};
use super::decode_loop;
use super::http::{BackendFailure, GenerationBackend, GenerationOutcome, GenerationSink};
use super::items::{QwenTemplate, ServeError, ServeRequest};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::snapshot_cache::SnapshotCache;
use anyhow::Context as _;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::MetalContext;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpLoadedModel, Qwen4ExpRuntimeError, Qwen4ExpSessionCapacity, Qwen4ExpTextRunner,
};
use qwen_llm::qwen4exp_text_session::Qwen4ExpTextSnapshot;
use qwen_llm::snapshot_policy::SnapshotPolicyConfig;
use qwen_llm::tokenizer::Tokenizer;
use std::io;
use std::time::Instant;

const FAMILY: &str = "Qwen3.8-Flash-Next";

pub(crate) struct FlashNextBackend {
    ctx: MetalContext,
    /// Serve holds the mapping for the process lifetime; the loaded model's
    /// CPU-resident PLE table borrows it.
    gguf: &'static GgufFile,
    tokenizer: Tokenizer,
    loaded: Qwen4ExpLoadedModel<'static>,
    model_id: String,
    default_max_tokens: usize,
    forward_limit: usize,
    vocab_size: u32,
    stop_tokens: Vec<i32>,
    im_start: Option<u32>,
    /// Owned by this backend and its one loaded model; restore additionally
    /// refuses a snapshot whose session geometry differs.
    cache: SnapshotCache<Qwen4ExpTextSnapshot>,
    pub(super) snapshot_cache_plan: super::SnapshotCachePlan,
    /// Deployment default for `x_qwen.template_style` (`--template-style`).
    pub(super) template_style: super::items::TemplateStyle,
}

impl FlashNextBackend {
    pub(crate) fn new(
        ctx: MetalContext,
        gguf: &'static GgufFile,
        model_id: String,
        default_max_tokens: usize,
        context_limit: usize,
        snapshot_cache_mib: Option<u64>,
        snapshot_policy: SnapshotPolicyConfig,
    ) -> anyhow::Result<Self> {
        let tokenizer = Tokenizer::from_gguf(gguf).context("load Qwen3.8-Flash-Next tokenizer")?;
        let config =
            Qwen4ExpConfig::from_gguf(gguf).context("bind Qwen3.8-Flash-Next request geometry")?;
        anyhow::ensure!(
            config == Qwen4ExpConfig::flash_next_reference(),
            "Qwen3.8-Flash-Next runtime requires the released architecture contract"
        );
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
        // Packed prefill sized to the whole context; when that allocation
        // is refused, prompts prefill token by token as on the run lane.
        let (loaded, packed_fallback) = match Qwen4ExpLoadedModel::load_with_decode_options(
            &ctx,
            gguf,
            capacity,
            Some(context_limit),
            decode_options,
        ) {
            Ok(loaded) => (loaded, None),
            Err(packed_error) => (
                Qwen4ExpLoadedModel::load_with_decode_options(
                    &ctx,
                    gguf,
                    capacity,
                    None,
                    decode_options,
                )
                .with_context(|| {
                    format!(
                        "load Qwen3.8-Flash-Next serve session (packed prefill refused: {packed_error})"
                    )
                })?,
                Some(packed_error.to_string()),
            ),
        };
        tracing::info!(
            target: "qwen_diag",
            "serve: qwen4exp resident forward_limit={} qsa_physical_capacity={} packed_prefill_capacity={:?} packed_fallback={:?} guarded_topk={} hc_up_mix={}",
            capacity.forward_limit(),
            capacity.qsa_physical_capacity(),
            loaded.packed_prefill_capacity(),
            packed_fallback,
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
            ctx,
            gguf,
            tokenizer,
            loaded,
            model_id,
            default_max_tokens,
            forward_limit: capacity.forward_limit(),
            vocab_size,
            stop_tokens,
            im_start,
            cache: SnapshotCache::new(snapshot_cache_plan.bytes, snapshot_cache_plan.policy),
            snapshot_cache_plan,
            template_style: super::items::TemplateStyle::House,
        })
    }
}

impl GenerationBackend for FlashNextBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn idle(&mut self) {
        super::log_expired_snapshots("qwen4exp", &self.cache.sweep());
    }

    /// Bind once through the family table so the bound request is what
    /// renders, selects the output protocol, and echoes.
    fn normalize_request(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        *request = crate::open_responses::bind_qwen_request(request, QwenTemplate::Qwen38, true)?;
        Ok(())
    }

    fn template_style_default(&self) -> Option<super::items::TemplateStyle> {
        Some(self.template_style)
    }

    fn output_protocol(&self, request: &ServeRequest) -> OutputProtocol {
        OutputProtocol::Qwen {
            preopened_reasoning: super::render::qwen_generation(request)
                == super::render::QwenGeneration::PreOpen,
            parse_tools: true,
            tool_grammar: ToolGrammar::QwenXml,
        }
    }

    fn render_prompt(&self, request: &ServeRequest) -> Result<String, ServeError> {
        Ok(super::render::render_qwen_serve_prompt(request))
    }

    fn generate(
        &mut self,
        request: &ServeRequest,
        prompt: &str,
        sink: &mut dyn GenerationSink,
    ) -> Result<GenerationOutcome, BackendFailure> {
        let max_tokens = request.max_output_tokens.unwrap_or(self.default_max_tokens);
        let mut sampler = super::backend::request_sampler(request)?;
        let tokenize_t0 = Instant::now();
        let prompt_ids =
            decode_loop::encode_checked(&self.tokenizer, prompt, false, self.vocab_size, FAMILY)?;
        let tokenize_ms = tokenize_t0.elapsed().as_secs_f64() * 1e3;
        let required = decode_loop::required_forwards(
            FAMILY,
            prompt_ids.len(),
            max_tokens,
            self.forward_limit,
        )?;
        let mut runner = self
            .loaded
            .create_runner(&self.ctx)
            .map_err(|error| ServeError::server_error(format!("bind {FAMILY} runner: {error}")))?;

        let mut run = || -> Result<GenerationOutcome, BackendFailure> {
            let restore_t0 = Instant::now();
            let matched_tokens = match self.cache.best_prefix(&prompt_ids) {
                Some((prefix_len, snapshot)) => match runner.restore_snapshot(&snapshot) {
                    Ok(()) if runner.next_position() == prefix_len => prefix_len,
                    outcome => {
                        tracing::warn!(
                            "serve: qwen4exp restore of {prefix_len} tokens failed, cold prefilling: {:?}",
                            outcome.err()
                        );
                        runner.reset().map_err(|error| {
                            ServeError::server_error(format!("reset {FAMILY} session: {error}"))
                        })?;
                        0
                    }
                },
                None => 0,
            };
            let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;

            // Transcript-boundary capture: stop prefill before the generation
            // header, snapshot, then finish the header.
            let transcript_at = self
                .im_start
                .and_then(|im_start| transcript_boundary(&prompt_ids, im_start));
            let transcript_split = transcript_at.filter(|&boundary| boundary > matched_tokens);
            // A restore ending exactly on the transcript boundary (a resent or
            // regenerated turn) already holds the transcript snapshot; it is
            // protected like a fresh capture below.
            let mut transcript_entry = transcript_at
                .filter(|&boundary| boundary == matched_tokens)
                .and_then(|boundary| self.cache.entry_for(&prompt_ids[..boundary]));
            let transcript_restored = transcript_entry.is_some();
            let prefill_t0 = Instant::now();
            let mut capture_ms = 0.0;
            if let Some(boundary) = transcript_split {
                prefill_segment(&mut runner, &prompt_ids[matched_tokens..boundary], sink)?;
                let capture_t0 = Instant::now();
                if capture_snapshot(
                    &mut self.cache,
                    &self.ctx,
                    &runner,
                    &prompt_ids[..boundary],
                    "transcript",
                ) {
                    transcript_entry = self.cache.entry_for(&prompt_ids[..boundary]);
                }
                capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
            }
            let start = transcript_split.unwrap_or(matched_tokens);
            prefill_segment(&mut runner, &prompt_ids[start..], sink)?;
            let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3 - capture_ms;
            let logits = runner
                .logits()
                .map_err(|error| ServeError::server_error(format!("{FAMILY} logits: {error}")))?
                .to_vec();
            // The transcript boundary a few header tokens earlier already
            // holds this prompt's reusable prefix.
            if transcript_entry.is_none() {
                let capture_t0 = Instant::now();
                capture_snapshot(&mut self.cache, &self.ctx, &runner, &prompt_ids, "prompt");
                capture_ms += capture_t0.elapsed().as_secs_f64() * 1e3;
            }

            let generation = decode_loop::decode_serial(
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
            )?;

            // Completed-turn capture under prompt + consumed generated tokens.
            // The transcript snapshot is what the next turn reuses when the
            // template re-renders this one; pin it so admitting the completed
            // boundary never evicts it.
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
                    let pinned = transcript_entry.filter(|&entry| self.cache.pin(entry));
                    capture_snapshot(&mut self.cache, &self.ctx, &runner, &consumed, "completed");
                    if let Some(entry) = pinned {
                        self.cache.unpin(entry);
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
                "serve phases: family=qwen4exp tokenize_ms={tokenize_ms:.1} restore_ms={restore_ms:.1} matched_tokens={matched_tokens} prefill_ms={prefill_ms:.1} prefill_tokens={} snapshot_capture_ms={capture_ms:.1}{transcript_phase} decode_ms={:.1} required_forwards={required} forward_limit={} transitions={} snapshot_cache_bytes={}",
                prompt_ids.len() - matched_tokens,
                generation.wall_ms,
                self.forward_limit,
                generation.transitions,
                self.cache.indexed_bytes(),
            );
            Ok(super::outcome::finish_generation(
                prompt_ids.len(),
                &generation,
                matched_tokens,
                restore_ms,
            ))
        };
        let outcome = run();
        // The workspace goes back reset whatever the outcome; a failed reset
        // leaves the backend without a session, which the next request
        // reports as a server error rather than serving stale state.
        let workspace = runner.into_workspace();
        self.loaded.restore_workspace(workspace).map_err(|error| {
            ServeError::server_error(format!("reset Qwen3.8-Flash-Next session: {error}"))
        })?;
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
    result.map_err(|error| match checkpoint_abort {
        Some(error) => BackendFailure::Aborted(error),
        None => ServeError::server_error(format!("prefill {FAMILY} prompt: {error}")).into(),
    })
}

/// Snapshot the runner's committed state under `tokens`; true when cached.
/// Pinned entries are never evicted to admit it.
fn capture_snapshot(
    cache: &mut SnapshotCache<Qwen4ExpTextSnapshot>,
    ctx: &MetalContext,
    runner: &Qwen4ExpTextRunner<'_, '_, '_>,
    tokens: &[u32],
    boundary: &'static str,
) -> bool {
    if cache.max_bytes() == 0 {
        return false;
    }
    let payload_bytes = match runner.snapshot_bytes() {
        Ok(bytes) => bytes,
        Err(error) => {
            tracing::warn!("serve: qwen4exp {boundary} snapshot estimate failed: {error}");
            return false;
        }
    };
    let Some(entry_bytes) = cache.strict_eligibility(tokens, payload_bytes) else {
        tracing::info!(
            target: "qwen_diag",
            "serve: qwen4exp {boundary} snapshot not cached (already present or over budget); payload_bytes={payload_bytes} cache_budget_bytes={}",
            cache.max_bytes(),
        );
        return false;
    };
    let signals = ctx.memory_signals();
    if let Err(reason) = super::snapshot_capture_admission(entry_bytes, signals) {
        tracing::warn!(
            "serve: qwen4exp {boundary} snapshot denied by memory headroom; reason={reason:?} payload_bytes={payload_bytes} metal_current_bytes={} metal_recommended_bytes={} process_remaining_bytes={:?}",
            signals.current_allocated_bytes,
            signals.recommended_max_bytes,
            signals.process_limit_remaining_bytes,
        );
        return false;
    }
    match runner.capture_snapshot() {
        Ok(snapshot) => {
            debug_assert_eq!(snapshot.payload_bytes(), payload_bytes);
            let inserted = cache.insert_strict(tokens.to_vec(), snapshot, entry_bytes);
            if !inserted {
                tracing::warn!("serve: qwen4exp {boundary} snapshot rejected at cache insertion");
            }
            inserted
        }
        Err(error) => {
            tracing::warn!("serve: qwen4exp {boundary} snapshot capture failed: {error}");
            false
        }
    }
}
