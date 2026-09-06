//! DFlash speculative decode loop and adaptive backoff.

use super::*;

/// **v0.77** DFlash adaptive-policy constants for `qwen run --drafter`.
///
/// Calibrated 2026-08-19 on M4 Max + Qwen3.8-27B Q4_K_M + incoai DFlash2
/// Q8_0 (block size 8); see the sweep tables in `qwen-bench`'s
/// `DFLASH2_ALPHA_WINDOW` doc comment and docs/PERF-LOG.md. Speculation
/// wins whenever mean emitted tokens/step exceeds the ctx-keyed
/// (draft + verify) / single_token premium; the trailing-α window is the
/// content-aware guard, since acceptance varies 2.5-5.8 at fixed ctx.
pub(crate) const DFLASH_ALPHA_WINDOW: usize = 16;

/// Sampled acceptance is often materially lower than greedy acceptance, so a
/// shorter observation window limits the cost of discovering a losing region.
/// Re-probes still allow speculation to resume when content changes.
pub(crate) const DFLASH_SAMPLED_ALPHA_WINDOW: usize = 8;

/// Break-even fit (mean emitted tokens/step at which Spec ties serial).
///
/// **2026-08-20 recalibration.** The previous form was
/// `2.9 + ctx/8000`, whose ctx term was fit on ABSOLUTE verify rows
/// compared ACROSS bench sessions — the exact procedure PERF-TOOLS
/// forbids, and the same phantom slope that produced (and then failed)
/// the V1 chunked-verify projection. Refit from a single-process
/// `--n-policy cycle` run (140+ samples per cell, ctx 464 -> 2062):
///
///   verify(8)    111.5 ms, within-session slope +0.81 ms/1K ctx (+0.73%/1K)
///   single_token  40.2 ms, within-session slope +0.40 ms/1K ctx (+1.00%/1K)
///   verify(1)     41.8 ms, within-session slope +0.28 ms/1K ctx
///
/// Break-even = (draft + verify) / single. Single-token cost grows
/// FASTER in relative terms than verify(8) does, because verify
/// amortizes one KV stream over 8 rows while serial decode re-reads it
/// every token. With the drafter's SWA window plateaued (>= 2048), the
/// derivative is `d(break-even)/d(1K ctx) = -0.011` — flat to slightly
/// DECLINING. Evaluated at both band ends the value is 3.12 / 3.10.
/// Positive-temperature Q4 DFlash2 measures the same short-context threshold:
/// draft ~12 ms + verify ~105 ms + read/sample/append/restore ~4 ms against
/// ~39 ms serial decode, or ~3.1 emitted tokens per speculative step.
///
/// So the ctx term is dropped, not merely reduced: speculation does not
/// get harder with context on this architecture, it gets marginally
/// easier. The hard `*_OFF_CTX` guard and the content-aware α-backoff
/// remain the safety nets.
///
/// The 2026-08-25 served-Q8 audit measured verify8/single ratios of
/// 1.86x/2.27x/2.71x at 8K/32K/64K. The verifier remains viable across the
/// band, but content acceptance and exact-fallback incidence still preclude a
/// default long-context promotion.
pub(crate) const DFLASH_BREAKEVEN_BASE: f64 = 3.1;

/// Trigger margin below break-even, sized ≈ 1 SE of the window mean.
pub(crate) const DFLASH_ALPHA_OFF_MARGIN: f64 = 0.6;

/// Hard ctx guard past the calibrated range.
pub(crate) const DFLASH_OFF_CTX_DEFAULT: usize = 16384;

pub(crate) const DFLASH_OFF_CTX_ENV: &str = "QWEN_DFLASH_OFF_CTX";

/// Off-steps between content-aware re-probe spec steps while in
/// trailing-alpha backoff. One probe = one draft+verify step (~125 ms),
/// so the steady backoff tax is bounded at ~16 ms/token above serial.
pub(crate) const DFLASH_REPROBE_INTERVAL: usize = 8;

/// Sampled probes have lower acceptance and include CPU distribution work;
/// space them out after a losing region while retaining eventual re-entry.
pub(crate) const DFLASH_SAMPLED_REPROBE_INTERVAL: usize = 32;

pub(crate) const DFLASH_LONG_ALPHA_OFF_MEAN: f64 = 2.9;

pub(crate) const DFLASH_LONG_FALLBACK_EARLY_WINDOW: usize = 4;

pub(crate) const DFLASH_LONG_FALLBACK_EARLY_LIMIT: usize = 2;

pub(crate) const DFLASH_LONG_FALLBACK_WINDOW: usize = 8;

pub(crate) const DFLASH_LONG_FALLBACK_LIMIT: usize = 4;

pub(crate) const DFLASH_LONG_REPROBE_INTERVAL: usize = 64;

pub(crate) const DFLASH_LONG_FALLBACK_REPROBE_INTERVAL: usize = 256;

pub(crate) const DFLASH_LONG_REENTRY_PROBES: usize = 2;

pub(crate) const DFLASH_LONG_REENTRY_MEAN: f64 = 3.1;

/// Domain separation for DFlash proposal sampling. Target sampling keeps the
/// request seed unchanged; stochastic proposals use this independent stream.
pub(crate) const DFLASH_PROPOSAL_SEED_XOR: u64 = 0x4446_4c41_5348_3251;

/// Default margin for the batched-verify exact fallback: a committed row
/// whose (top1 - top2) argmax gap is below this is re-evaluated through the
/// token-major path. Sized at ~2x the max batched-vs-token-major logit
/// delta observed by the shadow probe (9.5e-2, 2026-08-21).
pub(crate) const DFLASH_VERIFY_FALLBACK_MARGIN_DEFAULT: f32 = 0.2;

pub(crate) fn verify_fallback_margin(n_pos: usize) -> f32 {
    std::env::var("QWEN_DFLASH_VERIFY_MARGIN")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(if n_pos >= 4096 {
            // Ctx-aware: the batched-verify-vs-token-major divergence grows
            // with context and content — 9.5e-2 at 42 tokens, ~3.65e-1 at
            // 10.8K for BOTH the per-row and packed attention paths
            // (2026-08-22 shadow probes). 2x the observed max per tier.
            0.75
        } else {
            DFLASH_VERIFY_FALLBACK_MARGIN_DEFAULT
        })
}

pub(crate) fn verify_fallback_enabled() -> bool {
    std::env::var("QWEN_DFLASH_VERIFY_FALLBACK").map_or(true, |value| value != "0")
}

/// Ctx-keyed spec-vs-serial break-even in mean emitted tokens/step.
pub(crate) fn dflash_breakeven(_kv_n_pos: usize) -> f64 {
    DFLASH_BREAKEVEN_BASE
}

pub(crate) fn parse_dflash_off_ctx(value: Option<&OsStr>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(DFLASH_OFF_CTX_DEFAULT);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{DFLASH_OFF_CTX_ENV} is not valid UTF-8"))?;
    value
        .parse()
        .with_context(|| format!("{DFLASH_OFF_CTX_ENV}={value:?} is not a token count"))
}

pub(crate) fn configured_dflash_off_ctx() -> Result<usize> {
    parse_dflash_off_ctx(std::env::var_os(DFLASH_OFF_CTX_ENV).as_deref())
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DflashBackoffReason {
    Acceptance,
    Fallback,
}

impl DflashBackoffReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Acceptance => "acceptance",
            Self::Fallback => "fallback",
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct DflashAdaptiveState {
    pub(crate) alpha_window: Vec<usize>,
    pub(crate) fallback_window: Vec<bool>,
    pub(crate) reentry_accepts: Vec<usize>,
    pub(crate) reason: Option<DflashBackoffReason>,
    pub(crate) off_steps_since_probe: usize,
    pub(crate) long_mode: bool,
}

impl DflashAdaptiveState {
    pub(crate) fn prepare_mode(&mut self, long_mode: bool) {
        if long_mode && !self.long_mode {
            self.alpha_window.clear();
            self.fallback_window.clear();
            self.reentry_accepts.clear();
            self.long_mode = true;
        }
    }

    pub(crate) fn backoff_active(&self) -> bool {
        self.reason.is_some()
    }

    pub(crate) fn probe_due(&self, sampled_mode: bool, long_mode: bool) -> bool {
        let Some(reason) = self.reason else {
            return false;
        };
        let interval = if long_mode {
            match reason {
                DflashBackoffReason::Acceptance => DFLASH_LONG_REPROBE_INTERVAL,
                DflashBackoffReason::Fallback => DFLASH_LONG_FALLBACK_REPROBE_INTERVAL,
            }
        } else if sampled_mode {
            DFLASH_SAMPLED_REPROBE_INTERVAL
        } else {
            DFLASH_REPROBE_INTERVAL
        };
        self.off_steps_since_probe > 0 && self.off_steps_since_probe.is_multiple_of(interval)
    }

    pub(crate) fn record_off_step(&mut self) {
        self.off_steps_since_probe += 1;
    }

    pub(crate) fn begin_spec_step(&mut self) {
        self.off_steps_since_probe = 0;
    }

    pub(crate) fn enter_long_backoff(&mut self, reason: DflashBackoffReason) {
        self.reason = Some(reason);
        self.alpha_window.clear();
        self.fallback_window.clear();
        self.reentry_accepts.clear();
    }

    pub(crate) fn record_spec_step(
        &mut self,
        n_accepted: usize,
        fallback_ran: bool,
        was_probe: bool,
        reentry_eligible: bool,
        sampled_mode: bool,
        long_mode: bool,
        position: usize,
    ) {
        if long_mode {
            if was_probe {
                if fallback_ran {
                    self.reason = Some(DflashBackoffReason::Fallback);
                    self.reentry_accepts.clear();
                    return;
                }
                if !reentry_eligible {
                    self.reentry_accepts.clear();
                    return;
                }
                self.reentry_accepts.push(n_accepted);
                if self.reentry_accepts.len() > DFLASH_LONG_REENTRY_PROBES {
                    self.reentry_accepts.remove(0);
                }
                if self.reentry_accepts.len() == DFLASH_LONG_REENTRY_PROBES {
                    let mean_emitted = 1.0
                        + self.reentry_accepts.iter().sum::<usize>() as f64
                            / DFLASH_LONG_REENTRY_PROBES as f64;
                    if mean_emitted >= DFLASH_LONG_REENTRY_MEAN {
                        self.reason = None;
                        self.alpha_window.clear();
                        self.fallback_window.clear();
                        self.reentry_accepts.clear();
                    }
                }
                return;
            }

            self.alpha_window.push(n_accepted);
            if self.alpha_window.len() > DFLASH_ALPHA_WINDOW {
                self.alpha_window.remove(0);
            }
            self.fallback_window.push(fallback_ran);
            if self.fallback_window.len() > DFLASH_LONG_FALLBACK_WINDOW {
                self.fallback_window.remove(0);
            }
            let fallback_losing = self.fallback_window.len() == DFLASH_LONG_FALLBACK_WINDOW
                && self.fallback_window.iter().filter(|&&ran| ran).count()
                    >= DFLASH_LONG_FALLBACK_LIMIT;
            let early_fallback_losing = self.fallback_window.len()
                >= DFLASH_LONG_FALLBACK_EARLY_WINDOW
                && self.fallback_window
                    [self.fallback_window.len() - DFLASH_LONG_FALLBACK_EARLY_WINDOW..]
                    .iter()
                    .filter(|&&ran| ran)
                    .count()
                    >= DFLASH_LONG_FALLBACK_EARLY_LIMIT
                && 1.0
                    + self.alpha_window
                        [self.alpha_window.len() - DFLASH_LONG_FALLBACK_EARLY_WINDOW..]
                        .iter()
                        .sum::<usize>() as f64
                        / (DFLASH_LONG_FALLBACK_EARLY_WINDOW as f64)
                    < DFLASH_LONG_ALPHA_OFF_MEAN;
            let alpha_losing = self.alpha_window.len() == DFLASH_ALPHA_WINDOW
                && 1.0
                    + self.alpha_window.iter().sum::<usize>() as f64 / (DFLASH_ALPHA_WINDOW as f64)
                    < DFLASH_LONG_ALPHA_OFF_MEAN;
            if early_fallback_losing || fallback_losing {
                self.enter_long_backoff(DflashBackoffReason::Fallback);
            } else if alpha_losing {
                self.enter_long_backoff(DflashBackoffReason::Acceptance);
            }
            return;
        }

        let window_size = if sampled_mode {
            DFLASH_SAMPLED_ALPHA_WINDOW
        } else {
            DFLASH_ALPHA_WINDOW
        };
        self.alpha_window.push(n_accepted);
        if self.alpha_window.len() > window_size {
            self.alpha_window.remove(0);
        }
        self.fallback_window.push(fallback_ran);
        if self.fallback_window.len() > window_size {
            self.fallback_window.remove(0);
        }
        if self.alpha_window.len() == window_size {
            let mean_emitted =
                1.0 + self.alpha_window.iter().sum::<usize>() as f64 / window_size as f64;
            // Exact fallback repeats every committed target row serially;
            // those accepted tokens save no target evaluations.
            let mean_saved = self
                .alpha_window
                .iter()
                .zip(&self.fallback_window)
                .map(|(&accepted, &fallback)| if fallback { 0 } else { accepted + 1 })
                .sum::<usize>() as f64
                / window_size as f64;
            let threshold = dflash_breakeven(position) - DFLASH_ALPHA_OFF_MARGIN;
            if mean_saved < threshold {
                self.reason = Some(if mean_emitted < threshold {
                    DflashBackoffReason::Acceptance
                } else {
                    DflashBackoffReason::Fallback
                });
            } else if self.backoff_active() {
                self.reason = None;
            }
        }
    }
}

pub(crate) fn dflash_speculation_enabled(
    spec_disabled: bool,
    position: usize,
    off_ctx: usize,
    remaining_context_tokens: usize,
    block_size: usize,
    backoff_active: bool,
    backoff_probe_due: bool,
) -> bool {
    !spec_disabled
        && position < off_ctx
        && remaining_context_tokens >= block_size
        && (!backoff_active || backoff_probe_due)
}

pub(crate) fn dflash_prefix_replay_drafts<'a>(
    history: &'a [i32],
    emitted: &[i32],
    draft_tokens: usize,
) -> Option<&'a [i32]> {
    if history.get(..emitted.len())? != emitted {
        return None;
    }
    history.get(emitted.len()..emitted.len().checked_add(draft_tokens)?)
}

pub(crate) fn dflash_prefix_replay_allowed(
    spec_disabled: bool,
    position: usize,
    off_ctx: usize,
    remaining_context_tokens: usize,
    block_size: usize,
    long_mode: bool,
    backoff_active: bool,
    backoff_probe_due: bool,
) -> bool {
    !spec_disabled
        && position < off_ctx
        && remaining_context_tokens >= block_size
        && (!long_mode || !backoff_active || backoff_probe_due)
}

/// **v0.77** DFlash speculative-decode statistics for one request.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct DflashDecodeStats {
    pub(crate) off_ctx: usize,
    pub(crate) draft_ms: f64,
    pub(crate) draft_first_call_ms: f64,
    pub(crate) verify_ms: f64,
    pub(crate) sampled_logits_read_ms: f64,
    pub(crate) sample_ms: f64,
    pub(crate) append_ms: f64,
    pub(crate) restore_ms: f64,
    pub(crate) fallback_ms: f64,
    pub(crate) backoff_probe_ms: f64,
    pub(crate) serial_ms: f64,
    pub(crate) scratch_allocation_ms: f64,
    pub(crate) drafter_calls: usize,
    pub(crate) prefix_replay_steps: usize,
    pub(crate) prefix_replay_accepted_drafts: usize,
    pub(crate) prefix_replay_drafts_scored: usize,
    pub(crate) prefix_replay_mismatches: usize,
    pub(crate) verify_calls: usize,
    pub(crate) restore_calls: usize,
    pub(crate) spec_steps: usize,
    pub(crate) off_steps: usize,
    pub(crate) accepted_drafts: usize,
    pub(crate) drafts_scored: usize,
    pub(crate) physical_target_positions: usize,
    pub(crate) fallback_calls: usize,
    pub(crate) backoff_probe_steps: usize,
    pub(crate) backoff_reason: Option<DflashBackoffReason>,
    pub(crate) alpha_backoff: bool,
}

pub(crate) struct DflashGeneration {
    pub(crate) generation: GenerationResult,
    pub(crate) stats: DflashDecodeStats,
    pub(crate) sequence: Sequence,
}

pub(crate) struct DflashDecodeScratch {
    pub(crate) verify: MetalDFlashVerifyScratch,
    pub(crate) layer: MetalDFlashLayerMajorScratch,
    pub(crate) sampled_logits: Option<MetalTensor>,
    pub(crate) allocation_ms: f64,
}

pub(crate) fn allocate_dflash_decode_scratch(
    loaded: &LoadedModel,
    head: &MetalDFlashHead,
    sampled: bool,
) -> Result<DflashDecodeScratch> {
    let allocation_t0 = Instant::now();
    let verify = MetalDFlashVerifyScratch::fresh(
        loaded.context(),
        loaded.metal_model(),
        head.config.block_size,
        u32::try_from(head.target_layer_ids.len()).context("DFlash target layer count overflow")?,
    )
    .context("allocate dflash verify scratch")?;
    let layer = MetalDFlashLayerMajorScratch::fresh(
        loaded.context(),
        loaded.metal_model(),
        head.config.block_size,
    )
    .context("allocate dflash layer scratch")?;
    let sampled_logits = if sampled {
        Some(
            MetalTensor::zeros_f32(
                loaded.context(),
                vec![
                    head.config.block_size as u64,
                    loaded.arch().vocab_size as u64,
                ],
            )
            .context("allocate sampled dflash target logits")?,
        )
    } else {
        None
    };
    Ok(DflashDecodeScratch {
        verify,
        layer,
        sampled_logits,
        allocation_ms: allocation_t0.elapsed().as_secs_f64() * 1e3,
    })
}

/// **v0.77** DFlash speculative decode for `qwen run --drafter`.
///
/// Sampled DFlash 2 requests use maximal coupling against the selector's sparse
/// proposal distribution. Other drafters fall back to deterministic target-first
/// coupling. Both keep verification on the accelerated packed target path.
///
/// The drafter conditions on captured target hidden states for every
/// committed position: the caller must have prefilled with capture and
/// seeded `dsess` with the prompt columns.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_dflash<OnToken>(
    loaded: &LoadedModel,
    forward: &MetalForward<'_>,
    head: &MetalDFlashHead,
    dsess: MetalDFlashSession,
    scratch: DflashDecodeScratch,
    mut sequence: Sequence,
    logits: Vec<f32>,
    sampler: &mut Sampler,
    max_tokens: usize,
    stop_tokens: &[i32],
    capture_ring: Option<(MetalTensor, usize, usize, usize)>,
    prefix_replay: Option<&[i32]>,
    mut shadow_probe: Option<&mut Sequence>,
    mut on_token: OnToken,
) -> Result<DflashGeneration>
where
    OnToken: FnMut(i32) -> Result<()>,
{
    ensure!(max_tokens > 0, "max_tokens must be >= 1");
    let wall_t0 = Instant::now();
    let selection_t0 = Instant::now();
    let sampled_mode = sampler.config().temperature > 0.0;
    let off_ctx = configured_dflash_off_ctx()?;
    let mut proposal_sampler = if sampled_mode && head.selector.is_some() {
        let target_config = sampler.config();
        Some(
            Sampler::new(SamplingConfig {
                temperature: target_config.temperature,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: target_config.seed ^ DFLASH_PROPOSAL_SEED_XOR,
            })
            .context("initialize dflash proposal sampler")?,
        )
    } else {
        None
    };
    let mut carry = sampler
        .sample(&logits)
        .context("sample dflash initial token")?
        .token;
    let first_token_selection_ms = selection_t0.elapsed().as_secs_f64() * 1e3;
    let first_token_ready_ms = Some(wall_t0.elapsed().as_secs_f64() * 1e3);
    let mut first_token_callback_ms = None;
    let mut first_transition_ms = None;
    let mut tokens = Vec::with_capacity(max_tokens);
    let mut transitions = 0usize;
    let mut transition_wall_ms = 0.0;
    let mut stats = DflashDecodeStats::default();
    stats.off_ctx = off_ctx;

    let cfg = head.config;
    let n_block = cfg.block_size as usize;
    let k_layers = head.target_layer_ids.len();
    let n_target_features = k_layers * loaded.arch().hidden_size as usize;

    let DflashDecodeScratch {
        verify: mut verify_scratch,
        layer: mut layer_scratch,
        sampled_logits,
        allocation_ms,
    } = scratch;
    stats.scratch_allocation_ms = allocation_ms;

    let debug_scratch = if shadow_probe.is_some() && !sampled_mode {
        Some(
            MetalDFlashDebugScratch::fresh(
                loaded.context(),
                loaded.metal_model(),
                n_block as u32,
                k_layers as u32,
            )
            .context("allocate shadow probe debug scratch")?,
        )
    } else {
        None
    };

    let mut decoder = DFlashDecoder::new(forward, head, dsess);
    let mut sampled_logits_cpu = Vec::<f32>::new();

    // Adaptive policy state: `spec_disabled` is the hard OFF_CTX stop
    // (terminal). Content backoff periodically re-probes while off steps keep
    // the drafter cross-context and caller capture ring current.
    let mut spec_disabled = false;
    let mut adaptive = DflashAdaptiveState::default();

    let stop_reason = 'outer: loop {
        shutdown::checkpoint()?;
        tokens.push(carry);
        if stop_tokens.contains(&carry) {
            break StopReason::Eos;
        }
        on_token(carry)?;
        first_token_callback_ms.get_or_insert_with(|| wall_t0.elapsed().as_secs_f64() * 1e3);
        if tokens.len() == max_tokens {
            break StopReason::TokenLimit;
        }

        let transition_t0 = Instant::now();
        let position = sequence.position();
        let long_mode = off_ctx > DFLASH_OFF_CTX_DEFAULT && position >= DFLASH_OFF_CTX_DEFAULT;
        adaptive.prepare_mode(long_mode);
        let backoff_active = adaptive.backoff_active();
        let backoff_probe_due = adaptive.probe_due(sampled_mode, long_mode);
        let remaining_context_tokens = sequence.remaining_context_tokens();
        let prefix_replay_drafts = prefix_replay.and_then(|history| {
            dflash_prefix_replay_allowed(
                spec_disabled,
                position,
                off_ctx,
                remaining_context_tokens,
                n_block,
                long_mode,
                backoff_active,
                backoff_probe_due,
            )
            .then(|| dflash_prefix_replay_drafts(history, &tokens, n_block.saturating_sub(1)))?
        });
        let spec_enabled = prefix_replay_drafts.is_some()
            || dflash_speculation_enabled(
                spec_disabled,
                position,
                off_ctx,
                remaining_context_tokens,
                n_block,
                backoff_active,
                backoff_probe_due,
            );

        if !spec_enabled {
            // Off step: exact single-token decode with multi-hidden
            // capture so the caller's capture ring stays fed. The drafter
            // cross-context is appended only while re-entry is possible
            // (below the hard OFF_CTX guard, which is terminal).
            if position >= off_ctx || remaining_context_tokens < n_block {
                spec_disabled = true;
            }
            stats.off_steps += 1;
            adaptive.record_off_step();
            sequence.ensure_can_append(1)?;
            let serial_t0 = Instant::now();
            let hidden_dst = verify_scratch.hidden_capture_n_slot(0);
            let next = forward
                .single_token_with_multi_hidden(
                    carry,
                    position as u32,
                    unsafe { sequence.metal_session_mut() },
                    &head.target_layer_ids,
                    &hidden_dst,
                )
                .context("dflash off-mode capture single_token")?;
            if !spec_disabled {
                decoder
                    .session
                    .append_target_ctx_columns_now(
                        loaded.context(),
                        std::slice::from_ref(&(&hidden_dst, position as u32)),
                        n_target_features,
                    )
                    .context("dflash off-mode ctx append")?;
            }
            if let Some((ring, wstart, features, ring_window)) = capture_ring
                .as_ref()
                .filter(|(_, wstart, _, _)| position >= *wstart)
            {
                let offset = (position - *wstart) % *ring_window;
                let view = ring.view_subrange((offset * *features) as u64, vec![*features as u64]);
                let ring_encoder = loaded.context().queue.commandBuffer().expect("cmd");
                let ring_enc = KernelEncoder::begin(&ring_encoder);
                encode_scatter_offset_f32(
                    loaded.context(),
                    &ring_enc,
                    &hidden_dst,
                    &view,
                    0,
                    *features,
                )
                .context("off-mode ring scatter")?;
                ring_enc.end();
                ring_encoder.commit();
                ring_encoder.waitUntilCompleted();
            }
            stats.serial_ms += serial_t0.elapsed().as_secs_f64() * 1e3;
            sequence.advance_by(1)?;
            transitions += 1;
            let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
            transition_wall_ms += elapsed_ms;
            first_transition_ms.get_or_insert(elapsed_ms);
            carry = sampler
                .sample(&next)
                .context("sample dflash off-mode token")?
                .token;
            continue;
        }
        let backoff_probe_step = backoff_active && backoff_probe_due;
        if backoff_probe_step {
            stats.backoff_probe_steps += 1;
        }
        adaptive.begin_spec_step();

        // ---- Draft ----
        let drafter_pos = position as u32;
        let prefix_replay_step = prefix_replay_drafts.is_some();
        let (draft_tokens, sparse_proposals) = if let Some(replay) = prefix_replay_drafts {
            if let Some(proposal_sampler) = proposal_sampler.as_mut() {
                let draws = proposal_sampler
                    .draws()
                    .checked_add(replay.len())
                    .context("DFlash proposal draw count overflow")?;
                *proposal_sampler = Sampler::at_draw(proposal_sampler.config(), draws)
                    .context("advance DFlash proposal RNG across prefix replay")?;
            }
            let mut drafts = Vec::with_capacity(n_block);
            drafts.push(carry);
            drafts.extend_from_slice(replay);
            stats.prefix_replay_steps += 1;
            (drafts, None)
        } else {
            let draft_t0 = Instant::now();
            let drafted = if let Some(proposal_sampler) = proposal_sampler.as_mut() {
                let block = decoder
                    .draft_block_sampled(carry, drafter_pos, proposal_sampler)
                    .context("sample dflash2 proposal path")?;
                (block.draft_tokens, Some(block.proposals))
            } else {
                (
                    decoder
                        .draft_block(carry, drafter_pos)
                        .context("dflash draft_block")?,
                    None,
                )
            };
            let draft_ms = draft_t0.elapsed().as_secs_f64() * 1e3;
            if stats.drafter_calls == 0 {
                // One-time prompt projection through the drafter caches.
                stats.draft_first_call_ms = draft_ms;
            } else {
                stats.draft_ms += draft_ms;
            }
            stats.drafter_calls += 1;
            drafted
        };
        stats.spec_steps += 1;

        // ---- Packed verify: [carry, drafts...] ----
        let mut verify_input = Vec::with_capacity(n_block);
        verify_input.push(carry);
        verify_input.extend_from_slice(&draft_tokens[1..n_block]);
        let n_eff = verify_input.len();
        let n_drafts_scored = n_eff - 1;
        sequence.ensure_can_append(n_eff)?;

        let verify_t0 = Instant::now();
        let verify_logits = if sampled_mode {
            sampled_logits.as_ref()
        } else {
            debug_scratch.as_ref().map(|debug| &debug.debug_logits)
        };
        let verify_argmax = qwen_llm::metal_dflash::encode_packed_verify_layer_major_inner(
            forward,
            &head.target_layer_ids,
            &verify_input,
            drafter_pos,
            &mut verify_scratch,
            &mut layer_scratch,
            unsafe { sequence.metal_session_mut() },
            verify_logits,
            Some(n_eff as u32),
        )
        .context("dflash packed verify")?;
        stats.verify_ms += verify_t0.elapsed().as_secs_f64() * 1e3;
        stats.verify_calls += 1;
        stats.drafts_scored += n_drafts_scored;
        stats.physical_target_positions += n_eff;

        // ---- Accept-prefix ----
        let mut accepted = Vec::with_capacity(n_drafts_scored);
        let mut sampled_targets = Vec::with_capacity(n_eff);
        let mut terminal = None;
        if sampled_mode {
            let logits = sampled_logits
                .as_ref()
                .context("sampled dflash logits scratch is absent")?;
            let vocab = loaded.arch().vocab_size as usize;
            let read_t0 = Instant::now();
            sampled_logits_cpu.resize(n_eff * vocab, 0.0);
            unsafe {
                let source = logits
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(logits.offset as usize)
                    .cast::<f32>();
                std::ptr::copy_nonoverlapping(
                    source,
                    sampled_logits_cpu.as_mut_ptr(),
                    sampled_logits_cpu.len(),
                );
            }
            stats.sampled_logits_read_ms += read_t0.elapsed().as_secs_f64() * 1e3;
            let base = sampled_logits_cpu.as_ptr();
            let sample_t0 = Instant::now();
            for (row, &draft) in verify_input[1..].iter().enumerate() {
                let target_logits =
                    unsafe { std::slice::from_raw_parts(base.add(row * vocab), vocab) };
                let decision = if let Some(proposals) = sparse_proposals.as_ref() {
                    sampler
                        .couple_sparse_proposal(
                            target_logits,
                            proposals
                                .get(row)
                                .context("dflash2 sparse proposal row is absent")?,
                        )
                        .context("couple sparse dflash2 proposal to target sampler")?
                } else {
                    sampler
                        .couple_deterministic_proposal(target_logits, draft)
                        .context("couple deterministic dflash proposal to target sampler")?
                };
                match decision {
                    SpeculativeSamplingDecision::Accepted => sampled_targets.push(draft),
                    SpeculativeSamplingDecision::Rejected { correction } => {
                        sampled_targets.push(correction);
                        break;
                    }
                }
                accepted.push(draft);
                if stop_tokens.contains(&draft) {
                    terminal = Some(StopReason::Eos);
                    break;
                }
                if tokens.len() + accepted.len() == max_tokens {
                    terminal = Some(StopReason::TokenLimit);
                    break;
                }
            }
            if terminal.is_none() && accepted.len() == n_drafts_scored {
                let bonus_logits =
                    unsafe { std::slice::from_raw_parts(base.add(n_drafts_scored * vocab), vocab) };
                sampled_targets.push(
                    sampler
                        .sample(bonus_logits)
                        .context("sample dflash target bonus token")?
                        .token,
                );
            }
            stats.sample_ms += sample_t0.elapsed().as_secs_f64() * 1e3;
        } else {
            for (&draft, &target) in verify_input[1..].iter().zip(&verify_argmax) {
                if draft != target {
                    break;
                }
                accepted.push(draft);
                if stop_tokens.contains(&draft) {
                    terminal = Some(StopReason::Eos);
                    break;
                }
                if tokens.len() + accepted.len() == max_tokens {
                    terminal = Some(StopReason::TokenLimit);
                    break;
                }
            }
        }
        let mut n_accepted = accepted.len();
        let mut n_keep = if terminal.is_some() {
            n_accepted
        } else {
            n_accepted + 1
        };
        // ---- Margin-guarded exact fallback ----
        // The batched verify arithmetic diverges from token-major by up to
        // ~1e-1 absolute on low-confidence rows (shadow probe, 2026-08-21).
        // When any committed row's (top1 - top2) gap is below the margin the
        // batched argmax could flip, so replay the block through exact
        // token-major forwards and re-accept against their argmaxes.
        let mut fallback_ran = false;
        let mut fallback_targets: Vec<i32> = Vec::new();
        if !sampled_mode && verify_fallback_enabled() {
            let margin = verify_fallback_margin((drafter_pos as usize) + n_eff);
            let gaps = unsafe {
                let src = verify_scratch.verify_gap.buffer.contents().as_ptr() as *const f32;
                std::slice::from_raw_parts(src, n_eff)
            };
            let flagged = (0..=n_accepted).any(|i| !(gaps[i].is_finite() && gaps[i] >= margin));
            if std::env::var_os("QWEN_DFLASH_VERIFY_FALLBACK_DIAG").is_some() {
                eprintln!(
                    "[fallback-diag] step_pos={drafter_pos} n_eff={n_eff} n_accepted={n_accepted} gaps={:?} flagged={flagged}",
                    &gaps[..n_eff]
                );
            }
            if flagged {
                let fallback_t0 = Instant::now();
                fallback_ran = true;
                stats.fallback_calls += 1;
                qwen_llm::metal_dflash::encode_restore_to_pre_block(
                    forward,
                    &verify_scratch,
                    drafter_pos,
                    unsafe { sequence.metal_session_mut() },
                    Some(n_eff as u32),
                )
                .context("verify fallback restore to pre-block state")?;
                // Exact replay with adaptive stop: row 0 is the carry; each
                // further row replays a draft only while the reference stream
                // keeps accepting it. The replay stops at the first mismatch,
                // so the session advances exactly the committed-row count.
                fallback_targets.clear();
                fallback_targets.reserve(n_eff);
                accepted.clear();
                terminal = None;
                {
                    let mut replay_row = |row: usize, token: i32| -> Result<i32> {
                        let pos = drafter_pos + row as u32;
                        let hidden_dst = verify_scratch.hidden_capture_n_slot(row as u32);
                        let ref_logits = forward
                            .single_token_with_multi_hidden(
                                token,
                                pos,
                                unsafe { sequence.metal_session_mut() },
                                &head.target_layer_ids,
                                &hidden_dst,
                            )
                            .context("verify fallback exact row")?;
                        if let Some((ring, wstart, features, ring_window)) = capture_ring
                            .as_ref()
                            .filter(|(_, wstart, _, _)| (pos as usize) >= *wstart)
                        {
                            let offset = ((pos as usize - *wstart) % *ring_window) * *features;
                            let view = ring.view_subrange(offset as u64, vec![*features as u64]);
                            let ring_encoder = loaded.context().queue.commandBuffer().expect("cmd");
                            let ring_enc = KernelEncoder::begin(&ring_encoder);
                            encode_scatter_offset_f32(
                                loaded.context(),
                                &ring_enc,
                                &hidden_dst,
                                &view,
                                0,
                                *features,
                            )
                            .context("fallback ring scatter")?;
                            ring_enc.end();
                            ring_encoder.commit();
                            ring_encoder.waitUntilCompleted();
                        }
                        Ok(argmax_i32(&ref_logits))
                    };
                    let row0_target = replay_row(0, verify_input[0])?;
                    fallback_targets.push(row0_target);
                    for (i, &draft) in verify_input[1..].iter().enumerate() {
                        if draft != fallback_targets[i] {
                            break;
                        }
                        accepted.push(draft);
                        // Terminal checks BEFORE replaying the next row: the
                        // terminal token is emitted but never consumed, and
                        // the session must end exactly n_keep rows ahead of
                        // the pre-block state or the completed-boundary
                        // checkpoint fails KvPosition validation.
                        if stop_tokens.contains(&draft) {
                            terminal = Some(StopReason::Eos);
                            break;
                        }
                        if tokens.len() + accepted.len() == max_tokens {
                            terminal = Some(StopReason::TokenLimit);
                            break;
                        }
                        let target = replay_row(i + 1, draft)?;
                        fallback_targets.push(target);
                    }
                }
                n_accepted = accepted.len();
                n_keep = if terminal.is_some() {
                    n_accepted
                } else {
                    n_accepted + 1
                };
                if fallback_targets.len() < n_keep {
                    fallback_targets.resize(n_keep, 0);
                }
                stats.fallback_ms += fallback_t0.elapsed().as_secs_f64() * 1e3;
            }
        }
        if prefix_replay_step {
            stats.prefix_replay_accepted_drafts += n_accepted;
            stats.prefix_replay_drafts_scored += n_drafts_scored;
            if terminal.is_none() && n_accepted < n_drafts_scored {
                stats.prefix_replay_mismatches += 1;
            }
        }

        // ---- Append captured hiddens for committed positions ----
        // Columns 0..=n_accepted are carry + accepted drafts. The bonus
        // position is committed on the NEXT iteration, when it is carry.
        if n_keep > 0 {
            let append_t0 = Instant::now();
            let columns: Vec<(MetalTensor, u32)> = (0..n_keep)
                .map(|i| {
                    (
                        verify_scratch.hidden_capture_n_slot(i as u32),
                        drafter_pos + i as u32,
                    )
                })
                .collect();
            let column_refs: Vec<(&MetalTensor, u32)> =
                columns.iter().map(|(t, p)| (t, *p)).collect();
            decoder
                .session
                .append_target_ctx_columns_now(loaded.context(), &column_refs, n_target_features)
                .context("append dflash ctx columns")?;
            // Optional windowed capture-ring feed: scatter the same committed
            // hidden columns into the caller's ring at windowed offsets so
            // completed-boundary capture tails stay fresh during speculation.
            if let Some((ring, wstart, features, ring_window)) = capture_ring.as_ref() {
                let ring_encoder = loaded.context().queue.commandBuffer().expect("cmd");
                let ring_enc = KernelEncoder::begin(&ring_encoder);
                for i in 0..n_keep {
                    let position = drafter_pos + i as u32;
                    if (position as usize) < *wstart {
                        continue;
                    }
                    let offset = ((position as usize - *wstart) % *ring_window) * *features;
                    let view = ring.view_subrange(offset as u64, vec![*features as u64]);
                    encode_scatter_offset_f32(
                        loaded.context(),
                        &ring_enc,
                        &verify_scratch.hidden_capture_n_slot(i as u32),
                        &view,
                        0,
                        *features,
                    )
                    .context("ring scatter dflash ctx column")?;
                }
                ring_enc.end();
                ring_encoder.commit();
                ring_encoder.waitUntilCompleted();
            }
            stats.append_ms += append_t0.elapsed().as_secs_f64() * 1e3;
        }

        // ---- Restore target state on partial accept ----
        // Skipped when the margin fallback ran: the exact replay already
        // left the session at the committed-row state (and the
        // kv_n_pos contract the partial-accept restore validates no
        // longer holds).
        if !fallback_ran && n_keep < n_eff {
            let restore_t0 = Instant::now();
            qwen_llm::metal_dflash::encode_restore_after_partial_accept_inner(
                forward,
                &verify_scratch,
                n_keep as u32,
                drafter_pos,
                unsafe { sequence.metal_session_mut() },
                Some(n_eff as u32),
            )
            .context("dflash restore after partial accept")?;
            stats.restore_ms += restore_t0.elapsed().as_secs_f64() * 1e3;
            stats.restore_calls += 1;
        }
        sequence.advance_by(n_keep)?;
        transitions += n_keep;
        stats.accepted_drafts += n_accepted;
        let elapsed_ms = transition_t0.elapsed().as_secs_f64() * 1e3;
        transition_wall_ms += elapsed_ms;
        if backoff_probe_step {
            stats.backoff_probe_ms += elapsed_ms;
        }
        first_transition_ms.get_or_insert(elapsed_ms);

        // ---- Shadow-reference divergence probe ----
        // Replay the committed prefix through a separately prefilled
        // serial-reference session and compare the reference argmax against
        // the packed-verify row-0 argmax. Emits per-step evidence; the fork
        // step is the one where spec != ref.
        if let Some(shadow) = shadow_probe.as_deref_mut() {
            let committed = &verify_input[..n_keep];
            for (i, &tok) in committed.iter().enumerate() {
                let pos = drafter_pos + i as u32;
                let ref_logits = forward
                    .single_token(tok, pos, unsafe { shadow.metal_session_mut() })
                    .context("shadow probe single_token")?;
                shadow.advance_by(1)?;
                let ref_tok = argmax_i32(&ref_logits);
                let spec_tok = if fallback_ran {
                    fallback_targets[i]
                } else {
                    verify_argmax[i]
                };
                let mut top1 = ref_logits[0];
                let mut top2 = f32::NEG_INFINITY;
                for &v in &ref_logits[1..] {
                    if v > top1 {
                        top2 = top1;
                        top1 = v;
                    } else if v > top2 {
                        top2 = v;
                    }
                }
                let ref_gap = top1 - top2;
                let max_delta = match sampled_logits
                    .as_ref()
                    .or_else(|| debug_scratch.as_ref().map(|debug| &debug.debug_logits))
                {
                    Some(debug_logits) => {
                        let src = debug_logits.buffer.contents().as_ptr() as *const f32;
                        let v = loaded.arch().vocab_size as usize;
                        let mut max_delta = 0.0f32;
                        for (j, &r) in ref_logits.iter().enumerate() {
                            let d = unsafe { (*src.add(i * v + j) - r).abs() };
                            if d > max_delta {
                                max_delta = d;
                            }
                        }
                        max_delta
                    }
                    None => f32::NAN,
                };
                if spec_tok != ref_tok {
                    eprintln!(
                        "[shadow-probe] FLIP pos={pos} spec={spec_tok} ref={ref_tok} ref_gap={ref_gap:.6e} max_delta={max_delta:.6e}"
                    );
                }
                eprintln!("[shadow-probe] row pos={pos} gap={ref_gap:.6e} delta={max_delta:.6e}");
            }
        }
        for token in accepted {
            tokens.push(token);
            if !stop_tokens.contains(&token) {
                on_token(token)?;
            }
        }
        if let Some(reason) = terminal {
            break 'outer reason;
        }

        // ---- Content and exact-fallback backoff / re-entry ----
        adaptive.record_spec_step(
            n_accepted,
            fallback_ran,
            backoff_probe_step,
            !prefix_replay_step,
            sampled_mode,
            long_mode,
            sequence.position(),
        );
        stats.backoff_reason = adaptive.reason;
        if adaptive.reason.is_some() {
            stats.alpha_backoff = true;
        }

        carry = if sampled_mode {
            *sampled_targets
                .get(n_accepted)
                .context("sampled dflash target frontier is absent")?
        } else if fallback_ran {
            fallback_targets[n_accepted]
        } else {
            verify_argmax[n_accepted]
        };
    };

    ensure!(
        transitions.checked_add(1) == Some(tokens.len()),
        "dflash generation violated N-1 transition semantics"
    );
    Ok(DflashGeneration {
        generation: GenerationResult {
            tokens,
            wall_ms: wall_t0.elapsed().as_secs_f64() * 1e3,
            first_token_selection_ms,
            first_token_ready_ms,
            first_token_callback_ms,
            transitions,
            transition_ms: transition_wall_ms,
            first_transition_ms,
            stop_reason,
        },
        stats,
        sequence,
    })
}
