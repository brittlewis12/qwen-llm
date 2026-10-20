//! Request-scoped GLM-Flash DFlash packets. The caller emits `packet.tokens`
//! exactly once, then passes its nonterminal carry back to `step`.

use super::*;
use crate::metal_dflash::{
    DFlashAcceptDecision, DFlashAcceptError, DFlashAcceptStopReason, DFlashError, DFlashNextCarry,
    MetalDFlashDraftDecoder, MetalDFlashDraftTarget, MetalDFlashHead, MetalDFlashSession,
    decide_dflash_accept_prefix, dflash_capture_window_limit, dflash_capture_window_span,
};
use crate::sampling::{Sampler, SamplingError};
use objc2_metal::MTLResource;
use std::time::{Duration, Instant};

pub type PacketResult<T> = std::result::Result<T, Glm5NextDFlashError>;

#[derive(Debug, thiserror::Error)]
pub enum Glm5NextDFlashError {
    #[error("invalid GLM DFlash request: {0}")]
    Invalid(&'static str),
    #[error("GLM DFlash request is closed")]
    Closed,
    #[error("GLM DFlash request cancelled")]
    Cancelled,
    #[error(transparent)]
    Target(#[from] Glm5NextMetalError),
    #[error(transparent)]
    Draft(#[from] DFlashError),
    #[error(transparent)]
    Sampling(#[from] SamplingError),
    #[error(transparent)]
    Accept(#[from] DFlashAcceptError),
    #[error("{cause}; target rollback also failed: {restore}")]
    Recovery {
        cause: Box<Glm5NextDFlashError>,
        restore: Glm5NextMetalError,
    },
}

impl Glm5NextDFlashError {
    /// Retryable admission pressure, not malformed geometry or unavailable
    /// telemetry. The Target variant preserves the native typed refusal.
    pub fn is_memory_pressure(&self) -> bool {
        matches!(self, Self::Target(error) if error.is_memory_pressure())
    }
}

fn admit_request(
    draft_bytes: u64,
    bank_bytes: u64,
    checkpoint_bytes: u64,
    additional_cpu_reserve: u64,
    signals: crate::metal::MetalMemorySignals,
) -> PacketResult<u64> {
    let priced_bytes = draft_bytes
        .checked_add(bank_bytes)
        .and_then(|n| n.checked_add(checkpoint_bytes))
        .ok_or(Glm5NextDFlashError::Invalid("runner price overflow"))?;
    let admission = evaluate_metal_memory_admission_with_cpu_bytes(
        priced_bytes,
        additional_cpu_reserve,
        memory::DYNAMIC_RESERVE_BYTES,
        signals,
        true,
    );
    if let Some(denied) = admission.refusal() {
        return Err(Glm5NextMetalError::MemoryAdmission {
            denied,
            budget_bytes: admission_budget(&admission),
            advice: CapacityAdvice::NotEvaluated,
        }
        .into());
    }
    Ok(priced_bytes)
}

/// An explicit numerical contract, never inferred from weight names or margins.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextDFlashVerification {
    ConfiguredPacked,
    Serial,
    /// Run configured packed verification, then replace it with serial rows from
    /// the checkpoint BEFORE any acceptance draws. Useful for explicit fallback.
    PackedThenSerial,
}

#[derive(Clone, Debug)]
pub struct Glm5NextDFlashOptions {
    pub prompt_tokens: usize,
    /// Includes the first carry and any terminal EOS (even though EOS is hidden
    /// from callback tokens). The last emitted token remains unconsumed.
    pub max_new_tokens: usize,
    pub stop_ids: Vec<i32>,
    pub verification: Glm5NextDFlashVerification,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextDFlashStop {
    Eos,
    TokenLimit,
    Capacity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Glm5NextDFlashCarry {
    Initial(i32),
    Correction(i32),
    Bonus(i32),
}

impl Glm5NextDFlashCarry {
    pub fn token(self) -> i32 {
        match self {
            Self::Initial(t) | Self::Correction(t) | Self::Bonus(t) => t,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Glm5NextDFlashCosts {
    /// Synchronous host elapsed durations, NOT GPU timestamp measurements.
    pub prefill: Duration,
    pub draft: Duration,
    pub checkpoint: Duration,
    pub verify: Duration,
    pub accept: Duration,
    /// Initial prefill packet only: host elapsed time copying its final logits
    /// to CPU (including readback allocation) and calling Sampler::sample.
    /// Excludes sampler cloning, readback deallocation, GPU head execution and
    /// feature append. Also included in `accept`, not additive.
    /// Zero for later packets or when no initial selection runs.
    pub selection: Duration,
    pub commit: Duration,
    pub append: Duration,
    pub total: Duration,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Glm5NextDFlashCounters {
    pub prefill_tokens: usize,
    pub attempts: usize,
    pub packets: usize,
    /// Attempted calls, including a call that fails or is subsequently cancelled.
    pub draft_calls: usize,
    /// Prepared candidate prefix, excluding the carry (possibly later cancelled).
    pub drafted_tokens: usize,
    pub verified_rows: usize,
    pub serial_verifications: usize,
    pub anchor_only_packets: usize,
    pub accepted_drafts: usize,
    pub retained_inputs: usize,
    pub replayed_inputs: usize,
    pub full_commits: usize,
    pub output_tokens: usize,
    pub callback_tokens: usize,
    /// Published RNG draws only; failed/cancelled attempts publish neither RNG.
    pub target_draws: usize,
    pub proposal_draws: usize,
    pub cancellations: usize,
    pub failures: usize,
}

pub struct Glm5NextDFlashPacket {
    /// Exact newly generated history, INCLUDING the selected terminal stop ID.
    /// Append this to GenerationResult.tokens/accounting, not to callbacks. The
    /// old carry is excluded; an initial packet contains its first carry here.
    pub emitted_tokens: Vec<i32>,
    /// Callback-ready tokens: accepted drafts followed by the correction/bonus,
    /// never the old carry, and never a stop ID. Emit once before the next step.
    pub tokens: Vec<i32>,
    /// Already included in `tokens` unless it is a suppressed stop ID. This is a
    /// continuation handle, not an additional token to emit. Ignore on stop.
    pub next_carry: Option<Glm5NextDFlashCarry>,
    pub stop_reason: Option<Glm5NextDFlashStop>,
    pub accepted_drafts: usize,
    pub retain_input_count: usize,
    pub verification_arithmetic: Option<Glm5NextVerificationArithmetic>,
    pub retained_arithmetic: Option<Glm5NextVerificationArithmetic>,
    /// True for no-draft packets, including bounded sampled tails.
    pub anchor_only: bool,
    pub costs: Glm5NextDFlashCosts,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    Fresh,
    Ready(i32),
    Running,
    Stopped,
    Failed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PacketPlan {
    rows: usize,
    sampled: bool,
}

fn packet_plan(
    block: usize,
    verify_capacity: usize,
    target_free: usize,
    draft_free: usize,
    remaining: usize,
    sampled: bool,
) -> PacketResult<PacketPlan> {
    let mut rows = block
        .min(verify_capacity)
        .min(target_free)
        .min(remaining.saturating_add(1));
    if remaining == 0 {
        rows = 0;
    }
    // The shared sampled API draws every slot of the loaded block. Do not draw
    // and discard a suffix or change the head's convolution/attention geometry.
    if sampled && rows < block {
        rows = rows.min(1);
    }
    if rows.min(remaining) > draft_free {
        return Err(Glm5NextDFlashError::Invalid(
            "draft context capacity exhausted",
        ));
    }
    Ok(PacketPlan {
        rows,
        sampled: sampled && rows > 1,
    })
}

fn cancel(check: &mut dyn FnMut() -> bool) -> PacketResult<()> {
    if check() {
        Err(Glm5NextDFlashError::Cancelled)
    } else {
        Ok(())
    }
}

fn packet_from_decision(
    decision: DFlashAcceptDecision,
    remaining: usize,
    stop_ids: &[i32],
) -> (Glm5NextDFlashPacket, usize) {
    let next_carry = decision.next_carry.map(|carry| match carry {
        DFlashNextCarry::Correction(t) => Glm5NextDFlashCarry::Correction(t),
        DFlashNextCarry::Bonus(t) => Glm5NextDFlashCarry::Bonus(t),
    });
    let emitted_tokens: Vec<i32> = decision
        .accepted_drafts
        .iter()
        .copied()
        .chain(next_carry.map(Glm5NextDFlashCarry::token))
        .collect();
    let outputs = emitted_tokens.len();
    let mut stop_reason = decision.stop_reason.map(|stop| match stop {
        DFlashAcceptStopReason::Eos => Glm5NextDFlashStop::Eos,
        DFlashAcceptStopReason::TokenLimit => Glm5NextDFlashStop::TokenLimit,
    });
    if let Some(carry) = next_carry {
        if stop_ids.contains(&carry.token()) {
            stop_reason = Some(Glm5NextDFlashStop::Eos);
        } else if outputs == remaining {
            stop_reason = Some(Glm5NextDFlashStop::TokenLimit);
        }
    }
    let tokens = emitted_tokens
        .iter()
        .copied()
        .filter(|token| !stop_ids.contains(token))
        .collect();
    (
        Glm5NextDFlashPacket {
            emitted_tokens,
            tokens,
            next_carry,
            stop_reason,
            accepted_drafts: decision.accepted_drafts.len(),
            retain_input_count: decision.retain_input_count,
            verification_arithmetic: None,
            retained_arithmetic: None,
            anchor_only: false,
            costs: Glm5NextDFlashCosts::default(),
        },
        outputs,
    )
}

/// One request, borrowing a fresh target session and the already loaded drafter.
/// No target weights are retained outside the short-lived draft decoder borrow.
/// Storage is separately admitted once; no full-prefix snapshots or CPU hidden
/// copies are used. Draft context is append-only, bounded by this request's cap.
///
/// Invalid calls before work leave the runner unchanged. Any execution error or
/// cancellation closes it permanently and invalidates all draft cache watermarks.
/// Before target commit, cancellation/completed validation failures attempt compact
/// rollback. Fatal command failures never attempt restoration or fallback. A
/// failed rollback, failed prompt prefill, or post-commit feature-publication error
/// poisons the target; drop the runner and reset/recreate the target, never retry
/// that packet. Cancellation is checked at synchronous host stage boundaries
/// and before each native prompt chunk, never inside GPU commands. No cancellation
/// is polled between target commit, feature append and sampler publication (one
/// publication unit).
pub struct Glm5NextDFlashRunner<'a, 'w> {
    ctx: &'a MetalContext,
    target: &'a mut Glm5NextSession<'w>,
    head: &'a MetalDFlashHead,
    options: Glm5NextDFlashOptions,
    draft: MetalDFlashSession,
    bank: Glm5NextRowBank,
    checkpoint: Glm5NextAppendCheckpoint,
    sampler: Sampler,
    proposal_sampler: Option<Sampler>,
    initial_target_draws: usize,
    initial_proposal_draws: usize,
    remaining: usize,
    seed_rows: usize,
    verify_capacity: usize,
    n_features: usize,
    priced_bytes: u64,
    phase: Phase,
    counters: Glm5NextDFlashCounters,
    #[cfg(test)]
    fail_context_append: bool,
}

impl<'a, 'w> Glm5NextDFlashRunner<'a, 'w> {
    /// Greedy: temperature-zero target sampler, no proposal sampler. Sampled:
    /// positive target temperature plus a separately seeded, temperature-only
    /// proposal sampler and a DFlash2 selector. No seed or filter is invented.
    pub fn new(
        ctx: &'a MetalContext,
        target: &'a mut Glm5NextSession<'w>,
        head: &'a MetalDFlashHead,
        options: Glm5NextDFlashOptions,
        sampler: Sampler,
        proposal_sampler: Option<Sampler>,
    ) -> PacketResult<Self> {
        Self::new_with_cpu_reserve(ctx, target, head, options, sampler, proposal_sampler, 0)
    }

    /// Admit the aggregate draft session, row bank and checkpoint, plus future
    /// control/transport CPU bytes and ONE standard safety reserve, before any
    /// request buffers are allocated. Resident target/head storage is already
    /// reflected in telemetry. Admission and all allocations share one lock;
    /// constituent banks do not perform independent reserve-zero admissions.
    pub fn new_with_cpu_reserve(
        ctx: &'a MetalContext,
        target: &'a mut Glm5NextSession<'w>,
        head: &'a MetalDFlashHead,
        options: Glm5NextDFlashOptions,
        sampler: Sampler,
        proposal_sampler: Option<Sampler>,
        additional_cpu_reserve: u64,
    ) -> PacketResult<Self> {
        if target.is_poisoned() || target.position() != 0 {
            return Err(Glm5NextDFlashError::Invalid(
                "runner requires a fresh healthy target",
            ));
        }
        let c = &target.weights.config;
        let block = head.config.block_size as usize;
        if options.prompt_tokens == 0
            || options.prompt_tokens > target.capacity
            || target.capacity > i32::MAX as usize
            || c.vocab_size > i32::MAX as u32
            || block == 0
            || head.config.hidden_size != c.hidden_size
            || head.target_layer_ids.is_empty()
            || head.config.n_target_features_layers as usize != head.target_layer_ids.len()
            || head.fc.shape
                != [
                    c.hidden_size as u64 * head.target_layer_ids.len() as u64,
                    c.hidden_size as u64,
                ]
            || options
                .stop_ids
                .iter()
                .any(|&id| id < 0 || id as u32 >= c.vocab_size)
            || head.config.mask_token_id < 0
            || head.config.mask_token_id as u32 >= c.vocab_size
        {
            return Err(Glm5NextDFlashError::Invalid(
                "prompt, head or target geometry mismatch",
            ));
        }
        if head.fc.buffer.device().registryID() != ctx.device.registryID()
            || target.weights.embedding.buffer.device().registryID() != ctx.device.registryID()
        {
            return Err(Glm5NextDFlashError::Invalid("head/target device mismatch"));
        }
        let sampled = sampler.config().temperature > 0.0;
        if sampled != proposal_sampler.is_some() || (sampled && head.selector.is_none()) {
            return Err(Glm5NextDFlashError::Invalid(
                "sampled mode requires a selector and proposal sampler",
            ));
        }
        if let Some(proposal) = &proposal_sampler {
            let p = proposal.config();
            if p.temperature <= 0.0 || p.top_k != 0 || p.top_p != 1.0 || p.min_p != 0.0 {
                return Err(Glm5NextDFlashError::Invalid(
                    "proposal sampler must be temperature-only",
                ));
            }
        }
        let verify_capacity = match options.verification {
            Glm5NextDFlashVerification::Serial => block,
            _ => block.min(target.packed.as_ref().map_or(0, |p| p.rows())),
        };
        if verify_capacity == 0 {
            return Err(Glm5NextDFlashError::Invalid(
                "configured packed verification needs packed scratch",
            ));
        }
        let (_, seed_rows) =
            dflash_capture_window_span(options.prompt_tokens, dflash_capture_window_limit(head));
        let ctx_capacity = seed_rows
            .checked_add(
                options
                    .max_new_tokens
                    .saturating_sub(1)
                    .min(target.capacity - options.prompt_tokens),
            )
            .ok_or(Glm5NextDFlashError::Invalid("draft capacity overflow"))?;
        let plan = Glm5NextRowBankPlan::new(
            c,
            &head.target_layer_ids,
            seed_rows.max(verify_capacity),
            verify_capacity,
        )?;
        let n_features = (c.hidden_size as usize)
            .checked_mul(head.target_layer_ids.len())
            .ok_or(Glm5NextDFlashError::Invalid("feature extent overflow"))?;
        let draft_bytes = MetalDFlashSession::priced_bytes(
            ctx,
            head,
            c.hidden_size as u64,
            c.vocab_size as u64,
            ctx_capacity,
        )?;
        let bank_bytes = plan.priced_bytes(&device_price(ctx))?;
        let checkpoint_bytes = Glm5NextAppendCheckpoint::priced_bytes_for(c, &device_price(ctx))?;
        let (draft, bank, checkpoint, priced_bytes) = {
            let _allocation = ctx.begin_allocation_transaction();
            let priced_bytes = admit_request(
                draft_bytes,
                bank_bytes,
                checkpoint_bytes,
                additional_cpu_reserve,
                ctx.memory_signals(),
            )?;
            let draft = MetalDFlashSession::fresh(
                ctx,
                head,
                c.hidden_size as u64,
                c.vocab_size as u64,
                ctx_capacity,
            )?;
            let bank = Glm5NextRowBank::allocate_admitted(ctx, target.weights, plan, bank_bytes)?;
            let checkpoint =
                Glm5NextAppendCheckpoint::allocate_admitted(ctx, target, checkpoint_bytes)?;
            // Recheck actual residency, including allocator pools, without
            // charging owned scratch twice or losing the future CPU allowance.
            admit_request(0, 0, 0, additional_cpu_reserve, ctx.memory_signals())?;
            (draft, bank, checkpoint, priced_bytes)
        };
        Ok(Self {
            ctx,
            target,
            head,
            remaining: options.max_new_tokens,
            options,
            draft,
            bank,
            checkpoint,
            initial_target_draws: sampler.draws(),
            initial_proposal_draws: proposal_sampler.as_ref().map_or(0, Sampler::draws),
            sampler,
            proposal_sampler,
            seed_rows,
            verify_capacity,
            n_features,
            priced_bytes,
            phase: Phase::Fresh,
            counters: Glm5NextDFlashCounters::default(),
            #[cfg(test)]
            fail_context_append: false,
        })
    }

    pub fn counters(&self) -> Glm5NextDFlashCounters {
        self.counters
    }
    /// Device-buffer prices only; excludes the admitted safety/CPU reserves and
    /// already resident target/head storage.
    pub fn priced_bytes(&self) -> u64 {
        self.priced_bytes
    }
    pub fn sampler(&self) -> &Sampler {
        &self.sampler
    }
    pub fn proposal_sampler(&self) -> Option<&Sampler> {
        self.proposal_sampler.as_ref()
    }
    pub fn target_position(&self) -> usize {
        self.target.position()
    }
    pub fn target_is_poisoned(&self) -> bool {
        self.target.is_poisoned()
    }
    pub fn remaining_output_tokens(&self) -> usize {
        self.remaining
    }

    /// Call if the output bridge fails partway through emitting a packet. Target
    /// state may already include tokens the client never received; neither retry
    /// that packet nor resume this target history. No GPU work is submitted.
    pub fn abort_output(&mut self) {
        self.target.poisoned = true;
        if self.phase != Phase::Failed {
            self.failed(&Glm5NextDFlashError::Closed);
        }
    }

    fn failed(&mut self, error: &Glm5NextDFlashError) {
        self.phase = Phase::Failed;
        self.draft.target_ctx_n = 0;
        self.draft.ctx_h_ready_n = 0;
        self.draft.kv_ctx_ready_n = 0;
        if matches!(error, Glm5NextDFlashError::Cancelled) {
            self.counters.cancellations += 1;
        } else {
            self.counters.failures += 1;
        }
    }

    fn publish(&mut self, packet: &Glm5NextDFlashPacket, outputs: usize) {
        debug_assert_eq!(outputs, packet.emitted_tokens.len());
        self.remaining -= outputs;
        self.counters.output_tokens += outputs;
        self.counters.callback_tokens += packet.tokens.len();
        self.counters.target_draws = self.sampler.draws() - self.initial_target_draws;
        self.counters.proposal_draws =
            self.proposal_sampler.as_ref().map_or(0, Sampler::draws) - self.initial_proposal_draws;
        self.phase = if packet.stop_reason.is_some() {
            Phase::Stopped
        } else {
            Phase::Ready(
                packet
                    .next_carry
                    .expect("nonterminal packet has a carry")
                    .token(),
            )
        };
    }

    /// Capture the actual prompt execution's trailing window and sample the first
    /// unconsumed carry. Zero output budget executes neither prompt nor sampler.
    pub fn prefill(
        &mut self,
        prompt: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> PacketResult<Glm5NextDFlashPacket> {
        if self.phase != Phase::Fresh {
            return Err(Glm5NextDFlashError::Closed);
        }
        if prompt.len() != self.options.prompt_tokens {
            return Err(Glm5NextDFlashError::Invalid(
                "prompt length differs from admission",
            ));
        }
        self.target.validate_request(prompt)?;
        if let Err(error) = cancel(cancelled) {
            self.failed(&error);
            return Err(error);
        }
        if self.remaining == 0 {
            let packet = empty_packet(Glm5NextDFlashStop::TokenLimit);
            self.publish(&packet, 0);
            return Ok(packet);
        }
        self.phase = Phase::Running;
        let result = self.prefill_inner(prompt, cancelled);
        match result {
            Ok(packet) => {
                self.publish(&packet, 1);
                Ok(packet)
            }
            Err(error) => {
                self.target.poisoned = true;
                self.failed(&error);
                Err(error)
            }
        }
    }

    fn prefill_inner(
        &mut self,
        prompt: &[u32],
        cancelled: &mut dyn FnMut() -> bool,
    ) -> PacketResult<Glm5NextDFlashPacket> {
        let total = Instant::now();
        let started = Instant::now();
        let base = self.target.position();
        let result = self.target.prefill_with_captures_and_checkpoint(
            self.ctx,
            prompt,
            &mut self.bank,
            &mut || {
                if cancelled() {
                    Err("GLM DFlash request cancelled".into())
                } else {
                    Ok(())
                }
            },
        );
        self.counters.prefill_tokens = self.target.position() - base;
        let rows = result.map_err(|error| match error {
            Glm5NextMetalError::Cancelled(_) => Glm5NextDFlashError::Cancelled,
            other => Glm5NextDFlashError::Target(other),
        })?;
        let mut costs = Glm5NextDFlashCosts {
            prefill: started.elapsed(),
            ..Default::default()
        };
        cancel(cancelled)?;
        let start = prompt.len() - self.seed_rows;
        let offset = (start - rows.feature_positions().start) * self.n_features;
        let features = rows
            .features()
            .expect("capture plan validated")
            .view_subrange(
                offset as u64,
                vec![(self.seed_rows * self.n_features) as u64],
            );
        let started = Instant::now();
        self.draft.append_target_ctx_columns_contiguous_now(
            self.ctx,
            &features,
            start as u32,
            self.seed_rows,
            self.n_features,
        )?;
        costs.append = started.elapsed();
        cancel(cancelled)?;
        let mut trial = self.sampler.clone();
        let started = Instant::now();
        let logits = read_f32(rows.logits().expect("logit plan validated"))?;
        let carry = trial.sample(&logits)?.token;
        costs.selection = started.elapsed();
        costs.accept = costs.selection;
        drop(logits);
        let stop_reason = if self.options.stop_ids.contains(&carry) {
            Some(Glm5NextDFlashStop::Eos)
        } else if self.remaining == 1 {
            Some(Glm5NextDFlashStop::TokenLimit)
        } else {
            None
        };
        self.sampler = trial;
        costs.total = total.elapsed();
        Ok(Glm5NextDFlashPacket {
            emitted_tokens: vec![carry],
            tokens: if self.options.stop_ids.contains(&carry) {
                vec![]
            } else {
                vec![carry]
            },
            next_carry: Some(Glm5NextDFlashCarry::Initial(carry)),
            stop_reason,
            accepted_drafts: 0,
            retain_input_count: 0,
            verification_arithmetic: None,
            retained_arithmetic: None,
            anchor_only: false,
            costs,
        })
    }

    /// `carry` must be the previous packet's already-emitted, unconsumed carry.
    /// Stop packets must not be stepped. Full-width sampled packets use the shared
    /// sparse selector; bounded sampled tails use an explicit anchor-only packet,
    /// with no proposal draws. Greedy candidates may be truncated to a ragged width.
    pub fn step(
        &mut self,
        carry: i32,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> PacketResult<Glm5NextDFlashPacket> {
        match self.phase {
            Phase::Ready(expected) if expected == carry => {}
            Phase::Ready(_) => {
                return Err(Glm5NextDFlashError::Invalid(
                    "carry differs from published packet",
                ));
            }
            _ => return Err(Glm5NextDFlashError::Closed),
        }
        if let Err(error) = cancel(cancelled) {
            self.failed(&error);
            return Err(error);
        }
        let plan = packet_plan(
            self.head.config.block_size as usize,
            self.verify_capacity,
            self.target.capacity - self.target.position(),
            self.draft.target_ctx_capacity - self.draft.target_ctx_n,
            self.remaining,
            self.proposal_sampler.is_some(),
        )?;
        if plan.rows == 0 {
            let packet = empty_packet(Glm5NextDFlashStop::Capacity);
            self.publish(&packet, 0);
            return Ok(packet);
        }
        self.phase = Phase::Running;
        self.counters.attempts += 1;
        let result = self.step_inner(carry, plan, cancelled);
        match result {
            Ok((packet, outputs)) => {
                self.counters.packets += 1;
                self.counters.anchor_only_packets += usize::from(packet.anchor_only);
                self.publish(&packet, outputs);
                Ok(packet)
            }
            Err(error) => {
                self.failed(&error);
                Err(error)
            }
        }
    }

    fn step_inner(
        &mut self,
        carry: i32,
        plan: PacketPlan,
        cancelled: &mut dyn FnMut() -> bool,
    ) -> PacketResult<(Glm5NextDFlashPacket, usize)> {
        let total = Instant::now();
        let mut costs = Glm5NextDFlashCosts::default();
        let mut trial = self.sampler.clone();
        let mut proposal_trial = self.proposal_sampler.clone();
        let mut ids = vec![carry];
        let mut proposals = None;
        if plan.rows > 1 {
            let started = Instant::now();
            self.counters.draft_calls += 1;
            // This weight borrow ends before beginning mutable target verification.
            let w = self.target.weights;
            let target = MetalDFlashDraftTarget {
                ctx: self.ctx,
                hidden_size: w.config.hidden_size,
                vocab_size: w.config.vocab_size,
                token_embedding: &w.embedding,
                output: &w.output,
            };
            let mut decoder = MetalDFlashDraftDecoder::new(target, self.head, &mut self.draft);
            if plan.sampled {
                let block = decoder.draft_block_sampled(
                    carry,
                    self.target.position() as u32,
                    proposal_trial.as_mut().expect("sampled plan validated"),
                )?;
                if block.draft_tokens.len() != plan.rows || block.proposals.len() + 1 != plan.rows {
                    return Err(Glm5NextDFlashError::Invalid(
                        "sampled draft block extent mismatch",
                    ));
                }
                ids.extend_from_slice(&block.draft_tokens[1..]);
                proposals = Some(block.proposals);
            } else {
                let block = decoder.draft_block(carry, self.target.position() as u32)?;
                if block.len() < plan.rows {
                    return Err(Glm5NextDFlashError::Invalid(
                        "greedy draft block extent mismatch",
                    ));
                }
                ids.extend_from_slice(&block[1..plan.rows]);
            }
            costs.draft = started.elapsed();
        }
        self.counters.drafted_tokens += ids.len() - 1;
        cancel(cancelled)?;
        let inputs = ids
            .iter()
            .map(|&id| {
                u32::try_from(id).map_err(|_| Glm5NextDFlashError::Invalid("negative draft token"))
            })
            .collect::<PacketResult<Vec<_>>>()?;
        let vocab = self.target.weights.config.vocab_size as usize;
        let started = Instant::now();
        let mut tx =
            self.target
                .begin_append(self.ctx, &mut self.checkpoint, &mut self.bank, &inputs)?;
        costs.checkpoint = started.elapsed();
        let mut target_committed = false;
        let result = (|| -> PacketResult<(Glm5NextDFlashPacket, usize)> {
            cancel(cancelled)?;
            let started = Instant::now();
            let (logits, arithmetic) = match self.options.verification {
                Glm5NextDFlashVerification::ConfiguredPacked => {
                    let rows = tx.verify_packed()?;
                    self.counters.verified_rows += inputs.len();
                    (rows.read_logits()?, rows.arithmetic())
                }
                Glm5NextDFlashVerification::Serial
                | Glm5NextDFlashVerification::PackedThenSerial => {
                    if self.options.verification == Glm5NextDFlashVerification::PackedThenSerial {
                        tx.verify_packed()?;
                        self.counters.verified_rows += inputs.len();
                        cancel(cancelled)?;
                    }
                    let rows = tx.verify_serial()?;
                    self.counters.verified_rows += inputs.len();
                    self.counters.serial_verifications += 1;
                    (rows.read_logits()?, rows.arithmetic())
                }
            };
            costs.verify = started.elapsed();
            cancel(cancelled)?;
            let started = Instant::now();
            let decision = decide_dflash_accept_prefix(
                &ids,
                &logits,
                vocab,
                proposals.as_deref(),
                &mut trial,
                self.remaining,
                &self.options.stop_ids,
            )?;
            costs.accept = started.elapsed();
            cancel(cancelled)?;
            let count = decision.retain_input_count;
            let started = Instant::now();
            let committed = tx.commit_retained_inputs(count)?;
            target_committed = true;
            costs.commit = started.elapsed();
            let retained_arithmetic = committed.retained_arithmetic();
            let started = Instant::now();
            if let Some(rows) = committed.rows() {
                let positions = rows.feature_positions();
                #[cfg(test)]
                let _append_failure = self
                    .fail_context_append
                    .then(crate::metal::fail_next_completion_check);
                self.draft.append_target_ctx_columns_contiguous_now(
                    self.ctx,
                    rows.features().expect("capture plan validated"),
                    positions.start as u32,
                    positions.len(),
                    self.n_features,
                )?;
            }
            costs.append = started.elapsed();
            let (mut packet, outputs) =
                packet_from_decision(decision, self.remaining, &self.options.stop_ids);
            packet.verification_arithmetic = Some(arithmetic);
            packet.retained_arithmetic = retained_arithmetic;
            packet.anchor_only = plan.rows == 1;
            self.counters.accepted_drafts += packet.accepted_drafts;
            self.counters.retained_inputs += count;
            if count == inputs.len() {
                self.counters.full_commits += 1;
            } else {
                self.counters.replayed_inputs += count;
            }
            self.sampler = trial;
            self.proposal_sampler = proposal_trial;
            costs.total = total.elapsed();
            packet.costs = costs;
            Ok((packet, outputs))
        })();
        let result = match result {
            Err(cause) if !target_committed && !tx.is_fatal() => match tx.rollback() {
                Ok(()) => Err(cause),
                Err(restore) => Err(Glm5NextDFlashError::Recovery {
                    cause: Box::new(cause),
                    restore,
                }),
            },
            other => other,
        };
        drop(tx);
        if result.is_err() && target_committed {
            // Target and draft publication are not jointly rewindable after commit.
            self.target.poisoned = true;
        }
        result
    }
}

fn empty_packet(stop: Glm5NextDFlashStop) -> Glm5NextDFlashPacket {
    Glm5NextDFlashPacket {
        emitted_tokens: vec![],
        tokens: vec![],
        next_carry: None,
        stop_reason: Some(stop),
        accepted_drafts: 0,
        retain_input_count: 0,
        verification_arithmetic: None,
        retained_arithmetic: None,
        anchor_only: false,
        costs: Glm5NextDFlashCosts::default(),
    }
}

impl Drop for Glm5NextDFlashRunner<'_, '_> {
    fn drop(&mut self) {
        if self.phase == Phase::Running {
            self.target.poisoned = true;
        }
    }
}

#[cfg(test)]
#[path = "tests/dflash.rs"]
mod tests;
