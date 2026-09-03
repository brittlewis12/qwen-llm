//! Prefill chunk decisions, admission pricing, and prefill request allocation.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PrefillChunkArg {
    Fixed(usize),
    Auto,
}

impl PrefillChunkArg {
    pub(crate) fn is_auto(self) -> bool {
        self == Self::Auto
    }

    pub(crate) fn validate(self) -> Result<()> {
        ensure!(
            !matches!(self, Self::Fixed(0)),
            "--prefill-chunk must be >= 1 or auto"
        );
        Ok(())
    }
}

impl FromStr for PrefillChunkArg {
    type Err = String;

    fn from_str(value: &str) -> std::result::Result<Self, Self::Err> {
        if value == "auto" {
            return Ok(Self::Auto);
        }
        value
            .parse::<usize>()
            .map(Self::Fixed)
            .map_err(|_| format!("expected a positive integer or auto, got {value:?}"))
    }
}

impl Serialize for PrefillChunkArg {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Fixed(value) => serializer.serialize_u64(*value as u64),
            Self::Auto => serializer.serialize_str("auto"),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PrefillChunkDecision {
    pub(crate) policy: &'static str,
    pub(crate) profile: Option<&'static str>,
    pub(crate) classification: &'static str,
    pub(crate) reason: &'static str,
    pub(crate) candidate: Option<usize>,
    pub(crate) selected: usize,
    pub(crate) validated_prompt_range: Option<[usize; 2]>,
    pub(crate) evidence_baseline_chunk: Option<usize>,
    pub(crate) baseline: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) detail: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) plan: Option<PrefillPlanDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) admission: Option<PrefillAdmissionDecision>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct AutoPrefillProfile {
    pub(crate) name: &'static str,
    pub(crate) outer_chunk: usize,
    pub(crate) query_heads: u64,
    pub(crate) gdn_overlay_bytes: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PrefillPlanAllocationDecision {
    pub(crate) name: &'static str,
    pub(crate) deferred: bool,
    pub(crate) logical_bytes: u64,
    pub(crate) priced_bytes: u64,
    pub(crate) alignment: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PrefillPlanDecision {
    pub(crate) block_size: u32,
    pub(crate) matrix_max_pos: u64,
    pub(crate) matrix_query_rows: u32,
    pub(crate) eager_allocation_count: usize,
    pub(crate) deferred_allocation_count: usize,
    pub(crate) eager_logical_bytes: u64,
    pub(crate) deferred_logical_bytes: u64,
    pub(crate) maximum_logical_bytes: u64,
    pub(crate) priced_upper_bytes: u64,
    pub(crate) overlay: PrefillScratchOverlayTimingStats,
    pub(crate) allocations: Vec<PrefillPlanAllocationDecision>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct PrefillMemorySignalsDecision {
    pub(crate) recommended_max_bytes: u64,
    pub(crate) current_allocated_bytes: u64,
    pub(crate) process_limit_remaining_bytes: Option<u64>,
    pub(crate) working_set_headroom_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct PrefillAdmissionDecision {
    pub(crate) current_allocated_before_sequence: u64,
    pub(crate) current_allocated_after_sequence: u64,
    pub(crate) sequence_allocation_delta_bytes: u64,
    pub(crate) transient_reserve_bytes: u64,
    pub(crate) reserve_bytes: u64,
    pub(crate) required_bytes: Option<u64>,
    pub(crate) signals: PrefillMemorySignalsDecision,
    pub(crate) evaluator_reason: &'static str,
    pub(crate) admitted: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct PrefillAttentionQueryStats {
    pub(crate) outer_chunk_rows: usize,
    pub(crate) query_rows: usize,
    pub(crate) tiled_layer_calls: u64,
    pub(crate) query_tile_calls: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct PrefillScratchOverlayTimingStats {
    pub(crate) backing_bytes: u64,
    pub(crate) attention_bytes: u64,
    pub(crate) gdn_bytes: u64,
    pub(crate) saved_bytes: u64,
}

pub(crate) const AUTO_CHUNK_PROMPT_MIN: usize = 8192;

pub(crate) const AUTO_CHUNK_PROMPT_MAX: usize = 16384;

pub(crate) const AUTO_CHUNK_QUERY_ROWS: usize = 1024;

pub(crate) const AUTO_CHUNK_TRANSIENT_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

pub(crate) const AUTO_CHUNK_BASELINE: &str = "legacy_outer_1024_matrix_max_pos_v1";

pub(crate) fn request_schema_version(
    prefill_chunk: PrefillChunkArg,
    prompt_lookup: bool,
    has_query_topology: bool,
    has_scratch_overlay: bool,
    sampled: bool,
    sampling_attribution: bool,
    sampled_structural: bool,
) -> u32 {
    if sampled_structural {
        12
    } else if sampling_attribution {
        11
    } else if sampled {
        10
    } else if prefill_chunk.is_auto() {
        9
    } else if prompt_lookup || has_query_topology || has_scratch_overlay {
        8
    } else {
        7
    }
}

pub(crate) fn auto_prefill_cache_safe(
    cache_entries: usize,
    cache_prefix_tokens: Option<usize>,
) -> bool {
    cache_entries == 0 && cache_prefix_tokens.is_none()
}

pub(crate) fn cache_prefix_needs_extension(
    configured_prefix: usize,
    restored_prefix: usize,
) -> bool {
    configured_prefix > restored_prefix
}

pub(crate) fn auto_prefill_profile(
    arch: Arch,
    base_model_name: Option<&str>,
    file_type: Option<u64>,
) -> Option<AutoPrefillProfile> {
    if arch.kind != ArchKind::Moe
        || arch.expert_count != 256
        || arch.expert_used_count != 8
        || arch.full_attention_interval != 4
        || arch.attn_head_dim != 256
        || arch.partial_rotary_factor != 0.25
        || arch.gdn_n_k_heads != 16
        || arch.gdn_head_dim != 128
        || arch.gdn_conv_kernel != 4
        || arch.mtp_n_hidden_layers != 0
        || file_type != Some(15)
    {
        return None;
    }
    match base_model_name {
        Some("Qwen3.6 35B A3B")
            if arch.n_layer == 40
                && arch.hidden_size == 2048
                && arch.n_q_heads == 16
                && arch.n_kv_heads == 2
                && arch.gdn_n_v_heads == 32
                && arch.expert_feed_forward_length == 512
                && arch.expert_shared_feed_forward_length == 512 =>
        {
            Some(AutoPrefillProfile {
                name: "qwen3.6-35b-a3b-filetype15",
                outer_chunk: 2048,
                query_heads: 16,
                gdn_overlay_bytes: 235_405_312,
            })
        }
        Some("Qwen3.5 122B A10B")
            if arch.n_layer == 48
                && arch.hidden_size == 3072
                && arch.n_q_heads == 32
                && arch.n_kv_heads == 2
                && arch.gdn_n_v_heads == 64
                && arch.expert_feed_forward_length == 1024
                && arch.expert_shared_feed_forward_length == 1024 =>
        {
            Some(AutoPrefillProfile {
                name: "qwen3.5-122b-a10b-filetype15",
                outer_chunk: 4096,
                query_heads: 32,
                gdn_overlay_bytes: 807_403_520,
            })
        }
        _ => None,
    }
}

pub(crate) fn baseline_prefill_chunk(prompt_tokens: usize) -> usize {
    1024.min(prompt_tokens.max(1))
}

pub(crate) fn auto_prefill_chunk_decision(
    profile: Option<AutoPrefillProfile>,
    prompt_tokens: usize,
    environment_override: bool,
    cache_safe: bool,
) -> PrefillChunkDecision {
    let baseline = baseline_prefill_chunk(prompt_tokens);
    let (classification, reason, selected, profile_name, candidate) = match profile {
        None => ("baseline", "profile_not_allowlisted", baseline, None, None),
        Some(profile) if prompt_tokens < AUTO_CHUNK_PROMPT_MIN => (
            "baseline",
            "prompt_below_validated_range",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) if prompt_tokens > AUTO_CHUNK_PROMPT_MAX => (
            "baseline",
            "prompt_above_memory_bounded_range",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) if environment_override => (
            "baseline",
            "prefill_environment_override_present",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) if !cache_safe => (
            "baseline",
            "prefix_cache_interaction_unvalidated",
            baseline,
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
        Some(profile) => (
            "candidate",
            "matched_validated_profile",
            profile.outer_chunk.min(prompt_tokens.max(1)),
            Some(profile.name),
            Some(profile.outer_chunk),
        ),
    };
    PrefillChunkDecision {
        policy: "moe_allowlist_v2",
        profile: profile_name,
        classification,
        reason,
        candidate,
        selected,
        validated_prompt_range: profile_name
            .map(|_| [AUTO_CHUNK_PROMPT_MIN, AUTO_CHUNK_PROMPT_MAX]),
        evidence_baseline_chunk: profile_name.map(|_| 1024),
        baseline: AUTO_CHUNK_BASELINE,
        detail: None,
        plan: None,
        admission: None,
    }
}

pub(crate) fn prefill_environment_override_present() -> bool {
    prefill_environment_override_present_in(std::env::vars_os().map(|(key, _)| key))
}

pub(crate) fn prefill_environment_override_present_in<I, K>(keys: I) -> bool
where
    I: IntoIterator<Item = K>,
    K: AsRef<std::ffi::OsStr>,
{
    keys.into_iter()
        .any(|key| is_prefill_environment_key(key.as_ref()))
}

pub(crate) fn is_prefill_environment_key(key: &std::ffi::OsStr) -> bool {
    key.as_encoded_bytes().starts_with(b"QWEN_PREFILL_")
}

pub(crate) fn checked_overlay_alignment(value: u64) -> Result<u64> {
    value
        .checked_add(255)
        .map(|aligned| aligned & !255)
        .context("prefill overlay alignment overflow")
}

pub(crate) fn expected_auto_prefill_overlay(
    profile: AutoPrefillProfile,
    matrix_max_pos: usize,
) -> Result<PrefillScratchOverlayStats> {
    let matrix_max_pos =
        u64::try_from(matrix_max_pos).context("matrix max pos does not fit u64")?;
    let query_rows = AUTO_CHUNK_QUERY_ROWS as u64;
    let score_bytes = 2u64
        .checked_mul(query_rows)
        .and_then(|value| value.checked_mul(profile.query_heads))
        .and_then(|value| value.checked_mul(matrix_max_pos))
        .context("prefill score overlay byte overflow")?;
    let ml_bytes = 8u64
        .checked_mul(query_rows)
        .and_then(|value| value.checked_mul(profile.query_heads))
        .and_then(|value| value.checked_mul(matrix_max_pos.div_ceil(64)))
        .context("prefill sidecar overlay byte overflow")?;
    let attention_bytes = checked_overlay_alignment(
        checked_overlay_alignment(score_bytes)?
            .checked_add(ml_bytes)
            .context("prefill attention overlay byte overflow")?,
    )?;
    let gdn_bytes = profile.gdn_overlay_bytes;
    Ok(PrefillScratchOverlayStats {
        backing_bytes: attention_bytes.max(gdn_bytes),
        attention_bytes,
        gdn_bytes,
        saved_bytes: attention_bytes.min(gdn_bytes),
    })
}

pub(crate) fn overlay_timing_stats(
    value: PrefillScratchOverlayStats,
) -> PrefillScratchOverlayTimingStats {
    PrefillScratchOverlayTimingStats {
        backing_bytes: value.backing_bytes,
        attention_bytes: value.attention_bytes,
        gdn_bytes: value.gdn_bytes,
        saved_bytes: value.saved_bytes,
    }
}

pub(crate) fn price_prefill_allocations(
    allocations: impl IntoIterator<Item = (&'static str, bool, u64)>,
    mut price: impl FnMut(u64) -> Result<MetalBufferSizeAndAlign>,
) -> Result<(Vec<PrefillPlanAllocationDecision>, u64, u64, u64)> {
    let mut rows = Vec::new();
    let mut eager_logical_bytes = 0u64;
    let mut deferred_logical_bytes = 0u64;
    let mut priced_upper_bytes = 0u64;
    for (name, deferred, logical_bytes) in allocations {
        let priced = price(logical_bytes)?;
        ensure!(
            priced.size > 0 && priced.size >= logical_bytes && priced.alignment.is_power_of_two(),
            "auto-prefill allocation pricing is invalid"
        );
        if deferred {
            deferred_logical_bytes = deferred_logical_bytes
                .checked_add(logical_bytes)
                .context("deferred prefill logical byte overflow")?;
        } else {
            eager_logical_bytes = eager_logical_bytes
                .checked_add(logical_bytes)
                .context("eager prefill logical byte overflow")?;
        }
        priced_upper_bytes = priced_upper_bytes
            .checked_add(priced.size)
            .context("priced prefill byte overflow")?;
        rows.push(PrefillPlanAllocationDecision {
            name,
            deferred,
            logical_bytes,
            priced_bytes: priced.size,
            alignment: priced.alignment,
        });
    }
    Ok((
        rows,
        eager_logical_bytes,
        deferred_logical_bytes,
        priced_upper_bytes,
    ))
}

pub(crate) fn validate_auto_prefill_plan_topology(
    profile: AutoPrefillProfile,
    prompt_tokens: usize,
    block_size: u32,
    matrix_max_pos: u64,
    matrix_query_rows: u32,
    overlay: Option<PrefillScratchOverlayStats>,
) -> Result<PrefillScratchOverlayStats> {
    let expected_matrix_max_pos = prompt_tokens.max(profile.outer_chunk);
    ensure!(
        block_size == u32::try_from(profile.outer_chunk)?
            && matrix_max_pos == u64::try_from(expected_matrix_max_pos)?
            && matrix_query_rows == u32::try_from(AUTO_CHUNK_QUERY_ROWS)?,
        "auto-prefill candidate plan geometry drifted"
    );
    let expected_overlay = expected_auto_prefill_overlay(profile, expected_matrix_max_pos)?;
    let overlay = overlay.context("auto-prefill candidate plan has no scratch overlay")?;
    ensure!(
        overlay == expected_overlay,
        "auto-prefill candidate overlay geometry drifted"
    );
    Ok(overlay)
}

pub(crate) fn price_prefill_plan(
    ctx: &MetalContext,
    profile: AutoPrefillProfile,
    prompt_tokens: usize,
    plan: &PrefillScratchPlan,
) -> Result<PrefillPlanDecision> {
    let overlay = validate_auto_prefill_plan_topology(
        profile,
        prompt_tokens,
        plan.block_size(),
        plan.matrix_max_pos(),
        plan.matrix_query_rows(),
        plan.overlay(),
    )?;

    plan.allocation_count()
        .checked_add(plan.deferred_allocations().len())
        .context("auto-prefill allocation count overflow")?;
    let plan_allocations = plan
        .allocations()
        .iter()
        .map(|allocation| (allocation.name(), false, allocation.logical_bytes()))
        .chain(
            plan.deferred_allocations()
                .iter()
                .map(|allocation| (allocation.name(), true, allocation.logical_bytes())),
        );
    let (allocations, eager_logical_bytes, deferred_logical_bytes, priced_upper_bytes) =
        price_prefill_allocations(plan_allocations, |logical_bytes| {
            Ok(ctx.shared_buffer_size_and_align(logical_bytes)?)
        })?;
    let maximum_logical_bytes = plan.maximum_logical_bytes()?;
    ensure!(
        eager_logical_bytes.checked_add(deferred_logical_bytes) == Some(maximum_logical_bytes),
        "auto-prefill logical byte totals do not reconcile"
    );
    let independently_priced =
        plan.priced_upper_bound(|bytes| Ok(ctx.shared_buffer_size_and_align(bytes)?.size))?;
    ensure!(
        independently_priced == priced_upper_bytes,
        "auto-prefill priced byte totals do not reconcile"
    );
    Ok(PrefillPlanDecision {
        block_size: plan.block_size(),
        matrix_max_pos: plan.matrix_max_pos(),
        matrix_query_rows: plan.matrix_query_rows(),
        eager_allocation_count: plan.allocation_count(),
        deferred_allocation_count: plan.deferred_allocations().len(),
        eager_logical_bytes,
        deferred_logical_bytes,
        maximum_logical_bytes,
        priced_upper_bytes,
        overlay: overlay_timing_stats(overlay),
        allocations,
    })
}

pub(crate) fn prefill_memory_signals_decision(
    signals: MetalMemorySignals,
    admission: MetalMemoryAdmission,
) -> PrefillMemorySignalsDecision {
    PrefillMemorySignalsDecision {
        recommended_max_bytes: signals.recommended_max_bytes,
        current_allocated_bytes: signals.current_allocated_bytes,
        process_limit_remaining_bytes: signals.process_limit_remaining_bytes,
        working_set_headroom_bytes: admission.working_set_headroom_bytes,
    }
}

pub(crate) fn auto_prefill_reserve(
    before_sequence: u64,
    after_sequence: u64,
) -> std::result::Result<(u64, u64), &'static str> {
    let delta = after_sequence
        .checked_sub(before_sequence)
        .filter(|&delta| delta > 0)
        .ok_or("sequence_allocation_signal_invalid")?;
    let reserve = delta
        .checked_add(AUTO_CHUNK_TRANSIENT_RESERVE_BYTES)
        .ok_or("candidate_reserve_overflow")?;
    Ok((delta, reserve))
}

pub(crate) struct AllocatedPrefillRequestState<
    Scratch = MetalDFlashLayerMajorScratch,
    State = Sequence,
> {
    pub(crate) chunk: usize,
    pub(crate) decision: Option<PrefillChunkDecision>,
    pub(crate) scratch: Scratch,
    pub(crate) sequence: State,
    pub(crate) scratch_allocation_ms: f64,
    pub(crate) sequence_allocation_ms: f64,
    pub(crate) after_scratch_allocated: u64,
    pub(crate) after_sequence_allocated: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum CandidatePlanFailure {
    Unavailable(String),
    InvalidOrUnpriceable(String),
}

pub(crate) trait PrefillRequestAllocator {
    type Scratch;
    type Sequence;
    type Plan;

    fn current_allocated_size(&mut self) -> u64;
    fn allocate_legacy_scratch(
        &mut self,
        chunk: usize,
        prompt_tokens: usize,
    ) -> Result<Self::Scratch>;
    fn allocate_sequence(&mut self, capacity: usize) -> Result<Self::Sequence>;
    fn build_candidate_plan(
        &mut self,
        profile: AutoPrefillProfile,
        prompt_tokens: usize,
    ) -> std::result::Result<(Self::Plan, PrefillPlanDecision), CandidatePlanFailure>;
    fn memory_signals(&mut self) -> MetalMemorySignals;
    fn allocate_candidate_scratch(&mut self, plan: Self::Plan) -> Result<Self::Scratch>;
}

pub(crate) struct MetalPrefillRequestAllocator<'a> {
    pub(crate) loaded: &'a LoadedModel,
}

pub(crate) fn allocate_legacy_prefill_scratch(
    loaded: &LoadedModel,
    chunk: usize,
    prompt_tokens: usize,
) -> Result<MetalDFlashLayerMajorScratch> {
    MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        u32::try_from(chunk).context("prefill chunk does not fit u32")?,
        prompt_tokens.max(chunk),
    )
    .context("allocate legacy prefill scratch")
}

impl PrefillRequestAllocator for MetalPrefillRequestAllocator<'_> {
    type Scratch = MetalDFlashLayerMajorScratch;
    type Sequence = Sequence;
    type Plan = PrefillScratchPlan;

    fn current_allocated_size(&mut self) -> u64 {
        self.loaded.context().current_allocated_size()
    }

    fn allocate_legacy_scratch(
        &mut self,
        chunk: usize,
        prompt_tokens: usize,
    ) -> Result<Self::Scratch> {
        allocate_legacy_prefill_scratch(self.loaded, chunk, prompt_tokens)
    }

    fn allocate_sequence(&mut self, capacity: usize) -> Result<Self::Sequence> {
        self.loaded
            .create_sequence(SequenceConfig::new(capacity))
            .map_err(anyhow::Error::from)
    }

    fn build_candidate_plan(
        &mut self,
        profile: AutoPrefillProfile,
        prompt_tokens: usize,
    ) -> std::result::Result<(Self::Plan, PrefillPlanDecision), CandidatePlanFailure> {
        let block_size = u32::try_from(profile.outer_chunk)
            .map_err(|error| CandidatePlanFailure::Unavailable(error.to_string()))?;
        let matrix_max_pos = prompt_tokens.max(profile.outer_chunk);
        let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
            self.loaded.metal_model(),
            block_size,
            matrix_max_pos,
            PrefillScratchConfig {
                matrix_query_cap: Some(AUTO_CHUNK_QUERY_ROWS),
            },
        )
        .map_err(|error| CandidatePlanFailure::Unavailable(error.to_string()))?;
        let decision = price_prefill_plan(self.loaded.context(), profile, prompt_tokens, &plan)
            .map_err(|error| CandidatePlanFailure::InvalidOrUnpriceable(error.to_string()))?;
        Ok((plan, decision))
    }

    fn memory_signals(&mut self) -> MetalMemorySignals {
        self.loaded.context().memory_signals()
    }

    fn allocate_candidate_scratch(&mut self, plan: Self::Plan) -> Result<Self::Scratch> {
        MetalDFlashLayerMajorScratch::fresh_prefill_from_plan(
            self.loaded.context(),
            self.loaded.metal_model(),
            plan,
        )
        .context("allocate admitted auto-prefill scratch")
    }
}

pub(crate) fn allocate_scratch_then_sequence<A: PrefillRequestAllocator>(
    allocator: &mut A,
    capacity: usize,
    chunk: usize,
    prompt_tokens: usize,
    decision: Option<PrefillChunkDecision>,
) -> Result<AllocatedPrefillRequestState<A::Scratch, A::Sequence>> {
    let scratch_t0 = Instant::now();
    let scratch = allocator.allocate_legacy_scratch(chunk, prompt_tokens)?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = allocator.current_allocated_size();
    let sequence_t0 = Instant::now();
    let sequence = allocator.allocate_sequence(capacity)?;
    let sequence_allocation_ms = sequence_t0.elapsed().as_secs_f64() * 1e3;
    let after_sequence_allocated = allocator.current_allocated_size();
    Ok(AllocatedPrefillRequestState {
        chunk,
        decision,
        scratch,
        sequence,
        scratch_allocation_ms,
        sequence_allocation_ms,
        after_scratch_allocated,
        after_sequence_allocated,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn allocate_legacy_after_sequence<A: PrefillRequestAllocator>(
    allocator: &mut A,
    prompt_tokens: usize,
    mut decision: PrefillChunkDecision,
    reason: &'static str,
    detail: Option<String>,
    sequence: A::Sequence,
    sequence_allocation_ms: f64,
    after_sequence_allocated: u64,
) -> Result<AllocatedPrefillRequestState<A::Scratch, A::Sequence>> {
    let chunk = baseline_prefill_chunk(prompt_tokens);
    decision.classification = "baseline";
    decision.reason = reason;
    decision.selected = chunk;
    decision.detail = detail;
    let scratch_t0 = Instant::now();
    let scratch = allocator.allocate_legacy_scratch(chunk, prompt_tokens)?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = allocator.current_allocated_size();
    Ok(AllocatedPrefillRequestState {
        chunk,
        decision: Some(decision),
        scratch,
        sequence,
        scratch_allocation_ms,
        sequence_allocation_ms,
        after_scratch_allocated,
        after_sequence_allocated,
    })
}

pub(crate) fn allocate_prefill_request_state_with<A: PrefillRequestAllocator>(
    allocator: &mut A,
    requested: PrefillChunkArg,
    prompt_tokens: usize,
    capacity: usize,
    cache_safe: bool,
    profile: Option<AutoPrefillProfile>,
    environment_override: bool,
) -> Result<AllocatedPrefillRequestState<A::Scratch, A::Sequence>> {
    if let PrefillChunkArg::Fixed(requested) = requested {
        let chunk = requested.min(prompt_tokens.max(1));
        return allocate_scratch_then_sequence(allocator, capacity, chunk, prompt_tokens, None);
    }

    let mut decision =
        auto_prefill_chunk_decision(profile, prompt_tokens, environment_override, cache_safe);
    if decision.classification != "candidate" {
        return allocate_scratch_then_sequence(
            allocator,
            capacity,
            decision.selected,
            prompt_tokens,
            Some(decision),
        );
    }
    let profile = profile.context("auto-prefill candidate is missing its profile")?;

    let current_allocated_before_sequence = allocator.current_allocated_size();
    let sequence_t0 = Instant::now();
    let sequence = allocator.allocate_sequence(capacity)?;
    let sequence_allocation_ms = sequence_t0.elapsed().as_secs_f64() * 1e3;
    let current_allocated_after_sequence = allocator.current_allocated_size();
    let (sequence_allocation_delta_bytes, reserve_bytes) = match auto_prefill_reserve(
        current_allocated_before_sequence,
        current_allocated_after_sequence,
    ) {
        Ok(value) => value,
        Err(reason) => {
            return allocate_legacy_after_sequence(
                allocator,
                prompt_tokens,
                decision,
                reason,
                None,
                sequence,
                sequence_allocation_ms,
                current_allocated_after_sequence,
            );
        }
    };

    let (plan, plan_decision) = match allocator.build_candidate_plan(profile, prompt_tokens) {
        Ok(value) => value,
        Err(CandidatePlanFailure::Unavailable(detail)) => {
            return allocate_legacy_after_sequence(
                allocator,
                prompt_tokens,
                decision,
                "candidate_plan_unavailable",
                Some(detail),
                sequence,
                sequence_allocation_ms,
                current_allocated_after_sequence,
            );
        }
        Err(CandidatePlanFailure::InvalidOrUnpriceable(detail)) => {
            return allocate_legacy_after_sequence(
                allocator,
                prompt_tokens,
                decision,
                "candidate_plan_invalid_or_unpriceable",
                Some(detail),
                sequence,
                sequence_allocation_ms,
                current_allocated_after_sequence,
            );
        }
    };
    decision.plan = Some(plan_decision.clone());
    let signals = allocator.memory_signals();
    let admission = evaluate_metal_memory_admission(
        plan_decision.priced_upper_bytes,
        reserve_bytes,
        signals,
        true,
    );
    decision.admission = Some(PrefillAdmissionDecision {
        current_allocated_before_sequence,
        current_allocated_after_sequence,
        sequence_allocation_delta_bytes,
        transient_reserve_bytes: AUTO_CHUNK_TRANSIENT_RESERVE_BYTES,
        reserve_bytes,
        required_bytes: admission.required_bytes,
        signals: prefill_memory_signals_decision(signals, admission),
        evaluator_reason: admission.reason.as_str(),
        admitted: admission.admitted,
    });
    if !admission.admitted {
        let reason = if admission.required_bytes.is_none() {
            "candidate_required_bytes_overflow"
        } else {
            "memory_admission_denied"
        };
        return allocate_legacy_after_sequence(
            allocator,
            prompt_tokens,
            decision,
            reason,
            None,
            sequence,
            sequence_allocation_ms,
            current_allocated_after_sequence,
        );
    }

    decision.reason = "admitted";
    let scratch_t0 = Instant::now();
    let scratch = allocator.allocate_candidate_scratch(plan)?;
    let scratch_allocation_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
    let after_scratch_allocated = allocator.current_allocated_size();
    Ok(AllocatedPrefillRequestState {
        chunk: profile.outer_chunk,
        decision: Some(decision),
        scratch,
        sequence,
        scratch_allocation_ms,
        sequence_allocation_ms,
        after_scratch_allocated,
        after_sequence_allocated: current_allocated_after_sequence,
    })
}

pub(crate) fn allocate_prefill_request_state(
    loaded: &LoadedModel,
    requested: PrefillChunkArg,
    prompt_tokens: usize,
    capacity: usize,
    cache_safe: bool,
) -> Result<AllocatedPrefillRequestState> {
    let profile = auto_prefill_profile(
        loaded.arch(),
        loaded.gguf().get_str("general.base_model.0.name"),
        loaded.gguf().get_u64("general.file_type"),
    );
    let mut allocator = MetalPrefillRequestAllocator { loaded };
    allocate_prefill_request_state_with(
        &mut allocator,
        requested,
        prompt_tokens,
        capacity,
        cache_safe,
        profile,
        prefill_environment_override_present(),
    )
}

pub(crate) fn report_prefill_chunk_decision(
    decision: Option<&PrefillChunkDecision>,
    prompt_tokens: usize,
) {
    let Some(decision) = decision else {
        return;
    };
    eprintln!(
        "prefill_chunk: policy={} profile={} prompt_tokens={} selected={} reason={}",
        decision.policy,
        decision.profile.unwrap_or("none"),
        prompt_tokens,
        decision.selected,
        decision.reason,
    );
}
