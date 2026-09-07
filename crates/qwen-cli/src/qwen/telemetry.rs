//! Request timing, sampling attribution, request-stats records, fingerprints, and JSONL sinks.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StopReason {
    Eos,
    TokenLimit,
}

impl StopReason {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Eos => "eos",
            Self::TokenLimit => "token_limit",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct SamplingTelemetry {
    pub(crate) algorithm_version: u32,
    pub(crate) temperature: f32,
    pub(crate) top_k: usize,
    pub(crate) top_p: f32,
    pub(crate) min_p: f32,
    pub(crate) effective_seed: u64,
    pub(crate) draws: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SampledStructuralTelemetry {
    pub(crate) version: u32,
    pub(crate) algorithm_version: u32,
    pub(crate) path: &'static str,
    pub(crate) prompt_owned_bounded_calls: u64,
    pub(crate) borrowed_transition_calls: u64,
    pub(crate) resident_head_wait_calls: u64,
    pub(crate) validated_shared_row_calls: u64,
    pub(crate) fallback_calls: u64,
    pub(crate) input_logits_total: u64,
    pub(crate) input_logits_min: u64,
    pub(crate) input_logits_max: u64,
    pub(crate) retained_top_k_total: u64,
    pub(crate) retained_top_k_min: u64,
    pub(crate) retained_top_k_max: u64,
    pub(crate) max_heap_len: u64,
    pub(crate) max_heap_capacity: u64,
    pub(crate) full_candidate_vector_allocations: u64,
    pub(crate) transition_logits_copy_bytes: u64,
    pub(crate) extra_command_buffers: u64,
    pub(crate) gpu_sampling_dispatches: u64,
}

impl Default for SampledStructuralTelemetry {
    fn default() -> Self {
        Self {
            version: 1,
            algorithm_version: SAMPLER_ALGORITHM_VERSION,
            path: "bounded_topk_borrowed_transitions",
            prompt_owned_bounded_calls: 0,
            borrowed_transition_calls: 0,
            resident_head_wait_calls: 0,
            validated_shared_row_calls: 0,
            fallback_calls: 0,
            input_logits_total: 0,
            input_logits_min: 0,
            input_logits_max: 0,
            retained_top_k_total: 0,
            retained_top_k_min: 0,
            retained_top_k_max: 0,
            max_heap_len: 0,
            max_heap_capacity: 0,
            full_candidate_vector_allocations: 0,
            transition_logits_copy_bytes: 0,
            extra_command_buffers: 0,
            gpu_sampling_dispatches: 0,
        }
    }
}

impl SampledStructuralTelemetry {
    pub(crate) fn record_bounded(
        &mut self,
        evidence: BoundedTopKEvidence,
        prompt: bool,
    ) -> Result<()> {
        let calls = self
            .prompt_owned_bounded_calls
            .checked_add(self.borrowed_transition_calls)
            .context("sampled structural call count overflow")?;
        if prompt {
            self.prompt_owned_bounded_calls = self
                .prompt_owned_bounded_calls
                .checked_add(1)
                .context("prompt bounded-call count overflow")?;
        } else {
            self.borrowed_transition_calls = self
                .borrowed_transition_calls
                .checked_add(1)
                .context("borrowed transition count overflow")?;
        }
        if !evidence.used_bounded_path {
            self.fallback_calls = self
                .fallback_calls
                .checked_add(1)
                .context("sampled structural fallback count overflow")?;
        }
        let input = u64::try_from(evidence.input_logits).context("input logits do not fit u64")?;
        let retained =
            u64::try_from(evidence.retained_top_k).context("retained top-k does not fit u64")?;
        let heap_len =
            u64::try_from(evidence.max_heap_len).context("heap length does not fit u64")?;
        let heap_capacity =
            u64::try_from(evidence.heap_capacity).context("heap capacity does not fit u64")?;
        self.input_logits_total = self
            .input_logits_total
            .checked_add(input)
            .context("input logits total overflow")?;
        self.retained_top_k_total = self
            .retained_top_k_total
            .checked_add(retained)
            .context("retained top-k total overflow")?;
        if calls == 0 {
            self.input_logits_min = input;
            self.input_logits_max = input;
            self.retained_top_k_min = retained;
            self.retained_top_k_max = retained;
        } else {
            self.input_logits_min = self.input_logits_min.min(input);
            self.input_logits_max = self.input_logits_max.max(input);
            self.retained_top_k_min = self.retained_top_k_min.min(retained);
            self.retained_top_k_max = self.retained_top_k_max.max(retained);
        }
        self.max_heap_len = self.max_heap_len.max(heap_len);
        self.max_heap_capacity = self.max_heap_capacity.max(heap_capacity);
        Ok(())
    }

    pub(crate) fn record_prompt(&mut self, evidence: BoundedTopKEvidence) -> Result<()> {
        self.record_bounded(evidence, true)
    }

    pub(crate) fn record_transition(
        &mut self,
        bounded: BoundedTopKEvidence,
        row: StructuralRowEvidence,
    ) -> Result<()> {
        self.record_bounded(bounded, false)?;
        self.resident_head_wait_calls = self
            .resident_head_wait_calls
            .checked_add(row.resident_head_wait_calls)
            .context("resident-head wait count overflow")?;
        self.validated_shared_row_calls = self
            .validated_shared_row_calls
            .checked_add(row.validated_shared_row_calls)
            .context("validated Shared-row count overflow")?;
        self.transition_logits_copy_bytes = self
            .transition_logits_copy_bytes
            .checked_add(row.transition_logits_copy_bytes)
            .context("transition logits-copy byte count overflow")?;
        self.extra_command_buffers = self
            .extra_command_buffers
            .checked_add(row.extra_command_buffers)
            .context("extra command-buffer count overflow")?;
        self.gpu_sampling_dispatches = self
            .gpu_sampling_dispatches
            .checked_add(row.gpu_sampling_dispatches)
            .context("GPU sampling-dispatch count overflow")?;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
pub(crate) struct CountSummary {
    pub(crate) total: u64,
    pub(crate) min: u64,
    pub(crate) max: u64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SamplingClockProbe {
    pub(crate) batches: u32,
    pub(crate) iterations_per_batch: u64,
    pub(crate) pair_ns: [f64; 7],
    pub(crate) upper_pair_ns: f64,
    pub(crate) new_timer_spans: u64,
    pub(crate) observer_upper_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SamplerAttribution {
    pub(crate) calls: u64,
    pub(crate) timer_spans: u64,
    pub(crate) input_logits_total: u64,
    pub(crate) input_logits_min: u64,
    pub(crate) input_logits_max: u64,
    pub(crate) wall_ms: f64,
    pub(crate) shape_validation_ms: f64,
    pub(crate) candidate_alloc_ms: f64,
    pub(crate) candidate_fill_ms: f64,
    pub(crate) top_k_order_ms: f64,
    pub(crate) min_p_ms: f64,
    pub(crate) positive_infinity_ms: f64,
    pub(crate) temperature_scale_ms: f64,
    pub(crate) probability_weights_ms: f64,
    pub(crate) top_p_ms: f64,
    pub(crate) categorical_ms: f64,
    pub(crate) residual_ms: f64,
    pub(crate) candidate_capacity_bytes_total: u64,
    pub(crate) candidate_capacity_bytes_peak: u64,
    pub(crate) probability_capacity_bytes_total: u64,
    pub(crate) probability_capacity_bytes_peak: u64,
    pub(crate) after_top_k: CountSummary,
    pub(crate) after_min_p: CountSummary,
    pub(crate) after_positive_infinity: CountSummary,
    pub(crate) after_top_p: CountSummary,
    pub(crate) candidate_index: CountSummary,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct TransitionAttribution {
    pub(crate) calls: u64,
    pub(crate) new_timer_spans: u64,
    pub(crate) logits_bytes_per_call: u64,
    pub(crate) logits_bytes_total: u64,
    pub(crate) outer_wall_ms: f64,
    pub(crate) inner_wall_ms: f64,
    pub(crate) cpu_encode_ms: f64,
    pub(crate) completion_wait_ms: f64,
    pub(crate) gpu_ms_nested: f64,
    pub(crate) logits_alloc_zero_ms: f64,
    pub(crate) logits_copy_ms: f64,
    pub(crate) inner_residual_ms: f64,
    pub(crate) outer_wrapper_advance_ms: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct SamplingAttributionBounds {
    pub(crate) observer_upper_ms: f64,
    pub(crate) workspace_raw_ms: f64,
    pub(crate) workspace_adjusted_ms: f64,
    pub(crate) workspace_adjusted_fraction: f64,
    pub(crate) borrowed_raw_ms: f64,
    pub(crate) borrowed_adjusted_ms: f64,
    pub(crate) borrowed_adjusted_fraction: f64,
    pub(crate) combined_raw_ms: f64,
    pub(crate) combined_adjusted_ms: f64,
    pub(crate) combined_adjusted_fraction: f64,
    pub(crate) structural_raw_ms: f64,
    pub(crate) structural_adjusted_ms: f64,
    pub(crate) structural_adjusted_fraction: f64,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct SamplingAttribution {
    pub(crate) version: u32,
    pub(crate) prompt_token_ids_sha256: String,
    pub(crate) clock_probe: SamplingClockProbe,
    pub(crate) sampler: SamplerAttribution,
    pub(crate) transitions: TransitionAttribution,
    pub(crate) bounds: SamplingAttributionBounds,
}

#[derive(Debug, Default)]
pub(crate) struct CountAccumulator {
    pub(crate) total: u64,
    pub(crate) min: Option<u64>,
    pub(crate) max: u64,
}

impl CountAccumulator {
    pub(crate) fn record(&mut self, value: usize, label: &str) -> Result<()> {
        let value = u64::try_from(value).with_context(|| format!("{label} does not fit u64"))?;
        self.total = self
            .total
            .checked_add(value)
            .with_context(|| format!("{label} total overflow"))?;
        self.min = Some(self.min.map_or(value, |current| current.min(value)));
        self.max = self.max.max(value);
        Ok(())
    }

    pub(crate) fn finish(self) -> CountSummary {
        CountSummary {
            total: self.total,
            min: self.min.unwrap_or(0),
            max: self.max,
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct SamplerAttributionAccumulator {
    pub(crate) calls: u64,
    pub(crate) timer_spans: u64,
    pub(crate) input_logits: CountAccumulator,
    pub(crate) wall_ms: f64,
    pub(crate) shape_validation_ms: f64,
    pub(crate) candidate_alloc_ms: f64,
    pub(crate) candidate_fill_ms: f64,
    pub(crate) top_k_order_ms: f64,
    pub(crate) min_p_ms: f64,
    pub(crate) positive_infinity_ms: f64,
    pub(crate) temperature_scale_ms: f64,
    pub(crate) probability_weights_ms: f64,
    pub(crate) top_p_ms: f64,
    pub(crate) categorical_ms: f64,
    pub(crate) residual_ms: f64,
    pub(crate) candidate_capacity_bytes_total: u64,
    pub(crate) candidate_capacity_bytes_peak: u64,
    pub(crate) probability_capacity_bytes_total: u64,
    pub(crate) probability_capacity_bytes_peak: u64,
    pub(crate) after_top_k: CountAccumulator,
    pub(crate) after_min_p: CountAccumulator,
    pub(crate) after_positive_infinity: CountAccumulator,
    pub(crate) after_top_p: CountAccumulator,
    pub(crate) candidate_index: CountAccumulator,
}

impl SamplerAttributionAccumulator {
    pub(crate) fn record(&mut self, profile: SamplingPhaseProfile) -> Result<()> {
        ensure!(
            profile.input_logits > 0
                && profile.after_top_k > 0
                && profile.after_top_k <= profile.input_logits
                && profile.after_min_p > 0
                && profile.after_min_p <= profile.after_top_k
                && profile.after_positive_infinity > 0
                && profile.after_positive_infinity <= profile.after_min_p
                && profile.after_top_p > 0
                && profile.after_top_p <= profile.after_positive_infinity
                && profile.candidate_index < profile.after_top_p,
            "profiled sampler support accounting is invalid"
        );
        self.calls = self.calls.checked_add(1).context("sampler call overflow")?;
        self.timer_spans = self
            .timer_spans
            .checked_add(u64::from(profile.timer_spans))
            .context("sampler timer-span overflow")?;
        self.input_logits
            .record(profile.input_logits, "input logits")?;
        self.wall_ms += profile.total_ms;
        self.shape_validation_ms += profile.shape_validation_ms;
        self.candidate_alloc_ms += profile.candidate_alloc_ms;
        self.candidate_fill_ms += profile.candidate_fill_ms;
        self.top_k_order_ms += profile.top_k_order_ms;
        self.min_p_ms += profile.min_p_ms;
        self.positive_infinity_ms += profile.positive_infinity_ms;
        self.temperature_scale_ms += profile.temperature_scale_ms;
        self.probability_weights_ms += profile.probability_weights_ms;
        self.top_p_ms += profile.top_p_ms;
        self.categorical_ms += profile.categorical_ms;
        self.residual_ms += profile.residual_ms;

        let candidate_bytes = u64::try_from(profile.candidate_capacity_bytes)
            .context("candidate capacity bytes do not fit u64")?;
        self.candidate_capacity_bytes_total = self
            .candidate_capacity_bytes_total
            .checked_add(candidate_bytes)
            .context("candidate capacity-byte total overflow")?;
        self.candidate_capacity_bytes_peak =
            self.candidate_capacity_bytes_peak.max(candidate_bytes);
        let probability_bytes = u64::try_from(profile.probability_capacity_bytes)
            .context("probability capacity bytes do not fit u64")?;
        self.probability_capacity_bytes_total = self
            .probability_capacity_bytes_total
            .checked_add(probability_bytes)
            .context("probability capacity-byte total overflow")?;
        self.probability_capacity_bytes_peak =
            self.probability_capacity_bytes_peak.max(probability_bytes);

        self.after_top_k
            .record(profile.after_top_k, "after top-k")?;
        self.after_min_p
            .record(profile.after_min_p, "after min-p")?;
        self.after_positive_infinity.record(
            profile.after_positive_infinity,
            "after positive-infinity filter",
        )?;
        self.after_top_p
            .record(profile.after_top_p, "after top-p")?;
        self.candidate_index
            .record(profile.candidate_index, "candidate index")?;
        Ok(())
    }

    pub(crate) fn finish(self) -> SamplerAttribution {
        SamplerAttribution {
            calls: self.calls,
            timer_spans: self.timer_spans,
            input_logits_total: self.input_logits.total,
            input_logits_min: self.input_logits.min.unwrap_or(0),
            input_logits_max: self.input_logits.max,
            wall_ms: self.wall_ms,
            shape_validation_ms: self.shape_validation_ms,
            candidate_alloc_ms: self.candidate_alloc_ms,
            candidate_fill_ms: self.candidate_fill_ms,
            top_k_order_ms: self.top_k_order_ms,
            min_p_ms: self.min_p_ms,
            positive_infinity_ms: self.positive_infinity_ms,
            temperature_scale_ms: self.temperature_scale_ms,
            probability_weights_ms: self.probability_weights_ms,
            top_p_ms: self.top_p_ms,
            categorical_ms: self.categorical_ms,
            residual_ms: self.residual_ms,
            candidate_capacity_bytes_total: self.candidate_capacity_bytes_total,
            candidate_capacity_bytes_peak: self.candidate_capacity_bytes_peak,
            probability_capacity_bytes_total: self.probability_capacity_bytes_total,
            probability_capacity_bytes_peak: self.probability_capacity_bytes_peak,
            after_top_k: self.after_top_k.finish(),
            after_min_p: self.after_min_p.finish(),
            after_positive_infinity: self.after_positive_infinity.finish(),
            after_top_p: self.after_top_p.finish(),
            candidate_index: self.candidate_index.finish(),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct TransitionAttributionAccumulator {
    pub(crate) calls: u64,
    pub(crate) new_timer_spans: u64,
    pub(crate) logits_bytes_per_call: Option<u64>,
    pub(crate) logits_bytes_total: u64,
    pub(crate) inner_wall_ms: f64,
    pub(crate) cpu_encode_ms: f64,
    pub(crate) completion_wait_ms: f64,
    pub(crate) gpu_ms_nested: f64,
    pub(crate) logits_alloc_zero_ms: f64,
    pub(crate) logits_copy_ms: f64,
    pub(crate) inner_residual_ms: f64,
}

impl TransitionAttributionAccumulator {
    pub(crate) fn record(
        &mut self,
        token: TokenProfile,
        readback: LogitsReadbackProfile,
    ) -> Result<()> {
        self.calls = self
            .calls
            .checked_add(1)
            .context("transition attribution call overflow")?;
        self.new_timer_spans = self
            .new_timer_spans
            .checked_add(u64::from(readback.timer_spans))
            .context("transition timer-span overflow")?;
        let bytes = u64::try_from(readback.bytes).context("logits bytes do not fit u64")?;
        if let Some(expected) = self.logits_bytes_per_call {
            ensure!(
                bytes == expected,
                "profiled transition logits bytes changed: {bytes} != {expected}"
            );
        } else {
            self.logits_bytes_per_call = Some(bytes);
        }
        self.logits_bytes_total = self
            .logits_bytes_total
            .checked_add(bytes)
            .context("transition logits-byte total overflow")?;
        self.inner_wall_ms += token.total_ms;
        self.cpu_encode_ms += token.cpu_encode_ms;
        self.completion_wait_ms += token.cpu_to_gpu_complete_ms;
        self.gpu_ms_nested += token.gpu_kernel_ms;
        self.logits_alloc_zero_ms += readback.allocation_zero_fill_ms;
        self.logits_copy_ms += readback.copy_ms;
        self.inner_residual_ms += token.total_ms
            - token.cpu_encode_ms
            - token.cpu_to_gpu_complete_ms
            - readback.allocation_zero_fill_ms
            - readback.copy_ms;
        Ok(())
    }

    pub(crate) fn finish(self, outer_wall_ms: f64) -> TransitionAttribution {
        TransitionAttribution {
            calls: self.calls,
            new_timer_spans: self.new_timer_spans,
            logits_bytes_per_call: self.logits_bytes_per_call.unwrap_or(0),
            logits_bytes_total: self.logits_bytes_total,
            outer_wall_ms,
            inner_wall_ms: self.inner_wall_ms,
            cpu_encode_ms: self.cpu_encode_ms,
            completion_wait_ms: self.completion_wait_ms,
            gpu_ms_nested: self.gpu_ms_nested,
            logits_alloc_zero_ms: self.logits_alloc_zero_ms,
            logits_copy_ms: self.logits_copy_ms,
            inner_residual_ms: self.inner_residual_ms,
            outer_wrapper_advance_ms: outer_wall_ms - self.inner_wall_ms,
        }
    }
}

pub(crate) fn measure_sampling_clock_probe() -> SamplingClockProbe {
    const BATCHES: usize = 7;
    const ITERATIONS: usize = 100_000;
    const NEW_TIMER_SPANS: u64 = 1_662;
    let mut pair_ns = [0.0; BATCHES];
    for value in &mut pair_ns {
        let batch_t0 = Instant::now();
        for _ in 0..ITERATIONS {
            let pair_t0 = Instant::now();
            std::hint::black_box(pair_t0.elapsed());
        }
        *value = batch_t0.elapsed().as_secs_f64() * 1e9 / ITERATIONS as f64;
    }
    let upper_pair_ns = pair_ns.iter().copied().fold(0.0f64, f64::max).ceil();
    SamplingClockProbe {
        batches: BATCHES as u32,
        iterations_per_batch: ITERATIONS as u64,
        pair_ns,
        upper_pair_ns,
        new_timer_spans: NEW_TIMER_SPANS,
        observer_upper_ms: upper_pair_ns * NEW_TIMER_SPANS as f64 / 1e6,
    }
}

pub(crate) fn adjusted_bound(raw_ms: f64, observer_upper_ms: f64) -> f64 {
    (raw_ms - observer_upper_ms).max(0.0)
}

pub(crate) fn bound_fraction(adjusted_ms: f64, generation_ms: f64) -> f64 {
    if generation_ms > 0.0 {
        adjusted_ms / generation_ms
    } else {
        0.0
    }
}

pub(crate) fn finalize_sampling_attribution(
    prompt_ids: &[i32],
    clock_probe: SamplingClockProbe,
    sampler: SamplerAttributionAccumulator,
    transitions: TransitionAttributionAccumulator,
    outer_transition_ms: f64,
    generation_ms: f64,
) -> SamplingAttribution {
    let sampler = sampler.finish();
    let transitions = transitions.finish(outer_transition_ms);
    let observer_upper_ms = clock_probe.observer_upper_ms;
    let workspace_raw_ms = transitions.logits_alloc_zero_ms + sampler.candidate_alloc_ms;
    let borrowed_raw_ms = transitions.logits_alloc_zero_ms + transitions.logits_copy_ms;
    let combined_raw_ms = borrowed_raw_ms + sampler.candidate_alloc_ms;
    let structural_raw_ms = combined_raw_ms + sampler.candidate_fill_ms + sampler.top_k_order_ms;
    let workspace_adjusted_ms = adjusted_bound(workspace_raw_ms, observer_upper_ms);
    let borrowed_adjusted_ms = adjusted_bound(borrowed_raw_ms, observer_upper_ms);
    let combined_adjusted_ms = adjusted_bound(combined_raw_ms, observer_upper_ms);
    let structural_adjusted_ms = adjusted_bound(structural_raw_ms, observer_upper_ms);
    SamplingAttribution {
        version: 1,
        prompt_token_ids_sha256: token_ids_sha256_i32le(prompt_ids),
        clock_probe,
        sampler,
        transitions,
        bounds: SamplingAttributionBounds {
            observer_upper_ms,
            workspace_raw_ms,
            workspace_adjusted_ms,
            workspace_adjusted_fraction: bound_fraction(workspace_adjusted_ms, generation_ms),
            borrowed_raw_ms,
            borrowed_adjusted_ms,
            borrowed_adjusted_fraction: bound_fraction(borrowed_adjusted_ms, generation_ms),
            combined_raw_ms,
            combined_adjusted_ms,
            combined_adjusted_fraction: bound_fraction(combined_adjusted_ms, generation_ms),
            structural_raw_ms,
            structural_adjusted_ms,
            structural_adjusted_fraction: bound_fraction(structural_adjusted_ms, generation_ms),
        },
    }
}

impl SamplingTelemetry {
    pub(crate) fn sampled(config: SamplingConfig, draws: usize) -> Option<Self> {
        (config.temperature > 0.0).then_some(Self {
            algorithm_version: SAMPLER_ALGORITHM_VERSION,
            temperature: config.temperature,
            top_k: config.top_k,
            top_p: config.top_p,
            min_p: config.min_p,
            effective_seed: config.seed,
            draws,
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct MetalAllocationSample {
    pub(crate) current_bytes: u64,
    pub(crate) delta_from_model_ready_bytes: i64,
    pub(crate) delta_from_request_start_bytes: i64,
}

#[derive(Debug, Serialize)]
pub(crate) struct MetalAllocationSamples {
    pub(crate) process_model_ready: MetalAllocationSample,
    pub(crate) request_start: MetalAllocationSample,
    pub(crate) after_scratch: MetalAllocationSample,
    pub(crate) after_sequence: MetalAllocationSample,
    pub(crate) after_prefill: MetalAllocationSample,
    pub(crate) after_first_stdout_flush: MetalAllocationSample,
    pub(crate) request_end_before_state_drop: MetalAllocationSample,
    pub(crate) after_request_state_drop: MetalAllocationSample,
    pub(crate) current_allocated_sampled_max_bytes: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestTimingRow {
    pub(crate) schema_version: u32,
    pub(crate) request_epoch: &'static str,
    pub(crate) request_index: usize,
    pub(crate) tokenizer_reused: bool,
    pub(crate) pair_requested: bool,
    pub(crate) pair_id: Option<String>,
    pub(crate) pair_request_equal: Option<bool>,
    pub(crate) pair_generated_tokens_equal: Option<bool>,
    pub(crate) prefix_cache_used: bool,
    pub(crate) build_commit: &'static str,
    pub(crate) build_dirty: &'static str,
    pub(crate) build_source_state: &'static str,
    pub(crate) model: String,
    pub(crate) runtime_identity_kind: &'static str,
    pub(crate) runtime_model_id: String,
    pub(crate) runtime_tokenizer_id: String,
    pub(crate) greedy_gpu_selection_reason: &'static str,
    pub(crate) request_start_unix_ms: u64,
    pub(crate) runtime_and_model_load_ms: f64,
    pub(crate) stdout_sink: &'static str,
    pub(crate) ttft_endpoint: &'static str,
    pub(crate) prompt_source: PromptSource,
    pub(crate) prompt_bytes: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) requested_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_sha256: String,
    pub(crate) stop_reason: StopReason,
    pub(crate) decode_policy: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sampling: Option<SamplingTelemetry>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sampling_attribution: Option<SamplingAttribution>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sampled_structural: Option<SampledStructuralTelemetry>,
    pub(crate) terminal_token_target_transition_consumed: bool,
    pub(crate) no_special_tokens: bool,
    pub(crate) prefill_chunk_requested: PrefillChunkArg,
    pub(crate) prefill_chunk_effective: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_chunk_decision: Option<PrefillChunkDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_attention_query: Option<PrefillAttentionQueryStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_scratch_overlay: Option<PrefillScratchOverlayTimingStats>,
    pub(crate) max_context_tokens: usize,
    pub(crate) prompt_acquisition_ms: f64,
    pub(crate) tokenizer_init_ms: f64,
    pub(crate) tokenization_ms: f64,
    pub(crate) capacity_validation_ms: f64,
    pub(crate) scratch_allocation_ms: f64,
    pub(crate) sequence_allocation_ms: f64,
    pub(crate) prefill_ms: f64,
    pub(crate) first_token_selection_ms: f64,
    pub(crate) first_token_callback_duration_ms: f64,
    pub(crate) first_token_ready_ms: f64,
    pub(crate) ttft_ms: f64,
    pub(crate) generation_ms: f64,
    pub(crate) transition_count: usize,
    pub(crate) transition_ms: f64,
    pub(crate) transition_tps: f64,
    pub(crate) inference_complete_ms: f64,
    pub(crate) total_request_ms: f64,
    pub(crate) pso_cache: PipelineCachePhaseMetrics,
    pub(crate) metal_allocated: MetalAllocationSamples,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_lookup: Option<PromptLookupDecodeStats>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct PipelineCacheMetricDelta {
    pub(crate) misses: u64,
    pub(crate) miss_wall_ns: u64,
    pub(crate) compiler_wall_ns: u64,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct PipelineCachePhaseMetrics {
    pub(crate) prefill: PipelineCacheMetricDelta,
    pub(crate) generation: PipelineCacheMetricDelta,
    pub(crate) total: PipelineCacheMetricDelta,
}

#[derive(Debug)]
pub(crate) struct PreparedRequest {
    pub(crate) request_start_unix_ms: u64,
    pub(crate) request_t0: Instant,
    pub(crate) request_start_allocated: Option<u64>,
    pub(crate) pipeline_cache_start: Option<MetalPipelineCacheMetrics>,
    pub(crate) prompt: String,
    pub(crate) prompt_source: PromptSource,
    pub(crate) completed_checkpoint_eligible: bool,
    pub(crate) prompt_ids: Vec<i32>,
    pub(crate) prompt_acquisition_ms: f64,
    pub(crate) tokenizer_init_ms: f64,
    pub(crate) tokenization_ms: f64,
    pub(crate) tokenizer_reused: bool,
}

pub(crate) fn pipeline_cache_delta(
    after: MetalPipelineCacheMetrics,
    before: MetalPipelineCacheMetrics,
) -> PipelineCacheMetricDelta {
    let delta = after.saturating_delta_since(before);
    PipelineCacheMetricDelta {
        misses: delta.misses,
        miss_wall_ns: delta.miss_wall_ns,
        compiler_wall_ns: delta.compiler_wall_ns,
    }
}

pub(crate) fn pipeline_cache_phase_metrics(
    request_start: MetalPipelineCacheMetrics,
    prefill_entry: MetalPipelineCacheMetrics,
    prefill_exit: MetalPipelineCacheMetrics,
    generation_exit: MetalPipelineCacheMetrics,
) -> PipelineCachePhaseMetrics {
    PipelineCachePhaseMetrics {
        prefill: pipeline_cache_delta(prefill_exit, prefill_entry),
        generation: pipeline_cache_delta(generation_exit, prefill_exit),
        total: pipeline_cache_delta(generation_exit, request_start),
    }
}

#[derive(Debug)]
pub(crate) struct SingleTurnResult {
    pub(crate) row: Option<RequestTimingRow>,
    pub(crate) prompt: String,
    pub(crate) prompt_source: PromptSource,
    pub(crate) prompt_ids: Vec<i32>,
    pub(crate) generated: Vec<i32>,
    pub(crate) transitions: usize,
    pub(crate) stop_reason: StopReason,
    pub(crate) prefill_ms: f64,
    pub(crate) ttft_ms: f64,
    pub(crate) decode_tps: f64,
    pub(crate) transition_tps: f64,
    pub(crate) tokenizer_init_ms: f64,
    pub(crate) tokenization_ms: f64,
    pub(crate) decode_ms: f64,
    pub(crate) total_ms: f64,
}

pub(crate) fn allocation_delta(current: u64, model_ready: u64) -> i64 {
    if current >= model_ready {
        i64::try_from(current - model_ready).unwrap_or(i64::MAX)
    } else {
        -i64::try_from(model_ready - current).unwrap_or(i64::MAX)
    }
}

pub(crate) fn allocation_sample(
    current: u64,
    model_ready: u64,
    request_start: u64,
) -> MetalAllocationSample {
    MetalAllocationSample {
        current_bytes: current,
        delta_from_model_ready_bytes: allocation_delta(current, model_ready),
        delta_from_request_start_bytes: allocation_delta(current, request_start),
    }
}

pub(crate) fn metal_allocation_samples(
    model_ready: u64,
    request_start: u64,
    after_scratch: u64,
    after_sequence: u64,
    after_prefill: u64,
    after_first_stdout_flush: u64,
    request_end_before_state_drop: u64,
    after_request_state_drop: u64,
) -> MetalAllocationSamples {
    let sampled_max = [
        model_ready,
        request_start,
        after_scratch,
        after_sequence,
        after_prefill,
        after_first_stdout_flush,
        request_end_before_state_drop,
        after_request_state_drop,
    ]
    .into_iter()
    .max()
    .unwrap_or(model_ready);
    MetalAllocationSamples {
        process_model_ready: allocation_sample(model_ready, model_ready, request_start),
        request_start: allocation_sample(request_start, model_ready, request_start),
        after_scratch: allocation_sample(after_scratch, model_ready, request_start),
        after_sequence: allocation_sample(after_sequence, model_ready, request_start),
        after_prefill: allocation_sample(after_prefill, model_ready, request_start),
        after_first_stdout_flush: allocation_sample(
            after_first_stdout_flush,
            model_ready,
            request_start,
        ),
        request_end_before_state_drop: allocation_sample(
            request_end_before_state_drop,
            model_ready,
            request_start,
        ),
        after_request_state_drop: allocation_sample(
            after_request_state_drop,
            model_ready,
            request_start,
        ),
        current_allocated_sampled_max_bytes: sampled_max,
    }
}

pub(crate) fn validate_request_timing_invariants(
    first_token_ready_ms: f64,
    ttft_ms: f64,
    inference_complete_ms: f64,
    total_request_ms: f64,
    generated_tokens: usize,
    transition_count: usize,
) -> Result<()> {
    for (name, value) in [
        ("first_token_ready_ms", first_token_ready_ms),
        ("ttft_ms", ttft_ms),
        ("inference_complete_ms", inference_complete_ms),
        ("total_request_ms", total_request_ms),
    ] {
        ensure!(value.is_finite() && value >= 0.0, "invalid {name}: {value}");
    }
    ensure!(
        first_token_ready_ms <= ttft_ms
            && ttft_ms <= inference_complete_ms
            && inference_complete_ms <= total_request_ms,
        "request timing milestones are out of order"
    );
    ensure!(
        transition_count.checked_add(1) == Some(generated_tokens),
        "expected N-1 transitions for N generated tokens"
    );
    Ok(())
}

pub(crate) fn ensure_close_ms(actual: f64, expected: f64, label: &str) -> Result<()> {
    ensure!(
        (actual - expected).abs() <= 0.001,
        "{label} does not reconcile: actual={actual:.9} expected={expected:.9}"
    );
    Ok(())
}

pub(crate) fn validate_sampling_attribution_row(row: &RequestTimingRow) -> Result<()> {
    let Some(attribution) = row.sampling_attribution.as_ref() else {
        return Ok(());
    };
    let sampling = row
        .sampling
        .as_ref()
        .context("sampling attribution requires sampling telemetry")?;
    ensure!(
        row.schema_version == 11,
        "sampling attribution requires schema 11"
    );
    ensure!(
        row.decode_policy == "sampled_cpu"
            && row.stop_reason == StopReason::TokenLimit
            && row.generated_tokens == 128
            && row.transition_count == 127
            && sampling.draws == 128
            && row.runtime_model_id == "e6024ce53109fdf7"
            && row.runtime_tokenizer_id == "a4b0b26f8a8c9917"
            && row.greedy_gpu_selection_reason == "ineligible_request",
        "sampling attribution request shape or terminal semantics changed"
    );
    ensure!(
        sampling.algorithm_version == 1,
        "sampling attribution requires sampler algorithm version 1"
    );
    ensure!(
        attribution.version == 1
            && attribution.prompt_token_ids_sha256
                == "fb4bbb4dc66ca7d219099e2974e787ef976f80789cde3e48b8a905dceece1f9f",
        "sampling attribution version or prompt-token identity changed"
    );
    let clock = &attribution.clock_probe;
    ensure!(
        clock.batches == 7
            && clock.iterations_per_batch == 100_000
            && clock.new_timer_spans == 1_662,
        "sampling clock-probe shape changed"
    );
    for value in clock.pair_ns {
        ensure!(
            value.is_finite() && value >= 0.0,
            "invalid clock-pair observation {value}"
        );
    }
    let expected_upper = clock.pair_ns.iter().copied().fold(0.0f64, f64::max).ceil();
    ensure_close_ms(
        clock.upper_pair_ns / 1e6,
        expected_upper / 1e6,
        "clock upper bound",
    )?;
    ensure_close_ms(
        clock.observer_upper_ms,
        clock.upper_pair_ns * clock.new_timer_spans as f64 / 1e6,
        "clock observer bound",
    )?;

    let sampler = &attribution.sampler;
    ensure!(
        sampler.calls == 128
            && sampler.timer_spans == 1_408
            && sampler.input_logits_total == 128 * 248_320
            && sampler.input_logits_min == 248_320
            && sampler.input_logits_max == 248_320,
        "sampling attribution call, timer, or logits counts changed"
    );
    for (label, summary) in [
        ("after_top_k", sampler.after_top_k),
        ("after_min_p", sampler.after_min_p),
        ("after_positive_infinity", sampler.after_positive_infinity),
        ("after_top_p", sampler.after_top_p),
        ("candidate_index", sampler.candidate_index),
    ] {
        let calls = sampler.calls;
        ensure!(
            summary.min <= summary.max
                && summary.total >= calls.saturating_mul(summary.min)
                && summary.total <= calls.saturating_mul(summary.max),
            "invalid {label} count summary"
        );
    }
    ensure!(
        sampler.candidate_capacity_bytes_peak > 0
            && sampler.candidate_capacity_bytes_total >= sampler.candidate_capacity_bytes_peak
            && sampler.candidate_capacity_bytes_total
                <= sampler
                    .calls
                    .saturating_mul(sampler.candidate_capacity_bytes_peak)
            && sampler.probability_capacity_bytes_peak > 0
            && sampler.probability_capacity_bytes_total >= sampler.probability_capacity_bytes_peak
            && sampler.probability_capacity_bytes_total
                <= sampler
                    .calls
                    .saturating_mul(sampler.probability_capacity_bytes_peak),
        "invalid sampler capacity-byte accounting"
    );
    let sampler_non_residual = [
        sampler.wall_ms,
        sampler.shape_validation_ms,
        sampler.candidate_alloc_ms,
        sampler.candidate_fill_ms,
        sampler.top_k_order_ms,
        sampler.min_p_ms,
        sampler.positive_infinity_ms,
        sampler.temperature_scale_ms,
        sampler.probability_weights_ms,
        sampler.top_p_ms,
        sampler.categorical_ms,
    ];
    ensure!(
        sampler_non_residual
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && sampler.residual_ms.is_finite(),
        "invalid sampler attribution duration"
    );
    let sampler_phase_sum = sampler.shape_validation_ms
        + sampler.candidate_alloc_ms
        + sampler.candidate_fill_ms
        + sampler.top_k_order_ms
        + sampler.min_p_ms
        + sampler.positive_infinity_ms
        + sampler.temperature_scale_ms
        + sampler.probability_weights_ms
        + sampler.top_p_ms
        + sampler.categorical_ms
        + sampler.residual_ms;
    ensure_close_ms(sampler.wall_ms, sampler_phase_sum, "sampler phase sum")?;

    let transitions = &attribution.transitions;
    ensure!(
        transitions.calls == 127
            && transitions.new_timer_spans == 254
            && transitions.logits_bytes_per_call == 993_280
            && transitions.logits_bytes_total == 126_146_560,
        "sampling attribution transition or logits-byte counts changed"
    );
    let transition_non_residual = [
        transitions.outer_wall_ms,
        transitions.inner_wall_ms,
        transitions.cpu_encode_ms,
        transitions.completion_wait_ms,
        transitions.gpu_ms_nested,
        transitions.logits_alloc_zero_ms,
        transitions.logits_copy_ms,
    ];
    ensure!(
        transition_non_residual
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
            && transitions.inner_residual_ms.is_finite()
            && transitions.outer_wrapper_advance_ms.is_finite(),
        "invalid transition attribution duration"
    );
    ensure_close_ms(
        transitions.inner_wall_ms,
        transitions.cpu_encode_ms
            + transitions.completion_wait_ms
            + transitions.logits_alloc_zero_ms
            + transitions.logits_copy_ms
            + transitions.inner_residual_ms,
        "inner transition phase sum",
    )?;
    ensure_close_ms(
        transitions.outer_wall_ms,
        transitions.inner_wall_ms + transitions.outer_wrapper_advance_ms,
        "outer transition phase sum",
    )?;
    ensure_close_ms(
        transitions.outer_wall_ms,
        row.transition_ms,
        "request and attribution transition wall",
    )?;
    ensure!(
        transitions.gpu_ms_nested <= transitions.completion_wait_ms + 0.001,
        "nested GPU wall exceeds completion wait"
    );

    let bounds = attribution.bounds;
    let expected_workspace = transitions.logits_alloc_zero_ms + sampler.candidate_alloc_ms;
    let expected_borrowed = transitions.logits_alloc_zero_ms + transitions.logits_copy_ms;
    let expected_combined = expected_borrowed + sampler.candidate_alloc_ms;
    let expected_structural =
        expected_combined + sampler.candidate_fill_ms + sampler.top_k_order_ms;
    for (label, actual, expected) in [
        ("workspace raw", bounds.workspace_raw_ms, expected_workspace),
        ("borrowed raw", bounds.borrowed_raw_ms, expected_borrowed),
        ("combined raw", bounds.combined_raw_ms, expected_combined),
        (
            "structural raw",
            bounds.structural_raw_ms,
            expected_structural,
        ),
    ] {
        ensure_close_ms(actual, expected, label)?;
    }
    for (label, raw, adjusted, fraction) in [
        (
            "workspace",
            bounds.workspace_raw_ms,
            bounds.workspace_adjusted_ms,
            bounds.workspace_adjusted_fraction,
        ),
        (
            "borrowed",
            bounds.borrowed_raw_ms,
            bounds.borrowed_adjusted_ms,
            bounds.borrowed_adjusted_fraction,
        ),
        (
            "combined",
            bounds.combined_raw_ms,
            bounds.combined_adjusted_ms,
            bounds.combined_adjusted_fraction,
        ),
        (
            "structural",
            bounds.structural_raw_ms,
            bounds.structural_adjusted_ms,
            bounds.structural_adjusted_fraction,
        ),
    ] {
        let expected_adjusted = adjusted_bound(raw, bounds.observer_upper_ms);
        ensure_close_ms(adjusted, expected_adjusted, &format!("{label} adjusted"))?;
        let expected_fraction = bound_fraction(expected_adjusted, row.generation_ms);
        ensure!(
            (fraction - expected_fraction).abs() <= 1e-9,
            "{label} adjusted fraction does not reconcile"
        );
    }
    for value in [
        bounds.observer_upper_ms,
        bounds.workspace_raw_ms,
        bounds.workspace_adjusted_ms,
        bounds.workspace_adjusted_fraction,
        bounds.borrowed_raw_ms,
        bounds.borrowed_adjusted_ms,
        bounds.borrowed_adjusted_fraction,
        bounds.combined_raw_ms,
        bounds.combined_adjusted_ms,
        bounds.combined_adjusted_fraction,
        bounds.structural_raw_ms,
        bounds.structural_adjusted_ms,
        bounds.structural_adjusted_fraction,
    ] {
        ensure!(
            value.is_finite() && value >= 0.0,
            "invalid sampling attribution bound"
        );
    }
    ensure_close_ms(
        bounds.observer_upper_ms,
        clock.observer_upper_ms,
        "bound observer overhead",
    )?;
    Ok(())
}

pub(crate) fn validate_sampled_structural_row(row: &RequestTimingRow) -> Result<()> {
    let Some(structural) = row.sampled_structural.as_ref() else {
        ensure!(
            row.schema_version != 12,
            "schema 12 requires sampled structural telemetry"
        );
        return Ok(());
    };
    let sampling = row
        .sampling
        .as_ref()
        .context("sampled structural telemetry requires sampling telemetry")?;
    ensure!(
        SAMPLER_ALGORITHM_VERSION == 1
            && structural.algorithm_version == 1
            && sampling.algorithm_version == 1,
        "sampled structural requires sampler algorithm version 1"
    );
    ensure!(
        row.schema_version == 12
            && row.sampling_attribution.is_none()
            && row.decode_policy == "sampled_cpu"
            && sampling.draws == row.generated_tokens,
        "sampled structural schema or sampling contract changed"
    );
    ensure!(
        structural.version == 1 && structural.path == "bounded_topk_borrowed_transitions",
        "sampled structural version or path changed"
    );
    ensure!(
        structural.prompt_owned_bounded_calls == 1
            && structural.borrowed_transition_calls
                == u64::try_from(row.transition_count)
                    .context("transition count does not fit u64")?
            && structural.resident_head_wait_calls == structural.borrowed_transition_calls
            && structural.validated_shared_row_calls == structural.borrowed_transition_calls
            && structural.fallback_calls == 0,
        "sampled structural call accounting changed"
    );
    let calls = structural
        .prompt_owned_bounded_calls
        .checked_add(structural.borrowed_transition_calls)
        .context("sampled structural call count overflow")?;
    let generated_tokens =
        u64::try_from(row.generated_tokens).context("generated token count does not fit u64")?;
    let sampling_top_k =
        u64::try_from(sampling.top_k).context("sampling top-k does not fit u64")?;
    ensure!(
        calls == generated_tokens
            && structural.input_logits_min > 0
            && structural.input_logits_min == structural.input_logits_max
            && structural.retained_top_k_min > 0
            && structural.retained_top_k_min == structural.retained_top_k_max
            && structural.retained_top_k_min == sampling_top_k,
        "sampled structural support summaries changed"
    );
    ensure!(
        structural.input_logits_total
            == calls
                .checked_mul(structural.input_logits_min)
                .context("sampled structural input-logits product overflow")?
            && structural.retained_top_k_total
                == calls
                    .checked_mul(structural.retained_top_k_min)
                    .context("sampled structural retained-top-k product overflow")?
            && structural.max_heap_len == structural.retained_top_k_max
            && structural.max_heap_capacity >= structural.max_heap_len
            && structural.max_heap_capacity < structural.input_logits_min,
        "sampled structural heap or total accounting changed"
    );
    ensure!(
        structural.full_candidate_vector_allocations == 0
            && structural.transition_logits_copy_bytes == 0
            && structural.extra_command_buffers == 0
            && structural.gpu_sampling_dispatches == 0,
        "sampled structural path added excluded work"
    );
    Ok(())
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestOutput {
    pub(crate) id: String,
    /// Source line in the requests JSONL; the stable key for merging
    /// outcomes back onto inputs.
    pub(crate) line: usize,
    pub(crate) status: &'static str,
    pub(crate) input: JsonlInputLabel,
    pub(crate) prompt_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_sha256: String,
    pub(crate) generated_text: String,
    pub(crate) stop_reason: StopReason,
    pub(crate) terminal_token_target_transition_consumed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_partition: Option<GeneratedThinkingPartition>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct GeneratedThinkingPartition {
    pub(crate) delimiter: &'static str,
    pub(crate) delimiter_start_token_index: usize,
    pub(crate) delimiter_end_token_index_exclusive: usize,
    pub(crate) delimiter_token_aligned: bool,
    pub(crate) reasoning_tokens: Option<usize>,
    pub(crate) delimiter_tokens: Option<usize>,
    pub(crate) visible_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsRow {
    pub(crate) schema_version: u32,
    pub(crate) request_stats_contract: &'static str,
    pub(crate) id: String,
    pub(crate) line: usize,
    pub(crate) model: String,
    pub(crate) build_commit: &'static str,
    pub(crate) build_dirty: bool,
    pub(crate) build_source_state: &'static str,
    pub(crate) model_prefetch_policy: &'static str,
    pub(crate) model_prefetch_bytes_returned: u64,
    pub(crate) greedy_gpu_selection_reason: &'static str,
    pub(crate) arrival_ms: u64,
    pub(crate) finish_ms: u64,
    pub(crate) prompt_tokens: usize,
    pub(crate) prompt_hash: String,
    pub(crate) requested_tokens: usize,
    pub(crate) generated_tokens: usize,
    pub(crate) generated_token_sha256: String,
    pub(crate) decode_policy: &'static str,
    pub(crate) stop_reason: StopReason,
    pub(crate) terminal_token_target_transition_consumed: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) thinking_partition: Option<GeneratedThinkingPartition>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) sampling: Option<SamplingTelemetry>,
    pub(crate) cache_prefix_tokens: Option<usize>,
    pub(crate) cache_prefix_source: String,
    pub(crate) cache_prefix_hash: Option<String>,
    pub(crate) auto_cache_prefix_tokens: Option<usize>,
    pub(crate) auto_cache_future_hits: usize,
    pub(crate) cache_hit: bool,
    pub(crate) matched_prefix_tokens: usize,
    pub(crate) matched_prefix_hash: Option<String>,
    pub(crate) exact_cache_hit: bool,
    pub(crate) prefill_chunk: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_chunk_effective: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_chunk_decision: Option<PrefillChunkDecision>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_attention_query: Option<PrefillAttentionQueryStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prefill_scratch_overlay: Option<PrefillScratchOverlayTimingStats>,
    pub(crate) max_context_tokens: usize,
    pub(crate) no_special_tokens: bool,
    pub(crate) restore_ms: f64,
    pub(crate) prefix_inserted_bytes: u64,
    pub(crate) prefix_insert_ms: f64,
    pub(crate) prefill_ms: f64,
    pub(crate) decode_ms: f64,
    pub(crate) model_ttft_ms: f64,
    pub(crate) first_token_ms: f64,
    pub(crate) first_token_callback_ms: f64,
    pub(crate) first_decode_ms: f64,
    pub(crate) decode_tps: f64,
    pub(crate) decode_transitions: usize,
    pub(crate) transition_ms: f64,
    pub(crate) transition_tps: f64,
    pub(crate) total_ms: f64,
    pub(crate) cache_entries: usize,
    pub(crate) cache_bytes: u64,
    pub(crate) cache_max_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) prompt_lookup: Option<PromptLookupDecodeStats>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct PromptLookupDecodeStats {
    pub(crate) index_build_ms: f64,
    pub(crate) index_update_ms: f64,
    pub(crate) lookup_ms: f64,
    pub(crate) scratch_allocation_ms: f64,
    pub(crate) scratch_allocated_bytes: u64,
    pub(crate) scratch_peak_allocated_bytes: u64,
    pub(crate) serial_ms: f64,
    pub(crate) verify_ms: f64,
    pub(crate) restore_ms: f64,
    pub(crate) attempts: usize,
    pub(crate) abstentions: usize,
    pub(crate) verify_calls: usize,
    pub(crate) restore_calls: usize,
    pub(crate) accepted_drafts: usize,
    pub(crate) drafts_scored: usize,
    pub(crate) physical_target_positions: usize,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum RequestStatsStatus {
    Ok,
    // Reserved for future error / cancelled records. Kept explicit so schema
    // consumers see the full status vocabulary from v1.
    #[allow(dead_code)]
    Error,
    #[allow(dead_code)]
    Cancelled,
}

/// Envelope-owned finish reason enum, decoupled from the internal `StopReason`
/// so that adding a new internal variant or renaming does not silently mutate
/// the wire schema. Exhaustive `From<StopReason>` forces future variants to
/// be a compile-time decision.
#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RequestStatsFinishReason {
    Eos,
    TokenLimit,
}

impl From<StopReason> for RequestStatsFinishReason {
    fn from(reason: StopReason) -> Self {
        match reason {
            StopReason::Eos => Self::Eos,
            StopReason::TokenLimit => Self::TokenLimit,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsRequestRecord<'a> {
    pub(crate) schema: &'static str,
    pub(crate) schema_version: u32,
    pub(crate) record_type: &'static str,
    pub(crate) invocation_id: &'a str,
    pub(crate) request_index: u32,
    pub(crate) status: RequestStatsStatus,
    pub(crate) model: RequestStatsModel<'a>,
    pub(crate) input: RequestStatsInput<'a>,
    // Success-only fields are optional so future error/cancelled records need
    // only omit them, without a breaking restructuring.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) usage: Option<RequestStatsUsage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) finish: Option<RequestStatsFinish>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) timing_ms: Option<RequestStatsTiming>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) throughput_tps: Option<RequestStatsThroughput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) output_fingerprint: Option<RequestStatsOutputFingerprint>,
    pub(crate) build: RequestStatsBuild,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) diagnostics: Option<RequestStatsDiagnostics>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsModel<'a> {
    pub(crate) family: &'a str,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsInput<'a> {
    pub(crate) kind: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) template: Option<&'a str>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsUsage {
    pub(crate) input_tokens: u64,
    pub(crate) output_tokens: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsFinish {
    pub(crate) reason: RequestStatsFinishReason,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsTiming {
    pub(crate) total: f64,
    pub(crate) tokenization: f64,
    pub(crate) prefill: f64,
    pub(crate) decode: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsThroughput {
    pub(crate) prefill: f64,
    pub(crate) decode: f64,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsOutputFingerprint {
    pub(crate) algorithm: &'static str,
    pub(crate) value: String,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsBuild {
    pub(crate) commit: &'static str,
    pub(crate) dirty: bool,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsDiagnostics {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) deepseek_v4: Option<RequestStatsDeepSeekV4Diagnostics>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RequestStatsDeepSeekV4Diagnostics {
    pub(crate) schema_version: u32,
    pub(crate) prefill_mode: &'static str,
    pub(crate) prefill_chunk_cap: u64,
    pub(crate) transitions: u64,
    pub(crate) transition_tps: f64,
    pub(crate) load_ms: f64,
}

/// Case-insensitive parser for the `QWEN_BUILD_DIRTY` build-time env var.
/// Accepts `0`/`false`/`no` (any case, plus empty) as clean; anything else
/// is treated as dirty, biasing toward "assume unstable" if the value is
/// unexpected.
pub(crate) fn parse_build_dirty(raw: &str) -> bool {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return false;
    }
    !(trimmed.eq_ignore_ascii_case("0")
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no"))
}

/// The family-neutral measured core of one completed single-turn request.
/// Every lane already computes these; the record is a projection, not a new
/// instrument. Family-specific facts travel in `RequestStatsDiagnostics`.
pub(crate) struct RequestStatsMeasured {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub transitions: u64,
    pub stop_reason: StopReason,
    pub tokenizer_ms: f64,
    pub load_ms: f64,
    pub prefill_ms: f64,
    pub prefill_tps: f64,
    pub decode_ms: f64,
    pub decode_tps: f64,
    pub transition_tps: f64,
    pub total_ms: f64,
    pub output_fingerprint: GeneratedTokenSha256Digest,
}

/// Build the `qwen-llm.request-stats` v1 record for a completed single-turn
/// request. `input` is the representation the lane rendered (`raw` /
/// `messages` plus an optional template label); `diagnostics` is the lane's
/// own namespaced block, or `None` when it has nothing qualified to add.
pub(crate) fn build_single_turn_stats_record<'a>(
    invocation_id: &'a str,
    request_index: u32,
    family: &'a str,
    input: RequestStatsInput<'a>,
    measured: &RequestStatsMeasured,
    diagnostics: Option<RequestStatsDiagnostics>,
) -> RequestStatsRequestRecord<'a> {
    RequestStatsRequestRecord {
        schema: "qwen-llm.request-stats",
        schema_version: 1,
        record_type: "request_stats",
        invocation_id,
        request_index,
        status: RequestStatsStatus::Ok,
        model: RequestStatsModel { family },
        input,
        usage: Some(RequestStatsUsage {
            input_tokens: measured.input_tokens,
            output_tokens: measured.output_tokens,
        }),
        finish: Some(RequestStatsFinish {
            reason: measured.stop_reason.into(),
        }),
        timing_ms: Some(RequestStatsTiming {
            total: sanitize_finite_metric(measured.total_ms, "timing_ms.total"),
            tokenization: sanitize_finite_metric(measured.tokenizer_ms, "timing_ms.tokenization"),
            prefill: sanitize_finite_metric(measured.prefill_ms, "timing_ms.prefill"),
            decode: sanitize_finite_metric(measured.decode_ms, "timing_ms.decode"),
        }),
        throughput_tps: Some(RequestStatsThroughput {
            prefill: sanitize_finite_metric(measured.prefill_tps, "throughput_tps.prefill"),
            decode: sanitize_finite_metric(measured.decode_tps, "throughput_tps.decode"),
        }),
        output_fingerprint: Some(RequestStatsOutputFingerprint {
            algorithm: "sha256-qwen-generated-token-ids-v1",
            value: measured.output_fingerprint.hex(),
        }),
        build: RequestStatsBuild {
            commit: env!("QWEN_BUILD_COMMIT"),
            dirty: parse_build_dirty(env!("QWEN_BUILD_DIRTY")),
        },
        diagnostics,
    }
}

/// `input.kind` for the common core is restricted to `raw` / `messages`; the
/// template label is the lane's pinned protocol name when it has one.
pub(crate) fn request_stats_input(
    source: PromptSource,
    template: Option<&'static str>,
) -> RequestStatsInput<'static> {
    match source {
        PromptSource::Inline | PromptSource::File => RequestStatsInput {
            kind: "raw",
            template: None,
        },
        PromptSource::Messages => RequestStatsInput {
            kind: "messages",
            template,
        },
    }
}

/// Append one v1 record to the `--request-stats-jsonl` sidecar.
pub(crate) fn append_single_turn_stats_record(
    path: &Path,
    request_index: u32,
    family: &str,
    input: RequestStatsInput<'_>,
    measured: &RequestStatsMeasured,
    diagnostics: Option<RequestStatsDiagnostics>,
) -> Result<()> {
    let record = build_single_turn_stats_record(
        &INVOCATION_ID,
        request_index,
        family,
        input,
        measured,
        diagnostics,
    );
    append_jsonl_record(path, &record, "request stats jsonl")
}

pub(crate) fn unix_epoch_ms_u64() -> Result<u64> {
    let ms = unix_epoch_ms()?;
    u64::try_from(ms).context("Unix epoch milliseconds do not fit u64")
}

pub(crate) const TOKEN_HASH_SEED: u64 = 0xcbf29ce484222325;

pub(crate) const TOKEN_HASH_PRIME: u64 = 0x100000001b3;

pub(crate) fn token_hash_hex(tokens: &[i32]) -> String {
    let mut hash = TOKEN_HASH_SEED;
    for &token in tokens {
        hash ^= (token as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
        hash = hash.wrapping_mul(TOKEN_HASH_PRIME);
    }
    format!("{hash:016x}")
}

/// Type-safe wrapper for the generated-token-ids fingerprint. The tuple
/// field is scoped to a private child module so that neither crate root
/// code, nor tests, nor any other module can bypass `of()` to construct a
/// digest with arbitrary bytes. This is what actually enforces the
/// invariant that the emitted value under algorithm identifier
/// `sha256-qwen-generated-token-ids-v1` is the output of that exact algorithm.
pub(crate) mod fingerprint {
    use sha2::{Digest, Sha256};

    #[derive(Debug, Clone, Copy, Eq, PartialEq)]
    pub struct GeneratedTokenSha256Digest([u8; 32]);

    impl GeneratedTokenSha256Digest {
        /// Compute the canonical fingerprint over token IDs. Byte layout:
        /// `domain-separator || length_u64_le || (token_i32_le)*`. See
        /// `sha256-qwen-generated-token-ids-v1` algorithm identifier.
        pub fn of(tokens: &[i32]) -> Self {
            let mut digest = Sha256::new();
            digest.update(b"qwen-generated-token-ids-v1\0");
            digest.update((tokens.len() as u64).to_le_bytes());
            for token in tokens {
                digest.update(token.to_le_bytes());
            }
            Self(digest.finalize().into())
        }

        pub fn hex(&self) -> String {
            crate::hex_encode_bytes(&self.0)
        }

        #[cfg(test)]
        pub fn as_bytes(&self) -> &[u8; 32] {
            &self.0
        }
    }
}

pub(crate) fn generated_token_sha256(tokens: &[i32]) -> String {
    GeneratedTokenSha256Digest::of(tokens).hex()
}

pub(crate) fn hex_encode_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

/// Per-process invocation identifier: 16 bytes of /dev/urandom entropy
/// formatted as lowercase hex (matches UUID/ULID uniqueness class without
/// adding a dep). **Fails closed** if secure entropy is unavailable — better
/// to refuse to run than to emit records with a weak, potentially colliding
/// identifier that consumers might trust for cross-run correlation.
///
/// Generated on first access; force-init only when telemetry is actually
/// requested (see the --request-stats-jsonl pre-flight branch in the DS4
/// single-turn generator), so `--help`, `--version`, non-telemetry
/// invocations, and dispatch errors never touch /dev/urandom.
pub(crate) static INVOCATION_ID: LazyLock<String> = LazyLock::new(|| {
    generate_invocation_id_from(EntropySource::DevUrandom)
        .unwrap_or_else(|e| panic!("cannot initialize invocation id: {e}"))
});

#[derive(Copy, Clone, Debug)]
pub(crate) enum EntropySource {
    DevUrandom,
    #[cfg(test)]
    Deterministic([u8; 16]),
    #[cfg(test)]
    ForceFail,
}

pub(crate) fn generate_invocation_id_from(source: EntropySource) -> Result<String> {
    let mut buf = [0u8; 16];
    match source {
        EntropySource::DevUrandom => {
            let mut f = std::fs::File::open("/dev/urandom")
                .context("open /dev/urandom for invocation id entropy")?;
            f.read_exact(&mut buf)
                .context("read 16 bytes from /dev/urandom for invocation id")?;
        }
        #[cfg(test)]
        EntropySource::Deterministic(bytes) => {
            buf = bytes;
        }
        #[cfg(test)]
        EntropySource::ForceFail => {
            return Err(anyhow!(
                "test-injected entropy failure (secure randomness unavailable)"
            ));
        }
    }
    Ok(hex_encode_bytes(&buf))
}

/// Append a serialized JSONL record with cooperating-writer integrity:
///   * The record is fully serialized into a memory buffer first, so a
///     serialization failure never writes a partial line.
///   * An advisory exclusive `flock` is held across the tail-repair check,
///     the write, and the `fsync`. Cooperating processes cannot interleave
///     with each other. Non-cooperating writers (that ignore flock) are
///     out of scope.
///   * If the file already ends with a partial record (does not end with
///     `\n`), a leading `\n` is prepended to the buffer so the next record
///     starts on a fresh line — otherwise we would concatenate the new
///     record onto the abandoned partial one and corrupt that line.
///   * `sync_data` is invoked after the write, so delayed I/O failures
///     (e.g. `ENOSPC` on a filesystem with write-back caching) surface as
///     errors instead of being silently deferred past our success return.
///     `File::flush` is a no-op on Unix and Windows; `sync_data` is not.
///   * On write or fsync failure, best-effort rollback truncates the file
///     back to the original length. Rollback failure is chained into the
///     returned error rather than silently discarded.
pub(crate) fn append_jsonl_record<T: Serialize>(
    path: &Path,
    record: &T,
    label: &str,
) -> Result<()> {
    let payload =
        serde_json::to_vec(record).with_context(|| format!("serialize {label} record"))?;
    // Open with read+append so the SAME locked fd can serve both the tail
    // probe (via pread) and the write. open_append_file grants append-only,
    // which would fail pread with EBADF.
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} directory {}", parent.display()))?;
    }
    let mut file = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))?;
    let fd = file.as_raw_fd();
    // SAFETY: fd is valid for the duration of `file`; flock(2) accepts any
    // open file descriptor. LOCK_EX blocks until acquired.
    let lock_rc = unsafe { libc::flock(fd, libc::LOCK_EX) };
    if lock_rc != 0 {
        return Err(anyhow!(
            "acquire exclusive lock on {label} {}: {}",
            path.display(),
            std::io::Error::last_os_error()
        ));
    }
    let unlock = |fd: std::os::unix::io::RawFd| {
        // SAFETY: fd is valid for the caller-held `file`; LOCK_UN is defined.
        let _ = unsafe { libc::flock(fd, libc::LOCK_UN) };
    };
    let original_len = match file.metadata() {
        Ok(m) => m.len(),
        Err(e) => {
            unlock(fd);
            return Err(anyhow::Error::new(e).context(format!("stat {label} {}", path.display())));
        }
    };
    // Detect a pre-existing partial-record tail (file does not end with '\n').
    // Read through the SAME locked fd via pread(2) to avoid TOCTOU across the
    // rename/replace race a second open on the pathname would expose.
    let mut buf = Vec::with_capacity(payload.len() + 2);
    if original_len > 0 {
        let mut probe = [0u8; 1];
        let offset = (original_len - 1) as libc::off_t;
        // SAFETY: fd is valid; pread reads at a specific offset without
        // moving the file pointer, does not mutate the file, and returns
        // -1 on error with errno set.
        let n = unsafe { libc::pread(fd, probe.as_mut_ptr().cast(), probe.len(), offset) };
        if n < 0 {
            unlock(fd);
            return Err(anyhow!(
                "probe tail of {label} {}: {}",
                path.display(),
                std::io::Error::last_os_error()
            ));
        }
        // Concurrent truncation between metadata and pread would leave the
        // tail unprobed; refuse rather than risk concatenating onto it.
        if n != 1 {
            unlock(fd);
            return Err(anyhow!(
                "tail probe of {label} {} returned {n} bytes at offset {}; \
                 expected 1 (concurrent truncation between metadata and pread?)",
                path.display(),
                offset,
            ));
        }
        if probe[0] != b'\n' {
            // Isolate our record from the orphan tail rather than concatenating.
            buf.push(b'\n');
        }
    }
    buf.extend_from_slice(&payload);
    buf.push(b'\n');
    // Write + durability. sync_data persists file content; sync_containing_dir
    // persists a newly-created directory entry (fsync on the file alone does
    // NOT guarantee the entry survives a crash for a new file).
    let write_result = file
        .write_all(&buf)
        .and_then(|()| file.flush())
        .and_then(|()| file.sync_data())
        .and_then(|()| sync_containing_dir(path));
    if let Err(e) = write_result {
        // Best-effort rollback: truncate AND sync to persist the reverted
        // state. Chain both errors together — silently discarding either
        // would let a corrupt tail persist beyond the reported error.
        let rollback_err = file
            .set_len(original_len)
            .and_then(|()| file.sync_data())
            .err();
        unlock(fd);
        let mut chained =
            anyhow::Error::new(e).context(format!("append {label} record to {}", path.display()));
        if let Some(re) = rollback_err {
            chained = chained.context(format!(
                "rollback truncate/sync also failed for {}: {}",
                path.display(),
                re
            ));
        }
        return Err(chained);
    }
    unlock(fd);
    Ok(())
}

/// fsync the directory containing `path` so a newly-created entry is
/// persisted before we report success. `sync_data`/`fsync` on the file
/// itself is not enough for a new directory entry on most filesystems.
///
/// If the path's parent hierarchy was newly created (via `create_dir_all`),
/// sync each newly-materialized directory on the way up to an existing
/// ancestor, so the whole hierarchy is durable — not just the final leaf
/// directory containing the file.
pub(crate) fn sync_containing_dir(path: &Path) -> std::io::Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let leaf = if parent.as_os_str().is_empty() {
        Path::new(".")
    } else {
        parent
    };
    // Walk up from the leaf, syncing each directory. Stop at root or when
    // a further ancestor doesn't need syncing (best-effort: we always sync
    // the leaf; ancestors are synced too as a conservative default so that
    // a create_dir_all() hierarchy survives crash).
    let mut cur: Option<&Path> = Some(leaf);
    while let Some(dir) = cur {
        std::fs::File::open(dir)?.sync_all()?;
        cur = dir
            .parent()
            .filter(|p| !p.as_os_str().is_empty() && *p != dir);
    }
    Ok(())
}

/// Preflight the --request-stats-jsonl destination using the exact open
/// mode the emission path uses (read+append). An append-writable but
/// unreadable file cannot pass this check and then fail emission after
/// inference cost is paid. Also creates parent directories so append
/// itself doesn't fail on first record.
pub(crate) fn preflight_request_stats_jsonl(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).with_context(|| {
            format!(
                "create request stats jsonl (pre-flight) directory {}",
                parent.display()
            )
        })?;
    }
    let _ = OpenOptions::new()
        .create(true)
        .read(true)
        .append(true)
        .open(path)
        .with_context(|| {
            format!(
                "open request stats jsonl (pre-flight) with read+append {}",
                path.display()
            )
        })?;
    Ok(())
}

/// Coerce a metric to a well-defined finite JSON representation. Returns 0.0
/// for non-finite (NaN, ±∞) or negative values. Emits a `tracing::warn` so
/// callers notice degenerate measurements. Prevents JSON `null` in fields
/// documented as non-negative f64 (serde_json serializes NaN/∞ as `null`).
pub(crate) fn sanitize_finite_metric(value: f64, label: &str) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        tracing::warn!(
            metric = label,
            raw_value = value,
            "non-finite metric coerced to 0.0 for stats emission"
        );
        0.0
    }
}

pub(crate) fn open_append_file(path: &Path, label: &str) -> Result<std::fs::File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} directory {}", parent.display()))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))
}

pub(crate) fn unix_epoch_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_millis())
}

pub(crate) fn append_request_trace(
    path: &Path,
    arrival_ms: u128,
    prompt_tokens: usize,
    generated_tokens: usize,
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create trace directory {}", parent.display()))?;
    }
    let write_header = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open request trace {}", path.display()))?;
    if write_header {
        writeln!(file, "arrival_ms\ttokens\tid\tprompt_tokens")?;
    }
    let id = format!("{}-{arrival_ms}", std::process::id());
    writeln!(
        file,
        "{arrival_ms}\t{generated_tokens}\t{id}\t{prompt_tokens}"
    )?;
    Ok(())
}

pub(crate) fn argmax_i32(xs: &[f32]) -> i32 {
    // Ties resolve to the LOWEST index, matching the GPU argmax kernel
    // contract that packed-verify decode uses (`kernel_argmax_f32`). The
    // old max_by(total_cmp) kept the LAST element on ties, inverting the
    // two decode paths' tie semantics.
    let mut best = (0usize, xs[0]);
    for (i, &v) in xs.iter().enumerate().skip(1) {
        if v.total_cmp(&best.1) == std::cmp::Ordering::Greater {
            best = (i, v);
        }
    }
    best.0 as i32
}

#[cfg(test)]
mod argmax_tie_tests {
    use super::argmax_i32;

    #[test]
    fn ties_resolve_to_lowest_index() {
        assert_eq!(argmax_i32(&[1.0, 5.0, 5.0, 3.0]), 1);
        assert_eq!(argmax_i32(&[2.0, 2.0, 2.0]), 0);
        assert_eq!(argmax_i32(&[-0.0, 0.0, -0.0]), 1);
        assert_eq!(argmax_i32(&[0.0, -0.0, 0.0]), 0);
    }
}
