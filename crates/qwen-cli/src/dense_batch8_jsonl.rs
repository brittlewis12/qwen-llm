use super::*;
use qwen_llm::dense_batch8::DENSE_BATCH8_WIDTH;
use qwen_llm::runtime::DenseBatch8SequenceExecutor;

const PAD_TOKEN: i32 = 0;
const TRANSIENT_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const PREFIX_FANOUT_MIN_TOKENS: usize = 256;
const PREFIX_FANOUT_ENV: &str = "QWEN_DENSE_BATCH8_PREFIX_FANOUT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefixFanoutPlan {
    common_prefix_tokens: usize,
    selected_prefix_tokens: usize,
    reason: &'static str,
}

#[derive(Debug)]
struct LaneProgress {
    max_tokens: usize,
    generated: Vec<i32>,
    stop_reason: Option<StopReason>,
    logical_transitions: usize,
    padding_transitions: usize,
}

impl LaneProgress {
    fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            generated: Vec::with_capacity(max_tokens),
            stop_reason: None,
            logical_transitions: 0,
            padding_transitions: 0,
        }
    }

    fn is_active(&self) -> bool {
        self.stop_reason.is_none()
    }

    fn transition_token(&self) -> i32 {
        self.generated
            .last()
            .copied()
            .filter(|_| self.is_active())
            .unwrap_or(PAD_TOKEN)
    }

    /// Record one selected token and return whether its piece is visible.
    fn record_selection(&mut self, token: i32, stop_tokens: &[i32]) -> Result<bool> {
        ensure!(
            self.is_active(),
            "cannot select into a finished dense B=8 lane"
        );
        ensure!(
            self.generated.len() < self.max_tokens,
            "dense B=8 lane exceeded its token limit"
        );
        self.generated.push(token);
        if stop_tokens.contains(&token) {
            self.stop_reason = Some(StopReason::Eos);
            return Ok(false);
        }
        if self.generated.len() == self.max_tokens {
            self.stop_reason = Some(StopReason::TokenLimit);
        }
        Ok(true)
    }

    fn record_batch_step(
        &mut self,
        was_active: bool,
        selected: i32,
        stop_tokens: &[i32],
    ) -> Result<bool> {
        ensure!(
            was_active == self.is_active(),
            "dense B=8 active-lane snapshot drifted"
        );
        if was_active {
            self.logical_transitions += 1;
            self.record_selection(selected, stop_tokens)
        } else {
            self.padding_transitions += 1;
            Ok(false)
        }
    }

    fn validate_complete(&self, physical_batch_steps: usize) -> Result<()> {
        ensure!(
            self.stop_reason.is_some(),
            "dense B=8 lane did not terminate"
        );
        ensure!(
            self.logical_transitions.checked_add(1) == Some(self.generated.len()),
            "dense B=8 lane violated N-1 transition semantics"
        );
        ensure!(
            self.logical_transitions + self.padding_transitions == physical_batch_steps,
            "dense B=8 physical transition accounting drifted"
        );
        Ok(())
    }
}

struct Lane {
    id: String,
    prompt_tokens: usize,
    sequence: Sequence,
    progress: LaneProgress,
    generated_text: String,
}

#[derive(Debug, Serialize)]
struct CohortTelemetry {
    schema_version: u32,
    backend: &'static str,
    cohort_index: usize,
    width: usize,
    prompt_tokens_per_request: usize,
    requested_tokens_per_request: usize,
    generated_tokens: usize,
    productive_transitions: usize,
    padding_transitions: usize,
    physical_batch_steps: usize,
    common_prefix_tokens: usize,
    prefix_fanout_tokens: usize,
    prefix_fanout_reason: &'static str,
    prefix_fanout_min_tokens: usize,
    prefix_snapshot_bytes: u64,
    prefix_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    suffix_prefill_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    batch_transition_ms: f64,
    executor_gpu_ms: Option<f64>,
    aggregate_generated_tps: f64,
    sequence_allocation_delta_bytes: u64,
    remaining_sequence_required_bytes: u64,
    prefix_snapshot_required_bytes: u64,
    memory_admission_incremental_bytes: u64,
    memory_admission_reserve_bytes: u64,
    memory_admission_required_bytes: Option<u64>,
    memory_admission_reason: &'static str,
}

fn common_prefix_tokens(requests: &[PreparedJsonlRequest]) -> usize {
    let Some(first) = requests.first() else {
        return 0;
    };
    requests
        .iter()
        .skip(1)
        .fold(first.prompt_ids.len(), |len, request| {
            first.prompt_ids[..len]
                .iter()
                .zip(&request.prompt_ids)
                .take_while(|(left, right)| left == right)
                .count()
        })
}

fn plan_prefix_fanout(requests: &[PreparedJsonlRequest], enabled: bool) -> PrefixFanoutPlan {
    let common_prefix_tokens = common_prefix_tokens(requests);
    if !enabled {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "disabled",
        };
    }
    if common_prefix_tokens < PREFIX_FANOUT_MIN_TOKENS {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "below_minimum",
        };
    }
    PrefixFanoutPlan {
        common_prefix_tokens,
        selected_prefix_tokens: common_prefix_tokens,
        reason: "selected",
    }
}

fn align_prefix_fanout(
    mut plan: PrefixFanoutPlan,
    prompt_tokens: usize,
    prefill_chunk: usize,
) -> PrefixFanoutPlan {
    if plan.selected_prefix_tokens == 0 || plan.selected_prefix_tokens == prompt_tokens {
        return plan;
    }
    let aligned = plan.selected_prefix_tokens / prefill_chunk * prefill_chunk;
    if aligned < PREFIX_FANOUT_MIN_TOKENS {
        plan.selected_prefix_tokens = 0;
        plan.reason = "alignment_below_minimum";
    } else if aligned != plan.selected_prefix_tokens {
        plan.selected_prefix_tokens = aligned;
        plan.reason = "selected_chunk_aligned";
    }
    plan
}

fn parse_prefix_fanout_enabled(value: Option<&OsStr>) -> bool {
    let Some(value) = value.and_then(OsStr::to_str) else {
        return true;
    };
    !matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "0" | "false" | "no" | "off"
    )
}

pub(super) fn validate_cli(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    let Some(batch_size) = args.batch_size else {
        return Ok(());
    };
    ensure!(
        batch_size == DENSE_BATCH8_WIDTH,
        "--batch-size currently requires {DENSE_BATCH8_WIDTH}, got {batch_size}"
    );
    ensure!(!args.info, "--batch-size cannot be used with --info");
    let requests_path = args
        .requests_jsonl
        .as_deref()
        .context("--batch-size requires --requests-jsonl")?;
    ensure!(
        requests_path != Path::new("-"),
        "--batch-size requires a JSONL file; streaming stdin cohorts are not yet supported"
    );
    ensure!(
        matches!(args.prefill_chunk, PrefillChunkArg::Fixed(_)),
        "--batch-size currently requires a fixed --prefill-chunk"
    );
    ensure!(
        args.temperature == 0.0,
        "--batch-size currently requires greedy decoding (--temp 0)"
    );
    ensure!(
        !args.prompt_lookup,
        "--batch-size does not yet compose with --prompt-lookup"
    );
    ensure!(
        args.request_stats.is_none() && args.request_stats_jsonl.is_none(),
        "--batch-size does not yet emit per-request stats sidecars"
    );
    ensure!(
        args.trace_request.is_none(),
        "--batch-size does not yet emit request-trace rows"
    );
    ensure!(
        args.cache_prefix_tokens.is_none()
            && !explicit.prefix_cache_max_mib
            && !explicit.cache_prefix_auto_min_tokens,
        "--batch-size does not yet compose with prefix-cache configuration"
    );
    ensure!(
        args.durable_prefix_cache.is_none(),
        "--batch-size does not yet compose with durable prefix caching"
    );
    validate_greedy_gpu_mode(configured_greedy_gpu_argmax_mode())?;
    Ok(())
}

fn validate_greedy_gpu_mode(mode: GreedyGpuArgmaxMode) -> Result<()> {
    ensure!(
        mode != GreedyGpuArgmaxMode::ExplicitRollback,
        "{GREEDY_GPU_ARGMAX_ENV}=0 disables --batch-size because fixed B=8 requires GPU greedy selection"
    );
    Ok(())
}

pub(super) fn validate_model_family(
    batch_size: Option<usize>,
    model_family: Option<ModelFamily>,
) -> Result<()> {
    ensure!(
        batch_size.is_none() || model_family == Some(ModelFamily::Qwen35),
        "--batch-size currently requires a dense Qwen model"
    );
    Ok(())
}

pub(super) fn run_file(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    requests_path: &Path,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
) -> Result<usize> {
    validate_greedy_gpu_mode(greedy_gpu_mode)?;
    let requests = prepare_jsonl_requests(requests_path, tokenizer, args)?;
    validate_requests(&requests, args)?;
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let mut executor = loaded
        .create_dense_batch8_executor()
        .context("create dense B=8 executor")?;
    let mut completed = 0usize;
    for (cohort_index, cohort) in requests.chunks_exact(DENSE_BATCH8_WIDTH).enumerate() {
        shutdown::checkpoint()?;
        let cohort: &[PreparedJsonlRequest; DENSE_BATCH8_WIDTH] =
            cohort.try_into().expect("validated dense B=8 cohort width");
        let (outputs, telemetry) = run_cohort(
            loaded,
            tokenizer,
            &mut executor,
            cohort,
            args,
            &stop_tokens,
            cohort_index,
        )
        .with_context(|| format!("run dense B=8 cohort {cohort_index}"))?;
        shutdown::checkpoint()?;
        let mut encoded = Vec::new();
        for output in &outputs {
            serde_json::to_writer(&mut encoded, output).context("encode request output")?;
            encoded.push(b'\n');
        }
        stdout
            .write_all(&encoded)
            .context("write dense B=8 cohort outputs")?;
        stdout.flush()?;
        eprintln!(
            "dense_batch8: {}",
            serde_json::to_string(&telemetry).context("serialize dense B=8 telemetry")?
        );
        completed += outputs.len();
    }
    Ok(completed)
}

fn validate_requests(requests: &[PreparedJsonlRequest], args: &Args) -> Result<()> {
    ensure!(
        !requests.is_empty() && requests.len().is_multiple_of(DENSE_BATCH8_WIDTH),
        "--batch-size {DENSE_BATCH8_WIDTH} requires a non-empty request count divisible by {DENSE_BATCH8_WIDTH}; got {}",
        requests.len()
    );
    for request in requests {
        ensure!(
            request.request.cache_prefix_tokens.is_none(),
            "request {} sets cache_prefix_tokens, which is unsupported with --batch-size",
            request.id
        );
        ensure!(
            request.sampling.temperature == 0.0,
            "request {} uses temperature {}; --batch-size currently requires greedy decoding",
            request.id,
            request.sampling.temperature
        );
        ensure!(
            request.auto_cache_prefix_tokens.is_none() && request.auto_cache_future_hits == 0,
            "request {} carried prefix-cache lookahead into dense B=8 admission",
            request.id
        );
        jsonl_generation_capacity(request, args)?;
    }
    for (cohort_index, cohort) in requests.chunks_exact(DENSE_BATCH8_WIDTH).enumerate() {
        let expected_prompt = cohort[0].prompt_ids.len();
        let expected_tokens = cohort[0].request.tokens.unwrap_or(args.tokens);
        for (slot, request) in cohort.iter().enumerate().skip(1) {
            ensure!(
                request.prompt_ids.len() == expected_prompt,
                "dense B=8 cohort {cohort_index} slot {slot} prompt length {} != {expected_prompt}",
                request.prompt_ids.len()
            );
            let requested_tokens = request.request.tokens.unwrap_or(args.tokens);
            ensure!(
                requested_tokens == expected_tokens,
                "dense B=8 cohort {cohort_index} slot {slot} token limit {requested_tokens} != {expected_tokens}"
            );
        }
    }
    Ok(())
}

fn run_cohort(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    executor: &mut DenseBatch8SequenceExecutor<'_>,
    requests: &[PreparedJsonlRequest; DENSE_BATCH8_WIDTH],
    args: &Args,
    stop_tokens: &[i32],
    cohort_index: usize,
) -> Result<(Vec<RequestOutput>, CohortTelemetry)> {
    let prompt_tokens = requests[0].prompt_ids.len();
    let (requested_tokens, capacity) = jsonl_generation_capacity(&requests[0], args)?;
    let chunk = match args.prefill_chunk {
        PrefillChunkArg::Fixed(requested) => requested.min(prompt_tokens.max(1)),
        PrefillChunkArg::Auto => bail!("dense B=8 requires a fixed prefill chunk"),
    };
    ensure!(chunk > 0, "dense B=8 prefill chunk must be nonzero");
    let mut prefix_fanout = align_prefix_fanout(
        plan_prefix_fanout(
            requests,
            parse_prefix_fanout_enabled(std::env::var_os(PREFIX_FANOUT_ENV).as_deref()),
        ),
        prompt_tokens,
        chunk,
    );

    let mut scratch = allocate_legacy_prefill_scratch(loaded, chunk, prompt_tokens)?;
    let mut sequences = Vec::with_capacity(DENSE_BATCH8_WIDTH);
    let before_first_sequence = loaded.context().current_allocated_size();
    sequences.push(
        loaded
            .create_sequence(SequenceConfig::new(capacity))
            .context("allocate first dense B=8 sequence")?,
    );
    let after_first_sequence = loaded.context().current_allocated_size();
    let sequence_allocation_delta_bytes = after_first_sequence
        .checked_sub(before_first_sequence)
        .filter(|&bytes| bytes > 0)
        .context("dense B=8 sequence allocation did not produce a valid Metal byte delta")?;
    let remaining_sequence_required_bytes = sequence_allocation_delta_bytes
        .checked_mul((DENSE_BATCH8_WIDTH - 1) as u64)
        .context("dense B=8 remaining sequence byte estimate overflow")?;
    let mut prefix_snapshot_required_bytes = if prefix_fanout.selected_prefix_tokens > 0 {
        loaded
            .estimate_checkpoint_boundary_sizes(
                &sequences[0],
                prefix_fanout.selected_prefix_tokens,
                false,
                false,
            )
            .context("estimate dense B=8 prefix fanout snapshot")?
            .snapshot_bytes
    } else {
        0
    };
    let mut memory_admission_incremental_bytes = remaining_sequence_required_bytes
        .checked_add(prefix_snapshot_required_bytes)
        .context("dense B=8 fanout admission byte overflow")?;
    let mut memory_admission = evaluate_metal_memory_admission(
        memory_admission_incremental_bytes,
        TRANSIENT_RESERVE_BYTES,
        loaded.context().memory_signals(),
        true,
    );
    if !memory_admission.admitted && prefix_fanout.selected_prefix_tokens > 0 {
        let without_fanout = evaluate_metal_memory_admission(
            remaining_sequence_required_bytes,
            TRANSIENT_RESERVE_BYTES,
            loaded.context().memory_signals(),
            true,
        );
        if without_fanout.admitted {
            prefix_fanout.selected_prefix_tokens = 0;
            prefix_fanout.reason = "memory_fallback";
            prefix_snapshot_required_bytes = 0;
            memory_admission_incremental_bytes = remaining_sequence_required_bytes;
            memory_admission = without_fanout;
        }
    }
    ensure!(
        memory_admission.admitted,
        "dense B=8 memory admission denied: reason={} required_bytes={:?} working_set_headroom_bytes={:?} process_limit_remaining_bytes={:?}",
        memory_admission.reason.as_str(),
        memory_admission.required_bytes,
        memory_admission.working_set_headroom_bytes,
        memory_admission.signals.process_limit_remaining_bytes,
    );
    for _ in 1..DENSE_BATCH8_WIDTH {
        sequences.push(
            loaded
                .create_sequence(SequenceConfig::new(capacity))
                .context("allocate admitted dense B=8 sequence")?,
        );
    }

    let forward = loaded.forward();
    let prefill_t0 = Instant::now();
    let mut prompt_logits = Vec::with_capacity(DENSE_BATCH8_WIDTH);
    let mut prefix_snapshot_bytes = 0u64;
    let mut prefix_prefill_ms = 0.0;
    let mut prefix_snapshot_ms = 0.0;
    let mut prefix_restore_ms = 0.0;
    let mut suffix_prefill_ms = 0.0;
    if prefix_fanout.selected_prefix_tokens > 0 {
        let prefix_len = prefix_fanout.selected_prefix_tokens;
        let (prefix_logits, ms) = prefill_span(
            &forward,
            &mut sequences[0],
            &mut scratch,
            &requests[0].prompt_ids[..prefix_len],
            0,
        )
        .context("prefill dense B=8 shared prefix")?;
        prefix_prefill_ms = ms;

        let snapshot_t0 = Instant::now();
        let prepared = loaded
            .prepare_checkpoint_boundary(
                &sequences[0],
                requests[0].prompt_ids[..prefix_len].to_vec(),
                None,
                None,
            )
            .context("capture dense B=8 shared prefix")?;
        prefix_snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;
        prefix_snapshot_bytes = prepared.snapshot_bytes();
        ensure!(
            prefix_snapshot_bytes == prefix_snapshot_required_bytes,
            "dense B=8 prefix snapshot bytes {} != estimate {}",
            prefix_snapshot_bytes,
            prefix_snapshot_required_bytes,
        );

        for slot in 0..DENSE_BATCH8_WIDTH {
            shutdown::checkpoint()?;
            if slot > 0 {
                let restore_t0 = Instant::now();
                let restored = loaded
                    .restore_prepared_checkpoint(
                        &prepared,
                        &mut sequences[slot],
                        &requests[slot].prompt_ids,
                    )
                    .with_context(|| format!("restore dense B=8 shared prefix into slot {slot}"))?;
                prefix_restore_ms += restore_t0.elapsed().as_secs_f64() * 1e3;
                ensure!(
                    restored.matched_prefix_len == prefix_len
                        && restored.restored_prefix_len == prefix_len,
                    "dense B=8 slot {slot} restored an unexpected prefix boundary"
                );
            }
            let suffix = &requests[slot].prompt_ids[prefix_len..];
            if suffix.is_empty() {
                prompt_logits.push(prefix_logits.clone());
            } else {
                let (logits, ms) = prefill_span(
                    &forward,
                    &mut sequences[slot],
                    &mut scratch,
                    suffix,
                    prefix_len,
                )
                .with_context(|| format!("prefill dense B=8 cohort suffix slot {slot}"))?;
                suffix_prefill_ms += ms;
                prompt_logits.push(logits);
            }
        }
    } else {
        for (slot, (request, sequence)) in requests.iter().zip(&mut sequences).enumerate() {
            shutdown::checkpoint()?;
            let (logits, ms) =
                prefill_span(&forward, sequence, &mut scratch, &request.prompt_ids, 0)
                    .with_context(|| format!("prefill dense B=8 cohort slot {slot}"))?;
            suffix_prefill_ms += ms;
            prompt_logits.push(logits);
        }
    }
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
    drop(scratch);
    {
        let sequences: &mut [Sequence; DENSE_BATCH8_WIDTH] = sequences
            .as_mut_slice()
            .try_into()
            .expect("dense B=8 sequence width");
        let [s0, s1, s2, s3, s4, s5, s6, s7] = sequences;
        executor
            .validate(
                [PAD_TOKEN; DENSE_BATCH8_WIDTH],
                [s0, s1, s2, s3, s4, s5, s6, s7],
            )
            .context("validate dense B=8 cohort backend")?;
    }

    let decode_t0 = Instant::now();
    let mut lanes = Vec::with_capacity(DENSE_BATCH8_WIDTH);
    for ((request, sequence), logits) in requests.iter().zip(sequences).zip(prompt_logits) {
        let mut sampler =
            Sampler::new(request.sampling).context("initialize dense B=8 greedy sampler")?;
        let first = sampler
            .sample(&logits)
            .context("select dense B=8 first token")?
            .token;
        let mut progress = LaneProgress::new(requested_tokens);
        let visible = progress.record_selection(first, stop_tokens)?;
        let mut generated_text = String::new();
        if visible {
            generated_text.push_str(&tokenizer.decode_piece(first));
        }
        lanes.push(Lane {
            id: request.id.clone(),
            prompt_tokens,
            sequence,
            progress,
            generated_text,
        });
    }

    let mut physical_batch_steps = 0usize;
    let mut batch_transition_ms = 0.0;
    let mut executor_gpu_ms = Some(0.0);
    while lanes.iter().any(|lane| lane.progress.is_active()) {
        shutdown::checkpoint()?;
        let active: [bool; DENSE_BATCH8_WIDTH] =
            std::array::from_fn(|slot| lanes[slot].progress.is_active());
        let token_ids: [i32; DENSE_BATCH8_WIDTH] =
            std::array::from_fn(|slot| lanes[slot].progress.transition_token());
        let transition_t0 = Instant::now();
        let step = {
            let lanes: &mut [Lane; DENSE_BATCH8_WIDTH] = lanes
                .as_mut_slice()
                .try_into()
                .expect("dense B=8 lane width");
            let [l0, l1, l2, l3, l4, l5, l6, l7] = lanes;
            executor.step_greedy(
                token_ids,
                [
                    &mut l0.sequence,
                    &mut l1.sequence,
                    &mut l2.sequence,
                    &mut l3.sequence,
                    &mut l4.sequence,
                    &mut l5.sequence,
                    &mut l6.sequence,
                    &mut l7.sequence,
                ],
                || shutdown::checkpoint().is_err(),
            )?
        };
        shutdown::checkpoint()?;
        batch_transition_ms += transition_t0.elapsed().as_secs_f64() * 1e3;
        executor_gpu_ms = match (executor_gpu_ms, step.gpu_ms) {
            (Some(total), Some(step)) => Some(total + step),
            _ => None,
        };
        physical_batch_steps += 1;

        for (slot, lane) in lanes.iter_mut().enumerate() {
            let token = step.argmax_ids[slot];
            if lane
                .progress
                .record_batch_step(active[slot], token, stop_tokens)?
            {
                lane.generated_text.push_str(&tokenizer.decode_piece(token));
            }
        }
    }
    let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;

    let generated_tokens = lanes
        .iter()
        .map(|lane| lane.progress.generated.len())
        .sum::<usize>();
    let productive_transitions = lanes
        .iter()
        .map(|lane| lane.progress.logical_transitions)
        .sum::<usize>();
    let padding_transitions = lanes
        .iter()
        .map(|lane| lane.progress.padding_transitions)
        .sum::<usize>();
    let aggregate_generated_tps = if decode_ms > 0.0 {
        generated_tokens as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let mut outputs = Vec::with_capacity(DENSE_BATCH8_WIDTH);
    for lane in lanes {
        lane.progress
            .validate_complete(physical_batch_steps)
            .with_context(|| format!("validate dense B=8 lane {}", lane.id))?;
        ensure!(
            lane.sequence.position() == lane.prompt_tokens + physical_batch_steps,
            "dense B=8 lane {} sequence frontier drifted",
            lane.id
        );
        let stop_reason = lane
            .progress
            .stop_reason
            .expect("validated dense B=8 lane termination");
        outputs.push(RequestOutput {
            id: lane.id,
            prompt_tokens: lane.prompt_tokens,
            generated_tokens: lane.progress.generated.len(),
            generated_token_sha256: generated_token_sha256(&lane.progress.generated),
            generated_text: lane.generated_text,
            stop_reason,
            terminal_token_target_transition_consumed: false,
        });
    }
    Ok((
        outputs,
        CohortTelemetry {
            schema_version: 2,
            backend: "dense_qwen_static_batch8_v2",
            cohort_index,
            width: DENSE_BATCH8_WIDTH,
            prompt_tokens_per_request: prompt_tokens,
            requested_tokens_per_request: requested_tokens,
            generated_tokens,
            productive_transitions,
            padding_transitions,
            physical_batch_steps,
            common_prefix_tokens: prefix_fanout.common_prefix_tokens,
            prefix_fanout_tokens: prefix_fanout.selected_prefix_tokens,
            prefix_fanout_reason: prefix_fanout.reason,
            prefix_fanout_min_tokens: PREFIX_FANOUT_MIN_TOKENS,
            prefix_snapshot_bytes,
            prefix_prefill_ms,
            prefix_snapshot_ms,
            prefix_restore_ms,
            suffix_prefill_ms,
            prefill_ms,
            decode_ms,
            batch_transition_ms,
            executor_gpu_ms,
            aggregate_generated_tps,
            sequence_allocation_delta_bytes,
            remaining_sequence_required_bytes,
            prefix_snapshot_required_bytes,
            memory_admission_incremental_bytes,
            memory_admission_reserve_bytes: memory_admission.reserve_bytes,
            memory_admission_required_bytes: memory_admission.required_bytes,
            memory_admission_reason: memory_admission.reason.as_str(),
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    fn test_args() -> Args {
        Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "requests.jsonl",
            "--batch-size",
            "8",
            "--tokens",
            "3",
        ])
        .expect("valid dense B=8 test arguments")
    }

    fn prepared(id: &str, tokens: &[i32]) -> PreparedJsonlRequest {
        PreparedJsonlRequest {
            request: JsonlRequest {
                id: Some(id.to_string()),
                prompt: None,
                prompt_file: None,
                tokens: None,
                cache_prefix_tokens: None,
                sampling: None,
            },
            id: id.to_string(),
            line: 1,
            prompt_ids: tokens.to_vec(),
            sampling: SamplingConfig::default(),
            auto_cache_prefix_tokens: None,
            auto_cache_future_hits: 0,
        }
    }

    #[test]
    fn cli_contract_is_explicit_and_family_scoped() {
        let args = test_args();
        validate_cli(&args, ExplicitCliOptions::default()).unwrap();
        validate_model_family(args.batch_size, Some(ModelFamily::Qwen35)).unwrap();
        assert!(validate_model_family(args.batch_size, Some(ModelFamily::Qwen35Moe)).is_err());
        assert!(validate_model_family(args.batch_size, Some(ModelFamily::DeepSeek4)).is_err());
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(OsStr::new("0"))),
            GreedyGpuArgmaxMode::ExplicitRollback
        );
        assert!(validate_greedy_gpu_mode(GreedyGpuArgmaxMode::ExplicitRollback).is_err());
        validate_greedy_gpu_mode(GreedyGpuArgmaxMode::DefaultOff).unwrap();
        validate_greedy_gpu_mode(GreedyGpuArgmaxMode::ForceEnabled).unwrap();

        let mut invalid = test_args();
        invalid.batch_size = Some(4);
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        invalid = test_args();
        invalid.requests_jsonl = Some(PathBuf::from("-"));
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        invalid = test_args();
        invalid.temperature = 0.7;
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        invalid = test_args();
        invalid.info = true;
        assert!(validate_cli(&invalid, ExplicitCliOptions::default()).is_err());
        assert!(
            validate_cli(
                &test_args(),
                ExplicitCliOptions {
                    prefix_cache_max_mib: true,
                    ..ExplicitCliOptions::default()
                }
            )
            .is_err()
        );
    }

    #[test]
    fn request_admission_requires_uniform_full_cohorts() {
        let args = test_args();
        let mut requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| prepared(&format!("slot-{slot}"), &[1, 2]))
            .collect::<Vec<_>>();
        validate_requests(&requests, &args).unwrap();
        assert!(validate_requests(&requests[..7], &args).is_err());

        requests[3].prompt_ids.push(3);
        assert!(validate_requests(&requests, &args).is_err());
        requests[3].prompt_ids.pop();
        requests[4].request.tokens = Some(4);
        assert!(validate_requests(&requests, &args).is_err());
        requests[4].request.tokens = None;
        requests[5].sampling.temperature = 0.1;
        assert!(validate_requests(&requests, &args).is_err());
        requests[5].sampling.temperature = 0.0;
        requests[6].request.cache_prefix_tokens = Some(1);
        assert!(validate_requests(&requests, &args).is_err());
    }

    #[test]
    fn prefix_fanout_requires_an_eight_way_minimum_prefix() {
        assert!(parse_prefix_fanout_enabled(None));
        assert!(parse_prefix_fanout_enabled(Some(OsStr::new("1"))));
        assert!(!parse_prefix_fanout_enabled(Some(OsStr::new("off"))));
        let shared = (0..PREFIX_FANOUT_MIN_TOKENS as i32).collect::<Vec<_>>();
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut tokens = shared.clone();
                tokens.push(10_000 + slot as i32);
                prepared(&format!("slot-{slot}"), &tokens)
            })
            .collect::<Vec<_>>();
        assert_eq!(common_prefix_tokens(&requests), PREFIX_FANOUT_MIN_TOKENS);
        assert_eq!(
            plan_prefix_fanout(&requests, true),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected",
            }
        );
        assert_eq!(
            plan_prefix_fanout(&requests, false),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: 0,
                reason: "disabled",
            }
        );
        assert_eq!(
            align_prefix_fanout(plan_prefix_fanout(&requests, true), 257, 128),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected",
            }
        );

        let wider = requests
            .iter()
            .enumerate()
            .map(|(slot, request)| {
                let mut tokens = request.prompt_ids.clone();
                tokens.splice(
                    PREFIX_FANOUT_MIN_TOKENS..PREFIX_FANOUT_MIN_TOKENS,
                    [8, 9, 10, 11],
                );
                prepared(&format!("wider-{slot}"), &tokens)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            align_prefix_fanout(plan_prefix_fanout(&wider, true), 261, 256),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            align_prefix_fanout(plan_prefix_fanout(&wider, true), 261, 200),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: 0,
                reason: "alignment_below_minimum",
            }
        );

        let short = requests
            .iter()
            .enumerate()
            .map(|(slot, request)| prepared(&format!("short-{slot}"), &request.prompt_ids[1..]))
            .collect::<Vec<_>>();
        assert_eq!(
            plan_prefix_fanout(&short, true),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS - 1,
                selected_prefix_tokens: 0,
                reason: "below_minimum",
            }
        );
    }

    #[test]
    fn lane_progress_separates_logical_work_from_padding() {
        let mut one = LaneProgress::new(1);
        assert!(one.record_selection(10, &[99]).unwrap());
        assert_eq!(one.stop_reason, Some(StopReason::TokenLimit));
        one.validate_complete(0).unwrap();

        let mut transition_eos = LaneProgress::new(3);
        assert!(transition_eos.record_selection(10, &[99]).unwrap());
        assert!(!transition_eos.record_batch_step(true, 99, &[99]).unwrap());
        assert_eq!(transition_eos.stop_reason, Some(StopReason::Eos));
        transition_eos.validate_complete(1).unwrap();

        let mut full = LaneProgress::new(3);
        assert!(full.record_selection(10, &[99]).unwrap());
        assert!(full.record_batch_step(true, 11, &[99]).unwrap());
        assert!(full.record_batch_step(true, 12, &[99]).unwrap());
        assert_eq!(full.stop_reason, Some(StopReason::TokenLimit));
        assert_eq!(full.generated, [10, 11, 12]);
        full.validate_complete(2).unwrap();

        let mut early = LaneProgress::new(3);
        assert!(!early.record_selection(99, &[99]).unwrap());
        assert_eq!(early.stop_reason, Some(StopReason::Eos));
        assert_eq!(early.transition_token(), PAD_TOKEN);
        assert!(!early.record_batch_step(false, 7, &[99]).unwrap());
        assert!(!early.record_batch_step(false, 8, &[99]).unwrap());
        assert_eq!(early.generated, [99]);
        assert_eq!(early.logical_transitions, 0);
        assert_eq!(early.padding_transitions, 2);
        early.validate_complete(2).unwrap();
    }
}
