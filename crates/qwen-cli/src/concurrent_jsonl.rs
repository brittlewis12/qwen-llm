use super::*;
use qwen_llm::runtime::IndependentQueue2SequenceExecutor;

const WIDTH: usize = 2;

#[derive(Debug)]
struct LaneProgress {
    max_tokens: usize,
    generated: Vec<i32>,
    stop_reason: Option<StopReason>,
    transitions: usize,
}

impl LaneProgress {
    fn new(max_tokens: usize) -> Self {
        Self {
            max_tokens,
            generated: Vec::with_capacity(max_tokens),
            stop_reason: None,
            transitions: 0,
        }
    }

    fn is_active(&self) -> bool {
        self.stop_reason.is_none()
    }

    fn pending_token(&self) -> Result<i32> {
        ensure!(
            self.is_active(),
            "finished concurrent lane has no pending token"
        );
        self.generated
            .last()
            .copied()
            .context("active concurrent lane has no generated token")
    }

    fn record_selection(&mut self, token: i32, stop_tokens: &[i32]) -> Result<bool> {
        ensure!(
            self.is_active(),
            "cannot select into a finished concurrent lane"
        );
        ensure!(
            self.generated.len() < self.max_tokens,
            "concurrent lane exceeded its token limit"
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

    fn validate_complete(&self) -> Result<()> {
        ensure!(
            self.stop_reason.is_some(),
            "concurrent lane did not terminate"
        );
        ensure!(
            self.transitions.checked_add(1) == Some(self.generated.len()),
            "concurrent lane violated N-1 transition semantics"
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeWork {
    Pair,
    Serial(usize),
    Complete,
}

fn next_decode_work(active: [bool; WIDTH]) -> DecodeWork {
    match active {
        [true, true] => DecodeWork::Pair,
        [true, false] => DecodeWork::Serial(0),
        [false, true] => DecodeWork::Serial(1),
        [false, false] => DecodeWork::Complete,
    }
}

struct PreparedLane {
    id: String,
    prompt_tokens: usize,
    max_tokens: usize,
    sequence: Sequence,
    logits: Vec<f32>,
    sampling: SamplingConfig,
}

#[derive(Debug, Serialize)]
struct PairTelemetry {
    schema_version: u32,
    backend: &'static str,
    pair_index: usize,
    prompt_tokens: [usize; WIDTH],
    requested_tokens: [usize; WIDTH],
    generated_tokens: usize,
    productive_transitions: usize,
    paired_transitions: usize,
    serial_tail_transitions: usize,
    prepare_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    executor_gpu_ms: [Option<f64>; WIDTH],
    aggregate_generated_tps: f64,
    aggregate_transition_tps: f64,
}

pub(super) fn validate_cli(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    validate_cli_with_greedy_mode(args, explicit, configured_greedy_gpu_argmax_mode())
}

fn validate_cli_with_greedy_mode(
    args: &Args,
    explicit: ExplicitCliOptions,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
) -> Result<()> {
    let Some(concurrency) = args.concurrency else {
        return Ok(());
    };
    ensure!(
        concurrency == WIDTH,
        "--concurrency currently requires {WIDTH}, got {concurrency}"
    );
    ensure!(!args.info, "--concurrency cannot be used with --info");
    let requests_path = args
        .requests_jsonl
        .as_deref()
        .context("--concurrency requires --requests-jsonl")?;
    ensure!(
        requests_path != Path::new("-"),
        "--concurrency requires a regular JSONL file; streaming stdin is not yet supported"
    );
    let metadata = std::fs::metadata(requests_path)
        .with_context(|| format!("inspect requests JSONL {}", requests_path.display()))?;
    ensure!(
        metadata.is_file(),
        "--concurrency requires a regular JSONL file, got {}",
        requests_path.display()
    );
    ensure!(
        args.temperature == 0.0,
        "--concurrency currently requires greedy decoding (--temp 0)"
    );
    ensure!(
        !args.prompt_lookup,
        "--concurrency does not yet compose with --prompt-lookup"
    );
    ensure!(
        args.request_stats.is_none() && args.request_stats_jsonl.is_none(),
        "--concurrency does not yet emit per-request stats sidecars"
    );
    ensure!(
        args.trace_request.is_none(),
        "--concurrency does not yet emit request-trace rows"
    );
    ensure!(
        args.cache_prefix_tokens.is_none()
            && !explicit.prefix_cache_max_mib
            && !explicit.cache_prefix_auto_min_tokens,
        "--concurrency does not yet compose with prefix-cache configuration"
    );
    ensure!(
        args.durable_prefix_cache.is_none()
            && !explicit.durable_prefix_cache_max_mib
            && !explicit.durable_prefix_cache_max_entry_mib
            && !explicit.durable_prefix_cache_min_tokens,
        "--concurrency does not yet compose with durable prefix caching"
    );
    validate_greedy_gpu_mode(greedy_gpu_mode)
}

fn validate_greedy_gpu_mode(mode: GreedyGpuArgmaxMode) -> Result<()> {
    ensure!(
        mode != GreedyGpuArgmaxMode::ExplicitRollback,
        "{GREEDY_GPU_ARGMAX_ENV}=0 disables --concurrency because paired decode requires GPU greedy selection"
    );
    Ok(())
}

pub(super) fn validate_model_family(
    concurrency: Option<usize>,
    model_family: Option<ModelFamily>,
) -> Result<()> {
    ensure!(
        concurrency.is_none()
            || matches!(
                model_family,
                Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)
            ),
        "--concurrency currently requires a Qwen model"
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
    let max_capacity = requests
        .iter()
        .map(|request| jsonl_generation_capacity(request, args).map(|(_, capacity)| capacity))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("concurrent JSONL file contained no requests")?;
    let prefill_scratch_upper_bytes = requests
        .iter()
        .map(|request| prefill_scratch_upper_bytes(loaded, request, args))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("concurrent JSONL file contained no requests")?;
    let pair_count = requests.len() / WIDTH;
    let mut executor = if pair_count > 0 {
        let admission = loaded
            .admit_independent_queue2(max_capacity, prefill_scratch_upper_bytes)
            .context("admit two independent Qwen sessions")?;
        eprintln!(
            "concurrency_admission: width={WIDTH} prefill_scratch_upper_bytes={} reason={} required_bytes={:?} working_set_headroom_bytes={:?}",
            prefill_scratch_upper_bytes,
            admission.reason.as_str(),
            admission.required_bytes,
            admission.working_set_headroom_bytes,
        );
        Some(
            loaded
                .create_independent_queue2_executor()
                .context("create independent Qwen queue executor")?,
        )
    } else {
        None
    };
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;

    let mut completed = 0usize;
    let mut pairs = requests.chunks_exact(WIDTH);
    for (pair_index, pair) in pairs.by_ref().enumerate() {
        shutdown::checkpoint()?;
        let pair = [&pair[0], &pair[1]];
        let (outputs, telemetry) = run_pair(
            loaded,
            tokenizer,
            executor.as_mut().expect("planned concurrent executor"),
            pair,
            args,
            &stop_tokens,
            pair_index,
        )
        .with_context(|| format!("run concurrent request pair {pair_index}"))?;
        write_outputs(stdout, &outputs)?;
        eprintln!(
            "concurrency_pair: {}",
            serde_json::to_string(&telemetry).context("serialize concurrency telemetry")?
        );
        completed += WIDTH;
    }

    if let Some(request) = pairs.remainder().first() {
        shutdown::checkpoint()?;
        let (output, _) = run_jsonl_request(loaded, tokenizer, request, args, greedy_gpu_mode)
            .with_context(|| format!("run concurrent serial tail request {}", request.id))?;
        write_outputs(stdout, std::slice::from_ref(&output))?;
        completed += 1;
    }
    ensure!(
        completed == requests.len(),
        "concurrent JSONL completed {completed}/{} requests",
        requests.len()
    );
    Ok(completed)
}

fn price_prefill_scratch_plan(
    loaded: &LoadedModel,
    chunk: usize,
    prompt_tokens: usize,
    config: PrefillScratchConfig,
) -> Result<u64> {
    let plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        loaded.metal_model(),
        u32::try_from(chunk).context("prefill chunk does not fit u32")?,
        prompt_tokens.max(chunk),
        config,
    )?;
    plan.priced_upper_bound(|bytes| Ok(loaded.context().shared_buffer_size_and_align(bytes)?.size))
        .map_err(anyhow::Error::from)
}

fn prefill_scratch_upper_bytes(
    loaded: &LoadedModel,
    request: &PreparedJsonlRequest,
    args: &Args,
) -> Result<u64> {
    let prompt_tokens = request.prompt_ids.len();
    if let PrefillChunkArg::Fixed(requested) = args.prefill_chunk {
        return price_prefill_scratch_plan(
            loaded,
            requested.min(prompt_tokens.max(1)),
            prompt_tokens,
            PrefillScratchConfig::default(),
        );
    }

    let baseline = baseline_prefill_chunk(prompt_tokens);
    let mut upper = price_prefill_scratch_plan(
        loaded,
        baseline,
        prompt_tokens,
        PrefillScratchConfig::default(),
    )?;
    let profile = auto_prefill_profile(
        loaded.arch(),
        loaded.gguf().get_str("general.base_model.0.name"),
        loaded.gguf().get_u64("general.file_type"),
    );
    let decision = auto_prefill_chunk_decision(
        profile,
        prompt_tokens,
        prefill_environment_override_present(),
        true,
    );
    if decision.classification == "candidate"
        && let Ok(candidate) = price_prefill_scratch_plan(
            loaded,
            decision.selected,
            prompt_tokens,
            PrefillScratchConfig {
                matrix_query_cap: Some(AUTO_CHUNK_QUERY_ROWS),
            },
        )
    {
        upper = upper.max(candidate);
    }
    Ok(upper)
}

fn validate_requests(requests: &[PreparedJsonlRequest], args: &Args) -> Result<()> {
    ensure!(
        !requests.is_empty(),
        "--concurrency requires at least one request"
    );
    for request in requests {
        ensure!(
            request.request.cache_prefix_tokens.is_none(),
            "request {} sets cache_prefix_tokens, which is unsupported with --concurrency",
            request.id
        );
        ensure!(
            request.sampling.temperature == 0.0,
            "request {} uses temperature {}; --concurrency currently requires greedy decoding",
            request.id,
            request.sampling.temperature
        );
        ensure!(
            request.auto_cache_prefix_tokens.is_none() && request.auto_cache_future_hits == 0,
            "request {} carried prefix-cache lookahead into concurrent admission",
            request.id
        );
        jsonl_generation_capacity(request, args)?;
    }
    Ok(())
}

fn prepare_lane(
    loaded: &LoadedModel,
    request: &PreparedJsonlRequest,
    args: &Args,
) -> Result<(PreparedLane, f64)> {
    let (max_tokens, capacity) = jsonl_generation_capacity(request, args)?;
    let allocated = allocate_prefill_request_state(
        loaded,
        args.prefill_chunk,
        request.prompt_ids.len(),
        capacity,
        true,
    )?;
    report_prefill_chunk_decision(allocated.decision.as_ref(), request.prompt_ids.len());
    let mut scratch = allocated.scratch;
    let mut sequence = allocated.sequence;
    let (logits, prefill_ms) = prefill_span(
        &loaded.forward(),
        &mut sequence,
        &mut scratch,
        &request.prompt_ids,
        0,
    )?;
    drop(scratch);
    Ok((
        PreparedLane {
            id: request.id.clone(),
            prompt_tokens: request.prompt_ids.len(),
            max_tokens,
            sequence,
            logits,
            sampling: request.sampling,
        },
        prefill_ms,
    ))
}

fn start_lane(prepared: PreparedLane, tokenizer: &Tokenizer, stop_tokens: &[i32]) -> Result<Lane> {
    let mut sampler = Sampler::new(prepared.sampling).context("initialize concurrent sampler")?;
    let first = sampler
        .sample(&prepared.logits)
        .context("select concurrent first token")?
        .token;
    let mut progress = LaneProgress::new(prepared.max_tokens);
    let visible = progress.record_selection(first, stop_tokens)?;
    let mut generated_text = String::new();
    if visible {
        generated_text.push_str(&tokenizer.decode_piece(first));
    }
    Ok(Lane {
        id: prepared.id,
        prompt_tokens: prepared.prompt_tokens,
        sequence: prepared.sequence,
        progress,
        generated_text,
    })
}

fn run_pair(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    executor: &mut IndependentQueue2SequenceExecutor<'_>,
    requests: [&PreparedJsonlRequest; WIDTH],
    args: &Args,
    stop_tokens: &[i32],
    pair_index: usize,
) -> Result<([RequestOutput; WIDTH], PairTelemetry)> {
    let prepare_t0 = Instant::now();
    let (left, left_prefill_ms) = prepare_lane(loaded, requests[0], args)?;
    let (right, right_prefill_ms) = prepare_lane(loaded, requests[1], args)?;
    let requested_tokens = [left.max_tokens, right.max_tokens];
    let prepare_ms = prepare_t0.elapsed().as_secs_f64() * 1e3;
    let prefill_ms = left_prefill_ms + right_prefill_ms;

    let decode_t0 = Instant::now();
    let mut lanes = [
        start_lane(left, tokenizer, stop_tokens)?,
        start_lane(right, tokenizer, stop_tokens)?,
    ];
    let mut paired_transitions = 0usize;
    let mut serial_tail_transitions = 0usize;
    let mut executor_gpu_ms = [Some(0.0), Some(0.0)];
    let forward = loaded.forward();
    loop {
        shutdown::checkpoint()?;
        match next_decode_work(std::array::from_fn(|slot| lanes[slot].progress.is_active())) {
            DecodeWork::Pair => {
                let token_ids = [
                    lanes[0].progress.pending_token()?,
                    lanes[1].progress.pending_token()?,
                ];
                let [left, right] = &mut lanes;
                let step = executor.step_greedy(
                    token_ids,
                    [&mut left.sequence, &mut right.sequence],
                    || shutdown::checkpoint().is_err(),
                )?;
                shutdown::checkpoint()?;
                paired_transitions += 1;
                for slot in 0..WIDTH {
                    lanes[slot].progress.transitions += 1;
                    if lanes[slot]
                        .progress
                        .record_selection(step.argmax_ids[slot], stop_tokens)?
                    {
                        lanes[slot]
                            .generated_text
                            .push_str(&tokenizer.decode_piece(step.argmax_ids[slot]));
                    }
                    executor_gpu_ms[slot] = match (executor_gpu_ms[slot], step.gpu_ms[slot]) {
                        (Some(total), Some(step_ms)) => Some(total + step_ms),
                        _ => None,
                    };
                }
            }
            DecodeWork::Serial(slot) => {
                let lane = &mut lanes[slot];
                let token = lane.progress.pending_token()?;
                let position = lane.sequence.position();
                let selected = forward
                    .single_token_greedy(
                        token,
                        u32::try_from(position).context("position does not fit u32")?,
                        unsafe { lane.sequence.metal_session_mut() },
                    )
                    .context("decode concurrent serial tail")?
                    .into_token()
                    .map_err(anyhow::Error::new)?;
                lane.sequence.advance_by(1)?;
                lane.progress.transitions += 1;
                serial_tail_transitions += 1;
                if lane.progress.record_selection(selected, stop_tokens)? {
                    lane.generated_text
                        .push_str(&tokenizer.decode_piece(selected));
                }
            }
            DecodeWork::Complete => break,
        }
    }
    if paired_transitions == 0 {
        executor_gpu_ms = [None, None];
    }
    let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;

    for lane in &lanes {
        lane.progress
            .validate_complete()
            .with_context(|| format!("validate concurrent lane {}", lane.id))?;
        ensure!(
            lane.sequence.position() == lane.prompt_tokens + lane.progress.transitions,
            "concurrent lane {} sequence frontier drifted",
            lane.id
        );
    }
    let generated_tokens = lanes
        .iter()
        .map(|lane| lane.progress.generated.len())
        .sum::<usize>();
    let productive_transitions = paired_transitions
        .checked_mul(WIDTH)
        .and_then(|value| value.checked_add(serial_tail_transitions))
        .context("concurrent transition count overflow")?;
    ensure!(
        productive_transitions
            .checked_add(WIDTH)
            .map(|selected| selected == generated_tokens)
            .unwrap_or(false),
        "concurrent pair violated aggregate N-1 transition semantics"
    );
    let aggregate_generated_tps = if decode_ms > 0.0 {
        generated_tokens as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let aggregate_transition_tps = if decode_ms > 0.0 {
        productive_transitions as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let outputs = lanes.map(|lane| RequestOutput {
        id: lane.id,
        prompt_tokens: lane.prompt_tokens,
        generated_tokens: lane.progress.generated.len(),
        generated_token_sha256: generated_token_sha256(&lane.progress.generated),
        generated_text: lane.generated_text,
        stop_reason: lane
            .progress
            .stop_reason
            .expect("validated concurrent lane termination"),
        terminal_token_target_transition_consumed: false,
    });
    Ok((
        outputs,
        PairTelemetry {
            schema_version: 1,
            backend: "qwen_independent_queues_v1",
            pair_index,
            prompt_tokens: [requests[0].prompt_ids.len(), requests[1].prompt_ids.len()],
            requested_tokens,
            generated_tokens,
            productive_transitions,
            paired_transitions,
            serial_tail_transitions,
            prepare_ms,
            prefill_ms,
            decode_ms,
            executor_gpu_ms,
            aggregate_generated_tps,
            aggregate_transition_tps,
        },
    ))
}

fn write_outputs(stdout: &mut impl Write, outputs: &[RequestOutput]) -> Result<()> {
    let mut encoded = Vec::new();
    for output in outputs {
        serde_json::to_writer(&mut encoded, output).context("encode concurrent request output")?;
        encoded.push(b'\n');
    }
    stdout
        .write_all(&encoded)
        .context("write concurrent request outputs")?;
    stdout.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEST_FILE: AtomicU64 = AtomicU64::new(0);

    struct TestFile(PathBuf);

    impl TestFile {
        fn new() -> Self {
            let nonce = NEXT_TEST_FILE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "qwen-concurrent-jsonl-test-{}-{nonce}.jsonl",
                std::process::id()
            ));
            std::fs::write(&path, "{}\n").expect("create concurrent JSONL test file");
            Self(path)
        }
    }

    impl Drop for TestFile {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn test_args(path: &Path) -> Args {
        let mut args = Args::try_parse_from([
            "qwen",
            "--model",
            "model.gguf",
            "--requests-jsonl",
            "placeholder.jsonl",
            "--concurrency",
            "2",
            "--tokens",
            "3",
        ])
        .expect("valid concurrency test arguments");
        args.requests_jsonl = Some(path.to_path_buf());
        args
    }

    #[test]
    fn lane_progress_preserves_terminal_token_semantics() {
        let mut progress = LaneProgress::new(3);
        assert!(progress.record_selection(1, &[9]).unwrap());
        progress.transitions += 1;
        assert!(progress.record_selection(2, &[9]).unwrap());
        progress.transitions += 1;
        assert!(progress.record_selection(3, &[9]).unwrap());
        progress.validate_complete().unwrap();
        assert_eq!(progress.stop_reason, Some(StopReason::TokenLimit));

        let mut eos = LaneProgress::new(3);
        assert!(!eos.record_selection(9, &[9]).unwrap());
        eos.validate_complete().unwrap();
        assert_eq!(eos.stop_reason, Some(StopReason::Eos));
    }

    #[test]
    fn scheduler_pairs_only_while_both_lanes_are_active() {
        assert_eq!(next_decode_work([true, true]), DecodeWork::Pair);
        assert_eq!(next_decode_work([true, false]), DecodeWork::Serial(0));
        assert_eq!(next_decode_work([false, true]), DecodeWork::Serial(1));
        assert_eq!(next_decode_work([false, false]), DecodeWork::Complete);
    }

    #[test]
    fn cli_contract_is_greedy_file_scoped_and_cross_qwen() {
        let file = TestFile::new();
        let args = test_args(&file.0);
        assert_eq!(args.concurrency, Some(2));
        validate_cli_with_greedy_mode(
            &args,
            ExplicitCliOptions::default(),
            GreedyGpuArgmaxMode::DefaultOff,
        )
        .unwrap();
        validate_model_family(args.concurrency, Some(ModelFamily::Qwen35)).unwrap();
        validate_model_family(args.concurrency, Some(ModelFamily::Qwen35Moe)).unwrap();
        assert!(validate_model_family(args.concurrency, Some(ModelFamily::DeepSeek4)).is_err());
        assert!(validate_greedy_gpu_mode(GreedyGpuArgmaxMode::ExplicitRollback).is_err());

        let mut invalid = test_args(&file.0);
        invalid.concurrency = Some(3);
        let error = validate_cli_with_greedy_mode(
            &invalid,
            ExplicitCliOptions::default(),
            GreedyGpuArgmaxMode::DefaultOff,
        )
        .unwrap_err();
        assert!(error.to_string().contains("currently requires 2"));

        invalid = test_args(&file.0);
        invalid.temperature = 0.7;
        let error = validate_cli_with_greedy_mode(
            &invalid,
            ExplicitCliOptions::default(),
            GreedyGpuArgmaxMode::DefaultOff,
        )
        .unwrap_err();
        assert!(error.to_string().contains("greedy decoding"));

        invalid = test_args(&file.0);
        invalid.prompt_lookup = true;
        let error = validate_cli_with_greedy_mode(
            &invalid,
            ExplicitCliOptions::default(),
            GreedyGpuArgmaxMode::DefaultOff,
        )
        .unwrap_err();
        assert!(error.to_string().contains("prompt-lookup"));

        invalid = test_args(&file.0);
        invalid.requests_jsonl = Some(PathBuf::from("-"));
        let error = validate_cli_with_greedy_mode(
            &invalid,
            ExplicitCliOptions::default(),
            GreedyGpuArgmaxMode::DefaultOff,
        )
        .unwrap_err();
        assert!(error.to_string().contains("regular JSONL file"));
    }
}
