use super::*;
use qwen_llm::runtime::IndependentQueue2SequenceExecutor;
use std::sync::{Arc, mpsc};

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

struct DeepSeekPreparedLane {
    request: DeepSeekV4PreparedRequest,
    session: DeepSeekV4Session,
    logits: Vec<f32>,
    session_ms: f64,
    prefill_mode: &'static str,
    prefill_ms: f64,
}

struct DeepSeekCompletedLane {
    output: RequestOutput,
    line: usize,
    session_ms: f64,
    prefill_mode: &'static str,
    prefill_ms: f64,
    generation_ms: f64,
    transitions: usize,
    selector_telemetry: DeepSeekV4MultigroupSelectorTelemetry,
}

#[derive(Debug, Serialize)]
struct DeepSeekPairTelemetry {
    schema_version: u32,
    backend: &'static str,
    pair_index: usize,
    prompt_tokens: [usize; WIDTH],
    requested_tokens: [usize; WIDTH],
    generated_tokens: usize,
    productive_transitions: usize,
    pair_wall_ms: f64,
    serial_prefill_ms: f64,
    concurrent_generation_ms: f64,
    session_allocation_delta_bytes: u64,
    session_priced_upper_bytes: u64,
    runtime_allocation_upper_bytes: u64,
    aggregate_generated_tps: f64,
    aggregate_transition_tps: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DeepSeekWorkerControl {
    Prepare,
    Generate,
    Abort,
}

pub(super) fn validate_cli(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
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
    Ok(())
}

fn validate_greedy_gpu_mode(mode: GreedyGpuArgmaxMode) -> Result<()> {
    ensure!(
        mode != GreedyGpuArgmaxMode::ExplicitRollback,
        "{GREEDY_GPU_ARGMAX_ENV}=0 disables --concurrency because paired decode requires GPU greedy selection"
    );
    Ok(())
}

pub(super) fn validate_model_family(args: &Args, model_family: Option<ModelFamily>) -> Result<()> {
    validate_model_family_with_modes(
        args.concurrency,
        args.temperature,
        model_family,
        configured_greedy_gpu_argmax_mode(),
        qwen_llm::env_flag::read_default_off("QWEN_DSV4_RESIDENCY_SET"),
    )
}

fn validate_model_family_with_modes(
    concurrency: Option<usize>,
    temperature: f32,
    model_family: Option<ModelFamily>,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    deepseek_residency_set: bool,
) -> Result<()> {
    if concurrency.is_none() {
        return Ok(());
    }
    match model_family {
        Some(ModelFamily::Qwen35 | ModelFamily::Qwen35Moe) => {
            ensure!(
                temperature == 0.0,
                "Qwen --concurrency currently requires greedy decoding (--temp 0)"
            );
            validate_greedy_gpu_mode(greedy_gpu_mode)
        }
        Some(ModelFamily::DeepSeek4) => {
            ensure!(
                !deepseek_residency_set,
                "--concurrency requires QWEN_DSV4_RESIDENCY_SET=0 because DeepSeek residency sets are command-queue scoped"
            );
            Ok(())
        }
        None => bail!("--concurrency requires a supported Qwen or DeepSeek V4 model"),
    }
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

fn prepare_deepseek_lane(
    ctx: &MetalContext,
    residency: &Arc<DeepSeekV4MetalResidency>,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    request: DeepSeekV4PreparedRequest,
    prefill_chunk_tokens: usize,
    vocab_size: u32,
) -> Result<DeepSeekPreparedLane> {
    shutdown::checkpoint()?;
    ensure!(
        request.required_forwards <= residency.session_capacity().forward_limit(),
        "request {} requires {} forwards, beyond the shared session budget {}",
        request.id,
        request.required_forwards,
        residency.session_capacity().forward_limit(),
    );
    let session_t0 = Instant::now();
    let mut session = DeepSeekV4Session::new_shared(ctx, residency.clone()).with_context(|| {
        format!(
            "create concurrent DeepSeek session for request {}",
            request.id
        )
    })?;
    selector_plan.seal_session(&mut session, &request.id)?;
    let session_ms = session_t0.elapsed().as_secs_f64() * 1e3;

    let prefill_t0 = Instant::now();
    let packed_chunk_count =
        deepseek_v4_packed_chunk_count(request.prompt_token_ids.len(), prefill_chunk_tokens);
    let prefill_mode = if packed_chunk_count > 0 {
        execute_deepseek_v4_prompt_suffix(
            &mut session,
            ctx,
            &request.prompt_token_ids,
            prefill_chunk_tokens,
        )
        .with_context(|| format!("prefill concurrent request {}", request.id))?;
        if packed_chunk_count == 1 {
            "layer_major"
        } else {
            "layer_major_chunks"
        }
    } else {
        for (index, &token) in request.prompt_token_ids.iter().enumerate() {
            session.forward_token(ctx, token).with_context(|| {
                format!(
                    "forward concurrent request {} prompt token {index}",
                    request.id
                )
            })?;
        }
        "singleton"
    };
    let logits = copy_deepseek_v4_logits(&session, vocab_size, "concurrent prompt")
        .with_context(|| format!("copy request {} prompt logits", request.id))?;
    deepseek_v4_debug_dump_logits_sha256(&request.id, &logits);
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
    Ok(DeepSeekPreparedLane {
        request,
        session,
        logits,
        session_ms,
        prefill_mode,
        prefill_ms,
    })
}

fn generate_deepseek_lane(
    ctx: &MetalContext,
    tokenizer: &Tokenizer,
    vocab_size: u32,
    stop_tokens: &[i32],
    mut lane: DeepSeekPreparedLane,
) -> Result<DeepSeekCompletedLane> {
    let mut sampler = Sampler::new(lane.request.sampling)
        .with_context(|| format!("initialize sampler for request {}", lane.request.id))?;
    let mut generated_bytes = Vec::new();
    let mut transition_index = 0usize;
    let generation = generate_serial(
        lane.logits,
        lane.request.max_tokens,
        stop_tokens,
        &mut sampler,
        |token| {
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token)
                .with_context(|| format!("decode request {} token {token}", lane.request.id))?;
            generated_bytes.extend_from_slice(piece);
            Ok(())
        },
        |token| {
            let current_transition = transition_index;
            let token = checked_deepseek_v4_token_id(
                token,
                vocab_size,
                &format!(
                    "request {} generated transition {current_transition}",
                    lane.request.id
                ),
            )?;
            lane.session.forward_token(ctx, token).with_context(|| {
                format!(
                    "forward request {} generated transition {current_transition}",
                    lane.request.id
                )
            })?;
            let logits = copy_deepseek_v4_logits(&lane.session, vocab_size, "continuing")
                .with_context(|| {
                    format!(
                        "copy request {} continuing logits after transition {current_transition}",
                        lane.request.id
                    )
                })?;
            transition_index += 1;
            Ok(logits)
        },
    )?;
    let selector_telemetry = lane.session.multigroup_selector_telemetry();
    let output = RequestOutput {
        id: lane.request.id,
        prompt_tokens: lane.request.prompt_tokens,
        generated_tokens: generation.tokens.len(),
        generated_token_sha256: generated_token_sha256(&generation.tokens),
        generated_text: String::from_utf8_lossy(&generated_bytes).into_owned(),
        stop_reason: generation.stop_reason,
        terminal_token_target_transition_consumed: false,
    };
    Ok(DeepSeekCompletedLane {
        output,
        line: lane.request.line,
        session_ms: lane.session_ms,
        prefill_mode: lane.prefill_mode,
        prefill_ms: lane.prefill_ms,
        generation_ms: generation.wall_ms,
        transitions: generation.transitions,
        selector_telemetry,
    })
}

fn emit_deepseek_completion(
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    completion: &DeepSeekCompletedLane,
    prefill_chunk_tokens: usize,
    effective_concurrency: usize,
) -> Result<()> {
    selector_plan.emit_completion(&completion.output.id, completion.selector_telemetry)?;
    let prefill_tps = if completion.prefill_ms > 0.0 {
        completion.output.prompt_tokens as f64 / (completion.prefill_ms / 1e3)
    } else {
        0.0
    };
    let decode_tps = if completion.generation_ms > 0.0 {
        completion.output.generated_tokens as f64 / (completion.generation_ms / 1e3)
    } else {
        0.0
    };
    eprintln!(
        concat!(
            "deepseek_v4 stats: request={} line={} prompt_kind=raw prefill_mode={} prefill_chunk_cap={} prompt_tokens={} ",
            "generated_tokens={} transitions={} stop_reason={} session_ms={:.1} prefill_ms={:.1} prefill_tps={:.2} ",
            "generation_ms={:.1} decode_tps={:.2} concurrency={} build_commit={} build_dirty={}"
        ),
        completion.output.id,
        completion.line,
        completion.prefill_mode,
        prefill_chunk_tokens,
        completion.output.prompt_tokens,
        completion.output.generated_tokens,
        completion.transitions,
        completion.output.stop_reason.as_str(),
        completion.session_ms,
        completion.prefill_ms,
        prefill_tps,
        completion.generation_ms,
        decode_tps,
        effective_concurrency,
        env!("QWEN_BUILD_COMMIT"),
        env!("QWEN_BUILD_DIRTY"),
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_deepseek_worker(
    ctx: &MetalContext,
    residency: Arc<DeepSeekV4MetalResidency>,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    tokenizer: &Tokenizer,
    vocab_size: u32,
    stop_tokens: &[i32],
    request: DeepSeekV4PreparedRequest,
    prefill_chunk_tokens: usize,
    control: mpsc::Receiver<DeepSeekWorkerControl>,
    prepared_tx: mpsc::Sender<bool>,
) -> Result<DeepSeekCompletedLane> {
    match control
        .recv()
        .context("receive DeepSeek concurrency prepare control")?
    {
        DeepSeekWorkerControl::Prepare => {}
        DeepSeekWorkerControl::Abort => bail!("DeepSeek concurrency worker aborted before prepare"),
        DeepSeekWorkerControl::Generate => {
            bail!("DeepSeek concurrency worker received generation before prepare")
        }
    }
    let prepared = prepare_deepseek_lane(
        ctx,
        &residency,
        selector_plan,
        request,
        prefill_chunk_tokens,
        vocab_size,
    );
    prepared_tx
        .send(prepared.is_ok())
        .context("publish DeepSeek concurrency prepare status")?;
    match control
        .recv()
        .context("receive DeepSeek concurrency generation control")?
    {
        DeepSeekWorkerControl::Generate => {
            generate_deepseek_lane(ctx, tokenizer, vocab_size, stop_tokens, prepared?)
        }
        DeepSeekWorkerControl::Abort => match prepared {
            Ok(_) => bail!("DeepSeek concurrency worker aborted after peer setup failure"),
            Err(error) => Err(error),
        },
        DeepSeekWorkerControl::Prepare => {
            bail!("DeepSeek concurrency worker received duplicate prepare control")
        }
    }
}

fn run_deepseek_pair(
    contexts: [&MetalContext; WIDTH],
    residency: &Arc<DeepSeekV4MetalResidency>,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    tokenizer: &Tokenizer,
    vocab_size: u32,
    stop_tokens: &[i32],
    requests: [DeepSeekV4PreparedRequest; WIDTH],
    prefill_chunk_tokens: usize,
    session_priced_upper_bytes: u64,
    pair_index: usize,
) -> Result<([DeepSeekCompletedLane; WIDTH], DeepSeekPairTelemetry)> {
    let prompt_tokens = [requests[0].prompt_tokens, requests[1].prompt_tokens];
    let requested_tokens = [requests[0].max_tokens, requests[1].max_tokens];
    let [left_request, right_request] = requests;
    let pair_t0 = Instant::now();
    let before_sessions = contexts[0].current_allocated_size();
    let runtime_allocation_upper_bytes = session_priced_upper_bytes
        .checked_mul(WIDTH as u64)
        .and_then(|bytes| bytes.checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES))
        .context("DeepSeek two-session runtime byte overflow")?;
    let (completed, serial_prefill_ms, concurrent_generation_ms, session_allocation_delta_bytes) =
        std::thread::scope(|scope| -> Result<_> {
            let (left_control_tx, left_control_rx) = mpsc::channel();
            let (right_control_tx, right_control_rx) = mpsc::channel();
            let (left_prepared_tx, left_prepared_rx) = mpsc::channel();
            let (right_prepared_tx, right_prepared_rx) = mpsc::channel();
            let left_residency = residency.clone();
            let right_residency = residency.clone();
            let left_handle = scope.spawn(move || {
                run_deepseek_worker(
                    contexts[0],
                    left_residency,
                    selector_plan,
                    tokenizer,
                    vocab_size,
                    stop_tokens,
                    left_request,
                    prefill_chunk_tokens,
                    left_control_rx,
                    left_prepared_tx,
                )
            });
            let right_handle = scope.spawn(move || {
                run_deepseek_worker(
                    contexts[1],
                    right_residency,
                    selector_plan,
                    tokenizer,
                    vocab_size,
                    stop_tokens,
                    right_request,
                    prefill_chunk_tokens,
                    right_control_rx,
                    right_prepared_tx,
                )
            });

            let prefill_t0 = Instant::now();
            left_control_tx
                .send(DeepSeekWorkerControl::Prepare)
                .context("start DeepSeek concurrency lane 0 prefill")?;
            let left_prepared = left_prepared_rx.recv().unwrap_or(false);
            let right_prepared = if left_prepared {
                right_control_tx
                    .send(DeepSeekWorkerControl::Prepare)
                    .context("start DeepSeek concurrency lane 1 prefill")?;
                right_prepared_rx.recv().unwrap_or(false)
            } else {
                false
            };
            let serial_prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

            if !left_prepared || !right_prepared {
                let _ = left_control_tx.send(DeepSeekWorkerControl::Abort);
                let _ = right_control_tx.send(DeepSeekWorkerControl::Abort);
                let left_result = left_handle.join();
                let right_result = right_handle.join();
                if !left_prepared {
                    return match left_result {
                        Ok(Err(error)) => Err(error),
                        Ok(Ok(_)) => bail!("DeepSeek concurrency lane 0 failed without an error"),
                        Err(_) => bail!("DeepSeek concurrency lane 0 panicked during prefill"),
                    };
                }
                return match right_result {
                    Ok(Err(error)) => Err(error),
                    Ok(Ok(_)) => bail!("DeepSeek concurrency lane 1 failed without an error"),
                    Err(_) => bail!("DeepSeek concurrency lane 1 panicked during prefill"),
                };
            }

            let after_sessions = contexts[0].current_allocated_size();
            let session_allocation_delta_bytes = after_sessions
                .checked_sub(before_sessions)
                .context("DeepSeek concurrent session allocation counter regressed")?;
            ensure!(
                session_allocation_delta_bytes <= runtime_allocation_upper_bytes,
                "DeepSeek concurrent runtime allocation {session_allocation_delta_bytes} exceeds session-plus-reserve upper {runtime_allocation_upper_bytes}"
            );

            let generation_t0 = Instant::now();
            left_control_tx
                .send(DeepSeekWorkerControl::Generate)
                .context("start DeepSeek concurrency lane 0 generation")?;
            right_control_tx
                .send(DeepSeekWorkerControl::Generate)
                .context("start DeepSeek concurrency lane 1 generation")?;
            let left_result = left_handle.join();
            let right_result = right_handle.join();
            let concurrent_generation_ms = generation_t0.elapsed().as_secs_f64() * 1e3;
            let left =
                left_result.map_err(|_| anyhow!("DeepSeek concurrency lane 0 panicked"))??;
            let right =
                right_result.map_err(|_| anyhow!("DeepSeek concurrency lane 1 panicked"))??;
            Ok((
                [left, right],
                serial_prefill_ms,
                concurrent_generation_ms,
                session_allocation_delta_bytes,
            ))
        })?;
    let pair_wall_ms = pair_t0.elapsed().as_secs_f64() * 1e3;
    let generated_tokens = completed
        .iter()
        .map(|lane| lane.output.generated_tokens)
        .sum::<usize>();
    let productive_transitions = completed.iter().map(|lane| lane.transitions).sum::<usize>();
    ensure!(
        productive_transitions.checked_add(WIDTH) == Some(generated_tokens),
        "DeepSeek concurrent pair violated aggregate N-1 transition semantics"
    );
    let aggregate_generated_tps = if concurrent_generation_ms > 0.0 {
        generated_tokens as f64 / (concurrent_generation_ms / 1e3)
    } else {
        0.0
    };
    let aggregate_transition_tps = if concurrent_generation_ms > 0.0 {
        productive_transitions as f64 / (concurrent_generation_ms / 1e3)
    } else {
        0.0
    };
    Ok((
        completed,
        DeepSeekPairTelemetry {
            schema_version: 1,
            backend: "deepseek_v4_independent_queues_v1",
            pair_index,
            prompt_tokens,
            requested_tokens,
            generated_tokens,
            productive_transitions,
            pair_wall_ms,
            serial_prefill_ms,
            concurrent_generation_ms,
            session_allocation_delta_bytes,
            session_priced_upper_bytes,
            runtime_allocation_upper_bytes,
            aggregate_generated_tps,
            aggregate_transition_tps,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn run_deepseek_file(
    ctx: &MetalContext,
    residency: DeepSeekV4MetalResidency,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    tokenizer: &Tokenizer,
    vocab_size: u32,
    stop_tokens: &[i32],
    requests: Vec<DeepSeekV4PreparedRequest>,
    prefill_chunk_tokens: usize,
    session_priced_upper_bytes: u64,
) -> Result<usize> {
    ensure!(
        !requests.is_empty(),
        "DeepSeek concurrency requires requests"
    );
    let contexts = [
        ctx.with_new_command_queue()
            .context("create DeepSeek concurrency queue 0")?,
        ctx.with_new_command_queue()
            .context("create DeepSeek concurrency queue 1")?,
    ];
    let residency = Arc::new(residency);
    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let mut executed = 0usize;
    let mut requests = requests.into_iter();
    let mut pair_index = 0usize;
    while let Some(left) = requests.next() {
        let Some(right) = requests.next() else {
            let before_session = contexts[0].current_allocated_size();
            let lane = prepare_deepseek_lane(
                &contexts[0],
                &residency,
                selector_plan,
                left,
                prefill_chunk_tokens,
                vocab_size,
            )?;
            let after_session = contexts[0].current_allocated_size();
            let session_delta = after_session
                .checked_sub(before_session)
                .context("DeepSeek odd-tail allocation counter regressed")?;
            let session_runtime_upper = session_priced_upper_bytes
                .checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES)
                .context("DeepSeek odd-tail runtime byte overflow")?;
            ensure!(
                session_delta <= session_runtime_upper,
                "DeepSeek odd-tail runtime allocation {session_delta} exceeds session-plus-reserve upper {session_runtime_upper}"
            );
            let completion =
                generate_deepseek_lane(&contexts[0], tokenizer, vocab_size, stop_tokens, lane)?;
            emit_deepseek_completion(selector_plan, &completion, prefill_chunk_tokens, 1)?;
            write_outputs(&mut stdout, std::slice::from_ref(&completion.output))?;
            executed += 1;
            break;
        };
        let (completed, telemetry) = run_deepseek_pair(
            [&contexts[0], &contexts[1]],
            &residency,
            selector_plan,
            tokenizer,
            vocab_size,
            stop_tokens,
            [left, right],
            prefill_chunk_tokens,
            session_priced_upper_bytes,
            pair_index,
        )?;
        for lane in &completed {
            emit_deepseek_completion(selector_plan, lane, prefill_chunk_tokens, WIDTH)?;
        }
        let outputs = completed.map(|lane| lane.output);
        write_outputs(&mut stdout, &outputs)?;
        eprintln!(
            "deepseek_v4 concurrency_pair: {}",
            serde_json::to_string(&telemetry)
                .context("serialize DeepSeek concurrency telemetry")?
        );
        executed += WIDTH;
        pair_index += 1;
    }
    Ok(executed)
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
    fn cli_contract_is_file_scoped_and_family_capability_aware() {
        let file = TestFile::new();
        let args = test_args(&file.0);
        assert_eq!(args.concurrency, Some(2));
        validate_cli(&args, ExplicitCliOptions::default()).unwrap();
        validate_model_family_with_modes(
            args.concurrency,
            0.0,
            Some(ModelFamily::Qwen35),
            GreedyGpuArgmaxMode::DefaultOff,
            false,
        )
        .unwrap();
        validate_model_family_with_modes(
            args.concurrency,
            0.0,
            Some(ModelFamily::Qwen35Moe),
            GreedyGpuArgmaxMode::DefaultOff,
            false,
        )
        .unwrap();
        validate_model_family_with_modes(
            args.concurrency,
            0.7,
            Some(ModelFamily::DeepSeek4),
            GreedyGpuArgmaxMode::ExplicitRollback,
            false,
        )
        .unwrap();
        assert!(
            validate_model_family_with_modes(
                args.concurrency,
                0.0,
                Some(ModelFamily::Qwen35),
                GreedyGpuArgmaxMode::ExplicitRollback,
                false,
            )
            .is_err()
        );
        assert!(
            validate_model_family_with_modes(
                args.concurrency,
                0.0,
                Some(ModelFamily::DeepSeek4),
                GreedyGpuArgmaxMode::DefaultOff,
                true,
            )
            .is_err()
        );

        let mut invalid = test_args(&file.0);
        invalid.concurrency = Some(3);
        let error = validate_cli(&invalid, ExplicitCliOptions::default()).unwrap_err();
        assert!(error.to_string().contains("currently requires 2"));

        invalid = test_args(&file.0);
        invalid.temperature = 0.7;
        validate_cli(&invalid, ExplicitCliOptions::default()).unwrap();
        let error = validate_model_family_with_modes(
            invalid.concurrency,
            invalid.temperature,
            Some(ModelFamily::Qwen35),
            GreedyGpuArgmaxMode::DefaultOff,
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("greedy decoding"));

        invalid = test_args(&file.0);
        invalid.prompt_lookup = true;
        let error = validate_cli(&invalid, ExplicitCliOptions::default()).unwrap_err();
        assert!(error.to_string().contains("prompt-lookup"));

        invalid = test_args(&file.0);
        invalid.requests_jsonl = Some(PathBuf::from("-"));
        let error = validate_cli(&invalid, ExplicitCliOptions::default()).unwrap_err();
        assert!(error.to_string().contains("regular JSONL file"));
    }
}
