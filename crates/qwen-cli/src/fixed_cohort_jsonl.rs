use super::*;
use qwen_llm::dense_batch8::DENSE_BATCH8_WIDTH;
use qwen_llm::moe_batch16::MOE_BATCH16_WIDTH;
use qwen_llm::runtime::{DenseBatch8SequenceExecutor, MoeBatch16SequenceExecutor, RuntimeError};

const PAD_TOKEN: i32 = 0;
const TRANSIENT_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const PREFIX_FANOUT_MIN_TOKENS: usize = 256;
const DENSE_PREFIX_FANOUT_ENV: &str = "QWEN_DENSE_BATCH8_PREFIX_FANOUT";
const MOE_PREFIX_FANOUT_ENV: &str = "QWEN_MOE_BATCH16_PREFIX_FANOUT";

#[derive(Clone, Copy, Debug)]
struct FixedCohortStep<const WIDTH: usize> {
    argmax_ids: [i32; WIDTH],
    gpu_ms: Option<f64>,
}

trait FixedCohortExecutor<const WIDTH: usize> {
    const DISPLAY_NAME: &'static str;
    const PREFIX_FANOUT_ENV: &'static str;
    const COHORT_BACKEND: &'static str;
    const PLANNER_BACKEND: &'static str;
    const TELEMETRY_PREFIX: &'static str;
    const PLANNER_TELEMETRY_PREFIX: &'static str;

    fn validate(
        &self,
        token_ids: [i32; WIDTH],
        sequences: [&mut Sequence; WIDTH],
    ) -> Result<(), RuntimeError>;

    fn step_greedy(
        &mut self,
        token_ids: [i32; WIDTH],
        sequences: [&mut Sequence; WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<FixedCohortStep<WIDTH>, RuntimeError>;

    fn scratch_bytes(&self) -> Option<u64> {
        None
    }
}

impl FixedCohortExecutor<DENSE_BATCH8_WIDTH> for DenseBatch8SequenceExecutor<'_> {
    const DISPLAY_NAME: &'static str = "dense B=8";
    const PREFIX_FANOUT_ENV: &'static str = DENSE_PREFIX_FANOUT_ENV;
    const COHORT_BACKEND: &'static str = "dense_qwen_static_batch8_v2";
    const PLANNER_BACKEND: &'static str = "dense_qwen_fixed_cohort_planner_v1";
    const TELEMETRY_PREFIX: &'static str = "dense_batch8";
    const PLANNER_TELEMETRY_PREFIX: &'static str = "dense_batch8_planner";

    fn validate(
        &self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        sequences: [&mut Sequence; DENSE_BATCH8_WIDTH],
    ) -> Result<(), RuntimeError> {
        let [s0, s1, s2, s3, s4, s5, s6, s7] = sequences;
        DenseBatch8SequenceExecutor::validate(self, token_ids, [s0, s1, s2, s3, s4, s5, s6, s7])
    }

    fn step_greedy(
        &mut self,
        token_ids: [i32; DENSE_BATCH8_WIDTH],
        sequences: [&mut Sequence; DENSE_BATCH8_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<FixedCohortStep<DENSE_BATCH8_WIDTH>, RuntimeError> {
        let [s0, s1, s2, s3, s4, s5, s6, s7] = sequences;
        let step = DenseBatch8SequenceExecutor::step_greedy(
            self,
            token_ids,
            [s0, s1, s2, s3, s4, s5, s6, s7],
            cancelled,
        )?;
        Ok(FixedCohortStep {
            argmax_ids: step.argmax_ids,
            gpu_ms: step.gpu_ms,
        })
    }
}

impl FixedCohortExecutor<MOE_BATCH16_WIDTH> for MoeBatch16SequenceExecutor<'_> {
    const DISPLAY_NAME: &'static str = "MoE B=16";
    const PREFIX_FANOUT_ENV: &'static str = MOE_PREFIX_FANOUT_ENV;
    const COHORT_BACKEND: &'static str = "qwen_moe_static_batch16_v1";
    const PLANNER_BACKEND: &'static str = "qwen_moe_fixed_cohort_planner_v1";
    const TELEMETRY_PREFIX: &'static str = "moe_batch16";
    const PLANNER_TELEMETRY_PREFIX: &'static str = "moe_batch16_planner";

    fn validate(
        &self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        sequences: [&mut Sequence; MOE_BATCH16_WIDTH],
    ) -> Result<(), RuntimeError> {
        MoeBatch16SequenceExecutor::validate(self, token_ids, sequences)
    }

    fn step_greedy(
        &mut self,
        token_ids: [i32; MOE_BATCH16_WIDTH],
        sequences: [&mut Sequence; MOE_BATCH16_WIDTH],
        cancelled: impl Fn() -> bool,
    ) -> Result<FixedCohortStep<MOE_BATCH16_WIDTH>, RuntimeError> {
        let step = MoeBatch16SequenceExecutor::step_greedy(self, token_ids, sequences, cancelled)?;
        Ok(FixedCohortStep {
            argmax_ids: step.argmax_ids,
            gpu_ms: step.gpu_ms,
        })
    }

    fn scratch_bytes(&self) -> Option<u64> {
        Some(MoeBatch16SequenceExecutor::scratch_bytes(self))
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefixFanoutPlan {
    common_prefix_tokens: usize,
    selected_prefix_tokens: usize,
    reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CohortCompatibility {
    prompt_tokens: usize,
    requested_tokens: usize,
}

#[derive(Debug, Eq, PartialEq)]
enum PlannedWork<const WIDTH: usize> {
    Batch([usize; WIDTH]),
    Serial(usize),
}

impl<const WIDTH: usize> PlannedWork<WIDTH> {
    fn first_request_index(&self) -> usize {
        match self {
            Self::Batch(indices) => indices[0],
            Self::Serial(index) => *index,
        }
    }
}

#[derive(Debug)]
struct CohortPlan<const WIDTH: usize> {
    work: Vec<PlannedWork<WIDTH>>,
    compatibility_buckets: usize,
    full_cohorts: usize,
    serial_fallback_requests: usize,
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
            "cannot select into a finished fixed-cohort lane"
        );
        ensure!(
            self.generated.len() < self.max_tokens,
            "fixed-cohort lane exceeded its token limit"
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
            "fixed-cohort active-lane snapshot drifted"
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
            "fixed-cohort lane did not terminate"
        );
        ensure!(
            self.logical_transitions.checked_add(1) == Some(self.generated.len()),
            "fixed-cohort lane violated N-1 transition semantics"
        );
        ensure!(
            self.logical_transitions + self.padding_transitions == physical_batch_steps,
            "fixed-cohort physical transition accounting drifted"
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
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_scratch_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    executor_scratch_incremental_bytes: Option<u64>,
}

#[derive(Debug, Serialize)]
struct PlannerTelemetry {
    schema_version: u32,
    backend: &'static str,
    requests: usize,
    compatibility_buckets: usize,
    full_cohorts: usize,
    batched_requests: usize,
    serial_fallback_requests: usize,
    output_order: &'static str,
}

fn common_prefix_tokens(requests: &[&PreparedJsonlRequest]) -> usize {
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

fn plan_prefix_fanout(requests: &[&PreparedJsonlRequest], enabled: bool) -> PrefixFanoutPlan {
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

fn cohort_compatibility(
    request: &PreparedJsonlRequest,
    args: &Args,
) -> Result<CohortCompatibility> {
    let (requested_tokens, _) = jsonl_generation_capacity(request, args)?;
    Ok(CohortCompatibility {
        prompt_tokens: request.prompt_ids.len(),
        requested_tokens,
    })
}

fn plan_request_work<const WIDTH: usize>(
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> Result<CohortPlan<WIDTH>> {
    ensure!(WIDTH > 0, "fixed-cohort planner width must be nonzero");
    let mut buckets: Vec<(CohortCompatibility, Vec<usize>)> = Vec::new();
    for (index, request) in requests.iter().enumerate() {
        let key = cohort_compatibility(request, args)?;
        if let Some((_, indices)) = buckets.iter_mut().find(|(candidate, _)| *candidate == key) {
            indices.push(index);
        } else {
            buckets.push((key, vec![index]));
        }
    }

    let compatibility_buckets = buckets.len();
    let mut work = Vec::new();
    let mut full_cohorts = 0usize;
    let mut serial_fallback_requests = 0usize;
    for (_, indices) in buckets {
        let mut cohorts = indices.chunks_exact(WIDTH);
        for cohort in &mut cohorts {
            work.push(PlannedWork::Batch(
                cohort.try_into().expect("exact fixed-cohort planner chunk"),
            ));
            full_cohorts += 1;
        }
        for &index in cohorts.remainder() {
            work.push(PlannedWork::Serial(index));
            serial_fallback_requests += 1;
        }
    }
    work.sort_by_key(PlannedWork::first_request_index);
    ensure!(
        full_cohorts
            .checked_mul(WIDTH)
            .and_then(|batched| batched.checked_add(serial_fallback_requests))
            == Some(requests.len()),
        "fixed-cohort planner request accounting drifted"
    );
    Ok(CohortPlan {
        work,
        compatibility_buckets,
        full_cohorts,
        serial_fallback_requests,
    })
}

pub(super) fn validate_cli(args: &Args, explicit: ExplicitCliOptions) -> Result<()> {
    let Some(batch_size) = args.batch_size else {
        return Ok(());
    };
    ensure!(
        matches!(batch_size, DENSE_BATCH8_WIDTH | MOE_BATCH16_WIDTH),
        "--batch-size supports only {DENSE_BATCH8_WIDTH} (dense Qwen) or {MOE_BATCH16_WIDTH} (Qwen MoE), got {batch_size}"
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
    validate_greedy_gpu_mode(configured_greedy_gpu_argmax_mode(), args.batch_size)?;
    Ok(())
}

fn validate_greedy_gpu_mode(mode: GreedyGpuArgmaxMode, batch_size: Option<usize>) -> Result<()> {
    if mode == GreedyGpuArgmaxMode::ExplicitRollback {
        if batch_size == Some(DENSE_BATCH8_WIDTH) {
            bail!(
                "{GREEDY_GPU_ARGMAX_ENV}=0 disables --batch-size because fixed B=8 requires GPU greedy selection"
            );
        }
        bail!(
            "{GREEDY_GPU_ARGMAX_ENV}=0 disables --batch-size because MoE B=16 requires GPU greedy selection"
        );
    }
    Ok(())
}

pub(super) fn validate_model_family(
    batch_size: Option<usize>,
    model_family: Option<ModelFamily>,
) -> Result<()> {
    ensure!(
        match batch_size {
            None => true,
            Some(DENSE_BATCH8_WIDTH) => model_family == Some(ModelFamily::Qwen35),
            Some(MOE_BATCH16_WIDTH) => model_family == Some(ModelFamily::Qwen35Moe),
            Some(_) => false,
        },
        "unsupported --batch-size/model combination: batch size 8 requires qwen35 (dense), and batch size 16 requires qwen35moe"
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
    validate_greedy_gpu_mode(greedy_gpu_mode, args.batch_size)?;
    let requests_metadata = std::fs::metadata(requests_path)
        .with_context(|| format!("inspect requests JSONL {}", requests_path.display()))?;
    ensure!(
        requests_metadata.is_file(),
        "--batch-size requires a regular JSONL file, got {}",
        requests_path.display()
    );
    let requests = prepare_jsonl_requests(requests_path, tokenizer, args)?;
    validate_requests(&requests, args)?;
    match args.batch_size {
        Some(DENSE_BATCH8_WIDTH) => {
            let plan = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests, args)?;
            let executor = if plan.full_cohorts > 0 {
                Some(
                    loaded
                        .create_dense_batch8_executor()
                        .context("create dense B=8 executor")?,
                )
            } else {
                None
            };
            run_fixed_cohort_file(
                loaded,
                tokenizer,
                &requests,
                args,
                greedy_gpu_mode,
                stdout,
                plan,
                executor,
            )
        }
        Some(MOE_BATCH16_WIDTH) => {
            let plan = plan_request_work::<MOE_BATCH16_WIDTH>(&requests, args)?;
            let executor = if plan.full_cohorts > 0 {
                Some(
                    loaded
                        .create_moe_batch16_executor()
                        .context("create MoE B=16 executor")?,
                )
            } else {
                None
            };
            run_fixed_cohort_file(
                loaded,
                tokenizer,
                &requests,
                args,
                greedy_gpu_mode,
                stdout,
                plan,
                executor,
            )
        }
        Some(batch_size) => bail!("unsupported fixed-cohort batch size {batch_size}"),
        None => bail!("fixed-cohort JSONL execution requires --batch-size"),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_fixed_cohort_file<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
    plan: CohortPlan<WIDTH>,
    mut executor: Option<E>,
) -> Result<usize> {
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let planner_telemetry = PlannerTelemetry {
        schema_version: 1,
        backend: E::PLANNER_BACKEND,
        requests: requests.len(),
        compatibility_buckets: plan.compatibility_buckets,
        full_cohorts: plan.full_cohorts,
        batched_requests: plan.full_cohorts * WIDTH,
        serial_fallback_requests: plan.serial_fallback_requests,
        output_order: "input",
    };
    let mut pending_outputs = std::iter::repeat_with(|| None)
        .take(requests.len())
        .collect::<Vec<Option<RequestOutput>>>();
    let mut next_output = 0usize;
    let mut completed = 0usize;
    let mut cohort_index = 0usize;
    for work in plan.work {
        shutdown::checkpoint()?;
        match work {
            PlannedWork::Batch(indices) => {
                let cohort = indices.map(|index| &requests[index]);
                let (outputs, telemetry) = run_cohort(
                    loaded,
                    tokenizer,
                    executor.as_mut().expect("planned fixed-cohort executor"),
                    &cohort,
                    args,
                    &stop_tokens,
                    cohort_index,
                )
                .with_context(|| format!("run {} cohort {cohort_index}", E::DISPLAY_NAME))?;
                for (index, output) in indices.into_iter().zip(outputs) {
                    ensure!(
                        pending_outputs[index].replace(output).is_none(),
                        "{} planner produced request {index} twice",
                        E::DISPLAY_NAME
                    );
                }
                eprintln!(
                    "{}: {}",
                    E::TELEMETRY_PREFIX,
                    serde_json::to_string(&telemetry)
                        .with_context(|| format!("serialize {} telemetry", E::DISPLAY_NAME))?
                );
                cohort_index += 1;
            }
            PlannedWork::Serial(index) => {
                let (output, _) =
                    run_jsonl_request(loaded, tokenizer, &requests[index], args, greedy_gpu_mode)
                        .with_context(|| {
                        format!("run {} serial fallback request {index}", E::DISPLAY_NAME)
                    })?;
                ensure!(
                    pending_outputs[index].replace(output).is_none(),
                    "{} planner produced request {index} twice",
                    E::DISPLAY_NAME
                );
            }
        }
        shutdown::checkpoint()?;
        completed += flush_ready_outputs(stdout, &mut pending_outputs, &mut next_output)?;
    }
    ensure!(
        completed == requests.len() && next_output == requests.len(),
        "{} planner completed {completed}/{} requests",
        E::DISPLAY_NAME,
        requests.len()
    );
    eprintln!(
        "{}: {}",
        E::PLANNER_TELEMETRY_PREFIX,
        serde_json::to_string(&planner_telemetry)
            .with_context(|| format!("serialize {} planner telemetry", E::DISPLAY_NAME))?
    );
    Ok(completed)
}

fn flush_ready_outputs(
    stdout: &mut impl Write,
    pending: &mut [Option<RequestOutput>],
    next: &mut usize,
) -> Result<usize> {
    let mut encoded = Vec::new();
    let start = *next;
    while let Some(output) = pending.get_mut(*next).and_then(Option::take) {
        serde_json::to_writer(&mut encoded, &output).context("encode request output")?;
        encoded.push(b'\n');
        *next += 1;
    }
    if !encoded.is_empty() {
        stdout
            .write_all(&encoded)
            .context("write fixed-cohort ordered outputs")?;
        stdout.flush()?;
    }
    Ok(*next - start)
}

fn validate_requests(requests: &[PreparedJsonlRequest], args: &Args) -> Result<()> {
    ensure!(
        !requests.is_empty(),
        "--batch-size requires at least one request"
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
            "request {} carried prefix-cache lookahead into fixed-cohort admission",
            request.id
        );
        jsonl_generation_capacity(request, args)?;
    }
    Ok(())
}

fn run_cohort<const WIDTH: usize, E: FixedCohortExecutor<WIDTH>>(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    executor: &mut E,
    requests: &[&PreparedJsonlRequest; WIDTH],
    args: &Args,
    stop_tokens: &[i32],
    cohort_index: usize,
) -> Result<(Vec<RequestOutput>, CohortTelemetry)> {
    let prompt_tokens = requests[0].prompt_ids.len();
    let (requested_tokens, capacity) = jsonl_generation_capacity(requests[0], args)?;
    let chunk = match args.prefill_chunk {
        PrefillChunkArg::Fixed(requested) => requested.min(prompt_tokens.max(1)),
        PrefillChunkArg::Auto => bail!("{} requires a fixed prefill chunk", E::DISPLAY_NAME),
    };
    ensure!(
        chunk > 0,
        "{} prefill chunk must be nonzero",
        E::DISPLAY_NAME
    );
    let mut prefix_fanout = align_prefix_fanout(
        plan_prefix_fanout(
            requests,
            parse_prefix_fanout_enabled(std::env::var_os(E::PREFIX_FANOUT_ENV).as_deref()),
        ),
        prompt_tokens,
        chunk,
    );

    let mut scratch = allocate_legacy_prefill_scratch(loaded, chunk, prompt_tokens)?;
    let mut sequences = Vec::with_capacity(WIDTH);
    let before_first_sequence = loaded.context().current_allocated_size();
    sequences.push(
        loaded
            .create_sequence(SequenceConfig::new(capacity))
            .with_context(|| format!("allocate first {} sequence", E::DISPLAY_NAME))?,
    );
    let after_first_sequence = loaded.context().current_allocated_size();
    let sequence_allocation_delta_bytes = after_first_sequence
        .checked_sub(before_first_sequence)
        .filter(|&bytes| bytes > 0)
        .with_context(|| {
            format!(
                "{} sequence allocation did not produce a valid Metal byte delta",
                E::DISPLAY_NAME
            )
        })?;
    let remaining_sequence_required_bytes = sequence_allocation_delta_bytes
        .checked_mul((WIDTH - 1) as u64)
        .with_context(|| {
            format!(
                "{} remaining sequence byte estimate overflow",
                E::DISPLAY_NAME
            )
        })?;
    let mut prefix_snapshot_required_bytes = if prefix_fanout.selected_prefix_tokens > 0 {
        loaded
            .estimate_checkpoint_boundary_sizes(
                &sequences[0],
                prefix_fanout.selected_prefix_tokens,
                false,
                false,
            )
            .with_context(|| format!("estimate {} prefix fanout snapshot", E::DISPLAY_NAME))?
            .snapshot_bytes
    } else {
        0
    };
    let mut memory_admission_incremental_bytes = remaining_sequence_required_bytes
        .checked_add(prefix_snapshot_required_bytes)
        .with_context(|| format!("{} fanout admission byte overflow", E::DISPLAY_NAME))?;
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
        "{} memory admission denied: reason={} required_bytes={:?} working_set_headroom_bytes={:?} process_limit_remaining_bytes={:?}",
        E::DISPLAY_NAME,
        memory_admission.reason.as_str(),
        memory_admission.required_bytes,
        memory_admission.working_set_headroom_bytes,
        memory_admission.signals.process_limit_remaining_bytes,
    );
    for _ in 1..WIDTH {
        sequences.push(
            loaded
                .create_sequence(SequenceConfig::new(capacity))
                .with_context(|| format!("allocate admitted {} sequence", E::DISPLAY_NAME))?,
        );
    }

    let forward = loaded.forward();
    let prefill_t0 = Instant::now();
    let mut prompt_logits = Vec::with_capacity(WIDTH);
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
        .with_context(|| format!("prefill {} shared prefix", E::DISPLAY_NAME))?;
        prefix_prefill_ms = ms;

        let snapshot_t0 = Instant::now();
        let prepared = loaded
            .prepare_checkpoint_boundary(
                &sequences[0],
                requests[0].prompt_ids[..prefix_len].to_vec(),
                None,
                None,
            )
            .with_context(|| format!("capture {} shared prefix", E::DISPLAY_NAME))?;
        prefix_snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;
        prefix_snapshot_bytes = prepared.snapshot_bytes();
        ensure!(
            prefix_snapshot_bytes == prefix_snapshot_required_bytes,
            "{} prefix snapshot bytes {} != estimate {}",
            E::DISPLAY_NAME,
            prefix_snapshot_bytes,
            prefix_snapshot_required_bytes,
        );

        for slot in 0..WIDTH {
            shutdown::checkpoint()?;
            if slot > 0 {
                let restore_t0 = Instant::now();
                let restored = loaded
                    .restore_prepared_checkpoint(
                        &prepared,
                        &mut sequences[slot],
                        &requests[slot].prompt_ids,
                    )
                    .with_context(|| {
                        format!("restore {} shared prefix into slot {slot}", E::DISPLAY_NAME)
                    })?;
                prefix_restore_ms += restore_t0.elapsed().as_secs_f64() * 1e3;
                ensure!(
                    restored.matched_prefix_len == prefix_len
                        && restored.restored_prefix_len == prefix_len,
                    "{} slot {slot} restored an unexpected prefix boundary",
                    E::DISPLAY_NAME
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
                .with_context(|| {
                    format!("prefill {} cohort suffix slot {slot}", E::DISPLAY_NAME)
                })?;
                suffix_prefill_ms += ms;
                prompt_logits.push(logits);
            }
        }
    } else {
        for (slot, (request, sequence)) in requests.iter().zip(&mut sequences).enumerate() {
            shutdown::checkpoint()?;
            let (logits, ms) =
                prefill_span(&forward, sequence, &mut scratch, &request.prompt_ids, 0)
                    .with_context(|| format!("prefill {} cohort slot {slot}", E::DISPLAY_NAME))?;
            suffix_prefill_ms += ms;
            prompt_logits.push(logits);
        }
    }
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;
    drop(scratch);
    {
        let sequences: &mut [Sequence; WIDTH] = sequences
            .as_mut_slice()
            .try_into()
            .expect("fixed-cohort sequence width");
        executor
            .validate([PAD_TOKEN; WIDTH], sequences.each_mut())
            .with_context(|| format!("validate {} cohort backend", E::DISPLAY_NAME))?;
    }

    let decode_t0 = Instant::now();
    let mut lanes = Vec::with_capacity(WIDTH);
    for ((request, sequence), logits) in requests.iter().zip(sequences).zip(prompt_logits) {
        let mut sampler = Sampler::new(request.sampling)
            .with_context(|| format!("initialize {} greedy sampler", E::DISPLAY_NAME))?;
        let first = sampler
            .sample(&logits)
            .with_context(|| format!("select {} first token", E::DISPLAY_NAME))?
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
        let active: [bool; WIDTH] = std::array::from_fn(|slot| lanes[slot].progress.is_active());
        let token_ids: [i32; WIDTH] =
            std::array::from_fn(|slot| lanes[slot].progress.transition_token());
        let transition_t0 = Instant::now();
        let step = {
            let lanes: &mut [Lane; WIDTH] = lanes
                .as_mut_slice()
                .try_into()
                .expect("fixed-cohort lane width");
            executor.step_greedy(
                token_ids,
                lanes.each_mut().map(|lane| &mut lane.sequence),
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
    let mut outputs = Vec::with_capacity(WIDTH);
    for lane in lanes {
        lane.progress
            .validate_complete(physical_batch_steps)
            .with_context(|| format!("validate {} lane {}", E::DISPLAY_NAME, lane.id))?;
        ensure!(
            lane.sequence.position() == lane.prompt_tokens + physical_batch_steps,
            "{} lane {} sequence frontier drifted",
            E::DISPLAY_NAME,
            lane.id
        );
        let stop_reason = lane
            .progress
            .stop_reason
            .expect("validated fixed-cohort lane termination");
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
            backend: E::COHORT_BACKEND,
            cohort_index,
            width: WIDTH,
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
            executor_scratch_bytes: executor.scratch_bytes(),
            // The executor is created before the first sequence delta and this
            // admission snapshot, so its persistent Metal arena is already in
            // current_allocated_size and memory_signals. Adding it here would
            // charge the MoE scratch twice.
            executor_scratch_incremental_bytes: executor.scratch_bytes().map(|_| 0),
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

    fn output(id: &str) -> RequestOutput {
        RequestOutput {
            id: id.to_string(),
            prompt_tokens: 1,
            generated_tokens: 1,
            generated_token_sha256: id.to_string(),
            generated_text: id.to_string(),
            stop_reason: StopReason::TokenLimit,
            terminal_token_target_transition_consumed: false,
        }
    }

    #[test]
    fn cli_contract_is_explicit_and_family_scoped() {
        let args = test_args();
        validate_cli(&args, ExplicitCliOptions::default()).unwrap();
        validate_model_family(args.batch_size, Some(ModelFamily::Qwen35)).unwrap();
        assert!(validate_model_family(args.batch_size, Some(ModelFamily::Qwen35Moe)).is_err());
        assert!(validate_model_family(args.batch_size, Some(ModelFamily::DeepSeek4)).is_err());
        validate_model_family(Some(16), Some(ModelFamily::Qwen35Moe)).unwrap();
        assert!(validate_model_family(Some(16), Some(ModelFamily::Qwen35)).is_err());
        assert!(validate_model_family(Some(16), Some(ModelFamily::DeepSeek4)).is_err());
        let mut moe_args = test_args();
        moe_args.batch_size = Some(MOE_BATCH16_WIDTH);
        validate_cli(&moe_args, ExplicitCliOptions::default()).unwrap();
        assert_eq!(
            parse_greedy_gpu_argmax_mode(Some(OsStr::new("0"))),
            GreedyGpuArgmaxMode::ExplicitRollback
        );
        assert!(
            validate_greedy_gpu_mode(
                GreedyGpuArgmaxMode::ExplicitRollback,
                Some(DENSE_BATCH8_WIDTH)
            )
            .is_err()
        );
        assert!(
            validate_greedy_gpu_mode(
                GreedyGpuArgmaxMode::ExplicitRollback,
                Some(MOE_BATCH16_WIDTH)
            )
            .is_err()
        );
        validate_greedy_gpu_mode(GreedyGpuArgmaxMode::DefaultOff, Some(DENSE_BATCH8_WIDTH))
            .unwrap();
        validate_greedy_gpu_mode(GreedyGpuArgmaxMode::ForceEnabled, Some(MOE_BATCH16_WIDTH))
            .unwrap();

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
    fn planner_forms_compatible_cohorts_and_serializes_underfill() {
        let args = test_args();
        let mut requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| prepared(&format!("a-{slot}"), &[1, 2]))
            .chain((0..DENSE_BATCH8_WIDTH).map(|slot| {
                let mut request = prepared(&format!("b-{slot}"), &[1, 2, 3]);
                request.request.tokens = Some(4);
                request
            }))
            .collect::<Vec<_>>();
        requests.push(prepared("a-underfill", &[1, 2]));
        requests.push(prepared("c-underfill", &[1, 2, 3, 4]));
        validate_requests(&requests, &args).unwrap();
        let plan = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests, &args).unwrap();
        assert_eq!(plan.compatibility_buckets, 3);
        assert_eq!(plan.full_cohorts, 2);
        assert_eq!(plan.serial_fallback_requests, 2);
        assert_eq!(
            plan.work,
            vec![
                PlannedWork::Batch([0, 1, 2, 3, 4, 5, 6, 7]),
                PlannedWork::Batch([8, 9, 10, 11, 12, 13, 14, 15]),
                PlannedWork::Serial(16),
                PlannedWork::Serial(17),
            ]
        );
        let underfill = plan_request_work::<DENSE_BATCH8_WIDTH>(&requests[..7], &args).unwrap();
        assert_eq!(underfill.full_cohorts, 0);
        assert_eq!(underfill.serial_fallback_requests, 7);

        requests[5].sampling.temperature = 0.1;
        assert!(validate_requests(&requests, &args).is_err());
        requests[5].sampling.temperature = 0.0;
        requests[6].request.cache_prefix_tokens = Some(1);
        assert!(validate_requests(&requests, &args).is_err());
    }

    #[test]
    fn planner_forms_exact_sixteen_way_cohorts_and_serializes_remainders() {
        let mut args = test_args();
        args.batch_size = Some(MOE_BATCH16_WIDTH);
        let mut requests = (0..MOE_BATCH16_WIDTH)
            .map(|slot| prepared(&format!("full-{slot}"), &[1, 2, 3]))
            .collect::<Vec<_>>();
        let mut heterogeneous = prepared("heterogeneous", &[1, 2, 3, 4]);
        heterogeneous.request.tokens = Some(7);
        requests.push(heterogeneous);
        requests.push(prepared("underfill", &[1, 2, 3]));

        let plan = plan_request_work::<MOE_BATCH16_WIDTH>(&requests, &args).unwrap();
        assert_eq!(plan.compatibility_buckets, 2);
        assert_eq!(plan.full_cohorts, 1);
        assert_eq!(plan.serial_fallback_requests, 2);
        assert_eq!(
            plan.work,
            vec![
                PlannedWork::Batch([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]),
                PlannedWork::Serial(16),
                PlannedWork::Serial(17),
            ]
        );

        let underfill = plan_request_work::<MOE_BATCH16_WIDTH>(&requests[..15], &args).unwrap();
        assert_eq!(underfill.full_cohorts, 0);
        assert_eq!(underfill.serial_fallback_requests, 15);
    }

    #[test]
    fn prefix_fanout_requires_and_aligns_a_shared_minimum_prefix() {
        assert!(parse_prefix_fanout_enabled(None));
        assert!(parse_prefix_fanout_enabled(Some(OsStr::new("1"))));
        assert!(!parse_prefix_fanout_enabled(Some(OsStr::new("off"))));
        assert_eq!(DENSE_PREFIX_FANOUT_ENV, "QWEN_DENSE_BATCH8_PREFIX_FANOUT");
        assert_eq!(MOE_PREFIX_FANOUT_ENV, "QWEN_MOE_BATCH16_PREFIX_FANOUT");
        let shared = (0..PREFIX_FANOUT_MIN_TOKENS as i32).collect::<Vec<_>>();
        let requests = (0..DENSE_BATCH8_WIDTH)
            .map(|slot| {
                let mut tokens = shared.clone();
                tokens.push(10_000 + slot as i32);
                prepared(&format!("slot-{slot}"), &tokens)
            })
            .collect::<Vec<_>>();
        let request_refs = requests.iter().collect::<Vec<_>>();
        assert_eq!(
            common_prefix_tokens(&request_refs),
            PREFIX_FANOUT_MIN_TOKENS
        );
        assert_eq!(
            plan_prefix_fanout(&request_refs, true),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected",
            }
        );
        assert_eq!(
            plan_prefix_fanout(&request_refs, false),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                selected_prefix_tokens: 0,
                reason: "disabled",
            }
        );
        assert_eq!(
            align_prefix_fanout(plan_prefix_fanout(&request_refs, true), 257, 128),
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
        let wider_refs = wider.iter().collect::<Vec<_>>();
        assert_eq!(
            align_prefix_fanout(plan_prefix_fanout(&wider_refs, true), 261, 256),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS + 4,
                selected_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            align_prefix_fanout(plan_prefix_fanout(&wider_refs, true), 261, 200),
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
        let short_refs = short.iter().collect::<Vec<_>>();
        assert_eq!(
            plan_prefix_fanout(&short_refs, true),
            PrefixFanoutPlan {
                common_prefix_tokens: PREFIX_FANOUT_MIN_TOKENS - 1,
                selected_prefix_tokens: 0,
                reason: "below_minimum",
            }
        );
    }

    #[test]
    fn reorder_buffer_emits_only_contiguous_input_order() {
        let mut pending = std::iter::repeat_with(|| None)
            .take(3)
            .collect::<Vec<Option<RequestOutput>>>();
        let mut next = 0usize;
        let mut stdout = Vec::new();
        pending[2] = Some(output("two"));
        assert_eq!(
            flush_ready_outputs(&mut stdout, &mut pending, &mut next).unwrap(),
            0
        );
        pending[0] = Some(output("zero"));
        assert_eq!(
            flush_ready_outputs(&mut stdout, &mut pending, &mut next).unwrap(),
            1
        );
        pending[1] = Some(output("one"));
        assert_eq!(
            flush_ready_outputs(&mut stdout, &mut pending, &mut next).unwrap(),
            2
        );
        let rows = String::from_utf8(stdout).unwrap();
        let zero = rows.find("zero").unwrap();
        let one = rows.find("one").unwrap();
        let two = rows.find("two").unwrap();
        assert!(zero < one && one < two);
        assert_eq!(next, 3);
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
