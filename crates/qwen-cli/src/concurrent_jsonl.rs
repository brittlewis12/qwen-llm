use super::*;
use qwen_llm::deepseek_v4_metal::DeepSeekV4CausalSnapshot;
use qwen_llm::runtime::IndependentQueue2SequenceExecutor;
use std::sync::{Arc, mpsc};

const WIDTH: usize = 2;
const PREFIX_FANOUT_MIN_TOKENS: usize = 256;
const PREFIX_FANOUT_ENV: &str = "QWEN_CONCURRENCY_PREFIX_FANOUT";
const PAIR_PLANNER_ENV: &str = "QWEN_CONCURRENCY_PAIR_PLANNER";
const PAIR_PLANNER_WINDOW: usize = 16;
const PREFIX_FANOUT_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const FILE_ROOT_FANOUT_ENV: &str = "QWEN_CONCURRENCY_FILE_ROOT_FANOUT";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct PrefixFanoutPlan {
    common_prefix_tokens: usize,
    selected_prefix_tokens: usize,
    reason: &'static str,
}

#[derive(Debug, Serialize)]
struct FileRootTelemetry {
    schema_version: u32,
    backend: &'static str,
    enabled: bool,
    requests: usize,
    planned_pairs: usize,
    common_prefix_tokens: usize,
    selected_prefix_tokens: usize,
    planned_pair_uses: usize,
    planned_avoided_prefix_evaluations: usize,
    planned_avoided_prompt_tokens: usize,
    minimum_tokens: usize,
    minimum_pairs: usize,
    reason: &'static str,
    outcome: &'static str,
    actual_pair_uses: usize,
    actual_avoided_prefix_evaluations: usize,
    actual_avoided_prompt_tokens: usize,
    excluded_serial_requests: usize,
    snapshot_required_bytes: u64,
    max_pair_snapshot_required_bytes: u64,
    additional_memory_bytes: u64,
    memory_admission_required_bytes: Option<u64>,
    memory_admission_reason: &'static str,
    snapshot_bytes: u64,
    prefill_ms: f64,
    snapshot_ms: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PairWork {
    Pair([usize; WIDTH]),
    Serial(usize),
}

impl PairWork {
    fn first_request_index(self) -> usize {
        match self {
            Self::Pair(indices) => indices[0].min(indices[1]),
            Self::Serial(index) => index,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct PairSchedule {
    work: Vec<PairWork>,
    prefix_affinity_pairs: usize,
    depth_balanced_pairs: usize,
}

#[derive(Debug, Serialize)]
struct PairPlannerTelemetry {
    schema_version: u32,
    backend: &'static str,
    enabled: bool,
    requests: usize,
    prefix_affinity_pairs: usize,
    depth_balanced_pairs: usize,
    serial_requests: usize,
    planning_window: usize,
    strategy: &'static str,
    output_order: &'static str,
}

#[derive(Debug, Serialize)]
struct QwenPairPlannerTelemetry {
    schema_version: u32,
    backend: &'static str,
    enabled: bool,
    requests: usize,
    prefix_affinity_pairs: usize,
    depth_balanced_pairs: usize,
    serial_requests: usize,
    planning_window: usize,
    strategy: &'static str,
    output_order: &'static str,
    prefix_fanout_boundary_policy: &'static str,
    private_suffix_singleton_max_tokens: usize,
}

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
    request_indices: [usize; WIDTH],
    prompt_tokens: [usize; WIDTH],
    requested_tokens: [usize; WIDTH],
    generated_tokens: usize,
    productive_transitions: usize,
    paired_transitions: usize,
    serial_tail_transitions: usize,
    common_prefix_tokens: usize,
    prefix_fanout_tokens: usize,
    prefix_fanout_reason: &'static str,
    prefix_fanout_boundary_policy: &'static str,
    prefix_fanout_min_tokens: usize,
    prefix_snapshot_bytes: u64,
    prefix_prefill_tokens: usize,
    prefix_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    private_prefill_ms: f64,
    private_suffix_singleton_max_tokens: usize,
    private_suffix_singleton_lanes: usize,
    private_suffix_singleton_tokens: usize,
    private_suffix_packed_lanes: usize,
    private_suffix_packed_tokens: usize,
    prefix_snapshot_required_bytes: u64,
    prefix_memory_admission_required_bytes: Option<u64>,
    prefix_memory_admission_reason: &'static str,
    file_root_tokens: usize,
    file_root_restores: usize,
    file_root_restore_ms: f64,
    prepare_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    executor_gpu_ms: [Option<f64>; WIDTH],
    aggregate_generated_tps: f64,
    aggregate_transition_tps: f64,
}

struct PreparedPair {
    lanes: [PreparedLane; WIDTH],
    plan: PrefixFanoutPlan,
    prefix_snapshot_bytes: u64,
    prefix_prefill_tokens: usize,
    prefix_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    private_prefill_ms: f64,
    private_suffix_singleton_lanes: usize,
    private_suffix_singleton_tokens: usize,
    private_suffix_packed_lanes: usize,
    private_suffix_packed_tokens: usize,
    prefix_snapshot_required_bytes: u64,
    prefix_memory_admission_required_bytes: Option<u64>,
    prefix_memory_admission_reason: &'static str,
    file_root_tokens: usize,
    file_root_restores: usize,
    file_root_restore_ms: f64,
}

struct DeepSeekPreparedLane {
    request: DeepSeekV4PreparedRequest,
    session: DeepSeekV4Session,
    logits: Vec<f32>,
    session_ms: f64,
    prefill_mode: &'static str,
    evaluated_prefill_tokens: usize,
    prefill_ms: f64,
    prefix_prefill_ms: f64,
    private_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    prefix_snapshot_payload_bytes: u64,
    exact_prompt_logits_reused: bool,
}

struct DeepSeekCompletedLane {
    output: RequestOutput,
    line: usize,
    session_ms: f64,
    prefill_mode: &'static str,
    evaluated_prefill_tokens: usize,
    prefill_ms: f64,
    prefix_prefill_ms: f64,
    private_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    prefix_snapshot_payload_bytes: u64,
    exact_prompt_logits_reused: bool,
    generation_ms: f64,
    transitions: usize,
    selector_telemetry: DeepSeekV4MultigroupSelectorTelemetry,
}

#[derive(Debug, Serialize)]
struct DeepSeekPairTelemetry {
    schema_version: u32,
    backend: &'static str,
    pair_index: usize,
    request_indices: [usize; WIDTH],
    prompt_tokens: [usize; WIDTH],
    requested_tokens: [usize; WIDTH],
    generated_tokens: usize,
    productive_transitions: usize,
    common_prefix_tokens: usize,
    prefix_fanout_tokens: usize,
    prefix_fanout_reason: &'static str,
    prefix_fanout_min_tokens: usize,
    prefix_snapshot_payload_bytes: u64,
    prefix_snapshot_priced_upper_bytes: u64,
    prefix_restore_workspace_bytes: u64,
    prefix_prefill_ms: f64,
    prefix_snapshot_ms: f64,
    prefix_restore_ms: f64,
    private_prefill_ms: f64,
    exact_prompt_logits_reused: bool,
    fanout_memory_admission_required_bytes: Option<u64>,
    fanout_memory_admission_reason: &'static str,
    pair_wall_ms: f64,
    prepare_ms: f64,
    evaluated_prefill_tokens: usize,
    model_prefill_ms: f64,
    concurrent_generation_ms: f64,
    session_allocation_delta_bytes: u64,
    session_priced_upper_bytes: u64,
    runtime_allocation_upper_bytes: u64,
    aggregate_generated_tps: f64,
    aggregate_transition_tps: f64,
}

#[derive(Clone, Debug)]
enum DeepSeekWorkerControl {
    PrepareSerial,
    PrepareSharedSource {
        prefix_len: usize,
        model_content_id: DeepSeekV4ModelContentId,
    },
    PrepareSharedRestore {
        prefix_len: usize,
        model_content_id: DeepSeekV4ModelContentId,
        snapshot: Arc<DeepSeekV4CausalSnapshot>,
        prefix_logits: Arc<Vec<f32>>,
    },
    Generate,
    Abort,
}

#[derive(Clone)]
enum DeepSeekPrepareSignal {
    Ready,
    SharedSource {
        snapshot: Arc<DeepSeekV4CausalSnapshot>,
        prefix_logits: Arc<Vec<f32>>,
    },
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct QwenExecutionAdmissionRequirements {
    pub max_capacity: usize,
    pub prefill_scratch_upper_bytes: u64,
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
    let requests = prepare_jsonl_requests(requests_path, tokenizer, args)?;
    run_prepared(loaded, tokenizer, &requests, args, greedy_gpu_mode, stdout)
}

pub(super) fn run_prepared(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    requests: &[PreparedJsonlRequest],
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
) -> Result<usize> {
    validate_greedy_gpu_mode(greedy_gpu_mode)?;
    validate_requests(requests, args)?;
    let requirements = qwen_execution_admission_requirements(loaded, requests, args)?;
    let requested_tokens = requests
        .iter()
        .map(|request| jsonl_generation_capacity(request, args).map(|(tokens, _)| tokens))
        .collect::<Result<Vec<_>>>()?;
    let prompt_refs = requests
        .iter()
        .map(|request| request.prompt_ids.as_slice())
        .collect::<Vec<_>>();
    let planner_enabled = pair_planner_enabled(true)?;
    let fanout_enabled = prefix_fanout_enabled();
    let prefix_boundary_policy = qwen_prefix_fanout_boundary_policy()?;
    let schedule = plan_request_pairs(
        &prompt_refs,
        &requested_tokens,
        planner_enabled,
        |left, right| {
            plan_prefix_fanout(
                left,
                right,
                args.prefill_chunk,
                fanout_enabled,
                prefix_boundary_policy,
            )
            .selected_prefix_tokens
        },
    )?;
    let pair_count = schedule
        .work
        .iter()
        .filter(|work| matches!(work, PairWork::Pair(_)))
        .count();
    eprintln!(
        "concurrency_planner: {}",
        serde_json::to_string(&QwenPairPlannerTelemetry {
            schema_version: 2,
            backend: "qwen_pair_affinity_v2",
            enabled: planner_enabled,
            requests: requests.len(),
            prefix_affinity_pairs: schedule.prefix_affinity_pairs,
            depth_balanced_pairs: schedule.depth_balanced_pairs,
            serial_requests: requests.len() - pair_count * WIDTH,
            planning_window: PAIR_PLANNER_WINDOW,
            strategy: if planner_enabled {
                "bounded_prefix_affinity_then_depth"
            } else {
                "input_order"
            },
            output_order: "input",
            prefix_fanout_boundary_policy: prefix_boundary_policy.as_str(),
            private_suffix_singleton_max_tokens: PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS,
        })
        .context("serialize Qwen concurrency planner telemetry")?
    );
    let paired_prompt_refs = schedule
        .work
        .iter()
        .filter_map(|work| match *work {
            PairWork::Pair(indices) => Some(indices),
            PairWork::Serial(_) => None,
        })
        .flatten()
        .map(|index| requests[index].prompt_ids.as_slice())
        .collect::<Vec<_>>();
    let root_plan = qwen_file_root::plan(
        &paired_prompt_refs,
        args.prefill_chunk,
        pair_count,
        qwen_file_root::enabled(FILE_ROOT_FANOUT_ENV, true)?,
    );
    let root_reason = if root_plan.reason == "below_minimum_uses" {
        "below_minimum_pairs"
    } else {
        root_plan.reason
    };
    let mut file_root = None;
    let mut root_telemetry = FileRootTelemetry {
        schema_version: 1,
        backend: "qwen_file_root_v1",
        enabled: root_plan.enabled,
        requests: requests.len(),
        planned_pairs: pair_count,
        common_prefix_tokens: root_plan.common_prefix_tokens,
        selected_prefix_tokens: root_plan.selected_prefix_tokens,
        planned_pair_uses: root_plan.planned_uses,
        planned_avoided_prefix_evaluations: root_plan.planned_avoided_prefix_evaluations,
        planned_avoided_prompt_tokens: root_plan.planned_avoided_prompt_tokens,
        minimum_tokens: qwen_file_root::MIN_TOKENS,
        minimum_pairs: 2,
        reason: root_reason,
        outcome: "not_selected",
        actual_pair_uses: 0,
        actual_avoided_prefix_evaluations: 0,
        actual_avoided_prompt_tokens: 0,
        excluded_serial_requests: requests.len() - pair_count * WIDTH,
        snapshot_required_bytes: 0,
        max_pair_snapshot_required_bytes: 0,
        additional_memory_bytes: 0,
        memory_admission_required_bytes: None,
        memory_admission_reason: "not_requested",
        snapshot_bytes: 0,
        prefill_ms: 0.0,
        snapshot_ms: 0.0,
    };
    let mut executor = if pair_count > 0 {
        let admission = if root_plan.selected_prefix_tokens > 0 {
            let root_snapshot_required_bytes = loaded
                .estimate_checkpoint_boundary_sizes_unallocated(
                    requirements.max_capacity,
                    root_plan.selected_prefix_tokens,
                    false,
                    true,
                )
                .context("estimate file-root checkpoint")?
                .snapshot_bytes;
            let max_pair_snapshot_required_bytes = max_file_root_pair_snapshot_bytes(
                loaded,
                requests,
                &schedule,
                args,
                prefix_boundary_policy,
                root_plan.selected_prefix_tokens,
                requirements.max_capacity,
            )?;
            let additional_memory_bytes = root_snapshot_required_bytes
                .checked_add(max_pair_snapshot_required_bytes)
                .context("file-root plus pair checkpoint bytes overflow")?;
            let admission = loaded
                .qwen_execution_memory_admission_with_additional_bytes(
                    WIDTH,
                    requirements.max_capacity,
                    requirements.prefill_scratch_upper_bytes,
                    0,
                    additional_memory_bytes,
                )
                .context("price file-root Qwen execution")?;
            root_telemetry.snapshot_required_bytes = root_snapshot_required_bytes;
            root_telemetry.max_pair_snapshot_required_bytes = max_pair_snapshot_required_bytes;
            root_telemetry.additional_memory_bytes = additional_memory_bytes;
            root_telemetry.memory_admission_required_bytes = admission.required_bytes;
            root_telemetry.memory_admission_reason = admission.reason.as_str();
            if admission.admitted {
                let root_prompt = paired_prompt_refs
                    .first()
                    .copied()
                    .context("file-root plan has no paired source prompt")?;
                let (prepared, prefill_ms, snapshot_ms) = qwen_file_root::prepare(
                    loaded,
                    root_prompt,
                    args.prefill_chunk,
                    root_plan,
                    requirements.max_capacity,
                    root_snapshot_required_bytes,
                )?;
                root_telemetry.outcome = "prepared";
                root_telemetry.snapshot_bytes = prepared.checkpoint.snapshot_bytes();
                root_telemetry.prefill_ms = prefill_ms;
                root_telemetry.snapshot_ms = snapshot_ms;
                file_root = Some(prepared);
                admission
            } else {
                root_telemetry.outcome = "memory_fallback";
                root_telemetry.additional_memory_bytes = 0;
                loaded
                    .admit_independent_queue2(
                        requirements.max_capacity,
                        requirements.prefill_scratch_upper_bytes,
                    )
                    .context("admit two independent Qwen sessions after root fallback")?
            }
        } else {
            loaded
                .admit_independent_queue2(
                    requirements.max_capacity,
                    requirements.prefill_scratch_upper_bytes,
                )
                .context("admit two independent Qwen sessions")?
        };
        eprintln!(
            "concurrency_admission: width={WIDTH} prefill_scratch_upper_bytes={} additional_memory_bytes={} reason={} required_bytes={:?} working_set_headroom_bytes={:?}",
            requirements.prefill_scratch_upper_bytes,
            root_telemetry.additional_memory_bytes,
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
    let mut pair_index = 0usize;
    let mut pending_outputs = std::iter::repeat_with(|| None)
        .take(requests.len())
        .collect::<Vec<Option<RequestOutput>>>();
    let mut next_output = 0usize;
    let mut buffered_outputs = 0usize;
    for work in schedule.work {
        shutdown::checkpoint()?;
        match work {
            PairWork::Pair(indices) => {
                let pair = [&requests[indices[0]], &requests[indices[1]]];
                let (outputs, telemetry) = run_pair(
                    loaded,
                    tokenizer,
                    executor.as_mut().expect("planned concurrent executor"),
                    pair,
                    args,
                    &stop_tokens,
                    pair_index,
                    indices,
                    prefix_boundary_policy,
                    file_root.as_ref(),
                )
                .with_context(|| format!("run concurrent request pair {pair_index}"))?;
                for (index, output) in indices.into_iter().zip(outputs) {
                    ensure!(
                        pending_outputs[index].replace(output).is_none(),
                        "concurrent planner completed request {index} twice"
                    );
                }
                eprintln!(
                    "concurrency_pair: {}",
                    serde_json::to_string(&telemetry).context("serialize concurrency telemetry")?
                );
                if telemetry.file_root_restores > 0 {
                    root_telemetry.actual_pair_uses += 1;
                    root_telemetry.actual_avoided_prefix_evaluations =
                        root_telemetry.actual_pair_uses.saturating_sub(1);
                    root_telemetry.actual_avoided_prompt_tokens = root_telemetry
                        .selected_prefix_tokens
                        .saturating_mul(root_telemetry.actual_avoided_prefix_evaluations);
                }
                completed += WIDTH;
                buffered_outputs = buffered_outputs
                    .checked_add(WIDTH)
                    .context("concurrent output buffer accounting overflow")?;
                pair_index += 1;
            }
            PairWork::Serial(index) => {
                let request = &requests[index];
                let (output, _) =
                    run_jsonl_request(loaded, tokenizer, request, args, greedy_gpu_mode)
                        .with_context(|| {
                            format!("run concurrent serial tail request {}", request.id)
                        })?;
                ensure!(
                    pending_outputs[index].replace(output).is_none(),
                    "concurrent planner completed request {index} twice"
                );
                completed += 1;
                buffered_outputs = buffered_outputs
                    .checked_add(1)
                    .context("concurrent output buffer accounting overflow")?;
            }
        }
        let written = write_contiguous_outputs(stdout, &mut pending_outputs, &mut next_output)?;
        buffered_outputs = buffered_outputs
            .checked_sub(written)
            .context("concurrent output buffer accounting underflow")?;
        ensure!(
            buffered_outputs < PAIR_PLANNER_WINDOW,
            "concurrent output buffer exceeded its planning window"
        );
    }
    ensure!(
        completed == requests.len(),
        "concurrent JSONL completed {completed}/{} requests",
        requests.len()
    );
    ensure!(
        next_output == requests.len()
            && buffered_outputs == 0
            && pending_outputs.iter().all(Option::is_none),
        "concurrent JSONL output reorder buffer did not drain"
    );
    eprintln!(
        "concurrency_file_root: {}",
        serde_json::to_string(&root_telemetry).context("serialize Qwen file-root telemetry")?
    );
    Ok(completed)
}

pub(super) fn qwen_execution_admission_requirements(
    loaded: &LoadedModel,
    requests: &[PreparedJsonlRequest],
    args: &Args,
) -> Result<QwenExecutionAdmissionRequirements> {
    let max_capacity = requests
        .iter()
        .map(|request| jsonl_generation_capacity(request, args).map(|(_, capacity)| capacity))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("Qwen execution plan contained no requests")?;
    let prefill_scratch_upper_bytes = requests
        .iter()
        .map(|request| prefill_scratch_upper_bytes(loaded, request, args))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .max()
        .context("Qwen execution plan contained no requests")?;
    Ok(QwenExecutionAdmissionRequirements {
        max_capacity,
        prefill_scratch_upper_bytes,
    })
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

pub(super) fn prefill_scratch_upper_bytes(
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

fn common_prefix_tokens<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter()
        .zip(right)
        .take_while(|(left, right)| left == right)
        .count()
}

fn max_file_root_pair_snapshot_bytes(
    loaded: &LoadedModel,
    requests: &[PreparedJsonlRequest],
    schedule: &PairSchedule,
    args: &Args,
    boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    root_tokens: usize,
    max_capacity: usize,
) -> Result<u64> {
    let mut maximum = 0;
    for work in &schedule.work {
        let PairWork::Pair([left, right]) = *work else {
            continue;
        };
        let pair_plan = plan_prefix_fanout(
            &requests[left].prompt_ids,
            &requests[right].prompt_ids,
            args.prefill_chunk,
            prefix_fanout_enabled(),
            boundary_policy,
        );
        let prefix_tokens = pair_plan.selected_prefix_tokens.max(root_tokens);
        if prefix_tokens <= root_tokens {
            continue;
        }
        let bytes = loaded
            .estimate_checkpoint_boundary_sizes_unallocated(
                max_capacity,
                prefix_tokens,
                false,
                false,
            )
            .context("estimate pair checkpoint above file root")?
            .snapshot_bytes;
        maximum = maximum.max(bytes);
    }
    Ok(maximum)
}

fn pair_planner_enabled(default_enabled: bool) -> Result<bool> {
    let value = std::env::var_os(PAIR_PLANNER_ENV)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{PAIR_PLANNER_ENV} must be valid UTF-8"))
        })
        .transpose()?;
    parse_pair_planner_enabled(value.as_deref(), default_enabled)
}

fn parse_pair_planner_enabled(value: Option<&str>, default_enabled: bool) -> Result<bool> {
    let Some(value) = value else {
        return Ok(default_enabled);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{PAIR_PLANNER_ENV} must be a boolean"),
    }
}

fn plan_request_pairs<T: Ord>(
    prompts: &[&[T]],
    requested_tokens: &[usize],
    enabled: bool,
    selected_prefix_tokens: impl Fn(&[T], &[T]) -> usize,
) -> Result<PairSchedule> {
    ensure!(
        prompts.len() == requested_tokens.len(),
        "pair planner prompt/token-limit length mismatch"
    );
    let mut work = Vec::new();
    let mut prefix_affinity_pairs = 0usize;
    let mut depth_balanced_pairs = 0usize;
    for window_start in (0..prompts.len()).step_by(PAIR_PLANNER_WINDOW) {
        let window_end = (window_start + PAIR_PLANNER_WINDOW).min(prompts.len());
        let mut available = (window_start..window_end).collect::<Vec<_>>();
        let serial = (!available.len().is_multiple_of(WIDTH))
            .then(|| available.pop().expect("odd pair-planning window"));
        let mut window_work = Vec::new();
        if enabled {
            let mut edges = Vec::new();
            for (offset, &left) in available.iter().enumerate() {
                for &right in &available[offset + 1..] {
                    let prefix_tokens = selected_prefix_tokens(prompts[left], prompts[right]);
                    if prefix_tokens > 0 {
                        edges.push((
                            prefix_tokens,
                            requested_tokens[left].abs_diff(requested_tokens[right]),
                            left,
                            right,
                        ));
                    }
                }
            }
            edges.sort_by(|left, right| {
                right
                    .0
                    .cmp(&left.0)
                    .then_with(|| left.1.cmp(&right.1))
                    .then_with(|| left.2.cmp(&right.2))
                    .then_with(|| left.3.cmp(&right.3))
            });
            let mut matched = vec![false; window_end - window_start];
            for (_, _, left, right) in edges {
                let left_slot = left - window_start;
                let right_slot = right - window_start;
                if !matched[left_slot] && !matched[right_slot] {
                    matched[left_slot] = true;
                    matched[right_slot] = true;
                    window_work.push(PairWork::Pair([left, right]));
                    prefix_affinity_pairs += 1;
                }
            }
            available.retain(|&index| !matched[index - window_start]);
            available.sort_by_key(|&index| (requested_tokens[index], index));
            for pair in available.chunks_exact(WIDTH) {
                let mut pair = [pair[0], pair[1]];
                pair.sort_unstable();
                window_work.push(PairWork::Pair(pair));
                depth_balanced_pairs += 1;
            }
        } else {
            for pair in available.chunks_exact(WIDTH) {
                window_work.push(PairWork::Pair([pair[0], pair[1]]));
            }
        }
        if let Some(serial) = serial {
            window_work.push(PairWork::Serial(serial));
        }
        window_work.sort_by_key(|item| item.first_request_index());
        work.extend(window_work);
    }
    ensure!(
        work.iter()
            .map(|item| match item {
                PairWork::Pair(_) => WIDTH,
                PairWork::Serial(_) => 1,
            })
            .sum::<usize>()
            == prompts.len(),
        "pair planner request accounting drifted"
    );
    Ok(PairSchedule {
        work,
        prefix_affinity_pairs,
        depth_balanced_pairs,
    })
}

fn prefix_fanout_enabled() -> bool {
    std::env::var_os(PREFIX_FANOUT_ENV)
        .and_then(|value| value.into_string().ok())
        .map(|value| {
            !matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "0" | "false" | "no" | "off"
            )
        })
        .unwrap_or(true)
}

fn plan_prefix_fanout(
    left: &[i32],
    right: &[i32],
    prefill_chunk: PrefillChunkArg,
    enabled: bool,
    boundary_policy: QwenPrefixFanoutBoundaryPolicy,
) -> PrefixFanoutPlan {
    let common_prefix_tokens = common_prefix_tokens(left, right);
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
    let PrefillChunkArg::Fixed(chunk) = prefill_chunk else {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "auto_chunk_unsupported",
        };
    };
    if left == right {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: common_prefix_tokens,
            reason: "selected_identical",
        };
    }
    let chunk_aligned_prefix_tokens = common_prefix_tokens / chunk * chunk;
    let max_private_suffix_tokens = left
        .len()
        .saturating_sub(common_prefix_tokens)
        .max(right.len().saturating_sub(common_prefix_tokens));
    let selected_prefix_tokens = match boundary_policy {
        QwenPrefixFanoutBoundaryPolicy::ChunkAligned => chunk_aligned_prefix_tokens,
        QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
            if chunk_aligned_prefix_tokens >= PREFIX_FANOUT_MIN_TOKENS
                && max_private_suffix_tokens <= PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS =>
        {
            common_prefix_tokens
        }
        QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp => chunk_aligned_prefix_tokens,
        QwenPrefixFanoutBoundaryPolicy::ExactLcp => common_prefix_tokens,
    };
    if selected_prefix_tokens < PREFIX_FANOUT_MIN_TOKENS {
        PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "alignment_below_minimum",
        }
    } else {
        PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens,
            reason: match boundary_policy {
                QwenPrefixFanoutBoundaryPolicy::ExactLcp => "selected_exact_lcp",
                QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
                    if selected_prefix_tokens == common_prefix_tokens
                        && selected_prefix_tokens != chunk_aligned_prefix_tokens =>
                {
                    "selected_tiny_suffix_exact_lcp"
                }
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned
                | QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp
                    if selected_prefix_tokens == common_prefix_tokens =>
                {
                    "selected"
                }
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned
                | QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp => "selected_chunk_aligned",
            },
        }
    }
}

fn prepared_lane(
    request: &PreparedJsonlRequest,
    max_tokens: usize,
    sequence: Sequence,
    logits: Vec<f32>,
) -> PreparedLane {
    PreparedLane {
        id: request.id.clone(),
        prompt_tokens: request.prompt_ids.len(),
        max_tokens,
        sequence,
        logits,
        sampling: request.sampling,
    }
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

fn prepare_pair(
    loaded: &LoadedModel,
    requests: [&PreparedJsonlRequest; WIDTH],
    args: &Args,
    prefix_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    file_root: Option<&qwen_file_root::Prepared>,
) -> Result<PreparedPair> {
    let mut plan = plan_prefix_fanout(
        &requests[0].prompt_ids,
        &requests[1].prompt_ids,
        args.prefill_chunk,
        prefix_fanout_enabled(),
        prefix_boundary_policy,
    );
    let file_root_tokens = file_root
        .map(|root| root.plan.selected_prefix_tokens)
        .unwrap_or(0);
    if file_root_tokens > plan.selected_prefix_tokens {
        plan.selected_prefix_tokens = file_root_tokens;
        plan.reason = "selected_file_root";
    }
    if plan.selected_prefix_tokens == 0 {
        let (left, left_prefill_ms) = prepare_lane(loaded, requests[0], args)?;
        let (right, right_prefill_ms) = prepare_lane(loaded, requests[1], args)?;
        return Ok(PreparedPair {
            lanes: [left, right],
            plan,
            prefix_snapshot_bytes: 0,
            prefix_prefill_tokens: 0,
            prefix_prefill_ms: 0.0,
            prefix_snapshot_ms: 0.0,
            prefix_restore_ms: 0.0,
            private_prefill_ms: left_prefill_ms + right_prefill_ms,
            private_suffix_singleton_lanes: 0,
            private_suffix_singleton_tokens: 0,
            private_suffix_packed_lanes: 0,
            private_suffix_packed_tokens: 0,
            prefix_snapshot_required_bytes: 0,
            prefix_memory_admission_required_bytes: None,
            prefix_memory_admission_reason: "not_requested",
            file_root_tokens: 0,
            file_root_restores: 0,
            file_root_restore_ms: 0.0,
        });
    }

    let PrefillChunkArg::Fixed(requested_chunk) = args.prefill_chunk else {
        unreachable!("fanout planner rejects automatic chunks")
    };
    let (left_max_tokens, left_capacity) = jsonl_generation_capacity(requests[0], args)?;
    let (right_max_tokens, right_capacity) = jsonl_generation_capacity(requests[1], args)?;
    let max_prompt_tokens = requests
        .iter()
        .map(|request| request.prompt_ids.len())
        .max()
        .expect("fixed-width request pair");
    let chunk = requested_chunk.min(max_prompt_tokens.max(1));
    let mut scratch = allocate_legacy_prefill_scratch(loaded, chunk, max_prompt_tokens)
        .context("allocate concurrent shared-prefix scratch")?;
    let mut sequences = [
        loaded
            .create_sequence(SequenceConfig::new(left_capacity))
            .context("allocate concurrent shared-prefix source sequence")?,
        loaded
            .create_sequence(SequenceConfig::new(right_capacity))
            .context("allocate concurrent shared-prefix restore sequence")?,
    ];

    let prefix_len = plan.selected_prefix_tokens;
    let pair_snapshot_needed = prefix_len > file_root_tokens;
    let prefix_snapshot_required_bytes = if pair_snapshot_needed {
        loaded
            .estimate_checkpoint_boundary_sizes(&sequences[0], prefix_len, false, false)
            .context("estimate concurrent shared-prefix snapshot")?
            .snapshot_bytes
    } else {
        0
    };
    let prefix_memory_admission = if file_root.is_some() {
        None
    } else {
        Some(evaluate_metal_memory_admission(
            prefix_snapshot_required_bytes,
            PREFIX_FANOUT_RESERVE_BYTES,
            loaded.context().memory_signals(),
            true,
        ))
    };
    let prefix_memory_admission_required_bytes = prefix_memory_admission
        .as_ref()
        .and_then(|admission| admission.required_bytes);
    let prefix_memory_admission_reason = prefix_memory_admission
        .as_ref()
        .map(|admission| admission.reason.as_str())
        .unwrap_or("file_root_pre_admitted");
    if prefix_memory_admission
        .as_ref()
        .is_some_and(|admission| !admission.admitted)
    {
        plan.selected_prefix_tokens = 0;
        plan.reason = "memory_fallback";
        let forward = loaded.forward();
        let mut private_prefill_ms = 0.0;
        let (left_logits, left_ms) = prefill_span(
            &forward,
            &mut sequences[0],
            &mut scratch,
            &requests[0].prompt_ids,
            0,
        )
        .context("prefill concurrent memory-fallback lane 0")?;
        private_prefill_ms += left_ms;
        let (right_logits, right_ms) = prefill_span(
            &forward,
            &mut sequences[1],
            &mut scratch,
            &requests[1].prompt_ids,
            0,
        )
        .context("prefill concurrent memory-fallback lane 1")?;
        private_prefill_ms += right_ms;
        let [left_sequence, right_sequence] = sequences;
        return Ok(PreparedPair {
            lanes: [
                prepared_lane(requests[0], left_max_tokens, left_sequence, left_logits),
                prepared_lane(requests[1], right_max_tokens, right_sequence, right_logits),
            ],
            plan,
            prefix_snapshot_bytes: 0,
            prefix_prefill_tokens: 0,
            prefix_prefill_ms: 0.0,
            prefix_snapshot_ms: 0.0,
            prefix_restore_ms: 0.0,
            private_prefill_ms,
            private_suffix_singleton_lanes: 0,
            private_suffix_singleton_tokens: 0,
            private_suffix_packed_lanes: 0,
            private_suffix_packed_tokens: 0,
            prefix_snapshot_required_bytes,
            prefix_memory_admission_required_bytes,
            prefix_memory_admission_reason,
            file_root_tokens: 0,
            file_root_restores: 0,
            file_root_restore_ms: 0.0,
        });
    }

    shutdown::checkpoint()?;
    let forward = loaded.forward();
    let mut file_root_restores = 0usize;
    let mut file_root_restore_ms = 0.0;
    let (source_prefix_logits, prefix_prefill_ms) = if let Some(file_root) = file_root {
        let restore_t0 = Instant::now();
        let restored = loaded
            .restore_prepared_checkpoint(
                &file_root.checkpoint,
                &mut sequences[0],
                &requests[0].prompt_ids,
            )
            .context("restore file root into concurrent source lane")?;
        file_root_restore_ms += restore_t0.elapsed().as_secs_f64() * 1e3;
        file_root_restores += 1;
        ensure!(
            restored.matched_prefix_len == file_root_tokens
                && restored.restored_prefix_len == file_root_tokens,
            "file-root restore selected an unexpected source boundary"
        );
        if prefix_len == file_root_tokens {
            (restored.exact_final_logits, 0.0)
        } else {
            let (logits, bridge_ms) = prefill_span(
                &forward,
                &mut sequences[0],
                &mut scratch,
                &requests[0].prompt_ids[file_root_tokens..prefix_len],
                file_root_tokens,
            )
            .context("prefill pair prefix above file root")?;
            (Some(logits), bridge_ms)
        }
    } else {
        let (logits, prefill_ms) = prefill_span(
            &forward,
            &mut sequences[0],
            &mut scratch,
            &requests[0].prompt_ids[..prefix_len],
            0,
        )
        .context("prefill concurrent shared prefix")?;
        (Some(logits), prefill_ms)
    };
    shutdown::checkpoint()?;

    let mut pair_checkpoint = None;
    let mut prefix_snapshot_ms = 0.0;
    let mut prefix_snapshot_bytes = 0;
    if pair_snapshot_needed {
        let snapshot_t0 = Instant::now();
        let prepared = loaded
            .prepare_checkpoint_boundary(
                &sequences[0],
                requests[0].prompt_ids[..prefix_len].to_vec(),
                None,
                None,
            )
            .context("capture concurrent shared prefix")?;
        prefix_snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;
        prefix_snapshot_bytes = prepared.snapshot_bytes();
        ensure!(
            prefix_snapshot_bytes == prefix_snapshot_required_bytes,
            "concurrent prefix snapshot bytes {prefix_snapshot_bytes} != estimate {prefix_snapshot_required_bytes}"
        );
        pair_checkpoint = Some(prepared);
    }
    shutdown::checkpoint()?;

    let (prefix_restore_ms, target_prefix_logits) = if let Some(prepared) = pair_checkpoint.as_ref()
    {
        let restore_t0 = Instant::now();
        let restored = loaded
            .restore_prepared_checkpoint(prepared, &mut sequences[1], &requests[1].prompt_ids)
            .context("restore concurrent shared prefix")?;
        let prefix_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        ensure!(
            restored.matched_prefix_len == prefix_len && restored.restored_prefix_len == prefix_len,
            "concurrent shared-prefix restore selected an unexpected boundary"
        );
        (prefix_restore_ms, None)
    } else {
        let file_root = file_root.expect("root-only pair has file root");
        let restore_t0 = Instant::now();
        let restored = loaded
            .restore_prepared_checkpoint(
                &file_root.checkpoint,
                &mut sequences[1],
                &requests[1].prompt_ids,
            )
            .context("restore file root into concurrent restore lane")?;
        let restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        file_root_restore_ms += restore_ms;
        file_root_restores += 1;
        ensure!(
            restored.matched_prefix_len == prefix_len && restored.restored_prefix_len == prefix_len,
            "file-root restore selected an unexpected target boundary"
        );
        (0.0, restored.exact_final_logits)
    };
    shutdown::checkpoint()?;

    let source_prefix_logits = if requests[0].prompt_ids.len() == prefix_len {
        Some(source_prefix_logits.context("exact source prefix omitted final logits")?)
    } else {
        source_prefix_logits
    };
    let target_prefix_logits = if requests[1].prompt_ids.len() == prefix_len {
        Some(match target_prefix_logits {
            Some(logits) => logits,
            None => source_prefix_logits
                .as_ref()
                .context("exact restored prefix omitted final logits")?
                .clone(),
        })
    } else {
        target_prefix_logits
    };

    let mut private_prefill_ms = 0.0;
    let mut private_suffix_singleton_lanes = 0usize;
    let mut private_suffix_singleton_tokens = 0usize;
    let mut private_suffix_packed_lanes = 0usize;
    let mut private_suffix_packed_tokens = 0usize;
    let left_logits = if requests[0].prompt_ids.len() == prefix_len {
        source_prefix_logits.expect("validated exact source logits")
    } else {
        let result = prefill_private_suffix(
            &forward,
            &mut sequences[0],
            &mut scratch,
            &requests[0].prompt_ids[prefix_len..],
            prefix_len,
        )
        .context("prefill concurrent shared-prefix lane 0 suffix")?;
        private_prefill_ms += result.ms;
        match result.mode {
            PrivateSuffixExecutionMode::Packed => {
                private_suffix_packed_lanes += 1;
                private_suffix_packed_tokens += requests[0].prompt_ids.len() - prefix_len;
            }
            PrivateSuffixExecutionMode::Singleton => {
                private_suffix_singleton_lanes += 1;
                private_suffix_singleton_tokens += requests[0].prompt_ids.len() - prefix_len;
            }
        }
        result.logits
    };
    shutdown::checkpoint()?;
    let right_logits = if requests[1].prompt_ids.len() == prefix_len {
        target_prefix_logits.expect("validated exact target logits")
    } else {
        let result = prefill_private_suffix(
            &forward,
            &mut sequences[1],
            &mut scratch,
            &requests[1].prompt_ids[prefix_len..],
            prefix_len,
        )
        .context("prefill concurrent shared-prefix lane 1 suffix")?;
        private_prefill_ms += result.ms;
        match result.mode {
            PrivateSuffixExecutionMode::Packed => {
                private_suffix_packed_lanes += 1;
                private_suffix_packed_tokens += requests[1].prompt_ids.len() - prefix_len;
            }
            PrivateSuffixExecutionMode::Singleton => {
                private_suffix_singleton_lanes += 1;
                private_suffix_singleton_tokens += requests[1].prompt_ids.len() - prefix_len;
            }
        }
        result.logits
    };
    drop(pair_checkpoint);
    drop(scratch);
    let [left_sequence, right_sequence] = sequences;

    Ok(PreparedPair {
        lanes: [
            prepared_lane(requests[0], left_max_tokens, left_sequence, left_logits),
            prepared_lane(requests[1], right_max_tokens, right_sequence, right_logits),
        ],
        plan,
        prefix_snapshot_bytes,
        prefix_prefill_tokens: prefix_len.saturating_sub(file_root_tokens),
        prefix_prefill_ms,
        prefix_snapshot_ms,
        prefix_restore_ms,
        private_prefill_ms,
        private_suffix_singleton_lanes,
        private_suffix_singleton_tokens,
        private_suffix_packed_lanes,
        private_suffix_packed_tokens,
        prefix_snapshot_required_bytes,
        prefix_memory_admission_required_bytes,
        prefix_memory_admission_reason,
        file_root_tokens,
        file_root_restores,
        file_root_restore_ms,
    })
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
    request_indices: [usize; WIDTH],
    prefix_boundary_policy: QwenPrefixFanoutBoundaryPolicy,
    file_root: Option<&qwen_file_root::Prepared>,
) -> Result<([RequestOutput; WIDTH], PairTelemetry)> {
    let prepare_t0 = Instant::now();
    let prepared = prepare_pair(loaded, requests, args, prefix_boundary_policy, file_root)?;
    let [left, right] = prepared.lanes;
    let requested_tokens = [left.max_tokens, right.max_tokens];
    let prepare_ms = prepare_t0.elapsed().as_secs_f64() * 1e3;
    let prefill_ms = prepared.prefix_prefill_ms + prepared.private_prefill_ms;

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
            schema_version: 6,
            backend: "qwen_independent_queues_v6",
            pair_index,
            request_indices,
            prompt_tokens: [requests[0].prompt_ids.len(), requests[1].prompt_ids.len()],
            requested_tokens,
            generated_tokens,
            productive_transitions,
            paired_transitions,
            serial_tail_transitions,
            common_prefix_tokens: prepared.plan.common_prefix_tokens,
            prefix_fanout_tokens: prepared.plan.selected_prefix_tokens,
            prefix_fanout_reason: prepared.plan.reason,
            prefix_fanout_boundary_policy: prefix_boundary_policy.as_str(),
            prefix_fanout_min_tokens: PREFIX_FANOUT_MIN_TOKENS,
            prefix_snapshot_bytes: prepared.prefix_snapshot_bytes,
            prefix_prefill_tokens: prepared.prefix_prefill_tokens,
            prefix_prefill_ms: prepared.prefix_prefill_ms,
            prefix_snapshot_ms: prepared.prefix_snapshot_ms,
            prefix_restore_ms: prepared.prefix_restore_ms,
            private_prefill_ms: prepared.private_prefill_ms,
            private_suffix_singleton_max_tokens: PRIVATE_SUFFIX_SINGLETON_MAX_TOKENS,
            private_suffix_singleton_lanes: prepared.private_suffix_singleton_lanes,
            private_suffix_singleton_tokens: prepared.private_suffix_singleton_tokens,
            private_suffix_packed_lanes: prepared.private_suffix_packed_lanes,
            private_suffix_packed_tokens: prepared.private_suffix_packed_tokens,
            prefix_snapshot_required_bytes: prepared.prefix_snapshot_required_bytes,
            prefix_memory_admission_required_bytes: prepared.prefix_memory_admission_required_bytes,
            prefix_memory_admission_reason: prepared.prefix_memory_admission_reason,
            file_root_tokens: prepared.file_root_tokens,
            file_root_restores: prepared.file_root_restores,
            file_root_restore_ms: prepared.file_root_restore_ms,
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

fn write_contiguous_outputs(
    stdout: &mut impl Write,
    pending: &mut [Option<RequestOutput>],
    next_output: &mut usize,
) -> Result<usize> {
    let mut ready = Vec::new();
    while let Some(output) = pending.get_mut(*next_output).and_then(Option::take) {
        ready.push(output);
        *next_output += 1;
    }
    let written = ready.len();
    if !ready.is_empty() {
        write_outputs(stdout, &ready)?;
    }
    Ok(written)
}

fn plan_deepseek_prefix_fanout(
    left: &[u32],
    right: &[u32],
    chunk_tokens: usize,
    enabled: bool,
) -> PrefixFanoutPlan {
    let common_prefix_tokens = common_prefix_tokens(left, right);
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
    if left == right {
        return PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: common_prefix_tokens,
            reason: "selected_identical",
        };
    }

    let right_boundaries = deepseek_v4_prefill_chunk_ranges(right.len(), chunk_tokens)
        .into_iter()
        .map(|range| range.end)
        .collect::<std::collections::BTreeSet<_>>();
    let selected_prefix_tokens = deepseek_v4_prefill_chunk_ranges(left.len(), chunk_tokens)
        .into_iter()
        .map(|range| range.end)
        .filter(|&end| end <= common_prefix_tokens && right_boundaries.contains(&end))
        .max()
        .unwrap_or(0);
    if selected_prefix_tokens < PREFIX_FANOUT_MIN_TOKENS {
        PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens: 0,
            reason: "alignment_below_minimum",
        }
    } else {
        PrefixFanoutPlan {
            common_prefix_tokens,
            selected_prefix_tokens,
            reason: if selected_prefix_tokens == common_prefix_tokens {
                "selected"
            } else {
                "selected_chunk_aligned"
            },
        }
    }
}

fn transient_deepseek_model_content_id(
    residency: &Arc<DeepSeekV4MetalResidency>,
    pair_index: usize,
) -> Result<DeepSeekV4ModelContentId> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock precedes Unix epoch")?;
    let mut hasher = Sha256::new();
    hasher.update(b"qwen-dsv4-transient-concurrency-snapshot-v1\0");
    hasher.update(std::process::id().to_le_bytes());
    hasher.update((Arc::as_ptr(residency) as usize).to_le_bytes());
    hasher.update(pair_index.to_le_bytes());
    hasher.update(now.as_nanos().to_le_bytes());
    Ok(DeepSeekV4ModelContentId::new(hasher.finalize().into()))
}

fn deepseek_snapshot_restore_workspace_bytes(residency: &DeepSeekV4MetalResidency) -> Result<u64> {
    u64::from(residency.config().layer_count)
        .checked_mul(128)
        .and_then(|value| value.checked_mul(u64::from(residency.config().key_length)))
        .and_then(|value| value.checked_mul(std::mem::size_of::<u16>() as u64))
        .context("DeepSeek snapshot restore workspace byte overflow")
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
    let (mut session, session_ms) =
        create_deepseek_lane_session(ctx, residency, selector_plan, &request.id, None)?;

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
        evaluated_prefill_tokens: request.prompt_token_ids.len(),
        request,
        session,
        logits,
        session_ms,
        prefill_mode,
        prefill_ms,
        prefix_prefill_ms: 0.0,
        private_prefill_ms: prefill_ms,
        prefix_snapshot_ms: 0.0,
        prefix_restore_ms: 0.0,
        prefix_snapshot_payload_bytes: 0,
        exact_prompt_logits_reused: false,
    })
}

fn create_deepseek_lane_session(
    ctx: &MetalContext,
    residency: &Arc<DeepSeekV4MetalResidency>,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    request_id: &str,
    model_content_id: Option<DeepSeekV4ModelContentId>,
) -> Result<(DeepSeekV4Session, f64)> {
    let session_t0 = Instant::now();
    let mut session = match model_content_id {
        Some(model_content_id) => DeepSeekV4Session::new_shared_with_model_content_id(
            ctx,
            residency.clone(),
            model_content_id,
        ),
        None => DeepSeekV4Session::new_shared(ctx, residency.clone()),
    }
    .with_context(|| format!("create concurrent DeepSeek session for request {request_id}"))?;
    selector_plan.seal_session(&mut session, request_id)?;
    Ok((session, session_t0.elapsed().as_secs_f64() * 1e3))
}

#[allow(clippy::too_many_arguments)]
fn prepare_deepseek_shared_source(
    ctx: &MetalContext,
    residency: &Arc<DeepSeekV4MetalResidency>,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    request: DeepSeekV4PreparedRequest,
    prefill_chunk_tokens: usize,
    vocab_size: u32,
    prefix_len: usize,
    model_content_id: DeepSeekV4ModelContentId,
) -> Result<(
    DeepSeekPreparedLane,
    Arc<DeepSeekV4CausalSnapshot>,
    Arc<Vec<f32>>,
)> {
    shutdown::checkpoint()?;
    ensure!(
        prefix_len > 0 && prefix_len <= request.prompt_token_ids.len(),
        "request {} shared prefix {} is outside prompt length {}",
        request.id,
        prefix_len,
        request.prompt_token_ids.len()
    );
    let (mut session, session_ms) = create_deepseek_lane_session(
        ctx,
        residency,
        selector_plan,
        &request.id,
        Some(model_content_id),
    )?;

    let prefix_t0 = Instant::now();
    execute_deepseek_v4_prompt_suffix(
        &mut session,
        ctx,
        &request.prompt_token_ids[..prefix_len],
        prefill_chunk_tokens,
    )
    .with_context(|| format!("prefill shared prefix for request {}", request.id))?;
    let prefix_logits = Arc::new(
        copy_deepseek_v4_logits(&session, vocab_size, "concurrent shared prefix")
            .with_context(|| format!("copy request {} shared-prefix logits", request.id))?,
    );
    let prefix_prefill_ms = prefix_t0.elapsed().as_secs_f64() * 1e3;
    shutdown::checkpoint()?;

    let snapshot_t0 = Instant::now();
    let snapshot = Arc::new(
        session
            .capture_causal_snapshot()
            .with_context(|| format!("capture request {} shared prefix", request.id))?,
    );
    let prefix_snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;
    ensure!(
        snapshot.next_position() as usize == prefix_len
            && snapshot.prefix_tokens() == &request.prompt_token_ids[..prefix_len],
        "request {} captured an unexpected shared-prefix frontier",
        request.id
    );

    let suffix = &request.prompt_token_ids[prefix_len..];
    let (logits, private_prefill_ms, prefill_mode) = if suffix.is_empty() {
        (
            prefix_logits.as_ref().clone(),
            0.0,
            "shared_source_exact_logits",
        )
    } else {
        let suffix_t0 = Instant::now();
        execute_deepseek_v4_prompt_suffix(&mut session, ctx, suffix, prefill_chunk_tokens)
            .with_context(|| format!("prefill request {} private suffix", request.id))?;
        let logits = copy_deepseek_v4_logits(&session, vocab_size, "concurrent private suffix")
            .with_context(|| format!("copy request {} private-suffix logits", request.id))?;
        (
            logits,
            suffix_t0.elapsed().as_secs_f64() * 1e3,
            "shared_source_suffix",
        )
    };
    deepseek_v4_debug_dump_logits_sha256(&request.id, &logits);
    let prefix_snapshot_payload_bytes = snapshot.payload_bytes();
    Ok((
        DeepSeekPreparedLane {
            evaluated_prefill_tokens: request.prompt_token_ids.len(),
            request,
            session,
            logits,
            session_ms,
            prefill_mode,
            prefill_ms: prefix_prefill_ms + private_prefill_ms,
            prefix_prefill_ms,
            private_prefill_ms,
            prefix_snapshot_ms,
            prefix_restore_ms: 0.0,
            prefix_snapshot_payload_bytes,
            exact_prompt_logits_reused: false,
        },
        snapshot,
        prefix_logits,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prepare_deepseek_shared_restore(
    ctx: &MetalContext,
    residency: &Arc<DeepSeekV4MetalResidency>,
    selector_plan: &DeepSeekV4MultigroupSelectorPlan,
    request: DeepSeekV4PreparedRequest,
    prefill_chunk_tokens: usize,
    vocab_size: u32,
    prefix_len: usize,
    model_content_id: DeepSeekV4ModelContentId,
    snapshot: &DeepSeekV4CausalSnapshot,
    prefix_logits: &[f32],
) -> Result<DeepSeekPreparedLane> {
    shutdown::checkpoint()?;
    ensure!(
        prefix_len > 0
            && request.prompt_token_ids.len() >= prefix_len
            && &request.prompt_token_ids[..prefix_len] == snapshot.prefix_tokens(),
        "request {} does not match the restored shared prefix",
        request.id
    );
    let (mut session, session_ms) = create_deepseek_lane_session(
        ctx,
        residency,
        selector_plan,
        &request.id,
        Some(model_content_id),
    )?;
    let restore_t0 = Instant::now();
    session
        .restore_causal_snapshot(snapshot)
        .with_context(|| format!("restore request {} shared prefix", request.id))?;
    let prefix_restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
    ensure!(
        session.next_position() as usize == prefix_len,
        "request {} restored an unexpected shared-prefix frontier",
        request.id
    );
    shutdown::checkpoint()?;

    let suffix = &request.prompt_token_ids[prefix_len..];
    let (logits, private_prefill_ms, prefill_mode, exact_prompt_logits_reused) = if suffix
        .is_empty()
    {
        (
            prefix_logits.to_vec(),
            0.0,
            "shared_restore_exact_logits",
            true,
        )
    } else {
        let suffix_t0 = Instant::now();
        execute_deepseek_v4_prompt_suffix(&mut session, ctx, suffix, prefill_chunk_tokens)
            .with_context(|| format!("prefill request {} restored suffix", request.id))?;
        let logits = copy_deepseek_v4_logits(&session, vocab_size, "restored private suffix")
            .with_context(|| format!("copy request {} restored-suffix logits", request.id))?;
        (
            logits,
            suffix_t0.elapsed().as_secs_f64() * 1e3,
            "shared_restore_suffix",
            false,
        )
    };
    deepseek_v4_debug_dump_logits_sha256(&request.id, &logits);
    Ok(DeepSeekPreparedLane {
        evaluated_prefill_tokens: suffix.len(),
        request,
        session,
        logits,
        session_ms,
        prefill_mode,
        prefill_ms: private_prefill_ms,
        prefix_prefill_ms: 0.0,
        private_prefill_ms,
        prefix_snapshot_ms: 0.0,
        prefix_restore_ms,
        prefix_snapshot_payload_bytes: 0,
        exact_prompt_logits_reused,
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
        evaluated_prefill_tokens: lane.evaluated_prefill_tokens,
        prefill_ms: lane.prefill_ms,
        prefix_prefill_ms: lane.prefix_prefill_ms,
        private_prefill_ms: lane.private_prefill_ms,
        prefix_snapshot_ms: lane.prefix_snapshot_ms,
        prefix_restore_ms: lane.prefix_restore_ms,
        prefix_snapshot_payload_bytes: lane.prefix_snapshot_payload_bytes,
        exact_prompt_logits_reused: lane.exact_prompt_logits_reused,
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
        completion.evaluated_prefill_tokens as f64 / (completion.prefill_ms / 1e3)
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
            "deepseek_v4 stats: request={} line={} prompt_kind=raw prefill_mode={} prefill_chunk_cap={} prompt_tokens={} evaluated_prefill_tokens={} ",
            "generated_tokens={} transitions={} stop_reason={} session_ms={:.1} prefill_ms={:.1} prefill_tps={:.2} ",
            "generation_ms={:.1} decode_tps={:.2} concurrency={} build_commit={} build_dirty={}"
        ),
        completion.output.id,
        completion.line,
        completion.prefill_mode,
        prefill_chunk_tokens,
        completion.output.prompt_tokens,
        completion.evaluated_prefill_tokens,
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
    prepared_tx: mpsc::Sender<DeepSeekPrepareSignal>,
) -> Result<DeepSeekCompletedLane> {
    let prepare_control = control
        .recv()
        .context("receive DeepSeek concurrency prepare control")?;
    let prepared = match prepare_control {
        DeepSeekWorkerControl::PrepareSerial => prepare_deepseek_lane(
            ctx,
            &residency,
            selector_plan,
            request,
            prefill_chunk_tokens,
            vocab_size,
        )
        .map(|lane| (lane, DeepSeekPrepareSignal::Ready)),
        DeepSeekWorkerControl::PrepareSharedSource {
            prefix_len,
            model_content_id,
        } => prepare_deepseek_shared_source(
            ctx,
            &residency,
            selector_plan,
            request,
            prefill_chunk_tokens,
            vocab_size,
            prefix_len,
            model_content_id,
        )
        .map(|(lane, snapshot, prefix_logits)| {
            (
                lane,
                DeepSeekPrepareSignal::SharedSource {
                    snapshot,
                    prefix_logits,
                },
            )
        }),
        DeepSeekWorkerControl::PrepareSharedRestore {
            prefix_len,
            model_content_id,
            snapshot,
            prefix_logits,
        } => prepare_deepseek_shared_restore(
            ctx,
            &residency,
            selector_plan,
            request,
            prefill_chunk_tokens,
            vocab_size,
            prefix_len,
            model_content_id,
            &snapshot,
            prefix_logits.as_slice(),
        )
        .map(|lane| (lane, DeepSeekPrepareSignal::Ready)),
        DeepSeekWorkerControl::Abort => {
            bail!("DeepSeek concurrency worker aborted before prepare")
        }
        DeepSeekWorkerControl::Generate => {
            bail!("DeepSeek concurrency worker received generation before prepare")
        }
    };
    prepared_tx
        .send(
            prepared
                .as_ref()
                .map(|(_, signal)| signal.clone())
                .unwrap_or(DeepSeekPrepareSignal::Failed),
        )
        .context("publish DeepSeek concurrency prepare status")?;
    match control
        .recv()
        .context("receive DeepSeek concurrency generation control")?
    {
        DeepSeekWorkerControl::Generate => {
            let (lane, prepare_signal) = prepared?;
            drop(prepare_signal);
            generate_deepseek_lane(ctx, tokenizer, vocab_size, stop_tokens, lane)
        }
        DeepSeekWorkerControl::Abort => match prepared {
            Ok(_) => bail!("DeepSeek concurrency worker aborted after peer setup failure"),
            Err(error) => Err(error),
        },
        DeepSeekWorkerControl::PrepareSerial
        | DeepSeekWorkerControl::PrepareSharedSource { .. }
        | DeepSeekWorkerControl::PrepareSharedRestore { .. } => {
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
    request_indices: [usize; WIDTH],
) -> Result<([DeepSeekCompletedLane; WIDTH], DeepSeekPairTelemetry)> {
    let prompt_tokens = [requests[0].prompt_tokens, requests[1].prompt_tokens];
    let requested_tokens = [requests[0].max_tokens, requests[1].max_tokens];
    let mut prefix_fanout = plan_deepseek_prefix_fanout(
        &requests[0].prompt_token_ids,
        &requests[1].prompt_token_ids,
        prefill_chunk_tokens,
        prefix_fanout_enabled(),
    );
    let mut prefix_snapshot_priced_upper_bytes = 0u64;
    let mut prefix_restore_workspace_bytes = 0u64;
    let mut fanout_memory_admission_required_bytes = None;
    let mut fanout_memory_admission_reason = "not_requested";
    let mut transient_model_content_id = None;
    if prefix_fanout.selected_prefix_tokens > 0 {
        prefix_snapshot_priced_upper_bytes = causal_snapshot_record_bytes(
            residency.config(),
            residency.session_capacity(),
            u32::try_from(prefix_fanout.selected_prefix_tokens)
                .context("DeepSeek shared prefix does not fit u32")?,
            u64::MAX,
        )
        .context("estimate DeepSeek concurrent snapshot record")?;
        prefix_restore_workspace_bytes = deepseek_snapshot_restore_workspace_bytes(residency)?;
        let incremental_bytes = session_priced_upper_bytes
            .checked_mul(WIDTH as u64)
            .and_then(|bytes| bytes.checked_add(prefix_snapshot_priced_upper_bytes))
            .and_then(|bytes| bytes.checked_add(prefix_restore_workspace_bytes))
            .context("DeepSeek fanout admission byte overflow")?;
        let admission = evaluate_metal_memory_admission(
            incremental_bytes,
            DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
            contexts[0].memory_signals(),
            true,
        );
        fanout_memory_admission_required_bytes = admission.required_bytes;
        fanout_memory_admission_reason = admission.reason.as_str();
        if admission.admitted {
            transient_model_content_id =
                Some(transient_deepseek_model_content_id(residency, pair_index)?);
        } else {
            prefix_fanout.selected_prefix_tokens = 0;
            prefix_fanout.reason = "memory_fallback";
        }
    }
    let [left_request, right_request] = requests;
    let pair_t0 = Instant::now();
    let before_sessions = contexts[0].current_allocated_size();
    let runtime_allocation_upper_bytes = session_priced_upper_bytes
        .checked_mul(WIDTH as u64)
        .and_then(|bytes| bytes.checked_add(DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES))
        .context("DeepSeek two-session runtime byte overflow")?;
    let (completed, prepare_ms, concurrent_generation_ms, session_allocation_delta_bytes) =
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
            let prefix_len = prefix_fanout.selected_prefix_tokens;
            let left_control = match transient_model_content_id {
                Some(model_content_id) => DeepSeekWorkerControl::PrepareSharedSource {
                    prefix_len,
                    model_content_id,
                },
                None => DeepSeekWorkerControl::PrepareSerial,
            };
            left_control_tx
                .send(left_control)
                .context("start DeepSeek concurrency lane 0 prefill")?;
            let left_signal = left_prepared_rx
                .recv()
                .unwrap_or(DeepSeekPrepareSignal::Failed);
            let (left_prepared, right_control) = match (transient_model_content_id, left_signal) {
                (
                    Some(model_content_id),
                    DeepSeekPrepareSignal::SharedSource {
                        snapshot,
                        prefix_logits,
                    },
                ) => (
                    true,
                    Some(DeepSeekWorkerControl::PrepareSharedRestore {
                        prefix_len,
                        model_content_id,
                        snapshot,
                        prefix_logits,
                    }),
                ),
                (None, DeepSeekPrepareSignal::Ready) => {
                    (true, Some(DeepSeekWorkerControl::PrepareSerial))
                }
                _ => (false, None),
            };
            let right_prepared = if let Some(right_control) = right_control {
                right_control_tx
                    .send(right_control)
                    .context("start DeepSeek concurrency lane 1 prefill")?;
                matches!(
                    right_prepared_rx
                        .recv()
                        .unwrap_or(DeepSeekPrepareSignal::Failed),
                    DeepSeekPrepareSignal::Ready
                )
            } else {
                false
            };
            let prepare_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

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
                prepare_ms,
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
    let evaluated_prefill_tokens = completed
        .iter()
        .map(|lane| lane.evaluated_prefill_tokens)
        .sum::<usize>();
    let model_prefill_ms = completed.iter().map(|lane| lane.prefill_ms).sum::<f64>();
    let prefix_prefill_ms = completed
        .iter()
        .map(|lane| lane.prefix_prefill_ms)
        .sum::<f64>();
    let private_prefill_ms = completed
        .iter()
        .map(|lane| lane.private_prefill_ms)
        .sum::<f64>();
    let prefix_snapshot_ms = completed
        .iter()
        .map(|lane| lane.prefix_snapshot_ms)
        .sum::<f64>();
    let prefix_restore_ms = completed
        .iter()
        .map(|lane| lane.prefix_restore_ms)
        .sum::<f64>();
    let prefix_snapshot_payload_bytes = completed
        .iter()
        .map(|lane| lane.prefix_snapshot_payload_bytes)
        .sum::<u64>();
    let exact_prompt_logits_reused = completed.iter().any(|lane| lane.exact_prompt_logits_reused);
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
            schema_version: 3,
            backend: "deepseek_v4_independent_queues_v3",
            pair_index,
            request_indices,
            prompt_tokens,
            requested_tokens,
            generated_tokens,
            productive_transitions,
            common_prefix_tokens: prefix_fanout.common_prefix_tokens,
            prefix_fanout_tokens: prefix_fanout.selected_prefix_tokens,
            prefix_fanout_reason: prefix_fanout.reason,
            prefix_fanout_min_tokens: PREFIX_FANOUT_MIN_TOKENS,
            prefix_snapshot_payload_bytes,
            prefix_snapshot_priced_upper_bytes,
            prefix_restore_workspace_bytes,
            prefix_prefill_ms,
            prefix_snapshot_ms,
            prefix_restore_ms,
            private_prefill_ms,
            exact_prompt_logits_reused,
            fanout_memory_admission_required_bytes,
            fanout_memory_admission_reason,
            pair_wall_ms,
            prepare_ms,
            evaluated_prefill_tokens,
            model_prefill_ms,
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
    let prompt_refs = requests
        .iter()
        .map(|request| request.prompt_token_ids.as_slice())
        .collect::<Vec<_>>();
    let requested_tokens = requests
        .iter()
        .map(|request| request.max_tokens)
        .collect::<Vec<_>>();
    let planner_enabled = pair_planner_enabled(false)?;
    let fanout_enabled = prefix_fanout_enabled();
    let schedule = plan_request_pairs(
        &prompt_refs,
        &requested_tokens,
        planner_enabled,
        |left, right| {
            plan_deepseek_prefix_fanout(left, right, prefill_chunk_tokens, fanout_enabled)
                .selected_prefix_tokens
        },
    )?;
    let pair_count = schedule
        .work
        .iter()
        .filter(|work| matches!(work, PairWork::Pair(_)))
        .count();
    eprintln!(
        "deepseek_v4 concurrency_planner: {}",
        serde_json::to_string(&PairPlannerTelemetry {
            schema_version: 1,
            backend: "deepseek_v4_pair_affinity_v1",
            enabled: planner_enabled,
            requests: requests.len(),
            prefix_affinity_pairs: schedule.prefix_affinity_pairs,
            depth_balanced_pairs: schedule.depth_balanced_pairs,
            serial_requests: requests.len() - pair_count * WIDTH,
            planning_window: PAIR_PLANNER_WINDOW,
            strategy: if planner_enabled {
                "bounded_prefix_affinity_then_depth"
            } else {
                "input_order"
            },
            output_order: "input",
        })
        .context("serialize DeepSeek concurrency planner telemetry")?
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
    let request_count = requests.len();
    let mut requests = requests.into_iter().map(Some).collect::<Vec<_>>();
    let mut pending_outputs = std::iter::repeat_with(|| None)
        .take(request_count)
        .collect::<Vec<Option<RequestOutput>>>();
    let mut next_output = 0usize;
    let mut buffered_outputs = 0usize;
    let mut pair_index = 0usize;
    for work in schedule.work {
        shutdown::checkpoint()?;
        match work {
            PairWork::Pair(indices) => {
                let left = requests[indices[0]]
                    .take()
                    .context("DeepSeek pair planner reused its left request")?;
                let right = requests[indices[1]]
                    .take()
                    .context("DeepSeek pair planner reused its right request")?;
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
                    indices,
                )?;
                for lane in &completed {
                    emit_deepseek_completion(selector_plan, lane, prefill_chunk_tokens, WIDTH)?;
                }
                for (index, output) in indices.into_iter().zip(completed.map(|lane| lane.output)) {
                    ensure!(
                        pending_outputs[index].replace(output).is_none(),
                        "DeepSeek pair planner completed request {index} twice"
                    );
                }
                eprintln!(
                    "deepseek_v4 concurrency_pair: {}",
                    serde_json::to_string(&telemetry)
                        .context("serialize DeepSeek concurrency telemetry")?
                );
                executed += WIDTH;
                buffered_outputs = buffered_outputs
                    .checked_add(WIDTH)
                    .context("DeepSeek output buffer accounting overflow")?;
                pair_index += 1;
            }
            PairWork::Serial(index) => {
                let request = requests[index]
                    .take()
                    .context("DeepSeek pair planner reused its serial request")?;
                let before_session = contexts[0].current_allocated_size();
                let lane = prepare_deepseek_lane(
                    &contexts[0],
                    &residency,
                    selector_plan,
                    request,
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
                ensure!(
                    pending_outputs[index].replace(completion.output).is_none(),
                    "DeepSeek pair planner completed request {index} twice"
                );
                executed += 1;
                buffered_outputs = buffered_outputs
                    .checked_add(1)
                    .context("DeepSeek output buffer accounting overflow")?;
            }
        }
        let written =
            write_contiguous_outputs(&mut stdout, &mut pending_outputs, &mut next_output)?;
        buffered_outputs = buffered_outputs
            .checked_sub(written)
            .context("DeepSeek output buffer accounting underflow")?;
        ensure!(
            buffered_outputs < PAIR_PLANNER_WINDOW,
            "DeepSeek output buffer exceeded its planning window"
        );
    }
    ensure!(
        executed == request_count
            && next_output == request_count
            && buffered_outputs == 0
            && requests.iter().all(Option::is_none)
            && pending_outputs.iter().all(Option::is_none),
        "DeepSeek concurrency planner did not drain every request"
    );
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
    fn pair_planner_recovers_interleaved_prefix_affinity() {
        let mut a0 = vec![1; PREFIX_FANOUT_MIN_TOKENS];
        a0.push(10);
        let mut a1 = vec![1; PREFIX_FANOUT_MIN_TOKENS];
        a1.push(11);
        let mut b0 = vec![2; PREFIX_FANOUT_MIN_TOKENS];
        b0.push(10);
        let mut b1 = vec![2; PREFIX_FANOUT_MIN_TOKENS];
        b1.push(11);
        let prompts = [a0.as_slice(), b0.as_slice(), a1.as_slice(), b1.as_slice()];
        let schedule = plan_request_pairs(&prompts, &[20, 20, 21, 21], true, |left, right| {
            let common = common_prefix_tokens(left, right);
            if common >= PREFIX_FANOUT_MIN_TOKENS {
                common
            } else {
                0
            }
        })
        .unwrap();
        assert_eq!(
            schedule.work,
            vec![PairWork::Pair([0, 2]), PairWork::Pair([1, 3])]
        );
        assert_eq!(schedule.prefix_affinity_pairs, 2);
        assert_eq!(schedule.depth_balanced_pairs, 0);
    }

    #[test]
    fn pair_planner_prefers_the_deepest_available_prefix_edges() {
        let base = vec![7; PREFIX_FANOUT_MIN_TOKENS];
        let a = base.clone();
        let mut b = base.clone();
        b.push(0);
        let mut c = b.clone();
        c.push(0);
        let mut d = base;
        d.push(1);
        let prompts = [a.as_slice(), b.as_slice(), c.as_slice(), d.as_slice()];
        let schedule = plan_request_pairs(&prompts, &[20; 4], true, |left, right| {
            let common = common_prefix_tokens(left, right);
            if common >= PREFIX_FANOUT_MIN_TOKENS {
                common
            } else {
                0
            }
        })
        .unwrap();
        assert_eq!(
            schedule.work,
            vec![PairWork::Pair([0, 3]), PairWork::Pair([1, 2])]
        );
        assert_eq!(schedule.prefix_affinity_pairs, 2);
    }

    #[test]
    fn pair_planner_balances_nonprefix_generation_depth() {
        let prompt_storage = [vec![0], vec![1], vec![2], vec![3], vec![4]];
        let prompts = prompt_storage.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let schedule = plan_request_pairs(&prompts, &[2, 100, 3, 99, 50], true, |_, _| 0).unwrap();
        assert_eq!(
            schedule.work,
            vec![
                PairWork::Pair([0, 2]),
                PairWork::Pair([1, 3]),
                PairWork::Serial(4),
            ]
        );
        assert_eq!(schedule.prefix_affinity_pairs, 0);
        assert_eq!(schedule.depth_balanced_pairs, 2);
    }

    #[test]
    fn disabled_pair_planner_preserves_input_order() {
        let prompt_storage = [vec![0], vec![1], vec![2]];
        let prompts = prompt_storage.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let schedule = plan_request_pairs(&prompts, &[3, 1, 2], false, |_, _| 0).unwrap();
        assert_eq!(
            schedule.work,
            vec![PairWork::Pair([0, 1]), PairWork::Serial(2)]
        );
        assert_eq!(schedule.prefix_affinity_pairs, 0);
        assert_eq!(schedule.depth_balanced_pairs, 0);
    }

    #[test]
    fn pair_planner_policy_defaults_and_invalid_values_are_explicit() {
        assert!(parse_pair_planner_enabled(None, true).unwrap());
        assert!(!parse_pair_planner_enabled(None, false).unwrap());
        assert!(!parse_pair_planner_enabled(Some("off"), true).unwrap());
        assert!(parse_pair_planner_enabled(Some("YES"), false).unwrap());
        assert!(parse_pair_planner_enabled(Some("maybe"), true).is_err());
    }

    #[test]
    fn pair_planner_never_reorders_across_bounded_windows() {
        let prompt_storage = (0..(PAIR_PLANNER_WINDOW + WIDTH))
            .map(|index| {
                let mut prompt = vec![7; PREFIX_FANOUT_MIN_TOKENS];
                prompt.push(index);
                prompt
            })
            .collect::<Vec<_>>();
        let prompts = prompt_storage.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let schedule =
            plan_request_pairs(&prompts, &vec![20; prompts.len()], true, |left, right| {
                common_prefix_tokens(left, right)
            })
            .unwrap();
        assert!(schedule.work.iter().all(|work| match work {
            PairWork::Pair([left, right]) => {
                left / PAIR_PLANNER_WINDOW == right / PAIR_PLANNER_WINDOW
            }
            PairWork::Serial(_) => true,
        }));
    }

    #[test]
    fn prefix_fanout_requires_a_stable_shared_boundary() {
        let identical = vec![7; 300];
        assert_eq!(
            plan_prefix_fanout(
                &identical,
                &identical,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: 300,
                selected_prefix_tokens: 300,
                reason: "selected_identical",
            }
        );

        let mut right = vec![7; 900];
        right[700] = 8;
        assert_eq!(
            plan_prefix_fanout(
                &vec![7; 900],
                &right,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: 700,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan_prefix_fanout(
                &vec![7; 900],
                &right,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::ExactLcp,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: 700,
                selected_prefix_tokens: 700,
                reason: "selected_exact_lcp",
            }
        );
        assert_eq!(
            plan_prefix_fanout(
                &vec![7; 900],
                &right,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: 700,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
        let tiny_left = vec![7; 705];
        let mut tiny_right = tiny_left.clone();
        tiny_right[700] = 8;
        assert_eq!(
            plan_prefix_fanout(
                &tiny_left,
                &tiny_right,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp,
            ),
            PrefixFanoutPlan {
                common_prefix_tokens: 700,
                selected_prefix_tokens: 700,
                reason: "selected_tiny_suffix_exact_lcp",
            }
        );

        let shorter = vec![7; 300];
        let longer = vec![7; 600];
        let expected = PrefixFanoutPlan {
            common_prefix_tokens: 300,
            selected_prefix_tokens: 0,
            reason: "alignment_below_minimum",
        };
        assert_eq!(
            plan_prefix_fanout(
                &shorter,
                &longer,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            expected
        );
        assert_eq!(
            plan_prefix_fanout(
                &longer,
                &shorter,
                PrefillChunkArg::Fixed(512),
                true,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            ),
            expected
        );
    }

    #[test]
    fn tiny_suffix_exact_lcp_policy_freezes_six_token_boundary() {
        fn plan(
            common_prefix_tokens: usize,
            private_suffix_tokens: usize,
            boundary_policy: QwenPrefixFanoutBoundaryPolicy,
        ) -> PrefixFanoutPlan {
            let left = vec![7; common_prefix_tokens + private_suffix_tokens];
            let mut right = left.clone();
            right[common_prefix_tokens] = 8;
            plan_prefix_fanout(
                &left,
                &right,
                PrefillChunkArg::Fixed(512),
                true,
                boundary_policy,
            )
        }

        assert_eq!(
            plan(513, 6, QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 513,
                reason: "selected_tiny_suffix_exact_lcp",
            }
        );
        assert_eq!(
            plan(513, 7, QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan(513, 6, QwenPrefixFanoutBoundaryPolicy::ChunkAligned),
            PrefixFanoutPlan {
                common_prefix_tokens: 513,
                selected_prefix_tokens: 512,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan(300, 5, QwenPrefixFanoutBoundaryPolicy::TinySuffixExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 300,
                selected_prefix_tokens: 0,
                reason: "alignment_below_minimum",
            }
        );
        assert_eq!(
            plan(300, 5, QwenPrefixFanoutBoundaryPolicy::ExactLcp),
            PrefixFanoutPlan {
                common_prefix_tokens: 300,
                selected_prefix_tokens: 300,
                reason: "selected_exact_lcp",
            }
        );
    }

    #[test]
    fn prefix_fanout_fails_closed_for_policy_controls() {
        let prompt = vec![7; 512];
        assert_eq!(
            plan_prefix_fanout(
                &prompt,
                &prompt,
                PrefillChunkArg::Fixed(512),
                false,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            )
            .reason,
            "disabled"
        );
        assert_eq!(
            plan_prefix_fanout(
                &prompt,
                &prompt,
                PrefillChunkArg::Auto,
                true,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            )
            .reason,
            "auto_chunk_unsupported"
        );
        assert_eq!(
            plan_prefix_fanout(
                &vec![7; PREFIX_FANOUT_MIN_TOKENS - 1],
                &vec![7; PREFIX_FANOUT_MIN_TOKENS - 1],
                PrefillChunkArg::Fixed(128),
                true,
                QwenPrefixFanoutBoundaryPolicy::ChunkAligned,
            )
            .reason,
            "below_minimum"
        );
    }

    #[test]
    fn deepseek_prefix_fanout_intersects_real_chunk_boundaries() {
        let mut right = vec![7u32; 6_650];
        right[6_475] = 8;
        assert_eq!(
            plan_deepseek_prefix_fanout(&vec![7; 6_642], &right, 4_096, true),
            PrefixFanoutPlan {
                common_prefix_tokens: 6_475,
                selected_prefix_tokens: 6_144,
                reason: "selected_chunk_aligned",
            }
        );

        let shorter = vec![7u32; 6_000];
        let longer = vec![7u32; 8_000];
        assert_eq!(
            plan_deepseek_prefix_fanout(&shorter, &longer, 4_096, true),
            PrefixFanoutPlan {
                common_prefix_tokens: 6_000,
                selected_prefix_tokens: 4_096,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan_deepseek_prefix_fanout(&longer, &shorter, 4_096, true),
            PrefixFanoutPlan {
                common_prefix_tokens: 6_000,
                selected_prefix_tokens: 4_096,
                reason: "selected_chunk_aligned",
            }
        );
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
