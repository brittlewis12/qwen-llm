//! Coefficient sweeps and resident sweep cohorts.

use super::*;

pub(super) const RUN_COHORT_SCHEMA: &str = "qwen.lens.run_cohort";

pub(super) const RUN_COHORT_SCHEMA_VERSION: u32 = 1;

pub(super) const SWEEP_SCHEMA: &str = "qwen.lens.coefficient_sweep";

pub(super) const SWEEP_SCHEMA_VERSION: u32 = 3;

pub(super) const SWEEP_MANIFEST_NAME: &str = "manifest.json";

pub(super) const MAX_SWEEP_ARMS: usize = 64;

pub(super) const SWEEP_COHORT_SCHEMA: &str = "qwen.lens.coefficient_sweep_cohort";

pub(super) const SWEEP_COHORT_SCHEMA_VERSION: u32 = 1;

pub(super) const MIN_SWEEP_COHORT_REQUESTS: usize = 2;

pub(super) const MAX_SWEEP_COHORT_RECORD_BYTES: usize = 1024 * 1024;

pub(super) const MAX_SWEEP_COHORT_FILE_BYTES: usize = 32 * 1024 * 1024;

pub(super) const MAX_SWEEP_COHORT_ID_BYTES: usize = 128;

pub(super) const MAX_SWEEP_COHORT_MESSAGES_BYTES: usize = 16 * 1024 * 1024;

pub(super) const MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND: u64 = 1_000_000;

pub(crate) const MAX_SWEEP_BUNDLE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("sweep_input")
        .required(true)
        .multiple(false)
        .args([
            "prompt",
            "token_ids",
            "user",
            "messages",
            "open_responses",
            "requests_jsonl",
        ])
))]
pub(crate) struct CoefficientSweepArgs {
    /// Existing private identity cache; data exact bindings always hash retained bytes.
    #[arg(long)]
    pub(super) identity_cache: Option<PathBuf>,
    /// Ordinary dense or MoE Qwen GGUF model, loaded once for every arm.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,

    /// Strict Lens plan JSON file whose authored coefficients remain unchanged.
    #[arg(long)]
    pub(super) plan: PathBuf,

    /// Exact operation ID whose coefficient is replaced in each arm.
    #[arg(long)]
    pub(super) operation: String,

    /// Ordered finite coefficients; duplicates and zero controls are preserved.
    #[arg(
        long,
        value_delimiter = ',',
        required = true,
        allow_hyphen_values = true
    )]
    pub(super) coefficients: Vec<f32>,

    /// Raw untemplated text; tokenizer-configured specials are enabled by default.
    #[arg(long, visible_alias = "raw-prompt", allow_hyphen_values = true)]
    pub(super) prompt: Option<String>,

    /// Literal comma-separated token IDs; no specials are added.
    #[arg(long, value_delimiter = ',')]
    pub(super) token_ids: Option<Vec<i32>>,

    /// One user message rendered with the model-family template; '-' reads stdin once.
    #[arg(long, value_name = "TEXT|-")]
    pub(super) user: Option<String>,

    /// Add one system message before --user.
    #[arg(long, requires = "user")]
    pub(super) system: Option<String>,

    /// JSON message array or wrapper with a `messages` array.
    #[arg(long, value_name = "FILE|-")]
    pub(super) messages: Option<PathBuf>,

    /// Open Responses request JSON rendered by the exact qwen serve prompt path.
    #[arg(long, visible_alias = "responses-input", value_name = "FILE|-")]
    pub(super) open_responses: Option<PathBuf>,

    /// Strict message-file-only JSONL cohort; record paths are relative to this file.
    #[arg(
        long,
        value_name = "PATH",
        conflicts_with_all = [
            "prompt",
            "token_ids",
            "user",
            "system",
            "messages",
            "open_responses",
            "message_mode",
            "no_special_tokens"
        ]
    )]
    pub(super) requests_jsonl: Option<PathBuf>,

    /// Generation transition for --user/--messages; supported values depend on the model.
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["prompt", "token_ids", "open_responses", "requests_jsonl"]
    )]
    pub(super) message_mode: Option<LensMessageMode>,

    /// Disable tokenizer-configured specials for --prompt.
    #[arg(
        long,
        requires = "prompt",
        conflicts_with_all = [
            "token_ids",
            "user",
            "messages",
            "open_responses",
            "requests_jsonl"
        ]
    )]
    pub(super) no_special_tokens: bool,

    /// Maximum number of generated tokens per fresh arm.
    #[arg(long, default_value_t = 32)]
    pub(super) max_new_tokens: usize,

    /// Use one qualified passive-span schedule for every arm, or force serial prefill.
    #[arg(long, value_enum, default_value_t = PrefillExecution::Auto)]
    pub(super) prefill_execution: PrefillExecution,

    /// Native sampler temperature; each arm restarts from the same seed.
    #[arg(long, default_value_t = 0.0)]
    pub(super) temperature: f32,

    /// Native sampler top-k; zero disables the filter.
    #[arg(long, default_value_t = 0)]
    pub(super) top_k: usize,

    /// Native sampler nucleus threshold.
    #[arg(long, default_value_t = 1.0)]
    pub(super) top_p: f32,

    /// Native sampler minimum probability threshold.
    #[arg(long, default_value_t = 0.0)]
    pub(super) min_p: f32,

    /// Native sampler seed, reset for every arm.
    #[arg(long, default_value_t = 0)]
    pub(super) seed: u64,

    /// New immutable sweep directory, published only after every arm succeeds.
    #[arg(long)]
    pub(super) output: PathBuf,
}

impl CoefficientSweepArgs {
    pub(super) fn arm_run_args(&self) -> LensRunArgs {
        LensRunArgs {
            model: self.model.clone(),
            plan: self.plan.clone(),
            identity_cache: self.identity_cache.clone(),
            prompt: self.prompt.clone(),
            token_ids: self.token_ids.clone(),
            user: self.user.clone(),
            system: self.system.clone(),
            messages: self.messages.clone(),
            open_responses: self.open_responses.clone(),
            requests_jsonl: None,
            message_mode: self.message_mode,
            no_special_tokens: self.no_special_tokens,
            max_new_tokens: self.max_new_tokens,
            prefill_execution: self.prefill_execution,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            seed: self.seed,
            output: None,
            format: None,
            output_dir: None,
        }
    }
}

pub(super) struct LoadedRunCohortRequests {
    pub(super) canonical_path: PathBuf,
    pub(super) records: Vec<(usize, LensCohortRequest)>,
}

pub(super) struct PreflightRunCohortRequest {
    pub(super) source_line: usize,
    pub(super) id: String,
    pub(super) prepared_input: PreparedLensInput,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct RunCohortBounds {
    pub(super) request_count: usize,
    pub(super) aggregate_prompt_tokens: usize,
    pub(super) work_upper_bound: u64,
    pub(super) forward_upper_bound: u64,
    pub(super) sample_upper_bound: u64,
    pub(super) max_request_forwards: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunCohortManifest {
    pub(super) schema: String,
    pub(super) schema_version: u32,
    pub(super) producer: SweepProducer,
    pub(super) execution_policy: String,
    pub(super) requests_jsonl_path: PathBuf,
    pub(super) canonical_plan_path: PathBuf,
    pub(super) source_plan: LensPlan,
    pub(super) model_path: PathBuf,
    pub(super) sampler: RunSampler,
    pub(super) max_new_tokens: usize,
    pub(super) prefill_execution: PrefillExecution,
    pub(super) request_count: usize,
    pub(super) aggregate_prompt_tokens: usize,
    pub(super) work_upper_bound: u64,
    pub(super) forward_upper_bound: u64,
    pub(super) sample_upper_bound: u64,
    pub(super) max_request_forwards: usize,
    pub(super) cumulative_child_bytes: u64,
    pub(super) runs: Vec<RunCohortChild>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct RunCohortChild {
    pub(super) index: usize,
    pub(super) id: String,
    pub(super) source_line: usize,
    pub(super) path: String,
    pub(super) input_source: String,
    pub(super) prompt_token_count: usize,
    pub(super) artifact_byte_length: u64,
}

pub(super) struct RunCohortManifestBasis {
    pub(super) producer: SweepProducer,
    pub(super) requests_jsonl_path: PathBuf,
    pub(super) canonical_plan_path: PathBuf,
    pub(super) source_plan: LensPlan,
    pub(super) model_path: PathBuf,
    pub(super) sampler: RunSampler,
    pub(super) max_new_tokens: usize,
    pub(super) prefill_execution: PrefillExecution,
    pub(super) bounds: RunCohortBounds,
}

impl RunCohortManifestBasis {
    pub(super) fn build(
        &self,
        cumulative_child_bytes: u64,
        runs: Vec<RunCohortChild>,
    ) -> RunCohortManifest {
        RunCohortManifest {
            schema: RUN_COHORT_SCHEMA.into(),
            schema_version: RUN_COHORT_SCHEMA_VERSION,
            producer: self.producer.clone(),
            execution_policy:
                "resident_model_shared_prepared_plan_serial_request_order_fresh_sequence_and_sampler_request_local_binding_no_cross_request_batching"
                    .into(),
            requests_jsonl_path: self.requests_jsonl_path.clone(),
            canonical_plan_path: self.canonical_plan_path.clone(),
            source_plan: self.source_plan.clone(),
            model_path: self.model_path.clone(),
            sampler: self.sampler,
            max_new_tokens: self.max_new_tokens,
            prefill_execution: self.prefill_execution,
            request_count: self.bounds.request_count,
            aggregate_prompt_tokens: self.bounds.aggregate_prompt_tokens,
            work_upper_bound: self.bounds.work_upper_bound,
            forward_upper_bound: self.bounds.forward_upper_bound,
            sample_upper_bound: self.bounds.sample_upper_bound,
            max_request_forwards: self.bounds.max_request_forwards,
            cumulative_child_bytes,
            runs,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct SweepProducer {
    pub(crate) build_commit: String,
    pub(crate) build_dirty: String,
    pub(crate) build_source_state: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoefficientSweepArm {
    pub(crate) index: usize,
    pub(crate) coefficient: f32,
    pub(crate) artifact: String,
    pub(crate) byte_length: u64,
    pub(crate) blake3: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct CoefficientSweepManifest {
    pub(crate) schema: String,
    pub(crate) schema_version: u32,
    pub(crate) producer: SweepProducer,
    pub(crate) canonical_source_plan_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_plan: Option<LensPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) source_plan_canonical_json_blake3: Option<String>,
    pub(crate) operation_id: String,
    pub(crate) coefficients: Vec<f32>,
    pub(crate) arms: Vec<CoefficientSweepArm>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq)]
#[serde(deny_unknown_fields)]
pub(super) struct SweepCohortRequestRecord {
    pub(super) id: String,
    pub(super) messages: PathBuf,
    pub(super) message_mode: Option<LensMessageMode>,
}

pub(super) struct LoadedSweepCohortRequests {
    pub(super) canonical_path: PathBuf,
    pub(super) blake3: String,
    pub(super) records: Vec<(usize, SweepCohortRequestRecord)>,
}

pub(super) struct PreflightSweepCohortRequest {
    pub(super) source_line: usize,
    pub(super) id: String,
    pub(super) messages_path: PathBuf,
    pub(super) messages_blake3: String,
    pub(super) arm_args: LensRunArgs,
    pub(super) prepared_input: PreparedLensInput,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SweepCohortManifest {
    pub(super) schema: String,
    pub(super) schema_version: u32,
    pub(super) producer: SweepProducer,
    pub(super) requests_jsonl_path: PathBuf,
    pub(super) requests_jsonl_blake3: String,
    pub(super) canonical_source_plan_path: PathBuf,
    pub(super) source_plan: LensPlan,
    pub(super) source_plan_canonical_json_blake3: String,
    pub(super) model_path: PathBuf,
    pub(super) operation_id: String,
    pub(super) coefficients: Vec<f32>,
    pub(super) sampler: RunSampler,
    pub(super) max_new_tokens: usize,
    pub(super) prefill_execution: PrefillExecution,
    pub(super) execution_policy: String,
    pub(super) planned_request_count: usize,
    pub(super) planned_total_arm_count: usize,
    pub(super) transition_upper_bound: u64,
    pub(super) cumulative_serialized_child_bytes: u64,
    pub(super) sweeps: Vec<SweepCohortChild>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SweepCohortChild {
    pub(super) index: usize,
    pub(super) id: String,
    pub(super) source_line: usize,
    pub(super) path: String,
    pub(super) prompt_token_count: usize,
    pub(super) messages_path: PathBuf,
    pub(super) messages_blake3: String,
    pub(super) serialized_byte_length: u64,
    pub(super) manifest_byte_length: u64,
    pub(super) manifest_blake3: String,
}

pub(super) struct BuiltSweepBundle {
    pub(super) summaries: Vec<SweepArmSummary>,
    pub(super) serialized_byte_length: u64,
    pub(super) manifest_byte_length: u64,
    pub(super) manifest_blake3: String,
}

pub(super) struct CapturedSweepMessages {
    pub(super) path: PathBuf,
    pub(super) bytes: Vec<u8>,
    pub(super) blake3: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct SweepCohortPlanBounds {
    pub(super) request_count: usize,
    pub(super) total_arm_count: usize,
    pub(super) transition_upper_bound: u64,
}

pub(super) struct SweepCohortManifestBasis {
    pub(super) producer: SweepProducer,
    pub(super) requests_jsonl_path: PathBuf,
    pub(super) requests_jsonl_blake3: String,
    pub(super) canonical_source_plan_path: PathBuf,
    pub(super) source_plan: LensPlan,
    pub(super) source_plan_canonical_json_blake3: String,
    pub(super) model_path: PathBuf,
    pub(super) operation_id: String,
    pub(super) coefficients: Vec<f32>,
    pub(super) sampler: RunSampler,
    pub(super) max_new_tokens: usize,
    pub(super) prefill_execution: PrefillExecution,
    pub(super) bounds: SweepCohortPlanBounds,
}

impl SweepCohortManifestBasis {
    pub(super) fn build(
        &self,
        cumulative_serialized_child_bytes: u64,
        sweeps: Vec<SweepCohortChild>,
    ) -> SweepCohortManifest {
        SweepCohortManifest {
            schema: SWEEP_COHORT_SCHEMA.into(),
            schema_version: SWEEP_COHORT_SCHEMA_VERSION,
            producer: self.producer.clone(),
            requests_jsonl_path: self.requests_jsonl_path.clone(),
            requests_jsonl_blake3: self.requests_jsonl_blake3.clone(),
            canonical_source_plan_path: self.canonical_source_plan_path.clone(),
            source_plan: self.source_plan.clone(),
            source_plan_canonical_json_blake3: self.source_plan_canonical_json_blake3.clone(),
            model_path: self.model_path.clone(),
            operation_id: self.operation_id.clone(),
            coefficients: self.coefficients.clone(),
            sampler: self.sampler,
            max_new_tokens: self.max_new_tokens,
            prefill_execution: self.prefill_execution,
            execution_policy:
                "serial_prompts_serial_arms_fresh_sequence_and_sampler_no_batched_generation".into(),
            planned_request_count: self.bounds.request_count,
            planned_total_arm_count: self.bounds.total_arm_count,
            transition_upper_bound: self.bounds.transition_upper_bound,
            cumulative_serialized_child_bytes,
            sweeps,
        }
    }
}

pub(super) struct SweepArmSummary {
    pub(super) index: usize,
    pub(super) coefficient: f32,
    pub(super) decoded_text: String,
    pub(super) stop_reason: String,
    pub(super) operation_application_count: usize,
    pub(super) live_readout_count: usize,
}

pub(super) fn run_cohort(args: LensRunArgs) -> Result<()> {
    let requests = read_run_cohort_requests(
        args.requests_jsonl
            .as_deref()
            .context("--requests-jsonl is required in cohort mode")?,
    )?;
    let output_path = crate::resolve_output_path(
        args.output_dir
            .as_deref()
            .context("--output-dir is required in cohort mode")?,
    )?;
    ensure_new_bundle_output(&output_path)?;

    let plan_path = std::fs::canonicalize(&args.plan)
        .with_context(|| format!("resolve plan {}", args.plan.display()))?;
    let source_plan = parse_plan_bytes(&crate::read_regular_file_bounded(
        &plan_path,
        MAX_PLAN_BYTES,
    )?)
    .with_context(|| format!("parse Lens plan {}", plan_path.display()))?;
    validate_plan(&source_plan)?;
    validate_ordinary_plan(&source_plan)?;
    let plan_dir = plan_path.parent().unwrap_or_else(|| Path::new("."));
    let full_transports = open_full_transports(&source_plan, plan_dir)?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    ensure!(
        !crate::muse_lens_artifact::is_muse_architecture(gguf.architecture().as_deref()),
        "qwen-lens run cohorts currently support ordinary Qwen only; Muse Glimmer is not supported"
    );
    let family = ModelFamily::detect(&gguf).context("model has no supported Qwen architecture")?;
    ensure!(
        matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "qwen-lens run cohorts currently support ordinary Qwen only"
    );
    let model_context_tokens = gguf.declared_context_length()?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load model tokenizer")?;
    let sampler = run_sampler(&args);
    Sampler::new(SamplingConfig {
        temperature: sampler.temperature,
        top_k: sampler.top_k,
        top_p: sampler.top_p,
        min_p: sampler.min_p,
        seed: sampler.seed,
    })
    .context("validate run cohort sampler")?;

    let mut preflight_requests = Vec::new();
    preflight_requests
        .try_reserve_exact(requests.records.len())
        .context("allocate run cohort preflight requests")?;
    let mut aggregate_prompt_tokens = 0usize;
    let mut work_upper_bound = 0u64;
    let mut forward_upper_bound = 0u64;
    let mut sample_upper_bound = 0u64;
    let mut max_request_forwards = 0usize;
    for (source_line, request) in requests.records {
        let prepared_input =
            prepare_qwen_model_input(request.input_spec(), family, &gguf, &tokenizer)
                .with_context(|| format!("prepare run cohort request {:?}", request.id))?;
        validate_sweep_prompt(
            &prepared_input.token_ids,
            tokenizer.n_vocab(),
            args.max_new_tokens,
            model_context_tokens,
        )
        .with_context(|| format!("preflight run cohort request {:?}", request.id))?;
        let request_forwards =
            required_forward_count(prepared_input.token_ids.len(), args.max_new_tokens)?;
        aggregate_prompt_tokens = aggregate_prompt_tokens
            .checked_add(prepared_input.token_ids.len())
            .context("run cohort aggregate prompt token count overflow")?;
        work_upper_bound = work_upper_bound
            .checked_add(
                u64::try_from(prepared_input.token_ids.len())
                    .context("run cohort prompt token count")?
                    .checked_add(
                        u64::try_from(args.max_new_tokens)
                            .context("run cohort generation bound")?,
                    )
                    .context("run cohort request work bound overflow")?,
            )
            .context("run cohort work bound overflow")?;
        forward_upper_bound = forward_upper_bound
            .checked_add(u64::try_from(request_forwards).context("run cohort forward bound")?)
            .context("run cohort forward bound overflow")?;
        sample_upper_bound = sample_upper_bound
            .checked_add(u64::try_from(args.max_new_tokens).context("run cohort sample bound")?)
            .context("run cohort sample bound overflow")?;
        max_request_forwards = max_request_forwards.max(request_forwards);

        let bound_plan = bind_plan_positions(
            &source_plan,
            &prepared_input.rendering,
            prepared_input.token_ids.len(),
        )
        .with_context(|| format!("bind run cohort request {:?}", request.id))?;
        validate_reachable_scopes(
            &bound_plan.resolved,
            prepared_input.token_ids.len(),
            args.max_new_tokens,
        )
        .with_context(|| format!("validate run cohort request {:?} scopes", request.id))?;
        preflight_requests.push(PreflightRunCohortRequest {
            source_line,
            id: request.id,
            prepared_input,
        });
    }
    let bounds = RunCohortBounds {
        request_count: preflight_requests.len(),
        aggregate_prompt_tokens,
        work_upper_bound,
        forward_upper_bound,
        sample_upper_bound,
        max_request_forwards,
    };
    let manifest_basis = RunCohortManifestBasis {
        producer: current_sweep_producer(),
        requests_jsonl_path: requests.canonical_path,
        canonical_plan_path: plan_path.clone(),
        source_plan: source_plan.clone(),
        model_path: args.model.clone(),
        sampler,
        max_new_tokens: args.max_new_tokens,
        prefill_execution: args.prefill_execution,
        bounds,
    };
    crate::shutdown::checkpoint()?;
    let mut full_transports = bind_full_transports(
        full_transports,
        &source_plan,
        plan_dir,
        &gguf,
        args.identity_cache.as_deref(),
    )?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf_with_intent(
            gguf,
            args.model.clone(),
            LoadedModelConfig::default(),
            ModelLoadIntent::Reusable,
        )
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_runtime(loaded.gguf(), loaded.arch().kind, loaded.arch().n_layer)?;
    for request in &preflight_requests {
        ensure!(
            request
                .prepared_input
                .token_ids
                .iter()
                .all(|&token| token >= 0 && (token as u32) < loaded.arch().vocab_size),
            "run cohort request {:?} contains a token outside the deployed vocabulary",
            request.id
        );
    }
    let first_request = preflight_requests
        .first()
        .context("run cohort has no preflight requests")?;
    let first_bound_plan = bind_plan_positions(
        &source_plan,
        &first_request.prepared_input.rendering,
        first_request.prepared_input.token_ids.len(),
    )
    .with_context(|| format!("bind run cohort request {:?}", first_request.id))?;
    let execution = prepare_execution_plan(
        &first_bound_plan.resolved,
        plan_dir,
        &loaded,
        &mut full_transports,
    )?;
    drop(first_bound_plan);
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()?
        .into_iter()
        .collect::<HashSet<_>>();

    stage_and_publish_bundle(&output_path, |staging| {
        let mut children = Vec::new();
        children
            .try_reserve_exact(preflight_requests.len())
            .context("allocate run cohort manifest entries")?;
        let mut cumulative_child_bytes = 0u64;
        for (index, request) in preflight_requests.iter().enumerate() {
            let bound_plan = bind_plan_positions(
                &source_plan,
                &request.prepared_input.rendering,
                request.prepared_input.token_ids.len(),
            )
            .with_context(|| format!("bind run cohort request {:?}", request.id))?;
            let schedule =
                CompiledEventSchedule::compile(&bound_plan.resolved, loaded.arch().n_layer)?;
            let mut prefill = prepare_ordinary_prefill(
                &loaded,
                &schedule,
                &bound_plan.resolved,
                args.prefill_execution,
                RunExecutionScheduleBasis::EffectivePlan,
                request.prepared_input.token_ids.len(),
                args.max_new_tokens,
            )?;
            prefill.execution.validate_against_plan(
                "ordinary_qwen",
                &bound_plan.resolved,
                request.prepared_input.token_ids.len(),
            )?;
            let result = execute_ordinary_arm(
                &loaded,
                &tokenizer,
                &execution,
                &bound_plan.resolved,
                &schedule,
                &request.prepared_input.token_ids,
                args.max_new_tokens,
                sampler,
                &stop_tokens,
                &mut prefill,
            )?;
            let artifact = build_run_output(
                &args.model,
                sampler,
                args.max_new_tokens,
                "ordinary_qwen",
                &plan_path,
                bound_plan,
                &request.prepared_input,
                result,
                prefill.execution,
                None,
            );
            let path = format!("run-{index:06}.json");
            let artifact_bytes =
                write_new_run_output(&staging.join(&path), &artifact, MAX_RUN_ARTIFACT_BYTES)?;
            cumulative_child_bytes = cumulative_child_bytes
                .checked_add(
                    u64::try_from(artifact_bytes)
                        .context("run cohort child artifact byte length")?,
                )
                .context("run cohort child artifact byte count overflow")?;
            children.push(RunCohortChild {
                index,
                id: request.id.clone(),
                source_line: request.source_line,
                path,
                input_source: request.prepared_input.source.into(),
                prompt_token_count: request.prepared_input.token_ids.len(),
                artifact_byte_length: u64::try_from(artifact_bytes)
                    .context("run cohort child artifact byte length")?,
            });
        }
        let manifest = manifest_basis.build(cumulative_child_bytes, children);
        let manifest_bytes = serialize_run_cohort_manifest(&manifest)?;
        write_new_bundle_file(&staging.join(SWEEP_MANIFEST_NAME), &manifest_bytes)?;
        crate::sync_directory(staging)
    })?;
    println!(
        "runtime=ordinary_qwen requests={} aggregate_prompt_tokens={} work_upper_bound={} requested_prefill={}\nartifact={}",
        bounds.request_count,
        bounds.aggregate_prompt_tokens,
        bounds.work_upper_bound,
        match args.prefill_execution {
            PrefillExecution::Auto => "auto",
            PrefillExecution::Serial => "serial",
        },
        output_path.display()
    );
    Ok(())
}

pub(super) fn execute_ordinary_arm(
    loaded: &qwen_llm::runtime::LoadedModel,
    tokenizer: &Tokenizer,
    execution: &ExecutionPlan,
    plan: &LensPlan,
    schedule: &CompiledEventSchedule,
    prompt_token_ids: &[i32],
    max_new_tokens: usize,
    sampler_config: RunSampler,
    stop_tokens: &HashSet<i32>,
    prefill: &mut PreparedOrdinaryPrefill,
) -> Result<RunResult> {
    let schedule = schedule.bind(plan)?;
    let mut event = schedule.new_event()?;
    let capacity = ensure_request_fits_context(
        prompt_token_ids.len(),
        max_new_tokens,
        loaded.context_length()?,
    )?;
    ensure_qwen_sequence_admitted(loaded, capacity)?;
    let mut sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let forward = loaded.forward();
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: sampler_config.temperature,
        top_k: sampler_config.top_k,
        top_p: sampler_config.top_p,
        min_p: sampler_config.min_p,
        seed: sampler_config.seed,
    })?;
    let mut operation_applications = Vec::new();
    let mut live_readouts = Vec::new();
    let mut logits = Vec::new();

    let mut packed_span_index = 0usize;
    let mut index = 0usize;
    while index < prompt_token_ids.len() {
        if let Some(span) = prefill
            .execution
            .packed_spans()
            .get(packed_span_index)
            .copied()
            && span.start == index
        {
            let scratch = prefill
                .scratch
                .as_mut()
                .context("packed Lens prefill schedule has no scratch")?;
            loaded
                .prefill_prompt_only(
                    &mut sequence,
                    scratch,
                    &prompt_token_ids[span.start..span.end],
                )
                .with_context(|| {
                    format!(
                        "execute packed passive Lens prefill span {}..{}",
                        span.start, span.end
                    )
                })?;
            index = span.end;
            packed_span_index += 1;
            continue;
        }
        let token = prompt_token_ids[index];
        let phase = Phase::Prefill(index);
        schedule.populate(phase, &mut event)?;
        logits = forward_event(
            execution,
            &schedule,
            &forward,
            token,
            u32::try_from(index).context("prefill position exceeds runtime addressing")?,
            &mut sequence,
            phase,
            &event,
            phase_needs_logits(phase, prompt_token_ids.len()),
            &mut operation_applications,
            &mut live_readouts,
        )?;
        index += 1;
    }
    ensure!(
        packed_span_index == prefill.execution.packed_spans().len(),
        "packed Lens prefill schedule was not fully consumed"
    );
    let mut generated_token_ids = Vec::new();
    let mut stop_reason = String::from("max_new_tokens");
    for generated_index in 0..max_new_tokens {
        let sampled = sampler.sample(&logits)?.token;
        generated_token_ids.push(sampled);
        if stop_tokens.contains(&sampled) {
            stop_reason = String::from("stop_token");
            break;
        }
        if generated_index + 1 == max_new_tokens {
            break;
        }
        let phase = Phase::Decode(generated_index);
        schedule.populate(phase, &mut event)?;
        let position = prompt_token_ids
            .len()
            .checked_add(generated_index)
            .context("decode position overflow")?;
        logits = forward_event(
            execution,
            &schedule,
            &forward,
            sampled,
            u32::try_from(position).context("decode position exceeds runtime addressing")?,
            &mut sequence,
            phase,
            &event,
            phase_needs_logits(phase, prompt_token_ids.len()),
            &mut operation_applications,
            &mut live_readouts,
        )?;
    }
    Ok(RunResult {
        linear_transports: execution
            .plan
            .lenses
            .iter()
            .filter_map(|definition| {
                let id = definition.id();
                let prepared = execution.lenses.get(id)?;
                if let LoadedLens::Native(native) = &prepared.lens {
                    native
                        .producer_metadata
                        .as_ref()
                        .map(|metadata| serde_json::json!({"lens_id": id, "artifact":metadata}))
                } else {
                    None
                }
            })
            .collect(),
        prompt_token_ids: prompt_token_ids.to_vec(),
        decoded_text: tokenizer.decode(&generated_token_ids),
        generated_token_ids,
        stop_reason,
        operation_applications,
        live_readouts,
        native_hyper_captures: Vec::new(),
    })
}

pub(crate) fn run_coefficient_sweep(args: CoefficientSweepArgs) -> Result<()> {
    validate_coefficient_sweep_args(&args)?;
    if args.requests_jsonl.is_some() {
        return run_coefficient_sweep_cohort(args);
    }
    run_single_coefficient_sweep(args)
}

pub(super) fn run_single_coefficient_sweep(args: CoefficientSweepArgs) -> Result<()> {
    let arm_args = args.arm_run_args();
    validate_run_args(&arm_args)?;
    let output_path = crate::resolve_output_path(&args.output)?;
    ensure_new_bundle_output(&output_path)?;

    let plan_path = std::fs::canonicalize(&args.plan)
        .with_context(|| format!("resolve plan {}", args.plan.display()))?;
    let source_plan = parse_plan_bytes(&crate::read_regular_file_bounded(
        &plan_path,
        MAX_PLAN_BYTES,
    )?)
    .with_context(|| format!("parse Lens plan {}", plan_path.display()))?;
    validate_plan(&source_plan)?;
    validate_ordinary_plan(&source_plan)?;
    validate_sweep_source_operation(&source_plan, &args.operation)?;
    for &coefficient in &args.coefficients {
        let effective =
            plan_with_operation_coefficient(&source_plan, &args.operation, coefficient)?;
        validate_ordinary_plan(&effective)?;
    }
    let plan_dir = plan_path.parent().unwrap_or_else(|| Path::new("."));
    let full_transports = open_full_transports(&source_plan, plan_dir)?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    ensure!(
        !crate::muse_lens_artifact::is_muse_architecture(gguf.architecture().as_deref()),
        "qwen-lens sweep supports ordinary Qwen only; Muse Glimmer is not supported"
    );
    let family = ModelFamily::detect(&gguf).context("model has no supported Qwen architecture")?;
    ensure!(
        matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "qwen-lens sweep supports ordinary Qwen only"
    );

    let tokenizer = Tokenizer::from_gguf(&gguf).context("load model tokenizer")?;
    let prepared_input =
        prepare_qwen_model_input(arm_args.input_spec(), family, &gguf, &tokenizer)?;
    validate_sweep_prompt(
        &prepared_input.token_ids,
        tokenizer.n_vocab(),
        args.max_new_tokens,
        gguf.declared_context_length()?,
    )?;

    crate::shutdown::checkpoint()?;
    let mut full_transports = bind_full_transports(
        full_transports,
        &source_plan,
        plan_dir,
        &gguf,
        args.identity_cache.as_deref(),
    )?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf(gguf, args.model.clone())
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_runtime(loaded.gguf(), loaded.arch().kind, loaded.arch().n_layer)?;
    let prompt_token_ids = &prepared_input.token_ids;
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < loaded.arch().vocab_size),
        "prompt contains a token outside the model vocabulary"
    );

    let source_bound_plan = bind_plan_positions(
        &source_plan,
        &prepared_input.rendering,
        prompt_token_ids.len(),
    )?;
    let execution = prepare_execution_plan(
        &source_bound_plan.resolved,
        plan_dir,
        &loaded,
        &mut full_transports,
    )?;
    validate_reachable_scopes(&execution.plan, prompt_token_ids.len(), args.max_new_tokens)?;
    let schedule = CompiledEventSchedule::compile(&execution.plan, execution.n_layer)?;
    let mut prefill = prepare_ordinary_prefill(
        &loaded,
        &schedule,
        &source_bound_plan.resolved,
        args.prefill_execution,
        RunExecutionScheduleBasis::SweepSourcePlan,
        prompt_token_ids.len(),
        args.max_new_tokens,
    )?;
    prefill.execution.validate_against_plan(
        "ordinary_qwen",
        &source_bound_plan.resolved,
        prompt_token_ids.len(),
    )?;
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()?
        .into_iter()
        .collect::<HashSet<_>>();
    let sampler = run_sampler(&arm_args);
    let built = stage_and_publish_bundle(&output_path, |staging| {
        let mut output_budget = BundleOutputBudget::new(0)?;
        build_sweep_bundle(
            staging,
            &args,
            &arm_args,
            &plan_path,
            &source_plan,
            &source_bound_plan,
            &prepared_input,
            &loaded,
            &tokenizer,
            &execution,
            &schedule,
            &stop_tokens,
            sampler,
            &mut prefill,
            Some(&mut output_budget),
        )
    })?;
    print_sweep_summary(&args, &output_path, &built.summaries);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_sweep_bundle(
    root: &Path,
    args: &CoefficientSweepArgs,
    arm_args: &LensRunArgs,
    plan_path: &Path,
    source_plan: &LensPlan,
    source_bound_plan: &BoundLensPlan,
    prepared_input: &PreparedLensInput,
    loaded: &qwen_llm::runtime::LoadedModel,
    tokenizer: &Tokenizer,
    execution: &ExecutionPlan,
    schedule: &CompiledEventSchedule,
    stop_tokens: &HashSet<i32>,
    sampler: RunSampler,
    prefill: &mut PreparedOrdinaryPrefill,
    mut output_budget: Option<&mut BundleOutputBudget>,
) -> Result<BuiltSweepBundle> {
    let arms_path = root.join("arms");
    create_bundle_directory(&arms_path)?;
    crate::sync_directory(root)?;

    let prompt_token_ids = &prepared_input.token_ids;
    let mut arms = Vec::with_capacity(args.coefficients.len());
    let mut summaries = Vec::with_capacity(args.coefficients.len());
    let mut serialized_byte_length = 0u64;
    for (index, &coefficient) in args.coefficients.iter().enumerate() {
        let effective_authored_plan =
            plan_with_operation_coefficient(source_plan, &args.operation, coefficient)?;
        let effective_bound_plan = bind_plan_positions(
            &effective_authored_plan,
            &prepared_input.rendering,
            prompt_token_ids.len(),
        )?;
        ensure!(
            effective_bound_plan.position_bindings == source_bound_plan.position_bindings,
            "coefficient sweep changed semantic position bindings"
        );
        let result = execute_ordinary_arm(
            loaded,
            tokenizer,
            execution,
            &effective_bound_plan.resolved,
            schedule,
            prompt_token_ids,
            args.max_new_tokens,
            sampler,
            stop_tokens,
            prefill,
        )?;
        let summary = SweepArmSummary {
            index,
            coefficient,
            decoded_text: result.decoded_text.clone(),
            stop_reason: result.stop_reason.clone(),
            operation_application_count: result.operation_applications.len(),
            live_readout_count: result.live_readouts.len(),
        };
        let artifact = build_run_output(
            &arm_args.model,
            run_sampler(arm_args),
            arm_args.max_new_tokens,
            "ordinary_qwen",
            plan_path,
            effective_bound_plan,
            prepared_input,
            result,
            prefill.execution.clone(),
            None,
        );
        let bytes = serialize_run_output(&artifact)?;
        serialized_byte_length = charge_sweep_bundle_bytes(
            serialized_byte_length,
            bytes.len(),
            output_budget.as_deref_mut(),
        )?;
        let relative = format!("arms/{index:06}/run.json");
        let arm_path = arms_path.join(format!("{index:06}"));
        create_bundle_directory(&arm_path)?;
        write_new_bundle_file(&arm_path.join("run.json"), &bytes)?;
        crate::sync_directory(&arm_path)?;
        arms.push(CoefficientSweepArm {
            index,
            coefficient,
            artifact: relative,
            byte_length: bytes.len() as u64,
            blake3: blake3::hash(&bytes).to_hex().to_string(),
        });
        summaries.push(summary);
    }
    crate::sync_directory(&arms_path)?;

    let manifest = CoefficientSweepManifest {
        schema: SWEEP_SCHEMA.into(),
        schema_version: SWEEP_SCHEMA_VERSION,
        producer: current_sweep_producer(),
        canonical_source_plan_path: plan_path.to_path_buf(),
        source_plan: Some(source_plan.clone()),
        source_plan_canonical_json_blake3: Some(
            source_bound_plan
                .authored_plan_canonical_json_blake3
                .clone(),
        ),
        operation_id: args.operation.clone(),
        coefficients: args.coefficients.clone(),
        arms,
    };
    let manifest_bytes = serialize_sweep_manifest(&manifest)?;
    serialized_byte_length = charge_sweep_bundle_bytes(
        serialized_byte_length,
        manifest_bytes.len(),
        output_budget.as_deref_mut(),
    )?;
    write_new_bundle_file(&root.join(SWEEP_MANIFEST_NAME), &manifest_bytes)?;
    crate::sync_directory(root)?;
    Ok(BuiltSweepBundle {
        summaries,
        serialized_byte_length,
        manifest_byte_length: manifest_bytes.len() as u64,
        manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
    })
}

pub(super) fn charge_sweep_bundle_bytes(
    current: u64,
    byte_length: usize,
    output_budget: Option<&mut BundleOutputBudget>,
) -> Result<u64> {
    let byte_length = u64::try_from(byte_length).context("serialized sweep byte length")?;
    let next = current
        .checked_add(byte_length)
        .context("serialized sweep bundle byte count overflow")?;
    if let Some(output_budget) = output_budget {
        output_budget
            .charge(usize::try_from(byte_length).context("serialized sweep byte length")?)?;
    }
    Ok(next)
}

pub(super) fn current_sweep_producer() -> SweepProducer {
    SweepProducer {
        build_commit: env!("QWEN_BUILD_COMMIT").into(),
        build_dirty: env!("QWEN_BUILD_DIRTY").into(),
        build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
    }
}

pub(super) fn run_coefficient_sweep_cohort(args: CoefficientSweepArgs) -> Result<()> {
    let requests = read_sweep_cohort_requests(
        args.requests_jsonl
            .as_deref()
            .context("--requests-jsonl is required in cohort mode")?,
        args.coefficients.len(),
        args.max_new_tokens,
    )?;
    checked_sweep_cohort_child_count(requests.records.len(), args.coefficients.len())?;
    let output_path = crate::resolve_output_path(&args.output)?;
    ensure_new_bundle_output(&output_path)?;

    let plan_path = std::fs::canonicalize(&args.plan)
        .with_context(|| format!("resolve plan {}", args.plan.display()))?;
    let source_plan = parse_plan_bytes(&crate::read_regular_file_bounded(
        &plan_path,
        MAX_PLAN_BYTES,
    )?)
    .with_context(|| format!("parse Lens plan {}", plan_path.display()))?;
    validate_plan(&source_plan)?;
    validate_ordinary_plan(&source_plan)?;
    validate_sweep_source_operation(&source_plan, &args.operation)?;
    for &coefficient in &args.coefficients {
        let effective =
            plan_with_operation_coefficient(&source_plan, &args.operation, coefficient)?;
        validate_ordinary_plan(&effective)?;
    }
    let source_plan_blake3 = canonical_plan_blake3(&source_plan)?;
    let plan_dir = plan_path.parent().unwrap_or_else(|| Path::new("."));
    let full_transports = open_full_transports(&source_plan, plan_dir)?;

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    ensure!(
        !crate::muse_lens_artifact::is_muse_architecture(gguf.architecture().as_deref()),
        "qwen-lens sweep supports ordinary Qwen only; Muse Glimmer is not supported"
    );
    let family = ModelFamily::detect(&gguf).context("model has no supported Qwen architecture")?;
    ensure!(
        matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "qwen-lens sweep supports ordinary Qwen only"
    );
    let model_context_tokens = gguf.declared_context_length()?;
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load model tokenizer")?;

    let mut preflight_requests = Vec::new();
    preflight_requests
        .try_reserve_exact(requests.records.len())
        .context("allocate sweep cohort preflight requests")?;
    let mut admitted_transition_upper_bound = 0u64;
    for (source_line, request) in requests.records {
        let captured = capture_sweep_cohort_messages(&request.messages)
            .with_context(|| format!("capture sweep cohort request {:?} messages", request.id))?;
        let mut arm_args = args.arm_run_args();
        arm_args.messages = Some(captured.path.clone());
        arm_args.message_mode = request.message_mode;
        validate_run_args(&arm_args)
            .with_context(|| format!("validate sweep cohort request {:?}", request.id))?;
        let source = captured.path.display().to_string();
        let prepared_input = prepare_qwen_model_messages_bytes(
            &captured.bytes,
            &source,
            request.message_mode,
            family,
            &gguf,
            &tokenizer,
        )
        .with_context(|| format!("render sweep cohort request {:?}", request.id))?;
        validate_sweep_prompt(
            &prepared_input.token_ids,
            tokenizer.n_vocab(),
            args.max_new_tokens,
            model_context_tokens,
        )
        .with_context(|| format!("preflight sweep cohort request {:?}", request.id))?;
        admitted_transition_upper_bound = admitted_transition_upper_bound
            .checked_add(sweep_cohort_request_transition_upper_bound(
                prepared_input.token_ids.len(),
                args.coefficients.len(),
                args.max_new_tokens,
            )?)
            .context("cohort transition bound overflow")?;
        ensure!(
            admitted_transition_upper_bound <= MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND,
            "cohort transition upper bound {admitted_transition_upper_bound} exceeds limit {MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND}"
        );
        let source_bound_plan = bind_plan_positions(
            &source_plan,
            &prepared_input.rendering,
            prepared_input.token_ids.len(),
        )
        .with_context(|| format!("bind sweep cohort request {:?}", request.id))?;
        validate_reachable_scopes(
            &source_bound_plan.resolved,
            prepared_input.token_ids.len(),
            args.max_new_tokens,
        )
        .with_context(|| format!("validate sweep cohort request {:?} scopes", request.id))?;
        preflight_requests.push(PreflightSweepCohortRequest {
            source_line,
            id: request.id,
            messages_path: captured.path,
            messages_blake3: captured.blake3,
            arm_args,
            prepared_input,
        });
    }
    let plan_bounds = plan_sweep_cohort_bounds(
        &preflight_requests
            .iter()
            .map(|request| request.prepared_input.token_ids.len())
            .collect::<Vec<_>>(),
        args.coefficients.len(),
        args.max_new_tokens,
    )?;
    ensure!(
        plan_bounds.transition_upper_bound == admitted_transition_upper_bound,
        "incremental sweep cohort transition admission drifted from the final plan"
    );
    let sampler = run_sampler(&args.arm_run_args());
    let manifest_basis = SweepCohortManifestBasis {
        producer: current_sweep_producer(),
        requests_jsonl_path: requests.canonical_path.clone(),
        requests_jsonl_blake3: requests.blake3.clone(),
        canonical_source_plan_path: plan_path.clone(),
        source_plan: source_plan.clone(),
        source_plan_canonical_json_blake3: source_plan_blake3.clone(),
        model_path: args.model.clone(),
        operation_id: args.operation.clone(),
        coefficients: args.coefficients.clone(),
        sampler,
        max_new_tokens: args.max_new_tokens,
        prefill_execution: args.prefill_execution,
        bounds: plan_bounds,
    };
    let manifest_reserve_bytes =
        ensure_sweep_cohort_manifest_capacity(&manifest_basis, &preflight_requests)?;

    crate::shutdown::checkpoint()?;
    let mut full_transports = bind_full_transports(
        full_transports,
        &source_plan,
        plan_dir,
        &gguf,
        args.identity_cache.as_deref(),
    )?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf(gguf, args.model.clone())
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_runtime(loaded.gguf(), loaded.arch().kind, loaded.arch().n_layer)?;

    for request in &preflight_requests {
        ensure!(
            request
                .prepared_input
                .token_ids
                .iter()
                .all(|&token| token >= 0 && (token as u32) < loaded.arch().vocab_size),
            "sweep cohort request {:?} contains a token outside the deployed vocabulary",
            request.id
        );
    }
    let first_request = preflight_requests
        .first()
        .context("sweep cohort has no preflight requests")?;
    let first_bound_plan = bind_plan_positions(
        &source_plan,
        &first_request.prepared_input.rendering,
        first_request.prepared_input.token_ids.len(),
    )
    .with_context(|| format!("bind sweep cohort request {:?}", first_request.id))?;
    let execution = prepare_execution_plan(
        &first_bound_plan.resolved,
        plan_dir,
        &loaded,
        &mut full_transports,
    )?;
    drop(first_bound_plan);
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()?
        .into_iter()
        .collect::<HashSet<_>>();
    let request_count = preflight_requests.len();
    stage_and_publish_bundle(&output_path, |staging| {
        let sweeps_root = staging.join("sweeps");
        create_bundle_directory(&sweeps_root)?;
        crate::sync_directory(staging)?;
        let mut children = Vec::new();
        children
            .try_reserve_exact(preflight_requests.len())
            .context("allocate sweep cohort child manifest entries")?;
        let mut output_budget = BundleOutputBudget::new(manifest_reserve_bytes)?;
        for (index, request) in preflight_requests.iter().enumerate() {
            let source_bound_plan = bind_plan_positions(
                &source_plan,
                &request.prepared_input.rendering,
                request.prepared_input.token_ids.len(),
            )
            .with_context(|| format!("bind sweep cohort request {:?}", request.id))?;
            let schedule =
                CompiledEventSchedule::compile(&source_bound_plan.resolved, loaded.arch().n_layer)?;
            let mut prefill = PreparedOrdinaryPrefill::serial(
                args.prefill_execution,
                RunExecutionScheduleBasis::SweepSourcePlan,
                RunSerialReason::CohortSerialPolicy,
            );
            prefill.execution.validate_against_plan(
                "ordinary_qwen",
                &source_bound_plan.resolved,
                request.prepared_input.token_ids.len(),
            )?;
            let child_path = sweeps_root.join(format!("{index:06}"));
            create_bundle_directory(&child_path)?;
            let child = build_sweep_bundle(
                &child_path,
                &args,
                &request.arm_args,
                &plan_path,
                &source_plan,
                &source_bound_plan,
                &request.prepared_input,
                &loaded,
                &tokenizer,
                &execution,
                &schedule,
                &stop_tokens,
                sampler,
                &mut prefill,
                Some(&mut output_budget),
            )?;
            children.push(SweepCohortChild {
                index,
                id: request.id.clone(),
                source_line: request.source_line,
                path: format!("sweeps/{index:06}"),
                prompt_token_count: request.prepared_input.token_ids.len(),
                messages_path: request.messages_path.clone(),
                messages_blake3: request.messages_blake3.clone(),
                serialized_byte_length: child.serialized_byte_length,
                manifest_byte_length: child.manifest_byte_length,
                manifest_blake3: child.manifest_blake3,
            });
        }
        crate::sync_directory(&sweeps_root)?;
        for child in &children {
            verify_sweep_cohort_child_manifest(staging, child)?;
        }
        let manifest = manifest_basis.build(output_budget.consumed, children);
        let manifest_bytes = serialize_sweep_cohort_manifest(&manifest)?;
        ensure!(
            manifest_bytes.len() <= manifest_reserve_bytes,
            "sweep cohort manifest exceeded its preflight reservation"
        );
        output_budget.release_reservation();
        output_budget.charge(manifest_bytes.len())?;
        write_new_bundle_file(&staging.join(SWEEP_MANIFEST_NAME), &manifest_bytes)?;
        crate::sync_directory(staging)?;
        Ok(())
    })?;
    println!(
        "runtime=ordinary_qwen model={} operation={} requests={} arms_per_request={} requested_prefill={} effective_prefill=serial\nartifact={}",
        args.model.display(),
        args.operation,
        request_count,
        args.coefficients.len(),
        match args.prefill_execution {
            PrefillExecution::Auto => "auto",
            PrefillExecution::Serial => "serial",
        },
        output_path.display()
    );
    Ok(())
}

pub(super) fn validate_sweep_prompt(
    token_ids: &[i32],
    vocab_size: u32,
    max_new_tokens: usize,
    model_context_tokens: usize,
) -> Result<()> {
    ensure!(
        !token_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < vocab_size),
        "prompt contains a token outside the model vocabulary"
    );
    ensure_request_fits_context(token_ids.len(), max_new_tokens, model_context_tokens)?;
    Ok(())
}

pub(super) fn checked_sweep_cohort_child_count(
    request_count: usize,
    arm_count: usize,
) -> Result<usize> {
    request_count
        .checked_mul(arm_count)
        .context("cohort prompt/arm child count overflow")
}

pub(super) fn sweep_cohort_request_transition_upper_bound(
    prompt_tokens: usize,
    arm_count: usize,
    max_new_tokens: usize,
) -> Result<u64> {
    let arm_count = u64::try_from(arm_count).context("cohort arm count")?;
    u64::try_from(prompt_tokens)
        .context("cohort prompt token count")?
        .checked_add(u64::try_from(max_new_tokens).context("cohort generation bound")?)
        .and_then(|transitions| transitions.checked_mul(arm_count))
        .context("cohort request transition bound overflow")
}

pub(super) fn sweep_cohort_request_capacity(
    arm_count: usize,
    max_new_tokens: usize,
) -> Result<usize> {
    ensure!(arm_count > 0, "sweep cohort arm count must be positive");
    ensure!(
        max_new_tokens > 0,
        "sweep cohort generation bound must be positive"
    );
    let minimum_transitions =
        sweep_cohort_request_transition_upper_bound(1, arm_count, max_new_tokens)?;
    let transition_capacity = MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND / minimum_transitions;
    let minimal_child = SweepCohortChild {
        index: 0,
        id: "x".into(),
        source_line: 1,
        path: "x".into(),
        prompt_token_count: 1,
        messages_path: PathBuf::from("x"),
        messages_blake3: "0".repeat(64),
        serialized_byte_length: 1,
        manifest_byte_length: 1,
        manifest_blake3: "0".repeat(64),
    };
    let minimum_child_bytes = serde_json::to_vec(&minimal_child)
        .context("price minimum sweep cohort child metadata")?
        .len()
        .checked_add(1)
        .context("minimum sweep cohort child size overflow")?;
    let manifest_capacity = MAX_PLAN_BYTES / minimum_child_bytes;
    let capacity = usize::try_from(transition_capacity)
        .unwrap_or(usize::MAX)
        .min(manifest_capacity);
    ensure!(
        capacity >= MIN_SWEEP_COHORT_REQUESTS,
        "sweep cohort cannot admit {MIN_SWEEP_COHORT_REQUESTS} requests within its transition and manifest budgets"
    );
    Ok(capacity)
}

pub(super) fn plan_sweep_cohort_bounds(
    prompt_token_counts: &[usize],
    arm_count: usize,
    max_new_tokens: usize,
) -> Result<SweepCohortPlanBounds> {
    let total_arm_count = checked_sweep_cohort_child_count(prompt_token_counts.len(), arm_count)?;
    let transition_upper_bound = prompt_token_counts
        .iter()
        .try_fold(0u64, |total, &prompt| {
            total
                .checked_add(sweep_cohort_request_transition_upper_bound(
                    prompt,
                    arm_count,
                    max_new_tokens,
                )?)
                .context("cohort transition bound overflow")
        })?;
    ensure!(
        transition_upper_bound <= MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND,
        "cohort transition upper bound {transition_upper_bound} exceeds limit {MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND}"
    );
    Ok(SweepCohortPlanBounds {
        request_count: prompt_token_counts.len(),
        total_arm_count,
        transition_upper_bound,
    })
}

pub(super) fn ensure_sweep_cohort_manifest_capacity(
    basis: &SweepCohortManifestBasis,
    requests: &[PreflightSweepCohortRequest],
) -> Result<usize> {
    let placeholder_digest = "0".repeat(64);
    let mut children = Vec::new();
    children
        .try_reserve_exact(requests.len())
        .context("allocate sweep cohort manifest preflight")?;
    for (index, request) in requests.iter().enumerate() {
        children.push(SweepCohortChild {
            index,
            id: request.id.clone(),
            source_line: request.source_line,
            path: format!("sweeps/{index:06}"),
            prompt_token_count: request.prepared_input.token_ids.len(),
            messages_path: request.messages_path.clone(),
            messages_blake3: request.messages_blake3.clone(),
            serialized_byte_length: u64::MAX,
            manifest_byte_length: u64::MAX,
            manifest_blake3: placeholder_digest.clone(),
        });
    }
    let manifest = basis.build(MAX_SWEEP_BUNDLE_BYTES, children);
    let bytes = serde_json::to_vec(&manifest).context("price sweep cohort manifest")?;
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "planned sweep cohort manifest requires at most {} bytes, exceeding limit {MAX_PLAN_BYTES}",
        bytes.len()
    );
    Ok(bytes.len())
}

pub(super) fn read_run_cohort_requests(path: &Path) -> Result<LoadedRunCohortRequests> {
    let canonical_path = std::fs::canonicalize(path)
        .with_context(|| format!("resolve run cohort request file {}", path.display()))?;
    let (file, length) = crate::open_regular_file(&canonical_path)?;
    let bytes = crate::read_opened_file_exact(file, &canonical_path, length)?;
    let text = String::from_utf8(bytes)
        .with_context(|| format!("read {} as UTF-8", canonical_path.display()))?;
    let request_root = canonical_path
        .parent()
        .context("run cohort request file has no parent")?;
    let mut records = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in text.split('\n').enumerate() {
        let source_line = line_index + 1;
        let line = line.strip_suffix('\r').unwrap_or(line);
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let mut request: LensCohortRequest = serde_json::from_str(trimmed)
            .with_context(|| format!("parse {} line {source_line}", canonical_path.display()))?;
        ensure!(
            !request.id.is_empty(),
            "run cohort request ID on line {source_line} must not be empty"
        );
        ensure!(
            ids.insert(request.id.clone()),
            "run cohort request ID {:?} is duplicated",
            request.id
        );
        request
            .resolve_paths(request_root)
            .with_context(|| format!("resolve run cohort request {:?}", request.id))?;
        validate_lens_input_spec(request.input_spec())
            .with_context(|| format!("validate run cohort request {:?}", request.id))?;
        records.push((source_line, request));
    }
    ensure!(!records.is_empty(), "run cohort request file is empty");
    Ok(LoadedRunCohortRequests {
        canonical_path,
        records,
    })
}

pub(super) fn serialize_run_cohort_manifest(manifest: &RunCohortManifest) -> Result<Vec<u8>> {
    validate_run_cohort_manifest(manifest)?;
    let bytes = serde_json::to_vec(manifest).context("serialize run cohort manifest")?;
    let decoded: RunCohortManifest =
        serde_json::from_slice(&bytes).context("reparse run cohort manifest")?;
    ensure!(
        &decoded == manifest,
        "run cohort manifest failed canonical JSON round trip"
    );
    Ok(bytes)
}

pub(super) fn validate_run_cohort_manifest(manifest: &RunCohortManifest) -> Result<()> {
    ensure!(
        manifest.schema == RUN_COHORT_SCHEMA
            && manifest.schema_version == RUN_COHORT_SCHEMA_VERSION,
        "unsupported run cohort manifest schema"
    );
    ensure!(
        manifest.request_count > 0 && manifest.request_count == manifest.runs.len(),
        "run cohort request count is inconsistent"
    );
    ensure!(
        manifest.execution_policy
            == "resident_model_shared_prepared_plan_serial_request_order_fresh_sequence_and_sampler_request_local_binding_no_cross_request_batching",
        "run cohort execution policy is unsupported"
    );
    ensure!(
        manifest.requests_jsonl_path.is_absolute() && manifest.canonical_plan_path.is_absolute(),
        "run cohort source paths must be absolute"
    );
    validate_plan(&manifest.source_plan)?;
    Sampler::new(SamplingConfig {
        temperature: manifest.sampler.temperature,
        top_k: manifest.sampler.top_k,
        top_p: manifest.sampler.top_p,
        min_p: manifest.sampler.min_p,
        seed: manifest.sampler.seed,
    })
    .context("validate run cohort manifest sampler")?;
    let mut ids = BTreeSet::new();
    let mut aggregate_prompt_tokens = 0usize;
    let mut work_upper_bound = 0u64;
    let mut forward_upper_bound = 0u64;
    let mut cumulative_child_bytes = 0u64;
    let mut max_request_forwards = 0usize;
    for (index, child) in manifest.runs.iter().enumerate() {
        ensure!(
            child.index == index
                && child.path == format!("run-{index:06}.json")
                && !child.id.is_empty()
                && ids.insert(child.id.clone())
                && matches!(
                    child.input_source.as_str(),
                    "prompt" | "token_ids" | "user" | "messages" | "open_responses"
                )
                && child.prompt_token_count > 0
                && child.artifact_byte_length > 0
                && child.artifact_byte_length <= MAX_RUN_ARTIFACT_BYTES as u64,
            "run cohort child {index} metadata is invalid"
        );
        aggregate_prompt_tokens = aggregate_prompt_tokens
            .checked_add(child.prompt_token_count)
            .context("run cohort aggregate prompt token count overflow")?;
        work_upper_bound = work_upper_bound
            .checked_add(
                u64::try_from(child.prompt_token_count)
                    .context("run cohort child prompt token count")?
                    .checked_add(
                        u64::try_from(manifest.max_new_tokens)
                            .context("run cohort generation bound")?,
                    )
                    .context("run cohort child work bound overflow")?,
            )
            .context("run cohort work bound overflow")?;
        let child_forwards =
            required_forward_count(child.prompt_token_count, manifest.max_new_tokens)?;
        forward_upper_bound = forward_upper_bound
            .checked_add(u64::try_from(child_forwards).context("run cohort child forwards")?)
            .context("run cohort forward bound overflow")?;
        max_request_forwards = max_request_forwards.max(child_forwards);
        cumulative_child_bytes = cumulative_child_bytes
            .checked_add(child.artifact_byte_length)
            .context("run cohort child byte count overflow")?;
    }
    ensure!(
        aggregate_prompt_tokens == manifest.aggregate_prompt_tokens
            && manifest.max_new_tokens > 0
            && work_upper_bound == manifest.work_upper_bound
            && forward_upper_bound == manifest.forward_upper_bound
            && max_request_forwards == manifest.max_request_forwards
            && manifest.sample_upper_bound
                == u64::try_from(manifest.request_count)
                    .context("run cohort request count")?
                    .checked_mul(
                        u64::try_from(manifest.max_new_tokens)
                            .context("run cohort generation bound")?
                    )
                    .context("run cohort sample bound overflow")?
            && cumulative_child_bytes == manifest.cumulative_child_bytes,
        "run cohort aggregate bounds are inconsistent"
    );
    Ok(())
}

pub(super) fn read_sweep_cohort_requests(
    path: &Path,
    arm_count: usize,
    max_new_tokens: usize,
) -> Result<LoadedSweepCohortRequests> {
    let request_capacity = sweep_cohort_request_capacity(arm_count, max_new_tokens)?;
    let canonical_path = std::fs::canonicalize(path)
        .with_context(|| format!("resolve sweep request file {}", path.display()))?;
    let bytes = crate::read_regular_file_bounded(&canonical_path, MAX_SWEEP_COHORT_FILE_BYTES)?;
    let text = std::str::from_utf8(&bytes)
        .with_context(|| format!("read {} as UTF-8", canonical_path.display()))?;
    let request_root = canonical_path
        .parent()
        .context("sweep request file has no parent")?;
    let mut records = Vec::new();
    let mut ids = BTreeSet::new();
    for (line_index, line) in text.split('\n').enumerate() {
        let source_line = line_index + 1;
        let line = line.strip_suffix('\r').unwrap_or(line);
        ensure!(
            line.len() <= MAX_SWEEP_COHORT_RECORD_BYTES,
            "{} line {} exceeds JSONL record limit of {} bytes",
            canonical_path.display(),
            source_line,
            MAX_SWEEP_COHORT_RECORD_BYTES
        );
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        ensure!(
            records.len() < request_capacity,
            "sweep request cohort exceeds the resource-derived capacity of {request_capacity} records for this arm and generation budget"
        );
        let mut request: SweepCohortRequestRecord = serde_json::from_str(trimmed)
            .with_context(|| format!("parse {} line {source_line}", canonical_path.display()))?;
        validate_cohort_id(&request.id, source_line)?;
        ensure!(
            ids.insert(request.id.clone()),
            "sweep request ID {:?} is duplicated",
            request.id
        );
        ensure!(
            request.messages != Path::new("-"),
            "sweep request messages on line {source_line} cannot read stdin"
        );
        if request.messages.is_relative() {
            request.messages = request_root.join(&request.messages);
        }
        records.push((source_line, request));
    }
    ensure!(
        records.len() >= MIN_SWEEP_COHORT_REQUESTS,
        "sweep request cohort requires at least {MIN_SWEEP_COHORT_REQUESTS} nonblank records"
    );
    Ok(LoadedSweepCohortRequests {
        canonical_path,
        blake3: blake3::hash(&bytes).to_hex().to_string(),
        records,
    })
}

pub(super) fn capture_sweep_cohort_messages(path: &Path) -> Result<CapturedSweepMessages> {
    capture_sweep_cohort_messages_with(path, |path| {
        crate::read_regular_file_bounded(path, MAX_SWEEP_COHORT_MESSAGES_BYTES)
    })
}

pub(super) fn capture_sweep_cohort_messages_with(
    path: &Path,
    read: impl FnOnce(&Path) -> Result<Vec<u8>>,
) -> Result<CapturedSweepMessages> {
    let bytes = read(path)?;
    Ok(CapturedSweepMessages {
        path: path.to_path_buf(),
        blake3: blake3::hash(&bytes).to_hex().to_string(),
        bytes,
    })
}

pub(super) fn validate_cohort_id(id: &str, source_line: usize) -> Result<()> {
    ensure!(
        !id.is_empty() && id.len() <= MAX_SWEEP_COHORT_ID_BYTES,
        "cohort request ID on line {source_line} must contain 1..={MAX_SWEEP_COHORT_ID_BYTES} bytes"
    );
    let mut bytes = id.bytes();
    ensure!(
        bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')),
        "cohort request ID on line {source_line} must start with an ASCII alphanumeric and contain only ASCII alphanumerics, '.', '_', or '-'"
    );
    Ok(())
}

pub(super) fn serialize_sweep_cohort_manifest(manifest: &SweepCohortManifest) -> Result<Vec<u8>> {
    validate_sweep_cohort_manifest(manifest)?;
    let bytes =
        serde_json::to_vec(manifest).context("serialize coefficient sweep cohort manifest")?;
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "coefficient sweep cohort manifest exceeds {MAX_PLAN_BYTES} bytes"
    );
    let decoded = parse_sweep_cohort_manifest_bytes(&bytes)
        .context("reparse coefficient sweep cohort manifest")?;
    ensure!(
        serde_json::to_vec(&decoded)? == bytes,
        "coefficient sweep cohort manifest failed canonical JSON round trip"
    );
    Ok(bytes)
}

pub(super) fn verify_sweep_cohort_child_manifest(
    root: &Path,
    child: &SweepCohortChild,
) -> Result<()> {
    let length = usize::try_from(child.manifest_byte_length)
        .context("sweep cohort child manifest length does not fit this platform")?;
    let path = root.join(&child.path).join(SWEEP_MANIFEST_NAME);
    let bytes = crate::read_regular_file_exact(&path, length)?;
    ensure!(
        blake3::hash(&bytes).to_hex().as_str() == child.manifest_blake3,
        "sweep cohort child {} manifest BLAKE3 does not match",
        child.index
    );
    let manifest = parse_sweep_manifest_bytes(&bytes)
        .with_context(|| format!("validate sweep cohort child {} manifest", child.index))?;
    let expected_serialized_byte_length =
        manifest
            .arms
            .iter()
            .try_fold(child.manifest_byte_length, |total, arm| {
                total
                    .checked_add(arm.byte_length)
                    .context("sweep cohort child serialized byte count overflow")
            })?;
    ensure!(
        expected_serialized_byte_length == child.serialized_byte_length,
        "sweep cohort child {} serialized byte length does not match its manifest",
        child.index
    );
    Ok(())
}

pub(super) fn parse_sweep_cohort_manifest_bytes(bytes: &[u8]) -> Result<SweepCohortManifest> {
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "coefficient sweep cohort manifest exceeds {MAX_PLAN_BYTES} bytes"
    );
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("parse coefficient sweep cohort manifest JSON")?;
    let manifest: SweepCohortManifest =
        serde_json::from_value(value).context("bind coefficient sweep cohort manifest")?;
    validate_sweep_cohort_manifest(&manifest)?;
    validate_sweep_cohort_bundle_size(&manifest, bytes.len())?;
    Ok(manifest)
}

pub(super) fn validate_sweep_cohort_bundle_size(
    manifest: &SweepCohortManifest,
    manifest_byte_length: usize,
) -> Result<()> {
    let total = manifest
        .cumulative_serialized_child_bytes
        .checked_add(u64::try_from(manifest_byte_length).context("sweep cohort manifest length")?)
        .context("coefficient sweep cohort bundle byte count overflow")?;
    ensure!(
        total <= MAX_SWEEP_BUNDLE_BYTES,
        "coefficient sweep cohort bundle bytes {total} exceed limit {MAX_SWEEP_BUNDLE_BYTES}"
    );
    Ok(())
}

pub(super) fn validate_sweep_cohort_manifest(manifest: &SweepCohortManifest) -> Result<()> {
    ensure!(
        manifest.schema == SWEEP_COHORT_SCHEMA
            && manifest.schema_version == SWEEP_COHORT_SCHEMA_VERSION,
        "unsupported coefficient sweep cohort manifest schema"
    );
    validate_sweep_producer(&manifest.producer)?;
    ensure!(
        manifest.requests_jsonl_path.is_absolute()
            && manifest.canonical_source_plan_path.is_absolute(),
        "coefficient sweep cohort source paths must be absolute"
    );
    ensure!(
        manifest.requests_jsonl_blake3.len() == 64 && is_lower_hex(&manifest.requests_jsonl_blake3),
        "coefficient sweep cohort JSONL digest is invalid"
    );
    validate_plan(&manifest.source_plan)?;
    validate_ordinary_plan(&manifest.source_plan)?;
    ensure!(
        canonical_plan_blake3(&manifest.source_plan)? == manifest.source_plan_canonical_json_blake3,
        "coefficient sweep cohort source-plan digest is invalid"
    );
    validate_sweep_source_operation(&manifest.source_plan, &manifest.operation_id)
        .context("validate coefficient sweep cohort source operation")?;
    ensure!(
        !manifest.coefficients.is_empty()
            && manifest.coefficients.len() <= MAX_SWEEP_ARMS
            && manifest.coefficients.iter().all(|value| value.is_finite()),
        "coefficient sweep cohort coefficients are invalid"
    );
    ensure!(
        manifest.max_new_tokens > 0
            && manifest.execution_policy
                == "serial_prompts_serial_arms_fresh_sequence_and_sampler_no_batched_generation",
        "coefficient sweep cohort execution policy is invalid"
    );
    SamplingConfig {
        temperature: manifest.sampler.temperature,
        top_k: manifest.sampler.top_k,
        top_p: manifest.sampler.top_p,
        min_p: manifest.sampler.min_p,
        seed: manifest.sampler.seed,
    }
    .validate()
    .context("coefficient sweep cohort sampler is invalid")?;
    ensure!(
        manifest.planned_request_count == manifest.sweeps.len()
            && manifest.planned_request_count >= MIN_SWEEP_COHORT_REQUESTS,
        "coefficient sweep cohort planned request count is invalid"
    );
    let planned = plan_sweep_cohort_bounds(
        &manifest
            .sweeps
            .iter()
            .map(|child| child.prompt_token_count)
            .collect::<Vec<_>>(),
        manifest.coefficients.len(),
        manifest.max_new_tokens,
    )?;
    ensure!(
        manifest.planned_total_arm_count == planned.total_arm_count
            && manifest.transition_upper_bound == planned.transition_upper_bound,
        "coefficient sweep cohort planned aggregate bounds are inconsistent"
    );
    let mut ids = BTreeSet::new();
    let mut cumulative_serialized_child_bytes = 0u64;
    for (index, child) in manifest.sweeps.iter().enumerate() {
        validate_cohort_id(&child.id, child.source_line)?;
        ensure!(
            ids.insert(child.id.clone()),
            "coefficient sweep cohort child IDs repeat"
        );
        ensure!(
            child.index == index
                && child.source_line > 0
                && child.path == format!("sweeps/{index:06}")
                && child.prompt_token_count > 0
                && child.messages_path.is_absolute()
                && child.messages_blake3.len() == 64
                && is_lower_hex(&child.messages_blake3)
                && child.serialized_byte_length >= child.manifest_byte_length
                && child.serialized_byte_length <= MAX_SWEEP_BUNDLE_BYTES
                && child.manifest_byte_length > 0
                && child.manifest_byte_length <= MAX_PLAN_BYTES as u64
                && child.manifest_blake3.len() == 64
                && is_lower_hex(&child.manifest_blake3),
            "coefficient sweep cohort child {index} metadata is invalid"
        );
        cumulative_serialized_child_bytes = cumulative_serialized_child_bytes
            .checked_add(child.serialized_byte_length)
            .context("coefficient sweep cohort serialized child byte count overflow")?;
    }
    ensure!(
        cumulative_serialized_child_bytes == manifest.cumulative_serialized_child_bytes
            && cumulative_serialized_child_bytes <= MAX_SWEEP_BUNDLE_BYTES,
        "coefficient sweep cohort cumulative serialized child bytes are invalid"
    );
    Ok(())
}

pub(super) fn validate_sweep_producer(producer: &SweepProducer) -> Result<()> {
    ensure!(
        matches!(producer.build_commit.len(), 40 | 64)
            && is_lower_hex(&producer.build_commit)
            && matches!(producer.build_dirty.as_str(), "0" | "1")
            && producer
                .build_source_state
                .strip_prefix("git-source-sha256-v2:")
                .is_some_and(|digest| digest.len() == 64 && is_lower_hex(digest)),
        "coefficient sweep producer metadata is invalid"
    );
    Ok(())
}

pub(super) fn validate_coefficient_sweep_args(args: &CoefficientSweepArgs) -> Result<()> {
    ensure!(
        !args.coefficients.is_empty() && args.coefficients.len() <= MAX_SWEEP_ARMS,
        "--coefficients requires 1..={MAX_SWEEP_ARMS} values"
    );
    ensure!(
        args.coefficients.iter().all(|value| value.is_finite()),
        "--coefficients values must be finite"
    );
    ensure!(!args.operation.is_empty(), "--operation must not be empty");
    if args.requests_jsonl.is_some() {
        ensure!(
            args.prompt.is_none()
                && args.token_ids.is_none()
                && args.user.is_none()
                && args.system.is_none()
                && args.messages.is_none()
                && args.open_responses.is_none()
                && args.message_mode.is_none()
                && !args.no_special_tokens,
            "--requests-jsonl conflicts with all single-prompt input flags"
        );
    }
    Ok(())
}

pub(super) fn serialize_sweep_manifest(manifest: &CoefficientSweepManifest) -> Result<Vec<u8>> {
    validate_sweep_manifest(manifest)?;
    let bytes = serde_json::to_vec(manifest).context("serialize coefficient sweep manifest")?;
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "coefficient sweep manifest exceeds {MAX_PLAN_BYTES} bytes"
    );
    let decoded =
        parse_sweep_manifest_bytes(&bytes).context("reparse coefficient sweep manifest JSON")?;
    ensure!(
        serde_json::to_vec(&decoded)? == bytes,
        "coefficient sweep manifest failed canonical JSON round trip"
    );
    Ok(bytes)
}

pub(crate) fn parse_sweep_manifest_bytes(bytes: &[u8]) -> Result<CoefficientSweepManifest> {
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "coefficient sweep manifest exceeds {MAX_PLAN_BYTES} bytes"
    );
    let value: serde_json::Value =
        serde_json::from_slice(bytes).context("parse coefficient sweep manifest JSON")?;
    let manifest: CoefficientSweepManifest =
        serde_json::from_value(value).context("bind coefficient sweep manifest")?;
    validate_sweep_manifest(&manifest)?;
    validate_sweep_bundle_size(&manifest, bytes.len())?;
    Ok(manifest)
}

pub(super) fn validate_sweep_bundle_size(
    manifest: &CoefficientSweepManifest,
    manifest_byte_length: usize,
) -> Result<()> {
    let total = manifest.arms.iter().try_fold(
        u64::try_from(manifest_byte_length).context("sweep manifest length")?,
        |total, arm| {
            total
                .checked_add(arm.byte_length)
                .context("coefficient sweep bundle byte count overflow")
        },
    )?;
    ensure!(
        total <= MAX_SWEEP_BUNDLE_BYTES,
        "coefficient sweep bundle bytes {total} exceed limit {MAX_SWEEP_BUNDLE_BYTES}"
    );
    Ok(())
}

pub(super) fn validate_sweep_manifest(manifest: &CoefficientSweepManifest) -> Result<()> {
    ensure!(
        manifest.schema == SWEEP_SCHEMA && matches!(manifest.schema_version, 1 | 2 | 3),
        "unsupported coefficient sweep manifest schema"
    );
    ensure!(
        matches!(manifest.producer.build_commit.len(), 40 | 64)
            && is_lower_hex(&manifest.producer.build_commit)
            && matches!(manifest.producer.build_dirty.as_str(), "0" | "1")
            && manifest
                .producer
                .build_source_state
                .strip_prefix("git-source-sha256-v2:")
                .is_some_and(|digest| digest.len() == 64 && is_lower_hex(digest)),
        "coefficient sweep producer metadata is invalid"
    );
    ensure!(
        manifest.canonical_source_plan_path.is_absolute(),
        "coefficient sweep source plan path must be absolute"
    );
    match manifest.schema_version {
        1 => ensure!(
            manifest.source_plan.is_none() && manifest.source_plan_canonical_json_blake3.is_none(),
            "coefficient sweep v1 must not contain embedded source-plan provenance"
        ),
        2 | 3 => {
            let source_plan = manifest
                .source_plan
                .as_ref()
                .context("coefficient sweep v2/v3 requires embedded source plan")?;
            validate_plan(source_plan)
                .context("validate embedded coefficient-sweep source plan")?;
            validate_ordinary_plan(source_plan)
                .context("validate embedded coefficient-sweep ordinary-Qwen source plan")?;
            let digest = manifest
                .source_plan_canonical_json_blake3
                .as_deref()
                .context("coefficient sweep v2/v3 requires source-plan digest")?;
            ensure!(
                canonical_plan_blake3(source_plan)? == digest
                    && digest.len() == 64
                    && is_lower_hex(digest),
                "coefficient sweep embedded source-plan digest is invalid"
            );
        }
        _ => unreachable!(),
    }
    ensure!(
        !manifest.operation_id.is_empty(),
        "coefficient sweep operation ID must not be empty"
    );
    if let Some(source_plan) = &manifest.source_plan {
        validate_sweep_source_operation(source_plan, &manifest.operation_id)
            .context("validate embedded coefficient sweep source operation")?;
    }
    ensure!(
        !manifest.coefficients.is_empty() && manifest.coefficients.len() <= MAX_SWEEP_ARMS,
        "coefficient sweep requires 1..={MAX_SWEEP_ARMS} coefficients"
    );
    ensure!(
        manifest.coefficients.iter().all(|value| value.is_finite()),
        "coefficient sweep contains a non-finite coefficient"
    );
    ensure!(
        manifest.arms.len() == manifest.coefficients.len(),
        "coefficient sweep arm count differs from coefficient count"
    );
    for (index, (arm, coefficient)) in manifest.arms.iter().zip(&manifest.coefficients).enumerate()
    {
        ensure!(
            arm.index == index,
            "coefficient sweep arm index is not ordered"
        );
        ensure!(
            arm.coefficient.to_bits() == coefficient.to_bits(),
            "coefficient sweep arm coefficient differs from its ordered coefficient"
        );
        ensure!(
            arm.artifact == format!("arms/{index:06}/run.json"),
            "coefficient sweep arm path is not canonical"
        );
        ensure!(
            arm.byte_length > 0 && arm.byte_length <= MAX_RUN_ARTIFACT_BYTES as u64,
            "coefficient sweep arm byte length is invalid"
        );
        ensure!(
            arm.blake3.len() == 64 && is_lower_hex(&arm.blake3),
            "coefficient sweep arm BLAKE3 is invalid"
        );
    }
    Ok(())
}

pub(super) fn print_sweep_summary(
    args: &CoefficientSweepArgs,
    output: &Path,
    summaries: &[SweepArmSummary],
) {
    println!(
        "runtime=ordinary_qwen model={} operation={} arms={}",
        args.model.display(),
        args.operation,
        summaries.len()
    );
    for summary in summaries {
        println!(
            "arm={} coefficient={} generated_text={} stop_reason={} operation_applications={} live_readouts={}",
            summary.index,
            summary.coefficient,
            serde_json::to_string(&summary.decoded_text).expect("string serialization cannot fail"),
            summary.stop_reason,
            summary.operation_application_count,
            summary.live_readout_count
        );
    }
    println!("artifact={}", output.display());
}

pub(super) fn validate_sweep_source_operation(plan: &LensPlan, operation_id: &str) -> Result<()> {
    let operation = plan
        .operations
        .iter()
        .find(|operation| operation.id == operation_id)
        .with_context(|| format!("Lens plan has no operation {operation_id:?}"))?;
    ensure!(
        operation.action.coefficient() != 0.0,
        "sweep source operation {operation_id:?} must be nonzero so enabled arms share a conservative prefill topology"
    );
    Ok(())
}

pub(crate) fn validate_sweep_effective_plan(plan: &LensPlan, operation_id: &str) -> Result<()> {
    ensure!(
        plan.operations
            .iter()
            .any(|operation| operation.id == operation_id),
        "Lens plan has no operation {operation_id:?}"
    );
    validate_plan(plan)?;
    validate_ordinary_plan(plan)
}
