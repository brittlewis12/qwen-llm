//! Resident `--requests-jsonl` request loop and per-request generation.

use super::*;

/// One batch row. Exactly one input form: raw `prompt` / `prompt_file`
/// (tokenized as given), or templated `user` (+ optional `system`) rendered
/// through the model's pinned released template — the same renderer as
/// `qwen run --user`. The rendering controls `no_thinking` and
/// `reasoning_effort` apply to `user` rows only.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JsonlRequest {
    pub(crate) id: Option<String>,
    pub(crate) prompt: Option<String>,
    pub(crate) prompt_file: Option<PathBuf>,
    pub(crate) user: Option<String>,
    pub(crate) system: Option<String>,
    pub(crate) no_thinking: Option<bool>,
    pub(crate) reasoning_effort: Option<cli::RunReasoningEffort>,
    pub(crate) tokens: Option<usize>,
    pub(crate) cache_prefix_tokens: Option<usize>,
    pub(crate) sampling: Option<JsonlSampling>,
}

/// How a row's prompt was produced; echoed on its output row and carried
/// into stats with the same `raw` / `messages` vocabulary as the v1 record.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct JsonlInputLabel {
    pub(crate) kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) template: Option<&'static str>,
}

impl JsonlInputLabel {
    pub(crate) const RAW: Self = Self {
        kind: "raw",
        template: None,
    };
}

/// Resolve a row's input form. Returns the prompt text, whether the
/// tokenizer should add its own specials (never for rendered templates), and
/// the label. Rejects contradictory shapes explicitly: `deny_unknown_fields`
/// only catches unknown keys, not a raw row carrying rendering controls.
pub(crate) fn resolve_jsonl_request_input(
    request: &JsonlRequest,
    line: usize,
    protocol: Option<&crate::QwenUserPromptProtocol>,
) -> Result<(String, bool, JsonlInputLabel)> {
    let raw_forms = usize::from(request.prompt.is_some()) + usize::from(request.prompt_file.is_some());
    let templated = request.user.is_some();
    ensure!(
        raw_forms + usize::from(templated) == 1,
        "request line {line} must carry exactly one of prompt, prompt_file, or user"
    );
    if !templated {
        ensure!(
            request.system.is_none()
                && request.no_thinking.is_none()
                && request.reasoning_effort.is_none(),
            "request line {line}: system, no_thinking, and reasoning_effort apply to user rows only"
        );
        return Ok((request_prompt(request, line)?, true, JsonlInputLabel::RAW));
    }
    let protocol = protocol.with_context(|| {
        format!("request line {line}: templated user rows require an ordinary Qwen model")
    })?;
    ensure!(
        protocol.pinned(),
        "request line {line}: templated user rows require a model whose chat template is pinned (released Qwen3.5/3.6/3.8); this model's template is unrecognized, so submit a raw prompt instead"
    );
    let user = request.user.as_deref().expect("templated row has user");
    let prompt = protocol
        .render(
            user,
            request.system.as_deref(),
            request.no_thinking.unwrap_or(false),
            request.reasoning_effort,
        )
        .with_context(|| format!("render request line {line}"))?;
    Ok((
        prompt,
        false,
        JsonlInputLabel {
            kind: "messages",
            template: Some(protocol.label()),
        },
    ))
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JsonlSampling {
    #[serde(alias = "temp")]
    pub(crate) temperature: Option<f32>,
    pub(crate) top_k: Option<usize>,
    pub(crate) top_p: Option<f32>,
    pub(crate) min_p: Option<f32>,
    pub(crate) seed: Option<u64>,
}

#[derive(Debug)]
pub(crate) struct PreparedJsonlRequest {
    pub(crate) request: JsonlRequest,
    pub(crate) id: String,
    pub(crate) line: usize,
    pub(crate) input: JsonlInputLabel,
    pub(crate) prompt_ids: Vec<i32>,
    pub(crate) sampling: SamplingConfig,
    pub(crate) auto_cache_prefix_tokens: Option<usize>,
    pub(crate) auto_cache_future_hits: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum CachePrefixSource {
    None,
    Request,
    RequestDisabled,
    Cli,
    CliDisabled,
    Auto,
}

impl CachePrefixSource {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Request => "request",
            Self::RequestDisabled => "request_disabled",
            Self::Cli => "cli",
            Self::CliDisabled => "cli_disabled",
            Self::Auto => "auto",
        }
    }
}

pub(crate) fn run_requests_jsonl(
    model_path: &Path,
    requests_path: &Path,
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    args.prefill_chunk.validate()?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    // --request-stats-jsonl is DS4-single-turn-only today; reject on Qwen
    // batch rather than silently no-oping.
    ensure!(
        args.request_stats_jsonl.is_none(),
        "--request-stats-jsonl is not yet implemented on Qwen --requests-jsonl. \
         Use --request-stats for legacy per-request stats output."
    );
    cli_sampling_config(args)?;
    let policy = args.on_request_error;
    let accelerated_lane = args.batch_size.is_some()
        || args.concurrency.is_some()
        || args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    ensure!(
        policy == RequestErrorPolicy::Stop || !accelerated_lane,
        "--on-request-error continue is supported on the serial lane only; remove --batch-size, --concurrency, or --execution-mode auto"
    );

    let mut stdout = std::io::stdout().lock();
    // File requests are prepared from the header before the model loads:
    // every row is settled (parse, render, tokenize, sampling, capacity,
    // executor constraints) so a bad row under `stop` costs no load, and the
    // planners below only ever see rows that can run.
    let file = if requests_path == Path::new("-") {
        None
    } else {
        let tokenizer = Tokenizer::from_gguf(&gguf).context("load tokenizer")?;
        let file = prepare_jsonl_file(requests_path, &gguf, &tokenizer, args)?;
        if policy == RequestErrorPolicy::Stop
            && let Some(failure) = file.failures.first()
        {
            write_jsonl_row(&mut stdout, failure)?;
            bail!("{}", failure.message);
        }
        Some(file)
    };

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let requested_prefetch = args.model_prefetch.unwrap_or_default();
    let prefetch_policy = requested_prefetch.policy();
    let prefix_cache_max_bytes = if args.batch_size.is_some() || args.concurrency.is_some() {
        0
    } else {
        prefix_cache_max_bytes(args)?
    };
    let defaults = LoadedModelConfig::default();
    let loaded = runtime
        .load_open_model_with_config(
            gguf,
            LoadedModelConfig {
                prefix_cache_max_bytes,
                prefetch_policy,
                ..defaults
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    ensure!(
        loaded.prefetch_outcome().policy == prefetch_policy,
        "Qwen model prefetch policy changed during load"
    );
    eprintln!(
        "model_prefetch: requested={} effective={} bytes_returned={}",
        requested_prefetch.as_str(),
        prefetch_policy_label(loaded.prefetch_outcome().policy),
        loaded.prefetch_outcome().bytes_returned_total(),
    );
    let greedy_gpu_mode = configured_greedy_gpu_argmax_mode();
    let tokenizer = loaded.tokenizer().context("load tokenizer")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    let mut stats_file = args
        .request_stats
        .as_ref()
        .map(|path| open_append_file(path, "request stats"))
        .transpose()?;
    let mut n_requests = 0usize;
    let mut n_failed = 0usize;

    let model_family = match loaded.arch().kind {
        ArchKind::Dense => Some(ModelFamily::Qwen35),
        ArchKind::Moe => Some(ModelFamily::Qwen35Moe),
    };
    let auto_mode = args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    let (mut file_prepared, file_failures) = match file {
        Some(PreparedJsonlFile { prepared, failures }) => (Some(prepared), failures),
        None => (None, Vec::new()),
    };
    let mut auto_prepared = if auto_mode { file_prepared.take() } else { None };
    let auto_selection = if auto_mode {
        let requests = auto_prepared.as_deref().unwrap_or(&[]);
        let all_requests_accelerable = !requests.is_empty()
            && requests.iter().all(|request| {
                request.sampling.temperature == 0.0
                    && request.request.cache_prefix_tokens.is_none()
                    && request.auto_cache_prefix_tokens.is_none()
                    && request.auto_cache_future_hits == 0
            });
        let (dense_summary, moe_summary) = match (model_family, requests.is_empty()) {
            (_, true) => (
                fixed_cohort_jsonl::CohortPlanSummary::default(),
                fixed_cohort_jsonl::CohortPlanSummary::default(),
            ),
            (Some(ModelFamily::Qwen35), false) => (
                fixed_cohort_jsonl::plan_summary::<DENSE_BATCH8_WIDTH>(requests, args)?,
                fixed_cohort_jsonl::CohortPlanSummary::default(),
            ),
            (Some(ModelFamily::Qwen35Moe), false) => (
                fixed_cohort_jsonl::CohortPlanSummary::default(),
                fixed_cohort_jsonl::plan_summary::<MOE_BATCH16_WIDTH>(requests, args)?,
            ),
            (
                Some(ModelFamily::Qwen4Exp | ModelFamily::DeepSeek4 | ModelFamily::MuseGlimmer)
                | None,
                false,
            ) => (
                fixed_cohort_jsonl::CohortPlanSummary::default(),
                fixed_cohort_jsonl::CohortPlanSummary::default(),
            ),
        };
        let moe_plan = if model_family == Some(ModelFamily::Qwen35Moe) {
            loaded.inspect_moe_batch16_plan().ok()
        } else {
            None
        };
        let admission_requirements = if requests.is_empty() {
            None
        } else {
            Some(concurrent_jsonl::qwen_execution_admission_requirements(
                &loaded, requests, args,
            )?)
        };
        let admission = |width: usize,
                         max_capacity: usize,
                         executor_scratch_upper_bytes: u64|
         -> Result<bool> {
            let Some(requirements) = admission_requirements else {
                return Ok(false);
            };
            Ok(loaded
                .qwen_execution_memory_admission(
                    width,
                    max_capacity,
                    requirements.prefill_scratch_upper_bytes,
                    executor_scratch_upper_bytes,
                )?
                .admitted)
        };
        let concurrency2_memory_admitted = admission(
            2,
            admission_requirements
                .map(|requirements| requirements.max_capacity)
                .unwrap_or(0),
            1024 * 1024,
        )?;
        let dense_batch8_memory_admitted =
            if model_family == Some(ModelFamily::Qwen35) && loaded.inspect_dense_batch8().is_ok() {
                admission(
                    DENSE_BATCH8_WIDTH,
                    dense_summary.max_execution_capacity,
                    loaded
                        .dense_batch8_scratch_bytes()?
                        .saturating_add(1024 * 1024),
                )?
            } else {
                false
            };
        let moe_batch16_memory_admitted =
            if model_family == Some(ModelFamily::Qwen35Moe) && moe_plan.is_some() {
                admission(
                    MOE_BATCH16_WIDTH,
                    moe_summary.max_execution_capacity,
                    loaded
                        .moe_batch16_scratch_bytes()?
                        .saturating_add(1024 * 1024),
                )?
            } else {
                false
            };
        let selection = execution_selector::select_qwen(execution_selector::QwenSelectionInput {
            mode: args.execution_mode,
            family: model_family,
            arch: loaded.arch(),
            request_count: requests.len(),
            all_requests_accelerable,
            fixed_prefill_chunk: execution_selector::fixed_prefill_chunk(args.prefill_chunk),
            acceleration_blocker: if requests_path == Path::new("-") {
                Some("streaming_input")
            } else {
                execution_selector::qwen_acceleration_blocker(args, explicit, greedy_gpu_mode)
            },
            dense_full_cohorts: dense_summary.full_cohorts,
            dense_serial_remainders: dense_summary.serial_fallback_requests,
            moe_full_cohorts: moe_summary.full_cohorts,
            moe_serial_remainders: moe_summary.serial_fallback_requests,
            fixed_cohort_economics_rejected: match model_family {
                Some(ModelFamily::Qwen35) => dense_summary.economics_rejected_cohorts > 0,
                Some(ModelFamily::Qwen35Moe) => moe_summary.economics_rejected_cohorts > 0,
                Some(ModelFamily::Qwen4Exp | ModelFamily::DeepSeek4 | ModelFamily::MuseGlimmer)
                | None => false,
            },
            concurrency2_memory_admitted,
            dense_batch8_memory_admitted,
            moe_batch16_memory_admitted,
            moe_plan,
        });
        eprintln!(
            "execution_selection: {}",
            serde_json::to_string(&execution_selector::ExecutionSelectionRecord::new(
                model_family,
                (!requests.is_empty()).then_some(requests.len()),
                selection,
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.ragged_prompt_policy,
                    Some(ModelFamily::Qwen35Moe) => moe_summary.ragged_prompt_policy,
                    Some(
                        ModelFamily::Qwen4Exp | ModelFamily::DeepSeek4 | ModelFamily::MuseGlimmer,
                    )
                    | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.ragged_prompt_plan_decision,
                    Some(ModelFamily::Qwen35Moe) => moe_summary.ragged_prompt_plan_decision,
                    Some(
                        ModelFamily::Qwen4Exp | ModelFamily::DeepSeek4 | ModelFamily::MuseGlimmer,
                    )
                    | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.refill_policy,
                    Some(
                        ModelFamily::Qwen35Moe
                        | ModelFamily::Qwen4Exp
                        | ModelFamily::DeepSeek4
                        | ModelFamily::MuseGlimmer,
                    )
                    | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.planned_refill_arenas,
                    Some(
                        ModelFamily::Qwen35Moe
                        | ModelFamily::Qwen4Exp
                        | ModelFamily::DeepSeek4
                        | ModelFamily::MuseGlimmer,
                    )
                    | None => None,
                },
                match model_family {
                    Some(ModelFamily::Qwen35) => dense_summary.planned_refill_requests,
                    Some(
                        ModelFamily::Qwen35Moe
                        | ModelFamily::Qwen4Exp
                        | ModelFamily::DeepSeek4
                        | ModelFamily::MuseGlimmer,
                    )
                    | None => None,
                },
            ))
            .context("serialize execution selection")?
        );
        if selection.selected.accelerated() {
            let stats = loaded.set_prefix_cache_max_bytes(0);
            ensure!(
                stats.entries == 0 && stats.indexed_bytes == 0,
                "automatic execution selection found a populated prefix cache before request execution"
            );
        }
        Some(selection)
    } else {
        None
    };
    let effective_prefix_cache_max_bytes = loaded.prefix_cache_stats().max_indexed_bytes;

    eprintln!(
        "loaded {} in {:.1} ms; prefix_cache_max_mib={}",
        model_path.display(),
        load_ms,
        effective_prefix_cache_max_bytes / (1024 * 1024),
    );

    if args.concurrency.is_some() {
        let requests = file_prepared
            .as_deref()
            .expect("--concurrency requires a regular JSONL file");
        n_requests += concurrent_jsonl::run_prepared(
            &loaded,
            &tokenizer,
            requests,
            args,
            greedy_gpu_mode,
            &mut stdout,
        )?;
    } else if let Some(batch_size) = args.batch_size {
        let requests = file_prepared
            .as_deref()
            .expect("--batch-size requires a regular JSONL file");
        n_requests += fixed_cohort_jsonl::run_prepared_with_batch_size(
            &loaded,
            &tokenizer,
            requests,
            args,
            greedy_gpu_mode,
            &mut stdout,
            batch_size,
        )?;
    } else if let Some(selection) = auto_selection
        && selection.selected.accelerated()
    {
        let requests = auto_prepared
            .as_deref()
            .expect("accelerated automatic selection requires file lookahead");
        n_requests += match selection.selected {
            execution_selector::SelectedExecution::Concurrency2 => concurrent_jsonl::run_prepared(
                &loaded,
                &tokenizer,
                requests,
                args,
                greedy_gpu_mode,
                &mut stdout,
            )?,
            execution_selector::SelectedExecution::DenseBatch8 => fixed_cohort_jsonl::run_prepared(
                &loaded,
                &tokenizer,
                requests,
                args,
                greedy_gpu_mode,
                &mut stdout,
                DENSE_BATCH8_WIDTH,
            )?,
            execution_selector::SelectedExecution::MoeBatch16 => fixed_cohort_jsonl::run_prepared(
                &loaded,
                &tokenizer,
                requests,
                args,
                greedy_gpu_mode,
                &mut stdout,
                MOE_BATCH16_WIDTH,
            )?,
            execution_selector::SelectedExecution::Serial => {
                unreachable!("accelerated selection checked above")
            }
        };
    } else if requests_path == Path::new("-") {
        shutdown::checkpoint()?;
        let stdin = std::io::stdin();
        let reader = stdin.lock();
        if args.cache_prefix_auto_min_tokens > 0 {
            eprintln!(
                "prefix-cache auto admission needs request lookahead; disabled for stdin JSONL"
            );
        }
        let protocol = jsonl_user_prompt_protocol(&loaded)?;
        for (line_idx, line) in reader.lines().enumerate() {
            shutdown::checkpoint()?;
            let line_no = line_idx + 1;
            let line = line.with_context(|| format!("read requests line {line_no}"))?;
            let prepared_request =
                match prepare_jsonl_row(line_no, &line, &tokenizer, args, protocol.as_ref()) {
                    JsonlRowOutcome::Skipped => continue,
                    JsonlRowOutcome::Prepared(request) => request,
                    JsonlRowOutcome::Failed(failure) => {
                        write_jsonl_row(&mut stdout, &failure)?;
                        if policy == RequestErrorPolicy::Stop {
                            bail!("{}", failure.message);
                        }
                        n_failed += 1;
                        continue;
                    }
                };
            let (output, stats) = run_jsonl_request(
                &loaded,
                &tokenizer,
                &prepared_request,
                args,
                greedy_gpu_mode,
            )
            .with_context(|| format!("run request {}", prepared_request.id))?;

            serde_json::to_writer(&mut stdout, &output).context("write request output")?;
            writeln!(stdout)?;
            stdout.flush()?;

            if let Some(file) = stats_file.as_mut() {
                serde_json::to_writer(&mut *file, &stats).context("write request stats")?;
                writeln!(file)?;
                file.flush()?;
            }
            if let Some(path) = args.trace_request.as_ref() {
                append_request_trace(
                    path,
                    unix_epoch_ms()?,
                    stats.prompt_tokens,
                    stats.generated_tokens,
                )?;
            }
            n_requests += 1;
        }
    } else {
        let mut prepared = auto_prepared
            .take()
            .or(file_prepared)
            .expect("file requests were prepared before load");
        n_failed += file_failures.len();
        n_requests += run_prepared_jsonl_serial_with_failures(
            &loaded,
            &tokenizer,
            &mut prepared,
            &file_failures,
            args,
            greedy_gpu_mode,
            &mut stdout,
            &mut stats_file,
        )?;
    }

    ensure!(
        n_requests > 0 || n_failed > 0,
        "requests JSONL {} contained no requests",
        requests_path.display()
    );

    let stats = loaded.prefix_cache_stats();
    // JSONL aggregate stats line. No existing script pins this grammar
    // (the per-request stats in `stats: prompt_tokens=…` are the anchored
    // ones), but keep it flowing through `qwen_diag` for consistency and
    // so operators get a summary in interactive JSONL runs. Per-request
    // detail is available via `--request-stats <path>` when needed.
    tracing::info!(
        target: "qwen_diag",
        "stats: requests={} cache_entries={} cache_mib={:.1}/{:.1}",
        n_requests,
        stats.entries,
        stats.indexed_bytes as f64 / 1024.0 / 1024.0,
        stats.max_indexed_bytes as f64 / 1024.0 / 1024.0,
    );
    ensure!(
        n_failed == 0,
        "{n_failed} of {} requests failed to prepare; see the status=error rows on stdout",
        n_requests + n_failed
    );
    Ok(())
}

/// Serial execution that emits preparation failures in source order among
/// the successes: every failure whose line precedes the next runnable row is
/// written first, and the remainder after the last row.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_prepared_jsonl_serial_with_failures(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared: &mut [PreparedJsonlRequest],
    failures: &[JsonlRequestFailure],
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
    stats_file: &mut Option<std::fs::File>,
) -> Result<usize> {
    discover_auto_cache_prefixes(prepared, args.cache_prefix_auto_min_tokens);
    let mut pending = failures.iter().peekable();
    let mut completed = 0usize;
    for prepared_request in prepared.iter() {
        while let Some(failure) = pending.next_if(|failure| failure.line < prepared_request.line) {
            write_jsonl_row(stdout, failure)?;
        }
        completed += run_one_prepared_jsonl_request(
            loaded,
            tokenizer,
            prepared_request,
            args,
            greedy_gpu_mode,
            stdout,
            stats_file,
        )?;
    }
    for failure in pending {
        write_jsonl_row(stdout, failure)?;
    }
    Ok(completed)
}

#[allow(clippy::too_many_arguments)]
fn run_one_prepared_jsonl_request(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared_request: &PreparedJsonlRequest,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
    stdout: &mut impl Write,
    stats_file: &mut Option<std::fs::File>,
) -> Result<usize> {
    let (output, stats) =
        run_jsonl_request(loaded, tokenizer, prepared_request, args, greedy_gpu_mode)
            .with_context(|| format!("run request {}", prepared_request.id))?;
    write_jsonl_row(stdout, &output)?;
    if let Some(file) = stats_file.as_mut() {
        serde_json::to_writer(&mut *file, &stats).context("write request stats")?;
        writeln!(file)?;
        file.flush()?;
    }
    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(
            path,
            unix_epoch_ms()?,
            stats.prompt_tokens,
            stats.generated_tokens,
        )?;
    }
    Ok(1)
}

/// The row-rendering protocol for this loaded model, or `None` when the
/// family has no ordinary-Qwen chat rendering.
pub(crate) fn jsonl_user_prompt_protocol(
    loaded: &LoadedModel,
) -> Result<Option<crate::QwenUserPromptProtocol>> {
    jsonl_user_prompt_protocol_for_gguf(loaded.gguf())
}

/// Header-only form of `jsonl_user_prompt_protocol`, for preparing file
/// requests before the model loads.
pub(crate) fn jsonl_user_prompt_protocol_for_gguf(
    gguf: &GgufFile,
) -> Result<Option<crate::QwenUserPromptProtocol>> {
    match ModelFamily::detect(gguf) {
        Some(family @ (ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)) => {
            crate::QwenUserPromptProtocol::resolve(family, gguf)
        }
        _ => Ok(None),
    }
}

/// A row that could not be prepared. Emitted on stdout in place of a success
/// row so batch consumers see one typed outcome per attempted request.
/// Preparation failures are the recoverable class (parse, shape, render,
/// tokenize, sampling policy, capacity, executor constraints); execution,
/// admission, and I/O failures remain fatal to the batch.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct JsonlRequestFailure {
    pub(crate) id: String,
    pub(crate) line: usize,
    pub(crate) status: &'static str,
    pub(crate) code: &'static str,
    pub(crate) message: String,
}

impl JsonlRequestFailure {
    fn new(id: String, line: usize, code: &'static str, error: &anyhow::Error) -> Self {
        Self {
            id,
            line,
            status: "error",
            code,
            message: format!("{error:#}"),
        }
    }
}

/// What preparing one source line produced.
#[derive(Debug)]
pub(crate) enum JsonlRowOutcome {
    /// Blank or `#` comment line.
    Skipped,
    Prepared(PreparedJsonlRequest),
    Failed(JsonlRequestFailure),
}

/// `--on-request-error`: what a preparation failure does to the batch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub(crate) enum RequestErrorPolicy {
    /// Emit the failure row and stop; nothing after it is attempted.
    #[default]
    Stop,
    /// Emit the failure row and keep going; exit nonzero at the end.
    Continue,
}

/// Prepare one source line into a typed outcome. Only reading the line is
/// the caller's (fatal) concern; everything about the row itself is
/// reported as a `Failed` outcome with a stable code.
pub(crate) fn prepare_jsonl_row(
    line_no: usize,
    line: &str,
    tokenizer: &Tokenizer,
    args: &Args,
    protocol: Option<&crate::QwenUserPromptProtocol>,
) -> JsonlRowOutcome {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return JsonlRowOutcome::Skipped;
    }
    let fallback_id = format!("line-{line_no}");
    let mut request: JsonlRequest = match serde_json::from_str(trimmed) {
        Ok(request) => request,
        Err(error) => {
            return JsonlRowOutcome::Failed(JsonlRequestFailure::new(
                fallback_id,
                line_no,
                "parse",
                &anyhow::Error::new(error).context(format!("parse requests line {line_no}")),
            ));
        }
    };
    let id = request.id.clone().unwrap_or(fallback_id);
    let fail = |code, error: anyhow::Error| {
        JsonlRowOutcome::Failed(JsonlRequestFailure::new(id.clone(), line_no, code, &error))
    };
    let (prompt, tokenizer_specials, input) =
        match resolve_jsonl_request_input(&request, line_no, protocol) {
            Ok(resolved) => resolved,
            Err(error) => return fail("input", error),
        };
    let prompt_ids = match tokenizer.encode(&prompt, tokenizer_specials && !args.no_special_tokens)
    {
        Ok(ids) if ids.is_empty() => {
            return fail("tokenize", anyhow!("request {id} tokenized to zero tokens"));
        }
        Ok(ids) => ids,
        Err(error) => return fail("tokenize", anyhow::Error::new(error).context("tokenize prompt")),
    };
    let sampling = match request_sampling_config(&request, args) {
        Ok(sampling) => sampling,
        Err(error) => return fail("sampling", error),
    };
    if let Err(error) = validate_sampling_decode_policy(sampling, args.prompt_lookup) {
        return fail(
            "sampling",
            error.context(format!("validate decode policy for request {id}")),
        );
    }
    request.prompt = None;
    request.prompt_file = None;
    request.user = None;
    request.system = None;
    let prepared = PreparedJsonlRequest {
        request,
        id: id.clone(),
        line: line_no,
        input,
        prompt_ids,
        sampling,
        auto_cache_prefix_tokens: None,
        auto_cache_future_hits: 0,
    };
    // Generation budget and context capacity are row facts; settle them here
    // so planners and executors only ever see rows that can run.
    if let Err(error) = jsonl_generation_capacity(&prepared, args) {
        return fail("capacity", error);
    }
    // Forced accelerated executors admit only greedy rows without request
    // cache controls; report that as a row failure rather than a batch abort.
    if args.batch_size.is_some() || args.concurrency.is_some() {
        let executor = if args.batch_size.is_some() {
            "--batch-size"
        } else {
            "--concurrency"
        };
        if prepared.request.cache_prefix_tokens.is_some() {
            return fail(
                "executor_constraint",
                anyhow!("request {id} sets cache_prefix_tokens, which is unsupported with {executor}"),
            );
        }
        if prepared.sampling.temperature != 0.0 {
            return fail(
                "executor_constraint",
                anyhow!(
                    "request {id} uses temperature {}; {executor} currently requires greedy decoding",
                    prepared.sampling.temperature
                ),
            );
        }
    }
    JsonlRowOutcome::Prepared(prepared)
}

/// Prepared file requests split into the rows that can run and the rows
/// that cannot, both in source order. Header-only: runs before the model
/// loads.
pub(crate) struct PreparedJsonlFile {
    pub(crate) prepared: Vec<PreparedJsonlRequest>,
    pub(crate) failures: Vec<JsonlRequestFailure>,
}

pub(crate) fn prepare_jsonl_file(
    requests_path: &Path,
    gguf: &GgufFile,
    tokenizer: &Tokenizer,
    args: &Args,
) -> Result<PreparedJsonlFile> {
    let protocol = jsonl_user_prompt_protocol_for_gguf(gguf)?;
    let requests = std::fs::File::open(requests_path)
        .with_context(|| format!("open requests JSONL {}", requests_path.display()))?;
    let reader = std::io::BufReader::new(requests);
    let mut prepared = Vec::new();
    let mut failures = Vec::new();
    shutdown::checkpoint()?;
    for (line_idx, line) in reader.lines().enumerate() {
        shutdown::checkpoint()?;
        let line_no = line_idx + 1;
        let line = line.with_context(|| format!("read requests line {line_no}"))?;
        match prepare_jsonl_row(line_no, &line, tokenizer, args, protocol.as_ref()) {
            JsonlRowOutcome::Skipped => {}
            JsonlRowOutcome::Prepared(request) => prepared.push(request),
            JsonlRowOutcome::Failed(failure) => failures.push(failure),
        }
    }
    ensure!(
        !prepared.is_empty() || !failures.is_empty(),
        "requests JSONL {} contained no requests",
        requests_path.display()
    );
    Ok(PreparedJsonlFile { prepared, failures })
}

pub(crate) fn write_jsonl_row(stdout: &mut impl Write, row: &impl Serialize) -> Result<()> {
    serde_json::to_writer(&mut *stdout, row).context("write request output")?;
    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}

pub(crate) fn discover_auto_cache_prefixes(
    requests: &mut [PreparedJsonlRequest],
    min_tokens: usize,
) {
    if min_tokens == 0 || requests.len() < 2 {
        return;
    }

    for idx in 0..requests.len() {
        let mut future_lcps = Vec::new();
        for future in &requests[idx + 1..] {
            let lcp = longest_common_prefix_len(&requests[idx].prompt_ids, &future.prompt_ids);
            if lcp >= min_tokens {
                future_lcps.push(lcp);
            }
        }
        if future_lcps.is_empty() {
            continue;
        }

        future_lcps.sort_unstable();
        let mut best_len = 0usize;
        let mut best_hits = 0usize;
        let mut best_score = 0usize;
        for (pos, &len) in future_lcps.iter().enumerate() {
            let hits = future_lcps.len() - pos;
            let score = len.saturating_mul(hits);
            if score > best_score || (score == best_score && len > best_len) {
                best_len = len;
                best_hits = hits;
                best_score = score;
            }
        }

        requests[idx].auto_cache_prefix_tokens = Some(best_len.min(requests[idx].prompt_ids.len()));
        requests[idx].auto_cache_future_hits = best_hits;
    }
}

pub(crate) fn longest_common_prefix_len(a: &[i32], b: &[i32]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

pub(crate) fn selected_cache_prefix(
    request: &JsonlRequest,
    args: &Args,
    auto_cache_prefix_tokens: Option<usize>,
    prompt_len: usize,
) -> (Option<usize>, CachePrefixSource) {
    if let Some(n) = request.cache_prefix_tokens {
        return selected_cache_prefix_from_value(n, prompt_len, CachePrefixSource::Request);
    }
    if let Some(n) = args.cache_prefix_tokens {
        return selected_cache_prefix_from_value(n, prompt_len, CachePrefixSource::Cli);
    }
    if let Some(n) = auto_cache_prefix_tokens {
        return selected_cache_prefix_from_value(n, prompt_len, CachePrefixSource::Auto);
    }
    (None, CachePrefixSource::None)
}

pub(crate) fn selected_cache_prefix_from_value(
    n: usize,
    prompt_len: usize,
    source: CachePrefixSource,
) -> (Option<usize>, CachePrefixSource) {
    if n == 0 {
        let disabled = match source {
            CachePrefixSource::Request => CachePrefixSource::RequestDisabled,
            CachePrefixSource::Cli => CachePrefixSource::CliDisabled,
            _ => CachePrefixSource::None,
        };
        return (None, disabled);
    }
    (Some(n.min(prompt_len)).filter(|&n| n > 0), source)
}

pub(crate) fn jsonl_generation_capacity(
    prepared: &PreparedJsonlRequest,
    args: &Args,
) -> Result<(usize, usize)> {
    let n_generate = prepared.request.tokens.unwrap_or(args.tokens);
    ensure!(
        n_generate > 0,
        "tokens must be >= 1 for request {}",
        prepared.id
    );
    let prompt_and_generation = prepared
        .prompt_ids
        .len()
        .checked_add(n_generate)
        .context("sequence capacity overflow")?;
    let min_capacity = prompt_and_generation
        .checked_add(16)
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_and_generation,
        "max context {} is smaller than prompt {} + generation {} for request {}",
        capacity,
        prepared.prompt_ids.len(),
        n_generate,
        prepared.id,
    );
    Ok((n_generate, capacity))
}

pub(crate) fn run_jsonl_request(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    prepared: &PreparedJsonlRequest,
    args: &Args,
    greedy_gpu_mode: GreedyGpuArgmaxMode,
) -> Result<(RequestOutput, RequestStatsRow)> {
    let request = &prepared.request;
    let id = &prepared.id;
    let arrival_ms = unix_epoch_ms_u64()?;
    let total_t0 = Instant::now();
    let prompt_ids = &prepared.prompt_ids;

    let (n_generate, capacity) = jsonl_generation_capacity(prepared, args)?;

    let (cache_prefix_tokens, cache_prefix_source) = selected_cache_prefix(
        request,
        args,
        prepared.auto_cache_prefix_tokens,
        prompt_ids.len(),
    );
    let auto_prefill_cache_safe =
        auto_prefill_cache_safe(loaded.prefix_cache_stats().entries, cache_prefix_tokens);
    let allocated = allocate_prefill_request_state(
        loaded,
        args.prefill_chunk,
        prompt_ids.len(),
        capacity,
        auto_prefill_cache_safe,
    )?;
    let chunk = allocated.chunk;
    let prefill_chunk_decision = allocated.decision;
    let mut scratch = allocated.scratch;
    let mut sequence = allocated.sequence;
    let forward = loaded.forward();

    let prompt_hash = token_hash_hex(prompt_ids);
    let cache_prefix_hash = cache_prefix_tokens.map(|n| token_hash_hex(&prompt_ids[..n]));

    let mut cache_hit = false;
    let mut matched_prefix_tokens = 0usize;
    let mut exact_cache_hit = false;
    let restore_ms;
    let mut prefix_inserted_bytes = 0u64;
    let mut prefix_insert_ms = 0.0;
    let mut prefill_ms = 0.0;

    let logits = {
        let restore_t0 = Instant::now();
        let hit = loaded
            .restore_cached_prefix(&mut sequence, prompt_ids)
            .context("restore prefix cache")?;
        restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        if let Some(hit) = hit {
            cache_hit = true;
            matched_prefix_tokens = hit.matched_prefix_len;
            exact_cache_hit = hit.exact;
            let restored_prefix_tokens = hit.restored_prefix_len;
            if restored_prefix_tokens == prompt_ids.len() {
                hit.exact_final_logits.with_context(|| {
                    format!("exact prefix-cache hit for request {id} did not store logits")
                })?
            } else if let Some(prefix_len) = cache_prefix_tokens
                && cache_prefix_needs_extension(prefix_len, restored_prefix_tokens)
            {
                let prefix_suffix = &prompt_ids[restored_prefix_tokens..prefix_len];
                let (prefix_logits, ms) = prefill_owned(
                    loaded,
                    &mut sequence,
                    &mut scratch,
                    prefix_suffix,
                    restored_prefix_tokens,
                )?;
                prefill_ms += ms;

                let insert_t0 = Instant::now();
                let insert = loaded
                    .cache_sequence_prefix(
                        &sequence,
                        prompt_ids[..prefix_len].to_vec(),
                        Some(prefix_logits.clone()),
                    )
                    .context("insert prefix cache snapshot")?;
                prefix_insert_ms = insert_t0.elapsed().as_secs_f64() * 1e3;
                prefix_inserted_bytes = insert.snapshot_bytes;

                if prefix_len == prompt_ids.len() {
                    prefix_logits
                } else {
                    let suffix = &prompt_ids[prefix_len..];
                    let (logits, ms) =
                        prefill_owned(loaded, &mut sequence, &mut scratch, suffix, prefix_len)?;
                    prefill_ms += ms;
                    logits
                }
            } else {
                let suffix = &prompt_ids[restored_prefix_tokens..];
                let (logits, ms) = prefill_owned(
                    loaded,
                    &mut sequence,
                    &mut scratch,
                    suffix,
                    restored_prefix_tokens,
                )?;
                prefill_ms += ms;
                logits
            }
        } else if let Some(prefix_len) = cache_prefix_tokens {
            let prefix = &prompt_ids[..prefix_len];
            let (prefix_logits, ms) =
                prefill_owned(loaded, &mut sequence, &mut scratch, prefix, 0)?;
            prefill_ms += ms;

            let insert_t0 = Instant::now();
            let insert = loaded
                .cache_sequence_prefix(&sequence, prefix.to_vec(), Some(prefix_logits.clone()))
                .context("insert prefix cache snapshot")?;
            prefix_insert_ms = insert_t0.elapsed().as_secs_f64() * 1e3;
            prefix_inserted_bytes = insert.snapshot_bytes;

            if prefix_len == prompt_ids.len() {
                prefix_logits
            } else {
                let suffix = &prompt_ids[prefix_len..];
                let (logits, ms) =
                    prefill_owned(loaded, &mut sequence, &mut scratch, suffix, prefix_len)?;
                prefill_ms += ms;
                logits
            }
        } else {
            let (logits, ms) = prefill_owned(loaded, &mut sequence, &mut scratch, prompt_ids, 0)?;
            prefill_ms += ms;
            logits
        }
    };
    let prefill_attention_query =
        (scratch.attn_matrix_tiled_layer_calls() > 0).then(|| PrefillAttentionQueryStats {
            outer_chunk_rows: chunk,
            query_rows: scratch.attn_matrix_query_rows(),
            tiled_layer_calls: scratch.attn_matrix_tiled_layer_calls(),
            query_tile_calls: scratch.attn_matrix_query_tile_calls(),
        });
    let prefill_scratch_overlay =
        scratch
            .prefill_scratch_overlay_stats()
            .map(|stats| PrefillScratchOverlayTimingStats {
                backing_bytes: stats.backing_bytes,
                attention_bytes: stats.attention_bytes,
                gdn_bytes: stats.gdn_bytes,
                saved_bytes: stats.saved_bytes,
            });

    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()
        .context("load producer-declared stop tokens")?;
    let sampling_config = prepared.sampling;
    let mut sampler = Sampler::new(sampling_config).context("initialize request sampler")?;
    let greedy_gpu_decision =
        resolve_greedy_gpu_decision(greedy_gpu_mode, sampling_config, args.prompt_lookup);
    let (generation, generated_text, prompt_lookup_stats) = if args.prompt_lookup {
        let (result, generated_text) = decode_prompt_lookup(
            loaded,
            &forward,
            tokenizer,
            sequence,
            prompt_ids,
            logits,
            prompt_ids.len(),
            n_generate,
            &stop_tokens,
        )?;
        sequence = result.sequence;
        (result.generation, generated_text, Some(result.stats))
    } else {
        let (generation, generated_text) = decode_serial(
            loaded,
            tokenizer,
            &mut sequence,
            logits,
            prompt_ids.len(),
            n_generate,
            &stop_tokens,
            &mut sampler,
            greedy_gpu_decision.enabled,
        )?;
        (generation, generated_text, None)
    };
    let stop_reason = generation.stop_reason;
    let generated = generation.tokens;
    let generated_token_sha256 = generated_token_sha256(&generated);
    let thinking_partition = generated_thinking_partition(tokenizer, &generated);
    drop(sequence);
    let decode_ms = generation.wall_ms;
    let decode_tps = if decode_ms > 0.0 {
        generated.len() as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    let stats_now = loaded.prefix_cache_stats();
    let finish_ms = unix_epoch_ms_u64()?;
    let total_ms = total_t0.elapsed().as_secs_f64() * 1e3;
    report_prefill_chunk_decision(prefill_chunk_decision.as_ref(), prompt_ids.len());
    let stats = RequestStatsRow {
        schema_version: request_schema_version(
            args.prefill_chunk,
            args.prompt_lookup,
            prefill_attention_query.is_some(),
            prefill_scratch_overlay.is_some(),
            sampling_config.temperature > 0.0,
            false,
            false,
        ),
        request_stats_contract: "qwen_jsonl_v2",
        id: id.to_string(),
        line: prepared.line,
        model: loaded.path().display().to_string(),
        build_commit: env!("QWEN_BUILD_COMMIT"),
        build_dirty: parse_build_dirty(env!("QWEN_BUILD_DIRTY")),
        build_source_state: env!("QWEN_BUILD_SOURCE_STATE"),
        model_prefetch_policy: prefetch_policy_label(loaded.prefetch_outcome().policy),
        model_prefetch_bytes_returned: loaded.prefetch_outcome().bytes_returned_total(),
        greedy_gpu_selection_reason: greedy_gpu_decision.reason,
        arrival_ms,
        finish_ms,
        prompt_tokens: prompt_ids.len(),
        prompt_hash,
        requested_tokens: n_generate,
        generated_tokens: generated.len(),
        generated_token_sha256: generated_token_sha256.clone(),
        decode_policy: jsonl_decode_policy_label(
            sampling_config,
            args.prompt_lookup,
            greedy_gpu_decision.enabled,
        ),
        stop_reason,
        terminal_token_target_transition_consumed: false,
        thinking_partition: thinking_partition.clone(),
        sampling: SamplingTelemetry::sampled(sampling_config, sampler.draws()),
        cache_prefix_tokens,
        cache_prefix_source: cache_prefix_source.as_str().to_string(),
        cache_prefix_hash,
        auto_cache_prefix_tokens: prepared.auto_cache_prefix_tokens,
        auto_cache_future_hits: prepared.auto_cache_future_hits,
        cache_hit,
        matched_prefix_tokens,
        matched_prefix_hash: if matched_prefix_tokens > 0 {
            Some(token_hash_hex(&prompt_ids[..matched_prefix_tokens]))
        } else {
            None
        },
        exact_cache_hit,
        prefill_chunk: match args.prefill_chunk {
            PrefillChunkArg::Fixed(requested) => requested,
            PrefillChunkArg::Auto => chunk,
        },
        prefill_chunk_effective: args.prefill_chunk.is_auto().then_some(chunk),
        prefill_chunk_decision,
        prefill_attention_query,
        prefill_scratch_overlay,
        max_context_tokens: capacity,
        no_special_tokens: args.no_special_tokens,
        restore_ms,
        prefix_inserted_bytes,
        prefix_insert_ms,
        prefill_ms,
        decode_ms,
        model_ttft_ms: restore_ms
            + prefix_insert_ms
            + prefill_ms
            + generation.first_token_ready_ms.unwrap_or(0.0),
        first_token_ms: generation.first_token_ready_ms.unwrap_or(0.0),
        first_token_callback_ms: generation.first_token_callback_ms.unwrap_or(0.0),
        first_decode_ms: generation.first_transition_ms.unwrap_or(0.0),
        decode_tps,
        decode_transitions: generation.transitions,
        transition_ms: generation.transition_ms,
        transition_tps,
        total_ms,
        cache_entries: stats_now.entries,
        cache_bytes: stats_now.indexed_bytes,
        cache_max_bytes: stats_now.max_indexed_bytes,
        prompt_lookup: prompt_lookup_stats,
    };
    let output = RequestOutput {
        id: id.to_string(),
        line: prepared.line,
        status: "ok",
        input: prepared.input,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated.len(),
        generated_token_sha256,
        generated_text,
        stop_reason,
        terminal_token_target_transition_consumed: false,
        thinking_partition,
    };
    Ok((output, stats))
}

pub(crate) fn request_prompt(request: &JsonlRequest, line: usize) -> Result<String> {
    match (request.prompt.as_ref(), request.prompt_file.as_ref()) {
        (Some(_), Some(_)) => bail!("request line {line} has both prompt and prompt_file"),
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("read prompt_file {} on line {line}", path.display())),
        (None, None) => bail!("request line {line} has neither prompt nor prompt_file"),
    }
}

pub(crate) fn generated_thinking_partition(
    tokenizer: &Tokenizer,
    generated: &[i32],
) -> Option<GeneratedThinkingPartition> {
    let pieces = generated
        .iter()
        .map(|&token| tokenizer.decode_piece(token))
        .collect::<Vec<_>>();
    thinking_partition_from_pieces(&pieces)
}

pub(crate) fn thinking_partition_from_pieces(
    pieces: &[String],
) -> Option<GeneratedThinkingPartition> {
    const DELIMITER: &str = "</think>";
    let mut decoded = String::new();
    let mut boundaries = Vec::with_capacity(pieces.len() + 1);
    boundaries.push(0usize);
    for piece in pieces {
        decoded.push_str(piece);
        boundaries.push(decoded.len());
    }

    let delimiter_start = decoded.find(DELIMITER)?;
    let delimiter_end = delimiter_start + DELIMITER.len();
    let start_exact = boundaries
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, &offset)| (offset == delimiter_start).then_some(index));
    let end_exact = boundaries
        .iter()
        .enumerate()
        .find_map(|(index, &offset)| (offset == delimiter_end).then_some(index));
    let delimiter_start_token_index = start_exact.unwrap_or_else(|| {
        boundaries
            .iter()
            .rposition(|&offset| offset < delimiter_start)
            .unwrap_or(0)
    });
    let delimiter_end_token_index_exclusive = end_exact.unwrap_or_else(|| {
        boundaries
            .iter()
            .position(|&offset| offset > delimiter_end)
            .unwrap_or(pieces.len())
    });
    let delimiter_token_aligned = start_exact.is_some() && end_exact.is_some();
    let (reasoning_tokens, delimiter_tokens, visible_tokens) = match (start_exact, end_exact) {
        (Some(start), Some(end)) if start <= end => (
            Some(start),
            Some(end - start),
            Some(pieces.len().saturating_sub(end)),
        ),
        _ => (None, None, None),
    };

    Some(GeneratedThinkingPartition {
        delimiter: DELIMITER,
        delimiter_start_token_index,
        delimiter_end_token_index_exclusive,
        delimiter_token_aligned,
        reasoning_tokens,
        delimiter_tokens,
        visible_tokens,
    })
}
