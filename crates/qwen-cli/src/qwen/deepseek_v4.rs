//! DeepSeek V4 single-turn, JSONL, snapshot, and info paths.

use super::*;

pub(crate) const DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES: u64 = 1024 * 1024 * 1024;

pub(crate) const DEEPSEEK_V4_SNAPSHOT_IDENTITY_CACHE_DIR: &str = ".qwen-dsv4-model-identity-v2";

pub(crate) const DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE: &str = "Apple M4 Max";

pub(crate) const DEEPSEEK_V4_PREFETCH_ENV: &str = "QWEN_DSV4_PREFETCH";

pub(crate) const DEEPSEEK_V4_PREFETCH_AUTO_THRESHOLD: f64 = 0.98;

#[cfg(feature = "dsv4-diagnostics")]
pub(crate) const DEEPSEEK_V4_TEMPORAL_WINDOW_ENV: &str = "QWEN_DSV4_TEMPORAL_WINDOW";

#[cfg(feature = "dsv4-diagnostics")]
pub(crate) const DEEPSEEK_V4_TEMPORAL_JSON_ENV: &str = "QWEN_DSV4_TEMPORAL_JSON";

#[derive(Clone, Debug)]
pub(crate) struct DeepSeekV4MultigroupSelectorPlan {
    pub(crate) requested: DeepSeekV4MultigroupSelectorArg,
    pub(crate) device_name: String,
    pub(crate) device_qualified: bool,
    pub(crate) capacity: DeepSeekV4SessionCapacity,
    pub(crate) geometry: Option<DeepSeekV4MultigroupSelectorGeometry>,
}

#[derive(Debug, Serialize)]
pub(crate) struct DeepSeekV4MultigroupSelectorSessionRecord<'a> {
    pub(crate) schema_version: u32,
    pub(crate) kind: &'static str,
    pub(crate) scope: &'a str,
    pub(crate) requested: &'static str,
    pub(crate) sealed: bool,
    pub(crate) device_name: &'a str,
    pub(crate) device_qualified: bool,
    pub(crate) forward_limit: usize,
    pub(crate) physical_capacity_rows: usize,
    pub(crate) max_reachable_visible_rows: usize,
    pub(crate) frozen_min_visible_rows: usize,
    pub(crate) frozen_max_capacity_rows: usize,
    pub(crate) frozen_min_capacity_occupancy: &'static str,
    pub(crate) fallback: &'static str,
}

#[derive(Debug, Serialize)]
pub(crate) struct DeepSeekV4MultigroupSelectorCompletionRecord<'a> {
    pub(crate) schema_version: u32,
    pub(crate) kind: &'static str,
    pub(crate) scope: &'a str,
    pub(crate) requested: &'static str,
    pub(crate) sealed: bool,
    pub(crate) multigroup_invocations: u64,
    pub(crate) ineligible_singleton_radix4_invocations: u64,
}

impl DeepSeekV4MultigroupSelectorPlan {
    pub(crate) fn new(
        requested: DeepSeekV4MultigroupSelectorArg,
        device_name: impl Into<String>,
        capacity: DeepSeekV4SessionCapacity,
    ) -> Result<Self> {
        let device_name = device_name.into();
        let device_qualified = device_name == DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE;
        let geometry = match requested {
            DeepSeekV4MultigroupSelectorArg::Auto | DeepSeekV4MultigroupSelectorArg::Off => None,
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental => {
                ensure!(
                    device_qualified,
                    "--deepseek-v4-multigroup-selector=qualified-experimental requires {}, got {}",
                    DEEPSEEK_V4_MULTIGROUP_SELECTOR_QUALIFIED_DEVICE,
                    device_name,
                );
                Some(
                    capacity
                        .qualify_multigroup_selector_experiment()
                        .context("qualify DeepSeek V4 multi-group selector request geometry")?,
                )
            }
        };
        Ok(Self {
            requested,
            device_name,
            device_qualified,
            capacity,
            geometry,
        })
    }

    pub(crate) fn sealed(&self) -> bool {
        self.geometry.is_some()
    }

    pub(crate) fn session_record<'a>(
        &'a self,
        scope: &'a str,
    ) -> DeepSeekV4MultigroupSelectorSessionRecord<'a> {
        DeepSeekV4MultigroupSelectorSessionRecord {
            schema_version: 1,
            kind: "session_policy",
            scope,
            requested: self.requested.as_str(),
            sealed: self.sealed(),
            device_name: &self.device_name,
            device_qualified: self.device_qualified,
            forward_limit: self.capacity.forward_limit(),
            physical_capacity_rows: self.capacity.csa_physical_rows(),
            max_reachable_visible_rows: self.capacity.forward_limit() / 4,
            frozen_min_visible_rows: DEEPSEEK_V4_MULTIGROUP_SELECTOR_MIN_VISIBLE_ROWS,
            frozen_max_capacity_rows: DEEPSEEK_V4_MULTIGROUP_SELECTOR_MAX_CAPACITY_ROWS,
            frozen_min_capacity_occupancy: "3/4",
            fallback: "radix4_for_packed_and_ineligible_singleton",
        }
    }

    pub(crate) fn seal_session(&self, session: &mut DeepSeekV4Session, scope: &str) -> Result<()> {
        match self.requested {
            DeepSeekV4MultigroupSelectorArg::Auto => {}
            DeepSeekV4MultigroupSelectorArg::Off => session
                .disable_multigroup_selector()
                .context("disable the DeepSeek V4 multi-group selector")?,
            DeepSeekV4MultigroupSelectorArg::QualifiedExperimental => session
                .enable_multigroup_selector_experiment()
                .context("seal qualified experimental DeepSeek V4 multi-group selector")?,
        }
        let telemetry = session.multigroup_selector_telemetry();
        ensure!(
            telemetry.sealed() == self.sealed(),
            "DeepSeek V4 multi-group selector sealing did not match the request"
        );
        eprintln!(
            "deepseek_v4 selector: {}",
            serde_json::to_string(&self.session_record(scope))
                .context("serialize DeepSeek V4 selector session policy")?
        );
        Ok(())
    }

    pub(crate) fn completion_record<'a>(
        &'a self,
        scope: &'a str,
        telemetry: DeepSeekV4MultigroupSelectorTelemetry,
    ) -> Result<DeepSeekV4MultigroupSelectorCompletionRecord<'a>> {
        self.completion_record_from_values(
            scope,
            telemetry.sealed(),
            telemetry.multigroup_invocations(),
            telemetry.ineligible_radix4_invocations(),
        )
    }

    pub(crate) fn completion_record_from_values<'a>(
        &'a self,
        scope: &'a str,
        sealed: bool,
        multigroup_invocations: u64,
        ineligible_radix4_invocations: u64,
    ) -> Result<DeepSeekV4MultigroupSelectorCompletionRecord<'a>> {
        ensure!(
            sealed == self.sealed(),
            "DeepSeek V4 multi-group selector completion changed sealed policy"
        );
        if self.requested == DeepSeekV4MultigroupSelectorArg::Off {
            ensure!(
                multigroup_invocations == 0 && ineligible_radix4_invocations == 0,
                "disabled DeepSeek V4 multi-group selector recorded invocations"
            );
        }
        Ok(DeepSeekV4MultigroupSelectorCompletionRecord {
            schema_version: 1,
            kind: "session_completion",
            scope,
            requested: self.requested.as_str(),
            sealed,
            multigroup_invocations,
            ineligible_singleton_radix4_invocations: ineligible_radix4_invocations,
        })
    }

    pub(crate) fn emit_completion(
        &self,
        scope: &str,
        telemetry: DeepSeekV4MultigroupSelectorTelemetry,
    ) -> Result<()> {
        eprintln!(
            "deepseek_v4 selector: {}",
            serde_json::to_string(&self.completion_record(scope, telemetry)?)
                .context("serialize DeepSeek V4 selector completion")?
        );
        Ok(())
    }
}

/// CLI values for `--reasoning`, mapping to the DeepSeek V4 release
/// three-tier effort contract (vLLM `77434861`): `none` is chat mode; `low`
/// opens `<think>` with no effort bytes (the release thinking default);
/// `high` additionally prepends the "Absolute maximum" instruction (labeled
/// max in the earlier two-tier encoders); and
/// `max` prepends the stronger "Beyond maximum" instruction.
#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub(crate) enum ReasoningLevelArg {
    None,
    Low,
    High,
    Max,
}

pub(crate) fn deepseek_v4_encode_options(args: &Args) -> Result<DeepSeekV4EncodeOptions> {
    let reasoning = match args.reasoning {
        None | Some(ReasoningLevelArg::None) => DeepSeekV4Reasoning::None,
        Some(ReasoningLevelArg::Low) => DeepSeekV4Reasoning::Low,
        Some(ReasoningLevelArg::High) => DeepSeekV4Reasoning::High,
        Some(ReasoningLevelArg::Max) => DeepSeekV4Reasoning::Max,
    };
    ensure!(
        !(args.preserve_reasoning && matches!(reasoning, DeepSeekV4Reasoning::None)),
        "--preserve-reasoning requires --reasoning low, high, or max"
    );
    Ok(DeepSeekV4EncodeOptions {
        reasoning,
        preserve_reasoning: args.preserve_reasoning,
    })
}

pub(crate) fn validate_deepseek_v4_multigroup_selector_scope(args: &Args) -> Result<()> {
    if args.deepseek_v4_multigroup_selector
        != DeepSeekV4MultigroupSelectorArg::QualifiedExperimental
    {
        return Ok(());
    }
    ensure!(
        args.prompt.is_some()
            || args.prompt_file.is_some()
            || args.messages.is_some()
            || args.requests_jsonl.is_some(),
        "--deepseek-v4-multigroup-selector requires a generation request"
    );
    Ok(())
}

pub(crate) fn validate_deepseek_v4_multigroup_selector_family(
    requested: DeepSeekV4MultigroupSelectorArg,
    model_family: Option<ModelFamily>,
) -> Result<()> {
    ensure!(
        requested != DeepSeekV4MultigroupSelectorArg::QualifiedExperimental
            || model_family == Some(ModelFamily::DeepSeek4),
        "--deepseek-v4-multigroup-selector requires a DeepSeek V4 model"
    );
    Ok(())
}

pub(crate) fn validate_deepseek_v4_reasoning_scope(args: &Args) -> Result<()> {
    ensure!(
        (args.reasoning.is_none() && !args.preserve_reasoning) || has_messages_input(args),
        "--reasoning and --preserve-reasoning require --messages"
    );
    Ok(())
}

/// Options unsupported for every DeepSeek V4 execution mode. Mode-specific
/// options (`--requests-jsonl`, `--max-context-tokens`) are validated by the
/// single-turn and requests-mode validators respectively.
/// Legacy research options no family outside ordinary Qwen implements.
pub(crate) fn shared_unsupported_options(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Vec<&'static str> {
    let mut unsupported = Vec::new();
    if args.prompt_lookup {
        unsupported.push("--prompt-lookup");
    }
    if explicit.prefill_chunk || args.prefill_chunk != PrefillChunkArg::Fixed(1024) {
        unsupported.push("--prefill-chunk");
    }
    if explicit.prefix_cache_max_mib || args.prefix_cache_max_mib != 16 * 1024 {
        unsupported.push("--prefix-cache-max-mib");
    }
    if args.cache_prefix_tokens.is_some() {
        unsupported.push("--cache-prefix-tokens");
    }
    if explicit.cache_prefix_auto_min_tokens || args.cache_prefix_auto_min_tokens != 1024 {
        unsupported.push("--cache-prefix-auto-min-tokens");
    }
    if args.request_stats.is_some() {
        unsupported.push("--request-stats");
    }
    if args.request_timings.is_some() {
        unsupported.push("--request-timings");
    }
    if args.model_prefetch.is_some() {
        unsupported.push("--model-prefetch");
    }
    if args.request_timing_warm_followup {
        unsupported.push("--request-timing-warm-followup");
    }
    if args.messages_no_generation_prompt {
        unsupported.push("--messages-no-generation-prompt");
    }
    unsupported
}

/// Options a request-shaped serial single-turn lane (Muse Glimmer,
/// Flash-Next) does not implement, on top of `shared_unsupported_options`.
/// `--drafter` is settled earlier by `drafter_policy` for every family.
pub(crate) fn serial_lane_unsupported_options(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Vec<&'static str> {
    let mut unsupported = shared_unsupported_options(args, explicit);
    if args.requests_jsonl.is_some() {
        unsupported.push("--requests-jsonl");
    }
    if args.batch_size.is_some() {
        unsupported.push("--batch-size");
    }
    if args.concurrency.is_some() {
        unsupported.push("--concurrency");
    }
    if args.execution_mode.is_some() {
        unsupported.push("--execution-mode");
    }
    if args.durable_prefix_cache.is_some() {
        unsupported.push("--durable-prefix-cache");
    }
    if explicit.durable_prefix_cache_max_mib {
        unsupported.push("--durable-prefix-cache-max-mib");
    }
    if explicit.durable_prefix_cache_max_entry_mib {
        unsupported.push("--durable-prefix-cache-max-entry-mib");
    }
    if explicit.durable_prefix_cache_min_tokens {
        unsupported.push("--durable-prefix-cache-min-tokens");
    }
    if args.sampling_attribution {
        unsupported.push("--sampling-attribution");
    }
    if args.sampled_structural {
        unsupported.push("--sampled-structural");
    }
    if args.trace_request.is_some() {
        unsupported.push("--trace-request");
    }
    if args.messages_preserve_thinking || args.messages_strip_thinking {
        unsupported.push("legacy message thinking controls");
    }
    unsupported
}

/// DeepSeek V4-only controls must not be silently accepted elsewhere.
pub(crate) fn ensure_no_deepseek_v4_only_options(
    args: &Args,
    explicit: ExplicitCliOptions,
    unsupported: &mut Vec<&'static str>,
) -> Result<()> {
    if args.reasoning.is_some() || args.preserve_reasoning {
        unsupported.push("DeepSeek V4 reasoning controls");
    }
    if args.deepseek_v4_snapshot.is_some() {
        unsupported.push("--deepseek-v4-snapshot");
    }
    ensure!(
        !explicit.deepseek_v4_multigroup_selector
            && args.deepseek_v4_multigroup_selector == DeepSeekV4MultigroupSelectorArg::Auto,
        "--deepseek-v4-multigroup-selector applies only to DeepSeek V4"
    );
    Ok(())
}

pub(crate) fn validate_deepseek_v4_generation_mode(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    let mut unsupported = shared_unsupported_options(args, explicit);
    if args.requests_jsonl.is_some() {
        unsupported.push("--requests-jsonl");
    }
    if args.max_context_tokens.is_some() {
        unsupported.push("--max-context-tokens");
    }
    if args.messages_preserve_thinking
        && matches!(args.reasoning, None | Some(ReasoningLevelArg::None))
    {
        // Preserved history reasoning is a release thinking-mode contract;
        // accepting the flag in chat mode would silently no-op.
        unsupported.push("--messages-preserve-thinking (requires --reasoning low, high, or max)");
    }
    if args.durable_prefix_cache.is_none() {
        if explicit.durable_prefix_cache_max_mib {
            unsupported.push("--durable-prefix-cache-max-mib");
        }
        if explicit.durable_prefix_cache_max_entry_mib {
            unsupported.push("--durable-prefix-cache-max-entry-mib");
        }
        if explicit.durable_prefix_cache_min_tokens {
            unsupported.push("--durable-prefix-cache-min-tokens");
        }
    }
    ensure!(
        unsupported.is_empty(),
        "DeepSeek V4 currently supports bounded raw or ordinary-message single-turn generation only; unsupported options: {}",
        unsupported.join(", ")
    );
    ensure!(
        has_single_turn_input(args),
        "DeepSeek V4 generation requires --prompt, --prompt-file, --messages, or `qwen run --user`"
    );
    Ok(())
}

pub(crate) fn validate_deepseek_v4_requests_mode(
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    let mut unsupported = shared_unsupported_options(args, explicit);
    if args.durable_prefix_cache.is_some() {
        unsupported.push("--durable-prefix-cache");
    }
    if explicit.durable_prefix_cache_max_mib || args.durable_prefix_cache_max_mib != 32 * 1024 {
        unsupported.push("--durable-prefix-cache-max-mib");
    }
    if explicit.durable_prefix_cache_max_entry_mib
        || args.durable_prefix_cache_max_entry_mib != 16 * 1024
    {
        unsupported.push("--durable-prefix-cache-max-entry-mib");
    }
    if explicit.durable_prefix_cache_min_tokens || args.durable_prefix_cache_min_tokens != 1024 {
        unsupported.push("--durable-prefix-cache-min-tokens");
    }
    if args.messages_preserve_thinking {
        unsupported.push("--messages-preserve-thinking");
    }
    if args.messages_strip_thinking {
        unsupported.push("--messages-strip-thinking");
    }
    if args.deepseek_v4_snapshot.is_some() {
        // Also excluded at the parser level; kept as a defensive invariant.
        unsupported.push("--deepseek-v4-snapshot");
    }
    // --request-stats-jsonl is DS4-single-turn-only today. Reject on DS4 batch
    // rather than silently no-oping (which would break the mandatory/fail-closed
    // policy the flag advertises).
    if args.request_stats_jsonl.is_some() {
        unsupported.push("--request-stats-jsonl");
    }
    ensure!(
        unsupported.is_empty(),
        "DeepSeek V4 --requests-jsonl supports raw prompt requests only; unsupported options: {}",
        unsupported.join(", ")
    );
    let stdin = deepseek_v4_requests_reads_stdin(args)?;
    if stdin {
        ensure!(
            args.max_context_tokens.is_some(),
            "DeepSeek V4 --requests-jsonl from stdin cannot derive a context budget by lookahead; supply --max-context-tokens as the shared logical token capacity"
        );
    } else {
        ensure!(
            args.max_context_tokens.is_none(),
            "DeepSeek V4 --requests-jsonl file mode derives the forward budget from the request set; remove --max-context-tokens"
        );
    }
    Ok(())
}

pub(crate) fn deepseek_v4_requests_reads_stdin(args: &Args) -> Result<bool> {
    let path = args
        .requests_jsonl
        .as_ref()
        .context("DeepSeek V4 requests mode requires --requests-jsonl")?;
    Ok(path.as_os_str() == "-")
}

pub(crate) fn deepseek_v4_forward_budget_for_context_limit(context_tokens: usize) -> Result<usize> {
    ensure!(
        (2..=DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY).contains(&context_tokens),
        "--max-context-tokens must be in 2..={} for DeepSeek V4 requests",
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    Ok(context_tokens - 1)
}

pub(crate) fn validate_deepseek_v4_request_context_limit(
    request_id: &str,
    prompt_tokens: usize,
    max_tokens: usize,
    context_limit: usize,
) -> Result<()> {
    let required_context_tokens = prompt_tokens
        .checked_add(max_tokens)
        .context("DeepSeek V4 logical context requirement overflow")?;
    ensure!(
        required_context_tokens <= context_limit,
        "request {request_id} requires {required_context_tokens} logical context tokens ({prompt_tokens} prompt + {max_tokens} generation), beyond --max-context-tokens {context_limit}",
    );
    Ok(())
}

pub(crate) fn parse_deepseek_v4_prefill_chunk_tokens(value: Option<&str>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS);
    };
    let chunk_tokens = value
        .parse::<usize>()
        .with_context(|| format!("QWEN_DSV4_PREFILL_CHUNK_TOKENS={value:?} is not an integer"))?;
    ensure!(
        (1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&chunk_tokens),
        "QWEN_DSV4_PREFILL_CHUNK_TOKENS must be in 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS}, got {chunk_tokens}"
    );
    Ok(chunk_tokens)
}

pub(crate) fn deepseek_v4_prefill_chunk_tokens() -> Result<usize> {
    let value = std::env::var("QWEN_DSV4_PREFILL_CHUNK_TOKENS").ok();
    parse_deepseek_v4_prefill_chunk_tokens(value.as_deref())
}

pub(crate) fn deepseek_v4_prefill_chunk_ranges(
    prompt_tokens: usize,
    chunk_tokens: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < prompt_tokens {
        let remaining = prompt_tokens - start;
        let len = if remaining >= chunk_tokens {
            chunk_tokens
        } else if chunk_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS && remaining >= 2_048 {
            2_048
        } else {
            remaining
        };
        ranges.push(start..start + len);
        start += len;
    }
    ranges
}

pub(crate) fn deepseek_v4_packed_chunk_count(prompt_tokens: usize, chunk_tokens: usize) -> usize {
    if prompt_tokens < 2 {
        0
    } else {
        deepseek_v4_prefill_chunk_ranges(prompt_tokens, chunk_tokens).len()
    }
}

pub(crate) fn deepseek_v4_snapshot_publish_prefix(prompt_tokens: usize) -> Result<usize> {
    ensure!(
        prompt_tokens >= 2,
        "--deepseek-v4-snapshot requires at least two prompt tokens so restored state has an uncached endpoint token"
    );
    Ok(prompt_tokens - 1)
}

pub(crate) fn deepseek_v4_snapshot_restored_prefix_len(
    snapshot_prefix: &[u32],
    prompt_tokens: &[u32],
) -> Result<usize> {
    ensure!(
        snapshot_prefix.len() < prompt_tokens.len(),
        "DeepSeek V4 snapshot prefix has {} tokens but the request has {}; at least one uncached endpoint token is required because causal snapshots omit observations",
        snapshot_prefix.len(),
        prompt_tokens.len(),
    );
    ensure!(
        prompt_tokens.starts_with(snapshot_prefix),
        "DeepSeek V4 snapshot token prefix does not match this request"
    );
    Ok(snapshot_prefix.len())
}

pub(crate) fn deepseek_v4_snapshot_parent(path: &Path) -> Result<&Path> {
    ensure!(
        path.file_name().is_some(),
        "--deepseek-v4-snapshot requires a file path"
    );
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let metadata = std::fs::symlink_metadata(parent)
        .with_context(|| format!("inspect DeepSeek V4 snapshot parent {}", parent.display()))?;
    ensure!(
        metadata.file_type().is_dir(),
        "DeepSeek V4 snapshot parent {} is not a directory",
        parent.display()
    );
    ensure!(
        metadata.uid() == current_effective_uid() && metadata.mode() & 0o022 == 0,
        "DeepSeek V4 snapshot parent {} must be owned by the current user and not group/world-writable",
        parent.display()
    );
    Ok(parent)
}

pub(crate) fn deepseek_v4_snapshot_identity_cache(
    parent: &Path,
) -> Result<CheckpointIdentityCache> {
    let root = parent.join(DEEPSEEK_V4_SNAPSHOT_IDENTITY_CACHE_DIR);
    match std::fs::DirBuilder::new().mode(0o700).create(&root) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(error) => {
            return Err(error).with_context(|| {
                format!(
                    "create private DeepSeek V4 identity cache {}",
                    root.display()
                )
            });
        }
    }
    let metadata = std::fs::symlink_metadata(&root)
        .with_context(|| format!("inspect DeepSeek V4 identity cache {}", root.display()))?;
    ensure!(
        metadata.file_type().is_dir()
            && metadata.uid() == current_effective_uid()
            && metadata.mode() & 0o077 == 0,
        "DeepSeek V4 identity cache {} must be a current-user 0700 directory",
        root.display()
    );
    Ok(CheckpointIdentityCache::new(root))
}

pub(crate) fn advance_deepseek_v4_prompt_prefix(
    session: &mut DeepSeekV4Session,
    ctx: &MetalContext,
    token_ids: &[u32],
    chunk_tokens: usize,
) -> Result<()> {
    ensure!(
        !token_ids.is_empty(),
        "DeepSeek V4 snapshot prefix is empty"
    );
    for (chunk_index, range) in deepseek_v4_prefill_chunk_ranges(token_ids.len(), chunk_tokens)
        .into_iter()
        .enumerate()
    {
        shutdown::checkpoint()?;
        let chunk = &token_ids[range];
        session.advance_tokens(ctx, chunk).with_context(|| {
            format!("advance DeepSeek V4 snapshot prefix chunk {chunk_index} without logits")
        })?;
        shutdown::checkpoint()?;
    }
    Ok(())
}

pub(crate) fn execute_deepseek_v4_prompt_suffix(
    session: &mut DeepSeekV4Session,
    ctx: &MetalContext,
    token_ids: &[u32],
    chunk_tokens: usize,
) -> Result<usize> {
    ensure!(
        !token_ids.is_empty(),
        "DeepSeek V4 prompt suffix requires an endpoint token"
    );
    let chunks = deepseek_v4_prefill_chunk_ranges(token_ids.len(), chunk_tokens);
    let chunk_count = chunks.len();
    for (chunk_index, range) in chunks.into_iter().enumerate() {
        shutdown::checkpoint()?;
        let chunk = &token_ids[range];
        if chunk_index + 1 == chunk_count {
            session
                .prefill_tokens(ctx, chunk)
                .with_context(|| format!("prefill final DeepSeek V4 prompt chunk {chunk_index}"))?;
        } else {
            session.advance_tokens(ctx, chunk).with_context(|| {
                format!("advance DeepSeek V4 prompt chunk {chunk_index} without logits")
            })?;
        }
        shutdown::checkpoint()?;
    }
    Ok(chunk_count)
}

pub(crate) fn copy_deepseek_v4_logits(
    session: &DeepSeekV4Session,
    vocab_size: u32,
    purpose: &str,
) -> Result<Vec<f32>> {
    let logits = session
        .copy_logits_f32()
        .with_context(|| format!("copy {purpose} DeepSeek V4 logits"))?;
    ensure!(
        logits.len() == vocab_size as usize,
        "{purpose} DeepSeek V4 logits length {} differs from vocabulary {vocab_size}",
        logits.len(),
    );
    Ok(logits)
}

#[derive(Default)]
pub(crate) struct DeepSeekV4CliStageProfileAggregate {
    pub(crate) stages: [f64; 10],
    pub(crate) boundary_ms: f64,
    pub(crate) command_gpu_ms: f64,
    pub(crate) forward_wall_ms: f64,
}

pub(crate) fn emit_deepseek_v4_whole_profile(
    positions: &[u32],
    wall_ms: &[f64],
    gpu_ms: &[f64],
    outside_gpu_ms: &[f64],
    encode_cpu_ms: &[f64],
    wait_residual_ms: &[f64],
) {
    eprintln!(
        "deepseek_v4 whole_profile: positions={}..{} samples={} wall_median_ms={:.3} gpu_median_ms={:.3} outside_gpu_median_ms={:.3} encode_cpu_median_ms={:.3} wait_residual_median_ms={:.3} wall_ms={wall_ms:?} gpu_ms={gpu_ms:?}",
        positions[0],
        positions[positions.len() - 1],
        positions.len(),
        median_f64(wall_ms),
        median_f64(gpu_ms),
        median_f64(outside_gpu_ms),
        median_f64(encode_cpu_ms),
        median_f64(wait_residual_ms),
    );
}

pub(crate) fn emit_deepseek_v4_stage_profile(
    profile: &DeepSeekV4StageProfile,
    group: usize,
    groups: usize,
    aggregate: &mut DeepSeekV4CliStageProfileAggregate,
) {
    let mut stages = [0.0f64; 10];
    let mut boundary_ms = 0.0f64;
    for layer in &profile.sampled_layers {
        boundary_ms += layer.encoder_boundary_ms_scaled;
        for stage in &layer.stages {
            let index = match stage.kind {
                DeepSeekV4StageKind::AttentionHyperConnection => 0,
                DeepSeekV4StageKind::AttentionPrepare => 1,
                DeepSeekV4StageKind::AttentionCore => 2,
                DeepSeekV4StageKind::AttentionOutput => 3,
                DeepSeekV4StageKind::HyperConnectionBridge => 4,
                DeepSeekV4StageKind::MoeRouter => 5,
                DeepSeekV4StageKind::MoeRoutedExperts => 6,
                DeepSeekV4StageKind::MoeSharedExpert => 7,
                DeepSeekV4StageKind::MoeCombine => 8,
                DeepSeekV4StageKind::LayerTail => 9,
            };
            stages[index] += stage.duration_ms_scaled;
        }
    }
    let command_gpu_ms = profile
        .layers
        .iter()
        .map(|layer| layer.command_gpu_ms)
        .sum::<f64>();
    for (total, sample) in aggregate.stages.iter_mut().zip(stages) {
        *total += sample;
    }
    aggregate.boundary_ms += boundary_ms;
    aggregate.command_gpu_ms += command_gpu_ms;
    aggregate.forward_wall_ms += profile.forward_wall_ms;
    eprintln!(
        concat!(
            "deepseek_v4 stage_profile_group: position={} group={}/{} schedule=instrumented_per_layer ",
            "forward_wall_ms={:.3} command_gpu_ms={:.3} ",
            "attention_hc_ms={:.3} attention_prepare_ms={:.3} attention_core_ms={:.3} ",
            "attention_output_ms={:.3} bridge_ms={:.3} moe_router_ms={:.3} ",
            "moe_routed_ms={:.3} moe_shared_ms={:.3} moe_combine_ms={:.3} ",
            "layer_tail_ms={:.3} encoder_boundary_ms={:.3} sampled_layers={}"
        ),
        profile.position,
        group,
        groups,
        profile.forward_wall_ms,
        command_gpu_ms,
        stages[0],
        stages[1],
        stages[2],
        stages[3],
        stages[4],
        stages[5],
        stages[6],
        stages[7],
        stages[8],
        stages[9],
        boundary_ms,
        profile.sampled_layers.len(),
    );
    if group + 1 == groups {
        eprintln!(
            concat!(
                "deepseek_v4 stage_profile: positions={}..{} groups={} schedule=rotating_instrumented_per_layer ",
                "mean_forward_wall_ms={:.3} mean_command_gpu_ms={:.3} ",
                "attention_hc_ms={:.3} attention_prepare_ms={:.3} attention_core_ms={:.3} ",
                "attention_output_ms={:.3} bridge_ms={:.3} moe_router_ms={:.3} ",
                "moe_routed_ms={:.3} moe_shared_ms={:.3} moe_combine_ms={:.3} ",
                "layer_tail_ms={:.3} encoder_boundary_ms={:.3}"
            ),
            profile.position + 1 - groups as u32,
            profile.position,
            groups,
            aggregate.forward_wall_ms / groups as f64,
            aggregate.command_gpu_ms / groups as f64,
            aggregate.stages[0],
            aggregate.stages[1],
            aggregate.stages[2],
            aggregate.stages[3],
            aggregate.stages[4],
            aggregate.stages[5],
            aggregate.stages[6],
            aggregate.stages[7],
            aggregate.stages[8],
            aggregate.stages[9],
            aggregate.boundary_ms,
        );
    }
}

pub(crate) fn run_deepseek_v4_single_turn(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<()> {
    validate_deepseek_v4_generation_mode(args, explicit)?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    // Preflight the sidecar in the same open mode as emission (read+append),
    // and force INVOCATION_ID init here so entropy is required only when
    // telemetry is requested.
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        preflight_request_stats_jsonl(path)?;
        LazyLock::force(&INVOCATION_ID);
    }
    let request_start = std::time::Instant::now();
    let prefill_chunk_tokens = deepseek_v4_prefill_chunk_tokens()?;
    let prefetch_mode = configured_deepseek_v4_prefetch_mode()?;
    let sampling = cli_sampling_config(args)?;
    let arrival_ms = unix_epoch_ms()?;
    let encode_options = deepseek_v4_encode_options(args)?;
    let (prompt, prompt_source, durable_completed_eligible) =
        if let Some(path) = args.messages.as_ref() {
            let mut encode_options = encode_options;
            let inline_thinking = if args.messages_strip_thinking {
                DeepSeekV4InlineThinking::Strip
            } else if args.messages_preserve_thinking {
                // Tier presence is enforced by validate_deepseek_v4_generation_mode;
                // the flag maps onto the release encoder's drop_thinking=False lane.
                encode_options.preserve_reasoning = true;
                DeepSeekV4InlineThinking::PromoteToReasoning
            } else {
                DeepSeekV4InlineThinking::Verbatim
            };
            // Preserved reasoning renders assistant turns byte-faithfully, so the
            // next turn's re-rendered prompt strictly extends this turn's
            // completed transcript; only then is a completed-turn checkpoint
            // reusable.
            let completed_eligible = encode_options.preserve_reasoning;
            (
                load_deepseek_v4_0731_messages_prompt(
                    path,
                    args.messages_max,
                    encode_options,
                    inline_thinking,
                )
                .context("render DeepSeek V4 0731 messages")?,
                PromptSource::Messages,
                completed_eligible,
            )
        } else {
            let (prompt, source, _) = prompt_text(args)?;
            (prompt, source, false)
        };
    let prompt_kind = match prompt_source {
        PromptSource::Inline | PromptSource::File => "raw",
        PromptSource::Messages => match encode_options.reasoning {
            DeepSeekV4Reasoning::None => "messages_0731_chat",
            DeepSeekV4Reasoning::Low | DeepSeekV4Reasoning::High | DeepSeekV4Reasoning::Max => {
                "messages_0731_thinking"
            }
        },
    };

    let tokenizer_t0 = Instant::now();
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load DeepSeek V4 tokenizer")?;
    let tokenizer_ms = tokenizer_t0.elapsed().as_secs_f64() * 1e3;
    let prompt_ids = tokenizer
        .encode(&prompt, false)
        .context("tokenize raw DeepSeek V4 prompt")?;
    let required_forwards = required_forwards(
        "DeepSeek V4",
        prompt_ids.len(),
        args.tokens,
        Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY),
    )?;
    deepseek_v4_debug_dump_prompt_ids("single_turn", &prompt_ids);
    let vocab_size = tokenizer.n_vocab();
    let prompt_token_ids = prompt_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| checked_token_id(token, vocab_size, &format!("prompt[{index}]")))
        .collect::<Result<Vec<_>>>()?;
    let durable_store = deepseek_v4_checkpoint_store(args, staged_integrity)?;
    let durable_max_record_bytes = if durable_store.is_some() {
        durable_prefix_cache_max_entry_bytes(args)?
    } else {
        0
    };
    let mut durable_admitted = durable_store.is_some()
        && args.durable_prefix_cache_min_tokens > 0
        && prompt_token_ids.len() >= args.durable_prefix_cache_min_tokens;
    if let Some(publish_prefix) =
        deepseek_v4_durable_capture_prefix_len(prompt_token_ids.len(), durable_admitted)?
    {
        let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf)
            .context("bind durable DeepSeek V4 snapshot geometry")?;
        let session_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            required_forwards,
            model.config.context_length,
        )
        .context("derive durable DeepSeek V4 snapshot session capacity")?;
        let durable_record_admitted = causal_snapshot_record_bytes(
            &model.config,
            session_capacity,
            u32::try_from(publish_prefix).context("DeepSeek V4 durable prefix exceeds u32")?,
            durable_max_record_bytes,
        );
        if let Err(error) = durable_record_admitted.as_ref() {
            durable_admitted = false;
            eprintln!(
                "warning: durable DeepSeek V4 prefix capture is not admissible; generation will continue without publication: {error}"
            );
        }
    }
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared DeepSeek V4 stop tokens")?;
    for &token in &stop_tokens {
        checked_token_id(token, vocab_size, "stop")?;
    }
    // Probe store occupancy before resolving the strong model identity so an
    // empty store with no planned capture skips identity work entirely.
    let durable_probe_t0 = Instant::now();
    let durable_has_blobs: Option<bool> = match durable_store.as_ref() {
        None => None,
        Some(store) => match store.has_managed_blobs() {
            Ok(has_blobs) => Some(has_blobs),
            Err(error) => {
                eprintln!(
                    "warning: durable DeepSeek V4 prefix inventory failed after {:.1} ms; cold-prefilling: {error}",
                    durable_probe_t0.elapsed().as_secs_f64() * 1e3,
                );
                None
            }
        },
    };
    let durable_probe_ms = durable_probe_t0.elapsed().as_secs_f64() * 1e3;
    let durable_identity_needed = durable_store.is_some()
        && prompt_token_ids.len() >= 2
        && (durable_has_blobs == Some(true) || durable_admitted);
    let (snapshot_model_content_id, snapshot_file_exists) = if args.deepseek_v4_snapshot.is_some()
        || durable_identity_needed
    {
        let explicit_snapshot_path = args.deepseek_v4_snapshot.as_ref();
        let snapshot_parent = explicit_snapshot_path
            .map(|path| deepseek_v4_snapshot_parent(path))
            .transpose()?;
        let exists = if let Some(path) = explicit_snapshot_path {
            match path.symlink_metadata() {
                Ok(_) => true,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => false,
                Err(error) => {
                    return Err(error).with_context(|| {
                        format!("inspect DeepSeek V4 snapshot path {}", path.display())
                    });
                }
            }
        } else {
            false
        };
        if explicit_snapshot_path.is_some() && !exists {
            let publish_prefix = deepseek_v4_snapshot_publish_prefix(prompt_token_ids.len())?;
            let model = DeepSeekV4Model::from_gguf_flash_0731(&gguf)
                .context("bind DeepSeek V4 snapshot geometry")?;
            let session_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
                required_forwards,
                model.config.context_length,
            )
            .context("derive DeepSeek V4 snapshot session capacity")?;
            causal_snapshot_record_bytes(
                &model.config,
                session_capacity,
                u32::try_from(publish_prefix).context("DeepSeek V4 snapshot prefix exceeds u32")?,
                DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES,
            )
            .context("preflight DeepSeek V4 causal snapshot record budget")?;
        }
        let identity_cache = if let Some(store) = durable_store.as_ref() {
            store.identity_cache()
        } else {
            deepseek_v4_snapshot_identity_cache(
                snapshot_parent.expect("explicit snapshot path resolved a parent"),
            )?
        };
        let identity_t0 = Instant::now();
        let report = checkpoint_content_identity(&gguf, &identity_cache)
            .context("derive strong ordered-shard DeepSeek V4 model identity")?;
        eprintln!(
            "deepseek_v4: checkpoint model identity cache={} hashed_bytes={} elapsed_ms={:.1}",
            identity_cache_outcome_label(report.outcome),
            report.bytes_hashed,
            identity_t0.elapsed().as_secs_f64() * 1e3,
        );
        (
            Some(DeepSeekV4ModelContentId::new(report.content_id)),
            explicit_snapshot_path.is_some() && exists,
        )
    } else {
        (None, false)
    };

    eprintln!(
        "deepseek_v4: loading {} for generation; prompt_kind={} prompt_tokens={} max_generated_tokens={} reserved_forwards={}/{}",
        model_path.display(),
        prompt_kind,
        prompt_ids.len(),
        args.tokens,
        required_forwards,
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("init Metal context for DeepSeek V4")?;
    let load_plan =
        DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, required_forwards)
            .context("plan strict DeepSeek V4 Metal residency and session")?;
    let session_capacity = load_plan.session_capacity();
    eprintln!(
        "deepseek_v4: session capacity forwards={} csa_physical_rows={} hca_physical_rows={}",
        session_capacity.forward_limit(),
        session_capacity.csa_physical_rows(),
        session_capacity.hca_physical_rows(),
    );
    let selector_plan = DeepSeekV4MultigroupSelectorPlan::new(
        args.deepseek_v4_multigroup_selector,
        ctx.device.name().to_string(),
        session_capacity,
    )?;
    let restored_snapshot = if snapshot_file_exists {
        let snapshot_path = args
            .deepseek_v4_snapshot
            .as_ref()
            .expect("snapshot existence requires a snapshot path");
        let model_content_id = snapshot_model_content_id
            .expect("snapshot path resolved a model-content identity before planning");
        let snapshot_t0 = Instant::now();
        let snapshot = load_causal_snapshot_file(
            snapshot_path,
            DeepSeekV4SnapshotCodecConstraints {
                config: load_plan.config(),
                session_capacity: load_plan.session_capacity(),
                expected_model_content_id: model_content_id,
                max_record_bytes: DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES,
            },
        )
        .with_context(|| {
            format!(
                "load DeepSeek V4 causal snapshot {}",
                snapshot_path.display()
            )
        })?;
        let restored_prefix =
            deepseek_v4_snapshot_restored_prefix_len(snapshot.prefix_tokens(), &prompt_token_ids)?;
        eprintln!(
            "deepseek_v4: snapshot validated before residency path={} prefix_tokens={} record_load_ms={:.1}",
            snapshot_path.display(),
            restored_prefix,
            snapshot_t0.elapsed().as_secs_f64() * 1e3,
        );
        Some((snapshot, restored_prefix))
    } else {
        None
    };
    let memory_plan = load_plan.memory_plan().clone();
    let initial_memory_signals = ctx.memory_signals();
    eprintln!("deepseek_v4: memory plan; {memory_plan}");
    let admitted_load_plan = load_plan
        .admit(initial_memory_signals)
        .context("admit strict DeepSeek V4 Metal residency and session")?;
    let prefetch_outcome = apply_deepseek_v4_prefetch(&gguf, prefetch_mode)?;
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted_load_plan)
        .context("load admitted strict DeepSeek V4 Metal residency")?;
    shutdown::checkpoint()?;
    let (residency, memory_admission, after_residency_bytes) = realized.into_parts();
    let memory_signals = memory_admission.signals;
    eprintln!(
        "deepseek_v4: memory admission admitted={} reason={} recommended={} current={} process_remaining={:?} working_set_headroom={:?} required={:?}",
        memory_admission.admitted,
        memory_admission.reason.as_str(),
        memory_signals.recommended_max_bytes,
        memory_signals.current_allocated_bytes,
        memory_signals.process_limit_remaining_bytes,
        memory_admission.working_set_headroom_bytes,
        memory_admission.required_bytes,
    );
    let before_residency_bytes = memory_signals.current_allocated_bytes;
    memory_plan
        .reconcile_residency(before_residency_bytes, after_residency_bytes)
        .context("reconcile DeepSeek V4 residency allocation")?;
    ensure!(
        residency.config().vocab_size == vocab_size,
        "DeepSeek V4 tokenizer vocabulary {} differs from resident model vocabulary {}",
        vocab_size,
        residency.config().vocab_size,
    );
    let residency_report = residency.report().clone();
    let mut session = match snapshot_model_content_id {
        Some(model_content_id) => {
            DeepSeekV4Session::new_with_model_content_id(&ctx, residency, model_content_id)
        }
        None => DeepSeekV4Session::new(&ctx, residency),
    }
    .context("create DeepSeek V4 session")?;
    selector_plan.seal_session(&mut session, "single_turn")?;
    let after_session_bytes = ctx.current_allocated_size();
    memory_plan
        .reconcile_session(
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes,
        )
        .context("reconcile DeepSeek V4 session allocation")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "deepseek_v4: resident on {} in {:.1} ms; {}",
        ctx.describe(),
        load_ms,
        residency_report,
    );

    let prefill_t0 = Instant::now();
    let mut durable_prepared: Option<DeepSeekV4PreparedCheckpoint> = None;
    let mut durable_capture_kind = "prompt";
    let mut durable_capture_ms = 0.0;
    let mut durable_restore_ms = durable_probe_ms;
    let prefill_mode = if let Some(snapshot_path) = args.deepseek_v4_snapshot.as_ref() {
        let model_content_id = snapshot_model_content_id
            .expect("snapshot path resolved a model-content identity before residency");
        if snapshot_file_exists {
            let (snapshot, restored_prefix) = restored_snapshot
                .as_ref()
                .expect("existing snapshot was validated before residency");
            session
                .restore_causal_snapshot(snapshot)
                .context("restore DeepSeek V4 causal snapshot")?;
            let suffix_chunks = execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[*restored_prefix..],
                prefill_chunk_tokens,
            )?;
            eprintln!(
                "deepseek_v4: snapshot restore path={} restored_tokens={} suffix_tokens={} suffix_chunks={} payload_bytes={}",
                snapshot_path.display(),
                restored_prefix,
                prompt_token_ids.len() - *restored_prefix,
                suffix_chunks,
                snapshot.payload_bytes(),
            );
            "causal_snapshot_restore"
        } else {
            let publish_prefix = deepseek_v4_snapshot_publish_prefix(prompt_token_ids.len())?;
            advance_deepseek_v4_prompt_prefix(
                &mut session,
                &ctx,
                &prompt_token_ids[..publish_prefix],
                prefill_chunk_tokens,
            )?;
            let snapshot = session
                .capture_causal_snapshot()
                .context("capture DeepSeek V4 causal snapshot")?;
            let report = publish_causal_snapshot_file(
                snapshot_path,
                &snapshot,
                DeepSeekV4SnapshotCodecConstraints {
                    config: session.residency().config(),
                    session_capacity: session.capacity(),
                    expected_model_content_id: model_content_id,
                    max_record_bytes: DEEPSEEK_V4_SNAPSHOT_MAX_RECORD_BYTES,
                },
            )
            .with_context(|| {
                format!(
                    "publish DeepSeek V4 causal snapshot {}",
                    snapshot_path.display()
                )
            })?;
            execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[publish_prefix..],
                prefill_chunk_tokens,
            )?;
            eprintln!(
                "deepseek_v4: snapshot publish path={} outcome={} prefix_tokens={} payload_bytes={} record_bytes={}",
                snapshot_path.display(),
                match report.outcome {
                    DeepSeekV4SnapshotFileOutcome::Published => "published",
                    DeepSeekV4SnapshotFileOutcome::AlreadyPresent => "already_present",
                },
                publish_prefix,
                snapshot.payload_bytes(),
                report.record_bytes,
            );
            "causal_snapshot_publish"
        }
    } else {
        let (restored_prefix, durable_payload_bytes) = attempt_deepseek_v4_durable_restore(
            durable_store.as_ref(),
            durable_has_blobs,
            &mut session,
            &prompt_token_ids,
            durable_max_record_bytes,
            durable_probe_ms,
            &mut durable_restore_ms,
        )?;
        // Completed-turn runs skip the mid-prefill prompt boundary: the
        // post-decode transcript strictly covers it.
        let capture_boundary = deepseek_v4_durable_capture_prefix_len(
            prompt_token_ids.len(),
            durable_admitted && !durable_completed_eligible,
        )?;
        if let Some(publish_prefix) = capture_boundary {
            if restored_prefix < publish_prefix {
                advance_deepseek_v4_prompt_prefix(
                    &mut session,
                    &ctx,
                    &prompt_token_ids[restored_prefix..publish_prefix],
                    prefill_chunk_tokens,
                )?;
            }
            let capture_t0 = Instant::now();
            match session.prepare_durable_checkpoint() {
                Ok(prepared) => durable_prepared = Some(prepared),
                Err(error)
                    if causal_snapshot_capture_error_kind(&error)
                        == DeepSeekV4SnapshotCaptureErrorKind::Allocation =>
                {
                    eprintln!(
                        "warning: durable DeepSeek V4 prefix capture allocation failed; continuing without publication: {error}"
                    );
                }
                Err(error) => {
                    return Err(error).context("capture durable DeepSeek V4 causal snapshot");
                }
            }
            durable_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
            let suffix_chunks = execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[publish_prefix..],
                prefill_chunk_tokens,
            )?;
            if restored_prefix > 0 {
                eprintln!(
                    "deepseek_v4: durable restore restored_tokens={} promoted_tokens={} suffix_tokens={} suffix_chunks={} payload_bytes={}",
                    restored_prefix,
                    publish_prefix - restored_prefix,
                    prompt_token_ids.len() - publish_prefix,
                    suffix_chunks,
                    durable_payload_bytes,
                );
                "durable_prefix_restore"
            } else {
                "durable_prefix_capture"
            }
        } else if restored_prefix > 0 {
            let suffix_chunks = execute_deepseek_v4_prompt_suffix(
                &mut session,
                &ctx,
                &prompt_token_ids[restored_prefix..],
                prefill_chunk_tokens,
            )?;
            eprintln!(
                "deepseek_v4: durable restore restored_tokens={} promoted_tokens=0 suffix_tokens={} suffix_chunks={} payload_bytes={}",
                restored_prefix,
                prompt_token_ids.len() - restored_prefix,
                suffix_chunks,
                durable_payload_bytes,
            );
            "durable_prefix_restore"
        } else {
            let packed_chunk_count =
                deepseek_v4_packed_chunk_count(prompt_token_ids.len(), prefill_chunk_tokens);
            if packed_chunk_count > 0 {
                execute_deepseek_v4_prompt_suffix(
                    &mut session,
                    &ctx,
                    &prompt_token_ids,
                    prefill_chunk_tokens,
                )?;
                if packed_chunk_count == 1 {
                    "layer_major"
                } else {
                    "layer_major_chunks"
                }
            } else {
                for (index, &token) in prompt_token_ids.iter().enumerate() {
                    session
                        .forward_token(&ctx, token)
                        .with_context(|| format!("forward DeepSeek V4 prompt token {index}"))?;
                }
                "singleton"
            }
        }
    };
    let reconciliation = memory_plan
        .reconcile(DeepSeekV4MemorySamples {
            before_residency_bytes,
            after_residency_bytes,
            after_session_bytes,
            after_first_forward_bytes: ctx.current_allocated_size(),
        })
        .context("reconcile admitted DeepSeek V4 Metal memory")?;
    eprintln!("deepseek_v4: memory reconciliation; {reconciliation}");
    let logits = copy_deepseek_v4_logits(&session, vocab_size, "prompt")?;
    deepseek_v4_debug_dump_logits_sha256("single_turn", &logits);
    deepseek_v4_debug_dump_top_logits("single_turn", &logits, &tokenizer);
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let mut sampler = Sampler::new(sampling).context("initialize DeepSeek V4 sampler")?;
    const STAGE_PROFILE_GROUPS: usize = 4;
    let stage_profile_enabled = qwen_llm::env_flag::read_default_off("QWEN_DSV4_STAGE_PROFILE");
    const WHOLE_PROFILE_SAMPLES: usize = 8;
    let whole_profile_enabled = qwen_llm::env_flag::read_default_off("QWEN_DSV4_WHOLE_PROFILE");
    #[cfg(feature = "dsv4-diagnostics")]
    let temporal_window = configured_deepseek_v4_temporal_window()?;
    #[cfg(not(feature = "dsv4-diagnostics"))]
    let temporal_window = 0usize;
    ensure!(
        usize::from(stage_profile_enabled)
            + usize::from(whole_profile_enabled)
            + usize::from(temporal_window > 0)
            <= 1,
        "QWEN_DSV4_STAGE_PROFILE, QWEN_DSV4_WHOLE_PROFILE, and QWEN_DSV4_TEMPORAL_WINDOW are mutually exclusive"
    );
    #[cfg(feature = "dsv4-diagnostics")]
    let mut temporal_capture = dsv4_temporal::TemporalCapture::new(temporal_window);
    let mut profile_warmup_pending = stage_profile_enabled || whole_profile_enabled;
    let mut stage_profile_next_group = 0usize;
    let mut stage_profile_aggregate = DeepSeekV4CliStageProfileAggregate::default();
    let mut whole_profile_positions = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_wall_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_gpu_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_outside_gpu_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_encode_cpu_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let mut whole_profile_wait_residual_ms = Vec::with_capacity(WHOLE_PROFILE_SAMPLES);
    let stdout_handle = std::io::stdout();
    let mut stdout = stdout_handle.lock();
    let generation = generate_serial(
        logits,
        args.tokens,
        &stop_tokens,
        &mut sampler,
        |token| {
            let piece = tokenizer
                .try_decode_piece_bytes_exact(token)
                .with_context(|| format!("decode DeepSeek V4 token {token}"))?;
            stdout
                .write_all(piece)
                .with_context(|| format!("write DeepSeek V4 token {token}"))?;
            stdout.flush().context("flush DeepSeek V4 token")?;
            Ok(())
        },
        |token| {
            let token = checked_token_id(token, vocab_size, "generated")?;
            #[cfg(feature = "dsv4-diagnostics")]
            let capture_temporal = temporal_capture.should_capture();
            #[cfg(not(feature = "dsv4-diagnostics"))]
            let capture_temporal = false;
            if capture_temporal {
                #[cfg(feature = "dsv4-diagnostics")]
                {
                    let position = session.next_position();
                    session
                        .arm_decision_transcript(position)
                        .context("arm temporal DeepSeek V4 decision capture")?;
                    session
                        .forward_token(&ctx, token)
                        .context("forward temporal DeepSeek V4 token")?;
                    let transcript = session
                        .take_decision_transcript_and_reset()
                        .context("take temporal DeepSeek V4 decision capture")?;
                    temporal_capture.record(transcript)?;
                }
            } else if std::mem::take(&mut profile_warmup_pending) {
                session
                    .forward_token(&ctx, token)
                    .context("warm profiled DeepSeek V4 decode")?;
            } else if stage_profile_enabled && stage_profile_next_group < STAGE_PROFILE_GROUPS {
                let group = stage_profile_next_group;
                stage_profile_next_group += 1;
                let sampled_layers = (0..session.residency().config().attention_kinds.len())
                    .filter(|layer| layer % STAGE_PROFILE_GROUPS == group)
                    .collect::<Vec<_>>();
                let profile = session
                    .forward_token_stage_profiled(&ctx, token, &sampled_layers)
                    .context("stage-profile generated DeepSeek V4 token")?;
                emit_deepseek_v4_stage_profile(
                    &profile,
                    group,
                    STAGE_PROFILE_GROUPS,
                    &mut stage_profile_aggregate,
                );
            } else if whole_profile_enabled && whole_profile_positions.len() < WHOLE_PROFILE_SAMPLES
            {
                let profile = session
                    .forward_token_whole_profiled(&ctx, token)
                    .context("whole-profile generated DeepSeek V4 token")?;
                whole_profile_positions.push(profile.position);
                whole_profile_wall_ms.push(profile.forward_wall_ms);
                whole_profile_gpu_ms.push(profile.command_gpu_ms);
                whole_profile_outside_gpu_ms.push(profile.outside_gpu_ms());
                whole_profile_encode_cpu_ms.push(profile.encode_cpu_ms);
                whole_profile_wait_residual_ms.push(profile.wait_residual_ms());
                if whole_profile_positions.len() == WHOLE_PROFILE_SAMPLES {
                    emit_deepseek_v4_whole_profile(
                        &whole_profile_positions,
                        &whole_profile_wall_ms,
                        &whole_profile_gpu_ms,
                        &whole_profile_outside_gpu_ms,
                        &whole_profile_encode_cpu_ms,
                        &whole_profile_wait_residual_ms,
                    );
                }
            } else {
                session
                    .forward_token(&ctx, token)
                    .context("forward generated DeepSeek V4 token")?;
            }
            copy_deepseek_v4_logits(&session, vocab_size, "continuing")
        },
    )?;
    drop(stdout);
    if durable_completed_eligible
        && durable_admitted
        && durable_store.is_some()
        && durable_prepared.is_none()
        && args.deepseek_v4_snapshot.is_none()
    {
        if matches!(generation.stop_reason, StopReason::Eos) {
            // The session sits at the completed transcript boundary: every
            // prompt and generated token except the unconsumed terminal EOS.
            match causal_snapshot_record_bytes(
                session.residency().config(),
                session.capacity(),
                session.next_position(),
                durable_max_record_bytes,
            ) {
                Ok(_) => {
                    let capture_t0 = Instant::now();
                    match session.prepare_durable_checkpoint() {
                        Ok(prepared) => {
                            durable_prepared = Some(prepared);
                            durable_capture_kind = "completed";
                        }
                        Err(error)
                            if causal_snapshot_capture_error_kind(&error)
                                == DeepSeekV4SnapshotCaptureErrorKind::Allocation =>
                        {
                            eprintln!(
                                "warning: durable DeepSeek V4 completed capture allocation failed; continuing without publication: {error}"
                            );
                        }
                        Err(error) => {
                            return Err(error)
                                .context("capture completed DeepSeek V4 causal snapshot");
                        }
                    }
                    durable_capture_ms = capture_t0.elapsed().as_secs_f64() * 1e3;
                }
                Err(error) => eprintln!(
                    "warning: durable DeepSeek V4 completed capture is not admissible; continuing without publication: {error}"
                ),
            }
        } else {
            // A truncated turn's boundary can never prefix a retry of the
            // same prompt; publishing it would only pollute the budget.
            eprintln!(
                "durable_prefix_cache: family=deepseek_v4 publish=skipped capture=completed reason=token_limit"
            );
        }
    }
    if let (Some(store), Some(prepared)) = (durable_store.as_ref(), durable_prepared.as_ref()) {
        let publish_t0 = Instant::now();
        match store.publish_prepared(prepared, durable_max_record_bytes) {
            Ok(report) => eprintln!(
                concat!(
                    "durable_prefix_cache: family=deepseek_v4 publish={} capture={} ",
                    "matched_tokens={} blob_bytes={} evicted={} ",
                    "staging_examined={} staging_removed={} staging_allocated_bytes_reclaimed={} ",
                    "staging_live={} staging_legacy={} staging_foreign={} staging_truncated={} ",
                    "staged_integrity={} staged_integrity_us={} capture_ms={:.1} publish_ms={:.1}"
                ),
                publish_outcome_label(report.outcome),
                durable_capture_kind,
                prepared.next_position(),
                report.blob_bytes,
                report.evicted_entries,
                report.staging_entries_examined,
                report.staging_entries_removed,
                report.staging_allocated_bytes_reclaimed,
                report.staging_live_entries,
                report.staging_legacy_entries,
                report.staging_foreign_entries,
                report.staging_cleanup_truncated,
                report.staged_integrity.mode.as_str(),
                report.staged_integrity.elapsed.as_micros(),
                durable_capture_ms,
                publish_t0.elapsed().as_secs_f64() * 1e3,
            ),
            Err(error) => eprintln!(
                "warning: durable DeepSeek V4 prefix publication failed after response (restore_ms={:.1} capture_ms={:.1}): {error}",
                durable_restore_ms, durable_capture_ms,
            ),
        }
    }
    #[cfg(feature = "dsv4-diagnostics")]
    if temporal_capture.requested_tokens() > 0 {
        let mut report = temporal_capture.finish();
        let final_logits = copy_deepseek_v4_logits(&session, vocab_size, "temporal final")?;
        let final_logits_sha256 =
            hex_encode_bytes(&Sha256::digest(bytemuck::cast_slice(&final_logits)));
        let final_causal_digest = if args.deepseek_v4_snapshot.is_some() {
            let snapshot = session
                .capture_causal_snapshot()
                .context("capture final temporal DeepSeek V4 causal state")?;
            Some(hex_encode_bytes(snapshot.causal_digest()))
        } else {
            None
        };
        report.attach_final_state(final_logits_sha256, final_causal_digest);
        if let Some(path) = std::env::var_os(DEEPSEEK_V4_TEMPORAL_JSON_ENV) {
            let path = PathBuf::from(path);
            std::fs::write(
                &path,
                format!("{}\n", serde_json::to_string_pretty(&report)?),
            )
            .with_context(|| format!("write temporal report {}", path.display()))?;
        }
        eprintln!(
            "deepseek_v4 temporal: {}",
            serde_json::to_string(&report.summary())?
        );
    }
    selector_plan.emit_completion("single_turn", session.multigroup_selector_telemetry())?;

    let decode_tps = if generation.wall_ms > 0.0 {
        generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
    } else {
        0.0
    };
    let prefill_tps = if prefill_ms > 0.0 {
        prompt_ids.len() as f64 / (prefill_ms / 1e3)
    } else {
        0.0
    };
    let transition_tps = if generation.transition_ms > 0.0 {
        generation.transitions as f64 / (generation.transition_ms / 1e3)
    } else {
        0.0
    };
    let generated_ids_sha256 = generated_token_sha256(&generation.tokens);
    eprintln!(
        concat!(
            "deepseek_v4 stats: prompt_kind={} prefill_mode={} prefill_chunk_cap={} prompt_tokens={} generated_tokens={} transitions={} ",
            "stop_reason={} tokenizer_ms={:.1} prefetch_mode={} prefetch_ms={:.1} load_ms={:.1} prefill_ms={:.1} prefill_tps={:.2} ",
            "generation_ms={:.1} decode_tps={:.2} transition_tps={:.2} build_commit={} build_dirty={} generated_ids_sha256={}"
        ),
        prompt_kind,
        prefill_mode,
        prefill_chunk_tokens,
        prompt_ids.len(),
        generation.tokens.len(),
        generation.transitions,
        generation.stop_reason.as_str(),
        tokenizer_ms,
        prefetch_outcome.mode.as_str(),
        prefetch_outcome.wall_ms,
        load_ms,
        prefill_ms,
        prefill_tps,
        generation.wall_ms,
        decode_tps,
        transition_tps,
        env!("QWEN_BUILD_COMMIT"),
        env!("QWEN_BUILD_DIRTY"),
        generated_ids_sha256,
    );
    if std::env::var_os("QWEN_DSV4_GENERATED_IDS").is_some() {
        eprintln!("deepseek_v4 generated_ids: {:?}", generation.tokens);
    }
    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(path, arrival_ms, prompt_ids.len(), generation.tokens.len())?;
    }
    if let Some(path) = args.request_stats_jsonl.as_ref() {
        // Measure total request wall time at the outer boundary (not the sum
        // of phase timings, which can miss inter-phase gaps).
        let total_ms = request_start.elapsed().as_secs_f64() * 1e3;
        let measured = RequestStatsMeasured {
            input_tokens: prompt_ids.len() as u64,
            output_tokens: generation.tokens.len() as u64,
            transitions: generation.transitions as u64,
            stop_reason: generation.stop_reason,
            tokenizer_ms,
            load_ms,
            prefill_ms,
            prefill_tps,
            decode_ms: generation.wall_ms,
            decode_tps,
            transition_tps,
            total_ms,
            output_fingerprint: GeneratedTokenSha256Digest::of(&generation.tokens),
        };
        let record = build_deepseek_v4_single_turn_stats_record(
            &INVOCATION_ID,
            prompt_kind,
            prefill_mode,
            prefill_chunk_tokens as u64,
            &measured,
        );
        append_jsonl_record(path, &record, "request stats jsonl")?;
    }
    Ok(())
}

pub(crate) fn build_deepseek_v4_single_turn_stats_record<'a>(
    invocation_id: &'a str,
    prompt_kind: &'a str,
    prefill_mode: &'static str,
    prefill_chunk_cap: u64,
    measured: &RequestStatsMeasured,
) -> RequestStatsRequestRecord<'a> {
    // input.kind describes representation (raw vs messages); template surfaces
    // the specific chat-template variant when known (e.g., messages_0731_chat
    // -> kind=messages template=0731_chat).
    let (input_kind, input_template) = split_ds4_prompt_kind(prompt_kind);
    build_single_turn_stats_record(
        invocation_id,
        0,
        "deepseek_v4",
        RequestStatsInput {
            kind: input_kind,
            template: input_template,
        },
        measured,
        Some(RequestStatsDiagnostics {
            deepseek_v4: Some(RequestStatsDeepSeekV4Diagnostics {
                schema_version: 1,
                prefill_mode,
                prefill_chunk_cap,
                transitions: measured.transitions,
                transition_tps: sanitize_finite_metric(
                    measured.transition_tps,
                    "diagnostics.deepseek_v4.transition_tps",
                ),
                load_ms: sanitize_finite_metric(
                    measured.load_ms,
                    "diagnostics.deepseek_v4.load_ms",
                ),
            }),
        }),
    )
}

/// Split a DeepSeek V4 `prompt_kind` string into (input_kind, template).
/// The common `input.kind` vocabulary is restricted to `messages`, `raw`, or
/// `unknown` — new backend labels do NOT expand the common core by accident.
///
/// Known values:
///   - "messages_0731_chat"     -> ("messages", Some("0731_chat"))
///   - "messages_0731_thinking" -> ("messages", Some("0731_thinking"))
///   - "raw"                    -> ("raw", None)
///   - "messages_" or "messages" (empty suffix) -> ("unknown", None)
///   - anything else            -> ("unknown", None)
pub(crate) fn split_ds4_prompt_kind(prompt_kind: &str) -> (&str, Option<&str>) {
    if let Some(rest) = prompt_kind.strip_prefix("messages_") {
        if rest.is_empty() {
            ("unknown", None)
        } else {
            ("messages", Some(rest))
        }
    } else if prompt_kind == "raw" {
        ("raw", None)
    } else {
        ("unknown", None)
    }
}

pub(crate) struct DeepSeekV4PreparedRequest {
    pub(crate) id: String,
    pub(crate) line: usize,
    pub(crate) prompt_tokens: usize,
    pub(crate) prompt_token_ids: Vec<u32>,
    pub(crate) max_tokens: usize,
    pub(crate) required_forwards: usize,
    pub(crate) sampling: SamplingConfig,
}

/// DS4 request preparation deliberately diverges from the Qwen preparer in
/// two ways: prompts always tokenize without automatic specials (DS4 prompts
/// carry explicit BOS bytes), and `cache_prefix_tokens` fails closed until
/// the DS4 durable prefix lane lands.
pub(crate) fn prepare_deepseek_v4_jsonl_request_line(
    tokenizer: &Tokenizer,
    vocab_size: u32,
    args: &Args,
    line: &str,
    line_no: usize,
) -> Result<Option<DeepSeekV4PreparedRequest>> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with('#') {
        return Ok(None);
    }
    let request: JsonlRequest =
        serde_json::from_str(trimmed).with_context(|| format!("parse requests line {line_no}"))?;
    let id = request
        .id
        .clone()
        .unwrap_or_else(|| format!("line-{line_no}"));
    ensure!(
        request.cache_prefix_tokens.is_none(),
        "request {id} sets cache_prefix_tokens, which DeepSeek V4 requests do not support yet"
    );
    ensure!(
        request.user.is_none()
            && request.system.is_none()
            && request.no_thinking.is_none()
            && request.reasoning_effort.is_none(),
        "request {id}: templated user rows are not supported for DeepSeek V4 batch requests; submit a raw prompt"
    );
    let prompt = request_prompt(&request, line_no)
        .with_context(|| format!("resolve request {id} prompt at line {line_no}"))?;
    let prompt_ids = tokenizer
        .encode(&prompt, false)
        .with_context(|| format!("tokenize request {id}"))?;
    ensure!(
        !prompt_ids.is_empty(),
        "request {id} tokenized to zero tokens"
    );
    deepseek_v4_debug_dump_prompt_ids(&id, &prompt_ids);
    let prompt_token_ids = prompt_ids
        .iter()
        .enumerate()
        .map(|(index, &token)| {
            checked_token_id(token, vocab_size, &format!("{id} prompt[{index}]"))
        })
        .collect::<Result<Vec<_>>>()?;
    let max_tokens = request.tokens.unwrap_or(args.tokens);
    ensure!(max_tokens > 0, "request {id} requires tokens >= 1");
    let required_forwards = required_forwards(
        "DeepSeek V4",
        prompt_token_ids.len(),
        max_tokens,
        Some(DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY),
    )
    .with_context(|| format!("derive forward budget for request {id}"))?;
    let sampling = request_sampling_config(&request, args)
        .with_context(|| format!("validate sampling for request {id}"))?;
    Ok(Some(DeepSeekV4PreparedRequest {
        id,
        line: line_no,
        prompt_tokens: prompt_ids.len(),
        prompt_token_ids,
        max_tokens,
        required_forwards,
        sampling,
    }))
}

/// Executes prepared requests sequentially against one long-lived residency,
/// rebuilding a fresh session per request. Mirrors the Qwen JSONL contract:
/// buffered per-request output lines, fail-fast on the first error, and no
/// token streaming.
pub(crate) fn run_deepseek_v4_requests_jsonl(
    model_path: &Path,
    gguf: GgufFile,
    args: &Args,
    explicit: ExplicitCliOptions,
) -> Result<()> {
    validate_deepseek_v4_requests_mode(args, explicit)?;
    ensure!(args.tokens > 0, "--tokens must be >= 1");
    let prefill_chunk_tokens = deepseek_v4_prefill_chunk_tokens()?;
    let prefetch_mode = configured_deepseek_v4_prefetch_mode()?;
    cli_sampling_config(args)?;
    let stdin_mode = deepseek_v4_requests_reads_stdin(args)?;
    let requests_path = args
        .requests_jsonl
        .clone()
        .expect("requests mode requires --requests-jsonl");

    let tokenizer = Tokenizer::from_gguf(&gguf).context("load DeepSeek V4 tokenizer")?;
    let vocab_size = tokenizer.n_vocab();
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load producer-declared DeepSeek V4 stop tokens")?;
    for &token in &stop_tokens {
        checked_token_id(token, vocab_size, "stop")?;
    }

    let (prepared, forward_budget, logical_context_limit) = if stdin_mode {
        let context_limit = args
            .max_context_tokens
            .expect("stdin requests mode validated --max-context-tokens");
        let forward_budget = deepseek_v4_forward_budget_for_context_limit(context_limit)?;
        (None, forward_budget, Some(context_limit))
    } else {
        let raw = std::fs::read_to_string(&requests_path)
            .with_context(|| format!("read requests file {}", requests_path.display()))?;
        let mut prepared = Vec::new();
        for (index, line) in raw.lines().enumerate() {
            if let Some(request) = prepare_deepseek_v4_jsonl_request_line(
                &tokenizer,
                vocab_size,
                args,
                line,
                index + 1,
            )? {
                prepared.push(request);
            }
        }
        ensure!(
            !prepared.is_empty(),
            "requests file {} contains no requests",
            requests_path.display()
        );
        let budget = prepared
            .iter()
            .map(|request| request.required_forwards)
            .max()
            .expect("nonempty prepared requests");
        (Some(prepared), budget, None)
    };

    eprintln!(
        "deepseek_v4: loading {} for requests; source={} forward_budget={} logical_context_limit={:?} promoted_capacity={}",
        model_path.display(),
        if stdin_mode { "stdin" } else { "file" },
        forward_budget,
        logical_context_limit,
        DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
    );
    let load_t0 = Instant::now();
    let ctx = MetalContext::new().context("init Metal context for DeepSeek V4")?;
    let load_plan = DeepSeekV4MetalResidency::plan_for_forward_limit(&ctx, &gguf, forward_budget)
        .context("plan strict DeepSeek V4 Metal residency and session")?;
    let session_capacity = load_plan.session_capacity();
    eprintln!(
        "deepseek_v4: session capacity forwards={} csa_physical_rows={} hca_physical_rows={}",
        session_capacity.forward_limit(),
        session_capacity.csa_physical_rows(),
        session_capacity.hca_physical_rows(),
    );
    let selector_plan = DeepSeekV4MultigroupSelectorPlan::new(
        args.deepseek_v4_multigroup_selector,
        ctx.device.name().to_string(),
        session_capacity,
    )?;
    let memory_plan = load_plan.memory_plan().clone();
    let initial_memory_signals = ctx.memory_signals();
    eprintln!("deepseek_v4: memory plan; {memory_plan}");
    let auto_mode = args.execution_mode == Some(execution_selector::ExecutionModeArg::Auto);
    let auto_selection = if auto_mode {
        let request_count = prepared.as_ref().map_or(0, Vec::len);
        let two_session_memory_admitted = memory_plan
            .admission_for_sessions(initial_memory_signals, 2)
            .context("evaluate automatic DeepSeek V4 two-session admission")?
            .admitted;
        let selection =
            execution_selector::select_deepseek(execution_selector::DeepSeekSelectionInput {
                mode: args.execution_mode,
                request_count,
                stdin: stdin_mode,
                residency_set: qwen_llm::env_flag::read_default_off("QWEN_DSV4_RESIDENCY_SET"),
                two_session_memory_admitted,
            });
        eprintln!(
            "execution_selection: {}",
            serde_json::to_string(&execution_selector::ExecutionSelectionRecord::new(
                Some(ModelFamily::DeepSeek4),
                (!stdin_mode).then_some(request_count),
                selection,
                None,
                None,
                None,
                None,
                None,
            ))
            .context("serialize DeepSeek V4 execution selection")?
        );
        Some(selection)
    } else {
        None
    };
    let use_concurrency = args.concurrency.is_some()
        || auto_selection.is_some_and(|selection| {
            selection.selected == execution_selector::SelectedExecution::Concurrency2
        });
    let session_count = if use_concurrency { 2 } else { 1 };
    let admitted_load_plan = load_plan
        .admit_for_sessions(initial_memory_signals, session_count)
        .context("admit strict DeepSeek V4 Metal residency and session")?;
    let _prefetch_outcome = apply_deepseek_v4_prefetch(&gguf, prefetch_mode)?;
    let realized = DeepSeekV4MetalResidency::load_from_plan(&ctx, &gguf, admitted_load_plan)
        .context("load admitted strict DeepSeek V4 Metal residency")?;
    shutdown::checkpoint()?;
    let (residency, memory_admission, after_residency_bytes) = realized.into_parts();
    let memory_signals = memory_admission.signals;
    eprintln!(
        "deepseek_v4: memory admission admitted={} reason={} recommended={} current={} process_remaining={:?} working_set_headroom={:?} required={:?}",
        memory_admission.admitted,
        memory_admission.reason.as_str(),
        memory_signals.recommended_max_bytes,
        memory_signals.current_allocated_bytes,
        memory_signals.process_limit_remaining_bytes,
        memory_admission.working_set_headroom_bytes,
        memory_admission.required_bytes,
    );
    let before_residency_bytes = memory_signals.current_allocated_bytes;
    memory_plan
        .reconcile_residency(before_residency_bytes, after_residency_bytes)
        .context("reconcile DeepSeek V4 residency allocation")?;
    ensure!(
        residency.config().vocab_size == vocab_size,
        "DeepSeek V4 tokenizer vocabulary {} differs from resident model vocabulary {}",
        vocab_size,
        residency.config().vocab_size,
    );
    eprintln!(
        "deepseek_v4: resident on {} in {:.1} ms; {}",
        ctx.describe(),
        load_t0.elapsed().as_secs_f64() * 1e3,
        residency.report(),
    );

    if use_concurrency {
        let prepared = prepared.expect("DeepSeek concurrency requires file lookahead");
        let executed = concurrent_jsonl::run_deepseek_file(
            &ctx,
            residency,
            &selector_plan,
            &tokenizer,
            vocab_size,
            &stop_tokens,
            prepared,
            prefill_chunk_tokens,
            memory_plan.session_priced_upper_bytes(),
        )?;
        eprintln!("deepseek_v4: requests complete; executed={executed}");
        return Ok(());
    }

    let stdout_handle = std::io::stdout();
    let mut residency_slot = Some(residency);
    let mut reconcile_first_session = Some((before_residency_bytes, after_residency_bytes));
    let mut executed = 0usize;
    let mut execute = |request: DeepSeekV4PreparedRequest| -> Result<()> {
        shutdown::checkpoint()?;
        if let Some(limit) = logical_context_limit {
            validate_deepseek_v4_request_context_limit(
                &request.id,
                request.prompt_tokens,
                request.max_tokens,
                limit,
            )?;
        }
        ensure!(
            request.required_forwards <= session_capacity.forward_limit(),
            "request {} requires {} forwards, beyond the run's shared session budget {}",
            request.id,
            request.required_forwards,
            session_capacity.forward_limit(),
        );
        let arrival_ms = unix_epoch_ms()?;
        let residency = residency_slot
            .take()
            .expect("residency is returned after every request");
        let session_t0 = Instant::now();
        let mut session = DeepSeekV4Session::new(&ctx, residency)
            .with_context(|| format!("create DeepSeek V4 session for request {}", request.id))?;
        selector_plan.seal_session(&mut session, &request.id)?;
        let first_memory_sample =
            reconcile_first_session
                .take()
                .map(|(before, after_residency)| {
                    (before, after_residency, ctx.current_allocated_size())
                });
        let session_ms = session_t0.elapsed().as_secs_f64() * 1e3;

        let request_execution = (|| -> Result<(GenerationResult, Vec<u8>, &'static str, f64)> {
            if let Some((before, after_residency, after_session)) = first_memory_sample {
                memory_plan
                    .reconcile_session(before, after_residency, after_session)
                    .context("reconcile DeepSeek V4 request session allocation")?;
            }
            let prefill_t0 = Instant::now();
            let packed_chunk_count = deepseek_v4_packed_chunk_count(
                request.prompt_token_ids.len(),
                prefill_chunk_tokens,
            );
            let prefill_mode = if packed_chunk_count > 0 {
                execute_deepseek_v4_prompt_suffix(
                    &mut session,
                    &ctx,
                    &request.prompt_token_ids,
                    prefill_chunk_tokens,
                )
                .with_context(|| format!("prefill request {}", request.id))?;
                if packed_chunk_count == 1 {
                    "layer_major"
                } else {
                    "layer_major_chunks"
                }
            } else {
                for (index, &token) in request.prompt_token_ids.iter().enumerate() {
                    session.forward_token(&ctx, token).with_context(|| {
                        format!("forward request {} prompt token {index}", request.id)
                    })?;
                }
                "singleton"
            };
            if let Some((before, after_residency, after_session)) = first_memory_sample {
                let reconciliation = memory_plan
                    .reconcile(DeepSeekV4MemorySamples {
                        before_residency_bytes: before,
                        after_residency_bytes: after_residency,
                        after_session_bytes: after_session,
                        after_first_forward_bytes: ctx.current_allocated_size(),
                    })
                    .context("reconcile admitted DeepSeek V4 request memory")?;
                eprintln!("deepseek_v4: request memory reconciliation; {reconciliation}");
            }
            let logits = copy_deepseek_v4_logits(&session, vocab_size, "prompt")
                .with_context(|| format!("copy request {} prompt logits", request.id))?;
            deepseek_v4_debug_dump_logits_sha256(&request.id, &logits);
            deepseek_v4_debug_dump_top_logits(&request.id, &logits, &tokenizer);
            let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

            let mut sampler = Sampler::new(request.sampling)
                .with_context(|| format!("initialize sampler for request {}", request.id))?;
            let mut generated_bytes = Vec::new();
            let mut transition_index = 0usize;
            let generation = generate_serial(
                logits,
                request.max_tokens,
                &stop_tokens,
                &mut sampler,
                |token| {
                    let piece = tokenizer
                        .try_decode_piece_bytes_exact(token)
                        .with_context(|| format!("decode request {} token {token}", request.id))?;
                    generated_bytes.extend_from_slice(piece);
                    Ok(())
                },
                |token| {
                    let current_transition = transition_index;
                    let token = checked_token_id(
                        token,
                        vocab_size,
                        &format!(
                            "request {} generated transition {current_transition}",
                            request.id
                        ),
                    )?;
                    session.forward_token(&ctx, token).with_context(|| {
                        format!(
                            "forward request {} generated transition {current_transition}",
                            request.id
                        )
                    })?;
                    let logits = copy_deepseek_v4_logits(&session, vocab_size, "continuing")
                        .with_context(|| {
                            format!(
                                "copy request {} continuing logits after transition {current_transition}",
                                request.id
                            )
                        })?;
                    transition_index += 1;
                    Ok(logits)
                },
            )?;
            Ok((generation, generated_bytes, prefill_mode, prefill_ms))
        })();
        let selector_telemetry = session.multigroup_selector_telemetry();
        residency_slot = Some(session.into_residency()?);
        let (generation, generated_bytes, prefill_mode, prefill_ms) = request_execution
            .with_context(|| format!("execute request {} at line {}", request.id, request.line))?;
        selector_plan.emit_completion(&request.id, selector_telemetry)?;

        let decode_tps = if generation.wall_ms > 0.0 {
            generation.tokens.len() as f64 / (generation.wall_ms / 1e3)
        } else {
            0.0
        };
        let prefill_tps = if prefill_ms > 0.0 {
            request.prompt_tokens as f64 / (prefill_ms / 1e3)
        } else {
            0.0
        };
        let output = RequestOutput {
            id: request.id.clone(),
            input: JsonlInputLabel::RAW,
            prompt_tokens: request.prompt_tokens,
            generated_tokens: generation.tokens.len(),
            generated_token_sha256: generated_token_sha256(&generation.tokens),
            generated_text: String::from_utf8_lossy(&generated_bytes).into_owned(),
            stop_reason: generation.stop_reason,
            terminal_token_target_transition_consumed: false,
            thinking_partition: None,
        };
        {
            let mut stdout = stdout_handle.lock();
            serde_json::to_writer(&mut stdout, &output)
                .with_context(|| format!("serialize output for request {}", request.id))?;
            stdout
                .write_all(b"\n")
                .and_then(|_| stdout.flush())
                .with_context(|| format!("write output for request {}", request.id))?;
        }
        eprintln!(
            concat!(
                "deepseek_v4 stats: request={} line={} prompt_kind=raw prefill_mode={} prefill_chunk_cap={} prompt_tokens={} ",
                "generated_tokens={} transitions={} stop_reason={} session_ms={:.1} prefill_ms={:.1} prefill_tps={:.2} ",
                "generation_ms={:.1} decode_tps={:.2} build_commit={} build_dirty={}"
            ),
            output.id,
            request.line,
            prefill_mode,
            prefill_chunk_tokens,
            output.prompt_tokens,
            output.generated_tokens,
            generation.transitions,
            generation.stop_reason.as_str(),
            session_ms,
            prefill_ms,
            prefill_tps,
            generation.wall_ms,
            decode_tps,
            env!("QWEN_BUILD_COMMIT"),
            env!("QWEN_BUILD_DIRTY"),
        );
        if let Some(path) = args.trace_request.as_ref() {
            append_request_trace(
                path,
                arrival_ms,
                output.prompt_tokens,
                output.generated_tokens,
            )?;
        }
        executed += 1;
        Ok(())
    };

    if let Some(prepared) = prepared {
        for request in prepared {
            execute(request)?;
        }
    } else {
        shutdown::checkpoint()?;
        let stdin = std::io::stdin();
        for (index, line) in stdin.lock().lines().enumerate() {
            shutdown::checkpoint()?;
            let line = line.context("read requests line from stdin")?;
            if let Some(request) = prepare_deepseek_v4_jsonl_request_line(
                &tokenizer,
                vocab_size,
                args,
                &line,
                index + 1,
            )? {
                execute(request)?;
            }
        }
        ensure!(executed > 0, "stdin request stream contained no requests");
    }
    eprintln!("deepseek_v4: requests complete; executed={executed}");
    Ok(())
}

/// Debug observability: `QWEN_DSV4_PROMPT_IDS=1` dumps the exact input token
/// stream fed to the model, for cross-engine tokenization diffs.
pub(crate) fn deepseek_v4_debug_dump_prompt_ids(scope: &str, prompt_ids: &[i32]) {
    if std::env::var_os("QWEN_DSV4_PROMPT_IDS").is_some() {
        eprintln!(
            "deepseek_v4 prompt_ids: scope={scope:?} count={} ids={:?}",
            prompt_ids.len(),
            prompt_ids
        );
    }
}

/// Debug observability: `QWEN_DSV4_LOGITS_SHA256=1` emits a compact exact
/// identity for the complete first-token F32 logit vector.
pub(crate) fn deepseek_v4_debug_dump_logits_sha256(scope: &str, logits: &[f32]) {
    if std::env::var_os("QWEN_DSV4_LOGITS_SHA256").is_some() {
        eprintln!(
            "deepseek_v4 logits: scope={scope:?} count={} sha256_f32le={:x}",
            logits.len(),
            Sha256::digest(bytemuck::cast_slice(logits))
        );
    }
}

/// Debug observability: `QWEN_DSV4_TOP_LOGITS=N` dumps the top-N first-token
/// logits with decoded pieces, for greedy near-tie margin analysis.
pub(crate) fn deepseek_v4_debug_dump_top_logits(
    scope: &str,
    logits: &[f32],
    tokenizer: &Tokenizer,
) {
    let Some(value) = std::env::var_os("QWEN_DSV4_TOP_LOGITS") else {
        return;
    };
    let count = value
        .to_string_lossy()
        .parse::<usize>()
        .unwrap_or(5)
        .clamp(1, 50);
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    for (rank, &(token, logit)) in ranked.iter().take(count).enumerate() {
        let piece = tokenizer
            .try_decode_piece_bytes_exact(token as i32)
            .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
            .unwrap_or_else(|_| "<undecodable>".into());
        let margin = ranked[0].1 - logit;
        eprintln!(
            "deepseek_v4 first_token_logit: scope={scope:?} rank={rank} id={token} logit={logit:.6} margin_to_top={margin:.6} piece={piece:?}"
        );
    }
}

pub(crate) fn deepseek_v4_checkpoint_store(
    args: &Args,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<Option<DeepSeekV4CheckpointStore>> {
    let Some(root) = args.durable_prefix_cache.as_ref() else {
        return Ok(None);
    };
    let budget = mib_to_bytes(
        args.durable_prefix_cache_max_mib,
        "durable prefix cache byte budget",
    )?;
    Ok(Some(match staged_integrity {
        Some(mode) => DeepSeekV4CheckpointStore::with_staged_integrity(root, budget, mode),
        None => DeepSeekV4CheckpointStore::new(root, budget),
    }))
}

pub(crate) fn deepseek_v4_durable_capture_prefix_len(
    prompt_len: usize,
    admitted: bool,
) -> Result<Option<usize>> {
    if !admitted || prompt_len < 2 {
        return Ok(None);
    }
    deepseek_v4_snapshot_publish_prefix(prompt_len).map(Some)
}

/// Probe the durable store and restore the longest strict prefix into the
/// session, translating the runtime's fail-open policy into telemetry:
/// misses, store faults, and pre-mutation restore allocation failures all
/// return `(0, 0)` and cold-prefill; only invariant restore failures are
/// fatal. Returns `(restored_prefix_len, payload_bytes)`.
pub(crate) fn attempt_deepseek_v4_durable_restore(
    durable_store: Option<&DeepSeekV4CheckpointStore>,
    durable_has_blobs: Option<bool>,
    session: &mut DeepSeekV4Session,
    prompt_token_ids: &[u32],
    durable_max_record_bytes: u64,
    durable_probe_ms: f64,
    durable_restore_ms: &mut f64,
) -> Result<(usize, u64)> {
    let store = match (durable_store, durable_has_blobs) {
        (Some(store), Some(true)) => store,
        (Some(_), Some(false)) => {
            eprintln!(
                "durable_prefix_cache: family=deepseek_v4 store_empty=true restore_total_ms={durable_probe_ms:.1}",
            );
            return Ok((0, 0));
        }
        _ => return Ok((0, 0)),
    };
    let restore_t0 = Instant::now();
    match session.restore_durable_prefix(store, prompt_token_ids, durable_max_record_bytes) {
        Ok(attempt) => {
            *durable_restore_ms = durable_probe_ms + restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                concat!(
                    "durable_prefix_cache: family=deepseek_v4 checkpoint_hit={} ",
                    "matched={} restored={} candidates={} corrupt_removed={} ",
                    "restore_total_ms={:.1}"
                ),
                attempt.restored_prefix_len.is_some(),
                attempt.matched_prefix_len,
                attempt.restored_prefix_len.unwrap_or(0),
                attempt.candidates_examined,
                attempt.corrupt_entries_removed,
                *durable_restore_ms,
            );
            Ok((
                attempt.restored_prefix_len.unwrap_or(0),
                attempt.payload_bytes,
            ))
        }
        Err(DeepSeekV4DurableError::UnboundIdentity) => {
            // Occupied store, but the prompt was too short to resolve an
            // identity for this session; there is nothing to restore.
            eprintln!(
                "durable_prefix_cache: family=deepseek_v4 checkpoint_hit=false skipped=short_prompt restore_total_ms={durable_probe_ms:.1}",
            );
            Ok((0, 0))
        }
        Err(DeepSeekV4DurableError::Restore(error))
            if causal_snapshot_restore_error_kind(&error)
                == DeepSeekV4SnapshotRestoreErrorKind::Allocation =>
        {
            *durable_restore_ms = durable_probe_ms + restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "warning: durable DeepSeek V4 prefix restore allocation failed; cold-prefilling: {error}"
            );
            Ok((0, 0))
        }
        Err(DeepSeekV4DurableError::Restore(error)) => {
            Err(error).context("restore durable DeepSeek V4 causal snapshot")
        }
        Err(DeepSeekV4DurableError::Store(error)) => {
            *durable_restore_ms = durable_probe_ms + restore_t0.elapsed().as_secs_f64() * 1e3;
            eprintln!(
                "warning: durable DeepSeek V4 prefix lookup failed after {:.1} ms; cold-prefilling: {error}",
                *durable_restore_ms,
            );
            Ok((0, 0))
        }
    }
}

pub(crate) fn print_deepseek_v4_info(gguf: &qwen_llm::gguf::GgufFile) -> Result<()> {
    use std::collections::BTreeSet;

    let model = DeepSeekV4Model::from_gguf_flash_0731(gguf)
        .context("bind strict DeepSeek V4 Flash-0731 schema")?;
    let config = &model.config;
    let (local, csa, hca) = config.attention_counts();
    let hash_layers = model
        .blocks
        .iter()
        .filter(|block| matches!(block.moe.router, RouterWeights::TokenHash { .. }))
        .count();
    let csa_layers = model
        .blocks
        .iter()
        .filter(|block| matches!(block.attention.lane, AttentionLane::CompressedSparse { .. }))
        .count();
    let gate_up_types = model
        .blocks
        .iter()
        .flat_map(|block| [block.moe.gate_experts.dtype, block.moe.up_experts.dtype])
        .map(|dtype| dtype.to_string())
        .collect::<BTreeSet<_>>();
    let down_types = model
        .blocks
        .iter()
        .map(|block| block.moe.down_experts.dtype.to_string())
        .collect::<BTreeSet<_>>();

    println!(
        "deepseek4 target: {} layers, hidden={}, vocab={}, context={}",
        config.layer_count, config.hidden_size, config.vocab_size, config.context_length
    );
    println!(
        "attention: {local} local, {csa} CSA ratio-4, {hca} HCA ratio-128; heads={} shared-KV={}x{} local-window={} index-topk={}",
        config.attention_head_count,
        config.kv_head_count,
        config.key_length,
        config.sliding_window,
        config.indexer_top_k,
    );
    println!(
        "mHC: streams={} sinkhorn-iters={} epsilon={}; MoE: experts={} topk={} hash-layers={hash_layers}",
        config.hyper_connection_count,
        config.sinkhorn_iterations,
        config.hyper_connection_epsilon,
        config.expert_count,
        config.expert_used_count,
    );
    println!(
        "tokenizer: {}/{} bos={:?} eos={:?} pad={:?}",
        config.tokenizer_model,
        config.tokenizer_pre,
        config.bos_token_id,
        config.eos_token_id,
        config.padding_token_id,
    );
    println!(
        "trailing compression-ratio entries={} (target-only GGUF); routed gate/up types={gate_up_types:?}, down types={down_types:?}",
        config.compress_ratio_tail.len(),
    );
    println!(
        "strict tensor schema: validated all {} tensors; CSA indexers={csa_layers}",
        model.source_tensor_count
    );
    Ok(())
}

pub(crate) fn print_deepseek_v4_census(model_path: &Path) -> Result<()> {
    let gguf = qwen_llm::gguf::GgufFile::open(model_path)
        .with_context(|| format!("open DeepSeek V4 model {}", model_path.display()))?;
    ensure!(
        ModelFamily::detect(&gguf) == Some(ModelFamily::DeepSeek4),
        "--deepseek-census-json requires general.architecture=deepseek4"
    );
    let census = DeepSeekV4CensusV1::from_gguf_flash_0731(&gguf)
        .context("construct DeepSeek V4 schema/quant census")?;
    serde_json::to_writer_pretty(std::io::stdout().lock(), &census)?;
    println!();
    Ok(())
}
