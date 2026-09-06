use anyhow::{Context, Result, bail, ensure};
use clap::{Args, Parser, Subcommand, ValueEnum};
use qwen_llm::checkpoint_identity::{CheckpointIdentityCache, checkpoint_content_identity};
use qwen_llm::model::Arch;
use qwen_llm::runtime::{Runtime, SequenceConfig};
use qwen_llm::workspace_lens::{
    MAX_WORKSPACE_LENS_DIM_BATCH, MAX_WORKSPACE_LENS_TOKENS, WorkspaceLensBlockKind,
    WorkspaceLensReplayDiagnostic, WorkspaceLensRule,
};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::fs::{DirBuilder, File, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

mod full_lens;
mod lens_compare;
mod lens_input;
mod lens_inspect;
mod lens_run;
#[allow(dead_code)]
mod messages;
mod model_request;
mod muse_full_lens;
mod muse_full_lens_artifact;
mod muse_lens_artifact;
mod muse_lens_fit;
mod muse_lens_rows_artifact;
mod muse_lens_rows_fit;
mod muse_lens_run;
mod muse_published_full_lens;
mod muse_published_full_lens_artifact;
mod open_responses;
#[allow(dead_code)]
mod prompt_template;
mod published_pt;
#[allow(dead_code)]
mod template_lens;
use full_lens::{
    CompareTransferArgs, ImportFullArgs, ReadFullArgs, TraceFullArgs, compare_transfer,
    import_full, read_full as read_qwen_full, trace_full as trace_qwen_full,
    trace_full_batch as trace_qwen_full_batch,
};

const SHARD_SCHEMA: &str = "qwen.workspace_lens_row_shard";
const CHECKPOINT_SCHEMA: &str = "qwen.workspace_lens_row_checkpoint";
const SCHEMA_VERSION: u32 = 1;
const ESTIMATOR_VERSION: &str = "summed_causal_target_vjp_mean_source_positions_v1";
const ORIENTATION: &str = "source_layer_output_coordinate_source_coordinate";
const RULE_VERSION: &str = "qwen_workspace_rules_v1";
const PAYLOAD_NAME: &str = "rows.f32le";
const MANIFEST_NAME: &str = "shard.json";
const CHECKPOINT_NAME: &str = "checkpoint.json";
const TOKEN_READOUT_SCHEMA: &str = "qwen.workspace_lens_token_readouts";
const TOKEN_CHECKPOINT_SCHEMA: &str = "qwen.workspace_lens_token_checkpoint";
const TOKEN_PAYLOAD_NAME: &str = "readouts.f32le";
const TOKEN_MANIFEST_NAME: &str = "readouts.json";
const TOKEN_READOUT_VERSION: &str = "lm_head_row_times_output_norm_gamma_f32_v1";
const TOKEN_SCORE_SEMANTICS: &str =
    "selected_token_logit_ranking_numerator_no_rms_denominator_no_softmax_v1";
const TOKEN_ORIENTATION: &str = "source_layer_selected_token_source_coordinate";
const TOKEN_ARTIFACT_MAX_BYTES: usize = 128 * 1024 * 1024;
const TOKEN_ID_ARGUMENT_MAX_COUNT: usize = 65_536;
// Valid row-v1 and token artifacts are accepted up to this explicit JSON compatibility limit.
pub(crate) const JSON_FILE_MAX_BYTES: usize = 16 * 1024 * 1024;
const MAX_PROMPT_RECORDS: usize = 10_000;
// At six JSON bytes per escaped input byte, 10k skipped IDs occupy at most 7.32 MiB.
const MAX_PROMPT_ID_BYTES: usize = 128;
const MAX_PROMPT_RECORD_BYTES: usize = 1024 * 1024;
const MAX_SELECTED_CORPUS_BYTES: usize = 64 * 1024 * 1024;

pub(crate) struct ByteLimitedWriter<W> {
    inner: W,
    written: usize,
    max_bytes: usize,
}

impl<W> ByteLimitedWriter<W> {
    pub(crate) fn new(inner: W, max_bytes: usize) -> Self {
        Self {
            inner,
            written: 0,
            max_bytes,
        }
    }

    pub(crate) fn written(&self) -> usize {
        self.written
    }
}

impl<W: Write> Write for ByteLimitedWriter<W> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let remaining = self.max_bytes.saturating_sub(self.written);
        if bytes.len() > remaining {
            return Err(std::io::Error::other(format!(
                "serialized output exceeds bounded writer capacity {}",
                self.max_bytes
            )));
        }
        let written = self.inner.write(bytes)?;
        self.written = self
            .written
            .checked_add(written)
            .ok_or_else(|| std::io::Error::other("serialized output byte count overflow"))?;
        Ok(written)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

#[derive(Debug, Parser)]
#[command(name = "qwen-lens", about = "Native Qwen workspace-lens tools")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Compare two compatible trace or run artifacts by exact identities.
    Compare(lens_compare::CompareArgs),
    /// Inspect a bounded qwen.lens.trace artifact without loading a model.
    Inspect(lens_inspect::InspectArgs),
    /// Verify and inspect one coefficient-sweep bundle without loading a model.
    #[command(name = "inspect-sweep")]
    InspectSweep(lens_compare::InspectSweepArgs),
    /// Run one Lens request or a resident ordinary-Qwen request cohort.
    #[command(name = "run")]
    LensRun(lens_run::LensRunArgs),
    /// Sweep one operation coefficient with one resident ordinary-Qwen model load.
    #[command(name = "sweep")]
    CoefficientSweep(lens_run::CoefficientSweepArgs),
    /// Fit a resumable contiguous shard of J-lens or R-lens transport rows.
    FitRows(FitRowsArgs),
    /// Fit resumable projected J-lens or R-lens selected-token readouts.
    FitTokens(FitTokensArgs),
    /// Assemble complete Muse row shards into one self-contained F16 transport.
    #[command(name = "assemble-muse-full")]
    AssembleMuseFull(muse_full_lens::AssembleMuseFullArgs),
    /// Import one exact pinned published Muse full transport without executing pickle.
    #[command(name = "import-muse-full")]
    ImportMuseFull(muse_published_full_lens::ImportMuseFullArgs),
    /// Import one pinned published Qwen3.6/Qwen3.8 full J/R transport safely.
    ImportFull(ImportFullArgs),
    /// Compare the published J-lens with native deployed-checkpoint J directions.
    #[command(alias = "validate-transfer")]
    CompareTransfer(CompareTransferArgs),
    /// Read full-vocabulary logits through an imported published J/R transport.
    ReadFull(ReadFullArgs),
    /// Trace packed full-vocabulary published J/R top-k across layers and positions.
    #[command(name = "trace-full")]
    TraceFull(TraceFullArgs),
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum FitMethod {
    J,
    R,
}

impl FitMethod {
    fn rule(self) -> WorkspaceLensRule {
        match self {
            Self::J => WorkspaceLensRule::Jacobian,
            Self::R => WorkspaceLensRule::Relp,
        }
    }
}

#[derive(Debug, Args)]
struct FitRowsArgs {
    /// Dense Qwen3.8 or Muse Glimmer GGUF model (the first shard is sufficient).
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Strict JSONL corpus; 64 MiB selected, 1 MiB/record, 128 UTF-8 bytes/ID.
    #[arg(long)]
    prompts: PathBuf,

    /// New shard directory, or an incomplete shard directory with --resume.
    #[arg(long)]
    output: PathBuf,

    /// Identity cache; Muse accepts cache or fresh declarations without hashing weights.
    #[arg(long)]
    identity_cache: PathBuf,

    #[arg(long, value_enum)]
    method: FitMethod,

    /// Post-block residual target layer.
    #[arg(long)]
    target_layer: u32,

    /// Strictly increasing post-block source layers.
    #[arg(long, value_delimiter = ',', required = true)]
    source_layers: Vec<u32>,

    /// First target/output coordinate in this shard (inclusive).
    #[arg(long)]
    row_start: u32,

    /// Last target/output coordinate in this shard (exclusive).
    #[arg(long)]
    row_end: u32,

    /// Number of target/output rows propagated in one query-major batch.
    #[arg(long, default_value_t = 8)]
    dim_batch: usize,

    /// Leading attention-sink positions excluded from target and source means.
    #[arg(long, default_value_t = 4)]
    skip_first: usize,

    /// Tokenize/truncate each record to this bound (maximum 128).
    #[arg(long, default_value_t = MAX_WORKSPACE_LENS_TOKENS)]
    max_tokens: usize,

    /// Consume at most this many non-comment JSONL records (maximum 10000).
    #[arg(long, default_value_t = 25)]
    max_prompts: usize,

    /// Muse only: stop after N selected records, including skipped records.
    /// This execution budget starts at the resume cursor and is excluded from fit identity.
    #[arg(long)]
    records_this_run: Option<usize>,

    /// Disable the tokenizer's configured BOS/EOS insertion policy.
    #[arg(long)]
    no_special_tokens: bool,

    /// Resume an incomplete, configuration-identical output directory.
    #[arg(long)]
    resume: bool,
}

#[derive(Debug, Args)]
struct FitTokensArgs {
    /// Dense Qwen3.8 or Muse Glimmer GGUF model (the first shard is sufficient).
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Strict JSONL corpus; 64 MiB selected, 1 MiB/record, 128 UTF-8 bytes/ID.
    #[arg(long)]
    prompts: PathBuf,

    /// New readout directory; ordinary Qwen also permits --resume.
    #[arg(long)]
    output: PathBuf,

    /// Private cache directory for the strong ordered-GGUF content identity.
    #[arg(long)]
    identity_cache: PathBuf,

    #[arg(long, value_enum)]
    method: FitMethod,

    /// Post-block residual target layer.
    #[arg(long)]
    target_layer: u32,

    /// Strictly increasing post-block sources below the target layer.
    #[arg(long, value_delimiter = ',', required = true)]
    source_layers: Vec<u32>,

    /// Selected vocabulary token IDs, in output query order.
    #[arg(long, value_delimiter = ',', required = true)]
    token_ids: Vec<u32>,

    /// Selected-token query batch (ordinary default 8; Muse requires explicit 1).
    #[arg(long, default_value_t = 8)]
    dim_batch: usize,

    /// Leading attention-sink positions excluded from target and source means.
    #[arg(long, default_value_t = 4)]
    skip_first: usize,

    /// Tokenize/truncate each record to this bound (maximum 128).
    #[arg(long, default_value_t = MAX_WORKSPACE_LENS_TOKENS)]
    max_tokens: usize,

    /// Consume at most this many non-comment JSONL records (maximum 10000).
    #[arg(long, default_value_t = 25)]
    max_prompts: usize,

    /// Disable the tokenizer's configured BOS/EOS insertion policy.
    #[arg(long)]
    no_special_tokens: bool,

    /// Resume an incomplete ordinary-Qwen artifact; Muse rejects --resume.
    #[arg(long)]
    resume: bool,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct PromptRequest {
    id: Option<String>,
    prompt: Option<String>,
    token_ids: Option<Vec<i32>>,
}

#[derive(Clone, Debug)]
struct PreparedPrompt {
    id: String,
    token_ids: Vec<i32>,
    original_token_count: usize,
    truncated: bool,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct FitConfig {
    estimator_version: String,
    orientation: String,
    rule_version: String,
    method: FitMethod,
    model_content_blake3: String,
    model_locator_id: String,
    tokenizer_metadata_id: String,
    architecture: String,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    full_attention_interval: u32,
    target_layer: u32,
    source_layers: Vec<u32>,
    row_start: u32,
    row_end: u32,
    skip_first: usize,
    max_tokens: usize,
    add_special_tokens: bool,
    corpus_blake3: String,
    selected_records: usize,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct PayloadDescriptor {
    path: String,
    dtype: String,
    shape: [usize; 3],
    byte_length: u64,
    blake3: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ReplayDiagnostic {
    layer: u32,
    kind: String,
    residual_replay_max_abs_error: f32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SkippedPrompt {
    id: String,
    reason: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FitCheckpoint {
    schema: String,
    schema_version: u32,
    config_blake3: String,
    generation: u64,
    next_record: usize,
    used_prompts: u64,
    truncated_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    forward_seconds: f64,
    vjp_seconds: f64,
    sums: PayloadDescriptor,
    diagnostics: Vec<ReplayDiagnostic>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusSummary {
    selected_records: usize,
    used_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    truncated_prompts: u64,
    ordered_token_ids_blake3: String,
    add_special_tokens: bool,
    max_tokens: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FitSummary {
    estimator_version: String,
    orientation: String,
    rule_version: String,
    method: FitMethod,
    target_layer: u32,
    source_layers: Vec<u32>,
    row_start: u32,
    row_end: u32,
    skip_first: usize,
    valid_position_denominator: String,
    accumulator_dtype: String,
    storage_dtype: String,
    forward_seconds: f64,
    vjp_seconds: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelSummary {
    path: String,
    content_blake3: String,
    content_identity_outcome: String,
    content_bytes_hashed: u64,
    #[serde(
        rename = "research_identity_scheme",
        alias = "workspace_lens_identity_scheme"
    )]
    workspace_lens_identity_scheme: String,
    model_locator_id: String,
    tokenizer_metadata_id: String,
    content_authenticated: bool,
    architecture: String,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    full_attention_interval: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Provenance {
    build_commit: String,
    build_dirty: String,
    build_source_state: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenProvenance {
    build_commit: String,
    build_dirty: String,
    build_source_state: String,
    build_stamp_source: String,
    build_stamp_error: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct FitShardManifest {
    schema: String,
    schema_version: u32,
    status: String,
    config_blake3: String,
    config: FitConfig,
    model: ModelSummary,
    corpus: CorpusSummary,
    fit: FitSummary,
    payload: PayloadDescriptor,
    diagnostics: Vec<ReplayDiagnostic>,
    provenance: Provenance,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenReadoutSpec {
    readout_version: String,
    score_semantics: String,
    token_ids: Vec<u32>,
    target_covectors_blake3: String,
    target_covectors_dtype: String,
    target_covectors_shape: [usize; 2],
    lm_head_dtype: String,
    lm_head_shape: [usize; 2],
    output_norm_dtype: String,
    output_norm_shape: Vec<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenFitConfig {
    estimator_version: String,
    orientation: String,
    rule_version: String,
    method: FitMethod,
    model_content_blake3: String,
    model_locator_id: String,
    tokenizer_metadata_id: String,
    architecture: String,
    n_layers: u32,
    hidden_size: u32,
    vocab_size: u32,
    full_attention_interval: u32,
    target_layer: u32,
    source_layers: Vec<u32>,
    readouts: TokenReadoutSpec,
    skip_first: usize,
    max_tokens: usize,
    add_special_tokens: bool,
    corpus_blake3: String,
    selected_records: usize,
    build_source_state: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct DimBatchTiming {
    dim_batch: usize,
    prompt_count: u64,
    vjp_seconds: f64,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenFitCheckpoint {
    schema: String,
    schema_version: u32,
    config_blake3: String,
    generation: u64,
    next_record: usize,
    used_prompts: u64,
    truncated_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    forward_seconds: f64,
    vjp_seconds: f64,
    sums: PayloadDescriptor,
    diagnostics: Vec<ReplayDiagnostic>,
    dim_batch_timings: Vec<DimBatchTiming>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenFitSummary {
    estimator_version: String,
    orientation: String,
    rule_version: String,
    method: FitMethod,
    target_layer: u32,
    source_layers: Vec<u32>,
    skip_first: usize,
    valid_position_denominator: String,
    prompt_denominator: String,
    accumulator_dtype: String,
    storage_dtype: String,
    forward_seconds: f64,
    vjp_seconds: f64,
    dim_batch_timings: Vec<DimBatchTiming>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct TokenReadoutManifest {
    schema: String,
    schema_version: u32,
    status: String,
    config_blake3: String,
    config: TokenFitConfig,
    model: ModelSummary,
    corpus: CorpusSummary,
    fit: TokenFitSummary,
    readouts: TokenReadoutSpec,
    payload: PayloadDescriptor,
    diagnostics: Vec<ReplayDiagnostic>,
    provenance: TokenProvenance,
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Compare(args) => lens_compare::run(args),
        Command::Inspect(args) => lens_inspect::run(args),
        Command::InspectSweep(args) => lens_compare::inspect_sweep(args),
        Command::LensRun(args) => lens_run::run(args),
        Command::CoefficientSweep(args) => lens_run::run_coefficient_sweep(args),
        Command::FitRows(args) => fit_rows(args),
        Command::FitTokens(args) => fit_tokens(args),
        Command::AssembleMuseFull(args) => muse_full_lens::assemble(args),
        Command::ImportMuseFull(args) => muse_published_full_lens::import_full(args),
        Command::ImportFull(args) => import_full(args),
        Command::CompareTransfer(args) => compare_transfer(args),
        Command::ReadFull(args) => read_full(args),
        Command::TraceFull(args) => trace_full(args),
    }
}

fn fit_rows(args: FitRowsArgs) -> Result<()> {
    let gguf = qwen_llm::gguf::GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    if muse_lens_artifact::is_muse_architecture(gguf.architecture().as_deref()) {
        return muse_lens_rows_fit::fit_rows(args, gguf);
    }
    drop(gguf);
    fit_qwen_rows(args)
}

fn read_full(args: ReadFullArgs) -> Result<()> {
    if muse_full_lens::is_artifact(&args.full_lens)? {
        muse_full_lens::read_full(args)
    } else {
        read_qwen_full(args)
    }
}

fn trace_full(args: TraceFullArgs) -> Result<()> {
    let muse = muse_full_lens::is_artifact(&args.full_lens)?;
    if args.requests_jsonl.is_some() {
        ensure!(
            !muse,
            "--requests-jsonl currently supports ordinary Qwen full lenses"
        );
        return trace_qwen_full_batch(args);
    }
    if muse {
        ensure!(
            args.open_responses.is_none(),
            "--open-responses supports ordinary Qwen only; Muse Glimmer is not supported"
        );
        muse_full_lens::trace_full(args)
    } else {
        trace_qwen_full(args)
    }
}

fn fit_qwen_rows(mut args: FitRowsArgs) -> Result<()> {
    validate_args(&args)?;
    args.output = resolve_output_path(&args.output)?;
    let requests = read_prompt_requests(&args.prompts, args.max_prompts)?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_model(&args.model)
        .with_context(|| format!("load model {}", args.model.display()))?;
    let arch = loaded.arch();
    ensure!(
        args.target_layer < arch.n_layer,
        "--target-layer {} is out of range for {} layers",
        args.target_layer,
        arch.n_layer
    );
    ensure!(
        args.source_layers
            .iter()
            .all(|&source| source < args.target_layer),
        "every --source-layers entry must be below --target-layer {}",
        args.target_layer
    );
    ensure!(
        args.row_end <= arch.hidden_size,
        "row range {}..{} exceeds hidden size {}",
        args.row_start,
        args.row_end,
        arch.hidden_size
    );

    let tokenizer = loaded.tokenizer().context("load tokenizer from GGUF")?;
    let add_special_tokens = !args.no_special_tokens;
    let prompts = prepare_prompts(
        requests,
        &tokenizer,
        add_special_tokens,
        args.max_tokens,
        arch.vocab_size,
    )?;
    let corpus_blake3 = corpus_digest(&prompts);
    let identity = loaded.workspace_lens_identity();
    let content = checkpoint_content_identity(
        loaded.gguf(),
        &CheckpointIdentityCache::new(&args.identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve strong model identity using {}",
            args.identity_cache.display()
        )
    })?;
    let model_content_blake3 = hex(&content.content_id);
    let config = FitConfig {
        estimator_version: ESTIMATOR_VERSION.into(),
        orientation: ORIENTATION.into(),
        rule_version: RULE_VERSION.into(),
        method: args.method,
        model_content_blake3: model_content_blake3.clone(),
        model_locator_id: format!("{:016x}", identity.model_locator_id),
        tokenizer_metadata_id: format!("{:016x}", identity.tokenizer_metadata_id),
        architecture: "qwen3_hybrid_dense".into(),
        n_layers: arch.n_layer,
        hidden_size: arch.hidden_size,
        vocab_size: arch.vocab_size,
        full_attention_interval: arch.full_attention_interval,
        target_layer: args.target_layer,
        source_layers: args.source_layers.clone(),
        row_start: args.row_start,
        row_end: args.row_end,
        skip_first: args.skip_first,
        max_tokens: args.max_tokens,
        add_special_tokens,
        corpus_blake3: corpus_blake3.clone(),
        selected_records: prompts.len(),
    };
    let config_blake3 = digest_json(&config)?;
    let row_count = usize::try_from(args.row_end - args.row_start).context("row count")?;
    let hidden_size = arch.hidden_size as usize;
    let value_count = args
        .source_layers
        .len()
        .checked_mul(row_count)
        .and_then(|value| value.checked_mul(hidden_size))
        .context("fit shard value count overflow")?;

    let mut state = open_or_create_state(
        &args.output,
        args.resume,
        &config,
        &prompts,
        &config_blake3,
        value_count,
        args.source_layers.len(),
        row_count,
        hidden_size,
    )?;
    if let WorkerState::Complete(manifest) = state {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    let WorkerState::Active(ref mut active) = state else {
        unreachable!();
    };
    validate_active_state(
        active,
        &prompts,
        args.skip_first,
        arch,
        args.target_layer,
        &args.source_layers,
        value_count,
    )?;

    let output_rows: Vec<u32> = (args.row_start..args.row_end).collect();
    for (record_index, prompt) in prompts.iter().enumerate().skip(active.next_record) {
        if let Some(skipped) = skipped_prompt(prompt, args.skip_first) {
            active.skipped_prompts.push(skipped);
            active.next_record = record_index + 1;
            checkpoint_active(
                &args.output,
                &config_blake3,
                active,
                args.source_layers.len(),
                row_count,
                hidden_size,
            )?;
            continue;
        }
        eprintln!(
            "fit prompt {}/{} id={} tokens={} rows={}..{} dim_batch={}",
            record_index + 1,
            prompts.len(),
            prompt.id,
            prompt.token_ids.len(),
            args.row_start,
            args.row_end,
            args.dim_batch,
        );
        let mut sequence = loaded
            .create_sequence(SequenceConfig::new(prompt.token_ids.len()))
            .with_context(|| format!("create sequence for prompt {}", prompt.id))?;
        let mut workspace_lens = loaded
            .workspace_lens_session(&mut sequence)
            .with_context(|| format!("open workspace-lens session for prompt {}", prompt.id))?;
        let started = Instant::now();
        let forward = workspace_lens
            .forward_prompt_with_workspace_capture(&prompt.token_ids)
            .with_context(|| format!("capture workspace prompt {}", prompt.id))?;
        active.forward_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let rows = workspace_lens
            .workspace_fit_rows_batched(
                &forward,
                args.target_layer,
                &args.source_layers,
                &output_rows,
                args.skip_first,
                args.dim_batch,
                args.method.rule(),
            )
            .with_context(|| format!("fit workspace rows for prompt {}", prompt.id))?;
        active.vjp_seconds += started.elapsed().as_secs_f64();
        ensure!(
            rows.values.len() == active.sums.len(),
            "fitted row count {} != accumulator count {}",
            rows.values.len(),
            active.sums.len()
        );
        ensure!(
            rows.values.iter().all(|value| value.is_finite()),
            "prompt {} produced non-finite fitted rows",
            prompt.id
        );
        for (sum, value) in active.sums.iter_mut().zip(rows.values) {
            *sum += value;
        }
        merge_diagnostics(&mut active.diagnostics, &rows.diagnostics)?;
        active.used_prompts += 1;
        active.truncated_prompts += u64::from(prompt.truncated);
        active.next_record = record_index + 1;
        checkpoint_active(
            &args.output,
            &config_blake3,
            active,
            args.source_layers.len(),
            row_count,
            hidden_size,
        )?;
    }
    ensure!(active.used_prompts > 0, "no prompt had any valid positions");

    let scale = (active.used_prompts as f32).recip();
    let mut averaged = active.sums.clone();
    for value in &mut averaged {
        *value *= scale;
    }
    ensure!(
        averaged.iter().all(|value| value.is_finite()),
        "final averaged shard contains non-finite values"
    );
    let payload_bytes = encode_f32_le(&averaged)?;
    let payload_path = args.output.join(PAYLOAD_NAME);
    publish_immutable(&payload_path, &payload_bytes)?;
    let payload = payload_descriptor(
        PAYLOAD_NAME,
        &payload_bytes,
        [args.source_layers.len(), row_count, hidden_size],
    );
    let manifest = FitShardManifest {
        schema: SHARD_SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        status: "complete".into(),
        config_blake3: config_blake3.clone(),
        config: config.clone(),
        model: ModelSummary {
            path: args.model.display().to_string(),
            content_blake3: model_content_blake3,
            content_identity_outcome: format!("{:?}", content.outcome),
            content_bytes_hashed: content.bytes_hashed,
            workspace_lens_identity_scheme: "qwen_llm_model_locator_v1".into(),
            model_locator_id: config.model_locator_id.clone(),
            tokenizer_metadata_id: config.tokenizer_metadata_id.clone(),
            content_authenticated: true,
            architecture: config.architecture.clone(),
            n_layers: arch.n_layer,
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
            full_attention_interval: arch.full_attention_interval,
        },
        corpus: CorpusSummary {
            selected_records: prompts.len(),
            used_prompts: active.used_prompts,
            skipped_prompts: active.skipped_prompts.clone(),
            truncated_prompts: active.truncated_prompts,
            ordered_token_ids_blake3: corpus_blake3,
            add_special_tokens,
            max_tokens: args.max_tokens,
        },
        fit: FitSummary {
            estimator_version: ESTIMATOR_VERSION.into(),
            orientation: ORIENTATION.into(),
            rule_version: RULE_VERSION.into(),
            method: args.method,
            target_layer: args.target_layer,
            source_layers: args.source_layers.clone(),
            row_start: args.row_start,
            row_end: args.row_end,
            skip_first: args.skip_first,
            valid_position_denominator: "number_of_valid_source_positions".into(),
            accumulator_dtype: "f32".into(),
            storage_dtype: "f32_le".into(),
            forward_seconds: active.forward_seconds,
            vjp_seconds: active.vjp_seconds,
        },
        payload,
        diagnostics: active.diagnostics.clone(),
        provenance: Provenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
        },
    };
    let manifest_bytes = serialize_json_pretty_bounded(&manifest, "row shard manifest")?;
    publish_immutable(&args.output.join(MANIFEST_NAME), &manifest_bytes)?;
    sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

fn fit_tokens(mut args: FitTokensArgs) -> Result<()> {
    let gguf = qwen_llm::gguf::GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    if muse_lens_artifact::is_muse_architecture(gguf.architecture().as_deref()) {
        return muse_lens_fit::fit_tokens(args, gguf);
    }
    validate_token_args(&args)?;
    validate_token_build_identity(
        env!("QWEN_BUILD_SOURCE_STATE"),
        env!("QWEN_BUILD_STAMP_ERROR"),
    )?;
    args.output = resolve_output_path(&args.output)?;
    let requests = read_prompt_requests(&args.prompts, args.max_prompts)?;
    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_model(&args.model)
        .with_context(|| format!("load model {}", args.model.display()))?;
    let arch = loaded.arch();
    ensure!(
        args.target_layer < arch.n_layer,
        "--target-layer {} is out of range for {} layers",
        args.target_layer,
        arch.n_layer
    );
    ensure!(
        args.source_layers
            .iter()
            .all(|&source| source < args.target_layer),
        "every --source-layers entry must be below --target-layer {}",
        args.target_layer
    );
    if let Some(&token_id) = args
        .token_ids
        .iter()
        .find(|&&token_id| token_id >= arch.vocab_size)
    {
        bail!(
            "--token-ids entry {token_id} is outside vocab {}",
            arch.vocab_size
        );
    }
    let hidden_size = arch.hidden_size as usize;
    let (value_count, expected_shape) =
        token_artifact_layout(args.source_layers.len(), args.token_ids.len(), hidden_size)?;

    let tokenizer = loaded.tokenizer().context("load tokenizer from GGUF")?;
    let add_special_tokens = !args.no_special_tokens;
    let prompts = prepare_prompts(
        requests,
        &tokenizer,
        add_special_tokens,
        args.max_tokens,
        arch.vocab_size,
    )?;
    let corpus_blake3 = corpus_digest(&prompts);
    let identity = loaded.workspace_lens_identity();
    let content = checkpoint_content_identity(
        loaded.gguf(),
        &CheckpointIdentityCache::new(&args.identity_cache),
    )
    .with_context(|| {
        format!(
            "resolve strong model identity using {}",
            args.identity_cache.display()
        )
    })?;
    let model_content_blake3 = hex(&content.content_id);

    let selected = {
        let mut sequence = loaded
            .create_sequence(SequenceConfig::new(1))
            .context("create selected-token readout sequence")?;
        let workspace_lens = loaded
            .workspace_lens_session(&mut sequence)
            .context("open selected-token readout workspace-lens session")?;
        workspace_lens
            .selected_token_readouts(&args.token_ids)
            .context("derive selected-token target covectors")?
    };
    ensure!(
        selected.token_ids == args.token_ids
            && selected.hidden_size == hidden_size
            && selected.values.len()
                == args
                    .token_ids
                    .len()
                    .checked_mul(hidden_size)
                    .context("selected-token covector size overflow")?
            && selected.values.iter().all(|value| value.is_finite()),
        "selected-token target covectors have invalid metadata, shape, or values"
    );
    let readout_spec = TokenReadoutSpec {
        readout_version: TOKEN_READOUT_VERSION.into(),
        score_semantics: TOKEN_SCORE_SEMANTICS.into(),
        token_ids: selected.token_ids.clone(),
        target_covectors_blake3: token_covector_digest(
            &selected.token_ids,
            hidden_size,
            &selected.values,
        )?,
        target_covectors_dtype: "f32_le".into(),
        target_covectors_shape: [args.token_ids.len(), hidden_size],
        lm_head_dtype: format!("{:?}", selected.lm_head_dtype),
        lm_head_shape: selected.lm_head_shape,
        output_norm_dtype: format!("{:?}", selected.output_norm_dtype),
        output_norm_shape: selected.output_norm_shape.clone(),
    };
    let config = TokenFitConfig {
        estimator_version: ESTIMATOR_VERSION.into(),
        orientation: TOKEN_ORIENTATION.into(),
        rule_version: RULE_VERSION.into(),
        method: args.method,
        model_content_blake3: model_content_blake3.clone(),
        model_locator_id: format!("{:016x}", identity.model_locator_id),
        tokenizer_metadata_id: format!("{:016x}", identity.tokenizer_metadata_id),
        architecture: "qwen3_hybrid_dense".into(),
        n_layers: arch.n_layer,
        hidden_size: arch.hidden_size,
        vocab_size: arch.vocab_size,
        full_attention_interval: arch.full_attention_interval,
        target_layer: args.target_layer,
        source_layers: args.source_layers.clone(),
        readouts: readout_spec.clone(),
        skip_first: args.skip_first,
        max_tokens: args.max_tokens,
        add_special_tokens,
        corpus_blake3: corpus_blake3.clone(),
        selected_records: prompts.len(),
        build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
    };
    let config_blake3 = digest_json(&config)?;

    let mut state = open_or_create_token_state(
        &args.output,
        args.resume,
        &config,
        &prompts,
        &config_blake3,
        value_count,
        expected_shape,
        arch,
    )?;
    if let TokenWorkerState::Complete(manifest) = state {
        println!("{}", serde_json::to_string_pretty(&manifest)?);
        return Ok(());
    }
    let TokenWorkerState::Active(ref mut active) = state else {
        unreachable!();
    };
    validate_active_state(
        active,
        &prompts,
        args.skip_first,
        arch,
        args.target_layer,
        &args.source_layers,
        value_count,
    )?;

    for (record_index, prompt) in prompts.iter().enumerate().skip(active.next_record) {
        if let Some(skipped) = skipped_prompt(prompt, args.skip_first) {
            active.skipped_prompts.push(skipped);
            active.next_record = record_index + 1;
            checkpoint_token_active(&args.output, &config_blake3, active, expected_shape)?;
            continue;
        }
        eprintln!(
            "fit prompt {}/{} id={} tokens={} selected_tokens={} dim_batch={}",
            record_index + 1,
            prompts.len(),
            prompt.id,
            prompt.token_ids.len(),
            args.token_ids.len(),
            args.dim_batch,
        );
        let mut sequence = loaded
            .create_sequence(SequenceConfig::new(prompt.token_ids.len()))
            .with_context(|| format!("create sequence for prompt {}", prompt.id))?;
        let mut workspace_lens = loaded
            .workspace_lens_session(&mut sequence)
            .with_context(|| format!("open workspace-lens session for prompt {}", prompt.id))?;
        let started = Instant::now();
        let forward = workspace_lens
            .forward_prompt_with_workspace_capture(&prompt.token_ids)
            .with_context(|| format!("capture workspace prompt {}", prompt.id))?;
        active.forward_seconds += started.elapsed().as_secs_f64();
        let started = Instant::now();
        let readouts = workspace_lens
            .workspace_fit_readouts_batched(
                &forward,
                args.target_layer,
                &args.source_layers,
                &selected.values,
                args.skip_first,
                args.dim_batch,
                args.method.rule(),
            )
            .with_context(|| format!("fit workspace token readouts for prompt {}", prompt.id))?;
        let vjp_seconds = started.elapsed().as_secs_f64();
        let expected_valid_positions = prompt.token_ids.len() - args.skip_first - 1;
        ensure!(
            readouts.target_layer == args.target_layer
                && readouts.source_layers == args.source_layers
                && readouts.n_query == args.token_ids.len()
                && readouts.n_tokens == prompt.token_ids.len()
                && readouts.n_valid_positions == expected_valid_positions
                && readouts.hidden_size == hidden_size
                && readouts.values.len() == active.sums.len(),
            "prompt {} returned inconsistent fitted readout metadata or shape",
            prompt.id
        );
        ensure!(
            readouts.values.iter().all(|value| value.is_finite()),
            "prompt {} produced non-finite fitted readouts",
            prompt.id
        );
        for (sum, value) in active.sums.iter_mut().zip(readouts.values) {
            *sum += value;
        }
        ensure!(
            active.sums.iter().all(|value| value.is_finite()),
            "prompt {} overflowed the token readout accumulator",
            prompt.id
        );
        merge_diagnostics(&mut active.diagnostics, &readouts.diagnostics)?;
        record_dim_batch_timing(
            &mut active.token_dim_batch_timings,
            args.dim_batch,
            vjp_seconds,
        )?;
        active.vjp_seconds = canonical_vjp_seconds(&active.token_dim_batch_timings)?;
        active.used_prompts += 1;
        active.truncated_prompts += u64::from(prompt.truncated);
        active.next_record = record_index + 1;
        checkpoint_token_active(&args.output, &config_blake3, active, expected_shape)?;
    }
    ensure!(active.used_prompts > 0, "no prompt had any valid positions");

    let scale = (active.used_prompts as f32).recip();
    for value in &mut active.sums {
        *value *= scale;
    }
    ensure!(
        active.sums.iter().all(|value| value.is_finite()),
        "final averaged token readouts contain non-finite values"
    );
    let payload_bytes = encode_f32_le_fallible(&active.sums)?;
    publish_immutable(&args.output.join(TOKEN_PAYLOAD_NAME), &payload_bytes)?;
    let payload = payload_descriptor(TOKEN_PAYLOAD_NAME, &payload_bytes, expected_shape);
    let manifest = TokenReadoutManifest {
        schema: TOKEN_READOUT_SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        status: "complete".into(),
        config_blake3: config_blake3.clone(),
        config: config.clone(),
        model: ModelSummary {
            path: args.model.display().to_string(),
            content_blake3: model_content_blake3,
            content_identity_outcome: format!("{:?}", content.outcome),
            content_bytes_hashed: content.bytes_hashed,
            workspace_lens_identity_scheme: "qwen_llm_model_locator_v1".into(),
            model_locator_id: config.model_locator_id.clone(),
            tokenizer_metadata_id: config.tokenizer_metadata_id.clone(),
            content_authenticated: true,
            architecture: config.architecture.clone(),
            n_layers: arch.n_layer,
            hidden_size: arch.hidden_size,
            vocab_size: arch.vocab_size,
            full_attention_interval: arch.full_attention_interval,
        },
        corpus: CorpusSummary {
            selected_records: prompts.len(),
            used_prompts: active.used_prompts,
            skipped_prompts: active.skipped_prompts.clone(),
            truncated_prompts: active.truncated_prompts,
            ordered_token_ids_blake3: corpus_blake3,
            add_special_tokens,
            max_tokens: args.max_tokens,
        },
        fit: TokenFitSummary {
            estimator_version: ESTIMATOR_VERSION.into(),
            orientation: TOKEN_ORIENTATION.into(),
            rule_version: RULE_VERSION.into(),
            method: args.method,
            target_layer: args.target_layer,
            source_layers: args.source_layers.clone(),
            skip_first: args.skip_first,
            valid_position_denominator: "number_of_valid_source_positions".into(),
            prompt_denominator: "number_of_used_prompts".into(),
            accumulator_dtype: "f32".into(),
            storage_dtype: "f32_le".into(),
            forward_seconds: active.forward_seconds,
            vjp_seconds: active.vjp_seconds,
            dim_batch_timings: active.token_dim_batch_timings.clone(),
        },
        readouts: readout_spec,
        payload,
        diagnostics: active.diagnostics.clone(),
        provenance: TokenProvenance {
            build_commit: env!("QWEN_BUILD_COMMIT").into(),
            build_dirty: env!("QWEN_BUILD_DIRTY").into(),
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
            build_stamp_source: env!("QWEN_BUILD_STAMP_SOURCE").into(),
            build_stamp_error: env!("QWEN_BUILD_STAMP_ERROR").into(),
        },
    };
    validate_token_complete_manifest(&manifest, &config, &prompts, &config_blake3, expected_shape)?;
    publish_immutable(
        &args.output.join(TOKEN_MANIFEST_NAME),
        &serialize_json_pretty_bounded(&manifest, "token readout manifest")?,
    )?;
    sync_directory(&args.output)?;
    println!("{}", serde_json::to_string_pretty(&manifest)?);
    Ok(())
}

#[derive(Debug)]
struct ActiveState {
    generation: u64,
    next_record: usize,
    used_prompts: u64,
    truncated_prompts: u64,
    skipped_prompts: Vec<SkippedPrompt>,
    forward_seconds: f64,
    vjp_seconds: f64,
    sums: Vec<f32>,
    sums_path: Option<String>,
    diagnostics: Vec<ReplayDiagnostic>,
    token_dim_batch_timings: Vec<DimBatchTiming>,
}

#[derive(Debug)]
enum WorkerState {
    Active(ActiveState),
    Complete(Box<FitShardManifest>),
}

#[derive(Debug)]
enum TokenWorkerState {
    Active(ActiveState),
    Complete(Box<TokenReadoutManifest>),
}

fn validate_args(args: &FitRowsArgs) -> Result<()> {
    ensure!(args.max_prompts > 0, "--max-prompts must be nonzero");
    ensure!(
        args.max_prompts <= MAX_PROMPT_RECORDS,
        "--max-prompts {} exceeds artifact compatibility limit {}",
        args.max_prompts,
        MAX_PROMPT_RECORDS
    );
    ensure!(args.dim_batch > 0, "--dim-batch must be nonzero");
    ensure!(
        args.records_this_run.is_none(),
        "--records-this-run is supported only for Muse row fitting"
    );
    ensure!(
        args.dim_batch <= MAX_WORKSPACE_LENS_DIM_BATCH,
        "--dim-batch {} exceeds native workspace limit {}",
        args.dim_batch,
        MAX_WORKSPACE_LENS_DIM_BATCH
    );
    ensure!(args.max_tokens > 0, "--max-tokens must be nonzero");
    ensure!(
        args.max_tokens <= MAX_WORKSPACE_LENS_TOKENS,
        "--max-tokens {} exceeds native workspace limit {}",
        args.max_tokens,
        MAX_WORKSPACE_LENS_TOKENS
    );
    ensure!(
        args.skip_first
            .checked_add(2)
            .is_some_and(|minimum| minimum <= args.max_tokens),
        "--max-tokens must be at least --skip-first + 2"
    );
    ensure!(
        !args.source_layers.is_empty(),
        "--source-layers must not be empty"
    );
    ensure!(
        args.source_layers.windows(2).all(|pair| pair[0] < pair[1]),
        "--source-layers must be strictly increasing and unique"
    );
    ensure!(
        args.row_start < args.row_end,
        "row range must be nonempty and half-open"
    );
    ensure!(
        args.prompts != Path::new("-"),
        "stdin prompt corpora are not supported because resumable fitting requires replayable input"
    );
    Ok(())
}

fn validate_token_args(args: &FitTokensArgs) -> Result<()> {
    ensure!(args.max_prompts > 0, "--max-prompts must be nonzero");
    ensure!(
        args.max_prompts <= MAX_PROMPT_RECORDS,
        "--max-prompts {} exceeds artifact compatibility limit {}",
        args.max_prompts,
        MAX_PROMPT_RECORDS
    );
    ensure!(args.dim_batch > 0, "--dim-batch must be nonzero");
    ensure!(
        args.dim_batch <= MAX_WORKSPACE_LENS_DIM_BATCH,
        "--dim-batch {} exceeds native workspace limit {}",
        args.dim_batch,
        MAX_WORKSPACE_LENS_DIM_BATCH
    );
    ensure!(args.max_tokens > 0, "--max-tokens must be nonzero");
    ensure!(
        args.max_tokens <= MAX_WORKSPACE_LENS_TOKENS,
        "--max-tokens {} exceeds native workspace limit {}",
        args.max_tokens,
        MAX_WORKSPACE_LENS_TOKENS
    );
    ensure!(
        args.skip_first
            .checked_add(2)
            .is_some_and(|minimum| minimum <= args.max_tokens),
        "--max-tokens must be at least --skip-first + 2"
    );
    ensure!(
        !args.source_layers.is_empty(),
        "--source-layers must not be empty"
    );
    ensure!(
        args.source_layers.windows(2).all(|pair| pair[0] < pair[1]),
        "--source-layers must be strictly increasing and unique"
    );
    ensure!(!args.token_ids.is_empty(), "--token-ids must not be empty");
    ensure!(
        args.token_ids.len() <= TOKEN_ID_ARGUMENT_MAX_COUNT,
        "--token-ids count {} exceeds safety limit {}",
        args.token_ids.len(),
        TOKEN_ID_ARGUMENT_MAX_COUNT
    );
    let mut unique_token_ids = HashSet::new();
    unique_token_ids
        .try_reserve(args.token_ids.len())
        .context("allocate --token-ids uniqueness set")?;
    ensure!(
        args.token_ids
            .iter()
            .all(|&token_id| unique_token_ids.insert(token_id)),
        "--token-ids must be unique"
    );
    ensure!(
        args.prompts != Path::new("-"),
        "stdin prompt corpora are not supported because resumable fitting requires replayable input"
    );
    Ok(())
}

fn validate_token_build_identity(source_state: &str, stamp_error: &str) -> Result<()> {
    const PREFIX: &str = "git-source-sha256-v2:";
    let Some(digest) = source_state.strip_prefix(PREFIX) else {
        bail!("token fitting requires a verified git-source-sha256-v2 build identity");
    };
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "token fitting build source-state digest must be 64 lowercase hex characters"
    );
    ensure!(
        stamp_error == "none",
        "token fitting build stamp is not trustworthy: {stamp_error}"
    );
    Ok(())
}

fn validate_prompt_id(id: &str, line_number: usize) -> Result<()> {
    ensure!(!id.is_empty(), "prompt id on line {line_number} is empty");
    ensure!(
        id.len() <= MAX_PROMPT_ID_BYTES,
        "prompt id on line {line_number} is {} bytes; maximum is {MAX_PROMPT_ID_BYTES}",
        id.len()
    );
    Ok(())
}

fn read_prompt_requests(path: &Path, max_prompts: usize) -> Result<Vec<(usize, PromptRequest)>> {
    read_prompt_requests_with_corpus_limit(path, max_prompts, MAX_SELECTED_CORPUS_BYTES)
}

fn read_prompt_requests_with_corpus_limit(
    path: &Path,
    max_prompts: usize,
    maximum_selected_bytes: usize,
) -> Result<Vec<(usize, PromptRequest)>> {
    ensure!(
        max_prompts > 0 && max_prompts <= MAX_PROMPT_RECORDS,
        "prompt record limit must be in 1..={MAX_PROMPT_RECORDS}"
    );
    let file =
        File::open(path).with_context(|| format!("open prompt corpus {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut line_bytes = Vec::new();
    let mut requests = Vec::new();
    let mut selected_bytes = 0usize;
    let mut line_number = 0usize;
    while read_bounded_jsonl_record(
        &mut reader,
        &mut line_bytes,
        MAX_PROMPT_RECORD_BYTES,
        path,
        line_number + 1,
    )? {
        line_number += 1;
        let line = std::str::from_utf8(&line_bytes)
            .with_context(|| format!("read {} line {line_number} as UTF-8", path.display()))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        selected_bytes = selected_bytes
            .checked_add(line_bytes.len())
            .context("selected JSONL corpus byte count overflow")?;
        ensure!(
            selected_bytes <= maximum_selected_bytes,
            "selected JSONL corpus exceeds cumulative limit of {} bytes",
            maximum_selected_bytes
        );
        let request: PromptRequest = serde_json::from_str(trimmed)
            .with_context(|| format!("parse {} line {line_number}", path.display()))?;
        ensure!(
            request.prompt.is_some() ^ request.token_ids.is_some(),
            "{} line {} must contain exactly one of prompt or token_ids",
            path.display(),
            line_number
        );
        requests.push((line_number, request));
        if requests.len() == max_prompts {
            break;
        }
    }
    ensure!(!requests.is_empty(), "prompt corpus has no records");
    Ok(requests)
}

fn read_bounded_jsonl_record<R: BufRead>(
    reader: &mut R,
    output: &mut Vec<u8>,
    maximum_bytes: usize,
    path: &Path,
    line_number: usize,
) -> Result<bool> {
    output.clear();
    loop {
        let buffer = reader
            .fill_buf()
            .with_context(|| format!("read {} line {line_number}", path.display()))?;
        if buffer.is_empty() {
            return Ok(!output.is_empty());
        }
        let newline = buffer.iter().position(|&byte| byte == b'\n');
        let record_bytes = newline.unwrap_or(buffer.len());
        let next_length = output
            .len()
            .checked_add(record_bytes)
            .context("JSONL record length overflow")?;
        ensure!(
            next_length <= maximum_bytes,
            "{} line {} exceeds JSONL record limit of {} bytes",
            path.display(),
            line_number,
            maximum_bytes
        );
        output
            .try_reserve(record_bytes)
            .with_context(|| format!("allocate {} line {line_number}", path.display()))?;
        output.extend_from_slice(&buffer[..record_bytes]);
        let consumed = record_bytes + usize::from(newline.is_some());
        reader.consume(consumed);
        if newline.is_some() {
            return Ok(true);
        }
    }
}

fn prepare_prompts(
    requests: Vec<(usize, PromptRequest)>,
    tokenizer: &impl qwen_llm::tokenizer::Tokenize,
    add_special_tokens: bool,
    max_tokens: usize,
    vocab_size: u32,
) -> Result<Vec<PreparedPrompt>> {
    let mut ids = HashSet::new();
    let mut prompts = Vec::with_capacity(requests.len());
    for (line_number, request) in requests {
        let id = request.id.unwrap_or_else(|| format!("line-{line_number}"));
        validate_prompt_id(&id, line_number)?;
        ensure!(ids.insert(id.clone()), "duplicate prompt id {id:?}");
        let mut token_ids = match (request.prompt, request.token_ids) {
            (Some(prompt), None) => tokenizer
                .encode(&prompt, add_special_tokens)
                .with_context(|| format!("tokenize prompt {id}"))?,
            (None, Some(token_ids)) => token_ids,
            _ => unreachable!("request shape validated before model load"),
        };
        let original_token_count = token_ids.len();
        let truncated = original_token_count > max_tokens;
        token_ids.truncate(max_tokens);
        ensure!(!token_ids.is_empty(), "prompt {id} produced no tokens");
        if let Some(&token_id) = token_ids
            .iter()
            .find(|&&token_id| token_id < 0 || token_id as u32 >= vocab_size)
        {
            bail!("prompt {id} contains token {token_id} outside vocab {vocab_size}");
        }
        prompts.push(PreparedPrompt {
            id,
            token_ids,
            original_token_count,
            truncated,
        });
    }
    Ok(prompts)
}

fn corpus_digest(prompts: &[PreparedPrompt]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"qwen-workspace-lens-corpus-v1\0");
    for prompt in prompts {
        hasher.update(&(prompt.id.len() as u64).to_le_bytes());
        hasher.update(prompt.id.as_bytes());
        hasher.update(&(prompt.original_token_count as u64).to_le_bytes());
        hasher.update(&[u8::from(prompt.truncated)]);
        hasher.update(&(prompt.token_ids.len() as u64).to_le_bytes());
        for token_id in &prompt.token_ids {
            hasher.update(&token_id.to_le_bytes());
        }
    }
    hasher.finalize().to_hex().to_string()
}

fn token_covector_digest(token_ids: &[u32], hidden_size: usize, values: &[f32]) -> Result<String> {
    ensure!(
        values.len()
            == token_ids
                .len()
                .checked_mul(hidden_size)
                .context("token covector digest shape overflow")?,
        "token covector digest values do not match token and hidden dimensions"
    );
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"qwen-selected-token-readout-covectors-f32le-v2\0");
    hasher.update(
        &u64::try_from(token_ids.len())
            .context("token covector count")?
            .to_le_bytes(),
    );
    hasher.update(
        &u64::try_from(hidden_size)
            .context("token covector hidden size")?
            .to_le_bytes(),
    );
    for token_id in token_ids {
        hasher.update(&token_id.to_le_bytes());
    }
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
    Ok(hasher.finalize().to_hex().to_string())
}

fn token_artifact_layout(
    source_count: usize,
    token_count: usize,
    hidden_size: usize,
) -> Result<(usize, [usize; 3])> {
    let value_count = source_count
        .checked_mul(token_count)
        .and_then(|value| value.checked_mul(hidden_size))
        .context("token readout value count overflow")?;
    let byte_count = value_count
        .checked_mul(std::mem::size_of::<f32>())
        .context("token readout payload byte count overflow")?;
    ensure!(
        byte_count <= TOKEN_ARTIFACT_MAX_BYTES,
        "token readout payload {byte_count} bytes exceeds artifact limit {TOKEN_ARTIFACT_MAX_BYTES}"
    );
    Ok((value_count, [source_count, token_count, hidden_size]))
}

fn try_zeroed_f32(count: usize, name: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .with_context(|| format!("allocate {name} with {count} F32 values"))?;
    values.resize(count, 0.0);
    Ok(values)
}

fn record_dim_batch_timing(
    timings: &mut Vec<DimBatchTiming>,
    dim_batch: usize,
    vjp_seconds: f64,
) -> Result<()> {
    ensure!(
        dim_batch > 0
            && dim_batch <= MAX_WORKSPACE_LENS_DIM_BATCH
            && vjp_seconds.is_finite()
            && vjp_seconds >= 0.0,
        "invalid token dim-batch timing sample"
    );
    if let Some(timing) = timings
        .iter_mut()
        .find(|timing| timing.dim_batch == dim_batch)
    {
        timing.prompt_count = timing
            .prompt_count
            .checked_add(1)
            .context("token dim-batch prompt count overflow")?;
        timing.vjp_seconds += vjp_seconds;
        ensure!(
            timing.vjp_seconds.is_finite(),
            "token dim-batch VJP timing overflow"
        );
    } else {
        timings
            .try_reserve(1)
            .context("allocate token dim-batch timing group")?;
        timings.push(DimBatchTiming {
            dim_batch,
            prompt_count: 1,
            vjp_seconds,
        });
    }
    Ok(())
}

fn validate_dim_batch_timings(
    timings: &[DimBatchTiming],
    used_prompts: u64,
    total_vjp_seconds: f64,
) -> Result<()> {
    let mut seen = HashSet::new();
    seen.try_reserve(timings.len())
        .context("allocate token dim-batch validation set")?;
    let mut prompt_count = 0u64;
    for timing in timings {
        ensure!(
            timing.dim_batch > 0
                && timing.dim_batch <= MAX_WORKSPACE_LENS_DIM_BATCH
                && seen.insert(timing.dim_batch)
                && timing.prompt_count > 0
                && timing.vjp_seconds.is_finite()
                && timing.vjp_seconds >= 0.0,
            "invalid or duplicate token dim-batch timing group"
        );
        prompt_count = prompt_count
            .checked_add(timing.prompt_count)
            .context("token dim-batch grouped prompt count overflow")?;
    }
    ensure!(
        prompt_count == used_prompts,
        "token dim-batch prompt counts do not match used prompts"
    );
    let canonical_seconds = canonical_vjp_seconds(timings)?;
    ensure!(
        total_vjp_seconds.is_finite() && canonical_seconds.to_bits() == total_vjp_seconds.to_bits(),
        "token dim-batch VJP timings do not match total VJP time"
    );
    Ok(())
}

fn canonical_vjp_seconds(timings: &[DimBatchTiming]) -> Result<f64> {
    let mut total = 0.0f64;
    for timing in timings {
        ensure!(
            timing.vjp_seconds.is_finite() && timing.vjp_seconds >= 0.0,
            "invalid token dim-batch VJP timing"
        );
        total += timing.vjp_seconds;
        ensure!(total.is_finite(), "token canonical VJP timing overflow");
    }
    Ok(total)
}

fn digest_json(value: &impl Serialize) -> Result<String> {
    Ok(blake3::hash(&serde_json::to_vec(value)?)
        .to_hex()
        .to_string())
}

fn skipped_prompt(prompt: &PreparedPrompt, skip_first: usize) -> Option<SkippedPrompt> {
    (prompt.token_ids.len() < skip_first.checked_add(2)?).then(|| SkippedPrompt {
        id: prompt.id.clone(),
        reason: format!(
            "{} tokens leave no valid positions with skip_first={skip_first}",
            prompt.token_ids.len()
        ),
    })
}

fn validate_active_state(
    state: &ActiveState,
    prompts: &[PreparedPrompt],
    skip_first: usize,
    arch: Arch,
    target_layer: u32,
    source_layers: &[u32],
    expected_values: usize,
) -> Result<()> {
    ensure!(
        state.next_record <= prompts.len(),
        "checkpoint next_record {} exceeds corpus length {}",
        state.next_record,
        prompts.len()
    );
    ensure!(
        state.generation == u64::try_from(state.next_record).context("checkpoint cursor")?,
        "checkpoint generation {} does not match cursor {}",
        state.generation,
        state.next_record
    );
    let processed = &prompts[..state.next_record];
    let expected_skipped: Vec<_> = processed
        .iter()
        .filter_map(|prompt| skipped_prompt(prompt, skip_first))
        .collect();
    let expected_used = u64::try_from(state.next_record - expected_skipped.len())
        .context("checkpoint used prompt count")?;
    let expected_truncated = u64::try_from(
        processed
            .iter()
            .filter(|prompt| skipped_prompt(prompt, skip_first).is_none() && prompt.truncated)
            .count(),
    )
    .context("checkpoint truncated prompt count")?;
    ensure!(
        state.used_prompts == expected_used
            && state.skipped_prompts == expected_skipped
            && state.truncated_prompts == expected_truncated,
        "checkpoint cursor/counter metadata is inconsistent with the corpus prefix"
    );
    ensure!(
        state.forward_seconds.is_finite()
            && state.forward_seconds >= 0.0
            && state.vjp_seconds.is_finite()
            && state.vjp_seconds >= 0.0,
        "checkpoint timings must be finite and nonnegative"
    );
    ensure!(
        state.sums.len() == expected_values && state.sums.iter().all(|value| value.is_finite()),
        "checkpoint accumulator shape or values are invalid"
    );
    if state.used_prompts == 0 {
        ensure!(
            state.diagnostics.is_empty()
                && state.forward_seconds == 0.0
                && state.vjp_seconds == 0.0,
            "checkpoint without fitted prompts must not contain fit diagnostics or timings"
        );
    } else {
        let earliest_source = source_layers
            .first()
            .copied()
            .context("checkpoint validation has no source layers")?;
        validate_replay_schedule(
            &state.diagnostics,
            target_layer,
            earliest_source,
            arch.full_attention_interval,
        )?;
    }
    Ok(())
}

fn validate_replay_schedule(
    diagnostics: &[ReplayDiagnostic],
    target_layer: u32,
    earliest_source: u32,
    full_attention_interval: u32,
) -> Result<()> {
    ensure!(
        full_attention_interval > 0 && earliest_source < target_layer,
        "invalid replay schedule geometry"
    );
    let expected_count =
        usize::try_from(target_layer - earliest_source).context("replay diagnostic count")?;
    ensure!(
        diagnostics.len() == expected_count,
        "replay diagnostic schedule length {} != expected {}",
        diagnostics.len(),
        expected_count
    );
    for (offset, diagnostic) in diagnostics.iter().enumerate() {
        let layer = target_layer - u32::try_from(offset).context("replay diagnostic offset")?;
        let expected_kind = match layer % full_attention_interval {
            remainder if remainder == full_attention_interval - 1 => "attention",
            _ => "gdn",
        };
        ensure!(
            diagnostic.layer == layer
                && diagnostic.kind == expected_kind
                && diagnostic.residual_replay_max_abs_error.is_finite(),
            "invalid replay diagnostic at offset {}",
            offset
        );
    }
    Ok(())
}

fn validate_complete_manifest(
    manifest: &FitShardManifest,
    expected_config: &FitConfig,
    prompts: &[PreparedPrompt],
    expected_config_blake3: &str,
    expected_shape: [usize; 3],
) -> Result<()> {
    ensure!(
        manifest.schema == SHARD_SCHEMA,
        "unknown completed shard schema"
    );
    ensure!(
        manifest.schema_version == SCHEMA_VERSION,
        "unsupported completed shard version"
    );
    ensure!(
        manifest.status == "complete",
        "shard status is not complete"
    );
    ensure!(
        &manifest.config == expected_config,
        "completed shard embedded config does not match the requested fit"
    );
    let embedded_digest = digest_json(&manifest.config)?;
    ensure!(
        manifest.config_blake3 == embedded_digest
            && manifest.config_blake3 == expected_config_blake3,
        "completed shard config digest is inconsistent"
    );
    ensure!(
        manifest.payload.path == PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == expected_shape,
        "completed shard payload descriptor is not canonical"
    );
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .and_then(|values| values.checked_mul(4))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("completed shard byte count overflow")?;
    ensure!(
        manifest.payload.byte_length == expected_bytes,
        "completed shard payload byte length is inconsistent with its shape"
    );
    ensure!(
        manifest.model.content_blake3 == expected_config.model_content_blake3
            && manifest.model.model_locator_id == expected_config.model_locator_id
            && manifest.model.tokenizer_metadata_id == expected_config.tokenizer_metadata_id
            && manifest.model.architecture == expected_config.architecture
            && manifest.model.n_layers == expected_config.n_layers
            && manifest.model.hidden_size == expected_config.hidden_size
            && manifest.model.vocab_size == expected_config.vocab_size
            && manifest.model.full_attention_interval == expected_config.full_attention_interval
            && manifest.model.content_authenticated,
        "completed shard model summary disagrees with its config"
    );
    ensure!(
        manifest.corpus.selected_records == expected_config.selected_records
            && manifest.corpus.used_prompts > 0
            && manifest.corpus.ordered_token_ids_blake3 == expected_config.corpus_blake3
            && manifest.corpus.add_special_tokens == expected_config.add_special_tokens
            && manifest.corpus.max_tokens == expected_config.max_tokens
            && manifest.corpus.used_prompts
                + u64::try_from(manifest.corpus.skipped_prompts.len())
                    .context("completed skipped prompt count")?
                == u64::try_from(manifest.corpus.selected_records)
                    .context("completed selected record count")?
            && manifest.corpus.truncated_prompts <= manifest.corpus.used_prompts,
        "completed shard corpus summary disagrees with its config"
    );
    let expected_skipped: Vec<_> = prompts
        .iter()
        .filter_map(|prompt| skipped_prompt(prompt, expected_config.skip_first))
        .collect();
    let expected_used = u64::try_from(prompts.len() - expected_skipped.len())
        .context("completed used prompt count")?;
    let expected_truncated = u64::try_from(
        prompts
            .iter()
            .filter(|prompt| {
                skipped_prompt(prompt, expected_config.skip_first).is_none() && prompt.truncated
            })
            .count(),
    )
    .context("completed truncated prompt count")?;
    ensure!(
        manifest.corpus.used_prompts == expected_used
            && manifest.corpus.skipped_prompts == expected_skipped
            && manifest.corpus.truncated_prompts == expected_truncated,
        "completed shard corpus counters do not match the bound prompt corpus"
    );
    ensure!(
        manifest.fit.estimator_version == expected_config.estimator_version
            && manifest.fit.orientation == expected_config.orientation
            && manifest.fit.rule_version == expected_config.rule_version
            && manifest.fit.method == expected_config.method
            && manifest.fit.target_layer == expected_config.target_layer
            && manifest.fit.source_layers == expected_config.source_layers
            && manifest.fit.row_start == expected_config.row_start
            && manifest.fit.row_end == expected_config.row_end
            && manifest.fit.skip_first == expected_config.skip_first
            && manifest.fit.valid_position_denominator == "number_of_valid_source_positions"
            && manifest.fit.accumulator_dtype == "f32"
            && manifest.fit.storage_dtype == "f32_le"
            && manifest.fit.forward_seconds.is_finite()
            && manifest.fit.forward_seconds >= 0.0
            && manifest.fit.vjp_seconds.is_finite()
            && manifest.fit.vjp_seconds >= 0.0,
        "completed shard fit summary disagrees with its config"
    );
    let earliest_source = expected_config
        .source_layers
        .first()
        .copied()
        .context("completed shard config has no source layers")?;
    validate_replay_schedule(
        &manifest.diagnostics,
        expected_config.target_layer,
        earliest_source,
        expected_config.full_attention_interval,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && !manifest.provenance.build_dirty.is_empty()
            && !manifest.provenance.build_source_state.is_empty(),
        "completed shard provenance is incomplete"
    );
    Ok(())
}

fn validate_token_complete_manifest(
    manifest: &TokenReadoutManifest,
    expected_config: &TokenFitConfig,
    prompts: &[PreparedPrompt],
    expected_config_blake3: &str,
    expected_shape: [usize; 3],
) -> Result<()> {
    ensure!(
        manifest.schema == TOKEN_READOUT_SCHEMA,
        "unknown completed token readout schema"
    );
    ensure!(
        manifest.schema_version == SCHEMA_VERSION,
        "unsupported completed token readout version"
    );
    ensure!(
        manifest.status == "complete",
        "token readout status is not complete"
    );
    ensure!(
        &manifest.config == expected_config,
        "completed token readout embedded config does not match the requested fit"
    );
    let embedded_digest = digest_json(&manifest.config)?;
    ensure!(
        manifest.config_blake3 == embedded_digest
            && manifest.config_blake3 == expected_config_blake3,
        "completed token readout config digest is inconsistent"
    );
    validate_token_readout_spec(&manifest.config.readouts, expected_config)?;
    ensure!(
        manifest.readouts == expected_config.readouts,
        "completed token readout metadata disagrees with its config"
    );
    ensure!(
        manifest.payload.path == TOKEN_PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == expected_shape,
        "completed token readout payload descriptor is not canonical"
    );
    let expected_bytes = expected_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .and_then(|values| values.checked_mul(4))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("completed token readout byte count overflow")?;
    ensure!(
        expected_bytes <= TOKEN_ARTIFACT_MAX_BYTES as u64
            && manifest.payload.byte_length == expected_bytes,
        "completed token readout payload byte length is inconsistent with its shape"
    );
    ensure!(
        manifest.model.content_blake3 == expected_config.model_content_blake3
            && manifest.model.model_locator_id == expected_config.model_locator_id
            && manifest.model.tokenizer_metadata_id == expected_config.tokenizer_metadata_id
            && manifest.model.architecture == expected_config.architecture
            && manifest.model.n_layers == expected_config.n_layers
            && manifest.model.hidden_size == expected_config.hidden_size
            && manifest.model.vocab_size == expected_config.vocab_size
            && manifest.model.full_attention_interval == expected_config.full_attention_interval
            && manifest.model.workspace_lens_identity_scheme == "qwen_llm_model_locator_v1"
            && manifest.model.content_authenticated,
        "completed token readout model summary disagrees with its config"
    );
    ensure!(
        manifest.corpus.selected_records == expected_config.selected_records
            && manifest.corpus.used_prompts > 0
            && manifest.corpus.ordered_token_ids_blake3 == expected_config.corpus_blake3
            && manifest.corpus.add_special_tokens == expected_config.add_special_tokens
            && manifest.corpus.max_tokens == expected_config.max_tokens
            && manifest.corpus.used_prompts
                + u64::try_from(manifest.corpus.skipped_prompts.len())
                    .context("completed token skipped prompt count")?
                == u64::try_from(manifest.corpus.selected_records)
                    .context("completed token selected record count")?
            && manifest.corpus.truncated_prompts <= manifest.corpus.used_prompts,
        "completed token readout corpus summary disagrees with its config"
    );
    let expected_skipped: Vec<_> = prompts
        .iter()
        .filter_map(|prompt| skipped_prompt(prompt, expected_config.skip_first))
        .collect();
    let expected_used = u64::try_from(prompts.len() - expected_skipped.len())
        .context("completed token used prompt count")?;
    let expected_truncated = u64::try_from(
        prompts
            .iter()
            .filter(|prompt| {
                skipped_prompt(prompt, expected_config.skip_first).is_none() && prompt.truncated
            })
            .count(),
    )
    .context("completed token truncated prompt count")?;
    ensure!(
        manifest.corpus.used_prompts == expected_used
            && manifest.corpus.skipped_prompts == expected_skipped
            && manifest.corpus.truncated_prompts == expected_truncated,
        "completed token readout corpus counters do not match the bound prompt corpus"
    );
    ensure!(
        manifest.fit.estimator_version == expected_config.estimator_version
            && manifest.fit.orientation == expected_config.orientation
            && manifest.fit.rule_version == expected_config.rule_version
            && manifest.fit.method == expected_config.method
            && manifest.fit.target_layer == expected_config.target_layer
            && manifest.fit.source_layers == expected_config.source_layers
            && manifest.fit.skip_first == expected_config.skip_first
            && manifest.fit.valid_position_denominator == "number_of_valid_source_positions"
            && manifest.fit.prompt_denominator == "number_of_used_prompts"
            && manifest.fit.accumulator_dtype == "f32"
            && manifest.fit.storage_dtype == "f32_le"
            && manifest.fit.forward_seconds.is_finite()
            && manifest.fit.forward_seconds >= 0.0
            && manifest.fit.vjp_seconds.is_finite()
            && manifest.fit.vjp_seconds >= 0.0,
        "completed token readout fit summary disagrees with its config"
    );
    validate_dim_batch_timings(
        &manifest.fit.dim_batch_timings,
        manifest.corpus.used_prompts,
        manifest.fit.vjp_seconds,
    )?;
    let earliest_source = expected_config
        .source_layers
        .first()
        .copied()
        .context("completed token readout config has no source layers")?;
    validate_replay_schedule(
        &manifest.diagnostics,
        expected_config.target_layer,
        earliest_source,
        expected_config.full_attention_interval,
    )?;
    ensure!(
        !manifest.provenance.build_commit.is_empty()
            && !manifest.provenance.build_dirty.is_empty()
            && manifest.provenance.build_source_state == expected_config.build_source_state
            && expected_config.build_source_state == env!("QWEN_BUILD_SOURCE_STATE")
            && manifest.provenance.build_stamp_source == env!("QWEN_BUILD_STAMP_SOURCE")
            && manifest.provenance.build_stamp_error == env!("QWEN_BUILD_STAMP_ERROR"),
        "completed token readout provenance is incomplete"
    );
    validate_token_build_identity(
        &manifest.provenance.build_source_state,
        &manifest.provenance.build_stamp_error,
    )?;
    Ok(())
}

fn validate_token_readout_spec(spec: &TokenReadoutSpec, config: &TokenFitConfig) -> Result<()> {
    ensure!(
        spec.readout_version == TOKEN_READOUT_VERSION
            && spec.score_semantics == TOKEN_SCORE_SEMANTICS
            && spec.target_covectors_dtype == "f32_le"
            && spec.target_covectors_shape == [spec.token_ids.len(), config.hidden_size as usize]
            && spec.lm_head_shape == [config.hidden_size as usize, config.vocab_size as usize]
            && spec.output_norm_dtype == "F32"
            && spec.output_norm_shape == [u64::from(config.hidden_size)]
            && spec.target_covectors_blake3.len() == 64
            && spec.token_ids.len() <= TOKEN_ID_ARGUMENT_MAX_COUNT,
        "selected-token readout specification is internally inconsistent"
    );
    ensure!(
        !spec.token_ids.is_empty()
            && spec
                .token_ids
                .iter()
                .all(|&token_id| token_id < config.vocab_size),
        "selected-token readout IDs are empty or outside the configured vocabulary"
    );
    let mut unique = HashSet::new();
    unique
        .try_reserve(spec.token_ids.len())
        .context("allocate selected-token manifest uniqueness set")?;
    ensure!(
        spec.token_ids
            .iter()
            .all(|&token_id| unique.insert(token_id)),
        "selected-token readout IDs are not unique"
    );
    Ok(())
}

fn open_or_create_state(
    output: &Path,
    resume: bool,
    expected_config: &FitConfig,
    prompts: &[PreparedPrompt],
    config_blake3: &str,
    value_count: usize,
    n_sources: usize,
    n_rows: usize,
    hidden_size: usize,
) -> Result<WorkerState> {
    validate_output_leaf(output)?;
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "output {} must be a real directory",
            output.display()
        );
        ensure!(
            resume,
            "output {} already exists; pass --resume",
            output.display()
        );
    } else {
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "output parent {} does not exist",
            parent.display()
        );
        DirBuilder::new()
            .mode(0o700)
            .create(output)
            .with_context(|| format!("create output directory {}", output.display()))?;
        sync_directory(parent)?;
    }

    let manifest_path = output.join(MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: FitShardManifest = read_json_file(&manifest_path)?;
        validate_complete_manifest(
            &manifest,
            expected_config,
            prompts,
            config_blake3,
            [n_sources, n_rows, hidden_size],
        )?;
        let bytes = verify_payload(output, &manifest.payload)?;
        decode_f32_le(&bytes, value_count)?;
        return Ok(WorkerState::Complete(Box::new(manifest)));
    }
    ensure!(
        !output.join(TOKEN_MANIFEST_NAME).exists(),
        "output contains a completed token-readout artifact, not a row shard"
    );

    let checkpoint_path = output.join(CHECKPOINT_NAME);
    if !checkpoint_path.exists() {
        return Ok(WorkerState::Active(ActiveState {
            generation: 0,
            next_record: 0,
            used_prompts: 0,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 0.0,
            vjp_seconds: 0.0,
            sums: vec![0.0; value_count],
            sums_path: None,
            diagnostics: Vec::new(),
            token_dim_batch_timings: Vec::new(),
        }));
    }
    let checkpoint: FitCheckpoint = read_json_file(&checkpoint_path)?;
    ensure!(
        checkpoint.schema == CHECKPOINT_SCHEMA,
        "unknown checkpoint schema"
    );
    ensure!(
        checkpoint.schema_version == SCHEMA_VERSION,
        "unsupported checkpoint version"
    );
    ensure!(
        checkpoint.config_blake3 == config_blake3,
        "checkpoint config {} != requested {}",
        checkpoint.config_blake3,
        config_blake3
    );
    ensure!(
        checkpoint.sums.shape == [n_sources, n_rows, hidden_size],
        "checkpoint sums shape {:?} != expected {:?}",
        checkpoint.sums.shape,
        [n_sources, n_rows, hidden_size]
    );
    ensure!(
        checkpoint.sums.dtype == "f32_le",
        "checkpoint sums dtype must be f32_le"
    );
    ensure!(
        checkpoint.sums.path == format!("sums-{:08}.f32le", checkpoint.generation),
        "checkpoint sums path does not match generation {}",
        checkpoint.generation
    );
    let expected_bytes = value_count
        .checked_mul(4)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("checkpoint byte count overflow")?;
    ensure!(
        checkpoint.sums.byte_length == expected_bytes,
        "checkpoint sums byte length {} != expected {}",
        checkpoint.sums.byte_length,
        expected_bytes
    );
    let bytes = verify_payload(output, &checkpoint.sums)?;
    let sums = decode_f32_le(&bytes, value_count)?;
    Ok(WorkerState::Active(ActiveState {
        generation: checkpoint.generation,
        next_record: checkpoint.next_record,
        used_prompts: checkpoint.used_prompts,
        truncated_prompts: checkpoint.truncated_prompts,
        skipped_prompts: checkpoint.skipped_prompts,
        forward_seconds: checkpoint.forward_seconds,
        vjp_seconds: checkpoint.vjp_seconds,
        sums,
        sums_path: Some(checkpoint.sums.path),
        diagnostics: checkpoint.diagnostics,
        token_dim_batch_timings: Vec::new(),
    }))
}

fn checkpoint_active(
    output: &Path,
    config_blake3: &str,
    state: &mut ActiveState,
    n_sources: usize,
    n_rows: usize,
    hidden_size: usize,
) -> Result<()> {
    state.generation = state
        .generation
        .checked_add(1)
        .context("checkpoint generation overflow")?;
    let sums_name = format!("sums-{:08}.f32le", state.generation);
    let bytes = encode_f32_le(&state.sums)?;
    publish_immutable(&output.join(&sums_name), &bytes)?;
    let sums = payload_descriptor(&sums_name, &bytes, [n_sources, n_rows, hidden_size]);
    let checkpoint = FitCheckpoint {
        schema: CHECKPOINT_SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        config_blake3: config_blake3.into(),
        generation: state.generation,
        next_record: state.next_record,
        used_prompts: state.used_prompts,
        truncated_prompts: state.truncated_prompts,
        skipped_prompts: state.skipped_prompts.clone(),
        forward_seconds: state.forward_seconds,
        vjp_seconds: state.vjp_seconds,
        sums,
        diagnostics: state.diagnostics.clone(),
    };
    write_atomic_replace(
        &output.join(CHECKPOINT_NAME),
        &serialize_json_pretty_bounded(&checkpoint, "row checkpoint")?,
    )?;
    if let Some(previous) = state.sums_path.replace(sums_name) {
        let previous_path = output.join(previous);
        if previous_path.exists() {
            std::fs::remove_file(&previous_path)
                .with_context(|| format!("remove prior sums {}", previous_path.display()))?;
            sync_directory(output)?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn open_or_create_token_state(
    output: &Path,
    resume: bool,
    expected_config: &TokenFitConfig,
    prompts: &[PreparedPrompt],
    config_blake3: &str,
    value_count: usize,
    expected_shape: [usize; 3],
    arch: Arch,
) -> Result<TokenWorkerState> {
    validate_output_leaf(output)?;
    let (bounded_value_count, bounded_shape) =
        token_artifact_layout(expected_shape[0], expected_shape[1], expected_shape[2])?;
    ensure!(
        bounded_value_count == value_count && bounded_shape == expected_shape,
        "token readout state shape and value count are inconsistent"
    );
    if output.exists() {
        let metadata = std::fs::symlink_metadata(output)
            .with_context(|| format!("inspect output {}", output.display()))?;
        ensure!(
            metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
            "output {} must be a real directory",
            output.display()
        );
        ensure!(
            resume,
            "output {} already exists; pass --resume",
            output.display()
        );
    } else {
        let parent = output
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        ensure!(
            parent.is_dir(),
            "output parent {} does not exist",
            parent.display()
        );
        DirBuilder::new()
            .mode(0o700)
            .create(output)
            .with_context(|| format!("create output directory {}", output.display()))?;
        sync_directory(parent)?;
    }

    let manifest_path = output.join(TOKEN_MANIFEST_NAME);
    if manifest_path.exists() {
        let manifest: TokenReadoutManifest = read_json_file(&manifest_path)?;
        validate_token_complete_manifest(
            &manifest,
            expected_config,
            prompts,
            config_blake3,
            expected_shape,
        )?;
        let bytes = verify_payload(output, &manifest.payload)?;
        decode_f32_le(&bytes, value_count)?;
        return Ok(TokenWorkerState::Complete(Box::new(manifest)));
    }
    ensure!(
        !output.join(MANIFEST_NAME).exists(),
        "output contains a completed row-shard artifact, not token readouts"
    );

    let checkpoint_path = output.join(CHECKPOINT_NAME);
    if !checkpoint_path.exists() {
        return Ok(TokenWorkerState::Active(ActiveState {
            generation: 0,
            next_record: 0,
            used_prompts: 0,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 0.0,
            vjp_seconds: 0.0,
            sums: try_zeroed_f32(value_count, "token readout accumulator")?,
            sums_path: None,
            diagnostics: Vec::new(),
            token_dim_batch_timings: Vec::new(),
        }));
    }
    let checkpoint: TokenFitCheckpoint = read_json_file(&checkpoint_path)?;
    ensure!(
        checkpoint.schema == TOKEN_CHECKPOINT_SCHEMA,
        "unknown token readout checkpoint schema"
    );
    ensure!(
        checkpoint.schema_version == SCHEMA_VERSION,
        "unsupported token readout checkpoint version"
    );
    ensure!(
        checkpoint.config_blake3 == config_blake3,
        "token readout checkpoint config {} != requested {}",
        checkpoint.config_blake3,
        config_blake3
    );
    ensure!(
        checkpoint.sums.shape == expected_shape,
        "token readout checkpoint sums shape {:?} != expected {:?}",
        checkpoint.sums.shape,
        expected_shape
    );
    ensure!(
        checkpoint.sums.dtype == "f32_le",
        "token readout checkpoint sums dtype must be f32_le"
    );
    ensure!(
        checkpoint.sums.path == format!("token-sums-{:08}.f32le", checkpoint.generation),
        "token readout checkpoint sums path does not match generation {}",
        checkpoint.generation
    );
    let expected_bytes = value_count
        .checked_mul(4)
        .and_then(|bytes| u64::try_from(bytes).ok())
        .context("token readout checkpoint byte count overflow")?;
    ensure!(
        checkpoint.sums.byte_length == expected_bytes,
        "token readout checkpoint sums byte length {} != expected {}",
        checkpoint.sums.byte_length,
        expected_bytes
    );
    let bytes = verify_payload(output, &checkpoint.sums)?;
    let sums = decode_f32_le(&bytes, value_count)?;
    let state = ActiveState {
        generation: checkpoint.generation,
        next_record: checkpoint.next_record,
        used_prompts: checkpoint.used_prompts,
        truncated_prompts: checkpoint.truncated_prompts,
        skipped_prompts: checkpoint.skipped_prompts,
        forward_seconds: checkpoint.forward_seconds,
        vjp_seconds: checkpoint.vjp_seconds,
        sums,
        sums_path: Some(checkpoint.sums.path),
        diagnostics: checkpoint.diagnostics,
        token_dim_batch_timings: checkpoint.dim_batch_timings,
    };
    validate_active_state(
        &state,
        prompts,
        expected_config.skip_first,
        arch,
        expected_config.target_layer,
        &expected_config.source_layers,
        value_count,
    )?;
    validate_dim_batch_timings(
        &state.token_dim_batch_timings,
        state.used_prompts,
        state.vjp_seconds,
    )?;
    Ok(TokenWorkerState::Active(state))
}

fn checkpoint_token_active(
    output: &Path,
    config_blake3: &str,
    state: &mut ActiveState,
    shape: [usize; 3],
) -> Result<()> {
    state.vjp_seconds = canonical_vjp_seconds(&state.token_dim_batch_timings)?;
    state.generation = state
        .generation
        .checked_add(1)
        .context("token readout checkpoint generation overflow")?;
    ensure!(
        state.generation == u64::try_from(state.next_record).context("token checkpoint cursor")?,
        "token readout checkpoint generation does not match its cursor"
    );
    ensure!(
        state.sums.iter().all(|value| value.is_finite()),
        "token readout checkpoint accumulator contains non-finite values"
    );
    validate_dim_batch_timings(
        &state.token_dim_batch_timings,
        state.used_prompts,
        state.vjp_seconds,
    )?;
    let sums_name = format!("token-sums-{:08}.f32le", state.generation);
    let bytes = encode_f32_le_fallible(&state.sums)?;
    publish_immutable(&output.join(&sums_name), &bytes)?;
    let sums = payload_descriptor(&sums_name, &bytes, shape);
    let checkpoint = TokenFitCheckpoint {
        schema: TOKEN_CHECKPOINT_SCHEMA.into(),
        schema_version: SCHEMA_VERSION,
        config_blake3: config_blake3.into(),
        generation: state.generation,
        next_record: state.next_record,
        used_prompts: state.used_prompts,
        truncated_prompts: state.truncated_prompts,
        skipped_prompts: state.skipped_prompts.clone(),
        forward_seconds: state.forward_seconds,
        vjp_seconds: state.vjp_seconds,
        sums,
        diagnostics: state.diagnostics.clone(),
        dim_batch_timings: state.token_dim_batch_timings.clone(),
    };
    write_atomic_replace(
        &output.join(CHECKPOINT_NAME),
        &serialize_json_pretty_bounded(&checkpoint, "token readout checkpoint")?,
    )?;
    if let Some(previous) = state.sums_path.replace(sums_name) {
        let previous_path = output.join(previous);
        if previous_path.exists() {
            std::fs::remove_file(&previous_path)
                .with_context(|| format!("remove prior token sums {}", previous_path.display()))?;
            sync_directory(output)?;
        }
    }
    Ok(())
}

fn merge_diagnostics(
    aggregate: &mut Vec<ReplayDiagnostic>,
    current: &[WorkspaceLensReplayDiagnostic],
) -> Result<()> {
    if aggregate.is_empty() {
        aggregate.extend(current.iter().map(|diagnostic| ReplayDiagnostic {
            layer: diagnostic.layer,
            kind: block_kind(diagnostic.kind).into(),
            residual_replay_max_abs_error: diagnostic.residual_replay_max_abs_error,
        }));
        return Ok(());
    }
    ensure!(
        aggregate.len() == current.len(),
        "replay diagnostic schedule length changed"
    );
    for (aggregate, current) in aggregate.iter_mut().zip(current) {
        ensure!(
            aggregate.layer == current.layer && aggregate.kind == block_kind(current.kind),
            "replay diagnostic schedule changed at layer {}",
            current.layer
        );
        aggregate.residual_replay_max_abs_error = aggregate
            .residual_replay_max_abs_error
            .max(current.residual_replay_max_abs_error);
    }
    Ok(())
}

fn block_kind(kind: WorkspaceLensBlockKind) -> &'static str {
    match kind {
        WorkspaceLensBlockKind::Gdn => "gdn",
        WorkspaceLensBlockKind::Attention => "attention",
    }
}

fn payload_descriptor(path: &str, bytes: &[u8], shape: [usize; 3]) -> PayloadDescriptor {
    PayloadDescriptor {
        path: path.into(),
        dtype: "f32_le".into(),
        shape,
        byte_length: bytes.len() as u64,
        blake3: blake3::hash(bytes).to_hex().to_string(),
    }
}

fn verify_payload(directory: &Path, descriptor: &PayloadDescriptor) -> Result<Vec<u8>> {
    ensure!(
        Path::new(&descriptor.path).components().count() == 1,
        "payload path must be one relative filename"
    );
    let path = directory.join(&descriptor.path);
    let expected_length = usize::try_from(descriptor.byte_length)
        .context("payload byte length does not fit this platform")?;
    let bytes = read_regular_file_exact(&path, expected_length)?;
    ensure!(
        blake3::hash(&bytes).to_hex().as_str() == descriptor.blake3,
        "payload {} digest mismatch",
        path.display()
    );
    Ok(bytes)
}

fn encode_f32_le(values: &[f32]) -> Result<Vec<u8>> {
    let byte_count = values
        .len()
        .checked_mul(4)
        .context("F32 byte count overflow")?;
    let mut bytes = Vec::with_capacity(byte_count);
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn encode_f32_le_fallible(values: &[f32]) -> Result<Vec<u8>> {
    let byte_count = values
        .len()
        .checked_mul(4)
        .context("F32 byte count overflow")?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(byte_count)
        .context("allocate token F32 payload encoding")?;
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    Ok(bytes)
}

fn decode_f32_le(bytes: &[u8], expected_values: usize) -> Result<Vec<f32>> {
    ensure!(
        bytes.len()
            == expected_values
                .checked_mul(4)
                .context("F32 byte count overflow")?,
        "F32 payload length {} != expected {}",
        bytes.len(),
        expected_values * 4
    );
    let mut values = Vec::new();
    values
        .try_reserve_exact(expected_values)
        .context("allocate decoded F32 payload")?;
    for chunk in bytes.chunks_exact(4) {
        values.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "F32 payload contains non-finite values"
    );
    Ok(values)
}

fn read_json_file<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T> {
    serde_json::from_slice(&read_regular_file_bounded(path, JSON_FILE_MAX_BYTES)?)
        .with_context(|| format!("parse JSON {}", path.display()))
}

fn serialize_json_pretty_bounded(value: &impl Serialize, name: &str) -> Result<Vec<u8>> {
    let bytes = serde_json::to_vec_pretty(value).with_context(|| format!("serialize {name}"))?;
    ensure!(
        bytes.len() <= JSON_FILE_MAX_BYTES,
        "serialized {name} is {} bytes; artifact JSON limit is {}",
        bytes.len(),
        JSON_FILE_MAX_BYTES
    );
    Ok(bytes)
}

fn open_regular_file(path: &Path) -> Result<(File, usize)> {
    let lexical_metadata =
        std::fs::symlink_metadata(path).with_context(|| format!("inspect {}", path.display()))?;
    ensure!(
        lexical_metadata.file_type().is_file() && !lexical_metadata.file_type().is_symlink(),
        "{} must be a regular non-symlink file",
        path.display()
    );
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("open {}", path.display()))?;
    let metadata = file
        .metadata()
        .with_context(|| format!("inspect opened {}", path.display()))?;
    ensure!(
        metadata.file_type().is_file(),
        "{} must remain a regular file after open",
        path.display()
    );
    let length = usize::try_from(metadata.len())
        .with_context(|| format!("{} length does not fit this platform", path.display()))?;
    Ok((file, length))
}

fn read_opened_file_exact(mut file: File, path: &Path, expected_length: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected_length)
        .with_context(|| format!("allocate {} bytes for {}", expected_length, path.display()))?;
    bytes.resize(expected_length, 0);
    file.read_exact(&mut bytes)
        .with_context(|| format!("read exact contents of {}", path.display()))?;
    let mut extra = [0u8; 1];
    ensure!(
        file.read(&mut extra)
            .with_context(|| format!("check end of {}", path.display()))?
            == 0,
        "{} grew while it was being read",
        path.display()
    );
    Ok(bytes)
}

fn read_regular_file_exact(path: &Path, expected_length: usize) -> Result<Vec<u8>> {
    let (file, length) = open_regular_file(path)?;
    ensure!(
        length == expected_length,
        "{} length {} != expected {}",
        path.display(),
        length,
        expected_length
    );
    read_opened_file_exact(file, path, expected_length)
}

fn read_regular_file_bounded(path: &Path, maximum_length: usize) -> Result<Vec<u8>> {
    let (file, length) = open_regular_file(path)?;
    ensure!(
        length <= maximum_length,
        "{} length {} exceeds limit {}",
        path.display(),
        length,
        maximum_length
    );
    read_opened_file_exact(file, path, length)
}

fn resolve_output_path(output: &Path) -> Result<PathBuf> {
    let leaf = output
        .file_name()
        .filter(|leaf| !leaf.is_empty() && *leaf != "." && *leaf != "..")
        .context("output path must name a non-root directory leaf")?;
    let parent = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let canonical_parent = std::fs::canonicalize(parent)
        .with_context(|| format!("resolve existing output parent {}", parent.display()))?;
    let metadata = std::fs::symlink_metadata(&canonical_parent)
        .with_context(|| format!("inspect output parent {}", canonical_parent.display()))?;
    ensure!(
        metadata.file_type().is_dir() && !metadata.file_type().is_symlink(),
        "resolved output parent {} must be a real directory",
        canonical_parent.display()
    );
    let resolved = canonical_parent.join(leaf);
    validate_output_leaf(&resolved)?;
    Ok(resolved)
}

fn resolve_output_file_path(output: &Path) -> Result<PathBuf> {
    let resolved = resolve_output_path(output)?;
    if let Ok(metadata) = std::fs::symlink_metadata(&resolved) {
        ensure!(
            metadata.file_type().is_file() && !metadata.file_type().is_symlink(),
            "output {} must be a regular non-symlink file",
            resolved.display()
        );
    }
    Ok(resolved)
}

fn validate_output_leaf(output: &Path) -> Result<()> {
    ensure!(
        output.file_name().is_some() && output.parent().is_some(),
        "output path must name a non-root directory leaf"
    );
    match std::fs::symlink_metadata(output) {
        Ok(metadata) => ensure!(
            !metadata.file_type().is_symlink(),
            "output leaf {} must not be a symlink",
            output.display()
        ),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => {
            return Err(error).with_context(|| format!("inspect output leaf {}", output.display()));
        }
    }
    Ok(())
}

fn publish_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        let existing = read_regular_file_exact(path, bytes.len())?;
        ensure!(
            existing == bytes,
            "existing {} conflicts with publication",
            path.display()
        );
        return Ok(());
    }
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let name = path
        .file_name()
        .context("immutable output has no filename")?;
    let staging = parent.join(format!(
        ".{}.stage.{}.{}",
        name.to_string_lossy(),
        std::process::id(),
        nonce
    ));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(&staging)
        .with_context(|| format!("create staging file {}", staging.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write staging file {}", staging.display()))?;
    file.sync_all()
        .with_context(|| format!("sync staging file {}", staging.display()))?;
    drop(file);
    match std::fs::hard_link(&staging, path) {
        Ok(()) => {}
        Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
            let existing = read_regular_file_exact(path, bytes.len())?;
            if existing != bytes {
                let _ = std::fs::remove_file(&staging);
                bail!("existing {} conflicts with publication", path.display());
            }
        }
        Err(error) => {
            let _ = std::fs::remove_file(&staging);
            return Err(error).with_context(|| {
                format!(
                    "publish staging file {} to {}",
                    staging.display(),
                    path.display()
                )
            });
        }
    }
    std::fs::remove_file(&staging)
        .with_context(|| format!("remove staging file {}", staging.display()))?;
    sync_directory(parent)
}

fn write_atomic_replace(path: &Path, bytes: &[u8]) -> Result<()> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let name = path.file_name().context("atomic output has no filename")?;
    let temporary = parent.join(format!(
        ".{}.tmp.{}.{}",
        name.to_string_lossy(),
        std::process::id(),
        nonce
    ));
    let publish_result = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(&temporary)
            .with_context(|| format!("create staging file {}", temporary.display()))?;
        file.write_all(bytes)
            .with_context(|| format!("write staging file {}", temporary.display()))?;
        file.sync_all()
            .with_context(|| format!("sync staging file {}", temporary.display()))?;
        drop(file);
        std::fs::rename(&temporary, path).with_context(|| {
            format!(
                "publish staging file {} to {}",
                temporary.display(),
                path.display()
            )
        })
    })();
    if let Err(error) = publish_result {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    sync_directory(parent)
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .with_context(|| format!("open directory {} for sync", path.display()))?
        .sync_all()
        .with_context(|| format!("sync directory {}", path.display()))
}

fn hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").unwrap();
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_summary_keeps_the_historical_identity_wire_key() {
        let summary = ModelSummary {
            path: "model.gguf".into(),
            content_blake3: "11".repeat(32),
            content_identity_outcome: "computed".into(),
            content_bytes_hashed: 42,
            workspace_lens_identity_scheme: "qwen_llm_model_locator_v1".into(),
            model_locator_id: "22".repeat(8),
            tokenizer_metadata_id: "33".repeat(8),
            content_authenticated: true,
            architecture: "qwen35".into(),
            n_layers: 64,
            hidden_size: 5_120,
            vocab_size: 248_320,
            full_attention_interval: 4,
        };
        let encoded = serde_json::to_value(&summary).unwrap();
        assert_eq!(
            encoded["research_identity_scheme"],
            "qwen_llm_model_locator_v1"
        );
        assert!(encoded.get("workspace_lens_identity_scheme").is_none());

        let mut aliased = encoded;
        let object = aliased.as_object_mut().unwrap();
        let scheme = object.remove("research_identity_scheme").unwrap();
        object.insert("workspace_lens_identity_scheme".into(), scheme);
        let decoded: ModelSummary = serde_json::from_value(aliased).unwrap();
        assert_eq!(
            decoded.workspace_lens_identity_scheme,
            "qwen_llm_model_locator_v1"
        );
    }

    fn test_config() -> FitConfig {
        FitConfig {
            estimator_version: ESTIMATOR_VERSION.into(),
            orientation: ORIENTATION.into(),
            rule_version: RULE_VERSION.into(),
            method: FitMethod::R,
            model_content_blake3: "11".repeat(32),
            model_locator_id: "22".repeat(8),
            tokenizer_metadata_id: "33".repeat(8),
            architecture: "test_dense".into(),
            n_layers: 4,
            hidden_size: 2,
            vocab_size: 8,
            full_attention_interval: 4,
            target_layer: 3,
            source_layers: vec![0],
            row_start: 0,
            row_end: 2,
            skip_first: 0,
            max_tokens: 2,
            add_special_tokens: true,
            corpus_blake3: "44".repeat(32),
            selected_records: 2,
        }
    }

    fn test_fit_args() -> FitRowsArgs {
        FitRowsArgs {
            model: "model.gguf".into(),
            prompts: "prompts.jsonl".into(),
            output: "rows".into(),
            identity_cache: "identity-cache".into(),
            method: FitMethod::R,
            target_layer: 3,
            source_layers: vec![0],
            row_start: 0,
            row_end: 2,
            dim_batch: 2,
            skip_first: 0,
            max_tokens: 2,
            max_prompts: 1,
            records_this_run: None,
            no_special_tokens: false,
            resume: false,
        }
    }

    fn test_token_args() -> FitTokensArgs {
        FitTokensArgs {
            model: "model.gguf".into(),
            prompts: "prompts.jsonl".into(),
            output: "readouts".into(),
            identity_cache: "identity-cache".into(),
            method: FitMethod::R,
            target_layer: 3,
            source_layers: vec![0],
            token_ids: vec![7, 2],
            dim_batch: 2,
            skip_first: 0,
            max_tokens: 2,
            max_prompts: 1,
            no_special_tokens: false,
            resume: false,
        }
    }

    fn test_token_config(args: &FitTokensArgs) -> TokenFitConfig {
        let covectors = [1.0, 2.0, 3.0, 4.0];
        TokenFitConfig {
            estimator_version: ESTIMATOR_VERSION.into(),
            orientation: TOKEN_ORIENTATION.into(),
            rule_version: RULE_VERSION.into(),
            method: args.method,
            model_content_blake3: "11".repeat(32),
            model_locator_id: "22".repeat(8),
            tokenizer_metadata_id: "33".repeat(8),
            architecture: "test_dense".into(),
            n_layers: 4,
            hidden_size: 2,
            vocab_size: 8,
            full_attention_interval: 4,
            target_layer: args.target_layer,
            source_layers: args.source_layers.clone(),
            readouts: TokenReadoutSpec {
                readout_version: TOKEN_READOUT_VERSION.into(),
                score_semantics: TOKEN_SCORE_SEMANTICS.into(),
                token_ids: args.token_ids.clone(),
                target_covectors_blake3: token_covector_digest(&args.token_ids, 2, &covectors)
                    .unwrap(),
                target_covectors_dtype: "f32_le".into(),
                target_covectors_shape: [args.token_ids.len(), 2],
                lm_head_dtype: "Q6_K".into(),
                lm_head_shape: [2, 8],
                output_norm_dtype: "F32".into(),
                output_norm_shape: vec![2],
            },
            skip_first: args.skip_first,
            max_tokens: args.max_tokens,
            add_special_tokens: !args.no_special_tokens,
            corpus_blake3: "44".repeat(32),
            selected_records: 1,
            build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
        }
    }

    fn test_prompt() -> PreparedPrompt {
        PreparedPrompt {
            id: "valid".into(),
            token_ids: vec![1, 2],
            original_token_count: 2,
            truncated: false,
        }
    }

    fn test_diagnostics() -> Vec<ReplayDiagnostic> {
        vec![
            ReplayDiagnostic {
                layer: 3,
                kind: "attention".into(),
                residual_replay_max_abs_error: 0.1,
            },
            ReplayDiagnostic {
                layer: 2,
                kind: "gdn".into(),
                residual_replay_max_abs_error: 0.0,
            },
            ReplayDiagnostic {
                layer: 1,
                kind: "gdn".into(),
                residual_replay_max_abs_error: 0.0,
            },
        ]
    }

    fn test_directory(label: &str) -> PathBuf {
        std::fs::canonicalize(std::env::temp_dir())
            .unwrap()
            .join(format!(
                "qwen-lens-{label}-{}-{}",
                std::process::id(),
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos()
            ))
    }

    #[test]
    fn fit_args_bound_native_query_batch() {
        let mut args = test_fit_args();
        validate_args(&args).unwrap();
        args.dim_batch = MAX_WORKSPACE_LENS_DIM_BATCH + 1;
        assert!(validate_args(&args).is_err());
        args.dim_batch = 1;
        args.max_prompts = MAX_PROMPT_RECORDS + 1;
        assert!(validate_args(&args).is_err());
        args = test_fit_args();
        args.records_this_run = Some(1);
        assert!(validate_args(&args).is_err());
    }

    #[test]
    fn token_build_identity_validation_fails_closed() {
        let valid = format!("git-source-sha256-v2:{}", "a5".repeat(32));
        validate_token_build_identity(&valid, "none").unwrap();
        validate_token_build_identity(
            env!("QWEN_BUILD_SOURCE_STATE"),
            env!("QWEN_BUILD_STAMP_ERROR"),
        )
        .unwrap();
        assert!(validate_token_build_identity("unknown", "none").is_err());
        assert!(
            validate_token_build_identity(
                &format!("git-source-sha256-v2:{}", "A5".repeat(32)),
                "none"
            )
            .is_err()
        );
        assert!(validate_token_build_identity("git-source-sha256-v2:abcd", "none").is_err());
        assert!(validate_token_build_identity(&valid, "unknown").is_err());
        assert!(validate_token_build_identity(&valid, "git_identity_unavailable").is_err());
    }

    #[test]
    fn token_arguments_validate_uniqueness_and_preserve_caller_order() {
        let parsed = Cli::try_parse_from([
            "qwen-lens",
            "fit-tokens",
            "--model",
            "model.gguf",
            "--prompts",
            "prompts.jsonl",
            "--output",
            "out",
            "--identity-cache",
            "cache",
            "--method",
            "r",
            "--target-layer",
            "3",
            "--source-layers",
            "0,2",
            "--token-ids",
            "7,2,5",
            "--skip-first",
            "0",
            "--max-tokens",
            "2",
        ])
        .unwrap();
        let Command::FitTokens(parsed) = parsed.command else {
            panic!("expected fit-tokens command");
        };
        assert_eq!(parsed.token_ids, [7, 2, 5]);
        validate_token_args(&parsed).unwrap();

        let mut invalid = test_token_args();
        invalid.token_ids.clear();
        assert!(validate_token_args(&invalid).is_err());
        invalid.token_ids = vec![2, 2];
        assert!(validate_token_args(&invalid).is_err());
        invalid.token_ids = vec![0; TOKEN_ID_ARGUMENT_MAX_COUNT + 1];
        assert!(validate_token_args(&invalid).is_err());
        invalid.token_ids = vec![0];
        invalid.max_prompts = MAX_PROMPT_RECORDS + 1;
        assert!(validate_token_args(&invalid).is_err());
        validate_prompt_id(&"x".repeat(MAX_PROMPT_ID_BYTES), 1).unwrap();
        assert!(validate_prompt_id(&"x".repeat(MAX_PROMPT_ID_BYTES + 1), 1).is_err());
    }

    #[test]
    fn trace_full_cli_accepts_each_exact_input_form_and_rejects_mixing() {
        let prompt = Cli::try_parse_from([
            "qwen-lens",
            "trace-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--prompt",
            "--help",
            "--no-special-tokens",
            "--layers",
            "62,0",
        ])
        .unwrap();
        assert!(matches!(prompt.command, Command::TraceFull(_)));

        let token_ids = Cli::try_parse_from([
            "qwen-lens",
            "trace-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--token-ids",
            "1,2,3",
            "--top-k",
            "16",
        ])
        .unwrap();
        assert!(matches!(token_ids.command, Command::TraceFull(_)));

        let messages = Cli::try_parse_from([
            "qwen-lens",
            "trace-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--messages",
            "messages.json",
            "--message-mode",
            "no-thinking",
        ])
        .unwrap();
        assert!(matches!(messages.command, Command::TraceFull(_)));

        let responses = Cli::try_parse_from([
            "qwen-lens",
            "trace-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--open-responses",
            "request.json",
        ])
        .unwrap();
        assert!(matches!(responses.command, Command::TraceFull(_)));

        let user = Cli::try_parse_from([
            "qwen-lens",
            "trace-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--system",
            "policy",
            "--user",
            "request",
            "--message-mode",
            "xhigh",
        ])
        .unwrap();
        assert!(matches!(user.command, Command::TraceFull(_)));

        assert!(
            Cli::try_parse_from([
                "qwen-lens",
                "trace-full",
                "--model",
                "model.gguf",
                "--full-lens",
                "full-lens",
                "--prompt",
                "hello",
                "--message-mode",
                "thinking",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "qwen-lens",
                "trace-full",
                "--model",
                "model.gguf",
                "--full-lens",
                "full-lens",
                "--open-responses",
                "request.json",
                "--messages",
                "messages.json",
            ])
            .is_err()
        );

        assert!(
            Cli::try_parse_from([
                "qwen-lens",
                "trace-full",
                "--model",
                "model.gguf",
                "--full-lens",
                "full-lens",
                "--prompt",
                "hello",
                "--messages",
                "messages.json",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "qwen-lens",
                "trace-full",
                "--model",
                "model.gguf",
                "--full-lens",
                "full-lens",
                "--token-ids",
                "1,2",
                "--no-special-tokens",
            ])
            .is_err()
        );
    }

    #[test]
    fn full_readout_cli_requires_exactly_one_input_form() {
        let parsed = Cli::try_parse_from([
            "qwen-lens",
            "read-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--prompt",
            "The capital of France is",
            "--layers",
            "0,31,62",
            "--identity-cache",
            "identity-cache",
            "--allow-unvalidated-transfer",
            "--include-vector",
        ])
        .unwrap();
        assert!(matches!(parsed.command, Command::ReadFull(ref args) if args.include_vector));

        let hyphen_prompt = Cli::try_parse_from([
            "qwen-lens",
            "read-full",
            "--model",
            "model.gguf",
            "--full-lens",
            "full-lens",
            "--prompt",
            "--help",
            "--identity-cache",
            "identity-cache",
        ])
        .unwrap();
        assert!(matches!(hyphen_prompt.command, Command::ReadFull(_)));

        assert!(
            Cli::try_parse_from([
                "qwen-lens",
                "read-full",
                "--model",
                "model.gguf",
                "--full-lens",
                "full-lens",
                "--prompt",
                "hello",
                "--token-ids",
                "1,2",
                "--identity-cache",
                "identity-cache",
            ])
            .is_err()
        );
        assert!(
            Cli::try_parse_from([
                "qwen-lens",
                "read-full",
                "--model",
                "model.gguf",
                "--full-lens",
                "full-lens",
                "--identity-cache",
                "identity-cache",
            ])
            .is_err()
        );
    }

    #[test]
    fn token_covector_digest_binds_exact_ordered_f32_bits() {
        let values = [1.0, -0.0, 2.0, 3.0];
        let base = token_covector_digest(&[7, 2], 2, &values).unwrap();
        assert_eq!(base, token_covector_digest(&[7, 2], 2, &values).unwrap());
        assert_ne!(base, token_covector_digest(&[2, 7], 2, &values).unwrap());
        assert_ne!(base, token_covector_digest(&[7], 4, &values).unwrap());
        assert_ne!(
            base,
            token_covector_digest(&[7, 2], 2, &[1.0, 0.0, 2.0, 3.0]).unwrap()
        );
        assert!(token_covector_digest(&[7, 2], 3, &values).is_err());
    }

    #[test]
    fn token_config_digest_excludes_runtime_dim_batch_but_binds_token_order() {
        let args = test_token_args();
        let config = test_token_config(&args);
        let digest = digest_json(&config).unwrap();
        let mut runtime_changed = args;
        runtime_changed.dim_batch = MAX_WORKSPACE_LENS_DIM_BATCH;
        assert_eq!(
            digest,
            digest_json(&test_token_config(&runtime_changed)).unwrap()
        );
        runtime_changed.token_ids.reverse();
        assert_ne!(
            digest,
            digest_json(&test_token_config(&runtime_changed)).unwrap()
        );
        let mut implementation_changed = config;
        implementation_changed.build_source_state = "different-numerics".into();
        assert_ne!(digest, digest_json(&implementation_changed).unwrap());
    }

    #[test]
    fn token_artifact_preflight_enforces_128_mib_cap() {
        let (values, shape) = token_artifact_layout(63, 104, 5120).unwrap();
        assert_eq!(shape, [63, 104, 5120]);
        assert_eq!(values * 4, 134_184_960);
        assert!(token_artifact_layout(63, 105, 5120).is_err());
    }

    #[test]
    fn dim_batch_timing_validation_rejects_duplicates_ranges_and_bad_counts() {
        let valid = [
            DimBatchTiming {
                dim_batch: 2,
                prompt_count: 1,
                vjp_seconds: 1.0,
            },
            DimBatchTiming {
                dim_batch: 4,
                prompt_count: 2,
                vjp_seconds: 2.0,
            },
        ];
        validate_dim_batch_timings(&valid, 3, 3.0).unwrap();
        let mut invalid = valid.to_vec();
        invalid[1].dim_batch = 2;
        assert!(validate_dim_batch_timings(&invalid, 3, 3.0).is_err());
        let mut invalid = valid.to_vec();
        invalid[1].dim_batch = MAX_WORKSPACE_LENS_DIM_BATCH + 1;
        assert!(validate_dim_batch_timings(&invalid, 3, 3.0).is_err());
        assert!(validate_dim_batch_timings(&valid, 4, 3.0).is_err());
    }

    #[test]
    fn alternating_dim_batch_samples_have_stable_canonical_total() {
        let mut timings = Vec::new();
        let mut total = 0.0;
        for index in 0..10_000 {
            let dim_batch = if index % 2 == 0 { 2 } else { 7 };
            let seconds = if index % 3 == 0 { 0.1 } else { 0.000_001 };
            record_dim_batch_timing(&mut timings, dim_batch, seconds).unwrap();
            total = canonical_vjp_seconds(&timings).unwrap();
        }
        assert_eq!(timings[0].dim_batch, 2);
        assert_eq!(timings[1].dim_batch, 7);
        validate_dim_batch_timings(&timings, 10_000, total).unwrap();
        let noncanonical = f64::from_bits(total.to_bits() + 1);
        assert!(validate_dim_batch_timings(&timings, 10_000, noncanonical).is_err());
    }

    #[test]
    fn f32_payload_round_trips_and_rejects_non_finite_values() {
        let values = [1.25f32, -2.5, 0.0];
        let bytes = encode_f32_le(&values).unwrap();
        assert_eq!(decode_f32_le(&bytes, values.len()).unwrap(), values);
        assert!(decode_f32_le(&f32::NAN.to_le_bytes(), 1).is_err());
    }

    #[test]
    fn corpus_digest_binds_ids_boundaries_and_tokens() {
        let prompt = |id: &str, token_ids: &[i32]| PreparedPrompt {
            id: id.into(),
            token_ids: token_ids.to_vec(),
            original_token_count: token_ids.len(),
            truncated: false,
        };
        let base = corpus_digest(&[prompt("a", &[1, 2]), prompt("b", &[3])]);
        assert_ne!(
            base,
            corpus_digest(&[prompt("a", &[1]), prompt("b", &[2, 3])])
        );
        assert_ne!(
            base,
            corpus_digest(&[prompt("b", &[3]), prompt("a", &[1, 2])])
        );
        let truncated = PreparedPrompt {
            id: "a".into(),
            token_ids: vec![1, 2],
            original_token_count: 3,
            truncated: true,
        };
        assert_ne!(
            corpus_digest(&[prompt("a", &[1, 2])]),
            corpus_digest(&[truncated])
        );
    }

    #[test]
    fn diagnostic_merge_is_schedule_strict_and_takes_maximum() {
        let current = WorkspaceLensReplayDiagnostic {
            layer: 3,
            kind: WorkspaceLensBlockKind::Attention,
            residual_replay_max_abs_error: 0.2,
        };
        let mut aggregate = Vec::new();
        merge_diagnostics(&mut aggregate, std::slice::from_ref(&current)).unwrap();
        let mut lower = current;
        lower.residual_replay_max_abs_error = 0.1;
        merge_diagnostics(&mut aggregate, &[lower]).unwrap();
        assert_eq!(aggregate[0].residual_replay_max_abs_error, 0.2);
    }

    #[test]
    fn checkpoint_round_trip_restores_exact_sums_and_cursor() {
        let root = test_directory("checkpoint-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let mut active = ActiveState {
            generation: 1,
            next_record: 2,
            used_prompts: 1,
            truncated_prompts: 1,
            skipped_prompts: vec![SkippedPrompt {
                id: "short".into(),
                reason: "1 tokens leave no valid positions with skip_first=0".into(),
            }],
            forward_seconds: 1.25,
            vjp_seconds: 2.5,
            sums: vec![1.0, 2.0, 3.0, 4.0],
            sums_path: None,
            diagnostics: vec![
                ReplayDiagnostic {
                    layer: 3,
                    kind: "attention".into(),
                    residual_replay_max_abs_error: 0.1,
                },
                ReplayDiagnostic {
                    layer: 2,
                    kind: "gdn".into(),
                    residual_replay_max_abs_error: 0.0,
                },
                ReplayDiagnostic {
                    layer: 1,
                    kind: "gdn".into(),
                    residual_replay_max_abs_error: 0.0,
                },
            ],
            token_dim_batch_timings: Vec::new(),
        };
        let prompts = [
            PreparedPrompt {
                id: "valid".into(),
                token_ids: vec![1, 2],
                original_token_count: 3,
                truncated: true,
            },
            PreparedPrompt {
                id: "short".into(),
                token_ids: vec![1],
                original_token_count: 1,
                truncated: false,
            },
        ];
        let config = test_config();
        let config_digest = digest_json(&config).unwrap();
        checkpoint_active(&root, &config_digest, &mut active, 1, 2, 2).unwrap();
        let WorkerState::Active(restored) =
            open_or_create_state(&root, true, &config, &prompts, &config_digest, 4, 1, 2, 2)
                .unwrap()
        else {
            panic!("expected active checkpoint");
        };
        assert_eq!(restored.next_record, 2);
        assert_eq!(restored.used_prompts, 1);
        assert_eq!(restored.sums, [1.0, 2.0, 3.0, 4.0]);
        validate_active_state(
            &restored,
            &prompts,
            0,
            qwen_llm::model::QWEN3_0_8B,
            3,
            &[0],
            4,
        )
        .unwrap();
        let mut corrupted = restored;
        corrupted.next_record = 1;
        assert!(
            validate_active_state(
                &corrupted,
                &prompts,
                0,
                qwen_llm::model::QWEN3_0_8B,
                3,
                &[0],
                4,
            )
            .is_err()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn immutable_publication_survives_orphan_staging_and_is_idempotent() {
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-publication-test-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        std::fs::write(root.join(".rows.f32le.stage.crashed"), b"partial").unwrap();
        let final_path = root.join("rows.f32le");
        publish_immutable(&final_path, b"complete").unwrap();
        publish_immutable(&final_path, b"complete").unwrap();
        assert_eq!(
            read_regular_file_exact(&final_path, b"complete".len()).unwrap(),
            b"complete"
        );
        assert!(publish_immutable(&final_path, b"different").is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn token_checkpoint_round_trip_restores_exact_shape_values_and_cursor() {
        let root = test_directory("token-checkpoint-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let args = test_token_args();
        let config = test_token_config(&args);
        let config_digest = digest_json(&config).unwrap();
        let mut active = ActiveState {
            generation: 0,
            next_record: 1,
            used_prompts: 1,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 1.25,
            vjp_seconds: 2.5,
            sums: vec![1.0, 2.0, 3.0, 4.0],
            sums_path: None,
            diagnostics: test_diagnostics(),
            token_dim_batch_timings: vec![DimBatchTiming {
                dim_batch: 2,
                prompt_count: 1,
                vjp_seconds: 2.5,
            }],
        };
        checkpoint_token_active(&root, &config_digest, &mut active, [1, 2, 2]).unwrap();
        let prompts = [test_prompt()];
        let TokenWorkerState::Active(restored) = open_or_create_token_state(
            &root,
            true,
            &config,
            &prompts,
            &config_digest,
            4,
            [1, 2, 2],
            qwen_llm::model::QWEN3_0_8B,
        )
        .unwrap() else {
            panic!("expected active token checkpoint");
        };
        assert_eq!(restored.generation, 1);
        assert_eq!(restored.next_record, 1);
        assert_eq!(restored.sums, [1.0, 2.0, 3.0, 4.0]);
        assert!(
            open_or_create_token_state(
                &root,
                true,
                &config,
                &prompts,
                &config_digest,
                4,
                [2, 1, 2],
                qwen_llm::model::QWEN3_0_8B,
            )
            .is_err()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn token_resume_accepts_changed_dim_batch_and_preserves_grouped_timings() {
        let root = test_directory("token-changed-batch-resume-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let args = test_token_args();
        let mut config = test_token_config(&args);
        config.selected_records = 2;
        let config_digest = digest_json(&config).unwrap();
        let mut active = ActiveState {
            generation: 0,
            next_record: 1,
            used_prompts: 1,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 1.0,
            vjp_seconds: 1.0,
            sums: vec![1.0, 2.0, 3.0, 4.0],
            sums_path: None,
            diagnostics: test_diagnostics(),
            token_dim_batch_timings: vec![DimBatchTiming {
                dim_batch: 2,
                prompt_count: 1,
                vjp_seconds: 1.0,
            }],
        };
        checkpoint_token_active(&root, &config_digest, &mut active, [1, 2, 2]).unwrap();
        let prompts = [
            test_prompt(),
            PreparedPrompt {
                id: "second".into(),
                token_ids: vec![2, 3],
                original_token_count: 2,
                truncated: false,
            },
        ];
        let TokenWorkerState::Active(mut restored) = open_or_create_token_state(
            &root,
            true,
            &config,
            &prompts,
            &config_digest,
            4,
            [1, 2, 2],
            qwen_llm::model::QWEN3_0_8B,
        )
        .unwrap() else {
            panic!("expected active token checkpoint");
        };
        restored.next_record = 2;
        restored.used_prompts = 2;
        restored.forward_seconds += 1.0;
        record_dim_batch_timing(&mut restored.token_dim_batch_timings, 4, 2.0).unwrap();
        restored.vjp_seconds = canonical_vjp_seconds(&restored.token_dim_batch_timings).unwrap();
        checkpoint_token_active(&root, &config_digest, &mut restored, [1, 2, 2]).unwrap();
        let TokenWorkerState::Active(restored) = open_or_create_token_state(
            &root,
            true,
            &config,
            &prompts,
            &config_digest,
            4,
            [1, 2, 2],
            qwen_llm::model::QWEN3_0_8B,
        )
        .unwrap() else {
            panic!("expected changed-batch token checkpoint");
        };
        assert_eq!(
            restored.token_dim_batch_timings,
            [
                DimBatchTiming {
                    dim_batch: 2,
                    prompt_count: 1,
                    vjp_seconds: 1.0,
                },
                DimBatchTiming {
                    dim_batch: 4,
                    prompt_count: 1,
                    vjp_seconds: 2.0,
                },
            ]
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn token_resume_rejects_row_checkpoint_schema() {
        let root = test_directory("token-wrong-schema-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let args = test_token_args();
        let config = test_token_config(&args);
        let config_digest = digest_json(&config).unwrap();
        let mut active = ActiveState {
            generation: 0,
            next_record: 1,
            used_prompts: 1,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 1.0,
            vjp_seconds: 1.0,
            sums: vec![1.0, 2.0, 3.0, 4.0],
            sums_path: None,
            diagnostics: test_diagnostics(),
            token_dim_batch_timings: Vec::new(),
        };
        checkpoint_active(&root, &config_digest, &mut active, 1, 2, 2).unwrap();
        assert!(
            open_or_create_token_state(
                &root,
                true,
                &config,
                &[test_prompt()],
                &config_digest,
                4,
                [1, 2, 2],
                qwen_llm::model::QWEN3_0_8B,
            )
            .is_err()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn row_v1_resume_rejects_token_checkpoint_schema() {
        let root = test_directory("row-wrong-schema-regression-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let config = test_config();
        let config_digest = digest_json(&config).unwrap();
        let mut active = ActiveState {
            generation: 0,
            next_record: 1,
            used_prompts: 1,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 1.0,
            vjp_seconds: 1.0,
            sums: vec![1.0, 2.0, 3.0, 4.0],
            sums_path: None,
            diagnostics: test_diagnostics(),
            token_dim_batch_timings: vec![DimBatchTiming {
                dim_batch: 2,
                prompt_count: 1,
                vjp_seconds: 1.0,
            }],
        };
        checkpoint_token_active(&root, &config_digest, &mut active, [1, 2, 2]).unwrap();
        assert!(
            open_or_create_state(
                &root,
                true,
                &config,
                &[test_prompt()],
                &config_digest,
                4,
                1,
                2,
                2,
            )
            .is_err()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn token_resume_rejects_nonfinite_checkpoint_payload() {
        let root = test_directory("token-nonfinite-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let args = test_token_args();
        let config = test_token_config(&args);
        let config_digest = digest_json(&config).unwrap();
        let mut bytes = encode_f32_le(&[1.0, 2.0, 3.0]).unwrap();
        bytes.extend_from_slice(&f32::NAN.to_le_bytes());
        let sums_name = "token-sums-00000001.f32le";
        publish_immutable(&root.join(sums_name), &bytes).unwrap();
        let checkpoint = TokenFitCheckpoint {
            schema: TOKEN_CHECKPOINT_SCHEMA.into(),
            schema_version: SCHEMA_VERSION,
            config_blake3: config_digest.clone(),
            generation: 1,
            next_record: 1,
            used_prompts: 1,
            truncated_prompts: 0,
            skipped_prompts: Vec::new(),
            forward_seconds: 1.0,
            vjp_seconds: 1.0,
            sums: payload_descriptor(sums_name, &bytes, [1, 2, 2]),
            diagnostics: test_diagnostics(),
            dim_batch_timings: vec![DimBatchTiming {
                dim_batch: 2,
                prompt_count: 1,
                vjp_seconds: 1.0,
            }],
        };
        write_atomic_replace(
            &root.join(CHECKPOINT_NAME),
            &serde_json::to_vec_pretty(&checkpoint).unwrap(),
        )
        .unwrap();
        assert!(
            open_or_create_token_state(
                &root,
                true,
                &config,
                &[test_prompt()],
                &config_digest,
                4,
                [1, 2, 2],
                qwen_llm::model::QWEN3_0_8B,
            )
            .is_err()
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn token_payload_publication_is_immutable_and_crash_tolerant() {
        let root = test_directory("token-publication-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        std::fs::write(root.join(".readouts.f32le.stage.crashed"), b"partial").unwrap();
        let final_path = root.join(TOKEN_PAYLOAD_NAME);
        publish_immutable(&final_path, b"complete").unwrap();
        publish_immutable(&final_path, b"complete").unwrap();
        assert_eq!(
            read_regular_file_exact(&final_path, b"complete".len()).unwrap(),
            b"complete"
        );
        assert!(publish_immutable(&final_path, b"different").is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn output_path_resolution_canonicalizes_ancestors_and_rejects_trailing_symlink_leaf() {
        let root = test_directory("output-path-symlink-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let real = root.join("real");
        DirBuilder::new().mode(0o700).create(&real).unwrap();
        let ancestor_link = root.join("ancestor-link");
        std::os::unix::fs::symlink(&real, &ancestor_link).unwrap();
        assert_eq!(
            resolve_output_path(&ancestor_link.join("output")).unwrap(),
            std::fs::canonicalize(&real).unwrap().join("output")
        );

        let leaf_link = root.join("leaf-link");
        std::os::unix::fs::symlink(&real, &leaf_link).unwrap();
        let trailing_leaf = PathBuf::from(format!("{}/", leaf_link.display()));
        assert!(resolve_output_path(&trailing_leaf).is_err());
        assert!(resolve_output_path(Path::new("/")).is_err());
        assert!(resolve_output_path(Path::new("")).is_err());

        let directory_leaf = root.join("directory-output");
        DirBuilder::new()
            .mode(0o700)
            .create(&directory_leaf)
            .unwrap();
        assert!(resolve_output_file_path(&directory_leaf).is_err());
        assert_eq!(
            resolve_output_file_path(&root.join("result.json")).unwrap(),
            root.join("result.json")
        );
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn atomic_replace_removes_staging_file_when_rename_fails() {
        let root = test_directory("atomic-replace-cleanup-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let target = root.join("target");
        DirBuilder::new().mode(0o700).create(&target).unwrap();

        assert!(write_atomic_replace(&target, b"content").is_err());
        assert!(!std::fs::read_dir(&root).unwrap().any(|entry| {
            entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".target.tmp.")
        }));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn artifact_reads_reject_oversized_json_and_length_mismatch_before_reading() {
        let root = test_directory("bounded-read-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let oversized = root.join("oversized.json");
        File::create(&oversized)
            .unwrap()
            .set_len((JSON_FILE_MAX_BYTES + 1) as u64)
            .unwrap();
        assert!(read_json_file::<serde_json::Value>(&oversized).is_err());

        let payload = root.join("payload.f32le");
        std::fs::write(&payload, [0u8; 8]).unwrap();
        assert!(read_regular_file_exact(&payload, 4).is_err());
        let oversized_value = "x".repeat(JSON_FILE_MAX_BYTES);
        assert!(serialize_json_pretty_bounded(&oversized_value, "test JSON").is_err());
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn jsonl_record_limit_accepts_boundary_and_rejects_oversized_before_serde() {
        let root = test_directory("jsonl-record-limit-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let boundary_path = root.join("boundary.jsonl");
        let prefix = br#"{"token_ids":[1]}"#;
        let mut boundary = vec![b' '; MAX_PROMPT_RECORD_BYTES];
        boundary[..prefix.len()].copy_from_slice(prefix);
        boundary.push(b'\n');
        std::fs::write(&boundary_path, &boundary).unwrap();
        let requests = read_prompt_requests(&boundary_path, 1).unwrap();
        assert_eq!(requests.len(), 1);

        let oversized_path = root.join("oversized.jsonl");
        let oversized = vec![b'x'; MAX_PROMPT_RECORD_BYTES + 1];
        std::fs::write(&oversized_path, &oversized).unwrap();
        let error = read_prompt_requests(&oversized_path, 1).unwrap_err();
        assert!(error.to_string().contains("exceeds JSONL record limit"));
        assert!(!error.to_string().contains("parse"));
        std::fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn selected_jsonl_corpus_enforces_cumulative_byte_limit() {
        let root = test_directory("jsonl-cumulative-limit-test");
        DirBuilder::new().mode(0o700).create(&root).unwrap();
        let corpus_path = root.join("prompts.jsonl");
        let record = format!("{}{}", r#"{"token_ids":[1]}"#, " ".repeat(32));
        std::fs::write(&corpus_path, format!("{record}\n{record}\n")).unwrap();
        let exact_bytes = record.len().checked_mul(2).unwrap();
        assert_eq!(
            read_prompt_requests_with_corpus_limit(&corpus_path, 2, exact_bytes)
                .unwrap()
                .len(),
            2
        );
        let error =
            read_prompt_requests_with_corpus_limit(&corpus_path, 2, exact_bytes - 1).unwrap_err();
        assert!(error.to_string().contains("cumulative limit"));
        std::fs::remove_dir_all(&root).unwrap();
    }
}
