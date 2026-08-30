use crate::messages::{
    Qwen38GenerationMode, Qwen38ReasoningEffort, QwenGenerationMode, parse_strict_messages_input,
    render_qwen_messages_prompt_with_generation, render_qwen38_messages_prompt_with_generation,
    supports_qwen4exp_prompt_protocol,
};
use crate::template_lens::{TemplateLens, TemplateScore, TemplateVocabulary};
use anyhow::{Context, Result, bail, ensure};
use clap::{ArgGroup, Args, ValueEnum};
use objc2_metal::MTLBuffer;
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{MetalContext, MetalTensor, PostBlockIntervention};
use qwen_llm::model::ArchKind;
use qwen_llm::model_family::ModelFamily;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpFixedHyperAdd, Qwen4ExpLoadedModel, Qwen4ExpPostLayerHyperRequest,
    Qwen4ExpSessionCapacity, Qwen4ExpTextRunner,
};
use qwen_llm::runtime::{Runtime, SequenceConfig};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::Write;
use std::path::{Path, PathBuf};

const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;
const MAX_LENSES: usize = 64;
const MAX_DIRECTIONS: usize = 4096;
const MAX_OPERATIONS: usize = 4096;
const MAX_READOUTS: usize = 1024;
const MAX_SELECTOR_VALUES: usize = 4096;
const MAX_TOP_K: usize = 1024;
const MAX_NEW_TOKENS: usize = 4096;
const MAX_NATIVE_HYPER_CAPTURES: usize = 32;
const MAX_PUBLISHED_FULL_TOKEN_IDS: usize = 32;
const MAX_RUN_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;
const RUN_SCHEMA: &str = "qwen.lens.run";
const RUN_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("lens_input")
        .required(true)
        .multiple(false)
        .args(["prompt", "token_ids", "messages"])
))]
pub(crate) struct LensRunArgs {
    /// Ordinary Qwen, Qwen3.8-Flash-Next, or Muse Glimmer GGUF model.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,

    /// Strict Lens plan JSON file.
    #[arg(long)]
    pub(crate) plan: PathBuf,

    /// Private model-content identity cache (required for Muse Glimmer).
    #[arg(long)]
    pub(crate) identity_cache: Option<PathBuf>,

    /// Raw text prompt; tokenizer-configured specials are enabled by default.
    #[arg(long)]
    pub(crate) prompt: Option<String>,

    /// Literal comma-separated token IDs; no specials are added.
    #[arg(long, value_delimiter = ',')]
    pub(crate) token_ids: Option<Vec<i32>>,

    /// JSON message array or wrapper with a `messages` array.
    #[arg(long)]
    pub(crate) messages: Option<PathBuf>,

    /// Disable tokenizer-configured specials for --prompt.
    #[arg(long)]
    pub(crate) no_special_tokens: bool,

    /// Maximum number of generated tokens.
    #[arg(long, default_value_t = 32)]
    pub(crate) max_new_tokens: usize,

    /// Native sampler temperature; zero is deterministic greedy decoding.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) temperature: f32,

    /// Native sampler top-k; zero disables the filter.
    #[arg(long, default_value_t = 0)]
    pub(crate) top_k: usize,

    /// Native sampler nucleus threshold.
    #[arg(long, default_value_t = 1.0)]
    pub(crate) top_p: f32,

    /// Native sampler minimum probability threshold.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) min_p: f32,

    /// Native sampler seed.
    #[arg(long, default_value_t = 0)]
    pub(crate) seed: u64,

    /// Replace this JSON run artifact atomically after successful execution.
    #[arg(long)]
    pub(crate) output: Option<PathBuf>,

    /// Compact summary or the complete JSON run artifact on stdout.
    #[arg(long, value_enum)]
    pub(crate) format: Option<RunStdoutFormat>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum RunStdoutFormat {
    Summary,
    Json,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensPlan {
    pub(crate) version: u32,
    pub(crate) lenses: Vec<LensDefinition>,
    pub(crate) directions: Vec<DirectionDefinition>,
    pub(crate) operations: Vec<OperationDefinition>,
    #[serde(default)]
    pub(crate) readouts: Vec<ReadoutDefinition>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum LensDefinition {
    NativeSelected {
        id: String,
        artifact: PathBuf,
    },
    #[serde(rename = "published_full_transport", alias = "published_full_j")]
    PublishedFullTransport {
        id: String,
        artifact: PathBuf,
        token_ids: Vec<u32>,
        allow_unvalidated_transfer: bool,
    },
    WorkspaceTemplate {
        id: String,
        weights: PathBuf,
        labels: PathBuf,
    },
}

impl LensDefinition {
    fn id(&self) -> &str {
        match self {
            Self::NativeSelected { id, .. }
            | Self::PublishedFullTransport { id, .. }
            | Self::WorkspaceTemplate { id, .. } => id,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(untagged)]
pub(crate) enum DirectionDefinition {
    LensRow(LensRowDirectionDefinition),
    NativeHyper(NativeHyperDirectionDefinition),
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct LensRowDirectionDefinition {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) row: DirectionRow,
    pub(crate) normalization: Normalization,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct NativeHyperDirectionDefinition {
    pub(crate) id: String,
    pub(crate) source: NativeHyperDirectionSource,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum NativeHyperDirectionSource {
    NativeHyperF32 { path: PathBuf, layer: u32 },
}

impl DirectionDefinition {
    fn id(&self) -> &str {
        match self {
            Self::LensRow(direction) => &direction.id,
            Self::NativeHyper(direction) => &direction.id,
        }
    }

    fn lens_row(&self) -> Option<&LensRowDirectionDefinition> {
        match self {
            Self::LensRow(direction) => Some(direction),
            Self::NativeHyper(_) => None,
        }
    }

    fn native_hyper(&self) -> Option<&NativeHyperDirectionDefinition> {
        match self {
            Self::LensRow(_) => None,
            Self::NativeHyper(direction) => Some(direction),
        }
    }

    fn normalization(&self) -> Option<Normalization> {
        self.lens_row().map(|direction| direction.normalization)
    }
}

impl NativeHyperDirectionSource {
    fn path_and_layer(&self) -> (&Path, u32) {
        match self {
            Self::NativeHyperF32 { path, layer } => (path, *layer),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum DirectionRow {
    TokenId { token_id: i32 },
    TemplateRowId { template_row_id: usize },
    Label { label: String },
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Normalization {
    AsStored,
    UnitL2,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct OperationDefinition {
    pub(crate) id: String,
    pub(crate) scope: Scope,
    pub(crate) action: Action,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Action {
    FixedAdd {
        direction: String,
        coefficient: f32,
    },
    ResidualL2Fraction {
        direction: String,
        coefficient: f32,
    },
    ProjectionAblate {
        direction: String,
        coefficient: f32,
    },
    SourceToTarget {
        source: String,
        target: String,
        coefficient: f32,
    },
    CoordinateSwap {
        source: String,
        target: String,
        coefficient: f32,
    },
}

impl Action {
    fn coefficient(&self) -> f32 {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient,
        }
    }

    pub(crate) fn direction_ids<'a>(&'a self) -> impl Iterator<Item = &'a str> + 'a {
        match self {
            Self::FixedAdd { direction, .. }
            | Self::ResidualL2Fraction { direction, .. }
            | Self::ProjectionAblate { direction, .. } => {
                EitherDirectionIds::One(std::iter::once(direction.as_str()))
            }
            Self::SourceToTarget { source, target, .. }
            | Self::CoordinateSwap { source, target, .. } => {
                EitherDirectionIds::Two([source.as_str(), target.as_str()].into_iter())
            }
        }
    }
}

enum EitherDirectionIds<'a> {
    One(std::iter::Once<&'a str>),
    Two(std::array::IntoIter<&'a str, 2>),
}

impl<'a> Iterator for EitherDirectionIds<'a> {
    type Item = &'a str;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::One(iter) => iter.next(),
            Self::Two(iter) => iter.next(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadoutDefinition {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) scope: Scope,
    pub(crate) top_k: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub(crate) layers: Selector,
    #[serde(default)]
    pub(crate) prefill: Option<Selector>,
    #[serde(default)]
    pub(crate) decode: Option<Selector>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Selector {
    All,
    Values { values: Vec<u32> },
    Range { start: u32, end: u32 },
}

impl Selector {
    fn validate(&self, name: &str) -> Result<()> {
        match self {
            Self::All => Ok(()),
            Self::Values { values } => {
                ensure!(!values.is_empty(), "{name} values must not be empty");
                ensure!(
                    values.len() <= MAX_SELECTOR_VALUES,
                    "{name} has too many explicit values"
                );
                ensure!(
                    values.windows(2).all(|pair| pair[0] < pair[1]),
                    "{name} values must be sorted and unique"
                );
                Ok(())
            }
            Self::Range { start, end } => {
                ensure!(
                    start <= end,
                    "{name} range must be inclusive with start <= end"
                );
                ensure!(
                    u64::from(*end) - u64::from(*start) < MAX_SELECTOR_VALUES as u64,
                    "{name} range is too large"
                );
                Ok(())
            }
        }
    }

    fn expand(&self, upper_bound: u32, name: &str) -> Result<Vec<u32>> {
        self.validate(name)?;
        let values = match self {
            Self::All => (0..upper_bound).collect(),
            Self::Values { values } => values.clone(),
            Self::Range { start, end } => (*start..=*end).collect(),
        };
        ensure!(
            values.iter().all(|&value| value < upper_bound),
            "{name} contains a value outside 0..{upper_bound}"
        );
        Ok(values)
    }
}

impl Scope {
    fn validate(&self, name: &str) -> Result<()> {
        self.layers.validate(&format!("{name}.layers"))?;
        ensure!(
            self.prefill.is_some() || self.decode.is_some(),
            "{name} must select prefill and/or decode"
        );
        if let Some(selector) = &self.prefill {
            selector.validate(&format!("{name}.prefill"))?;
        }
        if let Some(selector) = &self.decode {
            selector.validate(&format!("{name}.decode"))?;
        }
        Ok(())
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RunOutput {
    schema: &'static str,
    schema_version: u32,
    runtime_kind: &'static str,
    model_path: PathBuf,
    canonical_plan_path: PathBuf,
    plan: LensPlan,
    input_source: &'static str,
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    sampler: RunSampler,
    max_new_tokens: usize,
    decoded_text: String,
    stop_reason: String,
    operation_applications: Vec<OperationApplication>,
    requested_live_readouts: Vec<ReadoutDefinition>,
    live_readouts: Vec<LiveReadout>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    native_hyper_captures: Vec<NativeHyperCapture>,
}

#[derive(Debug, Serialize)]
struct RunSampler {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    min_p: f32,
    seed: u64,
}

#[derive(Debug, Serialize)]
pub(crate) struct OperationApplication {
    pub(crate) id: String,
    pub(crate) layer: u32,
    pub(crate) phase: &'static str,
    pub(crate) index: usize,
}

#[derive(Debug, Serialize)]
pub(crate) struct NativeHyperCapture {
    operation_id: String,
    layer: u32,
    phase: &'static str,
    index: usize,
    position: usize,
    coordinate: &'static str,
    capture_stage: &'static str,
    shape: [usize; 2],
    flattening: &'static str,
    direction_normalization: &'static str,
    coefficient: f32,
    values: Vec<f32>,
}

#[derive(Debug, Serialize)]
pub(crate) struct LiveReadout {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) method: String,
    pub(crate) score_kind: &'static str,
    pub(crate) candidate_universe: &'static str,
    pub(crate) source_layer: u32,
    pub(crate) target_layer: Option<u32>,
    pub(crate) phase: &'static str,
    pub(crate) index: usize,
    pub(crate) scores: Vec<LiveScore>,
}

#[derive(Debug, Serialize)]
pub(crate) struct LiveScore {
    pub(crate) token_id: Option<i32>,
    pub(crate) row_id: usize,
    pub(crate) word_id: Option<i64>,
    pub(crate) label: Option<String>,
    pub(crate) score: f32,
}

pub(crate) struct RunResult {
    pub(crate) prompt_token_ids: Vec<i32>,
    pub(crate) generated_token_ids: Vec<i32>,
    pub(crate) decoded_text: String,
    pub(crate) stop_reason: String,
    pub(crate) operation_applications: Vec<OperationApplication>,
    pub(crate) live_readouts: Vec<LiveReadout>,
    pub(crate) native_hyper_captures: Vec<NativeHyperCapture>,
}

pub(crate) fn emit_run_output(
    args: &LensRunArgs,
    runtime_kind: &'static str,
    plan_path: &Path,
    plan: LensPlan,
    result: RunResult,
    output_path: Option<&Path>,
) -> Result<()> {
    let artifact = RunOutput {
        schema: RUN_SCHEMA,
        schema_version: RUN_SCHEMA_VERSION,
        runtime_kind,
        model_path: args.model.clone(),
        canonical_plan_path: plan_path.to_path_buf(),
        requested_live_readouts: plan.readouts.clone(),
        plan,
        input_source: input_source(args),
        prompt_token_ids: result.prompt_token_ids,
        generated_token_ids: result.generated_token_ids,
        sampler: RunSampler {
            temperature: args.temperature,
            top_k: args.top_k,
            top_p: args.top_p,
            min_p: args.min_p,
            seed: args.seed,
        },
        max_new_tokens: args.max_new_tokens,
        decoded_text: result.decoded_text,
        stop_reason: result.stop_reason,
        operation_applications: result.operation_applications,
        live_readouts: result.live_readouts,
        native_hyper_captures: result.native_hyper_captures,
    };
    let stdout_format = effective_run_stdout_format(args.format, output_path.is_some());
    let bytes = if output_path.is_some() || stdout_format == RunStdoutFormat::Json {
        let bytes = serde_json::to_vec(&artifact).context("serialize Lens run artifact")?;
        ensure!(
            bytes.len() <= MAX_RUN_ARTIFACT_BYTES,
            "serialized Lens run artifact is {} bytes; limit is {MAX_RUN_ARTIFACT_BYTES}",
            bytes.len()
        );
        Some(bytes)
    } else {
        None
    };
    if let (Some(path), Some(bytes)) = (output_path, &bytes) {
        super::write_atomic_replace(path, bytes)?;
    }
    match stdout_format {
        RunStdoutFormat::Summary => print_run_summary(&artifact, output_path),
        RunStdoutFormat::Json => {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout
                .write_all(bytes.as_deref().expect("JSON output was serialized"))
                .context("write Lens run JSON")?;
            stdout.write_all(b"\n").context("finish Lens run JSON")?;
        }
    }
    Ok(())
}

fn effective_run_stdout_format(
    explicit: Option<RunStdoutFormat>,
    has_output: bool,
) -> RunStdoutFormat {
    explicit.unwrap_or(if has_output {
        RunStdoutFormat::Summary
    } else {
        RunStdoutFormat::Json
    })
}

fn input_source(args: &LensRunArgs) -> &'static str {
    if args.prompt.is_some() {
        "prompt"
    } else if args.token_ids.is_some() {
        "token_ids"
    } else {
        "messages"
    }
}

fn print_run_summary(artifact: &RunOutput, output_path: Option<&Path>) {
    print!("{}", run_summary(artifact, output_path));
}

fn run_summary(artifact: &RunOutput, output_path: Option<&Path>) -> String {
    let mut summary = format!(
        "runtime={} model={}\ngenerated_text={}\nstop_reason={}\noperation_applications={} live_readouts={}\n",
        artifact.runtime_kind,
        artifact.model_path.display(),
        serde_json::to_string(&artifact.decoded_text).expect("string serialization cannot fail"),
        artifact.stop_reason,
        artifact.operation_applications.len(),
        artifact.live_readouts.len()
    );
    if let Some(path) = output_path {
        summary.push_str(&format!("artifact={}\n", path.display()));
    }
    summary
}

struct NativeLens {
    method: String,
    target_layer: u32,
    source_layers: Vec<u32>,
    token_ids: Vec<i32>,
    hidden_size: usize,
    values: Vec<f32>,
}

enum LoadedLens {
    Native(NativeLens),
    Template {
        lens: TemplateLens,
        vocabulary: TemplateVocabulary,
    },
}

struct PreparedLens {
    id: String,
    lens: LoadedLens,
}

struct PreparedDirection {
    rows: BTreeMap<u32, MetalTensor>,
}

struct ExecutionPlan {
    plan: LensPlan,
    lenses: HashMap<String, PreparedLens>,
    directions: HashMap<String, PreparedDirection>,
    coordinate_swaps: HashMap<String, PreparedDirection>,
    capture_layers: Vec<u32>,
    layer_slots: HashMap<u32, usize>,
    n_layer: u32,
    hidden_size: usize,
    capture: Option<MetalTensor>,
}

struct PreparedNativeHyperDirection {
    layer: u32,
    values: Vec<f32>,
}

struct Qwen4ExpExecutionPlan {
    plan: LensPlan,
    directions: HashMap<String, PreparedNativeHyperDirection>,
    operation_layers: HashMap<String, u32>,
    branch_count: usize,
    hidden_size: usize,
}

pub(crate) fn run(args: LensRunArgs) -> Result<()> {
    ensure!(
        args.max_new_tokens > 0 && args.max_new_tokens <= MAX_NEW_TOKENS,
        "--max-new-tokens must be in 1..={MAX_NEW_TOKENS}"
    );
    ensure!(
        args.prompt.is_some() || args.token_ids.is_some() || args.messages.is_some(),
        "exactly one prompt input is required"
    );
    ensure!(
        !args.no_special_tokens || args.prompt.is_some(),
        "--no-special-tokens is supported only with --prompt"
    );
    let output_path = args
        .output
        .as_deref()
        .map(super::resolve_output_file_path)
        .transpose()?;

    let plan_path = std::fs::canonicalize(&args.plan)
        .with_context(|| format!("resolve plan {}", args.plan.display()))?;
    let plan = parse_plan_bytes(&super::read_regular_file_bounded(
        &plan_path,
        MAX_PLAN_BYTES,
    )?)
    .with_context(|| format!("parse Lens plan {}", plan_path.display()))?;
    validate_plan(&plan)?;
    let plan_dir = plan_path.parent().unwrap_or_else(|| Path::new("."));

    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open model {}", args.model.display()))?;
    if crate::muse_lens_artifact::is_muse_architecture(gguf.architecture().as_deref()) {
        return crate::muse_lens_run::run(
            &args,
            plan,
            &plan_path,
            plan_dir,
            gguf,
            output_path.as_deref(),
        );
    }
    let family = ModelFamily::detect(&gguf).context("model has no supported Qwen architecture")?;
    if family == ModelFamily::Qwen4Exp {
        return run_qwen4exp(
            &args,
            plan,
            &plan_path,
            plan_dir,
            gguf,
            output_path.as_deref(),
        );
    }
    validate_ordinary_plan(&plan)?;

    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf(gguf, args.model.clone())
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_runtime(&loaded.gguf(), loaded.arch().kind, loaded.arch().n_layer)?;
    let tokenizer = loaded.tokenizer().context("load model tokenizer")?;
    let prompt_token_ids = input_token_ids(&args, family, &loaded.gguf(), &tokenizer)?;
    ensure!(
        !prompt_token_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        prompt_token_ids.len() <= MAX_NEW_TOKENS * 16,
        "prompt is too long for the bounded Lens runner"
    );
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < loaded.arch().vocab_size),
        "prompt contains a token outside the model vocabulary"
    );

    let mut execution = prepare_execution_plan(&plan, plan_dir, &loaded)?;
    validate_reachable_scopes(&execution.plan, prompt_token_ids.len(), args.max_new_tokens)?;
    let mut sequence = loaded.create_sequence(SequenceConfig::new(
        prompt_token_ids
            .len()
            .checked_add(args.max_new_tokens)
            .context("sequence capacity overflow")?,
    ))?;
    let forward = loaded.forward();
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    })?;
    let stop_tokens: HashSet<i32> = loaded.gguf().stop_token_ids()?.into_iter().collect();
    let mut operation_applications = Vec::new();
    let mut live_readouts = Vec::new();
    let mut logits = Vec::new();

    for (index, &token) in prompt_token_ids.iter().enumerate() {
        logits = forward_event(
            &mut execution,
            &forward,
            token,
            index as u32,
            &mut sequence,
            Phase::Prefill(index),
            &mut operation_applications,
            &mut live_readouts,
        )?;
    }
    let mut generated_token_ids = Vec::new();
    let mut stop_reason = String::from("max_new_tokens");
    for generated_index in 0..args.max_new_tokens {
        let sampled = sampler.sample(&logits)?.token;
        generated_token_ids.push(sampled);
        if stop_tokens.contains(&sampled) {
            stop_reason = String::from("stop_token");
            break;
        }
        if generated_index + 1 == args.max_new_tokens {
            break;
        }
        logits = forward_event(
            &mut execution,
            &forward,
            sampled,
            (prompt_token_ids.len() + generated_index) as u32,
            &mut sequence,
            Phase::Decode(generated_index),
            &mut operation_applications,
            &mut live_readouts,
        )?;
    }
    let decoded_text = tokenizer.decode(&generated_token_ids);
    let result = RunResult {
        prompt_token_ids,
        generated_token_ids,
        decoded_text,
        stop_reason,
        operation_applications,
        live_readouts,
        native_hyper_captures: Vec::new(),
    };
    emit_run_output(
        &args,
        "ordinary_qwen",
        &plan_path,
        plan,
        result,
        output_path.as_deref(),
    )
}

fn run_qwen4exp(
    args: &LensRunArgs,
    plan: LensPlan,
    plan_path: &Path,
    plan_dir: &Path,
    gguf: GgufFile,
    output_path: Option<&Path>,
) -> Result<()> {
    let config = Qwen4ExpConfig::from_gguf(&gguf).context("bind Flash-Next model geometry")?;
    ensure!(
        config == Qwen4ExpConfig::flash_next_reference(),
        "qwen-lens run requires the released Flash-Next architecture contract"
    );
    let tokenizer = Tokenizer::from_gguf(&gguf).context("load Flash-Next tokenizer")?;
    ensure!(
        tokenizer.n_vocab() == config.vocab_size,
        "Flash-Next tokenizer vocabulary {} differs from model {}",
        tokenizer.n_vocab(),
        config.vocab_size
    );
    let prompt_token_ids = input_token_ids(args, ModelFamily::Qwen4Exp, &gguf, &tokenizer)?;
    ensure!(
        !prompt_token_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        prompt_token_ids.len() <= MAX_NEW_TOKENS * 16,
        "prompt is too long for the bounded Lens runner"
    );
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < config.vocab_size),
        "prompt contains a token outside the Flash-Next vocabulary"
    );
    validate_reachable_scopes(&plan, prompt_token_ids.len(), args.max_new_tokens)?;
    let execution = prepare_qwen4exp_execution_plan(plan, plan_dir, &config)?;
    validate_qwen4exp_event_schedule(&execution, prompt_token_ids.len(), args.max_new_tokens)?;

    let required_forwards = prompt_token_ids
        .len()
        .checked_add(args.max_new_tokens.saturating_sub(1))
        .context("Flash-Next forward count overflow")?;
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, required_forwards)
        .context("derive Flash-Next serial session capacity")?;
    let stop_tokens = gguf
        .stop_token_ids()
        .context("load Flash-Next stop tokens")?;
    ensure!(
        stop_tokens
            .iter()
            .all(|&token| token >= 0 && (token as u32) < config.vocab_size),
        "Flash-Next stop-token metadata contains an invalid token"
    );
    let stop_tokens = stop_tokens.into_iter().collect::<HashSet<_>>();

    let context = MetalContext::new().context("initialize Metal for Flash-Next Lens run")?;
    let mut loaded = Qwen4ExpLoadedModel::load(&context, &gguf, capacity)
        .context("load Flash-Next serial Lens session")?;
    let mut runner = loaded
        .create_runner(&context)
        .context("bind Flash-Next serial Lens runner")?;
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    })?;
    let mut operation_applications = Vec::new();
    let mut native_hyper_captures = Vec::new();
    let mut logits = Vec::new();
    for (index, &token) in prompt_token_ids.iter().enumerate() {
        logits = qwen4exp_forward_event(
            &execution,
            &mut runner,
            u32::try_from(token).context("Flash-Next prompt token is negative")?,
            Phase::Prefill(index),
            &mut operation_applications,
            &mut native_hyper_captures,
        )?;
    }

    let mut generated_token_ids = Vec::new();
    let mut stop_reason = String::from("max_new_tokens");
    for generated_index in 0..args.max_new_tokens {
        let sampled = sampler.sample(&logits)?.token;
        generated_token_ids.push(sampled);
        if stop_tokens.contains(&sampled) {
            stop_reason = String::from("stop_token");
            break;
        }
        if generated_index + 1 == args.max_new_tokens {
            break;
        }
        logits = qwen4exp_forward_event(
            &execution,
            &mut runner,
            u32::try_from(sampled).context("Flash-Next sampled a negative token")?,
            Phase::Decode(generated_index),
            &mut operation_applications,
            &mut native_hyper_captures,
        )?;
    }

    let result = RunResult {
        prompt_token_ids,
        decoded_text: tokenizer.decode(&generated_token_ids),
        generated_token_ids,
        stop_reason,
        operation_applications,
        live_readouts: Vec::new(),
        native_hyper_captures,
    };
    let plan = execution.plan.clone();
    emit_run_output(args, "flash_next", plan_path, plan, result, output_path)
}

fn prepare_qwen4exp_execution_plan(
    plan: LensPlan,
    plan_dir: &Path,
    config: &Qwen4ExpConfig,
) -> Result<Qwen4ExpExecutionPlan> {
    ensure!(
        plan.lenses.is_empty(),
        "Flash-Next Lens plans cannot use ordinary J/R or template lenses"
    );
    ensure!(
        plan.readouts.is_empty(),
        "Flash-Next live readouts require a real hyper-space lens and are not yet supported"
    );
    ensure!(
        !plan.operations.is_empty(),
        "Flash-Next Lens plans must declare at least one fixed-add operation"
    );
    ensure!(
        plan.directions
            .iter()
            .all(|direction| direction.native_hyper().is_some()),
        "Flash-Next Lens plans require native_hyper_f32 directions"
    );
    let branch_count = config.hyper_connection.count as usize;
    let hidden_size = config.hidden_size as usize;
    let hyper_width = branch_count
        .checked_mul(hidden_size)
        .context("Flash-Next hyper width overflow")?;
    let mut directions = HashMap::new();
    for definition in &plan.directions {
        let definition = definition
            .native_hyper()
            .context("Flash-Next direction is not native hyper")?;
        let (path, layer) = definition.source.path_and_layer();
        ensure!(
            layer > 0 && layer < config.layer_count,
            "native hyper direction {} layer {} is outside 1..{}",
            definition.id,
            layer,
            config.layer_count
        );
        let values = load_native_hyper_direction(
            &resolve_plan_path(plan_dir, path),
            hyper_width,
            &definition.id,
        )?;
        ensure!(
            directions
                .insert(
                    definition.id.clone(),
                    PreparedNativeHyperDirection { layer, values }
                )
                .is_none()
        );
    }

    let mut operation_layers = HashMap::new();
    for operation in &plan.operations {
        let (direction_id, coefficient) = match &operation.action {
            Action::FixedAdd {
                direction,
                coefficient,
            } => (direction.as_str(), *coefficient),
            _ => bail!(
                "Flash-Next operation {} supports fixed_add only",
                operation.id
            ),
        };
        let direction = directions.get(direction_id).with_context(|| {
            format!(
                "Flash-Next operation {} has no native hyper direction {}",
                operation.id, direction_id
            )
        })?;
        let layers = operation.scope.layers.expand(
            config.layer_count,
            &format!("operation {} layers", operation.id),
        )?;
        ensure!(
            layers.len() == 1 && layers[0] == direction.layer,
            "Flash-Next operation {} must select only direction {} layer {}",
            operation.id,
            direction_id,
            direction.layer
        );
        ensure!(
            direction
                .values
                .iter()
                .all(|value| (*value * coefficient).is_finite()),
            "Flash-Next operation {} coefficient overflows its direction",
            operation.id
        );
        operation_layers.insert(operation.id.clone(), direction.layer);
    }
    Ok(Qwen4ExpExecutionPlan {
        plan,
        directions,
        operation_layers,
        branch_count,
        hidden_size,
    })
}

fn load_native_hyper_direction(path: &Path, width: usize, id: &str) -> Result<Vec<f32>> {
    let expected_bytes = width
        .checked_mul(std::mem::size_of::<f32>())
        .context("native hyper direction byte count overflow")?;
    let bytes = super::read_regular_file_exact(path, expected_bytes)
        .with_context(|| format!("read native hyper direction {id} from {}", path.display()))?;
    let values = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect::<Vec<_>>();
    normalize_direction(values, Normalization::AsStored, id)
}

fn validate_qwen4exp_event_schedule(
    execution: &Qwen4ExpExecutionPlan,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let mut capture_count = 0usize;
    for index in 0..prompt_len {
        if qwen4exp_matching_operation(execution, Phase::Prefill(index))?.is_some() {
            capture_count += 1;
        }
    }
    for index in 0..max_new_tokens.saturating_sub(1) {
        if qwen4exp_matching_operation(execution, Phase::Decode(index))?.is_some() {
            capture_count += 1;
        }
    }
    ensure!(
        capture_count <= MAX_NATIVE_HYPER_CAPTURES,
        "Flash-Next plan can emit {capture_count} native hyper captures, maximum is {MAX_NATIVE_HYPER_CAPTURES}"
    );
    Ok(())
}

fn qwen4exp_matching_operation<'a>(
    execution: &'a Qwen4ExpExecutionPlan,
    phase: Phase,
) -> Result<Option<(&'a OperationDefinition, u32)>> {
    let mut matched = None;
    for operation in &execution.plan.operations {
        let layer = execution.operation_layers[&operation.id];
        if !scope_matches(&operation.scope, phase, layer)? {
            continue;
        }
        ensure!(
            matched.is_none(),
            "Flash-Next operations overlap at {} index {}; one native hyper probe is supported per token event",
            phase.label(),
            phase.index()
        );
        matched = Some((operation, layer));
    }
    Ok(matched)
}

fn qwen4exp_forward_event(
    execution: &Qwen4ExpExecutionPlan,
    runner: &mut Qwen4ExpTextRunner<'_, '_, '_>,
    token: u32,
    phase: Phase,
    operation_applications: &mut Vec<OperationApplication>,
    native_hyper_captures: &mut Vec<NativeHyperCapture>,
) -> Result<Vec<f32>> {
    let Some((operation, layer)) = qwen4exp_matching_operation(execution, phase)? else {
        return Ok(runner
            .forward_token(token)
            .with_context(|| {
                format!(
                    "forward Flash-Next {} token {}",
                    phase.label(),
                    phase.index()
                )
            })?
            .to_vec());
    };
    let (direction_id, coefficient) = match &operation.action {
        Action::FixedAdd {
            direction,
            coefficient,
        } => (direction, *coefficient),
        _ => unreachable!("Flash-Next plan validation admits fixed_add only"),
    };
    let direction = &execution.directions[direction_id];
    let capture = runner
        .forward_token_with_post_layer_hyper_capture(
            token,
            Qwen4ExpPostLayerHyperRequest {
                layer,
                fixed_add: Some(Qwen4ExpFixedHyperAdd {
                    direction: &direction.values,
                    coefficient,
                }),
            },
        )
        .with_context(|| {
            format!(
                "apply Flash-Next operation {} at {} index {}",
                operation.id,
                phase.label(),
                phase.index()
            )
        })?;
    let logits = runner.logits()?.to_vec();
    operation_applications.push(OperationApplication {
        id: operation.id.clone(),
        layer,
        phase: phase.label(),
        index: phase.index(),
    });
    native_hyper_captures.push(NativeHyperCapture {
        operation_id: operation.id.clone(),
        layer,
        phase: phase.label(),
        index: phase.index(),
        position: capture.position,
        coordinate: "qwen4exp_persistent_post_layer_hyper_state",
        capture_stage: "after_fixed_add",
        shape: [execution.branch_count, execution.hidden_size],
        flattening: "branch_major_hidden_minor",
        direction_normalization: "as_stored",
        coefficient,
        values: capture.values,
    });
    Ok(logits)
}

fn parse_plan_bytes(bytes: &[u8]) -> Result<LensPlan> {
    // Parse through Value so serde_json's arbitrary-precision number marker is
    // resolved before serde buffers the internally tagged action enum.
    let value: serde_json::Value = serde_json::from_slice(bytes)?;
    Ok(serde_json::from_value(value)?)
}

fn validate_plan(plan: &LensPlan) -> Result<()> {
    ensure!(plan.version == 1, "Lens plan version must be 1");
    ensure!(
        !plan.lenses.is_empty()
            || plan
                .directions
                .iter()
                .any(|direction| direction.native_hyper().is_some()),
        "Lens plan must declare at least one lens or native hyper direction"
    );
    ensure!(plan.lenses.len() <= MAX_LENSES, "too many lenses");
    ensure!(
        plan.directions.len() <= MAX_DIRECTIONS,
        "too many directions"
    );
    ensure!(
        plan.operations.len() <= MAX_OPERATIONS,
        "too many operations"
    );
    ensure!(plan.readouts.len() <= MAX_READOUTS, "too many readouts");
    unique_ids(plan.lenses.iter().map(LensDefinition::id), "lens")?;
    unique_ids(
        plan.directions.iter().map(DirectionDefinition::id),
        "direction",
    )?;
    unique_ids(
        plan.operations.iter().map(|item| item.id.as_str()),
        "operation",
    )?;
    unique_ids(plan.readouts.iter().map(|item| item.id.as_str()), "readout")?;
    for lens in &plan.lenses {
        if let LensDefinition::PublishedFullTransport {
            id,
            token_ids,
            allow_unvalidated_transfer,
            ..
        } = lens
        {
            let unique = token_ids.iter().copied().collect::<HashSet<_>>();
            ensure!(
                !token_ids.is_empty()
                    && token_ids.len() <= MAX_PUBLISHED_FULL_TOKEN_IDS
                    && unique.len() == token_ids.len(),
                "published full transport lens {id} requires 1..={MAX_PUBLISHED_FULL_TOKEN_IDS} unique token IDs"
            );
            ensure!(
                *allow_unvalidated_transfer,
                "published full transport lens {id} requires allow_unvalidated_transfer=true for BF16-to-GGUF use"
            );
        }
    }
    let lens_ids: HashSet<&str> = plan.lenses.iter().map(LensDefinition::id).collect();
    let published_full_tokens = plan
        .lenses
        .iter()
        .filter_map(|lens| match lens {
            LensDefinition::PublishedFullTransport { id, token_ids, .. } => Some((
                id.as_str(),
                token_ids.iter().copied().collect::<HashSet<_>>(),
            )),
            _ => None,
        })
        .collect::<HashMap<_, _>>();
    let direction_ids: HashSet<&str> = plan
        .directions
        .iter()
        .map(DirectionDefinition::id)
        .collect();
    let direction_normalization: HashMap<&str, Option<Normalization>> = plan
        .directions
        .iter()
        .map(|direction| (direction.id(), direction.normalization()))
        .collect();
    for direction in &plan.directions {
        match direction {
            DirectionDefinition::LensRow(direction) => {
                ensure!(
                    lens_ids.contains(direction.lens.as_str()),
                    "direction {} references unknown lens {}",
                    direction.id,
                    direction.lens
                );
                if let Some(token_ids) = published_full_tokens.get(direction.lens.as_str()) {
                    let DirectionRow::TokenId { token_id } = direction.row else {
                        bail!(
                            "published full transport direction {} requires row.kind=token_id",
                            direction.id
                        );
                    };
                    ensure!(
                        token_id >= 0 && token_ids.contains(&(token_id as u32)),
                        "published full transport direction {} selects token {} absent from lens {}",
                        direction.id,
                        token_id,
                        direction.lens
                    );
                }
            }
            DirectionDefinition::NativeHyper(direction) => {
                let (path, _) = direction.source.path_and_layer();
                ensure!(
                    !path.as_os_str().is_empty(),
                    "native hyper direction {} path must not be empty",
                    direction.id
                );
            }
        }
    }
    for operation in &plan.operations {
        operation
            .scope
            .validate(&format!("operation {} scope", operation.id))?;
        ensure!(
            operation.action.coefficient().is_finite() && operation.action.coefficient() != 0.0,
            "operation {} coefficient must be finite and nonzero",
            operation.id
        );
        if let Action::CoordinateSwap {
            source,
            target,
            coefficient,
        } = &operation.action
        {
            ensure!(
                source != target,
                "coordinate-swap operation {} requires distinct source and target directions",
                operation.id
            );
            ensure!(
                (2.0 * *coefficient).is_finite(),
                "coordinate-swap operation {} coefficient overflows its reflection scale",
                operation.id
            );
        }
        for direction in operation.action.direction_ids() {
            ensure!(
                direction_ids.contains(direction),
                "operation {} references unknown direction {}",
                operation.id,
                direction
            );
        }
        if action_requires_unit_l2(&operation.action) {
            for direction in operation.action.direction_ids() {
                if let Some(normalization) =
                    direction_normalization.get(direction).copied().flatten()
                {
                    ensure!(
                        normalization == Normalization::UnitL2,
                        "operation {} requires unit_l2 direction {}",
                        operation.id,
                        direction
                    );
                }
            }
        }
    }
    for readout in &plan.readouts {
        readout
            .scope
            .validate(&format!("readout {} scope", readout.id))?;
        ensure!(
            readout.top_k > 0 && readout.top_k <= MAX_TOP_K,
            "readout {} top_k must be in 1..={MAX_TOP_K}",
            readout.id
        );
        ensure!(
            lens_ids.contains(readout.lens.as_str()),
            "readout {} references unknown lens {}",
            readout.id,
            readout.lens
        );
    }
    Ok(())
}

fn unique_ids<'a>(ids: impl Iterator<Item = &'a str>, kind: &str) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        ensure!(!id.is_empty(), "{kind} id must not be empty");
        ensure!(seen.insert(id), "duplicate {kind} id {id:?}");
    }
    Ok(())
}

fn validate_ordinary_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        !plan.lenses.is_empty(),
        "ordinary Qwen Lens plans must declare at least one lens"
    );
    ensure!(
        plan.directions
            .iter()
            .all(|direction| direction.lens_row().is_some()),
        "native hyper directions are supported only by Flash-Next"
    );
    Ok(())
}

fn validate_runtime(gguf: &GgufFile, kind: ArchKind, n_layer: u32) -> Result<()> {
    let family = ModelFamily::detect(gguf).context("model has no supported Qwen architecture")?;
    ensure!(
        matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "qwen-lens run supports ordinary Qwen dense or MoE models only"
    );
    ensure!(
        matches!(kind, ArchKind::Dense | ArchKind::Moe),
        "unsupported model runtime shape"
    );
    ensure!(n_layer > 0, "model has no transformer blocks");
    Ok(())
}

fn input_token_ids(
    args: &LensRunArgs,
    family: ModelFamily,
    gguf: &GgufFile,
    tokenizer: &Tokenizer,
) -> Result<Vec<i32>> {
    match (&args.prompt, &args.token_ids, &args.messages) {
        (Some(prompt), None, None) => Ok(tokenizer.encode(prompt, !args.no_special_tokens)?),
        (None, Some(token_ids), None) => {
            ensure!(!token_ids.is_empty(), "--token-ids must not be empty");
            ensure!(
                token_ids
                    .iter()
                    .all(|&id| id >= 0 && (id as u32) < tokenizer.n_vocab()),
                "--token-ids contains an ID outside the tokenizer vocabulary"
            );
            Ok(token_ids.clone())
        }
        (None, None, Some(path)) => {
            let raw = std::fs::read_to_string(path)
                .with_context(|| format!("read messages {}", path.display()))?;
            let messages = parse_strict_messages_input(&raw, &path.display().to_string())?;
            let qwen38 = if family == ModelFamily::Qwen4Exp {
                supports_qwen4exp_prompt_protocol(family, gguf)
            } else {
                qwen38_prompt_protocol(gguf)
            };
            ensure!(
                family != ModelFamily::Qwen4Exp || qwen38,
                "Flash-Next --messages requires the released qwen35 prompt protocol"
            );
            let rendered = if qwen38 {
                render_qwen38_messages_prompt_with_generation(
                    &messages,
                    true,
                    // Do not silently inject an effort instruction into the
                    // user's system prompt. Raw prompts remain available for
                    // callers that need byte-exact rendered input.
                    Qwen38GenerationMode::Thinking(Qwen38ReasoningEffort::Medium),
                )
            } else {
                render_qwen_messages_prompt_with_generation(
                    &messages,
                    false,
                    true,
                    QwenGenerationMode::Auto,
                )
            };
            // `false` prevents tokenizer-configured BOS/EOS insertion while
            // still recognizing special tokens present in the rendered text.
            Ok(tokenizer.encode(&rendered, false)?)
        }
        _ => bail!("exactly one of --prompt, --token-ids, or --messages is required"),
    }
}

fn qwen38_prompt_protocol(gguf: &GgufFile) -> bool {
    let named = [
        gguf.get_str("general.name"),
        gguf.get_str("general.base_model.0.name"),
    ]
    .into_iter()
    .flatten()
    .any(|value| value.to_ascii_lowercase().contains("qwen3.8"));
    named
        && gguf.get_str("tokenizer.ggml.model") == Some("gpt2")
        && gguf.get_str("tokenizer.ggml.pre") == Some("qwen35")
}

fn prepare_execution_plan(
    plan: &LensPlan,
    plan_dir: &Path,
    loaded: &qwen_llm::runtime::LoadedModel,
) -> Result<ExecutionPlan> {
    let arch = loaded.arch();
    let direction_defs = plan
        .directions
        .iter()
        .map(|direction| (direction.id(), direction))
        .collect::<HashMap<_, _>>();
    let mut direction_layers: BTreeMap<&str, BTreeSet<u32>> = BTreeMap::new();
    let mut required_lens_layers: HashMap<&str, BTreeSet<u32>> = HashMap::new();
    for operation in &plan.operations {
        let layers = operation
            .scope
            .layers
            .expand(arch.n_layer, "operation.layers")?;
        for direction_id in operation.action.direction_ids() {
            let direction = direction_defs[direction_id].lens_row().with_context(|| {
                format!("ordinary runtime cannot load native hyper direction {direction_id}")
            })?;
            direction_layers
                .entry(direction_id)
                .or_default()
                .extend(layers.iter().copied());
            required_lens_layers
                .entry(direction.lens.as_str())
                .or_default()
                .extend(layers.iter().copied());
        }
    }
    let mut readout_layers = BTreeSet::new();
    for readout in &plan.readouts {
        let layers = readout
            .scope
            .layers
            .expand(arch.n_layer, "readout.layers")?;
        required_lens_layers
            .entry(readout.lens.as_str())
            .or_default()
            .extend(layers.iter().copied());
        readout_layers.extend(layers);
    }

    let mut lenses = HashMap::new();
    for definition in &plan.lenses {
        let prepared = match definition {
            LensDefinition::NativeSelected { id, artifact } => PreparedLens {
                id: id.clone(),
                lens: LoadedLens::Native(load_native_lens(
                    &resolve_plan_path(plan_dir, artifact),
                    arch.n_layer,
                    arch.hidden_size as usize,
                    arch.vocab_size,
                )?),
            },
            LensDefinition::PublishedFullTransport {
                id,
                artifact,
                token_ids,
                allow_unvalidated_transfer: _,
            } => {
                let layers = required_lens_layers
                    .get(id.as_str())
                    .with_context(|| format!("published full transport lens {id} is not used"))?
                    .iter()
                    .copied()
                    .collect::<Vec<_>>();
                let projected = super::full_lens::project_full_token_directions(
                    &resolve_plan_path(plan_dir, artifact),
                    token_ids,
                    &layers,
                    loaded,
                )?;
                PreparedLens {
                    id: id.clone(),
                    lens: LoadedLens::Native(NativeLens {
                        method: projected.method,
                        target_layer: projected.target_layer,
                        source_layers: projected.source_layers,
                        token_ids: projected.token_ids,
                        hidden_size: projected.hidden_size,
                        values: projected.values,
                    }),
                }
            }
            LensDefinition::WorkspaceTemplate {
                id,
                weights,
                labels,
            } => {
                let weights_path = resolve_plan_path(plan_dir, weights);
                let lens = TemplateLens::open(&weights_path)
                    .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                let vocabulary =
                    TemplateVocabulary::load(&resolve_plan_path(plan_dir, labels), lens.n_rows())
                        .map_err(|error| anyhow::anyhow!(error.to_string()))?;
                ensure!(
                    lens.hidden_size() == arch.hidden_size as usize,
                    "template lens {} hidden size {} != model {}",
                    id,
                    lens.hidden_size(),
                    arch.hidden_size
                );
                PreparedLens {
                    id: id.clone(),
                    lens: LoadedLens::Template { lens, vocabulary },
                }
            }
        };
        ensure!(lenses.insert(prepared.id.clone(), prepared).is_none());
    }

    for readout in &plan.readouts {
        let layers = readout
            .scope
            .layers
            .expand(arch.n_layer, "readout.layers")?;
        let prepared_lens = &lenses[&readout.lens];
        for &layer in &layers {
            ensure!(
                lens_has_layer(prepared_lens, layer),
                "readout {} lens {} has no row for layer {}",
                readout.id,
                readout.lens,
                layer
            );
        }
    }
    let capture_layers: Vec<u32> = readout_layers.iter().copied().collect();
    let layer_slots = capture_layers
        .iter()
        .enumerate()
        .map(|(slot, &layer)| (layer, slot))
        .collect::<HashMap<_, _>>();

    let mut directions = HashMap::new();
    for (direction_id, layers) in direction_layers {
        let definition = direction_defs[direction_id];
        let definition = definition.lens_row().with_context(|| {
            format!("ordinary runtime cannot load native hyper direction {direction_id}")
        })?;
        let prepared_lens = &lenses[&definition.lens];
        let mut rows = BTreeMap::new();
        for layer in layers {
            let raw = lens_row(prepared_lens, &definition.row, layer)?;
            let normalized = normalize_direction(raw, definition.normalization, direction_id)?;
            let tensor = MetalTensor::from_bytes(
                loaded.context(),
                bytemuck::cast_slice(&normalized),
                vec![arch.hidden_size as u64],
                GgmlType::F32,
            )?;
            ensure!(rows.insert(layer, tensor).is_none());
        }
        directions.insert(direction_id.to_owned(), PreparedDirection { rows });
    }
    let coordinate_swaps = prepare_coordinate_swaps(
        &plan.operations,
        &directions,
        loaded.context(),
        arch.n_layer,
        arch.hidden_size as usize,
    )?;
    let capture = if capture_layers.is_empty() {
        None
    } else {
        Some(MetalTensor::zeros_f32(
            loaded.context(),
            vec![(capture_layers.len() * arch.hidden_size as usize) as u64],
        )?)
    };
    Ok(ExecutionPlan {
        plan: plan.clone(),
        lenses,
        directions,
        coordinate_swaps,
        capture_layers,
        layer_slots,
        n_layer: arch.n_layer,
        hidden_size: arch.hidden_size as usize,
        capture,
    })
}

fn prepare_coordinate_swaps(
    operations: &[OperationDefinition],
    directions: &HashMap<String, PreparedDirection>,
    context: &MetalContext,
    n_layer: u32,
    hidden_size: usize,
) -> Result<HashMap<String, PreparedDirection>> {
    let mut swaps = HashMap::new();
    let mut reflection_cache = HashMap::<(String, String, u32), MetalTensor>::new();
    for operation in operations {
        let Action::CoordinateSwap { source, target, .. } = &operation.action else {
            continue;
        };
        let pair = if source <= target {
            (source.clone(), target.clone())
        } else {
            (target.clone(), source.clone())
        };
        let layers = operation
            .scope
            .layers
            .expand(n_layer, &format!("operation {} layers", operation.id))?;
        let source = directions
            .get(source)
            .with_context(|| format!("coordinate swap {} has no source direction", operation.id))?;
        let target = directions
            .get(target)
            .with_context(|| format!("coordinate swap {} has no target direction", operation.id))?;
        let mut rows = BTreeMap::new();
        for layer in layers {
            let cache_key = (pair.0.clone(), pair.1.clone(), layer);
            if let Some(reflection) = reflection_cache.get(&cache_key) {
                ensure!(rows.insert(layer, reflection.clone()).is_none());
                continue;
            }
            let source = source.rows.get(&layer).with_context(|| {
                format!(
                    "coordinate swap {} source is unavailable at layer {layer}",
                    operation.id
                )
            })?;
            let target = target.rows.get(&layer).with_context(|| {
                format!(
                    "coordinate swap {} target is unavailable at layer {layer}",
                    operation.id
                )
            })?;
            let reflection = coordinate_swap_reflection_direction(
                &read_f32_tensor(source, hidden_size),
                &read_f32_tensor(target, hidden_size),
                &format!("{} at layer {layer}", operation.id),
            )?;
            let reflection = MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(&reflection),
                vec![hidden_size as u64],
                GgmlType::F32,
            )?;
            ensure!(
                reflection_cache
                    .insert(cache_key, reflection.clone())
                    .is_none()
            );
            ensure!(rows.insert(layer, reflection).is_none());
        }
        ensure!(
            swaps
                .insert(operation.id.clone(), PreparedDirection { rows })
                .is_none(),
            "duplicate coordinate-swap operation {}",
            operation.id
        );
    }
    Ok(swaps)
}

pub(crate) fn coordinate_swap_reflection_direction(
    source: &[f32],
    target: &[f32],
    id: &str,
) -> Result<Vec<f32>> {
    ensure!(
        !source.is_empty() && source.len() == target.len(),
        "coordinate swap {id} directions have incompatible shapes"
    );
    ensure!(
        source.iter().chain(target).all(|value| value.is_finite()),
        "coordinate swap {id} contains a non-finite direction"
    );
    let source_sq = source
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let target_sq = target
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let source_norm = source_sq.sqrt();
    let target_norm = target_sq.sqrt();
    ensure!(
        source_norm.is_finite()
            && target_norm.is_finite()
            && source_norm > 0.0
            && target_norm > 0.0,
        "coordinate swap {id} has a zero or non-finite direction norm"
    );
    let cosine = source
        .iter()
        .zip(target)
        .map(|(&source, &target)| f64::from(source) * f64::from(target))
        .sum::<f64>()
        / (source_norm * target_norm);
    let cosine = cosine.clamp(-1.0, 1.0);
    ensure!(
        1.0 - cosine * cosine > 1e-10,
        "coordinate swap {id} directions are linearly dependent"
    );
    let difference = source
        .iter()
        .zip(target)
        .map(|(&source, &target)| {
            (f64::from(source) / source_norm - f64::from(target) / target_norm) as f32
        })
        .collect();
    normalize_direction(difference, Normalization::UnitL2, id)
}

fn lens_has_layer(prepared: &PreparedLens, layer: u32) -> bool {
    match &prepared.lens {
        LoadedLens::Native(native) => native.source_layers.contains(&layer),
        LoadedLens::Template { lens, .. } => lens.layers().contains(&layer),
    }
}

fn action_requires_unit_l2(action: &Action) -> bool {
    matches!(
        action,
        Action::ResidualL2Fraction { .. }
            | Action::ProjectionAblate { .. }
            | Action::SourceToTarget { .. }
            | Action::CoordinateSwap { .. }
    )
}

fn resolve_plan_path(plan_dir: &Path, path: &Path) -> PathBuf {
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        plan_dir.join(path)
    }
}

fn load_native_lens(
    artifact: &Path,
    model_layers: u32,
    hidden_size: usize,
    vocab_size: u32,
) -> Result<NativeLens> {
    let manifest_path = if artifact.is_dir() {
        artifact.join("readouts.json")
    } else {
        artifact.to_path_buf()
    };
    let manifest: super::TokenReadoutManifest = super::read_json_file(&manifest_path)?;
    ensure!(
        manifest.schema == super::TOKEN_READOUT_SCHEMA
            && manifest.schema_version == super::SCHEMA_VERSION
            && manifest.status == "complete",
        "native lens {} is not a completed fit-tokens artifact",
        manifest_path.display()
    );
    let method = match manifest.config.method {
        super::FitMethod::J => "J",
        super::FitMethod::R => "R",
    };
    ensure!(
        manifest.config.orientation == super::TOKEN_ORIENTATION
            && manifest.readouts == manifest.config.readouts
            && manifest.readouts.target_covectors_dtype == "f32_le",
        "native lens {} has unsupported readout metadata",
        manifest_path.display()
    );
    super::validate_token_readout_spec(&manifest.readouts, &manifest.config)?;
    ensure!(
        manifest.config.n_layers == model_layers
            && manifest.config.hidden_size as usize == hidden_size
            && manifest.config.vocab_size == vocab_size,
        "native lens {} model geometry does not match the loaded model",
        manifest_path.display()
    );
    ensure!(
        manifest.config.target_layer < model_layers,
        "native lens target layer is outside model geometry"
    );
    let source_layers = manifest.config.source_layers.clone();
    ensure!(
        !source_layers.is_empty(),
        "native lens has no source layers"
    );
    ensure!(
        source_layers.windows(2).all(|pair| pair[0] < pair[1])
            && source_layers.iter().all(|&layer| layer < model_layers),
        "native lens source layers are not sorted or are out of bounds"
    );
    let token_ids = manifest
        .readouts
        .token_ids
        .iter()
        .map(|&id| id as i32)
        .collect::<Vec<_>>();
    ensure!(!token_ids.is_empty(), "native lens has no selected tokens");
    ensure!(
        token_ids
            .iter()
            .all(|&id| id >= 0 && (id as u32) < vocab_size)
            && token_ids.iter().copied().collect::<HashSet<_>>().len() == token_ids.len(),
        "native lens selected token IDs are invalid or duplicated"
    );
    let expected_shape = [source_layers.len(), token_ids.len(), hidden_size];
    ensure!(
        manifest.payload.path == super::TOKEN_PAYLOAD_NAME
            && manifest.payload.dtype == "f32_le"
            && manifest.payload.shape == expected_shape,
        "native lens payload shape or descriptor is not supported"
    );
    let expected_values = expected_shape
        .iter()
        .try_fold(1usize, |product, &dimension| product.checked_mul(dimension))
        .context("native lens payload size overflow")?;
    let expected_bytes = expected_values
        .checked_mul(4)
        .context("native lens byte size overflow")?;
    ensure!(
        expected_bytes <= super::TOKEN_ARTIFACT_MAX_BYTES
            && manifest.payload.byte_length == expected_bytes as u64,
        "native lens payload byte length does not match its shape"
    );
    let directory = manifest_path.parent().unwrap_or_else(|| Path::new("."));
    let payload_path = directory.join(&manifest.payload.path);
    let bytes = super::read_regular_file_exact(&payload_path, expected_bytes)?;
    let mut values = Vec::with_capacity(expected_values);
    for chunk in bytes.chunks_exact(4) {
        let value = f32::from_le_bytes(chunk.try_into().unwrap());
        ensure!(
            value.is_finite(),
            "native lens payload contains a non-finite value"
        );
        values.push(value);
    }
    Ok(NativeLens {
        method: method.into(),
        target_layer: manifest.config.target_layer,
        source_layers,
        token_ids,
        hidden_size,
        values,
    })
}

fn lens_row(prepared: &PreparedLens, selector: &DirectionRow, layer: u32) -> Result<Vec<f32>> {
    match (&prepared.lens, selector) {
        (LoadedLens::Native(native), DirectionRow::TokenId { token_id }) => {
            let layer_slot = native
                .source_layers
                .iter()
                .position(|&candidate| candidate == layer)
                .with_context(|| {
                    format!("native lens {} has no row for layer {layer}", prepared.id)
                })?;
            let token_slot = native
                .token_ids
                .iter()
                .position(|&candidate| candidate == *token_id)
                .with_context(|| {
                    format!("native lens {} has no token row {token_id}", prepared.id)
                })?;
            let offset = native_payload_offset(
                layer_slot,
                token_slot,
                native.token_ids.len(),
                native.hidden_size,
            );
            Ok(native.values[offset..offset + native.hidden_size].to_vec())
        }
        (
            LoadedLens::Template {
                lens,
                vocabulary: _,
            },
            DirectionRow::TemplateRowId { template_row_id },
        ) => lens
            .row_f32(layer, *template_row_id)
            .map_err(|error| anyhow::anyhow!(error.to_string())),
        (LoadedLens::Template { lens, vocabulary }, DirectionRow::Label { label }) => {
            let row_id = vocabulary
                .unique_row_id_for_label(label)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            lens.row_f32(layer, row_id)
                .map_err(|error| anyhow::anyhow!(error.to_string()))
        }
        (LoadedLens::Native(_), _) => bail!("native selected directions require row.kind=token_id"),
        (LoadedLens::Template { .. }, DirectionRow::TokenId { .. }) => {
            bail!("workspace-template directions require row.kind=template_row_id or label")
        }
    }
}

fn native_payload_offset(
    layer_slot: usize,
    token_slot: usize,
    token_count: usize,
    hidden_size: usize,
) -> usize {
    (layer_slot * token_count + token_slot) * hidden_size
}

pub(crate) fn normalize_direction(
    mut row: Vec<f32>,
    normalization: Normalization,
    id: &str,
) -> Result<Vec<f32>> {
    ensure!(!row.is_empty(), "direction {id} is empty");
    ensure!(
        row.iter().all(|value| value.is_finite()),
        "direction {id} has a non-finite value"
    );
    let norm_squared = row
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let norm = norm_squared.sqrt();
    ensure!(
        norm.is_finite() && norm > 0.0,
        "direction {id} has a zero or non-finite norm"
    );
    if normalization == Normalization::UnitL2 {
        for value in &mut row {
            *value = (f64::from(*value) / norm) as f32;
        }
    }
    ensure!(
        row.iter().all(|value| value.is_finite()),
        "direction {id} normalization overflowed"
    );
    Ok(row)
}

pub(crate) fn validate_reachable_scopes(
    plan: &LensPlan,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<()> {
    let prefill_bound =
        u32::try_from(prompt_len).context("prompt length exceeds selector range")?;
    let decode_bound = u32::try_from(max_new_tokens.saturating_sub(1))
        .context("decode selector range overflow")?;
    for operation in &plan.operations {
        validate_scope_reachable(&operation.scope, prefill_bound, decode_bound, &operation.id)?;
    }
    for readout in &plan.readouts {
        validate_scope_reachable(&readout.scope, prefill_bound, decode_bound, &readout.id)?;
    }
    Ok(())
}

fn validate_scope_reachable(
    scope: &Scope,
    prefill_bound: u32,
    decode_bound: u32,
    id: &str,
) -> Result<()> {
    if let Some(selector) = &scope.prefill {
        selector.expand(prefill_bound, &format!("{id}.prefill"))?;
    }
    if let Some(selector) = &scope.decode {
        ensure!(
            decode_bound > 0,
            "{id}.decode cannot reach a transition when max_new_tokens is 1"
        );
        selector.expand(decode_bound, &format!("{id}.decode"))?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Phase {
    Prefill(usize),
    Decode(usize),
}

impl Phase {
    fn label(self) -> &'static str {
        match self {
            Self::Prefill(_) => "prefill",
            Self::Decode(_) => "decode",
        }
    }

    fn index(self) -> usize {
        match self {
            Self::Prefill(index) | Self::Decode(index) => index,
        }
    }
}

fn forward_event(
    execution: &ExecutionPlan,
    forward: &qwen_llm::metal_forward::MetalForward<'_>,
    token: i32,
    position: u32,
    sequence: &mut qwen_llm::runtime::Sequence,
    phase: Phase,
    operation_applications: &mut Vec<OperationApplication>,
    live_readouts: &mut Vec<LiveReadout>,
) -> Result<Vec<f32>> {
    let mut interventions = Vec::new();
    for layer in 0..execution.n_layer {
        for operation in &execution.plan.operations {
            if scope_matches(&operation.scope, phase, layer)? {
                let intervention = action_to_intervention(
                    &operation.id,
                    &operation.action,
                    layer,
                    &execution.directions,
                    &execution.coordinate_swaps,
                )?;
                interventions.push((operation.id.clone(), layer, intervention));
            }
        }
    }
    let borrowed = interventions
        .iter()
        .map(|(_, _, op)| *op)
        .collect::<Vec<_>>();
    let capture = execution.capture.as_ref();
    let logits = if let Some(capture) = capture {
        forward.single_token_with_post_block_interventions(
            token,
            position,
            unsafe { sequence.metal_session_mut() },
            &execution.capture_layers,
            capture,
            &borrowed,
        )?
    } else if borrowed.is_empty() {
        forward.single_token(token, position, unsafe { sequence.metal_session_mut() })?
    } else {
        forward.single_token_with_post_block_interventions_no_capture(
            token,
            position,
            unsafe { sequence.metal_session_mut() },
            &borrowed,
        )?
    };
    sequence.advance_by(1)?;

    for (id, layer, _) in &interventions {
        operation_applications.push(OperationApplication {
            id: id.clone(),
            layer: *layer,
            phase: phase.label(),
            index: phase.index(),
        });
    }
    if let Some(capture) = capture {
        let values = read_f32_tensor(
            capture,
            execution.capture_layers.len() * execution.hidden_size,
        );
        for readout in &execution.plan.readouts {
            for &layer in &execution.capture_layers {
                if !scope_matches(&readout.scope, phase, layer)? {
                    continue;
                }
                let slot = execution.layer_slots[&layer];
                let row = &values[slot * execution.hidden_size..(slot + 1) * execution.hidden_size];
                let prepared = &execution.lenses[&readout.lens];
                let (score_kind, candidate_universe) = readout_score_semantics(prepared);
                let scores = score_readout(prepared, layer, row, readout.top_k)?;
                live_readouts.push(LiveReadout {
                    id: readout.id.clone(),
                    lens: readout.lens.clone(),
                    method: scores.0,
                    score_kind,
                    candidate_universe,
                    source_layer: layer,
                    target_layer: scores.1,
                    phase: phase.label(),
                    index: phase.index(),
                    scores: scores.2,
                });
            }
        }
    }
    Ok(logits)
}

fn scope_matches(scope: &Scope, phase: Phase, layer: u32) -> Result<bool> {
    let index = u32::try_from(phase.index()).context("event index exceeds u32")?;
    let phase_matches = match phase {
        Phase::Prefill(_) => scope
            .prefill
            .as_ref()
            .is_some_and(|selector| selector_contains(selector, index)),
        Phase::Decode(_) => scope
            .decode
            .as_ref()
            .is_some_and(|selector| selector_contains(selector, index)),
    };
    Ok(phase_matches && selector_contains(&scope.layers, layer))
}

fn selector_contains(selector: &Selector, value: u32) -> bool {
    match selector {
        Selector::All => true,
        Selector::Values { values } => values.binary_search(&value).is_ok(),
        Selector::Range { start, end } => (*start..=*end).contains(&value),
    }
}

fn action_to_intervention<'a>(
    operation_id: &str,
    action: &'a Action,
    layer: u32,
    directions: &'a HashMap<String, PreparedDirection>,
    coordinate_swaps: &'a HashMap<String, PreparedDirection>,
) -> Result<PostBlockIntervention<'a>> {
    let direction = |id: &str| -> Result<&'a MetalTensor> {
        directions
            .get(id)
            .and_then(|prepared| prepared.rows.get(&layer))
            .with_context(|| format!("direction {id} has no uploaded row for layer {layer}"))
    };
    Ok(match action {
        Action::FixedAdd {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Fixed {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ResidualL2Fraction {
            direction: id,
            coefficient,
        } => PostBlockIntervention::ResidualL2Relative {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::ProjectionAblate {
            direction: id,
            coefficient,
        } => PostBlockIntervention::Projection {
            layer,
            direction: direction(id)?,
            coefficient: *coefficient,
        },
        Action::SourceToTarget {
            source,
            target,
            coefficient,
        } => PostBlockIntervention::SourceToTarget {
            layer,
            source: direction(source)?,
            target: direction(target)?,
            coefficient: *coefficient,
        },
        Action::CoordinateSwap { coefficient, .. } => PostBlockIntervention::Projection {
            layer,
            direction: coordinate_swaps
                .get(operation_id)
                .and_then(|prepared| prepared.rows.get(&layer))
                .with_context(|| {
                    format!(
                        "coordinate swap {operation_id} has no reflection direction at layer {layer}"
                    )
                })?,
            coefficient: 2.0 * *coefficient,
        },
    })
}

pub(crate) fn read_f32_tensor(tensor: &MetalTensor, length: usize) -> Vec<f32> {
    let mut values = vec![0.0; length];
    unsafe {
        let source = tensor
            .buffer
            .contents()
            .as_ptr()
            .add(tensor.offset as usize) as *const f32;
        std::ptr::copy_nonoverlapping(source, values.as_mut_ptr(), length);
    }
    values
}

fn readout_score_semantics(prepared: &PreparedLens) -> (&'static str, &'static str) {
    match &prepared.lens {
        LoadedLens::Native(_) => (
            "selected_row_projection_numerator",
            "lens_artifact_selected_token_rows",
        ),
        LoadedLens::Template { .. } => ("cosine_similarity", "workspace_template_rows"),
    }
}

fn score_readout(
    prepared: &PreparedLens,
    layer: u32,
    activation: &[f32],
    top_k: usize,
) -> Result<(String, Option<u32>, Vec<LiveScore>)> {
    match &prepared.lens {
        LoadedLens::Native(native) => {
            let layer_slot = native
                .source_layers
                .iter()
                .position(|&candidate| candidate == layer)
                .with_context(|| {
                    format!(
                        "native lens {} has no readout row for layer {layer}",
                        prepared.id
                    )
                })?;
            let mut scores = native
                .token_ids
                .iter()
                .enumerate()
                .map(|(token_slot, &token_id)| {
                    let offset =
                        (layer_slot * native.token_ids.len() + token_slot) * native.hidden_size;
                    let score = activation
                        .iter()
                        .zip(&native.values[offset..offset + native.hidden_size])
                        .map(|(&left, &right)| f64::from(left) * f64::from(right))
                        .sum::<f64>();
                    (token_slot, token_id, score)
                })
                .collect::<Vec<_>>();
            scores.sort_unstable_by(|left, right| {
                right
                    .2
                    .total_cmp(&left.2)
                    .then_with(|| left.1.cmp(&right.1))
            });
            scores.truncate(top_k.min(scores.len()));
            Ok((
                native.method.clone(),
                Some(native.target_layer),
                scores
                    .into_iter()
                    .map(|(row_id, token_id, score)| LiveScore {
                        token_id: Some(token_id),
                        row_id,
                        word_id: None,
                        label: None,
                        score: score as f32,
                    })
                    .collect(),
            ))
        }
        LoadedLens::Template { lens, vocabulary } => {
            let scores: Vec<TemplateScore> = lens
                .score(layer, activation, vocabulary, top_k)
                .map_err(|error| anyhow::anyhow!(error.to_string()))?;
            Ok((
                "workspace_template_cosine".into(),
                None,
                scores
                    .into_iter()
                    .map(|score| LiveScore {
                        token_id: None,
                        row_id: score.row_id,
                        word_id: Some(score.word_id),
                        label: Some(score.text),
                        score: score.score,
                    })
                    .collect(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[derive(Debug, Parser)]
    struct RunArgsParser {
        #[command(flatten)]
        args: LensRunArgs,
    }

    fn minimal_plan() -> LensPlan {
        serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"native_selected","id":"j","artifact":"relative/j"}],
            "directions": [],
            "operations": [],
            "readouts": [{
                "id":"live",
                "lens":"j",
                "scope":{"layers":{"kind":"values","values":[1]},"prefill":{"kind":"all"}},
                "top_k":1
            }]
        }))
        .unwrap()
    }

    fn test_args() -> LensRunArgs {
        RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--prompt",
            "hello",
        ])
        .unwrap()
        .args
    }

    #[test]
    fn run_cli_uses_contextual_default_and_accepts_explicit_json_output() {
        let defaults = test_args();
        assert_eq!(defaults.format, None);
        assert!(defaults.output.is_none());

        let parsed = RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--token-ids",
            "1,2",
            "--output",
            "run.json",
            "--format",
            "json",
        ])
        .unwrap()
        .args;
        assert_eq!(parsed.format, Some(RunStdoutFormat::Json));
        assert_eq!(parsed.output.as_deref(), Some(Path::new("run.json")));
        assert_eq!(
            effective_run_stdout_format(None, false),
            RunStdoutFormat::Json
        );
        assert_eq!(
            effective_run_stdout_format(None, true),
            RunStdoutFormat::Summary
        );
        assert_eq!(
            effective_run_stdout_format(Some(RunStdoutFormat::Summary), false),
            RunStdoutFormat::Summary
        );
    }

    #[test]
    fn run_artifact_serializes_envelope_exact_plan_and_score_semantics() {
        let plan = minimal_plan();
        let artifact = RunOutput {
            schema: RUN_SCHEMA,
            schema_version: RUN_SCHEMA_VERSION,
            runtime_kind: "ordinary_qwen",
            model_path: "model.gguf".into(),
            canonical_plan_path: "/canonical/plan.json".into(),
            requested_live_readouts: plan.readouts.clone(),
            plan,
            input_source: "prompt",
            prompt_token_ids: vec![1, 2],
            generated_token_ids: vec![3],
            sampler: RunSampler {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7,
            },
            max_new_tokens: 1,
            decoded_text: "done".into(),
            stop_reason: "max_new_tokens".into(),
            operation_applications: Vec::new(),
            live_readouts: vec![LiveReadout {
                id: "live".into(),
                lens: "j".into(),
                method: "J".into(),
                score_kind: "selected_row_projection_numerator",
                candidate_universe: "lens_artifact_selected_token_rows",
                source_layer: 1,
                target_layer: Some(2),
                phase: "prefill",
                index: 0,
                scores: Vec::new(),
            }],
            native_hyper_captures: Vec::new(),
        };
        let value = serde_json::to_value(&artifact).unwrap();
        assert_eq!(value["schema"], RUN_SCHEMA);
        assert_eq!(value["schema_version"], 1);
        assert_eq!(value["plan"]["lenses"][0]["artifact"], "relative/j");
        assert_eq!(value["requested_live_readouts"], value["plan"]["readouts"]);
        assert_eq!(
            value["live_readouts"][0]["score_kind"],
            "selected_row_projection_numerator"
        );
        assert_eq!(
            value["live_readouts"][0]["candidate_universe"],
            "lens_artifact_selected_token_rows"
        );
        assert!(value["live_readouts"][0].get("probability").is_none());
    }

    #[test]
    fn summary_contains_required_counts_text_and_artifact_path() {
        let plan = minimal_plan();
        let artifact = RunOutput {
            schema: RUN_SCHEMA,
            schema_version: RUN_SCHEMA_VERSION,
            runtime_kind: "muse_glimmer",
            model_path: "muse.gguf".into(),
            canonical_plan_path: "/canonical/plan.json".into(),
            requested_live_readouts: plan.readouts.clone(),
            plan,
            input_source: "token_ids",
            prompt_token_ids: vec![1],
            generated_token_ids: vec![2],
            sampler: RunSampler {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 0,
            },
            max_new_tokens: 1,
            decoded_text: "line\nbreak".into(),
            stop_reason: "stop_token".into(),
            operation_applications: vec![OperationApplication {
                id: "op".into(),
                layer: 1,
                phase: "prefill",
                index: 0,
            }],
            live_readouts: Vec::new(),
            native_hyper_captures: Vec::new(),
        };
        assert_eq!(
            run_summary(&artifact, Some(Path::new("/tmp/run.json"))),
            concat!(
                "runtime=muse_glimmer model=muse.gguf\n",
                "generated_text=\"line\\nbreak\"\n",
                "stop_reason=stop_token\n",
                "operation_applications=1 live_readouts=0\n",
                "artifact=/tmp/run.json\n"
            )
        );
    }

    fn temporary_direction_path() -> PathBuf {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        std::env::temp_dir().join(format!(
            "qwen-lens-native-hyper-{}-{}.f32le",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn native_hyper_plan(path: &Path, operations: serde_json::Value) -> LensPlan {
        serde_json::from_value(json!({
            "version": 1,
            "lenses": [],
            "directions": [{
                "id": "hyper",
                "source": {
                    "kind": "native_hyper_f32",
                    "path": path,
                    "layer": 23
                }
            }],
            "operations": operations,
            "readouts": []
        }))
        .unwrap()
    }

    #[test]
    fn plan_selectors_expand_sorted_unique_and_inclusive() {
        assert_eq!(Selector::All.expand(4, "layers").unwrap(), vec![0, 1, 2, 3]);
        assert_eq!(
            Selector::Values { values: vec![1, 3] }
                .expand(4, "layers")
                .unwrap(),
            vec![1, 3]
        );
        assert_eq!(
            Selector::Range { start: 1, end: 3 }
                .expand(4, "layers")
                .unwrap(),
            vec![1, 2, 3]
        );
        assert!(
            Selector::Values { values: vec![2, 1] }
                .expand(4, "layers")
                .is_err()
        );
    }

    #[test]
    fn plan_rejects_missing_phase_and_retains_operation_order() {
        let plan: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [],
            "operations": [
                {"id":"second","scope":{"layers":{"kind":"values","values":[2]},"decode":{"kind":"values","values":[0]}},"action":{"kind":"projection_ablate","direction":"d","coefficient":1.0}},
                {"id":"first","scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"fixed_add","direction":"d","coefficient":1.0}}
            ],
            "readouts": []
        }))
        .unwrap();
        assert!(validate_plan(&plan).is_err());

        let plan: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [{"id":"d","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"unit_l2"}],
            "operations": [
                {"id":"second","scope":{"layers":{"kind":"values","values":[2]},"decode":{"kind":"values","values":[0]}},"action":{"kind":"projection_ablate","direction":"d","coefficient":1.0}},
                {"id":"first","scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"values","values":[0]}},"action":{"kind":"fixed_add","direction":"d","coefficient":1.0}}
            ],
            "readouts": []
        }))
        .unwrap();
        validate_plan(&plan).unwrap();
        assert_eq!(plan.operations[0].id, "second");
        assert_eq!(plan.operations[1].id, "first");
    }

    #[test]
    fn plan_scope_requires_reachable_prefill_or_decode() {
        let scope = Scope {
            layers: Selector::All,
            prefill: None,
            decode: Some(Selector::Values { values: vec![1] }),
        };
        assert!(validate_scope_reachable(&scope, 2, 1, "x").is_err());
        assert!(validate_scope_reachable(&scope, 2, 2, "x").is_ok());
        let scope = Scope {
            layers: Selector::All,
            prefill: None,
            decode: Some(Selector::Values { values: vec![2] }),
        };
        assert!(validate_scope_reachable(&scope, 2, 2, "x").is_err());
        let scope = Scope {
            layers: Selector::All,
            prefill: Some(Selector::Values { values: vec![1] }),
            decode: None,
        };
        assert!(validate_scope_reachable(&scope, 2, 1, "x").is_ok());
    }

    #[test]
    fn plan_relative_paths_resolve_against_the_plan_directory() {
        let base = Path::new("/tmp/lens-plan");
        assert_eq!(
            resolve_plan_path(base, Path::new("artifacts/readouts.json")),
            PathBuf::from("/tmp/lens-plan/artifacts/readouts.json")
        );
        assert_eq!(
            resolve_plan_path(base, Path::new("/absolute/weights.safetensors")),
            PathBuf::from("/absolute/weights.safetensors")
        );
    }

    #[test]
    fn plan_file_parser_accepts_fractional_coefficients() {
        let plan = parse_plan_bytes(
            br#"{
                "version": 1,
                "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
                "directions": [{"id":"d","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"as_stored"}],
                "operations": [{"id":"add","scope":{"layers":{"kind":"values","values":[0]},"prefill":{"kind":"all"}},"action":{"kind":"fixed_add","direction":"d","coefficient":0.0001}}],
                "readouts": []
            }"#,
        )
        .unwrap();
        assert_eq!(plan.operations[0].action.coefficient(), 0.0001);
    }

    #[test]
    fn coordinate_swap_requires_distinct_unit_directions() {
        let plan = |source: &str, target: &str, target_normalization: &str| {
            serde_json::from_value::<LensPlan>(json!({
                "version": 1,
                "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
                "directions": [
                    {"id":"source","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"unit_l2"},
                    {"id":"target","lens":"t","row":{"kind":"template_row_id","template_row_id":1},"normalization":target_normalization}
                ],
                "operations": [{
                    "id":"swap",
                    "scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"all"}},
                    "action":{"kind":"coordinate_swap","source":source,"target":target,"coefficient":1.0}
                }],
                "readouts": []
            }))
            .unwrap()
        };

        validate_plan(&plan("source", "target", "unit_l2")).unwrap();
        assert!(validate_plan(&plan("source", "source", "unit_l2")).is_err());
        assert!(validate_plan(&plan("source", "target", "as_stored")).is_err());
    }

    #[test]
    fn coordinate_swap_reflection_exchanges_two_lens_coordinates() {
        let source = [1.0_f32, 0.0, 0.0];
        let target = [0.6_f32, 0.8, 0.0];
        let mut activation = [2.0_f32, -1.0, 5.0];
        let source_before = activation
            .iter()
            .zip(source)
            .map(|(&x, v)| x * v)
            .sum::<f32>();
        let target_before = activation
            .iter()
            .zip(target)
            .map(|(&x, v)| x * v)
            .sum::<f32>();
        let reflection = coordinate_swap_reflection_direction(&source, &target, "test").unwrap();
        let projection = activation
            .iter()
            .zip(&reflection)
            .map(|(&x, &u)| x * u)
            .sum::<f32>();
        for (value, &direction) in activation.iter_mut().zip(&reflection) {
            *value -= 2.0 * projection * direction;
        }
        let source_after = activation
            .iter()
            .zip(source)
            .map(|(&x, v)| x * v)
            .sum::<f32>();
        let target_after = activation
            .iter()
            .zip(target)
            .map(|(&x, v)| x * v)
            .sum::<f32>();
        assert!((source_after - target_before).abs() < 1e-5);
        assert!((target_after - source_before).abs() < 1e-5);
        assert!((activation[2] - 5.0).abs() < 1e-6);
        assert!(coordinate_swap_reflection_direction(&source, &[2.0, 0.0, 0.0], "bad").is_err());
    }

    #[test]
    fn published_full_transport_plan_requires_explicit_transfer_and_unique_tokens() {
        let plan = |token_ids: serde_json::Value, allow_unvalidated_transfer: bool| {
            serde_json::from_value::<LensPlan>(json!({
                "version": 1,
                "lenses": [{
                    "kind": "published_full_transport",
                    "id": "j",
                    "artifact": "published",
                    "token_ids": token_ids,
                    "allow_unvalidated_transfer": allow_unvalidated_transfer
                }],
                "directions": [{
                    "id": "concept",
                    "lens": "j",
                    "row": {"kind": "token_id", "token_id": 42},
                    "normalization": "unit_l2"
                }],
                "operations": [{
                    "id": "add",
                    "scope": {
                        "layers": {"kind": "values", "values": [31]},
                        "prefill": {"kind": "all"}
                    },
                    "action": {
                        "kind": "residual_l2_fraction",
                        "direction": "concept",
                        "coefficient": 0.01
                    }
                }],
                "readouts": []
            }))
            .unwrap()
        };

        let valid = plan(json!([42, 43]), true);
        validate_plan(&valid).unwrap();
        validate_ordinary_plan(&valid).unwrap();
        assert!(matches!(
            &valid.lenses[0],
            LensDefinition::PublishedFullTransport { token_ids, .. } if token_ids == &[42, 43]
        ));

        assert!(validate_plan(&plan(json!([42, 43]), false)).is_err());
        assert!(validate_plan(&plan(json!([42, 42]), true)).is_err());
        assert!(validate_plan(&plan(json!([]), true)).is_err());
        assert!(validate_plan(&plan(json!([43]), true)).is_err());

        let legacy: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{
                "kind": "published_full_j",
                "id": "j",
                "artifact": "published",
                "token_ids": [42],
                "allow_unvalidated_transfer": true
            }],
            "directions": [],
            "operations": [],
            "readouts": [{
                "id": "read",
                "lens": "j",
                "scope": {"layers":{"kind":"values","values":[31]},"prefill":{"kind":"all"}},
                "top_k": 1
            }]
        }))
        .unwrap();
        assert!(matches!(
            legacy.lenses[0],
            LensDefinition::PublishedFullTransport { .. }
        ));
    }

    #[test]
    fn native_payload_indexing_is_layer_token_hidden_major() {
        let values = (0..2 * 3 * 4).map(|value| value as f32).collect::<Vec<_>>();
        let offset = native_payload_offset(1, 2, 3, 4);
        assert_eq!(&values[offset..offset + 4], &[20.0, 21.0, 22.0, 23.0]);
    }

    #[test]
    fn native_hyper_direction_syntax_is_additive_and_fail_closed() {
        let native = native_hyper_plan(
            Path::new("direction.f32le"),
            json!([{
                "id": "add",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
            }]),
        );
        validate_plan(&native).unwrap();
        assert!(native.directions[0].native_hyper().is_some());
        assert!(validate_ordinary_plan(&native).is_err());

        let legacy: LensPlan = serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [{"id":"d","lens":"t","row":{"kind":"template_row_id","template_row_id":0},"normalization":"as_stored"}],
            "operations": [],
            "readouts": []
        }))
        .unwrap();
        validate_plan(&legacy).unwrap();
        validate_ordinary_plan(&legacy).unwrap();
        assert!(legacy.directions[0].lens_row().is_some());

        let mixed = serde_json::from_value::<LensPlan>(json!({
            "version": 1,
            "lenses": [],
            "directions": [{
                "id": "bad",
                "lens": "x",
                "row": {"kind": "token_id", "token_id": 1},
                "normalization": "as_stored",
                "source": {"kind": "native_hyper_f32", "path": "x", "layer": 23}
            }],
            "operations": [],
            "readouts": []
        }));
        assert!(mixed.is_err());
    }

    #[test]
    fn native_hyper_payload_requires_exact_finite_nonzero_f32() {
        const WIDTH: usize = 10_240;
        let path = temporary_direction_path();
        let mut values = vec![0.0_f32; WIDTH];
        values[17] = 1.0;
        std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
        let loaded = load_native_hyper_direction(&path, WIDTH, "hyper").unwrap();
        assert_eq!(loaded, values);

        std::fs::write(&path, bytemuck::cast_slice(&values[..WIDTH - 1])).unwrap();
        assert!(load_native_hyper_direction(&path, WIDTH, "hyper").is_err());

        values.fill(0.0);
        std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
        assert!(load_native_hyper_direction(&path, WIDTH, "hyper").is_err());

        values[0] = f32::NAN;
        std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
        assert!(load_native_hyper_direction(&path, WIDTH, "hyper").is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn flash_plan_rejects_wrong_actions_layers_overlap_and_capture_excess() {
        const WIDTH: usize = 10_240;
        let path = temporary_direction_path();
        let mut values = vec![0.0_f32; WIDTH];
        values[0] = 1.0;
        std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
        let config = Qwen4ExpConfig::flash_next_reference();

        let valid = native_hyper_plan(
            &path,
            json!([{
                "id": "add",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
            }]),
        );
        validate_plan(&valid).unwrap();
        let execution = prepare_qwen4exp_execution_plan(valid, Path::new("/"), &config).unwrap();
        validate_qwen4exp_event_schedule(&execution, 1, 1).unwrap();
        assert_eq!(execution.directions["hyper"].layer, 23);

        let wrong_layer = native_hyper_plan(
            &path,
            json!([{
                "id": "add",
                "scope": {
                    "layers": {"kind": "values", "values": [22]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
            }]),
        );
        assert!(prepare_qwen4exp_execution_plan(wrong_layer, Path::new("/"), &config).is_err());

        let wrong_action = native_hyper_plan(
            &path,
            json!([{
                "id": "project",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "projection_ablate", "direction": "hyper", "coefficient": 1.0}
            }]),
        );
        validate_plan(&wrong_action).unwrap();
        assert!(prepare_qwen4exp_execution_plan(wrong_action, Path::new("/"), &config).is_err());

        let overlapping = native_hyper_plan(
            &path,
            json!([
                {
                    "id": "first",
                    "scope": {
                        "layers": {"kind": "values", "values": [23]},
                        "prefill": {"kind": "values", "values": [0]}
                    },
                    "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
                },
                {
                    "id": "second",
                    "scope": {
                        "layers": {"kind": "values", "values": [23]},
                        "prefill": {"kind": "values", "values": [0]}
                    },
                    "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.5}
                }
            ]),
        );
        let execution =
            prepare_qwen4exp_execution_plan(overlapping, Path::new("/"), &config).unwrap();
        assert!(validate_qwen4exp_event_schedule(&execution, 1, 1).is_err());

        let excessive = native_hyper_plan(
            &path,
            json!([{
                "id": "many",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "all"}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
            }]),
        );
        let execution =
            prepare_qwen4exp_execution_plan(excessive, Path::new("/"), &config).unwrap();
        validate_qwen4exp_event_schedule(&execution, 32, 1).unwrap();
        assert!(validate_qwen4exp_event_schedule(&execution, 33, 1).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn flash_decode_schedule_and_capture_metadata_are_explicit() {
        const WIDTH: usize = 10_240;
        let path = temporary_direction_path();
        let mut values = vec![0.0_f32; WIDTH];
        values[0] = 1.0;
        std::fs::write(&path, bytemuck::cast_slice(&values)).unwrap();
        let plan = native_hyper_plan(
            &path,
            json!([{
                "id": "decode-add",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "decode": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": 0.25}
            }]),
        );
        validate_reachable_scopes(&plan, 1, 2).unwrap();
        let execution = prepare_qwen4exp_execution_plan(
            plan,
            Path::new("/"),
            &Qwen4ExpConfig::flash_next_reference(),
        )
        .unwrap();
        validate_qwen4exp_event_schedule(&execution, 1, 2).unwrap();
        assert!(
            qwen4exp_matching_operation(&execution, Phase::Prefill(0))
                .unwrap()
                .is_none()
        );
        let (operation, layer) = qwen4exp_matching_operation(&execution, Phase::Decode(0))
            .unwrap()
            .unwrap();
        assert_eq!(operation.id, "decode-add");
        assert_eq!(layer, 23);

        let capture = NativeHyperCapture {
            operation_id: "decode-add".into(),
            layer: 23,
            phase: "decode",
            index: 0,
            position: 1,
            coordinate: "qwen4exp_persistent_post_layer_hyper_state",
            capture_stage: "after_fixed_add",
            shape: [4, 2_560],
            flattening: "branch_major_hidden_minor",
            direction_normalization: "as_stored",
            coefficient: 0.25,
            values: vec![1.0, 2.0],
        };
        let encoded = serde_json::to_value(capture).unwrap();
        assert_eq!(encoded["capture_stage"], "after_fixed_add");
        assert_eq!(encoded["direction_normalization"], "as_stored");
        assert!(encoded.get("normalization").is_none());
        assert_eq!(encoded["shape"], json!([4, 2560]));
        std::fs::remove_file(path).unwrap();
    }
}
