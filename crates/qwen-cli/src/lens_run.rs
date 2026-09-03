use crate::lens_input::{
    LensInputRendering, LensInputSpec, LensMessageMode, LensRenderedSpan, PreparedLensInput,
    is_known_lens_span, prepare_qwen_model_input, prepare_qwen_model_messages_bytes,
    validate_lens_input_spec,
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
use qwen_llm::runtime::{
    LoadedModelConfig, ModelLoadIntent, PackedPrefillScratch, Runtime, SequenceConfig,
};
use qwen_llm::sampling::{Sampler, SamplingConfig};
use qwen_llm::tensor::GgmlType;
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::ffi::CString;
use std::fs::{DirBuilder, OpenOptions};
use std::io::Write;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

mod event_schedule;

use event_schedule::{BoundEventSchedule, CompiledEvent, CompiledEventSchedule};

const MAX_PLAN_BYTES: usize = 16 * 1024 * 1024;
const MAX_LENSES: usize = 64;
const MAX_DIRECTIONS: usize = 4096;
const MAX_OPERATIONS: usize = 4096;
const MAX_READOUTS: usize = 1024;
const MAX_SELECTOR_VALUES: usize = 4096;
const MAX_RENDERED_SELECTOR_TEXT_BYTES: usize = 1024;
const MAX_RENDERED_SELECTORS_PER_PLAN: usize = 1024;
const MAX_RENDERED_SELECTOR_MATCH_WORK: usize = 1_000_000;
const MAX_TOP_K: usize = 1024;
const MAX_NATIVE_HYPER_CAPTURES: usize = 32;
const MAX_PUBLISHED_FULL_TOKEN_IDS: usize = 32;
pub(crate) const MAX_RUN_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;
const RUN_SCHEMA: &str = "qwen.lens.run";
const RUN_SCHEMA_VERSION: u32 = 5;
const SWEEP_SCHEMA: &str = "qwen.lens.coefficient_sweep";
const SWEEP_SCHEMA_VERSION: u32 = 3;
const SWEEP_MANIFEST_NAME: &str = "manifest.json";
const MAX_SWEEP_ARMS: usize = 64;
const SWEEP_COHORT_SCHEMA: &str = "qwen.lens.coefficient_sweep_cohort";
const SWEEP_COHORT_SCHEMA_VERSION: u32 = 1;
const MIN_SWEEP_COHORT_REQUESTS: usize = 2;
const MAX_SWEEP_COHORT_RECORD_BYTES: usize = 1024 * 1024;
const MAX_SWEEP_COHORT_FILE_BYTES: usize = 32 * 1024 * 1024;
const MAX_SWEEP_COHORT_ID_BYTES: usize = 128;
const MAX_SWEEP_COHORT_MESSAGES_BYTES: usize = 16 * 1024 * 1024;
const MAX_SWEEP_COHORT_TRANSITION_UPPER_BOUND: u64 = 1_000_000;
pub(crate) const MAX_SWEEP_BUNDLE_BYTES: u64 = 512 * 1024 * 1024;
const PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS: usize = 65;
const PACKED_PREFILL_CHUNK_CAP_TOKENS: usize = 1024;

pub(crate) fn required_forward_count(prompt_tokens: usize, max_new_tokens: usize) -> Result<usize> {
    ensure!(
        prompt_tokens > 0,
        "prompt must encode to at least one token"
    );
    ensure!(max_new_tokens > 0, "--max-new-tokens must be positive");
    prompt_tokens
        .checked_add(max_new_tokens - 1)
        .context("request forward count overflow")
}

fn ensure_request_fits_context(
    prompt_tokens: usize,
    max_new_tokens: usize,
    model_context_tokens: usize,
) -> Result<usize> {
    let required_forwards = required_forward_count(prompt_tokens, max_new_tokens)?;
    ensure!(
        required_forwards <= model_context_tokens,
        "request requires {required_forwards} token forwards ({prompt_tokens} prompt + {} maximum decode transitions), exceeding model context {model_context_tokens}",
        max_new_tokens - 1,
    );
    Ok(required_forwards)
}

pub(crate) fn ensure_qwen_sequence_admitted(
    loaded: &qwen_llm::runtime::LoadedModel,
    capacity: usize,
) -> Result<()> {
    let admission = loaded
        .qwen_execution_memory_admission(1, capacity, 0, 0)
        .context("price Lens sequence memory")?;
    ensure!(
        admission.admitted,
        "Lens sequence memory admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );
    Ok(())
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub(crate) enum PrefillExecution {
    Auto,
    Serial,
}

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("lens_input")
        .required(true)
        .multiple(false)
        .args(["prompt", "token_ids", "user", "messages", "open_responses"])
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

    /// Raw untemplated text; tokenizer-configured specials are enabled by default.
    #[arg(long, visible_alias = "raw-prompt", allow_hyphen_values = true)]
    pub(crate) prompt: Option<String>,

    /// Literal comma-separated token IDs; no specials are added.
    #[arg(long, value_delimiter = ',')]
    pub(crate) token_ids: Option<Vec<i32>>,

    /// One user message rendered with the model-family template; '-' reads stdin.
    #[arg(long, value_name = "TEXT|-")]
    pub(crate) user: Option<String>,

    /// Add one system message before --user.
    #[arg(long, requires = "user")]
    pub(crate) system: Option<String>,

    /// JSON message array or wrapper with a `messages` array.
    #[arg(long, value_name = "FILE|-")]
    pub(crate) messages: Option<PathBuf>,

    /// Open Responses request JSON rendered by the exact qwen serve prompt path.
    #[arg(long, visible_alias = "responses-input", value_name = "FILE|-")]
    pub(crate) open_responses: Option<PathBuf>,

    /// Generation transition for --user/--messages; supported values depend on the model.
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["prompt", "token_ids", "open_responses"]
    )]
    pub(crate) message_mode: Option<LensMessageMode>,

    /// Disable tokenizer-configured specials for --prompt.
    #[arg(
        long,
        requires = "prompt",
        conflicts_with_all = ["token_ids", "user", "messages", "open_responses"]
    )]
    pub(crate) no_special_tokens: bool,

    /// Maximum number of generated tokens.
    #[arg(long, default_value_t = 32)]
    pub(crate) max_new_tokens: usize,

    /// Use qualified packed passive spans when safe, or force the serial reference path.
    #[arg(long, value_enum, default_value_t = PrefillExecution::Auto)]
    pub(crate) prefill_execution: PrefillExecution,

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

impl LensRunArgs {
    pub(crate) fn input_spec(&self) -> LensInputSpec<'_> {
        LensInputSpec {
            prompt: self.prompt.as_deref(),
            token_ids: self.token_ids.as_deref(),
            user: self.user.as_deref(),
            system: self.system.as_deref(),
            messages: self.messages.as_deref(),
            open_responses: self.open_responses.as_deref(),
            no_special_tokens: self.no_special_tokens,
            message_mode: self.message_mode,
        }
    }
}

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
    /// Ordinary dense or MoE Qwen GGUF model, loaded once for every arm.
    #[arg(short = 'm', long)]
    model: PathBuf,

    /// Strict Lens plan JSON file whose authored coefficients remain unchanged.
    #[arg(long)]
    plan: PathBuf,

    /// Exact operation ID whose coefficient is replaced in each arm.
    #[arg(long)]
    operation: String,

    /// Ordered finite coefficients; duplicates and zero controls are preserved.
    #[arg(
        long,
        value_delimiter = ',',
        required = true,
        allow_hyphen_values = true
    )]
    coefficients: Vec<f32>,

    /// Raw untemplated text; tokenizer-configured specials are enabled by default.
    #[arg(long, visible_alias = "raw-prompt", allow_hyphen_values = true)]
    prompt: Option<String>,

    /// Literal comma-separated token IDs; no specials are added.
    #[arg(long, value_delimiter = ',')]
    token_ids: Option<Vec<i32>>,

    /// One user message rendered with the model-family template; '-' reads stdin once.
    #[arg(long, value_name = "TEXT|-")]
    user: Option<String>,

    /// Add one system message before --user.
    #[arg(long, requires = "user")]
    system: Option<String>,

    /// JSON message array or wrapper with a `messages` array.
    #[arg(long, value_name = "FILE|-")]
    messages: Option<PathBuf>,

    /// Open Responses request JSON rendered by the exact qwen serve prompt path.
    #[arg(long, visible_alias = "responses-input", value_name = "FILE|-")]
    open_responses: Option<PathBuf>,

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
    requests_jsonl: Option<PathBuf>,

    /// Generation transition for --user/--messages; supported values depend on the model.
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["prompt", "token_ids", "open_responses", "requests_jsonl"]
    )]
    message_mode: Option<LensMessageMode>,

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
    no_special_tokens: bool,

    /// Maximum number of generated tokens per fresh arm.
    #[arg(long, default_value_t = 32)]
    max_new_tokens: usize,

    /// Use one qualified passive-span schedule for every arm, or force serial prefill.
    #[arg(long, value_enum, default_value_t = PrefillExecution::Auto)]
    prefill_execution: PrefillExecution,

    /// Native sampler temperature; each arm restarts from the same seed.
    #[arg(long, default_value_t = 0.0)]
    temperature: f32,

    /// Native sampler top-k; zero disables the filter.
    #[arg(long, default_value_t = 0)]
    top_k: usize,

    /// Native sampler nucleus threshold.
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Native sampler minimum probability threshold.
    #[arg(long, default_value_t = 0.0)]
    min_p: f32,

    /// Native sampler seed, reset for every arm.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// New immutable sweep directory, published only after every arm succeeds.
    #[arg(long)]
    output: PathBuf,
}

impl CoefficientSweepArgs {
    fn arm_run_args(&self) -> LensRunArgs {
        LensRunArgs {
            model: self.model.clone(),
            plan: self.plan.clone(),
            identity_cache: None,
            prompt: self.prompt.clone(),
            token_ids: self.token_ids.clone(),
            user: self.user.clone(),
            system: self.system.clone(),
            messages: self.messages.clone(),
            open_responses: self.open_responses.clone(),
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
        }
    }
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) target_covector: Option<DirectionTargetCovector>,
    pub(crate) normalization: Normalization,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DirectionTargetCovector {
    DeployedLogitNumerator,
    RawLmHead,
    RawLmHeadOrthogonalToDeployedLogitNumerator,
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

impl LensRowDirectionDefinition {
    fn effective_target_covector(&self) -> DirectionTargetCovector {
        self.target_covector
            .unwrap_or(DirectionTargetCovector::DeployedLogitNumerator)
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
    pub(crate) fn coefficient(&self) -> f32 {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient,
        }
    }

    fn set_coefficient(&mut self, value: f32) {
        match self {
            Self::FixedAdd { coefficient, .. }
            | Self::ResidualL2Fraction { coefficient, .. }
            | Self::ProjectionAblate { coefficient, .. }
            | Self::SourceToTarget { coefficient, .. }
            | Self::CoordinateSwap { coefficient, .. } => *coefficient = value,
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
    Values {
        values: Vec<u32>,
    },
    Range {
        start: u32,
        end: u32,
    },
    RenderedSpans {
        selectors: Vec<RenderedSpanSelector>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RenderedSpanSelector {
    pub(crate) span_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) message_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    #[serde(default)]
    pub(crate) occurrence: RenderedSpanOccurrence,
    pub(crate) edge: RenderedSpanEdge,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RenderedSpanOccurrence {
    #[default]
    Unique,
    First,
    Last,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RenderedSpanEdge {
    Start,
    End,
}

impl Selector {
    fn validate(&self, name: &str, allow_rendered_spans: bool) -> Result<()> {
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
            Self::RenderedSpans { selectors } => {
                ensure!(
                    allow_rendered_spans,
                    "{name} does not support rendered-span selectors"
                );
                ensure!(
                    !selectors.is_empty() && selectors.len() <= MAX_SELECTOR_VALUES,
                    "{name} rendered_spans requires 1..={MAX_SELECTOR_VALUES} selectors"
                );
                ensure!(
                    selectors.iter().collect::<BTreeSet<_>>().len() == selectors.len(),
                    "{name} repeats an authored rendered-span selector"
                );
                for (index, selector) in selectors.iter().enumerate() {
                    selector.validate(&format!("{name}.selectors[{index}]"))?;
                }
                Ok(())
            }
        }
    }

    fn expand(&self, upper_bound: u32, name: &str) -> Result<Vec<u32>> {
        self.validate(name, false)?;
        let values = match self {
            Self::All => (0..upper_bound).collect(),
            Self::Values { values } => values.clone(),
            Self::Range { start, end } => (*start..=*end).collect(),
            Self::RenderedSpans { .. } => unreachable!("validation rejects unresolved selectors"),
        };
        ensure!(
            values.iter().all(|&value| value < upper_bound),
            "{name} contains a value outside 0..{upper_bound}"
        );
        Ok(values)
    }
}

impl RenderedSpanSelector {
    fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            is_known_lens_span(&self.span_kind),
            "{name}.span_kind is not a known renderer span"
        );
        ensure!(
            self.message_index
                .is_none_or(|index| index < MAX_SELECTOR_VALUES)
                && self
                    .tool_call_index
                    .is_none_or(|index| index < MAX_SELECTOR_VALUES),
            "{name} message/tool-call index exceeds the selector bound"
        );
        ensure!(
            self.role
                .as_deref()
                .is_none_or(|role| { matches!(role, "system" | "user" | "assistant" | "tool") }),
            "{name}.role is unsupported"
        );
        ensure!(
            self.channel.as_deref().is_none_or(|channel| {
                matches!(channel, "thinking" | "tool_call" | "tool_result")
            }),
            "{name}.channel is unsupported"
        );
        for (field, value) in [
            ("role", self.role.as_deref()),
            ("channel", self.channel.as_deref()),
            ("label", self.label.as_deref()),
        ] {
            ensure!(
                value.is_none_or(|value| {
                    !value.is_empty() && value.len() <= MAX_RENDERED_SELECTOR_TEXT_BYTES
                }),
                "{name}.{field} is empty or too long"
            );
        }
        Ok(())
    }

    fn matches(&self, span: &LensRenderedSpan) -> bool {
        span.kind == self.span_kind
            && self
                .message_index
                .is_none_or(|value| span.message_index == Some(value))
            && self
                .tool_call_index
                .is_none_or(|value| span.tool_call_index == Some(value))
            && self
                .role
                .as_deref()
                .is_none_or(|value| span.role.as_deref() == Some(value))
            && self
                .channel
                .as_deref()
                .is_none_or(|value| span.channel.as_deref() == Some(value))
            && self
                .label
                .as_deref()
                .is_none_or(|value| span.label.as_deref() == Some(value))
    }
}

impl Scope {
    fn validate(&self, name: &str, plan_version: u32) -> Result<()> {
        self.layers.validate(&format!("{name}.layers"), false)?;
        ensure!(
            self.prefill.is_some() || self.decode.is_some(),
            "{name} must select prefill and/or decode"
        );
        if let Some(selector) = &self.prefill {
            selector.validate(&format!("{name}.prefill"), plan_version == 2)?;
        }
        if let Some(selector) = &self.decode {
            selector.validate(&format!("{name}.decode"), false)?;
        }
        Ok(())
    }
}

pub(crate) fn bind_plan_positions(
    authored: &LensPlan,
    rendering: &LensInputRendering,
    prompt_len: usize,
) -> Result<BoundLensPlan> {
    validate_plan_position_selectors(authored)?;
    let rendered_selector_count = authored
        .operations
        .iter()
        .map(|operation| operation.scope.rendered_selector_count())
        .chain(
            authored
                .readouts
                .iter()
                .map(|readout| readout.scope.rendered_selector_count()),
        )
        .try_fold(0usize, |total, count| total.checked_add(count))
        .context("rendered-selector count overflow")?;
    ensure!(
        rendered_selector_count <= MAX_RENDERED_SELECTORS_PER_PLAN,
        "Lens plan has {rendered_selector_count} rendered selectors; limit is {MAX_RENDERED_SELECTORS_PER_PLAN}"
    );
    if rendered_selector_count > 0 {
        let match_work = rendered_selector_count
            .checked_mul(rendering.spans.len())
            .context("rendered-selector match-work overflow")?;
        ensure!(
            match_work <= MAX_RENDERED_SELECTOR_MATCH_WORK,
            "rendered selectors require {match_work} span comparisons; limit is {MAX_RENDERED_SELECTOR_MATCH_WORK}"
        );
    }
    let mut resolved = authored.clone();
    let mut position_bindings = Vec::new();
    for operation in &mut resolved.operations {
        bind_scope_prefill(
            &mut operation.scope,
            "operation",
            &operation.id,
            rendering,
            prompt_len,
            &mut position_bindings,
        )?;
    }
    for readout in &mut resolved.readouts {
        bind_scope_prefill(
            &mut readout.scope,
            "readout",
            &readout.id,
            rendering,
            prompt_len,
            &mut position_bindings,
        )?;
    }
    Ok(BoundLensPlan {
        authored: authored.clone(),
        resolved,
        authored_plan_canonical_json_blake3: canonical_plan_blake3(authored)?,
        position_bindings,
    })
}

impl Scope {
    fn rendered_selector_count(&self) -> usize {
        match &self.prefill {
            Some(Selector::RenderedSpans { selectors }) => selectors.len(),
            _ => 0,
        }
    }
}

fn validate_plan_position_selectors(plan: &LensPlan) -> Result<()> {
    ensure!(
        matches!(plan.version, 1 | 2),
        "Lens plan version must be 1 or 2"
    );
    ensure!(
        plan.operations.len() <= MAX_OPERATIONS && plan.readouts.len() <= MAX_READOUTS,
        "Lens plan has too many operations or readouts"
    );
    for operation in &plan.operations {
        operation
            .scope
            .validate(&format!("operation {} scope", operation.id), plan.version)?;
    }
    for readout in &plan.readouts {
        readout
            .scope
            .validate(&format!("readout {} scope", readout.id), plan.version)?;
    }
    Ok(())
}

fn bind_scope_prefill(
    scope: &mut Scope,
    owner_kind: &str,
    owner_id: &str,
    rendering: &LensInputRendering,
    prompt_len: usize,
    bindings: &mut Vec<PositionBinding>,
) -> Result<()> {
    let Some(Selector::RenderedSpans { selectors }) = scope.prefill.as_ref() else {
        return Ok(());
    };
    ensure!(
        !rendering.spans.is_empty(),
        "{owner_kind} {owner_id} rendered-span prefill selectors require renderer-authored spans; raw text and literal token IDs have none"
    );
    let selectors = selectors.clone();
    let mut values = BTreeSet::new();
    for (selector_index, selector) in selectors.into_iter().enumerate() {
        let mut match_count = 0usize;
        let mut first_match = None;
        let mut last_match = None;
        for matched in rendering
            .spans
            .iter()
            .enumerate()
            .filter(|(_, span)| selector.matches(span))
        {
            match_count += 1;
            first_match.get_or_insert(matched);
            last_match = Some(matched);
        }
        ensure!(
            match_count > 0,
            "{owner_kind} {owner_id} rendered selector {selector_index} matched no authored span"
        );
        let (rendering_span_index, span) = match selector.occurrence {
            RenderedSpanOccurrence::Unique => {
                ensure!(
                    match_count == 1,
                    "{owner_kind} {owner_id} rendered selector {selector_index} matched {} spans; choose first or last explicitly",
                    match_count
                );
                first_match.expect("nonempty matches")
            }
            RenderedSpanOccurrence::First => first_match.expect("nonempty matches"),
            RenderedSpanOccurrence::Last => last_match.expect("nonempty matches"),
        };
        let (token_start, token_end) = span
            .token_start
            .zip(span.token_end)
            .with_context(|| {
                format!(
                    "{owner_kind} {owner_id} rendered selector {selector_index} matched span {rendering_span_index} without an exact token range"
                )
            })?;
        ensure!(
            token_start < token_end && token_end <= prompt_len,
            "{owner_kind} {owner_id} rendered selector {selector_index} matched an invalid token range"
        );
        let position = match selector.edge {
            RenderedSpanEdge::Start => token_start,
            RenderedSpanEdge::End => token_end - 1,
        };
        let resolved_index =
            u32::try_from(position).context("resolved prefill position exceeds u32")?;
        ensure!(
            values.insert(resolved_index),
            "{owner_kind} {owner_id} rendered selectors resolve repeatedly to prefill position {position}"
        );
        bindings.push(PositionBinding {
            owner_kind: owner_kind.into(),
            owner_id: owner_id.into(),
            phase: "prefill".into(),
            selector_index,
            selector,
            rendering_span_index,
            matched_span: span.clone(),
            resolved_index,
            absolute_position: position,
        });
    }
    scope.prefill = Some(Selector::Values {
        values: values.into_iter().collect(),
    });
    Ok(())
}

pub(crate) fn canonical_plan_blake3(plan: &LensPlan) -> Result<String> {
    let bytes = serde_json::to_vec(plan).context("serialize canonical Lens plan JSON")?;
    ensure!(
        bytes.len() <= MAX_PLAN_BYTES,
        "canonical Lens plan exceeds limit"
    );
    Ok(blake3::hash(&bytes).to_hex().to_string())
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct PositionBinding {
    pub(crate) owner_kind: String,
    pub(crate) owner_id: String,
    pub(crate) phase: String,
    pub(crate) selector_index: usize,
    pub(crate) selector: RenderedSpanSelector,
    pub(crate) rendering_span_index: usize,
    pub(crate) matched_span: LensRenderedSpan,
    pub(crate) resolved_index: u32,
    pub(crate) absolute_position: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct BoundLensPlan {
    pub(crate) authored: LensPlan,
    pub(crate) resolved: LensPlan,
    pub(crate) authored_plan_canonical_json_blake3: String,
    pub(crate) position_bindings: Vec<PositionBinding>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunEffectivePrefill {
    Serial,
    DensePackedPassiveSpans,
}

impl RunEffectivePrefill {
    fn as_str(self) -> &'static str {
        match self {
            Self::Serial => "serial",
            Self::DensePackedPassiveSpans => "dense_packed_passive_spans",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunExecutionScheduleBasis {
    EffectivePlan,
    SweepSourcePlan,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunNumericalRelationship {
    SerialReference,
    PackedReductionTopologyDiffersFromSerial,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunSerialReason {
    RequestedSerial,
    NoEligiblePassiveSpan,
    DensePackedMemoryAdmissionDenied,
    MoePackedNotQualified,
    CohortSerialPolicy,
    FlashNextPackedNotImplemented,
    MusePackedNotImplemented,
}

impl RunSerialReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::RequestedSerial => "requested_serial",
            Self::NoEligiblePassiveSpan => "no_eligible_passive_span",
            Self::DensePackedMemoryAdmissionDenied => "dense_packed_memory_admission_denied",
            Self::MoePackedNotQualified => "moe_packed_not_qualified",
            Self::CohortSerialPolicy => "cohort_serial_policy",
            Self::FlashNextPackedNotImplemented => "flash_next_packed_not_implemented",
            Self::MusePackedNotImplemented => "muse_packed_not_implemented",
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunPackedPrefillSpan {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunExecution {
    requested_prefill: PrefillExecution,
    effective_prefill: RunEffectivePrefill,
    schedule_basis: RunExecutionScheduleBasis,
    numerical_relationship: RunNumericalRelationship,
    minimum_span_tokens: usize,
    chunk_cap_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    block_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attention_matrix_max_position: Option<u64>,
    scratch_priced_upper_bytes: u64,
    packed_spans: Vec<RunPackedPrefillSpan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    serial_reason: Option<RunSerialReason>,
}

fn expected_packed_prefill_spans(
    plan: &LensPlan,
    prompt_len: usize,
) -> Result<Vec<RunPackedPrefillSpan>> {
    let schedule = CompiledEventSchedule::compile(plan, u32::MAX)?;
    Ok(schedule
        .bind(plan)?
        .passive_prefill_spans(prompt_len, PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS)?
        .into_iter()
        .map(|span| RunPackedPrefillSpan {
            start: span.start,
            end: span.end,
        })
        .collect())
}

impl RunExecution {
    pub(crate) fn serial(
        requested_prefill: PrefillExecution,
        schedule_basis: RunExecutionScheduleBasis,
        automatic_reason: RunSerialReason,
    ) -> Self {
        let serial_reason = if requested_prefill == PrefillExecution::Serial {
            RunSerialReason::RequestedSerial
        } else {
            automatic_reason
        };
        Self {
            requested_prefill,
            effective_prefill: RunEffectivePrefill::Serial,
            schedule_basis,
            numerical_relationship: RunNumericalRelationship::SerialReference,
            minimum_span_tokens: PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS,
            chunk_cap_tokens: PACKED_PREFILL_CHUNK_CAP_TOKENS,
            block_tokens: None,
            attention_matrix_max_position: None,
            scratch_priced_upper_bytes: 0,
            packed_spans: Vec::new(),
            serial_reason: Some(serial_reason),
        }
    }

    fn dense_packed(
        schedule_basis: RunExecutionScheduleBasis,
        block_tokens: u32,
        attention_matrix_max_position: u64,
        scratch_priced_upper_bytes: u64,
        packed_spans: Vec<RunPackedPrefillSpan>,
    ) -> Self {
        Self {
            requested_prefill: PrefillExecution::Auto,
            effective_prefill: RunEffectivePrefill::DensePackedPassiveSpans,
            schedule_basis,
            numerical_relationship:
                RunNumericalRelationship::PackedReductionTopologyDiffersFromSerial,
            minimum_span_tokens: PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS,
            chunk_cap_tokens: PACKED_PREFILL_CHUNK_CAP_TOKENS,
            block_tokens: Some(block_tokens),
            attention_matrix_max_position: (attention_matrix_max_position > 0)
                .then_some(attention_matrix_max_position),
            scratch_priced_upper_bytes,
            packed_spans,
            serial_reason: None,
        }
    }

    pub(crate) fn runtime_serial(
        requested_prefill: PrefillExecution,
        automatic_reason: RunSerialReason,
    ) -> Self {
        Self::serial(
            requested_prefill,
            RunExecutionScheduleBasis::EffectivePlan,
            automatic_reason,
        )
    }

    pub(crate) fn validate(&self, runtime_kind: &str, prompt_len: usize) -> Result<()> {
        ensure!(
            self.minimum_span_tokens == PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS
                && self.chunk_cap_tokens == PACKED_PREFILL_CHUNK_CAP_TOKENS,
            "run execution policy constants are unsupported"
        );
        match self.effective_prefill {
            RunEffectivePrefill::Serial => {
                ensure!(
                    self.numerical_relationship == RunNumericalRelationship::SerialReference
                        && self.block_tokens.is_none()
                        && self.attention_matrix_max_position.is_none()
                        && self.scratch_priced_upper_bytes == 0
                        && self.packed_spans.is_empty()
                        && self.serial_reason.is_some(),
                    "serial run execution metadata contains packed state"
                );
                let reason_matches = match (self.requested_prefill, self.serial_reason) {
                    (PrefillExecution::Serial, Some(RunSerialReason::RequestedSerial)) => true,
                    (PrefillExecution::Auto, Some(reason)) => {
                        reason != RunSerialReason::RequestedSerial
                    }
                    _ => false,
                };
                ensure!(
                    reason_matches,
                    "serial run execution reason differs from the request"
                );
                let runtime_matches = match self.serial_reason {
                    Some(RunSerialReason::RequestedSerial) => true,
                    Some(
                        RunSerialReason::NoEligiblePassiveSpan
                        | RunSerialReason::DensePackedMemoryAdmissionDenied
                        | RunSerialReason::MoePackedNotQualified
                        | RunSerialReason::CohortSerialPolicy,
                    ) => runtime_kind == "ordinary_qwen",
                    Some(RunSerialReason::FlashNextPackedNotImplemented) => {
                        runtime_kind == "flash_next"
                    }
                    Some(RunSerialReason::MusePackedNotImplemented) => {
                        runtime_kind == "muse_glimmer"
                    }
                    None => false,
                };
                ensure!(
                    runtime_matches,
                    "serial run execution reason differs from the runtime"
                );
            }
            RunEffectivePrefill::DensePackedPassiveSpans => {
                ensure!(
                    runtime_kind == "ordinary_qwen"
                        && self.requested_prefill == PrefillExecution::Auto
                        && self.numerical_relationship
                            == RunNumericalRelationship::PackedReductionTopologyDiffersFromSerial
                        && self.serial_reason.is_none()
                        && self.scratch_priced_upper_bytes > 0
                        && !self.packed_spans.is_empty(),
                    "packed run execution metadata has an inconsistent runtime or policy"
                );
                let block_tokens = self
                    .block_tokens
                    .context("packed run execution requires block_tokens")?;
                ensure!(
                    self.attention_matrix_max_position
                        .is_none_or(|position| position >= u64::from(block_tokens)),
                    "packed run execution attention-matrix extent is smaller than its block"
                );
                let final_prompt_index = prompt_len
                    .checked_sub(1)
                    .context("packed run execution requires a nonempty prompt")?;
                let mut previous_end = 0usize;
                let mut longest = 0usize;
                for span in &self.packed_spans {
                    let length = span
                        .end
                        .checked_sub(span.start)
                        .context("packed run execution span is reversed")?;
                    ensure!(
                        span.start >= previous_end
                            && span.end <= final_prompt_index
                            && length >= self.minimum_span_tokens,
                        "packed run execution contains an invalid, overlapping, or final-token span"
                    );
                    previous_end = span.end;
                    longest = longest.max(length);
                }
                ensure!(
                    usize::try_from(block_tokens)? == longest.min(self.chunk_cap_tokens),
                    "packed run execution block size differs from its spans"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn validate_against_plan(
        &self,
        runtime_kind: &str,
        plan: &LensPlan,
        prompt_len: usize,
    ) -> Result<()> {
        self.validate(runtime_kind, prompt_len)?;
        let expected = expected_packed_prefill_spans(plan, prompt_len)?;
        match (self.effective_prefill, self.serial_reason) {
            (RunEffectivePrefill::DensePackedPassiveSpans, None) => ensure!(
                self.packed_spans == expected,
                "packed run execution spans differ from passive plan spans"
            ),
            (RunEffectivePrefill::Serial, Some(RunSerialReason::NoEligiblePassiveSpan)) => {
                ensure!(
                    expected.is_empty(),
                    "serial run claims no eligible passive span, but the plan has one"
                )
            }
            (
                RunEffectivePrefill::Serial,
                Some(RunSerialReason::DensePackedMemoryAdmissionDenied),
            ) => ensure!(
                !expected.is_empty(),
                "serial run claims packed-memory denial without an eligible passive span"
            ),
            _ => {}
        }
        Ok(())
    }

    fn effective_prefill(&self) -> RunEffectivePrefill {
        self.effective_prefill
    }

    fn serial_reason(&self) -> Option<RunSerialReason> {
        self.serial_reason
    }

    fn packed_spans(&self) -> &[RunPackedPrefillSpan] {
        &self.packed_spans
    }

    pub(crate) fn schedule_basis(&self) -> RunExecutionScheduleBasis {
        self.schedule_basis
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RunOutput {
    schema: &'static str,
    schema_version: u32,
    runtime_kind: &'static str,
    model_path: PathBuf,
    canonical_plan_path: PathBuf,
    authored_plan: LensPlan,
    authored_plan_canonical_json_blake3: String,
    plan: LensPlan,
    position_bindings: Vec<PositionBinding>,
    input_source: &'static str,
    add_special_tokens: Option<bool>,
    rendering: LensInputRendering,
    prompt_token_ids: Vec<i32>,
    generated_token_ids: Vec<i32>,
    sampler: RunSampler,
    max_new_tokens: usize,
    decoded_text: String,
    stop_reason: String,
    execution: RunExecution,
    operation_applications: Vec<OperationApplication>,
    requested_live_readouts: Vec<ReadoutDefinition>,
    live_readouts: Vec<LiveReadout>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    native_hyper_captures: Vec<NativeHyperCapture>,
    #[serde(skip_serializing_if = "Option::is_none")]
    execution_binding: Option<RunExecutionBinding>,
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
struct SweepCohortRequestRecord {
    id: String,
    messages: PathBuf,
    message_mode: Option<LensMessageMode>,
}

struct LoadedSweepCohortRequests {
    canonical_path: PathBuf,
    blake3: String,
    records: Vec<(usize, SweepCohortRequestRecord)>,
}

struct PreflightSweepCohortRequest {
    source_line: usize,
    id: String,
    messages_path: PathBuf,
    messages_blake3: String,
    arm_args: LensRunArgs,
    prepared_input: PreparedLensInput,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SweepCohortManifest {
    schema: String,
    schema_version: u32,
    producer: SweepProducer,
    requests_jsonl_path: PathBuf,
    requests_jsonl_blake3: String,
    canonical_source_plan_path: PathBuf,
    source_plan: LensPlan,
    source_plan_canonical_json_blake3: String,
    model_path: PathBuf,
    operation_id: String,
    coefficients: Vec<f32>,
    sampler: RunSampler,
    max_new_tokens: usize,
    prefill_execution: PrefillExecution,
    execution_policy: String,
    planned_request_count: usize,
    planned_total_arm_count: usize,
    transition_upper_bound: u64,
    cumulative_serialized_child_bytes: u64,
    sweeps: Vec<SweepCohortChild>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct SweepCohortChild {
    index: usize,
    id: String,
    source_line: usize,
    path: String,
    prompt_token_count: usize,
    messages_path: PathBuf,
    messages_blake3: String,
    serialized_byte_length: u64,
    manifest_byte_length: u64,
    manifest_blake3: String,
}

struct BuiltSweepBundle {
    summaries: Vec<SweepArmSummary>,
    serialized_byte_length: u64,
    manifest_byte_length: u64,
    manifest_blake3: String,
}

struct CapturedSweepMessages {
    path: PathBuf,
    bytes: Vec<u8>,
    blake3: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SweepCohortPlanBounds {
    request_count: usize,
    total_arm_count: usize,
    transition_upper_bound: u64,
}

struct SweepCohortManifestBasis {
    producer: SweepProducer,
    requests_jsonl_path: PathBuf,
    requests_jsonl_blake3: String,
    canonical_source_plan_path: PathBuf,
    source_plan: LensPlan,
    source_plan_canonical_json_blake3: String,
    model_path: PathBuf,
    operation_id: String,
    coefficients: Vec<f32>,
    sampler: RunSampler,
    max_new_tokens: usize,
    prefill_execution: PrefillExecution,
    bounds: SweepCohortPlanBounds,
}

impl SweepCohortManifestBasis {
    fn build(
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

struct SweepOutputBudget {
    consumed: u64,
    reserved: u64,
}

impl SweepOutputBudget {
    fn new(reserved: usize) -> Result<Self> {
        let reserved = u64::try_from(reserved).context("reserved sweep output bytes")?;
        ensure!(
            reserved <= MAX_SWEEP_BUNDLE_BYTES,
            "reserved sweep output bytes {reserved} exceed bundle limit {MAX_SWEEP_BUNDLE_BYTES}"
        );
        Ok(Self {
            consumed: 0,
            reserved,
        })
    }

    fn charge(&mut self, byte_length: usize) -> Result<()> {
        let consumed = self
            .consumed
            .checked_add(u64::try_from(byte_length).context("serialized child byte length")?)
            .context("cohort serialized child byte count overflow")?;
        let committed = consumed
            .checked_add(self.reserved)
            .context("cohort committed byte count overflow")?;
        ensure!(
            committed <= MAX_SWEEP_BUNDLE_BYTES,
            "coefficient sweep committed bytes {committed} exceed bundle limit {}",
            MAX_SWEEP_BUNDLE_BYTES
        );
        self.consumed = consumed;
        Ok(())
    }

    fn release_reservation(&mut self) {
        self.reserved = 0;
    }
}

struct SweepArmSummary {
    index: usize,
    coefficient: f32,
    decoded_text: String,
    stop_reason: String,
    operation_application_count: usize,
    live_readout_count: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
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

#[derive(Debug, Serialize)]
pub(crate) struct RunExecutionBinding {
    pub(crate) deployed_model_content_blake3: String,
    pub(crate) content_identity_outcome: String,
    pub(crate) weight_bytes_hashed: u64,
    pub(crate) published_lenses: Vec<RunPublishedLensBinding>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunPublishedLensBinding {
    pub(crate) lens_id: String,
    pub(crate) manifest: PathBuf,
    pub(crate) manifest_canonical_json_blake3: String,
    pub(crate) profile: String,
    pub(crate) method: String,
    pub(crate) target_layer: u32,
    pub(crate) fitted_checkpoint: String,
    pub(crate) fitted_checkpoint_revision: String,
    pub(crate) source_repository: String,
    pub(crate) source_revision: String,
    pub(crate) source_sha256: String,
    pub(crate) payload_blake3: String,
    pub(crate) claims_basis: String,
    pub(crate) transfer_validation_status: String,
    pub(crate) selected_token_ids: Vec<u32>,
    pub(crate) selected_matrices: Vec<RunPublishedMatrixBinding>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunPublishedMatrixBinding {
    pub(crate) source_layer: u32,
    pub(crate) blake3: String,
}

pub(crate) fn emit_run_output(
    args: &LensRunArgs,
    runtime_kind: &'static str,
    plan_path: &Path,
    bound_plan: BoundLensPlan,
    prepared_input: &PreparedLensInput,
    result: RunResult,
    execution: RunExecution,
    execution_binding: Option<RunExecutionBinding>,
    output_path: Option<&Path>,
) -> Result<()> {
    let artifact = build_run_output(
        args,
        runtime_kind,
        plan_path,
        bound_plan,
        prepared_input,
        result,
        execution,
        execution_binding,
    );
    let stdout_format = effective_run_stdout_format(args.format, output_path.is_some());
    let bytes = if output_path.is_some() || stdout_format == RunStdoutFormat::Json {
        Some(serialize_run_output(&artifact)?)
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

fn build_run_output(
    args: &LensRunArgs,
    runtime_kind: &'static str,
    plan_path: &Path,
    bound_plan: BoundLensPlan,
    prepared_input: &PreparedLensInput,
    result: RunResult,
    execution: RunExecution,
    execution_binding: Option<RunExecutionBinding>,
) -> RunOutput {
    let requested_live_readouts = bound_plan.resolved.readouts.clone();
    RunOutput {
        schema: RUN_SCHEMA,
        schema_version: RUN_SCHEMA_VERSION,
        runtime_kind,
        model_path: args.model.clone(),
        canonical_plan_path: plan_path.to_path_buf(),
        authored_plan: bound_plan.authored,
        authored_plan_canonical_json_blake3: bound_plan.authored_plan_canonical_json_blake3,
        requested_live_readouts,
        plan: bound_plan.resolved,
        position_bindings: bound_plan.position_bindings,
        input_source: prepared_input.source,
        add_special_tokens: prepared_input.add_special_tokens,
        rendering: prepared_input.rendering.clone(),
        prompt_token_ids: result.prompt_token_ids,
        generated_token_ids: result.generated_token_ids,
        sampler: run_sampler(args),
        max_new_tokens: args.max_new_tokens,
        decoded_text: result.decoded_text,
        stop_reason: result.stop_reason,
        execution,
        operation_applications: result.operation_applications,
        live_readouts: result.live_readouts,
        native_hyper_captures: result.native_hyper_captures,
        execution_binding,
    }
}

fn serialize_run_output(artifact: &RunOutput) -> Result<Vec<u8>> {
    if artifact.execution.schedule_basis() == RunExecutionScheduleBasis::EffectivePlan {
        artifact.execution.validate_against_plan(
            artifact.runtime_kind,
            &artifact.plan,
            artifact.prompt_token_ids.len(),
        )?;
    } else {
        artifact
            .execution
            .validate(artifact.runtime_kind, artifact.prompt_token_ids.len())?;
    }
    let bytes = serde_json::to_vec(artifact).context("serialize Lens run artifact")?;
    ensure!(
        bytes.len() <= MAX_RUN_ARTIFACT_BYTES,
        "serialized Lens run artifact is {} bytes; limit is {MAX_RUN_ARTIFACT_BYTES}",
        bytes.len()
    );
    Ok(bytes)
}

fn run_sampler(args: &LensRunArgs) -> RunSampler {
    RunSampler {
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    }
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

fn print_run_summary(artifact: &RunOutput, output_path: Option<&Path>) {
    print!("{}", run_summary(artifact, output_path));
}

fn run_summary(artifact: &RunOutput, output_path: Option<&Path>) -> String {
    let serial_reason = artifact
        .execution
        .serial_reason()
        .map_or("none", RunSerialReason::as_str);
    let mut summary = format!(
        "runtime={} model={}\nprefill_execution={} serial_reason={}\ngenerated_text={}\nstop_reason={}\noperation_applications={} live_readouts={}\n",
        artifact.runtime_kind,
        artifact.model_path.display(),
        artifact.execution.effective_prefill().as_str(),
        serial_reason,
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
    raw_lm_head: Option<NativeLens>,
}

struct PreparedDirection {
    rows: BTreeMap<u32, MetalTensor>,
}

struct ExecutionPlan {
    plan: LensPlan,
    lenses: HashMap<String, PreparedLens>,
    directions: HashMap<String, PreparedDirection>,
    coordinate_swaps: HashMap<String, PreparedDirection>,
    n_layer: u32,
    hidden_size: usize,
    capture: Option<MetalTensor>,
}

struct PreparedOrdinaryPrefill {
    execution: RunExecution,
    scratch: Option<PackedPrefillScratch>,
}

impl PreparedOrdinaryPrefill {
    fn serial(
        requested: PrefillExecution,
        schedule_basis: RunExecutionScheduleBasis,
        reason: RunSerialReason,
    ) -> Self {
        Self {
            execution: RunExecution::serial(requested, schedule_basis, reason),
            scratch: None,
        }
    }
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
    validate_run_args(&args)?;
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
        ensure!(
            args.open_responses.is_none(),
            "--open-responses supports ordinary Qwen only; Muse Glimmer is not supported"
        );
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
        ensure!(
            args.open_responses.is_none(),
            "--open-responses supports ordinary Qwen only; Flash-Next is not supported"
        );
        return run_qwen4exp(
            &args,
            plan,
            &plan_path,
            plan_dir,
            gguf,
            output_path.as_deref(),
        );
    }
    ensure!(
        matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe),
        "qwen-lens run supports ordinary Qwen, Muse Glimmer, or Flash-Next"
    );
    validate_ordinary_plan(&plan)?;

    let tokenizer = Tokenizer::from_gguf(&gguf).context("load model tokenizer")?;
    let prepared_input = prepare_qwen_model_input(args.input_spec(), family, &gguf, &tokenizer)?;
    let prompt_token_ids = &prepared_input.token_ids;
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < tokenizer.n_vocab()),
        "prompt contains a token outside the model vocabulary"
    );
    ensure_request_fits_context(
        prompt_token_ids.len(),
        args.max_new_tokens,
        gguf.declared_context_length()?,
    )?;

    let runtime = Runtime::metal().context("initialize Metal runtime")?;
    let loaded = runtime
        .load_opened_gguf_with_intent(
            gguf,
            args.model.clone(),
            LoadedModelConfig::default(),
            ModelLoadIntent::DisposableGeneration,
        )
        .with_context(|| format!("load model {}", args.model.display()))?;
    validate_runtime(loaded.gguf(), loaded.arch().kind, loaded.arch().n_layer)?;
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < loaded.arch().vocab_size),
        "prompt contains a token outside the model vocabulary"
    );

    let bound_plan = bind_plan_positions(&plan, &prepared_input.rendering, prompt_token_ids.len())?;
    let execution = prepare_execution_plan(&bound_plan.resolved, plan_dir, &loaded)?;
    validate_reachable_scopes(&execution.plan, prompt_token_ids.len(), args.max_new_tokens)?;
    let schedule = CompiledEventSchedule::compile(&execution.plan, execution.n_layer)?;
    let mut prefill = prepare_ordinary_prefill(
        &loaded,
        &schedule,
        &execution.plan,
        args.prefill_execution,
        RunExecutionScheduleBasis::EffectivePlan,
        prompt_token_ids.len(),
        args.max_new_tokens,
    )?;
    prefill.execution.validate_against_plan(
        "ordinary_qwen",
        &execution.plan,
        prompt_token_ids.len(),
    )?;
    let stop_tokens: HashSet<i32> = loaded.gguf().stop_token_ids()?.into_iter().collect();
    let result = execute_ordinary_arm(
        &loaded,
        &tokenizer,
        &execution,
        &execution.plan,
        &schedule,
        prompt_token_ids,
        args.max_new_tokens,
        run_sampler(&args),
        &stop_tokens,
        &mut prefill,
    )?;
    emit_run_output(
        &args,
        "ordinary_qwen",
        &plan_path,
        bound_plan,
        &prepared_input,
        result,
        prefill.execution,
        None,
        output_path.as_deref(),
    )
}

fn validate_run_args(args: &LensRunArgs) -> Result<()> {
    ensure!(args.max_new_tokens > 0, "--max-new-tokens must be positive");
    validate_lens_input_spec(args.input_spec())?;
    Ok(())
}

fn prepare_ordinary_prefill(
    loaded: &qwen_llm::runtime::LoadedModel,
    schedule: &CompiledEventSchedule,
    schedule_plan: &LensPlan,
    requested: PrefillExecution,
    schedule_basis: RunExecutionScheduleBasis,
    prompt_len: usize,
    max_new_tokens: usize,
) -> Result<PreparedOrdinaryPrefill> {
    if requested == PrefillExecution::Serial {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::RequestedSerial,
        ));
    }
    if loaded.arch().kind == ArchKind::Moe {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::MoePackedNotQualified,
        ));
    }

    let bound_schedule = schedule.bind(schedule_plan)?;
    let packed_spans = bound_schedule
        .passive_prefill_spans(prompt_len, PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS)?
        .into_iter()
        .map(|span| RunPackedPrefillSpan {
            start: span.start,
            end: span.end,
        })
        .collect::<Vec<_>>();
    let Some(longest_span) = packed_spans.iter().map(|span| span.end - span.start).max() else {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::NoEligiblePassiveSpan,
        ));
    };
    let block_tokens = u32::try_from(longest_span.min(PACKED_PREFILL_CHUNK_CAP_TOKENS))
        .context("packed Lens prefill block size exceeds u32")?;
    let scratch_plan = loaded
        .plan_packed_prefill_scratch(block_tokens, prompt_len)
        .context("plan dense packed Lens prefill scratch")?;
    let scratch_priced_upper_bytes = scratch_plan.priced_upper_bytes();
    let capacity = required_forward_count(prompt_len, max_new_tokens)?;
    let admission = loaded
        .qwen_execution_memory_admission(1, capacity, scratch_priced_upper_bytes, 0)
        .context("price dense packed Lens prefill memory")?;
    if !admission.admitted {
        return Ok(PreparedOrdinaryPrefill::serial(
            requested,
            schedule_basis,
            RunSerialReason::DensePackedMemoryAdmissionDenied,
        ));
    }
    let block_tokens = scratch_plan.block_size();
    let matrix_max_position = scratch_plan.matrix_max_pos();
    let scratch = loaded
        .allocate_packed_prefill_scratch(scratch_plan)
        .context("allocate dense packed Lens prefill scratch")?;
    let execution = RunExecution::dense_packed(
        schedule_basis,
        block_tokens,
        matrix_max_position,
        scratch_priced_upper_bytes,
        packed_spans,
    );
    execution.validate("ordinary_qwen", prompt_len)?;
    Ok(PreparedOrdinaryPrefill {
        execution,
        scratch: Some(scratch),
    })
}

fn execute_ordinary_arm(
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

fn run_single_coefficient_sweep(args: CoefficientSweepArgs) -> Result<()> {
    let arm_args = args.arm_run_args();
    validate_run_args(&arm_args)?;
    let output_path = super::resolve_output_path(&args.output)?;
    ensure_new_sweep_output(&output_path)?;

    let plan_path = std::fs::canonicalize(&args.plan)
        .with_context(|| format!("resolve plan {}", args.plan.display()))?;
    let source_plan = parse_plan_bytes(&super::read_regular_file_bounded(
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
    let execution = prepare_execution_plan(&source_bound_plan.resolved, plan_dir, &loaded)?;
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
    let built = stage_and_publish_sweep(&output_path, |staging| {
        let mut output_budget = SweepOutputBudget::new(0)?;
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
fn build_sweep_bundle(
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
    mut output_budget: Option<&mut SweepOutputBudget>,
) -> Result<BuiltSweepBundle> {
    let arms_path = root.join("arms");
    create_sweep_directory(&arms_path)?;
    super::sync_directory(root)?;

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
            arm_args,
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
        create_sweep_directory(&arm_path)?;
        write_new_sweep_file(&arm_path.join("run.json"), &bytes)?;
        super::sync_directory(&arm_path)?;
        arms.push(CoefficientSweepArm {
            index,
            coefficient,
            artifact: relative,
            byte_length: bytes.len() as u64,
            blake3: blake3::hash(&bytes).to_hex().to_string(),
        });
        summaries.push(summary);
    }
    super::sync_directory(&arms_path)?;

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
    write_new_sweep_file(&root.join(SWEEP_MANIFEST_NAME), &manifest_bytes)?;
    super::sync_directory(root)?;
    Ok(BuiltSweepBundle {
        summaries,
        serialized_byte_length,
        manifest_byte_length: manifest_bytes.len() as u64,
        manifest_blake3: blake3::hash(&manifest_bytes).to_hex().to_string(),
    })
}

fn charge_sweep_bundle_bytes(
    current: u64,
    byte_length: usize,
    output_budget: Option<&mut SweepOutputBudget>,
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

fn current_sweep_producer() -> SweepProducer {
    SweepProducer {
        build_commit: env!("QWEN_BUILD_COMMIT").into(),
        build_dirty: env!("QWEN_BUILD_DIRTY").into(),
        build_source_state: env!("QWEN_BUILD_SOURCE_STATE").into(),
    }
}

fn run_coefficient_sweep_cohort(args: CoefficientSweepArgs) -> Result<()> {
    let requests = read_sweep_cohort_requests(
        args.requests_jsonl
            .as_deref()
            .context("--requests-jsonl is required in cohort mode")?,
        args.coefficients.len(),
        args.max_new_tokens,
    )?;
    checked_sweep_cohort_child_count(requests.records.len(), args.coefficients.len())?;
    let output_path = super::resolve_output_path(&args.output)?;
    ensure_new_sweep_output(&output_path)?;

    let plan_path = std::fs::canonicalize(&args.plan)
        .with_context(|| format!("resolve plan {}", args.plan.display()))?;
    let source_plan = parse_plan_bytes(&super::read_regular_file_bounded(
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
    let execution = prepare_execution_plan(&first_bound_plan.resolved, plan_dir, &loaded)?;
    drop(first_bound_plan);
    let stop_tokens = loaded
        .gguf()
        .stop_token_ids()?
        .into_iter()
        .collect::<HashSet<_>>();
    let request_count = preflight_requests.len();
    stage_and_publish_sweep(&output_path, |staging| {
        let sweeps_root = staging.join("sweeps");
        create_sweep_directory(&sweeps_root)?;
        super::sync_directory(staging)?;
        let mut children = Vec::new();
        children
            .try_reserve_exact(preflight_requests.len())
            .context("allocate sweep cohort child manifest entries")?;
        let mut output_budget = SweepOutputBudget::new(manifest_reserve_bytes)?;
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
            create_sweep_directory(&child_path)?;
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
        super::sync_directory(&sweeps_root)?;
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
        write_new_sweep_file(&staging.join(SWEEP_MANIFEST_NAME), &manifest_bytes)?;
        super::sync_directory(staging)?;
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

fn validate_sweep_prompt(
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

fn checked_sweep_cohort_child_count(request_count: usize, arm_count: usize) -> Result<usize> {
    request_count
        .checked_mul(arm_count)
        .context("cohort prompt/arm child count overflow")
}

fn sweep_cohort_request_transition_upper_bound(
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

fn sweep_cohort_request_capacity(arm_count: usize, max_new_tokens: usize) -> Result<usize> {
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

fn plan_sweep_cohort_bounds(
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

fn ensure_sweep_cohort_manifest_capacity(
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

fn read_sweep_cohort_requests(
    path: &Path,
    arm_count: usize,
    max_new_tokens: usize,
) -> Result<LoadedSweepCohortRequests> {
    let request_capacity = sweep_cohort_request_capacity(arm_count, max_new_tokens)?;
    let canonical_path = std::fs::canonicalize(path)
        .with_context(|| format!("resolve sweep request file {}", path.display()))?;
    let bytes = super::read_regular_file_bounded(&canonical_path, MAX_SWEEP_COHORT_FILE_BYTES)?;
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
        validate_sweep_cohort_id(&request.id, source_line)?;
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

fn capture_sweep_cohort_messages(path: &Path) -> Result<CapturedSweepMessages> {
    capture_sweep_cohort_messages_with(path, |path| {
        super::read_regular_file_bounded(path, MAX_SWEEP_COHORT_MESSAGES_BYTES)
    })
}

fn capture_sweep_cohort_messages_with(
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

fn validate_sweep_cohort_id(id: &str, source_line: usize) -> Result<()> {
    ensure!(
        !id.is_empty() && id.len() <= MAX_SWEEP_COHORT_ID_BYTES,
        "sweep request ID on line {source_line} must contain 1..={MAX_SWEEP_COHORT_ID_BYTES} bytes"
    );
    let mut bytes = id.bytes();
    ensure!(
        bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_alphanumeric())
            && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-')),
        "sweep request ID on line {source_line} must start with an ASCII alphanumeric and contain only ASCII alphanumerics, '.', '_', or '-'"
    );
    Ok(())
}

fn serialize_sweep_cohort_manifest(manifest: &SweepCohortManifest) -> Result<Vec<u8>> {
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

fn verify_sweep_cohort_child_manifest(root: &Path, child: &SweepCohortChild) -> Result<()> {
    let length = usize::try_from(child.manifest_byte_length)
        .context("sweep cohort child manifest length does not fit this platform")?;
    let path = root.join(&child.path).join(SWEEP_MANIFEST_NAME);
    let bytes = super::read_regular_file_exact(&path, length)?;
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

fn parse_sweep_cohort_manifest_bytes(bytes: &[u8]) -> Result<SweepCohortManifest> {
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

fn validate_sweep_cohort_bundle_size(
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

fn validate_sweep_cohort_manifest(manifest: &SweepCohortManifest) -> Result<()> {
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
        validate_sweep_cohort_id(&child.id, child.source_line)?;
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

fn validate_sweep_producer(producer: &SweepProducer) -> Result<()> {
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

fn validate_coefficient_sweep_args(args: &CoefficientSweepArgs) -> Result<()> {
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

fn ensure_new_sweep_output(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => bail!("sweep output {} already exists", path.display()),
        Err(error) => {
            Err(error).with_context(|| format!("inspect sweep output {}", path.display()))
        }
    }
}

fn create_sweep_directory(path: &Path) -> Result<()> {
    DirBuilder::new()
        .mode(0o700)
        .create(path)
        .with_context(|| format!("create sweep directory {}", path.display()))
}

fn write_new_sweep_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("create sweep file {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write sweep file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync sweep file {}", path.display()))
}

fn stage_and_publish_sweep<T>(output: &Path, build: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
    let parent = output.parent().context("sweep output has no parent")?;
    let leaf = output.file_name().context("sweep output has no leaf")?;
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_nanos();
    let staging = parent.join(format!(
        ".{}.stage.{}.{}",
        leaf.to_string_lossy(),
        std::process::id(),
        nonce
    ));
    create_sweep_directory(&staging)?;
    if let Err(error) = super::sync_directory(parent) {
        if let Err(cleanup_error) = std::fs::remove_dir(&staging) {
            return Err(error.context(format!(
                "also failed to remove empty sweep staging directory {}: {cleanup_error}",
                staging.display()
            )));
        }
        return Err(error);
    }
    let result = build(&staging).and_then(|value| {
        publish_sweep_directory_exclusive(&staging, output)?;
        Ok(value)
    });
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            match std::fs::symlink_metadata(&staging) {
                Ok(_) => {
                    if let Err(cleanup_error) = std::fs::remove_dir_all(&staging) {
                        return Err(error.context(format!(
                            "also failed to remove sweep staging directory {}: {cleanup_error}",
                            staging.display()
                        )));
                    }
                    if let Err(sync_error) = super::sync_directory(parent) {
                        return Err(error.context(format!(
                            "removed sweep staging directory, but failed to sync {}: {sync_error}",
                            parent.display()
                        )));
                    }
                }
                Err(inspect_error) if inspect_error.kind() == std::io::ErrorKind::NotFound => {}
                Err(inspect_error) => {
                    return Err(error.context(format!(
                        "also failed to inspect sweep staging directory {}: {inspect_error}",
                        staging.display()
                    )));
                }
            }
            Err(error)
        }
    }
}

fn publish_sweep_directory_exclusive(staging: &Path, output: &Path) -> Result<()> {
    let old = CString::new(staging.as_os_str().as_bytes())?;
    let new = CString::new(output.as_os_str().as_bytes())?;
    let renamed = unsafe {
        libc::renameatx_np(
            libc::AT_FDCWD,
            old.as_ptr(),
            libc::AT_FDCWD,
            new.as_ptr(),
            libc::RENAME_EXCL,
        )
    };
    if renamed != 0 {
        return Err(std::io::Error::last_os_error()).with_context(|| {
            format!(
                "publish sweep directory {} to {}",
                staging.display(),
                output.display()
            )
        });
    }
    let parent = output.parent().context("sweep output has no parent")?;
    if let Err(error) = super::sync_directory(parent) {
        eprintln!(
            "warning: sweep {} is published, but its parent directory could not be synced: {error:#}",
            output.display()
        );
    }
    Ok(())
}

fn serialize_sweep_manifest(manifest: &CoefficientSweepManifest) -> Result<Vec<u8>> {
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

fn validate_sweep_bundle_size(
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

fn validate_sweep_manifest(manifest: &CoefficientSweepManifest) -> Result<()> {
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

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn print_sweep_summary(args: &CoefficientSweepArgs, output: &Path, summaries: &[SweepArmSummary]) {
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
    let prepared_input =
        prepare_qwen_model_input(args.input_spec(), ModelFamily::Qwen4Exp, &gguf, &tokenizer)?;
    let prompt_token_ids = &prepared_input.token_ids;
    ensure!(
        !prompt_token_ids.is_empty(),
        "prompt must encode to at least one token"
    );
    ensure!(
        prompt_token_ids
            .iter()
            .all(|&token| token >= 0 && (token as u32) < config.vocab_size),
        "prompt contains a token outside the Flash-Next vocabulary"
    );
    let required_forwards = ensure_request_fits_context(
        prompt_token_ids.len(),
        args.max_new_tokens,
        config.context_length as usize,
    )?;
    let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, required_forwards)
        .context("derive Flash-Next serial session capacity")?;
    let bound_plan = bind_plan_positions(&plan, &prepared_input.rendering, prompt_token_ids.len())?;
    validate_reachable_scopes(
        &bound_plan.resolved,
        prompt_token_ids.len(),
        args.max_new_tokens,
    )?;
    let execution =
        prepare_qwen4exp_execution_plan(bound_plan.resolved.clone(), plan_dir, &config)?;
    validate_qwen4exp_event_schedule(&execution, prompt_token_ids.len(), args.max_new_tokens)?;

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
        prompt_token_ids: prompt_token_ids.to_vec(),
        decoded_text: tokenizer.decode(&generated_token_ids),
        generated_token_ids,
        stop_reason,
        operation_applications,
        live_readouts: Vec::new(),
        native_hyper_captures,
    };
    emit_run_output(
        args,
        "flash_next",
        plan_path,
        bound_plan,
        &prepared_input,
        result,
        RunExecution::runtime_serial(
            args.prefill_execution,
            RunSerialReason::FlashNextPackedNotImplemented,
        ),
        None,
        output_path,
    )
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
        if !operation_enabled(operation) {
            continue;
        }
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
    ensure!(
        matches!(plan.version, 1 | 2),
        "Lens plan version must be 1 or 2"
    );
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
                if direction.target_covector.is_some() {
                    ensure!(
                        published_full_tokens.contains_key(direction.lens.as_str())
                            && matches!(direction.row, DirectionRow::TokenId { .. }),
                        "direction {} target_covector is supported only for published full transport token rows",
                        direction.id
                    );
                }
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
            .validate(&format!("operation {} scope", operation.id), plan.version)?;
        let coefficient = operation.action.coefficient();
        ensure!(
            coefficient.is_finite(),
            "operation {} coefficient must be finite",
            operation.id,
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
            .validate(&format!("readout {} scope", readout.id), plan.version)?;
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

fn validate_sweep_source_operation(plan: &LensPlan, operation_id: &str) -> Result<()> {
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

pub(crate) fn plan_with_operation_coefficient(
    source: &LensPlan,
    operation_id: &str,
    coefficient: f32,
) -> Result<LensPlan> {
    ensure!(coefficient.is_finite(), "sweep coefficient must be finite");
    let mut plan = source.clone();
    let operation = plan
        .operations
        .iter_mut()
        .find(|operation| operation.id == operation_id)
        .with_context(|| format!("Lens plan has no operation {operation_id:?}"))?;
    operation.action.set_coefficient(coefficient);
    validate_plan(&plan)?;
    Ok(plan)
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

pub(crate) fn validate_run_artifact_plan(plan: &LensPlan, runtime_kind: &str) -> Result<()> {
    validate_plan(plan)?;
    match runtime_kind {
        "ordinary_qwen" => validate_ordinary_plan(plan),
        "muse_glimmer" => validate_muse_artifact_plan(plan),
        "flash_next" => validate_flash_artifact_plan(plan),
        _ => bail!("run artifact has unsupported runtime kind {runtime_kind:?}"),
    }
}

fn validate_muse_artifact_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        !plan.operations.is_empty() || !plan.readouts.is_empty(),
        "Muse run artifact plan requires an operation or readout"
    );
    ensure!(
        plan.lenses.iter().all(|lens| matches!(
            lens,
            LensDefinition::NativeSelected { .. } | LensDefinition::PublishedFullTransport { .. }
        )),
        "Muse run artifact plan contains an unsupported lens kind"
    );
    ensure!(
        plan.directions.iter().all(|direction| matches!(
            direction,
            DirectionDefinition::LensRow(definition)
                if matches!(definition.row, DirectionRow::TokenId { .. })
                    && definition.target_covector.is_none()
        )),
        "Muse run artifact plan requires selected-token directions without target_covector overrides"
    );
    Ok(())
}

fn validate_flash_artifact_plan(plan: &LensPlan) -> Result<()> {
    ensure!(
        plan.lenses.is_empty()
            && plan.readouts.is_empty()
            && !plan.operations.is_empty()
            && plan
                .directions
                .iter()
                .all(|direction| direction.native_hyper().is_some()),
        "Flash-Next run artifact plan has unsupported lenses, directions, or readouts"
    );
    let direction_layers = plan
        .directions
        .iter()
        .filter_map(|direction| {
            direction.native_hyper().map(|definition| {
                let (_, layer) = definition.source.path_and_layer();
                (definition.id.as_str(), layer)
            })
        })
        .collect::<HashMap<_, _>>();
    for operation in &plan.operations {
        let Action::FixedAdd { direction, .. } = &operation.action else {
            bail!("Flash-Next run artifact operations require fixed_add")
        };
        let layer = direction_layers.get(direction.as_str()).with_context(|| {
            format!("Flash-Next operation {} lacks its direction", operation.id)
        })?;
        let exact_layer = match &operation.scope.layers {
            Selector::Values { values } => values.as_slice() == [*layer],
            Selector::Range { start, end } => start == layer && end == layer,
            Selector::All | Selector::RenderedSpans { .. } => false,
        };
        ensure!(
            exact_layer,
            "Flash-Next operation {} does not select exactly its direction layer {}",
            operation.id,
            layer
        );
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
    let mut raw_lens_layers: HashMap<&str, BTreeSet<u32>> = HashMap::new();
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
            if direction.effective_target_covector()
                != DirectionTargetCovector::DeployedLogitNumerator
            {
                raw_lens_layers
                    .entry(direction.lens.as_str())
                    .or_default()
                    .extend(layers.iter().copied());
            }
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
                raw_lm_head: None,
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
                    super::full_lens::FullTokenTargetCovector::DeployedLogitNumerator,
                )?;
                let raw_lm_head = raw_lens_layers
                    .get(id.as_str())
                    .map(|raw_layers| {
                        super::full_lens::project_full_token_directions(
                            &resolve_plan_path(plan_dir, artifact),
                            token_ids,
                            &raw_layers.iter().copied().collect::<Vec<_>>(),
                            loaded,
                            super::full_lens::FullTokenTargetCovector::RawLmHead,
                        )
                        .map(native_lens_from_projected_full)
                    })
                    .transpose()?;
                PreparedLens {
                    id: id.clone(),
                    lens: LoadedLens::Native(native_lens_from_projected_full(projected)),
                    raw_lm_head,
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
                    raw_lm_head: None,
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

    let mut directions = HashMap::new();
    for (direction_id, layers) in direction_layers {
        let definition = direction_defs[direction_id];
        let definition = definition.lens_row().with_context(|| {
            format!("ordinary runtime cannot load native hyper direction {direction_id}")
        })?;
        let prepared_lens = &lenses[&definition.lens];
        let mut rows = BTreeMap::new();
        for layer in layers {
            let raw = direction_row(prepared_lens, definition, layer)?;
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

fn native_lens_from_projected_full(
    projected: super::full_lens::ProjectedFullTokenDirections,
) -> NativeLens {
    NativeLens {
        method: projected.method,
        target_layer: projected.target_layer,
        source_layers: projected.source_layers,
        token_ids: projected.token_ids,
        hidden_size: projected.hidden_size,
        values: projected.values,
    }
}

fn direction_row(
    prepared: &PreparedLens,
    definition: &LensRowDirectionDefinition,
    layer: u32,
) -> Result<Vec<f32>> {
    match definition.effective_target_covector() {
        DirectionTargetCovector::DeployedLogitNumerator => {
            lens_row(prepared, &definition.row, layer)
        }
        DirectionTargetCovector::RawLmHead => {
            raw_lm_head_direction_row(prepared, &definition.row, layer, &definition.id)
        }
        DirectionTargetCovector::RawLmHeadOrthogonalToDeployedLogitNumerator => {
            let raw = raw_lm_head_direction_row(prepared, &definition.row, layer, &definition.id)?;
            let deployed = lens_row(prepared, &definition.row, layer)?;
            orthogonal_component(&raw, &deployed, &definition.id)
        }
    }
}

fn raw_lm_head_direction_row(
    prepared: &PreparedLens,
    selector: &DirectionRow,
    layer: u32,
    direction_id: &str,
) -> Result<Vec<f32>> {
    let raw = prepared
        .raw_lm_head
        .as_ref()
        .with_context(|| format!("direction {direction_id} requires a raw LM-head projection"))?;
    native_lens_row(raw, selector, layer, &prepared.id)
}

fn orthogonal_component(raw: &[f32], axis: &[f32], direction_id: &str) -> Result<Vec<f32>> {
    ensure!(
        !raw.is_empty() && raw.len() == axis.len(),
        "direction {direction_id} orthogonal decomposition has incompatible rows"
    );
    ensure!(
        raw.iter().chain(axis).all(|value| value.is_finite()),
        "direction {direction_id} orthogonal decomposition has non-finite input"
    );
    let raw_norm_squared = raw
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let axis_norm_squared = axis
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let dot = raw
        .iter()
        .zip(axis)
        .map(|(&left, &right)| f64::from(left) * f64::from(right))
        .sum::<f64>();
    ensure!(
        raw_norm_squared.is_finite()
            && raw_norm_squared > 0.0
            && axis_norm_squared.is_finite()
            && axis_norm_squared > 0.0
            && dot.is_finite(),
        "direction {direction_id} orthogonal decomposition has invalid geometry"
    );
    let scale = dot / axis_norm_squared;
    let orthogonal = raw
        .iter()
        .zip(axis)
        .map(|(&raw_value, &axis_value)| {
            (f64::from(raw_value) - scale * f64::from(axis_value)) as f32
        })
        .collect::<Vec<_>>();
    let orthogonal_norm_squared = orthogonal
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    ensure!(
        orthogonal_norm_squared.is_finite() && orthogonal_norm_squared > raw_norm_squared * 1.0e-12,
        "direction {direction_id} raw and deployed covectors are numerically collinear"
    );
    Ok(orthogonal)
}

fn lens_row(prepared: &PreparedLens, selector: &DirectionRow, layer: u32) -> Result<Vec<f32>> {
    match (&prepared.lens, selector) {
        (LoadedLens::Native(native), DirectionRow::TokenId { .. }) => {
            native_lens_row(native, selector, layer, &prepared.id)
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

fn native_lens_row(
    native: &NativeLens,
    selector: &DirectionRow,
    layer: u32,
    lens_id: &str,
) -> Result<Vec<f32>> {
    let DirectionRow::TokenId { token_id } = selector else {
        bail!("native selected directions require row.kind=token_id")
    };
    let layer_slot = native
        .source_layers
        .iter()
        .position(|&candidate| candidate == layer)
        .with_context(|| format!("native lens {lens_id} has no row for layer {layer}"))?;
    let token_slot = native
        .token_ids
        .iter()
        .position(|&candidate| candidate == *token_id)
        .with_context(|| format!("native lens {lens_id} has no token row {token_id}"))?;
    let offset = native_payload_offset(
        layer_slot,
        token_slot,
        native.token_ids.len(),
        native.hidden_size,
    );
    Ok(native.values[offset..offset + native.hidden_size].to_vec())
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

fn phase_needs_logits(phase: Phase, prompt_len: usize) -> bool {
    match phase {
        Phase::Prefill(index) => prompt_len.checked_sub(1) == Some(index),
        Phase::Decode(_) => true,
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EventForwardRoute {
    ProductionFullTail,
    ProductionFullTailDiscardLogits,
    SerialFullTailCapture,
    SerialFullTailNoCapture,
    SerialNoTailCapture,
    SerialNoTailNoCapture,
}

fn event_forward_route(
    needs_logits: bool,
    has_capture_plan: bool,
    has_active_readouts: bool,
    has_operation_topology: bool,
) -> EventForwardRoute {
    debug_assert!(!has_active_readouts || has_capture_plan);
    if has_active_readouts {
        if needs_logits {
            EventForwardRoute::SerialFullTailCapture
        } else {
            EventForwardRoute::SerialNoTailCapture
        }
    } else if has_capture_plan || has_operation_topology {
        if needs_logits {
            EventForwardRoute::SerialFullTailNoCapture
        } else {
            EventForwardRoute::SerialNoTailNoCapture
        }
    } else if needs_logits {
        EventForwardRoute::ProductionFullTail
    } else {
        EventForwardRoute::ProductionFullTailDiscardLogits
    }
}

fn forward_event(
    execution: &ExecutionPlan,
    schedule: &BoundEventSchedule<'_, '_>,
    forward: &qwen_llm::metal_forward::MetalForward<'_>,
    token: i32,
    position: u32,
    sequence: &mut qwen_llm::runtime::Sequence,
    phase: Phase,
    event: &CompiledEvent,
    needs_logits: bool,
    operation_applications: &mut Vec<OperationApplication>,
    live_readouts: &mut Vec<LiveReadout>,
) -> Result<Vec<f32>> {
    let mut interventions = Vec::new();
    for layer in 0..execution.n_layer {
        for &definition_index in event.operation_indices() {
            if !schedule.operation_selects_layer(definition_index, layer) {
                continue;
            }
            let operation = &schedule.plan().operations[definition_index];
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
    let borrowed = interventions
        .iter()
        .map(|(_, _, op)| *op)
        .collect::<Vec<_>>();
    sequence.check_position(position as usize)?;
    sequence.ensure_can_append(1)?;
    let has_readouts = !event.readout_indices().is_empty();
    let route = event_forward_route(
        needs_logits,
        execution.capture.is_some(),
        has_readouts,
        !event.operation_topology_indices().is_empty(),
    );
    let capture = if has_readouts {
        ensure!(
            !event.capture_layers().is_empty(),
            "active Lens readout selected no capture layers"
        );
        let capture_elements = event
            .capture_layers()
            .len()
            .checked_mul(execution.hidden_size)
            .context("event-local Lens capture size overflow")?;
        let capture_elements = u64::try_from(capture_elements)
            .context("event-local Lens capture exceeds Metal addressing")?;
        Some(
            execution
                .capture
                .as_ref()
                .context("active Lens readout has no capture storage")?
                .view_subrange(0, vec![capture_elements]),
        )
    } else {
        None
    };
    let logits = match route {
        EventForwardRoute::SerialFullTailCapture => {
            let capture = capture
                .as_ref()
                .context("active Lens readout has no capture view")?;
            forward.single_token_with_post_block_interventions(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                event.capture_layers(),
                capture,
                &borrowed,
            )?
        }
        EventForwardRoute::SerialFullTailNoCapture => forward
            .single_token_with_post_block_interventions_no_capture(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                &borrowed,
            )?,
        EventForwardRoute::ProductionFullTail => {
            forward.single_token(token, position, unsafe { sequence.metal_session_mut() })?
        }
        EventForwardRoute::SerialNoTailCapture => {
            let capture = capture
                .as_ref()
                .context("active Lens readout has no capture view")?;
            forward.single_token_with_post_block_interventions_no_tail(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                event.capture_layers(),
                capture,
                &borrowed,
            )?;
            Vec::new()
        }
        EventForwardRoute::SerialNoTailNoCapture => {
            forward.single_token_with_post_block_interventions_no_capture_no_tail(
                token,
                position,
                unsafe { sequence.metal_session_mut() },
                &borrowed,
            )?;
            Vec::new()
        }
        EventForwardRoute::ProductionFullTailDiscardLogits => {
            forward.single_token(token, position, unsafe { sequence.metal_session_mut() })?;
            Vec::new()
        }
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
    if let Some(capture) = capture.as_ref() {
        let values = read_f32_tensor(
            capture,
            event.capture_layers().len() * execution.hidden_size,
        );
        for &definition_index in event.readout_indices() {
            let readout = &schedule.plan().readouts[definition_index];
            for (slot, &layer) in event.capture_layers().iter().enumerate() {
                if !schedule.readout_selects_layer(definition_index, layer) {
                    continue;
                }
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

fn operation_enabled(operation: &OperationDefinition) -> bool {
    operation.action.coefficient() != 0.0
}

fn scope_matches(scope: &Scope, phase: Phase, layer: u32) -> Result<bool> {
    let index = u32::try_from(phase.index()).context("event index exceeds u32")?;
    let phase_matches = match phase {
        Phase::Prefill(_) => match &scope.prefill {
            Some(selector) => selector_contains(selector, index)?,
            None => false,
        },
        Phase::Decode(_) => match &scope.decode {
            Some(selector) => selector_contains(selector, index)?,
            None => false,
        },
    };
    Ok(phase_matches && selector_contains(&scope.layers, layer)?)
}

fn selector_contains(selector: &Selector, value: u32) -> Result<bool> {
    Ok(match selector {
        Selector::All => true,
        Selector::Values { values } => values.binary_search(&value).is_ok(),
        Selector::Range { start, end } => (*start..=*end).contains(&value),
        Selector::RenderedSpans { .. } => bail!("execution plan contains an unresolved selector"),
    })
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

    #[derive(Debug, Parser)]
    struct SweepArgsParser {
        #[command(flatten)]
        args: CoefficientSweepArgs,
    }

    #[test]
    fn request_capacity_follows_model_context_not_cli_constants() {
        assert_eq!(required_forward_count(1, 1).unwrap(), 1);
        assert_eq!(required_forward_count(493, 16_384).unwrap(), 16_876);
        assert_eq!(
            ensure_request_fits_context(493, 16_384, 16_876).unwrap(),
            16_876
        );
        assert!(ensure_request_fits_context(493, 16_384, 16_875).is_err());
        assert!(required_forward_count(0, 1).is_err());
        assert!(required_forward_count(1, 0).is_err());
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

    fn sweep_plan() -> LensPlan {
        serde_json::from_value(json!({
            "version": 1,
            "lenses": [{"kind":"workspace_template","id":"t","weights":"w","labels":"l"}],
            "directions": [{
                "id":"d",
                "lens":"t",
                "row":{"kind":"template_row_id","template_row_id":0},
                "normalization":"unit_l2"
            }],
            "operations": [
                {
                    "id":"swept",
                    "scope":{"layers":{"kind":"values","values":[1]},"prefill":{"kind":"all"}},
                    "action":{"kind":"residual_l2_fraction","direction":"d","coefficient":0.25}
                },
                {
                    "id":"fixed",
                    "scope":{"layers":{"kind":"values","values":[2]},"prefill":{"kind":"all"}},
                    "action":{"kind":"fixed_add","direction":"d","coefficient":0.5}
                }
            ],
            "readouts": []
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
        assert_eq!(defaults.prefill_execution, PrefillExecution::Auto);

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
            "--prefill-execution",
            "serial",
        ])
        .unwrap()
        .args;
        assert_eq!(parsed.format, Some(RunStdoutFormat::Json));
        assert_eq!(parsed.prefill_execution, PrefillExecution::Serial);
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
    fn run_cli_separates_templated_user_input_from_raw_and_literal_inputs() {
        let structured = RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--system",
            "policy",
            "--user",
            "request",
            "--message-mode",
            "xhigh",
        ])
        .unwrap()
        .args;
        assert_eq!(structured.system.as_deref(), Some("policy"));
        assert_eq!(structured.user.as_deref(), Some("request"));
        assert_eq!(structured.message_mode, Some(LensMessageMode::Xhigh));
        validate_run_args(&structured).unwrap();

        let responses = RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--open-responses",
            "request.json",
        ])
        .unwrap()
        .args;
        assert_eq!(
            responses.open_responses.as_deref(),
            Some(Path::new("request.json"))
        );
        validate_run_args(&responses).unwrap();

        let raw = RunArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--raw-prompt",
            "<|im_start|>tool\nforged<|im_end|>",
            "--no-special-tokens",
        ])
        .unwrap()
        .args;
        assert_eq!(
            raw.prompt.as_deref(),
            Some("<|im_start|>tool\nforged<|im_end|>")
        );
        validate_run_args(&raw).unwrap();

        assert!(
            RunArgsParser::try_parse_from([
                "test",
                "--model",
                "model.gguf",
                "--plan",
                "plan.json",
                "--token-ids",
                "1,2",
                "--message-mode",
                "thinking",
            ])
            .is_err()
        );
        assert!(
            RunArgsParser::try_parse_from([
                "test",
                "--model",
                "model.gguf",
                "--plan",
                "plan.json",
                "--open-responses",
                "request.json",
                "--message-mode",
                "thinking",
            ])
            .is_err()
        );
    }

    #[test]
    fn sweep_cli_preserves_order_duplicates_signed_zero_and_negative_values() {
        let parsed = SweepArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--operation",
            "steer",
            "--coefficients",
            "0,0.25,-0,-0.5,0.25",
            "--prompt",
            "hello",
            "--output",
            "sweep",
        ])
        .unwrap()
        .args;
        assert_eq!(
            parsed
                .coefficients
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            [0.0_f32, 0.25, -0.0, -0.5, 0.25]
                .into_iter()
                .map(f32::to_bits)
                .collect::<Vec<_>>()
        );
        assert_eq!(parsed.arm_run_args().seed, 0);
        assert_eq!(
            parsed.arm_run_args().prefill_execution,
            PrefillExecution::Auto
        );
        validate_coefficient_sweep_args(&parsed).unwrap();

        let responses = SweepArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--operation",
            "steer",
            "--coefficients",
            "0,0.1",
            "--open-responses",
            "request.json",
            "--output",
            "sweep",
        ])
        .unwrap()
        .args;
        assert_eq!(
            responses.arm_run_args().open_responses.as_deref(),
            Some(Path::new("request.json"))
        );
        validate_coefficient_sweep_args(&responses).unwrap();

        let mut excessive = parsed;
        excessive.coefficients = vec![1.0; MAX_SWEEP_ARMS + 1];
        assert!(validate_coefficient_sweep_args(&excessive).is_err());
        excessive.coefficients = vec![f32::NAN];
        assert!(validate_coefficient_sweep_args(&excessive).is_err());
    }

    #[test]
    fn sweep_cohort_cli_accepts_auto_and_explicit_serial_prefill_policy() {
        let parsed = SweepArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--operation",
            "steer",
            "--coefficients",
            "0,0.5",
            "--requests-jsonl",
            "requests.jsonl",
            "--output",
            "cohort",
        ])
        .unwrap()
        .args;
        assert_eq!(
            parsed.requests_jsonl.as_deref(),
            Some(Path::new("requests.jsonl"))
        );
        assert_eq!(parsed.prefill_execution, PrefillExecution::Auto);
        validate_coefficient_sweep_args(&parsed).unwrap();

        let explicit_serial = SweepArgsParser::try_parse_from([
            "test",
            "--model",
            "model.gguf",
            "--plan",
            "plan.json",
            "--operation",
            "steer",
            "--coefficients",
            "0",
            "--requests-jsonl",
            "requests.jsonl",
            "--prefill-execution",
            "serial",
            "--output",
            "cohort",
        ])
        .unwrap()
        .args;
        validate_coefficient_sweep_args(&explicit_serial).unwrap();
        assert!(
            SweepArgsParser::try_parse_from([
                "test",
                "--model",
                "model.gguf",
                "--plan",
                "plan.json",
                "--operation",
                "steer",
                "--coefficients",
                "0",
                "--requests-jsonl",
                "requests.jsonl",
                "--messages",
                "one.json",
                "--prefill-execution",
                "serial",
                "--output",
                "cohort",
            ])
            .is_err()
        );
    }

    #[test]
    fn sweep_cohort_jsonl_is_strict_bounded_and_resolves_relative_message_paths() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-cohort-jsonl-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("one.json"), "[]").unwrap();
        std::fs::write(root.join("two.json"), "[]").unwrap();
        let requests_path = root.join("requests.jsonl");
        std::fs::write(
            &requests_path,
            concat!(
                "{\"id\":\"first-safe\",\"messages\":\"one.json\"}\n",
                "\n",
                "{\"id\":\"second_safe\",\"messages\":\"two.json\",\"message_mode\":\"thinking\"}\n"
            ),
        )
        .unwrap();
        let loaded = read_sweep_cohort_requests(&requests_path, 2, 64).unwrap();
        assert!(loaded.canonical_path.is_absolute());
        assert_eq!(loaded.records.len(), 2);
        assert_eq!(loaded.records[0].0, 1);
        assert_eq!(loaded.records[1].0, 3);
        assert_eq!(
            loaded.records[0].1.messages,
            loaded.canonical_path.parent().unwrap().join("one.json")
        );
        assert_eq!(
            loaded.records[1].1.message_mode,
            Some(LensMessageMode::Thinking)
        );

        std::fs::write(
            &requests_path,
            concat!(
                "{\"id\":\"first\",\"messages\":\"one.json\",\"unknown\":true}\n",
                "{\"id\":\"second\",\"messages\":\"two.json\"}\n"
            ),
        )
        .unwrap();
        assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
        std::fs::write(
            &requests_path,
            concat!(
                "{\"id\":\"../unsafe\",\"messages\":\"one.json\"}\n",
                "{\"id\":\"second\",\"messages\":\"two.json\"}\n"
            ),
        )
        .unwrap();
        assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sweep_cohort_jsonl_rejects_invalid_records_and_accepts_resource_bounded_counts() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-cohort-bounds-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        std::fs::write(root.join("messages.json"), "[]").unwrap();
        let requests_path = root.join("requests.jsonl");
        std::fs::write(
            &requests_path,
            concat!(
                "{\"id\":\"same\",\"messages\":\"messages.json\"}\n",
                "{\"id\":\"same\",\"messages\":\"messages.json\"}\n"
            ),
        )
        .unwrap();
        assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
        std::fs::write(
            &requests_path,
            concat!(
                "{\"id\":\"one\",\"messages\":\"-\"}\n",
                "{\"id\":\"two\",\"messages\":\"messages.json\"}\n"
            ),
        )
        .unwrap();
        assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
        std::fs::write(
            &requests_path,
            "{\"id\":\"one\",\"messages\":\"messages.json\"}\n",
        )
        .unwrap();
        assert!(read_sweep_cohort_requests(&requests_path, 2, 64).is_err());
        assert!(checked_sweep_cohort_child_count(usize::MAX, 2).is_err());

        let expanded = (0..97)
            .map(|index| format!("{{\"id\":\"request-{index}\",\"messages\":\"messages.json\"}}\n"))
            .collect::<String>();
        std::fs::write(&requests_path, expanded).unwrap();
        let loaded = read_sweep_cohort_requests(&requests_path, 2, 64).unwrap();
        assert_eq!(loaded.records.len(), 97);
        assert_eq!(checked_sweep_cohort_child_count(97, 2).unwrap(), 194);
        assert!(read_sweep_cohort_requests(&requests_path, 1, 499_999).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sweep_cohort_messages_are_single_read_no_follow_and_content_bound() {
        static READS: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-cohort-capture-{}-{}",
            std::process::id(),
            READS.load(Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let path = root.join("messages.json");
        let original = br#"[{"role":"user","content":"original"}]"#;
        std::fs::write(&path, original).unwrap();
        READS.store(0, Ordering::Relaxed);
        let captured = capture_sweep_cohort_messages_with(&path, |path| {
            READS.fetch_add(1, Ordering::Relaxed);
            super::super::read_regular_file_bounded(path, MAX_SWEEP_COHORT_MESSAGES_BYTES)
        })
        .unwrap();
        std::fs::write(&path, br#"[{"role":"user","content":"replacement"}]"#).unwrap();
        assert_eq!(READS.load(Ordering::Relaxed), 1);
        assert_eq!(captured.bytes, original);
        assert_eq!(captured.blake3, blake3::hash(original).to_hex().to_string());
        let parsed = crate::messages::parse_strict_messages_input(
            std::str::from_utf8(&captured.bytes).unwrap(),
            &captured.path.display().to_string(),
        )
        .unwrap();
        assert_eq!(parsed[0].content, "original");

        let symlink = root.join("messages-link.json");
        std::os::unix::fs::symlink(&path, &symlink).unwrap();
        assert!(capture_sweep_cohort_messages(&symlink).is_err());
        let oversized = root.join("oversized.json");
        std::fs::File::create(&oversized)
            .unwrap()
            .set_len((MAX_SWEEP_COHORT_MESSAGES_BYTES + 1) as u64)
            .unwrap();
        assert!(capture_sweep_cohort_messages(&oversized).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sweep_cohort_aggregate_bounds_admit_campaign_and_reject_pathological_work() {
        let admitted = plan_sweep_cohort_bounds(&vec![4_096; 64], 3, 64).unwrap();
        assert_eq!(admitted.request_count, 64);
        assert_eq!(admitted.total_arm_count, 192);
        assert_eq!(admitted.transition_upper_bound, 798_720);

        assert!(plan_sweep_cohort_bounds(&vec![4_096; 64], 64, 64).is_err());
        assert!(plan_sweep_cohort_bounds(&vec![11_000; 32], 3, 64).is_err());
    }

    #[test]
    fn sweep_cohort_output_budget_overflow_cleans_staging() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-cohort-output-budget-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let output = root.join("cohort");
        assert!(
            stage_and_publish_sweep(&output, |staging| -> Result<()> {
                write_new_sweep_file(&staging.join("partial"), b"partial")?;
                let mut budget = SweepOutputBudget::new(0)?;
                budget.consumed = MAX_SWEEP_BUNDLE_BYTES - 1;
                budget.charge(2)
            })
            .is_err()
        );
        assert!(!output.exists());
        assert!(std::fs::read_dir(&root).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .contains(".stage.")
        }));
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sweep_output_budget_reserves_then_reconciles_outer_manifest() {
        let mut budget = SweepOutputBudget::new(8).unwrap();
        budget.consumed = MAX_SWEEP_BUNDLE_BYTES - 8;
        assert!(budget.charge(1).is_err());
        assert_eq!(budget.consumed, MAX_SWEEP_BUNDLE_BYTES - 8);
        budget.release_reservation();
        budget.charge(8).unwrap();
        assert_eq!(budget.consumed, MAX_SWEEP_BUNDLE_BYTES);
    }

    #[test]
    #[ignore = "requires a compatible local GGUF, Lens plan, operation ID, and two message fixtures"]
    fn sweep_cohort_model_bound_children_are_inspectable() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let model = std::env::var_os("QWEN_LENS_COHORT_MODEL").unwrap();
        let plan = std::env::var_os("QWEN_LENS_COHORT_PLAN").unwrap();
        let operation = std::env::var("QWEN_LENS_COHORT_OPERATION").unwrap();
        let first_messages =
            PathBuf::from(std::env::var_os("QWEN_LENS_COHORT_MESSAGES_FIRST").unwrap());
        let second_messages =
            PathBuf::from(std::env::var_os("QWEN_LENS_COHORT_MESSAGES_SECOND").unwrap());
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-cohort-model-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&root).unwrap();
        let requests_path = root.join("requests.jsonl");
        let records = [
            json!({"id":"first","messages":first_messages}),
            json!({"id":"second","messages":second_messages}),
        ];
        std::fs::write(
            &requests_path,
            records
                .iter()
                .map(|record| serde_json::to_string(record).unwrap())
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let output = root.join("cohort");
        run_coefficient_sweep(CoefficientSweepArgs {
            model: model.into(),
            plan: plan.into(),
            operation,
            coefficients: vec![0.0, 0.5],
            prompt: None,
            token_ids: None,
            user: None,
            system: None,
            messages: None,
            open_responses: None,
            requests_jsonl: Some(requests_path),
            message_mode: None,
            no_special_tokens: false,
            max_new_tokens: 1,
            prefill_execution: PrefillExecution::Auto,
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            min_p: 0.0,
            seed: 0,
            output: output.clone(),
        })
        .unwrap();
        let manifest_bytes = std::fs::read(output.join(SWEEP_MANIFEST_NAME)).unwrap();
        let manifest = parse_sweep_cohort_manifest_bytes(&manifest_bytes).unwrap();
        for child in &manifest.sweeps {
            verify_sweep_cohort_child_manifest(&output, child).unwrap();
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn every_action_coefficient_can_be_overridden() {
        let mut actions = vec![
            Action::FixedAdd {
                direction: "a".into(),
                coefficient: 1.0,
            },
            Action::ResidualL2Fraction {
                direction: "a".into(),
                coefficient: 1.0,
            },
            Action::ProjectionAblate {
                direction: "a".into(),
                coefficient: 1.0,
            },
            Action::SourceToTarget {
                source: "a".into(),
                target: "b".into(),
                coefficient: 1.0,
            },
            Action::CoordinateSwap {
                source: "a".into(),
                target: "b".into(),
                coefficient: 1.0,
            },
        ];
        for action in &mut actions {
            action.set_coefficient(-0.125);
            assert_eq!(action.coefficient().to_bits(), (-0.125_f32).to_bits());
        }
    }

    #[test]
    fn sweep_changes_only_the_selected_operation_and_supports_zero_controls() {
        let source = sweep_plan();
        validate_plan(&source).unwrap();
        let zero = plan_with_operation_coefficient(&source, "swept", -0.0).unwrap();
        assert_eq!(source.operations[0].action.coefficient(), 0.25);
        assert_eq!(
            zero.operations[0].action.coefficient().to_bits(),
            (-0.0_f32).to_bits()
        );
        assert_eq!(zero.operations[1], source.operations[1]);
        assert!(!operation_enabled(&zero.operations[0]));
        assert!(operation_enabled(&zero.operations[1]));
        validate_plan(&zero).unwrap();
        assert!(validate_sweep_source_operation(&zero, "swept").is_err());
        validate_sweep_source_operation(&source, "swept").unwrap();

        let mut wrong_zero = source.clone();
        wrong_zero.operations[1].action.set_coefficient(0.0);
        validate_plan(&wrong_zero).unwrap();
        assert!(!operation_enabled(&wrong_zero.operations[1]));
        assert!(plan_with_operation_coefficient(&source, "missing", 1.0).is_err());

        let first = plan_with_operation_coefficient(&source, "swept", 0.0).unwrap();
        let middle = plan_with_operation_coefficient(&source, "swept", 0.75).unwrap();
        let last = plan_with_operation_coefficient(&source, "swept", 0.0).unwrap();
        assert_eq!(first, last);
        assert_ne!(first, middle);
    }

    #[test]
    fn sweep_manifest_is_ordered_bounded_and_preserves_signed_zero() {
        let source_plan = sweep_plan();
        let coefficients = vec![0.0, 0.5, -0.0];
        let arms = coefficients
            .iter()
            .enumerate()
            .map(|(index, &coefficient)| CoefficientSweepArm {
                index,
                coefficient,
                artifact: format!("arms/{index:06}/run.json"),
                byte_length: 10,
                blake3: "0".repeat(64),
            })
            .collect();
        let manifest = CoefficientSweepManifest {
            schema: SWEEP_SCHEMA.into(),
            schema_version: SWEEP_SCHEMA_VERSION,
            producer: SweepProducer {
                build_commit: "a".repeat(40),
                build_dirty: "0".into(),
                build_source_state: format!("git-source-sha256-v2:{}", "b".repeat(64)),
            },
            canonical_source_plan_path: "/tmp/plan.json".into(),
            source_plan: Some(source_plan.clone()),
            source_plan_canonical_json_blake3: Some(canonical_plan_blake3(&source_plan).unwrap()),
            operation_id: "swept".into(),
            coefficients,
            arms,
        };
        let bytes = serialize_sweep_manifest(&manifest).unwrap();
        let decoded = parse_sweep_manifest_bytes(&bytes).unwrap();
        assert_eq!(decoded.coefficients[2].to_bits(), (-0.0_f32).to_bits());
        assert_eq!(decoded.arms[2].coefficient.to_bits(), (-0.0_f32).to_bits());
        let arm_bytes = decoded.arms.iter().map(|arm| arm.byte_length).sum::<u64>();
        let largest_manifest = usize::try_from(MAX_SWEEP_BUNDLE_BYTES - arm_bytes).unwrap();
        validate_sweep_bundle_size(&decoded, largest_manifest).unwrap();
        assert!(validate_sweep_bundle_size(&decoded, largest_manifest + 1).is_err());

        let mut malformed = decoded.clone();
        malformed.arms[1].artifact = "../run.json".into();
        assert!(validate_sweep_manifest(&malformed).is_err());

        let mut malformed_producer = decoded;
        malformed_producer.producer.build_commit = "not-a-commit".into();
        assert!(validate_sweep_manifest(&malformed_producer).is_err());
    }

    #[test]
    fn sweep_cohort_manifest_roundtrips_and_verifies_child_manifest_integrity() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let source_plan = sweep_plan();
        let child_manifest = CoefficientSweepManifest {
            schema: SWEEP_SCHEMA.into(),
            schema_version: SWEEP_SCHEMA_VERSION,
            producer: SweepProducer {
                build_commit: "a".repeat(40),
                build_dirty: "0".into(),
                build_source_state: format!("git-source-sha256-v2:{}", "b".repeat(64)),
            },
            canonical_source_plan_path: "/tmp/plan.json".into(),
            source_plan: Some(source_plan.clone()),
            source_plan_canonical_json_blake3: Some(canonical_plan_blake3(&source_plan).unwrap()),
            operation_id: "swept".into(),
            coefficients: vec![0.0],
            arms: vec![CoefficientSweepArm {
                index: 0,
                coefficient: 0.0,
                artifact: "arms/000000/run.json".into(),
                byte_length: 1,
                blake3: "c".repeat(64),
            }],
        };
        let child_bytes = serialize_sweep_manifest(&child_manifest).unwrap();
        let child_digest = blake3::hash(&child_bytes).to_hex().to_string();
        let children = (0..2)
            .map(|index| SweepCohortChild {
                index,
                id: format!("request-{index}"),
                source_line: index + 1,
                path: format!("sweeps/{index:06}"),
                prompt_token_count: 4,
                messages_path: format!("/tmp/messages-{index}.json").into(),
                messages_blake3: "e".repeat(64),
                serialized_byte_length: child_bytes.len() as u64 + 1,
                manifest_byte_length: child_bytes.len() as u64,
                manifest_blake3: child_digest.clone(),
            })
            .collect::<Vec<_>>();
        let manifest = SweepCohortManifest {
            schema: SWEEP_COHORT_SCHEMA.into(),
            schema_version: SWEEP_COHORT_SCHEMA_VERSION,
            producer: child_manifest.producer.clone(),
            requests_jsonl_path: "/tmp/requests.jsonl".into(),
            requests_jsonl_blake3: "d".repeat(64),
            canonical_source_plan_path: "/tmp/plan.json".into(),
            source_plan: source_plan.clone(),
            source_plan_canonical_json_blake3: canonical_plan_blake3(&source_plan).unwrap(),
            model_path: "model.gguf".into(),
            operation_id: "swept".into(),
            coefficients: vec![0.0],
            sampler: RunSampler {
                temperature: 0.0,
                top_k: 0,
                top_p: 1.0,
                min_p: 0.0,
                seed: 7,
            },
            max_new_tokens: 1,
            prefill_execution: PrefillExecution::Serial,
            execution_policy:
                "serial_prompts_serial_arms_fresh_sequence_and_sampler_no_batched_generation".into(),
            planned_request_count: 2,
            planned_total_arm_count: 2,
            transition_upper_bound: 10,
            cumulative_serialized_child_bytes: 2 * (child_bytes.len() as u64 + 1),
            sweeps: children,
        };
        let bytes = serialize_sweep_cohort_manifest(&manifest).unwrap();
        assert_eq!(parse_sweep_cohort_manifest_bytes(&bytes).unwrap(), manifest);
        let largest_manifest =
            usize::try_from(MAX_SWEEP_BUNDLE_BYTES - manifest.cumulative_serialized_child_bytes)
                .unwrap();
        validate_sweep_cohort_bundle_size(&manifest, largest_manifest).unwrap();
        assert!(validate_sweep_cohort_bundle_size(&manifest, largest_manifest + 1).is_err());
        let mut malformed_aggregate = manifest.clone();
        malformed_aggregate.cumulative_serialized_child_bytes += 1;
        assert!(validate_sweep_cohort_manifest(&malformed_aggregate).is_err());
        let mut malformed_binding = manifest.clone();
        malformed_binding.sweeps[0].messages_blake3 = "not-a-digest".into();
        assert!(validate_sweep_cohort_manifest(&malformed_binding).is_err());

        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-cohort-integrity-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(root.join("sweeps/000000")).unwrap();
        std::fs::write(root.join("sweeps/000000/manifest.json"), &child_bytes).unwrap();
        verify_sweep_cohort_child_manifest(&root, &manifest.sweeps[0]).unwrap();
        let mut drifted = manifest.sweeps[0].clone();
        drifted.manifest_blake3 = "e".repeat(64);
        assert!(verify_sweep_cohort_child_manifest(&root, &drifted).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn sweep_cohort_binds_semantic_positions_independently_per_prompt() {
        let plan = semantic_readout_plan(json!({
            "kind":"rendered_spans",
            "selectors":[{"span_kind":"message_content","role":"user","edge":"end"}]
        }));
        let rendering = |range| LensInputRendering {
            renderer: "qwen_chatml_messages_v1".into(),
            generation_mode: Some("auto".into()),
            spans: vec![rendered_span(
                "message_content",
                Some(0),
                "user",
                None,
                Some(range),
            )],
        };
        let first = bind_plan_positions(&plan, &rendering((1, 3)), 3).unwrap();
        let second = bind_plan_positions(&plan, &rendering((4, 8)), 8).unwrap();
        assert_eq!(first.position_bindings[0].resolved_index, 2);
        assert_eq!(second.position_bindings[0].resolved_index, 7);
        assert_ne!(first.resolved, second.resolved);
        assert_eq!(first.authored, second.authored);
    }

    #[test]
    fn sweep_staging_publishes_exclusively_and_cleans_its_failures() {
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        let root = std::env::temp_dir().join(format!(
            "qwen-lens-sweep-stage-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        DirBuilder::new().mode(0o700).create(&root).unwrap();

        let published = root.join("published");
        stage_and_publish_sweep(&published, |staging| {
            write_new_sweep_file(&staging.join("marker"), b"complete")?;
            super::super::sync_directory(staging)?;
            Ok(())
        })
        .unwrap();
        assert_eq!(
            std::fs::read(published.join("marker")).unwrap(),
            b"complete"
        );

        let failed = root.join("failed");
        assert!(
            stage_and_publish_sweep(&failed, |staging| -> Result<()> {
                let nested = staging.join("sweeps");
                create_sweep_directory(&nested)?;
                write_new_sweep_file(&nested.join("partial"), b"partial")?;
                bail!("injected failure")
            })
            .is_err()
        );
        assert!(!failed.exists());

        let raced = root.join("raced");
        assert!(
            stage_and_publish_sweep(&raced, |staging| {
                write_new_sweep_file(&staging.join("marker"), b"ours")?;
                create_sweep_directory(&raced)?;
                Ok(())
            })
            .is_err()
        );
        assert!(raced.is_dir());
        let leftovers = std::fs::read_dir(&root)
            .unwrap()
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.file_name().to_string_lossy().contains(".stage."))
            .count();
        assert_eq!(leftovers, 0);
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn run_artifact_serializes_envelope_exact_plan_and_score_semantics() {
        let plan = minimal_plan();
        let plan_digest = canonical_plan_blake3(&plan).unwrap();
        let artifact = RunOutput {
            schema: RUN_SCHEMA,
            schema_version: RUN_SCHEMA_VERSION,
            runtime_kind: "ordinary_qwen",
            model_path: "model.gguf".into(),
            canonical_plan_path: "/canonical/plan.json".into(),
            authored_plan: plan.clone(),
            authored_plan_canonical_json_blake3: plan_digest,
            requested_live_readouts: plan.readouts.clone(),
            plan,
            position_bindings: Vec::new(),
            input_source: "prompt",
            add_special_tokens: Some(true),
            rendering: LensInputRendering {
                renderer: "tokenizer_text".into(),
                generation_mode: None,
                spans: Vec::new(),
            },
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
            execution: RunExecution::runtime_serial(
                PrefillExecution::Auto,
                RunSerialReason::NoEligiblePassiveSpan,
            ),
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
            execution_binding: None,
        };
        let value = serde_json::to_value(&artifact).unwrap();
        assert_eq!(value["schema"], RUN_SCHEMA);
        assert_eq!(value["schema_version"], 5);
        assert_eq!(value["input_source"], "prompt");
        assert_eq!(value["add_special_tokens"], true);
        assert_eq!(value["rendering"]["renderer"], "tokenizer_text");
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
    fn run_execution_metadata_distinguishes_serial_controls_from_packed_spans() {
        let automatic_serial = RunExecution::runtime_serial(
            PrefillExecution::Auto,
            RunSerialReason::NoEligiblePassiveSpan,
        );
        automatic_serial.validate("ordinary_qwen", 8).unwrap();
        assert_eq!(
            automatic_serial.serial_reason(),
            Some(RunSerialReason::NoEligiblePassiveSpan)
        );

        let explicit_serial = RunExecution::runtime_serial(
            PrefillExecution::Serial,
            RunSerialReason::MusePackedNotImplemented,
        );
        explicit_serial.validate("ordinary_qwen", 8).unwrap();
        assert_eq!(
            explicit_serial.serial_reason(),
            Some(RunSerialReason::RequestedSerial)
        );

        let cohort_auto = RunExecution::serial(
            PrefillExecution::Auto,
            RunExecutionScheduleBasis::SweepSourcePlan,
            RunSerialReason::CohortSerialPolicy,
        );
        cohort_auto.validate("ordinary_qwen", 8).unwrap();
        assert_eq!(
            cohort_auto.serial_reason(),
            Some(RunSerialReason::CohortSerialPolicy)
        );

        let packed = RunExecution::dense_packed(
            RunExecutionScheduleBasis::EffectivePlan,
            70,
            141,
            1,
            vec![
                RunPackedPrefillSpan { start: 0, end: 65 },
                RunPackedPrefillSpan {
                    start: 70,
                    end: 140,
                },
            ],
        );
        packed.validate("ordinary_qwen", 141).unwrap();

        let mut final_token_overlap = packed.clone();
        final_token_overlap.packed_spans[1].end = 141;
        assert!(final_token_overlap.validate("ordinary_qwen", 141).is_err());
        let mut wrong_block = packed;
        wrong_block.block_tokens = Some(65);
        assert!(wrong_block.validate("ordinary_qwen", 141).is_err());

        let mut scheduled_plan = minimal_plan();
        scheduled_plan.readouts[0].scope.prefill = Some(Selector::Values { values: vec![70] });
        let scheduled = RunExecution::dense_packed(
            RunExecutionScheduleBasis::EffectivePlan,
            70,
            0,
            1,
            vec![
                RunPackedPrefillSpan { start: 0, end: 70 },
                RunPackedPrefillSpan {
                    start: 71,
                    end: 140,
                },
            ],
        );
        scheduled
            .validate_against_plan("ordinary_qwen", &scheduled_plan, 141)
            .unwrap();
        let mut undersized_matrix = scheduled.clone();
        undersized_matrix.attention_matrix_max_position = Some(1);
        assert!(undersized_matrix.validate("ordinary_qwen", 141).is_err());
        assert!(
            automatic_serial
                .validate_against_plan("ordinary_qwen", &scheduled_plan, 141)
                .is_err()
        );
        let mut false_schedule = scheduled;
        false_schedule.packed_spans[0].end = 69;
        false_schedule.block_tokens = Some(69);
        assert!(
            false_schedule
                .validate_against_plan("ordinary_qwen", &scheduled_plan, 141)
                .is_err()
        );
    }

    #[test]
    fn summary_contains_required_counts_text_and_artifact_path() {
        let plan = minimal_plan();
        let plan_digest = canonical_plan_blake3(&plan).unwrap();
        let artifact = RunOutput {
            schema: RUN_SCHEMA,
            schema_version: RUN_SCHEMA_VERSION,
            runtime_kind: "muse_glimmer",
            model_path: "muse.gguf".into(),
            canonical_plan_path: "/canonical/plan.json".into(),
            authored_plan: plan.clone(),
            authored_plan_canonical_json_blake3: plan_digest,
            requested_live_readouts: plan.readouts.clone(),
            plan,
            position_bindings: Vec::new(),
            input_source: "token_ids",
            add_special_tokens: None,
            rendering: LensInputRendering {
                renderer: "literal_token_ids".into(),
                generation_mode: None,
                spans: Vec::new(),
            },
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
            execution: RunExecution::runtime_serial(
                PrefillExecution::Auto,
                RunSerialReason::MusePackedNotImplemented,
            ),
            operation_applications: vec![OperationApplication {
                id: "op".into(),
                layer: 1,
                phase: "prefill",
                index: 0,
            }],
            live_readouts: Vec::new(),
            native_hyper_captures: Vec::new(),
            execution_binding: None,
        };
        assert_eq!(
            run_summary(&artifact, Some(Path::new("/tmp/run.json"))),
            concat!(
                "runtime=muse_glimmer model=muse.gguf\n",
                "prefill_execution=serial serial_reason=muse_packed_not_implemented\n",
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

    fn rendered_span(
        kind: &str,
        message_index: Option<usize>,
        role: &str,
        channel: Option<&str>,
        token_range: Option<(usize, usize)>,
    ) -> LensRenderedSpan {
        LensRenderedSpan {
            kind: kind.into(),
            message_index,
            tool_call_index: None,
            role: Some(role.into()),
            channel: channel.map(str::to_owned),
            label: None,
            byte_start: token_range.map_or(0, |range| range.0),
            byte_end: token_range.map_or(1, |range| range.1),
            token_start: token_range.map(|range| range.0),
            token_end: token_range.map(|range| range.1),
        }
    }

    fn semantic_readout_plan(prefill: serde_json::Value) -> LensPlan {
        serde_json::from_value(json!({
            "version": 2,
            "lenses": [{"kind":"native_selected","id":"j","artifact":"j"}],
            "directions": [],
            "operations": [],
            "readouts": [{
                "id":"live",
                "lens":"j",
                "scope":{"layers":{"kind":"values","values":[1]},"prefill":prefill},
                "top_k":1
            }]
        }))
        .unwrap()
    }

    #[test]
    fn rendered_span_selectors_bind_exact_content_edges_and_generated_markers() {
        let plan = semantic_readout_plan(json!({
            "kind":"rendered_spans",
            "selectors":[
                {"span_kind":"message_content","role":"user","occurrence":"last","edge":"end"},
                {"span_kind":"generated_assistant_start_marker","edge":"start"}
            ]
        }));
        validate_plan(&plan).unwrap();
        let rendering = LensInputRendering {
            renderer: "qwen_open_responses_annotated_v1".into(),
            generation_mode: Some("auto".into()),
            spans: vec![
                rendered_span("message_content", Some(0), "user", None, Some((1, 3))),
                rendered_span("message_content", Some(2), "user", None, Some((5, 7))),
                rendered_span(
                    "generated_assistant_start_marker",
                    None,
                    "assistant",
                    None,
                    Some((8, 9)),
                ),
            ],
        };
        let bound = bind_plan_positions(&plan, &rendering, 9).unwrap();
        assert_eq!(
            bound.resolved.readouts[0].scope.prefill,
            Some(Selector::Values { values: vec![6, 8] })
        );
        assert_eq!(bound.position_bindings.len(), 2);
        assert_eq!(bound.position_bindings[0].rendering_span_index, 1);
        assert_eq!(bound.position_bindings[0].resolved_index, 6);
        assert_eq!(bound.position_bindings[1].resolved_index, 8);
        assert_eq!(
            bound.authored_plan_canonical_json_blake3,
            canonical_plan_blake3(&plan).unwrap()
        );
    }

    #[test]
    fn tool_result_selectors_are_channel_portable_but_preserve_actual_roles() {
        let plan = semantic_readout_plan(json!({
            "kind":"rendered_spans",
            "selectors":[
                {"span_kind":"tool_result_content","channel":"tool_result","edge":"end"},
                {"span_kind":"message_end_marker","channel":"tool_result","edge":"start"}
            ]
        }));
        for role in ["user", "tool"] {
            let rendering = LensInputRendering {
                renderer: if role == "user" {
                    "qwen_open_responses_annotated_v1"
                } else {
                    "muse_glimmer_atem_annotated_v1"
                }
                .into(),
                generation_mode: Some("auto".into()),
                spans: vec![
                    rendered_span(
                        "tool_result_content",
                        Some(3),
                        role,
                        Some("tool_result"),
                        Some((4, 6)),
                    ),
                    rendered_span(
                        "message_end_marker",
                        Some(3),
                        role,
                        Some("tool_result"),
                        Some((6, 7)),
                    ),
                ],
            };
            let bound = bind_plan_positions(&plan, &rendering, 7).unwrap();
            assert_eq!(
                bound.resolved.readouts[0].scope.prefill,
                Some(Selector::Values { values: vec![5, 6] })
            );
            assert_eq!(
                bound.position_bindings[0].matched_span.role.as_deref(),
                Some(role)
            );
        }
    }

    #[test]
    fn rendered_span_selectors_fail_closed_on_ambiguity_missing_ranges_and_raw_input() {
        let rendering = LensInputRendering {
            renderer: "qwen_chatml_messages_v1".into(),
            generation_mode: Some("auto".into()),
            spans: vec![
                rendered_span("message_content", Some(0), "user", None, Some((1, 2))),
                rendered_span("message_content", Some(2), "user", None, Some((3, 4))),
            ],
        };
        let unique = semantic_readout_plan(json!({
            "kind":"rendered_spans",
            "selectors":[{"span_kind":"message_content","role":"user","edge":"end"}]
        }));
        assert!(bind_plan_positions(&unique, &rendering, 4).is_err());

        let last = semantic_readout_plan(json!({
            "kind":"rendered_spans",
            "selectors":[{"span_kind":"message_content","role":"user","occurrence":"last","edge":"end"}]
        }));
        assert_eq!(
            bind_plan_positions(&last, &rendering, 4)
                .unwrap()
                .resolved
                .readouts[0]
                .scope
                .prefill,
            Some(Selector::Values { values: vec![3] })
        );

        let mut missing_range = rendering.clone();
        missing_range.spans[1].token_start = None;
        missing_range.spans[1].token_end = None;
        assert!(bind_plan_positions(&last, &missing_range, 4).is_err());

        let raw = LensInputRendering {
            renderer: "tokenizer_text".into(),
            generation_mode: None,
            spans: Vec::new(),
        };
        assert!(bind_plan_positions(&last, &raw, 4).is_err());

        let duplicate_position = semantic_readout_plan(json!({
            "kind":"rendered_spans",
            "selectors":[
                {"span_kind":"message_content","message_index":2,"edge":"end"},
                {"span_kind":"message_content","role":"user","occurrence":"last","edge":"end"}
            ]
        }));
        assert!(bind_plan_positions(&duplicate_position, &rendering, 4).is_err());
    }

    #[test]
    fn rendered_span_binding_has_global_selector_and_match_work_bounds() {
        let make_selectors = |count: usize| {
            (0..count)
                .map(|message_index| RenderedSpanSelector {
                    span_kind: "message_content".into(),
                    message_index: Some(message_index),
                    tool_call_index: None,
                    role: Some("user".into()),
                    channel: None,
                    label: None,
                    occurrence: RenderedSpanOccurrence::Unique,
                    edge: RenderedSpanEdge::End,
                })
                .collect::<Vec<_>>()
        };
        let mut excessive = minimal_plan();
        excessive.version = 2;
        excessive.readouts[0].scope.prefill = Some(Selector::RenderedSpans {
            selectors: make_selectors(MAX_RENDERED_SELECTORS_PER_PLAN + 1),
        });
        let one_span = LensInputRendering {
            renderer: "qwen_chatml_messages_v1".into(),
            generation_mode: Some("auto".into()),
            spans: vec![rendered_span(
                "message_content",
                Some(0),
                "user",
                None,
                Some((0, 1)),
            )],
        };
        assert!(bind_plan_positions(&excessive, &one_span, 1).is_err());

        let mut expensive = minimal_plan();
        expensive.version = 2;
        expensive.readouts[0].scope.prefill = Some(Selector::RenderedSpans {
            selectors: make_selectors(1000),
        });
        let rendering = LensInputRendering {
            renderer: "qwen_chatml_messages_v1".into(),
            generation_mode: Some("auto".into()),
            spans: (0..1001)
                .map(|index| {
                    rendered_span(
                        "message_content",
                        Some(index),
                        "user",
                        None,
                        Some((index, index + 1)),
                    )
                })
                .collect(),
        };
        assert!(bind_plan_positions(&expensive, &rendering, 1001).is_err());
    }

    #[test]
    fn plan_v2_limits_semantic_selectors_to_prefill_and_keeps_numeric_v1_exact() {
        let semantic_prefill = json!({
            "kind":"rendered_spans",
            "selectors":[{"span_kind":"message_content","edge":"end"}]
        });
        let mut v1 = semantic_readout_plan(semantic_prefill.clone());
        v1.version = 1;
        assert!(validate_plan(&v1).is_err());

        let mut decode = semantic_readout_plan(json!({"kind":"values","values":[0]}));
        decode.readouts[0].scope.prefill = None;
        decode.readouts[0].scope.decode = Some(serde_json::from_value(semantic_prefill).unwrap());
        assert!(validate_plan(&decode).is_err());

        let numeric = minimal_plan();
        let raw = LensInputRendering {
            renderer: "literal_token_ids".into(),
            generation_mode: None,
            spans: Vec::new(),
        };
        let bound = bind_plan_positions(&numeric, &raw, 2).unwrap();
        assert_eq!(bound.authored, numeric);
        assert_eq!(bound.resolved, numeric);
        assert!(bound.position_bindings.is_empty());
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
    fn only_the_final_prefill_event_and_decode_events_need_logits() {
        assert!(phase_needs_logits(Phase::Prefill(0), 1));
        assert!(!phase_needs_logits(Phase::Prefill(0), 3));
        assert!(!phase_needs_logits(Phase::Prefill(1), 3));
        assert!(phase_needs_logits(Phase::Prefill(2), 3));
        assert!(phase_needs_logits(Phase::Decode(0), 3));
        assert!(phase_needs_logits(Phase::Decode(99), 3));
    }

    #[test]
    fn event_forward_routes_preserve_serial_and_production_topology() {
        use EventForwardRoute::*;

        assert_eq!(
            event_forward_route(true, false, false, false),
            ProductionFullTail
        );
        assert_eq!(
            event_forward_route(false, false, false, false),
            ProductionFullTailDiscardLogits
        );
        assert_eq!(
            event_forward_route(true, true, true, false),
            SerialFullTailCapture
        );
        assert_eq!(
            event_forward_route(false, true, true, true),
            SerialNoTailCapture
        );
        assert_eq!(
            event_forward_route(true, true, false, false),
            SerialFullTailNoCapture
        );
        assert_eq!(
            event_forward_route(false, true, false, false),
            SerialNoTailNoCapture
        );
        assert_eq!(
            event_forward_route(true, false, false, true),
            SerialFullTailNoCapture
        );
        assert_eq!(
            event_forward_route(false, false, false, true),
            SerialNoTailNoCapture
        );
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
    fn published_transport_direction_target_covectors_are_explicit_and_scoped() {
        let plan = |target_covector: Option<&str>, lens: serde_json::Value| {
            let mut direction = json!({
                "id": "concept",
                "lens": "j",
                "row": {"kind": "token_id", "token_id": 42},
                "normalization": "unit_l2"
            });
            if let Some(target_covector) = target_covector {
                direction["target_covector"] = json!(target_covector);
            }
            serde_json::from_value::<LensPlan>(json!({
                "version": 1,
                "lenses": [lens],
                "directions": [direction],
                "operations": [{
                    "id": "add",
                    "scope": {"layers":{"kind":"values","values":[31]},"prefill":{"kind":"all"}},
                    "action": {"kind":"residual_l2_fraction","direction":"concept","coefficient":0.1}
                }],
                "readouts": []
            }))
            .unwrap()
        };
        let published = || {
            json!({
                "kind": "published_full_transport",
                "id": "j",
                "artifact": "published",
                "token_ids": [42],
                "allow_unvalidated_transfer": true
            })
        };

        let implicit = plan(None, published());
        validate_plan(&implicit).unwrap();
        let implicit_direction = implicit.directions[0].lens_row().unwrap();
        assert_eq!(implicit_direction.target_covector, None);
        assert_eq!(
            implicit_direction.effective_target_covector(),
            DirectionTargetCovector::DeployedLogitNumerator
        );
        assert!(
            !serde_json::to_value(&implicit).unwrap()["directions"][0]
                .as_object()
                .unwrap()
                .contains_key("target_covector")
        );

        for (serialized, expected) in [
            (
                "deployed_logit_numerator",
                DirectionTargetCovector::DeployedLogitNumerator,
            ),
            ("raw_lm_head", DirectionTargetCovector::RawLmHead),
            (
                "raw_lm_head_orthogonal_to_deployed_logit_numerator",
                DirectionTargetCovector::RawLmHeadOrthogonalToDeployedLogitNumerator,
            ),
        ] {
            let explicit = plan(Some(serialized), published());
            validate_plan(&explicit).unwrap();
            assert_eq!(
                explicit.directions[0]
                    .lens_row()
                    .unwrap()
                    .effective_target_covector(),
                expected
            );
        }

        let unsupported = plan(
            Some("raw_lm_head"),
            json!({"kind":"native_selected","id":"j","artifact":"native"}),
        );
        assert!(validate_plan(&unsupported).is_err());
    }

    #[test]
    fn raw_direction_orthogonalization_removes_the_readout_axis() {
        let orthogonal = orthogonal_component(&[2.0, 3.0, 4.0], &[1.0, 0.0, 0.0], "test").unwrap();
        assert_eq!(orthogonal, [0.0, 3.0, 4.0]);
        let dot = orthogonal
            .iter()
            .zip([1.0_f32, 0.0, 0.0])
            .map(|(&left, right)| left * right)
            .sum::<f32>();
        assert_eq!(dot, 0.0);

        let axis = [0.3_f32, -0.7, 1.1, 0.2];
        let raw = [1.2_f32, 0.4, -0.5, 2.0];
        let orthogonal = normalize_direction(
            orthogonal_component(&raw, &axis, "test").unwrap(),
            Normalization::UnitL2,
            "test",
        )
        .unwrap();
        let axis = normalize_direction(axis.to_vec(), Normalization::UnitL2, "axis").unwrap();
        let normalized_dot = orthogonal
            .iter()
            .zip(axis)
            .map(|(&left, right)| left * right)
            .sum::<f32>();
        assert!(normalized_dot.abs() < 1.0e-6, "dot={normalized_dot}");

        assert!(orthogonal_component(&[1.0, 2.0], &[2.0, 4.0], "test").is_err());
        assert!(orthogonal_component(&[f32::NAN], &[1.0], "test").is_err());
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

        let disabled = native_hyper_plan(
            &path,
            json!([{
                "id": "disabled",
                "scope": {
                    "layers": {"kind": "values", "values": [23]},
                    "prefill": {"kind": "values", "values": [0]}
                },
                "action": {"kind": "fixed_add", "direction": "hyper", "coefficient": -0.0}
            }]),
        );
        validate_plan(&disabled).unwrap();
        let disabled = prepare_qwen4exp_execution_plan(disabled, Path::new("/"), &config).unwrap();
        validate_qwen4exp_event_schedule(&disabled, 1, 1).unwrap();
        assert!(
            qwen4exp_matching_operation(&disabled, Phase::Prefill(0))
                .unwrap()
                .is_none()
        );

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
