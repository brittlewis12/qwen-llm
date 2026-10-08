pub(crate) use crate::lens_intervention::Normalization;
pub(crate) use crate::lens_scope::{
    RenderedSpanEdge, RenderedSpanOccurrence, RenderedSpanSelector, Scope, Selector,
};

use crate::lens_input::{
    LensCohortRequest, LensInputRendering, LensInputSpec, LensMessageMode, LensRenderedSpan,
    PreparedLensInput, prepare_qwen_model_input, prepare_qwen_model_messages_bytes,
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

mod execute;
mod flash_next;
mod glm5_next;
pub(crate) use glm5_next::RAW_PROMPT_RENDERER as GLM5_NEXT_RAW_PROMPT_RENDERER;
mod lenses;
mod output;
mod plan;
mod sweep;
#[cfg(test)]
mod tests;
#[allow(unused_imports)]
pub(crate) use execute::*;
#[allow(unused_imports)]
pub(crate) use flash_next::*;
#[allow(unused_imports)]
pub(crate) use lenses::*;
#[allow(unused_imports)]
pub(crate) use output::*;
#[allow(unused_imports)]
pub(crate) use plan::*;
#[allow(unused_imports)]
pub(crate) use sweep::*;
#[cfg(test)]
pub(crate) use tests::generation_run_bytes;

const MAX_LENSES: usize = 64;
const MAX_READOUTS: usize = 1024;
const MAX_TOP_K: usize = 1024;
pub(crate) const MAX_RUN_ARTIFACT_BYTES: usize = 256 * 1024 * 1024;
const RUN_SCHEMA: &str = "qwen.lens.run";
const RUN_SCHEMA_VERSION: u32 = 5;

#[derive(Debug, Args)]
#[command(group(
    ArgGroup::new("lens_input")
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
pub(crate) struct LensRunArgs {
    /// Ordinary Qwen, Qwen3.8-Flash-Next, or Muse Glimmer GGUF model.
    #[arg(short = 'm', long)]
    pub(crate) model: PathBuf,

    /// Strict Lens plan JSON file.
    #[arg(long)]
    pub(crate) plan: PathBuf,

    /// Private native identity cache for legacy assets; data exact bindings always hash retained bytes.
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

    /// Strict Lens-input JSONL cohort; paths inside records are relative to this file.
    #[arg(long, value_name = "FILE", requires = "output_dir")]
    pub(crate) requests_jsonl: Option<PathBuf>,

    /// Generation transition for --user/--messages; supported values depend on the model.
    #[arg(
        long,
        value_enum,
        conflicts_with_all = ["prompt", "token_ids", "open_responses", "requests_jsonl"]
    )]
    pub(crate) message_mode: Option<LensMessageMode>,

    /// Typed unfinished assistant content: {"channel":"reasoning|final","text":"..."}.
    #[arg(long, value_name = "JSON", value_parser = parse_assistant_prefill,
        conflicts_with_all = ["prompt", "token_ids", "open_responses", "requests_jsonl"])]
    pub(crate) assistant_prefill: Option<crate::model_request::prefill::AssistantPrefill>,

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

    /// Record the N most probable tokens (full-vocabulary log-softmax of the
    /// unfiltered logits, before sampling) at every generated position.
    /// GLM-5.3-Flash only; zero records none.
    #[arg(long, default_value_t = 0)]
    pub(crate) logprobs_top_k: usize,

    /// Also record these token IDs' log-probabilities at every generated
    /// position (comma-separated). GLM-5.3-Flash only.
    #[arg(long, value_delimiter = ',')]
    pub(crate) logprobs_token_ids: Vec<u32>,

    /// Replace this JSON run artifact atomically after successful execution.
    #[arg(long, conflicts_with = "requests_jsonl")]
    pub(crate) output: Option<PathBuf>,

    /// Compact summary or the complete JSON run artifact on stdout.
    #[arg(long, value_enum, conflicts_with = "requests_jsonl")]
    pub(crate) format: Option<RunStdoutFormat>,

    /// Fresh immutable directory containing one ordinary run artifact per request.
    #[arg(long, requires = "requests_jsonl")]
    pub(crate) output_dir: Option<PathBuf>,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum RunStdoutFormat {
    Summary,
    Json,
}

fn parse_assistant_prefill(
    value: &str,
) -> std::result::Result<crate::model_request::prefill::AssistantPrefill, String> {
    let prefill: crate::model_request::prefill::AssistantPrefill =
        serde_json::from_str(value).map_err(|error| error.to_string())?;
    prefill.validate_qwen().map_err(|error| error.to_string())?;
    Ok(prefill)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ReadoutDefinition {
    pub(crate) id: String,
    pub(crate) lens: String,
    pub(crate) scope: Scope,
    pub(crate) top_k: usize,
}

/// Scalar residual readout against one plan direction; no lens is involved.
/// `site` (default post-block, omitted when serialized) and `point` (required
/// for a module site) are GLM-5.3-Flash only: a module site reads one
/// writer's output before or after that site's operations.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct DirectionReadoutDefinition {
    pub(crate) id: String,
    pub(crate) direction: String,
    pub(crate) scope: Scope,
    #[serde(default, skip_serializing_if = "OperationSite::is_post_block")]
    pub(crate) site: OperationSite,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) point: Option<ReadoutPoint>,
}

/// When a module-site direction readout reads its site.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ReadoutPoint {
    BeforeOperations,
    AfterOperations,
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
    /// GLM-5.3-Flash interventions and captures run on the serial decode
    /// path (the Exact lineage) for every prompt and generated token.
    Glm5NextSerialInterventions,
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
            Self::Glm5NextSerialInterventions => "glm5_next_serial_interventions",
        }
    }
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

/// `dot = h.v` with the post-block residual `h` and the prepared direction row
/// `v` (after its declared normalization), all accumulated in f64.
#[derive(Debug, Serialize)]
pub(crate) struct LiveDirectionReadout {
    pub(crate) id: String,
    pub(crate) direction: String,
    #[serde(skip_serializing_if = "OperationSite::is_post_block")]
    pub(crate) site: OperationSite,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) point: Option<ReadoutPoint>,
    pub(crate) source_layer: u32,
    pub(crate) phase: &'static str,
    pub(crate) index: usize,
    pub(crate) dot: f64,
    pub(crate) h_norm_l2: f64,
    pub(crate) v_norm_l2: f64,
}

pub(crate) fn direction_readout_scalars(
    residual: &[f32],
    direction: &[f32],
) -> Result<(f64, f64, f64)> {
    ensure!(
        !residual.is_empty() && residual.len() == direction.len(),
        "direction readout residual and direction widths differ"
    );
    let (mut dot, mut h_squared, mut v_squared) = (0.0f64, 0.0f64, 0.0f64);
    for (&h, &v) in residual.iter().zip(direction) {
        let (h, v) = (f64::from(h), f64::from(v));
        dot += h * v;
        h_squared += h * h;
        v_squared += v * v;
    }
    // JSON would publish a non-finite score as null; refuse it instead.
    ensure!(
        dot.is_finite() && h_squared.is_finite() && v_squared.is_finite(),
        "direction readout produced a non-finite value"
    );
    Ok((dot, h_squared.sqrt(), v_squared.sqrt()))
}

#[derive(Debug, Serialize)]
pub(crate) struct LiveScore {
    pub(crate) token_id: Option<i32>,
    pub(crate) row_id: usize,
    pub(crate) word_id: Option<i64>,
    pub(crate) label: Option<String>,
    pub(crate) score: f32,
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

/// Input features each family admits, checked before family dispatch:
/// typed assistant prefill is ordinary Qwen and GLM-5.3-Flash only; Open
/// Responses inputs are ordinary Qwen only; generation log-probabilities are
/// GLM-5.3-Flash only.
pub(crate) fn validate_family_inputs(family: ModelFamily, args: &LensRunArgs) -> Result<()> {
    let ordinary = matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe);
    let glm = family == ModelFamily::Glm5Next;
    ensure!(
        args.assistant_prefill.is_none() || ordinary || glm,
        "typed assistant prefill supports ordinary Qwen and GLM-5.3-Flash only"
    );
    ensure!(
        args.open_responses.is_none() || ordinary,
        "--open-responses supports ordinary Qwen only; {} is not supported",
        family.architecture_name()
    );
    ensure!(
        glm || (args.logprobs_top_k == 0 && args.logprobs_token_ids.is_empty()),
        "--logprobs-top-k and --logprobs-token-ids are supported only for GLM-5.3-Flash"
    );
    Ok(())
}

/// Unfiltered log-probabilities at one generated position.
#[derive(Debug, Serialize)]
pub(crate) struct GenerationLogprobs {
    /// Zero-based generated position (the sampled token's index).
    pub(crate) index: usize,
    pub(crate) sampled_token_id: i32,
    /// `[token_id, logprob]`, most probable first.
    pub(crate) top: Vec<(u32, f64)>,
    /// `[token_id, logprob]` for each requested ID, in request order.
    pub(crate) tracked: Vec<(u32, f64)>,
}

/// Full-vocabulary log-softmax (f64) of `logits`: the `top_k` most probable
/// IDs (ties by lower ID) and the `tracked` IDs.
pub(crate) fn generation_logprobs(
    logits: &[f32],
    index: usize,
    sampled: i32,
    top_k: usize,
    tracked: &[u32],
) -> Result<GenerationLogprobs> {
    ensure!(
        logits.iter().all(|v| v.is_finite()),
        "generation logits are not finite"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let log_sum = logits
        .iter()
        .map(|v| (f64::from(*v) - max).exp())
        .sum::<f64>()
        .ln();
    let logprob = |id: u32| f64::from(logits[id as usize]) - max - log_sum;
    let mut order: Vec<u32> = (0..logits.len() as u32).collect();
    let k = top_k.min(order.len());
    if k > 0 {
        order.select_nth_unstable_by(k - 1, |a, b| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        });
        order.truncate(k);
        order.sort_by(|a, b| {
            logits[*b as usize]
                .total_cmp(&logits[*a as usize])
                .then(a.cmp(b))
        });
    } else {
        order.clear();
    }
    for &id in tracked {
        ensure!(
            (id as usize) < logits.len(),
            "tracked token {id} is outside the vocabulary"
        );
    }
    Ok(GenerationLogprobs {
        index,
        sampled_token_id: sampled,
        top: order.into_iter().map(|id| (id, logprob(id))).collect(),
        tracked: tracked.iter().map(|&id| (id, logprob(id))).collect(),
    })
}

pub(crate) fn run(args: LensRunArgs) -> Result<()> {
    validate_run_args(&args)?;
    if args.requests_jsonl.is_some() {
        return run_cohort(args);
    }
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
    let family = ModelFamily::detect(&gguf).with_context(|| {
        format!(
            "qwen-lens run does not support model architecture {:?}",
            gguf.architecture()
        )
    })?;
    validate_family_inputs(family, &args)?;
    match family {
        ModelFamily::MuseGlimmer => {
            return crate::muse_lens_run::run(
                &args,
                plan,
                &plan_path,
                plan_dir,
                gguf,
                output_path.as_deref(),
            );
        }
        ModelFamily::Qwen4Exp => {
            return run_qwen4exp(
                &args,
                plan,
                &plan_path,
                plan_dir,
                gguf,
                output_path.as_deref(),
            );
        }
        ModelFamily::DeepSeek4 => bail!(
            "qwen-lens run supports ordinary Qwen, Muse Glimmer, or Flash-Next; DeepSeek V4 has no lens runtime"
        ),
        ModelFamily::K2Horizon => bail!(
            "qwen-lens run supports ordinary Qwen, Muse Glimmer, or Flash-Next; K2 Horizon is not supported by `qwen-lens run`; use its native `qwen-lens read-full --logit-lens` lane"
        ),
        ModelFamily::Glm5Next => {
            return glm5_next::run_glm5_next(
                &args,
                plan,
                &plan_path,
                plan_dir,
                gguf,
                output_path.as_deref(),
            );
        }
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => {}
    }
    validate_ordinary_plan(&plan)?;
    let full_transports = open_full_transports(&plan, plan_dir)?;

    let tokenizer = Tokenizer::from_gguf(&gguf).context("load model tokenizer")?;
    let (prepared_input, generation_input) = if let Some(prefill) = &args.assistant_prefill {
        let (input, record) = crate::lens_input::prepare_qwen_model_generation_input(
            args.input_spec(),
            prefill,
            family,
            &gguf,
            &tokenizer,
        )?;
        (input, Some(record))
    } else {
        (
            prepare_qwen_model_input(args.input_spec(), family, &gguf, &tokenizer)?,
            None,
        )
    };
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

    crate::shutdown::checkpoint()?;
    let mut full_transports = bind_full_transports(
        full_transports,
        &plan,
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
    let execution = prepare_execution_plan(
        &bound_plan.resolved,
        plan_dir,
        &loaded,
        &mut full_transports,
    )?;
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
        generation_input,
        result,
        prefill.execution,
        None,
        output_path.as_deref(),
    )
}

fn validate_run_args(args: &LensRunArgs) -> Result<()> {
    if let Some(prefill) = &args.assistant_prefill {
        ensure!(
            (args.user.is_some() || args.messages.is_some()) && args.requests_jsonl.is_none(),
            "--assistant-prefill requires --user or --messages, not cohort or raw input"
        );
        prefill.validate_qwen()?;
    }
    ensure!(args.max_new_tokens > 0, "--max-new-tokens must be positive");
    if args.requests_jsonl.is_some() {
        ensure!(
            args.prompt.is_none()
                && args.token_ids.is_none()
                && args.user.is_none()
                && args.system.is_none()
                && args.messages.is_none()
                && args.open_responses.is_none()
                && args.message_mode.is_none()
                && !args.no_special_tokens
                && args.output.is_none()
                && args.format.is_none()
                && args.output_dir.is_some(),
            "--requests-jsonl requires --output-dir and conflicts with single-request input and output flags"
        );
        return Ok(());
    }
    ensure!(
        args.output_dir.is_none(),
        "--output-dir requires --requests-jsonl"
    );
    validate_lens_input_spec(args.input_spec())?;
    Ok(())
}

fn create_bundle_directory(path: &Path) -> Result<()> {
    DirBuilder::new()
        .mode(0o700)
        .create(path)
        .with_context(|| format!("create bundle directory {}", path.display()))
}

fn write_new_bundle_file(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .with_context(|| format!("create bundle file {}", path.display()))?;
    file.write_all(bytes)
        .with_context(|| format!("write bundle file {}", path.display()))?;
    file.sync_all()
        .with_context(|| format!("sync bundle file {}", path.display()))
}

fn stage_and_publish_bundle<T>(output: &Path, build: impl FnOnce(&Path) -> Result<T>) -> Result<T> {
    let parent = output.parent().context("bundle output has no parent")?;
    let leaf = output.file_name().context("bundle output has no leaf")?;
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
    create_bundle_directory(&staging)?;
    if let Err(error) = super::sync_directory(parent) {
        if let Err(cleanup_error) = std::fs::remove_dir(&staging) {
            return Err(error.context(format!(
                "also failed to remove empty bundle staging directory {}: {cleanup_error}",
                staging.display()
            )));
        }
        return Err(error);
    }
    let result = build(&staging).and_then(|value| {
        publish_bundle_directory_exclusive(&staging, output)?;
        Ok(value)
    });
    match result {
        Ok(value) => Ok(value),
        Err(error) => {
            match std::fs::symlink_metadata(&staging) {
                Ok(_) => {
                    if let Err(cleanup_error) = std::fs::remove_dir_all(&staging) {
                        return Err(error.context(format!(
                            "also failed to remove bundle staging directory {}: {cleanup_error}",
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
                        "also failed to inspect bundle staging directory {}: {inspect_error}",
                        staging.display()
                    )));
                }
            }
            Err(error)
        }
    }
}

fn publish_bundle_directory_exclusive(staging: &Path, output: &Path) -> Result<()> {
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
                "publish bundle directory {} to {}",
                staging.display(),
                output.display()
            )
        });
    }
    let parent = output.parent().context("bundle output has no parent")?;
    if let Err(error) = super::sync_directory(parent) {
        eprintln!(
            "warning: bundle {} is published, but its parent directory could not be synced: {error:#}",
            output.display()
        );
    }
    Ok(())
}

fn is_lower_hex(value: &str) -> bool {
    value
        .bytes()
        .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn unique_ids<'a>(ids: impl Iterator<Item = &'a str>, kind: &str) -> Result<()> {
    let mut seen = HashSet::new();
    for id in ids {
        ensure!(!id.is_empty(), "{kind} id must not be empty");
        ensure!(seen.insert(id), "duplicate {kind} id {id:?}");
    }
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

fn lens_has_layer(prepared: &PreparedLens, layer: u32) -> bool {
    match &prepared.lens {
        LoadedLens::Native(native) => native.source_layers.contains(&layer),
        LoadedLens::Template { lens, .. } => lens.layers().contains(&layer),
    }
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
