//! Family-general pp/tg rows for the families that do not bind through the
//! Qwen `Runtime`: Flash-Next (`qwen4exp`), DeepSeek V4/V4.1, Muse Glimmer and K2
//! Horizon. Each adapter loads weights once with the production options
//! `qwen run` uses (`family_options`) and times the production calls,
//! including the host logits copy production makes at a prompt endpoint and
//! after every decoded token (llama-bench likewise copies logits). Rows record
//! what the timed call executed (`prefill_mode`, `decode_mode`).

use super::*;
use crate::family_options::{
    deepseek_v4_prefill_chunk_ranges, deepseek_v4_prefill_chunk_tokens,
    muse_runtime_options_from_env, qwen4exp_decode_options_from_env,
};
use anyhow::{bail, ensure};
use qwen_llm::deepseek_v4_metal::{DeepSeekV4MetalResidency, DeepSeekV4Session};
use qwen_llm::deepseek_v41::{
    forward::{Forward as V41Forward, PrefillExecution, PrefillReport},
    gguf as v41_weights,
};
use qwen_llm::glm5_next_metal::{Glm5NextSession, Glm5NextWeights, prefill_plan};
use qwen_llm::k2_horizon_runtime::{K2LoadedModel, K2Session};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::muse_glimmer_runtime::{MuseGlimmerLoadedModel, MuseGlimmerTextRunner};
use qwen_llm::muse_glimmer_text_session::MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity, Qwen4ExpTextRunner,
};

/// What one family's timed calls executed, recorded on every row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FamilySemantics {
    pub(crate) prefill_mode: &'static str,
    pub(crate) decode_mode: &'static str,
    pub(crate) prefill_chunk: Option<usize>,
}

pub(crate) trait FamilyBench {
    fn vocab_size(&self) -> usize;
    /// Fresh causal state at position zero; untimed.
    fn begin_rep(&mut self) -> Result<()>;
    fn begin_pp_rep(&mut self, _depth: usize) -> Result<()> {
        self.begin_rep()
    }
    /// Prefill `ids` after the current position through the production prompt
    /// path, including its endpoint logits copy.
    fn prefill(&mut self, ids: &[u32]) -> Result<()>;
    /// One step through the production generated-token path, including its
    /// logits copy.
    fn decode(&mut self, id: u32) -> Result<()>;
    /// Labels for the most recent prefill and decode calls.
    fn semantics(&self) -> FamilySemantics;
    fn prefill_report(&self) -> Option<qwen_llm::deepseek_v41::forward::PrefillReport> {
        None
    }
    fn v41_prefill_diagnostics(&self) -> Option<V41BenchPrefillRep> {
        None
    }
    /// Explicit per-case policy; only the paired V4.1 harness uses this.
    fn set_pair_execution(&mut self, _execution: PrefillExecution) -> Result<()> {
        bail!("paired execution requires the V4.1 adapter")
    }
    fn last_logits(&self) -> Option<&[f32]> {
        None
    }
    fn end_pair_case(&mut self) {}
}

/// Extent the loaded model and its session must hold for these shapes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FamilyBenchExtent {
    /// Deepest absolute position any row or the warm-up reaches.
    pub(crate) forwards: usize,
    /// Deepest absolute prefill endpoint (depth + pp, or the depth fill).
    /// Flash-Next sizes its packed and selected-range workspaces from this,
    /// as `qwen run` does from the whole prompt.
    pub(crate) prefill_endpoint: usize,
}

impl FamilyBenchExtent {
    pub(crate) fn for_shapes(
        pp: &[usize],
        tg: &[usize],
        depths: &[usize],
        warmup: (usize, usize),
    ) -> Result<Self> {
        let depth = depths.iter().copied().max().unwrap_or(0);
        let pp_max = pp.iter().copied().max().unwrap_or(0);
        let longest = pp_max.max(tg.iter().copied().max().unwrap_or(0));
        let timed = depth
            .checked_add(longest)
            .context("bench extent overflows usize")?;
        let pp_endpoint = if pp_max > 0 { depth + pp_max } else { 0 };
        Ok(Self {
            forwards: timed.max(warmup.0).max(warmup.1),
            prefill_endpoint: pp_endpoint.max(depth).max(warmup.0).max(2),
        })
    }
}

/// The family of a GGUF that does not bind through the Qwen runtime; `None`
/// for Qwen (and for unrecognized architectures, which the Qwen path refuses
/// with its own error).
pub(crate) fn non_qwen_family(model: &std::path::Path) -> Result<Option<ModelFamily>> {
    let gguf = GgufFile::open(model).with_context(|| format!("open {}", model.display()))?;
    Ok(ModelFamily::detect(&gguf)
        .filter(|family| !matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)))
}

/// Profiling-only Flash-Next modes `qwen run` executes differently; refused
/// here rather than silently ignored.
const QWEN4EXP_PROFILE_ONLY_ENV: [&str; 3] = [
    "QWEN4EXP_LAYER_PROFILE",
    "QWEN4EXP_PACKED_PREFILL_PROFILE",
    "QWEN4EXP_FULL_SHARD_PREFETCH",
];

/// Load `family` once with production options and run `body` against it.
pub(crate) fn with_family_bench<R>(
    family: ModelFamily,
    ctx: &MetalContext,
    gguf: &std::sync::Arc<GgufFile>,
    extent: FamilyBenchExtent,
    v41_prefill_execution: PrefillExecution,
    body: &mut dyn FnMut(&mut dyn FamilyBench) -> Result<R>,
) -> Result<R> {
    with_family_bench_planned(family, ctx, gguf, extent, v41_prefill_execution, None, body)
}

pub(crate) fn with_family_bench_planned<R>(
    family: ModelFamily,
    ctx: &MetalContext,
    gguf: &std::sync::Arc<GgufFile>,
    extent: FamilyBenchExtent,
    v41_prefill_execution: PrefillExecution,
    cold: Option<&qwen_llm::deepseek_v41::forward::ColdPrefillPlan>,
    body: &mut dyn FnMut(&mut dyn FamilyBench) -> Result<R>,
) -> Result<R> {
    match family {
        ModelFamily::DeepSeekV41 => {
            let plan = v41_weights::plan(gguf)?;
            validate_v41_extent(extent, plan.engram_hash.max_position)?;
            plan.validate_execution()?;
            let (planned, weights) = crate::family_options::v41_load_cold(
                v41_prefill_execution,
                cold,
                |target| {
                    Ok(
                        qwen_llm::deepseek_v41::forward::plan_cold_prefill_with_target(
                            ctx,
                            gguf,
                            extent.prefill_endpoint.min(extent.forwards),
                            extent.forwards,
                            &[v41_prefill_execution],
                            0,
                            target,
                        )?,
                    )
                },
                || ctx.memory_signals(),
                |storage| {
                    v41_weights::ResidentWeights::load_with_storage_plan(ctx, gguf.clone(), storage)
                        .context("load DeepSeek V4.1 weights")
                },
            )?;
            eprintln!(
                "[v41-retained-weight-resources] {}",
                serde_json::to_string(&weights.retained_resources)?
            );
            let forward = V41Forward::from_resident(
                ctx,
                &weights,
                weights.token_map.clone(),
                weights.raw_pad,
            )
            .map_err(anyhow::Error::msg)?;
            let request_plan = forward.plan_request_from_cold(&planned)?;
            let mut bench = DeepSeekV41Bench {
                vocab: weights.config.vocab,
                full_chunk_rows: request_plan.prefill().selected_rows,
                state: None,
                requested_execution: v41_prefill_execution,
                position: 0,
                ced_allowed: true,
                last_execution: PrefillExecution::FullForward,
                reports: Vec::new(),
                paired: false,
                last_logits: None,
                last_prefill: None,
                create: || Ok(forward.new_state_from_plan(&request_plan)?),
                prefill: |state: &mut qwen_llm::deepseek_v41::forward::State,
                          ids: &[u32],
                          execution| {
                    shutdown::checkpoint()?;
                    // Measurement hook (not a product flag): per-layer sampled
                    // stage envelopes for one prefill, printed to stderr.
                    let scope = std::env::var_os("QWEN_V41_PREFILL_STAGES")
                        .map(|_| qwen_llm::deepseek_v41::forward::profile::Scope::begin_stages())
                        .transpose()
                        .map_err(anyhow::Error::msg)?;
                    let result = forward
                        .prefill_with_execution(
                            state,
                            ids,
                            Some(shutdown::cancellation_flag()),
                            execution,
                        )
                        .map_err(anyhow::Error::from)?;
                    if let Some(scope) = scope {
                        let report = scope.finish();
                        eprintln!(
                            "[v41-prefill-stages] rows={} wall_ms={:.3}",
                            ids.len(),
                            report.token_wall_ms
                        );
                        for op in &report.operations {
                            eprintln!(
                                "[v41-prefill-stages] {} dims={:?} count={} gpu_ms={:?} encode_ms={:.3} wait_ms={:.3}",
                                op.operation,
                                op.dimensions,
                                op.count,
                                op.gpu_ms,
                                op.encode_ms,
                                op.commit_wait_ms
                            );
                        }
                    }
                    Ok(result)
                },
                step: |state: &mut qwen_llm::deepseek_v41::forward::State, token| {
                    shutdown::checkpoint()?;
                    // Measurement hooks (not product flags): per-layer sampled
                    // decode stage envelopes, and the step's dispatch census,
                    // printed to stderr for every decoded token. Stage sampling
                    // splits the step into ~160 encoders; on near-capacity
                    // residents (IQ3 at ~217 GiB) that costs seconds per token.
                    // QWEN_V41_DECODE_DISPATCHES instead times every dispatch in
                    // its own encoder and prints one JSON line per token, and
                    // QWEN_V41_DECODE_TIMING prints the unsplit command's times.
                    use qwen_llm::deepseek_v41::forward::profile::Scope;
                    let env = |name| std::env::var_os(name).is_some();
                    let dispatches = env("QWEN_V41_DECODE_DISPATCHES");
                    let scope = if dispatches {
                        Some(Scope::begin_dispatches())
                    } else if env("QWEN_V41_DECODE_STAGES") {
                        Some(Scope::begin_stages())
                    } else if env("QWEN_V41_DECODE_TIMING") {
                        Some(Scope::begin())
                    } else {
                        None
                    }
                    .transpose()
                    .map_err(anyhow::Error::msg)?;
                    let census = std::env::var_os("QWEN_V41_CENSUS").is_some();
                    if census {
                        qwen_llm::metal::dispatch_census_begin();
                    }
                    let logits = forward
                        .step(state, token, Some(shutdown::cancellation_flag()))
                        .map_err(anyhow::Error::msg)?;
                    if census {
                        let rows = qwen_llm::metal::dispatch_census_take();
                        let mut counts = std::collections::BTreeMap::new();
                        for row in &rows {
                            *counts.entry(row.kernel.clone()).or_insert(0usize) += 1;
                        }
                        let mut top: Vec<_> = counts.into_iter().collect();
                        top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                        top.truncate(40);
                        eprintln!(
                            "[v41-decode-census] dispatches={} top={:?}",
                            rows.len(),
                            top
                        );
                    }
                    let report = scope.map(Scope::finish);
                    if let Some(report) = report.as_ref().filter(|_| dispatches) {
                        let origin = report
                            .dispatches
                            .iter()
                            .filter_map(|r| r.gpu_start)
                            .min()
                            .unwrap_or(0);
                        let rel = |t: Option<u64>| t.map(|t| t.saturating_sub(origin));
                        let rows: Vec<_> = report
                            .dispatches
                            .iter()
                            .map(|r| {
                                serde_json::json!([
                                    r.label,
                                    r.layer,
                                    r.kernel,
                                    r.grid,
                                    r.threads,
                                    r.concurrent,
                                    r.gpu_ns,
                                    rel(r.gpu_start),
                                    rel(r.gpu_end)
                                ])
                            })
                            .collect();
                        let step_ms = report
                            .operations
                            .iter()
                            .find(|op| op.operation == "gpu_step")
                            .and_then(|op| op.gpu_ms);
                        eprintln!(
                            "[v41-decode-dispatches] {}",
                            serde_json::json!({
                                "wall_ms": report.token_wall_ms,
                                "split_command_gpu_ms": step_ms,
                                "overflow": report.dispatch_overflow,
                                "rows": rows,
                            })
                        );
                    } else if let Some(report) = report {
                        eprintln!(
                            "[v41-decode-stages] wall_ms={:.3} pageins={:?} disk_read_bytes={:?}",
                            report.token_wall_ms,
                            report.process_pageins,
                            report.process_disk_read_bytes
                        );
                        for op in &report.operations {
                            eprintln!(
                                "[v41-decode-stages] {} dims={:?} count={} gpu_ms={:?} encode_ms={:.3} wait_ms={:.3} wall_ms={:.3}",
                                op.operation,
                                op.dimensions,
                                op.count,
                                op.gpu_ms,
                                op.encode_ms,
                                op.commit_wait_ms,
                                op.wall_ms
                            );
                        }
                    }
                    Ok(logits)
                },
            };
            body(&mut bench)
        }
        ModelFamily::Qwen4Exp => {
            for name in QWEN4EXP_PROFILE_ONLY_ENV {
                ensure!(
                    std::env::var_os(name).is_none(),
                    "{name} is a qwen run profiling mode; unset it for qwen-bench"
                );
            }
            let config = Qwen4ExpConfig::from_gguf(gguf).context("bind Flash-Next geometry")?;
            ensure!(
                config == Qwen4ExpConfig::flash_next_reference(),
                "Qwen3.8-Flash-Next runtime requires the released architecture contract"
            );
            let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, extent.forwards)
                .context("derive Flash-Next session capacity")?;
            let options = qwen4exp_decode_options_from_env()?;
            // Same admission as `qwen run`: packed prefill sized to the
            // deepest prefill endpoint, or scalar when refused (logged).
            let mut loaded = match Qwen4ExpLoadedModel::load_with_decode_options(
                ctx,
                gguf,
                capacity,
                Some(extent.prefill_endpoint),
                options,
            ) {
                Ok(loaded) => loaded,
                Err(packed_error) => {
                    eprintln!(
                        "[family-bench] qwen4exp packed prefill unavailable ({packed_error}); retrying scalar admission"
                    );
                    Qwen4ExpLoadedModel::load_with_decode_options(
                        ctx, gguf, capacity, None, options,
                    )
                    .context("load Flash-Next (scalar fallback)")?
                }
            };
            let runner = loaded
                .create_runner(ctx)
                .context("bind Flash-Next execution graph")?;
            let mut bench = Qwen4ExpBench {
                runner,
                vocab: config.vocab_size as usize,
                last_prefill_mode: "none",
            };
            body(&mut bench)
        }
        ModelFamily::DeepSeek4 => {
            eprintln!(
                "[family-bench] deepseek_v4: load-time page-cache prefetch (QWEN_DSV4_PREFETCH) is not applied; rows time steady state after the model warm-up"
            );
            let chunk = deepseek_v4_prefill_chunk_tokens()?;
            let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(ctx, gguf, extent.forwards)
                .context("plan DeepSeek V4 residency and session")?;
            let admitted = plan
                .admit(ctx.memory_signals())
                .context("admit DeepSeek V4 residency and session")?;
            let (residency, _admission, _after) =
                DeepSeekV4MetalResidency::load_from_plan(ctx, gguf, admitted)
                    .context("load DeepSeek V4 residency")?
                    .into_parts();
            let vocab = residency.config().vocab_size as usize;
            let mut bench = DeepSeekV4Bench {
                ctx,
                residency: Some(residency),
                session: None,
                chunk,
                vocab,
                last_prefill_mode: "none",
            };
            body(&mut bench)
        }
        ModelFamily::MuseGlimmer => {
            let options = muse_runtime_options_from_env()?;
            let mut loaded =
                MuseGlimmerLoadedModel::load_with_options(ctx, gguf, extent.forwards, options)
                    .context("load Muse Glimmer with production math")?;
            let vocab = loaded.config().vocab_size as usize;
            let math = loaded.math_options();
            let runner = loaded
                .create_runner(ctx)
                .context("bind Muse Glimmer execution graph")?;
            let mut bench = MuseBench {
                runner,
                vocab,
                matrix_prefill: math.matrix_prefill,
                split_decode: math.split_decode,
                last_prefill_tokens: 0,
            };
            body(&mut bench)
        }
        ModelFamily::K2Horizon => {
            let capacity = u32::try_from(extent.forwards).context("K2 bench extent exceeds u32")?;
            let model =
                K2LoadedModel::load(ctx, gguf, capacity).context("load K2 Horizon weights")?;
            let mut bench = K2Bench {
                vocab: model.config().vocab_size as usize,
                model: &model,
                session: None,
                last_prefill: None,
            };
            body(&mut bench)
        }
        ModelFamily::Glm5Next => {
            // The same layout admission as `qwen run` (synthetic tokens need
            // no tokenizer), then the session preflight before any weight is
            // mapped.
            let layout = qwen_llm::glm5_next::Glm5NextArtifactLayout::inspect(gguf)
                .context("admit GLM-5.3-Flash artifact")?;
            let context = layout.config().context_length as usize;
            ensure!(
                extent.forwards <= context,
                "GLM-5.3-Flash rows must stay within the checkpoint context {context} (need {})",
                extent.forwards
            );
            let maximum = prefill_plan::candidates(
                extent.prefill_endpoint.min(extent.forwards),
                layout.packed_prefill(),
            )[0];
            let (prefill_rows, _) = prefill_plan::select(maximum, |rows| {
                qwen_llm::glm5_next_metal::preflight_session(
                    ctx,
                    gguf,
                    layout.model(),
                    extent.forwards,
                    rows,
                )
            })
            .context("admit the GLM-5.3-Flash bench session")?;
            let weights = Glm5NextWeights::load(ctx, gguf).context("load GLM-5.3-Flash weights")?;
            let mut bench = Glm5NextBench {
                ctx,
                vocab: weights.config.vocab_size as usize,
                weights: &weights,
                capacity: extent.forwards,
                prefill_rows,
                prefill_frozen: false,
                session: None,
            };
            body(&mut bench)
        }
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => {
            bail!("Qwen models bind through the Qwen runtime path, not the family adapter")
        }
    }
}

fn validate_v41_extent(extent: FamilyBenchExtent, max_position: u64) -> Result<()> {
    ensure!(
        extent.forwards > 0 && u64::try_from(extent.forwards)? <= max_position,
        "DeepSeek V4.1 rows must stay within the checkpoint context {max_position} (need {})",
        extent.forwards
    );
    Ok(())
}

struct DeepSeekV41Bench<S, Create, Step, Prefill> {
    vocab: usize,
    full_chunk_rows: usize,
    state: Option<S>,
    create: Create,
    step: Step,
    prefill: Prefill,
    requested_execution: PrefillExecution,
    position: usize,
    ced_allowed: bool,
    last_execution: PrefillExecution,
    reports: Vec<V41BenchPrefillCall>,
    paired: bool,
    last_logits: Option<Vec<f32>>,
    last_prefill: Option<PrefillReport>,
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct V41BenchPrefillRep {
    requested_execution: PrefillExecution,
    calls: Vec<V41BenchPrefillCall>,
}

impl V41BenchPrefillRep {
    pub(crate) fn validate_pair(&self, execution: PrefillExecution, tokens: usize) -> Result<()> {
        ensure!(
            self.requested_execution == execution,
            "paired requested policy changed"
        );
        ensure!(
            self.calls.len() == 1,
            "paired case must execute one whole cold prefill"
        );
        let call = &self.calls[0];
        ensure!(
            call.start_position == 0 && call.tokens == tokens,
            "paired prompt extent changed"
        );
        ensure!(
            call.diagnostics.report.execution == execution,
            "paired actual policy differs from requested policy"
        );
        ensure!(
            call.diagnostics.report.fallback.is_none(),
            "paired prefill fallback: {:?}",
            call.diagnostics.report.fallback
        );
        if execution == PrefillExecution::ShrinkingCed {
            let report = &call.diagnostics.report;
            let counters = report
                .counters
                .as_ref()
                .context("paired CED omitted work counters")?;
            ensure!(
                counters.encoded_commands > 0
                    && counters.encoded_commands == counters.completed_commands,
                "paired CED has incomplete command coverage"
            );
            let memory = report
                .memory
                .as_ref()
                .context("paired CED omitted its memory ledger")?;
            let observed = report
                .observed_incremental_metal_bytes
                .context("paired CED omitted allocation observations")?;
            ensure!(
                memory.incremental_metal_bytes > 0 && observed <= memory.incremental_metal_bytes,
                "paired CED allocation observation exceeds its accepted bound"
            );
        }
        Ok(())
    }
}

#[derive(Clone, Debug, serde::Serialize)]
struct V41BenchPrefillCall {
    start_position: usize,
    tokens: usize,
    #[serde(flatten)]
    diagnostics: crate::family_options::V41PrefillDiagnostics,
}

impl<S, Create, Step, Prefill> FamilyBench for DeepSeekV41Bench<S, Create, Step, Prefill>
where
    Create: FnMut() -> Result<S>,
    Step: FnMut(&mut S, u32) -> Result<Vec<f32>>,
    Prefill: FnMut(&mut S, &[u32], PrefillExecution) -> Result<(Vec<f32>, PrefillReport)>,
{
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn begin_rep(&mut self) -> Result<()> {
        self.state = None;
        self.last_prefill = None;
        self.state = Some((self.create)()?);
        self.position = 0;
        self.ced_allowed = true;
        self.last_execution = PrefillExecution::FullForward;
        self.reports.clear();
        self.last_logits = None;
        Ok(())
    }
    fn begin_pp_rep(&mut self, depth: usize) -> Result<()> {
        self.begin_rep()?;
        self.ced_allowed = depth == 0;
        Ok(())
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .context("DeepSeek V4.1 bench used before begin_rep")?;
        let execution = if self.ced_allowed && self.position == 0 {
            self.requested_execution
        } else {
            PrefillExecution::FullForward
        };
        let (logits, mut report) = (self.prefill)(state, ids, execution)?;
        self.last_prefill = Some(report.clone());
        if self.paired {
            self.last_logits = Some(logits);
        } else {
            std::hint::black_box(logits);
        }
        if execution != self.requested_execution {
            report.fallback = Some(
                "CED is cold whole-prompt only; nonzero-depth pp and suffix calls use full forward"
                    .into(),
            );
        }
        self.last_execution = report.execution;
        let diagnostics = if self.paired {
            Some(crate::family_options::V41PrefillDiagnostics {
                requested_execution: self.requested_execution,
                report,
                capacity: None,
            })
        } else {
            crate::family_options::V41PrefillDiagnostics::new(self.requested_execution, report)
        };
        if let Some(diagnostics) = diagnostics {
            self.reports.push(V41BenchPrefillCall {
                start_position: self.position,
                tokens: ids.len(),
                diagnostics,
            });
        }
        self.position += ids.len();
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        let state = self
            .state
            .as_mut()
            .context("DeepSeek V4.1 bench used before begin_rep")?;
        std::hint::black_box((self.step)(state, id)?);
        self.position += 1;
        Ok(())
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: crate::family_options::v41_prefill_mode(self.last_execution),
            decode_mode: "gpu_step+logits_copy",
            prefill_chunk: (self.last_execution == PrefillExecution::FullForward).then(|| {
                self.last_prefill
                    .as_ref()
                    .map_or(self.full_chunk_rows, |r| r.geometry.selected_rows)
            }),
        }
    }
    fn prefill_report(&self) -> Option<PrefillReport> {
        self.last_prefill.clone()
    }
    fn v41_prefill_diagnostics(&self) -> Option<V41BenchPrefillRep> {
        (self.requested_execution == PrefillExecution::ShrinkingCed || !self.reports.is_empty())
            .then(|| V41BenchPrefillRep {
                requested_execution: self.requested_execution,
                calls: self.reports.clone(),
            })
    }
    fn set_pair_execution(&mut self, execution: PrefillExecution) -> Result<()> {
        ensure!(
            self.state.is_none(),
            "paired policy must be selected without a live state"
        );
        self.paired = true;
        self.requested_execution = execution;
        Ok(())
    }
    fn last_logits(&self) -> Option<&[f32]> {
        self.last_logits.as_deref()
    }
    fn end_pair_case(&mut self) {
        self.state = None;
        self.last_logits = None;
    }
}

struct Glm5NextBench<'ctx, 'w> {
    ctx: &'ctx MetalContext,
    weights: &'w Glm5NextWeights,
    session: Option<Glm5NextSession<'w>>,
    capacity: usize,
    prefill_rows: usize,
    prefill_frozen: bool,
    vocab: usize,
}

/// Freeze only after creation succeeds, including the first warm-up session.
/// Later repetitions re-admit that same width rather than mixing execution modes.
fn create_glm_bench_session<T>(
    rows: &mut usize,
    frozen: &mut bool,
    mut create: impl FnMut(usize) -> qwen_llm::glm5_next_metal::Result<T>,
) -> qwen_llm::glm5_next_metal::Result<T> {
    if *frozen {
        return create(*rows);
    }
    let (selected, session) = prefill_plan::select(*rows, create)?;
    *rows = selected;
    *frozen = true;
    Ok(session)
}

impl<'w> Glm5NextBench<'_, 'w> {
    fn session(&mut self) -> Result<&mut Glm5NextSession<'w>> {
        self.session
            .as_mut()
            .context("GLM bench used before begin_rep")
    }
}

impl FamilyBench for Glm5NextBench<'_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    /// A fresh session (recurrent state, latent cache, pools) per rep, as
    /// `qwen run` creates one per request.
    fn begin_rep(&mut self) -> Result<()> {
        self.session = None;
        self.session = Some(create_glm_bench_session(
            &mut self.prefill_rows,
            &mut self.prefill_frozen,
            |rows| Glm5NextSession::with_prefill_rows(self.ctx, self.weights, self.capacity, rows),
        )?);
        Ok(())
    }
    /// Production prompt path: packed chunks (fast lineage), last-row logits.
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let ctx = self.ctx;
        std::hint::black_box(self.session()?.prefill_packed(ctx, ids)?);
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        let ctx = self.ctx;
        std::hint::black_box(self.session()?.forward(ctx, id)?);
        Ok(())
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: if self.prefill_rows > 0 {
                "packed_fast+last_logits"
            } else {
                "serial+logits_copy+route_status_checks"
            },
            decode_mode: "serial+logits_copy+route_status_checks",
            prefill_chunk: Some(self.prefill_rows),
        }
    }
}

struct Qwen4ExpBench<'ctx, 'model, 'gguf> {
    runner: Qwen4ExpTextRunner<'ctx, 'model, 'gguf>,
    vocab: usize,
    last_prefill_mode: &'static str,
}

impl FamilyBench for Qwen4ExpBench<'_, '_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn begin_rep(&mut self) -> Result<()> {
        Ok(self.runner.reset()?)
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let logits = self
            .runner
            .prefill_continuation_with_command_checkpoint(ids, || Ok(()))?
            .to_vec();
        std::hint::black_box(logits);
        self.last_prefill_mode = match self.runner.last_prefill_timing() {
            Some(timing) if timing.packed_token_count == timing.token_count => "packed",
            Some(timing) if timing.packed_token_count > 0 => "packed+scalar_tail",
            Some(_) => "scalar",
            None => "unreported",
        };
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        let logits = self.runner.forward_token(id)?.to_vec();
        std::hint::black_box(logits);
        Ok(())
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: self.last_prefill_mode,
            decode_mode: "forward_token+logits_copy",
            prefill_chunk: None,
        }
    }
}

struct DeepSeekV4Bench<'ctx> {
    ctx: &'ctx MetalContext,
    residency: Option<DeepSeekV4MetalResidency>,
    session: Option<DeepSeekV4Session>,
    chunk: usize,
    vocab: usize,
    last_prefill_mode: &'static str,
}

impl DeepSeekV4Bench<'_> {
    fn session(&mut self) -> Result<&mut DeepSeekV4Session> {
        self.session
            .as_mut()
            .context("DeepSeek V4 bench used before begin_rep")
    }

    /// The host copy `qwen run` makes at a prompt endpoint and per token.
    fn copy_logits(&mut self) -> Result<()> {
        let vocab = self.vocab;
        let logits = self.session()?.copy_logits_f32()?;
        ensure!(
            logits.len() == vocab,
            "DeepSeek V4 logits length {} differs from vocabulary {vocab}",
            logits.len()
        );
        std::hint::black_box(logits);
        Ok(())
    }
}

impl FamilyBench for DeepSeekV4Bench<'_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    /// A session per rep over the long-lived residency, as the JSONL
    /// requests lane and serve do.
    fn begin_rep(&mut self) -> Result<()> {
        if let Some(session) = self.session.take() {
            self.residency = Some(session.into_residency()?);
        }
        let residency = self
            .residency
            .take()
            .context("DeepSeek V4 residency was not returned by the previous rep")?;
        self.session = Some(DeepSeekV4Session::new(self.ctx, residency)?);
        Ok(())
    }
    /// Production prompt path: packed chunks without logits, the final chunk
    /// with logits; prompts under two tokens forward one token at a time.
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let ctx = self.ctx;
        let chunk = self.chunk;
        let session = self.session()?;
        let scalar = ids.len() < 2;
        if scalar {
            for &id in ids {
                session.forward_token(ctx, id)?;
            }
        } else {
            let ranges = deepseek_v4_prefill_chunk_ranges(ids.len(), chunk);
            let last = ranges.len() - 1;
            for (index, range) in ranges.into_iter().enumerate() {
                if index == last {
                    session.prefill_tokens(ctx, &ids[range])?;
                } else {
                    session.advance_tokens(ctx, &ids[range])?;
                }
            }
        }
        self.last_prefill_mode = if scalar {
            "forward_token+logits_copy"
        } else {
            "packed_chunks+final_logits_copy"
        };
        self.copy_logits()
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        let ctx = self.ctx;
        self.session()?.forward_token(ctx, id)?;
        self.copy_logits()
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: self.last_prefill_mode,
            decode_mode: "forward_token+logits_copy",
            prefill_chunk: Some(self.chunk),
        }
    }
}

struct MuseBench<'ctx, 'model> {
    runner: MuseGlimmerTextRunner<'ctx, 'model>,
    vocab: usize,
    matrix_prefill: bool,
    split_decode: bool,
    last_prefill_tokens: usize,
}

impl FamilyBench for MuseBench<'_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn begin_rep(&mut self) -> Result<()> {
        Ok(self.runner.reset()?)
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        std::hint::black_box(self.runner.prefill(ids)?);
        self.last_prefill_tokens = ids.len();
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        std::hint::black_box(self.runner.forward_token(id)?);
        Ok(())
    }
    /// The prefill label `qwen run` derives for the same prompt length.
    fn semantics(&self) -> FamilySemantics {
        let tokens = self.last_prefill_tokens;
        let packed =
            tokens / MUSE_GLIMMER_PACKED_PREFILL_QUANTUM * MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
        let tail = tokens - packed;
        FamilySemantics {
            prefill_mode: match (packed, tail, self.matrix_prefill) {
                (0, _, _) => "scalar_tail",
                (_, 0, false) => "packed_exact",
                (_, _, false) => "packed_exact+scalar_tail",
                (_, 0, true) => "packed_matrix_online",
                (_, _, true) => "packed_matrix_online+scalar_tail",
            },
            decode_mode: if self.split_decode {
                "split_decode+logits_copy"
            } else {
                "forward_token+logits_copy"
            },
            prefill_chunk: None,
        }
    }
}

struct K2Bench<'s, 'a> {
    model: &'s K2LoadedModel<'a>,
    session: Option<K2Session<'s, 'a>>,
    vocab: usize,
    last_prefill: Option<(&'static str, usize)>,
}

impl<'s, 'a> K2Bench<'s, 'a> {
    fn session(&mut self) -> Result<&mut K2Session<'s, 'a>> {
        self.session
            .as_mut()
            .context("K2 bench used before begin_rep")
    }
}

impl FamilyBench for K2Bench<'_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    /// A fresh session (KV and admission) per rep, as the request lane does.
    fn begin_rep(&mut self) -> Result<()> {
        self.session = None;
        self.session = Some(self.model.create_session(0)?);
        Ok(())
    }
    /// Production prompt path: the plan for this call's length, chunks
    /// without the head, the final chunk with logits.
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let info = self.model.prefill_info(ids.len());
        let chunk = info.chunk_tokens.max(1);
        let session = self.session()?;
        let chunks: Vec<&[u32]> = ids.chunks(chunk).collect();
        let last = chunks.len().saturating_sub(1);
        for (index, part) in chunks.into_iter().enumerate() {
            if index == last {
                std::hint::black_box(session.append(part)?);
            } else {
                session.advance(part)?;
            }
        }
        self.last_prefill = Some((info.mode, chunk));
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        std::hint::black_box(self.session()?.append(&[id])?);
        Ok(())
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: self.last_prefill.map_or("none", |(mode, _)| mode),
            decode_mode: "append+logits_copy+source_checks",
            prefill_chunk: self.last_prefill.map(|(_, chunk)| chunk),
        }
    }
}

fn synthetic_ids(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let vocab = u32::try_from(vocab).unwrap_or(u32::MAX);
    synthetic_prompt_ids(n, vocab, seed)
        .into_iter()
        .map(|id| id as u32)
        .collect()
}

/// Salt so the depth fill and the timed tokens are different sequences.
pub(crate) const DEPTH_SEED_SALT: u64 = 0x9e37_79b9_7f4a_7c15;

/// `pp512`, or `pp512@d8192` when timed after a depth fill.
pub(crate) fn shape_label(kind: &str, n: usize, depth: usize) -> String {
    if depth == 0 {
        format!("{kind}{n}")
    } else {
        format!("{kind}{n}@d{depth}")
    }
}

/// One llama-bench `pp<n>` row at `depth`: per rep a fresh state, an untimed
/// depth fill, then the timed prefill.
pub(crate) fn run_family_pp_row(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n_prompt: usize,
    depth: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    ensure!(n_prompt > 0, "--pp values must be >= 1");
    let vocab = bench.vocab_size();
    let depth_ids = synthetic_ids(depth, vocab, seed ^ DEPTH_SEED_SALT);
    let ids = synthetic_ids(n_prompt, vocab, seed);
    let rep = |bench: &mut dyn FamilyBench| -> Result<f64> {
        shutdown::checkpoint()?;
        bench.begin_pp_rep(depth)?;
        if depth > 0 {
            bench.prefill(&depth_ids)?;
        }
        let census = std::env::var_os("QWEN_V41_CENSUS").is_some();
        if census {
            qwen_llm::metal::dispatch_census_begin();
        }
        let t0 = Instant::now();
        bench.prefill(&ids)?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        if census {
            let rows = qwen_llm::metal::dispatch_census_take();
            let mut counts = std::collections::BTreeMap::new();
            for row in &rows {
                *counts.entry(row.kernel.clone()).or_insert(0usize) += 1;
            }
            let mut top: Vec<_> = counts.into_iter().collect();
            top.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
            top.truncate(30);
            eprintln!("[suite] v41 census dispatches={} top={:?}", rows.len(), top);
        }
        Ok(wall_ms)
    };
    if !no_warmup {
        rep(bench).context("family pp warmup")?;
    }
    let mut wall = Vec::with_capacity(runs);
    let mut diagnostics = Vec::new();
    for _ in 0..runs {
        wall.push(rep(bench).context("family pp rep")?);
        if let Some(report) = bench.v41_prefill_diagnostics() {
            diagnostics.push(report);
        }
    }
    let mut row = family_row(
        row_ctx,
        bench.semantics(),
        shape_label("pp", n_prompt, depth),
        n_prompt,
        depth,
        &wall,
        true,
    );
    if diagnostics.is_empty() {
        row.prefill_geometry = bench
            .prefill_report()
            .as_ref()
            .map(crate::family_options::v41_execution_geometry);
    }
    attach_v41_diagnostics(&mut row, diagnostics);
    Ok(row)
}

/// One llama-bench `tg<n>` row at `depth`: per rep a fresh state, an untimed
/// depth fill, then `n` timed single-token steps on random tokens.
pub(crate) fn run_family_tg_row(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n_gen: usize,
    depth: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    ensure!(n_gen > 0, "--tg values must be >= 1");
    let vocab = bench.vocab_size();
    let depth_ids = synthetic_ids(depth, vocab, seed ^ DEPTH_SEED_SALT);
    let mut rep_index = 0u64;
    let mut rep = |bench: &mut dyn FamilyBench| -> Result<f64> {
        shutdown::checkpoint()?;
        rep_index += 1;
        let ids = synthetic_ids(n_gen, vocab, seed.wrapping_add(rep_index));
        bench.begin_rep()?;
        if depth > 0 {
            bench.prefill(&depth_ids)?;
            if let Some(report) = bench.prefill_report() {
                eprintln!("[bench-depth-prefill] {}", serde_json::to_string(&report)?);
            }
        }
        let t0 = Instant::now();
        for &id in &ids {
            bench.decode(id)?;
        }
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };
    if !no_warmup {
        rep(bench).context("family tg warmup")?;
    }
    let mut wall = Vec::with_capacity(runs);
    let mut diagnostics = Vec::new();
    for _ in 0..runs {
        wall.push(rep(bench).context("family tg rep")?);
        if let Some(report) = bench.v41_prefill_diagnostics() {
            diagnostics.push(report);
        }
    }
    let mut row = family_row(
        row_ctx,
        bench.semantics(),
        shape_label("tg", n_gen, depth),
        n_gen,
        depth,
        &wall,
        false,
    );
    attach_v41_diagnostics(&mut row, diagnostics);
    Ok(row)
}

pub(crate) fn attach_v41_diagnostics(row: &mut BenchRow, repetitions: Vec<V41BenchPrefillRep>) {
    if repetitions.is_empty() {
        return;
    }
    let reports: Vec<_> = repetitions
        .iter()
        .filter_map(|r| r.calls.last())
        .map(|c| &c.diagnostics.report)
        .collect();
    row.prefill_geometry = reports
        .first()
        .filter(|first| {
            reports.len() == row.n_repetitions
                && reports
                    .iter()
                    .all(|r| r.geometry == first.geometry && r.execution == first.execution)
        })
        .map(|r| crate::family_options::v41_execution_geometry(r));
    if !reports.first().is_some_and(|first| {
        reports.len() == row.n_repetitions
            && reports
                .iter()
                .all(|r| r.geometry.selected_rows == first.geometry.selected_rows)
    }) {
        row.prefill_chunk = None;
    }
    if row.prefill_mode.is_some() {
        let mut actual = repetitions.iter().filter_map(|rep| {
            rep.calls
                .last()
                .map(|call| call.diagnostics.report.execution)
        });
        if let Some(first) = actual.next()
            && actual.any(|execution| execution != first)
        {
            row.prefill_mode = Some("mixed_full_forward_shrinking_ced+final_logits_copy");
            row.prefill_chunk = None;
        }
    }
    row.diagnostics = Some(serde_json::json!({
        "deepseek_v41": {"prefill_repetitions": repetitions}
    }));
    tracing::info!(target: "qwen_diag", family="deepseek_v41", test=%row.test,
        prefill_report=%row.diagnostics.as_ref().unwrap(), "bench cold prefill execution");
}

pub(crate) fn family_row(
    row_ctx: &SuiteRowContext,
    semantics: FamilySemantics,
    test: String,
    n_tokens: usize,
    n_depth: usize,
    wall_ms: &[f64],
    prefill: bool,
) -> BenchRow {
    let ts: Vec<f64> = wall_ms
        .iter()
        .map(|ms| n_tokens as f64 * 1000.0 / ms)
        .collect();
    let wall_mean = sample_mean(wall_ms);
    BenchRow {
        schema_version: BENCH_SCHEMA_VERSION,
        engine: "qwen-llm",
        build_commit: row_ctx.build_commit,
        build_dirty: row_ctx.build_dirty,
        build_identity: recorded_build_identity(),
        test_time: utc_iso8601_now(),
        model_filename: row_ctx.model_filename.clone(),
        model_size: row_ctx.model_size,
        model_n_params: row_ctx.model_n_params,
        arch_kind: row_ctx.arch_kind,
        family: row_ctx.family,
        test,
        n_tokens,
        n_depth,
        n_repetitions: wall_ms.len(),
        avg_ts: sample_mean(&ts),
        stddev_ts: sample_stdev(&ts),
        samples_ts: ts,
        samples_ns: wall_ms.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: None,
        avg_scratch_alloc_ns: None,
        avg_gpu_ns: None,
        kernel_trace_command_buffers_per_token: None,
        kernel_trace_encoders_per_token: None,
        kernel_trace_concurrent_encoders_per_token: None,
        kernel_trace_dispatches_per_token: None,
        decode_gb_per_s: None,
        prefill_geometry: None,
        prefill_chunk: if prefill {
            semantics.prefill_chunk
        } else {
            None
        },
        decode_mode: (!prefill).then_some(semantics.decode_mode),
        prefill_mode: prefill.then_some(semantics.prefill_mode),
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
        diagnostics: None,
    }
}

#[cfg(test)]
pub(crate) fn v41_pair_test_bench<S>(
    create: impl FnMut() -> Result<S>,
    prefill: impl FnMut(&mut S, &[u32], PrefillExecution) -> Result<(Vec<f32>, PrefillReport)>,
) -> impl FamilyBench {
    DeepSeekV41Bench {
        vocab: 257,
        full_chunk_rows: crate::family_options::V41_PREFILL_ROWS,
        state: None,
        create,
        step: |_: &mut S, _| bail!("paired fixture must not decode"),
        prefill,
        requested_execution: PrefillExecution::FullForward,
        position: 0,
        ced_allowed: true,
        last_execution: PrefillExecution::FullForward,
        reports: Vec::new(),
        paired: false,
        last_logits: None,
        last_prefill: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_report(fallback: Option<&str>) -> PrefillReport {
        PrefillReport {
            geometry: qwen_llm::deepseek_v41::forward::PrefillGeometry {
                start: 0,
                tokens: 3,
                selected_rows: 4096,
            },
            admission: None,
            execution: PrefillExecution::FullForward,
            fallback: fallback.map(str::to_owned),
            counters: None,
            memory: None,
            observed_incremental_metal_bytes: None,
        }
    }

    fn ced_report() -> PrefillReport {
        use qwen_llm::deepseek_v41::forward::{CedCounters, CedMemoryEstimate};
        PrefillReport {
            geometry: qwen_llm::deepseek_v41::forward::PrefillGeometry {
                start: 0,
                tokens: 3,
                selected_rows: 4096,
            },
            admission: None,
            execution: PrefillExecution::ShrinkingCed,
            fallback: None,
            counters: Some(CedCounters {
                decoder_output_rows: 3,
                completed_commands: 2,
                ..Default::default()
            }),
            memory: Some(CedMemoryEstimate {
                incremental_metal_bytes: 123,
                ..Default::default()
            }),
            observed_incremental_metal_bytes: Some(456),
        }
    }

    fn v41_row_context() -> SuiteRowContext {
        SuiteRowContext {
            build_commit: "fixture",
            build_dirty: 0,
            model_filename: "fixture.gguf".into(),
            model_size: 123,
            model_n_params: 456,
            arch_kind: "deepseek_v41",
            family: "deepseek_v41",
            power: None,
            qwen_env: Default::default(),
        }
    }

    fn v41_test_bench(
        prefill: impl FnMut(&mut usize, &[u32], PrefillExecution) -> Result<(Vec<f32>, PrefillReport)>,
    ) -> impl FamilyBench {
        DeepSeekV41Bench {
            vocab: 257,
            full_chunk_rows: crate::family_options::V41_PREFILL_ROWS,
            state: None,
            requested_execution: PrefillExecution::ShrinkingCed,
            position: 0,
            ced_allowed: true,
            last_execution: PrefillExecution::FullForward,
            reports: Vec::new(),
            paired: false,
            last_logits: None,
            last_prefill: None,
            create: || Ok(0usize),
            prefill,
            step: |position: &mut usize, _| {
                *position += 1;
                Ok(vec![0.; 257])
            },
        }
    }

    #[test]
    fn v41_ced_cold_pp_and_tg_depth_keep_actual_reports() {
        let mut calls = 0;
        let mut bench = v41_test_bench(|position, ids, execution| {
            assert_eq!(
                *position, 0,
                "CED is one whole cold prompt, never a reconstruction"
            );
            assert_eq!(execution, PrefillExecution::ShrinkingCed);
            *position += ids.len();
            calls += 1;
            Ok((vec![0.; 257], ced_report()))
        });
        let ctx = v41_row_context();
        let pp = run_family_pp_row(&mut bench, &ctx, 3, 0, 2, false, 42).unwrap();
        assert_eq!(pp.prefill_mode, Some("shrinking_ced+final_logits_copy"));
        assert_eq!(pp.prefill_chunk, None);
        let json = serde_json::to_value(pp).unwrap();
        let reps = &json["diagnostics"]["deepseek_v41"]["prefill_repetitions"];
        assert_eq!(reps.as_array().unwrap().len(), 2, "warmup is not a sample");
        let call = &reps[0]["calls"][0];
        assert_eq!(call["execution"], "shrinking_ced");
        assert_eq!(call["start_position"], 0);
        assert_eq!(call["tokens"], 3);
        assert_eq!(call["counters"]["completed_commands"], 2);
        assert_eq!(call["memory"]["incremental_metal_bytes"], 123);
        assert_eq!(call["observed_incremental_metal_bytes"], 456);

        let tg = run_family_tg_row(&mut bench, &ctx, 4, 2, 2, false, 42).unwrap();
        assert_eq!(tg.decode_mode, Some("gpu_step+logits_copy"));
        assert_eq!(tg.prefill_mode, None);
        let reps = &tg.diagnostics.as_ref().unwrap()["deepseek_v41"]["prefill_repetitions"];
        assert_eq!(reps[0]["calls"][0]["execution"], "shrinking_ced");
        assert_eq!(reps[0]["calls"][0]["tokens"], 2);
        let tg_zero = run_family_tg_row(&mut bench, &ctx, 4, 0, 1, true, 42).unwrap();
        assert_eq!(
            tg_zero.diagnostics.unwrap()["deepseek_v41"]["prefill_repetitions"][0]["calls"],
            serde_json::json!([])
        );
        drop(bench);
        assert_eq!(calls, 6);
    }

    #[test]
    fn v41_ced_pp_at_depth_is_explicit_full_forward() {
        let mut calls = Vec::new();
        let mut bench = v41_test_bench(|position, ids, execution| {
            assert_eq!(execution, PrefillExecution::FullForward);
            calls.push((*position, ids.len()));
            *position += ids.len();
            Ok((vec![0.; 257], full_report(None)))
        });
        let row = run_family_pp_row(&mut bench, &v41_row_context(), 3, 2, 1, true, 42).unwrap();
        assert_eq!(
            row.prefill_mode,
            Some(crate::family_options::V41_PREFILL_MODE)
        );
        let reports = &row.diagnostics.unwrap()["deepseek_v41"]["prefill_repetitions"][0]["calls"];
        assert_eq!(reports.as_array().unwrap().len(), 2);
        for call in reports.as_array().unwrap() {
            assert_eq!(call["requested_execution"], "shrinking_ced");
            assert_eq!(call["execution"], "full_forward");
            assert!(
                call["fallback"]
                    .as_str()
                    .unwrap()
                    .contains("cold whole-prompt only")
            );
        }
        drop(bench);
        assert_eq!(calls, [(0, 2), (2, 3)]);
    }

    #[test]
    fn v41_ced_library_fallback_and_mixed_reps_are_not_mislabeled() {
        for runs in [1, 2] {
            let mut call = 0;
            let mut bench = v41_test_bench(|position, ids, execution| {
                assert_eq!(execution, PrefillExecution::ShrinkingCed);
                assert_eq!(*position, 0);
                *position += ids.len();
                call += 1;
                let report = if call == 1 {
                    full_report(Some("CED memory admission: test refusal"))
                } else {
                    ced_report()
                };
                Ok((vec![0.; 257], report))
            });
            let row =
                run_family_pp_row(&mut bench, &v41_row_context(), 3, 0, runs, true, 42).unwrap();
            assert_eq!(
                row.prefill_mode,
                Some(if runs == 1 {
                    crate::family_options::V41_PREFILL_MODE
                } else {
                    "mixed_full_forward_shrinking_ced+final_logits_copy"
                })
            );
            assert_eq!(
                row.prefill_chunk,
                if runs == 1 {
                    Some(crate::family_options::V41_PREFILL_ROWS)
                } else {
                    None
                }
            );
            assert_eq!(row.prefill_geometry.is_some(), runs == 1);
            assert_eq!(row.n_repetitions, runs);
            assert_eq!(row.samples_ns.len(), runs);
            assert_eq!(row.samples_ts.len(), runs);
            let reports = &row.diagnostics.unwrap()["deepseek_v41"]["prefill_repetitions"];
            assert_eq!(reports.as_array().unwrap().len(), runs);
            assert_eq!(reports[0]["calls"][0]["execution"], "full_forward");
            assert_eq!(
                reports[0]["calls"][0]["fallback"],
                "CED memory admission: test refusal"
            );
            assert!(reports[0]["calls"][0]["counters"].is_null());
            if runs == 2 {
                assert_eq!(reports[1]["calls"][0]["execution"], "shrinking_ced");
            }
        }
    }

    #[test]
    fn v41_full_reps_with_different_widths_keep_raw_geometry_not_last_aggregate() {
        let mut call = 0;
        let mut bench = v41_test_bench(|position, ids, _| {
            *position += ids.len();
            call += 1;
            let mut report = full_report(Some("optional CED admission refusal"));
            report.geometry.selected_rows = if call == 1 { 4096 } else { 2048 };
            Ok((vec![0.; 257], report))
        });
        let row = run_family_pp_row(&mut bench, &v41_row_context(), 3, 0, 2, true, 42).unwrap();
        assert!(row.prefill_geometry.is_none());
        assert!(row.prefill_chunk.is_none());
        assert_eq!(
            row.prefill_mode,
            Some(crate::family_options::V41_PREFILL_MODE)
        );
        assert_eq!(row.n_repetitions, 2);
        assert_eq!(row.samples_ns.len(), 2);
        assert_eq!(row.samples_ts.len(), 2);
        let reps = &row.diagnostics.unwrap()["deepseek_v41"]["prefill_repetitions"];
        assert_eq!(reps.as_array().unwrap().len(), 2);
        for (index, width) in [4096, 2048].into_iter().enumerate() {
            assert_eq!(
                reps[index]["calls"][0]["input_geometry"]["selected_rows"],
                width
            );
            assert_eq!(reps[index]["calls"][0]["execution"], "full_forward");
            assert_eq!(
                reps[index]["calls"][0]["fallback"],
                "optional CED admission refusal"
            );
        }
    }

    fn glm_pressure() -> qwen_llm::glm5_next_metal::Glm5NextMetalError {
        let denied = qwen_llm::metal::evaluate_metal_memory_admission(
            2,
            0,
            qwen_llm::metal::MetalMemorySignals {
                host_budget: Some(qwen_llm::metal::HostMemoryBudget {
                    physical_bytes: 1024 << 30,
                    conservative_available_bytes: 1024 << 30,
                    external_gap_bytes: 0,
                    private_shadow_bytes: 0,
                }),
                recommended_max_bytes: 1,
                current_allocated_bytes: 0,
                process_limit_remaining_bytes: Some(1),
            },
            true,
        )
        .refusal()
        .unwrap();
        qwen_llm::glm5_next_metal::Glm5NextMetalError::MemoryAdmission {
            denied,
            budget_bytes: 1,
            advice: qwen_llm::glm5_next_metal::CapacityAdvice::NotEvaluated,
        }
    }

    #[test]
    fn glm_bench_freezes_only_successful_creation_and_never_changes_later_reps() {
        let (mut rows, mut frozen) = (2048, false);
        let mut attempts = Vec::new();
        let error = create_glm_bench_session(&mut rows, &mut frozen, |rows| {
            attempts.push(rows);
            Err::<(), _>(glm_pressure())
        })
        .unwrap_err();
        assert!(error.is_memory_pressure());
        assert_eq!(attempts, prefill_plan::candidates(2048, true));
        assert_eq!((rows, frozen), (2048, false));

        attempts.clear();
        let session = create_glm_bench_session(&mut rows, &mut frozen, |rows| {
            attempts.push(rows);
            if rows > 512 {
                Err(glm_pressure())
            } else {
                Ok(rows)
            }
        })
        .unwrap();
        assert_eq!(attempts, [2048, 1024, 512]);
        assert_eq!((rows, frozen, session), (512, true, 512));

        attempts.clear();
        assert!(
            create_glm_bench_session(&mut rows, &mut frozen, |rows| {
                attempts.push(rows);
                Err::<(), _>(glm_pressure())
            })
            .unwrap_err()
            .is_memory_pressure()
        );
        assert_eq!(attempts, [512]);
        assert_eq!((rows, frozen), (512, true));
        assert_eq!(
            create_glm_bench_session(&mut rows, &mut frozen, |rows| Ok(rows)).unwrap(),
            512
        );
    }

    #[test]
    fn glm_bench_nonpressure_creation_does_not_freeze_or_retry() {
        let (mut rows, mut frozen) = (2048, false);
        let mut attempts = Vec::new();
        let error = create_glm_bench_session(&mut rows, &mut frozen, |rows| {
            attempts.push(rows);
            Err::<(), _>(qwen_llm::glm5_next_metal::Glm5NextMetalError::Metal(
                qwen_llm::metal::MetalError::NoBuffer(4096),
            ))
        })
        .unwrap_err();
        assert!(!error.is_memory_pressure());
        assert_eq!(attempts, [2048]);
        assert_eq!((rows, frozen), (2048, false));
    }

    #[test]
    fn glm_bench_workspace_is_capped_to_prefill_endpoint_not_decode_extent() {
        let extent = FamilyBenchExtent::for_shapes(&[3], &[8192], &[0], (1, 1)).unwrap();
        assert_eq!(extent.forwards, 8192);
        let maximum = prefill_plan::candidates(extent.prefill_endpoint, true)[0];
        assert_eq!(maximum, 3);
        let (mut rows, mut frozen) = (maximum, false);
        assert_eq!(
            create_glm_bench_session(&mut rows, &mut frozen, |rows| Ok(rows)).unwrap(),
            3
        );
        assert_eq!((rows, frozen), (3, true));
        let (mut rows, mut frozen) = (0, false);
        create_glm_bench_session(&mut rows, &mut frozen, |rows| {
            assert_eq!(rows, 0);
            Ok(())
        })
        .unwrap();
        assert_eq!((rows, frozen), (0, true));
    }

    #[test]
    fn v41_standard_harness_keeps_fresh_state_depth_and_common_columns() {
        use std::cell::RefCell;
        let repetitions = RefCell::new(Vec::<Vec<u32>>::new());
        let mut bench = DeepSeekV41Bench {
            vocab: 257,
            full_chunk_rows: crate::family_options::V41_PREFILL_ROWS,
            state: None,
            requested_execution: PrefillExecution::FullForward,
            position: 0,
            ced_allowed: true,
            last_execution: PrefillExecution::FullForward,
            reports: Vec::new(),
            paired: false,
            last_logits: None,
            last_prefill: None,
            create: || {
                repetitions.borrow_mut().push(vec![]);
                Ok(0usize)
            },
            prefill: |position: &mut usize, tokens: &[u32], execution| {
                assert_eq!(execution, PrefillExecution::FullForward);
                let mut reps = repetitions.borrow_mut();
                let ids = reps.last_mut().unwrap();
                assert_eq!(*position, ids.len());
                ids.extend_from_slice(tokens);
                let geometry = qwen_llm::deepseek_v41::forward::PrefillGeometry {
                    start: *position,
                    tokens: tokens.len(),
                    selected_rows: 2048,
                };
                *position += tokens.len();
                Ok((
                    vec![0.; 257],
                    PrefillReport {
                        geometry,
                        ..full_report(None)
                    },
                ))
            },
            step: |position: &mut usize, token| {
                let mut reps = repetitions.borrow_mut();
                let ids = reps.last_mut().unwrap();
                assert_eq!(*position, ids.len());
                ids.push(token);
                *position += 1;
                Ok(vec![0.; 257])
            },
        };
        assert!(bench.decode(1).is_err());
        let row_ctx = SuiteRowContext {
            build_commit: "fixture",
            build_dirty: 0,
            model_filename: "fixture.gguf".into(),
            model_size: 123,
            model_n_params: 456,
            arch_kind: "deepseek_v41",
            family: "deepseek_v41",
            power: None,
            qwen_env: Default::default(),
        };
        let pp = run_family_pp_row(&mut bench, &row_ctx, 3, 2, 2, false, 42).unwrap();
        let tg = run_family_tg_row(&mut bench, &row_ctx, 4, 2, 2, false, 42).unwrap();
        assert_eq!(
            repetitions
                .borrow()
                .iter()
                .map(Vec::len)
                .collect::<Vec<_>>(),
            [5, 5, 5, 6, 6, 6]
        );
        assert_eq!(pp.test, "pp3@d2");
        assert_eq!(tg.test, "tg4@d2");
        assert_eq!(pp.prefill_chunk, Some(2048));
        let geometry = pp.prefill_geometry.as_ref().unwrap();
        assert_eq!(geometry["engine_rows_max"], 4096);
        assert_eq!(geometry["selected_rows"], 2048);
        assert_eq!(geometry["actual_chunks"], 1);
        assert_eq!(geometry["chunk_ranges"], serde_json::json!([[2, 5]]));
        assert_eq!(
            pp.prefill_mode,
            Some(crate::family_options::V41_PREFILL_MODE)
        );
        assert_eq!(tg.decode_mode, Some("gpu_step+logits_copy"));
        let wide = run_family_pp_row(&mut bench, &row_ctx, 4096, 73, 1, true, 42).unwrap();
        let geometry = wide.prefill_geometry.as_ref().unwrap();
        assert_eq!(wide.prefill_chunk, Some(2048));
        assert_eq!(geometry["actual_chunks"], 2);
        assert_eq!(
            geometry["chunk_ranges"],
            serde_json::json!([[73, 2121], [2121, 4169]])
        );
        for row in [pp, tg] {
            let json = serde_json::to_value(row).unwrap();
            assert!(json.get("diagnostics").is_none());
            assert_eq!(json["schema_version"], BENCH_SCHEMA_VERSION);
            assert_eq!(json["family"], "deepseek_v41");
            assert_eq!(json["n_repetitions"], 2);
            assert_eq!(json["samples_ns"].as_array().unwrap().len(), 2);
        }
    }

    #[test]
    fn v41_bench_extent_uses_validated_alias_without_expert_count_gate() {
        for experts in [272, 320, 384] {
            let gguf =
                crate::deepseek_v41_fixture::gguf(experts, "deepseek41.max_position_embeddings");
            assert_eq!(ModelFamily::detect(&gguf), Some(ModelFamily::DeepSeekV41));
            let plan = v41_weights::plan(&gguf).unwrap();
            assert_eq!(plan.engram_hash.max_position, 64);
            let extent = FamilyBenchExtent::for_shapes(&[32], &[16], &[32], (16, 16)).unwrap();
            validate_v41_extent(extent, plan.engram_hash.max_position).unwrap();
            assert!(
                validate_v41_extent(
                    FamilyBenchExtent {
                        forwards: 65,
                        ..extent
                    },
                    plan.engram_hash.max_position
                )
                .is_err()
            );
        }
    }

    /// The admitted extent is the deepest position any row or the warm-up
    /// reaches (no arbitrary margin, so context-edge shapes stay valid), and
    /// the prefill endpoint is the deepest absolute prefill end.
    #[test]
    fn extent_is_the_deepest_position_and_prefill_endpoint() {
        let warmup = (512, 16);
        let extent =
            FamilyBenchExtent::for_shapes(&[512, 4096], &[128], &[0, 8192], warmup).unwrap();
        assert_eq!(extent.forwards, 8192 + 4096);
        assert_eq!(extent.prefill_endpoint, 8192 + 4096);
        let tg_depth = FamilyBenchExtent::for_shapes(&[], &[128], &[0, 8192], (16, 16)).unwrap();
        assert_eq!(tg_depth.forwards, 8192 + 128);
        assert_eq!(
            tg_depth.prefill_endpoint, 8192,
            "the depth fill is the endpoint"
        );
        let tg_only = FamilyBenchExtent::for_shapes(&[], &[128], &[0], (16, 16)).unwrap();
        assert_eq!((tg_only.forwards, tg_only.prefill_endpoint), (128, 16));
        let small = FamilyBenchExtent::for_shapes(&[8], &[], &[0], (8, 16)).unwrap();
        assert_eq!(small.forwards, 16, "the tg warm-up must fit");
    }

    #[test]
    fn shape_labels_follow_llama_bench_with_depth_suffix() {
        assert_eq!(shape_label("pp", 512, 0), "pp512");
        assert_eq!(shape_label("tg", 128, 8192), "tg128@d8192");
    }

    #[test]
    fn suite_depth_defaults_to_zero_and_accepts_lists() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["qwen-bench", "suite", "-m", "model.gguf", "--pp", "512"];
            argv.extend_from_slice(extra);
            let Cmd::Suite(args) = Args::try_parse_from(argv).expect("parse suite").cmd else {
                panic!("expected suite command");
            };
            args.depth
        };
        assert_eq!(parse(&[]), vec![0]);
        assert_eq!(parse(&["-d", "0,8192,32768"]), vec![0, 8192, 32768]);
    }
}
