//! Focused artifact-selection probe: tokenizer fingerprints, paired
//! teacher-forced observations, and three-session 48K memory admission.

use super::*;
use anyhow::{Context, ensure};
use qwen_llm::{
    deepseek_v41::{
        forward::{Forward as V41Forward, PrefillExecution},
        gguf as v41_weights,
    },
    gguf::GgufFile,
    glm5_next_metal::{Glm5NextSession, Glm5NextWeights, prefill_plan},
    metal::MetalContext,
    tokenizer::{NativeTokenizer, token_ids_sha256_i32le},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub(crate) enum ArtifactChoiceMode {
    Identity,
    Quality,
    Fit,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub(crate) enum ArtifactFamily {
    V41,
    Glm53,
}

impl ArtifactFamily {
    fn as_str(self) -> &'static str {
        match self {
            Self::V41 => "deepseek-v4.1-flash-1017",
            Self::Glm53 => "glm-5.3-flash",
        }
    }

    fn check_architecture(self, g: &GgufFile) -> Result<()> {
        let architecture = g.architecture().unwrap_or_default();
        match self {
            Self::V41 => ensure!(
                matches!(architecture.as_str(), "deepseek4" | "deepseek41"),
                "family V41 does not match GGUF architecture {architecture:?}"
            ),
            Self::Glm53 => ensure!(
                matches!(architecture.as_str(), "glm5-next" | "glm5next"),
                "family GLM53 does not match GGUF architecture {architecture:?}"
            ),
        }
        Ok(())
    }
}

#[derive(Clone)]
struct Prompt {
    id: &'static str,
    stratum: &'static str,
    path: PathBuf,
    utf8_sha256: String,
    ids: Vec<i32>,
}

pub(crate) fn run(args: ArtifactChoiceArgs) -> Result<()> {
    let gguf = Arc::new(
        GgufFile::open(&args.model).with_context(|| format!("open {}", args.model.display()))?,
    );
    args.family.check_architecture(&gguf)?;
    let tokenizer_sha256 = tokenizer_identity_sha256(&gguf)?;

    let report = match args.mode {
        ArtifactChoiceMode::Identity => identity_report(&args, &gguf, tokenizer_sha256),
        ArtifactChoiceMode::Quality => {
            let expected = args
                .expected_tokenizer_sha256
                .as_deref()
                .context("quality mode requires --expected-tokenizer-sha256")?;
            ensure!(
                tokenizer_sha256 == expected,
                "tokenizer identity mismatch: expected {expected}, found {tokenizer_sha256}"
            );
            quality_report(&args, gguf, tokenizer_sha256)?
        }
        ArtifactChoiceMode::Fit => fit_report(&args, gguf, tokenizer_sha256)?,
    };
    write_report(args.out.as_deref(), &report)
}

fn identity_report(args: &ArtifactChoiceArgs, gguf: &GgufFile, tokenizer_sha256: String) -> Value {
    let shards: Vec<Value> = gguf
        .shards
        .iter()
        .map(|shard| {
            let bytes = fs::metadata(&shard.path).map(|m| m.len()).ok();
            json!({"path":shard.path.display().to_string(),"bytes":bytes})
        })
        .collect();
    let shard_bytes_total = shards
        .iter()
        .filter_map(|s| s["bytes"].as_u64())
        .sum::<u64>();
    json!({
        "schema_version": 1,
        "kind": "artifact_identity",
        "family": args.family.as_str(),
        "model": args.model.display().to_string(),
        "architecture": gguf.architecture(),
        "tokenizer_sha256": tokenizer_sha256,
        "shards": shards,
        "shard_bytes_total": shard_bytes_total,
    })
}

fn quality_report(
    args: &ArtifactChoiceArgs,
    gguf: Arc<GgufFile>,
    tokenizer_sha256: String,
) -> Result<Value> {
    let depth = args.depth;
    let targets = args.targets;
    ensure!(
        depth > 0 && targets > 0,
        "depth and target count must be positive"
    );
    let tokenizer = NativeTokenizer::from_gguf(&gguf).context("open native GGUF tokenizer")?;
    let mut prompts = load_prompts(
        args.cohort_dir
            .as_deref()
            .context("quality mode requires --cohort-dir")?,
        &tokenizer,
        depth,
        targets,
    )?;
    let context_capacity = depth
        .checked_add(targets)
        .and_then(|n| n.checked_add(16))
        .context("quality context capacity overflow")?;
    let ctx = MetalContext::new().context("initialize MetalContext")?;
    let thermal_start = thermal_level();
    let score_start = Instant::now();

    let load_summary = match args.family {
        ArtifactFamily::V41 => score_v41(
            args,
            &gguf,
            &ctx,
            &mut prompts,
            depth,
            targets,
            context_capacity,
        )?,
        ArtifactFamily::Glm53 => {
            score_glm53(&gguf, &ctx, &mut prompts, depth, targets, context_capacity)?
        }
    };
    let prompt_values: Vec<Value> = prompts.iter().map(|prompt| prompt.report.clone()).collect();
    Ok(json!({
        "schema_version": 1,
        "kind": "artifact_quality",
        "family": args.family.as_str(),
        "model": args.model.display().to_string(),
        "architecture": gguf.architecture(),
        "tokenizer_sha256": tokenizer_sha256,
        "depth": depth,
        "targets_per_prompt": targets,
        "device": serde_json::to_value(ctx.device_facts())?,
        "thermal_pressure_start": thermal_start,
        "score_run_seconds": score_start.elapsed().as_secs_f64(),
        "load": load_summary,
        "prompts": prompt_values,
    }))
}

#[derive(Clone)]
struct PromptScored {
    prompt: Prompt,
    report: Value,
}

fn load_prompts(
    cohort_dir: &Path,
    tokenizer: &NativeTokenizer,
    depth: usize,
    targets: usize,
) -> Result<Vec<PromptScored>> {
    const FILES: [(&str, &str, &str); 4] = [
        ("rust-agent-corpus", "code", "rust-agent-corpus.txt"),
        ("metal-routed-kernels", "code", "metal-routed-kernels.txt"),
        ("python-dflash-profile", "code", "python-dflash-profile.txt"),
        ("performance-log", "prose", "performance-log.txt"),
    ];
    FILES
        .into_iter()
        .map(|(id, stratum, filename)| {
            let path = cohort_dir.join(filename);
            let text = fs::read_to_string(&path)
                .with_context(|| format!("read cohort prompt {}", path.display()))?;
            let utf8_sha256 = sha256(text.as_bytes());
            let ids = tokenizer.encode(&text, false)?;
            ensure!(
                ids.len() >= depth + targets + 1,
                "{id} token count {} is shorter than depth+targets ({})",
                ids.len(),
                depth + targets
            );
            Ok(PromptScored {
                prompt: Prompt {
                    id,
                    stratum,
                    path,
                    utf8_sha256,
                    ids,
                },
                report: Value::Null,
            })
        })
        .collect()
}

fn score_v41(
    _args: &ArtifactChoiceArgs,
    gguf: &Arc<GgufFile>,
    ctx: &MetalContext,
    prompts: &mut [PromptScored],
    depth: usize,
    targets: usize,
    capacity: usize,
) -> Result<Value> {
    let execution = crate::family_options::v41_prefill_execution_from_env()?;
    let cold = qwen_llm::deepseek_v41::forward::plan_cold_prefill_with_target(
        ctx,
        gguf,
        depth,
        capacity,
        &[execution],
        0,
        Some(4_294_967_296),
    )
    .context("plan V4.1 resident model and request")?;
    let weights = v41_weights::ResidentWeights::load_with_storage_plan(
        ctx,
        gguf.clone(),
        &cold.resident_storage,
    )
    .context("load V4.1 resident weights")?;
    let forward = V41Forward::from_resident_with_cold_plan(
        ctx,
        &weights,
        weights.token_map.clone(),
        weights.raw_pad,
        &cold,
    )
    .map_err(anyhow::Error::msg)
    .context("bind V4.1 forward")?;
    let request_plan = forward
        .plan_request_from_cold(&cold)
        .context("plan V4.1 48K request")?;
    let mut prompt_seconds = Vec::with_capacity(prompts.len());
    let mut decode_seconds = Vec::with_capacity(prompts.len());
    for item in prompts {
        let token_ids: Vec<u32> = item
            .prompt
            .ids
            .iter()
            .map(|&id| u32::try_from(id).context("negative token id"))
            .collect::<Result<_>>()?;
        let mut state = forward
            .new_state_from_plan(&request_plan)
            .context("allocate V4.1 quality state")?;
        let prefill_start = Instant::now();
        let (mut logits, prefill_report) = forward
            .prefill_with_execution(&mut state, &token_ids[..depth], None, execution)
            .map_err(anyhow::Error::msg)
            .context("V4.1 quality prefix prefill")?;
        prompt_seconds.push(prefill_start.elapsed().as_secs_f64());
        let (observations, decode_seconds_one) = score_targets_v41(
            &forward,
            &mut state,
            &token_ids,
            &mut logits,
            depth,
            targets,
        )?;
        decode_seconds.push(decode_seconds_one);
        item.report = prompt_report(item, observations, format!("{prefill_report:?}"));
    }
    Ok(json!({
        "retained_weight_bytes": weights.bytes.resident_bytes(),
        "disk_engram_bytes": weights.bytes.disk_engram,
        "engram_streamed_from_disk": weights.bytes.disk_engram > 0,
        "prefill_execution_requested": format!("{execution:?}"),
        "prefill_selected_rows": request_plan.prefill().selected_rows,
        "request_memory_plan": format!("{:?}", request_plan.memory()),
        "retained_weight_resources": format!("{:?}", weights.retained_resources),
        "prompt_prefill_seconds": prompt_seconds,
        "teacher_forced_decode_seconds": decode_seconds,
        "memory_signals_after_score": memory_signals(ctx),
        "thermal_pressure_end": thermal_level(),
    }))
}

fn score_targets_v41(
    forward: &V41Forward<'_>,
    state: &mut qwen_llm::deepseek_v41::forward::State,
    token_ids: &[u32],
    logits: &mut Vec<f32>,
    depth: usize,
    targets: usize,
) -> Result<(Vec<Value>, f64)> {
    let start = Instant::now();
    let mut rows = Vec::with_capacity(targets);
    for position in depth..depth + targets {
        rows.push(score_row(logits, position, token_ids[position])?);
        if position + 1 < depth + targets {
            *logits = forward
                .step(state, token_ids[position], None)
                .map_err(anyhow::Error::msg)
                .context("V4.1 teacher-forced decode")?;
        }
    }
    Ok((rows, start.elapsed().as_secs_f64()))
}

fn score_glm53(
    gguf: &Arc<GgufFile>,
    ctx: &MetalContext,
    prompts: &mut [PromptScored],
    depth: usize,
    targets: usize,
    capacity: usize,
) -> Result<Value> {
    let layout = qwen_llm::glm5_next::Glm5NextArtifactLayout::inspect(gguf)
        .context("inspect GLM-5.3 artifact")?;
    let maximum = prefill_plan::candidates(capacity, layout.packed_prefill())[0];
    let (prefill_rows, _) = prefill_plan::select(maximum, |rows| {
        qwen_llm::glm5_next_metal::preflight_session(ctx, gguf, layout.model(), capacity, rows)
    })
    .context("preflight GLM-5.3 quality request")?;
    let weights = Glm5NextWeights::load(ctx, gguf).context("load GLM-5.3 weights")?;
    let mut prompt_seconds = Vec::with_capacity(prompts.len());
    let mut decode_seconds = Vec::with_capacity(prompts.len());
    for item in prompts {
        let token_ids: Vec<u32> = item
            .prompt
            .ids
            .iter()
            .map(|&id| u32::try_from(id).context("negative token id"))
            .collect::<Result<_>>()?;
        let mut session = Glm5NextSession::with_prefill_rows(ctx, &weights, capacity, prefill_rows)
            .context("allocate GLM-5.3 quality session")?;
        let prefill_start = Instant::now();
        let mut logits = session
            .prefill_packed(ctx, &token_ids[..depth])
            .context("GLM-5.3 quality prefix prefill")?;
        prompt_seconds.push(prefill_start.elapsed().as_secs_f64());
        let decode_start = Instant::now();
        let mut rows = Vec::with_capacity(targets);
        for position in depth..depth + targets {
            rows.push(score_row(&logits, position, token_ids[position])?);
            if position + 1 < depth + targets {
                logits = session
                    .forward(ctx, token_ids[position])
                    .context("GLM-5.3 teacher-forced decode")?;
            }
        }
        decode_seconds.push(decode_start.elapsed().as_secs_f64());
        item.report = prompt_report(item, rows, "packed_fast+last_logits".into());
    }
    Ok(json!({
        "retained_weight_bytes": weights.retained_bytes,
        "prefill_rows": prefill_rows,
        "prefill_execution": "packed_fast+last_logits",
        "engram_streamed_from_disk": false,
        "prompt_prefill_seconds": prompt_seconds,
        "teacher_forced_decode_seconds": decode_seconds,
        "memory_signals_after_score": memory_signals(ctx),
        "thermal_pressure_end": thermal_level(),
    }))
}

fn score_row(logits: &[f32], position: usize, target: u32) -> Result<Value> {
    ensure!(!logits.is_empty(), "empty logits at token {position}");
    ensure!(
        logits.iter().all(|value| value.is_finite()),
        "non-finite logits at token {position}"
    );
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = max
        + logits
            .iter()
            .map(|&value| (f64::from(value) - max).exp())
            .sum::<f64>()
            .ln();
    let target_index = usize::try_from(target)?;
    let target_logit = *logits
        .get(target_index)
        .with_context(|| format!("target {target} exceeds vocab {}", logits.len()))?;
    let top1 = logits
        .iter()
        .enumerate()
        .max_by(|(ia, a), (ib, b)| a.total_cmp(b).then_with(|| ib.cmp(ia)))
        .map(|(i, _)| i)
        .context("empty logits")?;
    Ok(json!({
        "position": position,
        "target_id": target,
        "nll": lse - f64::from(target_logit),
        "top1_id": top1,
    }))
}

fn prompt_report(item: &PromptScored, observations: Vec<Value>, execution: String) -> Value {
    let nll: Vec<f64> = observations
        .iter()
        .filter_map(|row| row["nll"].as_f64())
        .collect();
    let hits = observations
        .iter()
        .filter(|row| row["top1_id"] == row["target_id"])
        .count();
    json!({
        "id": item.prompt.id,
        "stratum": item.prompt.stratum,
        "source": item.prompt.path.file_name().and_then(|name| name.to_str()),
        "source_utf8_sha256": item.prompt.utf8_sha256,
        "token_count": item.prompt.ids.len(),
        "token_ids_sha256_i32le": token_ids_sha256_i32le(&item.prompt.ids),
        "targets": observations.len(),
        "mean_nll": nll.iter().sum::<f64>() / nll.len().max(1) as f64,
        "top1_accuracy": hits as f64 / observations.len().max(1) as f64,
        "prefill_execution": execution,
        "observations": observations,
    })
}

fn fit_report(
    args: &ArtifactChoiceArgs,
    gguf: Arc<GgufFile>,
    tokenizer_sha256: String,
) -> Result<Value> {
    let ctx = MetalContext::new().context("initialize MetalContext")?;
    let thermal_start = thermal_level();
    let start = Instant::now();
    let capacity = args.depth;
    ensure!(capacity > 0, "fit depth must be positive");
    let tokenizer = NativeTokenizer::from_gguf(&gguf).context("open fit tokenizer")?;
    let mut prompts = load_prompts(
        args.cohort_dir
            .as_deref()
            .context("fit mode requires --cohort-dir")?,
        &tokenizer,
        capacity,
        1,
    )?;
    let streams: Vec<Vec<u32>> = prompts
        .iter_mut()
        .take(3)
        .map(|item| {
            item.prompt.ids[..capacity]
                .iter()
                .map(|&id| u32::try_from(id).context("negative token id"))
                .collect::<Result<_>>()
        })
        .collect::<Result<_>>()?;
    let (agents_admitted, per_agent_plan, load_summary, selected) = match args.family {
        ArtifactFamily::V41 => fit_v41(&gguf, &ctx, capacity, &streams)?,
        ArtifactFamily::Glm53 => fit_glm53(&gguf, &ctx, capacity, &streams)?,
    };
    let prefill_workspace_agents = load_summary["prefill_workspace_agents"]
        .as_u64()
        .unwrap_or(0) as usize;
    Ok(json!({
        "schema_version": 1,
        "kind": "artifact_fit",
        "family": args.family.as_str(),
        "model": args.model.display().to_string(),
        "architecture": gguf.architecture(),
        "tokenizer_sha256": tokenizer_sha256,
        "capacity": capacity,
        "requested_agents": 3,
        "agents_admitted": agents_admitted,
        "fits_three_agent_48k": agents_admitted == 3 && prefill_workspace_agents == 3,
        "selected_prefill_geometry": selected,
        "per_agent_memory_plan": per_agent_plan,
        "load": load_summary,
        "elapsed_seconds": start.elapsed().as_secs_f64(),
        "device": serde_json::to_value(ctx.device_facts())?,
        "memory_signals_end": memory_signals(&ctx),
        "thermal_pressure_start": thermal_start,
        "thermal_pressure_end": thermal_level(),
    }))
}

fn fit_v41(
    gguf: &Arc<GgufFile>,
    ctx: &MetalContext,
    capacity: usize,
    streams: &[Vec<u32>],
) -> Result<(usize, Value, Value, Value)> {
    let execution = crate::family_options::v41_prefill_execution_from_env()?;
    let cold = qwen_llm::deepseek_v41::forward::plan_cold_prefill_with_target(
        ctx,
        gguf,
        capacity,
        capacity,
        &[execution],
        0,
        Some(4_294_967_296),
    )
    .context("plan V4.1 48K fit")?;
    let weights = v41_weights::ResidentWeights::load_with_storage_plan(
        ctx,
        gguf.clone(),
        &cold.resident_storage,
    )
    .context("load V4.1 resident weights for fit")?;
    let forward = V41Forward::from_resident_with_cold_plan(
        ctx,
        &weights,
        weights.token_map.clone(),
        weights.raw_pad,
        &cold,
    )
    .map_err(anyhow::Error::msg)
    .context("bind V4.1 fit forward")?;
    let mut states = Vec::with_capacity(3);
    let mut per_agent_plan = Value::Null;
    let mut failure = None;
    for _ in 0..3 {
        let plan = match forward.plan_request_from_cold(&cold) {
            Ok(plan) => plan,
            Err(error) => {
                failure = Some(format!("request plan: {error:#}"));
                break;
            }
        };
        per_agent_plan = json!({
            "selected_prefill_rows": plan.prefill().selected_rows,
            "request_memory": format!("{:?}", plan.memory()),
        });
        match forward.new_state_from_plan(&plan) {
            Ok(state) => states.push(state),
            Err(error) => {
                failure = Some(format!("{error:#}"));
                break;
            }
        }
    }
    let mut workspace_times = Vec::new();
    let mut prefill_reports = Vec::new();
    if states.len() == 3 {
        for (index, state) in states.iter_mut().enumerate() {
            let prefill_start = Instant::now();
            match forward.prefill_with_execution(state, &streams[index], None, execution) {
                Ok((_, report)) => {
                    workspace_times.push(prefill_start.elapsed().as_secs_f64());
                    prefill_reports.push(format!("{report:?}"));
                }
                Err(error) => {
                    failure = Some(format!("48K prefill workspace for agent {index}: {error}"));
                    break;
                }
            }
        }
    }
    let selected =
        json!({"prefill_execution":format!("{execution:?}"),"prefill_rows":cold.full_chunk_rows});
    let load = json!({
        "retained_weight_bytes": weights.bytes.resident_bytes(),
        "disk_engram_bytes": weights.bytes.disk_engram,
        "engram_streamed_from_disk": weights.bytes.disk_engram > 0,
        "retained_weight_resources": format!("{:?}", weights.retained_resources),
        "allocation_failure": failure,
        "prefill_workspace_agents": workspace_times.len(),
        "prefill_seconds": workspace_times,
        "prefill_reports": prefill_reports,
        "memory_signals_after_sessions": memory_signals(ctx),
    });
    Ok((states.len(), per_agent_plan, load, selected))
}

fn fit_glm53(
    gguf: &Arc<GgufFile>,
    ctx: &MetalContext,
    capacity: usize,
    streams: &[Vec<u32>],
) -> Result<(usize, Value, Value, Value)> {
    let layout = qwen_llm::glm5_next::Glm5NextArtifactLayout::inspect(gguf)
        .context("inspect GLM-5.3 fit artifact")?;
    let maximum = prefill_plan::candidates(capacity, layout.packed_prefill())[0];
    let weights = Glm5NextWeights::load(ctx, gguf).context("load GLM-5.3 weights for fit")?;
    let mut sessions = Vec::with_capacity(3);
    let mut rows_used = Vec::with_capacity(3);
    let mut per_agent_plan = Value::Null;
    let mut failure = None;
    for _ in 0..3 {
        match prefill_plan::select(maximum, |rows| {
            Glm5NextSession::with_prefill_rows(ctx, &weights, capacity, rows)
        }) {
            Ok((rows, session)) => {
                let ledger = Glm5NextSession::plan_memory(ctx, &weights, capacity, rows)?;
                per_agent_plan = json!({
                    "prefill_rows": rows,
                    "cumulative_peak_bytes": ledger.peak_bytes(),
                    "session_extra_bytes": ledger.peak_bytes().saturating_sub(weights.retained_bytes),
                    "ledger": format!("{ledger:?}"),
                });
                rows_used.push(rows);
                sessions.push(session);
            }
            Err(error) => {
                failure = Some(format!("{error:#}"));
                break;
            }
        }
    }
    let mut prefill_times = Vec::new();
    let mut prefill_failure = None;
    if sessions.len() == 3 {
        for (index, session) in sessions.iter_mut().enumerate() {
            let prefill_start = Instant::now();
            match session.prefill_packed(ctx, &streams[index]) {
                Ok(_) => prefill_times.push(prefill_start.elapsed().as_secs_f64()),
                Err(error) => {
                    prefill_failure =
                        Some(format!("48K prefill workspace for agent {index}: {error}"));
                    break;
                }
            }
        }
    }
    let load = json!({
        "retained_weight_bytes": weights.retained_bytes,
        "allocation_failure": failure,
        "prefill_workspace_agents": prefill_times.len(),
        "prefill_seconds": prefill_times,
        "prefill_failure": prefill_failure,
        "memory_signals_after_sessions": memory_signals(ctx),
    });
    Ok((
        sessions.len(),
        per_agent_plan,
        load,
        json!({"maximum_prefill_rows":maximum,"rows_per_agent":rows_used}),
    ))
}

fn memory_signals(ctx: &MetalContext) -> Value {
    let signals = ctx.memory_signals();
    json!({
        "recommended_max_bytes": signals.recommended_max_bytes,
        "current_allocated_bytes": signals.current_allocated_bytes,
        "process_limit_remaining_bytes": signals.process_limit_remaining_bytes,
        "host_remaining_bytes": signals.host_remaining_bytes(),
        "signals": format!("{signals:?}"),
    })
}

fn thermal_level() -> Option<String> {
    std::process::Command::new("notifyutil")
        .args(["-g", "com.apple.system.thermalpressurelevel"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn write_report(path: Option<&Path>, report: &Value) -> Result<()> {
    let encoded = serde_json::to_string_pretty(report)?;
    if let Some(path) = path {
        fs::write(path, &encoded).with_context(|| format!("write {}", path.display()))?;
        println!("{}", path.display());
    } else {
        println!("{encoded}");
    }
    Ok(())
}

fn sha256(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn tokenizer_identity_sha256(gguf: &GgufFile) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(b"artifact-choice-tokenizer-v2\0");
    // Report the source architecture separately: GLM releases use both
    // `glm5next` and `glm5-next`, which dispatch to identical GLM4 tokenization.
    for key in ["tokenizer.ggml.model", "tokenizer.ggml.pre"] {
        let value = gguf
            .get_str(key)
            .with_context(|| format!("missing {key}"))?;
        hash_identity_field(&mut digest, key, value.as_bytes());
    }
    let tokens = gguf
        .get_string_array_full("tokenizer.ggml.tokens")?
        .context("missing tokenizer.ggml.tokens")?;
    hash_identity_field(
        &mut digest,
        "tokenizer.ggml.tokens.count",
        &(tokens.len() as u64).to_le_bytes(),
    );
    for token in tokens {
        hash_identity_field(&mut digest, "tokenizer.ggml.tokens.item", token.as_bytes());
    }
    let token_types = gguf
        .get_i64_array("tokenizer.ggml.token_type")?
        .context("missing tokenizer.ggml.token_type")?;
    hash_identity_field(
        &mut digest,
        "tokenizer.ggml.token_type.count",
        &(token_types.len() as u64).to_le_bytes(),
    );
    for token_type in token_types {
        digest.update(token_type.to_le_bytes());
    }
    let merges = gguf
        .get_string_array_full("tokenizer.ggml.merges")?
        .context("missing tokenizer.ggml.merges")?;
    hash_identity_field(
        &mut digest,
        "tokenizer.ggml.merges.count",
        &(merges.len() as u64).to_le_bytes(),
    );
    for merge in merges {
        hash_identity_field(&mut digest, "tokenizer.ggml.merges.item", merge.as_bytes());
    }
    for key in ["tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id"] {
        let mut value = vec![0];
        if let Some(token_id) = gguf.get_u64(key) {
            value[0] = 1;
            value.extend_from_slice(&token_id.to_le_bytes());
        }
        hash_identity_field(&mut digest, key, &value);
    }
    for key in [
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
    ] {
        let value = gguf.get_bool(key)?.unwrap_or(false);
        hash_identity_field(&mut digest, key, &[u8::from(value)]);
    }
    Ok(hex(&digest.finalize()))
}

fn hash_identity_field(digest: &mut Sha256, name: &str, bytes: &[u8]) {
    digest.update((name.len() as u64).to_le_bytes());
    digest.update(name.as_bytes());
    digest.update((bytes.len() as u64).to_le_bytes());
    digest.update(bytes);
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
