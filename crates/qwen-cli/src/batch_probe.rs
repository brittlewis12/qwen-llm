use anyhow::{Context, Result, anyhow, bail, ensure};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice};
use qwen_llm::deepseek_v4_metal::{
    DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES, DEEPSEEK_V4_PREFILL_MAX_TOKENS,
    DeepSeekV4MetalResidency, DeepSeekV4Session, DeepSeekV4WholeTokenProfile,
};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{KernelEncoder, MetalContext, MetalTensor, evaluate_metal_memory_admission};
use qwen_llm::metal_dflash::prefill_tokens_prompt_only_profiled;
use qwen_llm::metal_forward::{MetalForward, MetalModel, MetalSession};
use qwen_llm::model::LayerKind;
use qwen_llm::model_family::ModelFamily;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::PathBuf;
use std::sync::{Arc, mpsc};
use std::time::Instant;

const QWEN_PROBE_FIXED_SESSION_UPPER_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const QWEN_PROBE_DYNAMIC_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Parser, Debug)]
pub struct QueueOverlapProbeArgs {
    /// Path to a supported Qwen or DeepSeek V4 GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Independent request counts to probe in one loaded process.
    #[arg(long, value_delimiter = ',', default_value = "2")]
    clients: Vec<usize>,
    /// Prompt tokens consumed before the timed decode window.
    #[arg(long, default_value = "1")]
    target_ctx: usize,
    /// Decode transitions per client and arm.
    #[arg(long, default_value = "1")]
    window: usize,
    /// Feed each selected argmax back into the next transition instead of
    /// using deterministic teacher-forced tokens.
    #[arg(long)]
    generated_feedback: bool,
    /// Counterbalanced serialized/independent-queue pairs.
    #[arg(long, default_value = "2")]
    runs: usize,
    /// Deterministic prompt and decode-token seed.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// Optional pretty-JSON output file; the report is always printed.
    #[arg(long)]
    json_out: Option<PathBuf>,
}

#[derive(Serialize)]
struct ModeCapability {
    mode: &'static str,
    state: &'static str,
    detail: &'static str,
}

#[derive(Clone, Serialize)]
struct ArmMetrics {
    wall_ms: f64,
    aggregate_tokens_per_second: f64,
    makespan_tokens_per_second_per_client: f64,
    gpu_sum_ms: f64,
    gpu_span_ms: Option<f64>,
    gpu_concurrency_factor: Option<f64>,
    metal_allocated_before_sessions: u64,
    metal_allocated_after_sessions: u64,
    metal_session_delta_bytes: u64,
}

#[derive(Serialize)]
struct PairSample {
    order: &'static str,
    serialized: ArmMetrics,
    independent_queues: ArmMetrics,
    evidence_match: bool,
}

#[derive(Serialize)]
struct ProbeMedians {
    serialized_aggregate_tokens_per_second: f64,
    independent_aggregate_tokens_per_second: f64,
    aggregate_speedup: f64,
    independent_gpu_concurrency_factor: Option<f64>,
}

#[derive(Serialize)]
struct ProbeRow {
    clients: usize,
    samples: Vec<PairSample>,
    medians: ProbeMedians,
    all_evidence_match: bool,
    planned_session_bytes_each: Option<u64>,
}

#[derive(Serialize)]
struct QueueOverlapProbeReport {
    schema_version: u32,
    report_kind: &'static str,
    build: Value,
    model: String,
    family: &'static str,
    device: String,
    target_context: usize,
    decode_tokens_per_client: usize,
    runs: usize,
    warmup_pairs: usize,
    token_policy: &'static str,
    workload_kind: &'static str,
    scheduler_policy: &'static str,
    graph_policy: &'static str,
    host_submission_policy: &'static str,
    measurement_authority: &'static str,
    capabilities: Vec<ModeCapability>,
    rows: Vec<ProbeRow>,
}

struct ArmResult<E> {
    metrics: ArmMetrics,
    evidence: E,
}

#[derive(Eq, PartialEq)]
struct QwenEvidence {
    argmax_ids: Vec<Vec<i32>>,
    final_logits_sha256: Vec<String>,
}

#[derive(Eq, PartialEq)]
struct DeepSeekEvidence {
    argmax_ids: Vec<Vec<u32>>,
    final_logits_sha256: Vec<String>,
}

fn median(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut values = values.into_iter().collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

fn token_for(seed: u64, client: usize, position: usize, vocab: usize) -> u32 {
    let mixed = seed
        .wrapping_add((client as u64 + 1).wrapping_mul(7_919))
        .wrapping_add((position as u64 + 1).wrapping_mul(104_729));
    (mixed % vocab as u64) as u32
}

fn argmax_f32(values: &[f32], label: &str) -> Result<u32> {
    ensure!(!values.is_empty(), "{label} logits are empty");
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{label} logits contain a non-finite value"
    );
    let mut best = 0usize;
    for index in 1..values.len() {
        if values[index] > values[best] {
            best = index;
        }
    }
    u32::try_from(best).context("argmax token exceeds u32")
}

fn checked_transitions(clients: usize, window: usize) -> Result<usize> {
    clients
        .checked_mul(window)
        .context("queue-overlap transition count overflow")
}

fn observed_allocation_delta(before: u64, after: u64, label: &str) -> Result<u64> {
    after
        .checked_sub(before)
        .with_context(|| format!("Metal allocation counter regressed during {label}"))
}

fn require_incremental_admission(
    ctx: &MetalContext,
    session_upper_bytes: u64,
    clients: usize,
    reserve_bytes: u64,
    label: &str,
) -> Result<u64> {
    let session_bytes = session_upper_bytes
        .checked_mul(u64::try_from(clients).context("queue-overlap client count exceeds u64")?)
        .context("queue-overlap multi-session byte estimate overflow")?;
    let admission =
        evaluate_metal_memory_admission(session_bytes, reserve_bytes, ctx.memory_signals(), true);
    ensure!(
        admission.admitted,
        "{label} memory admission denied for {clients} sessions: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );
    Ok(session_bytes)
}

fn qwen_session_upper_bytes(model: &MetalModel, capacity: usize) -> Result<u64> {
    let arch = &model.arch;
    let attention_layers = (0..arch.n_layer)
        .filter(|&layer| arch.layer_kind(layer) == LayerKind::GatedAttention)
        .count() as u64;
    let kv_elements = u64::try_from(capacity)
        .context("Qwen queue-overlap capacity exceeds u64")?
        .checked_mul(attention_layers)
        .and_then(|value| value.checked_mul(2))
        .and_then(|value| value.checked_mul(u64::from(arch.n_kv_heads)))
        .and_then(|value| value.checked_mul(u64::from(arch.attn_head_dim)))
        .context("Qwen queue-overlap KV estimate overflow")?;
    let kv_bytes = kv_elements
        .checked_mul(std::mem::size_of::<half::f16>() as u64)
        .context("Qwen queue-overlap KV byte estimate overflow")?;
    QWEN_PROBE_FIXED_SESSION_UPPER_BYTES
        .checked_add(kv_bytes)
        .context("Qwen queue-overlap session estimate overflow")
}

fn validate_gpu_interval(start: f64, end: f64, label: &str) -> Result<f64> {
    ensure!(
        start.is_finite() && end.is_finite() && start > 0.0 && end > start,
        "invalid {label} GPU interval: start={start} end={end}"
    );
    Ok((end - start) * 1e3)
}

fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|value| (*value).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "non-string panic payload".to_string())
}

fn capabilities(family: ModelFamily) -> Vec<ModeCapability> {
    let (static_state, static_detail) = match family {
        ModelFamily::Qwen35 => (
            "whole_model_probe",
            "fixed B=8 dense whole-model reuse is measured; product scheduling remains pending",
        ),
        ModelFamily::Qwen35Moe => (
            "primitive_only",
            "projection and guarded block replay exist; full-model static backend required",
        ),
        ModelFamily::Qwen4Exp => (
            "unsupported",
            "Flash-Next queue-overlap probing requires a family-specific session backend",
        ),
        ModelFamily::DeepSeek4 => (
            "primitive_only",
            "shared immutable residency exists; layer-synchronous backend required",
        ),
    };
    vec![
        ModeCapability {
            mode: "resident_serialized",
            state: "production_lifecycle",
            detail: "one loaded model and private sequence state; probe graph is reported separately",
        },
        ModeCapability {
            mode: "independent_queues",
            state: "probe_only",
            detail: "complete singleton graphs overlap without dispatch-level weight reuse",
        },
        ModeCapability {
            mode: "static_layer_batch",
            state: static_state,
            detail: static_detail,
        },
        ModeCapability {
            mode: "continuous_ragged",
            state: "blocked",
            detail: "requires a promoted static layer-batch backend",
        },
    ]
}

fn summarize_rows<E: PartialEq>(
    clients: usize,
    pairs: Vec<(PairSample, E, E)>,
    planned_session_bytes_each: Option<u64>,
) -> ProbeRow {
    let all_evidence_match = pairs
        .iter()
        .all(|(_, serial, independent)| serial == independent);
    let samples = pairs
        .into_iter()
        .map(|(sample, _, _)| sample)
        .collect::<Vec<_>>();
    let serial_tps = median(
        samples
            .iter()
            .map(|sample| sample.serialized.aggregate_tokens_per_second),
    );
    let independent_tps = median(
        samples
            .iter()
            .map(|sample| sample.independent_queues.aggregate_tokens_per_second),
    );
    let overlap = samples
        .iter()
        .filter_map(|sample| sample.independent_queues.gpu_concurrency_factor)
        .collect::<Vec<_>>();
    let paired_speedup = median(samples.iter().map(|sample| {
        sample.independent_queues.aggregate_tokens_per_second
            / sample.serialized.aggregate_tokens_per_second
    }));
    ProbeRow {
        clients,
        medians: ProbeMedians {
            serialized_aggregate_tokens_per_second: serial_tps,
            independent_aggregate_tokens_per_second: independent_tps,
            aggregate_speedup: paired_speedup,
            independent_gpu_concurrency_factor: (!overlap.is_empty()).then(|| median(overlap)),
        },
        samples,
        all_evidence_match,
        planned_session_bytes_each,
    }
}

fn prepare_qwen_sessions(
    ctx: &MetalContext,
    model: &MetalModel,
    clients: usize,
    target_ctx: usize,
    window: usize,
    seed: u64,
) -> Result<Vec<MetalSession>> {
    let forward = MetalForward::new(ctx, model);
    let capacity = target_ctx
        .checked_add(window)
        .and_then(|value| value.checked_add(8))
        .context("Qwen queue-overlap session capacity overflow")?;
    let mut sessions = Vec::with_capacity(clients);
    for client in 0..clients {
        crate::shutdown::checkpoint()?;
        let prompt = super::synthetic_prompt_ids(
            target_ctx,
            model.arch.vocab_size,
            seed.wrapping_add(client as u64),
        );
        let mut session = MetalSession::fresh(ctx, model, capacity)?;
        let chunk = super::default_prefill_chunk(model.arch.kind, target_ctx);
        let mut scratch = super::fresh_prefill_scratch_for_prompt(ctx, model, chunk, target_ctx)?;
        prefill_tokens_prompt_only_profiled(&forward, &prompt, 0, &mut session, &mut scratch)?;
        sessions.push(session);
    }
    Ok(sessions)
}

fn digest_qwen_logits(session: &MetalSession, vocab_size: usize) -> String {
    let byte_len = vocab_size * std::mem::size_of::<f32>();
    let offset = usize::try_from(session.logits.offset).expect("logit offset fits usize");
    let bytes = unsafe {
        std::slice::from_raw_parts(
            session
                .logits
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset),
            byte_len,
        )
    };
    format!("{:x}", Sha256::digest(bytes))
}

fn run_qwen_serialized(
    ctx: &MetalContext,
    model: &MetalModel,
    clients: usize,
    target_ctx: usize,
    window: usize,
    seed: u64,
    session_upper_bytes: u64,
    generated_feedback: bool,
) -> Result<ArmResult<QwenEvidence>> {
    let admitted_session_bytes = require_incremental_admission(
        ctx,
        session_upper_bytes,
        clients,
        QWEN_PROBE_DYNAMIC_RESERVE_BYTES,
        "Qwen serialized queue-overlap",
    )?;
    let allocated_before = ctx.current_allocated_size();
    let mut sessions = prepare_qwen_sessions(ctx, model, clients, target_ctx, window, seed)?;
    let forward = MetalForward::new(ctx, model);
    let mut ids = Vec::with_capacity(clients);
    let mut argmax = Vec::with_capacity(clients);
    for _ in 0..clients {
        ids.push(MetalTensor::zeros_i32(ctx, vec![1])?);
        argmax.push(MetalTensor::zeros_i32(ctx, vec![1])?);
    }
    let allocated_after = ctx.current_allocated_size();
    let mut argmax_ids = vec![Vec::with_capacity(window); clients];
    let mut next_tokens = (0..clients)
        .map(|client| token_for(seed, client, target_ctx, model.arch.vocab_size as usize) as i32)
        .collect::<Vec<_>>();
    let mut gpu_sum_ms = 0.0;
    let started = Instant::now();
    for step in 0..window {
        crate::shutdown::checkpoint()?;
        for client in 0..clients {
            let token = if generated_feedback {
                next_tokens[client]
            } else {
                token_for(
                    seed,
                    client,
                    target_ctx + step,
                    model.arch.vocab_size as usize,
                ) as i32
            };
            unsafe {
                ids[client]
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<i32>()
                    .write(token);
            }
            let command = ctx
                .queue
                .commandBuffer()
                .context("serialized probe command")?;
            let encoder = KernelEncoder::begin(&command);
            let position = u32::try_from(
                target_ctx
                    .checked_add(step)
                    .context("Qwen queue-overlap position overflow")?,
            )
            .context("Qwen queue-overlap position exceeds u32")?;
            forward.encode_single_token_argmax(
                &encoder,
                position,
                &mut sessions[client],
                &ids[client],
                &argmax[client],
            )?;
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            ensure!(
                command.error().is_none(),
                "serialized Qwen probe command failed: {:?}",
                command.error()
            );
            gpu_sum_ms += validate_gpu_interval(
                command.GPUStartTime(),
                command.GPUEndTime(),
                "serialized Qwen command",
            )?;
            let selected = unsafe { *argmax[client].buffer.contents().as_ptr().cast::<i32>() };
            argmax_ids[client].push(selected);
            if generated_feedback {
                next_tokens[client] = selected;
            }
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let transitions = checked_transitions(clients, window)?;
    let allocation_delta = observed_allocation_delta(
        allocated_before,
        allocated_after,
        "Qwen serialized session construction",
    )?;
    ensure!(
        allocation_delta <= admitted_session_bytes,
        "Qwen serialized session allocation delta {allocation_delta} exceeds admitted session bytes {admitted_session_bytes}"
    );
    Ok(ArmResult {
        metrics: ArmMetrics {
            wall_ms,
            aggregate_tokens_per_second: transitions as f64 * 1e3 / wall_ms,
            makespan_tokens_per_second_per_client: window as f64 * 1e3 / wall_ms,
            gpu_sum_ms,
            gpu_span_ms: None,
            gpu_concurrency_factor: None,
            metal_allocated_before_sessions: allocated_before,
            metal_allocated_after_sessions: allocated_after,
            metal_session_delta_bytes: allocation_delta,
        },
        evidence: QwenEvidence {
            argmax_ids,
            final_logits_sha256: sessions
                .iter()
                .map(|session| digest_qwen_logits(session, model.arch.vocab_size as usize))
                .collect(),
        },
    })
}

fn run_qwen_independent(
    ctx: &MetalContext,
    model: &MetalModel,
    contexts: &[MetalContext],
    target_ctx: usize,
    window: usize,
    seed: u64,
    session_upper_bytes: u64,
    generated_feedback: bool,
) -> Result<ArmResult<QwenEvidence>> {
    let clients = contexts.len();
    let admitted_session_bytes = require_incremental_admission(
        ctx,
        session_upper_bytes,
        clients,
        QWEN_PROBE_DYNAMIC_RESERVE_BYTES,
        "Qwen independent queue-overlap",
    )?;
    let allocated_before = ctx.current_allocated_size();
    let mut sessions = prepare_qwen_sessions(ctx, model, clients, target_ctx, window, seed)?;
    let forward = MetalForward::new(ctx, model);
    let mut ids = Vec::with_capacity(clients);
    let mut argmax = Vec::with_capacity(clients);
    for _ in 0..clients {
        ids.push(MetalTensor::zeros_i32(ctx, vec![1])?);
        argmax.push(MetalTensor::zeros_i32(ctx, vec![1])?);
    }
    let allocated_after = ctx.current_allocated_size();
    let mut argmax_ids = vec![Vec::with_capacity(window); clients];
    let mut next_tokens = (0..clients)
        .map(|client| token_for(seed, client, target_ctx, model.arch.vocab_size as usize) as i32)
        .collect::<Vec<_>>();
    let mut gpu_sum_ms = 0.0;
    let mut gpu_span_ms = 0.0;
    let started = Instant::now();
    for step in 0..window {
        crate::shutdown::checkpoint()?;
        let mut commands = Vec::with_capacity(clients);
        for client in 0..clients {
            let token = if generated_feedback {
                next_tokens[client]
            } else {
                token_for(
                    seed,
                    client,
                    target_ctx + step,
                    model.arch.vocab_size as usize,
                ) as i32
            };
            unsafe {
                let ptr = ids[client].buffer.contents().as_ptr().cast::<i32>();
                ptr.write(token);
            }
            let command = contexts[client]
                .queue
                .commandBuffer()
                .context("queue-overlap command buffer")?;
            let encoder = KernelEncoder::begin(&command);
            let position = u32::try_from(
                target_ctx
                    .checked_add(step)
                    .context("Qwen queue-overlap position overflow")?,
            )
            .context("Qwen queue-overlap position exceeds u32")?;
            forward.encode_single_token_argmax(
                &encoder,
                position,
                &mut sessions[client],
                &ids[client],
                &argmax[client],
            )?;
            encoder.end();
            commands.push(command);
        }
        for command in &commands {
            command.commit();
        }
        for command in &commands {
            command.waitUntilCompleted();
        }
        for command in &commands {
            ensure!(
                command.error().is_none(),
                "queue-overlap Qwen command failed: {:?}",
                command.error()
            );
        }
        let mut min_start = f64::INFINITY;
        let mut max_end: f64 = 0.0;
        for command in &commands {
            let start = command.GPUStartTime();
            let end = command.GPUEndTime();
            gpu_sum_ms += validate_gpu_interval(start, end, "independent Qwen command")?;
            min_start = min_start.min(start);
            max_end = max_end.max(end);
        }
        gpu_span_ms += validate_gpu_interval(min_start, max_end, "Qwen command envelope")?;
        for client in 0..clients {
            let value = unsafe { *argmax[client].buffer.contents().as_ptr().cast::<i32>() };
            argmax_ids[client].push(value);
            if generated_feedback {
                next_tokens[client] = value;
            }
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let transitions = checked_transitions(clients, window)?;
    let allocation_delta = observed_allocation_delta(
        allocated_before,
        allocated_after,
        "Qwen independent session construction",
    )?;
    ensure!(
        allocation_delta <= admitted_session_bytes,
        "Qwen independent session allocation delta {allocation_delta} exceeds admitted session bytes {admitted_session_bytes}"
    );
    Ok(ArmResult {
        metrics: ArmMetrics {
            wall_ms,
            aggregate_tokens_per_second: transitions as f64 * 1e3 / wall_ms,
            makespan_tokens_per_second_per_client: window as f64 * 1e3 / wall_ms,
            gpu_sum_ms,
            gpu_span_ms: Some(gpu_span_ms),
            gpu_concurrency_factor: (gpu_span_ms > 0.0).then_some(gpu_sum_ms / gpu_span_ms),
            metal_allocated_before_sessions: allocated_before,
            metal_allocated_after_sessions: allocated_after,
            metal_session_delta_bytes: allocation_delta,
        },
        evidence: QwenEvidence {
            argmax_ids,
            final_logits_sha256: sessions
                .iter()
                .map(|session| digest_qwen_logits(session, model.arch.vocab_size as usize))
                .collect(),
        },
    })
}

fn run_qwen(
    ctx: &MetalContext,
    gguf: &GgufFile,
    args: &QueueOverlapProbeArgs,
) -> Result<Vec<ProbeRow>> {
    let bound = Model::from_gguf(gguf).context("bind Qwen queue-overlap model")?;
    let model = MetalModel::load(ctx, gguf, &bound).context("load Qwen queue-overlap weights")?;
    ensure!(
        !model.has_queue_scoped_residency_set(),
        "Qwen queue-overlap probe cannot use a model with a queue-scoped residency set"
    );
    let max_clients = args.clients.iter().copied().max().unwrap_or(1);
    let mut contexts = Vec::with_capacity(max_clients);
    for _ in 0..max_clients {
        contexts.push(
            ctx.with_new_command_queue()
                .context("create Qwen queue-overlap command queue")?,
        );
    }
    let capacity = args
        .target_ctx
        .checked_add(args.window)
        .and_then(|value| value.checked_add(8))
        .context("Qwen queue-overlap session capacity overflow")?;
    let session_upper_bytes = qwen_session_upper_bytes(&model, capacity)?;
    let mut rows = Vec::with_capacity(args.clients.len());
    for &clients in &args.clients {
        let warmup_seed = args.seed.wrapping_add(0x9e37_79b9);
        let warmup_serialized = run_qwen_serialized(
            ctx,
            &model,
            clients,
            args.target_ctx,
            args.window,
            warmup_seed,
            session_upper_bytes,
            args.generated_feedback,
        )?;
        let warmup_independent = run_qwen_independent(
            ctx,
            &model,
            &contexts[..clients],
            args.target_ctx,
            args.window,
            warmup_seed,
            session_upper_bytes,
            args.generated_feedback,
        )?;
        ensure!(
            warmup_serialized.evidence == warmup_independent.evidence,
            "Qwen queue-overlap warmup evidence changed for clients={clients}"
        );
        let mut pairs = Vec::with_capacity(args.runs);
        for run in 0..args.runs {
            let run_seed = args.seed.wrapping_add((run / 2) as u64 * 65_537);
            let (serialized, independent, order) = if run.is_multiple_of(2) {
                (
                    run_qwen_serialized(
                        ctx,
                        &model,
                        clients,
                        args.target_ctx,
                        args.window,
                        run_seed,
                        session_upper_bytes,
                        args.generated_feedback,
                    )?,
                    run_qwen_independent(
                        ctx,
                        &model,
                        &contexts[..clients],
                        args.target_ctx,
                        args.window,
                        run_seed,
                        session_upper_bytes,
                        args.generated_feedback,
                    )?,
                    "serialized_independent",
                )
            } else {
                let independent = run_qwen_independent(
                    ctx,
                    &model,
                    &contexts[..clients],
                    args.target_ctx,
                    args.window,
                    run_seed,
                    session_upper_bytes,
                    args.generated_feedback,
                )?;
                let serialized = run_qwen_serialized(
                    ctx,
                    &model,
                    clients,
                    args.target_ctx,
                    args.window,
                    run_seed,
                    session_upper_bytes,
                    args.generated_feedback,
                )?;
                (serialized, independent, "independent_serialized")
            };
            let evidence_match = serialized.evidence == independent.evidence;
            ensure!(
                evidence_match,
                "Qwen queue-overlap evidence changed for clients={clients} run={run}"
            );
            eprintln!(
                "queue-overlap: family=qwen clients={clients} run={run} order={order} serial_ms={:.3} independent_ms={:.3}",
                serialized.metrics.wall_ms, independent.metrics.wall_ms,
            );
            pairs.push((
                PairSample {
                    order,
                    serialized: serialized.metrics,
                    independent_queues: independent.metrics,
                    evidence_match,
                },
                serialized.evidence,
                independent.evidence,
            ));
        }
        rows.push(summarize_rows(clients, pairs, Some(session_upper_bytes)));
    }
    Ok(rows)
}

fn digest_logits(session: &DeepSeekV4Session) -> Result<String> {
    let logits = session.copy_logits_f32()?;
    Ok(format!(
        "{:x}",
        Sha256::digest(bytemuck::cast_slice(&logits))
    ))
}

fn prepare_deepseek_session(
    ctx: &MetalContext,
    residency: &Arc<DeepSeekV4MetalResidency>,
    client: usize,
    target_ctx: usize,
    window: usize,
    seed: u64,
) -> Result<DeepSeekV4Session> {
    let vocab_size = residency.config().vocab_size as usize;
    crate::shutdown::checkpoint()?;
    let mut session = DeepSeekV4Session::new_shared(ctx, residency.clone())?;
    let prompt = (0..target_ctx)
        .map(|position| token_for(seed, client, position, vocab_size))
        .collect::<Vec<_>>();
    for chunk in prompt.chunks(DEEPSEEK_V4_PREFILL_MAX_TOKENS) {
        crate::shutdown::checkpoint()?;
        session.advance_tokens(ctx, chunk)?;
    }
    ensure!(
        session.next_position() as usize == target_ctx,
        "DeepSeek queue-overlap prefill stopped at {}, expected {target_ctx}",
        session.next_position()
    );
    ensure!(
        session.capacity().forward_limit() >= target_ctx + window,
        "DeepSeek queue-overlap session capacity is too small"
    );
    Ok(session)
}

fn prepare_deepseek_sessions(
    contexts: &[MetalContext],
    residency: &Arc<DeepSeekV4MetalResidency>,
    target_ctx: usize,
    window: usize,
    seed: u64,
) -> Result<Vec<DeepSeekV4Session>> {
    contexts
        .iter()
        .enumerate()
        .map(|(client, ctx)| {
            prepare_deepseek_session(ctx, residency, client, target_ctx, window, seed)
        })
        .collect()
}

fn run_deepseek_serialized(
    contexts: &[MetalContext],
    residency: &Arc<DeepSeekV4MetalResidency>,
    target_ctx: usize,
    window: usize,
    seed: u64,
    session_upper_bytes: u64,
    generated_feedback: bool,
) -> Result<ArmResult<DeepSeekEvidence>> {
    let clients = contexts.len();
    let admitted_session_bytes = require_incremental_admission(
        &contexts[0],
        session_upper_bytes,
        clients,
        DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
        "DeepSeek serialized queue-overlap",
    )?;
    let allocated_before = contexts[0].current_allocated_size();
    let mut sessions = prepare_deepseek_sessions(contexts, residency, target_ctx, window, seed)?;
    let allocated_after = contexts[0].current_allocated_size();
    let vocab_size = residency.config().vocab_size as usize;
    let mut argmax_ids = vec![Vec::with_capacity(window); clients];
    let mut next_tokens = (0..clients)
        .map(|client| token_for(seed, client, target_ctx, vocab_size))
        .collect::<Vec<_>>();
    let mut gpu_sum_ms = 0.0;
    let started = Instant::now();
    for step in 0..window {
        crate::shutdown::checkpoint()?;
        for client in 0..clients {
            let token = if generated_feedback {
                next_tokens[client]
            } else {
                token_for(seed, client, target_ctx + step, vocab_size)
            };
            let profile =
                sessions[client].forward_token_whole_profiled(&contexts[client], token)?;
            gpu_sum_ms += profile.command_gpu_ms;
            if generated_feedback {
                let logits = sessions[client].copy_logits_f32()?;
                let selected = argmax_f32(&logits, "serialized DeepSeek")?;
                argmax_ids[client].push(selected);
                next_tokens[client] = selected;
            }
        }
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let evidence = DeepSeekEvidence {
        argmax_ids,
        final_logits_sha256: sessions
            .iter()
            .map(digest_logits)
            .collect::<Result<Vec<_>>>()?,
    };
    let transitions = checked_transitions(clients, window)?;
    let allocation_delta = observed_allocation_delta(
        allocated_before,
        allocated_after,
        "DeepSeek serialized session construction",
    )?;
    ensure!(
        allocation_delta <= admitted_session_bytes,
        "DeepSeek serialized session allocation delta {allocation_delta} exceeds admitted session bytes {admitted_session_bytes}"
    );
    Ok(ArmResult {
        metrics: ArmMetrics {
            wall_ms,
            aggregate_tokens_per_second: transitions as f64 * 1e3 / wall_ms,
            makespan_tokens_per_second_per_client: window as f64 * 1e3 / wall_ms,
            gpu_sum_ms,
            gpu_span_ms: None,
            gpu_concurrency_factor: None,
            metal_allocated_before_sessions: allocated_before,
            metal_allocated_after_sessions: allocated_after,
            metal_session_delta_bytes: allocation_delta,
        },
        evidence,
    })
}

fn run_deepseek_independent(
    contexts: &[MetalContext],
    residency: &Arc<DeepSeekV4MetalResidency>,
    target_ctx: usize,
    window: usize,
    seed: u64,
    session_upper_bytes: u64,
    generated_feedback: bool,
) -> Result<ArmResult<DeepSeekEvidence>> {
    let clients = contexts.len();
    let admitted_session_bytes = require_incremental_admission(
        &contexts[0],
        session_upper_bytes,
        clients,
        DEEPSEEK_V4_DYNAMIC_MEMORY_RESERVE_BYTES,
        "DeepSeek independent queue-overlap",
    )?;
    let allocated_before = contexts[0].current_allocated_size();
    let (wall_ms, allocated_after, completed) = std::thread::scope(|scope| -> Result<_> {
        let (ready_tx, ready_rx) = mpsc::channel();
        let (done_tx, done_rx) = mpsc::channel();
        let mut start_senders = Vec::with_capacity(clients);
        let mut handles = Vec::with_capacity(clients);
        for (client, ctx) in contexts.iter().enumerate() {
            let residency = residency.clone();
            let ready_tx = ready_tx.clone();
            let done_tx = done_tx.clone();
            let (start_tx, start_rx) = mpsc::channel();
            start_senders.push(start_tx);
            handles.push(scope.spawn(
                move || -> Result<(Vec<DeepSeekV4WholeTokenProfile>, Vec<u32>, String)> {
                    let mut session = match catch_unwind(AssertUnwindSafe(|| {
                        prepare_deepseek_session(
                            ctx, &residency, client, target_ctx, window, seed,
                        )
                    })) {
                        Ok(result) => result,
                        Err(payload) => Err(anyhow!(
                            "DeepSeek queue-overlap worker {client} setup panicked: {}",
                            panic_message(payload)
                        )),
                    };
                    ready_tx
                        .send(client)
                        .context("publish DeepSeek queue-overlap worker readiness")?;
                    let mut profiles = Vec::with_capacity(window);
                    let mut argmax_ids = Vec::with_capacity(window);
                    let mut next_token = token_for(
                        seed,
                        client,
                        target_ctx,
                        residency.config().vocab_size as usize,
                    );
                    for step in 0..window {
                        start_rx
                            .recv()
                            .context("receive DeepSeek queue-overlap step start")?;
                        if session.is_ok() {
                            let active = session
                                .as_mut()
                                .expect("checked DeepSeek queue-overlap session state");
                            let result = match catch_unwind(AssertUnwindSafe(
                                || -> Result<(DeepSeekV4WholeTokenProfile, Option<u32>)> {
                                    crate::shutdown::checkpoint()?;
                                    let token = if generated_feedback {
                                        next_token
                                    } else {
                                        token_for(
                                            seed,
                                            client,
                                            target_ctx + step,
                                            residency.config().vocab_size as usize,
                                        )
                                    };
                                    let profile = active
                                        .forward_token_whole_profiled(ctx, token)
                                        .map_err(anyhow::Error::from)?;
                                    let selected = if generated_feedback {
                                        Some(argmax_f32(
                                            &active.copy_logits_f32()?,
                                            "independent DeepSeek",
                                        )?)
                                    } else {
                                        None
                                    };
                                    Ok((profile, selected))
                                },
                            )) {
                                Ok(result) => result,
                                Err(payload) => Err(anyhow!(
                                    "DeepSeek queue-overlap worker {client} step {step} panicked: {}",
                                    panic_message(payload)
                                )),
                            };
                            match result {
                                Ok((profile, selected)) => {
                                    profiles.push(profile);
                                    if let Some(selected) = selected {
                                        argmax_ids.push(selected);
                                        next_token = selected;
                                    }
                                }
                                Err(error) => session = Err(error),
                            }
                        }
                        done_tx
                            .send(client)
                            .context("publish DeepSeek queue-overlap step completion")?;
                    }
                    let session = session?;
                    Ok((profiles, argmax_ids, digest_logits(&session)?))
                },
            ));
        }
        drop(ready_tx);
        drop(done_tx);
        for _ in 0..clients {
            ready_rx
                .recv()
                .context("DeepSeek queue-overlap worker exited before readiness")?;
        }
        let allocated_after = contexts[0].current_allocated_size();
        let started = Instant::now();
        for step in 0..window {
            for start in &start_senders {
                start
                    .send(())
                    .with_context(|| format!("start DeepSeek queue-overlap step {step}"))?;
            }
            let mut completed_clients = vec![false; clients];
            for _ in 0..clients {
                let client = done_rx.recv().with_context(|| {
                    format!("DeepSeek queue-overlap worker exited during step {step}")
                })?;
                ensure!(
                    client < clients && !completed_clients[client],
                    "invalid duplicate DeepSeek queue-overlap completion for client {client} at step {step}"
                );
                completed_clients[client] = true;
            }
        }
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        drop(start_senders);
        let mut completed = Vec::with_capacity(clients);
        for handle in handles {
            completed.push(
                handle
                    .join()
                    .map_err(|_| anyhow!("DeepSeek queue-overlap worker panicked"))??,
            );
        }
        Ok((wall_ms, allocated_after, completed))
    })?;
    let evidence = DeepSeekEvidence {
        argmax_ids: completed
            .iter()
            .map(|(_, argmax_ids, _)| argmax_ids.clone())
            .collect(),
        final_logits_sha256: completed
            .iter()
            .map(|(_, _, digest)| digest.clone())
            .collect(),
    };
    let profiles = completed
        .iter()
        .map(|(profiles, _, _)| profiles.as_slice())
        .collect::<Vec<_>>();
    ensure!(
        profiles.iter().all(|profiles| profiles.len() == window),
        "DeepSeek queue-overlap worker returned an incomplete profile window"
    );
    let mut gpu_sum_ms = 0.0;
    for profile in profiles.iter().flat_map(|profiles| profiles.iter()) {
        gpu_sum_ms += validate_gpu_interval(
            profile.command_gpu_start_seconds,
            profile.command_gpu_end_seconds,
            "DeepSeek command",
        )?;
    }
    let mut gpu_span_ms = 0.0;
    for step in 0..window {
        let start = profiles
            .iter()
            .map(|profiles| profiles[step].command_gpu_start_seconds)
            .fold(f64::INFINITY, f64::min);
        let end = profiles
            .iter()
            .map(|profiles| profiles[step].command_gpu_end_seconds)
            .fold(0.0, f64::max);
        gpu_span_ms += validate_gpu_interval(start, end, "DeepSeek command envelope")?;
    }
    let transitions = checked_transitions(clients, window)?;
    let allocation_delta = observed_allocation_delta(
        allocated_before,
        allocated_after,
        "DeepSeek independent session construction",
    )?;
    ensure!(
        allocation_delta <= admitted_session_bytes,
        "DeepSeek independent session allocation delta {allocation_delta} exceeds admitted session bytes {admitted_session_bytes}"
    );
    Ok(ArmResult {
        metrics: ArmMetrics {
            wall_ms,
            aggregate_tokens_per_second: transitions as f64 * 1e3 / wall_ms,
            makespan_tokens_per_second_per_client: window as f64 * 1e3 / wall_ms,
            gpu_sum_ms,
            gpu_span_ms: Some(gpu_span_ms),
            gpu_concurrency_factor: (gpu_span_ms > 0.0).then_some(gpu_sum_ms / gpu_span_ms),
            metal_allocated_before_sessions: allocated_before,
            metal_allocated_after_sessions: allocated_after,
            metal_session_delta_bytes: allocation_delta,
        },
        evidence,
    })
}

fn run_deepseek(
    ctx: &MetalContext,
    gguf: &GgufFile,
    args: &QueueOverlapProbeArgs,
) -> Result<(Vec<ProbeRow>, u64)> {
    ensure!(
        !super::env_flag_enabled("QWEN_DSV4_RESIDENCY_SET"),
        "queue-overlap probe uses multiple queues; unset QWEN_DSV4_RESIDENCY_SET"
    );
    let max_clients = args.clients.iter().copied().max().unwrap_or(1);
    let mut contexts = Vec::with_capacity(max_clients);
    for _ in 0..max_clients {
        contexts.push(
            ctx.with_new_command_queue()
                .context("create DeepSeek queue-overlap command queue")?,
        );
    }
    let forward_limit = args
        .target_ctx
        .checked_add(args.window)
        .context("DeepSeek queue-overlap forward limit overflow")?;
    let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(ctx, gguf, forward_limit)?;
    let planned_session_bytes = plan.memory_plan().session_priced_upper_bytes();
    let admitted = plan.admit_for_sessions(ctx.memory_signals(), max_clients)?;
    let realized = DeepSeekV4MetalResidency::load_from_plan(ctx, gguf, admitted)?;
    let residency = Arc::new(realized.into_residency());
    let mut rows = Vec::with_capacity(args.clients.len());
    for &clients in &args.clients {
        let warmup_seed = args.seed.wrapping_add(0x9e37_79b9);
        let warmup_serialized = run_deepseek_serialized(
            &contexts[..clients],
            &residency,
            args.target_ctx,
            args.window,
            warmup_seed,
            planned_session_bytes,
            args.generated_feedback,
        )?;
        let warmup_independent = run_deepseek_independent(
            &contexts[..clients],
            &residency,
            args.target_ctx,
            args.window,
            warmup_seed,
            planned_session_bytes,
            args.generated_feedback,
        )?;
        ensure!(
            warmup_serialized.evidence == warmup_independent.evidence,
            "DeepSeek queue-overlap warmup evidence changed for clients={clients}"
        );
        let mut pairs = Vec::with_capacity(args.runs);
        for run in 0..args.runs {
            let run_seed = args.seed.wrapping_add((run / 2) as u64 * 65_537);
            let (serialized, independent, order) = if run.is_multiple_of(2) {
                (
                    run_deepseek_serialized(
                        &contexts[..clients],
                        &residency,
                        args.target_ctx,
                        args.window,
                        run_seed,
                        planned_session_bytes,
                        args.generated_feedback,
                    )?,
                    run_deepseek_independent(
                        &contexts[..clients],
                        &residency,
                        args.target_ctx,
                        args.window,
                        run_seed,
                        planned_session_bytes,
                        args.generated_feedback,
                    )?,
                    "serialized_independent",
                )
            } else {
                let independent = run_deepseek_independent(
                    &contexts[..clients],
                    &residency,
                    args.target_ctx,
                    args.window,
                    run_seed,
                    planned_session_bytes,
                    args.generated_feedback,
                )?;
                let serialized = run_deepseek_serialized(
                    &contexts[..clients],
                    &residency,
                    args.target_ctx,
                    args.window,
                    run_seed,
                    planned_session_bytes,
                    args.generated_feedback,
                )?;
                (serialized, independent, "independent_serialized")
            };
            let evidence_match = serialized.evidence == independent.evidence;
            ensure!(
                evidence_match,
                "DeepSeek queue-overlap evidence changed for clients={clients} run={run}"
            );
            eprintln!(
                "queue-overlap: family=deepseek4 clients={clients} run={run} order={order} serial_ms={:.3} independent_ms={:.3}",
                serialized.metrics.wall_ms, independent.metrics.wall_ms,
            );
            pairs.push((
                PairSample {
                    order,
                    serialized: serialized.metrics,
                    independent_queues: independent.metrics,
                    evidence_match,
                },
                serialized.evidence,
                independent.evidence,
            ));
        }
        rows.push(summarize_rows(clients, pairs, Some(planned_session_bytes)));
    }
    Ok((rows, planned_session_bytes))
}

pub fn run(args: QueueOverlapProbeArgs, build: Value) -> Result<()> {
    ensure!(args.target_ctx > 0, "--target-ctx must be nonzero");
    ensure!(args.window > 0, "--window must be nonzero");
    ensure!(args.runs > 0, "--runs must be nonzero");
    ensure!(
        args.runs.is_multiple_of(2),
        "--runs must be even so each seed executes both AB and BA"
    );
    ensure!(!args.clients.is_empty(), "--clients must not be empty");
    ensure!(
        args.clients
            .iter()
            .all(|clients| (1..=16).contains(clients)),
        "--clients values must be in 1..=16"
    );
    let endpoint = args
        .target_ctx
        .checked_add(args.window)
        .context("queue-overlap endpoint overflow")?;
    ensure!(
        u32::try_from(endpoint).is_ok(),
        "queue-overlap endpoint exceeds the u32 position contract"
    );
    for &clients in &args.clients {
        checked_transitions(clients, args.window)?;
    }
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open queue-overlap model {}", args.model.display()))?;
    let family = ModelFamily::detect(&gguf).context("unsupported queue-overlap model family")?;
    let ctx = MetalContext::new().context("create queue-overlap Metal context")?;
    let rows = match family {
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => run_qwen(&ctx, &gguf, &args)?,
        ModelFamily::Qwen4Exp => {
            bail!("queue-overlap probing is not supported for Qwen3.8-Flash-Next")
        }
        ModelFamily::DeepSeek4 => run_deepseek(&ctx, &gguf, &args)?.0,
    };
    let (graph_policy, host_submission_policy) = match family {
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => (
            "monolithic_encode_single_token_argmax",
            "single_host_thread_encode_all_then_commit_all",
        ),
        ModelFamily::Qwen4Exp => {
            unreachable!("Qwen3.8-Flash-Next queue-overlap requests fail above")
        }
        ModelFamily::DeepSeek4 => (
            "forward_token_whole_profiled",
            "one_host_thread_per_client_lockstep",
        ),
    };
    let report = QueueOverlapProbeReport {
        schema_version: 2,
        report_kind: "cross_family_queue_overlap_probe",
        build,
        model: args.model.display().to_string(),
        family: family.architecture_name(),
        device: ctx.device.name().to_string(),
        target_context: args.target_ctx,
        decode_tokens_per_client: args.window,
        runs: args.runs,
        warmup_pairs: 1,
        token_policy: if args.generated_feedback {
            "backend_native_prompt_deterministic_first_token_then_greedy_feedback"
        } else {
            "backend_native_prompt_deterministic_teacher_forced_decode_distinct_per_client"
        },
        workload_kind: if args.generated_feedback {
            "independent_generated_continuations"
        } else {
            "independent_singleton_requests"
        },
        scheduler_policy: "per_step_lockstep",
        graph_policy,
        host_submission_policy,
        measurement_authority: "diagnostic_queue_overlap_not_production_throughput",
        capabilities: capabilities(family),
        rows,
    };
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(path) = &args.json_out {
        std::fs::write(path, format!("{json}\n"))
            .with_context(|| format!("write queue-overlap report {}", path.display()))?;
    }
    println!("{json}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_policy_separates_clients_and_is_reproducible() {
        assert_eq!(token_for(7, 2, 11, 100_000), token_for(7, 2, 11, 100_000));
        assert_ne!(token_for(7, 1, 11, 100_000), token_for(7, 2, 11, 100_000));
        assert_ne!(token_for(7, 2, 10, 100_000), token_for(7, 2, 11, 100_000));
    }

    #[test]
    fn generated_feedback_argmax_is_tie_stable_and_finite() {
        assert_eq!(argmax_f32(&[-1.0, 3.0, 3.0, 2.0], "test").unwrap(), 1);
        assert!(argmax_f32(&[0.0, f32::NAN], "test").is_err());
    }

    #[test]
    fn capability_table_never_mislabels_queue_overlap_as_batching() {
        for family in [
            ModelFamily::Qwen35,
            ModelFamily::Qwen35Moe,
            ModelFamily::Qwen4Exp,
            ModelFamily::DeepSeek4,
        ] {
            let rows = capabilities(family);
            let queues = rows
                .iter()
                .find(|row| row.mode == "independent_queues")
                .unwrap();
            assert_eq!(queues.state, "probe_only");
            let batching = rows
                .iter()
                .find(|row| row.mode == "static_layer_batch")
                .unwrap();
            assert_ne!(batching.state, "production");
        }
        let flash_next = capabilities(ModelFamily::Qwen4Exp);
        assert_eq!(
            flash_next
                .iter()
                .find(|row| row.mode == "static_layer_batch")
                .unwrap()
                .state,
            "unsupported"
        );
    }
}
