use super::dense_block_batch::read_f16_prefix;
use super::{
    cosine_max_abs, env_flag_default_on, env_flag_enabled, f32_rms_delta,
    fresh_gdn_replay_sessions_with_capacity, read_f32_tensor,
};
use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_llm::dense_batch8::{DENSE_BATCH8_WIDTH, DenseBatch8Executor};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{MetalContext, evaluate_metal_memory_admission};
use qwen_llm::metal_forward::{MetalForward, MetalModel, MetalSession};
use qwen_llm::model::ArchKind;
use qwen_llm::tensor::GgmlType;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const BATCH: usize = DENSE_BATCH8_WIDTH;
const SESSION_ADMISSION_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const DYNAMIC_RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const RAMP_CAPACITY: usize = 512;

#[derive(Parser, Debug)]
pub struct DecodeDenseWholeBatchArgs {
    /// Path to a dense Qwen GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Untimed teacher-forced transitions before measurement.
    #[arg(long, default_value = "2")]
    warmup_steps: usize,
    /// Measured teacher-forced transitions per slot and arm.
    #[arg(long, default_value = "6")]
    steps: usize,
    /// High-occupancy static-B=8 ramp before the compared streams.
    #[arg(long, default_value = "500")]
    ramp_ms: u64,
    /// Deterministic per-slot token seed.
    #[arg(long, default_value = "1")]
    seed: u64,
    /// Report numerical evidence without enforcing the diagnostic gates.
    #[arg(long)]
    no_check: bool,
}

struct StepResult {
    wall_ms: f64,
    gpu_ms: f64,
    argmax: Vec<i32>,
    logits: Vec<Vec<f32>>,
}

#[derive(Default)]
struct StepEvidence {
    min_cos_logits: f64,
    max_abs_logits: f32,
    max_relative_rms_logits: f64,
    min_cos_x: f64,
    max_abs_x: f32,
    max_relative_rms_x: f64,
}

#[derive(Default)]
struct StateEvidence {
    min_cos_state: f64,
    max_abs_state: f32,
    min_cos_conv: f64,
    max_abs_conv: f32,
    min_cos_kv: f64,
    max_abs_kv: f32,
}

fn token_for(seed: u64, slot: usize, position: usize, vocab: usize) -> i32 {
    let mixed = seed
        .wrapping_add((slot as u64 + 1).wrapping_mul(7_919))
        .wrapping_add((position as u64 + 1).wrapping_mul(104_729));
    (mixed % vocab as u64) as i32
}

fn tokens_for(seed: u64, position: usize, vocab: usize) -> [i32; BATCH] {
    std::array::from_fn(|slot| token_for(seed, slot, position, vocab))
}

fn ensure_frontier(sessions: &[MetalSession], expected: usize, label: &str) -> Result<()> {
    for (slot, session) in sessions.iter().enumerate() {
        ensure!(
            session
                .kv_n_pos
                .iter()
                .all(|&position| position == expected),
            "{label} slot {slot} has a nonuniform attention frontier: {:?}",
            session.kv_n_pos
        );
    }
    Ok(())
}

fn run_static_step(
    executor: &mut DenseBatch8Executor<'_>,
    sessions: &mut [MetalSession],
    tokens: &[i32; BATCH],
    position: u32,
) -> Result<StepResult> {
    let started = Instant::now();
    let step = executor.step_with_cancel(*tokens, position, sessions, || {
        crate::shutdown::checkpoint().is_err()
    })?;
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let gpu_ms = step.gpu_ms.context("static whole-model GPU timestamp")?;
    let logits = sessions
        .iter()
        .map(|session| read_f32_tensor(&session.logits))
        .collect();
    Ok(StepResult {
        wall_ms,
        gpu_ms,
        argmax: step.argmax_ids.to_vec(),
        logits,
    })
}

fn run_serial_step(
    forward: &MetalForward<'_>,
    sessions: &mut [MetalSession],
    tokens: &[i32; BATCH],
    position: u32,
) -> Result<StepResult> {
    ensure!(
        sessions.len() == BATCH,
        "serialized whole-model batch must be 8"
    );
    ensure_frontier(sessions, position as usize, "serialized pre-encode")?;
    let started = Instant::now();
    let mut argmax = Vec::with_capacity(BATCH);
    let mut gpu_ms = 0.0;
    for slot in 0..BATCH {
        let (token, profile) = forward
            .single_token_argmax_profiled(tokens[slot], position, &mut sessions[slot])
            .with_context(|| format!("serialized whole-model slot {slot}"))?;
        argmax.push(token);
        gpu_ms += profile.gpu_kernel_ms;
    }
    let wall_ms = started.elapsed().as_secs_f64() * 1e3;
    let logits = sessions
        .iter()
        .map(|session| read_f32_tensor(&session.logits))
        .collect();
    Ok(StepResult {
        wall_ms,
        gpu_ms,
        argmax,
        logits,
    })
}

fn relative_rms(reference: &[f32], candidate: &[f32]) -> f64 {
    let rms = (reference
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        / reference.len() as f64)
        .sqrt();
    if rms > 0.0 {
        f32_rms_delta(reference, candidate) / rms
    } else {
        0.0
    }
}

fn compare_step(
    serial: &StepResult,
    static_batch: &StepResult,
    serial_sessions: &[MetalSession],
    static_sessions: &[MetalSession],
    expected_frontier: usize,
    enforce: bool,
) -> Result<StepEvidence> {
    ensure_frontier(serial_sessions, expected_frontier, "serialized post-encode")?;
    ensure_frontier(static_sessions, expected_frontier, "static post-encode")?;
    ensure!(
        serial.argmax == static_batch.argmax,
        "static whole-model argmax mismatch: serial={:?} static={:?}",
        serial.argmax,
        static_batch.argmax
    );
    let mut evidence = StepEvidence {
        min_cos_logits: 1.0,
        min_cos_x: 1.0,
        ..StepEvidence::default()
    };
    for slot in 0..BATCH {
        let serial_logits = &serial.logits[slot];
        let static_logits = &static_batch.logits[slot];
        ensure!(
            serial_logits.iter().all(|value| value.is_finite())
                && static_logits.iter().all(|value| value.is_finite()),
            "whole-model logits contain non-finite values at slot {slot}"
        );
        let (cos_logits, abs_logits) = cosine_max_abs(serial_logits, static_logits);
        evidence.min_cos_logits = evidence.min_cos_logits.min(cos_logits);
        evidence.max_abs_logits = evidence.max_abs_logits.max(abs_logits);
        evidence.max_relative_rms_logits = evidence
            .max_relative_rms_logits
            .max(relative_rms(serial_logits, static_logits));

        let serial_x = read_f32_tensor(&serial_sessions[slot].x);
        let static_x = read_f32_tensor(&static_sessions[slot].x);
        ensure!(
            serial_x.iter().all(|value| value.is_finite())
                && static_x.iter().all(|value| value.is_finite()),
            "whole-model residual contains non-finite values at slot {slot}"
        );
        let (cos_x, abs_x) = cosine_max_abs(&serial_x, &static_x);
        evidence.min_cos_x = evidence.min_cos_x.min(cos_x);
        evidence.max_abs_x = evidence.max_abs_x.max(abs_x);
        evidence.max_relative_rms_x = evidence
            .max_relative_rms_x
            .max(relative_rms(&serial_x, &static_x));
    }
    if enforce {
        ensure!(
            evidence.min_cos_logits >= 0.999
                && evidence.max_relative_rms_logits <= 0.02
                && evidence.min_cos_x >= 0.999
                && evidence.max_relative_rms_x <= 0.02,
            "static whole-model numerical gate failed: logits_cos={} logits_rel_rms={} x_cos={} x_rel_rms={}",
            evidence.min_cos_logits,
            evidence.max_relative_rms_logits,
            evidence.min_cos_x,
            evidence.max_relative_rms_x,
        );
    }
    Ok(evidence)
}

fn update_pair_metrics(min_cos: &mut f64, max_abs: &mut f32, a: &[f32], b: &[f32]) -> Result<()> {
    ensure!(
        a.iter().all(|value| value.is_finite()) && b.iter().all(|value| value.is_finite()),
        "persistent state contains a non-finite value"
    );
    let (cos, abs) = cosine_max_abs(a, b);
    *min_cos = min_cos.min(cos);
    *max_abs = max_abs.max(abs);
    Ok(())
}

fn audit_persistent_state(
    serial: &[MetalSession],
    static_batch: &[MetalSession],
    used_positions: usize,
    kv_dim: usize,
    enforce: bool,
) -> Result<StateEvidence> {
    let mut evidence = StateEvidence {
        min_cos_state: 1.0,
        min_cos_conv: 1.0,
        min_cos_kv: 1.0,
        ..StateEvidence::default()
    };
    let kv_elements = used_positions
        .checked_mul(kv_dim)
        .context("whole-model KV audit length overflow")?;
    for slot in 0..BATCH {
        for layer in 0..serial[slot].gdn_state.len() {
            let serial_state = read_f32_tensor(&serial[slot].gdn_state[layer]);
            let static_state = read_f32_tensor(&static_batch[slot].gdn_state[layer]);
            update_pair_metrics(
                &mut evidence.min_cos_state,
                &mut evidence.max_abs_state,
                &serial_state,
                &static_state,
            )?;
            let serial_conv = read_f32_tensor(&serial[slot].gdn_conv[layer]);
            let static_conv = read_f32_tensor(&static_batch[slot].gdn_conv[layer]);
            update_pair_metrics(
                &mut evidence.min_cos_conv,
                &mut evidence.max_abs_conv,
                &serial_conv,
                &static_conv,
            )?;
        }
        for layer in 0..serial[slot].kv_k.len() {
            let serial_k = read_f16_prefix(&serial[slot].kv_k[layer], kv_elements)?;
            let static_k = read_f16_prefix(&static_batch[slot].kv_k[layer], kv_elements)?;
            update_pair_metrics(
                &mut evidence.min_cos_kv,
                &mut evidence.max_abs_kv,
                &serial_k,
                &static_k,
            )?;
            let serial_v = read_f16_prefix(&serial[slot].kv_v[layer], kv_elements)?;
            let static_v = read_f16_prefix(&static_batch[slot].kv_v[layer], kv_elements)?;
            update_pair_metrics(
                &mut evidence.min_cos_kv,
                &mut evidence.max_abs_kv,
                &serial_v,
                &static_v,
            )?;
        }
    }
    if enforce {
        ensure!(
            evidence.min_cos_state >= 0.999
                && evidence.max_abs_state <= 0.1
                && evidence.min_cos_conv >= 0.999
                && evidence.max_abs_conv <= 0.2
                && evidence.min_cos_kv >= 0.999
                && evidence.max_abs_kv <= 0.1,
            "static whole-model state gate failed: state_cos={} state_abs={} conv_cos={} conv_abs={} kv_cos={} kv_abs={}",
            evidence.min_cos_state,
            evidence.max_abs_state,
            evidence.min_cos_conv,
            evidence.max_abs_conv,
            evidence.min_cos_kv,
            evidence.max_abs_kv,
        );
    }
    Ok(evidence)
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    if values.len().is_multiple_of(2) {
        (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
    } else {
        values[values.len() / 2]
    }
}

fn ramp_static(
    ctx: &MetalContext,
    forward: &MetalForward<'_>,
    executor: &mut DenseBatch8Executor<'_>,
    ramp_ms: u64,
    seed: u64,
) -> Result<usize> {
    let mut sessions =
        fresh_gdn_replay_sessions_with_capacity(ctx, forward.model, BATCH, RAMP_CAPACITY)?;
    ensure!(
        sessions
            .iter()
            .all(|session| session.kv_k.iter().all(|kv| kv.dtype == GgmlType::F16)),
        "whole-model probe currently requires F16 KV"
    );
    let started = Instant::now();
    let target = Duration::from_millis(ramp_ms);
    let mut position = 0usize;
    while position == 0 || (started.elapsed() < target && position < RAMP_CAPACITY) {
        crate::shutdown::checkpoint()?;
        let tokens = tokens_for(seed, position, forward.model.arch.vocab_size as usize);
        executor.step_with_cancel(tokens, position as u32, &mut sessions, || {
            crate::shutdown::checkpoint().is_err()
        })?;
        position += 1;
    }
    Ok(position)
}

pub fn run(args: DecodeDenseWholeBatchArgs) -> Result<()> {
    ensure!(args.steps > 0, "--steps must be nonzero");
    let total_steps = args
        .warmup_steps
        .checked_add(args.steps)
        .context("whole-model step count overflow")?;
    ensure!(
        total_steps <= RAMP_CAPACITY,
        "whole-model short-frontier probe supports at most {RAMP_CAPACITY} total steps"
    );
    for flag in [
        "QWEN_DECODE_GDN_NOOP_FRONT",
        "QWEN_DECODE_GDN_NOOP_OUT",
        "QWEN_DECODE_GDN_NOOP_QKV",
        "QWEN_DECODE_GDN_NOOP_Z",
        "QWEN_DECODE_GDN_NOOP_BETA",
        "QWEN_DECODE_GDN_NOOP_ALPHA",
    ] {
        ensure!(
            !env_flag_enabled(flag),
            "decode-dense-whole-batch does not support {flag}"
        );
    }
    ensure!(
        env_flag_default_on("QWEN_DECODE_DENSE_CONCURRENT_GDN"),
        "decode-dense-whole-batch requires the production concurrent-GDN baseline"
    );
    let fused_post_norm = env_flag_enabled("QWEN_DECODE_FUSED_RESIDUAL_RMSNORM");

    let ctx = MetalContext::new().context("create dense whole-model Metal context")?;
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open dense whole-model model {}", args.model.display()))?;
    let bound = Model::from_gguf(&gguf).context("bind dense whole-model model")?;
    let model = MetalModel::load(&ctx, &gguf, &bound).context("load dense whole-model model")?;
    ensure!(
        model.arch.kind == ArchKind::Dense,
        "decode-dense-whole-batch requires a dense Qwen model"
    );
    ensure!(
        !model.has_queue_scoped_residency_set(),
        "whole-model probe does not support a queue-scoped residency set"
    );
    let admission = evaluate_metal_memory_admission(
        SESSION_ADMISSION_BYTES,
        DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    ensure!(
        admission.admitted,
        "whole-model B=8 admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );

    let forward = MetalForward::new(&ctx, &model);
    let mut ramp_executor = DenseBatch8Executor::new(&ctx, &model)?;
    let ramp_repetitions = ramp_static(
        &ctx,
        &forward,
        &mut ramp_executor,
        args.ramp_ms,
        args.seed.wrapping_add(0x9e37_79b9),
    )?;
    drop(ramp_executor);
    let mut executor = DenseBatch8Executor::new(&ctx, &model)?;

    let capacity = total_steps
        .checked_add(1)
        .context("whole-model capacity overflow")?;
    let allocated_before_sessions = ctx.current_allocated_size();
    let mut serial = fresh_gdn_replay_sessions_with_capacity(&ctx, &model, BATCH, capacity)?;
    let mut static_batch = fresh_gdn_replay_sessions_with_capacity(&ctx, &model, BATCH, capacity)?;
    let allocated_after_sessions = ctx.current_allocated_size();
    ensure!(
        serial.iter().chain(&static_batch).all(|session| {
            session.kv_k.iter().all(|kv| kv.dtype == GgmlType::F16)
                && session.kv_v.iter().all(|kv| kv.dtype == GgmlType::F16)
        }),
        "whole-model probe currently requires F16 KV"
    );

    println!(
        "[decode-dense-whole-batch] model={} batch={} warmup_steps={} measured_steps={} ramp_ms={} ramp_repetitions={} baseline=production_concurrent_gdn candidate=layer_major_static_b8 frontier=short_equal attention=causal_multikey lm_head=batched fused_post_norm={} measurement_authority=serialized_short_frontier_only independent_queue_gate=pending allocation_delta_bytes={}",
        args.model.display(),
        BATCH,
        args.warmup_steps,
        args.steps,
        args.ramp_ms,
        ramp_repetitions,
        fused_post_norm,
        allocated_after_sessions.saturating_sub(allocated_before_sessions),
    );
    println!(
        "phase\tposition\torder\tserial_wall_ms\tserial_gpu_sum_ms\tstatic_wall_ms\tstatic_gpu_ms\twall_speedup\tgpu_speedup\tmin_cos_logits\tmax_abs_logits\tmax_relative_rms_logits\tmin_cos_x\tmax_abs_x\tmax_relative_rms_x\targmax"
    );

    let mut serial_wall = Vec::with_capacity(args.steps);
    let mut serial_gpu = Vec::with_capacity(args.steps);
    let mut static_wall = Vec::with_capacity(args.steps);
    let mut static_gpu = Vec::with_capacity(args.steps);
    let mut paired_speedup = Vec::with_capacity(args.steps);
    let mut worst_step = StepEvidence {
        min_cos_logits: 1.0,
        min_cos_x: 1.0,
        ..StepEvidence::default()
    };
    for position in 0..total_steps {
        crate::shutdown::checkpoint()?;
        let tokens = tokens_for(args.seed, position, model.arch.vocab_size as usize);
        let (serial_step, static_step, order) = if position.is_multiple_of(2) {
            (
                run_serial_step(&forward, &mut serial, &tokens, position as u32)?,
                run_static_step(&mut executor, &mut static_batch, &tokens, position as u32)?,
                "serial_static",
            )
        } else {
            let static_step =
                run_static_step(&mut executor, &mut static_batch, &tokens, position as u32)?;
            let serial_step = run_serial_step(&forward, &mut serial, &tokens, position as u32)?;
            (serial_step, static_step, "static_serial")
        };
        let evidence = compare_step(
            &serial_step,
            &static_step,
            &serial,
            &static_batch,
            position + 1,
            !args.no_check,
        )?;
        worst_step.min_cos_logits = worst_step.min_cos_logits.min(evidence.min_cos_logits);
        worst_step.max_abs_logits = worst_step.max_abs_logits.max(evidence.max_abs_logits);
        worst_step.max_relative_rms_logits = worst_step
            .max_relative_rms_logits
            .max(evidence.max_relative_rms_logits);
        worst_step.min_cos_x = worst_step.min_cos_x.min(evidence.min_cos_x);
        worst_step.max_abs_x = worst_step.max_abs_x.max(evidence.max_abs_x);
        worst_step.max_relative_rms_x = worst_step
            .max_relative_rms_x
            .max(evidence.max_relative_rms_x);
        let phase = if position < args.warmup_steps {
            "warmup"
        } else {
            serial_wall.push(serial_step.wall_ms);
            serial_gpu.push(serial_step.gpu_ms);
            static_wall.push(static_step.wall_ms);
            static_gpu.push(static_step.gpu_ms);
            paired_speedup.push(serial_step.wall_ms / static_step.wall_ms);
            "measured"
        };
        println!(
            "{phase}\t{position}\t{order}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.9}\t{:.6}\t{:.9}\t{:.9}\t{:.6}\t{:.9}\t{}",
            serial_step.wall_ms,
            serial_step.gpu_ms,
            static_step.wall_ms,
            static_step.gpu_ms,
            serial_step.wall_ms / static_step.wall_ms,
            serial_step.gpu_ms / static_step.gpu_ms,
            evidence.min_cos_logits,
            evidence.max_abs_logits,
            evidence.max_relative_rms_logits,
            evidence.min_cos_x,
            evidence.max_abs_x,
            evidence.max_relative_rms_x,
            serial_step
                .argmax
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
    }

    let kv_dim = model.arch.n_kv_heads as usize * model.arch.attn_head_dim as usize;
    let state =
        audit_persistent_state(&serial, &static_batch, total_steps, kv_dim, !args.no_check)?;
    let serial_median_wall = median(&serial_wall);
    let static_median_wall = median(&static_wall);
    let serial_median_gpu = median(&serial_gpu);
    let static_median_gpu = median(&static_gpu);
    println!(
        "summary\tbatch={}\tmeasurement_authority=serialized_short_frontier_only\tindependent_queue_gate=pending\tserial_median_wall_ms={serial_median_wall:.4}\tstatic_median_wall_ms={static_median_wall:.4}\tpaired_median_wall_speedup={:.4}\tserial_aggregate_tps={:.3}\tstatic_aggregate_tps={:.3}\tserial_median_gpu_sum_ms={serial_median_gpu:.4}\tstatic_median_gpu_ms={static_median_gpu:.4}\tgpu_speedup={:.4}\tworst_min_cos_logits={:.9}\tworst_max_abs_logits={:.6}\tworst_max_relative_rms_logits={:.9}\tworst_min_cos_x={:.9}\tworst_max_abs_x={:.6}\tworst_max_relative_rms_x={:.9}\tmin_cos_state={:.9}\tmax_abs_state={:.6}\tmin_cos_conv={:.9}\tmax_abs_conv={:.6}\tmin_cos_kv={:.9}\tmax_abs_kv={:.6}",
        BATCH,
        median(&paired_speedup),
        BATCH as f64 * 1e3 / serial_median_wall,
        BATCH as f64 * 1e3 / static_median_wall,
        serial_median_gpu / static_median_gpu,
        worst_step.min_cos_logits,
        worst_step.max_abs_logits,
        worst_step.max_relative_rms_logits,
        worst_step.min_cos_x,
        worst_step.max_abs_x,
        worst_step.max_relative_rms_x,
        state.min_cos_state,
        state.max_abs_state,
        state.min_cos_conv,
        state.max_abs_conv,
        state.min_cos_kv,
        state.max_abs_kv,
    );
    Ok(())
}
