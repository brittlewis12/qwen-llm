use super::{
    cosine_max_abs, env_flag_default_on, env_flag_enabled, f32_rms_delta,
    fresh_gdn_replay_sessions_with_capacity, fresh_prefill_scratch_for_prompt, read_f32_tensor,
};
use anyhow::{Context, Result, ensure};
use clap::Parser;
use objc2_metal::MTLBuffer;
use qwen_llm::dense_batch8::{DENSE_BATCH8_WIDTH, DenseBatch8Executor};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{MetalContext, evaluate_metal_memory_admission};
use qwen_llm::metal_dflash::{
    PrefillScratchConfig, plan_prefill_scratch_with_matrix_max_pos_configured,
    prefill_tokens_with_multi_hidden,
};
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
    /// Restore both arms at this causal frontier before measured continuation.
    #[arg(long, default_value = "0")]
    frontier_tokens: usize,
    /// Packed-prefill chunk used to construct a nonzero frontier snapshot.
    #[arg(long, default_value = "1024")]
    prefill_chunk: usize,
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

struct SessionArms {
    serial: Vec<MetalSession>,
    static_batch: Vec<MetalSession>,
    prefill_ms: f64,
    restore_ms: f64,
    snapshot_bytes: u64,
    allocation_delta_bytes: u64,
    session_allocation_bytes: u64,
    scratch_upper_bytes: u64,
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
    kv_start_position: usize,
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
    let kv_start_elements = kv_start_position
        .checked_mul(kv_dim)
        .context("whole-model KV audit start overflow")?;
    let kv_elements = used_positions
        .checked_sub(kv_start_position)
        .context("whole-model KV audit range underflow")?
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
            let serial_k =
                read_f16_range(&serial[slot].kv_k[layer], kv_start_elements, kv_elements)?;
            let static_k = read_f16_range(
                &static_batch[slot].kv_k[layer],
                kv_start_elements,
                kv_elements,
            )?;
            update_pair_metrics(
                &mut evidence.min_cos_kv,
                &mut evidence.max_abs_kv,
                &serial_k,
                &static_k,
            )?;
            let serial_v =
                read_f16_range(&serial[slot].kv_v[layer], kv_start_elements, kv_elements)?;
            let static_v = read_f16_range(
                &static_batch[slot].kv_v[layer],
                kv_start_elements,
                kv_elements,
            )?;
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

fn read_f16_range(
    tensor: &qwen_llm::metal::MetalTensor,
    start_elements: usize,
    elements: usize,
) -> Result<Vec<f32>> {
    ensure!(tensor.dtype == GgmlType::F16, "KV audit requires F16");
    ensure!(tensor.offset.is_multiple_of(2), "unaligned F16 KV tensor");
    ensure!(
        tensor.offset.checked_add(tensor.n_bytes()) <= Some(u64::try_from(tensor.buffer.length())?),
        "KV audit tensor exceeds its backing buffer"
    );
    let end_elements = start_elements
        .checked_add(elements)
        .context("KV audit element range overflow")?;
    ensure!(
        (end_elements as u64).checked_mul(2) <= Some(tensor.n_bytes()),
        "KV audit range exceeds tensor bytes"
    );
    let base = usize::try_from(tensor.offset / 2)?;
    let ptr = tensor.buffer.contents().as_ptr().cast::<u16>();
    ensure!(!ptr.is_null(), "KV audit tensor is not CPU visible");
    Ok((start_elements..end_elements)
        .map(|index| half::f16::from_bits(unsafe { *ptr.add(base + index) }).to_f32())
        .collect())
}

fn ensure_tensor_prefix_exact(
    source: &qwen_llm::metal::MetalTensor,
    restored: &qwen_llm::metal::MetalTensor,
    bytes: usize,
    label: &str,
) -> Result<()> {
    ensure!(
        bytes as u64 <= source.n_bytes() && bytes as u64 <= restored.n_bytes(),
        "{label} exact-restore range exceeds a tensor"
    );
    let source_start = usize::try_from(source.offset)?;
    let restored_start = usize::try_from(restored.offset)?;
    ensure!(
        source_start.checked_add(bytes) <= Some(source.buffer.length()),
        "{label} source range exceeds its buffer"
    );
    ensure!(
        restored_start.checked_add(bytes) <= Some(restored.buffer.length()),
        "{label} restored range exceeds its buffer"
    );
    let source_ptr = source.buffer.contents().as_ptr().cast::<u8>();
    let restored_ptr = restored.buffer.contents().as_ptr().cast::<u8>();
    ensure!(
        !source_ptr.is_null() && !restored_ptr.is_null(),
        "{label} tensor is not CPU visible"
    );
    let source_bytes = unsafe { std::slice::from_raw_parts(source_ptr.add(source_start), bytes) };
    let restored_bytes =
        unsafe { std::slice::from_raw_parts(restored_ptr.add(restored_start), bytes) };
    ensure!(
        source_bytes == restored_bytes,
        "{label} restore is not exact"
    );
    Ok(())
}

fn ensure_restored_source_exact(
    source: &MetalSession,
    restored: &MetalSession,
    frontier_tokens: usize,
    kv_dim: usize,
) -> Result<()> {
    ensure!(
        source.kv_n_pos == restored.kv_n_pos,
        "restored KV frontiers differ from the source"
    );
    ensure!(
        source.gdn_state.len() == restored.gdn_state.len()
            && source.gdn_conv.len() == restored.gdn_conv.len()
            && source.kv_k.len() == restored.kv_k.len()
            && source.kv_v.len() == restored.kv_v.len(),
        "restored persistent inventory differs from the source"
    );
    for layer in 0..source.gdn_state.len() {
        ensure_tensor_prefix_exact(
            &source.gdn_state[layer],
            &restored.gdn_state[layer],
            usize::try_from(source.gdn_state[layer].n_bytes())?,
            "GDN state",
        )?;
        ensure_tensor_prefix_exact(
            &source.gdn_conv[layer],
            &restored.gdn_conv[layer],
            usize::try_from(source.gdn_conv[layer].n_bytes())?,
            "GDN convolution",
        )?;
    }
    let kv_bytes = frontier_tokens
        .checked_mul(kv_dim)
        .and_then(|elements| elements.checked_mul(2))
        .context("exact restored KV byte count overflow")?;
    for layer in 0..source.kv_k.len() {
        ensure_tensor_prefix_exact(&source.kv_k[layer], &restored.kv_k[layer], kv_bytes, "KV K")?;
        ensure_tensor_prefix_exact(&source.kv_v[layer], &restored.kv_v[layer], kv_bytes, "KV V")?;
    }
    Ok(())
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

fn qwen_environment_names() -> String {
    let mut names = std::env::vars_os()
        .filter_map(|(name, _)| {
            let name = name.to_string_lossy();
            name.starts_with("QWEN_").then(|| name.into_owned())
        })
        .collect::<Vec<_>>();
    names.sort();
    if names.is_empty() {
        "none".to_string()
    } else {
        names.join(",")
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

fn sessions_at_frontier(
    ctx: &MetalContext,
    model: &MetalModel,
    forward: &MetalForward<'_>,
    frontier_tokens: usize,
    continuation_tokens: usize,
    prefill_chunk: usize,
    seed: u64,
) -> Result<SessionArms> {
    let capacity = frontier_tokens
        .checked_add(continuation_tokens)
        .and_then(|value| value.checked_add(1))
        .context("whole-model capacity overflow")?;
    let allocated_before_sessions = ctx.current_allocated_size();
    if frontier_tokens == 0 {
        let serial = fresh_gdn_replay_sessions_with_capacity(ctx, model, BATCH, capacity)?;
        let static_batch = fresh_gdn_replay_sessions_with_capacity(ctx, model, BATCH, capacity)?;
        return Ok(SessionArms {
            serial,
            static_batch,
            prefill_ms: 0.0,
            restore_ms: 0.0,
            snapshot_bytes: 0,
            allocation_delta_bytes: ctx
                .current_allocated_size()
                .saturating_sub(allocated_before_sessions),
            session_allocation_bytes: 0,
            scratch_upper_bytes: 0,
        });
    }

    ensure!(prefill_chunk > 0, "--prefill-chunk must be nonzero");
    let prefix = (0..frontier_tokens)
        .map(|position| {
            token_for(
                seed.wrapping_add(0x517c_c1b7),
                0,
                position,
                model.arch.vocab_size as usize,
            )
        })
        .collect::<Vec<_>>();
    let before_source = ctx.current_allocated_size();
    let mut source =
        MetalSession::fresh(ctx, model, capacity).context("frontier source session")?;
    let session_allocation_bytes = ctx
        .current_allocated_size()
        .checked_sub(before_source)
        .filter(|&bytes| bytes > 0)
        .context("frontier source allocation delta is invalid")?;
    let block_size = u32::try_from(prefill_chunk.min(frontier_tokens))
        .context("frontier prefill chunk does not fit u32")?;
    let scratch_plan = plan_prefill_scratch_with_matrix_max_pos_configured(
        model,
        block_size,
        frontier_tokens,
        PrefillScratchConfig::default(),
    )
    .context("plan frontier prefill scratch")?;
    let scratch_upper_bytes = scratch_plan
        .priced_upper_bound(|bytes| Ok(ctx.shared_buffer_size_and_align(bytes)?.size))
        .context("price frontier prefill scratch")?;
    let remaining_session_upper_bytes = session_allocation_bytes
        .checked_mul((BATCH * 2 - 1) as u64)
        .context("frontier session upper bound overflow")?;
    let prefill_required_bytes = scratch_upper_bytes
        .checked_add(remaining_session_upper_bytes)
        .context("frontier prefill admission overflow")?;
    let prefill_admission = evaluate_metal_memory_admission(
        prefill_required_bytes,
        DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    ensure!(
        prefill_admission.admitted,
        "frontier setup admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        prefill_admission.reason.as_str(),
        prefill_admission.required_bytes,
        prefill_admission.working_set_headroom_bytes,
        prefill_admission.signals.process_limit_remaining_bytes,
    );
    let mut scratch = fresh_prefill_scratch_for_prompt(
        ctx,
        model,
        prefill_chunk.min(frontier_tokens),
        frontier_tokens,
    )
    .context("frontier prefill scratch")?;
    let prefill_started = Instant::now();
    prefill_tokens_with_multi_hidden(forward, &prefix, 0, &mut source, &mut scratch, &[], None)
        .context("construct long-frontier state")?;
    let prefill_ms = prefill_started.elapsed().as_secs_f64() * 1e3;
    ensure_frontier(
        std::slice::from_ref(&source),
        frontier_tokens,
        "snapshot source",
    )?;
    let identity = source.snapshot_identity(0x6465_6e73_655f_6238, 0x6c6f_6e67_5f67_6174);
    let snapshot = source
        .snapshot(identity.clone(), prefix, None)
        .context("capture long-frontier snapshot")?;
    let snapshot_bytes = snapshot.n_bytes();
    drop(scratch);

    let restore_started = Instant::now();
    let before_first_session = ctx.current_allocated_size();
    let mut sessions = Vec::with_capacity(BATCH * 2);
    let mut first = MetalSession::fresh(ctx, model, capacity).context("first restored session")?;
    first
        .restore_from(&snapshot, &identity)
        .context("restore first long-frontier session")?;
    let one_session_bytes = ctx
        .current_allocated_size()
        .checked_sub(before_first_session)
        .filter(|&bytes| bytes > 0)
        .context("long-frontier session allocation delta is invalid")?;
    ensure_restored_source_exact(
        &source,
        &first,
        frontier_tokens,
        model.arch.n_kv_heads as usize * model.arch.attn_head_dim as usize,
    )?;
    drop(source);
    sessions.push(first);
    let remaining_session_bytes = one_session_bytes
        .checked_mul((BATCH * 2 - 1) as u64)
        .context("long-frontier remaining session estimate overflow")?;
    let admission = evaluate_metal_memory_admission(
        remaining_session_bytes,
        DYNAMIC_RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    ensure!(
        admission.admitted,
        "long-frontier B=8 admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
        admission.reason.as_str(),
        admission.required_bytes,
        admission.working_set_headroom_bytes,
        admission.signals.process_limit_remaining_bytes,
    );
    for slot in 1..BATCH * 2 {
        crate::shutdown::checkpoint()?;
        let mut session =
            MetalSession::fresh(ctx, model, capacity).context("restored session allocation")?;
        session
            .restore_from(&snapshot, &identity)
            .with_context(|| format!("restore long-frontier session {slot}"))?;
        sessions.push(session);
    }
    let restore_ms = restore_started.elapsed().as_secs_f64() * 1e3;
    let static_batch = sessions.split_off(BATCH);
    Ok(SessionArms {
        serial: sessions,
        static_batch,
        prefill_ms,
        restore_ms,
        snapshot_bytes,
        allocation_delta_bytes: ctx
            .current_allocated_size()
            .saturating_sub(allocated_before_sessions),
        session_allocation_bytes,
        scratch_upper_bytes,
    })
}

pub fn run(args: DecodeDenseWholeBatchArgs) -> Result<()> {
    ensure!(args.steps > 0, "--steps must be nonzero");
    ensure!(
        args.frontier_tokens <= 16 * 1024,
        "--frontier-tokens is bounded to 16384 by this gate"
    );
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
    let canonical_long_gate = !args.no_check
        && args.frontier_tokens == 16 * 1024
        && args.warmup_steps == 1
        && args.steps == 4
        && args.ramp_ms == 500
        && args.prefill_chunk == 1024
        && args.seed == 1;
    let evidence_scope = if canonical_long_gate {
        "restored_16k_checked"
    } else if args.no_check {
        "diagnostic_no_check"
    } else if args.frontier_tokens == 0 {
        "serialized_short_frontier"
    } else {
        "diagnostic_restored_frontier"
    };

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
    let arms = sessions_at_frontier(
        &ctx,
        &model,
        &forward,
        args.frontier_tokens,
        total_steps,
        args.prefill_chunk,
        args.seed,
    )?;
    let mut serial = arms.serial;
    let mut static_batch = arms.static_batch;
    ensure!(
        serial.iter().chain(&static_batch).all(|session| {
            session.kv_k.iter().all(|kv| kv.dtype == GgmlType::F16)
                && session.kv_v.iter().all(|kv| kv.dtype == GgmlType::F16)
        }),
        "whole-model probe currently requires F16 KV"
    );

    println!(
        "[decode-dense-whole-batch] model={} batch={} warmup_steps={} measured_steps={} ramp_ms={} ramp_repetitions={} seed={} prefill_chunk={} build_commit={} build_dirty={} build_source_state={} qwen_environment_names={} baseline=production_concurrent_gdn candidate=layer_major_static_b8 frontier_tokens={} frontier_source={} frontier_prefill_ms={:.3} frontier_restore_setup_ms={:.3} snapshot_bytes={} snapshot_identity_scope=in_process_abi_only source_restore_exact={} frontier_session_allocation_bytes={} frontier_scratch_upper_bytes={} attention=causal_multikey lm_head=batched fused_post_norm={} checks_enforced={} evidence_scope={} independent_queue_gate=pending post_setup_metal_allocation_delta_bytes={}",
        args.model.display(),
        BATCH,
        args.warmup_steps,
        args.steps,
        args.ramp_ms,
        ramp_repetitions,
        args.seed,
        args.prefill_chunk,
        env!("QWEN_BUILD_COMMIT"),
        env!("QWEN_BUILD_DIRTY"),
        env!("QWEN_BUILD_SOURCE_STATE"),
        qwen_environment_names(),
        args.frontier_tokens,
        if args.frontier_tokens == 0 {
            "fresh"
        } else {
            "packed_prefill_snapshot"
        },
        arms.prefill_ms,
        arms.restore_ms,
        arms.snapshot_bytes,
        args.frontier_tokens > 0,
        arms.session_allocation_bytes,
        arms.scratch_upper_bytes,
        fused_post_norm,
        !args.no_check,
        evidence_scope,
        arms.allocation_delta_bytes,
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
    for step_index in 0..total_steps {
        crate::shutdown::checkpoint()?;
        let position = args
            .frontier_tokens
            .checked_add(step_index)
            .context("whole-model position overflow")?;
        let position_u32 =
            u32::try_from(position).context("whole-model position does not fit u32")?;
        let tokens = tokens_for(args.seed, position, model.arch.vocab_size as usize);
        let (serial_step, static_step, order) = if position.is_multiple_of(2) {
            (
                run_serial_step(&forward, &mut serial, &tokens, position_u32)?,
                run_static_step(&mut executor, &mut static_batch, &tokens, position_u32)?,
                "serial_static",
            )
        } else {
            let static_step =
                run_static_step(&mut executor, &mut static_batch, &tokens, position_u32)?;
            let serial_step = run_serial_step(&forward, &mut serial, &tokens, position_u32)?;
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
        let phase = if step_index < args.warmup_steps {
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
    let used_positions = args
        .frontier_tokens
        .checked_add(total_steps)
        .context("whole-model audited position overflow")?;
    let state = audit_persistent_state(
        &serial,
        &static_batch,
        args.frontier_tokens,
        used_positions,
        kv_dim,
        !args.no_check,
    )?;
    let serial_median_wall = median(&serial_wall);
    let static_median_wall = median(&static_wall);
    let serial_median_gpu = median(&serial_gpu);
    let static_median_gpu = median(&static_gpu);
    println!(
        "summary\tbatch={}\tfrontier_tokens={}\tchecks_enforced={}\tevidence_scope={}\tindependent_queue_gate=pending\tserial_median_wall_ms={serial_median_wall:.4}\tstatic_median_wall_ms={static_median_wall:.4}\tpaired_median_wall_speedup={:.4}\tserial_aggregate_tps={:.3}\tstatic_aggregate_tps={:.3}\tserial_median_gpu_sum_ms={serial_median_gpu:.4}\tstatic_median_gpu_ms={static_median_gpu:.4}\tgpu_speedup={:.4}\tworst_min_cos_logits={:.9}\tworst_max_abs_logits={:.6}\tworst_max_relative_rms_logits={:.9}\tworst_min_cos_x={:.9}\tworst_max_abs_x={:.6}\tworst_max_relative_rms_x={:.9}\tmin_cos_state={:.9}\tmax_abs_state={:.6}\tmin_cos_conv={:.9}\tmax_abs_conv={:.6}\tmin_cos_kv_continuation={:.9}\tmax_abs_kv_continuation={:.6}",
        BATCH,
        args.frontier_tokens,
        !args.no_check,
        evidence_scope,
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
