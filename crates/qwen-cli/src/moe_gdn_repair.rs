use super::{
    GdnLayerReplayScratch, collect_gdn_layers, cosine_max_abs, encode_gdn_layer_replay,
    f32_rms_delta, fresh_gdn_replay_sessions_with_capacity, fresh_prefill_scratch_for_prompt,
    read_f32_tensor,
};
use anyhow::{Context, Result, ensure};
use clap::Parser;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal::{
        BlitEncoder, KernelEncoder, MetalContext, MetalTensor, encode_argmax_f32,
        encode_get_rows_f32, encode_mat_vec_q6_k_batch_f32,
        encode_moe_swiglu_q4_K_f32_packed_slots, encode_rms_norm_mul_f32,
        evaluate_metal_memory_admission,
    },
    metal_dflash::prefill_tokens_with_multi_hidden,
    metal_forward::{
        MetalBlock, MetalForward, MetalModel, MetalSession, RMS_EPS, SessionSnapshot,
        encode_mat_mat_dispatch,
    },
    model::ArchKind,
};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::time::Instant;

const BATCH: usize = 16;
const SESSION_ALLOWANCE_BYTES: u64 = 6 * 1024 * 1024 * 1024;
const RESERVE_BYTES: u64 = 2 * 1024 * 1024 * 1024;
const SNAPSHOT_MODEL_ID: u64 = 0x6d6f_655f_7265_7061;
const SNAPSHOT_TOKENIZER_ID: u64 = 0x6972_5f62_3136_0001;

#[derive(Parser, Debug)]
pub struct DecodeMoeGdnRepairArgs {
    #[arg(short = 'm', long)]
    model: PathBuf,
    #[arg(long, default_value = "1024")]
    frontier_tokens: usize,
    #[arg(long, default_value = "1024")]
    prefill_chunk: usize,
    #[arg(long, default_value = "2")]
    warmup_steps: usize,
    #[arg(long, default_value = "64")]
    steps: usize,
    #[arg(long, default_value = "1")]
    seed: u64,
}

struct Scratch {
    ids: MetalTensor,
    rows: MetalTensor,
    logits: MetalTensor,
    argmax: MetalTensor,
    moe_topk_idx: MetalTensor,
    moe_inner: MetalTensor,
    gdn: GdnLayerReplayScratch,
}

impl Scratch {
    fn new(ctx: &MetalContext, model: &MetalModel) -> Result<Self> {
        let h = model.arch.hidden_size as usize;
        let n_v = model.arch.gdn_n_v_heads as usize;
        let n_k = model.arch.gdn_n_k_heads as usize;
        let head_dim = model.arch.gdn_head_dim as usize;
        let conv_dim = (2 * n_k + n_v) * head_dim;
        let v_dim = n_v * head_dim;
        let topk = model.arch.expert_used_count.min(model.arch.expert_count) as usize;
        let f_exp = model.arch.expert_feed_forward_length as usize;
        Ok(Self {
            ids: MetalTensor::zeros_i32(ctx, vec![BATCH as u64])?,
            rows: MetalTensor::zeros_f32(ctx, vec![(BATCH * h) as u64])?,
            logits: MetalTensor::zeros_f32(
                ctx,
                vec![(BATCH * model.arch.vocab_size as usize) as u64],
            )?,
            argmax: MetalTensor::zeros_i32(ctx, vec![BATCH as u64])?,
            moe_topk_idx: MetalTensor::zeros_i32(ctx, vec![(BATCH * topk) as u64])?,
            moe_inner: MetalTensor::zeros_f32(ctx, vec![(BATCH * topk * f_exp) as u64])?,
            gdn: GdnLayerReplayScratch::new(ctx, BATCH, h, conv_dim, v_dim)?,
        })
    }

    fn write_ids(&self, ids: &[i32]) -> Result<()> {
        ensure!(ids.len() == BATCH, "repair input width drift");
        let ptr = self.ids.buffer.contents().as_ptr().cast::<i32>();
        ensure!(!ptr.is_null(), "repair IDs are not CPU-visible");
        unsafe {
            std::ptr::copy_nonoverlapping(
                ids.as_ptr(),
                ptr.add((self.ids.offset / 4) as usize),
                BATCH,
            );
        }
        Ok(())
    }

    fn read_argmax(&self) -> Result<Vec<i32>> {
        let ptr = self.argmax.buffer.contents().as_ptr().cast::<i32>();
        ensure!(!ptr.is_null(), "repair argmax is not CPU-visible");
        let mut ids = vec![0i32; BATCH];
        unsafe {
            std::ptr::copy_nonoverlapping(
                ptr.add((self.argmax.offset / 4) as usize),
                ids.as_mut_ptr(),
                BATCH,
            );
        }
        ensure!(ids.iter().all(|&id| id >= 0), "repair argmax reported NaN");
        Ok(ids)
    }
}

struct Step {
    wall_ms: f64,
    gpu_ms: f64,
    ids: Vec<i32>,
}

struct SessionSetup {
    serial: Vec<MetalSession>,
    candidate: Vec<MetalSession>,
    prefill_ms: f64,
    restore_ms: f64,
    snapshot_bytes: u64,
    prefix: Vec<i32>,
}

fn token_for(seed: u64, slot: usize, position: usize, vocab: usize) -> i32 {
    let mixed = seed
        .wrapping_add((slot as u64 + 1).wrapping_mul(7_919))
        .wrapping_add((position as u64 + 1).wrapping_mul(104_729));
    (mixed % vocab as u64) as i32
}

fn ensure_frontier(sessions: &[MetalSession], position: usize) -> Result<()> {
    ensure!(sessions.len() == BATCH, "repair session width drift");
    for (slot, session) in sessions.iter().enumerate() {
        ensure!(
            session.kv_n_pos.iter().all(|&actual| actual == position),
            "repair lane {slot} frontier drift: {:?}",
            session.kv_n_pos
        );
    }
    Ok(())
}

fn sessions_at_frontier(
    ctx: &MetalContext,
    model: &MetalModel,
    forward: &MetalForward<'_>,
    frontier: usize,
    capacity: usize,
    chunk: usize,
    seed: u64,
) -> Result<SessionSetup> {
    let prefix = (0..frontier)
        .map(|position| {
            token_for(
                seed.wrapping_add(0x517c_c1b7),
                0,
                position,
                model.arch.vocab_size as usize,
            )
        })
        .collect::<Vec<_>>();
    let mut source = MetalSession::fresh(ctx, model, capacity)?;
    let mut scratch = fresh_prefill_scratch_for_prompt(ctx, model, chunk, frontier)?;
    let started = Instant::now();
    prefill_tokens_with_multi_hidden(forward, &prefix, 0, &mut source, &mut scratch, &[], None)?;
    let prefill_ms = started.elapsed().as_secs_f64() * 1e3;
    let identity = source.snapshot_identity(SNAPSHOT_MODEL_ID, SNAPSHOT_TOKENIZER_ID);
    let snapshot = source.snapshot(identity.clone(), prefix.clone(), None)?;
    let snapshot_bytes = snapshot.n_bytes();
    drop(scratch);
    drop(source);

    let admission = evaluate_metal_memory_admission(
        SESSION_ALLOWANCE_BYTES,
        RESERVE_BYTES,
        ctx.memory_signals(),
        true,
    );
    ensure!(
        admission.admitted,
        "repair admission denied: reason={} required={:?}",
        admission.reason.as_str(),
        admission.required_bytes
    );
    let started = Instant::now();
    let mut sessions = fresh_gdn_replay_sessions_with_capacity(ctx, model, BATCH * 2, capacity)?;
    for (slot, session) in sessions.iter_mut().enumerate() {
        session
            .restore_from(&snapshot, &identity)
            .with_context(|| format!("restore repair lane {slot}"))?;
    }
    let restore_ms = started.elapsed().as_secs_f64() * 1e3;
    let candidate = sessions.split_off(BATCH);
    Ok(SessionSetup {
        serial: sessions,
        candidate,
        prefill_ms,
        restore_ms,
        snapshot_bytes,
        prefix,
    })
}

fn serial_step(
    forward: &MetalForward<'_>,
    sessions: &mut [MetalSession],
    ids: &[i32],
    position: u32,
) -> Result<Step> {
    let started = Instant::now();
    ensure_frontier(sessions, position as usize)?;
    let mut output = Vec::with_capacity(BATCH);
    let mut gpu_ms = 0.0;
    for slot in 0..BATCH {
        let (id, profile) =
            forward.single_token_argmax_profiled(ids[slot], position, &mut sessions[slot])?;
        output.push(id);
        gpu_ms += profile.gpu_kernel_ms;
    }
    Ok(Step {
        wall_ms: started.elapsed().as_secs_f64() * 1e3,
        gpu_ms,
        ids: output,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_production_blocks(
    ctx: &MetalContext,
    forward: &MetalForward<'_>,
    model: &MetalModel,
    command: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
    sessions: &mut [MetalSession],
    scratch: &Scratch,
    position: u32,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    let gdn_layers = collect_gdn_layers(model);
    for (block_i, block) in model.blocks.iter().enumerate() {
        match block {
            MetalBlock::Gdn(gdn) => {
                let gdn_i = gdn_layers
                    .iter()
                    .find(|layer| layer.block_i == block_i)
                    .context("repair GDN index missing")?
                    .gdn_i;
                encode_gdn_layer_replay(
                    ctx,
                    forward,
                    command,
                    gdn,
                    gdn_i,
                    sessions,
                    &scratch.gdn,
                    h,
                    conv_dim,
                    v_dim,
                )?;
            }
            MetalBlock::Attn(_) => {
                let encoder = KernelEncoder::begin(command);
                for session in sessions.iter_mut() {
                    forward.encode_moe_mixer_prep_by_index(&encoder, block_i, position, session)?;
                }
                encoder.end();
            }
        }
        if packed_gateup_enabled() {
            encode_packed_gateup_tail(
                forward, model, command, block, block_i, sessions, scratch, h,
            )?;
        } else {
            for session in sessions.iter_mut() {
                forward
                    .encode_moe_ffn_after_mixer_production_by_index(command, block_i, session)?;
            }
        }
    }
    Ok(())
}

fn encode_packed_gateup_tail(
    forward: &MetalForward<'_>,
    model: &MetalModel,
    command: &objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
    block: &MetalBlock,
    block_i: usize,
    sessions: &mut [MetalSession],
    scratch: &Scratch,
    h: usize,
) -> Result<()> {
    let moe = match block {
        MetalBlock::Gdn(block) => block.ffn_moe.as_ref(),
        MetalBlock::Attn(block) => block.ffn_moe.as_ref(),
    }
    .context("packed gate/up requires MoE weights")?;
    ensure!(
        moe.gate_exps.dtype == qwen_llm::tensor::GgmlType::Q4_K
            && moe.up_exps.dtype == qwen_llm::tensor::GgmlType::Q4_K,
        "packed gate/up requires Q4_K/Q4_K expert banks"
    );
    let topk = model.arch.expert_used_count.min(model.arch.expert_count) as usize;
    let f_exp = model.arch.expert_feed_forward_length as usize;
    let n_expert = model.arch.expert_count as usize;
    let hidden_bytes = (h * std::mem::size_of::<f32>()) as u64;
    let idx_bytes = (topk * std::mem::size_of::<i32>()) as u64;
    let inner_bytes = (topk * f_exp * std::mem::size_of::<f32>()) as u64;

    let encoder = KernelEncoder::begin(command);
    for session in sessions.iter_mut() {
        forward.encode_moe_route_prepare_by_index(&encoder, block_i, session)?;
    }
    encoder.end();

    let blit = BlitEncoder::begin(command);
    for (slot, session) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &session.h.buffer,
            session.h.offset,
            &scratch.rows.buffer,
            scratch.rows.offset + slot as u64 * hidden_bytes,
            hidden_bytes,
        );
        blit.copy_buffer(
            &session.moe_topk_idx.buffer,
            session.moe_topk_idx.offset,
            &scratch.moe_topk_idx.buffer,
            scratch.moe_topk_idx.offset + slot as u64 * idx_bytes,
            idx_bytes,
        );
    }
    blit.end();

    let encoder = KernelEncoder::begin_concurrent(command);
    encode_moe_swiglu_q4_K_f32_packed_slots(
        forward.ctx,
        &encoder,
        &moe.gate_exps,
        &moe.up_exps,
        &scratch.rows,
        &scratch.moe_topk_idx,
        &scratch.moe_inner,
        h,
        f_exp,
        n_expert,
        topk,
        BATCH,
    )?;
    let shared_inner_fused = sessions
        .iter_mut()
        .map(|session| forward.encode_moe_shared_gate_up_by_index(&encoder, block_i, session))
        .collect::<Result<Vec<_>, _>>()?;
    encoder.end();

    let blit = BlitEncoder::begin(command);
    for (slot, session) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &scratch.moe_inner.buffer,
            scratch.moe_inner.offset + slot as u64 * inner_bytes,
            &session.moe_inner.buffer,
            session.moe_inner.offset,
            inner_bytes,
        );
    }
    blit.end();

    for (session, shared_inner_fused) in sessions.iter_mut().zip(shared_inner_fused) {
        forward.encode_moe_ffn_after_external_routed_inner_by_index(
            command,
            block_i,
            session,
            shared_inner_fused,
        )?;
    }
    Ok(())
}

fn candidate_step(
    ctx: &MetalContext,
    forward: &MetalForward<'_>,
    model: &MetalModel,
    sessions: &mut [MetalSession],
    scratch: &Scratch,
    ids: &[i32],
    position: u32,
) -> Result<Step> {
    let started = Instant::now();
    ensure_frontier(sessions, position as usize)?;
    scratch.write_ids(ids)?;
    let h = model.arch.hidden_size as usize;
    let vocab = model.arch.vocab_size as usize;
    let n_v = model.arch.gdn_n_v_heads as usize;
    let n_k = model.arch.gdn_n_k_heads as usize;
    let head_dim = model.arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let command = ctx.queue.commandBuffer().context("repair command buffer")?;

    let encoder = KernelEncoder::begin(&command);
    encode_get_rows_f32(
        ctx,
        &encoder,
        &model.token_embd,
        &scratch.ids,
        &scratch.rows,
        BATCH,
        h,
    )?;
    encoder.end();
    let row_bytes = (h * std::mem::size_of::<f32>()) as u64;
    let blit = BlitEncoder::begin(&command);
    for (slot, session) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &scratch.rows.buffer,
            scratch.rows.offset + slot as u64 * row_bytes,
            &session.x.buffer,
            session.x.offset,
            row_bytes,
        );
    }
    blit.end();
    encode_production_blocks(
        ctx, forward, model, &command, sessions, scratch, position, h, conv_dim, v_dim,
    )?;
    let encoder = KernelEncoder::begin(&command);
    for session in sessions.iter() {
        encode_rms_norm_mul_f32(
            ctx,
            &encoder,
            &session.x,
            &model.output_norm,
            &session.h,
            RMS_EPS,
        )?;
    }
    encoder.end();
    let batched_head = batched_head_enabled();
    let blit = BlitEncoder::begin(&command);
    for (slot, session) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &session.h.buffer,
            session.h.offset,
            &scratch.rows.buffer,
            scratch.rows.offset + slot as u64 * row_bytes,
            row_bytes,
        );
    }
    blit.end();
    let encoder = KernelEncoder::begin(&command);
    if batched_head {
        encode_mat_mat_dispatch(
            ctx,
            &encoder,
            &model.lm_head,
            &scratch.rows,
            &scratch.logits,
            h,
            vocab,
            BATCH,
        )?;
    } else {
        encode_mat_vec_q6_k_batch_f32(
            ctx,
            &encoder,
            &model.lm_head,
            &scratch.rows,
            &scratch.logits,
            h,
            vocab,
            BATCH,
        )?;
    }
    encode_argmax_f32(
        ctx,
        &encoder,
        &scratch.logits,
        &scratch.argmax,
        BATCH,
        vocab,
    )?;
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    ensure!(
        command.status() == MTLCommandBufferStatus::Completed && command.error().is_none(),
        "repair command failed: {:?}",
        command.error()
    );
    let ids = scratch.read_argmax()?;
    Ok(Step {
        wall_ms: started.elapsed().as_secs_f64() * 1e3,
        gpu_ms: (command.GPUEndTime() - command.GPUStartTime()) * 1e3,
        ids,
    })
}

fn batched_head_enabled() -> bool {
    !matches!(
        std::env::var("QWEN_BENCH_MOE_BATCHED_HEAD").as_deref(),
        Ok("0") | Ok("false") | Ok("FALSE") | Ok("no") | Ok("NO")
    )
}

fn packed_gateup_enabled() -> bool {
    matches!(
        std::env::var("QWEN_BENCH_MOE_PACKED_GATEUP").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE") | Ok("yes") | Ok("YES")
    )
}

fn relative_rms(reference: &[f32], candidate: &[f32]) -> f64 {
    let rms = (reference
        .iter()
        .map(|value| (*value as f64).powi(2))
        .sum::<f64>()
        / reference.len() as f64)
        .sqrt();
    f32_rms_delta(reference, candidate) / rms.max(f64::MIN_POSITIVE)
}

#[derive(Clone, Copy, Debug)]
struct NumericEvidence {
    logit_cos: f64,
    logit_rel_rms: f64,
    x_cos: f64,
    x_rel_rms: f64,
    logits_bitwise_equal: bool,
    x_bitwise_equal: bool,
}

impl NumericEvidence {
    fn exact() -> Self {
        Self {
            logit_cos: 1.0,
            logit_rel_rms: 0.0,
            x_cos: 1.0,
            x_rel_rms: 0.0,
            logits_bitwise_equal: true,
            x_bitwise_equal: true,
        }
    }

    fn include(&mut self, observed: Self) {
        self.logit_cos = self.logit_cos.min(observed.logit_cos);
        self.logit_rel_rms = self.logit_rel_rms.max(observed.logit_rel_rms);
        self.x_cos = self.x_cos.min(observed.x_cos);
        self.x_rel_rms = self.x_rel_rms.max(observed.x_rel_rms);
        self.logits_bitwise_equal &= observed.logits_bitwise_equal;
        self.x_bitwise_equal &= observed.x_bitwise_equal;
    }
}

fn ensure_finite(values: &[f32], label: &str, slot: usize) -> Result<()> {
    if let Some((index, value)) = values
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        anyhow::bail!("non-finite {label} at slot {slot}, index {index}: {value:?}");
    }
    Ok(())
}

fn f32_bits_equal(left: &[f32], right: &[f32]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .all(|(left, right)| left.to_bits() == right.to_bits())
}

fn numeric_evidence(
    serial: &[MetalSession],
    candidate: &[MetalSession],
    logits: &MetalTensor,
    vocab: usize,
) -> Result<NumericEvidence> {
    let flat = read_f32_tensor(logits);
    ensure!(
        flat.len() == BATCH * vocab,
        "candidate logit shape drift: got {}, expected {}",
        flat.len(),
        BATCH * vocab
    );
    let mut evidence = NumericEvidence::exact();
    for slot in 0..BATCH {
        let serial_logits = read_f32_tensor(&serial[slot].logits);
        let candidate_logits = &flat[slot * vocab..(slot + 1) * vocab];
        ensure_finite(&serial_logits, "serial logits", slot)?;
        ensure_finite(candidate_logits, "candidate logits", slot)?;
        evidence.logit_cos = evidence
            .logit_cos
            .min(cosine_max_abs(&serial_logits, candidate_logits).0);
        evidence.logit_rel_rms = evidence
            .logit_rel_rms
            .max(relative_rms(&serial_logits, candidate_logits));
        evidence.logits_bitwise_equal &= f32_bits_equal(&serial_logits, candidate_logits);
        let serial_x = read_f32_tensor(&serial[slot].x);
        let candidate_x = read_f32_tensor(&candidate[slot].x);
        ensure_finite(&serial_x, "serial residual", slot)?;
        ensure_finite(&candidate_x, "candidate residual", slot)?;
        evidence.x_cos = evidence
            .x_cos
            .min(cosine_max_abs(&serial_x, &candidate_x).0);
        evidence.x_rel_rms = evidence
            .x_rel_rms
            .max(relative_rms(&serial_x, &candidate_x));
        evidence.x_bitwise_equal &= f32_bits_equal(&serial_x, &candidate_x);
    }
    Ok(evidence)
}

fn hash_ids(ids: &[i32]) -> String {
    let mut hasher = Sha256::new();
    hasher.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        hasher.update(id.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
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

fn ensure_finite_f32_arena(bytes: &[u8], label: &str, slot: usize) -> Result<()> {
    ensure!(
        bytes.len().is_multiple_of(std::mem::size_of::<f32>()),
        "{label} byte length is not F32-aligned"
    );
    for (index, value) in bytes
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().expect("four-byte chunk")))
        .enumerate()
    {
        ensure!(
            value.is_finite(),
            "non-finite {label} at slot {slot}, index {index}: {value:?}"
        );
    }
    Ok(())
}

fn snapshot_mismatch(left: &SessionSnapshot, right: &SessionSnapshot) -> Option<&'static str> {
    if left.identity != right.identity {
        Some("identity")
    } else if left.prefix_tokens != right.prefix_tokens {
        Some("prefix_tokens")
    } else if left.pending_token != right.pending_token {
        Some("pending_token")
    } else if left.kv_n_pos != right.kv_n_pos {
        Some("kv_n_pos")
    } else if left.kv_k_arena != right.kv_k_arena {
        Some("kv_k_arena")
    } else if left.kv_v_arena != right.kv_v_arena {
        Some("kv_v_arena")
    } else if left.gdn_conv_arena != right.gdn_conv_arena {
        Some("gdn_conv_arena")
    } else if left.gdn_state_arena != right.gdn_state_arena {
        Some("gdn_state_arena")
    } else {
        match (&left.final_logits, &right.final_logits) {
            (Some(left), Some(right)) if f32_bits_equal(left, right) => None,
            (None, None) => None,
            _ => Some("final_logits"),
        }
    }
}

fn final_causal_state_evidence(
    serial: &[MetalSession],
    candidate: &[MetalSession],
    serial_histories: &[Vec<i32>],
    candidate_histories: &[Vec<i32>],
    serial_pending: &[i32],
    candidate_pending: &[i32],
    candidate_logits: &MetalTensor,
    vocab: usize,
) -> Result<Option<String>> {
    ensure!(
        serial.len() == BATCH
            && candidate.len() == BATCH
            && serial_histories.len() == BATCH
            && candidate_histories.len() == BATCH
            && serial_pending.len() == BATCH
            && candidate_pending.len() == BATCH,
        "final causal-state width drift"
    );
    let candidate_logits = read_f32_tensor(candidate_logits);
    ensure!(
        candidate_logits.len() == BATCH * vocab,
        "final candidate logit shape drift"
    );
    for slot in 0..BATCH {
        let serial_logits = read_f32_tensor(&serial[slot].logits);
        let candidate_slot_logits = &candidate_logits[slot * vocab..(slot + 1) * vocab];
        ensure_finite(&serial_logits, "final serial logits", slot)?;
        ensure_finite(candidate_slot_logits, "final candidate logits", slot)?;
        let identity = serial[slot].snapshot_identity(SNAPSHOT_MODEL_ID, SNAPSHOT_TOKENIZER_ID);
        let mut left = serial[slot].snapshot(
            identity.clone(),
            serial_histories[slot].clone(),
            Some(serial_logits),
        )?;
        let mut right = candidate[slot].snapshot(
            identity,
            candidate_histories[slot].clone(),
            Some(candidate_slot_logits.to_vec()),
        )?;
        left.pending_token = Some(serial_pending[slot]);
        right.pending_token = Some(candidate_pending[slot]);
        ensure_finite_f32_arena(&left.gdn_conv_arena, "serial GDN convolution", slot)?;
        ensure_finite_f32_arena(&right.gdn_conv_arena, "candidate GDN convolution", slot)?;
        ensure_finite_f32_arena(&left.gdn_state_arena, "serial GDN state", slot)?;
        ensure_finite_f32_arena(&right.gdn_state_arena, "candidate GDN state", slot)?;
        if let Some(section) = snapshot_mismatch(&left, &right) {
            return Ok(Some(format!("slot:{slot},section:{section}")));
        }
    }
    Ok(None)
}

fn median(values: &[f64]) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    (values[values.len() / 2 - 1] + values[values.len() / 2]) * 0.5
}

pub fn run(args: DecodeMoeGdnRepairArgs) -> Result<()> {
    ensure!(
        args.steps > 0 && args.steps.is_multiple_of(2),
        "--steps must be positive and even"
    );
    let total_steps = args.warmup_steps + args.steps;
    let capacity = args.frontier_tokens + total_steps + 1;
    let ctx = MetalContext::new()?;
    let gguf = GgufFile::open(&args.model)?;
    let bound = Model::from_gguf(&gguf)?;
    let model = MetalModel::load(&ctx, &gguf, &bound)?;
    ensure!(
        model.arch.kind == ArchKind::Moe,
        "repair probe requires Qwen MoE"
    );
    ensure!(
        model.arch.hidden_size == 2_048
            && model.arch.vocab_size == 248_320
            && model.blocks.len() == 40,
        "repair probe is scoped to the 35B-A3B anchor"
    );
    let forward = MetalForward::new(&ctx, &model);
    let setup = sessions_at_frontier(
        &ctx,
        &model,
        &forward,
        args.frontier_tokens,
        capacity,
        args.prefill_chunk,
        args.seed,
    )?;
    let mut serial = setup.serial;
    let mut candidate = setup.candidate;
    let scratch = Scratch::new(&ctx, &model)?;
    let mut serial_ids = (0..BATCH)
        .map(|slot| {
            token_for(
                args.seed,
                slot,
                args.frontier_tokens,
                model.arch.vocab_size as usize,
            )
        })
        .collect::<Vec<_>>();
    let mut candidate_ids = serial_ids.clone();
    let mut serial_histories = vec![setup.prefix.clone(); BATCH];
    let mut candidate_histories = serial_histories.clone();
    let mut serial_selected_ids = Vec::with_capacity(total_steps * BATCH);
    let mut candidate_selected_ids = Vec::with_capacity(total_steps * BATCH);
    let mut serial_wall = Vec::new();
    let mut candidate_wall = Vec::new();
    let mut serial_gpu = Vec::new();
    let mut candidate_gpu = Vec::new();
    let mut first_divergence = None;
    let mut worst = NumericEvidence::exact();
    let mut same_history_steps = 0usize;
    let exact_selector =
        std::env::var("QWEN_BENCH_GDN_REPLAY_EXACT").unwrap_or_else(|_| "none".into());
    let batched_head = batched_head_enabled();
    let strict_exact = exact_selector == "all" && !batched_head;
    ensure!(
        !packed_gateup_enabled() || strict_exact,
        "packed gate/up requires exact=all and the exact Q6 head"
    );
    println!(
        "[decode-moe-gdn-repair] exact={} batch={BATCH} batched_head={} packed_gateup={} strict_exact={} frontier={} prefill_ms={:.3} restore_ms={:.3} snapshot_bytes={} steps={} warmup={} build_commit={} build_dirty={} build_source_state={} qwen_environment_names={} generated_id_hash_scope=warmup_plus_measured_step_major_lane_major_i32le",
        exact_selector,
        batched_head,
        packed_gateup_enabled(),
        strict_exact,
        args.frontier_tokens,
        setup.prefill_ms,
        setup.restore_ms,
        setup.snapshot_bytes,
        args.steps,
        args.warmup_steps,
        env!("QWEN_BUILD_COMMIT"),
        env!("QWEN_BUILD_DIRTY"),
        env!("QWEN_BUILD_SOURCE_STATE"),
        qwen_environment_names(),
    );
    for step in 0..total_steps {
        let position = (args.frontier_tokens + step) as u32;
        let same_history = first_divergence.is_none() && serial_ids == candidate_ids;
        let (left, right) = if step.is_multiple_of(2) {
            (
                serial_step(&forward, &mut serial, &serial_ids, position)?,
                candidate_step(
                    &ctx,
                    &forward,
                    &model,
                    &mut candidate,
                    &scratch,
                    &candidate_ids,
                    position,
                )?,
            )
        } else {
            let right = candidate_step(
                &ctx,
                &forward,
                &model,
                &mut candidate,
                &scratch,
                &candidate_ids,
                position,
            )?;
            let left = serial_step(&forward, &mut serial, &serial_ids, position)?;
            (left, right)
        };
        for slot in 0..BATCH {
            serial_histories[slot].push(serial_ids[slot]);
            candidate_histories[slot].push(candidate_ids[slot]);
        }
        serial_selected_ids.extend_from_slice(&left.ids);
        candidate_selected_ids.extend_from_slice(&right.ids);
        let mismatch = left
            .ids
            .iter()
            .zip(&right.ids)
            .enumerate()
            .find(|(_, (a, b))| a != b)
            .map(|(slot, (&a, &b))| (step, slot, a, b));
        if same_history {
            let observed = numeric_evidence(
                &serial,
                &candidate,
                &scratch.logits,
                model.arch.vocab_size as usize,
            )?;
            worst.include(observed);
            same_history_steps += 1;
        }
        if first_divergence.is_none() {
            first_divergence = mismatch;
        }
        if step >= args.warmup_steps {
            serial_wall.push(left.wall_ms);
            candidate_wall.push(right.wall_ms);
            serial_gpu.push(left.gpu_ms);
            candidate_gpu.push(right.gpu_ms);
        }
        serial_ids = left.ids;
        candidate_ids = right.ids;
    }
    let serial_wall = median(&serial_wall);
    let candidate_wall = median(&candidate_wall);
    let first = first_divergence
        .map(|(step, slot, a, b)| format!("step:{step},slot:{slot},serial:{a},candidate:{b}"))
        .unwrap_or_else(|| "none".into());
    let serial_id_sha256 = hash_ids(&serial_selected_ids);
    let candidate_id_sha256 = hash_ids(&candidate_selected_ids);
    let final_state_mismatch = if first_divergence.is_none() {
        final_causal_state_evidence(
            &serial,
            &candidate,
            &serial_histories,
            &candidate_histories,
            &serial_ids,
            &candidate_ids,
            &scratch.logits,
            model.arch.vocab_size as usize,
        )?
    } else {
        Some("skipped_after_history_divergence".into())
    };
    let final_state = final_state_mismatch.as_deref().unwrap_or("exact");
    let generated_ids_exact = serial_selected_ids == candidate_selected_ids;
    println!(
        "summary\tserial_wall_ms={serial_wall:.4}\tcandidate_wall_ms={candidate_wall:.4}\tspeedup={:.4}\tserial_tps={:.3}\tcandidate_tps={:.3}\tserial_gpu_ms={:.4}\tcandidate_gpu_ms={:.4}\tfirst_divergence={first}\tsame_history_steps={same_history_steps}\tsame_history_finite=true\tlogits_bitwise_exact={}\tx_bitwise_exact={}\tfinal_causal_state={final_state}\tserial_generated_ids_sha256={serial_id_sha256}\tcandidate_generated_ids_sha256={candidate_id_sha256}\tgenerated_ids_exact={}\tmin_cos_logits={:.9}\tmax_rel_rms_logits={:.9}\tmin_cos_x={:.9}\tmax_rel_rms_x={:.9}",
        serial_wall / candidate_wall,
        BATCH as f64 * 1e3 / serial_wall,
        BATCH as f64 * 1e3 / candidate_wall,
        median(&serial_gpu),
        median(&candidate_gpu),
        worst.logits_bitwise_equal,
        worst.x_bitwise_equal,
        generated_ids_exact,
        worst.logit_cos,
        worst.logit_rel_rms,
        worst.x_cos,
        worst.x_rel_rms,
    );
    if strict_exact {
        ensure!(
            first_divergence.is_none(),
            "strict exact ID divergence: {first}"
        );
        ensure!(
            same_history_steps == total_steps,
            "strict exact evidence covered {same_history_steps}/{total_steps} steps"
        );
        ensure!(worst.logits_bitwise_equal, "strict exact logits differed");
        ensure!(worst.x_bitwise_equal, "strict exact residuals differed");
        ensure!(
            generated_ids_exact && serial_id_sha256 == candidate_id_sha256,
            "strict exact generated-ID trace differed"
        );
        ensure!(
            final_state_mismatch.is_none(),
            "strict exact causal state differed: {final_state}"
        );
    }
    Ok(())
}
