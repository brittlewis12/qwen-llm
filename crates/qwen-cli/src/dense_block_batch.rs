use super::{
    GdnLayerReplayScratch, collect_gdn_layers, cosine_max_abs, encode_gdn_layer_baseline,
    encode_gdn_layer_replay, env_flag_enabled, f32_rms_delta, fresh_gdn_replay_sessions,
    read_f32_tensor,
};
use anyhow::{Context, Result, anyhow, ensure};
use clap::Parser;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
use qwen_llm::gguf::GgufFile;
use qwen_llm::loader::Model;
use qwen_llm::metal::{
    BlitEncoder, KernelEncoder, MetalContext, MetalTensor, encode_add_inplace_f32,
    encode_ffn_swiglu_q4_K_f32, encode_silu_mul_f32,
};
use qwen_llm::metal_forward::{
    MetalBlock, MetalForward, MetalModel, MetalSession, encode_mat_mat_dispatch,
    encode_mat_vec_dispatch,
};
use qwen_llm::model::ArchKind;
use qwen_llm::tensor::GgmlType;
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
pub struct DecodeDenseBlockBatchArgs {
    /// Path to a dense Qwen GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Independent static batch sizes to measure.
    #[arg(long, value_delimiter = ',', default_value = "2,4,6,8")]
    batches: Vec<usize>,
    /// Optional absolute GDN block index; defaults to the first GDN block.
    #[arg(long)]
    block: Option<usize>,
    /// Timed repetitions after warmup.
    #[arg(long, default_value = "7")]
    iters: usize,
    /// Untimed warmup repetitions.
    #[arg(long, default_value = "2")]
    warmup: usize,
    /// High-occupancy device ramp before the first timed batch.
    #[arg(long, default_value = "500")]
    ramp_ms: u64,
    /// Skip the baseline/candidate state comparison.
    #[arg(long)]
    no_check: bool,
}

struct DenseFfnBatchScratch {
    h: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    inner: MetalTensor,
    out: MetalTensor,
}

impl DenseFfnBatchScratch {
    fn new(ctx: &MetalContext, max_batch: usize, hidden: usize, ffn: usize) -> Result<Self> {
        let hidden_elements = max_batch
            .checked_mul(hidden)
            .context("dense static-batch hidden scratch overflow")?;
        let ffn_elements = max_batch
            .checked_mul(ffn)
            .context("dense static-batch FFN scratch overflow")?;
        Ok(Self {
            h: MetalTensor::zeros_f32(ctx, vec![hidden_elements as u64])?,
            gate: MetalTensor::zeros_f32(ctx, vec![ffn_elements as u64])?,
            up: MetalTensor::zeros_f32(ctx, vec![ffn_elements as u64])?,
            inner: MetalTensor::zeros_f32(ctx, vec![ffn_elements as u64])?,
            out: MetalTensor::zeros_f32(ctx, vec![hidden_elements as u64])?,
        })
    }
}

fn encode_dense_block_baseline(
    ctx: &MetalContext,
    forward: &MetalForward<'_>,
    encoder: &KernelEncoder,
    block: &MetalBlock,
    gdn_index: usize,
    sessions: &mut [MetalSession],
) -> Result<()> {
    let MetalBlock::Gdn(gdn) = block else {
        return Err(anyhow!("dense static-batch baseline requires a GDN block"));
    };
    encode_gdn_layer_baseline(ctx, forward, encoder, gdn, gdn_index, sessions)?;
    let hidden = forward.model.arch.hidden_size as usize;
    let ffn = forward.model.arch.intermediate_size as usize;
    for session in sessions {
        if gdn.ffn_gate.dtype == GgmlType::Q4_K && gdn.ffn_up.dtype == GgmlType::Q4_K {
            encode_ffn_swiglu_q4_K_f32(
                ctx,
                encoder,
                &gdn.ffn_gate,
                &gdn.ffn_up,
                &session.h,
                &session.ffn_inner,
                hidden,
                ffn,
            )?;
        } else {
            encode_mat_vec_dispatch(
                ctx,
                encoder,
                &gdn.ffn_gate,
                &session.h,
                &session.ffn_gate,
                hidden,
                ffn,
            )?;
            encode_mat_vec_dispatch(
                ctx,
                encoder,
                &gdn.ffn_up,
                &session.h,
                &session.ffn_up,
                hidden,
                ffn,
            )?;
            encode_silu_mul_f32(
                ctx,
                encoder,
                &session.ffn_gate,
                &session.ffn_up,
                &session.ffn_inner,
            )?;
        }
        encode_mat_vec_dispatch(
            ctx,
            encoder,
            &gdn.ffn_down,
            &session.ffn_inner,
            &session.ffn_out,
            ffn,
            hidden,
        )?;
        encode_add_inplace_f32(ctx, encoder, &session.x, &session.ffn_out)?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_dense_block_batch(
    ctx: &MetalContext,
    forward: &MetalForward<'_>,
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    block: &MetalBlock,
    gdn_index: usize,
    sessions: &mut [MetalSession],
    gdn_scratch: &GdnLayerReplayScratch,
    ffn_scratch: &DenseFfnBatchScratch,
    hidden: usize,
    conv_dim: usize,
    value_dim: usize,
    ffn: usize,
) -> Result<()> {
    let MetalBlock::Gdn(gdn) = block else {
        return Err(anyhow!("dense static-batch candidate requires a GDN block"));
    };
    let batch = sessions.len();
    encode_gdn_layer_replay(
        ctx,
        forward,
        command,
        gdn,
        gdn_index,
        sessions,
        gdn_scratch,
        hidden,
        conv_dim,
        value_dim,
    )?;

    let hidden_row_bytes = u64::try_from(
        hidden
            .checked_mul(std::mem::size_of::<f32>())
            .context("dense static-batch hidden row overflow")?,
    )?;
    let blit = BlitEncoder::begin(command);
    for (slot, session) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &session.h.buffer,
            session.h.offset,
            &ffn_scratch.h.buffer,
            ffn_scratch.h.offset + slot as u64 * hidden_row_bytes,
            hidden_row_bytes,
        );
    }
    blit.end();

    let h = ffn_scratch
        .h
        .view_subrange(0, vec![(batch * hidden) as u64]);
    let gate = ffn_scratch
        .gate
        .view_subrange(0, vec![(batch * ffn) as u64]);
    let up = ffn_scratch.up.view_subrange(0, vec![(batch * ffn) as u64]);
    let inner = ffn_scratch
        .inner
        .view_subrange(0, vec![(batch * ffn) as u64]);
    let out = ffn_scratch
        .out
        .view_subrange(0, vec![(batch * hidden) as u64]);
    let encoder = KernelEncoder::begin(command);
    encode_mat_mat_dispatch(ctx, &encoder, &gdn.ffn_gate, &h, &gate, hidden, ffn, batch)?;
    encode_mat_mat_dispatch(ctx, &encoder, &gdn.ffn_up, &h, &up, hidden, ffn, batch)?;
    encode_silu_mul_f32(ctx, &encoder, &gate, &up, &inner)?;
    encode_mat_mat_dispatch(
        ctx,
        &encoder,
        &gdn.ffn_down,
        &inner,
        &out,
        ffn,
        hidden,
        batch,
    )?;
    for (slot, session) in sessions.iter().enumerate() {
        let out_row = ffn_scratch
            .out
            .view_subrange((slot * hidden) as u64, vec![hidden as u64]);
        encode_add_inplace_f32(ctx, &encoder, &session.x, &out_row)?;
    }
    encoder.end();
    Ok(())
}

fn wait_success(
    command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    label: &str,
) -> Result<()> {
    command.commit();
    command.waitUntilCompleted();
    ensure!(
        command.error().is_none(),
        "{label} command failed: {:?}",
        command.error()
    );
    Ok(())
}

fn write_pattern(
    tensor: &MetalTensor,
    slot: usize,
    salt: usize,
    bias: f32,
    scale: f32,
) -> Result<()> {
    ensure!(
        tensor.offset.is_multiple_of(4),
        "unaligned F32 probe tensor"
    );
    let elements = usize::try_from(tensor.n_elements())?;
    let offset = usize::try_from(tensor.offset / 4)?;
    let ptr = tensor.buffer.contents().as_ptr().cast::<f32>();
    ensure!(
        !ptr.is_null(),
        "dense static-batch tensor is not CPU visible"
    );
    for index in 0..elements {
        let mixed = index
            .wrapping_mul(131)
            .wrapping_add(slot.wrapping_mul(7_919))
            .wrapping_add(salt.wrapping_mul(104_729));
        let centered = (mixed % 251) as f32 - 125.0;
        unsafe { ptr.add(offset + index).write(bias + centered * scale) };
    }
    Ok(())
}

fn reset_probe_sessions(sessions: &[MetalSession], gdn_index: usize) -> Result<()> {
    for (slot, session) in sessions.iter().enumerate() {
        write_pattern(&session.x, slot, 1, 0.03125 + slot as f32 * 0.0005, 1e-5)?;
        write_pattern(&session.gdn_conv[gdn_index], slot, 2, 0.0, 1e-6)?;
        write_pattern(&session.gdn_state[gdn_index], slot, 3, 0.0, 1e-7)?;
    }
    Ok(())
}

struct ArmStats {
    avg_wall_ms: f64,
    avg_gpu_ms: f64,
    p50_gpu_ms: f64,
    p90_gpu_ms: f64,
}

#[derive(Default)]
struct ArmSamples {
    wall_ms: Vec<f64>,
    gpu_ms: Vec<f64>,
}

impl ArmSamples {
    fn append(&mut self, mut other: Self) {
        self.wall_ms.append(&mut other.wall_ms);
        self.gpu_ms.append(&mut other.gpu_ms);
    }

    fn summarize(mut self) -> ArmStats {
        self.gpu_ms.sort_by(f64::total_cmp);
        ArmStats {
            avg_wall_ms: self.wall_ms.iter().sum::<f64>() / self.wall_ms.len() as f64,
            avg_gpu_ms: self.gpu_ms.iter().sum::<f64>() / self.gpu_ms.len() as f64,
            p50_gpu_ms: percentile(&self.gpu_ms, 0.5),
            p90_gpu_ms: percentile(&self.gpu_ms, 0.9),
        }
    }
}

fn percentile(sorted: &[f64], percentile: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * percentile).round() as usize;
    sorted[index]
}

fn time_with_reset(
    ctx: &MetalContext,
    warmup: usize,
    iters: usize,
    sessions: &mut [MetalSession],
    gdn_index: usize,
    mut encode: impl FnMut(
        &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        &mut [MetalSession],
    ) -> Result<()>,
) -> Result<ArmSamples> {
    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    for repetition in 0..warmup + iters {
        reset_probe_sessions(sessions, gdn_index)?;
        let started = Instant::now();
        let command = ctx
            .queue
            .commandBuffer()
            .context("dense static-batch timed command")?;
        encode(&command, sessions)?;
        command.commit();
        command.waitUntilCompleted();
        ensure!(
            command.error().is_none(),
            "dense static-batch timed command failed: {:?}",
            command.error()
        );
        let wall_ms = started.elapsed().as_secs_f64() * 1e3;
        let gpu_start = command.GPUStartTime();
        let gpu_end = command.GPUEndTime();
        ensure!(
            gpu_start.is_finite() && gpu_end.is_finite() && gpu_start > 0.0 && gpu_end > gpu_start,
            "invalid dense static-batch GPU interval: start={gpu_start} end={gpu_end}"
        );
        if repetition >= warmup {
            wall_samples.push(wall_ms);
            gpu_samples.push((gpu_end - gpu_start) * 1e3);
        }
    }
    Ok(ArmSamples {
        wall_ms: wall_samples,
        gpu_ms: gpu_samples,
    })
}

fn ensure_finite(label: &str, values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|value| value.is_finite()),
        "{label} contains a non-finite value"
    );
    Ok(())
}

fn ramp_device(
    ctx: &MetalContext,
    ramp_ms: u64,
    sessions: &mut [MetalSession],
    gdn_index: usize,
    mut encode: impl FnMut(
        &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        &mut [MetalSession],
    ) -> Result<()>,
) -> Result<usize> {
    let target = std::time::Duration::from_millis(ramp_ms);
    let started = Instant::now();
    let mut repetitions = 0usize;
    while repetitions == 0 || started.elapsed() < target {
        crate::shutdown::checkpoint()?;
        reset_probe_sessions(sessions, gdn_index)?;
        let command = ctx
            .queue
            .commandBuffer()
            .context("dense static-batch ramp command")?;
        encode(&command, sessions)?;
        wait_success(&command, "dense static-batch ramp")?;
        repetitions += 1;
    }
    Ok(repetitions)
}

pub fn run(args: DecodeDenseBlockBatchArgs) -> Result<()> {
    ensure!(args.iters > 0, "--iters must be nonzero");
    ensure!(!args.batches.is_empty(), "--batches must not be empty");
    ensure!(
        args.batches.iter().all(|batch| (1..=16).contains(batch)),
        "--batches values must be in 1..=16"
    );
    let mut batches = args.batches;
    batches.sort_unstable();
    batches.dedup();
    let max_batch = *batches.last().expect("validated nonempty batches");
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
            "decode-dense-block-batch does not support {flag}"
        );
    }

    let ctx = MetalContext::new().context("create dense static-batch Metal context")?;
    let gguf = GgufFile::open(&args.model)
        .with_context(|| format!("open dense static-batch model {}", args.model.display()))?;
    let bound = Model::from_gguf(&gguf).context("bind dense static-batch model")?;
    let model = MetalModel::load(&ctx, &gguf, &bound).context("load dense static-batch model")?;
    ensure!(
        model.arch.kind == ArchKind::Dense,
        "decode-dense-block-batch requires a dense Qwen model"
    );
    let layers = collect_gdn_layers(&model);
    let layer = if let Some(block) = args.block {
        *layers
            .iter()
            .find(|layer| layer.block_i == block)
            .ok_or_else(|| anyhow!("block {block} is not a GDN block"))?
    } else {
        *layers.first().context("dense model has no GDN block")?
    };
    let block = &model.blocks[layer.block_i];
    let hidden = model.arch.hidden_size as usize;
    let n_v = model.arch.gdn_n_v_heads as usize;
    let n_k = model.arch.gdn_n_k_heads as usize;
    let head_dim = model.arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let value_dim = n_v * head_dim;
    let ffn = model.arch.intermediate_size as usize;
    let forward = MetalForward::new(&ctx, &model);
    let gdn_scratch = GdnLayerReplayScratch::new(&ctx, max_batch, hidden, conv_dim, value_dim)?;
    let ffn_scratch = DenseFfnBatchScratch::new(&ctx, max_batch, hidden, ffn)?;

    if !args.no_check {
        for &batch in &batches {
            let mut baseline = fresh_gdn_replay_sessions(&ctx, &model, batch)?;
            let mut candidate = fresh_gdn_replay_sessions(&ctx, &model, batch)?;
            reset_probe_sessions(&baseline, layer.gdn_i)?;
            reset_probe_sessions(&candidate, layer.gdn_i)?;

            let command = ctx
                .queue
                .commandBuffer()
                .context("dense static-batch baseline command")?;
            let encoder = KernelEncoder::begin(&command);
            encode_dense_block_baseline(
                &ctx,
                &forward,
                &encoder,
                block,
                layer.gdn_i,
                &mut baseline,
            )?;
            encoder.end();
            wait_success(&command, "dense static-batch baseline")?;

            let command = ctx
                .queue
                .commandBuffer()
                .context("dense static-batch candidate command")?;
            encode_dense_block_batch(
                &ctx,
                &forward,
                &command,
                block,
                layer.gdn_i,
                &mut candidate,
                &gdn_scratch,
                &ffn_scratch,
                hidden,
                conv_dim,
                value_dim,
                ffn,
            )?;
            wait_success(&command, "dense static-batch candidate")?;

            let mut min_cos_x = 1.0f64;
            let mut max_abs_x = 0.0f32;
            let mut max_relative_rms_x = 0.0f64;
            let mut min_cos_state = 1.0f64;
            let mut max_abs_state = 0.0f32;
            let mut min_cos_conv = 1.0f64;
            let mut max_abs_conv = 0.0f32;
            for slot in 0..batch {
                let baseline_x = read_f32_tensor(&baseline[slot].x);
                let candidate_x = read_f32_tensor(&candidate[slot].x);
                let baseline_state = read_f32_tensor(&baseline[slot].gdn_state[layer.gdn_i]);
                let candidate_state = read_f32_tensor(&candidate[slot].gdn_state[layer.gdn_i]);
                let baseline_conv = read_f32_tensor(&baseline[slot].gdn_conv[layer.gdn_i]);
                let candidate_conv = read_f32_tensor(&candidate[slot].gdn_conv[layer.gdn_i]);
                ensure_finite("baseline x", &baseline_x)?;
                ensure_finite("candidate x", &candidate_x)?;
                ensure_finite("baseline GDN state", &baseline_state)?;
                ensure_finite("candidate GDN state", &candidate_state)?;
                ensure_finite("baseline GDN convolution", &baseline_conv)?;
                ensure_finite("candidate GDN convolution", &candidate_conv)?;
                let (cos_x, abs_x) = cosine_max_abs(&baseline_x, &candidate_x);
                let baseline_rms = (baseline_x
                    .iter()
                    .map(|value| (*value as f64).powi(2))
                    .sum::<f64>()
                    / baseline_x.len() as f64)
                    .sqrt();
                let relative_rms_x = if baseline_rms > 0.0 {
                    f32_rms_delta(&baseline_x, &candidate_x) / baseline_rms
                } else {
                    0.0
                };
                let (cos_state, abs_state) = cosine_max_abs(&baseline_state, &candidate_state);
                let (cos_conv, abs_conv) = cosine_max_abs(&baseline_conv, &candidate_conv);
                min_cos_x = min_cos_x.min(cos_x);
                max_abs_x = max_abs_x.max(abs_x);
                max_relative_rms_x = max_relative_rms_x.max(relative_rms_x);
                min_cos_state = min_cos_state.min(cos_state);
                max_abs_state = max_abs_state.max(abs_state);
                min_cos_conv = min_cos_conv.min(cos_conv);
                max_abs_conv = max_abs_conv.max(abs_conv);
            }
            println!(
                "check\tblock={}\tbatch={batch}\tmin_cos_x={min_cos_x:.9}\tmax_abs_x={max_abs_x:.6}\tmax_relative_rms_x={max_relative_rms_x:.9}\tmin_cos_state={min_cos_state:.9}\tmax_abs_state={max_abs_state:.6}\tmin_cos_conv={min_cos_conv:.9}\tmax_abs_conv={max_abs_conv:.6}",
                layer.block_i,
            );
            ensure!(
                min_cos_x >= 0.99999
                    && max_abs_x <= 0.2
                    && max_relative_rms_x <= 1e-3
                    && min_cos_state >= 0.999
                    && max_abs_state <= 0.01
                    && min_cos_conv >= 0.999
                    && max_abs_conv <= 0.01,
                "dense static-batch correctness gate failed at batch {batch}"
            );
        }
    }

    let mut ramp_sessions = fresh_gdn_replay_sessions(&ctx, &model, max_batch)?;
    let ramp_repetitions = ramp_device(
        &ctx,
        args.ramp_ms,
        &mut ramp_sessions,
        layer.gdn_i,
        |command, sessions| {
            encode_dense_block_batch(
                &ctx,
                &forward,
                command,
                block,
                layer.gdn_i,
                sessions,
                &gdn_scratch,
                &ffn_scratch,
                hidden,
                conv_dim,
                value_dim,
                ffn,
            )
        },
    )?;
    drop(ramp_sessions);

    println!(
        "[decode-dense-block-batch] model={} block={} gdn_i={} batches={} ramp_ms={} ramp_repetitions={} warmup={} iters={} batched_projections=6",
        args.model.display(),
        layer.block_i,
        layer.gdn_i,
        batches
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(","),
        args.ramp_ms,
        ramp_repetitions,
        args.warmup,
        args.iters,
    );
    println!(
        "mode\tbatch\torder\tavg_wall_ms\tavg_gpu_ms\tp50_gpu_ms\tp90_gpu_ms\tavg_gpu_ms_per_slot\taggregate_speedup\tsaving_pct"
    );
    for (ordinal, batch) in batches.into_iter().enumerate() {
        let mut baseline = fresh_gdn_replay_sessions(&ctx, &model, batch)?;
        let mut candidate = fresh_gdn_replay_sessions(&ctx, &model, batch)?;
        let baseline_timing = |sessions: &mut [MetalSession]| {
            time_with_reset(
                &ctx,
                args.warmup,
                args.iters,
                sessions,
                layer.gdn_i,
                |command, sessions| {
                    let encoder = KernelEncoder::begin(command);
                    encode_dense_block_baseline(
                        &ctx,
                        &forward,
                        &encoder,
                        block,
                        layer.gdn_i,
                        sessions,
                    )?;
                    encoder.end();
                    Ok(())
                },
            )
        };
        let candidate_timing = |sessions: &mut [MetalSession]| {
            time_with_reset(
                &ctx,
                args.warmup,
                args.iters,
                sessions,
                layer.gdn_i,
                |command, sessions| {
                    encode_dense_block_batch(
                        &ctx,
                        &forward,
                        command,
                        block,
                        layer.gdn_i,
                        sessions,
                        &gdn_scratch,
                        &ffn_scratch,
                        hidden,
                        conv_dim,
                        value_dim,
                        ffn,
                    )
                },
            )
        };
        let mut baseline_samples = ArmSamples::default();
        let mut candidate_samples = ArmSamples::default();
        let order = if ordinal.is_multiple_of(2) {
            baseline_samples.append(baseline_timing(&mut baseline)?);
            candidate_samples.append(candidate_timing(&mut candidate)?);
            candidate_samples.append(candidate_timing(&mut candidate)?);
            baseline_samples.append(baseline_timing(&mut baseline)?);
            "ABBA"
        } else {
            candidate_samples.append(candidate_timing(&mut candidate)?);
            baseline_samples.append(baseline_timing(&mut baseline)?);
            baseline_samples.append(baseline_timing(&mut baseline)?);
            candidate_samples.append(candidate_timing(&mut candidate)?);
            "BAAB"
        };
        let baseline_stats = baseline_samples.summarize();
        let candidate_stats = candidate_samples.summarize();
        let speedup = baseline_stats.avg_gpu_ms / candidate_stats.avg_gpu_ms;
        let saving_pct = (1.0 - candidate_stats.avg_gpu_ms / baseline_stats.avg_gpu_ms) * 100.0;
        println!(
            "baseline_serial\t{batch}\t{order}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t1.0000\t0.0",
            baseline_stats.avg_wall_ms,
            baseline_stats.avg_gpu_ms,
            baseline_stats.p50_gpu_ms,
            baseline_stats.p90_gpu_ms,
            baseline_stats.avg_gpu_ms / batch as f64,
        );
        println!(
            "static_layer_batch\t{batch}\t{order}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{speedup:.4}\t{saving_pct:.1}",
            candidate_stats.avg_wall_ms,
            candidate_stats.avg_gpu_ms,
            candidate_stats.p50_gpu_ms,
            candidate_stats.p90_gpu_ms,
            candidate_stats.avg_gpu_ms / batch as f64,
        );
    }
    Ok(())
}
