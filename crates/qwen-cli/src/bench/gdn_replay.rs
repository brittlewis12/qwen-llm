//! GDN layer/chain replay probes.

use super::*;

pub(crate) struct GdnLayerReplayScratch {
    pub(crate) h_pack: MetalTensor,
    pub(crate) qkv_pack: MetalTensor,
    pub(crate) z_pack: MetalTensor,
    pub(crate) normed_pack: MetalTensor,
    pub(crate) out_pack: MetalTensor,
}

impl GdnLayerReplayScratch {
    pub(crate) fn new(
        ctx: &MetalContext,
        max_tokens: usize,
        h: usize,
        conv_dim: usize,
        v_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            h_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * h) as u64])?,
            qkv_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * conv_dim) as u64])?,
            z_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * v_dim) as u64])?,
            normed_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * v_dim) as u64])?,
            out_pack: MetalTensor::zeros_f32(ctx, vec![(max_tokens * h) as u64])?,
        })
    }
}

pub(crate) fn fill_gdn_replay_inputs(ctx: &MetalContext, sessions: &[MetalSession]) -> Result<()> {
    let cmd = ctx.queue.commandBuffer().context("gdn replay fill cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    for (slot, s) in sessions.iter().enumerate() {
        let v = 0.03125 + (slot as f32) * 0.0009765625;
        encode_fill_f32(ctx, &enc, &s.x, v)?;
        for conv in &s.gdn_conv {
            encode_fill_f32(ctx, &enc, conv, 0.0)?;
        }
        for state in &s.gdn_state {
            encode_fill_f32(ctx, &enc, state, 0.0)?;
        }
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(())
}

pub(crate) fn gdn_replay_beta_projection_fused(
    gb: &qwen_llm::metal_forward::MetalGdnBlock,
) -> bool {
    env_flag_default_on("QWEN_DECODE_GDN_FUSED_BETA_PROJ")
        && gb.beta_proj.dtype == GgmlType::F32
        && !env_flag_enabled("QWEN_DECODE_GDN_NOOP_FRONT")
        && !env_flag_enabled("QWEN_DECODE_GDN_NOOP_BETA")
}

pub(crate) fn gdn_replay_exact_projection(name: &str) -> bool {
    let Ok(value) = std::env::var("QWEN_BENCH_GDN_REPLAY_EXACT") else {
        return false;
    };
    value.split(',').any(|item| {
        let item = item.trim();
        item == "all" || item == name || (item == "front" && matches!(name, "qkv" | "z"))
    })
}

pub(crate) fn read_f32_tensor(t: &MetalTensor) -> Vec<f32> {
    let n = t.n_elements() as usize;
    let mut xs = vec![0.0f32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
    }
    xs
}

pub(crate) fn read_f32_tensor_prefix(t: &MetalTensor, n: usize) -> Vec<f32> {
    let n = n.min(t.n_elements() as usize);
    let mut xs = vec![0.0f32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
    }
    xs
}

pub(crate) fn read_i32_tensor_prefix(t: &MetalTensor, n: usize) -> Vec<i32> {
    let n = n.min(t.n_elements() as usize);
    let mut xs = vec![0i32; n];
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const i32).add((t.offset / 4) as usize);
        std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
    }
    xs
}

pub(crate) fn fmt_i32_csv(xs: &[i32]) -> String {
    xs.iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(",")
}

pub(crate) struct RouteFingerprint {
    pub(crate) idx: Vec<i32>,
    pub(crate) weight: Vec<f32>,
    pub(crate) shared_gate: f32,
    pub(crate) logit_margin: f32,
    pub(crate) logits: Vec<f32>,
}

pub(crate) fn topk_logit_margin(logits: &[f32], topk: usize) -> f32 {
    if topk == 0 || logits.len() <= topk {
        return 0.0;
    }
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked[topk - 1].1 - ranked[topk].1
}

pub(crate) fn read_route_fingerprint(
    s: &MetalSession,
    topk: usize,
    n_expert: usize,
) -> RouteFingerprint {
    let idx = read_i32_tensor_prefix(&s.moe_topk_idx, topk);
    let weight = read_f32_tensor_prefix(&s.moe_topk_weight, topk);
    let shared_gate = read_f32_tensor_prefix(&s.moe_shared_gate, 1)
        .into_iter()
        .next()
        .unwrap_or(0.0);
    let logits = read_f32_tensor_prefix(&s.moe_router_probs, n_expert);
    RouteFingerprint {
        idx,
        weight,
        shared_gate,
        logit_margin: topk_logit_margin(&logits, topk),
        logits,
    }
}

pub(crate) fn f32_max_abs_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub(crate) fn f32_rms_delta(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let ss = a
        .iter()
        .zip(b)
        .take(n)
        .map(|(x, y)| {
            let d = *x as f64 - *y as f64;
            d * d
        })
        .sum::<f64>();
    (ss / n as f64).sqrt()
}

pub(crate) fn route_weight_max_abs(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

pub(crate) fn same_i32_set(a: &[i32], b: &[i32]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut aa = a.to_vec();
    let mut bb = b.to_vec();
    aa.sort_unstable();
    bb.sort_unstable();
    aa == bb
}

pub(crate) fn cosine_max_abs(a: &[f32], b: &[f32]) -> (f64, f32) {
    let max_abs = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    let dot: f64 = a.iter().zip(b).map(|(x, y)| *x as f64 * *y as f64).sum();
    let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
    let cos = if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        1.0
    };
    (cos, max_abs)
}

pub(crate) fn fresh_gdn_replay_sessions(
    ctx: &MetalContext,
    mm: &MetalModel,
    n: usize,
) -> Result<Vec<MetalSession>> {
    fresh_gdn_replay_sessions_with_capacity(ctx, mm, n, 32)
}

pub(crate) fn fresh_gdn_replay_sessions_with_capacity(
    ctx: &MetalContext,
    mm: &MetalModel,
    n: usize,
    kv_capacity: usize,
) -> Result<Vec<MetalSession>> {
    let mut sessions = Vec::with_capacity(n);
    for i in 0..n {
        sessions.push(
            MetalSession::fresh(ctx, mm, kv_capacity)
                .with_context(|| format!("fresh replay session {i}"))?,
        );
    }
    Ok(sessions)
}

pub(crate) fn encode_gdn_layer_baseline(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    enc: &KernelEncoder,
    gb: &qwen_llm::metal_forward::MetalGdnBlock,
    gdn_i: usize,
    sessions: &mut [MetalSession],
) -> Result<()> {
    for s in sessions {
        encode_rms_norm_mul_f32(ctx, enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)?;
        mf.encode_gdn(enc, gb, gdn_i, s)?;
        encode_add_inplace_f32(ctx, enc, &s.x, &s.mixer_out)?;
        encode_rms_norm_mul_f32(ctx, enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)?;
    }
    Ok(())
}

pub(crate) fn encode_gdn_layer_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    gb: &qwen_llm::metal_forward::MetalGdnBlock,
    gdn_i: usize,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    encode_gdn_layer_replay_with_post_norm(
        ctx, mf, cmd, gb, gdn_i, sessions, scratch, h, conv_dim, v_dim, false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_gdn_layer_replay_with_post_norm(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    gb: &qwen_llm::metal_forward::MetalGdnBlock,
    gdn_i: usize,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    fused_post_norm: bool,
) -> Result<()> {
    let tokens = sessions.len();
    let enc = KernelEncoder::begin(cmd);
    for s in sessions.iter() {
        encode_rms_norm_mul_f32(ctx, &enc, &s.x, &gb.attn_norm, &s.h, RMS_EPS)?;
    }
    enc.end();

    let row_bytes = (h * std::mem::size_of::<f32>()) as u64;
    let blit = BlitEncoder::begin(cmd);
    for (tok, s) in sessions.iter().enumerate() {
        blit.copy_buffer(
            &s.h.buffer,
            s.h.offset,
            &scratch.h_pack.buffer,
            scratch.h_pack.offset + tok as u64 * row_bytes,
            row_bytes,
        );
    }
    blit.end();

    let enc = KernelEncoder::begin(cmd);
    let h_pack = scratch.h_pack.view_subrange(0, vec![(tokens * h) as u64]);
    let qkv_pack = scratch
        .qkv_pack
        .view_subrange(0, vec![(tokens * conv_dim) as u64]);
    let z_pack = scratch
        .z_pack
        .view_subrange(0, vec![(tokens * v_dim) as u64]);
    if gdn_replay_exact_projection("qkv") {
        encode_mat_vec_q8_0_batch_f32(
            ctx,
            &enc,
            &gb.in_proj_qkv,
            &h_pack,
            &qkv_pack,
            h,
            conv_dim,
            tokens,
        )?;
    } else {
        encode_mat_mat_dispatch(
            ctx,
            &enc,
            &gb.in_proj_qkv,
            &h_pack,
            &qkv_pack,
            h,
            conv_dim,
            tokens,
        )?;
    }
    if gdn_replay_exact_projection("z") {
        encode_mat_vec_q8_0_batch_f32(
            ctx,
            &enc,
            &gb.in_proj_z,
            &h_pack,
            &z_pack,
            h,
            v_dim,
            tokens,
        )?;
    } else {
        encode_mat_mat_dispatch(ctx, &enc, &gb.in_proj_z, &h_pack, &z_pack, h, v_dim, tokens)?;
    }

    for (tok, s) in sessions.iter_mut().enumerate() {
        if gdn_replay_beta_projection_fused(gb) {
            encode_mat_vec_f32_sigmoid(
                ctx,
                &enc,
                &gb.beta_proj,
                &s.h,
                &s.gdn_beta,
                h,
                s.gdn_beta.n_elements() as usize,
            )?;
        } else {
            encode_mat_vec_dispatch(
                ctx,
                &enc,
                &gb.beta_proj,
                &s.h,
                &s.gdn_b,
                h,
                s.gdn_b.n_elements() as usize,
            )?;
        }
        encode_mat_vec_dispatch(
            ctx,
            &enc,
            &gb.alpha_proj,
            &s.h,
            &s.gdn_a,
            h,
            s.gdn_a.n_elements() as usize,
        )?;
        if !gdn_replay_beta_projection_fused(gb) {
            encode_sigmoid_f32(ctx, &enc, &s.gdn_b, &s.gdn_beta)?;
        }
        encode_gdn_decay_chain_f32(ctx, &enc, &s.gdn_a, &gb.dt_bias, &gb.a_log, &s.gdn_alpha)?;

        let qkv_row = scratch
            .qkv_pack
            .view_subrange((tok * conv_dim) as u64, vec![conv_dim as u64]);
        let z_row = scratch
            .z_pack
            .view_subrange((tok * v_dim) as u64, vec![v_dim as u64]);
        let normed_row = scratch
            .normed_pack
            .view_subrange((tok * v_dim) as u64, vec![v_dim as u64]);
        let alpha = s.gdn_alpha.clone();
        let beta = s.gdn_beta.clone();
        mf.encode_gdn_tail(
            &enc,
            gb,
            gdn_i,
            s,
            &qkv_row,
            &z_row,
            &alpha,
            &beta,
            &normed_row,
        )?;
    }

    let normed_pack = scratch
        .normed_pack
        .view_subrange(0, vec![(tokens * v_dim) as u64]);
    let out_pack = scratch.out_pack.view_subrange(0, vec![(tokens * h) as u64]);
    if gdn_replay_exact_projection("out") {
        encode_mat_vec_q8_0_batch_f32(
            ctx,
            &enc,
            &gb.out_proj,
            &normed_pack,
            &out_pack,
            v_dim,
            h,
            tokens,
        )?;
    } else {
        encode_mat_mat_dispatch(
            ctx,
            &enc,
            &gb.out_proj,
            &normed_pack,
            &out_pack,
            v_dim,
            h,
            tokens,
        )?;
    }

    for (tok, s) in sessions.iter().enumerate() {
        let out_row = scratch
            .out_pack
            .view_subrange((tok * h) as u64, vec![h as u64]);
        if fused_post_norm {
            encode_residual_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.x,
                &out_row,
                &gb.post_attn_norm,
                &s.h,
                RMS_EPS,
            )?;
        } else {
            encode_add_inplace_f32(ctx, &enc, &s.x, &out_row)?;
            encode_rms_norm_mul_f32(ctx, &enc, &s.x, &gb.post_attn_norm, &s.h, RMS_EPS)?;
        }
    }
    enc.end();
    Ok(())
}

#[derive(Clone, Copy)]
pub(crate) struct SelectedGdnLayer<'a> {
    pub(crate) block_i: usize,
    pub(crate) gdn_i: usize,
    pub(crate) gb: &'a qwen_llm::metal_forward::MetalGdnBlock,
}

pub(crate) fn collect_gdn_layers(mm: &MetalModel) -> Vec<SelectedGdnLayer<'_>> {
    mm.blocks
        .iter()
        .enumerate()
        .filter_map(|(block_i, b)| match b {
            MetalBlock::Gdn(gb) => Some((block_i, gb)),
            MetalBlock::Attn(_) => None,
        })
        .enumerate()
        .map(|(gdn_i, (block_i, gb))| SelectedGdnLayer { block_i, gdn_i, gb })
        .collect()
}

pub(crate) fn run_decode_gdn_layer_replay(args: DecodeGdnLayerReplayArgs) -> Result<()> {
    let DecodeGdnLayerReplayArgs {
        model,
        mut tokens,
        block,
        mut gdn_indexes,
        sample_gdn_layers,
        iters,
        warmup,
        no_check,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens.is_empty() || tokens.contains(&0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    if block.is_some() && (!gdn_indexes.is_empty() || sample_gdn_layers) {
        return Err(anyhow!(
            "--block conflicts with --gdn-index and --sample-gdn-layers"
        ));
    }
    if sample_gdn_layers && !gdn_indexes.is_empty() {
        return Err(anyhow!("--sample-gdn-layers conflicts with --gdn-index"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    gdn_indexes.sort_unstable();
    gdn_indexes.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;

    let gdn_layers = collect_gdn_layers(&mm);
    if gdn_layers.is_empty() {
        return Err(anyhow!("model has no GDN blocks"));
    }

    let selected: Vec<_> = if let Some(wanted) = block {
        vec![
            *gdn_layers
                .iter()
                .find(|layer| layer.block_i == wanted)
                .ok_or_else(|| anyhow!("block {wanted} is not a GDN block"))?,
        ]
    } else if !gdn_indexes.is_empty() {
        let mut xs = Vec::with_capacity(gdn_indexes.len());
        for &gdn_i in &gdn_indexes {
            xs.push(*gdn_layers.get(gdn_i).ok_or_else(|| {
                anyhow!(
                    "gdn-index {gdn_i} outside available 0..{}",
                    gdn_layers.len().saturating_sub(1)
                )
            })?);
        }
        xs
    } else if sample_gdn_layers {
        let mut idxs = vec![0, gdn_layers.len() / 2, gdn_layers.len() - 1];
        idxs.sort_unstable();
        idxs.dedup();
        idxs.into_iter().map(|i| gdn_layers[i]).collect()
    } else {
        vec![gdn_layers[0]]
    };

    println!(
        "[decode-gdn-layer-replay] model={} selected={} gdn_layers={} layers={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        selected.len(),
        gdn_layers.len(),
        mm.blocks.len(),
        h,
        conv_dim,
        v_dim,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );
    println!(
        "mode\tblock\tgdn_i\ttokens\tencoders_per_batch\tdispatches_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\tsaving_ms_per_tok\tsaving_pct"
    );

    for layer in selected {
        if !no_check {
            let check_tokens = max_tokens;
            let mut base = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
            let mut replay = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
            fill_gdn_replay_inputs(&ctx, &base)?;
            fill_gdn_replay_inputs(&ctx, &replay)?;
            let scratch = GdnLayerReplayScratch::new(&ctx, check_tokens, h, conv_dim, v_dim)?;

            let cmd = ctx
                .queue
                .commandBuffer()
                .context("gdn replay check baseline cmd")?;
            let enc = KernelEncoder::begin(&cmd);
            encode_gdn_layer_baseline(&ctx, &mf, &enc, layer.gb, layer.gdn_i, &mut base)?;
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();

            let cmd = ctx
                .queue
                .commandBuffer()
                .context("gdn replay check replay cmd")?;
            encode_gdn_layer_replay(
                &ctx,
                &mf,
                &cmd,
                layer.gb,
                layer.gdn_i,
                &mut replay,
                &scratch,
                h,
                conv_dim,
                v_dim,
            )?;
            cmd.commit();
            cmd.waitUntilCompleted();

            let mut min_cos = 1.0f64;
            let mut max_abs_all = 0.0f32;
            let mut worst_slot = 0usize;
            for i in 0..check_tokens {
                let (cos, max_abs) =
                    cosine_max_abs(&read_f32_tensor(&base[i].h), &read_f32_tensor(&replay[i].h));
                if cos < min_cos || max_abs > max_abs_all {
                    worst_slot = i;
                }
                min_cos = min_cos.min(cos);
                max_abs_all = max_abs_all.max(max_abs);
            }
            println!(
                "check\t{}\t{}\t{}\tmin_cos_h={min_cos:.9}\tmax_abs_h={max_abs_all:.6}\tworst_slot={worst_slot}",
                layer.block_i, layer.gdn_i, check_tokens
            );
            if min_cos < 0.999 || max_abs_all > 1e-2 {
                return Err(anyhow!(
                    "GDN layer replay check failed: block={} gdn_i={} min_cos_h={min_cos:.9} max_abs_h={max_abs_all:.6}",
                    layer.block_i,
                    layer.gdn_i
                ));
            }
        }

        let mut baseline_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
        let mut replay_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
        fill_gdn_replay_inputs(&ctx, &baseline_sessions)?;
        fill_gdn_replay_inputs(&ctx, &replay_sessions)?;
        let scratch = GdnLayerReplayScratch::new(&ctx, max_tokens, h, conv_dim, v_dim)?;

        for &n_tokens in &tokens {
            let baseline = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                let enc = KernelEncoder::begin(cmd);
                encode_gdn_layer_baseline(
                    &ctx,
                    &mf,
                    &enc,
                    layer.gb,
                    layer.gdn_i,
                    &mut baseline_sessions[..n_tokens],
                )?;
                enc.end();
                Ok(())
            })?;
            let replay = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
                encode_gdn_layer_replay(
                    &ctx,
                    &mf,
                    cmd,
                    layer.gb,
                    layer.gdn_i,
                    &mut replay_sessions[..n_tokens],
                    &scratch,
                    h,
                    conv_dim,
                    v_dim,
                )
            })?;

            let baseline_per_tok = baseline.avg_gpu_ms / n_tokens as f64;
            let replay_per_tok = replay.avg_gpu_ms / n_tokens as f64;
            let save = baseline_per_tok - replay_per_tok;
            let pct = if baseline_per_tok > 0.0 {
                save / baseline_per_tok * 100.0
            } else {
                0.0
            };
            println!(
                "baseline_seq\t{}\t{}\t{}\t1.00\t15.00\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t0.0",
                layer.block_i,
                layer.gdn_i,
                n_tokens,
                baseline.avg_wall_ms,
                baseline.avg_gpu_ms,
                baseline_per_tok,
                baseline.p50_gpu_ms / n_tokens as f64,
                baseline.p90_gpu_ms / n_tokens as f64,
                baseline.max_gpu_ms / n_tokens as f64
            );
            println!(
                "replay_batched_qkv_z_out\t{}\t{}\t{}\t3.00\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
                layer.block_i,
                layer.gdn_i,
                n_tokens,
                12.0 + 3.0 / n_tokens as f64,
                replay.avg_wall_ms,
                replay.avg_gpu_ms,
                replay_per_tok,
                replay.p50_gpu_ms / n_tokens as f64,
                replay.p90_gpu_ms / n_tokens as f64,
                replay.max_gpu_ms / n_tokens as f64,
                save,
                pct
            );
        }
    }

    Ok(())
}

pub(crate) fn encode_gdn_chain_baseline(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    enc: &KernelEncoder,
    layers: &[SelectedGdnLayer<'_>],
    sessions: &mut [MetalSession],
) -> Result<()> {
    for layer in layers {
        encode_gdn_layer_baseline(ctx, mf, enc, layer.gb, layer.gdn_i, sessions)?;
    }
    Ok(())
}

pub(crate) fn encode_gdn_chain_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    layers: &[SelectedGdnLayer<'_>],
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    for layer in layers {
        encode_gdn_layer_replay(
            ctx,
            mf,
            cmd,
            layer.gb,
            layer.gdn_i,
            sessions,
            scratch,
            h,
            conv_dim,
            v_dim,
        )?;
    }
    Ok(())
}

pub(crate) fn run_decode_gdn_chain_replay(args: DecodeGdnChainReplayArgs) -> Result<()> {
    let DecodeGdnChainReplayArgs {
        model,
        mut tokens,
        start_gdn,
        n_layers,
        iters,
        warmup,
        no_check,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if n_layers == 0 {
        return Err(anyhow!("--layers must be >= 1"));
    }
    if tokens.is_empty() || tokens.contains(&0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;

    let gdn_layers = collect_gdn_layers(&mm);
    let end_gdn = start_gdn
        .checked_add(n_layers)
        .ok_or_else(|| anyhow!("start_gdn + layers overflow"))?;
    if end_gdn > gdn_layers.len() {
        return Err(anyhow!(
            "GDN chain {}..{} outside available 0..{}",
            start_gdn,
            end_gdn,
            gdn_layers.len()
        ));
    }
    let selected = &gdn_layers[start_gdn..end_gdn];
    let block_list = selected
        .iter()
        .map(|layer| layer.block_i.to_string())
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "[decode-gdn-chain-replay] model={} start_gdn={} chain_layers={} blocks={} gdn_layers={} layers={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        start_gdn,
        n_layers,
        block_list,
        gdn_layers.len(),
        mm.blocks.len(),
        h,
        conv_dim,
        v_dim,
        tokens
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        warmup,
        iters
    );

    if !no_check {
        let check_tokens = max_tokens;
        let mut base = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
        let mut replay = fresh_gdn_replay_sessions(&ctx, &mm, check_tokens)?;
        fill_gdn_replay_inputs(&ctx, &base)?;
        fill_gdn_replay_inputs(&ctx, &replay)?;
        let scratch = GdnLayerReplayScratch::new(&ctx, check_tokens, h, conv_dim, v_dim)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("gdn chain check baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_gdn_chain_baseline(&ctx, &mf, &enc, selected, &mut base)?;
        enc.end();
        cmd.commit();
        cmd.waitUntilCompleted();

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("gdn chain check replay cmd")?;
        encode_gdn_chain_replay(
            &ctx,
            &mf,
            &cmd,
            selected,
            &mut replay,
            &scratch,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        cmd.waitUntilCompleted();

        let mut min_cos = 1.0f64;
        let mut max_abs_all = 0.0f32;
        let mut worst_slot = 0usize;
        for i in 0..check_tokens {
            let (cos, max_abs) =
                cosine_max_abs(&read_f32_tensor(&base[i].h), &read_f32_tensor(&replay[i].h));
            if cos < min_cos || max_abs > max_abs_all {
                worst_slot = i;
            }
            min_cos = min_cos.min(cos);
            max_abs_all = max_abs_all.max(max_abs);
        }
        println!(
            "check\ttokens={check_tokens}\tmin_cos_h={min_cos:.9}\tmax_abs_h={max_abs_all:.6}\tworst_slot={worst_slot}"
        );
        if min_cos < 0.999 || max_abs_all > 5e-2 {
            return Err(anyhow!(
                "GDN chain replay check failed: min_cos_h={min_cos:.9} max_abs_h={max_abs_all:.6}"
            ));
        }
    }

    let mut baseline_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
    let mut replay_sessions = fresh_gdn_replay_sessions(&ctx, &mm, max_tokens)?;
    fill_gdn_replay_inputs(&ctx, &baseline_sessions)?;
    fill_gdn_replay_inputs(&ctx, &replay_sessions)?;
    let scratch = GdnLayerReplayScratch::new(&ctx, max_tokens, h, conv_dim, v_dim)?;

    println!(
        "mode\ttokens\tchain_layers\tencoders_per_batch\tdispatches_per_token\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\tsaving_ms_per_tok\tsaving_pct"
    );
    for &n_tokens in &tokens {
        let baseline = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            fill_gdn_replay_inputs(&ctx, &baseline_sessions[..n_tokens])?;
            let enc = KernelEncoder::begin(cmd);
            encode_gdn_chain_baseline(
                &ctx,
                &mf,
                &enc,
                selected,
                &mut baseline_sessions[..n_tokens],
            )?;
            enc.end();
            Ok(())
        })?;
        let replay = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            fill_gdn_replay_inputs(&ctx, &replay_sessions[..n_tokens])?;
            encode_gdn_chain_replay(
                &ctx,
                &mf,
                cmd,
                selected,
                &mut replay_sessions[..n_tokens],
                &scratch,
                h,
                conv_dim,
                v_dim,
            )
        })?;

        let baseline_per_tok = baseline.avg_gpu_ms / n_tokens as f64;
        let replay_per_tok = replay.avg_gpu_ms / n_tokens as f64;
        let save = baseline_per_tok - replay_per_tok;
        let pct = if baseline_per_tok > 0.0 {
            save / baseline_per_tok * 100.0
        } else {
            0.0
        };
        println!(
            "baseline_seq\t{}\t{}\t1.00\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t0.0",
            n_tokens,
            n_layers,
            15.0 * n_layers as f64,
            baseline.avg_wall_ms,
            baseline.avg_gpu_ms,
            baseline_per_tok,
            baseline.p50_gpu_ms / n_tokens as f64,
            baseline.p90_gpu_ms / n_tokens as f64,
            baseline.max_gpu_ms / n_tokens as f64,
        );
        println!(
            "replay_batched_qkv_z_out\t{}\t{}\t{}\t{:.2}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
            n_tokens,
            n_layers,
            3 * n_layers,
            (12.0 + 3.0 / n_tokens as f64) * n_layers as f64,
            replay.avg_wall_ms,
            replay.avg_gpu_ms,
            replay_per_tok,
            replay.p50_gpu_ms / n_tokens as f64,
            replay.p90_gpu_ms / n_tokens as f64,
            replay.max_gpu_ms / n_tokens as f64,
            save,
            pct
        );
    }

    Ok(())
}
