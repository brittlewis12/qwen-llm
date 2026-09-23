//! Dense block-slice replay, trace, and margin probes.

use super::*;

pub(crate) fn encode_block_slice_baseline(
    mf: &MetalForward<'_>,
    enc: &KernelEncoder,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    sessions: &mut [MetalSession],
) -> Result<()> {
    for block_i in start_block..start_block + n_blocks {
        for s in sessions.iter_mut() {
            mf.encode_moe_block_by_index(enc, block_i, position, s)?;
        }
    }
    Ok(())
}

pub(crate) fn encode_block_slice_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    gdn_layers: &[SelectedGdnLayer<'_>],
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    for block_i in start_block..start_block + n_blocks {
        encode_one_block_replay(
            ctx, mf, mm, cmd, block_i, position, sessions, scratch, gdn_layers, h, conv_dim, v_dim,
        )?;
    }
    Ok(())
}

pub(crate) fn encode_one_block_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    block_i: usize,
    position: u32,
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    gdn_layers: &[SelectedGdnLayer<'_>],
    h: usize,
    conv_dim: usize,
    v_dim: usize,
) -> Result<()> {
    match &mm.blocks[block_i] {
        MetalBlock::Gdn(gb) => {
            let gdn_i = gdn_layers
                .iter()
                .find(|layer| layer.block_i == block_i)
                .map(|layer| layer.gdn_i)
                .ok_or_else(|| anyhow!("missing GDN index for block {block_i}"))?;
            encode_gdn_layer_replay(
                ctx, mf, cmd, gb, gdn_i, sessions, scratch, h, conv_dim, v_dim,
            )?;
            let enc = KernelEncoder::begin(cmd);
            for s in sessions.iter_mut() {
                mf.encode_moe_ffn_after_mixer_by_index(&enc, block_i, s)?;
            }
            enc.end();
        }
        MetalBlock::Attn(_) => {
            let enc = KernelEncoder::begin(cmd);
            for s in sessions.iter_mut() {
                mf.encode_moe_block_by_index(&enc, block_i, position, s)?;
            }
            enc.end();
        }
    }
    Ok(())
}

pub(crate) fn run_decode_block_slice_replay(args: DecodeBlockSliceReplayArgs) -> Result<()> {
    let DecodeBlockSliceReplayArgs {
        model,
        mut tokens,
        start_block,
        n_blocks,
        position,
        iters,
        warmup,
        no_check,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
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
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-replay currently requires an MoE model"
        ));
    }
    let end_block = start_block
        .checked_add(n_blocks)
        .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
    if end_block > mm.blocks.len() {
        return Err(anyhow!(
            "block slice {}..{} outside available 0..{}",
            start_block,
            end_block,
            mm.blocks.len()
        ));
    }
    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let gdn_layers = collect_gdn_layers(&mm);
    let n_gdn_in_slice = (start_block..end_block)
        .filter(|&i| matches!(mm.blocks[i], MetalBlock::Gdn(_)))
        .count();
    let n_attn_in_slice = n_blocks - n_gdn_in_slice;
    let kv_capacity = (position as usize)
        .checked_add(32)
        .ok_or_else(|| anyhow!("position + KV slack overflow"))?;

    println!(
        "[decode-block-slice-replay] model={} start_block={} blocks={} gdn_blocks={} attn_blocks={} position={} kv_capacity={} h={} conv_dim={} v_dim={} tokens={} warmup={} iters={}",
        model.display(),
        start_block,
        n_blocks,
        n_gdn_in_slice,
        n_attn_in_slice,
        position,
        kv_capacity,
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
        let mut base =
            fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, check_tokens, kv_capacity)?;
        let mut replay =
            fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, check_tokens, kv_capacity)?;
        fill_gdn_replay_inputs(&ctx, &base)?;
        fill_gdn_replay_inputs(&ctx, &replay)?;
        let scratch = GdnLayerReplayScratch::new(&ctx, check_tokens, h, conv_dim, v_dim)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block slice check baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(&mf, &enc, start_block, n_blocks, position, &mut base)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block slice check replay cmd")?;
        encode_block_slice_replay(
            &ctx,
            &mf,
            &mm,
            &cmd,
            start_block,
            n_blocks,
            position,
            &mut replay,
            &scratch,
            &gdn_layers,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        let mut min_cos = 1.0f64;
        let mut max_abs_all = 0.0f32;
        let mut worst_slot = 0usize;
        for i in 0..check_tokens {
            let (cos, max_abs) =
                cosine_max_abs(&read_f32_tensor(&base[i].x), &read_f32_tensor(&replay[i].x));
            if cos < min_cos || max_abs > max_abs_all {
                worst_slot = i;
            }
            min_cos = min_cos.min(cos);
            max_abs_all = max_abs_all.max(max_abs);
        }
        println!(
            "check\ttokens={check_tokens}\tmin_cos_x={min_cos:.9}\tmax_abs_x={max_abs_all:.6}\tworst_slot={worst_slot}"
        );
        if min_cos < 0.999 || max_abs_all > 5e-2 {
            return Err(anyhow!(
                "block slice replay check failed: min_cos_x={min_cos:.9} max_abs_x={max_abs_all:.6}"
            ));
        }
    }

    let mut baseline_sessions =
        fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, max_tokens, kv_capacity)?;
    let mut replay_sessions =
        fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, max_tokens, kv_capacity)?;
    fill_gdn_replay_inputs(&ctx, &baseline_sessions)?;
    fill_gdn_replay_inputs(&ctx, &replay_sessions)?;
    let scratch = GdnLayerReplayScratch::new(&ctx, max_tokens, h, conv_dim, v_dim)?;

    println!(
        "mode\ttokens\tblocks\tgdn_blocks\tattn_blocks\tavg_wall_ms\tavg_gpu_ms\tavg_gpu_ms_per_tok\tp50_gpu_ms_per_tok\tp90_gpu_ms_per_tok\tmax_gpu_ms_per_tok\tsaving_ms_per_tok\tsaving_pct"
    );
    for &n_tokens in &tokens {
        let baseline = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            let enc = KernelEncoder::begin(cmd);
            encode_block_slice_baseline(
                &mf,
                &enc,
                start_block,
                n_blocks,
                position,
                &mut baseline_sessions[..n_tokens],
            )?;
            enc.end();
            Ok(())
        })?;
        let replay = time_cmd_reps_stats(&ctx, warmup, iters, |cmd| {
            encode_block_slice_replay(
                &ctx,
                &mf,
                &mm,
                cmd,
                start_block,
                n_blocks,
                position,
                &mut replay_sessions[..n_tokens],
                &scratch,
                &gdn_layers,
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
            "baseline_seq\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t0.0000\t0.0",
            n_tokens,
            n_blocks,
            n_gdn_in_slice,
            n_attn_in_slice,
            baseline.avg_wall_ms,
            baseline.avg_gpu_ms,
            baseline_per_tok,
            baseline.p50_gpu_ms / n_tokens as f64,
            baseline.p90_gpu_ms / n_tokens as f64,
            baseline.max_gpu_ms / n_tokens as f64,
        );
        println!(
            "replay_gdn_batched\t{}\t{}\t{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.4}\t{:.1}",
            n_tokens,
            n_blocks,
            n_gdn_in_slice,
            n_attn_in_slice,
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

pub(crate) fn run_decode_block_slice_trace(args: DecodeBlockSliceTraceArgs) -> Result<()> {
    let DecodeBlockSliceTraceArgs {
        model,
        tokens,
        start_block,
        n_blocks,
        position,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-trace currently requires an MoE model"
        ));
    }
    let end_block = start_block
        .checked_add(n_blocks)
        .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
    if end_block > mm.blocks.len() {
        return Err(anyhow!(
            "block slice {}..{} outside available 0..{}",
            start_block,
            end_block,
            mm.blocks.len()
        ));
    }

    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let kv_capacity = (position as usize)
        .checked_add(32)
        .ok_or_else(|| anyhow!("position + KV slack overflow"))?;
    let gdn_layers = collect_gdn_layers(&mm);

    println!(
        "[decode-block-slice-trace] model={} start_block={} blocks={} position={} kv_capacity={} tokens={} topk={} experts={} h={} conv_dim={} v_dim={}",
        model.display(),
        start_block,
        n_blocks,
        position,
        kv_capacity,
        tokens,
        topk,
        n_expert,
        h,
        conv_dim,
        v_dim
    );

    let mut base = fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, tokens, kv_capacity)?;
    let mut replay = fresh_gdn_replay_sessions_with_capacity(&ctx, &mm, tokens, kv_capacity)?;
    fill_gdn_replay_inputs(&ctx, &base)?;
    fill_gdn_replay_inputs(&ctx, &replay)?;
    let scratch = GdnLayerReplayScratch::new(&ctx, tokens, h, conv_dim, v_dim)?;

    println!(
        "block\tkind\tslot\troute_order_equal\troute_set_equal\tbase_idx\treplay_idx\tweight_max_abs\tshared_abs\tbase_margin\treplay_margin\tlogit_max_abs\tlogit_rms\th_cos\th_max_abs\tx_cos\tx_max_abs"
    );

    let mut route_order_mismatches = 0usize;
    let mut route_set_mismatches = 0usize;
    let mut first_order_mismatch: Option<(usize, usize)> = None;
    let mut first_set_mismatch: Option<(usize, usize)> = None;
    let mut min_h_cos = 1.0f64;
    let mut min_x_cos = 1.0f64;
    let mut max_h_abs = 0.0f32;
    let mut max_x_abs = 0.0f32;
    let mut max_logit_abs = 0.0f32;
    let mut max_logit_rms = 0.0f64;
    let mut min_base_margin = f32::INFINITY;
    let mut min_replay_margin = f32::INFINITY;
    let mut replay_margin_lt_1e3 = 0usize;
    let mut replay_margin_lt_5e3 = 0usize;

    for block_i in start_block..end_block {
        let kind = match &mm.blocks[block_i] {
            MetalBlock::Gdn(_) => "gdn",
            MetalBlock::Attn(_) => "attn",
        };

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block trace baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(&mf, &enc, block_i, 1, position, &mut base)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block trace replay cmd")?;
        encode_one_block_replay(
            &ctx,
            &mf,
            &mm,
            &cmd,
            block_i,
            position,
            &mut replay,
            &scratch,
            &gdn_layers,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        for slot in 0..tokens {
            let base_route = read_route_fingerprint(&base[slot], topk, n_expert);
            let replay_route = read_route_fingerprint(&replay[slot], topk, n_expert);
            let route_order_equal = base_route.idx == replay_route.idx;
            let route_set_equal = same_i32_set(&base_route.idx, &replay_route.idx);
            if !route_order_equal {
                route_order_mismatches += 1;
                first_order_mismatch.get_or_insert((block_i, slot));
            }
            if !route_set_equal {
                route_set_mismatches += 1;
                first_set_mismatch.get_or_insert((block_i, slot));
            }
            let weight_max_abs = route_weight_max_abs(&base_route.weight, &replay_route.weight);
            let shared_abs = (base_route.shared_gate - replay_route.shared_gate).abs();
            let logit_max_abs = f32_max_abs_delta(&base_route.logits, &replay_route.logits);
            let logit_rms = f32_rms_delta(&base_route.logits, &replay_route.logits);
            max_logit_abs = max_logit_abs.max(logit_max_abs);
            max_logit_rms = max_logit_rms.max(logit_rms);
            min_base_margin = min_base_margin.min(base_route.logit_margin);
            min_replay_margin = min_replay_margin.min(replay_route.logit_margin);
            if replay_route.logit_margin < 0.001 {
                replay_margin_lt_1e3 += 1;
            }
            if replay_route.logit_margin < 0.005 {
                replay_margin_lt_5e3 += 1;
            }
            let (h_cos, h_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].h),
                &read_f32_tensor(&replay[slot].h),
            );
            let (x_cos, x_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].x),
                &read_f32_tensor(&replay[slot].x),
            );
            min_h_cos = min_h_cos.min(h_cos);
            min_x_cos = min_x_cos.min(x_cos);
            max_h_abs = max_h_abs.max(h_abs);
            max_x_abs = max_x_abs.max(x_abs);
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.9}\t{:.6}\t{:.9}\t{:.6}",
                block_i,
                kind,
                slot,
                route_order_equal,
                route_set_equal,
                fmt_i32_csv(&base_route.idx),
                fmt_i32_csv(&replay_route.idx),
                weight_max_abs,
                shared_abs,
                base_route.logit_margin,
                replay_route.logit_margin,
                logit_max_abs,
                logit_rms,
                h_cos,
                h_abs,
                x_cos,
                x_abs
            );
        }
    }

    let first_order = first_order_mismatch
        .map(|(block, slot)| format!("block={block},slot={slot}"))
        .unwrap_or_else(|| "none".to_string());
    let first_set = first_set_mismatch
        .map(|(block, slot)| format!("block={block},slot={slot}"))
        .unwrap_or_else(|| "none".to_string());
    println!(
        "summary\troute_order_mismatches={}\troute_set_mismatches={}\tfirst_order_mismatch={}\tfirst_set_mismatch={}\tmax_logit_abs={:.6}\tmax_logit_rms={:.6}\tmin_base_margin={:.6}\tmin_replay_margin={:.6}\treplay_margin_lt_1e3={}\treplay_margin_lt_5e3={}\tmin_h_cos={:.9}\tmax_h_abs={:.6}\tmin_x_cos={:.9}\tmax_x_abs={:.6}",
        route_order_mismatches,
        route_set_mismatches,
        first_order,
        first_set,
        max_logit_abs,
        max_logit_rms,
        min_base_margin,
        min_replay_margin,
        replay_margin_lt_1e3,
        replay_margin_lt_5e3,
        min_h_cos,
        max_h_abs,
        min_x_cos,
        max_x_abs
    );

    Ok(())
}

pub(crate) struct BlockSliceTraceSummary {
    pub(crate) route_order_mismatches: usize,
    pub(crate) route_set_mismatches: usize,
    pub(crate) first_set_mismatch: Option<(usize, usize)>,
    pub(crate) max_logit_abs: f32,
    pub(crate) max_logit_rms: f64,
    pub(crate) min_base_margin: f32,
    pub(crate) min_replay_margin: f32,
    pub(crate) replay_margin_lt_1e3: usize,
    pub(crate) replay_margin_lt_5e3: usize,
    pub(crate) min_x_cos: f64,
    pub(crate) max_x_abs: f32,
}

impl Default for BlockSliceTraceSummary {
    fn default() -> Self {
        Self {
            route_order_mismatches: 0,
            route_set_mismatches: 0,
            first_set_mismatch: None,
            max_logit_abs: 0.0,
            max_logit_rms: 0.0,
            min_base_margin: f32::INFINITY,
            min_replay_margin: f32::INFINITY,
            replay_margin_lt_1e3: 0,
            replay_margin_lt_5e3: 0,
            min_x_cos: 1.0,
            max_x_abs: 0.0,
        }
    }
}

pub(crate) fn trace_block_slice_summary(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    tokens: usize,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    topk: usize,
    n_expert: usize,
    gdn_layers: &[SelectedGdnLayer<'_>],
) -> Result<BlockSliceTraceSummary> {
    let kv_capacity = (position as usize)
        .checked_add(32)
        .ok_or_else(|| anyhow!("position + KV slack overflow"))?;
    let mut base = fresh_gdn_replay_sessions_with_capacity(ctx, mm, tokens, kv_capacity)?;
    let mut replay = fresh_gdn_replay_sessions_with_capacity(ctx, mm, tokens, kv_capacity)?;
    fill_gdn_replay_inputs(ctx, &base)?;
    fill_gdn_replay_inputs(ctx, &replay)?;
    let scratch = GdnLayerReplayScratch::new(ctx, tokens, h, conv_dim, v_dim)?;
    let mut out = BlockSliceTraceSummary::default();

    for block_i in start_block..start_block + n_blocks {
        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block margin sweep baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(mf, &enc, block_i, 1, position, &mut base)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("block margin sweep replay cmd")?;
        encode_one_block_replay(
            ctx,
            mf,
            mm,
            &cmd,
            block_i,
            position,
            &mut replay,
            &scratch,
            gdn_layers,
            h,
            conv_dim,
            v_dim,
        )?;
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        for slot in 0..tokens {
            let base_route = read_route_fingerprint(&base[slot], topk, n_expert);
            let replay_route = read_route_fingerprint(&replay[slot], topk, n_expert);
            let route_order_equal = base_route.idx == replay_route.idx;
            let route_set_equal = same_i32_set(&base_route.idx, &replay_route.idx);
            if !route_order_equal {
                out.route_order_mismatches += 1;
            }
            if !route_set_equal {
                out.route_set_mismatches += 1;
                out.first_set_mismatch.get_or_insert((block_i, slot));
            }
            let logit_max_abs = f32_max_abs_delta(&base_route.logits, &replay_route.logits);
            let logit_rms = f32_rms_delta(&base_route.logits, &replay_route.logits);
            out.max_logit_abs = out.max_logit_abs.max(logit_max_abs);
            out.max_logit_rms = out.max_logit_rms.max(logit_rms);
            out.min_base_margin = out.min_base_margin.min(base_route.logit_margin);
            out.min_replay_margin = out.min_replay_margin.min(replay_route.logit_margin);
            if replay_route.logit_margin < 0.001 {
                out.replay_margin_lt_1e3 += 1;
            }
            if replay_route.logit_margin < 0.005 {
                out.replay_margin_lt_5e3 += 1;
            }
            let (x_cos, x_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].x),
                &read_f32_tensor(&replay[slot].x),
            );
            out.min_x_cos = out.min_x_cos.min(x_cos);
            out.max_x_abs = out.max_x_abs.max(x_abs);
        }
    }

    Ok(out)
}

pub(crate) fn copy_recurrent_session_state(
    ctx: &MetalContext,
    src: &MetalSession,
    dst: &mut MetalSession,
) -> Result<()> {
    if src.gdn_conv.len() != dst.gdn_conv.len()
        || src.gdn_state.len() != dst.gdn_state.len()
        || src.kv_k.len() != dst.kv_k.len()
        || src.kv_v.len() != dst.kv_v.len()
    {
        return Err(anyhow!("session state vector lengths differ"));
    }
    let cmd = ctx
        .queue
        .commandBuffer()
        .context("copy session state cmd")?;
    let blit = BlitEncoder::begin(&cmd);
    for (a, b) in src.gdn_conv.iter().zip(dst.gdn_conv.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    for (a, b) in src.gdn_state.iter().zip(dst.gdn_state.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    for (a, b) in src.kv_k.iter().zip(dst.kv_k.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    for (a, b) in src.kv_v.iter().zip(dst.kv_v.iter()) {
        blit.copy_buffer(&a.buffer, a.offset, &b.buffer, b.offset, a.n_bytes());
    }
    blit.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    dst.kv_n_pos.clone_from(&src.kv_n_pos);
    Ok(())
}

pub(crate) fn reset_block_slice_sessions(
    ctx: &MetalContext,
    src: &[MetalSession],
    dst: &mut [MetalSession],
) -> Result<()> {
    if src.len() != dst.len() {
        return Err(anyhow!(
            "session reset length mismatch: {} source versus {} destination",
            src.len(),
            dst.len()
        ));
    }

    for (a, b) in src.iter().zip(dst.iter()) {
        if a.gdn_conv.len() != b.gdn_conv.len()
            || a.gdn_state.len() != b.gdn_state.len()
            || a.kv_k.len() != b.kv_k.len()
            || a.kv_v.len() != b.kv_v.len()
            || a.x.n_bytes() != b.x.n_bytes()
        {
            return Err(anyhow!("block-slice session reset shape mismatch"));
        }
    }

    let cmd = ctx
        .queue
        .commandBuffer()
        .context("reset block-slice sessions cmd")?;
    let blit = BlitEncoder::begin(&cmd);
    for (a, b) in src.iter().zip(dst.iter()) {
        blit.copy_buffer(
            &a.x.buffer,
            a.x.offset,
            &b.x.buffer,
            b.x.offset,
            a.x.n_bytes(),
        );
        for (src_t, dst_t) in a.gdn_conv.iter().zip(b.gdn_conv.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
        for (src_t, dst_t) in a.gdn_state.iter().zip(b.gdn_state.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
        for (src_t, dst_t) in a.kv_k.iter().zip(b.kv_k.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
        for (src_t, dst_t) in a.kv_v.iter().zip(b.kv_v.iter()) {
            blit.copy_buffer(
                &src_t.buffer,
                src_t.offset,
                &dst_t.buffer,
                dst_t.offset,
                src_t.n_bytes(),
            );
        }
    }
    blit.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    for (a, b) in src.iter().zip(dst.iter_mut()) {
        b.kv_n_pos.clone_from(&a.kv_n_pos);
    }
    Ok(())
}

pub(crate) fn prepare_moe_session_to_block(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    token_id: i32,
    position: u32,
    start_block: usize,
    session: &mut MetalSession,
    h: usize,
) -> Result<()> {
    unsafe {
        let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
        *ptr = token_id;
    }
    let cmd = ctx
        .queue
        .commandBuffer()
        .context("prepare block-slice session cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    encode_get_rows_f32(
        ctx,
        &enc,
        &mm.token_embd,
        &session.ids_buf,
        &session.x,
        1,
        h,
    )?;
    for block_i in 0..start_block {
        mf.encode_moe_block_by_index(&enc, block_i, position, session)?;
    }
    enc.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    Ok(())
}

pub(crate) fn prepare_real_block_slice_sessions(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    prefix_sessions: &[MetalSession],
    prompt_ids: &[Vec<i32>],
    slot_positions: &[usize],
    tokens: usize,
    start_block: usize,
    kv_capacity: usize,
    h: usize,
) -> Result<Vec<MetalSession>> {
    if slot_positions.len() < tokens {
        return Err(anyhow!(
            "slot_positions has {} entries, need {tokens}",
            slot_positions.len()
        ));
    }
    let mut sessions = fresh_gdn_replay_sessions_with_capacity(ctx, mm, tokens, kv_capacity)?;
    for slot in 0..tokens {
        copy_recurrent_session_state(ctx, &prefix_sessions[slot], &mut sessions[slot])?;
        let ids = if prompt_ids.len() == 1 {
            &prompt_ids[0]
        } else {
            &prompt_ids[slot]
        };
        let pos = slot_positions[slot];
        let token_id = ids[pos];
        prepare_moe_session_to_block(
            ctx,
            mf,
            mm,
            token_id,
            pos as u32,
            start_block,
            &mut sessions[slot],
            h,
        )?;
    }
    Ok(sessions)
}

pub(crate) struct ValidatedReplayStats {
    pub(crate) avg_wall_ms: f64,
    pub(crate) avg_gpu_ms: f64,
    pub(crate) p50_wall_ms: f64,
    pub(crate) p50_gpu_ms: f64,
    pub(crate) avg_fallback_slots: f64,
}

pub(crate) fn time_validated_block_slice_replay(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    seed_sessions: &[MetalSession],
    sessions: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    gdn_layers: &[SelectedGdnLayer<'_>],
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    topk: usize,
    n_expert: usize,
    margin_threshold: f32,
    warmup: usize,
    iters: usize,
) -> Result<ValidatedReplayStats> {
    let mut wall_samples = Vec::with_capacity(iters);
    let mut gpu_samples = Vec::with_capacity(iters);
    let mut fallback_samples = Vec::with_capacity(iters);

    for rep in 0..warmup + iters {
        let timed = rep >= warmup;
        reset_block_slice_sessions(ctx, seed_sessions, sessions)?;
        let start = Instant::now();
        let mut gpu_ms = 0.0f64;
        let mut fallback_slots = vec![false; sessions.len()];

        for block_i in start_block..start_block + n_blocks {
            let cmd = ctx
                .queue
                .commandBuffer()
                .context("validated replay block cmd")?;
            encode_one_block_replay(
                ctx, mf, mm, &cmd, block_i, position, sessions, scratch, gdn_layers, h, conv_dim,
                v_dim,
            )?;
            cmd.commit();
            qwen_llm::metal::wait_completed(&cmd)?;
            gpu_ms += (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3;

            for (slot, session) in sessions.iter().enumerate() {
                let route = read_route_fingerprint(session, topk, n_expert);
                if route.logit_margin < margin_threshold {
                    fallback_slots[slot] = true;
                }
            }
        }

        if timed {
            wall_samples.push(start.elapsed().as_secs_f64() * 1e3);
            gpu_samples.push(gpu_ms);
            fallback_samples.push(fallback_slots.iter().filter(|&&v| v).count() as f64);
        }
    }

    let avg_wall_ms = wall_samples.iter().sum::<f64>() / iters as f64;
    let avg_gpu_ms = gpu_samples.iter().sum::<f64>() / iters as f64;
    wall_samples.sort_by(|a, b| a.total_cmp(b));
    gpu_samples.sort_by(|a, b| a.total_cmp(b));
    Ok(ValidatedReplayStats {
        avg_wall_ms,
        avg_gpu_ms,
        p50_wall_ms: percentile(&wall_samples, 0.50),
        p50_gpu_ms: percentile(&gpu_samples, 0.50),
        avg_fallback_slots: fallback_samples.iter().sum::<f64>() / iters as f64,
    })
}

pub(crate) fn trace_prepared_block_slice_summary(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
    position: u32,
    base: &mut [MetalSession],
    replay: &mut [MetalSession],
    scratch: &GdnLayerReplayScratch,
    h: usize,
    conv_dim: usize,
    v_dim: usize,
    topk: usize,
    n_expert: usize,
    gdn_layers: &[SelectedGdnLayer<'_>],
) -> Result<BlockSliceTraceSummary> {
    if base.len() != replay.len() {
        return Err(anyhow!("base/replay slot counts differ"));
    }
    let mut out = BlockSliceTraceSummary::default();
    for block_i in start_block..start_block + n_blocks {
        let cmd = ctx
            .queue
            .commandBuffer()
            .context("real margin baseline cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        encode_block_slice_baseline(mf, &enc, block_i, 1, position, base)?;
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        let cmd = ctx
            .queue
            .commandBuffer()
            .context("real margin replay cmd")?;
        encode_one_block_replay(
            ctx, mf, mm, &cmd, block_i, position, replay, scratch, gdn_layers, h, conv_dim, v_dim,
        )?;
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;

        for slot in 0..base.len() {
            let base_route = read_route_fingerprint(&base[slot], topk, n_expert);
            let replay_route = read_route_fingerprint(&replay[slot], topk, n_expert);
            let route_order_equal = base_route.idx == replay_route.idx;
            let route_set_equal = same_i32_set(&base_route.idx, &replay_route.idx);
            if !route_order_equal {
                out.route_order_mismatches += 1;
            }
            if !route_set_equal {
                out.route_set_mismatches += 1;
                out.first_set_mismatch.get_or_insert((block_i, slot));
            }
            let logit_max_abs = f32_max_abs_delta(&base_route.logits, &replay_route.logits);
            let logit_rms = f32_rms_delta(&base_route.logits, &replay_route.logits);
            out.max_logit_abs = out.max_logit_abs.max(logit_max_abs);
            out.max_logit_rms = out.max_logit_rms.max(logit_rms);
            out.min_base_margin = out.min_base_margin.min(base_route.logit_margin);
            out.min_replay_margin = out.min_replay_margin.min(replay_route.logit_margin);
            if replay_route.logit_margin < 0.001 {
                out.replay_margin_lt_1e3 += 1;
            }
            if replay_route.logit_margin < 0.005 {
                out.replay_margin_lt_5e3 += 1;
            }
            let (x_cos, x_abs) = cosine_max_abs(
                &read_f32_tensor(&base[slot].x),
                &read_f32_tensor(&replay[slot].x),
            );
            out.min_x_cos = out.min_x_cos.min(x_cos);
            out.max_x_abs = out.max_x_abs.max(x_abs);
        }
    }
    Ok(out)
}

pub(crate) fn run_decode_block_slice_margin_sweep(
    args: DecodeBlockSliceMarginSweepArgs,
) -> Result<()> {
    let DecodeBlockSliceMarginSweepArgs {
        model,
        tokens,
        mut start_blocks,
        n_blocks,
        mut positions,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }
    if positions.is_empty() {
        return Err(anyhow!("--position must include at least one entry"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-margin-sweep currently requires an MoE model"
        ));
    }
    if start_blocks.is_empty() {
        start_blocks = (0..mm.blocks.len())
            .step_by(n_blocks)
            .filter(|&i| i + n_blocks <= mm.blocks.len())
            .collect();
    }
    start_blocks.sort_unstable();
    start_blocks.dedup();
    positions.sort_unstable();
    positions.dedup();
    for &start_block in &start_blocks {
        let end_block = start_block
            .checked_add(n_blocks)
            .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
        if end_block > mm.blocks.len() {
            return Err(anyhow!(
                "block slice {}..{} outside available 0..{}",
                start_block,
                end_block,
                mm.blocks.len()
            ));
        }
    }

    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let gdn_layers = collect_gdn_layers(&mm);

    println!(
        "[decode-block-slice-margin-sweep] model={} windows={} positions={} tokens={} blocks={} topk={} experts={} h={} conv_dim={} v_dim={}",
        model.display(),
        start_blocks.len(),
        positions
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        tokens,
        n_blocks,
        topk,
        n_expert,
        h,
        conv_dim,
        v_dim
    );
    println!(
        "start_block\tend_block\tposition\ttokens\troute_order_mismatches\troute_set_mismatches\tfirst_set_mismatch\tmax_logit_abs\tmax_logit_rms\tmin_base_margin\tmin_replay_margin\treplay_margin_lt_1e3\treplay_margin_lt_5e3\tmin_x_cos\tmax_x_abs"
    );

    for &position in &positions {
        for &start_block in &start_blocks {
            let summary = trace_block_slice_summary(
                &ctx,
                &mf,
                &mm,
                start_block,
                n_blocks,
                position,
                tokens,
                h,
                conv_dim,
                v_dim,
                topk,
                n_expert,
                &gdn_layers,
            )?;
            let first_set = summary
                .first_set_mismatch
                .map(|(block, slot)| format!("block={block},slot={slot}"))
                .unwrap_or_else(|| "none".to_string());
            println!(
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.9}\t{:.6}",
                start_block,
                start_block + n_blocks,
                position,
                tokens,
                summary.route_order_mismatches,
                summary.route_set_mismatches,
                first_set,
                summary.max_logit_abs,
                summary.max_logit_rms,
                summary.min_base_margin,
                summary.min_replay_margin,
                summary.replay_margin_lt_1e3,
                summary.replay_margin_lt_5e3,
                summary.min_x_cos,
                summary.max_x_abs,
            );
        }
    }

    Ok(())
}

pub(crate) fn run_decode_block_slice_real_margin(
    args: DecodeBlockSliceRealMarginArgs,
) -> Result<()> {
    let DecodeBlockSliceRealMarginArgs {
        model,
        file,
        tokens,
        mut slot_counts,
        mut contexts,
        stride,
        mut start_blocks,
        n_blocks,
        timing_iters,
        timing_warmup,
        margin_threshold,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if slot_counts.is_empty() {
        slot_counts.push(tokens);
    }
    slot_counts.sort_unstable();
    slot_counts.dedup();
    if slot_counts.contains(&0) {
        return Err(anyhow!("--slot-counts must all be >= 1"));
    }
    let max_slots = *slot_counts.iter().max().expect("non-empty slot_counts");
    if max_slots > tokens {
        return Err(anyhow!(
            "largest --slot-counts value ({max_slots}) exceeds --tokens {tokens}; set --tokens to the maximum slots to prepare"
        ));
    }
    if stride == 0 {
        return Err(anyhow!("--stride must be >= 1"));
    }
    if file.is_empty() {
        return Err(anyhow!("--file must include at least one prompt"));
    }
    if file.len() > 1 && file.len() < max_slots {
        return Err(anyhow!(
            "got {} --file entries, need at least {max_slots} for the requested slot counts",
            file.len(),
        ));
    }
    if n_blocks == 0 {
        return Err(anyhow!("--blocks must be >= 1"));
    }
    if timing_iters == 0 && timing_warmup != 1 {
        return Err(anyhow!(
            "--timing-warmup is only meaningful when --timing-iters > 0"
        ));
    }
    if !(margin_threshold.is_finite() && margin_threshold >= 0.0) {
        return Err(anyhow!(
            "--margin-threshold must be a finite non-negative value"
        ));
    }
    if contexts.is_empty() {
        return Err(anyhow!("--context must include at least one entry"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-block-slice-real-margin currently requires an MoE model"
        ));
    }
    if start_blocks.is_empty() {
        start_blocks = (0..mm.blocks.len())
            .step_by(n_blocks)
            .filter(|&i| i + n_blocks <= mm.blocks.len())
            .collect();
    }
    start_blocks.sort_unstable();
    start_blocks.dedup();
    contexts.sort_unstable();
    contexts.dedup();
    for &start_block in &start_blocks {
        let end_block = start_block
            .checked_add(n_blocks)
            .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
        if end_block > mm.blocks.len() {
            return Err(anyhow!(
                "block slice {}..{} outside available 0..{}",
                start_block,
                end_block,
                mm.blocks.len()
            ));
        }
    }

    let tok = NativeTokenizer::from_gguf(&g).context("open native GGUF tokenizer")?;
    let mut prompt_ids = Vec::with_capacity(file.len());
    for path in &file {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let ids = tok
            .encode(&text, false)
            .with_context(|| format!("tokenize {}", path.display()))?;
        prompt_ids.push(ids);
    }
    let max_context = *contexts.last().expect("non-empty contexts");
    let last_needed = if file.len() == 1 {
        max_context
            .checked_add((max_slots - 1).saturating_mul(stride))
            .ok_or_else(|| anyhow!("context + (tokens - 1) * stride overflow"))?
    } else {
        max_context
    };
    if file.len() == 1 {
        if prompt_ids[0].len() <= last_needed {
            return Err(anyhow!(
                "prompt {} has {} tokens, need at least {} for context+stride sweep",
                file[0].display(),
                prompt_ids[0].len(),
                last_needed + 1
            ));
        }
    } else {
        for (path, ids) in file.iter().zip(prompt_ids.iter()).take(max_slots) {
            if ids.len() <= last_needed {
                return Err(anyhow!(
                    "prompt {} has {} tokens, need at least {} for context sweep",
                    path.display(),
                    ids.len(),
                    last_needed + 1
                ));
            }
        }
    }

    let mf = MetalForward::new(&ctx, &mm);
    let arch = &mm.arch;
    let h = arch.hidden_size as usize;
    let n_v = arch.gdn_n_v_heads as usize;
    let n_k = arch.gdn_n_k_heads as usize;
    let head_dim = arch.gdn_head_dim as usize;
    let conv_dim = (2 * n_k + n_v) * head_dim;
    let v_dim = n_v * head_dim;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let n_expert = arch.expert_count as usize;
    let gdn_layers = collect_gdn_layers(&mm);

    println!(
        "[decode-block-slice-real-margin] model={} files={} first_prompt_tokens={} contexts={} windows={} tokens={} slot_counts={} stride={} blocks={} topk={} experts={} h={} conv_dim={} v_dim={} timing_iters={} timing_warmup={} margin_threshold={:.6}",
        model.display(),
        file.len(),
        prompt_ids[0].len(),
        contexts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        start_blocks.len(),
        tokens,
        slot_counts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        stride,
        n_blocks,
        topk,
        n_expert,
        h,
        conv_dim,
        v_dim,
        timing_iters,
        timing_warmup,
        margin_threshold,
    );
    println!(
        "start_block\tend_block\tcontext\tslots\troute_order_mismatches\troute_set_mismatches\tfirst_set_mismatch\tmax_logit_abs\tmax_logit_rms\tmin_base_margin\tmin_replay_margin\treplay_margin_lt_1e3\treplay_margin_lt_5e3\tmin_x_cos\tmax_x_abs\tbaseline_wall_ms_per_tok\treplay_wall_ms_per_tok\tgross_wall_save_pct\tvalidated_wall_ms_per_tok\tvalidated_gpu_ms_per_tok\tfallback_slots_avg\tfallback_pct\tnet_wall_save_pct\tbaseline_p50_wall_ms_per_tok\treplay_p50_wall_ms_per_tok\tvalidated_p50_wall_ms_per_tok\tvalidated_p50_gpu_ms_per_tok\tp50_net_wall_save_pct"
    );

    for &context in &contexts {
        let kv_capacity = last_needed + 32;
        let slot_positions: Vec<usize> = (0..max_slots)
            .map(|slot| {
                if prompt_ids.len() == 1 {
                    context + slot * stride
                } else {
                    context
                }
            })
            .collect();
        let mut prefix_sessions = Vec::with_capacity(max_slots);
        if prompt_ids.len() == 1 {
            let ids = &prompt_ids[0];
            let mut prev_pos = slot_positions[0];
            let mut first = MetalSession::fresh(&ctx, &mm, kv_capacity)
                .context("real prefix session slot 0")?;
            for (p, &token_id) in ids.iter().take(prev_pos).enumerate() {
                let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut first)?;
            }
            prefix_sessions.push(first);
            for slot in 1..max_slots {
                let pos = slot_positions[slot];
                let mut s = MetalSession::fresh(&ctx, &mm, kv_capacity)
                    .with_context(|| format!("real prefix session slot {slot}"))?;
                copy_recurrent_session_state(&ctx, &prefix_sessions[slot - 1], &mut s)?;
                for (p, &token_id) in ids.iter().enumerate().take(pos).skip(prev_pos) {
                    let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut s)?;
                }
                prefix_sessions.push(s);
                prev_pos = pos;
            }
        } else {
            for slot in 0..max_slots {
                let ids = &prompt_ids[slot];
                let pos = slot_positions[slot];
                let mut s = MetalSession::fresh(&ctx, &mm, kv_capacity)
                    .with_context(|| format!("real prefix session slot {slot}"))?;
                for (p, &token_id) in ids.iter().take(pos).enumerate() {
                    let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut s)?;
                }
                prefix_sessions.push(s);
            }
        }

        for &start_block in &start_blocks {
            let slice_has_attention = block_slice_has_attention(&mm, start_block, n_blocks)?;
            for &slot_count in &slot_counts {
                if prompt_ids.len() == 1 && slot_count > 1 && slice_has_attention {
                    return Err(anyhow!(
                        "single-file strided real-margin slots currently require GDN-only block slices; slice {}..{} contains attention",
                        start_block,
                        start_block + n_blocks
                    ));
                }
                let mut base = prepare_real_block_slice_sessions(
                    &ctx,
                    &mf,
                    &mm,
                    &prefix_sessions,
                    &prompt_ids,
                    &slot_positions,
                    slot_count,
                    start_block,
                    kv_capacity,
                    h,
                )?;
                let mut replay = prepare_real_block_slice_sessions(
                    &ctx,
                    &mf,
                    &mm,
                    &prefix_sessions,
                    &prompt_ids,
                    &slot_positions,
                    slot_count,
                    start_block,
                    kv_capacity,
                    h,
                )?;

                let scratch = GdnLayerReplayScratch::new(&ctx, slot_count, h, conv_dim, v_dim)?;
                let summary = trace_prepared_block_slice_summary(
                    &ctx,
                    &mf,
                    &mm,
                    start_block,
                    n_blocks,
                    context as u32,
                    &mut base,
                    &mut replay,
                    &scratch,
                    h,
                    conv_dim,
                    v_dim,
                    topk,
                    n_expert,
                    &gdn_layers,
                )?;
                let timing = if timing_iters > 0 {
                    let timing_base_seed = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let mut timing_base = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let timing_replay_seed = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let mut timing_replay = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let timing_validated_seed = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let mut timing_validated = prepare_real_block_slice_sessions(
                        &ctx,
                        &mf,
                        &mm,
                        &prefix_sessions,
                        &prompt_ids,
                        &slot_positions,
                        slot_count,
                        start_block,
                        kv_capacity,
                        h,
                    )?;
                    let timing_scratch =
                        GdnLayerReplayScratch::new(&ctx, slot_count, h, conv_dim, v_dim)?;
                    let baseline = time_cmd_reps_stats(&ctx, timing_warmup, timing_iters, |cmd| {
                        reset_block_slice_sessions(&ctx, &timing_base_seed, &mut timing_base)?;
                        let enc = KernelEncoder::begin(cmd);
                        encode_block_slice_baseline(
                            &mf,
                            &enc,
                            start_block,
                            n_blocks,
                            context as u32,
                            &mut timing_base,
                        )?;
                        enc.end();
                        Ok(())
                    })?;
                    let replay_stats =
                        time_cmd_reps_stats(&ctx, timing_warmup, timing_iters, |cmd| {
                            reset_block_slice_sessions(
                                &ctx,
                                &timing_replay_seed,
                                &mut timing_replay,
                            )?;
                            encode_block_slice_replay(
                                &ctx,
                                &mf,
                                &mm,
                                cmd,
                                start_block,
                                n_blocks,
                                context as u32,
                                &mut timing_replay,
                                &timing_scratch,
                                &gdn_layers,
                                h,
                                conv_dim,
                                v_dim,
                            )
                        })?;
                    let validated = time_validated_block_slice_replay(
                        &ctx,
                        &mf,
                        &mm,
                        start_block,
                        n_blocks,
                        context as u32,
                        &timing_validated_seed,
                        &mut timing_validated,
                        &timing_scratch,
                        &gdn_layers,
                        h,
                        conv_dim,
                        v_dim,
                        topk,
                        n_expert,
                        margin_threshold,
                        timing_warmup,
                        timing_iters,
                    )?;
                    let baseline_per_tok = baseline.avg_wall_ms / slot_count as f64;
                    let replay_per_tok = replay_stats.avg_wall_ms / slot_count as f64;
                    let validated_wall_per_tok = validated.avg_wall_ms / slot_count as f64;
                    let validated_gpu_per_tok = validated.avg_gpu_ms / slot_count as f64;
                    let baseline_p50_wall_per_tok = baseline.p50_wall_ms / slot_count as f64;
                    let replay_p50_wall_per_tok = replay_stats.p50_wall_ms / slot_count as f64;
                    let validated_p50_wall_per_tok = validated.p50_wall_ms / slot_count as f64;
                    let validated_p50_gpu_per_tok = validated.p50_gpu_ms / slot_count as f64;
                    let fallback_pct = validated.avg_fallback_slots / slot_count as f64;
                    let fallback_exact_per_tok = fallback_pct * baseline_per_tok;
                    let gross_save_pct = if baseline_per_tok > 0.0 {
                        (baseline_per_tok - replay_per_tok) / baseline_per_tok * 100.0
                    } else {
                        0.0
                    };
                    let net_save_pct = if baseline_per_tok > 0.0 {
                        (baseline_per_tok - validated_wall_per_tok - fallback_exact_per_tok)
                            / baseline_per_tok
                            * 100.0
                    } else {
                        0.0
                    };
                    let p50_net_save_pct = if baseline_p50_wall_per_tok > 0.0 {
                        (baseline_p50_wall_per_tok
                            - validated_p50_wall_per_tok
                            - fallback_exact_per_tok)
                            / baseline_p50_wall_per_tok
                            * 100.0
                    } else {
                        0.0
                    };
                    Some((
                        baseline_per_tok,
                        replay_per_tok,
                        gross_save_pct,
                        validated_wall_per_tok,
                        validated_gpu_per_tok,
                        validated.avg_fallback_slots,
                        fallback_pct * 100.0,
                        net_save_pct,
                        baseline_p50_wall_per_tok,
                        replay_p50_wall_per_tok,
                        validated_p50_wall_per_tok,
                        validated_p50_gpu_per_tok,
                        p50_net_save_pct,
                    ))
                } else {
                    None
                };
                let first_set = summary
                    .first_set_mismatch
                    .map(|(block, slot)| format!("block={block},slot={slot}"))
                    .unwrap_or_else(|| "none".to_string());
                let timing_cols = timing.map_or_else(
                    || "\t".repeat(13),
                    |(
                        baseline_per_tok,
                        replay_per_tok,
                        gross_save_pct,
                        validated_wall_per_tok,
                        validated_gpu_per_tok,
                        fallback_slots_avg,
                        fallback_pct,
                        net_save_pct,
                        baseline_p50_wall_per_tok,
                        replay_p50_wall_per_tok,
                        validated_p50_wall_per_tok,
                        validated_p50_gpu_per_tok,
                        p50_net_save_pct,
                    )| {
                        format!(
                            "\t{baseline_per_tok:.4}\t{replay_per_tok:.4}\t{gross_save_pct:.2}\t{validated_wall_per_tok:.4}\t{validated_gpu_per_tok:.4}\t{fallback_slots_avg:.2}\t{fallback_pct:.2}\t{net_save_pct:.2}\t{baseline_p50_wall_per_tok:.4}\t{replay_p50_wall_per_tok:.4}\t{validated_p50_wall_per_tok:.4}\t{validated_p50_gpu_per_tok:.4}\t{p50_net_save_pct:.2}"
                        )
                    },
                );
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{}\t{}\t{:.9}\t{:.6}{}",
                    start_block,
                    start_block + n_blocks,
                    context,
                    slot_count,
                    summary.route_order_mismatches,
                    summary.route_set_mismatches,
                    first_set,
                    summary.max_logit_abs,
                    summary.max_logit_rms,
                    summary.min_base_margin,
                    summary.min_replay_margin,
                    summary.replay_margin_lt_1e3,
                    summary.replay_margin_lt_5e3,
                    summary.min_x_cos,
                    summary.max_x_abs,
                    timing_cols,
                );
            }
        }
    }

    Ok(())
}

pub(crate) fn moe_for_block(mm: &MetalModel, block_idx: usize) -> Result<Option<&MetalMoeFfn>> {
    let block = mm
        .blocks
        .get(block_idx)
        .ok_or_else(|| anyhow!("block index {block_idx} >= {}", mm.blocks.len()))?;
    Ok(match block {
        MetalBlock::Gdn(b) => b.ffn_moe.as_ref(),
        MetalBlock::Attn(b) => b.ffn_moe.as_ref(),
    })
}

pub(crate) fn block_slice_has_attention(
    mm: &MetalModel,
    start_block: usize,
    n_blocks: usize,
) -> Result<bool> {
    let end_block = start_block
        .checked_add(n_blocks)
        .ok_or_else(|| anyhow!("start_block + blocks overflow"))?;
    if end_block > mm.blocks.len() {
        return Err(anyhow!(
            "block slice {}..{} outside available 0..{}",
            start_block,
            end_block,
            mm.blocks.len()
        ));
    }
    Ok(mm.blocks[start_block..end_block]
        .iter()
        .any(|block| matches!(block, MetalBlock::Attn(_))))
}

pub(crate) fn cpu_route_fingerprint(
    moe: &MetalMoeFfn,
    h_cpu: &[f32],
    topk: usize,
    n_expert: usize,
) -> RouteFingerprint {
    let mut logits = mat_vec_pub(&moe.gate_inp_cpu, h_cpu.len(), n_expert, h_cpu);
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    ranked.truncate(topk);
    let max_top = ranked.first().map(|&(_, v)| v).unwrap_or(f32::NEG_INFINITY);
    let mut sum = 0.0f32;
    let mut exp_vals = Vec::with_capacity(ranked.len());
    for &(_, v) in &ranked {
        let e = (v - max_top).exp();
        exp_vals.push(e);
        sum += e;
    }
    let inv = 1.0 / sum.max(6.103_515_6e-5);
    let shared_dot: f32 = h_cpu
        .iter()
        .zip(&moe.gate_inp_shexp_cpu[..h_cpu.len()])
        .map(|(a, b)| a * b)
        .sum();
    let idx = ranked.iter().map(|&(expert, _)| expert as i32).collect();
    let weight = exp_vals.into_iter().map(|v| v * inv).collect();
    let logit_margin = topk_logit_margin(&logits, topk);
    RouteFingerprint {
        idx,
        weight,
        shared_gate: 1.0 / (1.0 + (-shared_dot).exp()),
        logit_margin,
        logits: std::mem::take(&mut logits),
    }
}

pub(crate) fn seed_session_current_token(
    ctx: &MetalContext,
    mm: &MetalModel,
    session: &mut MetalSession,
    token_id: i32,
    h: usize,
) -> Result<()> {
    unsafe {
        let ptr = session.ids_buf.buffer.contents().as_ptr() as *mut i32;
        *ptr = token_id;
    }
    let cmd = ctx.queue.commandBuffer().context("router check seed cmd")?;
    let enc = KernelEncoder::begin(&cmd);
    encode_get_rows_f32(
        ctx,
        &enc,
        &mm.token_embd,
        &session.ids_buf,
        &session.x,
        1,
        h,
    )?;
    enc.end();
    cmd.commit();
    qwen_llm::metal::wait_completed(&cmd)?;
    Ok(())
}
