//! Pipelined decode-window probes.

use super::*;

pub(crate) fn run_decode_window(args: DecodeWindowArgs) -> Result<()> {
    let DecodeWindowArgs {
        model,
        target_ctx,
        prefill_warm,
        window,
        streams,
        stage_timestamps,
        stage_split_attn_route,
        stage_split_attn_detail,
        stage_split_gdn_after,
        ready_file,
        go_file,
        pipelined,
        concurrent_gdn_proj,
        concurrent_attn_proj,
    } = args;
    if streams == 0 {
        return Err(anyhow!("--streams must be >= 1"));
    }
    if streams > 1 && (pipelined || concurrent_gdn_proj || concurrent_attn_proj) {
        return Err(anyhow!(
            "--streams is a concurrency discriminator; combine it only with the default decode path"
        ));
    }
    if stage_timestamps && streams > 1 {
        return Err(anyhow!(
            "--stage-timestamps is a single-stream attribution probe; do not combine it with --streams"
        ));
    }
    if stage_timestamps && (pipelined || concurrent_gdn_proj || concurrent_attn_proj) {
        return Err(anyhow!(
            "--stage-timestamps profiles the default MoE decode shape; do not combine it with other decode-window experiments"
        ));
    }
    if stage_split_attn_route && !stage_timestamps {
        return Err(anyhow!(
            "--stage-split-attn-route requires --stage-timestamps"
        ));
    }
    if stage_split_attn_detail && !stage_timestamps {
        return Err(anyhow!(
            "--stage-split-attn-detail requires --stage-timestamps"
        ));
    }
    if stage_split_gdn_after && !stage_timestamps {
        return Err(anyhow!(
            "--stage-split-gdn-after requires --stage-timestamps"
        ));
    }
    let ctx = MetalContext::new()?;
    eprintln!("[decode-window] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;
    let mut s = MetalSession::fresh(&ctx, &mm, target_ctx + window + 16)?;
    let mf = MetalForward::new(&ctx, &mm);

    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s)?;
    }
    let mut s = MetalSession::fresh(&ctx, &mm, target_ctx + window + 16)?;
    if prefill_warm && target_ctx > 1 {
        let ids = vec![0i32; target_ctx];
        let chunk = default_prefill_chunk(mm.arch.kind, target_ctx);
        let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, chunk, ids.len())
            .context("prefill-warm scratch")?;
        let t0 = Instant::now();
        prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
            .context("prefill-warm")?;
        eprintln!(
            "[decode-window] prefill-warm to {target_ctx} in {:.1}s",
            t0.elapsed().as_secs_f64()
        );
    } else {
        let _ = mf.single_token(0, 0, &mut s)?;
        for p in 1..(target_ctx as u32) {
            let _ = mf.single_token(0, p, &mut s)?;
        }
    }

    let mut s_opt = Some(s);
    let mut multi_stream_sessions = None;
    if streams > 1 {
        let mut sessions = Vec::with_capacity(streams);
        sessions.push(s_opt.take().expect("primary session present"));
        for stream_idx in 1..streams {
            eprintln!(
                "[decode-window] ramping stream {}/{} to ctx={}",
                stream_idx + 1,
                streams,
                target_ctx
            );
            let mut sx = MetalSession::fresh(&ctx, &mm, target_ctx + window + 16)?;
            let _ = mf.single_token(0, 0, &mut sx)?;
            for p in 1..(target_ctx as u32) {
                shutdown::checkpoint()?;
                let _ = mf.single_token(0, p, &mut sx)?;
            }
            sessions.push(sx);
        }
        multi_stream_sessions = Some(sessions);
    }

    if let Some(parent) = ready_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if let Some(parent) = go_file.parent() {
        std::fs::create_dir_all(parent)?;
    }
    if go_file.exists() {
        std::fs::remove_file(&go_file)?;
    }
    std::fs::write(
        &ready_file,
        format!("ready ctx={} window={}\n", target_ctx, window),
    )?;
    eprintln!(
        "[decode-window] ready at ctx={} waiting for {:?}",
        target_ctx, go_file
    );
    while !go_file.exists() {
        shutdown::checkpoint()?;
        std::thread::sleep(Duration::from_millis(25));
    }
    eprintln!(
        "[decode-window] go signal received; running {} decode tokens",
        window
    );

    if pipelined && (concurrent_gdn_proj || concurrent_attn_proj) {
        return Err(anyhow!(
            "--pipelined is separate from the concurrent projection experiments; use one mode at a time"
        ));
    }

    if let Some(mut sessions) = multi_stream_sessions {
        return run_decode_window_multi_stream(&ctx, &mf, &mut sessions, target_ctx, window);
    }

    let mut s = s_opt.expect("single-stream session present");

    if stage_timestamps {
        if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
            return Err(anyhow!(
                "--stage-timestamps currently supports MoE models only"
            ));
        }
        return run_decode_window_stage_timestamps(
            &mf,
            &mut s,
            target_ctx,
            window,
            stage_split_attn_route,
            stage_split_attn_detail,
            stage_split_gdn_after,
        );
    }

    fn median(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) * 0.5
        }
    }

    fn p95(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let idx = ((v.len() - 1) as f64 * 0.95).round() as usize;
        v[idx]
    }

    if pipelined {
        let ids_ping = [
            MetalTensor::zeros_i32(&ctx, vec![1])?,
            MetalTensor::zeros_i32(&ctx, vec![1])?,
        ];
        let argmax_ping = [
            MetalTensor::zeros_i32(&ctx, vec![1])?,
            MetalTensor::zeros_i32(&ctx, vec![1])?,
        ];

        let mut encode_ms = Vec::with_capacity(window);
        let mut wait_ms = Vec::with_capacity(window);
        let mut gpu_ms = Vec::with_capacity(window);
        let total_t = Instant::now();
        let mut next_tok = 0i32;
        let mut pos = target_ctx as u32;

        unsafe {
            let ptr = ids_ping[0].buffer.contents().as_ptr() as *mut i32;
            *ptr = next_tok;
        }
        let first_cmd = ctx.queue.commandBuffer().expect("command buffer");
        let first_encode_t = Instant::now();
        let first_enc = qwen_llm::metal::KernelEncoder::begin(&first_cmd);
        mf.encode_single_token_argmax(&first_enc, pos, &mut s, &ids_ping[0], &argmax_ping[0])?;
        first_enc.end();
        encode_ms.push(first_encode_t.elapsed().as_secs_f64() * 1e3);
        first_cmd.commit();
        let mut pending_cmd = first_cmd;
        let mut pending_slot = 0usize;
        pos += 1;

        for _step in 1..window {
            let next_slot = pending_slot ^ 1;
            let next_cmd = ctx.queue.commandBuffer().expect("command buffer");
            let next_encode_t = Instant::now();
            let next_enc = qwen_llm::metal::KernelEncoder::begin(&next_cmd);
            mf.encode_single_token_argmax(
                &next_enc,
                pos,
                &mut s,
                &ids_ping[next_slot],
                &argmax_ping[next_slot],
            )?;
            next_enc.end();
            encode_ms.push(next_encode_t.elapsed().as_secs_f64() * 1e3);

            let wait_t = Instant::now();
            pending_cmd.waitUntilCompleted();
            wait_ms.push(wait_t.elapsed().as_secs_f64() * 1e3);
            gpu_ms.push((pending_cmd.GPUEndTime() - pending_cmd.GPUStartTime()) * 1e3);
            next_tok = unsafe {
                let src = argmax_ping[pending_slot].buffer.contents().as_ptr() as *const i32;
                *src
            };
            unsafe {
                let ptr = ids_ping[next_slot].buffer.contents().as_ptr() as *mut i32;
                *ptr = next_tok;
            }
            next_cmd.commit();
            pending_cmd = next_cmd;
            pending_slot = next_slot;
            pos += 1;
        }

        let wait_t = Instant::now();
        pending_cmd.waitUntilCompleted();
        wait_ms.push(wait_t.elapsed().as_secs_f64() * 1e3);
        gpu_ms.push((pending_cmd.GPUEndTime() - pending_cmd.GPUStartTime()) * 1e3);

        let total_ms = total_t.elapsed().as_secs_f64() * 1e3;
        let avg_total = total_ms / window as f64;
        let avg_gpu = gpu_ms.iter().sum::<f64>() / window as f64;
        let avg_enc = encode_ms.iter().sum::<f64>() / window as f64;
        let _avg_wait = wait_ms.iter().sum::<f64>() / window as f64;
        let med_gpu = median(&gpu_ms);
        let med_enc = median(&encode_ms);
        let med_wait = median(&wait_ms);
        let p95_gpu = p95(&gpu_ms);
        let p95_enc = p95(&encode_ms);
        let p95_wait = p95(&wait_ms);
        eprintln!(
            "[decode-window] ctx={} window={} pipelined avg_total={:.2} ms avg_gpu={:.2} ms avg_cpu_enc={:.2} ms t/s={:.1}",
            target_ctx,
            window,
            avg_total,
            avg_gpu,
            avg_enc,
            1000.0 / avg_total
        );
        eprintln!(
            "[decode-window] pipelined med_gpu={:.2} ms med_cpu_enc={:.2} ms med_wait={:.2} ms gpu/total(avg)={:.1}%",
            med_gpu,
            med_enc,
            med_wait,
            100.0 * avg_gpu / avg_total
        );
        eprintln!(
            "[decode-window] pipelined p95_gpu={:.2} ms p95_cpu_enc={:.2} ms p95_wait={:.2} ms",
            p95_gpu, p95_enc, p95_wait,
        );
        return Ok(());
    }

    let mut samples = Vec::with_capacity(window);
    let mut prev_tok = 0i32;
    for i in 0..window {
        let pos = target_ctx as u32 + i as u32;
        let (logits, p) = match (mm.arch.kind, concurrent_gdn_proj, concurrent_attn_proj) {
            (qwen_llm::model::ArchKind::Dense, true, true) => {
                mf.single_token_profiled_concurrent_gdn_attn_dense(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Dense, true, false) => {
                mf.single_token_profiled_concurrent_gdn_dense(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Dense, false, true) => {
                mf.single_token_profiled_concurrent_attn_dense(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Moe, true, false) => {
                mf.single_token_profiled_concurrent_gdn_moe(prev_tok, pos, &mut s)?
            }
            (qwen_llm::model::ArchKind::Moe, _, true) => {
                return Err(anyhow!(
                    "--concurrent-attn-proj decode-window is currently dense-only"
                ));
            }
            _ => mf.single_token_profiled(prev_tok, pos, &mut s)?,
        };
        samples.push(p);
        prev_tok = argmax_i32(&logits);
    }

    let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / window as f64;
    let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / window as f64;
    let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / window as f64;
    let avg_wait = samples
        .iter()
        .map(|p| p.cpu_to_gpu_complete_ms)
        .sum::<f64>()
        / window as f64;
    let totals: Vec<f64> = samples.iter().map(|p| p.total_ms).collect();
    let gpus: Vec<f64> = samples.iter().map(|p| p.gpu_kernel_ms).collect();
    let encs: Vec<f64> = samples.iter().map(|p| p.cpu_encode_ms).collect();
    let waits: Vec<f64> = samples.iter().map(|p| p.cpu_to_gpu_complete_ms).collect();
    let med_total = median(&totals);
    let med_gpu = median(&gpus);
    let med_enc = median(&encs);
    let med_wait = median(&waits);
    let p95_total = p95(&totals);
    let p95_gpu = p95(&gpus);
    let p95_enc = p95(&encs);
    let p95_wait = p95(&waits);
    eprintln!(
        "[decode-window] ctx={} window={}{}{}{} avg_total={:.2} ms avg_gpu={:.2} ms avg_cpu_enc={:.2} ms t/s={:.1}",
        target_ctx,
        window,
        if concurrent_gdn_proj {
            " concurrent_gdn"
        } else {
            ""
        },
        if concurrent_attn_proj {
            " concurrent_attn"
        } else {
            ""
        },
        if concurrent_gdn_proj && concurrent_attn_proj {
            "_both"
        } else {
            ""
        },
        avg_total,
        avg_gpu,
        avg_enc,
        1000.0 / avg_total
    );
    eprintln!(
        "[decode-window] med_total={:.2} ms med_gpu={:.2} ms med_cpu_enc={:.2} ms med_wait={:.2} ms gpu/total={:.1}%",
        med_total,
        med_gpu,
        med_enc,
        med_wait,
        100.0 * med_gpu / med_total
    );
    eprintln!(
        "[decode-window] p95_total={:.2} ms p95_gpu={:.2} ms p95_cpu_enc={:.2} ms p95_wait={:.2} ms",
        p95_total, p95_gpu, p95_enc, p95_wait,
    );
    eprintln!(
        "[decode-window] avg_wait={:.2} ms gpu/total(avg)={:.1}%",
        avg_wait,
        100.0 * avg_gpu / avg_total
    );
    Ok(())
}

#[derive(Default)]
pub(crate) struct StageAgg {
    pub(crate) count: usize,
    pub(crate) ms: f64,
}

pub(crate) fn run_decode_window_stage_timestamps(
    mf: &MetalForward,
    session: &mut MetalSession,
    target_ctx: usize,
    window: usize,
    split_attn_route: bool,
    split_attn_detail: bool,
    split_gdn_after: bool,
) -> Result<()> {
    let mut prev_tok = 0i32;
    let mut total_ms = 0.0f64;
    let mut gpu_ms = 0.0f64;
    let mut enc_ms = 0.0f64;
    let mut wait_ms = 0.0f64;
    let mut raw_cov = 0.0f64;
    let mut raw_span_ms = 0.0f64;
    let mut sampled_ticks = 0u64;
    let mut family = BTreeMap::<(String, String, bool), StageAgg>::new();
    let mut block =
        BTreeMap::<(String, String, Option<usize>, Option<usize>, bool), StageAgg>::new();

    for i in 0..window {
        let pos = target_ctx as u32 + i as u32;
        let (tok, profile) = mf.single_token_argmax_stage_profiled_concurrent_gdn_moe(
            prev_tok,
            pos,
            session,
            split_attn_route,
            split_attn_detail,
            split_gdn_after,
        )?;
        prev_tok = tok;
        total_ms += profile.token.total_ms;
        gpu_ms += profile.token.gpu_kernel_ms;
        enc_ms += profile.token.cpu_encode_ms;
        wait_ms += profile.token.cpu_to_gpu_complete_ms;
        raw_cov += profile.raw_coverage_assuming_ns;
        raw_span_ms += profile.raw_span_ms_assuming_ns;
        sampled_ticks = sampled_ticks.saturating_add(profile.sampled_span_ticks);

        for stage in profile.stages {
            let fam_key = (
                stage.block_kind.clone(),
                stage.family.clone(),
                stage.concurrent,
            );
            let fam = family.entry(fam_key).or_default();
            fam.count += 1;
            fam.ms += stage.duration_ms_scaled;

            let block_key = (
                stage.block_kind,
                stage.family,
                stage.block_index,
                stage.local_index,
                stage.concurrent,
            );
            let blk = block.entry(block_key).or_default();
            blk.count += 1;
            blk.ms += stage.duration_ms_scaled;
        }
    }

    let avg_total = total_ms / window as f64;
    let avg_gpu = gpu_ms / window as f64;
    let avg_enc = enc_ms / window as f64;
    let avg_wait = wait_ms / window as f64;
    eprintln!(
        "[decode-stage] ctx={} window={}{}{}{} avg_total={:.2} ms avg_gpu={:.2} ms avg_cpu_enc={:.2} ms t/s={:.1}",
        target_ctx,
        window,
        if split_attn_route {
            " split_attn_route"
        } else {
            ""
        },
        if split_attn_detail {
            " split_attn_detail"
        } else {
            ""
        },
        if split_gdn_after {
            " split_gdn_after"
        } else {
            ""
        },
        avg_total,
        avg_gpu,
        avg_enc,
        1000.0 / avg_total,
    );
    eprintln!(
        "[decode-stage] avg_wait={:.2} ms raw_coverage_assuming_ns={:.3} raw_span_ms={:.3} sampled_ticks={}",
        avg_wait,
        raw_cov / window as f64,
        raw_span_ms / window as f64,
        sampled_ticks,
    );

    println!(
        "row\tblock_kind\tfamily\tblock_index\tlocal_index\tconcurrent\tcount\tavg_ms\tpct_gpu"
    );
    for ((block_kind, family_name, concurrent), agg) in family {
        println!(
            "family\t{}\t{}\t\t\t{}\t{}\t{:.4}\t{:.2}",
            block_kind,
            family_name,
            concurrent,
            agg.count,
            agg.ms / window as f64,
            100.0 * agg.ms / gpu_ms,
        );
    }
    for ((block_kind, family_name, block_index, local_index, concurrent), agg) in block {
        let block_index = block_index.map(|v| v.to_string()).unwrap_or_default();
        let local_index = local_index.map(|v| v.to_string()).unwrap_or_default();
        println!(
            "block\t{}\t{}\t{}\t{}\t{}\t{}\t{:.4}\t{:.2}",
            block_kind,
            family_name,
            block_index,
            local_index,
            concurrent,
            agg.count,
            agg.ms / window as f64,
            100.0 * agg.ms / gpu_ms,
        );
    }
    Ok(())
}

pub(crate) fn run_decode_window_multi_stream(
    ctx: &MetalContext,
    mf: &MetalForward,
    sessions: &mut [MetalSession],
    target_ctx: usize,
    window: usize,
) -> Result<()> {
    fn median(values: &[f64]) -> f64 {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.total_cmp(b));
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) * 0.5
        }
    }

    let streams = sessions.len();
    let mut queues = Vec::with_capacity(streams);
    let mut ids = Vec::with_capacity(streams);
    let mut argmax = Vec::with_capacity(streams);
    for _ in 0..streams {
        queues.push(
            ctx.device
                .newCommandQueue()
                .context("stream command queue")?,
        );
        ids.push(MetalTensor::zeros_f32(ctx, vec![1])?);
        argmax.push(MetalTensor::zeros_f32(ctx, vec![1])?);
    }

    let mut prev_tokens = vec![0i32; streams];
    let mut encode_ms = Vec::with_capacity(window);
    let mut wait_ms = Vec::with_capacity(window);
    let mut gpu_span_ms = Vec::with_capacity(window);
    let mut gpu_sum_ms = Vec::with_capacity(window);
    let total_t = Instant::now();

    for step in 0..window {
        let pos = target_ctx as u32 + step as u32;
        let encode_t = Instant::now();
        let mut cmds = Vec::with_capacity(streams);
        for stream_idx in 0..streams {
            unsafe {
                let ptr = ids[stream_idx].buffer.contents().as_ptr() as *mut i32;
                *ptr = prev_tokens[stream_idx];
            }
            let cmd = queues[stream_idx]
                .commandBuffer()
                .context("stream command buffer")?;
            let enc = KernelEncoder::begin(&cmd);
            mf.encode_single_token_argmax(
                &enc,
                pos,
                &mut sessions[stream_idx],
                &ids[stream_idx],
                &argmax[stream_idx],
            )?;
            enc.end();
            cmd.commit();
            cmds.push(cmd);
        }
        encode_ms.push(encode_t.elapsed().as_secs_f64() * 1e3);

        let wait_t = Instant::now();
        for cmd in &cmds {
            cmd.waitUntilCompleted();
        }
        wait_ms.push(wait_t.elapsed().as_secs_f64() * 1e3);

        let mut min_start = f64::INFINITY;
        let mut max_end = 0.0f64;
        let mut sum = 0.0f64;
        for cmd in &cmds {
            let start = cmd.GPUStartTime();
            let end = cmd.GPUEndTime();
            min_start = min_start.min(start);
            max_end = max_end.max(end);
            sum += (end - start) * 1e3;
        }
        gpu_span_ms.push((max_end - min_start) * 1e3);
        gpu_sum_ms.push(sum);

        for stream_idx in 0..streams {
            prev_tokens[stream_idx] = unsafe {
                let src = argmax[stream_idx].buffer.contents().as_ptr() as *const i32;
                *src
            };
        }
    }

    let total_ms = total_t.elapsed().as_secs_f64() * 1e3;
    let aggregate_tokens = streams * window;
    let avg_total = total_ms / window as f64;
    let avg_span = gpu_span_ms.iter().sum::<f64>() / window as f64;
    let avg_sum = gpu_sum_ms.iter().sum::<f64>() / window as f64;
    let avg_enc = encode_ms.iter().sum::<f64>() / window as f64;
    let avg_wait = wait_ms.iter().sum::<f64>() / window as f64;
    eprintln!(
        "[decode-window] ctx={} window={} streams={} avg_step={:.2} ms agg_t/s={:.1} per_stream_t/s={:.1}",
        target_ctx,
        window,
        streams,
        avg_total,
        aggregate_tokens as f64 / (total_ms * 1e-3),
        window as f64 / (total_ms * 1e-3)
    );
    eprintln!(
        "[decode-window] streams med_gpu_span={:.2} ms med_gpu_sum={:.2} ms med_cpu_enc={:.2} ms med_wait={:.2} ms",
        median(&gpu_span_ms),
        median(&gpu_sum_ms),
        median(&encode_ms),
        median(&wait_ms),
    );
    eprintln!(
        "[decode-window] streams avg_gpu_span={:.2} ms avg_gpu_sum={:.2} ms avg_cpu_enc={:.2} ms avg_wait={:.2} ms overlap_eff={:.2}x",
        avg_span,
        avg_sum,
        avg_enc,
        avg_wait,
        if avg_span > 0.0 {
            avg_sum / avg_span
        } else {
            0.0
        }
    );
    Ok(())
}
