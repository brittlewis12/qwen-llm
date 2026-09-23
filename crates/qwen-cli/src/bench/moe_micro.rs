//! MoE router, down, gate/up, and batch-sweep microbenchmarks.

use super::*;

pub(crate) fn run_decode_moe_router_repack_check(
    args: DecodeMoeRouterRepackCheckArgs,
) -> Result<()> {
    let DecodeMoeRouterRepackCheckArgs {
        model,
        file,
        tokens,
        mut contexts,
    } = args;
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if file.is_empty() {
        return Err(anyhow!("--file must include at least one prompt"));
    }
    if file.len() < tokens {
        return Err(anyhow!(
            "got {} --file entries, need at least --tokens {tokens}",
            file.len()
        ));
    }
    if contexts.is_empty() {
        return Err(anyhow!("--context must include at least one entry"));
    }
    contexts.sort_unstable();
    contexts.dedup();

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    if mm.arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!(
            "decode-moe-router-repack-check currently requires an MoE model"
        ));
    }
    let mf = MetalForward::new(&ctx, &mm);
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
    for (path, ids) in file.iter().zip(prompt_ids.iter()).take(tokens) {
        if ids.len() <= max_context {
            return Err(anyhow!(
                "prompt {} has {} tokens, need at least {} for context sweep",
                path.display(),
                ids.len(),
                max_context + 1
            ));
        }
    }

    let h = mm.arch.hidden_size as usize;
    let topk = mm.arch.expert_used_count.min(mm.arch.expert_count) as usize;
    let n_expert = mm.arch.expert_count as usize;
    let first_moe = (0..mm.blocks.len())
        .find_map(|i| moe_for_block(&mm, i).ok().flatten())
        .ok_or_else(|| anyhow!("model has no MoE FFN blocks"))?;
    let router_dtype = first_moe.gate_inp.dtype;

    println!(
        "[decode-moe-router-repack-check] model={} files={} contexts={} slots={} blocks={} topk={} experts={} router_dtype={:?} env_QWEN_MOE_ROUTER_F16={}",
        model.display(),
        file.len(),
        contexts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(","),
        tokens,
        mm.blocks.len(),
        topk,
        n_expert,
        router_dtype,
        env_flag_enabled("QWEN_MOE_ROUTER_F16"),
    );
    println!(
        "context\tslots\troute_checks\troute_order_mismatches\troute_set_mismatches\tfirst_set_mismatch\tmax_logit_abs\tmax_logit_rms\tmin_f32_margin\tmin_gpu_margin\tmax_weight_abs\tmax_shared_abs"
    );

    for &context in &contexts {
        let kv_capacity = context + 32;
        let mut sessions = Vec::with_capacity(tokens);
        for (slot, ids) in prompt_ids.iter().enumerate().take(tokens) {
            let mut s = MetalSession::fresh(&ctx, &mm, kv_capacity)
                .with_context(|| format!("router check session slot {slot}"))?;
            for (p, &token_id) in ids.iter().take(context).enumerate() {
                let _ = mf.single_token_argmax_profiled(token_id, p as u32, &mut s)?;
            }
            seed_session_current_token(&ctx, &mm, &mut s, ids[context], h)?;
            sessions.push(s);
        }

        let mut route_checks = 0usize;
        let mut order_mismatches = 0usize;
        let mut set_mismatches = 0usize;
        let mut first_set_mismatch: Option<(usize, usize)> = None;
        let mut max_logit_abs = 0.0f32;
        let mut max_logit_rms = 0.0f64;
        let mut min_f32_margin = f32::INFINITY;
        let mut min_gpu_margin = f32::INFINITY;
        let mut max_weight_abs = 0.0f32;
        let mut max_shared_abs = 0.0f32;

        for block_i in 0..mm.blocks.len() {
            let Some(moe) = moe_for_block(&mm, block_i)? else {
                continue;
            };
            for (slot, session) in sessions.iter_mut().enumerate() {
                let cmd = ctx.queue.commandBuffer().context("router mixer cmd")?;
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_mixer_prep_by_index(&enc, block_i, context as u32, session)?;
                enc.end();
                cmd.commit();
                qwen_llm::metal::wait_completed(&cmd)?;

                let h_cpu = read_f32_tensor_prefix(&session.h, h);
                let f32_route = cpu_route_fingerprint(moe, &h_cpu, topk, n_expert);

                let cmd = ctx.queue.commandBuffer().context("router route cmd")?;
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_route_prepare_by_index(&enc, block_i, session)?;
                enc.end();
                cmd.commit();
                qwen_llm::metal::wait_completed(&cmd)?;

                let gpu_route = read_route_fingerprint(session, topk, n_expert);
                route_checks += 1;
                if f32_route.idx != gpu_route.idx {
                    order_mismatches += 1;
                }
                if !same_i32_set(&f32_route.idx, &gpu_route.idx) {
                    set_mismatches += 1;
                    first_set_mismatch.get_or_insert((block_i, slot));
                }
                max_logit_abs =
                    max_logit_abs.max(f32_max_abs_delta(&f32_route.logits, &gpu_route.logits));
                max_logit_rms =
                    max_logit_rms.max(f32_rms_delta(&f32_route.logits, &gpu_route.logits));
                min_f32_margin = min_f32_margin.min(f32_route.logit_margin);
                min_gpu_margin = min_gpu_margin.min(gpu_route.logit_margin);
                max_weight_abs =
                    max_weight_abs.max(route_weight_max_abs(&f32_route.weight, &gpu_route.weight));
                max_shared_abs =
                    max_shared_abs.max((f32_route.shared_gate - gpu_route.shared_gate).abs());

                let cmd = ctx.queue.commandBuffer().context("router ffn cmd")?;
                let enc = KernelEncoder::begin(&cmd);
                mf.encode_moe_ffn_after_mixer_by_index(&enc, block_i, session)?;
                enc.end();
                cmd.commit();
                qwen_llm::metal::wait_completed(&cmd)?;
            }
        }

        let first_set = first_set_mismatch
            .map(|(block, slot)| format!("block={block},slot={slot}"))
            .unwrap_or_else(|| "none".to_string());
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
            context,
            tokens,
            route_checks,
            order_mismatches,
            set_mismatches,
            first_set,
            max_logit_abs,
            max_logit_rms,
            min_f32_margin,
            min_gpu_margin,
            max_weight_abs,
            max_shared_abs,
        );
    }

    Ok(())
}

pub(crate) fn run_moe_down_micro(args: MoeDownMicroArgs) -> Result<()> {
    let MoeDownMicroArgs {
        model,
        iters,
        warmup,
        tokens,
        fused_routed_q4q5,
        route_capture_ctx,
        route_capture_token_pattern,
        synthetic_f_exp,
        synthetic_h,
        synthetic_layers,
        k512_r2,
        legacy_k512,
        check_k512_r2,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }
    if fused_routed_q4q5 && tokens != 1 {
        return Err(anyhow!("--fused-routed-q4q5 currently requires --tokens 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    if arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!("moe-down-micro requires an MoE model"));
    }
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let moe_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        })
        .collect();
    let q5_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.down_exps.dtype == GgmlType::Q5_K)
        .collect();
    let q4q5_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| {
            moe.gate_exps.dtype == GgmlType::Q4_K
                && moe.up_exps.dtype == GgmlType::Q4_K
                && moe.down_exps.dtype == GgmlType::Q5_K
        })
        .collect();
    let bench_moes = if fused_routed_q4q5 {
        &q4q5_moes
    } else {
        &q5_moes
    };
    if bench_moes.is_empty() {
        let wanted = if fused_routed_q4q5 {
            "Q4_K/Q4_K/Q5_K routed expert banks"
        } else {
            "Q5_K routed down expert banks"
        };
        return Err(anyhow!("model has no eligible {wanted}"));
    }
    if synthetic_f_exp.is_some() != synthetic_h.is_some() {
        return Err(anyhow!(
            "--synthetic-f-exp and --synthetic-h must be passed together"
        ));
    }
    let synthetic = synthetic_f_exp.is_some();
    if fused_routed_q4q5 && synthetic {
        return Err(anyhow!(
            "--fused-routed-q4q5 requires real Q4/Q5 expert banks"
        ));
    }
    if synthetic && route_capture_ctx.is_some() {
        return Err(anyhow!(
            "--route-capture-ctx is only supported for real routed-down layers"
        ));
    }
    let f_run = synthetic_f_exp.unwrap_or(f_exp);
    let h_run = synthetic_h.unwrap_or(h);
    let layer_count = synthetic_layers.unwrap_or(bench_moes.len());
    if synthetic && layer_count == 0 {
        return Err(anyhow!("--synthetic-layers must be >= 1"));
    }
    if f_run % 256 != 0 {
        return Err(anyhow!("routed-down f_exp must be divisible by 256"));
    }
    let synthetic_weight = if synthetic {
        Some(MetalTensor::zeros_dtype(
            &ctx,
            vec![f_run as u64, h_run as u64, n_expert as u64, 1],
            GgmlType::Q5_K,
        )?)
    } else {
        None
    };

    let slots = tokens * topk;
    let inner = MetalTensor::zeros_f32(&ctx, vec![(slots * f_run) as u64])?;
    let x = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
    let topk_idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
    let topk_w = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
    let out = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
    unsafe {
        let x_ptr = x.buffer.contents().as_ptr() as *mut f32;
        for i in 0..(tokens * h_run) {
            *x_ptr.add(i) = ((i % 31) as f32 - 15.0) * 0.01;
        }
        let inner_ptr = inner.buffer.contents().as_ptr() as *mut f32;
        for i in 0..(slots * f_run) {
            *inner_ptr.add(i) = ((i % 17) as f32 - 8.0) * 0.0125;
        }
        let idx_ptr = topk_idx.buffer.contents().as_ptr() as *mut i32;
        let w_ptr = topk_w.buffer.contents().as_ptr() as *mut f32;
        for slot in 0..slots {
            *idx_ptr.add(slot) = ((slot * 17) % n_expert) as i32;
            *w_ptr.add(slot) = 1.0 / topk as f32;
        }
    }

    let mut captured_route_stats = None;
    let captured_routes: Option<CapturedDownRouteTensors> = if let Some(capture_ctx) =
        route_capture_ctx
    {
        let mf = MetalForward::new(&ctx, &mm);
        let mut capture_s = MetalSession::fresh(&ctx, &mm, capture_ctx + tokens + 16)
            .context("route-capture session")?;
        for pos in 0..capture_ctx {
            let _ = mf.single_token(0, pos as u32, &mut capture_s)?;
        }
        let mut all_routes_by_token = Vec::with_capacity(tokens);
        for tok in 0..tokens {
            let token_id = capture_replay_token(route_capture_token_pattern, tok, arch.vocab_size);
            let all_routes = mf.capture_moe_gateup_replay_for_token(
                token_id,
                (capture_ctx + tok) as u32,
                &mut capture_s,
            )?;
            if all_routes.len() != moe_blocks.len() {
                return Err(anyhow!(
                    "captured {} MoE route rows, expected {}",
                    all_routes.len(),
                    moe_blocks.len()
                ));
            }
            all_routes_by_token.push(all_routes);
        }
        let eligible_indices: Vec<_> = moe_blocks
            .iter()
            .enumerate()
            .filter_map(|(i, moe)| {
                let eligible = if fused_routed_q4q5 {
                    moe.gate_exps.dtype == GgmlType::Q4_K
                        && moe.up_exps.dtype == GgmlType::Q4_K
                        && moe.down_exps.dtype == GgmlType::Q5_K
                } else {
                    moe.down_exps.dtype == GgmlType::Q5_K
                };
                if eligible { Some(i) } else { None }
            })
            .collect();
        if eligible_indices.len() != bench_moes.len() {
            return Err(anyhow!(
                "captured {} eligible route rows, expected {}",
                eligible_indices.len(),
                bench_moes.len()
            ));
        }
        captured_route_stats = Some(summarize_moe_route_batch(
            &all_routes_by_token,
            &eligible_indices,
            n_expert,
            topk,
        )?);

        let mut tensors = Vec::with_capacity(eligible_indices.len());
        for &moe_i in &eligible_indices {
            let idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
            let weight = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
            let hidden = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
            unsafe {
                let idx_ptr = idx.buffer.contents().as_ptr() as *mut i32;
                let w_ptr = weight.buffer.contents().as_ptr() as *mut f32;
                let h_ptr = hidden.buffer.contents().as_ptr() as *mut f32;
                for (tok, routes) in all_routes_by_token.iter().enumerate().take(tokens) {
                    let route = &routes[moe_i];
                    if route.topk_idx.len() != topk || route.topk_weight.len() != topk {
                        return Err(anyhow!(
                            "captured route has idx={} weight={}, expected topk={topk}",
                            route.topk_idx.len(),
                            route.topk_weight.len()
                        ));
                    }
                    if route.hidden.len() != h_run {
                        return Err(anyhow!(
                            "captured hidden has {}, expected h={h_run}",
                            route.hidden.len()
                        ));
                    }
                    std::ptr::copy_nonoverlapping(
                        route.hidden.as_ptr(),
                        h_ptr.add(tok * h_run),
                        h_run,
                    );
                    for slot in 0..topk {
                        let expert = route.topk_idx[slot];
                        if expert < 0 || expert as usize >= n_expert {
                            return Err(anyhow!(
                                "captured expert id {expert} outside n_expert={n_expert}"
                            ));
                        }
                        let out_slot = tok * topk + slot;
                        *idx_ptr.add(out_slot) = expert;
                        *w_ptr.add(out_slot) = route.topk_weight[slot];
                    }
                }
            }
            tensors.push((idx, weight, hidden));
        }
        Some((capture_ctx, tensors))
    } else {
        None
    };

    let active_weight_bytes: f64 = if let Some(weight) = synthetic_weight.as_ref() {
        weight.n_bytes() as f64 * layer_count as f64 * tokens as f64 * topk as f64 / n_expert as f64
    } else if fused_routed_q4q5 {
        bench_moes
            .iter()
            .map(|moe| {
                (moe.gate_exps.n_bytes() + moe.up_exps.n_bytes() + moe.down_exps.n_bytes()) as f64
                    * topk as f64
                    * tokens as f64
                    / n_expert as f64
            })
            .sum()
    } else {
        bench_moes
            .iter()
            .map(|moe| {
                moe.down_exps.n_bytes() as f64 * tokens as f64 * topk as f64 / n_expert as f64
            })
            .sum()
    };
    let inner_bytes = (slots * f_run * std::mem::size_of::<f32>()) as f64;
    let out_bytes = (tokens * h_run * std::mem::size_of::<f32>()) as f64;
    if k512_r2 && legacy_k512 {
        return Err(anyhow!("--k512-r2 conflicts with --legacy-k512"));
    }
    let use_k512_r2 = k512_r2 || (f_run == 512 && !legacy_k512);
    if use_k512_r2 && f_run != 512 {
        return Err(anyhow!(
            "QWEN_MOE_DOWN_Q5_K512_R2 requires f_exp=512, got {f_run}"
        ));
    }
    if check_k512_r2 {
        if f_run != 512 {
            return Err(anyhow!(
                "QWEN_MOE_DOWN_Q5_K512_R2_CHECK requires f_exp=512, got {f_run}"
            ));
        }
        let ref_out = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
        let alt_out = MetalTensor::zeros_f32(&ctx, vec![(tokens * h_run) as u64])?;
        let cmd = ctx.queue.commandBuffer().context("moe-down check cmd")?;
        let enc = KernelEncoder::begin(&cmd);
        if let Some(weight) = synthetic_weight.as_ref() {
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx, &enc, weight, &inner, &topk_idx, &topk_w, &ref_out, f_run, h_run, n_expert,
                topk, tokens,
            )?;
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                &ctx, &enc, weight, &inner, &topk_idx, &topk_w, &alt_out, f_run, h_run, n_expert,
                topk, tokens,
            )?;
        } else {
            let moe = q5_moes[0];
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                &ctx,
                &enc,
                &moe.down_exps,
                &inner,
                &topk_idx,
                &topk_w,
                &ref_out,
                f_run,
                h_run,
                n_expert,
                topk,
                tokens,
            )?;
            encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                &ctx,
                &enc,
                &moe.down_exps,
                &inner,
                &topk_idx,
                &topk_w,
                &alt_out,
                f_run,
                h_run,
                n_expert,
                topk,
                tokens,
            )?;
        }
        enc.end();
        cmd.commit();
        qwen_llm::metal::wait_completed(&cmd)?;
        let read = |t: &MetalTensor| -> Vec<f32> {
            let n = t.n_elements() as usize;
            let mut xs = vec![0.0f32; n];
            unsafe {
                let src = (t.buffer.contents().as_ptr() as *const f32).add((t.offset / 4) as usize);
                std::ptr::copy_nonoverlapping(src, xs.as_mut_ptr(), n);
            }
            xs
        };
        let a = read(&ref_out);
        let b = read(&alt_out);
        let max_abs = a
            .iter()
            .zip(&b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let dot: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
        let na = a.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let nb = b.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let cos = if na > 0.0 && nb > 0.0 {
            dot / (na * nb)
        } else {
            1.0
        };
        println!("[moe-down-micro-check] max_abs={max_abs:.6} cos={cos:.9}");
    }

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        if fused_routed_q4q5 {
            for (layer_i, moe) in bench_moes.iter().enumerate() {
                let (route_idx, route_w, layer_x) = captured_routes
                    .as_ref()
                    .map(|(_, routes)| (&routes[layer_i].0, &routes[layer_i].1, &routes[layer_i].2))
                    .unwrap_or((&topk_idx, &topk_w, &x));
                encode_moe_fused_routed_q4q5_token_f32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    &moe.down_exps,
                    layer_x,
                    route_idx,
                    route_w,
                    &out,
                    h_run,
                    f_run,
                    n_expert,
                    topk,
                    tokens,
                )?;
            }
        } else if let Some(weight) = synthetic_weight.as_ref() {
            for _ in 0..layer_count {
                if use_k512_r2 {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                        &ctx, enc, weight, &inner, &topk_idx, &topk_w, &out, f_run, h_run,
                        n_expert, topk, tokens,
                    )?;
                } else {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        &ctx, enc, weight, &inner, &topk_idx, &topk_w, &out, f_run, h_run,
                        n_expert, topk, tokens,
                    )?;
                }
            }
        } else {
            for (layer_i, moe) in bench_moes.iter().enumerate() {
                let (route_idx, route_w) = captured_routes
                    .as_ref()
                    .map(|(_, routes)| (&routes[layer_i].0, &routes[layer_i].1))
                    .unwrap_or((&topk_idx, &topk_w));
                if use_k512_r2 {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &inner,
                        route_idx,
                        route_w,
                        &out,
                        f_run,
                        h_run,
                        n_expert,
                        topk,
                        tokens,
                    )?;
                } else {
                    encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                        &ctx,
                        enc,
                        &moe.down_exps,
                        &inner,
                        route_idx,
                        route_w,
                        &out,
                        f_run,
                        h_run,
                        n_expert,
                        topk,
                        tokens,
                    )?;
                }
            }
        }
        Ok(())
    })?;

    let weight_gb = active_weight_bytes / 1e9;
    let activation_gb = if fused_routed_q4q5 {
        out_bytes * 2.0 * layer_count as f64 / 1e9
    } else {
        (inner_bytes + out_bytes) * layer_count as f64 / 1e9
    };
    let route_mode = captured_routes
        .as_ref()
        .map(|(ctx, _)| format!("captured(ctx={ctx},pattern={route_capture_token_pattern:?})"))
        .unwrap_or_else(|| "synthetic".to_string());
    println!(
        "[moe-down-micro] model={} mode={} route_mode={} kernel={} q5_layers={} layers={} h={} f_exp={} n_expert={} topk={} tokens={} warmup={} iters={}",
        model.display(),
        if synthetic { "synthetic" } else { "real" },
        route_mode,
        if fused_routed_q4q5 {
            "fused_q4q5"
        } else if use_k512_r2 {
            "k512_r2"
        } else {
            "default"
        },
        bench_moes.len(),
        layer_count,
        h_run,
        f_run,
        n_expert,
        topk,
        tokens,
        warmup,
        iters
    );
    if let Some(stats) = captured_route_stats {
        println!(
            "[moe-down-route-stats] layers={} tokens={} slots_per_layer={} avg_unique_experts={:.2} avg_max_slots={:.2} avg_reuse={:.2}",
            stats.layers,
            stats.tokens,
            stats.slots_per_layer,
            stats.avg_unique_experts,
            stats.avg_max_slots,
            stats.avg_reuse
        );
    }
    println!("phase\tactive_weight_gb\tactivation_gb\tavg_wall_ms\tavg_gpu_ms\tweight_gb_s");
    let phase = if fused_routed_q4q5 {
        "q4q5_fused_routed"
    } else {
        "q5_down_weighted_sum"
    };
    println!(
        "{phase}\t{weight_gb:.4}\t{activation_gb:.4}\t{wall:.4}\t{gpu:.4}\t{:.1}",
        weight_gb / (gpu / 1e3)
    );
    Ok(())
}

pub(crate) fn run_moe_batch_sweep(args: MoeBatchSweepArgs) -> Result<()> {
    let MoeBatchSweepArgs {
        model,
        iters,
        warmup,
        mut tokens,
        route_capture_ctx,
        route_capture_stride,
        file,
        route_capture_token_pattern,
        mut slot_orders,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens.is_empty() || tokens.contains(&0) {
        return Err(anyhow!("--tokens entries must be >= 1"));
    }
    if route_capture_stride == 0 {
        return Err(anyhow!("--route-capture-stride must be >= 1"));
    }
    if slot_orders.is_empty() {
        return Err(anyhow!("--slot-order must contain at least one entry"));
    }
    tokens.sort_unstable();
    tokens.dedup();
    slot_orders.sort_unstable();
    slot_orders.dedup();
    let max_tokens = *tokens.last().expect("non-empty tokens");

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    if arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!("moe-batch-sweep requires an MoE model"));
    }
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let moe_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        })
        .collect();
    let q4_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K)
        .collect();
    let q4_indices: Vec<_> = moe_blocks
        .iter()
        .enumerate()
        .filter_map(|(i, moe)| {
            if moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    let q5_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.down_exps.dtype == GgmlType::Q5_K)
        .collect();
    let q5_indices: Vec<_> = moe_blocks
        .iter()
        .enumerate()
        .filter_map(|(i, moe)| {
            if moe.down_exps.dtype == GgmlType::Q5_K {
                Some(i)
            } else {
                None
            }
        })
        .collect();
    if q4_moes.is_empty() || q5_moes.is_empty() {
        return Err(anyhow!(
            "moe-batch-sweep requires Q4_K gate/up and Q5_K down expert banks"
        ));
    }
    if !f_exp.is_multiple_of(256) {
        return Err(anyhow!("routed f_exp must be divisible by 256"));
    }
    let use_k512_r2 = f_exp == 512;

    let last_capture_pos = route_capture_ctx + (max_tokens - 1) * route_capture_stride;
    let prompt_ids = if file.is_empty() {
        Vec::new()
    } else {
        let tok = NativeTokenizer::from_gguf(&g).context("open native GGUF tokenizer")?;
        let mut all = Vec::with_capacity(file.len());
        for path in &file {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("read {}", path.display()))?;
            let ids = tok
                .encode(&text, false)
                .with_context(|| format!("tokenize {}", path.display()))?;
            all.push(ids);
        }
        all
    };
    if prompt_ids.len() == 1 {
        let need = last_capture_pos + 1;
        if prompt_ids[0].len() < need {
            return Err(anyhow!(
                "prompt file has {} tokens, need at least last capture position + 1 = {need}",
                prompt_ids[0].len()
            ));
        }
    } else if prompt_ids.len() > 1 {
        if prompt_ids.len() < max_tokens {
            return Err(anyhow!(
                "got {} --file entries, need at least max(tokens) = {max_tokens}",
                prompt_ids.len()
            ));
        }
        let need = route_capture_ctx + 1;
        for (path, ids) in file.iter().zip(prompt_ids.iter()) {
            if ids.len() < need {
                return Err(anyhow!(
                    "prompt file {} has {} tokens, need at least route_capture_ctx + 1 = {need}",
                    path.display(),
                    ids.len()
                ));
            }
        }
    }

    let mf = MetalForward::new(&ctx, &mm);
    let mut all_routes_by_token = Vec::with_capacity(max_tokens);
    if prompt_ids.len() > 1 {
        for ids in prompt_ids.iter().take(max_tokens) {
            let mut capture_s = MetalSession::fresh(&ctx, &mm, route_capture_ctx + 17)
                .context("route-capture session")?;
            for (pos, &token_id) in ids.iter().take(route_capture_ctx).enumerate() {
                let _ = mf.single_token(token_id, pos as u32, &mut capture_s)?;
            }
            let all_routes = mf.capture_moe_gateup_replay_for_token(
                ids[route_capture_ctx],
                route_capture_ctx as u32,
                &mut capture_s,
            )?;
            if all_routes.len() != moe_blocks.len() {
                return Err(anyhow!(
                    "captured {} MoE route rows, expected {}",
                    all_routes.len(),
                    moe_blocks.len()
                ));
            }
            all_routes_by_token.push(all_routes);
        }
    } else {
        let mut capture_s = MetalSession::fresh(&ctx, &mm, last_capture_pos + 17)
            .context("route-capture session")?;
        for pos in 0..route_capture_ctx {
            let token_id = prompt_ids.first().map(|ids| ids[pos]).unwrap_or(0);
            let _ = mf.single_token(token_id, pos as u32, &mut capture_s)?;
        }
        let mut next_pos = route_capture_ctx;
        for tok in 0..max_tokens {
            let target_pos = route_capture_ctx + tok * route_capture_stride;
            for pos in next_pos..target_pos {
                let token_id = prompt_ids.first().map(|ids| ids[pos]).unwrap_or(0);
                let _ = mf.single_token(token_id, pos as u32, &mut capture_s)?;
            }
            let token_id = prompt_ids
                .first()
                .map(|ids| ids[target_pos])
                .unwrap_or_else(|| {
                    capture_replay_token(route_capture_token_pattern, tok, arch.vocab_size)
                });
            let all_routes = mf.capture_moe_gateup_replay_for_token(
                token_id,
                target_pos as u32,
                &mut capture_s,
            )?;
            if all_routes.len() != moe_blocks.len() {
                return Err(anyhow!(
                    "captured {} MoE route rows, expected {}",
                    all_routes.len(),
                    moe_blocks.len()
                ));
            }
            all_routes_by_token.push(all_routes);
            next_pos = target_pos + 1;
        }
    }

    let gateup_weight_gb_per_token: f64 = q4_moes
        .iter()
        .map(|moe| (moe.gate_exps.n_bytes() + moe.up_exps.n_bytes()) as f64)
        .sum::<f64>()
        * topk as f64
        / n_expert as f64
        / 1e9;
    let down_weight_gb_per_token: f64 = q5_moes
        .iter()
        .map(|moe| moe.down_exps.n_bytes() as f64)
        .sum::<f64>()
        * topk as f64
        / n_expert as f64
        / 1e9;

    println!(
        "[moe-batch-sweep] model={} route_mode={} slot_orders={} q4_layers={} q5_layers={} h={} f_exp={} n_expert={} topk={} warmup={} iters={}",
        model.display(),
        if file.len() > 1 {
            format!(
                "captured(ctx={},independent_files={})",
                route_capture_ctx,
                file.len()
            )
        } else if let Some(path) = file.first() {
            format!(
                "captured(ctx={},stride={},file={})",
                route_capture_ctx,
                route_capture_stride,
                path.display()
            )
        } else {
            format!(
                "captured(ctx={},stride={},pattern={:?})",
                route_capture_ctx, route_capture_stride, route_capture_token_pattern
            )
        },
        slot_orders
            .iter()
            .map(|order| order.label())
            .collect::<Vec<_>>()
            .join(","),
        q4_moes.len(),
        q5_moes.len(),
        h,
        f_exp,
        n_expert,
        topk,
        warmup,
        iters
    );
    println!(
        "slot_order\ttokens\tgateup_gpu_ms\tdown_gpu_ms\tcombined_ms_per_token\tgateup_gb_s\tdown_gb_s\tq4_unique\tq4_max\tq4_reuse\tq5_unique\tq5_max\tq5_reuse"
    );

    for &n_tokens in &tokens {
        let routes = &all_routes_by_token[..n_tokens];
        let q4_stats = summarize_moe_route_batch(routes, &q4_indices, n_expert, topk)?;
        let q5_stats = summarize_moe_route_batch(routes, &q5_indices, n_expert, topk)?;
        let slots = n_tokens * topk;
        for &slot_order in &slot_orders {
            let gateup_inputs =
                captured_gateup_tensors(&ctx, routes, &q4_indices, h, n_expert, topk, slot_order)?;
            let down_inputs =
                captured_down_tensors(&ctx, routes, &q5_indices, n_expert, topk, slot_order)?;
            let inner = MetalTensor::zeros_f32(&ctx, vec![(slots * f_exp) as u64])?;
            let down_out = MetalTensor::zeros_f32(&ctx, vec![(n_tokens * h) as u64])?;
            unsafe {
                let inner_ptr = inner.buffer.contents().as_ptr() as *mut f32;
                for i in 0..(slots * f_exp) {
                    *inner_ptr.add(i) = ((i % 17) as f32 - 8.0) * 0.0125;
                }
            }

            let (_gateup_wall, gateup_gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
                for (layer_i, moe) in q4_moes.iter().enumerate() {
                    let (layer_x, route_idx) =
                        (&gateup_inputs[layer_i].0, &gateup_inputs[layer_i].1);
                    if n_tokens == 1 {
                        encode_moe_swiglu_q4_K_f32(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            layer_x,
                            route_idx,
                            &inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                        )?;
                    } else {
                        encode_moe_swiglu_q4_K_f32_packed_slots(
                            &ctx,
                            enc,
                            &moe.gate_exps,
                            &moe.up_exps,
                            layer_x,
                            route_idx,
                            &inner,
                            h,
                            f_exp,
                            n_expert,
                            topk,
                            n_tokens,
                        )?;
                    }
                }
                Ok(())
            })?;

            let (_down_wall, down_gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
                for (layer_i, moe) in q5_moes.iter().enumerate() {
                    let (route_idx, route_w) = (&down_inputs[layer_i].0, &down_inputs[layer_i].1);
                    if use_k512_r2 {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots_k512_r2(
                            &ctx,
                            enc,
                            &moe.down_exps,
                            &inner,
                            route_idx,
                            route_w,
                            &down_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            n_tokens,
                        )?;
                    } else {
                        encode_moe_down_weighted_sum_q5_K_f32_packed_slots(
                            &ctx,
                            enc,
                            &moe.down_exps,
                            &inner,
                            route_idx,
                            route_w,
                            &down_out,
                            f_exp,
                            h,
                            n_expert,
                            topk,
                            n_tokens,
                        )?;
                    }
                }
                Ok(())
            })?;

            let combined_per_token = (gateup_gpu + down_gpu) / n_tokens as f64;
            let gateup_gb_s = gateup_weight_gb_per_token * n_tokens as f64 / (gateup_gpu / 1e3);
            let down_gb_s = down_weight_gb_per_token * n_tokens as f64 / (down_gpu / 1e3);
            println!(
                "{}\t{}\t{:.4}\t{:.4}\t{:.4}\t{:.1}\t{:.1}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}\t{:.2}",
                slot_order.label(),
                n_tokens,
                gateup_gpu,
                down_gpu,
                combined_per_token,
                gateup_gb_s,
                down_gb_s,
                q4_stats.avg_unique_experts,
                q4_stats.avg_max_slots,
                q4_stats.avg_reuse,
                q5_stats.avg_unique_experts,
                q5_stats.avg_max_slots,
                q5_stats.avg_reuse
            );
        }
    }

    Ok(())
}

pub(crate) fn run_moe_gateup_micro(args: MoeGateupMicroArgs) -> Result<()> {
    let MoeGateupMicroArgs {
        model,
        iters,
        warmup,
        tokens,
        route_capture_ctx,
        route_capture_token_pattern,
    } = args;
    if iters == 0 {
        return Err(anyhow!("--iters must be >= 1"));
    }
    if tokens == 0 {
        return Err(anyhow!("--tokens must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model")?;
    let arch = &mm.arch;
    if arch.kind != qwen_llm::model::ArchKind::Moe {
        return Err(anyhow!("moe-gateup-micro requires an MoE model"));
    }
    let h = arch.hidden_size as usize;
    let f_exp = arch.expert_feed_forward_length as usize;
    let n_expert = arch.expert_count as usize;
    let topk = arch.expert_used_count.min(arch.expert_count) as usize;
    let moe_blocks: Vec<_> = mm
        .blocks
        .iter()
        .filter_map(|b| match b {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        })
        .collect();
    let q4_moes: Vec<_> = moe_blocks
        .iter()
        .copied()
        .filter(|moe| moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K)
        .collect();
    if q4_moes.is_empty() {
        return Err(anyhow!("model has no Q4_K routed gate/up expert banks"));
    }

    let slots = tokens * topk;
    let x = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;
    let topk_idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
    let inner = MetalTensor::zeros_f32(&ctx, vec![(slots * f_exp) as u64])?;
    unsafe {
        let x_ptr = x.buffer.contents().as_ptr() as *mut f32;
        for i in 0..(tokens * h) {
            *x_ptr.add(i) = ((i % 31) as f32 - 15.0) * 0.01;
        }
        let idx_ptr = topk_idx.buffer.contents().as_ptr() as *mut i32;
        for slot in 0..slots {
            *idx_ptr.add(slot) = ((slot * 17) % n_expert) as i32;
        }
    }

    let mut captured_route_stats = None;
    let captured_inputs: Option<(usize, Vec<(MetalTensor, MetalTensor)>)> =
        if let Some(capture_ctx) = route_capture_ctx {
            let mf = MetalForward::new(&ctx, &mm);
            let mut capture_s = MetalSession::fresh(&ctx, &mm, capture_ctx + tokens + 16)
                .context("route-capture session")?;
            for pos in 0..capture_ctx {
                let _ = mf.single_token(0, pos as u32, &mut capture_s)?;
            }
            let mut all_routes_by_token = Vec::with_capacity(tokens);
            for tok in 0..tokens {
                let token_id =
                    capture_replay_token(route_capture_token_pattern, tok, arch.vocab_size);
                let all_routes = mf.capture_moe_gateup_replay_for_token(
                    token_id,
                    (capture_ctx + tok) as u32,
                    &mut capture_s,
                )?;
                if all_routes.len() != moe_blocks.len() {
                    return Err(anyhow!(
                        "captured {} MoE route rows, expected {}",
                        all_routes.len(),
                        moe_blocks.len()
                    ));
                }
                all_routes_by_token.push(all_routes);
            }
            let q4_indices: Vec<_> = moe_blocks
                .iter()
                .enumerate()
                .filter_map(|(i, moe)| {
                    if moe.gate_exps.dtype == GgmlType::Q4_K && moe.up_exps.dtype == GgmlType::Q4_K
                    {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();
            if q4_indices.len() != q4_moes.len() {
                return Err(anyhow!(
                    "captured {} Q4 route rows, expected {}",
                    q4_indices.len(),
                    q4_moes.len()
                ));
            }
            captured_route_stats = Some(summarize_moe_route_batch(
                &all_routes_by_token,
                &q4_indices,
                n_expert,
                topk,
            )?);

            let mut tensors = Vec::with_capacity(q4_indices.len());
            for &moe_i in &q4_indices {
                let hidden_t = MetalTensor::zeros_f32(&ctx, vec![(tokens * h) as u64])?;
                let idx = MetalTensor::zeros_f32(&ctx, vec![slots as u64])?;
                unsafe {
                    let dst = hidden_t.buffer.contents().as_ptr() as *mut f32;
                    let ptr = idx.buffer.contents().as_ptr() as *mut i32;
                    for (tok, routes) in all_routes_by_token.iter().enumerate().take(tokens) {
                        let route = &routes[moe_i];
                        if route.topk_idx.len() != topk {
                            return Err(anyhow!(
                                "captured route has {} experts, expected topk={topk}",
                                route.topk_idx.len()
                            ));
                        }
                        if route.hidden.len() != h {
                            return Err(anyhow!(
                                "captured hidden has {} elements, expected h={h}",
                                route.hidden.len()
                            ));
                        }
                        std::ptr::copy_nonoverlapping(route.hidden.as_ptr(), dst.add(tok * h), h);
                        for (slot, &expert) in route.topk_idx.iter().enumerate() {
                            if expert < 0 || expert as usize >= n_expert {
                                return Err(anyhow!(
                                    "captured expert id {expert} outside n_expert={n_expert}"
                                ));
                            }
                            *ptr.add(tok * topk + slot) = expert;
                        }
                    }
                }
                tensors.push((hidden_t, idx));
            }
            Some((capture_ctx, tensors))
        } else {
            None
        };

    let active_weight_bytes: f64 = q4_moes
        .iter()
        .map(|moe| {
            (moe.gate_exps.n_bytes() + moe.up_exps.n_bytes()) as f64 * topk as f64 * tokens as f64
                / n_expert as f64
        })
        .sum();
    let activation_gb = ((tokens * h + slots * f_exp) * std::mem::size_of::<f32>()) as f64
        * q4_moes.len() as f64
        / 1e9;

    let (wall, gpu) = time_gpu_reps(&ctx, warmup, iters, |enc| {
        for (layer_i, moe) in q4_moes.iter().enumerate() {
            let (layer_x, route_idx) = captured_inputs
                .as_ref()
                .map(|(_, idx)| (&idx[layer_i].0, &idx[layer_i].1))
                .unwrap_or((&x, &topk_idx));
            if tokens == 1 {
                encode_moe_swiglu_q4_K_f32(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    layer_x,
                    route_idx,
                    &inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                )?;
            } else {
                encode_moe_swiglu_q4_K_f32_packed_slots(
                    &ctx,
                    enc,
                    &moe.gate_exps,
                    &moe.up_exps,
                    layer_x,
                    route_idx,
                    &inner,
                    h,
                    f_exp,
                    n_expert,
                    topk,
                    tokens,
                )?;
            }
        }
        Ok(())
    })?;

    let weight_gb = active_weight_bytes / 1e9;
    let route_mode = captured_inputs
        .as_ref()
        .map(|(ctx, _)| format!("captured(ctx={ctx},pattern={route_capture_token_pattern:?})"))
        .unwrap_or_else(|| "synthetic".to_string());
    println!(
        "[moe-gateup-micro] model={} mode={} q4_layers={} h={} f_exp={} n_expert={} topk={} tokens={} warmup={} iters={}",
        model.display(),
        route_mode,
        q4_moes.len(),
        h,
        f_exp,
        n_expert,
        topk,
        tokens,
        warmup,
        iters
    );
    if let Some(stats) = captured_route_stats {
        println!(
            "[moe-gateup-route-stats] layers={} tokens={} slots_per_layer={} avg_unique_experts={:.2} avg_max_slots={:.2} avg_reuse={:.2}",
            stats.layers,
            stats.tokens,
            stats.slots_per_layer,
            stats.avg_unique_experts,
            stats.avg_max_slots,
            stats.avg_reuse
        );
    }
    println!("phase\tactive_weight_gb\tactivation_gb\tavg_wall_ms\tavg_gpu_ms\tweight_gb_s");
    println!(
        "q4_gateup_swiglu\t{weight_gb:.4}\t{activation_gb:.4}\t{wall:.4}\t{gpu:.4}\t{:.1}",
        weight_gb / (gpu / 1e3)
    );
    Ok(())
}
