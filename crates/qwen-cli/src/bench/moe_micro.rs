//! MoE router, down, gate/up, and batch-sweep microbenchmarks.

use super::*;

fn comparison_max_abs(a: &[f32], b: &[f32]) -> f32 {
    qwen_llm::compare::report_max_abs_diff_f32(a.iter(), b.iter())
}

fn report_max_f32(current: f32, candidate: f32) -> f32 {
    if current.is_finite() && candidate.is_finite() {
        current.max(candidate)
    } else {
        f32::NAN
    }
}

fn report_max_f64(current: f64, candidate: f64) -> f64 {
    if current.is_finite() && candidate.is_finite() {
        current.max(candidate)
    } else {
        f64::NAN
    }
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
        let max_abs = comparison_max_abs(&a, &b);
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
        route_census_only,
        route_plan,
        route_rows_per_agent,
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
    if route_census_only {
        if file.len() != 1 {
            return Err(anyhow!(
                "--route-census-only requires exactly one --file token stream"
            ));
        }
        let plan_path = route_plan
            .as_ref()
            .ok_or_else(|| anyhow!("--route-census-only requires --route-plan"))?;
        if route_rows_per_agent == 0 {
            return Err(anyhow!("--route-rows-per-agent must be positive"));
        }
        let plan: serde_json::Value = serde_json::from_slice(
            &std::fs::read(plan_path).with_context(|| format!("read {}", plan_path.display()))?,
        )
        .with_context(|| format!("parse {}", plan_path.display()))?;
        let agents = plan["agents"]
            .as_array()
            .ok_or_else(|| anyhow!("route plan needs an agents array"))?;
        if agents.is_empty() {
            return Err(anyhow!("route plan has no agents"));
        }
        if route_rows_per_agent < 2048 {
            return Err(anyhow!(
                "route-census needs at least 2,048 rows per agent for the crossover grid"
            ));
        }
        let tokenizer = NativeTokenizer::from_gguf(&g).context("open native GGUF tokenizer")?;
        let text = std::fs::read_to_string(&file[0])
            .with_context(|| format!("read {}", file[0].display()))?;
        let ids = tokenizer
            .encode(&text, false)
            .with_context(|| format!("tokenize {}", file[0].display()))?;
        let roots_end = route_capture_ctx
            .checked_mul(agents.len())
            .ok_or_else(|| anyhow!("route plan root length overflow"))?;
        if ids.len() <= roots_end + 2048 {
            return Err(anyhow!(
                "token stream has {} tokens; route plan roots need {} plus suffix material",
                ids.len(),
                roots_end
            ));
        }
        let mf = MetalForward::new(&ctx, &mm);
        let mut counts_by_agent = vec![vec![vec![0u64; n_expert]; moe_blocks.len()]; agents.len()];
        let mut counts_by_agent_by_width =
            vec![std::collections::BTreeMap::<usize, Vec<Vec<u64>>>::new(); agents.len()];
        let mut rows_by_agent = vec![0usize; agents.len()];
        let mut calls_by_agent = vec![0usize; agents.len()];
        let capture_widths = [32usize, 161, 512, 2048];
        for (agent_index, agent_plan) in agents.iter().enumerate() {
            let calls = agent_plan["calls"]
                .as_array()
                .ok_or_else(|| anyhow!("route plan agent {agent_index} needs calls"))?;
            let root_start = agent_index * route_capture_ctx;
            let capacity = route_capture_ctx + route_rows_per_agent + 17;
            let mut session = MetalSession::fresh(&ctx, &mm, capacity)
                .with_context(|| format!("route-census agent {agent_index} session"))?;
            let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, 1024, route_capture_ctx)
                .context("route-census packed-root scratch")?;
            prefill_tokens_with_multi_hidden(
                &mf,
                &ids[root_start..root_start + route_capture_ctx],
                0,
                &mut session,
                &mut scratch,
                &[],
                None,
            )
            .with_context(|| format!("route-census agent {agent_index} root prefill"))?;
            let mut position = route_capture_ctx;
            'calls: for call in calls {
                let width = call["append_tokens"]
                    .as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| anyhow!("route plan call has invalid append_tokens"))?;
                if width == 0 {
                    continue;
                }
                let available = ids.len() - width;
                let suffix_start = (agent_index * 977) % available;
                let take = width.min(route_rows_per_agent - rows_by_agent[agent_index]);
                for &token_id in &ids[suffix_start..suffix_start + take] {
                    let routes = mf.capture_moe_gateup_replay_for_token(
                        token_id,
                        position as u32,
                        &mut session,
                    )?;
                    if routes.len() != moe_blocks.len() {
                        return Err(anyhow!(
                            "captured {} route rows, expected {} MoE blocks",
                            routes.len(),
                            moe_blocks.len()
                        ));
                    }
                    for (block, route) in routes.iter().enumerate() {
                        if route.topk_idx.len() != topk {
                            return Err(anyhow!(
                                "block {block} captured {} routes, expected topk={topk}",
                                route.topk_idx.len()
                            ));
                        }
                        for &expert in &route.topk_idx {
                            if expert < 0 || expert as usize >= n_expert {
                                return Err(anyhow!(
                                    "block {block} returned invalid expert id {expert}"
                                ));
                            }
                            counts_by_agent[agent_index][block][expert as usize] += 1;
                        }
                    }
                    rows_by_agent[agent_index] += 1;
                    position += 1;
                    if capture_widths.contains(&rows_by_agent[agent_index]) {
                        counts_by_agent_by_width[agent_index].insert(
                            rows_by_agent[agent_index],
                            counts_by_agent[agent_index].clone(),
                        );
                    }
                }
                calls_by_agent[agent_index] += 1;
                if rows_by_agent[agent_index] >= route_rows_per_agent {
                    break 'calls;
                }
            }
        }
        let weights = moe_blocks
            .iter()
            .enumerate()
            .map(|(block, moe)| {
                serde_json::json!({
                    "moe_index": block,
                    "gate_dtype": format!("{:?}", moe.gate_exps.dtype),
                    "up_dtype": format!("{:?}", moe.up_exps.dtype),
                    "down_dtype": format!("{:?}", moe.down_exps.dtype),
                    "gate_bytes_per_expert": moe.gate_exps.n_bytes() / n_expert as u64,
                    "up_bytes_per_expert": moe.up_exps.n_bytes() / n_expert as u64,
                    "down_bytes_per_expert": moe.down_exps.n_bytes() / n_expert as u64,
                })
            })
            .collect::<Vec<_>>();
        let mut total_counts = vec![vec![0u64; n_expert]; moe_blocks.len()];
        for agent in &counts_by_agent {
            for (block, per_expert) in agent.iter().enumerate() {
                for (expert, &count) in per_expert.iter().enumerate() {
                    total_counts[block][expert] += count;
                }
            }
        }
        println!(
            "{}",
            serde_json::json!({
                "kind": "moe_route_census",
                "model": model,
                "root_tokens": route_capture_ctx,
                "captured_tokens_per_agent": rows_by_agent,
                "calls_used_per_agent": calls_by_agent,
                "route_plan_family": plan["family"],
                "route_plan_seed": plan["seed"],
                "expert_count": n_expert,
                "topk": topk,
                "moe_blocks": weights,
                "counts_per_moe_block": total_counts,
                "counts_per_agent_and_moe_block": counts_by_agent,
                "counts_by_agent_and_width": counts_by_agent_by_width,
                "route_semantics": "serve_realistic CSV population; each agent owns its own packed code-corpus root, append token slices use the same agent*977 offset and call lengths as the serve driver; appended route rows are B1 continuations"
            })
        );
        return Ok(());
    }
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

#[cfg(test)]
mod comparison_contract_tests {
    use super::{comparison_max_abs, report_max_f32, report_max_f64};

    #[test]
    fn reporter_marks_uncomparable_inputs_non_finite() {
        assert_eq!(comparison_max_abs(&[1.0, 3.0], &[1.5, 1.0]), 2.0);
        assert!(comparison_max_abs(&[1.0], &[1.0, 2.0]).is_nan());
        assert!(comparison_max_abs(&[1.0], &[f32::NAN]).is_nan());
        assert!(report_max_f32(0.0, f32::NAN).is_nan());
        assert!(report_max_f64(0.0, f64::INFINITY).is_nan());
    }
}
