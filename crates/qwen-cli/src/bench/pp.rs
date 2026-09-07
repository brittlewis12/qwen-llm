//! Prompt-processing benches, FFN A/B, pp-wait, and MoE residency helpers.

use super::*;

#[derive(Clone, Copy, Debug)]
pub(crate) struct MoeRouteBatchStats {
    pub(crate) layers: usize,
    pub(crate) tokens: usize,
    pub(crate) slots_per_layer: usize,
    pub(crate) avg_unique_experts: f64,
    pub(crate) avg_max_slots: f64,
    pub(crate) avg_reuse: f64,
}

pub(crate) fn summarize_moe_route_batch(
    routes_by_token: &[Vec<MoeRouteReplayRow>],
    eligible_indices: &[usize],
    n_expert: usize,
    topk: usize,
) -> Result<MoeRouteBatchStats> {
    let tokens = routes_by_token.len();
    if tokens == 0 || eligible_indices.is_empty() {
        return Err(anyhow!("captured route batch is empty"));
    }

    let mut unique_sum = 0usize;
    let mut max_sum = 0usize;
    let mut reuse_sum = 0.0f64;
    let slots_per_layer = tokens * topk;
    for &moe_i in eligible_indices {
        let mut counts = vec![0usize; n_expert];
        for routes in routes_by_token {
            let route = routes
                .get(moe_i)
                .ok_or_else(|| anyhow!("captured route missing layer {moe_i}"))?;
            if route.topk_idx.len() != topk {
                return Err(anyhow!(
                    "captured route has {} experts, expected topk={topk}",
                    route.topk_idx.len()
                ));
            }
            for &expert in &route.topk_idx {
                if expert < 0 || expert as usize >= n_expert {
                    return Err(anyhow!(
                        "captured expert id {expert} outside n_expert={n_expert}"
                    ));
                }
                counts[expert as usize] += 1;
            }
        }
        let unique = counts.iter().filter(|&&c| c > 0).count();
        let max_count = counts.iter().copied().max().unwrap_or(0);
        unique_sum += unique;
        max_sum += max_count;
        reuse_sum += if unique > 0 {
            slots_per_layer as f64 / unique as f64
        } else {
            0.0
        };
    }
    let layers = eligible_indices.len();
    Ok(MoeRouteBatchStats {
        layers,
        tokens,
        slots_per_layer,
        avg_unique_experts: unique_sum as f64 / layers as f64,
        avg_max_slots: max_sum as f64 / layers as f64,
        avg_reuse: reuse_sum / layers as f64,
    })
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub(crate) enum CaptureTokenPattern {
    Zero,
    Ramp,
}

#[derive(Copy, Clone, Debug, Eq, Ord, PartialEq, PartialOrd, ValueEnum)]
pub(crate) enum MoeBatchSlotOrder {
    Exact,
    ExpertSortedPerfOnly,
}

impl MoeBatchSlotOrder {
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Exact => "exact",
            Self::ExpertSortedPerfOnly => "expert-sorted-perf-only",
        }
    }
}

pub(crate) fn capture_replay_token(
    pattern: CaptureTokenPattern,
    tok: usize,
    vocab_size: u32,
) -> i32 {
    match pattern {
        CaptureTokenPattern::Zero => 0,
        CaptureTokenPattern::Ramp => ((1 + tok * 7919) % vocab_size as usize) as i32,
    }
}

pub(crate) fn captured_gateup_tensors(
    ctx: &MetalContext,
    routes_by_token: &[Vec<MoeRouteReplayRow>],
    eligible_indices: &[usize],
    h: usize,
    n_expert: usize,
    topk: usize,
    slot_order: MoeBatchSlotOrder,
) -> Result<Vec<(MetalTensor, MetalTensor)>> {
    let tokens = routes_by_token.len();
    let slots = tokens * topk;
    let mut tensors = Vec::with_capacity(eligible_indices.len());
    for &moe_i in eligible_indices {
        let hidden_t = MetalTensor::zeros_f32(ctx, vec![(tokens * h) as u64])?;
        let idx = MetalTensor::zeros_f32(ctx, vec![slots as u64])?;
        unsafe {
            let dst = hidden_t.buffer.contents().as_ptr() as *mut f32;
            let ptr = idx.buffer.contents().as_ptr() as *mut i32;
            let mut sorted_slots = Vec::new();
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots.reserve(slots);
            }
            for (tok, routes) in routes_by_token.iter().enumerate() {
                let route = routes
                    .get(moe_i)
                    .ok_or_else(|| anyhow!("captured route missing layer {moe_i}"))?;
                if route.hidden.len() != h {
                    return Err(anyhow!(
                        "captured hidden has {} elements, expected h={h}",
                        route.hidden.len()
                    ));
                }
                if route.topk_idx.len() != topk {
                    return Err(anyhow!(
                        "captured route has {} experts, expected topk={topk}",
                        route.topk_idx.len()
                    ));
                }
                std::ptr::copy_nonoverlapping(route.hidden.as_ptr(), dst.add(tok * h), h);
                for (slot, &expert) in route.topk_idx.iter().enumerate() {
                    if expert < 0 || expert as usize >= n_expert {
                        return Err(anyhow!(
                            "captured expert id {expert} outside n_expert={n_expert}"
                        ));
                    }
                    match slot_order {
                        MoeBatchSlotOrder::Exact => {
                            *ptr.add(tok * topk + slot) = expert;
                        }
                        MoeBatchSlotOrder::ExpertSortedPerfOnly => {
                            sorted_slots.push((expert, tok, slot));
                        }
                    }
                }
            }
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots.sort_unstable();
                for (out_slot, (expert, _, _)) in sorted_slots.iter().copied().enumerate() {
                    *ptr.add(out_slot) = expert;
                }
            }
        }
        tensors.push((hidden_t, idx));
    }
    Ok(tensors)
}

pub(crate) fn captured_down_tensors(
    ctx: &MetalContext,
    routes_by_token: &[Vec<MoeRouteReplayRow>],
    eligible_indices: &[usize],
    n_expert: usize,
    topk: usize,
    slot_order: MoeBatchSlotOrder,
) -> Result<Vec<(MetalTensor, MetalTensor)>> {
    let tokens = routes_by_token.len();
    let slots = tokens * topk;
    let mut tensors = Vec::with_capacity(eligible_indices.len());
    for &moe_i in eligible_indices {
        let idx = MetalTensor::zeros_f32(ctx, vec![slots as u64])?;
        let weight = MetalTensor::zeros_f32(ctx, vec![slots as u64])?;
        unsafe {
            let idx_ptr = idx.buffer.contents().as_ptr() as *mut i32;
            let w_ptr = weight.buffer.contents().as_ptr() as *mut f32;
            let mut sorted_slots = Vec::new();
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots.reserve(slots);
            }
            for (tok, routes) in routes_by_token.iter().enumerate() {
                let route = routes
                    .get(moe_i)
                    .ok_or_else(|| anyhow!("captured route missing layer {moe_i}"))?;
                if route.topk_idx.len() != topk || route.topk_weight.len() != topk {
                    return Err(anyhow!(
                        "captured route has idx={} weight={}, expected topk={topk}",
                        route.topk_idx.len(),
                        route.topk_weight.len()
                    ));
                }
                for slot in 0..topk {
                    let expert = route.topk_idx[slot];
                    if expert < 0 || expert as usize >= n_expert {
                        return Err(anyhow!(
                            "captured expert id {expert} outside n_expert={n_expert}"
                        ));
                    }
                    match slot_order {
                        MoeBatchSlotOrder::Exact => {
                            let out_slot = tok * topk + slot;
                            *idx_ptr.add(out_slot) = expert;
                            *w_ptr.add(out_slot) = route.topk_weight[slot];
                        }
                        MoeBatchSlotOrder::ExpertSortedPerfOnly => {
                            sorted_slots.push((expert, tok, slot, route.topk_weight[slot]));
                        }
                    }
                }
            }
            if slot_order == MoeBatchSlotOrder::ExpertSortedPerfOnly {
                sorted_slots
                    .sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
                for (out_slot, (expert, _, _, route_weight)) in
                    sorted_slots.iter().copied().enumerate()
                {
                    *idx_ptr.add(out_slot) = expert;
                    *w_ptr.add(out_slot) = route_weight;
                }
            }
        }
        tensors.push((idx, weight));
    }
    Ok(tensors)
}

pub(crate) fn pp_warm_moe_weight_banks(ctx: &MetalContext, mf: &MetalForward<'_>) -> Result<usize> {
    let stride_bytes = std::env::var("QWEN_PP_TOUCH_STRIDE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(16 * 1024);
    let sink = MetalTensor::zeros_f32(ctx, vec![256])?;
    let cmd = ctx.queue.commandBuffer().context("warmup command buffer")?;
    let enc = KernelEncoder::begin(&cmd);
    let mut touched = 0usize;
    for block in &mf.model.blocks {
        let moe = match block {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        };
        let Some(moe) = moe else { continue };
        for tensor in [&moe.gate_exps, &moe.up_exps, &moe.down_exps] {
            encode_touch_bytes_f32(ctx, &enc, tensor, &sink, stride_bytes)?;
            touched += 1;
        }
    }
    enc.end();
    cmd.commit();
    cmd.waitUntilCompleted();
    Ok(touched)
}

pub(crate) fn buffer_as_allocation(
    buffer: &ProtocolObject<dyn MTLBuffer>,
) -> &ProtocolObject<dyn MTLAllocation> {
    ProtocolObject::from_ref(buffer)
}

pub(crate) struct PpResidencySetGuard {
    pub(crate) queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    pub(crate) set: Retained<ProtocolObject<dyn MTLResidencySet>>,
}

impl Drop for PpResidencySetGuard {
    fn drop(&mut self) {
        self.queue.removeResidencySet(&self.set);
        self.set.endResidency();
    }
}

pub(crate) fn pp_register_moe_residency_set(
    ctx: &MetalContext,
    mf: &MetalForward<'_>,
) -> Result<(PpResidencySetGuard, usize, u64)> {
    let mut seen = HashSet::new();
    let mut buffers = Vec::new();
    for block in &mf.model.blocks {
        let moe = match block {
            MetalBlock::Gdn(g) => g.ffn_moe.as_ref(),
            MetalBlock::Attn(a) => a.ffn_moe.as_ref(),
        };
        let Some(moe) = moe else { continue };
        for tensor in [&moe.gate_exps, &moe.up_exps, &moe.down_exps] {
            let ptr = Retained::as_ptr(&tensor.buffer) as *const _ as usize;
            if seen.insert(ptr) {
                buffers.push(&*tensor.buffer);
            }
        }
    }
    if buffers.is_empty() {
        return Err(anyhow!(
            "no MoE expert-bank buffers found for residency set"
        ));
    }

    let desc = MTLResidencySetDescriptor::new();
    desc.setLabel(Some(&NSString::from_str("qwen-bench-pp-moe-banks")));
    // SAFETY: initialCapacity is advisory only; we pass the exact number of
    // unique allocations we are about to register.
    unsafe { desc.setInitialCapacity(buffers.len()) };
    let set = ctx
        .device
        .newResidencySetWithDescriptor_error(&desc)
        .map_err(|e: Retained<NSError>| anyhow!(e.localizedDescription().to_string()))?;
    for buffer in &buffers {
        set.addAllocation(buffer_as_allocation(buffer));
    }
    set.commit();
    set.requestResidency();
    ctx.queue.addResidencySet(&set);
    let bytes = set.allocatedSize();
    let guard = PpResidencySetGuard {
        queue: ctx.queue.clone(),
        set,
    };
    Ok((guard, buffers.len(), bytes))
}

pub(crate) fn print_prefill_lowering_summary(mm: &MetalModel) {
    let mat_mat_gdn = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
        )
    };
    let mat_mat_attn = |dtype: GgmlType| {
        matches!(
            dtype,
            GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q6_K | GgmlType::Q8_0
        )
    };
    let mat_mat_dense_ffn = |dtype: GgmlType| matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K);

    let mut gdn_total = 0usize;
    let mut gdn_batched = 0usize;
    let mut attn_total = 0usize;
    let mut attn_batched = 0usize;
    let mut dense_ffn_total = 0usize;
    let mut dense_ffn_batched = 0usize;
    let mut moe_total = 0usize;
    let mut moe_gpu_supported = 0usize;
    let mut first_gdn = None;
    let mut first_attn = None;
    let mut first_moe = None;

    for block in &mm.blocks {
        match block {
            MetalBlock::Gdn(g) => {
                gdn_total += 1;
                let gdn_ok = mat_mat_gdn(g.in_proj_qkv.dtype)
                    && mat_mat_gdn(g.in_proj_z.dtype)
                    && mat_mat_gdn(g.out_proj.dtype);
                if gdn_ok {
                    gdn_batched += 1;
                }
                first_gdn.get_or_insert(format!(
                    "qkv={:?} z={:?} out={:?}",
                    g.in_proj_qkv.dtype, g.in_proj_z.dtype, g.out_proj.dtype
                ));
                if let Some(moe) = &g.ffn_moe {
                    moe_total += 1;
                    let moe_ok = matches!(moe.gate_exps.dtype, GgmlType::Q4_K | GgmlType::Q5_K)
                        && moe.gate_exps.dtype == moe.up_exps.dtype
                        && matches!(moe.down_exps.dtype, GgmlType::Q5_K | GgmlType::Q6_K);
                    if moe_ok {
                        moe_gpu_supported += 1;
                    }
                    first_moe.get_or_insert(format!(
                        "gate={:?} up={:?} down={:?}",
                        moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype
                    ));
                } else {
                    dense_ffn_total += 1;
                    if mat_mat_dense_ffn(g.ffn_gate.dtype)
                        && mat_mat_dense_ffn(g.ffn_up.dtype)
                        && mat_mat_dense_ffn(g.ffn_down.dtype)
                    {
                        dense_ffn_batched += 1;
                    }
                }
            }
            MetalBlock::Attn(a) => {
                attn_total += 1;
                let attn_ok = mat_mat_attn(a.q.dtype)
                    && mat_mat_attn(a.k.dtype)
                    && mat_mat_attn(a.v.dtype)
                    && mat_mat_attn(a.o.dtype);
                if attn_ok {
                    attn_batched += 1;
                }
                first_attn.get_or_insert(format!(
                    "q={:?} k={:?} v={:?} o={:?}",
                    a.q.dtype, a.k.dtype, a.v.dtype, a.o.dtype
                ));
                if let Some(moe) = &a.ffn_moe {
                    moe_total += 1;
                    let moe_ok = matches!(moe.gate_exps.dtype, GgmlType::Q4_K | GgmlType::Q5_K)
                        && moe.gate_exps.dtype == moe.up_exps.dtype
                        && matches!(moe.down_exps.dtype, GgmlType::Q5_K | GgmlType::Q6_K);
                    if moe_ok {
                        moe_gpu_supported += 1;
                    }
                    first_moe.get_or_insert(format!(
                        "gate={:?} up={:?} down={:?}",
                        moe.gate_exps.dtype, moe.up_exps.dtype, moe.down_exps.dtype
                    ));
                } else {
                    dense_ffn_total += 1;
                    if mat_mat_dense_ffn(a.ffn_gate.dtype)
                        && mat_mat_dense_ffn(a.ffn_up.dtype)
                        && mat_mat_dense_ffn(a.ffn_down.dtype)
                    {
                        dense_ffn_batched += 1;
                    }
                }
            }
        }
    }

    eprintln!(
        "[pp] lowering: gdn_batched={gdn_batched}/{gdn_total} attn_batched={attn_batched}/{attn_total} dense_ffn_batched={dense_ffn_batched}/{dense_ffn_total} moe_gpu_token_loop={moe_gpu_supported}/{moe_total}"
    );
    if let Some(s) = first_gdn {
        eprintln!("[pp] dtype sample gdn: {s}");
    }
    if let Some(s) = first_attn {
        eprintln!("[pp] dtype sample attn: {s}");
    }
    if let Some(s) = first_moe {
        eprintln!("[pp] dtype sample moe: {s}");
    }
}

pub(crate) fn run_pp(args: PpArgs) -> Result<()> {
    let PpArgs {
        model,
        n_prompt,
        prompt,
        file,
        messages,
        messages_max,
        messages_preserve_thinking,
        messages_strip_thinking,
        messages_no_generation_prompt,
        runs,
        no_warmup,
        prefill_chunk,
        with_tail,
        seed,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    let json_mode = matches!(output, OutputFormat::Json);

    let runtime = Runtime::metal().context("init Runtime")?;
    crate::text_log!(json_mode, "[pp] device: {}", runtime.describe());
    let power = capture_power_snapshot();
    crate::text_log!(json_mode, "[pp] power: {}", power_snapshot_summary(power.as_ref()));

    let loaded = runtime
        .load_model(&model)
        .with_context(|| format!("load {}", model.display()))?;
    shutdown::checkpoint()?;
    let ctx = loaded.context();
    let g = loaded.gguf();
    let mm = loaded.metal_model();
    let arch = loaded.arch();

    let (ids, source_label) = if let Some(prompt) = prompt {
        let tok = loaded.tokenizer().context("open tokenizer")?;
        let ids = tok.encode(&prompt, false).context("tokenize prompt")?;
        (ids, format!("text prompt ({} chars)", prompt.len()))
    } else if let Some(path) = file {
        let prompt =
            std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
        let tok = loaded.tokenizer().context("open tokenizer")?;
        let ids = tok.encode(&prompt, false).context("tokenize prompt file")?;
        (
            ids,
            format!("file prompt:{} ({} chars)", path.display(), prompt.len()),
        )
    } else if let Some(path) = messages {
        let prompt = load_messages_prompt(
            &path,
            messages_max,
            crate::prompt_template::qwen_template_for_gguf(loaded.gguf())?,
            messages_thinking_mode(messages_preserve_thinking, messages_strip_thinking),
            !messages_no_generation_prompt,
        )?;
        let tok = loaded.tokenizer().context("open tokenizer")?;
        let ids = tok
            .encode(&prompt, false)
            .context("tokenize rendered messages prompt")?;
        (
            ids,
            format!("messages:{} ({} chars)", path.display(), prompt.len()),
        )
    } else {
        if n_prompt == 0 {
            return Err(anyhow!("--n-prompt must be >= 1"));
        }
        (
            synthetic_prompt_ids(n_prompt, arch.vocab_size, seed),
            format!("synthetic token ids (seed={seed})"),
        )
    };
    if ids.is_empty() {
        return Err(anyhow!("prompt tokenized to an empty sequence"));
    }

    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    let mf = loaded.forward();
    let cap = ids.len() + 16;
    crate::text_log!(json_mode, 
        "[pp] model={} source={} n_prompt={} runs={} chunk={} tail={}",
        model.display(),
        source_label,
        ids.len(),
        runs,
        prefill_chunk,
        if with_tail { "final-logits" } else { "skip" }
    );
    if !json_mode {
        print_prefill_lowering_summary(mm);
    }

    let residency_guard = if env_flag_enabled("QWEN_PP_RESIDENCY_SET")
        && arch.kind == qwen_llm::model::ArchKind::Moe
    {
        let (guard, allocations, bytes) =
            pp_register_moe_residency_set(ctx, &mf).context("register MoE residency set")?;
        crate::text_log!(json_mode, 
            "[pp] residency: registered {allocations} MoE expert-bank allocations ({:.2} GiB tracked)",
            bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );
        Some(guard)
    } else {
        None
    };

    if residency_guard.is_some() && env_flag_enabled("QWEN_PP_WARM_MOE_BANKS") {
        crate::text_log!(json_mode, "[pp] residency-set active; skipping QWEN_PP_WARM_MOE_BANKS touch pass");
    } else if env_flag_enabled("QWEN_PP_WARM_MOE_BANKS")
        && arch.kind == qwen_llm::model::ArchKind::Moe
    {
        let touched =
            pp_warm_moe_weight_banks(ctx, &mf).context("warm grouped MoE weight banks")?;
        crate::text_log!(json_mode, "[pp] warmup: touched {touched} MoE expert-bank tensors via GPU residency pass");
    }

    if !no_warmup {
        let mut s = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("session warmup")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("warmup prefill scratch")?;
        if with_tail {
            let _ = prefill_tokens_with_multi_hidden(
                &mf,
                &ids,
                0,
                unsafe { s.metal_session_mut() },
                &mut scratch,
                &[],
                None,
            )
            .context("warmup prefill with tail")?;
        } else {
            let _ = prefill_tokens_prompt_only_profiled(
                &mf,
                &ids,
                0,
                unsafe { s.metal_session_mut() },
                &mut scratch,
            )
            .context("warmup prompt-only prefill")?;
        }
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    for run_idx in 0..runs {
        shutdown::checkpoint()?;
        let mut s = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("session run")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("timed prefill scratch")?;

        let t0 = Instant::now();
        let gpu_ms = if with_tail {
            let (_, gpu_ms) = prefill_tokens_with_multi_hidden_profiled(
                &mf,
                &ids,
                0,
                unsafe { s.metal_session_mut() },
                &mut scratch,
                &[],
                None,
            )
            .context("timed prefill with tail")?;
            gpu_ms
        } else {
            prefill_tokens_prompt_only_profiled(
                &mf,
                &ids,
                0,
                unsafe { s.metal_session_mut() },
                &mut scratch,
            )
            .context("timed prompt-only prefill")?
        };
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let ts = ids.len() as f64 * 1000.0 / wall_ms;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ts);
        crate::text_log!(json_mode, 
            "[pp] run {:>2}: wall {:>8.1} ms  gpu {:>8.1} ms  {:>7.2} t/s",
            run_idx + 1,
            wall_ms,
            gpu_ms,
            ts
        );
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let ts_mean = sample_mean(&ts_samples);
    let ts_sd = sample_stdev(&ts_samples);

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let row = BenchRow {
            schema_version: BENCH_SCHEMA_VERSION,
            engine: "qwen-llm",
            build_commit: commit,
            build_dirty: dirty,
            build_identity: recorded_build_identity(),
            test_time: utc_iso8601_now(),
            model_filename: model.display().to_string(),
            model_size: model_weight_bytes(g),
            model_n_params: g
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
            arch_kind: match arch.kind {
                qwen_llm::model::ArchKind::Dense => "dense",
                qwen_llm::model::ArchKind::Moe => "moe",
            },
            test: format!("pp{}", ids.len()),
            n_tokens: ids.len(),
            n_repetitions: runs,
            avg_ts: ts_mean,
            stddev_ts: ts_sd,
            samples_ts: ts_samples.clone(),
            samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
            avg_ns: (wall_mean * 1e6) as u64,
            avg_compute_ns: Some((wall_mean * 1e6) as u64),
            avg_session_alloc_ns: None,
            avg_scratch_alloc_ns: None,
            avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
            kernel_trace_command_buffers_per_token: None,
            kernel_trace_encoders_per_token: None,
            kernel_trace_concurrent_encoders_per_token: None,
            kernel_trace_dispatches_per_token: None,
            // pp is not a steady-state-bandwidth measurement, so we don't
            // emit a derived GB/s for prefill rows. Digest tools can compute
            // their own if they want, but the canonical bandwidth comparison
            // is on decode.
            decode_gb_per_s: None,
            prefill_chunk: Some(prefill_chunk),
            decode_mode: None,
            prefill_mode: Some("packed"),
            power,
            qwen_env: capture_qwen_env(),
        };
        // Wrap in an array to match `llama-bench -o json`.
        let arr = vec![row];
        let json = serde_json::to_string(&arr).context("serialize pp bench row")?;
        println!("{json}");
    } else {
        eprintln!();
        eprintln!("[pp] === results ===");
        eprintln!(
            "[pp] prompt: {} tokens in {:.1} ms avg = {:.2} ms/token = {:.2} +/- {:.2} t/s",
            ids.len(),
            wall_mean,
            wall_mean / ids.len() as f64,
            ts_mean,
            ts_sd
        );
        eprintln!(
            "[pp] gpu:    {:.1} ms avg = {:.2} ms/token = {:.1}% of wall",
            gpu_mean,
            gpu_mean / ids.len() as f64,
            100.0 * gpu_mean / wall_mean.max(1e-9)
        );
        eprintln!(
            "[pp] note: session and scratch allocation are outside the timed interval; tail={}.",
            if with_tail {
                "included"
            } else {
                "skipped to match llama-bench pp logits policy"
            }
        );
    }

    Ok(())
}

pub(crate) fn run_pp_ffn_ab_once(
    ctx: &MetalContext,
    mm: &MetalModel,
    mf: &MetalForward<'_>,
    ids: &[i32],
    prefill_chunk: usize,
    fused: bool,
) -> Result<(f64, f64, f64)> {
    let mut s = MetalSession::fresh(ctx, mm, ids.len() + 16).context("session run")?;
    let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
        .context("timed prefill scratch")?;

    let t0 = Instant::now();
    let gpu_ms = with_prefill_dense_ffn_fused_swiglu_q4_override(fused, || {
        prefill_tokens_prompt_only_profiled(mf, ids, 0, &mut s, &mut scratch)
    })
    .context("timed prompt-only prefill")?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ts = ids.len() as f64 * 1000.0 / wall_ms;
    Ok((wall_ms, gpu_ms, ts))
}

pub(crate) fn run_pp_ffn_ab(args: PpFfnAbArgs) -> Result<()> {
    let PpFfnAbArgs {
        model,
        n_prompt,
        prefill_chunk,
        pairs,
        no_warmup,
        seed,
    } = args;
    if n_prompt == 0 {
        return Err(anyhow!("--n-prompt must be >= 1"));
    }
    if pairs == 0 {
        return Err(anyhow!("--pairs must be >= 1"));
    }

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[pp-ffn-ab] device: {}", ctx.describe());
    let power = capture_power_snapshot();
    eprintln!(
        "[pp-ffn-ab] power: {}",
        power_snapshot_summary(power.as_ref())
    );

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    if m.arch.kind != qwen_llm::model::ArchKind::Dense {
        return Err(anyhow!("pp-ffn-ab is a dense FFN harness; got MoE model"));
    }
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let mf = MetalForward::new(&ctx, &mm);
    let ids = synthetic_prompt_ids(n_prompt, m.arch.vocab_size, seed);
    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(m.arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    eprintln!(
        "[pp-ffn-ab] model={} n_prompt={} pairs={} chunk={} warmup={}",
        model.display(),
        ids.len(),
        pairs,
        prefill_chunk,
        if no_warmup { "skip" } else { "base+fused" }
    );
    print_prefill_lowering_summary(&mm);

    if !no_warmup {
        for fused in [false, true] {
            let _ = run_pp_ffn_ab_once(&ctx, &mm, &mf, &ids, prefill_chunk, fused)
                .with_context(|| format!("warmup fused={fused}"))?;
        }
    }

    println!("pair\torder\tvariant\twall_ms\tgpu_ms\ttokens_s");
    for pair_idx in 0..pairs {
        let order = if pair_idx % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        for (order_idx, fused) in order.into_iter().enumerate() {
            let (wall_ms, gpu_ms, ts) =
                run_pp_ffn_ab_once(&ctx, &mm, &mf, &ids, prefill_chunk, fused)
                    .with_context(|| format!("timed pair={pair_idx} fused={fused}"))?;
            println!(
                "{pair_idx}\t{order_idx}\t{}\t{wall_ms:.1}\t{gpu_ms:.1}\t{ts:.2}",
                if fused { "fused" } else { "base" }
            );
        }
    }

    Ok(())
}

pub(crate) fn run_pp_wait(args: PpWaitArgs) -> Result<()> {
    let PpWaitArgs {
        model,
        n_prompt,
        prefill_chunk,
        with_tail,
        seed,
        no_warmup,
        ready_file,
        go_file,
        output,
    } = args;
    if n_prompt == 0 {
        return Err(anyhow!("--n-prompt must be >= 1"));
    }
    let json_mode = matches!(output, OutputFormat::Json);

    let ctx = MetalContext::new().context("init MetalContext")?;
    crate::text_log!(json_mode, "[pp-wait] device: {}", ctx.describe());
    let power = capture_power_snapshot();
    crate::text_log!(json_mode, 
        "[pp-wait] power: {}",
        power_snapshot_summary(power.as_ref())
    );

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    shutdown::checkpoint()?;
    let ids = synthetic_prompt_ids(n_prompt, m.arch.vocab_size, seed);
    let prefill_chunk =
        prefill_chunk.unwrap_or_else(|| default_prefill_chunk(m.arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + 16;
    crate::text_log!(json_mode, 
        "[pp-wait] model={} n_prompt={} chunk={} tail={} pid={}",
        model.display(),
        ids.len(),
        prefill_chunk,
        if with_tail { "final-logits" } else { "skip" },
        std::process::id(),
    );
    if !json_mode {
        print_prefill_lowering_summary(&mm);
    }

    if !no_warmup {
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, ids.len())
            .context("warmup prefill scratch")?;
        if with_tail {
            let _ = prefill_tokens_with_multi_hidden(&mf, &ids, 0, &mut s, &mut scratch, &[], None)
                .context("warmup prefill with tail")?;
        } else {
            let _ = prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
                .context("warmup prompt-only prefill")?;
        }
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
        format!(
            "ready pid={} model={} n_prompt={} chunk={} tail={}\n",
            std::process::id(),
            model.display(),
            ids.len(),
            prefill_chunk,
            if with_tail { "final-logits" } else { "skip" }
        ),
    )?;
    crate::text_log!(json_mode, "[pp-wait] ready; waiting for {:?}", go_file);
    while !go_file.exists() {
        shutdown::checkpoint()?;
        std::thread::sleep(Duration::from_millis(25));
    }
    crate::text_log!(json_mode, "[pp-wait] go signal received; running timed prefill");

    let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
    let mut scratch = fresh_prefill_scratch_for_prompt(&ctx, &mm, prefill_chunk, ids.len())
        .context("timed prefill scratch")?;
    let t0 = Instant::now();
    let gpu_ms = if with_tail {
        let (_, gpu_ms) = prefill_tokens_with_multi_hidden_profiled(
            &mf,
            &ids,
            0,
            &mut s,
            &mut scratch,
            &[],
            None,
        )
        .context("timed prefill with tail")?;
        gpu_ms
    } else {
        prefill_tokens_prompt_only_profiled(&mf, &ids, 0, &mut s, &mut scratch)
            .context("timed prompt-only prefill")?
    };
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    let ts = ids.len() as f64 * 1000.0 / wall_ms;

    if json_mode {
        let (commit, dirty) = qwen_build_identity();
        let row = BenchRow {
            schema_version: BENCH_SCHEMA_VERSION,
            engine: "qwen-llm",
            build_commit: commit,
            build_dirty: dirty,
            build_identity: recorded_build_identity(),
            test_time: utc_iso8601_now(),
            model_filename: model.display().to_string(),
            model_size: model_weight_bytes(&g),
            model_n_params: g
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
            arch_kind: match m.arch.kind {
                qwen_llm::model::ArchKind::Dense => "dense",
                qwen_llm::model::ArchKind::Moe => "moe",
            },
            test: format!("pp{}", ids.len()),
            n_tokens: ids.len(),
            n_repetitions: 1,
            avg_ts: ts,
            stddev_ts: 0.0,
            samples_ts: vec![ts],
            samples_ns: vec![(wall_ms * 1e6) as u64],
            avg_ns: (wall_ms * 1e6) as u64,
            avg_compute_ns: Some((wall_ms * 1e6) as u64),
            avg_session_alloc_ns: None,
            avg_scratch_alloc_ns: None,
            avg_gpu_ns: Some((gpu_ms * 1e6) as u64),
            kernel_trace_command_buffers_per_token: None,
            kernel_trace_encoders_per_token: None,
            kernel_trace_concurrent_encoders_per_token: None,
            kernel_trace_dispatches_per_token: None,
            decode_gb_per_s: None,
            prefill_chunk: Some(prefill_chunk),
            decode_mode: None,
            prefill_mode: Some("packed"),
            power,
            qwen_env: capture_qwen_env(),
        };
        println!(
            "{}",
            serde_json::to_string(&vec![row]).context("serialize pp-wait row")?
        );
    } else {
        eprintln!(
            "[pp-wait] run: wall {:>8.1} ms  gpu {:>8.1} ms  {:>7.2} t/s",
            wall_ms, gpu_ms, ts
        );
    }
    Ok(())
}
