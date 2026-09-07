//! Combined pp/tg suite rows.

use super::*;

pub(crate) fn arch_kind_label(kind: qwen_llm::model::ArchKind) -> &'static str {
    match kind {
        qwen_llm::model::ArchKind::Dense => "dense",
        qwen_llm::model::ArchKind::Moe => "moe",
    }
}

pub(crate) struct SuiteRowContext {
    pub(crate) build_commit: &'static str,
    pub(crate) build_dirty: u8,
    pub(crate) model_filename: String,
    pub(crate) model_size: u64,
    pub(crate) model_n_params: u64,
    pub(crate) arch_kind: &'static str,
    pub(crate) power: Option<PowerSnapshot>,
    pub(crate) qwen_env: std::collections::BTreeMap<String, String>,
}

pub(crate) fn run_suite_pp_row(
    loaded: &LoadedModel,
    row_ctx: &SuiteRowContext,
    n_prompt: usize,
    runs: usize,
    no_warmup: bool,
    prefill_chunk_override: Option<usize>,
    seed: u64,
) -> Result<BenchRow> {
    if n_prompt == 0 {
        return Err(anyhow!("--pp values must be >= 1"));
    }
    let arch = loaded.arch();
    let ctx = loaded.context();
    let mm = loaded.metal_model();
    let mf = loaded.forward();
    let ids = synthetic_prompt_ids(n_prompt, arch.vocab_size, seed);
    let prefill_chunk =
        prefill_chunk_override.unwrap_or_else(|| default_prefill_chunk(arch.kind, ids.len()));
    if prefill_chunk == 0 {
        return Err(anyhow!("--prefill-chunk must be >= 1"));
    }
    let cap = ids.len() + 16;

    if !no_warmup {
        let mut seq = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("suite pp warmup session")?;
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("suite pp warmup scratch")?;
        let _ = prefill_tokens_prompt_only_profiled(
            &mf,
            &ids,
            0,
            unsafe { seq.metal_session_mut() },
            &mut scratch,
        )
        .context("suite pp warmup")?;
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    let mut session_alloc_samples = Vec::with_capacity(runs);
    let mut scratch_alloc_samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        shutdown::checkpoint()?;
        let session_t0 = Instant::now();
        let mut seq = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("suite pp session")?;
        let session_alloc_ms = session_t0.elapsed().as_secs_f64() * 1e3;
        let scratch_t0 = Instant::now();
        let mut scratch = fresh_prefill_scratch_for_prompt(ctx, mm, prefill_chunk, ids.len())
            .context("suite pp scratch")?;
        let scratch_alloc_ms = scratch_t0.elapsed().as_secs_f64() * 1e3;
        seq.ensure_can_append(ids.len())
            .context("suite pp sequence capacity")?;
        let t0 = Instant::now();
        let gpu_ms = prefill_tokens_prompt_only_profiled(
            &mf,
            &ids,
            0,
            unsafe { seq.metal_session_mut() },
            &mut scratch,
        )
        .context("suite pp timed prefill")?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        seq.advance_by(ids.len())
            .context("suite pp advance sequence")?;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(ids.len() as f64 * 1000.0 / wall_ms);
        session_alloc_samples.push(session_alloc_ms);
        scratch_alloc_samples.push(scratch_alloc_ms);
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let session_alloc_mean = sample_mean(&session_alloc_samples);
    let scratch_alloc_mean = sample_mean(&scratch_alloc_samples);
    Ok(BenchRow {
        schema_version: BENCH_SCHEMA_VERSION,
        engine: "qwen-llm",
        build_commit: row_ctx.build_commit,
        build_dirty: row_ctx.build_dirty,
        build_identity: recorded_build_identity(),
        test_time: utc_iso8601_now(),
        model_filename: row_ctx.model_filename.clone(),
        model_size: row_ctx.model_size,
        model_n_params: row_ctx.model_n_params,
        arch_kind: row_ctx.arch_kind,
        test: format!("pp{}", ids.len()),
        n_tokens: ids.len(),
        n_repetitions: runs,
        avg_ts: sample_mean(&ts_samples),
        stddev_ts: sample_stdev(&ts_samples),
        samples_ts: ts_samples,
        samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: Some((session_alloc_mean * 1e6) as u64),
        avg_scratch_alloc_ns: Some((scratch_alloc_mean * 1e6) as u64),
        avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
        kernel_trace_command_buffers_per_token: None,
        kernel_trace_encoders_per_token: None,
        kernel_trace_concurrent_encoders_per_token: None,
        kernel_trace_dispatches_per_token: None,
        decode_gb_per_s: None,
        prefill_chunk: Some(prefill_chunk),
        decode_mode: None,
        prefill_mode: Some("packed"),
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    })
}

pub(crate) fn run_suite_tg_row(
    loaded: &LoadedModel,
    row_ctx: &SuiteRowContext,
    n_gen: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    if n_gen == 0 {
        return Err(anyhow!("--tg values must be >= 1"));
    }
    let arch = loaded.arch();
    let mf = loaded.forward();
    let cap = n_gen + 16;
    let vocab = arch.vocab_size.max(1);
    let trace_counts = env_flag_enabled("QWEN_DECODE_TRACE_COUNTS");
    let mut rng_state = if seed == 0 { 1u64 } else { seed };
    let mut next_rand_tok = || -> i32 {
        rng_state ^= rng_state << 13;
        rng_state ^= rng_state >> 7;
        rng_state ^= rng_state << 17;
        (rng_state % vocab as u64) as i32
    };
    let run_once = |first_tok: i32,
                    ranges: &mut dyn FnMut() -> i32|
     -> Result<(f64, f64, Option<KernelTraceCounters>, f64)> {
        let session_t0 = Instant::now();
        let mut seq = loaded
            .create_sequence(SequenceConfig::new(cap))
            .context("suite tg session")?;
        let session_alloc_ms = session_t0.elapsed().as_secs_f64() * 1e3;
        seq.ensure_can_append(n_gen)
            .context("suite tg sequence capacity")?;
        let _kernel_trace_guard = trace_counts.then(kernel_trace_begin);
        let t0 = Instant::now();
        let mut tok = first_tok;
        let mut gpu_ms_acc = 0.0;
        for pos in 0..n_gen {
            let (_argmax, prof) = mf.single_token_argmax_profiled(tok, pos as u32, unsafe {
                seq.metal_session_mut()
            })?;
            gpu_ms_acc += prof.gpu_kernel_ms;
            tok = ranges();
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        seq.advance_by(n_gen).context("suite tg advance sequence")?;
        let counts = trace_counts.then(kernel_trace_snapshot);
        Ok((wall_ms, gpu_ms_acc, counts, session_alloc_ms))
    };

    if !no_warmup {
        let first = next_rand_tok();
        let _ = run_once(first, &mut next_rand_tok).context("suite tg warmup")?;
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut gpu_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    let mut session_alloc_samples = Vec::with_capacity(runs);
    let mut trace_samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        shutdown::checkpoint()?;
        let first = next_rand_tok();
        let (wall_ms, gpu_ms, trace, session_alloc_ms) =
            run_once(first, &mut next_rand_tok).context("suite tg run")?;
        wall_samples.push(wall_ms);
        gpu_samples.push(gpu_ms);
        ts_samples.push(n_gen as f64 * 1000.0 / wall_ms);
        session_alloc_samples.push(session_alloc_ms);
        if let Some(trace) = trace {
            trace_samples.push(trace);
        }
    }

    let wall_mean = sample_mean(&wall_samples);
    let gpu_mean = sample_mean(&gpu_samples);
    let session_alloc_mean = sample_mean(&session_alloc_samples);
    let trace_per_token = if trace_counts && !trace_samples.is_empty() {
        let denom = (trace_samples.len() * n_gen) as f64;
        let encoders: u64 = trace_samples.iter().map(|t| t.encoders).sum();
        let concurrent_encoders: u64 = trace_samples.iter().map(|t| t.concurrent_encoders).sum();
        let dispatches: u64 = trace_samples.iter().map(|t| t.dispatches).sum();
        Some((
            1.0,
            encoders as f64 / denom,
            concurrent_encoders as f64 / denom,
            dispatches as f64 / denom,
        ))
    } else {
        None
    };
    let decode_gb_per_s = if matches!(arch.kind, qwen_llm::model::ArchKind::Dense)
        && row_ctx.model_size > 0
        && wall_mean > 0.0
    {
        Some((row_ctx.model_size as f64 / 1e9) / (wall_mean / 1000.0 / n_gen as f64))
    } else {
        None
    };

    Ok(BenchRow {
        schema_version: BENCH_SCHEMA_VERSION,
        engine: "qwen-llm",
        build_commit: row_ctx.build_commit,
        build_dirty: row_ctx.build_dirty,
        build_identity: recorded_build_identity(),
        test_time: utc_iso8601_now(),
        model_filename: row_ctx.model_filename.clone(),
        model_size: row_ctx.model_size,
        model_n_params: row_ctx.model_n_params,
        arch_kind: row_ctx.arch_kind,
        test: format!("tg{n_gen}"),
        n_tokens: n_gen,
        n_repetitions: runs,
        avg_ts: sample_mean(&ts_samples),
        stddev_ts: sample_stdev(&ts_samples),
        samples_ts: ts_samples,
        samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: Some((session_alloc_mean * 1e6) as u64),
        avg_scratch_alloc_ns: None,
        avg_gpu_ns: Some((gpu_mean * 1e6) as u64),
        kernel_trace_command_buffers_per_token: trace_per_token.map(|t| t.0),
        kernel_trace_encoders_per_token: trace_per_token.map(|t| t.1),
        kernel_trace_concurrent_encoders_per_token: trace_per_token.map(|t| t.2),
        kernel_trace_dispatches_per_token: trace_per_token.map(|t| t.3),
        decode_gb_per_s,
        prefill_chunk: None,
        decode_mode: Some("apples-lcpp"),
        prefill_mode: None,
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    })
}

pub(crate) fn run_suite(args: SuiteArgs) -> Result<()> {
    let SuiteArgs {
        model,
        pp,
        tg,
        runs,
        no_warmup,
        prefill_chunk,
        seed,
        output,
    } = args;
    if runs == 0 {
        return Err(anyhow!("--runs must be >= 1"));
    }
    if pp.is_empty() && tg.is_empty() {
        return Err(anyhow!("provide at least one --pp or --tg shape"));
    }
    let json_mode = matches!(output, OutputFormat::Json);

    let runtime = Runtime::metal().context("init Runtime")?;
    crate::text_log!(json_mode, "[suite] device: {}", runtime.describe());
    let power = capture_power_snapshot();
    crate::text_log!(
        json_mode,
        "[suite] power: {}",
        power_snapshot_summary(power.as_ref())
    );

    let loaded = runtime
        .load_model(&model)
        .with_context(|| format!("load {}", model.display()))?;
    let g = loaded.gguf();
    let (commit, dirty) = qwen_build_identity();
    let row_ctx = SuiteRowContext {
        build_commit: commit,
        build_dirty: dirty,
        model_filename: model.display().to_string(),
        model_size: model_weight_bytes(g),
        model_n_params: g
            .get_u64("general.parameter_count")
            .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
        arch_kind: arch_kind_label(loaded.arch().kind),
        power,
        qwen_env: capture_qwen_env(),
    };

    crate::text_log!(
        json_mode,
        "[suite] model={} pp={:?} tg={:?} runs={} warmup={} seed={}",
        model.display(),
        pp,
        tg,
        runs,
        if no_warmup { "skip" } else { "on" },
        seed,
    );

    let mut rows = Vec::with_capacity(pp.len() + tg.len());
    for &n_prompt in &pp {
        let row = run_suite_pp_row(
            &loaded,
            &row_ctx,
            n_prompt,
            runs,
            no_warmup,
            prefill_chunk,
            seed,
        )?;
        crate::text_log!(
            json_mode,
            "[suite] {:>8}: {:>8.2} t/s  {:>8.2} ms/token",
            row.test,
            row.avg_ts,
            1000.0 / row.avg_ts.max(1e-9),
        );
        rows.push(row);
    }
    for &n_gen in &tg {
        let row = run_suite_tg_row(&loaded, &row_ctx, n_gen, runs, no_warmup, seed)?;
        crate::text_log!(
            json_mode,
            "[suite] {:>8}: {:>8.2} t/s  {:>8.2} ms/token",
            row.test,
            row.avg_ts,
            1000.0 / row.avg_ts.max(1e-9),
        );
        rows.push(row);
    }

    if json_mode {
        println!(
            "{}",
            serde_json::to_string(&rows).context("serialize suite rows")?
        );
    }
    Ok(())
}
