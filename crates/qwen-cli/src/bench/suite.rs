//! Combined pp/tg suite rows for every supported family.
//!
//! Qwen models run through the Qwen runtime with the production prefill
//! allocator `qwen run` uses; every other family runs through its adapter in
//! `family_bench`, loaded with the same production options. Rows follow
//! llama-bench semantics: fresh state per rep, an untimed fill of `--depth`
//! tokens, then the timed prefill (`pp<N>`) or `N` single-token steps on
//! random tokens (`tg<N>`).

use super::*;
use crate::family_bench::{
    DEPTH_SEED_SALT, FamilyBenchExtent, run_family_pp_row, run_family_tg_row, shape_label,
    with_family_bench,
};
use crate::prefill_plan::{PrefillChunkArg, allocate_prefill_request_state};
use anyhow::ensure;
use qwen_llm::model_family::ModelFamily;

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
    pub(crate) family: &'static str,
    pub(crate) power: Option<PowerSnapshot>,
    pub(crate) qwen_env: std::collections::BTreeMap<String, String>,
}

struct QwenPpSample {
    wall_ms: f64,
    session_alloc_ms: f64,
    scratch_alloc_ms: f64,
    chunk: usize,
    reason: Option<&'static str>,
}

/// One Qwen `pp<n>` row at `depth`. Each rep allocates the request the way
/// `qwen run` does for a fresh `depth + n`-token prompt (auto chunk, plan and
/// admission, or `--prefill-chunk`), fills `depth` untimed, then times the
/// production `LoadedModel::prefill` of `n` tokens, which includes the final
/// norm, head and endpoint logits copy (as llama-bench's pp does).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_suite_pp_row(
    loaded: &LoadedModel,
    row_ctx: &SuiteRowContext,
    n_prompt: usize,
    depth: usize,
    runs: usize,
    no_warmup: bool,
    prefill_chunk_override: Option<usize>,
    seed: u64,
) -> Result<BenchRow> {
    ensure!(n_prompt > 0, "--pp values must be >= 1");
    ensure!(
        prefill_chunk_override != Some(0),
        "--prefill-chunk must be >= 1"
    );
    let arch = loaded.arch();
    let depth_ids = synthetic_prompt_ids(depth, arch.vocab_size, seed ^ DEPTH_SEED_SALT);
    let ids = synthetic_prompt_ids(n_prompt, arch.vocab_size, seed);
    let total = depth + ids.len();
    let requested = prefill_chunk_override.map_or(PrefillChunkArg::Auto, PrefillChunkArg::Fixed);

    let rep = || -> Result<QwenPpSample> {
        shutdown::checkpoint()?;
        let allocated = allocate_prefill_request_state(loaded, requested, total, total, true)
            .context("suite pp production request allocation")?;
        let chunk = allocated.chunk;
        let reason = allocated.decision.as_ref().map(|decision| decision.reason);
        let session_alloc_ms = allocated.sequence_allocation_ms;
        let scratch_alloc_ms = allocated.scratch_allocation_ms;
        let mut scratch = allocated.scratch;
        let mut seq = allocated.sequence;
        if depth > 0 {
            let logits = loaded
                .prefill(&mut seq, &mut scratch, &depth_ids)
                .context("suite pp depth fill")?;
            std::hint::black_box(logits);
        }
        let t0 = Instant::now();
        let logits = loaded
            .prefill(&mut seq, &mut scratch, &ids)
            .context("suite pp timed prefill")?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(logits);
        Ok(QwenPpSample {
            wall_ms,
            session_alloc_ms,
            scratch_alloc_ms,
            chunk,
            reason,
        })
    };

    if !no_warmup {
        rep().context("suite pp warmup")?;
    }
    let samples = (0..runs)
        .map(|_| rep().context("suite pp rep"))
        .collect::<Result<Vec<_>>>()?;
    let chunk = samples.first().map(|sample| sample.chunk);
    ensure!(
        samples.iter().all(|sample| Some(sample.chunk) == chunk),
        "production prefill allocation changed chunk between reps"
    );
    eprintln!(
        "[suite] {} prefill_chunk={} decision={}",
        shape_label("pp", n_prompt, depth),
        chunk.unwrap_or(0),
        samples
            .first()
            .and_then(|sample| sample.reason)
            .unwrap_or("fixed_chunk"),
    );

    let wall_samples: Vec<f64> = samples.iter().map(|sample| sample.wall_ms).collect();
    let ts_samples: Vec<f64> = wall_samples
        .iter()
        .map(|wall| ids.len() as f64 * 1000.0 / wall)
        .collect();
    let wall_mean = sample_mean(&wall_samples);
    let session_alloc_mean = sample_mean(
        &samples
            .iter()
            .map(|s| s.session_alloc_ms)
            .collect::<Vec<_>>(),
    );
    let scratch_alloc_mean = sample_mean(
        &samples
            .iter()
            .map(|s| s.scratch_alloc_ms)
            .collect::<Vec<_>>(),
    );
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
        family: row_ctx.family,
        test: shape_label("pp", ids.len(), depth),
        n_tokens: ids.len(),
        n_depth: depth,
        n_repetitions: runs,
        avg_ts: sample_mean(&ts_samples),
        stddev_ts: sample_stdev(&ts_samples),
        samples_ts: ts_samples,
        samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: Some((session_alloc_mean * 1e6) as u64),
        avg_scratch_alloc_ns: Some((scratch_alloc_mean * 1e6) as u64),
        avg_gpu_ns: None,
        kernel_trace_command_buffers_per_token: None,
        kernel_trace_encoders_per_token: None,
        kernel_trace_concurrent_encoders_per_token: None,
        kernel_trace_dispatches_per_token: None,
        decode_gb_per_s: None,
        prefill_chunk: chunk,
        decode_mode: None,
        prefill_mode: Some("production_prefill+endpoint_logits"),
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    })
}

/// One Qwen `tg<n>` row at `depth`: a fresh sequence per rep (filled with
/// `depth` tokens through the production allocator and prefill, untimed),
/// then `n` production `LoadedModel::decode_token` steps on random tokens,
/// each returning the full logits as `qwen run` consumes them.
pub(crate) fn run_suite_tg_row(
    loaded: &LoadedModel,
    row_ctx: &SuiteRowContext,
    n_gen: usize,
    depth: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    ensure!(n_gen > 0, "--tg values must be >= 1");
    let arch = loaded.arch();
    let cap = depth + n_gen;
    let vocab = arch.vocab_size.max(1);
    let depth_ids = synthetic_prompt_ids(depth, arch.vocab_size, seed ^ DEPTH_SEED_SALT);
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
     -> Result<(f64, Option<KernelTraceCounters>, f64)> {
        let (mut seq, session_alloc_ms) = if depth == 0 {
            let session_t0 = Instant::now();
            let seq = loaded
                .create_sequence(SequenceConfig::new(cap))
                .context("suite tg session")?;
            (seq, session_t0.elapsed().as_secs_f64() * 1e3)
        } else {
            let allocated =
                allocate_prefill_request_state(loaded, PrefillChunkArg::Auto, depth, cap, true)
                    .context("suite tg depth allocation")?;
            let session_alloc_ms = allocated.sequence_allocation_ms;
            let mut scratch = allocated.scratch;
            let mut seq = allocated.sequence;
            let logits = loaded
                .prefill(&mut seq, &mut scratch, &depth_ids)
                .context("suite tg depth fill")?;
            std::hint::black_box(logits);
            (seq, session_alloc_ms)
        };
        let _kernel_trace_guard = trace_counts.then(kernel_trace_begin);
        let t0 = Instant::now();
        let mut tok = first_tok;
        for _ in 0..n_gen {
            let logits = loaded
                .decode_token(&mut seq, tok)
                .context("suite tg decode token")?;
            std::hint::black_box(logits);
            tok = ranges();
        }
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
        let counts = trace_counts.then(kernel_trace_snapshot);
        Ok((wall_ms, counts, session_alloc_ms))
    };

    if !no_warmup {
        let first = next_rand_tok();
        let _ = run_once(first, &mut next_rand_tok).context("suite tg warmup")?;
    }

    let mut wall_samples = Vec::with_capacity(runs);
    let mut ts_samples = Vec::with_capacity(runs);
    let mut session_alloc_samples = Vec::with_capacity(runs);
    let mut trace_samples = Vec::with_capacity(runs);
    for _ in 0..runs {
        shutdown::checkpoint()?;
        let first = next_rand_tok();
        let (wall_ms, trace, session_alloc_ms) =
            run_once(first, &mut next_rand_tok).context("suite tg run")?;
        wall_samples.push(wall_ms);
        ts_samples.push(n_gen as f64 * 1000.0 / wall_ms);
        session_alloc_samples.push(session_alloc_ms);
        if let Some(trace) = trace {
            trace_samples.push(trace);
        }
    }

    let wall_mean = sample_mean(&wall_samples);
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
        family: row_ctx.family,
        test: shape_label("tg", n_gen, depth),
        n_tokens: n_gen,
        n_depth: depth,
        n_repetitions: runs,
        avg_ts: sample_mean(&ts_samples),
        stddev_ts: sample_stdev(&ts_samples),
        samples_ts: ts_samples,
        samples_ns: wall_samples.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: Some((session_alloc_mean * 1e6) as u64),
        avg_scratch_alloc_ns: None,
        avg_gpu_ns: None,
        kernel_trace_command_buffers_per_token: trace_per_token.map(|t| t.0),
        kernel_trace_encoders_per_token: trace_per_token.map(|t| t.1),
        kernel_trace_concurrent_encoders_per_token: trace_per_token.map(|t| t.2),
        kernel_trace_dispatches_per_token: trace_per_token.map(|t| t.3),
        decode_gb_per_s,
        prefill_chunk: None,
        decode_mode: Some("decode_token+logits_copy"),
        prefill_mode: None,
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    })
}

/// Decode steps in the model-level warm-up.
const WARMUP_TG: usize = 16;

fn log_row(json_mode: bool, row: &BenchRow) {
    crate::text_log!(
        json_mode,
        "[suite] {:>14}: {:>8.2} t/s  {:>8.2} ms/token",
        row.test,
        row.avg_ts,
        1000.0 / row.avg_ts.max(1e-9),
    );
}

pub(crate) fn run_suite(args: SuiteArgs) -> Result<()> {
    let SuiteArgs {
        model,
        pp,
        tg,
        depth: depths,
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
    ensure!(!depths.is_empty(), "--depth needs at least one value");
    let json_mode = matches!(output, OutputFormat::Json);

    let probe = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let family = ModelFamily::detect(&probe).with_context(|| {
        format!(
            "{}: architecture {:?} is not a supported family",
            model.display(),
            probe.get_str("general.architecture")
        )
    })?;
    let (commit, dirty) = qwen_build_identity();
    let power = capture_power_snapshot();
    crate::text_log!(
        json_mode,
        "[suite] family={} model={} pp={:?} tg={:?} depth={:?} runs={} warmup={} seed={}",
        family.record_label(),
        model.display(),
        pp,
        tg,
        depths,
        runs,
        if no_warmup { "skip" } else { "on" },
        seed,
    );
    crate::text_log!(
        json_mode,
        "[suite] power: {}",
        power_snapshot_summary(power.as_ref())
    );

    // Model-level warm-up before the first row, so first-touch weight paging
    // and GPU clock ramp-up after load do not land in the first row's
    // samples (observed: Flash-Next's first pp512 sample ran at half speed
    // after its row warmup). Shapes stay within the admitted extent.
    let warmup_pp = pp.first().copied().unwrap_or(WARMUP_TG);
    let mut rows = Vec::with_capacity((pp.len() + tg.len()) * depths.len());
    if matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe) {
        drop(probe);
        let runtime = Runtime::metal().context("init Runtime")?;
        crate::text_log!(json_mode, "[suite] device: {}", runtime.describe());
        let loaded = runtime
            .load_model(&model)
            .with_context(|| format!("load {}", model.display()))?;
        let g = loaded.gguf();
        let row_ctx = SuiteRowContext {
            build_commit: commit,
            build_dirty: dirty,
            model_filename: model.display().to_string(),
            model_size: model_weight_bytes(g),
            model_n_params: g
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| g.tensors.iter().map(|t| t.n_elements()).sum()),
            arch_kind: arch_kind_label(loaded.arch().kind),
            family: family.record_label(),
            power,
            qwen_env: capture_qwen_env(),
        };
        if !no_warmup {
            crate::text_log!(json_mode, "[suite] model warm-up (untimed)");
            run_suite_pp_row(
                &loaded,
                &row_ctx,
                warmup_pp,
                0,
                1,
                true,
                prefill_chunk,
                seed,
            )
            .context("suite model warm-up pp")?;
            run_suite_tg_row(&loaded, &row_ctx, WARMUP_TG, 0, 1, true, seed)
                .context("suite model warm-up tg")?;
        }
        for &depth in &depths {
            for &n_prompt in &pp {
                let row = run_suite_pp_row(
                    &loaded,
                    &row_ctx,
                    n_prompt,
                    depth,
                    runs,
                    no_warmup,
                    prefill_chunk,
                    seed,
                )?;
                log_row(json_mode, &row);
                rows.push(row);
            }
            for &n_gen in &tg {
                let row = run_suite_tg_row(&loaded, &row_ctx, n_gen, depth, runs, no_warmup, seed)?;
                log_row(json_mode, &row);
                rows.push(row);
            }
        }
    } else {
        ensure!(
            prefill_chunk.is_none(),
            "--prefill-chunk applies to Qwen models; {} uses its production chunking",
            family.record_label()
        );
        let ctx = MetalContext::new().context("init Metal")?;
        crate::text_log!(json_mode, "[suite] device: {}", ctx.describe());
        let row_ctx = SuiteRowContext {
            build_commit: commit,
            build_dirty: dirty,
            model_filename: model.display().to_string(),
            model_size: model_weight_bytes(&probe),
            model_n_params: probe
                .get_u64("general.parameter_count")
                .unwrap_or_else(|| probe.tensors.iter().map(|t| t.n_elements()).sum()),
            arch_kind: family.record_label(),
            family: family.record_label(),
            power,
            qwen_env: capture_qwen_env(),
        };
        let extent = FamilyBenchExtent::for_shapes(&pp, &tg, &depths, (warmup_pp, WARMUP_TG))?;
        rows = with_family_bench(family, &ctx, &probe, extent, &mut |bench| {
            let mut rows = Vec::with_capacity((pp.len() + tg.len()) * depths.len());
            if !no_warmup {
                crate::text_log!(json_mode, "[suite] model warm-up (untimed)");
                run_family_pp_row(bench, &row_ctx, warmup_pp, 0, 1, true, seed)
                    .context("suite model warm-up pp")?;
                run_family_tg_row(bench, &row_ctx, WARMUP_TG, 0, 1, true, seed)
                    .context("suite model warm-up tg")?;
            }
            for &depth in &depths {
                for &n_prompt in &pp {
                    let row =
                        run_family_pp_row(bench, &row_ctx, n_prompt, depth, runs, no_warmup, seed)?;
                    log_row(json_mode, &row);
                    rows.push(row);
                }
                for &n_gen in &tg {
                    let row =
                        run_family_tg_row(bench, &row_ctx, n_gen, depth, runs, no_warmup, seed)?;
                    log_row(json_mode, &row);
                    rows.push(row);
                }
            }
            Ok(rows)
        })?;
    }

    if json_mode {
        println!(
            "{}",
            serde_json::to_string(&rows).context("serialize suite rows")?
        );
    }
    Ok(())
}
