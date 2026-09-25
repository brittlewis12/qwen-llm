//! Family-general pp/tg rows for the families that do not bind through the
//! Qwen `Runtime`: Flash-Next (`qwen4exp`), DeepSeek V4, Muse Glimmer and K2
//! Horizon. Each adapter loads weights once with the production options
//! `qwen run` uses (`family_options`) and times the production calls,
//! including the host logits copy production makes at a prompt endpoint and
//! after every decoded token (llama-bench likewise copies logits). Rows record
//! what the timed call executed (`prefill_mode`, `decode_mode`).

use super::*;
use crate::family_options::{
    deepseek_v4_prefill_chunk_ranges, deepseek_v4_prefill_chunk_tokens,
    muse_runtime_options_from_env, qwen4exp_decode_options_from_env,
};
use anyhow::{bail, ensure};
use qwen_llm::deepseek_v4_metal::{DeepSeekV4MetalResidency, DeepSeekV4Session};
use qwen_llm::k2_horizon_runtime::{K2LoadedModel, K2Session};
use qwen_llm::model_family::ModelFamily;
use qwen_llm::muse_glimmer_runtime::{MuseGlimmerLoadedModel, MuseGlimmerTextRunner};
use qwen_llm::muse_glimmer_text_session::MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
use qwen_llm::qwen4exp::Qwen4ExpConfig;
use qwen_llm::qwen4exp_runtime::{
    Qwen4ExpLoadedModel, Qwen4ExpSessionCapacity, Qwen4ExpTextRunner,
};

/// What one family's timed calls executed, recorded on every row.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FamilySemantics {
    pub(crate) prefill_mode: &'static str,
    pub(crate) decode_mode: &'static str,
    pub(crate) prefill_chunk: Option<usize>,
}

pub(crate) trait FamilyBench {
    fn vocab_size(&self) -> usize;
    /// Fresh causal state at position zero; untimed.
    fn begin_rep(&mut self) -> Result<()>;
    /// Prefill `ids` after the current position through the production prompt
    /// path, including its endpoint logits copy.
    fn prefill(&mut self, ids: &[u32]) -> Result<()>;
    /// One step through the production generated-token path, including its
    /// logits copy.
    fn decode(&mut self, id: u32) -> Result<()>;
    /// Labels for the most recent prefill and decode calls.
    fn semantics(&self) -> FamilySemantics;
}

/// Extent the loaded model and its session must hold for these shapes.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FamilyBenchExtent {
    /// Deepest absolute position any row or the warm-up reaches.
    pub(crate) forwards: usize,
    /// Deepest absolute prefill endpoint (depth + pp, or the depth fill).
    /// Flash-Next sizes its packed and selected-range workspaces from this,
    /// as `qwen run` does from the whole prompt.
    pub(crate) prefill_endpoint: usize,
}

impl FamilyBenchExtent {
    pub(crate) fn for_shapes(
        pp: &[usize],
        tg: &[usize],
        depths: &[usize],
        warmup: (usize, usize),
    ) -> Result<Self> {
        let depth = depths.iter().copied().max().unwrap_or(0);
        let pp_max = pp.iter().copied().max().unwrap_or(0);
        let longest = pp_max.max(tg.iter().copied().max().unwrap_or(0));
        let timed = depth
            .checked_add(longest)
            .context("bench extent overflows usize")?;
        let pp_endpoint = if pp_max > 0 { depth + pp_max } else { 0 };
        Ok(Self {
            forwards: timed.max(warmup.0).max(warmup.1),
            prefill_endpoint: pp_endpoint.max(depth).max(warmup.0).max(2),
        })
    }
}

/// The family of a GGUF that does not bind through the Qwen runtime; `None`
/// for Qwen (and for unrecognized architectures, which the Qwen path refuses
/// with its own error).
pub(crate) fn non_qwen_family(model: &std::path::Path) -> Result<Option<ModelFamily>> {
    let gguf = GgufFile::open(model).with_context(|| format!("open {}", model.display()))?;
    Ok(ModelFamily::detect(&gguf)
        .filter(|family| !matches!(family, ModelFamily::Qwen35 | ModelFamily::Qwen35Moe)))
}

/// Profiling-only Flash-Next modes `qwen run` executes differently; refused
/// here rather than silently ignored.
const QWEN4EXP_PROFILE_ONLY_ENV: [&str; 3] = [
    "QWEN4EXP_LAYER_PROFILE",
    "QWEN4EXP_PACKED_PREFILL_PROFILE",
    "QWEN4EXP_FULL_SHARD_PREFETCH",
];

/// Load `family` once with production options and run `body` against it.
pub(crate) fn with_family_bench<R>(
    family: ModelFamily,
    ctx: &MetalContext,
    gguf: &GgufFile,
    extent: FamilyBenchExtent,
    body: &mut dyn FnMut(&mut dyn FamilyBench) -> Result<R>,
) -> Result<R> {
    match family {
        ModelFamily::Qwen4Exp => {
            for name in QWEN4EXP_PROFILE_ONLY_ENV {
                ensure!(
                    std::env::var_os(name).is_none(),
                    "{name} is a qwen run profiling mode; unset it for qwen-bench"
                );
            }
            let config = Qwen4ExpConfig::from_gguf(gguf).context("bind Flash-Next geometry")?;
            ensure!(
                config == Qwen4ExpConfig::flash_next_reference(),
                "Qwen3.8-Flash-Next runtime requires the released architecture contract"
            );
            let capacity = Qwen4ExpSessionCapacity::for_forward_limit(&config, extent.forwards)
                .context("derive Flash-Next session capacity")?;
            let options = qwen4exp_decode_options_from_env()?;
            // Same admission as `qwen run`: packed prefill sized to the
            // deepest prefill endpoint, or scalar when refused (logged).
            let mut loaded = match Qwen4ExpLoadedModel::load_with_decode_options(
                ctx,
                gguf,
                capacity,
                Some(extent.prefill_endpoint),
                options,
            ) {
                Ok(loaded) => loaded,
                Err(packed_error) => {
                    eprintln!(
                        "[family-bench] qwen4exp packed prefill unavailable ({packed_error}); retrying scalar admission"
                    );
                    Qwen4ExpLoadedModel::load_with_decode_options(
                        ctx, gguf, capacity, None, options,
                    )
                    .context("load Flash-Next (scalar fallback)")?
                }
            };
            let runner = loaded
                .create_runner(ctx)
                .context("bind Flash-Next execution graph")?;
            let mut bench = Qwen4ExpBench {
                runner,
                vocab: config.vocab_size as usize,
                last_prefill_mode: "none",
            };
            body(&mut bench)
        }
        ModelFamily::DeepSeek4 => {
            eprintln!(
                "[family-bench] deepseek_v4: load-time page-cache prefetch (QWEN_DSV4_PREFETCH) is not applied; rows time steady state after the model warm-up"
            );
            let chunk = deepseek_v4_prefill_chunk_tokens()?;
            let plan = DeepSeekV4MetalResidency::plan_for_forward_limit(ctx, gguf, extent.forwards)
                .context("plan DeepSeek V4 residency and session")?;
            let admitted = plan
                .admit(ctx.memory_signals())
                .context("admit DeepSeek V4 residency and session")?;
            let (residency, _admission, _after) =
                DeepSeekV4MetalResidency::load_from_plan(ctx, gguf, admitted)
                    .context("load DeepSeek V4 residency")?
                    .into_parts();
            let vocab = residency.config().vocab_size as usize;
            let mut bench = DeepSeekV4Bench {
                ctx,
                residency: Some(residency),
                session: None,
                chunk,
                vocab,
            };
            body(&mut bench)
        }
        ModelFamily::MuseGlimmer => {
            let options = muse_runtime_options_from_env()?;
            let mut loaded =
                MuseGlimmerLoadedModel::load_with_options(ctx, gguf, extent.forwards, options)
                    .context("load Muse Glimmer with production math")?;
            let vocab = loaded.config().vocab_size as usize;
            let math = loaded.math_options();
            let runner = loaded
                .create_runner(ctx)
                .context("bind Muse Glimmer execution graph")?;
            let mut bench = MuseBench {
                runner,
                vocab,
                matrix_prefill: math.matrix_prefill,
                split_decode: math.split_decode,
                last_prefill_tokens: 0,
            };
            body(&mut bench)
        }
        ModelFamily::K2Horizon => {
            let capacity = u32::try_from(extent.forwards).context("K2 bench extent exceeds u32")?;
            let model =
                K2LoadedModel::load(ctx, gguf, capacity).context("load K2 Horizon weights")?;
            let mut bench = K2Bench {
                vocab: model.config().vocab_size as usize,
                model: &model,
                session: None,
                last_prefill: None,
            };
            body(&mut bench)
        }
        ModelFamily::Qwen35 | ModelFamily::Qwen35Moe => {
            bail!("Qwen models bind through the Qwen runtime path, not the family adapter")
        }
    }
}

struct Qwen4ExpBench<'ctx, 'model, 'gguf> {
    runner: Qwen4ExpTextRunner<'ctx, 'model, 'gguf>,
    vocab: usize,
    last_prefill_mode: &'static str,
}

impl FamilyBench for Qwen4ExpBench<'_, '_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn begin_rep(&mut self) -> Result<()> {
        Ok(self.runner.reset()?)
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let logits = self
            .runner
            .prefill_continuation_with_command_checkpoint(ids, || Ok(()))?
            .to_vec();
        std::hint::black_box(logits);
        self.last_prefill_mode = match self.runner.last_prefill_timing() {
            Some(timing) if timing.packed_token_count == timing.token_count => "packed",
            Some(timing) if timing.packed_token_count > 0 => "packed+scalar_tail",
            Some(_) => "scalar",
            None => "unreported",
        };
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        let logits = self.runner.forward_token(id)?.to_vec();
        std::hint::black_box(logits);
        Ok(())
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: self.last_prefill_mode,
            decode_mode: "forward_token+logits_copy",
            prefill_chunk: None,
        }
    }
}

struct DeepSeekV4Bench<'ctx> {
    ctx: &'ctx MetalContext,
    residency: Option<DeepSeekV4MetalResidency>,
    session: Option<DeepSeekV4Session>,
    chunk: usize,
    vocab: usize,
}

impl DeepSeekV4Bench<'_> {
    fn session(&mut self) -> Result<&mut DeepSeekV4Session> {
        self.session
            .as_mut()
            .context("DeepSeek V4 bench used before begin_rep")
    }

    /// The host copy `qwen run` makes at a prompt endpoint and per token.
    fn copy_logits(&mut self) -> Result<()> {
        let vocab = self.vocab;
        let logits = self.session()?.copy_logits_f32()?;
        ensure!(
            logits.len() == vocab,
            "DeepSeek V4 logits length {} differs from vocabulary {vocab}",
            logits.len()
        );
        std::hint::black_box(logits);
        Ok(())
    }
}

impl FamilyBench for DeepSeekV4Bench<'_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    /// A session per rep over the long-lived residency, as the JSONL
    /// requests lane and serve do.
    fn begin_rep(&mut self) -> Result<()> {
        if let Some(session) = self.session.take() {
            self.residency = Some(session.into_residency()?);
        }
        let residency = self
            .residency
            .take()
            .context("DeepSeek V4 residency was not returned by the previous rep")?;
        self.session = Some(DeepSeekV4Session::new(self.ctx, residency)?);
        Ok(())
    }
    /// Production prompt path: packed chunks without logits, the final chunk
    /// with logits; prompts under two tokens forward one token at a time.
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let ctx = self.ctx;
        let chunk = self.chunk;
        let session = self.session()?;
        if ids.len() < 2 {
            for &id in ids {
                session.forward_token(ctx, id)?;
            }
        } else {
            let ranges = deepseek_v4_prefill_chunk_ranges(ids.len(), chunk);
            let last = ranges.len() - 1;
            for (index, range) in ranges.into_iter().enumerate() {
                if index == last {
                    session.prefill_tokens(ctx, &ids[range])?;
                } else {
                    session.advance_tokens(ctx, &ids[range])?;
                }
            }
        }
        self.copy_logits()
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        let ctx = self.ctx;
        self.session()?.forward_token(ctx, id)?;
        self.copy_logits()
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: "packed_chunks+final_logits_copy",
            decode_mode: "forward_token+logits_copy",
            prefill_chunk: Some(self.chunk),
        }
    }
}

struct MuseBench<'ctx, 'model> {
    runner: MuseGlimmerTextRunner<'ctx, 'model>,
    vocab: usize,
    matrix_prefill: bool,
    split_decode: bool,
    last_prefill_tokens: usize,
}

impl FamilyBench for MuseBench<'_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    fn begin_rep(&mut self) -> Result<()> {
        Ok(self.runner.reset()?)
    }
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        std::hint::black_box(self.runner.prefill(ids)?);
        self.last_prefill_tokens = ids.len();
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        std::hint::black_box(self.runner.forward_token(id)?);
        Ok(())
    }
    /// The prefill label `qwen run` derives for the same prompt length.
    fn semantics(&self) -> FamilySemantics {
        let tokens = self.last_prefill_tokens;
        let packed =
            tokens / MUSE_GLIMMER_PACKED_PREFILL_QUANTUM * MUSE_GLIMMER_PACKED_PREFILL_QUANTUM;
        let tail = tokens - packed;
        FamilySemantics {
            prefill_mode: match (packed, tail, self.matrix_prefill) {
                (0, _, _) => "scalar_tail",
                (_, 0, false) => "packed_exact",
                (_, _, false) => "packed_exact+scalar_tail",
                (_, 0, true) => "packed_matrix_online",
                (_, _, true) => "packed_matrix_online+scalar_tail",
            },
            decode_mode: if self.split_decode {
                "split_decode+logits_copy"
            } else {
                "forward_token+logits_copy"
            },
            prefill_chunk: None,
        }
    }
}

struct K2Bench<'s, 'a> {
    model: &'s K2LoadedModel<'a>,
    session: Option<K2Session<'s, 'a>>,
    vocab: usize,
    last_prefill: Option<(&'static str, usize)>,
}

impl<'s, 'a> K2Bench<'s, 'a> {
    fn session(&mut self) -> Result<&mut K2Session<'s, 'a>> {
        self.session
            .as_mut()
            .context("K2 bench used before begin_rep")
    }
}

impl FamilyBench for K2Bench<'_, '_> {
    fn vocab_size(&self) -> usize {
        self.vocab
    }
    /// A fresh session (KV and admission) per rep, as the request lane does.
    fn begin_rep(&mut self) -> Result<()> {
        self.session = None;
        self.session = Some(self.model.create_session(0)?);
        Ok(())
    }
    /// Production prompt path: the plan for this call's length, chunks
    /// without the head, the final chunk with logits.
    fn prefill(&mut self, ids: &[u32]) -> Result<()> {
        let info = self.model.prefill_info(ids.len());
        let chunk = info.chunk_tokens.max(1);
        let session = self.session()?;
        let chunks: Vec<&[u32]> = ids.chunks(chunk).collect();
        let last = chunks.len().saturating_sub(1);
        for (index, part) in chunks.into_iter().enumerate() {
            if index == last {
                std::hint::black_box(session.append(part)?);
            } else {
                session.advance(part)?;
            }
        }
        self.last_prefill = Some((info.mode, chunk));
        Ok(())
    }
    fn decode(&mut self, id: u32) -> Result<()> {
        std::hint::black_box(self.session()?.append(&[id])?);
        Ok(())
    }
    fn semantics(&self) -> FamilySemantics {
        FamilySemantics {
            prefill_mode: self.last_prefill.map_or("none", |(mode, _)| mode),
            decode_mode: "append+logits_copy+source_checks",
            prefill_chunk: self.last_prefill.map(|(_, chunk)| chunk),
        }
    }
}

fn synthetic_ids(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let vocab = u32::try_from(vocab).unwrap_or(u32::MAX);
    synthetic_prompt_ids(n, vocab, seed)
        .into_iter()
        .map(|id| id as u32)
        .collect()
}

/// Salt so the depth fill and the timed tokens are different sequences.
pub(crate) const DEPTH_SEED_SALT: u64 = 0x9e37_79b9_7f4a_7c15;

/// `pp512`, or `pp512@d8192` when timed after a depth fill.
pub(crate) fn shape_label(kind: &str, n: usize, depth: usize) -> String {
    if depth == 0 {
        format!("{kind}{n}")
    } else {
        format!("{kind}{n}@d{depth}")
    }
}

/// One llama-bench `pp<n>` row at `depth`: per rep a fresh state, an untimed
/// depth fill, then the timed prefill.
pub(crate) fn run_family_pp_row(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n_prompt: usize,
    depth: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    ensure!(n_prompt > 0, "--pp values must be >= 1");
    let vocab = bench.vocab_size();
    let depth_ids = synthetic_ids(depth, vocab, seed ^ DEPTH_SEED_SALT);
    let ids = synthetic_ids(n_prompt, vocab, seed);
    let rep = |bench: &mut dyn FamilyBench| -> Result<f64> {
        shutdown::checkpoint()?;
        bench.begin_rep()?;
        if depth > 0 {
            bench.prefill(&depth_ids)?;
        }
        let t0 = Instant::now();
        bench.prefill(&ids)?;
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };
    if !no_warmup {
        rep(bench).context("family pp warmup")?;
    }
    let mut wall = Vec::with_capacity(runs);
    for _ in 0..runs {
        wall.push(rep(bench).context("family pp rep")?);
    }
    Ok(family_row(
        row_ctx,
        bench.semantics(),
        shape_label("pp", n_prompt, depth),
        n_prompt,
        depth,
        &wall,
        true,
    ))
}

/// One llama-bench `tg<n>` row at `depth`: per rep a fresh state, an untimed
/// depth fill, then `n` timed single-token steps on random tokens.
pub(crate) fn run_family_tg_row(
    bench: &mut dyn FamilyBench,
    row_ctx: &SuiteRowContext,
    n_gen: usize,
    depth: usize,
    runs: usize,
    no_warmup: bool,
    seed: u64,
) -> Result<BenchRow> {
    ensure!(n_gen > 0, "--tg values must be >= 1");
    let vocab = bench.vocab_size();
    let depth_ids = synthetic_ids(depth, vocab, seed ^ DEPTH_SEED_SALT);
    let mut rep_index = 0u64;
    let mut rep = |bench: &mut dyn FamilyBench| -> Result<f64> {
        shutdown::checkpoint()?;
        rep_index += 1;
        let ids = synthetic_ids(n_gen, vocab, seed.wrapping_add(rep_index));
        bench.begin_rep()?;
        if depth > 0 {
            bench.prefill(&depth_ids)?;
        }
        let t0 = Instant::now();
        for &id in &ids {
            bench.decode(id)?;
        }
        Ok(t0.elapsed().as_secs_f64() * 1e3)
    };
    if !no_warmup {
        rep(bench).context("family tg warmup")?;
    }
    let mut wall = Vec::with_capacity(runs);
    for _ in 0..runs {
        wall.push(rep(bench).context("family tg rep")?);
    }
    Ok(family_row(
        row_ctx,
        bench.semantics(),
        shape_label("tg", n_gen, depth),
        n_gen,
        depth,
        &wall,
        false,
    ))
}

fn family_row(
    row_ctx: &SuiteRowContext,
    semantics: FamilySemantics,
    test: String,
    n_tokens: usize,
    n_depth: usize,
    wall_ms: &[f64],
    prefill: bool,
) -> BenchRow {
    let ts: Vec<f64> = wall_ms
        .iter()
        .map(|ms| n_tokens as f64 * 1000.0 / ms)
        .collect();
    let wall_mean = sample_mean(wall_ms);
    BenchRow {
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
        test,
        n_tokens,
        n_depth,
        n_repetitions: wall_ms.len(),
        avg_ts: sample_mean(&ts),
        stddev_ts: sample_stdev(&ts),
        samples_ts: ts,
        samples_ns: wall_ms.iter().map(|w| (*w * 1e6) as u64).collect(),
        avg_ns: (wall_mean * 1e6) as u64,
        avg_compute_ns: Some((wall_mean * 1e6) as u64),
        avg_session_alloc_ns: None,
        avg_scratch_alloc_ns: None,
        avg_gpu_ns: None,
        kernel_trace_command_buffers_per_token: None,
        kernel_trace_encoders_per_token: None,
        kernel_trace_concurrent_encoders_per_token: None,
        kernel_trace_dispatches_per_token: None,
        decode_gb_per_s: None,
        prefill_chunk: if prefill {
            semantics.prefill_chunk
        } else {
            None
        },
        decode_mode: (!prefill).then_some(semantics.decode_mode),
        prefill_mode: prefill.then_some(semantics.prefill_mode),
        power: row_ctx.power.clone(),
        qwen_env: row_ctx.qwen_env.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The admitted extent is the deepest position any row or the warm-up
    /// reaches (no arbitrary margin, so context-edge shapes stay valid), and
    /// the prefill endpoint is the deepest absolute prefill end.
    #[test]
    fn extent_is_the_deepest_position_and_prefill_endpoint() {
        let warmup = (512, 16);
        let extent =
            FamilyBenchExtent::for_shapes(&[512, 4096], &[128], &[0, 8192], warmup).unwrap();
        assert_eq!(extent.forwards, 8192 + 4096);
        assert_eq!(extent.prefill_endpoint, 8192 + 4096);
        let tg_depth = FamilyBenchExtent::for_shapes(&[], &[128], &[0, 8192], (16, 16)).unwrap();
        assert_eq!(tg_depth.forwards, 8192 + 128);
        assert_eq!(
            tg_depth.prefill_endpoint, 8192,
            "the depth fill is the endpoint"
        );
        let tg_only = FamilyBenchExtent::for_shapes(&[], &[128], &[0], (16, 16)).unwrap();
        assert_eq!((tg_only.forwards, tg_only.prefill_endpoint), (128, 16));
        let small = FamilyBenchExtent::for_shapes(&[8], &[], &[0], (8, 16)).unwrap();
        assert_eq!(small.forwards, 16, "the tg warm-up must fit");
    }

    #[test]
    fn shape_labels_follow_llama_bench_with_depth_suffix() {
        assert_eq!(shape_label("pp", 512, 0), "pp512");
        assert_eq!(shape_label("tg", 128, 8192), "tg128@d8192");
    }

    #[test]
    fn suite_depth_defaults_to_zero_and_accepts_lists() {
        let parse = |extra: &[&str]| {
            let mut argv = vec!["qwen-bench", "suite", "-m", "model.gguf", "--pp", "512"];
            argv.extend_from_slice(extra);
            let Cmd::Suite(args) = Args::try_parse_from(argv).expect("parse suite").cmd else {
                panic!("expected suite command");
            };
            args.depth
        };
        assert_eq!(parse(&[]), vec![0]);
        assert_eq!(parse(&["-d", "0,8192,32768"]), vec![0, 8192, 32768]);
    }
}
