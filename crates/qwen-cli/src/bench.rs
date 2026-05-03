//! `qwen-bench` — end-to-end decode throughput harness for qwen-llm.
//!
//! Replaces the prior stub (load-time only) with a real bench surface.
//! Three modes:
//!
//!   `decode`       — Run a prompt + N-token decode loop; report ms/token,
//!                    GPU vs wall split, per-token series, and (when an
//!                    oracle is available) cos vs oracle for correctness.
//!   `ctx-sweep`    — Ramp KV to each checkpoint context length, time a
//!                    short window of decodes there, report ms/token and
//!                    effective bandwidth at each point.
//!   `phase`        — Phase-resolved profile at one chosen context.
//!
//! Designed so "measure → change → measure" is `cargo run --release -p
//! qwen-cli --bin qwen-bench -- decode -m ... --tokens 64`, not "run the
//! right ignored test by name". Per Jeff & Sanjay (and the v0.32
//! re-sequencing review): the bench harness IS leverage, not hygiene.

use anyhow::{anyhow, Context, Result};
use clap::{Parser, Subcommand};
use qwen_llm::{
    gguf::GgufFile,
    loader::Model,
    metal::MetalContext,
    metal_forward::{MetalForward, MetalModel, MetalSession},
    tokenizer::Tokenizer,
};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(
    name = "qwen-bench",
    version,
    about = "end-to-end throughput benchmark for qwen-llm"
)]
struct Args {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Decode N tokens after a prompt; report per-token timings.
    Decode(DecodeArgs),
    /// Sweep context length (ramp + measure window).
    CtxSweep(CtxSweepArgs),
    /// Phase-resolved profile at one context length (uses the
    /// `phase_sum` GPU time, NOT the per-phase-cmdbuf wall artifact).
    Phase(PhaseArgs),
}

#[derive(Parser, Debug)]
struct DecodeArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Prompt text. If absent, uses a fixed warmup prompt.
    #[arg(short = 'p', long)]
    prompt: Option<String>,
    /// Number of tokens to generate after the prompt.
    #[arg(long, default_value = "64")]
    tokens: usize,
    /// Optional oracle file (raw f32 logits at last position from
    /// llama.cpp/llm `--snapshot`). If provided, compares cos.
    #[arg(long)]
    oracle: Option<PathBuf>,
    /// Skip the warmup pass (default is to do one warmup, then re-init
    /// the session for the timed run, exactly like the ignored tests).
    #[arg(long)]
    no_warmup: bool,
}

#[derive(Parser, Debug)]
struct CtxSweepArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Comma-separated context checkpoints to measure at.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,64,256,1024,4096,8192,16384"
    )]
    checkpoints: Vec<usize>,
    /// How many tokens to time at each checkpoint.
    #[arg(long, default_value = "5")]
    window: usize,
}

#[derive(Parser, Debug)]
struct PhaseArgs {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Context length to profile at.
    #[arg(long, default_value = "4096")]
    ctx: usize,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    match args.cmd {
        Cmd::Decode(a) => run_decode(a),
        Cmd::CtxSweep(a) => run_ctx_sweep(a),
        Cmd::Phase(a) => run_phase(a),
    }
}

fn run_decode(args: DecodeArgs) -> Result<()> {
    let DecodeArgs {
        model,
        prompt,
        tokens,
        oracle,
        no_warmup,
    } = args;
    let prompt =
        prompt.unwrap_or_else(|| "The quick brown fox jumps over the lazy dog".to_string());

    let ctx = MetalContext::new().context("init MetalContext")?;
    eprintln!("[bench] device: {}", ctx.describe());

    let g = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let m = Model::from_gguf(&g).context("parse model arch from gguf")?;
    let mm = MetalModel::load(&ctx, &g, &m).context("metal-load model weights")?;
    let tok = Tokenizer::open(&model).context("open tokenizer")?;

    let ids = tok.encode(&prompt, false).context("tokenize prompt")?;
    eprintln!(
        "[bench] model={} prompt={:?} ({} tokens), gen={} tokens",
        model.display(),
        prompt,
        ids.len(),
        tokens
    );

    let mf = MetalForward::new(&ctx, &mm);
    let cap = ids.len() + tokens + 16;

    if !no_warmup {
        // One warmup pass to compile pipeline state objects + warm caches.
        let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session warmup")?;
        let _ = mf.single_token(ids[0], 0, &mut s)?;
    }

    let mut s = MetalSession::fresh(&ctx, &mm, cap).context("session run")?;
    let mut per_token_ms: Vec<f64> = Vec::with_capacity(ids.len() + tokens);
    let mut last_logits: Vec<f32> = Vec::new();

    let t0 = Instant::now();

    // Prefill: feed all prompt tokens through (sequential single_token; we
    // don't have batched prefill yet — that's a future product).
    for (i, &tid) in ids.iter().enumerate() {
        let tt = Instant::now();
        last_logits = mf.single_token(tid, i as u32, &mut s)?;
        per_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
    }
    let prefill_wall = t0.elapsed().as_secs_f64() * 1e3;
    let prefill_avg = prefill_wall / ids.len() as f64;

    // Decode loop: greedy argmax sampling on the CPU side. (A backend
    // sampler kernel is on the roadmap; this is the trivial CPU baseline.)
    let mut gen_ids: Vec<i32> = Vec::with_capacity(tokens);
    let t1 = Instant::now();
    for k in 0..tokens {
        let pos = ids.len() + k;
        let next = argmax_i32(&last_logits);
        gen_ids.push(next);
        let tt = Instant::now();
        last_logits = mf.single_token(next, pos as u32, &mut s)?;
        per_token_ms.push(tt.elapsed().as_secs_f64() * 1e3);
    }
    let decode_wall = t1.elapsed().as_secs_f64() * 1e3;

    let total_wall = t0.elapsed().as_secs_f64() * 1e3;

    // Decode-only steady-state: skip the very first decode (cache-cold for
    // some downstream PSO + heavily warm-up sensitive).
    let decode_steady_ms: f64 = if tokens > 1 {
        per_token_ms[ids.len() + 1..].iter().sum::<f64>() / (tokens - 1) as f64
    } else {
        decode_wall
    };

    eprintln!();
    eprintln!("[bench] === results ===");
    eprintln!("[bench] prefill: {} tokens in {prefill_wall:.1} ms = {prefill_avg:.2} ms/token = {:.1} t/s", ids.len(), 1000.0 / prefill_avg);
    eprintln!("[bench] decode:  {tokens} tokens in {decode_wall:.1} ms = {:.2} ms/token (avg) = {:.1} t/s",
              decode_wall / tokens as f64, 1000.0 * tokens as f64 / decode_wall);
    eprintln!(
        "[bench] steady:  {decode_steady_ms:.2} ms/token (excl. first decode) = {:.2} t/s",
        1000.0 / decode_steady_ms
    );
    eprintln!("[bench] total:   {total_wall:.1} ms wall");

    // Print first/last few per-token times for spot-checking.
    let n_show = 5usize.min(per_token_ms.len());
    eprintln!(
        "[bench] per-token (first {n_show}): {:?}",
        &per_token_ms[..n_show]
    );
    if per_token_ms.len() > 2 * n_show {
        let m = per_token_ms.len();
        eprintln!(
            "[bench] per-token (last  {n_show}): {:?}",
            &per_token_ms[m - n_show..]
        );
    }

    if let Some(oracle_path) = oracle {
        let bytes = std::fs::read(&oracle_path)
            .with_context(|| format!("read oracle {}", oracle_path.display()))?;
        if bytes.len() % 4 != 0 || bytes.len() / 4 != last_logits.len() {
            return Err(anyhow!(
                "oracle size {} bytes ({} f32) != logits len {}",
                bytes.len(),
                bytes.len() / 4,
                last_logits.len()
            ));
        }
        let oracle: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let (cos, max_abs, argmax_ours, argmax_oracle) = compare_logits(&last_logits, &oracle);
        eprintln!(
            "[bench] oracle:  cos={cos:.6}  max|Δ|={max_abs:.4}  argmax: ours={argmax_ours} oracle={argmax_oracle} {}",
            if argmax_ours == argmax_oracle { "✓" } else { "✗ MISMATCH" }
        );
    }

    if !gen_ids.is_empty() {
        let text = tok.decode(&gen_ids);
        eprintln!("[bench] generated: {:?}", text);
    }

    Ok(())
}

fn run_ctx_sweep(args: CtxSweepArgs) -> Result<()> {
    let CtxSweepArgs {
        model,
        checkpoints,
        window,
    } = args;
    let ctx = MetalContext::new()?;
    eprintln!("[bench] device: {}", ctx.describe());
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&ctx, &g, &m)?;

    let max_n = *checkpoints
        .iter()
        .max()
        .ok_or_else(|| anyhow!("no checkpoints"))?;
    let mut s = MetalSession::fresh(&ctx, &mm, max_n + window + 16)?;
    let mf = MetalForward::new(&ctx, &mm);

    // Warmup pipeline state cache.
    for i in 0..3 {
        let _ = mf.single_token(0, i as u32, &mut s)?;
    }
    let mut s = MetalSession::fresh(&ctx, &mm, max_n + window + 16)?;
    // One pre-warmed token at position 0 to populate everything.
    let _ = mf.single_token(0, 0, &mut s)?;

    println!("[ctx-sweep] === per-token decode cost vs context ===");
    println!("[ctx-sweep] context  total_ms  gpu_ms  cpu_enc_ms  t/s");

    let mut prev_pos = 1u32;
    for &target in &checkpoints {
        for p in prev_pos..(target as u32) {
            let _ = mf.single_token(0, p, &mut s)?;
        }
        prev_pos = target as u32;

        let mut samples = Vec::with_capacity(window);
        for i in 0..window {
            let pos = prev_pos + i as u32;
            let (_, p) = mf.single_token_profiled(0, pos, &mut s)?;
            samples.push(p);
        }
        prev_pos += window as u32;

        let avg_total = samples.iter().map(|p| p.total_ms).sum::<f64>() / window as f64;
        let avg_gpu = samples.iter().map(|p| p.gpu_kernel_ms).sum::<f64>() / window as f64;
        let avg_enc = samples.iter().map(|p| p.cpu_encode_ms).sum::<f64>() / window as f64;
        println!(
            "[ctx-sweep] {target:>7}  {avg_total:>8.2}  {avg_gpu:>6.2}  {avg_enc:>10.2}  {:>4.1}",
            1000.0 / avg_total
        );
    }

    Ok(())
}

fn run_phase(args: PhaseArgs) -> Result<()> {
    let PhaseArgs { model, ctx: target } = args;
    let mctx = MetalContext::new()?;
    let g = GgufFile::open(&model)?;
    let m = Model::from_gguf(&g)?;
    let mm = MetalModel::load(&mctx, &g, &m)?;

    let mf = MetalForward::new(&mctx, &mm);
    {
        let mut s = MetalSession::fresh(&mctx, &mm, 32)?;
        for i in 0..3 {
            let _ = mf.single_token(0, i as u32, &mut s)?;
        }
    }
    let mut s = MetalSession::fresh(&mctx, &mm, target + 16)?;
    for p in 0..(target as u32) {
        let _ = mf.single_token(0, p, &mut s)?;
    }
    let (_, wall_artifact, phases) = mf.single_token_phase_profiled(0, target as u32, &mut s)?;
    let phase_sum: f64 = phases.iter().map(|p| p.1).sum();
    println!(
        "[phase ctx={target}] phase_sum={phase_sum:.2} ms (production-realistic GPU)  \
         wall_artifact={wall_artifact:.2} ms (DO NOT use as prod ms/token)"
    );
    for (name, ms) in &phases {
        let pct = ms / phase_sum * 100.0;
        println!("[phase ctx={target}]   {name:25} {ms:7.2} ms  ({pct:5.1}%)");
    }
    Ok(())
}

fn argmax_i32(logits: &[f32]) -> i32 {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &v) in logits.iter().enumerate() {
        if v > best.1 {
            best = (i, v);
        }
    }
    best.0 as i32
}

fn compare_logits(ours: &[f32], oracle: &[f32]) -> (f64, f32, usize, usize) {
    debug_assert_eq!(ours.len(), oracle.len());
    let mut max_abs = 0.0f32;
    let mut argmax_ours = 0usize;
    let mut argmax_oracle = 0usize;
    let mut max_ours = f32::NEG_INFINITY;
    let mut max_oracle = f32::NEG_INFINITY;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for i in 0..ours.len() {
        max_abs = max_abs.max((ours[i] - oracle[i]).abs());
        if ours[i] > max_ours {
            max_ours = ours[i];
            argmax_ours = i;
        }
        if oracle[i] > max_oracle {
            max_oracle = oracle[i];
            argmax_oracle = i;
        }
        dot += ours[i] as f64 * oracle[i] as f64;
        na += (ours[i] as f64).powi(2);
        nb += (oracle[i] as f64).powi(2);
    }
    let cos = dot / (na.sqrt() * nb.sqrt() + 1e-30);
    (cos, max_abs, argmax_ours, argmax_oracle)
}
