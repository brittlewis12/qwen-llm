//! CPU-only sampler replay: times the production sampler on identical
//! captured full-vocabulary logits rows, with its phase profile, so sampler
//! changes are judged on the same inputs rather than on decode trajectories
//! that diverge. No Metal, no model.

use anyhow::{Context, Result, ensure};
use clap::Parser;
use qwen_llm::sampling::{Sampler, SamplingConfig};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser, Debug)]
pub struct SamplerReplayArgs {
    /// Little-endian F32 logits rows (for example from the GLM acquisition
    /// test `capture_release_sampling_logits`).
    #[arg(long)]
    logits: PathBuf,
    /// Entries per row (vocabulary size).
    #[arg(long)]
    vocab: usize,
    /// Temperature (0 is greedy).
    #[arg(long, default_value = "1.0")]
    temp: f32,
    /// Top-k (0 is off).
    #[arg(long, default_value = "0")]
    top_k: usize,
    #[arg(long, default_value = "0.95")]
    top_p: f32,
    #[arg(long, default_value = "0.0")]
    min_p: f32,
    #[arg(long, default_value = "42")]
    seed: u64,
    /// Timed passes over every row after one warm-up pass.
    #[arg(long, default_value = "5")]
    repeats: usize,
    /// Acquisition sidecar (`qwen.sampler_replay_logits.v1`): replays each
    /// recorded request with its own seed and requires the recorded tokens,
    /// so a sampler change is checked against the stream that produced the
    /// rows.
    #[arg(long)]
    sidecar: Option<PathBuf>,
}

/// Replays every recorded request with its seed; returns the rows checked.
fn verify_recorded_stream(
    path: &PathBuf,
    rows: &[Vec<f32>],
    config: SamplingConfig,
) -> Result<usize> {
    let sidecar: serde_json::Value = serde_json::from_slice(
        &std::fs::read(path).with_context(|| format!("read {}", path.display()))?,
    )?;
    ensure!(
        sidecar["schema"] == "qwen.sampler_replay_logits.v1",
        "unknown sidecar schema"
    );
    let mut next = 0;
    for request in sidecar["requests"].as_array().context("sidecar requests")? {
        let seed = request["seed"].as_u64().context("request seed")?;
        let recorded = request["sampled"].as_array().context("request tokens")?;
        let mut sampler = Sampler::new(SamplingConfig { seed, ..config })?;
        for expected in recorded {
            let row = rows
                .get(next)
                .context("sidecar names more rows than the file holds")?;
            let token = sampler.sample(row)?.token;
            ensure!(
                Some(i64::from(token)) == expected.as_i64(),
                "row {next}: sampled {token}, recorded {expected}"
            );
            next += 1;
        }
    }
    ensure!(
        next == rows.len(),
        "sidecar covers {next} of {} rows",
        rows.len()
    );
    Ok(next)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let index = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[index]
}

pub fn run(args: SamplerReplayArgs, identity: serde_json::Value) -> Result<()> {
    ensure!(
        args.vocab > 0 && args.repeats > 0,
        "vocab and repeats must be positive"
    );
    let bytes =
        std::fs::read(&args.logits).with_context(|| format!("read {}", args.logits.display()))?;
    let row_bytes = args.vocab * 4;
    ensure!(
        !bytes.is_empty() && bytes.len() % row_bytes == 0,
        "{} bytes is not a whole number of {}-entry F32 rows",
        bytes.len(),
        args.vocab
    );
    let rows: Vec<Vec<f32>> = bytes
        .chunks_exact(row_bytes)
        .map(|row| {
            row.chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        })
        .collect();
    let config = SamplingConfig {
        temperature: args.temp,
        top_k: args.top_k,
        top_p: args.top_p,
        min_p: args.min_p,
        seed: args.seed,
    }
    .validate()?;
    let verified_rows = args
        .sidecar
        .as_ref()
        .map(|path| verify_recorded_stream(path, &rows, config))
        .transpose()?;
    // Warm-up pass, then timed passes; one sampler per pass so every pass
    // draws the same RNG stream over the same rows.
    let mut tokens = Vec::new();
    let mut sampler = Sampler::new(config)?;
    for row in &rows {
        tokens.push(sampler.sample(row)?.token);
    }
    let mut per_call = Vec::with_capacity(rows.len() * args.repeats);
    for _ in 0..args.repeats {
        let mut sampler = Sampler::new(config)?;
        for (row, &expected) in rows.iter().zip(&tokens) {
            let started = Instant::now();
            let token = sampler.sample(row)?.token;
            per_call.push(started.elapsed().as_secs_f64() * 1e3);
            ensure!(token == expected, "replay is not deterministic");
        }
    }
    per_call.sort_by(f64::total_cmp);
    // Phase attribution on the same rows (separately timed; the profile's
    // own timers add overhead, so totals are reported apart).
    let mut sampler = Sampler::new(config)?;
    let mut phases = serde_json::Map::new();
    let mut kept = Vec::new();
    for row in &rows {
        let (_, profile) = sampler.sample_profiled(row)?;
        kept.push(profile.after_top_p as f64);
        for (name, value) in [
            ("candidate_fill_ms", profile.candidate_fill_ms),
            ("top_k_order_ms", profile.top_k_order_ms),
            ("min_p_ms", profile.min_p_ms),
            ("probability_weights_ms", profile.probability_weights_ms),
            ("top_p_ms", profile.top_p_ms),
            ("categorical_ms", profile.categorical_ms),
            ("total_ms", profile.total_ms),
        ] {
            let entry = phases.entry(name).or_insert(serde_json::json!(0.0));
            *entry = serde_json::json!(entry.as_f64().unwrap() + value / rows.len() as f64);
        }
    }
    kept.sort_by(f64::total_cmp);
    let mean = per_call.iter().sum::<f64>() / per_call.len() as f64;
    let report = serde_json::json!({
        "schema": "qwen.sampler_replay.v1",
        "build_identity": identity,
        "logits": args.logits, "rows": rows.len(), "vocab": args.vocab,
        "config": {"temperature": config.temperature, "top_k": config.top_k, "top_p": config.top_p,
            "min_p": config.min_p, "seed": config.seed},
        "repeats": args.repeats,
        "recorded_stream_rows_verified": verified_rows,
        "per_call_ms": {"mean": mean, "p50": percentile(&per_call, 0.5),
            "p90": percentile(&per_call, 0.9), "max": percentile(&per_call, 1.0)},
        "profiled_mean_ms": phases,
        "nucleus_size": {"p50": percentile(&kept, 0.5), "p90": percentile(&kept, 0.9),
            "max": percentile(&kept, 1.0)},
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
