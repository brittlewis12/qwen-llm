//! End-to-end load-spike: measures the full `Runtime::load_model_with_config`
//! wall time with the configured `PrefetchPolicy`, in a genuine cold-cache
//! starting state.
//!
//! Differs from `load_spike.rs`: this exercises the real production
//! load path (`GgufFile::open` -> model bind/load-plan preparation ->
//! optional prefetch -> prepared Metal load), including full Metal
//! buffer creation, not just a bare mmap-page walk. This is the number
//! that actually reflects "how long does the user wait before they can
//! call `sequence.step()`".
//!
//! Usage:
//!
//!   cargo run --release -p qwen-llm --example runtime_load_spike -- \
//!     <path-to-model.gguf> [--policy off|always|cold-only] \
//!     [--workers 4] [--chunk-mib 16] [--invalidate]
//!
//! Falsifier arms:
//!
//!   D  policy=off,       invalidate=true
//!   A  policy=always,    invalidate=true
//!   AC policy=cold-only, invalidate=true  (should behave like A when cold)
//!   W  policy=always,    invalidate=false (warm-load regression check)

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qwen_llm::cache_probe::{invalidate_file_cache, probe_file_residency};
use qwen_llm::pid_metrics::{PidDelta, PidSnapshot};
use qwen_llm::runtime::{DEFAULT_COLD_ONLY_THRESHOLD, LoadedModelConfig, PrefetchPolicy, Runtime};

#[derive(Debug)]
struct Args {
    model: PathBuf,
    policy: PrefetchPolicy,
    workers: usize,
    chunk_bytes: usize,
    invalidate: bool,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<PathBuf> = None;
    let mut policy = PrefetchPolicy::Off;
    let mut workers = 0usize;
    let mut chunk_mib = 0usize;
    let mut invalidate = false;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--policy" => {
                let v = it.next().ok_or("--policy requires a value")?;
                policy = match v.as_str() {
                    "off" => PrefetchPolicy::Off,
                    "always" => PrefetchPolicy::Always,
                    "cold-only" => PrefetchPolicy::cold_only(DEFAULT_COLD_ONLY_THRESHOLD)
                        .expect("DEFAULT_COLD_ONLY_THRESHOLD is a valid fraction"),
                    other => return Err(format!("unknown policy: {other}")),
                };
            }
            "--workers" => {
                workers = it
                    .next()
                    .ok_or("--workers value")?
                    .parse()
                    .map_err(|e| format!("--workers: {e}"))?;
            }
            "--chunk-mib" => {
                chunk_mib = it
                    .next()
                    .ok_or("--chunk-mib value")?
                    .parse()
                    .map_err(|e| format!("--chunk-mib: {e}"))?;
            }
            "--invalidate" => invalidate = true,
            other if other.starts_with("--") => {
                return Err(format!("unknown flag: {other}"));
            }
            other => {
                if model.is_some() {
                    return Err(format!("unexpected positional: {other}"));
                }
                model = Some(PathBuf::from(other));
            }
        }
    }
    Ok(Args {
        model: model.ok_or("missing model path")?,
        policy,
        workers,
        chunk_bytes: chunk_mib * 1024 * 1024,
        invalidate,
    })
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

fn fmt_wall(d: Duration) -> String {
    format!("{:>7.3} s", d.as_secs_f64())
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "usage: runtime_load_spike <model.gguf> [--policy off|always|cold-only] [--workers N] [--chunk-mib N] [--invalidate]"
            );
            std::process::exit(2);
        }
    };

    let file_size = match std::fs::metadata(&args.model) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("error: metadata({}): {e}", args.model.display());
            std::process::exit(1);
        }
    };

    println!("model:       {}", args.model.display());
    println!("size:        {:.2} GiB", gib(file_size));
    println!("policy:      {:?}", args.policy);
    if !matches!(args.policy, PrefetchPolicy::Off) {
        println!(
            "workers:     {}",
            if args.workers == 0 { 4 } else { args.workers }
        );
        println!(
            "chunk:       {} MiB",
            if args.chunk_bytes == 0 {
                16
            } else {
                args.chunk_bytes / (1024 * 1024)
            }
        );
    }
    println!("invalidate:  {}", args.invalidate);
    println!();
    let _ = std::io::stdout().flush();

    // ---- Untimed cache reset ----
    if let Ok(r) = probe_file_residency(&args.model) {
        println!(
            "pre-arm residency: {}/{} pages ({:.1}%)",
            r.resident_pages,
            r.total_pages,
            r.resident_fraction() * 100.0,
        );
    }
    if args.invalidate {
        match invalidate_file_cache(&args.model) {
            Ok(rep) => {
                println!(
                    "invalidate: {}/{} -> {}/{}",
                    rep.before.resident_pages,
                    rep.before.total_pages,
                    rep.after.resident_pages,
                    rep.after.total_pages,
                );
            }
            Err(e) => eprintln!("warn: invalidate: {e}"),
        }
    }
    println!();

    // ---- The timed path: full runtime load ----
    let config = LoadedModelConfig {
        prefetch_policy: args.policy,
        prefetch_workers: args.workers,
        prefetch_chunk_bytes: args.chunk_bytes,
        ..LoadedModelConfig::default()
    };

    let runtime = Runtime::metal().expect("runtime metal init");
    println!("device: {}", runtime.describe());
    println!();

    let pid_a = PidSnapshot::now().expect("pid snapshot start");
    let t0 = Instant::now();

    let loaded = match runtime.load_model_with_config(&args.model, config) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("error: load_model_with_config: {e}");
            std::process::exit(1);
        }
    };

    let total = t0.elapsed();
    let pid_b = PidSnapshot::now().expect("pid snapshot end");
    let pid_delta = PidDelta::between(pid_a, pid_b);

    let out = loaded.prefetch_outcome();
    println!("=== phase breakdown ===");
    let mut prefetch_wall = Duration::ZERO;
    if !matches!(out.policy, PrefetchPolicy::Off) {
        prefetch_wall = out.total_wall;
        println!("  prefetch:   {}", fmt_wall(prefetch_wall));
        for s in &out.shards {
            if s.skipped {
                println!(
                    "    shard {}: SKIPPED (pre-resident {:.1}%){}",
                    s.path.display(),
                    s.pre_resident_fraction * 100.0,
                    s.skipped_reason
                        .as_ref()
                        .map(|r| format!(" [{r}]"))
                        .unwrap_or_default(),
                );
            } else {
                let gbps = if s.wall.as_secs_f64() > 0.0 {
                    s.bytes_returned as f64 / s.wall.as_secs_f64() / 1e9
                } else {
                    0.0
                };
                println!(
                    "    shard {}: {:.2} GiB in {} ({:.2} GB/s effective, pre-resident {:.1}%)",
                    s.path.display(),
                    gib(s.bytes_returned),
                    fmt_wall(s.wall),
                    gbps,
                    s.pre_resident_fraction * 100.0,
                );
            }
        }
    } else {
        println!("  prefetch:   (Off)");
    }
    let open_plus_metal = total - prefetch_wall;
    println!(
        "  open+metal: {}  (inferred: total - prefetch)",
        fmt_wall(open_plus_metal)
    );
    println!();
    println!("TOTAL LOAD:   {}", fmt_wall(total));
    println!(
        "rusage: pageins={:>7}  diskR={:>7.2} GiB  diskW={:>5.1} MiB  \u{0394}RSS={:+7.2} GiB  \u{0394}footprint={:+7.2} GiB",
        pid_delta.pageins,
        gib(pid_delta.diskio_bytesread),
        pid_delta.diskio_byteswritten as f64 / (1u64 << 20) as f64,
        pid_delta.resident_size_delta as f64 / (1u64 << 30) as f64,
        pid_delta.phys_footprint_delta as f64 / (1u64 << 30) as f64,
    );

    if let Ok(r) = probe_file_residency(&args.model) {
        println!(
            "post-arm residency: {}/{} pages ({:.1}%)",
            r.resident_pages,
            r.total_pages,
            r.resident_fraction() * 100.0,
        );
    }
}
