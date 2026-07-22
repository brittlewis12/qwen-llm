//! First-byte spike: measure the *full* user-perceived cold-start
//! wall — from process start-ish to the first decoded token — with
//! and without cache prefetch.
//!
//! Answers the concern surfaced in the review of the previous
//! `runtime_load_spike`: v0.591 documented that whole-shard prefault
//! *increased* first-byte latency by 368–435 ms even though it saved
//! load-model time. This example measures whether the pread-based
//! warmer has the same regression, or whether the load-model saving
//! propagates to first-byte.
//!
//! Phases (all wall time):
//!
//!   load     Runtime::load_model_with_config (includes prefetch if enabled)
//!   tok      Tokenizer::from_gguf + prompt encode
//!   session  MetalSession::fresh (KV/GDN scratch alloc)
//!   prefill  MetalForward::single_token per prompt token (positions 0..N-1)
//!   decode1  MetalForward::single_token at position N -> first output token
//!
//! Report: per-phase wall + total first-byte, plus pid_metrics deltas
//! per phase for physical-disk attribution.
//!
//! Usage:
//!
//!   cargo run --release -p qwen-llm --example first_byte_spike -- \
//!     <model.gguf> [--policy off|always|cold-only] [--invalidate] \
//!     [--prompt "text"] [--workers 4] [--chunk-mib 16]

use std::io::Write;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qwen_llm::cache_probe::{invalidate_file_cache, probe_file_residency};
use qwen_llm::metal_forward::MetalSession;
use qwen_llm::pid_metrics::{PidDelta, PidSnapshot};
use qwen_llm::runtime::{LoadedModelConfig, PrefetchPolicy, Runtime, SequenceConfig};

const DEFAULT_PROMPT: &str = "The capital of France is";

#[derive(Debug)]
struct Args {
    model: PathBuf,
    policy: PrefetchPolicy,
    workers: usize,
    chunk_bytes: usize,
    invalidate: bool,
    prompt: String,
    max_context: usize,
    /// Additional decoded tokens beyond the first-byte token. 0 stops
    /// after first-byte (the default). Larger values measure the
    /// sustained decode throughput, which is where a v0.591-style
    /// warm-decode regression would surface.
    tokens: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<PathBuf> = None;
    let mut policy = PrefetchPolicy::Off;
    let mut workers = 0usize;
    let mut chunk_mib = 0usize;
    let mut invalidate = false;
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut max_context = 512usize;
    let mut tokens = 0usize;

    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        match a.as_str() {
            "--policy" => {
                let v = it.next().ok_or("--policy value")?;
                policy = match v.as_str() {
                    "off" => PrefetchPolicy::Off,
                    "always" => PrefetchPolicy::Always,
                    "cold-only" => {
                        PrefetchPolicy::cold_only(0.5).expect("0.5 is a valid threshold")
                    }
                    other => return Err(format!("unknown policy: {other}")),
                };
            }
            "--workers" => {
                workers = it
                    .next()
                    .ok_or("--workers value")?
                    .parse()
                    .map_err(|e| format!("--workers: {e}"))?
            }
            "--chunk-mib" => {
                chunk_mib = it
                    .next()
                    .ok_or("--chunk-mib value")?
                    .parse()
                    .map_err(|e| format!("--chunk-mib: {e}"))?
            }
            "--invalidate" => invalidate = true,
            "--prompt" => prompt = it.next().ok_or("--prompt value")?,
            "--max-context" => {
                max_context = it
                    .next()
                    .ok_or("--max-context value")?
                    .parse()
                    .map_err(|e| format!("--max-context: {e}"))?
            }
            "--tokens" => {
                tokens = it
                    .next()
                    .ok_or("--tokens value")?
                    .parse()
                    .map_err(|e| format!("--tokens: {e}"))?
            }
            other if other.starts_with("--") => return Err(format!("unknown flag: {other}")),
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
        prompt,
        max_context,
        tokens,
    })
}

fn gib(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}
fn fmt_wall(d: Duration) -> String {
    format!("{:>7.3} s", d.as_secs_f64())
}

fn print_phase(label: &str, wall: Duration, delta: PidDelta) {
    println!(
        "{label:<12} {}  pageins={:>7}  diskR={:>6.2} GiB",
        fmt_wall(wall),
        delta.pageins,
        gib(delta.diskio_bytesread),
    );
}

fn argmax(logits: &[f32]) -> usize {
    let mut best_i = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best_i = i;
        }
    }
    best_i
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };

    let file_size = std::fs::metadata(&args.model).map(|m| m.len()).unwrap_or(0);

    println!("model:      {}", args.model.display());
    println!("size:       {:.2} GiB", gib(file_size));
    println!("policy:     {:?}", args.policy);
    println!("invalidate: {}", args.invalidate);
    println!("prompt:     {:?}", args.prompt);
    println!();
    let _ = std::io::stdout().flush();

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
            Ok(rep) => println!(
                "invalidate: {}/{} -> {}/{}",
                rep.before.resident_pages,
                rep.before.total_pages,
                rep.after.resident_pages,
                rep.after.total_pages,
            ),
            Err(e) => eprintln!("warn: invalidate: {e}"),
        }
    }
    println!();

    // Timer starts here — this is what the user would see as "TTFT".
    let pid_start = PidSnapshot::now().expect("pid_start");
    let t_start = Instant::now();

    // Runtime init (usually tiny; measured to be honest about it).
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let runtime = Runtime::metal().expect("runtime metal");
    let init_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("runtime:", init_wall, PidDelta::between(pid_a, pid_b));

    // Load model (with configured prefetch policy).
    let config = LoadedModelConfig {
        prefetch_policy: args.policy,
        prefetch_workers: args.workers,
        prefetch_chunk_bytes: args.chunk_bytes,
        ..LoadedModelConfig::default()
    };
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let loaded = runtime
        .load_model_with_config(&args.model, config)
        .expect("load_model_with_config");
    let load_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("load:", load_wall, PidDelta::between(pid_a, pid_b));

    // Report the internal prefetch breakdown for observability.
    let outcome = loaded.prefetch_outcome();
    if !matches!(outcome.policy, PrefetchPolicy::Off) {
        let bytes_returned = outcome.bytes_returned_total();
        println!(
            "  prefetch:    {}  {} shards prefetched, {} skipped, {:.2} GiB returned",
            fmt_wall(outcome.total_wall),
            outcome.shards_prefetched(),
            outcome.shards_skipped(),
            gib(bytes_returned),
        );
    }

    // Tokenizer + encode.
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let tok = loaded.tokenizer().expect("tokenizer");
    let ids = tok.encode(&args.prompt, false).expect("encode");
    let tok_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("tok+encode:", tok_wall, PidDelta::between(pid_a, pid_b));
    println!("  prompt encoded to {} tokens: {:?}", ids.len(), ids);
    assert!(!ids.is_empty(), "empty prompt encoding");

    // Session (KV/GDN scratch).
    let seq_config = SequenceConfig::new(args.max_context);
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let mut sequence = loaded.create_sequence(seq_config).expect("create_sequence");
    let sess_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("session:", sess_wall, PidDelta::between(pid_a, pid_b));

    let forward = loaded.forward();

    // Prefill: single-token per prompt token, positions 0..ids.len()-1.
    // (Slower than bulk prefill; matches trellis3_t9b_ppl pattern which
    // exercises the same forward path we care about here.)
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let mut last_logits: Vec<f32> = Vec::new();
    for (i, &token) in ids.iter().enumerate() {
        // SAFETY: raw session access is `unsafe` because it lets a caller
        // bypass the `LoadedModel` ownership check (post-commit 468bc37).
        // Here `sequence` was created by `loaded.create_sequence(..)` above,
        // so we are the only owner and the model provenance is preserved.
        let session = unsafe { sequence.metal_session_mut() };
        last_logits = forward
            .single_token(token, i as u32, session)
            .expect("single_token");
    }
    let prefill_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("prefill:", prefill_wall, PidDelta::between(pid_a, pid_b));
    sequence.advance_by(ids.len()).expect("advance");

    // First decoded token: argmax over the last position's logits.
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let first_token = argmax(&last_logits) as i32;
    let decode_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("decode1:", decode_wall, PidDelta::between(pid_a, pid_b));

    let first_byte_wall = t_start.elapsed();

    // Sustained decode: N more tokens after first-byte. Feeds
    // argmax-selected tokens back through single_token to measure
    // steady-state decode throughput. This is the phase that would
    // surface a v0.591-style warm-decode regression if pread-warming
    // caused one (measured on cold decode; if warm and cold decode
    // land the same tok/s, no regression).
    let mut decoded_tokens: Vec<i32> = Vec::with_capacity(args.tokens);
    if args.tokens > 0 {
        decoded_tokens.push(first_token);
        let pid_a = PidSnapshot::now().unwrap();
        let t = Instant::now();
        let mut tok = first_token;
        for step in 0..args.tokens {
            // Position of THIS token's forward pass: prompt.len() +
            // 1 (for first_token) + step (for the tokens we've already
            // decoded).
            let pos = (ids.len() + 1 + step) as u32;
            // SAFETY: sequence is our own; see prefill safety note.
            let session = unsafe { sequence.metal_session_mut() };
            let logits = forward
                .single_token(tok, pos, session)
                .expect("sustained decode single_token");
            sequence.advance_by(1).expect("advance");
            tok = argmax(&logits) as i32;
            decoded_tokens.push(tok);
        }
        let decode_n_wall = t.elapsed();
        let pid_b = PidSnapshot::now().unwrap();
        let tok_per_s = args.tokens as f64 / decode_n_wall.as_secs_f64();
        let ms_per_tok = decode_n_wall.as_secs_f64() * 1e3 / args.tokens as f64;
        println!(
            "decode {:>3}:   {}  ({:.2} tok/s, {:.2} ms/tok)",
            args.tokens,
            fmt_wall(decode_n_wall),
            tok_per_s,
            ms_per_tok,
        );
        print_phase(
            "  (rusage)",
            Duration::ZERO,
            PidDelta::between(pid_a, pid_b),
        );
    }

    let total = t_start.elapsed();
    let pid_end = PidSnapshot::now().unwrap();
    let total_delta = PidDelta::between(pid_start, pid_end);

    println!();
    println!("FIRST BYTE:  {}", fmt_wall(first_byte_wall));
    if args.tokens > 0 {
        println!("TOTAL (fb+{}): {}", args.tokens, fmt_wall(total));
    }
    println!(
        "rusage total: pageins={:>7}  diskR={:>6.2} GiB  diskW={:>4.1} MiB  \u{0394}RSS={:+7.2} GiB",
        total_delta.pageins,
        gib(total_delta.diskio_bytesread),
        total_delta.diskio_byteswritten as f64 / (1u64 << 20) as f64,
        total_delta.resident_size_delta as f64 / (1u64 << 30) as f64,
    );

    let piece = tok.decode_piece(first_token);
    println!("first token: id={first_token} piece={piece:?}");
    if !decoded_tokens.is_empty() {
        let decoded = tok.decode(&decoded_tokens);
        println!("decoded: {decoded:?}");
    }

    if let Ok(r) = probe_file_residency(&args.model) {
        println!(
            "post-arm residency: {}/{} pages ({:.1}%)",
            r.resident_pages,
            r.total_pages,
            r.resident_fraction() * 100.0,
        );
    }
}
