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
//!     <model.gguf> [--policy default|off|always|cold-only] [--invalidate] \
//!     [--prompt "text"] [--workers 4] [--chunk-mib 16]

use std::io::Write;
use std::os::unix::fs::MetadataExt;

#[path = "support/diag_subscriber.rs"]
mod diag_subscriber;
use diag_subscriber::install_example_diag_subscriber;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use qwen_llm::cache_probe::{invalidate_file_cache, probe_file_residency};
use qwen_llm::pid_metrics::{PidDelta, PidSnapshot};
use qwen_llm::runtime::{
    DEFAULT_COLD_ONLY_THRESHOLD, LoadedModelConfig, PrefetchAction, PrefetchOutcome,
    PrefetchPolicy, PrefetchSuppressionReason, Runtime, SequenceConfig,
};
use sha2::{Digest, Sha256};

const DEFAULT_PROMPT: &str = "The capital of France is";
const HARNESS_SOURCE: &str = include_str!("first_byte_spike.rs");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PolicySelection {
    DefaultConfig,
    Explicit(PrefetchPolicy),
}

#[derive(Debug)]
struct Args {
    model: PathBuf,
    policy: PolicySelection,
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
    intent: LoadIntent,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LoadIntent {
    ForceOnly,
    DisposableSingleTurn,
}

fn parse_args_from(args: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut model: Option<PathBuf> = None;
    let mut policy = PolicySelection::Explicit(PrefetchPolicy::Off);
    let mut workers = 0usize;
    let mut chunk_mib = 0usize;
    let mut invalidate = false;
    let mut prompt = DEFAULT_PROMPT.to_string();
    let mut max_context = 512usize;
    let mut tokens = 0usize;
    let mut intent = LoadIntent::ForceOnly;

    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--policy" => {
                let v = it.next().ok_or("--policy value")?;
                policy = match v.as_str() {
                    "default" => PolicySelection::DefaultConfig,
                    "off" => PolicySelection::Explicit(PrefetchPolicy::Off),
                    "always" => PolicySelection::Explicit(PrefetchPolicy::Always),
                    "cold-only" => PolicySelection::Explicit(
                        PrefetchPolicy::cold_only(DEFAULT_COLD_ONLY_THRESHOLD)
                            .expect("DEFAULT_COLD_ONLY_THRESHOLD is a valid fraction"),
                    ),
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
            "--intent" => {
                let v = it.next().ok_or("--intent value")?;
                intent = match v.as_str() {
                    "force-only" => LoadIntent::ForceOnly,
                    "disposable" => LoadIntent::DisposableSingleTurn,
                    other => return Err(format!("unknown intent: {other}")),
                };
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
    let args = Args {
        model: model.ok_or("missing model path")?,
        policy,
        workers,
        chunk_bytes: chunk_mib * 1024 * 1024,
        invalidate,
        prompt,
        max_context,
        tokens,
        intent,
    };
    if matches!(args.policy, PolicySelection::DefaultConfig)
        && (args.workers != 0 || args.chunk_bytes != 0)
    {
        return Err("--policy default cannot override workers or chunk size".to_string());
    }
    Ok(args)
}

fn parse_args() -> Result<Args, String> {
    parse_args_from(std::env::args().skip(1))
}

fn loaded_model_config(args: &Args) -> Result<LoadedModelConfig, String> {
    match args.policy {
        PolicySelection::DefaultConfig => {
            if args.workers != 0 || args.chunk_bytes != 0 {
                return Err("--policy default cannot override workers or chunk size".to_string());
            }
            Ok(LoadedModelConfig::default())
        }
        PolicySelection::Explicit(prefetch_policy) => Ok(LoadedModelConfig {
            prefetch_policy,
            prefetch_workers: args.workers,
            prefetch_chunk_bytes: args.chunk_bytes,
            ..LoadedModelConfig::default()
        }),
    }
}

fn policy_source(selection: PolicySelection) -> &'static str {
    match selection {
        PolicySelection::DefaultConfig => "default",
        PolicySelection::Explicit(_) => "explicit",
    }
}

fn policy_fields(policy: PrefetchPolicy) -> (&'static str, Option<f64>) {
    match policy {
        PrefetchPolicy::Off => ("off", None),
        PrefetchPolicy::Always => ("always", None),
        PrefetchPolicy::ColdOnly { threshold } => ("cold-only", Some(threshold.value())),
    }
}

fn configured_policy_line(config: LoadedModelConfig) -> String {
    let (policy, threshold) = policy_fields(config.prefetch_policy);
    let threshold = threshold.map_or_else(|| "none".to_string(), |value| value.to_string());
    format!(
        "policy_configured={policy} threshold={threshold} workers={} chunk_bytes={}",
        config.prefetch_workers, config.prefetch_chunk_bytes
    )
}

fn observed_policy_line(outcome: &PrefetchOutcome) -> String {
    let (policy, threshold) = policy_fields(outcome.policy);
    let threshold = threshold.map_or_else(|| "none".to_string(), |value| value.to_string());
    let (action, suppressed) = match outcome.action {
        PrefetchAction::ConfiguredPolicy => ("configured-policy", false),
        PrefetchAction::Suppressed { .. } => ("suppressed", true),
    };
    format!(
        "policy_observed={policy} threshold={threshold} action={action} suppressed={suppressed}"
    )
}

fn prefetch_exact_line(outcome: &PrefetchOutcome) -> String {
    format!(
        "prefetch_exact=events:{},prefetched:{},skipped:{},bytes:{}",
        outcome.shards.len(),
        outcome.shards_prefetched(),
        outcome.shards_skipped(),
        outcome.bytes_returned_total()
    )
}

fn timing_exact_line(load_wall: Duration, first_byte_wall: Duration) -> String {
    format!(
        "timing_exact=load_us:{},first_byte_us:{}",
        load_wall.as_micros(),
        first_byte_wall.as_micros()
    )
}

fn rusage_exact_line(delta: PidDelta) -> String {
    format!(
        "rusage_exact=pageins:{},disk_read_bytes:{},disk_write_bytes:{},rss_delta_bytes:{},footprint_delta_bytes:{}",
        delta.pageins,
        delta.diskio_bytesread,
        delta.diskio_byteswritten,
        delta.resident_size_delta,
        delta.phys_footprint_delta
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TargetIdentity {
    device: u64,
    inode: u64,
    size_bytes: u64,
    mtime_ns: i128,
}

impl TargetIdentity {
    fn capture(path: &PathBuf) -> std::io::Result<Self> {
        let metadata = std::fs::metadata(path)?;
        Ok(Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            size_bytes: metadata.len(),
            mtime_ns: i128::from(metadata.mtime()) * 1_000_000_000
                + i128::from(metadata.mtime_nsec()),
        })
    }

    fn stable(self) -> String {
        format!(
            "dev:{},ino:{},size:{},mtime_ns:{}",
            self.device, self.inode, self.size_bytes, self.mtime_ns
        )
    }
}

fn harness_source_sha256() -> String {
    format!("{:x}", Sha256::digest(HARNESS_SOURCE.as_bytes()))
}

fn stable_before_invalidate_lines(
    selection: PolicySelection,
    config: LoadedModelConfig,
    identity: TargetIdentity,
) -> Vec<String> {
    vec![
        format!("harness_source_sha256={}", harness_source_sha256()),
        format!("policy_source={}", policy_source(selection)),
        configured_policy_line(config),
        format!("target_identity_before_invalidate={}", identity.stable()),
    ]
}

fn stable_after_invalidate_line(identity: TargetIdentity) -> String {
    format!("target_identity_after_invalidate={}", identity.stable())
}

fn stable_after_load_lines(outcome: &PrefetchOutcome, identity: TargetIdentity) -> Vec<String> {
    vec![
        observed_policy_line(outcome),
        format!("target_identity_after_load={}", identity.stable()),
        prefetch_exact_line(outcome),
    ]
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
    install_example_diag_subscriber();
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    let config = match loaded_model_config(&args) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("error: {error}");
            std::process::exit(2);
        }
    };
    let default_mode = matches!(args.policy, PolicySelection::DefaultConfig);
    let target_identity_before = if default_mode {
        Some(TargetIdentity::capture(&args.model).expect("target identity before invalidate"))
    } else {
        None
    };

    let file_size = std::fs::metadata(&args.model).map(|m| m.len()).unwrap_or(0);

    println!("model:      {}", args.model.display());
    println!("size:       {:.2} GiB", gib(file_size));
    println!("policy:     {:?}", config.prefetch_policy);
    println!("intent:     {:?}", args.intent);
    println!("invalidate: {}", args.invalidate);
    println!("prompt:     {:?}", args.prompt);
    if let Some(identity) = target_identity_before {
        for line in stable_before_invalidate_lines(args.policy, config, identity) {
            println!("{line}");
        }
    }
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
            Err(error) if default_mode => panic!("invalidate failed: {error}"),
            Err(error) => eprintln!("warn: invalidate: {error}"),
        }
    }
    if let Some(before) = target_identity_before {
        let after = TargetIdentity::capture(&args.model).expect("target identity after invalidate");
        assert_eq!(before, after, "target identity changed across invalidation");
        println!("{}", stable_after_invalidate_line(after));
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
    let pid_a = PidSnapshot::now().unwrap();
    let t = Instant::now();
    let loaded = match args.intent {
        LoadIntent::ForceOnly => runtime
            .load_model_with_config(&args.model, config)
            .expect("load_model_with_config"),
        LoadIntent::DisposableSingleTurn => runtime
            .load_model_for_disposable_single_turn_with_config(&args.model, config)
            .expect("load_model_for_disposable_single_turn_with_config"),
    };
    let load_wall = t.elapsed();
    let pid_b = PidSnapshot::now().unwrap();
    print_phase("load:", load_wall, PidDelta::between(pid_a, pid_b));

    // Report the internal prefetch breakdown for observability.
    let outcome = loaded.prefetch_outcome();
    assert_eq!(outcome.policy, config.prefetch_policy);
    if let Some(before) = target_identity_before {
        assert_eq!(outcome.action, PrefetchAction::ConfiguredPolicy);
        let after = TargetIdentity::capture(&args.model).expect("target identity after load");
        assert_eq!(before, after, "target identity changed across load");
        for line in stable_after_load_lines(outcome, after) {
            println!("{line}");
        }
    }
    if let PrefetchAction::Suppressed {
        reason: PrefetchSuppressionReason::AuthenticatedDisposableAutoA3bDirectPread,
    } = outcome.action
    {
        println!("  prefetch:    suppressed (authenticated disposable Auto A3B direct pread)");
    } else if !matches!(outcome.policy, PrefetchPolicy::Off) {
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
    if default_mode {
        println!("{}", timing_exact_line(load_wall, first_byte_wall));
        println!("{}", rusage_exact_line(total_delta));
    }

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

#[cfg(test)]
mod tests {
    use super::*;
    use qwen_llm::runtime::ShardPrefetch;

    fn args(values: &[&str]) -> Result<Args, String> {
        parse_args_from(values.iter().map(|value| (*value).to_string()))
    }

    #[test]
    fn default_policy_selection_is_exact() {
        let default_args = args(&["model.gguf", "--policy", "default"]).unwrap();
        assert_eq!(default_args.policy, PolicySelection::DefaultConfig);
        let default_config = loaded_model_config(&default_args).unwrap();
        assert_eq!(default_config, LoadedModelConfig::default());
        assert_eq!(policy_source(default_args.policy), "default");
        assert_eq!(
            configured_policy_line(default_config),
            "policy_configured=cold-only threshold=0.9 workers=0 chunk_bytes=0"
        );

        let omitted_args = args(&["model.gguf"]).unwrap();
        assert_eq!(
            omitted_args.policy,
            PolicySelection::Explicit(PrefetchPolicy::Off)
        );
        let omitted_config = loaded_model_config(&omitted_args).unwrap();
        assert_eq!(omitted_config.prefetch_policy, PrefetchPolicy::Off);
        assert_eq!(omitted_config.prefetch_workers, 0);
        assert_eq!(omitted_config.prefetch_chunk_bytes, 0);
        assert_eq!(policy_source(omitted_args.policy), "explicit");

        let cold_only = PrefetchPolicy::cold_only(DEFAULT_COLD_ONLY_THRESHOLD).unwrap();
        for (name, expected) in [
            ("off", PrefetchPolicy::Off),
            ("always", PrefetchPolicy::Always),
            ("cold-only", cold_only),
        ] {
            let parsed = args(&["model.gguf", "--policy", name]).unwrap();
            assert_eq!(parsed.policy, PolicySelection::Explicit(expected));
            assert_eq!(
                loaded_model_config(&parsed).unwrap().prefetch_policy,
                expected
            );
        }

        let expected_error = "--policy default cannot override workers or chunk size";
        assert_eq!(
            args(&["model.gguf", "--policy", "default", "--workers", "4"]).unwrap_err(),
            expected_error
        );
        assert_eq!(
            args(&["model.gguf", "--policy", "default", "--chunk-mib", "16"]).unwrap_err(),
            expected_error
        );
        let zero_override = args(&[
            "model.gguf",
            "--policy",
            "default",
            "--workers",
            "0",
            "--chunk-mib",
            "0",
        ])
        .unwrap();
        assert_eq!(
            loaded_model_config(&zero_override).unwrap(),
            LoadedModelConfig::default()
        );

        let outcome = PrefetchOutcome {
            policy: default_config.prefetch_policy,
            action: PrefetchAction::ConfiguredPolicy,
            shards: vec![ShardPrefetch {
                path: PathBuf::from("model.gguf"),
                bytes_returned: 16_817_244_384,
                wall: Duration::from_secs(1),
                pre_resident_fraction: 0.0,
                skipped: false,
                skipped_reason: None,
            }],
            total_wall: Duration::from_secs(1),
        };
        assert_eq!(
            observed_policy_line(&outcome),
            "policy_observed=cold-only threshold=0.9 action=configured-policy suppressed=false"
        );
        assert_eq!(
            prefetch_exact_line(&outcome),
            "prefetch_exact=events:1,prefetched:1,skipped:0,bytes:16817244384"
        );

        let identity = TargetIdentity {
            device: 1,
            inode: 2,
            size_bytes: 16_817_244_384,
            mtime_ns: 3,
        };
        assert_eq!(identity.stable(), "dev:1,ino:2,size:16817244384,mtime_ns:3");
        let source_hash = harness_source_sha256();
        let rusage = PidDelta {
            pageins: 4,
            diskio_bytesread: 5,
            diskio_byteswritten: 6,
            resident_size_delta: -7,
            phys_footprint_delta: 8,
        };
        let mut stable_lines = stable_before_invalidate_lines(
            PolicySelection::DefaultConfig,
            default_config,
            identity,
        );
        stable_lines.push(stable_after_invalidate_line(identity));
        stable_lines.extend(stable_after_load_lines(&outcome, identity));
        stable_lines.push(timing_exact_line(
            Duration::from_micros(9),
            Duration::from_micros(10),
        ));
        stable_lines.push(rusage_exact_line(rusage));
        assert_eq!(
            stable_lines,
            vec![
                format!("harness_source_sha256={source_hash}"),
                "policy_source=default".to_string(),
                "policy_configured=cold-only threshold=0.9 workers=0 chunk_bytes=0"
                    .to_string(),
                "target_identity_before_invalidate=dev:1,ino:2,size:16817244384,mtime_ns:3"
                    .to_string(),
                "target_identity_after_invalidate=dev:1,ino:2,size:16817244384,mtime_ns:3"
                    .to_string(),
                "policy_observed=cold-only threshold=0.9 action=configured-policy suppressed=false"
                    .to_string(),
                "target_identity_after_load=dev:1,ino:2,size:16817244384,mtime_ns:3"
                    .to_string(),
                "prefetch_exact=events:1,prefetched:1,skipped:0,bytes:16817244384"
                    .to_string(),
                "timing_exact=load_us:9,first_byte_us:10".to_string(),
                "rusage_exact=pageins:4,disk_read_bytes:5,disk_write_bytes:6,rss_delta_bytes:-7,footprint_delta_bytes:8"
                    .to_string(),
            ]
        );

        let capture_path =
            std::env::temp_dir().join(format!("qwen-first-byte-identity-{}", std::process::id()));
        std::fs::write(&capture_path, b"identity").unwrap();
        let captured = TargetIdentity::capture(&capture_path).unwrap();
        assert_eq!(captured.size_bytes, 8);
        assert!(captured.device > 0);
        assert!(captured.inode > 0);
        std::fs::remove_file(capture_path).unwrap();

        assert_eq!(source_hash.len(), 64);
        assert!(source_hash.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_eq!(source_hash, source_hash.to_ascii_lowercase());
        let source_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/first_byte_spike.rs");
        let disk_source = std::fs::read(source_path).unwrap();
        assert_eq!(source_hash, format!("{:x}", Sha256::digest(disk_source)));
    }
}
