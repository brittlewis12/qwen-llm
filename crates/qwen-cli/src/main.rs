//! `qwen` — interactive CLI for the qwen-llm engine.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use qwen_llm::metal_dflash::{MetalDFlashLayerMajorScratch, prefill_tokens_with_multi_hidden};
use qwen_llm::metal_forward::MetalForward;
use qwen_llm::runtime::{LoadedModel, LoadedModelConfig, Runtime, Sequence, SequenceConfig};
use qwen_llm::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

#[derive(Parser, Debug)]
#[command(name = "qwen", version, about = "qwen-llm inference CLI")]
struct Args {
    /// Path to a GGUF file (Qwen 3.5 / 3.6 family).
    #[arg(short = 'm', long)]
    model: Option<std::path::PathBuf>,

    /// Print device info and exit.
    #[arg(long)]
    info: bool,

    /// Raw prompt text for a single-turn greedy generation.
    #[arg(short = 'p', long, conflicts_with = "prompt_file")]
    prompt: Option<String>,

    /// Read raw prompt text from a file.
    #[arg(long, conflicts_with = "prompt")]
    prompt_file: Option<PathBuf>,

    /// Read JSONL request objects from a file or '-' while keeping one model loaded.
    #[arg(long, conflicts_with_all = ["prompt", "prompt_file"])]
    requests_jsonl: Option<PathBuf>,

    /// Number of greedy tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    tokens: usize,

    /// Prompt prefill chunk size. The safe default mirrors qwen-bench.
    #[arg(long, default_value_t = 1024)]
    prefill_chunk: usize,

    /// Override sequence capacity. Defaults to prompt + generated tokens + slack.
    #[arg(long)]
    max_context_tokens: Option<usize>,

    /// Prefix-cache byte budget in MiB; oversized snapshots are retained alone.
    #[arg(long, default_value_t = 16 * 1024)]
    prefix_cache_max_mib: u64,

    /// Cache this many exact prompt tokens as the reusable prefix for requests.
    #[arg(long)]
    cache_prefix_tokens: Option<usize>,

    /// Append per-request JSON stats for multi-request runs.
    ///
    /// Timing fields are model-internal; this JSONL mode writes each completion
    /// after full decode rather than streaming the first token to stdout.
    #[arg(long)]
    request_stats: Option<PathBuf>,

    /// Do not ask the tokenizer to add model-defined special tokens.
    #[arg(long)]
    no_special_tokens: bool,

    /// Append a FIFO request-trace row after single-turn generation completes.
    ///
    /// Format is compatible with `scripts/profile/replay_economics.py
    /// --request-trace`: `arrival_ms tokens id ...`.
    #[arg(long)]
    trace_request: Option<PathBuf>,
}

#[derive(Debug, Deserialize)]
struct JsonlRequest {
    id: Option<String>,
    prompt: Option<String>,
    prompt_file: Option<PathBuf>,
    tokens: Option<usize>,
    cache_prefix_tokens: Option<usize>,
}

#[derive(Debug, Serialize)]
struct RequestOutput {
    id: String,
    prompt_tokens: usize,
    generated_tokens: usize,
    generated_text: String,
}

#[derive(Debug, Serialize)]
struct RequestStatsRow {
    schema_version: u32,
    id: String,
    line: usize,
    model: String,
    arrival_ms: u64,
    finish_ms: u64,
    prompt_tokens: usize,
    prompt_hash: String,
    requested_tokens: usize,
    generated_tokens: usize,
    cache_prefix_tokens: Option<usize>,
    cache_prefix_hash: Option<String>,
    cache_hit: bool,
    matched_prefix_tokens: usize,
    matched_prefix_hash: Option<String>,
    exact_cache_hit: bool,
    prefill_chunk: usize,
    max_context_tokens: usize,
    no_special_tokens: bool,
    restore_ms: f64,
    prefix_inserted_bytes: u64,
    prefix_insert_ms: f64,
    prefill_ms: f64,
    decode_ms: f64,
    model_ttft_ms: f64,
    first_decode_ms: f64,
    decode_tps: f64,
    total_ms: f64,
    cache_entries: usize,
    cache_bytes: u64,
    cache_max_bytes: u64,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    if args.info {
        let runtime = Runtime::metal()?;
        println!("device: {}", runtime.describe());
        return Ok(());
    }

    let Some(model_path) = args.model.as_ref() else {
        eprintln!("usage: qwen -m <path-to-gguf> -p <prompt>  (or `qwen --info`)");
        std::process::exit(2);
    };

    if let Some(prompt) = prompt_text(&args)? {
        return run_single_turn(model_path, &prompt, &args);
    }

    if let Some(path) = args.requests_jsonl.as_ref() {
        return run_requests_jsonl(model_path, path, &args);
    }

    print_model_info(model_path)
}

fn prompt_text(args: &Args) -> Result<Option<String>> {
    if let Some(prompt) = args.prompt.as_ref() {
        return Ok(Some(prompt.clone()));
    }
    if let Some(path) = args.prompt_file.as_ref() {
        return Ok(Some(std::fs::read_to_string(path).with_context(|| {
            format!("read prompt file {}", path.display())
        })?));
    }
    Ok(None)
}

fn run_single_turn(model_path: &Path, prompt: &str, args: &Args) -> Result<()> {
    ensure!(args.prefill_chunk > 0, "--prefill-chunk must be >= 1");

    let arrival_ms = unix_epoch_ms()?;
    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_model_with_config(
            model_path,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    let tokenizer = loaded.tokenizer().context("load tokenizer")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    let prompt_ids = tokenizer
        .encode(prompt, !args.no_special_tokens)
        .context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("prompt tokenized to zero tokens");
    }

    let min_capacity = prompt_ids
        .len()
        .checked_add(args.tokens)
        .and_then(|v| v.checked_add(16))
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_ids.len() + args.tokens,
        "max context {} is smaller than prompt {} + generation {}",
        capacity,
        prompt_ids.len(),
        args.tokens
    );

    let chunk = args.prefill_chunk.min(prompt_ids.len().max(1));
    let block_size = u32::try_from(chunk).context("prefill chunk does not fit u32")?;
    let matrix_max_pos = prompt_ids.len().max(chunk);
    let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        block_size,
        matrix_max_pos,
    )
    .context("allocate prefill scratch")?;
    let mut sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let forward = loaded.forward();

    let prefill_t0 = Instant::now();
    let mut logits = prefill_tokens_with_multi_hidden(
        &forward,
        &prompt_ids,
        0,
        sequence.metal_session_mut(),
        &mut scratch,
        &[],
        None,
    )
    .context("prefill prompt")?;
    sequence.advance_by(prompt_ids.len())?;
    let prefill_ms = prefill_t0.elapsed().as_secs_f64() * 1e3;

    let eos = tokenizer.eos();
    let mut generated = Vec::with_capacity(args.tokens);
    let mut stdout = std::io::stdout().lock();
    let decode_t0 = Instant::now();
    let mut first_decode_ms = 0.0;
    for i in 0..args.tokens {
        let token = argmax_i32(&logits);
        let step_t0 = Instant::now();
        logits = forward
            .single_token(
                token,
                u32::try_from(prompt_ids.len() + i).context("position does not fit u32")?,
                sequence.metal_session_mut(),
            )
            .context("decode token")?;
        sequence.advance_by(1)?;
        if i == 0 {
            first_decode_ms = step_t0.elapsed().as_secs_f64() * 1e3;
        }

        generated.push(token);
        write!(stdout, "{}", tokenizer.decode_piece(token))?;
        stdout.flush()?;

        if Some(token) == eos {
            break;
        }
    }
    if !generated.is_empty() {
        writeln!(stdout)?;
    }
    let decode_ms = decode_t0.elapsed().as_secs_f64() * 1e3;
    let ttft_ms = prefill_ms + first_decode_ms;
    let decode_tps = if decode_ms > 0.0 {
        generated.len() as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };

    eprintln!(
        "stats: prompt_tokens={} generated_tokens={} load_ms={:.1} prefill_ms={:.1} ttft_ms={:.1} decode_tps={:.2} cache_entries={} cache_mib={:.1}/{:.1}",
        prompt_ids.len(),
        generated.len(),
        load_ms,
        prefill_ms,
        ttft_ms,
        decode_tps,
        loaded.prefix_cache_stats().entries,
        loaded.prefix_cache_stats().total_bytes as f64 / 1024.0 / 1024.0,
        loaded.prefix_cache_stats().max_bytes as f64 / 1024.0 / 1024.0,
    );

    if let Some(path) = args.trace_request.as_ref() {
        append_request_trace(path, arrival_ms, prompt_ids.len(), generated.len())?;
    }

    Ok(())
}

fn run_requests_jsonl(model_path: &Path, requests_path: &Path, args: &Args) -> Result<()> {
    ensure!(args.prefill_chunk > 0, "--prefill-chunk must be >= 1");

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_model_with_config(
            model_path,
            LoadedModelConfig {
                prefix_cache_max_bytes: prefix_cache_max_bytes(args)?,
            },
        )
        .with_context(|| format!("load model {}", model_path.display()))?;
    let tokenizer = loaded.tokenizer().context("load tokenizer")?;
    let load_ms = load_t0.elapsed().as_secs_f64() * 1e3;

    let stdin;
    let reader: Box<dyn BufRead> = if requests_path == Path::new("-") {
        stdin = std::io::stdin();
        Box::new(stdin.lock())
    } else {
        let requests = std::fs::File::open(requests_path)
            .with_context(|| format!("open requests JSONL {}", requests_path.display()))?;
        Box::new(std::io::BufReader::new(requests))
    };
    let mut stats_file = args
        .request_stats
        .as_ref()
        .map(|path| open_append_file(path, "request stats"))
        .transpose()?;
    let mut stdout = std::io::stdout().lock();
    let mut n_requests = 0usize;

    eprintln!(
        "loaded {} in {:.1} ms; prefix_cache_max_mib={}",
        model_path.display(),
        load_ms,
        args.prefix_cache_max_mib,
    );

    for (line_idx, line) in reader.lines().enumerate() {
        let line_no = line_idx + 1;
        let line = line.with_context(|| format!("read requests line {line_no}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let request: JsonlRequest = serde_json::from_str(trimmed)
            .with_context(|| format!("parse requests line {line_no}"))?;
        let id = request
            .id
            .clone()
            .unwrap_or_else(|| format!("line-{line_no}"));
        let (output, stats) = run_jsonl_request(&loaded, &tokenizer, &request, &id, line_no, args)
            .with_context(|| format!("run request {id}"))?;

        serde_json::to_writer(&mut stdout, &output).context("write request output")?;
        writeln!(stdout)?;
        stdout.flush()?;

        if let Some(file) = stats_file.as_mut() {
            serde_json::to_writer(&mut *file, &stats).context("write request stats")?;
            writeln!(file)?;
            file.flush()?;
        }
        if let Some(path) = args.trace_request.as_ref() {
            append_request_trace(
                path,
                unix_epoch_ms()?,
                stats.prompt_tokens,
                stats.generated_tokens,
            )?;
        }
        n_requests += 1;
    }

    ensure!(
        n_requests > 0,
        "requests JSONL {} contained no requests",
        requests_path.display()
    );

    let stats = loaded.prefix_cache_stats();
    eprintln!(
        "stats: requests={} cache_entries={} cache_mib={:.1}/{:.1}",
        n_requests,
        stats.entries,
        stats.total_bytes as f64 / 1024.0 / 1024.0,
        stats.max_bytes as f64 / 1024.0 / 1024.0,
    );
    Ok(())
}

fn run_jsonl_request(
    loaded: &LoadedModel,
    tokenizer: &Tokenizer,
    request: &JsonlRequest,
    id: &str,
    line: usize,
    args: &Args,
) -> Result<(RequestOutput, RequestStatsRow)> {
    let arrival_ms = unix_epoch_ms_u64()?;
    let total_t0 = Instant::now();
    let prompt = request_prompt(request, line)?;
    let prompt_ids = tokenizer
        .encode(&prompt, !args.no_special_tokens)
        .context("tokenize prompt")?;
    if prompt_ids.is_empty() {
        bail!("request {id} tokenized to zero tokens");
    }

    let n_generate = request.tokens.unwrap_or(args.tokens);
    let min_capacity = prompt_ids
        .len()
        .checked_add(n_generate)
        .and_then(|v| v.checked_add(16))
        .context("sequence capacity overflow")?;
    let capacity = args.max_context_tokens.unwrap_or(min_capacity);
    ensure!(
        capacity >= prompt_ids.len() + n_generate,
        "max context {} is smaller than prompt {} + generation {} for request {}",
        capacity,
        prompt_ids.len(),
        n_generate,
        id,
    );

    let chunk = args.prefill_chunk.min(prompt_ids.len().max(1));
    let block_size = u32::try_from(chunk).context("prefill chunk does not fit u32")?;
    let matrix_max_pos = prompt_ids.len().max(chunk);
    let mut scratch = MetalDFlashLayerMajorScratch::fresh_prefill_with_matrix_max_pos(
        loaded.context(),
        loaded.metal_model(),
        block_size,
        matrix_max_pos,
    )
    .context("allocate prefill scratch")?;
    let mut sequence = loaded.create_sequence(SequenceConfig::new(capacity))?;
    let forward = loaded.forward();

    let cache_prefix_tokens = request
        .cache_prefix_tokens
        .or(args.cache_prefix_tokens)
        .map(|n| n.min(prompt_ids.len()))
        .filter(|&n| n > 0);
    let prompt_hash = token_hash_hex(&prompt_ids);
    let cache_prefix_hash = cache_prefix_tokens.map(|n| token_hash_hex(&prompt_ids[..n]));

    let mut cache_hit = false;
    let mut matched_prefix_tokens = 0usize;
    let mut exact_cache_hit = false;
    let restore_ms;
    let mut prefix_inserted_bytes = 0u64;
    let mut prefix_insert_ms = 0.0;
    let mut prefill_ms = 0.0;

    let logits = {
        let restore_t0 = Instant::now();
        let hit = loaded
            .restore_cached_prefix(&mut sequence, &prompt_ids)
            .context("restore prefix cache")?;
        restore_ms = restore_t0.elapsed().as_secs_f64() * 1e3;
        if let Some(hit) = hit {
            cache_hit = true;
            matched_prefix_tokens = hit.matched_prefix_len;
            exact_cache_hit = hit.exact;
            if matched_prefix_tokens == prompt_ids.len() {
                hit.exact_final_logits.with_context(|| {
                    format!("exact prefix-cache hit for request {id} did not store logits")
                })?
            } else {
                let suffix = &prompt_ids[matched_prefix_tokens..];
                let (logits, ms) = prefill_span(
                    &forward,
                    &mut sequence,
                    &mut scratch,
                    suffix,
                    matched_prefix_tokens,
                )?;
                prefill_ms += ms;
                logits
            }
        } else if let Some(prefix_len) = cache_prefix_tokens {
            let prefix = &prompt_ids[..prefix_len];
            let (prefix_logits, ms) =
                prefill_span(&forward, &mut sequence, &mut scratch, prefix, 0)?;
            prefill_ms += ms;

            let insert_t0 = Instant::now();
            let insert = loaded
                .cache_sequence_prefix(&sequence, prefix.to_vec(), Some(prefix_logits.clone()))
                .context("insert prefix cache snapshot")?;
            prefix_insert_ms = insert_t0.elapsed().as_secs_f64() * 1e3;
            prefix_inserted_bytes = insert.snapshot_bytes;

            if prefix_len == prompt_ids.len() {
                prefix_logits
            } else {
                let suffix = &prompt_ids[prefix_len..];
                let (logits, ms) =
                    prefill_span(&forward, &mut sequence, &mut scratch, suffix, prefix_len)?;
                prefill_ms += ms;
                logits
            }
        } else {
            let (logits, ms) = prefill_span(&forward, &mut sequence, &mut scratch, &prompt_ids, 0)?;
            prefill_ms += ms;
            logits
        }
    };

    let (generated, generated_text, decode_ms, first_decode_ms) = decode_greedy(
        &forward,
        tokenizer,
        &mut sequence,
        logits,
        prompt_ids.len(),
        n_generate,
    )?;
    let decode_tps = if decode_ms > 0.0 {
        generated.len() as f64 / (decode_ms / 1e3)
    } else {
        0.0
    };
    let stats_now = loaded.prefix_cache_stats();
    let finish_ms = unix_epoch_ms_u64()?;
    let stats = RequestStatsRow {
        schema_version: 1,
        id: id.to_string(),
        line,
        model: loaded.path().display().to_string(),
        arrival_ms,
        finish_ms,
        prompt_tokens: prompt_ids.len(),
        prompt_hash,
        requested_tokens: n_generate,
        generated_tokens: generated.len(),
        cache_prefix_tokens,
        cache_prefix_hash,
        cache_hit,
        matched_prefix_tokens,
        matched_prefix_hash: if matched_prefix_tokens > 0 {
            Some(token_hash_hex(&prompt_ids[..matched_prefix_tokens]))
        } else {
            None
        },
        exact_cache_hit,
        prefill_chunk: args.prefill_chunk,
        max_context_tokens: capacity,
        no_special_tokens: args.no_special_tokens,
        restore_ms,
        prefix_inserted_bytes,
        prefix_insert_ms,
        prefill_ms,
        decode_ms,
        model_ttft_ms: restore_ms + prefix_insert_ms + prefill_ms + first_decode_ms,
        first_decode_ms,
        decode_tps,
        total_ms: total_t0.elapsed().as_secs_f64() * 1e3,
        cache_entries: stats_now.entries,
        cache_bytes: stats_now.total_bytes,
        cache_max_bytes: stats_now.max_bytes,
    };
    let output = RequestOutput {
        id: id.to_string(),
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated.len(),
        generated_text,
    };
    Ok((output, stats))
}

fn request_prompt(request: &JsonlRequest, line: usize) -> Result<String> {
    match (request.prompt.as_ref(), request.prompt_file.as_ref()) {
        (Some(_), Some(_)) => bail!("request line {line} has both prompt and prompt_file"),
        (Some(prompt), None) => Ok(prompt.clone()),
        (None, Some(path)) => std::fs::read_to_string(path)
            .with_context(|| format!("read prompt_file {} on line {line}", path.display())),
        (None, None) => bail!("request line {line} has neither prompt nor prompt_file"),
    }
}

fn prefill_span(
    forward: &MetalForward<'_>,
    sequence: &mut Sequence,
    scratch: &mut MetalDFlashLayerMajorScratch,
    token_ids: &[i32],
    start_position: usize,
) -> Result<(Vec<f32>, f64)> {
    ensure!(!token_ids.is_empty(), "cannot prefill an empty token span");
    sequence.check_position(start_position)?;
    let t0 = Instant::now();
    let logits = prefill_tokens_with_multi_hidden(
        forward,
        token_ids,
        u32::try_from(start_position).context("position does not fit u32")?,
        sequence.metal_session_mut(),
        scratch,
        &[],
        None,
    )
    .context("prefill prompt span")?;
    sequence.advance_by(token_ids.len())?;
    Ok((logits, t0.elapsed().as_secs_f64() * 1e3))
}

fn decode_greedy(
    forward: &MetalForward<'_>,
    tokenizer: &Tokenizer,
    sequence: &mut Sequence,
    mut logits: Vec<f32>,
    start_position: usize,
    max_tokens: usize,
) -> Result<(Vec<i32>, String, f64, f64)> {
    sequence.check_position(start_position)?;
    let eos = tokenizer.eos();
    let mut generated = Vec::with_capacity(max_tokens);
    let mut generated_text = String::new();
    let decode_t0 = Instant::now();
    let mut first_decode_ms = 0.0;
    for i in 0..max_tokens {
        let token = argmax_i32(&logits);
        let step_t0 = Instant::now();
        logits = forward
            .single_token(
                token,
                u32::try_from(start_position + i).context("position does not fit u32")?,
                sequence.metal_session_mut(),
            )
            .context("decode token")?;
        sequence.advance_by(1)?;
        if i == 0 {
            first_decode_ms = step_t0.elapsed().as_secs_f64() * 1e3;
        }
        generated.push(token);
        generated_text.push_str(&tokenizer.decode_piece(token));
        if Some(token) == eos {
            break;
        }
    }
    Ok((
        generated,
        generated_text,
        decode_t0.elapsed().as_secs_f64() * 1e3,
        first_decode_ms,
    ))
}

fn prefix_cache_max_bytes(args: &Args) -> Result<u64> {
    args.prefix_cache_max_mib
        .checked_mul(1024 * 1024)
        .context("prefix cache byte budget overflow")
}

fn unix_epoch_ms_u64() -> Result<u64> {
    let ms = unix_epoch_ms()?;
    u64::try_from(ms).context("Unix epoch milliseconds do not fit u64")
}

const TOKEN_HASH_SEED: u64 = 0xcbf29ce484222325;
const TOKEN_HASH_PRIME: u64 = 0x100000001b3;

fn token_hash_hex(tokens: &[i32]) -> String {
    let mut hash = TOKEN_HASH_SEED;
    for &token in tokens {
        hash ^= (token as u32 as u64).wrapping_add(0x9e3779b97f4a7c15);
        hash = hash.wrapping_mul(TOKEN_HASH_PRIME);
    }
    format!("{hash:016x}")
}

fn open_append_file(path: &Path, label: &str) -> Result<std::fs::File> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {label} directory {}", parent.display()))?;
    }
    OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open {label} {}", path.display()))
}

fn unix_epoch_ms() -> Result<u128> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before Unix epoch")?
        .as_millis())
}

fn append_request_trace(
    path: &Path,
    arrival_ms: u128,
    prompt_tokens: usize,
    generated_tokens: usize,
) -> Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create trace directory {}", parent.display()))?;
    }
    let write_header = std::fs::metadata(path).map_or(true, |m| m.len() == 0);
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("open request trace {}", path.display()))?;
    if write_header {
        writeln!(file, "arrival_ms\ttokens\tid\tprompt_tokens")?;
    }
    let id = format!("{}-{arrival_ms}", std::process::id());
    writeln!(
        file,
        "{arrival_ms}\t{generated_tokens}\t{id}\t{prompt_tokens}"
    )?;
    Ok(())
}

fn argmax_i32(xs: &[f32]) -> i32 {
    xs.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i as i32)
        .unwrap_or(0)
}

fn print_model_info(model_path: &Path) -> Result<()> {
    let gguf = qwen_llm::gguf::GgufFile::open(&model_path)?;
    println!(
        "loaded {}: arch={} {} tensors, {} shard(s), mmap={} MiB, primary tensor-data starts at {}",
        model_path.display(),
        gguf.architecture().unwrap_or_else(|| "?".into()),
        gguf.tensors.len(),
        gguf.shard_count(),
        gguf.total_mapped_len() / (1024 * 1024),
        gguf.primary_shard().tensor_data_start,
    );

    // Group tensors by layer index. The GDN-layer test is "has ssm_* tensor",
    // the full-attn-layer test is "has attn_q/k/v/o.weight" (NOT attn_qkv,
    // which is GDN's combined input projection in this naming scheme).
    use std::collections::BTreeMap;
    let mut by_layer: BTreeMap<u32, Vec<&str>> = BTreeMap::new();
    for t in &gguf.tensors {
        if let Some(rest) = t.name.strip_prefix("blk.") {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<u32>() {
                    by_layer.entry(idx).or_default().push(&rest[dot + 1..]);
                }
            }
        }
    }
    let n_layers = by_layer.len();
    let mut gdn = 0usize;
    let mut attn = 0usize;
    for tensors in by_layer.values() {
        let has_ssm = tensors.iter().any(|s| s.starts_with("ssm_"));
        let has_attn_q = tensors
            .iter()
            .any(|s| *s == "attn_q.weight" || *s == "attn_qkv_real.weight");
        if has_ssm {
            gdn += 1;
        } else if has_attn_q {
            attn += 1;
        }
    }
    println!("blocks: {n_layers} total — {gdn} GDN, {attn} full-attn");

    // Show layer-0 and layer-3 tensor inventories: 0 should be GDN, 3 full-attn.
    for sample in [0u32, 3] {
        if let Some(t) = by_layer.get(&sample) {
            println!("blk.{sample} tensors ({}):", t.len());
            for name in t {
                println!("  blk.{sample}.{name}");
            }
        }
    }

    // Show metadata keys related to the architecture.
    let interesting_keys = [
        "qwen35.block_count",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.value_length",
        "qwen35.embedding_length",
        "qwen35.feed_forward_length",
        "qwen35.context_length",
        "qwen35.ssm.conv_kernel",
        "qwen35.ssm.inner_size",
        "qwen35.ssm.state_size",
        "qwen35.ssm.time_step_rank",
        "qwen35.ssm.group_count",
    ];
    println!("relevant metadata:");
    for k in interesting_keys {
        if let Some(v) = gguf.get_u64(k) {
            println!("  {k} = {v}");
        }
    }

    Ok(())
}
