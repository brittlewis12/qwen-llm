//! `qwen` — interactive CLI for the qwen-llm engine.

use anyhow::{Context, Result, bail, ensure};
use clap::Parser;
use qwen_llm::metal_dflash::{MetalDFlashLayerMajorScratch, prefill_tokens_with_multi_hidden};
use qwen_llm::runtime::{Runtime, SequenceConfig};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

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

    /// Number of greedy tokens to generate.
    #[arg(short = 'n', long, default_value_t = 64)]
    tokens: usize,

    /// Prompt prefill chunk size. The safe default mirrors qwen-bench.
    #[arg(long, default_value_t = 1024)]
    prefill_chunk: usize,

    /// Override sequence capacity. Defaults to prompt + generated tokens + slack.
    #[arg(long)]
    max_context_tokens: Option<usize>,

    /// Do not ask the tokenizer to add model-defined special tokens.
    #[arg(long)]
    no_special_tokens: bool,
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

    let load_t0 = Instant::now();
    let runtime = Runtime::metal().context("init Metal runtime")?;
    let loaded = runtime
        .load_model(model_path)
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
