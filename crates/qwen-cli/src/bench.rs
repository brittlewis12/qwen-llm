//! `qwen-bench` — apples-to-apples vs `~/code/llama.cpp/build/bin/llama-bench`.
//!
//! Same prompt corpus, same model file, sweep context, log pp/tg/peak-RSS.
//! v1 stub: just opens the file and reports load time. Decode/prefill
//! benchmarks land alongside the kernels they exercise.

use anyhow::Result;
use clap::Parser;

#[derive(Parser, Debug)]
#[command(
    name = "qwen-bench",
    version,
    about = "throughput benchmark for qwen-llm"
)]
struct Args {
    /// Path to a GGUF file.
    #[arg(short = 'm', long)]
    model: std::path::PathBuf,

    /// Prompt-processing batch sizes to benchmark.
    #[arg(long, value_delimiter = ',', default_value = "512")]
    pp: Vec<usize>,

    /// Token-generation lengths to benchmark.
    #[arg(long, value_delimiter = ',', default_value = "128")]
    tg: Vec<usize>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();

    let load_start = std::time::Instant::now();
    let gguf = qwen_llm::gguf::GgufFile::open(&args.model)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1e3;

    println!(
        "model:    {} ({} MiB mmap'd)",
        args.model.display(),
        gguf.mmap.len() / (1024 * 1024)
    );
    println!("load:     {load_ms:.2} ms (mmap-only, no copies)");
    println!("pp set:   {:?}", args.pp);
    println!("tg set:   {:?}", args.tg);
    println!("(decode + prefill benchmarks land alongside the kernels they exercise.)");

    Ok(())
}
