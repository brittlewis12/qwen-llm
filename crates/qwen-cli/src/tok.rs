//! `qwen-tok` — vocab-only tokenizer utility. Cheap (no model weights
//! loaded, no Metal context, no kernel JIT). Reads a supported GGUF's
//! tokenizer and reports token count + optionally the token ids.
//!
//! Usage:
//!   qwen-tok -m <gguf> --text "your prompt here"
//!   qwen-tok -m <gguf> --file path/to/prompt.txt
//!   qwen-tok -m <gguf> --file - < prompt.txt          # stdin
//!   qwen-tok -m <gguf> --file foo.txt --ids           # also print token ids
//!
//! Exit status: 0 on success.

use anyhow::{Context, Result, bail};
use clap::Parser;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "qwen-tok",
    version,
    about = "tokenize text with a supported GGUF's vocab"
)]
struct Args {
    /// Path to a supported GGUF (any quant; only tokenizer metadata is used).
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Inline prompt text (mutually exclusive with --file).
    #[arg(long, conflicts_with = "file")]
    text: Option<String>,
    /// Read prompt from a file, or `-` for stdin (mutually exclusive with --text).
    #[arg(long)]
    file: Option<String>,
    /// Print token ids in addition to the count (one per line).
    #[arg(long)]
    ids: bool,
    /// Add the BOS / system special token (matches add_special=true at
    /// inference). Default: false (matches the bench harness which
    /// passes add_special=false to `Tokenizer::encode`).
    #[arg(long)]
    add_special: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let text = match (&args.text, &args.file) {
        (Some(t), None) => t.clone(),
        (None, Some(p)) if p == "-" => {
            let mut s = String::new();
            std::io::stdin()
                .read_to_string(&mut s)
                .context("read stdin")?;
            s
        }
        (None, Some(p)) => std::fs::read_to_string(p).with_context(|| format!("read {p:?}"))?,
        (None, None) => bail!("specify --text <STRING> or --file <PATH|->"),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
    };

    let tok = qwen_llm::tokenizer::Tokenizer::open(&args.model)
        .with_context(|| format!("open tokenizer from {:?}", args.model))?;
    let ids = tok
        .encode(&text, args.add_special)
        .context("tokenize text")?;

    println!("{} tokens", ids.len());
    if args.ids {
        for id in &ids {
            println!("{id}");
        }
    }
    Ok(())
}
