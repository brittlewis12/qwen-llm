//! `qwen-tok` — vocab-only tokenizer utility. Cheap (no model weights
//! loaded, no Metal context, no kernel JIT). Reads a supported GGUF's
//! tokenizer and reports token count + optionally the token ids.
//!
//! Usage:
//!   qwen-tok -m <gguf> --text "your prompt here"
//!   qwen-tok -m <gguf> --file path/to/prompt.txt
//!   qwen-tok -m <gguf> --file - < prompt.txt          # stdin
//!   qwen-tok -m <gguf> --file foo.txt --ids           # also print token ids
//!   qwen-tok -m <gguf> --decode-ids path/to/ids.txt
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
    about = "tokenize text or decode token IDs with a supported GGUF's vocab"
)]
struct Args {
    /// Path to a supported GGUF (any quant; only tokenizer metadata is used).
    #[arg(short = 'm', long)]
    model: PathBuf,
    /// Inline prompt text (mutually exclusive with --file).
    #[arg(long, conflicts_with_all = ["file", "decode_ids"])]
    text: Option<String>,
    /// Read prompt from a file, or `-` for stdin (mutually exclusive with --text).
    #[arg(long, conflicts_with = "decode_ids")]
    file: Option<String>,
    /// Decode whitespace-separated token IDs from a file, or `-` for stdin.
    #[arg(long, conflicts_with_all = ["text", "file", "ids", "add_special"])]
    decode_ids: Option<String>,
    /// Print token ids in addition to the count (one per line).
    #[arg(long)]
    ids: bool,
    /// Add the BOS / system special token (matches add_special=true at
    /// inference). Default: false (matches the bench harness which
    /// passes add_special=false to `Tokenizer::encode`).
    #[arg(long)]
    add_special: bool,
}

fn read_path_or_stdin(path: &str) -> Result<String> {
    if path == "-" {
        let mut text = String::new();
        std::io::stdin()
            .read_to_string(&mut text)
            .context("read stdin")?;
        Ok(text)
    } else {
        std::fs::read_to_string(path).with_context(|| format!("read {path:?}"))
    }
}

fn parse_token_ids(input: &str) -> Result<Vec<i32>> {
    input
        .split_whitespace()
        .map(|value| {
            value
                .parse()
                .with_context(|| format!("parse token ID {value:?}"))
        })
        .collect()
}

fn main() -> Result<()> {
    let args = Args::parse();
    let tok = qwen_llm::tokenizer::Tokenizer::open(&args.model)
        .with_context(|| format!("open tokenizer from {:?}", args.model))?;

    if let Some(path) = args.decode_ids.as_deref() {
        let ids = parse_token_ids(&read_path_or_stdin(path)?)?;
        print!("{}", tok.decode(&ids));
        return Ok(());
    }

    let text = match (&args.text, &args.file) {
        (Some(t), None) => t.clone(),
        (None, Some(path)) => read_path_or_stdin(path)?,
        (None, None) => bail!("specify --text, --file, or --decode-ids"),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with"),
    };

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_id_stream_accepts_whitespace_and_rejects_non_integers() {
        assert_eq!(parse_token_ids("1\n-2  3\t").unwrap(), [1, -2, 3]);
        assert!(parse_token_ids("1 nope").is_err());
    }
}
