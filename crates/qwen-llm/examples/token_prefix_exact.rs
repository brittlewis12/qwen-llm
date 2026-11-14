//! CPU-only exactness witness with a released tokenizer and serve-plan shapes.
use qwen_llm::tokenizer::{NativeTokenizer, PartitionedEncoding};
use std::path::PathBuf;
fn head(s: &str, n: usize) -> &str {
    let mut n = n.min(s.len());
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    &s[..n]
}
fn main() -> anyhow::Result<()> {
    let a: Vec<_> = std::env::args_os().collect();
    anyhow::ensure!(a.len() == 3, "usage: token_prefix_exact MODEL CORPUS");
    let native = NativeTokenizer::open(PathBuf::from(&a[1]))?;
    let source = std::fs::read_to_string(PathBuf::from(&a[2]))?;
    let csv = include_str!("../../../docs/bench/2026-11-08-real-opencode-agent-calls.csv");
    let mut previous: Vec<Option<PartitionedEncoding>> = vec![None, None, None];
    let mut texts: Vec<_> = (0..3)
        .map(|i| {
            format!(
                "<｜begin▁of▁sentence｜><｜User｜>agent{i}:{}<｜Assistant｜></think>",
                head(&source, 196608)
            )
        })
        .collect();
    let mut cases = 0;
    let mut reused = 0;
    let mut boundaries = 0;
    for (i, row) in csv
        .lines()
        .skip(1)
        .filter(|r| r.starts_with("v41/") && r.split(',').nth(4) == Some("primary"))
        .take(64)
        .enumerate()
    {
        let n: usize = row.split(',').nth(7).unwrap().parse()?;
        let c = i % 3;
        if texts[c].len() > 500000 {
            let end = head(&texts[c], 196608).len();
            texts[c].truncate(end);
            previous[c] = None;
        }
        let adversarial = [
            "a",
            "ab",
            "e\u{301}👩‍💻",
            "<｜Assistant｜>",
            "<｜Assis",
            "tant｜>",
            "\r\n  ",
            "中\0x",
        ][i % 8];
        texts[c].push_str(&format!(
            "reply-{i}<｜end▁of▁sentence｜><｜User｜>{}{adversarial}<｜Assistant｜></think>",
            head(&source, n.min(8192))
        ));
        let current = native
            .encode_partitioned(&texts[c], previous[c].as_ref())?
            .ok_or_else(|| {
                anyhow::anyhow!("released tokenizer does not support exact partitioning")
            })?;
        anyhow::ensure!(
            current.ids() == native.encode(&texts[c], false)?,
            "full ids differ in case {i}"
        );
        reused += usize::from(current.reused_tokens() > 0);
        for (cut, _) in texts[c].match_indices("<｜Assistant｜>") {
            if let Some(ids) = current.exact_prefix(&texts[c][..cut]) {
                anyhow::ensure!(
                    ids == native.encode(&texts[c][..cut], false)?,
                    "prefix ids differ"
                );
                boundaries += 1;
            }
        }
        previous[c] = Some(current);
        cases += 1;
    }
    anyhow::ensure!(
        cases > 20 && reused > 10 && boundaries > 20,
        "inadequate witness coverage"
    );
    println!(
        "TOKEN_PREFIX_EXACT cases={cases} incremental_hits={reused} boundaries={boundaries} full_ids=true prefix_ids=true"
    );
    Ok(())
}
