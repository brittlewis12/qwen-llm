//! Tokenizer throughput subcommand.

use super::*;

pub(crate) fn run_tok(args: TokArgs) -> Result<()> {
    let TokArgs {
        model,
        prompt,
        file,
        messages,
        messages_max,
        messages_preserve_thinking,
        messages_strip_thinking,
        messages_no_generation_prompt,
        iters,
        add_special,
        print_token_ids,
    } = args;
    if iters == 0 {
        anyhow::bail!("--iters must be > 0");
    }
    let (source, text) = match (prompt, file, messages) {
        (Some(prompt), None, None) => ("inline".to_string(), prompt),
        (None, Some(path), None) => {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("read {}", path.display()))?;
            (format!("file:{}", path.display()), text)
        }
        (None, None, Some(path)) => (
            format!("messages:{}", path.display()),
            load_messages_prompt(
                &path,
                messages_max,
                messages_thinking_mode(messages_preserve_thinking, messages_strip_thinking),
                !messages_no_generation_prompt,
            )?,
        ),
        (None, None, None) => ("default".to_string(), default_tok_prompt()),
        _ => unreachable!("clap conflicts_with"),
    };

    let t0 = Instant::now();
    let ffi = LlamaCppTokenizer::open(&model).context("open llama.cpp FFI tokenizer")?;
    let ffi_load = t0.elapsed();

    let t0 = Instant::now();
    let gguf = GgufFile::open(&model).with_context(|| format!("open {}", model.display()))?;
    let native = NativeTokenizer::from_gguf(&gguf).context("open native GGUF tokenizer")?;
    let native_load = t0.elapsed();

    let ffi_ids = ffi.encode(&text, add_special).context("ffi encode")?;
    let native_ids = native.encode(&text, add_special).context("native encode")?;
    if ffi_ids != native_ids {
        let first = ffi_ids
            .iter()
            .zip(&native_ids)
            .position(|(a, b)| a != b)
            .unwrap_or_else(|| ffi_ids.len().min(native_ids.len()));
        anyhow::bail!(
            "native/ffi encode mismatch at token {first}: ffi_len={} native_len={} ffi={:?} native={:?}",
            ffi_ids.len(),
            native_ids.len(),
            ffi_ids.get(first),
            native_ids.get(first)
        );
    }
    let ffi_text = ffi.try_decode(&ffi_ids).context("ffi decode")?;
    let native_text = native.try_decode(&native_ids).context("native decode")?;
    if ffi_text != native_text {
        anyhow::bail!(
            "native/ffi decode mismatch: ffi_len={} native_len={}",
            ffi_text.len(),
            native_text.len()
        );
    }
    if print_token_ids {
        println!("[tok] token_ids={}", serde_json::to_string(&native_ids)?);
        println!(
            "[tok] token_ids_sha256_i32le={}",
            token_ids_sha256_i32le(&native_ids)
        );
    }

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ffi.encode(&text, add_special).context("ffi encode timed")?);
    }
    let ffi_encode = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            native
                .encode(&text, add_special)
                .context("native encode timed")?,
        );
    }
    let native_encode = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(ffi.try_decode(&ffi_ids).context("ffi decode timed")?);
    }
    let ffi_decode = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..iters {
        std::hint::black_box(
            native
                .try_decode(&native_ids)
                .context("native decode timed")?,
        );
    }
    let native_decode = t0.elapsed();

    let n_tokens = ffi_ids.len() * iters;
    println!("[tok] model={}", model.display());
    println!(
        "[tok] source={} chars={} tokens={} iters={} add_special={}",
        source,
        text.len(),
        ffi_ids.len(),
        iters,
        add_special
    );
    println!(
        "[tok] load_ms: ffi={:.3} native={:.3}",
        ffi_load.as_secs_f64() * 1000.0,
        native_load.as_secs_f64() * 1000.0
    );
    print_tok_rate("ffi encode", ffi_encode, n_tokens);
    print_tok_rate("native encode", native_encode, n_tokens);
    print_tok_rate("ffi decode", ffi_decode, n_tokens);
    print_tok_rate("native decode", native_decode, n_tokens);
    Ok(())
}

pub(crate) fn default_tok_prompt() -> String {
    "<|im_start|>user\nHello, world!\n\n```rust\nfn main() { println!(\"hi\"); }\n```\n数字123 combining e\u{301} emoji🙂\u{fe0f}\n<|im_end|>"
        .to_string()
}

pub(crate) fn print_tok_rate(label: &str, elapsed: Duration, n_tokens: usize) {
    let secs = elapsed.as_secs_f64();
    let tok_s = n_tokens as f64 / secs.max(f64::MIN_POSITIVE);
    println!(
        "[tok] {label}: {:.3} ms total, {:.0} tok/s",
        secs * 1000.0,
        tok_s
    );
}

#[cfg(test)]
mod tok_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn auto_preserves_thinking_only_for_qwen36() {
        assert!(messages_auto_preserve_thinking(
            &json!({ "model": "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf" })
        ));
        assert!(messages_auto_preserve_thinking(
            &json!({ "preserve_thinking": true, "model": "anything" })
        ));
        assert!(!messages_auto_preserve_thinking(
            &json!({ "model": "/Users/tito/models/Qwen3.5-27B-Q4_K_M.gguf" })
        ));
        assert!(!messages_auto_preserve_thinking(&serde_json::Value::Null));
    }

    #[test]
    fn strip_think_only_strips_leading_qwen_block() {
        assert_eq!(strip_think("<think>hidden</think>shown"), "shown");
        assert_eq!(strip_think("plain text"), "plain text");
        assert_eq!(strip_think("  plain text  "), "  plain text  ");
        assert_eq!(
            strip_think("prefix </think> shown"),
            "prefix </think> shown"
        );
    }

    #[test]
    fn render_messages_prompt_respects_thinking_mode() {
        let messages = vec![
            ChatMessage {
                role: "user".into(),
                content: "hi".into(),
                ..Default::default()
            },
            ChatMessage {
                role: "assistant".into(),
                content: "<think>hidden</think>shown".into(),
                ..Default::default()
            },
        ];
        let stripped = render_qwen_messages_prompt(&messages, false, true);
        let preserved = render_qwen_messages_prompt(&messages, true, true);
        assert!(stripped.contains("shown<|im_end|>"));
        assert!(!stripped.contains("hidden"));
        assert!(preserved.contains("<think>hidden</think>shown"));
        assert!(preserved.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn parse_messages_input_accepts_top_level_metadata() {
        let value = json!({
            "model": "/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf",
            "preserve_thinking": true,
            "messages": [
                {"role": "user", "content": "hi"}
            ]
        });
        let (messages, meta) = parse_messages_input(value).expect("parse messages");
        assert_eq!(messages.len(), 1);
        assert_eq!(
            meta.get("model").and_then(|v| v.as_str()),
            Some("/Users/tito/models/Qwen3.6-27B-Q4_K_M.gguf")
        );
        assert_eq!(
            meta.get("preserve_thinking").and_then(|v| v.as_bool()),
            Some(true)
        );
    }

    #[test]
    fn mtp_token_ids_require_output() {
        let missing_output = Args::try_parse_from([
            "qwen-bench",
            "mtp",
            "-m",
            "model.gguf",
            "--include-token-ids",
        ]);
        assert!(missing_output.is_err());

        let parsed = Args::try_parse_from([
            "qwen-bench",
            "mtp",
            "-m",
            "model.gguf",
            "--output",
            "fixture.json",
            "--include-token-ids",
        ])
        .expect("parse token fixture arguments");
        let Cmd::Mtp(args) = parsed.cmd else {
            panic!("expected mtp command");
        };
        assert!(args.include_token_ids);
        assert_eq!(args.output, Some(PathBuf::from("fixture.json")));
    }

    #[test]
    fn tok_print_token_ids_is_explicit() {
        let parsed = Args::try_parse_from([
            "qwen-bench",
            "tok",
            "--model",
            "model.gguf",
            "--prompt",
            "def",
            "--print-token-ids",
        ])
        .expect("parse tokenizer fixture arguments");
        let Cmd::Tok(args) = parsed.cmd else {
            panic!("expected tok command");
        };
        assert!(args.print_token_ids);
    }

    #[test]
    fn pld_terminal_window_counts_target_transitions() {
        let drafts = [11, 12, 13, 14, 15, 16, 17];
        let count = |emitted, stop| {
            terminal_draft_window(&drafts, emitted, 128, stop).map(|window| window.count)
        };
        assert_eq!(count(1, &[99]), None);
        assert_eq!(count(121, &[99]), Some(7));
        assert_eq!(count(125, &[99]), Some(3));
        assert_eq!(count(1, &[14]), Some(4));

        for remaining in 1..=DRAFT_TOKENS {
            let window = terminal_draft_window(&drafts, 128 - remaining, 128, &[99])
                .expect("output-limit window");
            assert_eq!(window.count, remaining);
            assert_eq!(window.cause, PromptLookupTerminalCause::OutputLimit);
        }
        for stop_index in 0..DRAFT_TOKENS {
            let window = terminal_draft_window(&drafts, 1, 128, &[drafts[stop_index]])
                .expect("stop-token window");
            assert_eq!(window.count, stop_index + 1);
            assert_eq!(window.cause, PromptLookupTerminalCause::StopToken);
        }
        let output_first = terminal_draft_window(&drafts, 125, 128, &[16])
            .expect("output limit precedes proposed stop");
        assert_eq!(output_first.count, 3);
        assert_eq!(output_first.cause, PromptLookupTerminalCause::OutputLimit);
    }
}
