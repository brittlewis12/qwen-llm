use super::*;
use crate::lens_input::LensMessageMode;
use crate::prompt_template::QwenPromptTemplate;
use serde_json::{Value, json};

fn parse(value: &Value) -> Result<RunDocument> {
    parse_run_bytes(&serde_json::to_vec(value)?, Path::new("prefill-run.json"))
}

fn fixture(channel: &str, text: &str) -> Value {
    mode_fixture(
        QwenPromptTemplate::Qwen36,
        Some(LensMessageMode::Thinking),
        Some((channel, text)),
    )
}

fn mode_fixture(
    protocol: QwenPromptTemplate,
    mode: Option<LensMessageMode>,
    prefill: Option<(&str, &str)>,
) -> Value {
    let prefill = prefill.map(|(channel, text)| {
        serde_json::from_value(json!({"channel":channel,"text":text})).unwrap()
    });
    serde_json::from_slice(&lens_run::generation_run_bytes(
        protocol,
        mode,
        prefill.as_ref(),
    ))
    .unwrap()
}

#[test]
fn typed_prefill_writer_roundtrips_through_strict_offline_reader() {
    for (protocol, modes) in [
        (
            QwenPromptTemplate::Qwen36,
            vec![
                None,
                Some(LensMessageMode::Auto),
                Some(LensMessageMode::Thinking),
                Some(LensMessageMode::NoThinking),
            ],
        ),
        (
            QwenPromptTemplate::Qwen38,
            vec![
                None,
                Some(LensMessageMode::Low),
                Some(LensMessageMode::Medium),
                Some(LensMessageMode::Xhigh),
                Some(LensMessageMode::NoThinking),
            ],
        ),
    ] {
        for mode in modes {
            let legacy = mode_fixture(protocol, mode, None);
            assert!(legacy.get("generation_input").is_none());
            parse(&legacy).unwrap();
            for channel in ["reasoning", "final"] {
                if channel == "reasoning" && mode == Some(LensMessageMode::NoThinking) {
                    continue;
                }
                for text in ["", " ", "  Answer:\n", "\u{1f642}"] {
                    let value = mode_fixture(protocol, mode, Some((channel, text)));
                    let document = parse(&value).unwrap();
                    assert_eq!(
                        value["generation_input"]["assistant_prefill"],
                        json!({"channel":channel,"text":text})
                    );
                    assert_eq!(document.decoded_text, "continuation only");
                    assert_eq!(document.generated_token_ids, vec![7]);
                    compare_runs(&document, &document, 1).unwrap();
                    ensure_sweep_run_context(&document, &document).unwrap();
                }
            }
        }
    }
}

#[test]
fn strict_reader_rejects_prefill_context_mutations_and_deletion() {
    let value = fixture("final", "  Answer:\n");
    for (pointer, replacement) in [
        ("/schema_version", json!(4)),
        ("/runtime_kind", json!("flash_next")),
        ("/input_source", json!("prompt")),
        ("/add_special_tokens", json!(true)),
        ("/generation_input/kind", json!("raw_text")),
        ("/generation_input/prompt_text", json!("changed")),
        ("/generation_input/prompt_bytes/0", json!(0)),
        ("/generation_input/prompt_digest", json!("0".repeat(64))),
        ("/generation_input/token_ids/0", json!(999)),
        ("/generation_input/template", json!("qwen3.5_messages_v1")),
        ("/generation_input/output_initial_state", json!("pre_open")),
        (
            "/generation_input/assistant_prefill/channel",
            json!("reasoning"),
        ),
        ("/generation_input/assistant_prefill/text", json!("changed")),
        (
            "/generation_input/rendering/generation_mode",
            json!("no_thinking"),
        ),
    ] {
        let mut changed = value.clone();
        *changed.pointer_mut(pointer).unwrap() = replacement;
        assert!(parse(&changed).is_err(), "{pointer}");
    }
    let mut changed = value.clone();
    let text = changed["generation_input"]["prompt_text"]
        .as_str()
        .unwrap()
        .replace("</think>", "<xthink>");
    changed["generation_input"]["prompt_bytes"] = json!(text.as_bytes());
    changed["generation_input"]["prompt_text"] = json!(text);
    assert!(parse(&changed).is_err());
    for channel in ["reasoning", "final"] {
        let mut changed = fixture(channel, "prefix");
        changed.as_object_mut().unwrap().remove("generation_input");
        assert!(parse(&changed).is_err());
    }
    let mut changed = fixture("final", "");
    changed["generation_input"] = Value::Null;
    assert!(parse(&changed).is_err());
    let mut changed = value.clone();
    changed["generation_input"]["extra"] = json!(true);
    assert!(parse(&changed).is_err());
    let mut changed = value;
    changed["generation_input"]
        .as_object_mut()
        .unwrap()
        .remove("assistant_prefill");
    assert!(parse(&changed).is_err());
}

#[test]
fn comparison_distinguishes_retained_intent_even_when_prompt_bytes_are_identical() {
    // These empty prefills change no bytes. The record is intent, not proof of
    // authenticity: removing it is indistinguishable from legacy input.
    for protocol in [QwenPromptTemplate::Qwen36, QwenPromptTemplate::Qwen38] {
        for (channel, mode) in [
            ("reasoning", LensMessageMode::Thinking),
            ("final", LensMessageMode::NoThinking),
        ] {
            let mut value = mode_fixture(protocol, Some(mode), Some((channel, "")));
            let explicit = parse(&value).unwrap();
            value.as_object_mut().unwrap().remove("generation_input");
            let legacy = parse(&value).unwrap();
            assert_eq!(explicit.prompt_token_ids, legacy.prompt_token_ids);
            assert_eq!(explicit.rendering, legacy.rendering);
            assert!(compare_runs(&explicit, &legacy, 1).is_err());
            assert!(ensure_sweep_run_context(&explicit, &legacy).is_err());
        }
    }
}
