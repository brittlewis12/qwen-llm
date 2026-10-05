use super::*;
use serde_json::{Value, json};

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../tests/fixtures/glm53_chat_hf.json")).unwrap()
}

fn cases(fixture: &Value) -> &[Value] {
    fixture["cases"].as_array().unwrap()
}

/// The fixture case through the product path: document parse, effort
/// parse, render.
fn native(case: &Value) -> Result<String> {
    let mut document = json!({"messages": case["messages"]});
    if let Some(tools) = case.get("tools") {
        document["tools"] = tools.clone();
    }
    let document = parse_document(&serde_json::to_vec(&document).unwrap())?;
    assert_eq!(document.clear_thinking, None);
    let effort = Effort::parse(case.get("reasoning_effort").and_then(Value::as_str))?;
    render(
        &document.messages,
        RenderOptions {
            effort,
            clear_thinking: case["clear_thinking"].as_bool().unwrap_or(false),
            add_generation_prompt: case["add_generation_prompt"].as_bool().unwrap(),
        },
    )
}

#[test]
fn fixture_pins_the_upstream_release_and_the_gguf_conversion() {
    let f = fixture();
    assert_eq!(f["schema"], "glm53.text_chat_template_oracle.v1");
    assert_eq!(f["revision"], REVISION);
    assert_eq!(f["template_sha256"], TEMPLATE_SHA256);
    assert_eq!(f["gguf_template_sha256"], GGUF_TEMPLATE_SHA256);
    assert_eq!(f["generation_config_sha256"], GENERATION_CONFIG_SHA256);
    let generation = &f["generation_config"];
    assert_eq!(generation["eos_token_id"], json!(CHAT_STOPS));
    assert_eq!(generation["temperature"], json!(TEMPERATURE));
    assert_eq!(generation["top_p"].as_f64().unwrap() as f32, TOP_P);
    assert!(cases(&f).iter().all(|case| case.get("error").is_none()));
}

#[test]
fn renders_every_supported_case_byte_for_byte_and_refuses_the_rest() {
    let f = fixture();
    let max = cases(&f)
        .iter()
        .find(|case| case["name"] == "effort-max")
        .unwrap()["rendered"]
        .as_str()
        .unwrap();
    let mut rendered = 0;
    for case in cases(&f) {
        let name = case["name"].as_str().unwrap();
        let expected = case["rendered"].as_str().unwrap();
        match case["native"].as_str().unwrap() {
            "render" => {
                assert_eq!(native(case).unwrap(), expected, "{name}");
                rendered += 1;
            }
            refusal => {
                let code = match refusal {
                    "refuse:effort" => {
                        // Upstream silently renders every unknown effort as Max.
                        assert_eq!(expected, max, "{name}");
                        "glm5_next_chat_effort"
                    }
                    "refuse:last_turn"
                    | "refuse:empty"
                    | "refuse:role"
                    | "refuse:content_parts"
                    | "refuse:input" => "glm5_next_chat_input",
                    "refuse:tools" => "glm5_next_chat_tools",
                    other => panic!("{name}: unknown native policy {other}"),
                };
                assert_eq!(native(case).unwrap_err().code(), code, "{name}");
            }
        }
    }
    assert!(rendered >= 35, "{rendered}");
}

#[test]
fn only_null_assistant_content_separates_the_gguf_template_from_upstream() {
    let f = fixture();
    let differ: Vec<_> = cases(&f)
        .iter()
        .filter(|case| case.get("gguf_rendered").is_some())
        .collect();
    assert_eq!(differ.len(), 1);
    let case = differ[0];
    assert_eq!(case["name"], "history-content-null");
    assert!(
        case["gguf_rendered"]
            .as_str()
            .unwrap()
            .contains("</think>None<|user|>")
    );
    assert_eq!(native(case).unwrap(), case["rendered"].as_str().unwrap());
}

#[test]
fn assistant_strip_is_pythons_str_strip_at_every_code_point() {
    let f = fixture();
    let python: std::collections::BTreeSet<u32> = f["python_isspace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_u64().unwrap() as u32)
        .collect();
    for code in 0..=0x10_ffff_u32 {
        if let Some(c) = char::from_u32(code) {
            assert_eq!(python_space(c), python.contains(&code), "U+{code:04X}");
        }
    }
}

#[test]
fn effort_levels_are_explicit_and_default_to_max() {
    assert_eq!(Effort::parse(None).unwrap(), Effort::Max);
    for level in Effort::LEVELS {
        assert_eq!(Effort::parse(Some(level)).unwrap().as_str(), level);
    }
    assert_eq!(Effort::default(), Effort::Max);
    for bad in ["medium", "none", "minimal", "xhigh", "Max", " max", ""] {
        let error = Effort::parse(Some(bad)).unwrap_err();
        assert_eq!(error.code(), "glm5_next_chat_effort", "{bad}");
        assert!(error.to_string().contains("no non-thinking mode"));
    }
}

#[test]
fn documents_refuse_tools_and_anything_the_renderer_would_drop() {
    for (document, code) in [
        (
            r#"{"messages":[{"role":"user","content":"x"}],"tools":[]}"#,
            "glm5_next_chat_tools",
        ),
        (
            r#"[{"role":"assistant","content":null,"tool_calls":[]},{"role":"user","content":"x"}]"#,
            "glm5_next_chat_tools",
        ),
        (
            r#"[{"role":"tool","content":"r"},{"role":"user","content":"x"}]"#,
            "glm5_next_chat_tools",
        ),
        (
            r#"[{"role":"user","content":"x","reasoning_content":"r"}]"#,
            "glm5_next_chat_input",
        ),
        (
            r#"[{"role":"system","content":"s","reasoning_content":""},{"role":"user","content":"x"}]"#,
            "glm5_next_chat_input",
        ),
        (
            r#"[{"role":"user","content":"x","name":"n"}]"#,
            "glm5_next_chat_input",
        ),
        (r#"[{"role":"user"}]"#, "glm5_next_chat_input"),
        (
            r#"[{"role":"user","content":null}]"#,
            "glm5_next_chat_input",
        ),
        (r#"[{"role":"user","content":1}]"#, "glm5_next_chat_input"),
        (
            r#"[{"role":"developer","content":"d"}]"#,
            "glm5_next_chat_input",
        ),
        (
            r#"[{"role":"user","role":"assistant","content":"x"}]"#,
            "glm5_next_chat_input",
        ),
        (
            r#"{"messages":[],"clear_thinking":"yes"}"#,
            "glm5_next_chat_input",
        ),
        (
            r#"{"messages":[],"reasoning_effort":"low"}"#,
            "glm5_next_chat_input",
        ),
        ("{}", "glm5_next_chat_input"),
    ] {
        assert_eq!(
            parse_document(document.as_bytes()).unwrap_err().code(),
            code,
            "{document}"
        );
    }
    let document = parse_document(
        br#"{"messages":[{"role":"assistant"},{"role":"user","content":"q"}],"clear_thinking":true}"#,
    )
    .unwrap();
    assert_eq!(document.clear_thinking, Some(true));
    assert_eq!(
        document.messages[0],
        Message::Assistant {
            content: String::new(),
            reasoning: None
        }
    );
}

#[test]
fn a_reply_extends_its_prompt_when_rendered_back_as_history() {
    // The model ends a turn by sampling `<|user|>`, the next turn's opener:
    // prompt + generated text + the next user turn is exactly the next
    // prompt, so a live session can continue instead of re-prefilling.
    let options = RenderOptions::generate(Effort::High, false);
    let first = vec![Message::User("Q1".into())];
    let prompt = render(&first, options).unwrap();
    let generated = "R1</think>A1";
    let mut second = first.clone();
    second.push(Message::Assistant {
        content: "A1".into(),
        reasoning: Some("R1".into()),
    });
    second.push(Message::User("Q2".into()));
    let next = render(&second, options).unwrap();
    assert_eq!(
        next,
        format!("{prompt}{generated}<|user|>Q2<|assistant|><think>")
    );
    // clear_thinking rewrites the earlier turn, so it cannot extend.
    let cleared = render(&second, RenderOptions::generate(Effort::High, true)).unwrap();
    assert!(!cleared.starts_with(&format!("{prompt}{generated}")));
}

#[test]
#[ignore = "CPU/header-only; requires GLM53_GGUF (GLM-5.3-Flash shard 1), no inference or GPU"]
fn gguf_profile_verifies_and_native_tokens_match_hf() {
    let path = crate::test_fixtures::GLM53_FLASH_UD_IQ3_XXS.required();
    let gguf = GgufFile::open(&path).unwrap();
    let artifact = crate::glm5_next::admission::Glm5NextPreparedArtifact::inspect(&gguf).unwrap();
    let stops = artifact.generation_stops().unwrap();
    let profile = verify_profile(&gguf, artifact.tokenizer(), &stops).unwrap();
    assert_eq!(profile.template_sha256, GGUF_TEMPLATE_SHA256);
    assert_eq!(profile.template_source, "unsloth_gguf");
    eprintln!("tokenizer_metadata_id={}", profile.tokenizer_metadata_id);
    let error = verify_profile(&gguf, artifact.tokenizer(), &[154_820, 154_827]).unwrap_err();
    assert_eq!(error.code(), "glm5_next_chat_profile_unverified");
    let f = fixture();
    let mut checked = 0;
    for case in cases(&f).iter().filter(|case| case["native"] == "render") {
        let ids = artifact
            .tokenizer()
            .encode(&native(case).unwrap(), false)
            .unwrap();
        assert_eq!(json!(ids), case["token_ids"], "{}", case["name"]);
        checked += 1;
    }
    assert!(checked >= 35);
}
