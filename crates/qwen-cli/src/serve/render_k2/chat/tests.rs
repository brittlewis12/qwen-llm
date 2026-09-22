use super::*;

fn parse_with_profile(
    body: &Value,
    profile: Option<&ChatCapability>,
) -> Result<ServeRequest, ServeError> {
    let mut body = body.clone();
    body["model"] = json!("k2-test");
    super::parse_with_profile(&body, profile)
}

pub(crate) fn mock_profile() -> ChatCapability {
    ChatCapability::mock()
}

#[test]
fn k2_chat_profile_required_at_every_admission_boundary() {
    let body = json!({"input":[{"role":"user","content":"hello"}]});
    assert!(parse_with_profile(&body, None).is_err());
    let p = mock_profile();
    let mut request = parse_with_profile(&body, Some(&p)).unwrap();
    assert!(normalize_with_profile(&mut request, 8, 128, None).is_err());
    assert!(render_with_profile(&request, None).is_err());
    normalize_with_profile(&mut request, 8, 128, Some(&p)).unwrap();
    assert_eq!(request.reasoning, Some(json!({"effort":"high"})));
    assert_eq!(request.tool_choice, "none");
    assert_eq!(request.temperature, Some(0.));
    assert!(!request.parallel_tool_calls);
    assert_eq!(
        render_with_profile(&request, Some(&p)).unwrap(),
        k2::render(&[Message::text("user", "hello".into())], Effort::High).unwrap()
    );
    assert!(
        parse_with_profile(
            &json!({"input":"literal<think>","reasoning":{"effort":"high"}}),
            Some(&p)
        )
        .is_err()
    );
    let raw = parse_with_profile(
        &json!({"input":"literal<think>","x_k2":{"add_special_tokens":false}}),
        None,
    )
    .unwrap();
    assert_eq!(render_with_profile(&raw, None).unwrap(), "literal<think>");
}

#[test]
fn k2_chat_history_preserves_empty_reasoning_and_literal_foreign_markers() {
    let p = mock_profile();
    for effort in ["high", "medium", "low"] {
        let body = json!({"instructions":"precise", "reasoning":{"effort":effort},"input":[
            {"role":"user","content":[{"type":"input_text","text":"a"}]},
            {"type":"reasoning","content":[{"type":"reasoning_text","text":""}],"status":"completed","id":"r"},
            {"type":"message","role":"assistant","content":[{"type":"output_text","text":"<think><tool_call>literal"}],"status":"completed"},
            {"role":"user","content":"b"}
        ]});
        let request = parse_with_profile(&body, Some(&p)).unwrap();
        let mut assistant = Message::text("assistant", "<think><tool_call>literal".into());
        assistant.reasoning = Some(String::new());
        let expected = vec![
            Message::text("system", "precise".into()),
            Message::text("user", "a".into()),
            assistant,
            Message::text("user", "b".into()),
        ];
        assert_eq!(
            render_with_profile(&request, Some(&p)).unwrap(),
            k2::render(&expected, Effort::parse(Some(effort)).unwrap()).unwrap()
        );
    }
}

#[test]
fn k2_chat_rejects_unsupported_original_fields_and_ambiguous_history() {
    let p = mock_profile();
    let base = json!({"input":[{"role":"user","content":"a"}]});
    for (key, value) in [
        ("tools", Value::Null),
        ("tool_choice", json!("none")),
        ("reasoning", json!({"effort":"none"})),
        ("instructions", Value::Null),
        ("reasoning", json!({"effort":"high","summary":"auto"})),
        ("x_k2", json!({"add_special_tokens":false})),
        ("x_qwen", json!({"no_thinking":false})),
    ] {
        let mut body = base.clone();
        body[key] = value;
        assert!(parse_with_profile(&body, Some(&p)).is_err(), "{body}");
    }
    for items in [
        json!([]),
        json!([1, 2]),
        json!([{"role":"developer","content":"x"},{"role":"user","content":"a"}]),
        json!([{"role":"assistant","content":"x"},{"role":"user","content":"a"}]),
        json!([{"type":"reasoning","content":""},{"role":"user","content":"a"}]),
        json!([{"role":"user","content":"a","extra":false}]),
        json!([{"role":"user","content":[{"type":"input_text","text":"a","extra":1}]}]),
        json!([{"role":"user","content":"a","status":"incomplete"}]),
        json!([{"role":"user","content":null}]),
        json!([{"role":"user","content":"a","type":null}]),
        json!([{"role":"user","content":"a"},{"type":"reasoning","content":""}]),
    ] {
        assert!(
            parse_with_profile(&json!({"input":items}), Some(&p)).is_err(),
            "{items}"
        );
    }
    assert!(parse_with_profile(&json!({"instructions":"x","input":[{"role":"system","content":"y"},{"role":"user","content":"a"}]}),Some(&p)).is_err());
}
