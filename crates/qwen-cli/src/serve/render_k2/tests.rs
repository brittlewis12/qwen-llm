use super::*;
use serde_json::json;

#[test]
fn raw_origin_and_special_policy_survive_normalization_without_templates() {
    let text = "<think>literal</think> <|ifm|begin_of_text|> e\u{301}";
    let mut request = parse_request(&json!({"model":"k2", "input":text})).unwrap();
    normalize(&mut request, 8, 32).unwrap();
    assert_eq!(render(&request).unwrap().as_bytes(), text.as_bytes());
    assert_eq!(request.k2_add_special_tokens, Some(true));
    assert_eq!(request.max_output_tokens, Some(8));
    assert_eq!(sampling(&request), SamplingConfig::default());
    assert!(!request.parallel_tool_calls);
    assert_eq!(request.tool_choice, "none");
    let explicit =
        parse_request(&json!({"model":"k2", "input":text,"x_k2":{"add_special_tokens":false}}))
            .unwrap();
    assert_eq!(explicit.k2_raw_input.as_deref(), Some(text));
    assert_eq!(explicit.k2_add_special_tokens, Some(false));
    let generic =
        crate::open_responses::items::parse_request(&json!({"model":"qwen","input":text})).unwrap();
    assert!(render(&generic).is_err());
    assert!(
        crate::open_responses::items::parse_request(
            &json!({"model":"qwen","input":text,"x_k2":{"add_special_tokens":false}})
        )
        .is_err()
    );
}

#[test]
fn controls_are_honored_and_unsupported_presence_including_null_fails() {
    let mut request = parse_request(&json!({"model":"k2","input":"text","max_output_tokens":3,"temperature":0.6,"top_p":0.8,
        "x_qwen":{"seed":9,"top_k":12,"min_p":0.2,"stats":true},"store":false,"truncation":"disabled"})).unwrap();
    normalize(&mut request, 8, 32).unwrap();
    assert_eq!(request.max_output_tokens, Some(3));
    assert!(request.echo_stats);
    assert_eq!(
        sampling(&request),
        SamplingConfig {
            temperature: 0.6,
            top_p: 0.8,
            top_k: 12,
            min_p: 0.2,
            seed: 9
        }
    );
    for key in [
        "instructions",
        "tools",
        "reasoning",
        "previous_response_id",
        "messages",
        "tool_choice",
        "parallel_tool_calls",
        "unknown",
    ] {
        for value in [Value::Null, json!([]), json!(false), json!("")] {
            let mut body = json!({"model":"k2","input":"text"});
            body[key] = value;
            assert!(parse_request(&body).is_err(), "{key}");
        }
    }
    for extension in [
        json!({"thinking":false}),
        json!({"no_thinking":false}),
        json!({"history_thinking":"preserve"}),
        json!({"stats":null}),
        json!({"unknown":true}),
    ] {
        assert!(parse_request(&json!({"model":"k2","input":"text","x_qwen":extension})).is_err());
    }
    for extension in [
        json!(null),
        json!(true),
        json!({"add_special_tokens":null}),
        json!({"add_special_tokens":0}),
        json!({"typo":false}),
    ] {
        assert!(parse_request(&json!({"model":"k2","input":"text","x_k2":extension})).is_err());
    }
}

#[test]
fn raw_only_input_and_effective_output_limits_fail_closed() {
    for input in [
        json!(null),
        json!(""),
        json!([0, 42]),
        json!([{"role":"user","content":"text"}]),
    ] {
        assert!(parse_request(&json!({"model":"k2","input":input})).is_err());
    }
    for (key, value) in [
        ("store", json!(true)),
        ("truncation", json!("auto")),
        ("max_output_tokens", json!(0)),
        ("temperature", json!(null)),
    ] {
        let mut body = json!({"model":"k2","input":"text"});
        body[key] = value;
        assert!(parse_request(&body).is_err());
    }
    let mut request =
        parse_request(&json!({"model":"k2","input":"text","max_output_tokens":33})).unwrap();
    assert!(normalize(&mut request, 8, 32).is_err());
    let mut request = parse_request(&json!({"model":"k2","input":"text"})).unwrap();
    assert!(normalize(&mut request, 0, 32).is_err());
}
