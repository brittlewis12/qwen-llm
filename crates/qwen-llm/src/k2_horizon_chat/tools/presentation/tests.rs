use super::*;

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../../../tests/fixtures/k2_tools_hf.json")).unwrap()
}

#[test]
fn native_schema_presentations_match_pinned_upstream() {
    let f = fixture();
    let mut compared = 0;
    for case in f["cases"].as_array().unwrap() {
        let definitions = case["tools"]
            .as_array()
            .filter(|v| !v.is_empty())
            .or_else(|| case["messages"][0]["tools"].as_array())
            .cloned()
            .unwrap_or_default();
        let format = case
            .get("tool_presentation_format")
            .map(|v| serde_json::from_value(v.clone()))
            .transpose();
        if let Some(expected) = case.get("tool_definitions") {
            let presentation = format.as_ref().unwrap().unwrap_or_default();
            let call_format = case
                .get("tool_call_format")
                .map(|v| serde_json::from_value(v.clone()).unwrap())
                .unwrap_or_default();
            let first = &case["messages"][0];
            let content = if first["role"] == "system" {
                first["content"].as_str().unwrap()
            } else {
                ""
            };
            let system =
                render_tool_system(&definitions, content, presentation, call_format).unwrap();
            assert_eq!(system, case["tool_system"], "{}", case["name"]);
            if !definitions.is_empty() {
                assert!(
                    case["body"].as_str().unwrap().starts_with(&system),
                    "native system turn must match full upstream prompt prefix"
                );
            }
            let actual =
                render_tool_definitions(&definitions, format.unwrap().unwrap_or_default()).unwrap();
            assert_eq!(&Value::String(actual), expected, "{}", case["name"]);
            compared += 1;
        } else if case["name"].as_str().unwrap().starts_with("schema-") {
            assert!(
                render_tool_definitions(&definitions, format.unwrap().unwrap_or_default()).is_err(),
                "{}",
                case["name"]
            );
        }
    }
    assert_eq!(compared, 56);
}

#[test]
fn schema_fallback_is_global_and_never_masks_later_validation() {
    let good =
        serde_json::json!({"name":"good","parameters":{"properties":{"x":{"type":"string"}}}});
    let fallback = serde_json::json!({"name":"fallback","parameters":true});
    let bad = serde_json::json!({"name":"bad","parameters":{"required":["missing"]}});
    let definitions = [good, fallback.clone()];
    let json = render_tool_definitions(&definitions, ToolPresentationFormat::Json).unwrap();
    for format in [
        ToolPresentationFormat::Markdown,
        ToolPresentationFormat::Xml,
    ] {
        assert_eq!(render_tool_definitions(&definitions, format).unwrap(), json);
        assert!(render_tool_definitions(&[fallback.clone(), bad.clone()], format).is_err());
    }
    assert_eq!(
        ToolPresentationFormat::default(),
        ToolPresentationFormat::Markdown
    );
}

#[test]
fn excessive_schema_nesting_returns_error_without_truncation() {
    let mut spec = serde_json::json!({});
    for _ in 0..130 {
        spec = serde_json::json!({"items":spec});
    }
    let tool = serde_json::json!({"name":"deep","parameters":spec});
    for format in [
        ToolPresentationFormat::Markdown,
        ToolPresentationFormat::Xml,
        ToolPresentationFormat::Json,
    ] {
        assert!(
            render_tool_definitions(&[tool.clone()], format)
                .unwrap_err()
                .to_string()
                .contains("nesting safety")
        );
    }
}

#[test]
fn flat_reference_chains_have_a_separate_expansion_guard() {
    let mut defs = Map::new();
    for index in 0..150 {
        defs.insert(
            format!("Node{index}"),
            serde_json::json!({"type":"object","properties":{
                "next":{"$ref":format!("#/$defs/Node{}", index+1)}
            }}),
        );
    }
    let tool = serde_json::json!({"name":"chain","parameters":{"$defs":defs,"properties":{"root":{"$ref":"#/$defs/Node0"}}}});
    assert!(render_tool_definitions(&[tool.clone()], ToolPresentationFormat::Json).is_ok());
    for format in [
        ToolPresentationFormat::Markdown,
        ToolPresentationFormat::Xml,
    ] {
        assert!(
            render_tool_definitions(&[tool.clone()], format)
                .unwrap_err()
                .to_string()
                .contains("reference expansion")
        );
    }
}

#[test]
#[ignore = "CPU only: K2_GGUF verified final artifact and native tool-template token fixtures"]
fn cpu_native_tool_template_token_ids() {
    let source = crate::gguf::GgufFile::open(std::env::var("K2_GGUF").unwrap()).unwrap();
    crate::k2_horizon_chat::verify_profile(&source).unwrap();
    let tokenizer = crate::tokenizer::NativeTokenizer::from_gguf(&source).unwrap();
    let f = fixture();
    let mut count = 0;
    for case in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c.get("error").is_none())
    {
        assert_eq!(
            serde_json::json!(
                tokenizer
                    .encode(case["body"].as_str().unwrap(), true)
                    .unwrap()
            ),
            case["token_ids"],
            "{}",
            case["name"]
        );
        count += 1;
    }
    assert_eq!(count, 56);
}
