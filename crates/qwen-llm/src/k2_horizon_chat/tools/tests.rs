use super::*;
use std::collections::BTreeMap;

fn fixture() -> Value {
    serde_json::from_str(include_str!("../../../tests/fixtures/k2_tools_hf.json")).unwrap()
}

#[test]
fn native_tool_fixture_inventory_is_named_and_classified() {
    let f = fixture();
    assert_eq!(f["schema"], "k2.tools_template_oracle.v1");
    assert_eq!(f["revision"], super::super::REVISION);
    assert_eq!(f["template_sha256"], super::super::TEMPLATE_SHA256);
    assert_eq!(
        f["generation_config_sha256"],
        super::super::GENERATION_CONFIG_SHA256
    );
    let mut required = BTreeMap::new();
    for name in [
        "tools-default",
        "tools-system-fallback",
        "tools-explicit-empty-fallback",
        "tools-top-level-priority",
        "tools-empty",
        "tools-tool-choice-ignored",
        "tools-result-string",
        "tools-result-null",
        "tools-nonobject-parameters-fallback",
        "tools-xml_typed-local-ref-history",
        "tools-xml_typed-duplicate-definition",
    ] {
        required.insert(name.to_owned(), false);
    }
    for name in [
        "tools-result-empty-list",
        "tools-reject-tool_presentation",
        "tools-reject-tool_calling_format",
        "tools-reject-tool_format",
        "tools-reject-tool_presentation_format",
        "tools-reject-tool_call_format",
        "tools-reject-string-arguments",
        "tools-reject-missing-name",
    ] {
        required.insert(name.to_owned(), true);
    }
    for presentation in ["markdown", "json", "xml"] {
        for format in ["xml", "json", "xml_typed"] {
            required.insert(format!("tools-{presentation}-{format}-history"), false);
        }
        for suffix in ["recursive-ref", "json-fallback"] {
            required.insert(format!("tools-{presentation}-{suffix}"), false);
        }
    }
    for format in ["xml", "json", "xml_typed"] {
        for suffix in ["verbatim-delimiters", "numeric-json"] {
            required.insert(format!("tools-{format}-{suffix}"), false);
        }
    }
    let cases = f["cases"].as_array().unwrap();
    for presentation in ["markdown", "xml", "json"] {
        for name in [
            "annotations",
            "root-variants",
            "reference-chain",
            "boolean-parameters",
            "container-fallback",
            "external-ref",
            "whitespace",
            "whole-set-fallback",
        ] {
            required.insert(format!("schema-{name}-{presentation}"), false);
        }
        for name in [
            "fallback-must-still-validate",
            "missing-required",
            "string-required",
            "undefined-required",
        ] {
            required.insert(format!("schema-{name}-{presentation}"), true);
        }
    }
    let actual: BTreeMap<_, _> = cases
        .iter()
        .map(|case| {
            (
                case["name"].as_str().unwrap().to_owned(),
                case.get("error").is_some(),
            )
        })
        .collect();
    assert_eq!(actual.len(), cases.len(), "duplicate fixture names");
    assert_eq!(actual, required);
    let by_name = |name: &str| cases.iter().find(|case| case["name"] == name).unwrap();
    for presentation in ["markdown", "xml"] {
        assert_eq!(
            by_name(&format!("tools-{presentation}-json-fallback"))["body"],
            by_name("tools-json-json-fallback")["body"]
        );
    }
    assert_eq!(
        by_name("tools-default")["body"],
        by_name("tools-tool-choice-ignored")["body"],
        "tool_choice is not an upstream control"
    );
}

fn calls(message: &Value) -> Vec<ToolCall> {
    message["tool_calls"]
        .as_array()
        .unwrap()
        .iter()
        .map(|call| serde_json::from_value(call.get("function").unwrap_or(call).clone()).unwrap())
        .collect()
}

#[test]
fn native_calls_and_results_match_upstream_all_formats_without_an_engine() {
    let f = fixture();
    let mut call_count = 0;
    let mut result_count = 0;
    for case in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|case| case.get("error").is_none())
    {
        let format = case
            .get("tool_call_format")
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .unwrap_or_default();
        let definitions = case["tools"].as_array().cloned().unwrap_or_default();
        let mut rendered_calls = Vec::new();
        let mut rendered_results = Vec::new();
        for message in case["messages"].as_array().unwrap() {
            if message.get("tool_calls").is_some() {
                rendered_calls
                    .push(render_tool_calls(&calls(message), format, &definitions).unwrap());
                call_count += 1;
            }
            if message["role"] == "tool" {
                rendered_results.push(render_tool_result(&message["content"]).unwrap());
                result_count += 1;
            }
        }
        assert_eq!(
            serde_json::json!(rendered_calls),
            case["call_blocks"],
            "{}",
            case["name"]
        );
        assert_eq!(
            serde_json::json!(rendered_results),
            case["tool_results"],
            "{}",
            case["name"]
        );
        for block in rendered_calls.into_iter().chain(rendered_results) {
            assert!(
                case["body"].as_str().unwrap().contains(&block),
                "macro must occur in full render"
            );
        }
    }
    assert_eq!(call_count, 19);
    assert_eq!(result_count, 36);
}

#[test]
fn native_tool_json_types_keep_strings_numbers_order_and_signed_zero() {
    let f = fixture();
    let numbers = f["json_number_oracle"].as_array().unwrap();
    assert_eq!(numbers.len(), 276);
    for number in numbers {
        assert_eq!(
            json::encode(&number["value"]).unwrap(),
            number["expected"],
            "{number}"
        );
    }
    assert_eq!(
        render_tool_calls(&[], ToolCallFormat::Xml, &[]).unwrap(),
        f["empty_call_block"]
    );
    let value: Value = serde_json::from_str(
        r#"{"z":"123","a":[1,1.0,-0.0,1e-5,1e-4,1e15,1e16,123456789012345678901234567890]}"#,
    )
    .unwrap();
    assert_eq!(
        json::encode(&value).unwrap(),
        r#"{"z": "123", "a": [1, 1.0, -0.0, 1e-05, 0.0001, 1000000000000000.0, 1e+16, 123456789012345678901234567890]}"#
    );
    assert!(json::encode(&serde_json::from_str::<Value>("1e999").unwrap()).is_err());
    assert!(render_tool_result(&serde_json::json!([])).is_err());
    for document in [
        r#"{"name":"f","arguments":"{}"}"#,
        r#"{"name":"f","arguments":[]}"#,
        r#"{"name":"f","arguments":{},"unexpected":1}"#,
    ] {
        assert!(serde_json::from_str::<ToolCall>(document).is_err());
    }
    assert_eq!(ToolCallFormat::default(), ToolCallFormat::Xml);
    for value in ["yaml", "xml-typed", "none"] {
        assert!(serde_json::from_value::<ToolCallFormat>(Value::String(value.into())).is_err());
    }
}

#[test]
fn typed_calls_follow_schema_refs_and_union_actual_value_types() {
    let definitions = serde_json::json!([{"name":"f","parameters":{
    "$defs":{"Count":{"type":"integer"}}, "properties":{
        "count":{"$ref":"#/$defs/Count"},
        "label":{"$ref":"#/$defs/Count","type":"string"},
        "choice":{"type":["integer","string"]},
        "nested":{"type":"array","items":{"anyOf":[{"type":"string"},{"type":"null"}]}},
        "cyclic":{"$ref":"#/$defs/Cycle"}
    }}}]);
    let defs = definitions.as_array().unwrap();
    for (key, value, expected) in [
        ("count", serde_json::json!(2), "integer"),
        ("label", serde_json::json!("123"), "string"),
        ("choice", serde_json::json!("123"), "string"),
        ("choice", serde_json::json!(3), "integer"),
        ("choice", serde_json::json!(3.0), "number"),
        ("nested", serde_json::json!(["x"]), "array"),
        ("cyclic", serde_json::json!(null), "Cycle"),
        ("unknown", serde_json::json!(true), "any"),
    ] {
        assert_eq!(
            types::argument_type(defs, "f", key, &value).unwrap(),
            expected
        );
    }
}
