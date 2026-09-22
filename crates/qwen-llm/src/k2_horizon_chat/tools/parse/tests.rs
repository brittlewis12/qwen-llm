use super::*;

fn definitions() -> Vec<Value> {
    vec![
        serde_json::json!({"name":"lookup","parameters":{"type":"object","properties":{
            "text":{"type":"string"},"count":{"type":"integer"},"flag":{"type":"boolean"},
            "items":{"type":"array"},"object":{"type":"object"},"none":{"type":"null"},
            "choice":{"type":["integer","string"]}
        }}}),
    ]
}
fn json_block(payload: &str) -> String {
    format!("{TOOL_BLOCK_OPEN}{CALL_OPEN}{payload}{CALL_CLOSE}{BLOCK_CLOSE}")
}
fn call(arguments: Value) -> ToolCall {
    ToolCall {
        name: "lookup".into(),
        arguments: arguments.as_object().unwrap().clone(),
    }
}

#[test]
fn complete_calls_preserve_types_and_json_marker_strings() {
    let reserved = vec![call(
        serde_json::json!({"object":{"$serde_json::private::Number":"123"}}),
    )];
    let rendered = render_tool_calls(&reserved, ToolCallFormat::Json, &definitions()).unwrap();
    assert_eq!(
        parse_tool_calls(&rendered, ToolCallFormat::Json, &definitions()).unwrap(),
        ParsedToolBlock::Complete(reserved)
    );
    let calls = vec![call(
        serde_json::json!({"text":" 123 &amp; ","count":2,"flag":false,"items":["</ifm|arg_value>"],"object":{"z":1,"text":"</ifm|arg_value></ifm|tool_call>"},"none":null}),
    )];
    for format in [
        ToolCallFormat::Xml,
        ToolCallFormat::Json,
        ToolCallFormat::XmlTyped,
    ] {
        let rendered = render_tool_calls(&calls, format, &definitions()).unwrap();
        assert_eq!(
            parse_tool_calls(&rendered, format, &definitions()).unwrap(),
            ParsedToolBlock::Complete(calls.clone())
        );
        for cut in 0..rendered.len() {
            if rendered.is_char_boundary(cut) {
                assert_eq!(
                    parse_tool_calls(&rendered[..cut], format, &definitions()).unwrap(),
                    ParsedToolBlock::Incomplete,
                    "format={format:?} cut={cut}"
                );
            }
        }
    }
    let calls = vec![call(
        serde_json::json!({"text":"</ifm|tool_call></ifm|tool_calls> \" \\ \u{1f389}"}),
    )];
    let rendered = render_tool_calls(&calls, ToolCallFormat::Json, &definitions()).unwrap();
    assert_eq!(
        parse_tool_calls(&rendered, ToolCallFormat::Json, &definitions()).unwrap(),
        ParsedToolBlock::Complete(calls)
    );
}

#[test]
fn malformed_json_never_salvages_calls_or_loses_duplicate_keys() {
    for payload in [
        r#"{"name":"lookup","name":"lookup","arguments":{}}"#,
        r#"{"name":"lookup","arguments":{"text":"a","text":"b"}}"#,
        r#"{"name":"lookup","arguments":{"object":{"a":1,"\u0061":2}}}"#,
        r#"{"name":"lookup","arguments":{"items":[{"a":1,"a":2}]}}"#,
        r#"{"name":"unknown","arguments":{}}"#,
        r#"{"name":"lookup","arguments":{},"extra":true}"#,
        r#"{"name":"lookup","arguments":"{}"}"#,
        r#"{"name":"lookup","arguments":[]}"#,
        r#"{"name":"lookup","arguments":{"count":1e999}}"#,
        r#"{"name":"lookup","arguments":{,}}"#,
    ] {
        assert!(
            parse_tool_calls(&json_block(payload), ToolCallFormat::Json, &definitions()).is_err(),
            "{payload}"
        );
    }
    let good = json_block(r#"{"name":"lookup","arguments":{}}"#);
    for suffix in ["junk", "<ifm|tool_calls></ifm|tool_calls>"] {
        assert!(
            parse_tool_calls(
                &format!("{good}{suffix}"),
                ToolCallFormat::Json,
                &definitions()
            )
            .is_err()
        );
    }
    let corrupt_second = good.replace(
        BLOCK_CLOSE,
        &format!("{CALL_OPEN}{{bad}}{CALL_CLOSE}{BLOCK_CLOSE}"),
    );
    assert!(parse_tool_calls(&corrupt_second, ToolCallFormat::Json, &definitions()).is_err());
    assert!(
        parse_tool_calls(
            &format!("{TOOL_BLOCK_OPEN}{BLOCK_CLOSE}"),
            ToolCallFormat::Json,
            &definitions()
        )
        .is_err()
    );
}

#[test]
fn xml_ambiguity_and_type_labels_are_not_guessed() {
    let calls = vec![call(serde_json::json!({"choice":"123"}))];
    let xml = render_tool_calls(&calls, ToolCallFormat::Xml, &definitions()).unwrap();
    assert!(
        parse_tool_calls(&xml, ToolCallFormat::Xml, &definitions())
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
    let typed = render_tool_calls(&calls, ToolCallFormat::XmlTyped, &definitions()).unwrap();
    assert_eq!(
        parse_tool_calls(&typed, ToolCallFormat::XmlTyped, &definitions()).unwrap(),
        ParsedToolBlock::Complete(calls)
    );
    let calls = vec![call(serde_json::json!({"count":3}))];
    let typed = render_tool_calls(&calls, ToolCallFormat::XmlTyped, &definitions()).unwrap();
    assert!(
        parse_tool_calls(
            &typed.replace("<ifm|arg_type>integer", "<ifm|arg_type>string"),
            ToolCallFormat::XmlTyped,
            &definitions()
        )
        .is_err()
    );
    let xml = render_tool_calls(&calls, ToolCallFormat::Xml, &definitions()).unwrap();
    let argument = "<ifm|arg_key>count</ifm|arg_key>\n<ifm|arg_value>3</ifm|arg_value>\n";
    assert!(
        parse_tool_calls(
            &xml.replace(argument, &argument.repeat(2)),
            ToolCallFormat::Xml,
            &definitions()
        )
        .is_err()
    );
    let unknown = vec![serde_json::json!({"name":"lookup","parameters":{}})];
    assert!(
        parse_tool_calls(&xml, ToolCallFormat::Xml, &unknown)
            .unwrap_err()
            .to_string()
            .contains("ambiguous")
    );
    let text = vec![call(serde_json::json!({"text":"not JSON"}))];
    let xml = render_tool_calls(&text, ToolCallFormat::Xml, &unknown).unwrap();
    assert_eq!(
        parse_tool_calls(&xml, ToolCallFormat::Xml, &unknown).unwrap(),
        ParsedToolBlock::Complete(text)
    );
}

#[test]
fn pinned_history_blocks_decode_where_the_dialect_is_unambiguous() {
    let f: Value =
        serde_json::from_str(include_str!("../../../../tests/fixtures/k2_tools_hf.json")).unwrap();
    let mut count = 0;
    for case in f["cases"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c.get("error").is_none())
    {
        let name = case["name"].as_str().unwrap();
        let format = case
            .get("tool_call_format")
            .map(|v| serde_json::from_value(v.clone()).unwrap())
            .unwrap_or_default();
        let defs = case["tools"].as_array().cloned().unwrap_or_default();
        for (message, block) in case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|m| m.get("tool_calls").is_some())
            .zip(case["call_blocks"].as_array().unwrap())
        {
            let parsed = parse_tool_calls(block.as_str().unwrap(), format, &defs);
            if name.contains("duplicate-definition")
                || (name.contains("verbatim-delimiters") && format != ToolCallFormat::Json)
            {
                assert!(parsed.is_err(), "{name}");
            } else {
                let calls: Vec<ToolCall> = message["tool_calls"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|c| {
                        serde_json::from_value(c.get("function").unwrap_or(c).clone()).unwrap()
                    })
                    .collect();
                assert_eq!(parsed.unwrap(), ParsedToolBlock::Complete(calls), "{name}");
            }
            count += 1;
        }
    }
    assert_eq!(count, 19);
}

#[test]
fn nested_json_duplicates_and_bounded_reference_coercion_fail_closed() {
    let defs = vec![
        serde_json::json!({"name":"lookup","parameters":{"$defs":{"Number":{"type":"integer"}},"properties":{"count":{"$ref":"#/$defs/Number"},"object":{"const":{}}}}}),
    ];
    let calls = vec![call(serde_json::json!({"count":1234567890123456789u64}))];
    let xml = render_tool_calls(&calls, ToolCallFormat::Xml, &defs).unwrap();
    assert_eq!(
        parse_tool_calls(&xml, ToolCallFormat::Xml, &defs).unwrap(),
        ParsedToolBlock::Complete(calls)
    );
    let bad = "<ifm|tool_calls><ifm|tool_call>lookup\n<ifm|arg_key>object</ifm|arg_key><ifm|arg_value>{\"a\":1,\"a\":2}</ifm|arg_value></ifm|tool_call></ifm|tool_calls>";
    assert!(parse_tool_calls(bad, ToolCallFormat::Xml, &defs).is_err());
}

#[test]
fn argument_reference_and_input_depth_guards_are_explicit() {
    let mut refs = Map::new();
    for index in 0..140 {
        refs.insert(
            format!("T{index}"),
            serde_json::json!({"$ref":format!("#/$defs/T{}", index+1)}),
        );
    }
    let defs = vec![
        serde_json::json!({"name":"lookup","parameters":{"$defs":refs,"properties":{"count":{"$ref":"#/$defs/T0"}}}}),
    ];
    let block = "<ifm|tool_calls><ifm|tool_call>lookup\n<ifm|arg_key>count</ifm|arg_key><ifm|arg_value>2</ifm|arg_value></ifm|tool_call></ifm|tool_calls>";
    assert!(
        parse_tool_calls(block, ToolCallFormat::Xml, &defs)
            .unwrap_err()
            .to_string()
            .contains("safety limit")
    );
    let mut spec = Value::Null;
    for _ in 0..130 {
        spec = serde_json::json!({"items":spec});
    }
    let defs = vec![serde_json::json!({"name":"lookup","parameters":spec})];
    assert!(
        parse_tool_calls(block, ToolCallFormat::Xml, &defs)
            .unwrap_err()
            .to_string()
            .contains("nesting safety")
    );
}

#[test]
fn direct_json_containers_keep_scalar_syntax_and_number_oracles() {
    let f: Value =
        serde_json::from_str(include_str!("../../../../tests/fixtures/k2_tools_hf.json")).unwrap();
    for number in f["json_number_oracle"].as_array().unwrap() {
        let text = number["expected"].as_str().unwrap();
        assert_eq!(
            json::encode(&json_decode::complete(text).unwrap()).unwrap(),
            text
        );
    }
    for text in [
        r#"{"a":[true,false,null,-2.3e-7],"s":"\uD83D\uDE80\t\"\\"}"#,
        "[]",
        "{}",
        "\r\n [1,2] \t",
    ] {
        assert_eq!(
            json_decode::complete(text).unwrap(),
            serde_json::from_str::<Value>(text).unwrap()
        );
    }
    for text in [
        "[1,]",
        r#"{"a":1,}"#,
        r#"{"a" 1}"#,
        r#"{"a":01}"#,
        r#"{"a":+1}"#,
        "[true false]",
        "True",
        r#""\uD800""#,
        "1e",
        "1.",
        "-",
        "[1}\n",
        "1 2",
    ] {
        assert!(json_decode::complete(text).is_err(), "{text}");
    }
}

#[test]
fn mathematical_integer_membership_and_native_float_labels_are_separate() {
    for text in ["3.0", "1000.0", "-0.0", "1e+20"] {
        let value: Value = serde_json::from_str(text).unwrap();
        let calls = vec![call(serde_json::json!({"count":value.clone()}))];
        for format in [ToolCallFormat::Xml, ToolCallFormat::XmlTyped] {
            let rendered = render_tool_calls(&calls, format, &definitions()).unwrap();
            assert_eq!(
                parse_tool_calls(&rendered, format, &definitions()).unwrap(),
                ParsedToolBlock::Complete(calls.clone())
            );
        }
        let calls = vec![call(serde_json::json!({"choice":value}))];
        let rendered = render_tool_calls(&calls, ToolCallFormat::XmlTyped, &definitions()).unwrap();
        assert!(rendered.contains("<ifm|arg_type>number</ifm|arg_type>"));
        assert_eq!(
            parse_tool_calls(&rendered, ToolCallFormat::XmlTyped, &definitions()).unwrap(),
            ParsedToolBlock::Complete(calls)
        );
    }
    let block = |text| {
        format!(
            "<ifm|tool_calls><ifm|tool_call>lookup\n<ifm|arg_key>count</ifm|arg_key><ifm|arg_value>{text}</ifm|arg_value></ifm|tool_call></ifm|tool_calls>"
        )
    };
    assert!(matches!(
        parse_tool_calls(&block("1e3"), ToolCallFormat::Xml, &definitions()).unwrap(),
        ParsedToolBlock::Complete(_)
    ));
    for fraction in [
        "3.0000000000000001",
        "1000000000000000000000001.1",
        "1e-400",
    ] {
        assert!(
            parse_tool_calls(&block(fraction), ToolCallFormat::Xml, &definitions()).is_err(),
            "{fraction}"
        );
    }
}
