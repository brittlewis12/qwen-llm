//! Emission-side parser for Qwen XML-parameter tool calls.
//!
//! Normative fixtures: `serve_tool_render_fixtures_v1.json`
//! (`qwen36_raw_echo_identity`, `malformed_corpus`). Frozen decisions it
//! implements:
//!
//! - **Raw-span retention:** the caller keeps the original emission bytes;
//!   parsing only *interprets* them into `function_call` items. Self-echo
//!   renders the retained bytes verbatim, so completed-checkpoint identity
//!   survives parsing (S0 F1).
//! - **No salvage:** any deviation after the first `<tool_call>` yields
//!   zero calls and the entire emission stays visible text; checkpoint
//!   capture proceeds safely.
//!
//! Grammar (template-oracle format instruction, "NO suffix"):
//! `visible-prose` then one-or-more `<tool_call>\n<function=NAME>\n`
//! (`<parameter=KEY>\nVALUE\n</parameter>\n`)* `</function>\n</tool_call>`
//! blocks separated by `\n`, ending the emission.

use serde_json::{Map, Value};

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParsedCall {
    pub(crate) name: String,
    pub(crate) arguments: Map<String, Value>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ParsedEmission {
    /// Prose before the first call block (whole emission when no valid
    /// calls exist). Byte-verbatim slice of the input.
    pub(crate) visible: String,
    pub(crate) calls: Vec<ParsedCall>,
}

const CALL_OPEN: &str = "<tool_call>\n";
const CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_OPEN: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>\n";
const PARAM_OPEN: &str = "<parameter=";
const PARAM_CLOSE: &str = "\n</parameter>\n";

/// Parse a complete visible emission span (post-reasoning split). Never
/// fails: malformed structure returns the whole span as visible text.
pub(crate) fn parse_emission(emission: &str) -> ParsedEmission {
    let Some(first_call) = emission.find("<tool_call>") else {
        return ParsedEmission {
            visible: emission.to_owned(),
            calls: Vec::new(),
        };
    };
    match parse_call_blocks(&emission[first_call..]) {
        Some(calls) if !calls.is_empty() => ParsedEmission {
            visible: emission[..first_call].to_owned(),
            calls,
        },
        _ => ParsedEmission {
            visible: emission.to_owned(),
            calls: Vec::new(),
        },
    }
}

fn parse_call_blocks(mut rest: &str) -> Option<Vec<ParsedCall>> {
    let mut calls = Vec::new();
    loop {
        rest = rest.strip_prefix(CALL_OPEN)?;
        rest = rest.strip_prefix(FUNCTION_OPEN)?;
        let name_end = rest.find(">\n")?;
        let name = &rest[..name_end];
        if name.is_empty() || name.contains('<') || name.contains('\n') {
            return None;
        }
        rest = &rest[name_end + 2..];

        let mut arguments = Map::new();
        while let Some(after_open) = rest.strip_prefix(PARAM_OPEN) {
            let key_end = after_open.find(">\n")?;
            let key = &after_open[..key_end];
            if key.is_empty() || key.contains('<') || key.contains('\n') {
                return None;
            }
            let value_region = &after_open[key_end + 2..];
            let value_end = value_region.find(PARAM_CLOSE)?;
            let raw_value = &value_region[..value_end];
            arguments.insert(key.to_owned(), decode_parameter_value(raw_value));
            rest = &value_region[value_end + PARAM_CLOSE.len()..];
        }

        rest = rest.strip_prefix(FUNCTION_CLOSE)?;
        rest = rest.strip_prefix(CALL_CLOSE)?;
        calls.push(ParsedCall {
            name: name.to_owned(),
            arguments,
        });
        if rest.is_empty() {
            return Some(calls);
        }
        // Blocks are separated by exactly one newline; anything else after
        // a completed block violates the NO-suffix contract.
        rest = rest.strip_prefix('\n')?;
        if rest.is_empty() {
            // Trailing newline after the final block is tolerated: the
            // decode loop's terminal boundary may land there.
            return Some(calls);
        }
    }
}

/// Parameter values round-trip through the structured render as compact
/// JSON for mappings/sequences/numbers/booleans and raw text otherwise
/// (fixture `qwen36_mapping_parameter_renders_compact_json`). JSON string
/// literals stay raw text so quoted prose is never silently unwrapped.
fn decode_parameter_value(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(parsed @ (Value::Object(_) | Value::Array(_) | Value::Number(_) | Value::Bool(_))) => {
            parsed
        }
        _ => Value::String(raw.to_owned()),
    }
}

/// Structured re-render of parsed calls (normalized template glue; the
/// verbatim path renders retained raw bytes instead). Fixture:
/// `qwen36_assistant_*` cases.
/// Serialize a JSON value the way Python's `json.dumps(ensure_ascii=False)`
/// does with default separators (`", "` and `": "`), which is what both
/// Transformers' and llama.cpp's `tojson` filters emit. Key order is kept.
pub(crate) fn python_json(value: &Value) -> String {
    fn write(value: &Value, out: &mut String) {
        match value {
            Value::Object(map) => {
                out.push('{');
                for (index, (key, item)) in map.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(&serde_json::to_string(key).expect("serialize key"));
                    out.push_str(": ");
                    write(item, out);
                }
                out.push('}');
            }
            Value::Array(items) => {
                out.push('[');
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    write(item, out);
                }
                out.push(']');
            }
            scalar => out.push_str(&serde_json::to_string(scalar).expect("serialize scalar")),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// Render one parameter value as the released Qwen templates do: mappings and
/// sequences through `tojson`, everything else through Jinja's `string`
/// filter (Python `str()`: `True`/`False`/`None`, numbers verbatim, strings
/// as-is).
fn python_parameter_value(value: &Value) -> String {
    match value {
        Value::Object(_) | Value::Array(_) => python_json(value),
        Value::String(text) => text.clone(),
        Value::Bool(true) => "True".into(),
        Value::Bool(false) => "False".into(),
        Value::Null => "None".into(),
        Value::Number(number) => number.to_string(),
    }
}

/// Legacy unpinned rendering: compact JSON for non-string values.
fn compact_parameter_value(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        other => serde_json::to_string(other).expect("serialize value"),
    }
}

pub(crate) fn render_calls(visible: &str, calls: &[ParsedCall]) -> String {
    render_calls_for(visible, calls, false)
}

/// Render assistant tool calls in the XML-parameter form. `released` selects
/// the pinned templates' Python value semantics; `false` keeps the legacy
/// compact form frozen in `serve_tool_render_fixtures_v1.json`.
pub(crate) fn render_calls_for(visible: &str, calls: &[ParsedCall], released: bool) -> String {
    let mut output = String::from(visible);
    for (index, call) in calls.iter().enumerate() {
        if index == 0 {
            if !visible.trim().is_empty() {
                output.push_str("\n\n");
            }
        } else {
            output.push('\n');
        }
        output.push_str(CALL_OPEN);
        output.push_str(FUNCTION_OPEN);
        output.push_str(&call.name);
        output.push_str(">\n");
        for (key, value) in &call.arguments {
            output.push_str(PARAM_OPEN);
            output.push_str(key);
            output.push_str(">\n");
            output.push_str(&if released {
                python_parameter_value(value)
            } else {
                compact_parameter_value(value)
            });
            output.push_str(PARAM_CLOSE);
        }
        output.push_str(FUNCTION_CLOSE);
        output.push_str(CALL_CLOSE);
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn fixture() -> Value {
        serde_json::from_str(include_str!(
            "../../tests/fixtures/serve_tool_render_fixtures_v1.json"
        ))
        .expect("parse tool render fixtures")
    }

    fn case(name: &str) -> Value {
        fixture()["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["name"] == name)
            .unwrap_or_else(|| panic!("missing case {name}"))
            .clone()
    }

    #[test]
    fn raw_echo_identity_case_parses_and_reconstructs_byte_exact() {
        let case = case("qwen36_raw_echo_identity");
        let emission = case["emission"].as_str().unwrap();
        // The reasoning split runs first in the pipeline.
        let split = crate::open_responses::render::split_reasoning(emission);
        assert_eq!(split.reasoning, case["expect_parse"]["reasoning"].as_str(),);
        let parsed = parse_emission(split.visible);
        assert_eq!(
            parsed.visible,
            case["expect_parse"]["visible_before_calls"]
                .as_str()
                .unwrap(),
        );
        assert_eq!(parsed.calls.len(), 1);
        assert_eq!(parsed.calls[0].name, "fs_list");
        assert_eq!(parsed.calls[0].arguments["path"], json!("/tmp"));
        // Byte-exact reconstruction: think block + structured render.
        let reconstructed = format!(
            "<think>{}</think>{}",
            split.reasoning.unwrap(),
            render_calls(parsed.visible.trim_end_matches('\n'), &parsed.calls),
        );
        // The fixture emission carries visible "\n\nListing now.\n\n" whose
        // trailing separator is the structured glue; verify verbatim path
        // instead: retained raw bytes are the identity guarantee.
        assert_eq!(
            format!(
                "<think>{}</think>{}",
                split.reasoning.unwrap(),
                split.visible
            ),
            emission,
            "verbatim retention must reproduce the emission"
        );
        // And the structured path reproduces it when glue conventions match.
        assert_eq!(reconstructed, emission);
    }

    #[test]
    fn malformed_corpus_never_salvages() {
        let case = case("malformed_corpus");
        for entry in case["corpus"].as_array().unwrap() {
            let emission = entry["emission"].as_str().unwrap();
            let parsed = parse_emission(emission);
            assert!(
                parsed.calls.is_empty(),
                "salvaged calls from defect {:?}",
                entry["defect"]
            );
            assert_eq!(
                parsed.visible, emission,
                "visible must be the full emission for defect {:?}",
                entry["defect"]
            );
        }
    }

    #[test]
    fn structured_render_matches_frozen_turn_fixtures() {
        for name in [
            "qwen36_assistant_single_call_with_content",
            "qwen36_assistant_call_no_content",
            "qwen36_assistant_parallel_calls_sequential_items",
            "qwen36_mapping_parameter_renders_compact_json",
        ] {
            let case = case(name);
            let turn = &case["turn"];
            let calls: Vec<ParsedCall> = turn["calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|call| ParsedCall {
                    name: call["name"].as_str().unwrap().to_owned(),
                    arguments: call["arguments"].as_object().unwrap().clone(),
                })
                .collect();
            let body = render_calls(turn["visible"].as_str().unwrap(), &calls);
            let rendered = match turn["reasoning"].as_str() {
                Some(reasoning) => {
                    format!("<|im_start|>assistant\n<think>{reasoning}</think>{body}<|im_end|>\n")
                }
                None => format!("<|im_start|>assistant\n{body}<|im_end|>\n"),
            };
            assert_eq!(
                rendered,
                case["rendered_turn"].as_str().unwrap(),
                "structured render diverged for {name}"
            );
        }
    }

    #[test]
    fn parallel_blocks_and_scalar_decoding_round_trip() {
        let emission = "<tool_call>\n<function=alpha>\n<parameter=count>\n2\n</parameter>\n<parameter=flag>\ntrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=beta>\n</function>\n</tool_call>";
        let parsed = parse_emission(emission);
        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[0].arguments["count"], json!(2));
        assert_eq!(parsed.calls[0].arguments["flag"], json!(true));
        assert!(parsed.calls[1].arguments.is_empty());
        assert_eq!(render_calls("", &parsed.calls), emission);

        // Quoted JSON strings stay raw text (never unwrapped).
        let quoted = "<tool_call>\n<function=echo>\n<parameter=text>\n\"quoted\"\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_emission(quoted);
        assert_eq!(parsed.calls[0].arguments["text"], json!("\"quoted\""));
        assert_eq!(render_calls("", &parsed.calls), quoted);

        // Multiline values survive.
        let multiline = "<tool_call>\n<function=fs_write>\n<parameter=content>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
        let parsed = parse_emission(multiline);
        assert_eq!(
            parsed.calls[0].arguments["content"],
            json!("line one\nline two")
        );
        assert_eq!(render_calls("", &parsed.calls), multiline);
    }

    #[test]
    fn trailing_newline_after_final_block_is_tolerated() {
        let emission = "<tool_call>\n<function=alpha>\n</function>\n</tool_call>\n";
        let parsed = parse_emission(emission);
        assert_eq!(parsed.calls.len(), 1);
    }
}
