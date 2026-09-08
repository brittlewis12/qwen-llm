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

/// Parameter values decode as JSON for mappings/sequences/numbers/booleans
/// and raw text otherwise; the template's argument style re-renders them
/// (`ArgumentStyle`). JSON string literals stay raw text so quoted prose is
/// never silently unwrapped.
fn decode_parameter_value(raw: &str) -> Value {
    match serde_json::from_str::<Value>(raw) {
        Ok(parsed @ (Value::Object(_) | Value::Array(_) | Value::Number(_) | Value::Bool(_))) => {
            parsed
        }
        _ => Value::String(raw.to_owned()),
    }
}

/// Python's `repr(float)` for a JSON number: shortest round-trip digits,
/// fixed notation for exponents in `-4..16`, otherwise `d.ddde±XX` with a
/// two-digit signed exponent, and always at least one fractional digit.
/// Integer-looking JSON numbers keep their text (Python parses them as
/// `int`).
pub(crate) fn python_number(number: &serde_json::Number) -> String {
    let text = number.to_string();
    if !text.contains(['.', 'e', 'E']) {
        return text;
    }
    let Ok(value) = text.parse::<f64>() else {
        return text;
    };
    if !value.is_finite() {
        return text;
    }
    if value == 0.0 {
        return if value.is_sign_negative() {
            "-0.0".into()
        } else {
            "0.0".into()
        };
    }
    // Shortest round-trip digits and decimal exponent from Rust's `{:e}`.
    let sci = format!("{value:e}");
    let (mantissa, exponent) = sci.split_once('e').expect("scientific form");
    let exponent: i32 = exponent.parse().expect("exponent");
    let negative = mantissa.starts_with('-');
    let digits: String = mantissa.chars().filter(|c| c.is_ascii_digit()).collect();
    let mut out = String::new();
    if negative {
        out.push('-');
    }
    if (-4..16).contains(&exponent) {
        if exponent >= 0 {
            let int_len = exponent as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.push_str(&"0".repeat(int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.push_str(&"0".repeat((-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        out.push_str(&format!("{:02}", exponent.abs()));
    }
    out
}

/// Serialize a JSON value the way Python's `json.dumps(ensure_ascii=False)`
/// does with default separators (`", "` and `": "`), which is what both
/// Transformers' and llama.cpp's `tojson` filters emit. Key order is kept;
/// numbers follow Python float repr.
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
            Value::Number(number) => out.push_str(&python_number(number)),
            scalar => out.push_str(&serde_json::to_string(scalar).expect("serialize scalar")),
        }
    }
    let mut out = String::new();
    write(value, &mut out);
    out
}

/// How a released template stringifies non-string call arguments.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ArgumentStyle {
    /// Qwen3.6: `tojson` for mappings/sequences, Jinja `string` (Python
    /// `str()`: `True`/`False`/`None`) for other scalars.
    PythonStr,
    /// Qwen3.5 and Qwen3.8: `tojson` for every non-string value.
    ToJson,
}

fn parameter_value(value: &Value, style: ArgumentStyle) -> String {
    match (style, value) {
        (_, Value::String(text)) => text.clone(),
        (ArgumentStyle::ToJson, other) => python_json(other),
        (ArgumentStyle::PythonStr, Value::Object(_) | Value::Array(_)) => python_json(value),
        (ArgumentStyle::PythonStr, Value::Bool(true)) => "True".into(),
        (ArgumentStyle::PythonStr, Value::Bool(false)) => "False".into(),
        (ArgumentStyle::PythonStr, Value::Null) => "None".into(),
        (ArgumentStyle::PythonStr, Value::Number(number)) => python_number(number),
    }
}

/// Structured re-render of parsed calls under the `tojson` argument rule
/// (Qwen3.5/3.8), where a parsed JSON scalar re-renders as the bytes it was
/// decoded from. Fixture: `qwen36_assistant_*` cases (string arguments,
/// style-independent).
#[cfg(test)]
fn render_calls(visible: &str, calls: &[ParsedCall]) -> String {
    render_calls_for(visible, calls, ArgumentStyle::ToJson)
}

/// Render assistant tool calls in the XML-parameter form with the argument
/// stringification of the given template family.
pub(crate) fn render_calls_for(
    visible: &str,
    calls: &[ParsedCall],
    style: ArgumentStyle,
) -> String {
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
            output.push_str(&parameter_value(value, style));
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

    #[test]
    fn python_number_matches_python_repr() {
        for (json, expected) in [
            ("1e10", "10000000000.0"),
            ("1e-7", "1e-07"),
            ("1.0", "1.0"),
            ("1.00", "1.0"),
            ("-0.0", "-0.0"),
            ("0.0", "0.0"),
            ("2.5", "2.5"),
            ("-2.5e16", "-2.5e+16"),
            ("1e16", "1e+16"),
            ("123456789012345.6", "123456789012345.6"),
            ("0.001", "0.001"),
            ("0.0001", "0.0001"),
            ("0.00001", "1e-05"),
            ("12345678901234567890", "12345678901234567890"),
            ("42", "42"),
        ] {
            let value: Value = serde_json::from_str(json).unwrap();
            let Value::Number(number) = value else {
                panic!()
            };
            assert_eq!(python_number(&number), expected, "{json}");
        }
    }

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

/// DeepSeek V4 DSML tool calls.
///
/// Grammar (vLLM `rust/src/parser/src/tool/deepseek_dsml` at the pinned
/// revision, whose event model this mirrors): visible prose runs until
/// `<｜DSML｜tool_calls>`; inside the block each complete
/// `<｜DSML｜invoke name="…">` … `</｜DSML｜invoke>` yields one call whose
/// `<｜DSML｜parameter name="…" string="true|false">…</｜DSML｜parameter>`
/// entries become a JSON object — `string="true"` values verbatim,
/// `string="false"` values parsed as JSON (falling back to the raw text
/// when they are not JSON); anything after `</｜DSML｜tool_calls>` is
/// ignored. Like `parse_emission`, a malformed block returns the whole span
/// as visible text rather than failing.
pub(crate) const DSML_TOOL_CALLS_OPEN: &str = "<｜DSML｜tool_calls>";
const DSML_TOOL_CALLS_CLOSE: &str = "</｜DSML｜tool_calls>";
const DSML_INVOKE_OPEN: &str = "<｜DSML｜invoke name=\"";
const DSML_INVOKE_CLOSE: &str = "</｜DSML｜invoke>";
const DSML_PARAMETER_OPEN: &str = "<｜DSML｜parameter name=\"";
const DSML_PARAMETER_CLOSE: &str = "</｜DSML｜parameter>";

pub(crate) fn parse_dsml_emission(emission: &str) -> ParsedEmission {
    let Some(first_call) = emission.find(DSML_TOOL_CALLS_OPEN) else {
        return ParsedEmission {
            visible: emission.to_owned(),
            calls: Vec::new(),
        };
    };
    match parse_dsml_block(&emission[first_call + DSML_TOOL_CALLS_OPEN.len()..]) {
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

fn parse_dsml_block(mut rest: &str) -> Option<Vec<ParsedCall>> {
    let mut calls = Vec::new();
    loop {
        rest = rest.trim_start();
        if let Some(_after) = rest.strip_prefix(DSML_TOOL_CALLS_CLOSE) {
            // Trailing text after the block is ignored (reference: IgnoredRest).
            return Some(calls);
        }
        let after_open = rest.strip_prefix(DSML_INVOKE_OPEN)?;
        let name_end = after_open.find("\">")?;
        let name = &after_open[..name_end];
        if name.is_empty() || name.contains('<') || name.contains('\n') {
            return None;
        }
        let body_region = &after_open[name_end + 2..];
        let body_end = body_region.find(DSML_INVOKE_CLOSE)?;
        let arguments = parse_dsml_parameters(&body_region[..body_end])?;
        calls.push(ParsedCall {
            name: name.to_owned(),
            arguments,
        });
        rest = &body_region[body_end + DSML_INVOKE_CLOSE.len()..];
    }
}

fn parse_dsml_parameters(mut body: &str) -> Option<Map<String, Value>> {
    let mut arguments = Map::new();
    loop {
        body = body.trim_start();
        if body.is_empty() {
            return Some(arguments);
        }
        let after_open = body.strip_prefix(DSML_PARAMETER_OPEN)?;
        let key_end = after_open.find('"')?;
        let key = &after_open[..key_end];
        if key.is_empty() || key.contains('<') || key.contains('\n') {
            return None;
        }
        let attrs = after_open[key_end + 1..].strip_prefix(" string=\"")?;
        let is_string = if let Some(rest) = attrs.strip_prefix("true\">") {
            body = rest;
            true
        } else if let Some(rest) = attrs.strip_prefix("false\">") {
            body = rest;
            false
        } else {
            return None;
        };
        let value_end = body.find(DSML_PARAMETER_CLOSE)?;
        let raw = &body[..value_end];
        let value = if is_string {
            Value::String(raw.to_owned())
        } else {
            serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_owned()))
        };
        arguments.insert(key.to_owned(), value);
        body = &body[value_end + DSML_PARAMETER_CLOSE.len()..];
    }
}

#[cfg(test)]
mod dsml_tests {
    use super::*;

    const BLOCK: &str = "Checking.\n\n<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"get_weather\">\n<｜DSML｜parameter name=\"city\" string=\"true\">Paris</｜DSML｜parameter>\n<｜DSML｜parameter name=\"units\" string=\"true\">c</｜DSML｜parameter>\n</｜DSML｜invoke>\n<｜DSML｜invoke name=\"lookup\">\n<｜DSML｜parameter name=\"ids\" string=\"false\">[1, 2]</｜DSML｜parameter>\n<｜DSML｜parameter name=\"limit\" string=\"false\">10</｜DSML｜parameter>\n<｜DSML｜parameter name=\"exact\" string=\"false\">true</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";

    #[test]
    fn parses_string_and_json_parameters_with_prose_prefix() {
        let parsed = parse_dsml_emission(BLOCK);
        assert_eq!(parsed.visible, "Checking.\n\n");
        assert_eq!(parsed.calls.len(), 2);
        assert_eq!(parsed.calls[0].name, "get_weather");
        assert_eq!(parsed.calls[0].arguments["city"], "Paris");
        assert_eq!(parsed.calls[1].name, "lookup");
        assert_eq!(parsed.calls[1].arguments["ids"], serde_json::json!([1, 2]));
        assert_eq!(parsed.calls[1].arguments["limit"], 10);
        assert_eq!(parsed.calls[1].arguments["exact"], true);
    }

    #[test]
    fn string_parameters_are_verbatim_even_when_they_look_like_json() {
        let text = "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"q\" string=\"true\">[1, 2]</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>";
        let parsed = parse_dsml_emission(text);
        assert_eq!(parsed.calls[0].arguments["q"], "[1, 2]");
    }

    #[test]
    fn trailing_text_after_the_block_is_ignored_and_no_block_is_all_visible() {
        let parsed = parse_dsml_emission(&format!("{BLOCK}\nstray"));
        assert_eq!(parsed.calls.len(), 2);
        let parsed = parse_dsml_emission("plain answer");
        assert_eq!(parsed.visible, "plain answer");
        assert!(parsed.calls.is_empty());
    }

    #[test]
    fn malformed_blocks_fall_through_as_visible_text() {
        for text in [
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"q\" string=\"maybe\">x</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\nnot a parameter\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n",
            "<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>",
        ] {
            let parsed = parse_dsml_emission(text);
            assert_eq!(parsed.visible, text, "{text}");
            assert!(parsed.calls.is_empty());
        }
    }
}
