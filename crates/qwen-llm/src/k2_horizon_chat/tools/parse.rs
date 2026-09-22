//! Decode one terminal IFM tool block, never salvage a prefix of malformed output.
use super::*;

mod coerce;
pub(super) mod json_decode;
#[cfg(test)]
mod tests;

pub const TOOL_BLOCK_OPEN: &str = "<ifm|tool_calls>";
const BLOCK_CLOSE: &str = "</ifm|tool_calls>";
const CALL_OPEN: &str = "<ifm|tool_call>";
const CALL_CLOSE: &str = "</ifm|tool_call>";

#[derive(Clone, Debug, PartialEq)]
pub enum ParsedToolBlock {
    Complete(Vec<ToolCall>),
    Incomplete,
}

/// Interpret generated calls only after the reasoning partition has closed.
/// Definitions bind function names and XML outer-value types; this is not full
/// JSON Schema argument validation or authorization to execute a call.
pub fn parse_tool_calls(
    block: &str,
    format: ToolCallFormat,
    definitions: &[Value],
) -> Result<ParsedToolBlock> {
    super::presentation::check_input_depth(definitions)?;
    match parse(block, format, definitions) {
        Ok(calls) => Ok(ParsedToolBlock::Complete(calls)),
        Err(Failure::Incomplete) => Ok(ParsedToolBlock::Incomplete),
        Err(Failure::Malformed(error)) => Err(error),
    }
}

enum Failure {
    Incomplete,
    Malformed(super::super::ChatError),
}
impl From<super::super::ChatError> for Failure {
    fn from(value: super::super::ChatError) -> Self {
        Self::Malformed(value)
    }
}
type ParseResult<T> = std::result::Result<T, Failure>;
fn malformed(message: impl Into<String>) -> Failure {
    error(message).into()
}

struct Cursor<'a>(&'a str);
impl<'a> Cursor<'a> {
    fn whitespace(&mut self) {
        self.0 = self.0.trim_start_matches(|c: char| c.is_ascii_whitespace());
    }
    fn expect(&mut self, tag: &str) -> ParseResult<()> {
        if let Some(rest) = self.0.strip_prefix(tag) {
            self.0 = rest;
            Ok(())
        } else if tag.starts_with(self.0) {
            Err(Failure::Incomplete)
        } else {
            Err(malformed(format!("invalid tool block: expected {tag}")))
        }
    }
    fn tagged(&mut self, open: &str, close: &str) -> ParseResult<&'a str> {
        self.whitespace();
        self.expect(open)?;
        let end = self.0.find(close).ok_or(Failure::Incomplete)?;
        let value = &self.0[..end];
        self.0 = &self.0[end + close.len()..];
        Ok(value)
    }
}

fn definition<'a>(definitions: &'a [Value], name: &str) -> ParseResult<&'a Value> {
    let mut found = None;
    for tool in definitions {
        let function = tool.get("function").unwrap_or(tool);
        if function["name"] == name {
            if found.is_some() {
                return Err(malformed("ambiguous duplicate tool definition"));
            }
            found = Some(function);
        }
    }
    found.ok_or_else(|| malformed(format!("undeclared tool function {name:?}")))
}

fn parse(block: &str, format: ToolCallFormat, definitions: &[Value]) -> ParseResult<Vec<ToolCall>> {
    let mut cursor = Cursor(block);
    cursor.whitespace();
    cursor.expect(TOOL_BLOCK_OPEN)?;
    let mut calls = Vec::new();
    loop {
        cursor.whitespace();
        if cursor.0.starts_with(BLOCK_CLOSE) {
            cursor.expect(BLOCK_CLOSE)?;
            cursor.whitespace();
            if !cursor.0.is_empty() {
                return Err(malformed(
                    "tool block must be terminal; trailing output is invalid",
                ));
            }
            if calls.is_empty() {
                return Err(malformed("generated tool block contains no calls"));
            }
            return Ok(calls);
        }
        if BLOCK_CLOSE.starts_with(cursor.0) {
            return Err(Failure::Incomplete);
        }
        cursor.expect(CALL_OPEN)?;
        cursor.whitespace();
        let call = if format == ToolCallFormat::Json {
            let (value, bytes) = json_decode::prefix(cursor.0)?;
            cursor.0 = &cursor.0[bytes..];
            let Value::Object(mut fields) = value else {
                return Err(malformed("tool call must be an object"));
            };
            if fields.len() != 2 {
                return Err(malformed("tool call requires only name and arguments"));
            }
            let Some(Value::String(name)) = fields.remove("name") else {
                return Err(malformed("tool call requires a string name"));
            };
            let Some(Value::Object(arguments)) = fields.remove("arguments") else {
                return Err(malformed("tool call requires object arguments"));
            };
            let call = ToolCall { name, arguments };
            definition(definitions, &call.name)?;
            // Match the native encoder's finite binary64 floating-point contract.
            json::encode(&Value::Object(call.arguments.clone()))?;
            call
        } else {
            let end = cursor.0.find('\n').ok_or(Failure::Incomplete)?;
            let name = cursor.0[..end]
                .strip_suffix('\r')
                .unwrap_or(&cursor.0[..end]);
            if name.is_empty() || name.contains(['<', '>']) {
                return Err(malformed("invalid XML tool function name"));
            }
            let function = definition(definitions, name)?;
            cursor.0 = &cursor.0[end + 1..];
            let mut arguments = Map::new();
            loop {
                cursor.whitespace();
                if cursor.0.starts_with(CALL_CLOSE) {
                    break;
                }
                if CALL_CLOSE.starts_with(cursor.0) {
                    return Err(Failure::Incomplete);
                }
                let key = cursor.tagged("<ifm|arg_key>", "</ifm|arg_key>")?;
                if key.is_empty() || key.contains(['<', '>']) || arguments.contains_key(key) {
                    return Err(malformed("invalid or duplicate XML tool argument key"));
                }
                let declared_type = if format == ToolCallFormat::XmlTyped {
                    Some(cursor.tagged("<ifm|arg_type>", "</ifm|arg_type>")?)
                } else {
                    None
                };
                cursor.whitespace();
                cursor.expect("<ifm|arg_value>")?;
                let rest = cursor.0;
                let raw = if coerce::structured(function, key, declared_type)?
                    && rest.trim_start().starts_with(['{', '['])
                {
                    let (_, bytes) = json_decode::prefix(rest)?;
                    cursor.0 = &rest[bytes..];
                    cursor.whitespace();
                    cursor.expect("</ifm|arg_value>")?;
                    &rest[..bytes]
                } else {
                    let end = rest.find("</ifm|arg_value>").ok_or(Failure::Incomplete)?;
                    cursor.0 = &rest[end + "</ifm|arg_value>".len()..];
                    &rest[..end]
                };
                let value = coerce::argument(function, key, raw, declared_type)?;
                if let Some(label) = declared_type {
                    let expected = types::argument_type(definitions, name, key, &value)?;
                    if label != expected {
                        return Err(malformed(format!(
                            "XML argument type {label:?} differs from native type {expected:?}"
                        )));
                    }
                }
                arguments.insert(key.into(), value);
            }
            ToolCall {
                name: name.into(),
                arguments,
            }
        };
        cursor.whitespace();
        cursor.expect(CALL_CLOSE)?;
        calls.push(call);
    }
}
