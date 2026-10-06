//! Build containers directly: serde Value's arbitrary-precision visitor treats
//! a literal "$serde_json::private::Number" object key as an internal number tag.
//! Serde still owns scalar JSON syntax, escapes and numeric parsing. The
//! decoder is family-neutral: [`crate::tool_schema::decode_json_prefix`].
use super::*;
use crate::tool_schema::JsonDecodeError;

pub(super) fn prefix(text: &str) -> ParseResult<(Value, usize)> {
    crate::tool_schema::decode_json_prefix(text).map_err(|failure| match failure {
        JsonDecodeError::Incomplete => Failure::Incomplete,
        JsonDecodeError::Malformed(message) => malformed(message),
    })
}
pub(in crate::k2_horizon_chat::tools) fn complete(text: &str) -> Result<Value> {
    crate::tool_schema::decode_json(text).map_err(error)
}
