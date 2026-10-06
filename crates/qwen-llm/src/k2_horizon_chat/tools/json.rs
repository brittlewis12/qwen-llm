//! HF's tojson uses Python json.dumps(ensure_ascii=False), including spaces and
//! Python float exponent formatting. Ordinary compact serde JSON is not identical.
//! The encoder is family-neutral: [`crate::tool_schema::python_json`].
use super::*;

pub(super) fn encode(value: &Value) -> Result<String> {
    crate::tool_schema::python_json(value).map_err(error)
}
