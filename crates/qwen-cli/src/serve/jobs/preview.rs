//! Bounded display excerpts from immutable authored input, never rendered prompts.

use serde::Serialize;
use serde_json::Value;

const MAX_PREVIEW_SCALARS: usize = 240;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(crate) struct RequestPreview {
    pub(crate) message_index: usize,
    pub(crate) message_count: usize,
    pub(crate) text: String,
    pub(crate) truncated: bool,
}

pub(super) fn request_preview(request: &Value) -> Option<RequestPreview> {
    if request.get("schema_version")?.as_u64()? != 1
        || request.pointer("/input/kind")?.as_str()? != "messages"
    {
        return None;
    }
    let messages = request.pointer("/input/messages")?.as_array()?;
    let (message_index, message) = messages
        .iter()
        .enumerate()
        .rev()
        .find(|(_, message)| message.get("role").and_then(Value::as_str) == Some("user"))?;
    let mut chars = message.get("content")?.as_str()?.chars();
    let text = chars.by_ref().take(MAX_PREVIEW_SCALARS).collect();
    Some(RequestPreview {
        message_index,
        message_count: messages.len(),
        text,
        truncated: chars.next().is_some(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn request(content: Value) -> Value {
        json!({"schema_version":1,"input":{"kind":"messages","messages":[
            {"role":"system","content":"not the preview"},
            {"role":"user","content":"not the latest user"},
            {"role":"assistant","content":"not the preview","reasoning":"private"},
            {"role":"user","content":content}
        ],"assistant_prefill":{"text":"not the preview"}}})
    }

    #[test]
    fn preserves_authored_text_and_never_substitutes_other_content() {
        let authored: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/lens_http_v1/request.json"
        ))
        .unwrap();
        let fixture: Value = serde_json::from_str(include_str!(
            "../../../tests/fixtures/lens_http_v1/request_preview.json"
        ))
        .unwrap();
        assert_eq!(
            serde_json::to_value(request_preview(&authored).unwrap()).unwrap(),
            fixture
        );
        for text in ["", " \n\t ", "<script>not markup</script>"] {
            assert_eq!(
                request_preview(&request(json!(text))),
                Some(RequestPreview {
                    message_index: 3,
                    message_count: 4,
                    text: text.into(),
                    truncated: false,
                })
            );
        }
        for content in [Value::Null, json!([{"text":"unsupported"}]), json!(4)] {
            assert_eq!(request_preview(&request(content)), None);
        }
        let mut value = request(json!("last"));
        value["input"]["messages"][3]
            .as_object_mut()
            .unwrap()
            .remove("content");
        assert_eq!(request_preview(&value), None);
        value["input"]["messages"] = json!([{"role":"system","content":"only system"}]);
        assert_eq!(request_preview(&value), None);
        assert_eq!(request_preview(&json!({})), None);
        value["schema_version"] = 2.into();
        assert_eq!(request_preview(&value), None);
    }

    #[test]
    fn bounds_unicode_scalars_without_splitting_utf8_or_hiding_truncation() {
        for scalar in ['a', '\u{1f680}', '\u{0000}'] {
            for count in [239, 240, 241, 100_000] {
                let text: String = std::iter::repeat_n(scalar, count).collect();
                let preview = request_preview(&request(json!(text))).unwrap();
                assert_eq!(preview.text.chars().count(), count.min(240));
                assert_eq!(preview.truncated, count > 240);
                assert!(preview.text.len() <= 960);
                assert!(serde_json::to_vec(&preview).unwrap().len() < 1600);
            }
        }
    }
}
