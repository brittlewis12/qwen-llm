//! Pinned IFM final-checkpoint no-tools template. Raw checkpoints remain separate.
use crate::gguf::GgufFile;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

pub mod tools;

pub const REVISION: &str = "2c9659a84c4eea6f9f60462221fe762c8c84d75c";
pub const TEMPLATE_SHA256: &str =
    "a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590";
pub const RENDERER: &str = "k2_horizon_ifm_no_tools_v1";
pub const CHAT_STOPS: [i32; 2] = [1, 250019];
const CHAT_MARKERS: [&str; 8] = [
    "<|ifm|im_start|>",
    "<|ifm|im_end|>",
    "<ifm|think>",
    "</ifm|think>",
    "<ifm|think_fast>",
    "</ifm|think_fast>",
    "<ifm|think_faster>",
    "</ifm|think_faster>",
];
pub const GGUF_TEMPLATE_SHA256: &str =
    "f6e3cd6dbf0f95016fff531f41f921dee541025733c14580a881cf5a5f9fa750";
pub const GENERATION_CONFIG_SHA256: &str =
    "2da7d47641f4509da4ae47711e31d8b5f0f3f801ee08d87e7e9f07f814bdc4a3";

#[derive(Debug, thiserror::Error)]
#[error("K2 chat: {0}")]
pub struct ChatError(String);
type Result<T> = std::result::Result<T, ChatError>;
fn error(message: impl Into<String>) -> ChatError {
    ChatError(message.into())
}

fn ensure_chat_markers(
    encode: impl FnMut(&str) -> std::result::Result<Vec<i32>, String>,
) -> Result<()> {
    let markers = CHAT_MARKERS.map(|marker| (marker, None));
    crate::tokenizer::require_single_token_markers(&markers, encode)
        .map_err(|reason| error(format!("{reason}; use raw input")))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Effort {
    High,
    Medium,
    Low,
}

impl Effort {
    pub fn parse(value: Option<&str>) -> Result<Self> {
        match value.unwrap_or("high") {
            "high" => Ok(Self::High),
            "medium" => Ok(Self::Medium),
            "low" => Ok(Self::Low),
            _ => Err(error(
                "reasoning effort must be high, medium, or low; no non-thinking mode is released",
            )),
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::High => "ifm|think",
            Self::Medium => "ifm|think_fast",
            Self::Low => "ifm|think_faster",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChatInput {
    pub messages: Vec<Message>,
    pub effort: Effort,
}

fn present_string<'de, D: Deserializer<'de>>(
    d: D,
) -> std::result::Result<Option<String>, D::Error> {
    String::deserialize(d).map(Some)
}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Message {
    pub role: String,
    pub content: String,
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub think: Option<String>,
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub think_fast: Option<String>,
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub think_faster: Option<String>,
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning_content: Option<String>,
    #[serde(
        default,
        deserialize_with = "present_string",
        skip_serializing_if = "Option::is_none"
    )]
    pub reasoning: Option<String>,
}

impl Message {
    pub fn text(role: &str, content: String) -> Self {
        Self {
            role: role.into(),
            content,
            think: None,
            think_fast: None,
            think_faster: None,
            reasoning_content: None,
            reasoning: None,
        }
    }
    fn thinking(&self) -> Option<(&str, &str)> {
        [
            (&self.think, "ifm|think"),
            (&self.think_fast, "ifm|think_fast"),
            (&self.think_faster, "ifm|think_faster"),
            (&self.reasoning_content, "ifm|think"),
            (&self.reasoning, "ifm|think"),
        ]
        .into_iter()
        .find_map(|(text, tag)| text.as_deref().map(|text| (tag, text)))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Wrapper {
    messages: Vec<Message>,
}
#[derive(Deserialize)]
#[serde(untagged)]
enum Document {
    Array(Vec<Message>),
    Object(Wrapper),
}

pub fn parse_messages(bytes: &[u8]) -> Result<Vec<Message>> {
    match serde_json::from_slice(bytes)
        .map_err(|e| error(format!("invalid no-tools message document: {e}")))?
    {
        Document::Array(messages) => Ok(messages),
        Document::Object(wrapper) => Ok(wrapper.messages),
    }
}

/// Returns the upstream bytes after its one leading BOS. The caller enables
/// native automatic BOS; authored marker-like content is preserved, never deduped.
pub fn render(messages: &[Message], effort: Effort) -> Result<String> {
    if messages.last().is_none_or(|m| m.role != "user") {
        return Err(error(
            "no-tools generation requires a nonempty conversation ending in a user turn",
        ));
    }
    let mut out = String::new();
    for (index, message) in messages.iter().enumerate() {
        out.push_str(&render_message_body(message, index)?);
        out.push_str("<|ifm|im_end|>");
    }
    out.push_str("<|ifm|im_start|>assistant\n<");
    out.push_str(effort.tag());
    out.push_str(">\n");
    Ok(out)
}

fn render_message_body(message: &Message, index: usize) -> Result<String> {
    let mut out = String::new();
    match message.role.as_str() {
        "system" if index != 0 => {
            return Err(error("only one leading system message is supported"));
        }
        "system" | "user" => {
            if message.thinking().is_some() {
                return Err(error("thinking fields belong only to assistant history"));
            }
            out.push_str("<|ifm|im_start|>");
            out.push_str(&message.role);
            out.push('\n');
            out.push_str(&message.content);
        }
        "assistant" => {
            let (tag, thinking) = message.thinking().ok_or_else(|| {
                error(
                    "assistant history requires an explicit string thinking field (empty is valid)",
                )
            })?;
            // Jinja preserves the newline before its generation block.
            out.push_str("<|ifm|im_start|>assistant\n<");
            out.push_str(tag);
            out.push_str(">\n");
            out.push_str(thinking);
            out.push_str("</");
            out.push_str(tag);
            out.push('>');
            out.push_str(&message.content);
        }
        _ => {
            return Err(error(
                "only string system/user/assistant turns are supported; tools and developer roles are unavailable",
            ));
        }
    }
    Ok(out)
}

#[derive(Debug, Serialize)]
pub struct VerifiedChatProfile {
    renderer: &'static str,
    source_revision: &'static str,
    template_sha256: &'static str,
    gguf_template_sha256: &'static str,
    generation_config_sha256: &'static str,
    reference_policy: &'static str,
}

/// Chat eligibility depends on the native markers, stops, and embedded
/// renderer template. Raw input remains available for other checkpoints.
pub fn verify_profile(source: &GgufFile) -> Result<VerifiedChatProfile> {
    crate::k2_horizon::K2HorizonModel::from_gguf(source).map_err(|e| error(e.to_string()))?;
    let tokenizer = crate::tokenizer::NativeTokenizer::from_gguf(source)
        .map_err(|e| error(format!("chat markers unavailable: {e}")))?;
    ensure_chat_markers(|marker| tokenizer.encode(marker, false).map_err(|e| e.to_string()))?;
    if tokenizer.eos() != Some(CHAT_STOPS[0])
        || tokenizer
            .encode("<|ifm|im_end|>", false)
            .map_err(|e| error(e.to_string()))?
            != [CHAT_STOPS[1]]
    {
        return Err(error(format!(
            "chat stop tokens {:?} are unavailable; use raw input",
            CHAT_STOPS
        )));
    }
    let template = source
        .get_str("tokenizer.chat_template")
        .ok_or_else(|| error("chat template unavailable; use raw input"))?;
    let template_digest = format!("{:x}", Sha256::digest(template.as_bytes()));
    if template_digest != GGUF_TEMPLATE_SHA256 {
        return Err(error(
            "chat template differs from pinned K2 renderer format; use raw input",
        ));
    }
    Ok(VerifiedChatProfile {
        renderer: RENDERER,
        source_revision: REVISION,
        template_sha256: TEMPLATE_SHA256,
        gguf_template_sha256: GGUF_TEMPLATE_SHA256,
        generation_config_sha256: GENERATION_CONFIG_SHA256,
        reference_policy: "template_and_generation_config_from_pinned_upstream_revision_publisher_declares_conversion_source",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/k2_chat_hf.json")).unwrap()
    }
    #[test]
    fn k2_chat_marker_gate_rejects_missing_and_split_markers_while_raw_input_works() {
        let missing = ensure_chat_markers(|marker| {
            Ok(if marker == CHAT_MARKERS[0] {
                vec![]
            } else {
                vec![7]
            })
        });
        assert!(
            missing
                .unwrap_err()
                .to_string()
                .contains("expected one token")
        );
        let split = ensure_chat_markers(|marker| {
            Ok(if marker == CHAT_MARKERS[1] {
                vec![7, 8]
            } else {
                vec![7]
            })
        });
        assert!(
            split
                .unwrap_err()
                .to_string()
                .contains("expected one token")
        );
        assert!(render(&[Message::text("user", "raw works".into())], Effort::High).is_ok());
    }
    #[test]
    fn k2_chat_bytes_match_pinned_upstream_no_tools_template() {
        let f = fixture();
        assert_eq!(f["revision"], REVISION);
        assert_eq!(f["template_sha256"], TEMPLATE_SHA256);
        assert_eq!(f["generation_config_sha256"], GENERATION_CONFIG_SHA256);
        assert_eq!(
            f["generation_config"]["eos_token_id"],
            serde_json::json!(CHAT_STOPS)
        );
        for case in f["cases"].as_array().unwrap() {
            let result = (|| {
                let messages = parse_messages(&serde_json::to_vec(&case["messages"]).unwrap())?;
                render(&messages, Effort::parse(case["reasoning_effort"].as_str())?)
            })();
            if case.get("error").is_some() {
                assert!(result.is_err(), "{}", case["name"]);
            } else {
                assert_eq!(
                    result.unwrap(),
                    case["body"].as_str().unwrap(),
                    "{}",
                    case["name"]
                );
            }
        }
    }
    #[test]
    fn k2_chat_rejects_unsupported_presence_and_ambiguous_documents() {
        for doc in [
            r#"{"messages":[],"tools":[]}"#,
            r#"[{"role":"user","content":"x","tools":null}]"#,
            r#"[{"role":"user","role":"assistant","content":"x"}]"#,
            r#"[{"role":"assistant","content":"x","think":null,"reasoning":"y"}]"#,
            r#"[{"role":"user","content":[{"type":"text","text":"x"}]}]"#,
        ] {
            assert!(parse_messages(doc.as_bytes()).is_err(), "{doc}");
        }
        for doc in [
            "[]",
            r#"[{"role":"developer","content":"x"},{"role":"user","content":"y"}]"#,
            r#"[{"role":"system","content":"x"},{"role":"system","content":"y"},{"role":"user","content":"z"}]"#,
        ] {
            assert!(render(&parse_messages(doc.as_bytes()).unwrap(), Effort::High).is_err());
        }
    }
    #[test]
    fn cpu_k2_chat_artifact_structure_and_native_tokens() {
        for file_fixture in [
            crate::test_fixtures::catalog::K2_HORIZON_7B_Q8_0,
            crate::test_fixtures::catalog::K2_HORIZON_7B_Q4_K_M,
        ] {
            let Some(path) = file_fixture.path_or_skip() else {
                continue;
            };
            if path.file_name().and_then(|name| name.to_str()) != Some(file_fixture.filename()) {
                eprintln!(
                    "skipping: fixture alias for {} resolved a different published filename",
                    file_fixture.id
                );
                continue;
            }
            let source = GgufFile::open(&path).unwrap();
            verify_profile(&source).unwrap();
            let tokenizer = crate::tokenizer::NativeTokenizer::from_gguf(&source).unwrap();
            for case in fixture()["cases"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|c| c.get("error").is_none())
            {
                let body = render(
                    &parse_messages(&serde_json::to_vec(&case["messages"]).unwrap()).unwrap(),
                    Effort::parse(case["reasoning_effort"].as_str()).unwrap(),
                )
                .unwrap();
                assert_eq!(
                    serde_json::json!(tokenizer.encode(&body, true).unwrap()),
                    case["token_ids"],
                    "{}",
                    case["name"]
                );
            }
            assert_eq!(tokenizer.encode("<|ifm|im_end|>", false).unwrap(), [250019]);
        }
    }
}
