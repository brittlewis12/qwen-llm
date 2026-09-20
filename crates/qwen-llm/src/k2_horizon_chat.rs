//! Pinned IFM final-checkpoint no-tools template. Raw checkpoints remain separate.
use crate::checkpoint_identity::verified_checkpoint_content_identity;
use crate::gguf::GgufFile;
use serde::{Deserialize, Deserializer, Serialize};
use sha2::{Digest, Sha256};

pub const REVISION: &str = "2c9659a84c4eea6f9f60462221fe762c8c84d75c";
pub const TEMPLATE_SHA256: &str =
    "a892cd0b0195599f283a8c706787520d9a6747640efb2f4dec4144b0abb62590";
pub const RENDERER: &str = "k2_horizon_ifm_no_tools_v1";
pub const CHAT_STOPS: [i32; 2] = [1, 250019];
pub const GGUF_TEMPLATE_SHA256: &str =
    "f6e3cd6dbf0f95016fff531f41f921dee541025733c14580a881cf5a5f9fa750";
pub const GENERATION_CONFIG_SHA256: &str =
    "2da7d47641f4509da4ae47711e31d8b5f0f3f801ee08d87e7e9f07f814bdc4a3";
const CONTENT_ID: &str = "719ae3a7c9386c25db2c33b50be15d715f883aa5a762495d5a65776660179e99";

#[derive(Debug, thiserror::Error)]
#[error("K2 chat: {0}")]
pub struct ChatError(String);
type Result<T> = std::result::Result<T, ChatError>;
fn error(message: impl Into<String>) -> ChatError {
    ChatError(message.into())
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
                let (tag, thinking) = message.thinking().ok_or_else(|| error("assistant history requires an explicit string thinking field (empty is valid)"))?;
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
        out.push_str("<|ifm|im_end|>");
    }
    out.push_str("<|ifm|im_start|>assistant\n<");
    out.push_str(effort.tag());
    out.push_str(">\n");
    Ok(out)
}

#[derive(Debug, Serialize)]
pub struct VerifiedChatProfile {
    renderer: &'static str,
    source_revision: &'static str,
    template_sha256: &'static str,
    gguf_template_sha256: &'static str,
    generation_config_sha256: &'static str,
    checkpoint_content_blake3: String,
    tokenizer_metadata_id: String,
    verification: &'static str,
    reference_policy: &'static str,
}

/// Chat eligibility is exact artifact provenance, not architecture/name/template
/// heuristics. Other compatible checkpoints can always use native raw input.
pub fn verify_profile(source: &GgufFile) -> Result<VerifiedChatProfile> {
    crate::k2_horizon::K2HorizonModel::from_gguf(source).map_err(|e| error(e.to_string()))?;
    let tokenizer = format!(
        "{:016x}",
        crate::runtime::tokenizer_metadata_identity(source)
    );
    if tokenizer != "51ebd8140ea2abd9" {
        return Err(error(
            "chat_profile_unverified: tokenizer profile; use raw input",
        ));
    }
    let template = source
        .get_str("tokenizer.chat_template")
        .ok_or_else(|| error("chat_profile_unverified: missing template"))?;
    let template_digest = format!("{:x}", Sha256::digest(template.as_bytes()));
    if template_digest != GGUF_TEMPLATE_SHA256 {
        return Err(error(
            "chat_profile_unverified: template differs from pinned release",
        ));
    }
    let report = verified_checkpoint_content_identity(source).map_err(|e| error(e.to_string()))?;
    let content: String = report
        .content_id
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    if !identity_matches(&tokenizer, &template_digest, &content) {
        return Err(error(
            "chat_profile_unverified: checkpoint bytes; use raw input",
        ));
    }
    Ok(VerifiedChatProfile {
        renderer: RENDERER,
        source_revision: REVISION,
        template_sha256: TEMPLATE_SHA256,
        gguf_template_sha256: GGUF_TEMPLATE_SHA256,
        generation_config_sha256: GENERATION_CONFIG_SHA256,
        checkpoint_content_blake3: content,
        tokenizer_metadata_id: tokenizer,
        verification: "retained_bytes_hashed_no_filename_cache_or_downloader_claims",
        reference_policy: "template_and_generation_config_from_pinned_upstream_revision_publisher_declares_conversion_source",
    })
}

fn identity_matches(tokenizer: &str, template: &str, content: &str) -> bool {
    tokenizer == "51ebd8140ea2abd9" && template == GGUF_TEMPLATE_SHA256 && content == CONTENT_ID
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> serde_json::Value {
        serde_json::from_str(include_str!("../tests/fixtures/k2_chat_hf.json")).unwrap()
    }
    #[test]
    fn k2_chat_identity_requires_all_facts_not_header_similarity() {
        assert!(identity_matches(
            "51ebd8140ea2abd9",
            GGUF_TEMPLATE_SHA256,
            CONTENT_ID
        ));
        assert!(!identity_matches(
            "pretraining",
            GGUF_TEMPLATE_SHA256,
            CONTENT_ID
        ));
        assert!(!identity_matches(
            "51ebd8140ea2abd9",
            TEMPLATE_SHA256,
            CONTENT_ID
        ));
        assert!(!identity_matches(
            "51ebd8140ea2abd9",
            GGUF_TEMPLATE_SHA256,
            &"0".repeat(64)
        ));
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
    #[ignore = "CPU only: K2_GGUF metadata/content identity and native token-ID oracle, never Metal"]
    fn cpu_k2_chat_artifact_identity_and_native_tokens() {
        let source = GgufFile::open(std::env::var("K2_GGUF").unwrap()).unwrap();
        let template = source.get_str("tokenizer.chat_template").unwrap_or("");
        let content = verified_checkpoint_content_identity(&source).unwrap();
        eprintln!(
            "template SHA256={:x}; content={}; tokenizer={:016x}; template_bytes={}",
            Sha256::digest(template.as_bytes()),
            content
                .content_id
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
            crate::runtime::tokenizer_metadata_identity(&source),
            template.len()
        );
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
        assert_eq!(
            verify_profile(&source).unwrap().checkpoint_content_blake3,
            CONTENT_ID
        );
    }
}
