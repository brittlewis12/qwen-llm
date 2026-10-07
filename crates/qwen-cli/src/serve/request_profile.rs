//! Immutable CPU request semantics, independent of the resident model owner.

use super::items::{QwenTemplate, ServeError, ServeRequest, TemplateStyle};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::{render, render_ds4, render_glm5_next, render_k2, render_muse};
use crate::release_identity::ReleaseIdentity;
use crate::release_sampling::{decimal, release_sampling};
use qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile;
use qwen_llm::sampling::SamplingConfig;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum RequestProfile {
    UnboundQwen,
    OrdinaryQwen {
        template: QwenTemplate,
        no_thinking_supported: bool,
        style: TemplateStyle,
        /// The identified release's defaults for omitted sampling fields;
        /// `None` leaves them to the backend's legacy greedy fallbacks.
        sampling: Option<SamplingConfig>,
    },
    FlashNext {
        style: TemplateStyle,
    },
    DeepSeekV4 {
        style: TemplateStyle,
        /// As for `OrdinaryQwen`.
        sampling: Option<SamplingConfig>,
    },
    Muse {
        template: MuseGlimmerChatTemplateProfile,
        default_max_tokens: usize,
        eos_token_id: i32,
        eot_token_id: i32,
    },
    K2 {
        chat: Option<Arc<render_k2::ChatCapability>>,
        default_max_tokens: usize,
        capacity: usize,
        max_piece_bytes: usize,
    },
    /// Verified GLM-5.3-Flash chat and tools over a session of fixed capacity.
    Glm5Next {
        default_max_tokens: usize,
        capacity: usize,
        max_piece_bytes: usize,
    },
}

impl RequestProfile {
    pub(crate) fn decode(&self, body: &[u8]) -> Result<Value, ServeError> {
        match self {
            Self::K2 { .. } => render_k2::tools::decode_request_json(body),
            // Tool schemas and replayed arguments reach GLM's renderer as
            // sent: no key read as serde's internal number tag, duplicate
            // keys refused rather than collapsed.
            Self::Glm5Next { .. } => {
                let text = std::str::from_utf8(body).map_err(|error| {
                    ServeError::invalid_request(
                        None,
                        format!("request body is not UTF-8 JSON: {error}"),
                    )
                })?;
                qwen_llm::tool_schema::decode_json(text).map_err(|error| {
                    ServeError::invalid_request(None, format!("request body is not JSON: {error}"))
                })
            }
            _ => serde_json::from_slice(body).map_err(|error| {
                ServeError::invalid_request(None, format!("request body is not JSON: {error}"))
            }),
        }
    }

    pub(crate) fn parse(&self, body: &Value) -> Result<ServeRequest, ServeError> {
        match self {
            Self::K2 { chat, .. } => render_k2::parse_with_profile(body, chat.as_deref()),
            Self::Glm5Next { .. } => {
                render_glm5_next::check_raw_tool_definitions(body)?;
                super::items::parse_request_lossless(body)
            }
            _ => super::items::parse_request(body),
        }
    }

    pub(crate) fn template_style_default(&self) -> Option<TemplateStyle> {
        match self {
            Self::OrdinaryQwen { style, .. }
            | Self::FlashNext { style }
            | Self::DeepSeekV4 { style, .. } => Some(*style),
            Self::UnboundQwen | Self::Muse { .. } | Self::K2 { .. } | Self::Glm5Next { .. } => None,
        }
    }

    pub(crate) fn normalize(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        match self {
            // Ordinary Qwen binds a clone during rendering; Flash-Next binds
            // before rendering, protocol selection and response echoes.
            Self::UnboundQwen => Ok(()),
            Self::OrdinaryQwen { sampling, .. } => {
                if let Some(release) = sampling {
                    fill_sampling_defaults(request, *release);
                }
                Ok(())
            }
            Self::FlashNext { .. } => {
                *request =
                    crate::open_responses::bind_qwen_request(request, QwenTemplate::Qwen38, true)?;
                fill_sampling_defaults(request, flash_next_release());
                Ok(())
            }
            Self::DeepSeekV4 { sampling, .. } => {
                if let Some(release) = sampling {
                    fill_sampling_defaults(request, *release);
                }
                if request.template_style != Some(TemplateStyle::Upstream) {
                    request.history_reasoning_missing = 0;
                }
                Ok(())
            }
            Self::Muse {
                default_max_tokens, ..
            } => render_muse::normalize_request(request, *default_max_tokens),
            Self::K2 {
                chat,
                default_max_tokens,
                capacity,
                max_piece_bytes,
            } => {
                render_k2::normalize_with_profile(
                    request,
                    *default_max_tokens,
                    *capacity,
                    chat.as_deref(),
                )?;
                if request.k2_tools.is_some() {
                    render_k2::tools::byte_budget(
                        request.max_output_tokens.unwrap(),
                        *max_piece_bytes,
                    )?;
                }
                Ok(())
            }
            Self::Glm5Next {
                default_max_tokens,
                capacity,
                max_piece_bytes,
            } => render_glm5_next::normalize_request(
                request,
                *default_max_tokens,
                *capacity,
                *max_piece_bytes,
            ),
        }
    }

    pub(crate) fn render(&self, request: &ServeRequest) -> Result<String, ServeError> {
        match self {
            Self::UnboundQwen | Self::FlashNext { .. } => {
                Ok(render::render_qwen_serve_prompt(request))
            }
            Self::OrdinaryQwen {
                template,
                no_thinking_supported,
                ..
            } => {
                let bound = crate::open_responses::bind_qwen_request(
                    request,
                    *template,
                    *no_thinking_supported,
                )?;
                Ok(render::render_qwen_serve_prompt(&bound))
            }
            Self::DeepSeekV4 { .. } => render_ds4::render_deepseek_v4_serve_prompt(request),
            Self::Muse { template, .. } => {
                render_muse::render_muse_glimmer_serve_prompt(request, *template)
            }
            Self::K2 { chat, .. } => render_k2::render_with_profile(request, chat.as_deref()),
            Self::Glm5Next { .. } => render_glm5_next::render(request),
        }
    }

    pub(crate) fn output(&self, request: &ServeRequest) -> OutputProtocol {
        match self {
            Self::UnboundQwen => OutputProtocol::Qwen {
                preopened_reasoning: false,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
            },
            Self::OrdinaryQwen { template, .. } => OutputProtocol::Qwen {
                preopened_reasoning: qwen_preopens(*template, request),
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
            },
            Self::FlashNext { .. } => OutputProtocol::Qwen {
                preopened_reasoning: render::qwen_generation(request)
                    == render::QwenGeneration::PreOpen,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
            },
            Self::DeepSeekV4 { .. } => OutputProtocol::Qwen {
                preopened_reasoning: render_ds4::preopens_reasoning(request).unwrap_or(false),
                parse_tools: !request.model_request.tools.is_empty(),
                tool_grammar: ToolGrammar::DeepSeekDsml,
            },
            Self::Muse {
                eos_token_id,
                eot_token_id,
                ..
            } => OutputProtocol::MuseAtem {
                eos_token_id: *eos_token_id,
                eot_token_id: *eot_token_id,
                declared_tools: request
                    .model_request
                    .tools
                    .iter()
                    .map(|tool| tool.name.clone())
                    .collect(),
            },
            Self::K2 {
                chat: Some(_),
                max_piece_bytes,
                ..
            } => render_k2::tools::output_protocol(request, *max_piece_bytes),
            Self::K2 { chat: None, .. } => OutputProtocol::RawText,
            Self::Glm5Next {
                max_piece_bytes, ..
            } => render_glm5_next::output_protocol(request, *max_piece_bytes),
        }
    }
}

/// Flash-Next's release defaults (`release_sampling`); serve seed 42.
pub(crate) fn flash_next_release() -> SamplingConfig {
    release_sampling(ReleaseIdentity::FlashNext, 42)
}

/// A request's sampler: each field it sets, else the release default. A
/// backend samples from exactly this, so a request that bypassed
/// normalization still gets the release's defaults.
pub(crate) fn sampling_with_defaults(
    request: &ServeRequest,
    release: SamplingConfig,
) -> SamplingConfig {
    SamplingConfig {
        temperature: request.temperature.unwrap_or(release.temperature),
        top_k: request.top_k.unwrap_or(release.top_k),
        top_p: request.top_p.unwrap_or(release.top_p),
        min_p: request.min_p.unwrap_or(release.min_p),
        seed: request.seed.unwrap_or(release.seed),
    }
}

/// Absent sampling fields take the release defaults, and the response echoes
/// what was sampled.
pub(crate) fn fill_sampling_defaults(request: &mut ServeRequest, release: SamplingConfig) {
    if request.temperature.is_none() {
        request.temperature = Some(release.temperature);
        request.temperature_echo = Some(decimal(release.temperature));
    }
    if request.top_p.is_none() {
        request.top_p = Some(release.top_p);
        request.top_p_echo = Some(decimal(release.top_p));
    }
    request.top_k.get_or_insert(release.top_k);
    request.min_p.get_or_insert(release.min_p);
    request.seed.get_or_insert(release.seed);
}

pub(super) fn qwen_preopens(template: QwenTemplate, request: &ServeRequest) -> bool {
    let mut bound = request.clone();
    bound.template = template;
    render::qwen_generation(&bound) == render::QwenGeneration::PreOpen
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profiles_are_owned_cpu_metadata() {
        fn assert_owned<T: Send + Sync + 'static>() {}
        assert_owned::<RequestProfile>();
    }

    const GLM: RequestProfile = RequestProfile::Glm5Next {
        default_max_tokens: 64,
        capacity: 4096,
        max_piece_bytes: 512,
    };

    /// A GLM request body decodes once, losslessly, from raw bytes: a
    /// literal "$serde_json::private::Number" object (numeric or not, with
    /// sibling keys) in a tool schema or a replayed call's arguments reaches
    /// the rendered prompt as sent, and duplicate keys anywhere are a 400.
    /// Other families keep serde decoding, where the replayed sentinel stays
    /// a 400 (their renderers decode the same way).
    #[test]
    fn glm_bodies_decode_losslessly_from_raw_bytes() {
        let body = |value: &str| {
            let arguments = serde_json::to_string(&format!(r#"{{"x":{value}}}"#)).unwrap();
            format!(
                r#"{{"model":"m","tools":[{{"type":"function","name":"f","parameters":{{"type":"object","properties":{{"x":{{"type":"object","default":{value}}}}}}}}}],"input":[{{"role":"user","content":"q"}},{{"type":"function_call","call_id":"c","name":"f","arguments":{arguments}}},{{"type":"function_call_output","call_id":"c","output":"r"}}]}}"#
            )
        };
        let sentinel = r#""$serde_json::private::Number""#;
        for value in [
            format!(r#"{{{sentinel}:"7"}}"#),
            format!(r#"{{{sentinel}:"not-a-number"}}"#),
            format!(r#"{{{sentinel}:"7","y":1}}"#),
        ] {
            let raw = body(&value);
            let decoded = GLM.decode(raw.as_bytes()).unwrap();
            let mut request = GLM
                .parse(&decoded)
                .unwrap_or_else(|e| panic!("{raw}: {e:?}"));
            let expected = qwen_llm::tool_schema::decode_json(&value).unwrap();
            assert_eq!(
                request.model_request.tools[0].parameters["properties"]["x"]["default"],
                expected
            );
            GLM.normalize(&mut request).unwrap();
            let prompt = GLM.render(&request).unwrap();
            // Python tojson's separators, with the literal key kept.
            let printed = qwen_llm::tool_schema::python_json(&expected).unwrap();
            assert!(printed.contains("$serde_json::private::Number"));
            assert_eq!(prompt.matches(&printed).count(), 2, "{prompt}");
        }
        for duplicate in [
            r#"{"model":"m","input":"q","input":"r"}"#.to_owned(),
            body(r#"{"a":1,"a":2}"#),
            r#"{"model":"m","input":"q","tools":[{"type":"function","name":"f","parameters":{"type":"object","type":"array"}}]}"#.to_owned(),
        ] {
            let error = GLM
                .decode(duplicate.as_bytes())
                .and_then(|decoded| GLM.parse(&decoded))
                .unwrap_err();
            assert_eq!(error.status, 400, "{duplicate}");
        }
        // Qwen keeps serde decoding; the sentinel in replayed arguments is
        // refused there (fail closed), not mis-rendered.
        let qwen = RequestProfile::UnboundQwen;
        let raw = body(&format!(r#"{{{sentinel}:"not-a-number"}}"#));
        let error = qwen
            .decode(raw.as_bytes())
            .and_then(|decoded| qwen.parse(&decoded))
            .unwrap_err();
        assert_eq!(error.status, 400);
    }
}
