//! Immutable CPU request semantics, independent of the resident model owner.

use super::items::{QwenTemplate, ServeError, ServeRequest, TemplateStyle};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::{render, render_ds4, render_glm5_next, render_k2, render_muse};
use qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile;
use serde_json::Value;
use std::sync::Arc;

#[derive(Clone)]
pub(crate) enum RequestProfile {
    UnboundQwen,
    OrdinaryQwen {
        template: QwenTemplate,
        no_thinking_supported: bool,
        style: TemplateStyle,
    },
    FlashNext {
        style: TemplateStyle,
    },
    DeepSeekV4 {
        style: TemplateStyle,
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
    /// Verified GLM-5.3-Flash text chat over a session of fixed capacity.
    Glm5Next {
        default_max_tokens: usize,
        capacity: usize,
    },
}

impl RequestProfile {
    pub(crate) fn decode(&self, body: &[u8]) -> Result<Value, ServeError> {
        match self {
            Self::K2 { .. } => render_k2::tools::decode_request_json(body),
            _ => serde_json::from_slice(body).map_err(|error| {
                ServeError::invalid_request(None, format!("request body is not JSON: {error}"))
            }),
        }
    }

    pub(crate) fn parse(&self, body: &Value) -> Result<ServeRequest, ServeError> {
        match self {
            Self::K2 { chat, .. } => render_k2::parse_with_profile(body, chat.as_deref()),
            _ => super::items::parse_request(body),
        }
    }

    pub(crate) fn template_style_default(&self) -> Option<TemplateStyle> {
        match self {
            Self::OrdinaryQwen { style, .. }
            | Self::FlashNext { style }
            | Self::DeepSeekV4 { style } => Some(*style),
            Self::UnboundQwen | Self::Muse { .. } | Self::K2 { .. } | Self::Glm5Next { .. } => None,
        }
    }

    pub(crate) fn normalize(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        match self {
            // Ordinary Qwen binds a clone during rendering; Flash-Next binds
            // before rendering, protocol selection and response echoes.
            Self::UnboundQwen | Self::OrdinaryQwen { .. } => Ok(()),
            Self::FlashNext { .. } => {
                *request =
                    crate::open_responses::bind_qwen_request(request, QwenTemplate::Qwen38, true)?;
                Ok(())
            }
            Self::DeepSeekV4 { .. } => {
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
            } => render_glm5_next::normalize_request(request, *default_max_tokens, *capacity),
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
            Self::Glm5Next { .. } => OutputProtocol::Glm5NextChat,
        }
    }
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
}
