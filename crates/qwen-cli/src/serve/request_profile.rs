//! Immutable CPU request semantics, independent of the resident model owner.

use super::items::{QwenTemplate, ServeError, ServeRequest, TemplateStyle};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::{render, render_ds4, render_glm5_next, render_k2, render_muse};
use crate::release_identity::ReleaseIdentity;
use crate::release_sampling::{decimal, fresh_seed, release_sampling};
use qwen_llm::muse_glimmer::MuseGlimmerChatTemplateProfile;
use qwen_llm::sampling::SamplingConfig;
use serde_json::Value;
use std::sync::Arc;

/// A backend's output limits, for ceilings computed before generation: the
/// token limit an omitted `max_output_tokens` resolves to, and the longest
/// piece its decoder emits for one token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OutputLimits {
    pub(crate) default_max_tokens: usize,
    pub(crate) max_piece_bytes: usize,
}

impl OutputLimits {
    /// The token limit a request resolves to (as the backends resolve it).
    pub(crate) fn max_tokens(self, request: &ServeRequest) -> usize {
        request.max_output_tokens.unwrap_or(self.default_max_tokens)
    }

    /// Test profiles' limits.
    #[cfg(test)]
    pub(crate) const TEST: Self = Self {
        default_max_tokens: 4096,
        max_piece_bytes: 64,
    };
}

#[derive(Clone)]
pub(crate) enum RequestProfile {
    /// The trait default for test backends: no output limits, so no tool
    /// byte ceiling (every production backend declares its own profile).
    UnboundQwen,
    OrdinaryQwen {
        template: QwenTemplate,
        no_thinking_supported: bool,
        style: TemplateStyle,
        /// The identified release's defaults for omitted sampling fields;
        /// `None` leaves them to the backend's legacy greedy fallbacks.
        sampling: Option<SamplingConfig>,
        limits: OutputLimits,
    },
    FlashNext {
        style: TemplateStyle,
        limits: OutputLimits,
    },
    DeepSeekV4 {
        style: TemplateStyle,
        /// As for `OrdinaryQwen`.
        sampling: Option<SamplingConfig>,
        limits: OutputLimits,
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
            | Self::FlashNext { style, .. }
            | Self::DeepSeekV4 { style, .. } => Some(*style),
            Self::UnboundQwen | Self::Muse { .. } | Self::K2 { .. } | Self::Glm5Next { .. } => None,
        }
    }

    pub(crate) fn normalize(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        if request.prefill_lineage.is_some() && !matches!(self, Self::Glm5Next { .. }) {
            return Err(ServeError::invalid_request(
                Some("x_qwen.prefill_lineage"),
                "x_qwen.prefill_lineage is supported only by GLM-5.3-Flash; this family reads prompts one way",
            ));
        }
        // The tool block's ceiling must be computable before generation
        // whenever the output protocol parses tools: always for ordinary
        // Qwen and Flash-Next (undeclared tool syntax still parses), for
        // DeepSeek V4 when tools are declared.
        let parses_tools = match self {
            Self::OrdinaryQwen { .. } | Self::FlashNext { .. } => true,
            Self::DeepSeekV4 { .. } => !request.model_request.tools.is_empty(),
            _ => false,
        };
        if let Self::OrdinaryQwen { limits, .. }
        | Self::FlashNext { limits, .. }
        | Self::DeepSeekV4 { limits, .. } = self
            && parses_tools
            && super::output_memory::tool_byte_ceiling(
                limits.max_tokens(request),
                limits.max_piece_bytes,
            )
            .is_none()
        {
            return Err(ServeError::invalid_request(
                Some("max_output_tokens"),
                "max_output_tokens must be >= 1 and small enough to bound the tool block",
            ));
        }
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

    /// [`Self::render`] plus the boundaries the family renderer recorded
    /// (GLM-5.3 only; other families render text only).
    pub(crate) fn render_prepared(
        &self,
        request: &ServeRequest,
    ) -> Result<(String, Option<super::http::PromptBoundaries>), ServeError> {
        match self {
            Self::Glm5Next { .. } => render_glm5_next::render_with_boundaries(request)
                .map(|(prompt, boundaries)| (prompt, Some(boundaries))),
            _ => self.render(request).map(|prompt| (prompt, None)),
        }
    }

    pub(crate) fn output(&self, request: &ServeRequest) -> OutputProtocol {
        // Normalization admitted the ceiling before any response bytes;
        // were it uncomputable here, fail closed (any tool block exceeds 0).
        let ceiling = |limits: &OutputLimits| {
            Some(
                super::output_memory::tool_byte_ceiling(
                    limits.max_tokens(request),
                    limits.max_piece_bytes,
                )
                .unwrap_or(0),
            )
        };
        match self {
            Self::UnboundQwen => OutputProtocol::Qwen {
                preopened_reasoning: false,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
                tool_byte_ceiling: None,
            },
            Self::OrdinaryQwen {
                template, limits, ..
            } => OutputProtocol::Qwen {
                preopened_reasoning: qwen_preopens(*template, request),
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
                tool_byte_ceiling: ceiling(limits),
            },
            Self::FlashNext { limits, .. } => OutputProtocol::Qwen {
                preopened_reasoning: render::qwen_generation(request)
                    == render::QwenGeneration::PreOpen,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
                tool_byte_ceiling: ceiling(limits),
            },
            Self::DeepSeekV4 { limits, .. } => OutputProtocol::Qwen {
                preopened_reasoning: render_ds4::preopens_reasoning(request).unwrap_or(false),
                parse_tools: !request.model_request.tools.is_empty(),
                tool_grammar: ToolGrammar::DeepSeekDsml,
                tool_byte_ceiling: ceiling(limits),
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

/// A request's sampler: each field it sets, else the release default, and a
/// fresh seed when it chose none. A backend samples from exactly this, so a
/// request that bypassed normalization still gets the release's defaults.
pub(crate) fn sampling_with_defaults(
    request: &ServeRequest,
    release: SamplingConfig,
) -> SamplingConfig {
    SamplingConfig {
        temperature: request.temperature.unwrap_or(release.temperature),
        top_k: request.top_k.unwrap_or(release.top_k),
        top_p: request.top_p.unwrap_or(release.top_p),
        min_p: request.min_p.unwrap_or(release.min_p),
        seed: request.seed.unwrap_or_else(fresh_seed),
    }
}

/// Absent sampling fields take the release defaults and an absent seed a
/// fresh draw; the response echoes what was sampled (the seed under
/// `x_qwen.stats`).
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
    request.seed.get_or_insert_with(fresh_seed);
}

pub(super) fn qwen_preopens(template: QwenTemplate, request: &ServeRequest) -> bool {
    let mut bound = request.clone();
    bound.template = template;
    render::qwen_generation(&bound) == render::QwenGeneration::PreOpen
}

#[cfg(test)]
mod tests {

    /// Qwen-family tool ceilings: the resolved token limit (explicit or the
    /// backend default) times the longest piece times 3; an uncomputable
    /// ceiling is refused at normalization (400) when tools are declared.
    #[test]
    fn qwen_family_tool_ceilings_resolve_the_token_limit_and_refuse_overflow() {
        let limits = OutputLimits {
            default_max_tokens: 100,
            max_piece_bytes: 16,
        };
        let profiles = [
            RequestProfile::OrdinaryQwen {
                template: QwenTemplate::Qwen35,
                no_thinking_supported: true,
                style: TemplateStyle::House,
                sampling: None,
                limits,
            },
            RequestProfile::FlashNext {
                style: TemplateStyle::House,
                limits,
            },
            RequestProfile::DeepSeekV4 {
                style: TemplateStyle::House,
                sampling: None,
                limits,
            },
        ];
        let tools = serde_json::json!([{"type": "function", "name": "f",
            "parameters": {"type": "object", "properties": {}}}]);
        for profile in &profiles {
            for (max, ceiling) in [(None, 100 * 16 * 3), (Some(7), 7 * 16 * 3)] {
                let mut body = serde_json::json!({"model": "m", "input": "hi", "tools": tools});
                if let Some(max) = max {
                    body["max_output_tokens"] = max.into();
                }
                let mut request = profile.parse(&body).unwrap();
                profile.normalize(&mut request).unwrap();
                match profile.output(&request) {
                    OutputProtocol::Qwen {
                        tool_byte_ceiling, ..
                    } => assert_eq!(tool_byte_ceiling, Some(ceiling)),
                    other => panic!("{other:?}"),
                }
            }
            let mut body = serde_json::json!({"model": "m", "input": "hi", "tools": tools,
                "max_output_tokens": u64::MAX});
            if let Ok(mut request) = profile.parse(&body) {
                let error = profile.normalize(&mut request).unwrap_err();
                assert_eq!(
                    (error.status, error.param.as_deref()),
                    (400, Some("max_output_tokens"))
                );
            }
            body["max_output_tokens"] = (usize::MAX / 2).into();
            let mut request = profile.parse(&body).unwrap();
            let error = profile.normalize(&mut request).unwrap_err();
            assert_eq!(error.status, 400);
            // Without declared tools: ordinary Qwen and Flash-Next still parse
            // tool syntax, so the ceiling is still required; DeepSeek V4 does
            // not parse tools then.
            let body = serde_json::json!({"model": "m", "input": "hi",
                "max_output_tokens": usize::MAX / 2});
            let mut request = profile.parse(&body).unwrap();
            let refused = profile.normalize(&mut request).is_err();
            assert_eq!(
                refused,
                !matches!(profile, RequestProfile::DeepSeekV4 { .. })
            );
        }
    }
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
