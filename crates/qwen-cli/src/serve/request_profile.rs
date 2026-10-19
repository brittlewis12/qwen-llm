//! Immutable CPU request semantics, independent of the resident model owner.

use super::items::{QwenTemplate, ServeError, ServeRequest, TemplateStyle};
use super::output_partition::{OutputProtocol, ToolGrammar};
use super::{render, render_ds4, render_glm5_next, render_k2, render_muse};
use crate::release_identity::ReleaseIdentity;
use crate::release_sampling::{SamplingDefaults, decimal, fresh_seed, release_sampling};
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

/// The request and protocol after family normalization and fallible protocol
/// resolution. Both values are immutable once constructed.
pub(crate) struct ValidatedRequest {
    request: ServeRequest,
    output: OutputProtocol,
}

impl ValidatedRequest {
    pub(crate) fn request(&self) -> &ServeRequest {
        &self.request
    }

    pub(crate) fn output(&self) -> &OutputProtocol {
        &self.output
    }
}

#[derive(Clone)]
pub(crate) enum RequestProfile {
    OrdinaryQwen {
        template: QwenTemplate,
        no_thinking_supported: bool,
        style: TemplateStyle,
        /// The identified release's defaults for omitted sampling fields;
        /// `None` leaves them to the backend's legacy greedy fallbacks.
        sampling: Option<SamplingDefaults>,
        limits: OutputLimits,
    },
    FlashNext {
        style: TemplateStyle,
        limits: OutputLimits,
    },
    DeepSeekV4 {
        style: TemplateStyle,
        /// As for `OrdinaryQwen`.
        sampling: Option<SamplingDefaults>,
        limits: OutputLimits,
    },
    DeepSeekV41 {
        limits: OutputLimits,
        sampling: Option<SamplingDefaults>,
    },
    Muse {
        template: MuseGlimmerChatTemplateProfile,
        default_max_tokens: usize,
        max_piece_bytes: usize,
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
    #[cfg(test)]
    pub(crate) fn test_qwen() -> Self {
        Self::OrdinaryQwen {
            template: QwenTemplate::Generic,
            no_thinking_supported: true,
            style: TemplateStyle::House,
            sampling: None,
            limits: OutputLimits::TEST,
        }
    }

    #[cfg(test)]
    pub(crate) fn validate(
        &self,
        mut request: ServeRequest,
    ) -> Result<ValidatedRequest, ServeError> {
        self.normalize(&mut request)?;
        let output = self.resolve_output(&request)?;
        Ok(ValidatedRequest { request, output })
    }

    pub(crate) fn validate_with(
        &self,
        mut request: ServeRequest,
        normalize: impl FnOnce(&mut ServeRequest) -> Result<(), ServeError>,
        resolve_output: impl FnOnce(&ServeRequest) -> Result<OutputProtocol, ServeError>,
    ) -> Result<ValidatedRequest, ServeError> {
        normalize(&mut request)?;
        let output = resolve_output(&request)?;
        Ok(ValidatedRequest { request, output })
    }

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
            Self::DeepSeekV41 { .. } => {
                if let Some(options) = body["x_qwen"].as_object() {
                    if options
                        .get("history_thinking")
                        .is_some_and(|v| !v.is_null())
                    {
                        return Err(ServeError::invalid_request(
                            Some("x_qwen.history_thinking"),
                            "V4.1 uses official recipe history semantics; history_thinking overrides are unsupported",
                        ));
                    }
                }
                super::items::parse_request(body)
            }
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
            Self::DeepSeekV41 { .. }
            | Self::Muse { .. }
            | Self::K2 { .. }
            | Self::Glm5Next { .. } => None,
        }
    }

    pub(crate) fn normalize(&self, request: &mut ServeRequest) -> Result<(), ServeError> {
        let max_piece_bytes = match self {
            Self::OrdinaryQwen { limits, .. }
            | Self::FlashNext { limits, .. }
            | Self::DeepSeekV4 { limits, .. }
            | Self::DeepSeekV41 { limits, .. } => Some(limits.max_piece_bytes),
            Self::K2 {
                max_piece_bytes, ..
            }
            | Self::Glm5Next {
                max_piece_bytes, ..
            } => Some(*max_piece_bytes),
            Self::Muse {
                max_piece_bytes, ..
            } => Some(*max_piece_bytes),
        };
        if max_piece_bytes == Some(0) {
            return Err(ServeError::server_error(
                "backend reports a zero decoded-piece byte bound",
            ));
        }
        if request.prefill_lineage.is_some() && !matches!(self, Self::Glm5Next { .. }) {
            return Err(ServeError::invalid_request(
                Some("x_qwen.prefill_lineage"),
                "x_qwen.prefill_lineage is supported only by GLM-5.3-Flash; this family reads prompts one way",
            ));
        }
        match self {
            // Ordinary Qwen binds a clone during rendering; Flash-Next binds
            // before rendering, protocol selection and response echoes.
            Self::DeepSeekV41 { limits, sampling } => {
                super::protocol_v41::validate(request)?;
                if request.template_style.is_some() {
                    return Err(ServeError::invalid_request(
                        Some("x_qwen.template_style"),
                        "V4.1 uses the official recipe; template_style overrides are unsupported",
                    ));
                }
                if let Some(defaults) = sampling {
                    fill_sampling_defaults(request, *defaults);
                }
                request
                    .max_output_tokens
                    .get_or_insert(limits.default_max_tokens);
                Ok(())
            }
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
                ..
            } => {
                render_k2::normalize_with_profile(
                    request,
                    *default_max_tokens,
                    *capacity,
                    chat.as_deref(),
                )?;
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
            Self::FlashNext { .. } => Ok(render::render_qwen_serve_prompt(request)),
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
            Self::DeepSeekV41 { .. } => super::protocol_v41::render(request),
            Self::DeepSeekV4 { .. } => render_ds4::render_deepseek_v4_serve_prompt(request),
            Self::Muse { template, .. } => {
                render_muse::render_muse_glimmer_serve_prompt(request, *template)
            }
            Self::K2 { chat, .. } => render_k2::render_with_profile(request, chat.as_deref()),
            Self::Glm5Next { .. } => render_glm5_next::render(request),
        }
    }

    /// [`Self::render`] plus the boundaries the family renderer recorded
    /// (GLM-5.3 and V4.1; other families render text only).
    pub(crate) fn render_prepared(
        &self,
        request: &ServeRequest,
    ) -> Result<(String, Option<super::http::PromptBoundaries>), ServeError> {
        match self {
            Self::DeepSeekV41 { .. } => super::protocol_v41::render_with_boundaries(request)
                .map(|(prompt, boundaries)| (prompt, Some(boundaries))),
            Self::Glm5Next { .. } => render_glm5_next::render_with_boundaries(request)
                .map(|(prompt, boundaries)| (prompt, Some(boundaries))),
            _ => self.render(request).map(|prompt| (prompt, None)),
        }
    }

    pub(crate) fn resolve_output(
        &self,
        request: &ServeRequest,
    ) -> Result<OutputProtocol, ServeError> {
        // Normalization admitted the ceiling before any response bytes;
        // were it uncomputable here, fail closed (any tool block exceeds 0).
        let ceiling = |limits: &OutputLimits| {
            super::output_memory::OutputByteCeiling::from_tokens_and_piece(
                limits.max_tokens(request),
                limits.max_piece_bytes,
            )
            .ok_or_else(|| {
                ServeError::invalid_request(
                    Some("max_output_tokens"),
                    "max_output_tokens must be >= 1 and small enough to bound the tool block",
                )
            })
        };
        Ok(match self {
            Self::DeepSeekV41 { limits, .. } => OutputProtocol::DeepSeekV41 {
                thinking: super::protocol_v41::thinking(request)?.is_some(),
                max_bytes: ceiling(limits)?,
            },
            Self::OrdinaryQwen {
                template, limits, ..
            } => OutputProtocol::Qwen {
                preopened_reasoning: qwen_preopens(*template, request),
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
                tool_byte_ceiling: Some(ceiling(limits)?),
            },
            Self::FlashNext { limits, .. } => OutputProtocol::Qwen {
                preopened_reasoning: render::qwen_generation(request)
                    == render::QwenGeneration::PreOpen,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
                tool_byte_ceiling: Some(ceiling(limits)?),
            },
            Self::DeepSeekV4 { limits, .. } => {
                let parses_tools = !request.model_request.tools.is_empty();
                OutputProtocol::Qwen {
                    preopened_reasoning: render_ds4::preopens_reasoning(request)?,
                    parse_tools: parses_tools,
                    tool_grammar: ToolGrammar::DeepSeekDsml,
                    tool_byte_ceiling: match ceiling(limits) {
                        Ok(ceiling) => Some(ceiling),
                        Err(error) if parses_tools => return Err(error),
                        Err(_) => None,
                    },
                }
            }
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
            } => render_k2::tools::output_protocol(request, *max_piece_bytes)?,
            Self::K2 { chat: None, .. } => OutputProtocol::RawText,
            Self::Glm5Next {
                max_piece_bytes, ..
            } => render_glm5_next::try_output_protocol(request, *max_piece_bytes)?,
        })
    }

    /// Test-only compatibility helper for fixtures that have already
    /// normalized a request.
    #[cfg(test)]
    pub(crate) fn output(&self, request: &ServeRequest) -> OutputProtocol {
        self.resolve_output(request)
            .expect("normalized fixture has a resolvable output protocol")
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
    release: impl Into<SamplingDefaults>,
) -> SamplingConfig {
    let release = release.into();
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
pub(crate) fn fill_sampling_defaults(
    request: &mut ServeRequest,
    release: impl Into<SamplingDefaults>,
) {
    let release = release.into();
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

    #[test]
    fn v41_refuses_recognized_but_unhonored_recipe_controls() {
        let profile = RequestProfile::DeepSeekV41 {
            limits: OutputLimits::TEST,
            sampling: None,
        };
        for (key, value) in [
            ("history_thinking", serde_json::json!("preserve")),
            ("history_thinking", serde_json::json!("strip")),
            ("template_style", serde_json::json!("house")),
            ("template_style", serde_json::json!("upstream")),
            ("prefill_lineage", serde_json::json!("exact")),
            ("prefill_lineage", serde_json::json!("fast")),
        ] {
            let body = serde_json::json!({"model":"m", "input":"hi", "x_qwen":{key:value}});
            let error = profile
                .parse(&body)
                .and_then(|mut request| profile.normalize(&mut request))
                .unwrap_err();
            assert_eq!(error.status, 400, "{key}");
            assert_eq!(
                error.param.as_deref(),
                Some(format!("x_qwen.{key}").as_str())
            );
        }
        let mut request = profile
            .parse(
                &serde_json::json!({"model":"m", "input":"hi", "x_qwen":{"history_thinking":null}}),
            )
            .unwrap();
        profile.normalize(&mut request).unwrap();
    }

    #[test]
    fn v41_unknown_extensions_follow_shared_parser_policy() {
        let profile = RequestProfile::DeepSeekV41 {
            limits: OutputLimits::TEST,
            sampling: None,
        };
        let base = serde_json::json!({"model":"m", "input":"hi"});
        let expected = super::super::items::parse_request(&base).unwrap();
        for key in [
            "ced",
            "prefix_reuse",
            "prefix_cache",
            "snapshot_cache",
            "durable_prefix_cache",
        ] {
            for value in [
                serde_json::json!(true),
                serde_json::json!(false),
                Value::Null,
            ] {
                let mut body = base.clone();
                body["x_qwen"] = serde_json::json!({key:value});
                let shared = super::super::items::parse_request(&body).unwrap();
                let mut request = profile.parse(&body).unwrap();
                assert_eq!(shared, expected, "{key}");
                assert_eq!(request, shared, "{key}");
                profile.normalize(&mut request).unwrap();
                assert_eq!(
                    profile.render(&request).unwrap(),
                    profile.render(&expected).unwrap()
                );
            }
        }
    }

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
                    } => assert_eq!(
                        tool_byte_ceiling.map(|ceiling| ceiling.get()),
                        Some(ceiling)
                    ),
                    other => panic!("{other:?}"),
                }
            }
            let mut body = serde_json::json!({"model": "m", "input": "hi", "tools": tools,
                "max_output_tokens": u64::MAX});
            if let Ok(request) = profile.parse(&body) {
                let error = profile.validate(request).err().unwrap();
                assert_eq!(
                    (error.status, error.param.as_deref()),
                    (400, Some("max_output_tokens"))
                );
            }
            body["max_output_tokens"] = (usize::MAX / 2).into();
            let request = profile.parse(&body).unwrap();
            let error = profile.validate(request).err().unwrap();
            assert_eq!(error.status, 400);
            // Without declared tools: ordinary Qwen and Flash-Next still parse
            // tool syntax, so the ceiling is still required; DeepSeek V4 does
            // not parse tools then.
            let body = serde_json::json!({"model": "m", "input": "hi",
                "max_output_tokens": usize::MAX / 2});
            let request = profile.parse(&body).unwrap();
            let refused = profile.validate(request).is_err();
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
        let qwen = RequestProfile::test_qwen();
        let raw = body(&format!(r#"{{{sentinel}:"not-a-number"}}"#));
        let error = qwen
            .decode(raw.as_bytes())
            .and_then(|decoded| qwen.parse(&decoded))
            .unwrap_err();
        assert_eq!(error.status, 400);
    }
}
