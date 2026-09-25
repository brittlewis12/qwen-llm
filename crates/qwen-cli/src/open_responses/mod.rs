//! Pure Open Responses parsing and Qwen prompt rendering.

#![allow(dead_code)] // each binary consumes a different subset of the shared protocol core

pub(crate) mod items;
pub(crate) mod render;
pub(crate) mod tool_parse;

use crate::model_request::Turn;
use items::{QwenTemplate, ServeError, ServeRequest, TemplateStyle};

pub(crate) fn bind_qwen_request(
    request: &ServeRequest,
    template: QwenTemplate,
    no_thinking_supported: bool,
) -> Result<ServeRequest, ServeError> {
    let mut request = request.clone();
    request.template = template;
    // Qwen templates give `<think>` meaning, so reasoning must travel as
    // items for history to replay faithfully.
    if let Some(turn) = request.model_request.turns.iter().position(
        |turn| matches!(turn, Turn::Assistant { visible, .. } if visible.contains("<think>")),
    ) {
        return Err(ServeError::invalid_request(
            Some("input"),
            format!(
                "assistant turn {turn}: content must not embed <think>; reasoning travels as reasoning items"
            ),
        ));
    }
    if request.thinking_requested && request.no_thinking {
        return Err(ServeError::invalid_request(
            Some("x_qwen.thinking"),
            "x_qwen.thinking cannot be combined with x_qwen.no_thinking",
        ));
    }
    if request.thinking_requested && !template.verified() {
        return Err(ServeError::invalid_request(
            Some("x_qwen.thinking"),
            "x_qwen.thinking requires an identified Qwen release",
        ));
    }
    if request.no_thinking && !no_thinking_supported {
        return Err(ServeError::invalid_request(
            Some("x_qwen.no_thinking"),
            "x_qwen.no_thinking is not validated for the loaded model identity",
        ));
    }
    // Identified releases render history reasoning as generated in every
    // mode. The unverified contract re-renders history in the current mode,
    // so a no-thinking request would silently drop supplied reasoning.
    if request.no_thinking
        && !template.verified()
        && request.model_request.turns.iter().any(|turn| {
            matches!(turn, Turn::Assistant { reasoning: Some(reasoning), .. } if !reasoning.is_empty())
        })
    {
        return Err(ServeError::invalid_request(
            Some("input"),
            "reasoning history requires an identified Qwen release when x_qwen.no_thinking is set",
        ));
    }
    if request.template_style == Some(TemplateStyle::Upstream) {
        if !template.verified() {
            return Err(ServeError::invalid_request(
                Some("x_qwen.template_style"),
                "template_style upstream requires an identified Qwen release",
            ));
        }
        // Qwen history is already mode-independent in the released
        // templates; upstream only restores each template's own default for
        // history reasoning (Qwen3.5 has no preserve option, Qwen3.6
        // defaults it off, Qwen3.8 on). An explicit history_thinking wins.
        if request.history_thinking.is_none() && !request.strip_history_thinking {
            request.strip_history_thinking =
                matches!(template, QwenTemplate::Qwen35 | QwenTemplate::Qwen36);
        }
    }
    // Tools and tool history: the same family rule `qwen run --messages`
    // applies, so a model refuses identically on both lanes.
    if request.model_request.has_tool_surface() {
        // Blame the field the caller actually sent: declared `tools`, or
        // replayed call/result items under `input`.
        let param = if request.model_request.tools.is_empty() {
            "input"
        } else {
            "tools"
        };
        crate::prompt_template::qwen_tools_support(template)
            .require()
            .map_err(|error| ServeError::unsupported(param, error))?;
    }
    if template != QwenTemplate::Qwen38 && request.reasoning_effort.is_some() {
        return Err(ServeError::invalid_request(
            Some("reasoning.effort"),
            "reasoning.effort is only supported for validated Qwen3.8 identities",
        ));
    }
    if template == QwenTemplate::Qwen38 {
        // Bind once through the family table; renderers consume the bound
        // mode instead of re-parsing the string.
        let mode = crate::messages::Qwen38GenerationMode::parse(
            request.reasoning_effort.as_deref(),
            request.no_thinking,
        )
        .map_err(|error| ServeError::unsupported("reasoning.effort", error))?;
        request.qwen38_mode = Some(mode);
    }
    Ok(request)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Upstream restores each Qwen template's own history-reasoning default
    /// (3.5/3.6 strip before the last query, 3.8 keeps); house keeps it for
    /// every release; an explicit history_thinking overrides either style;
    /// the generic contract has no upstream template.
    #[test]
    fn template_style_resolves_history_thinking_per_release() {
        use items::{HistoryThinking, TemplateStyle};
        let bind = |template, style, history| {
            let request = ServeRequest {
                template_style: Some(style),
                history_thinking: history,
                strip_history_thinking: history == Some(HistoryThinking::Strip),
                ..ServeRequest::default()
            };
            bind_qwen_request(&request, template, true).map(|bound| bound.strip_history_thinking)
        };
        for (template, upstream_strips) in [
            (QwenTemplate::Qwen35, true),
            (QwenTemplate::Qwen36, true),
            (QwenTemplate::Qwen38, false),
        ] {
            assert_eq!(bind(template, TemplateStyle::House, None), Ok(false));
            assert_eq!(
                bind(template, TemplateStyle::Upstream, None),
                Ok(upstream_strips)
            );
            for style in [TemplateStyle::House, TemplateStyle::Upstream] {
                assert_eq!(
                    bind(template, style, Some(HistoryThinking::Preserve)),
                    Ok(false)
                );
                assert_eq!(
                    bind(template, style, Some(HistoryThinking::Strip)),
                    Ok(true)
                );
            }
        }
        assert!(bind(QwenTemplate::Generic, TemplateStyle::Upstream, None).is_err());
        assert_eq!(
            bind(QwenTemplate::Generic, TemplateStyle::House, None),
            Ok(false)
        );
    }

    #[test]
    fn model_binding_applies_the_same_reasoning_gates_without_a_backend() {
        let mut request = ServeRequest {
            reasoning_effort: Some("low".into()),
            ..ServeRequest::default()
        };
        assert!(bind_qwen_request(&request, QwenTemplate::Generic, true).is_err());
        assert_eq!(
            bind_qwen_request(&request, QwenTemplate::Qwen38, true)
                .unwrap()
                .template,
            QwenTemplate::Qwen38
        );
        request.reasoning_effort = Some("invalid".into());
        assert!(bind_qwen_request(&request, QwenTemplate::Qwen38, true).is_err());
        request.reasoning_effort = None;
        request.no_thinking = true;
        assert!(bind_qwen_request(&request, QwenTemplate::Qwen38, false).is_err());
    }

    /// Tools and replayed tool turns bind through the family rule shared
    /// with `qwen run --messages`: pinned templates render them, an
    /// unpinned template refuses with the advertised code.
    #[test]
    fn tool_surface_binds_through_the_family_input_capability() {
        use crate::model_request::{ToolCall, ToolDefinition, Turn};
        let declared = ServeRequest {
            model_request: crate::model_request::ModelRequest {
                tools: vec![ToolDefinition {
                    name: "ping".into(),
                    description: None,
                    parameters: serde_json::Value::Null,
                    strict: None,
                }],
                ..Default::default()
            },
            ..ServeRequest::default()
        };
        let replayed = ServeRequest {
            model_request: crate::model_request::ModelRequest {
                turns: vec![
                    Turn::User("Ping.".into()),
                    Turn::Assistant {
                        reasoning: None,
                        visible: String::new(),
                        calls: vec![ToolCall {
                            call_id: "c1".into(),
                            name: "ping".into(),
                            arguments: "{}".into(),
                        }],
                    },
                ],
                ..Default::default()
            },
            ..ServeRequest::default()
        };
        let advertised = crate::prompt_template::qwen_tools_support(QwenTemplate::Generic)
            .require()
            .unwrap_err();
        for (request, param) in [(&declared, "tools"), (&replayed, "input")] {
            let error = bind_qwen_request(request, QwenTemplate::Generic, false).unwrap_err();
            assert_eq!(error.status, 400);
            assert_eq!(error.code, Some(advertised.code));
            assert_eq!(error.message, advertised.message);
            assert_eq!(error.param.as_deref(), Some(param));
            for template in [
                QwenTemplate::Qwen35,
                QwenTemplate::Qwen36,
                QwenTemplate::Qwen38,
            ] {
                assert_eq!(
                    bind_qwen_request(request, template, true).unwrap().template,
                    template
                );
            }
        }
        // Plain chat on the same unpinned template still binds.
        assert!(bind_qwen_request(&ServeRequest::default(), QwenTemplate::Generic, false).is_ok());
        // Reasoning refusals now carry their capability code too.
        let effort = ServeRequest {
            reasoning_effort: Some("turbo".into()),
            ..ServeRequest::default()
        };
        assert_eq!(
            bind_qwen_request(&effort, QwenTemplate::Qwen38, true)
                .unwrap_err()
                .code,
            Some("reasoning_effort_invalid")
        );
    }
}
