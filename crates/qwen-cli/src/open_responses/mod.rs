//! Pure Open Responses parsing and Qwen prompt rendering.

#![allow(dead_code)] // each binary consumes a different subset of the shared protocol core

pub(crate) mod items;
pub(crate) mod render;
pub(crate) mod tool_parse;

use items::{QwenTemplate, ServeError, ServeRequest};

pub(crate) fn bind_qwen_request(
    request: &ServeRequest,
    template: QwenTemplate,
    no_thinking_supported: bool,
) -> Result<ServeRequest, ServeError> {
    let mut request = request.clone();
    request.template = template;
    if request.thinking_requested && request.no_thinking {
        return Err(ServeError::invalid_request(
            Some("x_qwen.thinking"),
            "x_qwen.thinking cannot be combined with x_qwen.no_thinking",
        ));
    }
    if request.thinking_requested && !template.verified() {
        return Err(ServeError::invalid_request(
            Some("x_qwen.thinking"),
            "x_qwen.thinking requires a pinned Qwen template",
        ));
    }
    if request.no_thinking && !no_thinking_supported {
        return Err(ServeError::invalid_request(
            Some("x_qwen.no_thinking"),
            "x_qwen.no_thinking is not validated for the loaded model identity",
        ));
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
