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
    if request.no_thinking && !no_thinking_supported {
        return Err(ServeError::invalid_request(
            Some("x_qwen.no_thinking"),
            "x_qwen.no_thinking is not validated for the loaded model identity",
        ));
    }
    if template != QwenTemplate::Qwen38 && request.reasoning_effort.is_some() {
        return Err(ServeError::invalid_request(
            Some("reasoning.effort"),
            "reasoning.effort is only supported for validated Qwen3.8 identities",
        ));
    }
    if template == QwenTemplate::Qwen38 {
        match (request.no_thinking, request.reasoning_effort.as_deref()) {
            (true, Some(_)) => {
                return Err(ServeError::invalid_request(
                    Some("reasoning.effort"),
                    "reasoning.effort cannot be combined with x_qwen.no_thinking",
                ));
            }
            (_, None | Some("none" | "low" | "medium" | "xhigh")) => {}
            (_, Some(other)) => {
                return Err(ServeError::invalid_request(
                    Some("reasoning.effort"),
                    format!(
                        "Qwen3.8 supports reasoning.effort none|low|medium|xhigh; got {other:?}"
                    ),
                ));
            }
        }
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
}
