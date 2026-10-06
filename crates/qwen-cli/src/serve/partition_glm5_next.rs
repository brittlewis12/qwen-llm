//! GLM-5.3-Flash output with declared tools: the prompt-preopened reasoning
//! grammar (`<think>` … first `</think>`), then visible text routed through
//! `glm5_next_chat::ToolOutputStream`, which holds back tool markers and
//! publishes calls together once the turn's whole tool block parses. A tool
//! marker inside reasoning stays reasoning text.
use super::items::ServeError;
use super::output_partition::GenerationEnd;
use super::partition::PartitionEvent;
use super::partition_preopened::PreopenedPartition;
use qwen_llm::glm5_next_chat::{ToolDefinition, ToolOutputEnd, ToolOutputStream};

pub(crate) struct Glm5NextToolsPartition {
    reasoning: PreopenedPartition,
    tools: ToolOutputStream,
    failure: Option<ServeError>,
}

impl Glm5NextToolsPartition {
    pub(crate) fn new(definitions: Vec<ToolDefinition>, max_bytes: usize) -> Self {
        Self {
            reasoning: super::render_glm5_next::partition(),
            tools: ToolOutputStream::new(definitions, max_bytes),
            failure: None,
        }
    }

    fn route(&mut self, incoming: Vec<PartitionEvent>, events: &mut Vec<PartitionEvent>) {
        for event in incoming {
            if self.failure.is_some() {
                return;
            }
            if let PartitionEvent::Visible(text) = event {
                match self.tools.push_visible(&text) {
                    Ok(text) if !text.is_empty() => events.push(PartitionEvent::Visible(text)),
                    Ok(_) => {}
                    Err(e) => self.failure = Some(ServeError::server_error(e.to_string())),
                }
            } else {
                events.push(event);
            }
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8], events: &mut Vec<PartitionEvent>) {
        let mut incoming = Vec::new();
        self.reasoning.push(bytes, &mut incoming);
        self.route(incoming, events);
    }

    pub(crate) fn finish(
        mut self,
        end: GenerationEnd,
        events: &mut Vec<PartitionEvent>,
    ) -> Result<(), ServeError> {
        if let Some(error) = self.failure.take() {
            return Err(error);
        }
        let mut incoming = Vec::new();
        let reasoning =
            std::mem::replace(&mut self.reasoning, super::render_glm5_next::partition());
        reasoning.finish(end, &mut incoming)?;
        self.route(incoming, events);
        if let Some(error) = self.failure {
            return Err(error);
        }
        let result = self
            .tools
            .finish(if end.is_token_limit() {
                ToolOutputEnd::TokenLimit
            } else {
                ToolOutputEnd::Stop
            })
            .map_err(|e| ServeError::server_error(e.to_string()))?;
        if !result.visible_tail.is_empty() {
            events.push(PartitionEvent::Visible(result.visible_tail));
        }
        events.extend(result.calls.into_iter().map(|call| {
            PartitionEvent::FunctionCall(super::tool_parse::ParsedCall {
                name: call.name,
                arguments: call.arguments,
            })
        }));
        Ok(())
    }
}
