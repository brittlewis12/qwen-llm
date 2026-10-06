//! GLM-5.3-Flash output with declared tools: the prompt-preopened reasoning
//! grammar (`<think>` … first `</think>`), then visible text routed through
//! `glm5_next_chat::ToolOutputStream`, which holds back tool markers and
//! publishes calls together once the turn's whole tool block parses. A tool
//! marker inside reasoning stays reasoning text.
//!
//! Memory: the block is capped at the request's byte bound, and admitted as
//! it grows. Each further [`ADMISSION_STEP`] of buffered text must find
//! three steps of process headroom (the text, its parsed values and their
//! serialized arguments live together at finish), or the turn fails with
//! `memory_admission_denied` (503) instead of growing unpriced.
use super::items::ServeError;
use super::output_partition::GenerationEnd;
use super::partition::PartitionEvent;
use super::partition_preopened::PreopenedPartition;
use qwen_llm::glm5_next_chat::{ToolDefinition, ToolOutputEnd, ToolOutputStream};

/// Buffered tool bytes admitted per step.
pub(crate) const ADMISSION_STEP: usize = 1 << 20;

pub(crate) struct Glm5NextToolsPartition {
    reasoning: PreopenedPartition,
    tools: ToolOutputStream,
    failure: Option<ServeError>,
    /// Buffered bytes admitted so far.
    admitted: usize,
    /// Process memory headroom (None: the host omits it).
    headroom: fn() -> Option<u64>,
}

impl Glm5NextToolsPartition {
    pub(crate) fn new(definitions: Vec<ToolDefinition>, max_bytes: usize) -> Self {
        Self::with_headroom(
            definitions,
            max_bytes,
            qwen_llm::metal::MetalContext::process_limit_bytes_remaining,
        )
    }

    fn with_headroom(
        definitions: Vec<ToolDefinition>,
        max_bytes: usize,
        headroom: fn() -> Option<u64>,
    ) -> Self {
        Self {
            reasoning: super::render_glm5_next::partition(),
            tools: ToolOutputStream::new(definitions, max_bytes),
            failure: None,
            admitted: 0,
            headroom,
        }
    }

    /// Admit the buffered block's growth step by step.
    fn admit_growth(&mut self) {
        while self.failure.is_none() && self.tools.buffered_bytes() > self.admitted {
            let need = (3 * ADMISSION_STEP) as u64;
            match super::transport_memory::admit_resident_transport(need, (self.headroom)()) {
                Ok(()) => self.admitted += ADMISSION_STEP,
                Err(error) => self.failure = Some(error),
            }
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
        self.admit_growth();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::output_partition::GenerationEnd;
    use serde_json::json;

    fn weather() -> ToolDefinition {
        ToolDefinition::from_value(
            &json!({"name": "get_weather", "parameters": {"type": "object",
            "properties": {"city": {"type": "string"}}}}),
        )
        .unwrap()
    }

    /// A growing tool block is admitted step by step: ample headroom
    /// publishes the call; headroom below three steps fails the turn with
    /// memory_admission_denied once the block holds any bytes; plain text with
    /// no block never asks.
    #[test]
    fn tool_block_growth_is_admitted_incrementally() {
        let call = "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        let run = |headroom: fn() -> Option<u64>, output: &str| {
            let mut partition =
                Glm5NextToolsPartition::with_headroom(vec![weather()], 1 << 24, headroom);
            let mut events = Vec::new();
            partition.push(output.as_bytes(), &mut events);
            partition
                .finish(GenerationEnd::StopToken(154_829), &mut events)
                .map(|()| events)
        };
        let events = run(|| Some(1 << 40), call).unwrap();
        assert!(
            events
                .iter()
                .any(|e| matches!(e, PartitionEvent::FunctionCall(_)))
        );
        let error = run(|| Some(1 << 20), call).unwrap_err();
        assert_eq!(error.status, 503);
        assert_eq!(error.code, Some("memory_admission_denied"));
        // Text with no tool block buffers nothing and asks nothing.
        assert!(run(|| Some(0x10), "</think>Plain answer.").is_ok());
    }
}
