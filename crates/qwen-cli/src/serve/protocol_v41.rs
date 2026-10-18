//! Adapter for the revision-pinned, MIT-licensed official DeepSeek recipe.
//! Serving keeps the existing Responses subset (auto/allowed tools, text only).
//! Omitted effort selects chat; low/high/xhigh/max enable official thinking.
//! Cancellation discards withheld tool calls rather than manufacturing a finish.
use super::items::{ServeError, ServeRequest};
use super::partition::PartitionEvent;
#[cfg(test)]
use super::request_profile::{OutputLimits, RequestProfile};
use super::tool_parse::ParsedCall;
use crate::model_request::Turn;
use deepseek_recipe::stream::state_machine::{
    OutputAction, OutputActionSegment, ParsingOptions, ReasoningStage, StateMachine,
};
use deepseek_recipe_core::conversation::{Conversation, ReasoningEffort};
use deepseek_recipe_core::messages::{InputMessage, ToolCall};
use deepseek_recipe_core::tools::ToolDefinition;
use deepseek_recipe_encoding::{PromptEncoding, dsv41::DeepseekV41Encoding};

/// CPU-only fixture profile without deployment sampling defaults.
#[cfg(test)]
pub(crate) fn profile(limits: OutputLimits) -> RequestProfile {
    RequestProfile::DeepSeekV41 {
        limits,
        sampling: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn v41_official_renderer_parser_roundtrip_every_byte_boundary() {
        let profile = profile(OutputLimits::TEST);
        let request = profile.parse(&json!({"model":"m","input":[
            {"role":"user","content":"question"},
            {"type":"reasoning","content":"plan"},
            {"role":"assistant","content":"answer"},
            {"type":"function_call","call_id":"c","name":"lookup","arguments":"{\"s\":\"caf\u{00e9}\\n\\\"\",\"n\":2}"},
            {"type":"function_call","call_id":"d","name":"lookup","arguments":"{}"},
            {"type":"function_call_output","call_id":"c","output":"found"},
            {"type":"function_call_output","call_id":"d","output":"second"},
            {"role":"user","content":"continue"}
        ],"tools":[{"type":"function","name":"lookup","parameters":{"type":"object"}}],"reasoning":{"effort":"high"}})).unwrap();
        let prompt = profile.render(&request).unwrap();
        assert!(prompt.contains("Reasoning Effort: 75"));
        assert!(prompt.contains("found"));
        assert!(prompt.ends_with("<think>"));
        let start = prompt.find("<think>plan").unwrap() + "<think>".len();
        let end = prompt[start..]
            .find("</\u{ff5c}DSML\u{ff5c} calls>")
            .unwrap()
            + start
            + "</\u{ff5c}DSML\u{ff5c} calls>".len();
        let output = &prompt[start..end];
        for split in 0..=output.len() {
            let mut partition = Partition::new(true, 100_000);
            let mut events = vec![];
            partition.push(&output.as_bytes()[..split], &mut events);
            partition.push(&output.as_bytes()[split..], &mut events);
            partition.finish(&mut events).unwrap();
            let calls: Vec<_> = events
                .iter()
                .filter_map(|event| match event {
                    PartitionEvent::FunctionCall(call) => Some(call),
                    _ => None,
                })
                .collect();
            assert_eq!(calls.len(), 2, "split={split}: {events:?}");
            assert_eq!(calls[0].name, "lookup");
            assert_eq!(calls[0].arguments["s"], "caf\u{e9}\n\"");
            assert_eq!(calls[0].arguments["n"], 2);
            assert!(calls[1].arguments.is_empty());
        }
    }

    #[test]
    fn v41_abort_discards_pending_calls_and_runtime_is_not_ds4() {
        use super::super::output_partition::{OutputPartition, OutputProtocol};
        let mut partition = OutputPartition::new(OutputProtocol::DeepSeekV41 {
            thinking: true,
            max_bytes: 1024,
        });
        let mut events = vec![];
        partition.push(b"plan</think><", &mut events);
        partition.abort(&mut events);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, PartitionEvent::FunctionCall(_)))
        );
        let mut partition = Partition::new(false, 1);
        partition.push(b"too big", &mut vec![]);
        assert!(partition.failure().is_some());
    }
}

fn invalid(detail: impl Into<String>) -> ServeError {
    ServeError::invalid_request(Some("input"), detail)
}

pub(crate) fn thinking(request: &ServeRequest) -> Result<Option<ReasoningEffort>, ServeError> {
    match request.reasoning_effort.as_deref() {
        None | Some("none") => Ok(None),
        Some("low") => Ok(Some(ReasoningEffort::Low)),
        Some("high") => Ok(Some(ReasoningEffort::High)),
        Some("xhigh") => Ok(Some(ReasoningEffort::Xhigh)),
        Some("max") => Ok(Some(ReasoningEffort::Max)),
        _ => Err(ServeError::invalid_request(
            Some("reasoning.effort"),
            "V4.1 supports none, low, high, xhigh, max",
        )),
    }
}

pub(crate) fn validate(request: &ServeRequest) -> Result<(), ServeError> {
    if request.no_thinking || request.thinking_requested || request.strip_history_thinking {
        return Err(invalid(
            "V4.1 uses reasoning.effort and official recipe history semantics",
        ));
    }
    thinking(request)?;
    Ok(())
}

pub(crate) fn render(request: &ServeRequest) -> Result<String, ServeError> {
    validate(request)?;
    let effort = thinking(request)?;
    let mut conversation = Conversation {
        thinking_mode: effort.is_some(),
        reasoning_effort: effort,
        ..Conversation::default()
    };
    if let Some(content) = &request.model_request.system {
        conversation.messages.push(InputMessage::System {
            content: content.clone(),
        });
    }
    for turn in &request.model_request.turns {
        match turn {
            Turn::User(content) => conversation.messages.push(InputMessage::User {
                content: content.clone(),
                image_sources: vec![],
            }),
            Turn::Assistant {
                reasoning,
                visible,
                calls,
            } => conversation.messages.push(InputMessage::Assistant {
                content: visible.clone(),
                reasoning_content: reasoning.clone(),
                tool_calls: (!calls.is_empty()).then(|| {
                    calls
                        .iter()
                        .map(|call| ToolCall {
                            id: call.call_id.clone(),
                            name: call.name.clone(),
                            arguments: call.arguments.clone(),
                        })
                        .collect()
                }),
            }),
            Turn::ToolResults(results) => {
                for result in results {
                    conversation.messages.push(InputMessage::Tool {
                        content: result.output.clone(),
                        image_sources: vec![],
                        tool_call_id: result.call_id.clone(),
                    });
                }
            }
        }
    }
    conversation.tools = request
        .model_request
        .tools
        .iter()
        .map(|tool| ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.parameters.clone(),
            strict: tool.strict,
        })
        .collect();
    Ok(DeepseekV41Encoding::new()
        .render_conversation(&conversation)
        .prompt)
}

pub(crate) struct Partition {
    machine: Option<StateMachine>,
    utf8: super::utf8::Utf8Assembler,
    pending: Vec<u8>,
    name: String,
    arguments: String,
    in_call: bool,
    calls: Vec<ParsedCall>,
    reasoning: bool,
    bytes: usize,
    max_bytes: usize,
    error: Option<ServeError>,
}

impl Partition {
    pub(crate) fn new(thinking: bool, max_bytes: usize) -> Self {
        Self {
            machine: Some(StateMachine::new(ParsingOptions {
                reasoning_initial_stage: thinking.then_some(ReasoningStage::Start),
                ..ParsingOptions::default()
            })),
            utf8: super::utf8::Utf8Assembler::new(),
            pending: vec![],
            name: String::new(),
            arguments: String::new(),
            in_call: false,
            calls: vec![],
            reasoning: thinking,
            bytes: 0,
            max_bytes,
            error: None,
        }
    }

    pub(crate) fn failure(&self) -> Option<&ServeError> {
        self.error.as_ref()
    }

    pub(crate) fn push(&mut self, bytes: &[u8], events: &mut Vec<PartitionEvent>) {
        if self.error.is_some() {
            return;
        }
        self.bytes = self.bytes.saturating_add(bytes.len());
        if self.bytes > self.max_bytes {
            self.error = Some(ServeError::server_error(
                "V4.1 output exceeds admitted byte limit",
            ));
            return;
        }
        let text = self.utf8.push(bytes);
        self.feed(&text, events);
    }

    fn feed(&mut self, text: &str, events: &mut Vec<PartitionEvent>) {
        self.pending.extend_from_slice(text.as_bytes());
        let actions = self.machine.as_mut().unwrap().feed(text);
        self.apply(actions, events);
    }

    fn apply(&mut self, actions: Vec<OutputActionSegment>, events: &mut Vec<PartitionEvent>) {
        for segment in actions {
            // Actions consume bytes of the entire retained source, not just
            // the latest feed, and never count Unicode scalar values.
            if segment.len > self.pending.len() {
                self.error = Some(ServeError::server_error(
                    "recipe action exceeds source bytes",
                ));
                return;
            }
            let source: Vec<_> = self.pending.drain(..segment.len).collect();
            let Ok(text) = std::str::from_utf8(&source) else {
                self.error = Some(ServeError::server_error(
                    "recipe action splits a UTF-8 character",
                ));
                return;
            };
            match segment.action {
                OutputAction::Reasoning => events.push(PartitionEvent::Reasoning(text.into())),
                OutputAction::Raw => {
                    self.close_reasoning(events);
                    events.push(PartitionEvent::Visible(text.into()));
                }
                OutputAction::ToolCallBegin => {
                    self.close_reasoning(events);
                    if self.in_call {
                        self.error = Some(ServeError::server_error("unfinished V4.1 tool call"));
                        return;
                    }
                    self.in_call = true;
                    self.name.clear();
                    self.arguments.clear();
                }
                OutputAction::ToolName => self.name.push_str(text),
                OutputAction::LabelToolCallArguments { label, .. } => {
                    self.arguments.push_str(label)
                }
                OutputAction::RawToolCallArguments { string } => {
                    if string {
                        let escaped = serde_json::to_string(text).unwrap();
                        self.arguments.push_str(&escaped[1..escaped.len() - 1]);
                    } else {
                        self.arguments.push_str(text);
                    }
                }
                OutputAction::ToolCallArgumentsEnd { output } => {
                    if let Some(output) = output {
                        self.arguments.push_str(output);
                    }
                    match serde_json::from_str(&self.arguments) {
                        Ok(arguments) if crate::model_request::tool_name_is_valid(&self.name) => {
                            self.calls.push(ParsedCall {
                                name: self.name.clone(),
                                arguments,
                            })
                        }
                        _ => {
                            self.error = Some(ServeError::server_error(
                                "invalid V4.1 tool arguments or name",
                            ));
                            return;
                        }
                    }
                    self.in_call = false;
                }
                OutputAction::ToSpace => events.push(PartitionEvent::Visible(" ".into())),
                OutputAction::Label(label) => events.push(PartitionEvent::Visible(label.into())),
                OutputAction::Skip if text.ends_with("</think>") => self.close_reasoning(events),
                OutputAction::Skip
                | OutputAction::SkipInvalid { .. }
                | OutputAction::StopSequence
                | OutputAction::ToolNameEnd => {}
            }
        }
    }

    fn close_reasoning(&mut self, events: &mut Vec<PartitionEvent>) {
        if self.reasoning {
            self.reasoning = false;
            events.push(PartitionEvent::ReasoningClosed);
        }
    }

    pub(crate) fn finish(mut self, events: &mut Vec<PartitionEvent>) -> Result<(), ServeError> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let text = self.utf8.finish();
        self.feed(&text, events);
        let actions = self.machine.take().unwrap().finish();
        self.apply(actions, events);
        if let Some(error) = self.error {
            return Err(error);
        }
        if self.in_call {
            return Err(ServeError::server_error("incomplete V4.1 tool call"));
        }
        events.extend(self.calls.into_iter().map(PartitionEvent::FunctionCall));
        Ok(())
    }
}
