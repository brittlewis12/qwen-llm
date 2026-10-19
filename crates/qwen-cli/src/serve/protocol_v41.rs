//! Adapter for the revision-pinned, MIT-licensed official DeepSeek recipe.
//! Serving keeps the existing Responses subset (auto/allowed tools, text only).
//! Omitted effort selects chat; low/high/xhigh/max enable official thinking.
//! Cancellation discards withheld tool calls rather than manufacturing a finish.
use super::items::{ServeError, ServeRequest};
use super::output_partition::GenerationEnd;
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
    use crate::serve::output_partition::{OutputPartition, OutputProtocol};
    use serde_json::json;

    fn tool_fixture() -> String {
        concat!(
            "<\u{ff5c}DSML\u{ff5c} calls>\n",
            "<\u{ff5c}DSML\u{ff5c}invoke name=\"first\">\n",
            "</\u{ff5c}DSML\u{ff5c}invoke>\n",
            "<\u{ff5c}DSML\u{ff5c}invoke name=\"lookup\">\n",
            "<\u{ff5c}DSML\u{ff5c}parameter name=\"s\" string=\"true\">caf\u{e9}\n\"",
            "</\u{ff5c}DSML\u{ff5c}parameter>\n",
            "<\u{ff5c}DSML\u{ff5c}parameter name=\"n\" string=\"false\">2",
            "</\u{ff5c}DSML\u{ff5c}parameter>\n",
            "</\u{ff5c}DSML\u{ff5c}invoke>\n",
            "</\u{ff5c}DSML\u{ff5c} calls>"
        )
        .into()
    }

    fn output_partition(thinking: bool, max_bytes: usize) -> OutputPartition {
        OutputPartition::new(OutputProtocol::DeepSeekV41 {
            thinking,
            max_bytes: crate::serve::output_memory::OutputByteCeiling::from_bytes(max_bytes)
                .unwrap(),
        })
    }

    fn calls(events: &[PartitionEvent]) -> Vec<&ParsedCall> {
        events
            .iter()
            .filter_map(|event| match event {
                PartitionEvent::FunctionCall(call) => Some(call),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn v41_tool_termination_every_byte_cut_and_split() {
        let output = tool_fixture();
        let opener_len = "<\u{ff5c}DSML\u{ff5c}".len();
        for thinking in [false, true] {
            for cut in 0..=output.len() {
                for split in 0..=cut {
                    for end in [
                        Some(GenerationEnd::TokenLimit),
                        Some(GenerationEnd::StopToken(1)),
                        None,
                    ] {
                        let mut partition = output_partition(thinking, 100_000);
                        let mut events = vec![];
                        partition.push(&output.as_bytes()[..split], &mut events);
                        partition.push(&output.as_bytes()[split..cut], &mut events);
                        assert!(partition.failure().is_none(), "cut={cut} split={split}");
                        assert!(calls(&events).is_empty());
                        if let Some(end) = end {
                            let result = partition.finish(end, &mut events);
                            assert_eq!(
                                result.is_ok(),
                                end.is_token_limit() || cut < opener_len || cut == output.len(),
                                "thinking={thinking} cut={cut} split={split} end={end:?}: {result:?}"
                            );
                        } else {
                            let before = events.clone();
                            partition.abort(&mut events);
                            assert_eq!(events, before);
                        }
                        if cut < opener_len && end.is_some() {
                            let text = published_text(&events, thinking);
                            assert_eq!(text, String::from_utf8_lossy(&output.as_bytes()[..cut]));
                        } else {
                            assert!(
                                events.iter().all(|event| matches!(
                                    event,
                                    PartitionEvent::ReasoningClosed
                                        | PartitionEvent::FunctionCall(_)
                                )),
                                "cut={cut} split={split} end={end:?}: {events:?}"
                            );
                        }
                        let calls = calls(&events);
                        if cut == output.len() && end.is_some() {
                            assert_eq!(calls.len(), 2);
                            assert_eq!(calls[0].name, "first");
                            assert!(calls[0].arguments.is_empty());
                            assert_eq!(calls[1].name, "lookup");
                            assert_eq!(
                                calls[1].arguments,
                                json!({"s":"caf\u{e9}\n\"", "n":2})
                                    .as_object()
                                    .unwrap()
                                    .clone()
                            );
                        } else {
                            assert!(calls.is_empty(), "cut={cut} split={split} end={end:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn v41_recipe_alias_and_completed_call_before_partial_are_atomic() {
        let output = tool_fixture().replace(" calls>", "tool_calls>");
        let opener_len = "<\u{ff5c}DSML\u{ff5c}".len();
        for cut in 0..=output.len() {
            for end in [GenerationEnd::TokenLimit, GenerationEnd::StopToken(1)] {
                let mut partition = output_partition(false, 100_000);
                let mut events = vec![];
                for byte in &output.as_bytes()[..cut] {
                    partition.push(&[*byte], &mut events);
                }
                if let OutputPartition::DeepSeekV41(inner) = &partition {
                    if cut >= output.find("lookup").unwrap() {
                        assert!(
                            !inner.calls.is_empty(),
                            "fixture has an earlier completed call"
                        );
                    }
                }
                let result = partition.finish(end, &mut events);
                assert_eq!(
                    result.is_ok(),
                    end.is_token_limit() || cut < opener_len || cut == output.len()
                );
                assert_eq!(
                    calls(&events).len(),
                    if cut == output.len() { 2 } else { 0 }
                );
                if cut < opener_len {
                    assert_eq!(
                        published_text(&events, false),
                        String::from_utf8_lossy(&output.as_bytes()[..cut])
                    );
                } else {
                    assert!(
                        events
                            .iter()
                            .all(|event| matches!(event, PartitionEvent::FunctionCall(_)))
                    );
                }
            }
        }
    }

    #[test]
    fn v41_stored_parse_and_byte_limit_errors_survive_all_finishes() {
        let valid = tool_fixture();
        for output in [
            valid.replace("name=\"lookup\"", "name=\"bad name\""),
            valid.replace("false\">2", "false\">not_json"),
        ] {
            for split in 0..=output.len() {
                for end in [GenerationEnd::TokenLimit, GenerationEnd::StopToken(1)] {
                    let mut partition = output_partition(false, 100_000);
                    let mut events = vec![];
                    partition.push(&output.as_bytes()[..split], &mut events);
                    partition.push(&output.as_bytes()[split..], &mut events);
                    let failure = partition.failure().cloned().expect("stored parse error");
                    partition.push(b"ignored", &mut events);
                    assert_eq!(partition.finish(end, &mut events).unwrap_err(), failure);
                    assert!(events.is_empty());
                }
            }
        }
        for end in [GenerationEnd::TokenLimit, GenerationEnd::StopToken(1)] {
            // A zero byte ceiling is invalid backend configuration now; the
            // request profile refuses it before partition construction.
            for cut in 1..=valid.len() {
                let mut partition = output_partition(false, cut);
                let mut events = vec![];
                partition.push(&valid.as_bytes()[..cut], &mut events);
                assert!(partition.failure().is_none());
                partition.push(b"x", &mut events);
                let failure = partition
                    .failure()
                    .cloned()
                    .expect("stored byte limit error");
                partition.push(b"ignored", &mut events);
                assert_eq!(partition.finish(end, &mut events).unwrap_err(), failure);
                assert!(events.is_empty());
            }
        }
    }

    #[test]
    fn v41_stored_typed_memory_errors_take_precedence_over_truncation() {
        for headroom in [None, Some(1)] {
            let failure =
                crate::serve::transport_memory::admit_resident_transport(2, headroom).unwrap_err();
            for end in [GenerationEnd::TokenLimit, GenerationEnd::StopToken(1)] {
                let mut partition = Partition::new(false, 100_000);
                let mut events = vec![];
                partition.push("<\u{ff5c}DSML\u{ff5c} calls>".as_bytes(), &mut events);
                // Inject a typed refusal without adding an admission policy.
                partition.error = Some(failure.clone());
                assert_eq!(partition.failure(), Some(&failure));
                assert_eq!(partition.finish(end, &mut events).unwrap_err(), failure);
                assert!(events.is_empty());
            }
        }
    }

    fn published_text(events: &[PartitionEvent], thinking: bool) -> String {
        events
            .iter()
            .map(|event| match event {
                PartitionEvent::Reasoning(text) if thinking => text.as_str(),
                PartitionEvent::Visible(text) if !thinking => text.as_str(),
                other => panic!("unexpected {other:?}"),
            })
            .collect()
    }

    #[test]
    fn v41_plain_text_finish_preserves_recipe_suffix_and_utf8_semantics() {
        let mut cases: Vec<Vec<u8>> = [
            "answer",
            "caf\u{e9}\n",
            "answer\n\n",
            "literal <x>",
            "answer <",
            "answer </",
            "answer <nottool",
            "answer <\u{ff5c}DSMX",
            "answer\n<",
            "answer\n</",
            "answer\n</think",
            "answer <think>",
        ]
        .into_iter()
        .map(|text| text.as_bytes().to_vec())
        .collect();
        for marker in ["<\u{ff5c}DSML\u{ff5c}", "</think>"] {
            for cut in 1..marker.len() {
                let mut bytes = b"safe\n".to_vec();
                bytes.extend_from_slice(&marker.as_bytes()[..cut]);
                cases.push(bytes);
            }
        }
        for character in ["\u{e9}", "\u{1f389}"] {
            for cut in 1..character.len() {
                let mut bytes = b"safe".to_vec();
                bytes.extend_from_slice(&character.as_bytes()[..cut]);
                cases.push(bytes);
            }
        }
        for thinking in [false, true] {
            for bytes in &cases {
                for split in 0..=bytes.len() {
                    for end in [GenerationEnd::TokenLimit, GenerationEnd::StopToken(1)] {
                        let mut partition = output_partition(thinking, 100_000);
                        let mut events = vec![];
                        partition.push(&bytes[..split], &mut events);
                        partition.push(&bytes[split..], &mut events);
                        partition.finish(end, &mut events).unwrap();
                        assert_eq!(
                            published_text(&events, thinking),
                            String::from_utf8_lossy(bytes),
                            "split={split} end={end:?}"
                        );
                    }
                }
            }
        }
    }

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
            partition
                .finish(GenerationEnd::StopToken(1), &mut events)
                .unwrap();
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
            max_bytes: crate::serve::output_memory::OutputByteCeiling::from_bytes(1024).unwrap(),
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

    #[test]
    fn v41_shared_prelude_preserves_official_bytes_across_tasks_and_modes() {
        use deepseek_recipe_encoding::v4::{ASSISTANT_SP_TOKEN, USER_SP_TOKEN};
        let marker_text = format!("literal {ASSISTANT_SP_TOKEN}<think>{USER_SP_TOKEN}");
        let profile = profile(OutputLimits::TEST);
        for effort in ["none", "low", "high", "xhigh", "max"] {
            for (system, tools) in [(false, false), (true, false), (false, true), (true, true)] {
                let mut shared = None;
                for task in ["task A", "different task B"] {
                    let user = format!("{task} {marker_text}");
                    let mut body = json!({"model":"m", "input":user,
                        "reasoning":{"effort":effort}});
                    if system {
                        body["instructions"] = json!(marker_text);
                    }
                    if tools {
                        body["tools"] = json!([{"type":"function", "name":"lookup",
                            "description":marker_text, "parameters":{"type":"object"}}]);
                    }
                    let request = profile.parse(&body).unwrap();
                    let mut official = Conversation {
                        thinking_mode: effort != "none",
                        reasoning_effort: thinking(&request).unwrap(),
                        ..Default::default()
                    };
                    if system {
                        official.messages.push(InputMessage::System {
                            content: marker_text.clone(),
                        });
                    }
                    if tools {
                        let tool = &request.model_request.tools[0];
                        official.tools.push(ToolDefinition {
                            name: tool.name.clone(),
                            description: tool.description.clone(),
                            parameters: tool.parameters.clone(),
                            strict: tool.strict,
                        });
                    }
                    let encoder = DeepseekV41Encoding::new();
                    let prelude = encoder.render_conversation(&official).prompt;
                    official.messages.push(InputMessage::User {
                        content: user.clone(),
                        image_sources: vec![],
                    });
                    let (prompt, boundaries) = render_with_boundaries(&request).unwrap();
                    assert_eq!(prompt, encoder.render_conversation(&official).prompt);
                    let header = boundaries.generation_header_start.unwrap();
                    assert!(prompt[..header].ends_with(&user));
                    if system || tools {
                        let end = boundaries.shared_prefix_end.unwrap();
                        assert!(end > 0 && end < header);
                        let prefix = &prompt[..end];
                        assert_eq!(Some(prefix), prelude.strip_suffix(&prompt[header..]));
                        assert!(prefix.contains(&marker_text));
                        assert_eq!(prefix.contains("Available Tool Schemas"), tools);
                        assert_eq!(prefix.contains("Reasoning Effort:"), effort != "none");
                        if let Some(previous) = &shared {
                            assert_eq!(prefix, previous);
                        }
                        shared = Some(prefix.to_owned());
                    } else {
                        assert_eq!(boundaries.shared_prefix_end, None);
                    }
                }
            }
        }
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
    render_with_boundaries(request).map(|(prompt, _)| prompt)
}

pub(crate) fn render_with_boundaries(
    request: &ServeRequest,
) -> Result<(String, super::http::PromptBoundaries), ServeError> {
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
    let encoder = DeepseekV41Encoding::new();
    let prelude = (!conversation.messages.is_empty() || !conversation.tools.is_empty())
        .then(|| encoder.render_conversation(&conversation).prompt);
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
    let prompt = encoder.render_conversation(&conversation).prompt;
    // The pinned recipe unconditionally appends this exact generation suffix
    // (tool choice is Auto here). Never search transcript text for markers.
    use deepseek_recipe_encoding::v4::{
        ASSISTANT_SP_TOKEN, THINKING_END_TOKEN, THINKING_START_TOKEN,
    };
    let header = format!(
        "{ASSISTANT_SP_TOKEN}{}",
        if conversation.thinking_mode {
            THINKING_START_TOKEN
        } else {
            THINKING_END_TOKEN
        }
    );
    let transcript_len = prompt
        .strip_suffix(&header)
        .ok_or_else(|| ServeError::server_error("V4.1 recipe generation suffix changed"))?
        .len();
    let shared_prefix_end = prelude
        .as_deref()
        .and_then(|prelude| prelude.strip_suffix(&header))
        .filter(|prefix| {
            !prefix.is_empty() && prefix.len() < transcript_len && prompt.starts_with(prefix)
        })
        .map(str::len);
    let boundaries = super::http::PromptBoundaries {
        generation_header_start: Some(transcript_len),
        shared_prefix_end,
    };
    Ok((prompt, boundaries))
}

pub(crate) struct Partition {
    machine: Option<StateMachine>,
    utf8: super::utf8::Utf8Assembler,
    pending: Vec<u8>,
    name: String,
    arguments: String,
    in_call: bool,
    in_tool_block: bool,
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
            in_tool_block: false,
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
                OutputAction::Skip if text.ends_with("<\u{ff5c}DSML\u{ff5c}") => {
                    self.in_tool_block = true;
                }
                OutputAction::Skip
                    if matches!(
                        text,
                        "</\u{ff5c}DSML\u{ff5c} calls>" | "</\u{ff5c}DSML\u{ff5c}tool_calls>"
                    ) =>
                {
                    self.in_tool_block = false;
                }
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

    pub(crate) fn finish(
        mut self,
        end: GenerationEnd,
        events: &mut Vec<PartitionEvent>,
    ) -> Result<(), ServeError> {
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        // The recipe ends arguments on the generic DSML close prefix, not
        // the whole invoke delimiter. Commit only a closed terminal block,
        // as GLM/K2/Muse do: a partial later call withholds earlier calls too.
        // Only the recipe's recognized DSML opener confirms a tool block.
        // Outside it, preserve ordinary suffix and UTF-8 finalization, even
        // for ambiguous marker prefixes such as '<' or '</think'.
        if self.in_call || self.in_tool_block {
            return if end.is_token_limit() {
                Ok(())
            } else {
                Err(ServeError::server_error("incomplete V4.1 tool call"))
            };
        }
        let text = self.utf8.finish();
        self.feed(&text, events);
        if let Some(error) = self.error.take() {
            return Err(error);
        }
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
