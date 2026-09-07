//! Family-owned decoding grammar over exact generated token bytes.

use super::items::ServeError;
use super::partition::{PartitionEvent, StreamPartition, safe_emit_len};
use super::partition_muse::MuseAtemPartition;
use super::tool_parse::parse_emission;
use super::utf8::Utf8Assembler;

/// How a family's tool calls appear in the visible stream: the marker that
/// opens the first call block, and the parser for the buffered block.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ToolGrammar {
    /// `<tool_call>{json}</tool_call>` blocks (Qwen released templates).
    QwenXml,
    /// `<｜DSML｜tool_calls>` … `</｜DSML｜tool_calls>` (DeepSeek V4).
    DeepSeekDsml,
}

impl ToolGrammar {
    fn open_marker(self) -> &'static str {
        match self {
            Self::QwenXml => "<tool_call>",
            Self::DeepSeekDsml => super::tool_parse::DSML_TOOL_CALLS_OPEN,
        }
    }

    fn parse(self, buffer: &str) -> super::tool_parse::ParsedEmission {
        match self {
            Self::QwenXml => parse_emission(buffer),
            Self::DeepSeekDsml => super::tool_parse::parse_dsml_emission(buffer),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GenerationEnd {
    StopToken(i32),
    TokenLimit,
}

impl GenerationEnd {
    pub(crate) fn is_token_limit(self) -> bool {
        matches!(self, Self::TokenLimit)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum OutputProtocol {
    /// `<think>`-partitioned text with an optional tool block; the Qwen
    /// families and DeepSeek V4 share this shape and differ in grammar.
    Qwen {
        preopened_reasoning: bool,
        parse_tools: bool,
        tool_grammar: ToolGrammar,
    },
    MuseAtem {
        eos_token_id: i32,
        eot_token_id: i32,
        declared_tools: Vec<String>,
    },
}

pub(crate) enum OutputPartition {
    Qwen(QwenOutputPartition),
    Muse(MuseAtemPartition),
}

impl OutputPartition {
    pub(crate) fn new(protocol: OutputProtocol) -> Self {
        match protocol {
            OutputProtocol::Qwen {
                preopened_reasoning,
                parse_tools,
                tool_grammar,
            } => Self::Qwen(QwenOutputPartition::new(
                preopened_reasoning,
                parse_tools,
                tool_grammar,
            )),
            OutputProtocol::MuseAtem {
                eos_token_id,
                eot_token_id,
                declared_tools,
            } => Self::Muse(MuseAtemPartition::new(
                eos_token_id,
                eot_token_id,
                declared_tools,
            )),
        }
    }

    pub(crate) fn push(&mut self, bytes: &[u8], events: &mut Vec<PartitionEvent>) {
        match self {
            Self::Qwen(partition) => partition.push(bytes, events),
            Self::Muse(partition) => partition.push(bytes, events),
        }
    }

    pub(crate) fn finish(
        self,
        end: GenerationEnd,
        events: &mut Vec<PartitionEvent>,
    ) -> Result<(), ServeError> {
        match self {
            Self::Qwen(partition) => {
                partition.finish(events);
                Ok(())
            }
            Self::Muse(partition) => partition.finish(end, events),
        }
    }

    pub(crate) fn abort(self, events: &mut Vec<PartitionEvent>) {
        match self {
            Self::Qwen(partition) => partition.abort(events),
            Self::Muse(partition) => partition.abort(events),
        }
    }
}

pub(crate) struct QwenOutputPartition {
    reasoning: StreamPartition,
    utf8: Utf8Assembler,
    parse_tools: bool,
    tool_grammar: ToolGrammar,
    pending_visible: String,
    tool_buffer: String,
    in_tool_span: bool,
}

impl QwenOutputPartition {
    fn new(preopened_reasoning: bool, parse_tools: bool, tool_grammar: ToolGrammar) -> Self {
        Self {
            reasoning: if preopened_reasoning {
                StreamPartition::with_preopened_reasoning()
            } else {
                StreamPartition::new()
            },
            utf8: Utf8Assembler::new(),
            parse_tools,
            tool_grammar,
            pending_visible: String::new(),
            tool_buffer: String::new(),
            in_tool_span: false,
        }
    }

    fn push(&mut self, bytes: &[u8], events: &mut Vec<PartitionEvent>) {
        let text = self.utf8.push(bytes);
        if !text.is_empty() {
            self.push_text(&text, events);
        }
    }

    fn push_text(&mut self, text: &str, events: &mut Vec<PartitionEvent>) {
        let mut split = Vec::new();
        self.reasoning.push(text, &mut split);
        self.route_split(split, events);
    }

    fn route_split(&mut self, split: Vec<PartitionEvent>, events: &mut Vec<PartitionEvent>) {
        for event in split {
            match event {
                PartitionEvent::Visible(text) if self.parse_tools => {
                    self.push_visible(&text, events)
                }
                PartitionEvent::FunctionCall(_) => {
                    unreachable!("reasoning splitter cannot parse calls")
                }
                event => events.push(event),
            }
        }
    }

    fn push_visible(&mut self, text: &str, events: &mut Vec<PartitionEvent>) {
        if self.in_tool_span {
            self.tool_buffer.push_str(text);
            return;
        }
        self.pending_visible.push_str(text);
        if let Some(index) = self.pending_visible.find(self.tool_grammar.open_marker()) {
            let prose = self.pending_visible[..index].to_owned();
            let calls = self.pending_visible[index..].to_owned();
            self.pending_visible.clear();
            if !prose.is_empty() {
                events.push(PartitionEvent::Visible(prose));
            }
            self.tool_buffer.push_str(&calls);
            self.in_tool_span = true;
            return;
        }
        let safe = safe_emit_len(&self.pending_visible, self.tool_grammar.open_marker());
        if safe > 0 {
            let text = self.pending_visible[..safe].to_owned();
            self.pending_visible.drain(..safe);
            events.push(PartitionEvent::Visible(text));
        }
    }

    fn flush_reasoning(&mut self, events: &mut Vec<PartitionEvent>) {
        let tail = self.utf8.finish();
        if !tail.is_empty() {
            self.push_text(&tail, events);
        }
        let mut split = Vec::new();
        std::mem::replace(&mut self.reasoning, StreamPartition::new()).finish(&mut split);
        self.route_split(split, events);
    }

    fn finish(mut self, events: &mut Vec<PartitionEvent>) {
        self.flush_reasoning(events);
        if !self.parse_tools {
            return;
        }
        if !self.pending_visible.is_empty() {
            events.push(PartitionEvent::Visible(std::mem::take(
                &mut self.pending_visible,
            )));
        }
        if !self.in_tool_span {
            return;
        }
        let buffer = std::mem::take(&mut self.tool_buffer);
        let parsed = self.tool_grammar.parse(&buffer);
        if parsed.calls.is_empty() {
            events.push(PartitionEvent::Visible(buffer));
            return;
        }
        if !parsed.visible.is_empty() {
            events.push(PartitionEvent::Visible(parsed.visible));
        }
        events.extend(parsed.calls.into_iter().map(PartitionEvent::FunctionCall));
    }

    fn abort(mut self, events: &mut Vec<PartitionEvent>) {
        self.flush_reasoning(events);
        let mut raw = std::mem::take(&mut self.pending_visible);
        if self.in_tool_span {
            raw.push_str(&self.tool_buffer);
        }
        if !raw.is_empty() {
            events.push(PartitionEvent::Visible(raw));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(protocol: OutputProtocol, chunks: &[&[u8]], end: GenerationEnd) -> Vec<PartitionEvent> {
        let mut partition = OutputPartition::new(protocol);
        let mut events = Vec::new();
        for chunk in chunks {
            partition.push(chunk, &mut events);
        }
        partition.finish(end, &mut events).unwrap();
        events
    }

    #[test]
    fn qwen_partition_owns_reasoning_and_tool_syntax() {
        let events = run(
            OutputProtocol::Qwen {
                preopened_reasoning: false,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
            },
            &[
                b"<think>plan</think>answer<tool_",
                b"call>\n<function=ping>\n</function>\n</tool_call>",
            ],
            GenerationEnd::StopToken(1),
        );
        assert!(matches!(events[0], PartitionEvent::Reasoning(_)));
        assert!(
            events
                .iter()
                .any(|event| matches!(event, PartitionEvent::Visible(text) if text == "answer"))
        );
        assert!(events.iter().any(
            |event| matches!(event, PartitionEvent::FunctionCall(call) if call.name == "ping")
        ));
    }

    #[test]
    fn qwen_plain_protocol_never_interprets_tool_like_text() {
        let call = b"<tool_call>\n<function=ping>\n</function>\n</tool_call>";
        let events = run(
            OutputProtocol::Qwen {
                preopened_reasoning: false,
                parse_tools: false,
                tool_grammar: ToolGrammar::QwenXml,
            },
            &[call],
            GenerationEnd::StopToken(1),
        );
        assert_eq!(
            events,
            [PartitionEvent::Visible(
                String::from_utf8_lossy(call).into()
            )]
        );
    }

    #[test]
    fn preopened_qwen_reasoning_can_close_into_owned_tool_syntax() {
        let events = run(
            OutputProtocol::Qwen {
                preopened_reasoning: true,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
            },
            &[
                b"plan</think>answer<tool_call>\n<function=ping>\n",
                b"</function>\n</tool_call>",
            ],
            GenerationEnd::StopToken(1),
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, PartitionEvent::Reasoning(text) if text == "plan"))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, PartitionEvent::ReasoningClosed))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event, PartitionEvent::Visible(text) if text == "answer"))
        );
        assert!(events.iter().any(
            |event| matches!(event, PartitionEvent::FunctionCall(call) if call.name == "ping")
        ));
    }

    #[test]
    fn preopened_plain_protocol_leaves_tool_syntax_visible() {
        let events = run(
            OutputProtocol::Qwen {
                preopened_reasoning: true,
                parse_tools: false,
                tool_grammar: ToolGrammar::QwenXml,
            },
            &[b"plan</think><tool_call>\n<function=ping>\n</function>\n</tool_call>"],
            GenerationEnd::StopToken(1),
        );
        let visible = events
            .iter()
            .filter_map(|event| match event {
                PartitionEvent::Visible(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(
            visible,
            "<tool_call>\n<function=ping>\n</function>\n</tool_call>"
        );
        assert!(
            events
                .iter()
                .all(|event| !matches!(event, PartitionEvent::FunctionCall(_)))
        );
    }

    #[test]
    fn abort_flushes_qwen_ambiguity_but_never_muse_protocol_bytes() {
        let mut qwen = OutputPartition::new(OutputProtocol::Qwen {
            preopened_reasoning: false,
            parse_tools: true,
            tool_grammar: ToolGrammar::QwenXml,
        });
        let mut qwen_events = Vec::new();
        qwen.push(b"answer<tool_", &mut qwen_events);
        qwen.abort(&mut qwen_events);
        let visible = qwen_events
            .iter()
            .filter_map(|event| match event {
                PartitionEvent::Visible(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(visible, "answer<tool_");

        let mut muse = OutputPartition::new(OutputProtocol::MuseAtem {
            eos_token_id: 1,
            eot_token_id: 2,
            declared_tools: Vec::new(),
        });
        let mut muse_events = Vec::new();
        muse.push(b" to=user<|message|>answer<", &mut muse_events);
        muse.abort(&mut muse_events);
        let visible = muse_events
            .iter()
            .filter_map(|event| match event {
                PartitionEvent::Visible(text) => Some(text.as_str()),
                _ => None,
            })
            .collect::<String>();
        assert_eq!(visible, "answer");
        assert!(muse_events.iter().all(|event| match event {
            PartitionEvent::Reasoning(text) | PartitionEvent::Visible(text) => {
                !text.contains("<|") && !text.contains("<atem:")
            }
            PartitionEvent::ReasoningClosed => true,
            PartitionEvent::FunctionCall(_) => false,
        }));
    }
}
