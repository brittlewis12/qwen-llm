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

    /// Separator the renderer itself inserts between visible content and the
    /// call block, so it belongs to the call syntax rather than the visible
    /// text. The DS4 release encoder renders `content + "\n\n" + block`
    /// verbatim; Qwen templates trim content, so their separator needs no
    /// ownership rule.
    fn call_separator(self) -> Option<&'static str> {
        match self {
            Self::QwenXml => None,
            Self::DeepSeekDsml => Some("\n\n"),
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
    /// Literal completion text: UTF-8 assembly only, no control-marker parsing.
    RawText,
    K2Chat {
        effort: qwen_llm::k2_horizon_chat::Effort,
    },
    K2Tools {
        effort: qwen_llm::k2_horizon_chat::Effort,
        config: qwen_llm::k2_horizon_chat::tools::ToolConfig,
        max_bytes: usize,
    },
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

impl OutputProtocol {
    /// Whether this generation reasons, so the family renders history
    /// reasoning as model input (K2 and Muse always; Qwen and DeepSeek V4
    /// when the prompt opens `<think>`). A no-thinking generation renders
    /// history preclosed or dropped, where absent reasoning changes nothing.
    pub(crate) fn reasons(&self) -> bool {
        match self {
            Self::RawText => false,
            Self::K2Chat { .. } | Self::K2Tools { .. } | Self::MuseAtem { .. } => true,
            Self::Qwen {
                preopened_reasoning,
                ..
            } => *preopened_reasoning,
        }
    }
}

pub(crate) enum OutputPartition {
    Raw(Utf8Assembler),
    K2(super::partition_k2::K2Partition),
    K2Tools(super::partition_k2::K2ToolsPartition),
    Qwen(QwenOutputPartition),
    Muse(MuseAtemPartition),
}

impl OutputPartition {
    pub(crate) fn new(protocol: OutputProtocol) -> Self {
        match protocol {
            OutputProtocol::RawText => Self::Raw(Utf8Assembler::new()),
            OutputProtocol::K2Chat { effort } => {
                Self::K2(super::partition_k2::K2Partition::new(effort))
            }
            OutputProtocol::K2Tools {
                effort,
                config,
                max_bytes,
            } => Self::K2Tools(super::partition_k2::K2ToolsPartition::new(
                effort, config, max_bytes,
            )),
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
            Self::Raw(utf8) => {
                let text = utf8.push(bytes);
                if !text.is_empty() {
                    events.push(PartitionEvent::Visible(text));
                }
            }
            Self::Qwen(partition) => partition.push(bytes, events),
            Self::K2(partition) => partition.push(bytes, events),
            Self::K2Tools(partition) => partition.push(bytes, events),
            Self::Muse(partition) => partition.push(bytes, events),
        }
    }

    pub(crate) fn finish(
        self,
        end: GenerationEnd,
        events: &mut Vec<PartitionEvent>,
    ) -> Result<(), ServeError> {
        match self {
            Self::Raw(mut utf8) => {
                let text = utf8.finish();
                if !text.is_empty() {
                    events.push(PartitionEvent::Visible(text));
                }
                Ok(())
            }
            Self::Qwen(partition) => {
                partition.finish(events);
                Ok(())
            }
            Self::Muse(partition) => partition.finish(end, events),
            Self::K2(partition) => partition.finish(end, events),
            Self::K2Tools(partition) => partition.finish(end, events),
        }
    }

    pub(crate) fn abort(self, events: &mut Vec<PartitionEvent>) {
        match self {
            Self::Raw(mut utf8) => {
                let text = utf8.finish();
                if !text.is_empty() {
                    events.push(PartitionEvent::Visible(text));
                }
            }
            Self::Qwen(partition) => partition.abort(events),
            Self::Muse(partition) => partition.abort(events),
            Self::K2(partition) => partition.abort(events),
            Self::K2Tools(_) => {}
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
    /// The grammar's call separator seen right before the open marker: part
    /// of the call syntax when calls parse, visible text when they do not.
    tool_separator: &'static str,
    in_tool_span: bool,
}

#[cfg(test)]
mod raw_tests {
    use super::*;

    fn visible(events: Vec<PartitionEvent>) -> String {
        events
            .into_iter()
            .map(|event| match event {
                PartitionEvent::Visible(text) => text,
                other => panic!("raw text emitted non-visible event {other:?}"),
            })
            .collect()
    }

    #[test]
    fn raw_preserves_every_marker_and_utf8_across_all_chunk_sizes() {
        let text = "<think>literal</think><tool_call>{\"name\":\"x\"}</tool_call><|ifm|end_of_text|>\u{2192}\u{1f389}<thi";
        for chunk in 1..=text.len() {
            for end in [GenerationEnd::StopToken(1), GenerationEnd::TokenLimit] {
                let mut partition = OutputPartition::new(OutputProtocol::RawText);
                let mut events = Vec::new();
                for bytes in text.as_bytes().chunks(chunk) {
                    partition.push(bytes, &mut events);
                }
                partition.finish(end, &mut events).unwrap();
                assert_eq!(visible(events), text);
            }
        }
    }

    #[test]
    fn raw_invalid_and_dangling_utf8_flush_once_on_finish_or_abort() {
        let bytes = [b'x', 0xff, b'<', 0xf0, 0x9f];
        for abort in [false, true] {
            let mut partition = OutputPartition::new(OutputProtocol::RawText);
            let mut events = Vec::new();
            for byte in bytes {
                partition.push(&[byte], &mut events);
            }
            if abort {
                partition.abort(&mut events);
            } else {
                partition
                    .finish(GenerationEnd::TokenLimit, &mut events)
                    .unwrap();
            }
            assert_eq!(visible(events), String::from_utf8_lossy(&bytes));
        }
    }
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
            tool_separator: "",
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
        let marker = self.tool_grammar.open_marker();
        let separator = self.tool_grammar.call_separator();
        if let Some(index) = self.pending_visible.find(marker) {
            let mut prose = self.pending_visible[..index].to_owned();
            let calls = self.pending_visible[index..].to_owned();
            self.pending_visible.clear();
            if let Some(separator) = separator
                && prose.ends_with(separator)
            {
                prose.truncate(prose.len() - separator.len());
                self.tool_separator = separator;
            }
            if !prose.is_empty() {
                events.push(PartitionEvent::Visible(prose));
            }
            self.tool_buffer.push_str(&calls);
            self.in_tool_span = true;
            return;
        }
        // Also hold back a trailing separator that may precede the marker.
        let mut safe = safe_emit_len(&self.pending_visible, marker);
        if let Some(separator) = separator {
            safe = safe.min(safe_emit_len(
                &self.pending_visible,
                &format!("{separator}{marker}"),
            ));
        }
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
            events.push(PartitionEvent::Visible(format!(
                "{}{buffer}",
                self.tool_separator
            )));
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
            raw.push_str(self.tool_separator);
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

    /// The missing-reasoning diagnostic fires exactly for reasoning
    /// generations: every K2 chat/tools and Muse request, Qwen and DS4 when
    /// the prompt opens `<think>` (the backends' own predicates), never raw.
    #[test]
    fn reasons_tracks_thinking_generations() {
        use crate::open_responses::items::{QwenTemplate, parse_request};
        use crate::open_responses::render::{QwenGeneration, qwen_generation};
        use qwen_llm::k2_horizon_chat::Effort;
        use serde_json::json;

        let qwen = |preopened_reasoning| OutputProtocol::Qwen {
            preopened_reasoning,
            parse_tools: true,
            tool_grammar: ToolGrammar::QwenXml,
        };
        assert!(!OutputProtocol::RawText.reasons());
        assert!(
            OutputProtocol::K2Chat {
                effort: Effort::Low
            }
            .reasons()
        );
        assert!(
            OutputProtocol::MuseAtem {
                eos_token_id: 1,
                eot_token_id: 2,
                declared_tools: Vec::new()
            }
            .reasons()
        );
        assert!(qwen(true).reasons() && !qwen(false).reasons());

        let preopens = |template: QwenTemplate, body: serde_json::Value| {
            let request = parse_request(&body).unwrap();
            let bound = crate::open_responses::bind_qwen_request(&request, template, true).unwrap();
            qwen_generation(&bound) == QwenGeneration::PreOpen
        };
        let input = json!([{"role": "user", "content": "q"}]);
        for (template, x_qwen, effort, expected) in [
            (QwenTemplate::Qwen36, json!({}), None, true),
            (
                QwenTemplate::Qwen36,
                json!({"no_thinking": true}),
                None,
                false,
            ),
            (QwenTemplate::Qwen35, json!({}), None, false),
            (QwenTemplate::Qwen35, json!({"thinking": true}), None, true),
            (QwenTemplate::Qwen38, json!({}), None, true),
            (QwenTemplate::Qwen38, json!({}), Some("none"), false),
            (QwenTemplate::Generic, json!({}), None, false),
        ] {
            let mut body = json!({"model": "m", "input": input, "x_qwen": x_qwen});
            if let Some(effort) = effort {
                body["reasoning"] = json!({"effort": effort});
            }
            assert_eq!(
                preopens(template, body),
                expected,
                "{} {x_qwen} {effort:?}",
                template.label()
            );
        }
        for (effort, expected) in [(None, false), (Some("high"), true)] {
            let mut body = json!({"model": "ds", "input": input});
            if let Some(effort) = effort {
                body["reasoning"] = json!({"effort": effort});
            }
            let request = parse_request(&body).unwrap();
            assert_eq!(
                crate::serve::render_ds4::preopens_reasoning(&request).unwrap(),
                expected
            );
        }
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

    /// The DSML call separator is withheld from visible text only when a
    /// call actually parses. A malformed block, a token-limit cut, an abort
    /// and plain text ending in newlines all keep every visible byte, at
    /// every chunking.
    #[test]
    fn dsml_separator_is_visible_unless_a_call_parses() {
        let dsml = ToolGrammar::DeepSeekDsml;
        let protocol = OutputProtocol::Qwen {
            preopened_reasoning: false,
            parse_tools: true,
            tool_grammar: dsml,
        };
        let visible = |events: &[PartitionEvent]| {
            events
                .iter()
                .filter_map(|event| match event {
                    PartitionEvent::Visible(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect::<String>()
        };
        let calls = |events: &[PartitionEvent]| {
            events
                .iter()
                .filter(|event| matches!(event, PartitionEvent::FunctionCall(_)))
                .count()
        };
        let marker = dsml.open_marker();
        let malformed = format!("Checking.\n\n{marker}\n<not a call>");
        let valid = format!(
            "Checking.\n\n{marker}\n<｜DSML｜invoke name=\"ping\">\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>"
        );
        for chunk in 1..=valid.len() {
            let pieces = |text: &str| -> Vec<Vec<u8>> {
                text.as_bytes().chunks(chunk).map(<[u8]>::to_vec).collect()
            };
            let drive = |text: &str, end: Option<GenerationEnd>| {
                let mut partition = OutputPartition::new(protocol.clone());
                let mut events = Vec::new();
                for piece in pieces(text) {
                    partition.push(&piece, &mut events);
                }
                match end {
                    Some(end) => partition.finish(end, &mut events).unwrap(),
                    None => partition.abort(&mut events),
                }
                events
            };
            for end in [
                Some(GenerationEnd::StopToken(1)),
                Some(GenerationEnd::TokenLimit),
                None,
            ] {
                // Plain text ending in newlines, and a malformed block.
                for text in ["Plain answer.\n\n", "Plain answer.\n", malformed.as_str()] {
                    let events = drive(text, end);
                    assert_eq!(visible(&events), text, "chunk {chunk} {end:?} {text:?}");
                    assert_eq!(calls(&events), 0);
                }
            }
            let events = drive(&valid, Some(GenerationEnd::StopToken(1)));
            assert_eq!(visible(&events), "Checking.", "chunk {chunk}");
            assert_eq!(calls(&events), 1);
            // An abort mid-block keeps the separator with the raw bytes.
            let cut = &valid[..valid.find("</｜DSML｜invoke>").unwrap()];
            assert_eq!(visible(&drive(cut, None)), cut, "chunk {chunk}");
        }
    }
}
