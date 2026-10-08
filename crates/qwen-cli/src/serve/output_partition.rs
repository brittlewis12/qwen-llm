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
    /// GLM-5.3-Flash text chat: reasoning pre-opened by the prompt.
    Glm5NextChat,
    /// GLM-5.3-Flash chat with declared tools: calls after `</think>`.
    Glm5NextTools {
        definitions: Vec<qwen_llm::glm5_next_chat::ToolDefinition>,
        max_bytes: usize,
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
            Self::K2Chat { .. }
            | Self::K2Tools { .. }
            | Self::MuseAtem { .. }
            | Self::Glm5NextChat
            | Self::Glm5NextTools { .. } => true,
            Self::Qwen {
                preopened_reasoning,
                ..
            } => *preopened_reasoning,
        }
    }
}

pub(crate) enum OutputPartition {
    Raw(Utf8Assembler),
    Preopened(super::partition_preopened::PreopenedPartition),
    K2Tools(super::partition_k2::K2ToolsPartition),
    Glm5NextTools(super::partition_glm5_next::Glm5NextToolsPartition),
    Qwen(QwenOutputPartition),
    Muse(MuseAtemPartition),
}

impl OutputPartition {
    pub(crate) fn new(protocol: OutputProtocol) -> Self {
        Self::with_headroom(
            protocol,
            qwen_llm::metal::MetalContext::process_limit_bytes_remaining,
        )
    }

    /// [`Self::new`], with the process headroom its admitted tool buffers
    /// (Qwen/DS4, GLM) check (serve passes the backend's; tests inject).
    pub(crate) fn with_headroom(protocol: OutputProtocol, headroom: super::http::Headroom) -> Self {
        match protocol {
            OutputProtocol::RawText => Self::Raw(Utf8Assembler::new()),
            OutputProtocol::K2Chat { effort } => {
                Self::Preopened(super::partition_k2::k2_partition(effort))
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
            } => Self::Qwen(QwenOutputPartition::with_headroom(
                preopened_reasoning,
                parse_tools,
                tool_grammar,
                headroom,
            )),
            OutputProtocol::Glm5NextChat => Self::Preopened(super::render_glm5_next::partition()),
            OutputProtocol::Glm5NextTools {
                definitions,
                max_bytes,
            } => Self::Glm5NextTools(
                super::partition_glm5_next::Glm5NextToolsPartition::with_headroom(
                    definitions,
                    max_bytes,
                    headroom,
                ),
            ),
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
            Self::Preopened(partition) => partition.push(bytes, events),
            Self::K2Tools(partition) => partition.push(bytes, events),
            Self::Glm5NextTools(partition) => partition.push(bytes, events),
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
            Self::Qwen(partition) => partition.finish(events),
            Self::Muse(partition) => partition.finish(end, events),
            Self::Preopened(partition) => partition.finish(end, events),
            Self::K2Tools(partition) => partition.finish(end, events),
            Self::Glm5NextTools(partition) => partition.finish(end, events),
        }
    }

    /// A stored admission refusal or parse failure that already decides the
    /// turn (generation can stop; `finish` returns it).
    pub(crate) fn failure(&self) -> Option<&ServeError> {
        match self {
            Self::Qwen(partition) => partition.failure(),
            Self::Glm5NextTools(partition) => partition.failure(),
            Self::Raw(_) | Self::Preopened(_) | Self::K2Tools(_) | Self::Muse(_) => None,
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
            Self::Preopened(partition) => partition.abort(events),
            Self::K2Tools(_) | Self::Glm5NextTools(_) => {}
        }
    }
}

/// Qwen-XML and DeepSeek-V4-DSML output: the reasoning splitter, then
/// visible text with the tool span held until the turn ends and parsed
/// whole.
///
/// Memory (map #14 packet 2, as GLM's tool block): before the tool buffer
/// grows past its capacity, the capacity grows by whole
/// [`super::partition_glm5_next::ADMISSION_STEP`]s and the process must have
/// headroom for the block's whole outstanding peak at the new capacity,
/// `tool_block_peak_bytes` (GLM's 256x model; Qwen XML and DSML measured
/// within it, worst 147.5 for the same shapes) less what the buffer holds;
/// again before the parse. A refusal is stored and ends the turn with its
/// typed error (503 `memory_admission_denied`; 500 when the signal is
/// unavailable), never a raw-text fallback.
pub(crate) struct QwenOutputPartition {
    reasoning: StreamPartition,
    utf8: Utf8Assembler,
    parse_tools: bool,
    tool_grammar: ToolGrammar,
    pending_visible: String,
    tool_buffer: String,
    failure: Option<ServeError>,
    headroom: super::http::Headroom,
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
    fn with_headroom(
        preopened_reasoning: bool,
        parse_tools: bool,
        tool_grammar: ToolGrammar,
        headroom: super::http::Headroom,
    ) -> Self {
        Self {
            reasoning: if preopened_reasoning {
                StreamPartition::with_preopened_reasoning()
            } else if tool_grammar == ToolGrammar::DeepSeekDsml {
                // A DS4 prompt always ends in `<think>` or `</think>`, so a
                // generation that is not pre-opened starts after `</think>`:
                // any tag it writes is content, and it has no reasoning.
                StreamPartition::with_closed_reasoning()
            } else {
                StreamPartition::new()
            },
            utf8: Utf8Assembler::new(),
            parse_tools,
            tool_grammar,
            pending_visible: String::new(),
            tool_buffer: String::new(),
            failure: None,
            headroom,
            tool_separator: "",
            in_tool_span: false,
        }
    }

    /// Headroom for the tool block's outstanding peak at `capacity`, given
    /// what the buffer already holds.
    fn admit_outstanding(&self, capacity: usize) -> Result<(), ServeError> {
        let peak = qwen_llm::glm5_next_chat::checked_tool_block_peak_bytes(capacity)
            .ok_or_else(|| super::output_memory::size_overflow("tool block"))?;
        let outstanding = peak.saturating_sub(self.tool_buffer.capacity());
        super::transport_memory::admit_resident_transport(outstanding as u64, (self.headroom)())
    }

    /// Before the tool buffer grows by `additional` bytes: admit and reserve
    /// whole steps; on refusal, store it and leave the buffer unchanged.
    fn grow_tool_buffer(&mut self, additional: usize) -> bool {
        use super::partition_glm5_next::ADMISSION_STEP;
        let Some(len) = self.tool_buffer.len().checked_add(additional) else {
            self.failure = Some(super::output_memory::size_overflow("tool block"));
            return false;
        };
        if len <= self.tool_buffer.capacity() {
            return true;
        }
        let Some(capacity) = len.div_ceil(ADMISSION_STEP).checked_mul(ADMISSION_STEP) else {
            self.failure = Some(super::output_memory::size_overflow("tool block"));
            return false;
        };
        if let Err(error) = self.admit_outstanding(capacity) {
            self.failure = Some(error);
            return false;
        }
        if let Err(error) = self
            .tool_buffer
            .try_reserve_exact(capacity - self.tool_buffer.len())
        {
            self.failure = Some(super::output_memory::allocation_failure(
                "tool block",
                error,
            ));
            return false;
        }
        true
    }

    /// A stored refusal (generation should stop; the turn fails with it).
    pub(crate) fn failure(&self) -> Option<&ServeError> {
        self.failure.as_ref()
    }

    fn push(&mut self, bytes: &[u8], events: &mut Vec<PartitionEvent>) {
        if self.failure.is_some() {
            return;
        }
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
        if self.failure.is_some() {
            return;
        }
        if self.in_tool_span {
            if self.grow_tool_buffer(text.len()) {
                self.tool_buffer.push_str(text);
            }
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
            self.in_tool_span = true;
            if self.grow_tool_buffer(calls.len()) {
                self.tool_buffer.push_str(&calls);
            }
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

    fn finish(mut self, events: &mut Vec<PartitionEvent>) -> Result<(), ServeError> {
        if let Some(failure) = self.failure.take() {
            return Err(failure);
        }
        self.flush_reasoning(events);
        if let Some(failure) = self.failure.take() {
            return Err(failure);
        }
        if !self.parse_tools {
            return Ok(());
        }
        if !self.pending_visible.is_empty() {
            events.push(PartitionEvent::Visible(std::mem::take(
                &mut self.pending_visible,
            )));
        }
        if !self.in_tool_span {
            return Ok(());
        }
        // The parse's peak, checked again against current headroom.
        self.admit_outstanding(self.tool_buffer.capacity())?;
        let buffer = std::mem::take(&mut self.tool_buffer);
        let parsed = self.tool_grammar.parse(&buffer);
        if parsed.calls.is_empty() {
            events.push(PartitionEvent::Visible(format!(
                "{}{buffer}",
                self.tool_separator
            )));
            return Ok(());
        }
        if !parsed.visible.is_empty() {
            events.push(PartitionEvent::Visible(parsed.visible));
        }
        events.extend(parsed.calls.into_iter().map(PartitionEvent::FunctionCall));
        Ok(())
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

    /// Map #14 packet 2: the Qwen-XML and DSML tool buffers admit each
    /// 64 KiB step at the 256x outstanding peak against fresh headroom and
    /// again before the parse; a refusal is stored (generation can stop),
    /// publishes no raw tool text, and fails the turn with its typed error.
    #[test]
    fn qwen_and_dsml_tool_buffers_admit_each_step_and_refuse_typed() {
        use std::sync::atomic::{AtomicU64, Ordering};
        static HEADROOM: AtomicU64 = AtomicU64::new(u64::MAX);
        let step = super::super::partition_glm5_next::ADMISSION_STEP;
        let peak = |bytes: usize| qwen_llm::glm5_next_chat::tool_block_peak_bytes(bytes) as u64;
        for (grammar, open, body) in [
            (
                ToolGrammar::QwenXml,
                "<tool_call>\n<function=f>\n<parameter=a>\n",
                "\n</parameter>\n</function>\n</tool_call>",
            ),
            (
                ToolGrammar::DeepSeekDsml,
                "<｜DSML｜tool_calls>\n<｜DSML｜invoke name=\"f\">\n<｜DSML｜parameter name=\"a\" string=\"true\">",
                "</｜DSML｜parameter>\n</｜DSML｜invoke>\n</｜DSML｜tool_calls>",
            ),
        ] {
            let partition = || {
                OutputPartition::with_headroom(
                    OutputProtocol::Qwen {
                        preopened_reasoning: false,
                        parse_tools: true,
                        tool_grammar: grammar,
                    },
                    || Some(HEADROOM.load(Ordering::SeqCst)),
                )
            };
            let value = "x".repeat(step);
            // Enough for the first step, not the second.
            HEADROOM.store(peak(step), Ordering::SeqCst);
            let mut p = partition();
            let mut events = Vec::new();
            p.push(format!("before {open}").as_bytes(), &mut events);
            assert!(p.failure().is_none(), "{grammar:?}: first step admitted");
            p.push(value.as_bytes(), &mut events);
            let failure = p.failure().cloned().expect("second step refused");
            assert_eq!(
                (failure.status, failure.code),
                (503, Some("memory_admission_denied"))
            );
            p.push(body.as_bytes(), &mut events);
            let error = p
                .finish(GenerationEnd::StopToken(0), &mut events)
                .unwrap_err();
            assert_eq!(error, failure);
            let published: String = events
                .iter()
                .filter_map(|e| match e {
                    PartitionEvent::Visible(text) => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(published, "before ", "{grammar:?}: no raw tool text");

            // Admitted growth, then too little headroom for the parse.
            HEADROOM.store(u64::MAX, Ordering::SeqCst);
            let mut p = partition();
            let mut events = Vec::new();
            p.push(format!("{open}{value}{body}").as_bytes(), &mut events);
            assert!(p.failure().is_none());
            HEADROOM.store(1, Ordering::SeqCst);
            let error = p
                .finish(GenerationEnd::StopToken(0), &mut events)
                .unwrap_err();
            assert_eq!(error.code, Some("memory_admission_denied"));

            // With headroom the call parses and publishes.
            let mut p = partition();
            HEADROOM.store(u64::MAX, Ordering::SeqCst);
            let mut events = Vec::new();
            p.push(format!("{open}{value}{body}").as_bytes(), &mut events);
            p.finish(GenerationEnd::StopToken(0), &mut events).unwrap();
            assert!(
                events
                    .iter()
                    .any(|e| matches!(e, PartitionEvent::FunctionCall(_))),
                "{grammar:?}"
            );
        }
        // An unreadable signal fails closed as telemetry (500).
        let mut p = OutputPartition::with_headroom(
            OutputProtocol::Qwen {
                preopened_reasoning: false,
                parse_tools: true,
                tool_grammar: ToolGrammar::QwenXml,
            },
            || None,
        );
        p.push(b"<tool_call>\n", &mut Vec::new());
        assert_eq!(
            p.failure().map(|f| f.code),
            Some(Some("memory_signal_unavailable"))
        );
    }

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
