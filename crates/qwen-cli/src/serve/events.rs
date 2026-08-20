//! Response object construction and sequence-numbered streaming events.
//!
//! Every streamed event carries a monotonic `sequence_number` (SERVE.md,
//! review defect 4) and is delivered through [`EventWrite`], so tests
//! assert exact event order and payloads without any socket. SSE framing
//! (`event:`/`data:` lines, heartbeat comments, `[DONE]`) lives in
//! [`SseWriter`]; the HTTP slice owns the socket.
//!
//! Part-type note: reasoning item content parts use `reasoning_text`
//! (matching the OpenAI-lineage `response.reasoning.delta` events this
//! module emits). The gate-5 conformance run adjudicates this choice; it is
//! isolated behind `REASONING_PART_TYPE`.

use super::items::ServeRequest;
use super::partition::PartitionEvent;
use super::partition::safe_emit_len;
use super::tool_parse::{ParsedCall, parse_emission};
use serde_json::{Value, json};
use std::io::{self, Write};
use std::time::{Duration, Instant};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const REASONING_PART_TYPE: &str = "reasoning_text";
const TOOL_CALL_OPEN: &str = "<tool_call>";

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopReason {
    Eos,
    TokenLimit,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Usage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
    /// Tokens restored from checkpoints (spec `cached_tokens`).
    pub(crate) cached_tokens: usize,
}

impl Usage {
    fn to_json(&self) -> Value {
        json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_tokens_details": {"cached_tokens": self.cached_tokens},
            // This surface exposes model reasoning text but has no separate
            // hidden/provider reasoning-token accounting channel.
            "output_tokens_details": {"reasoning_tokens": 0},
        })
    }
}

/// Request-derived fields echoed into every response envelope; the
/// conformance suite validates the full ResponseResource field set
/// (gate 5), so absent-but-required keys are spec violations.
#[derive(Debug, Clone)]
pub(crate) struct EnvelopeEcho {
    pub(crate) temperature: f64,
    pub(crate) top_p: f64,
    pub(crate) max_output_tokens: Option<u64>,
    pub(crate) instructions: Value,
    pub(crate) tools: Value,
    pub(crate) tool_choice: Value,
    pub(crate) reasoning: Value,
    pub(crate) parallel_tool_calls: bool,
}

impl Default for EnvelopeEcho {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            max_output_tokens: None,
            instructions: Value::Null,
            tools: json!([]),
            tool_choice: json!("auto"),
            reasoning: Value::Null,
            parallel_tool_calls: true,
        }
    }
}

/// Optional `x_qwen.stats` echo (SERVE.md review R4).
#[derive(Debug, Clone, Default)]
pub(crate) struct ServeStats {
    pub(crate) matched_tokens: usize,
    pub(crate) restore_ms: f64,
    pub(crate) prompt_tokens: usize,
}

impl ServeStats {
    fn to_json(&self) -> Value {
        json!({
            "version": "serve_stats_v1",
            "matched_tokens": self.matched_tokens,
            "restore_ms": self.restore_ms,
            "prompt_tokens": self.prompt_tokens,
        })
    }
}

pub(crate) trait EventWrite {
    fn event(&mut self, event_type: &str, payload: Value) -> io::Result<()>;
    /// Transport-level keepalive (SSE comment). No-op for collectors.
    fn comment(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// Test/collection sink.
#[derive(Debug, Default)]
pub(crate) struct CollectEvents(pub(crate) Vec<(String, Value)>);

impl EventWrite for CollectEvents {
    fn event(&mut self, event_type: &str, payload: Value) -> io::Result<()> {
        self.0.push((event_type.to_owned(), payload));
        Ok(())
    }
}

/// SSE framing over any writer, flushed per event (serial loopback server;
/// flush latency is the streaming contract).
pub(crate) struct SseWriter<W: Write>(pub(crate) W);

impl<W: Write> SseWriter<W> {
    pub(crate) fn heartbeat(&mut self) -> io::Result<()> {
        self.0.write_all(b": ping\n\n")?;
        self.0.flush()
    }

    pub(crate) fn done(&mut self) -> io::Result<()> {
        self.0.write_all(b"data: [DONE]\n\n")?;
        self.0.flush()
    }
}

impl<W: Write> EventWrite for SseWriter<W> {
    fn event(&mut self, event_type: &str, payload: Value) -> io::Result<()> {
        write!(self.0, "event: {event_type}\ndata: {payload}\n\n")?;
        self.0.flush()
    }
    fn comment(&mut self) -> io::Result<()> {
        self.heartbeat()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OpenItem {
    None,
    Reasoning,
    Message,
}

/// Streams one response as ordered events while accumulating the final
/// response object. Drive with [`PartitionEvent`]s, then [`finish`].
pub(crate) struct ResponseStream<'a, W: EventWrite> {
    writer: &'a mut W,
    sequence_number: u64,
    response_id: String,
    model: String,
    created_at: u64,
    completed_at: Option<u64>,
    echo: EnvelopeEcho,
    open: OpenItem,
    output_index: usize,
    reasoning_item_id: String,
    message_item_id: String,
    reasoning_text: String,
    visible_text: String,
    reasoning_opened: bool,
    message_opened: bool,
    last_activity: Instant,
    /// Visible bytes held back while a `<tool_call>` prefix is ambiguous.
    pending_visible: String,
    /// Emission bytes from the first `<tool_call>` onward (S2: the format
    /// guarantees no suffix after calls, so everything after the opener is
    /// call syntax until proven malformed).
    tool_buffer: String,
    in_tool_span: bool,
    tool_items: Vec<Value>,
    /// Exact executable subset. Empty means no executable calls.
    allowed_tools: Vec<String>,
}

impl<'a, W: EventWrite> ResponseStream<'a, W> {
    /// Set the exact executable tool set. Empty means no executable calls.
    pub(crate) fn set_allowed_tools(&mut self, allowed: Vec<String>) {
        self.allowed_tools = allowed;
    }

    pub(crate) fn begin(
        writer: &'a mut W,
        response_id: String,
        model: String,
        created_at: u64,
        echo: EnvelopeEcho,
    ) -> io::Result<Self> {
        let mut stream = Self {
            writer,
            sequence_number: 0,
            response_id,
            model,
            created_at,
            completed_at: None,
            echo,
            open: OpenItem::None,
            output_index: 0,
            reasoning_item_id: String::new(),
            message_item_id: String::new(),
            reasoning_text: String::new(),
            visible_text: String::new(),
            reasoning_opened: false,
            message_opened: false,
            last_activity: Instant::now(),
            pending_visible: String::new(),
            tool_buffer: String::new(),
            in_tool_span: false,
            tool_items: Vec::new(),
            allowed_tools: Vec::new(),
        };
        stream.reasoning_item_id = format!("rs_{}", stream.response_id);
        stream.message_item_id = format!("msg_{}", stream.response_id);
        let created = stream.response_envelope("in_progress", Value::Array(vec![]), None, None);
        stream.emit("response.created", json!({"response": created.clone()}))?;
        stream.emit("response.in_progress", json!({"response": created}))?;
        Ok(stream)
    }

    fn next_sequence(&mut self) -> u64 {
        self.sequence_number += 1;
        self.sequence_number
    }

    fn emit(&mut self, event_type: &str, mut payload: Value) -> io::Result<()> {
        let sequence = self.next_sequence();
        payload["type"] = json!(event_type);
        payload["sequence_number"] = json!(sequence);
        self.last_activity = Instant::now();
        self.writer.event(event_type, payload)
    }

    /// Rate-limited transport keepalive; called between prefill chunks
    /// (SERVE.md gate 3, client-timeout defense). Write failure here is the
    /// cancellation signal.
    pub(crate) fn heartbeat_if_idle(&mut self) -> io::Result<()> {
        if self.last_activity.elapsed() >= Duration::from_secs(1) {
            self.writer.comment()?;
            self.last_activity = Instant::now();
        }
        Ok(())
    }

    /// Mid-stream failure: close open items as incomplete, emit
    /// `response.failed` with the spec error inside the response envelope.
    pub(crate) fn fail(mut self, error: &super::items::ServeError) -> io::Result<()> {
        // A failed generation must never publish an executable call. Preserve
        // every generated byte for diagnostics, but surface buffered tool
        // syntax as incomplete visible text instead of parsing it.
        let mut pending = std::mem::take(&mut self.pending_visible);
        if self.in_tool_span {
            pending.push_str(&std::mem::take(&mut self.tool_buffer));
            self.in_tool_span = false;
        }
        if !pending.is_empty() {
            self.emit_visible(&pending)?;
        }
        match self.open {
            OpenItem::Reasoning => self.close_reasoning("incomplete")?,
            OpenItem::Message => self.close_message("incomplete")?,
            OpenItem::None => {}
        }
        let mut output = Vec::new();
        if self.reasoning_opened {
            output.push(self.reasoning_item_json("incomplete"));
        }
        if self.message_opened {
            output.push(self.message_item_json("incomplete"));
        }
        output.extend(self.tool_items.iter().cloned());
        self.completed_at = Some(now_unix());
        let mut envelope = self.response_envelope("failed", Value::Array(output), None, None);
        envelope["error"] = error.to_json()["error"].clone();
        self.emit("response.failed", json!({"response": envelope}))
    }

    fn response_envelope(
        &self,
        status: &str,
        output: Value,
        usage: Option<&Usage>,
        incomplete_reason: Option<&str>,
    ) -> Value {
        let terminal = matches!(status, "completed" | "incomplete" | "failed");
        json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": if terminal {
                self.completed_at.map_or(Value::Null, |value| json!(value))
            } else {
                Value::Null
            },
            "model": self.model,
            "status": status,
            "output": output,
            "store": false,
            "usage": usage.map_or(Value::Null, Usage::to_json),
            "incomplete_details": incomplete_reason.map_or(Value::Null, |reason| json!({"reason": reason})),
            "error": Value::Null,
            "previous_response_id": Value::Null,
            "instructions": self.echo.instructions.clone(),
            "tools": self.echo.tools.clone(),
            "tool_choice": self.echo.tool_choice.clone(),
            "truncation": "disabled",
            "parallel_tool_calls": self.echo.parallel_tool_calls,
            "text": json!({"format": {"type": "text"}}),
            "temperature": self.echo.temperature,
            "top_p": self.echo.top_p,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "top_logprobs": 0,
            "reasoning": self.echo.reasoning.clone(),
            "max_output_tokens": self.echo.max_output_tokens.map_or(Value::Null, |v| json!(v)),
            "max_tool_calls": Value::Null,
            "background": false,
            "service_tier": "default",
            "metadata": json!({}),
            "safety_identifier": Value::Null,
            "prompt_cache_key": Value::Null,
        })
    }

    fn open_reasoning(&mut self) -> io::Result<()> {
        debug_assert_eq!(self.open, OpenItem::None);
        self.reasoning_opened = true;
        self.open = OpenItem::Reasoning;
        let item = json!({
            "id": self.reasoning_item_id,
            "type": "reasoning",
            "status": "in_progress",
            "summary": [],
            "content": [],
        });
        self.emit(
            "response.output_item.added",
            json!({"output_index": self.output_index, "item": item}),
        )?;
        self.emit(
            "response.content_part.added",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "part": {"type": REASONING_PART_TYPE, "text": ""},
            }),
        )
    }

    fn open_message(&mut self) -> io::Result<()> {
        debug_assert_eq!(self.open, OpenItem::None);
        self.message_opened = true;
        self.open = OpenItem::Message;
        let item = json!({
            "id": self.message_item_id,
            "type": "message",
            "role": "assistant",
            "status": "in_progress",
            "content": [],
        });
        self.emit(
            "response.output_item.added",
            json!({"output_index": self.output_index, "item": item}),
        )?;
        self.emit(
            "response.content_part.added",
            json!({
                "item_id": self.message_item_id,
                "output_index": self.output_index,
                "content_index": 0,
                "part": {"type": "output_text", "text": "", "annotations": []},
            }),
        )
    }

    fn close_reasoning(&mut self, status: &str) -> io::Result<()> {
        debug_assert_eq!(self.open, OpenItem::Reasoning);
        let item_id = self.reasoning_item_id.clone();
        let output_index = self.output_index;
        let text = self.reasoning_text.clone();
        self.emit(
            "response.reasoning.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            }),
        )?;
        let part = json!({"type": REASONING_PART_TYPE, "text": self.reasoning_text});
        self.emit(
            "response.content_part.done",
            json!({
                "item_id": self.reasoning_item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": part,
            }),
        )?;
        let item = self.reasoning_item_json(status);
        self.emit(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        )?;
        self.open = OpenItem::None;
        self.output_index += 1;
        Ok(())
    }

    fn close_message(&mut self, status: &str) -> io::Result<()> {
        debug_assert_eq!(self.open, OpenItem::Message);
        let item_id = self.message_item_id.clone();
        let output_index = self.output_index;
        let text = self.visible_text.clone();
        self.emit(
            "response.output_text.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "content_index": 0,
                "text": text,
            }),
        )?;
        let part = json!({"type": "output_text", "text": self.visible_text, "annotations": []});
        self.emit(
            "response.content_part.done",
            json!({
                "item_id": self.message_item_id,
                "output_index": output_index,
                "content_index": 0,
                "part": part,
            }),
        )?;
        let item = self.message_item_json(status);
        self.emit(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item}),
        )?;
        self.open = OpenItem::None;
        self.output_index += 1;
        Ok(())
    }

    fn reasoning_item_json(&self, status: &str) -> Value {
        json!({
            "id": self.reasoning_item_id,
            "type": "reasoning",
            "status": status,
            "summary": [],
            "content": [{"type": REASONING_PART_TYPE, "text": self.reasoning_text}],
        })
    }

    fn message_item_json(&self, status: &str) -> Value {
        json!({
            "id": self.message_item_id,
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": self.visible_text, "annotations": []}],
        })
    }

    pub(crate) fn on_partition(&mut self, event: &PartitionEvent) -> io::Result<()> {
        match event {
            PartitionEvent::Reasoning(text) => {
                if self.open != OpenItem::Reasoning {
                    debug_assert_eq!(self.open, OpenItem::None, "reasoning after message");
                    self.open_reasoning()?;
                }
                self.reasoning_text.push_str(text);
                let payload = json!({
                    "item_id": self.reasoning_item_id,
                    "output_index": self.output_index,
                    "content_index": 0,
                    "delta": text,
                });
                self.emit("response.reasoning.delta", payload.clone())?;
                // Compatibility alias consumed by stock
                // @ai-sdk/open-responses. Keep the gated event above as the
                // canonical contract and let output_item.done close the item.
                self.emit("response.reasoning_text.delta", payload)
            }
            PartitionEvent::ReasoningClosed => {
                if self.open == OpenItem::Reasoning {
                    self.close_reasoning("completed")?;
                }
                Ok(())
            }
            PartitionEvent::Visible(text) => {
                if self.open == OpenItem::Reasoning {
                    self.close_reasoning("completed")?;
                }
                if self.in_tool_span {
                    self.tool_buffer.push_str(text);
                    return Ok(());
                }
                self.pending_visible.push_str(text);
                if let Some(index) = self.pending_visible.find(TOOL_CALL_OPEN) {
                    let prose: String = self.pending_visible[..index].to_owned();
                    let call_span: String = self.pending_visible[index..].to_owned();
                    self.pending_visible.clear();
                    if !prose.is_empty() {
                        self.emit_visible(&prose)?;
                    }
                    self.tool_buffer.push_str(&call_span);
                    self.in_tool_span = true;
                    return Ok(());
                }
                let safe = safe_emit_len(&self.pending_visible, TOOL_CALL_OPEN);
                if safe > 0 {
                    let chunk: String = self.pending_visible[..safe].to_owned();
                    self.pending_visible.drain(..safe);
                    self.emit_visible(&chunk)?;
                }
                Ok(())
            }
        }
    }

    fn emit_visible(&mut self, text: &str) -> io::Result<()> {
        if self.open != OpenItem::Message {
            self.open_message()?;
        }
        self.visible_text.push_str(text);
        let payload = json!({
            "item_id": self.message_item_id,
            "output_index": self.output_index,
            "content_index": 0,
            "delta": text,
        });
        self.emit("response.output_text.delta", payload)
    }

    /// Emit one `function_call` item's full lifecycle.
    fn emit_function_call(&mut self, index: usize, call: &ParsedCall) -> io::Result<()> {
        let item_id = format!("fc_{}_{index}", self.response_id);
        let call_id = format!("call_{}_{index}", self.response_id);
        let arguments =
            serde_json::to_string(&Value::Object(call.arguments.clone())).expect("serialize args");
        let output_index = self.output_index;
        self.emit(
            "response.output_item.added",
            json!({
                "output_index": output_index,
                "item": {
                    "id": item_id,
                    "type": "function_call",
                    "status": "in_progress",
                    "call_id": call_id,
                    "name": call.name,
                    "arguments": "",
                },
            }),
        )?;
        self.emit(
            "response.function_call_arguments.delta",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "call_id": call_id,
                "delta": arguments,
            }),
        )?;
        self.emit(
            "response.function_call_arguments.done",
            json!({
                "item_id": item_id,
                "output_index": output_index,
                "call_id": call_id,
                "arguments": arguments,
            }),
        )?;
        let item = json!({
            "id": item_id,
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": call.name,
            "arguments": arguments,
        });
        self.emit(
            "response.output_item.done",
            json!({"output_index": output_index, "item": item.clone()}),
        )?;
        self.output_index += 1;
        self.tool_items.push(item);
        Ok(())
    }

    /// Resolve buffered tool-span bytes at end of generation: parse into
    /// calls, or (no salvage, S2 fixture `malformed_corpus`) flush the raw
    /// bytes back as visible text.
    fn resolve_tool_span(&mut self) -> io::Result<()> {
        let pending = std::mem::take(&mut self.pending_visible);
        if !pending.is_empty() {
            self.emit_visible(&pending)?;
        }
        if !self.in_tool_span {
            return Ok(());
        }
        let buffer = std::mem::take(&mut self.tool_buffer);
        self.in_tool_span = false;
        let parsed = parse_emission(&buffer);
        if parsed.calls.is_empty() {
            self.emit_visible(&buffer)?;
            return Ok(());
        }
        if !parsed.visible.is_empty() {
            self.emit_visible(&parsed.visible)?;
        }
        if self.open == OpenItem::Message {
            self.close_message("completed")?;
        }
        let mut emitted_calls = 0;
        for (index, call) in parsed.calls.iter().enumerate() {
            if !self.allowed_tools.contains(&call.name) {
                // Spec: allowed_tools is a hard constraint. The call is
                // dropped (not re-surfaced as text) and logged; enforcement
                // is post-generation, so its tokens were already spent.
                tracing::info!(
                    target: "qwen_diag",
                    "serve: suppressed call to disallowed tool {:?}",
                    call.name,
                );
                continue;
            }
            if !self.echo.parallel_tool_calls && emitted_calls > 0 {
                tracing::info!(target: "qwen_diag", "serve: suppressed parallel tool call {:?}", call.name);
                continue;
            }
            let call = call.clone();
            self.emit_function_call(index, &call)?;
            emitted_calls += 1;
        }
        Ok(())
    }

    /// Close open items, emit the terminal response event, and return the
    /// final response object (shared with the non-stream path).
    pub(crate) fn finish(
        mut self,
        stop_reason: StopReason,
        usage: Usage,
        stats: Option<&ServeStats>,
    ) -> io::Result<Value> {
        self.resolve_tool_span()?;
        let truncated_in_reasoning = self.open == OpenItem::Reasoning;
        match self.open {
            OpenItem::Reasoning => self.close_reasoning(match stop_reason {
                StopReason::Eos => "completed",
                StopReason::TokenLimit => "incomplete",
            })?,
            OpenItem::Message => self.close_message(match stop_reason {
                StopReason::Eos => "completed",
                StopReason::TokenLimit => "incomplete",
            })?,
            OpenItem::None => {}
        }
        let mut output = Vec::new();
        if self.reasoning_opened {
            let status = if truncated_in_reasoning && stop_reason == StopReason::TokenLimit {
                "incomplete"
            } else {
                "completed"
            };
            output.push(self.reasoning_item_json(status));
        }
        if self.message_opened {
            let status = if stop_reason == StopReason::TokenLimit
                && !truncated_in_reasoning
                && self.tool_items.is_empty()
            {
                "incomplete"
            } else {
                "completed"
            };
            output.push(self.message_item_json(status));
        }
        output.extend(self.tool_items.iter().cloned());
        let (status, event_type, incomplete_reason) = match stop_reason {
            StopReason::Eos => ("completed", "response.completed", None),
            StopReason::TokenLimit => (
                "incomplete",
                "response.incomplete",
                Some("max_output_tokens"),
            ),
        };
        self.completed_at = Some(now_unix());
        let mut envelope = self.response_envelope(
            status,
            Value::Array(output),
            Some(&usage),
            incomplete_reason,
        );
        if let Some(stats) = stats {
            envelope["x_qwen"] = stats.to_json();
        }
        self.emit(event_type, json!({"response": envelope.clone()}))?;
        Ok(envelope)
    }
}

/// Build the non-stream response object by replaying the same machinery
/// into a discard sink (identical output shape by construction).
pub(crate) fn build_response_object(
    request: &ServeRequest,
    response_id: String,
    created_at: u64,
    partition_events: &[PartitionEvent],
    stop_reason: StopReason,
    usage: Usage,
    stats: Option<&ServeStats>,
) -> io::Result<Value> {
    let mut sink = CollectEvents::default();
    let mut stream = ResponseStream::begin(
        &mut sink,
        response_id,
        request.model.clone(),
        created_at,
        envelope_echo(request),
    )?;
    stream.set_allowed_tools(request.allowed_tools.clone());
    for event in partition_events {
        stream.on_partition(event)?;
    }
    stream.finish(stop_reason, usage, stats)
}

/// Envelope echo derived from a validated request.
pub(crate) fn envelope_echo(request: &ServeRequest) -> EnvelopeEcho {
    let tools = request
        .tools
        .iter()
        .map(|tool| {
            let mut entry = serde_json::Map::new();
            entry.insert("type".into(), json!("function"));
            entry.insert("name".into(), json!(tool.name));
            if let Some(description) = &tool.description {
                entry.insert("description".into(), json!(description));
            }
            if !tool.parameters.is_null() {
                entry.insert("parameters".into(), tool.parameters.clone());
            }
            if let Some(strict) = tool.strict {
                entry.insert("strict".into(), json!(strict));
            }
            Value::Object(entry)
        })
        .collect();
    EnvelopeEcho {
        temperature: request.temperature_echo.unwrap_or(0.0),
        top_p: request.top_p_echo.unwrap_or(1.0),
        max_output_tokens: request.max_output_tokens.map(|value| value as u64),
        instructions: request
            .instructions
            .clone()
            .map_or(Value::Null, Value::String),
        tools: Value::Array(tools),
        tool_choice: request.tool_choice.clone(),
        reasoning: request.reasoning.clone().unwrap_or(Value::Null),
        parallel_tool_calls: request.parallel_tool_calls,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::partition::StreamPartition;

    fn drive(pieces: &[&str], stop_reason: StopReason) -> (Vec<(String, Value)>, Value) {
        let mut sink = CollectEvents::default();
        let mut stream = ResponseStream::begin(
            &mut sink,
            "resp_test".into(),
            "qwen-test".into(),
            1_755_500_000,
            EnvelopeEcho::default(),
        )
        .unwrap();
        stream.set_allowed_tools(vec![
            "fs_list".into(),
            "ping".into(),
            "a".into(),
            "b".into(),
        ]);
        let mut partition = StreamPartition::new();
        let mut events = Vec::new();
        for piece in pieces {
            events.clear();
            partition.push(piece, &mut events);
            for event in &events {
                stream.on_partition(event).unwrap();
            }
        }
        events.clear();
        partition.finish(&mut events);
        for event in &events {
            stream.on_partition(event).unwrap();
        }
        let envelope = stream
            .finish(
                stop_reason,
                Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cached_tokens: 0,
                },
                None,
            )
            .unwrap();
        (sink.0, envelope)
    }

    #[test]
    fn reasoning_then_visible_emits_spec_event_order_with_monotonic_sequence() {
        let (events, envelope) = drive(
            &["<think>\nplan\n</think>\n\nanswer", " tail"],
            StopReason::Eos,
        );
        let types: Vec<&str> = events.iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(
            types,
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.reasoning.delta",
                "response.reasoning_text.delta",
                "response.reasoning.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        for (index, (_, payload)) in events.iter().enumerate() {
            assert_eq!(
                payload["sequence_number"].as_u64(),
                Some(index as u64 + 1),
                "sequence_number must be monotonic from 1"
            );
            assert_eq!(payload["type"].as_str().unwrap(), types[index]);
        }
        assert_eq!(envelope["status"], "completed");
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["content"][0]["text"], "\nplan\n");
        assert_eq!(output[1]["content"][0]["text"], "\n\nanswer tail");
        assert_eq!(envelope["usage"]["total_tokens"], 15);
    }

    #[test]
    fn tool_call_emission_streams_function_call_items() {
        let (events, envelope) = drive(
            &[
                "<think>\nuse the tool\n</think>\n\nListing now.\n",
                "<tool_call>\n<function=fs_list>\n<parameter=path>\n/tmp\n",
                "</parameter>\n</function>\n</tool_call>",
            ],
            StopReason::Eos,
        );
        let types: Vec<&str> = events.iter().map(|(t, _)| t.as_str()).collect();
        assert!(types.contains(&"response.function_call_arguments.delta"));
        assert!(types.contains(&"response.function_call_arguments.done"));
        // No visible delta may leak the call syntax.
        for (event_type, payload) in &events {
            if event_type.as_str() == "response.output_text.delta" {
                assert!(
                    !payload["delta"].as_str().unwrap().contains("<tool_call>"),
                    "call syntax leaked into visible deltas"
                );
            }
        }
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 3, "reasoning + message + function_call");
        assert_eq!(output[2]["type"], "function_call");
        assert_eq!(output[2]["name"], "fs_list");
        assert_eq!(output[2]["arguments"], "{\"path\":\"/tmp\"}");
        assert!(output[2]["call_id"].as_str().unwrap().starts_with("call_"));
        assert_eq!(output[1]["content"][0]["text"], "\n\nListing now.\n");
        assert_eq!(envelope["status"], "completed");
    }

    #[test]
    fn allowed_tools_suppresses_disallowed_calls() {
        let mut sink = CollectEvents::default();
        let mut stream = ResponseStream::begin(
            &mut sink,
            "resp_test".into(),
            "qwen-test".into(),
            1_755_500_000,
            EnvelopeEcho::default(),
        )
        .unwrap();
        stream.set_allowed_tools(vec!["allowed".into()]);
        let mut partition = crate::serve::partition::StreamPartition::new();
        let mut events = Vec::new();
        partition.push(
            concat!(
                "<tool_call>\n<function=blocked>\n</function>\n</tool_call>\n",
                "<tool_call>\n<function=allowed>\n</function>\n</tool_call>",
            ),
            &mut events,
        );
        for event in &events {
            stream.on_partition(event).unwrap();
        }
        events.clear();
        partition.finish(&mut events);
        for event in &events {
            stream.on_partition(event).unwrap();
        }
        let envelope = stream
            .finish(StopReason::Eos, Usage::default(), None)
            .unwrap();
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "only the allowed call may be emitted");
        assert_eq!(output[0]["name"], "allowed");
    }

    #[test]
    fn call_without_prose_emits_no_message_item() {
        let (_, envelope) = drive(
            &["<tool_call>\n<function=ping>\n</function>\n</tool_call>"],
            StopReason::Eos,
        );
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "function_call");
        assert_eq!(output[0]["arguments"], "{}");
    }

    #[test]
    fn malformed_call_syntax_stays_visible_text() {
        let (_, envelope) = drive(
            &["Sure.\n<tool_call>\n<function=fs_list>\n<parameter=path>\n/tm"],
            StopReason::TokenLimit,
        );
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 1, "no function_call salvaged");
        assert_eq!(output[0]["type"], "message");
        assert_eq!(
            output[0]["content"][0]["text"],
            "Sure.\n<tool_call>\n<function=fs_list>\n<parameter=path>\n/tm"
        );
        assert_eq!(envelope["status"], "incomplete");
    }

    #[test]
    fn parallel_calls_emit_ordered_items_with_distinct_ids() {
        let (_, envelope) = drive(
            &[concat!(
                "<tool_call>\n<function=a>\n</function>\n</tool_call>\n",
                "<tool_call>\n<function=b>\n</function>\n</tool_call>",
            )],
            StopReason::Eos,
        );
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 2);
        assert_eq!(output[0]["name"], "a");
        assert_eq!(output[1]["name"], "b");
        assert_ne!(output[0]["call_id"], output[1]["call_id"]);
    }

    #[test]
    fn visible_only_stream_never_opens_a_reasoning_item() {
        let (events, envelope) = drive(&["plain answer"], StopReason::Eos);
        assert!(
            events.iter().all(|(t, _)| !t.contains("reasoning")),
            "no reasoning events expected"
        );
        assert_eq!(envelope["output"].as_array().unwrap().len(), 1);
    }

    #[test]
    fn token_limit_mid_reasoning_marks_item_and_response_incomplete() {
        let (events, envelope) = drive(&["<think>\nran out of bud"], StopReason::TokenLimit);
        assert_eq!(events.last().unwrap().0, "response.incomplete");
        assert_eq!(envelope["status"], "incomplete");
        assert_eq!(
            envelope["incomplete_details"]["reason"],
            "max_output_tokens"
        );
        let output = envelope["output"].as_array().unwrap();
        assert_eq!(output.len(), 1);
        assert_eq!(output[0]["type"], "reasoning");
        assert_eq!(output[0]["status"], "incomplete");
    }

    #[test]
    fn sse_framing_and_terminals_are_exact() {
        let mut bytes = Vec::new();
        {
            let mut writer = SseWriter(&mut bytes);
            writer.heartbeat().unwrap();
            writer
                .event("response.created", serde_json::json!({"a": 1}))
                .unwrap();
            writer.done().unwrap();
        }
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            ": ping\n\nevent: response.created\ndata: {\"a\":1}\n\ndata: [DONE]\n\n"
        );
    }

    #[test]
    fn non_stream_builder_matches_streamed_envelope() {
        let full = "<think>\nplan\n</think>\n\nanswer";
        let (_, streamed) = drive(&[full], StopReason::Eos);
        let mut partition = StreamPartition::new();
        let mut events = Vec::new();
        partition.push(full, &mut events);
        partition.finish(&mut events);
        let request = ServeRequest {
            model: "qwen-test".into(),
            ..ServeRequest::default()
        };
        let built = build_response_object(
            &request,
            "resp_test".into(),
            1_755_500_000,
            &events,
            StopReason::Eos,
            Usage {
                input_tokens: 10,
                output_tokens: 5,
                cached_tokens: 0,
            },
            None,
        )
        .unwrap();
        assert_eq!(built, streamed);
    }

    #[test]
    fn request_echo_and_no_tools_constraint_are_truthful() {
        let request = crate::serve::items::parse_request(&json!({
            "model":"qwen-test", "input":"q", "instructions":"be terse",
            "temperature":0.7, "top_p":0.8, "parallel_tool_calls":false,
            "reasoning":{"effort":"high"}, "tool_choice":"auto"
        }))
        .unwrap();
        let emission = "<tool_call>\n<function=undeclared>\n</function>\n</tool_call>";
        let mut partition = StreamPartition::new();
        let mut parts = Vec::new();
        partition.push(emission, &mut parts);
        partition.finish(&mut parts);
        let envelope = build_response_object(
            &request,
            "resp_echo".into(),
            1,
            &parts,
            StopReason::Eos,
            Usage::default(),
            None,
        )
        .unwrap();
        assert_eq!(envelope["instructions"], "be terse");
        assert_eq!(envelope["tool_choice"], "auto");
        assert_eq!(envelope["reasoning"]["effort"], "high");
        assert_eq!(envelope["parallel_tool_calls"], false);
        assert_eq!(envelope["temperature"], 0.7);
        assert!(
            envelope["output"].as_array().unwrap().is_empty(),
            "undeclared call became an item"
        );
        assert_eq!(
            envelope["usage"]["output_tokens_details"]["reasoning_tokens"],
            0
        );
    }

    #[test]
    fn failure_resolves_ambiguous_visible_bytes_before_terminal() {
        let mut sink = CollectEvents::default();
        let mut stream = ResponseStream::begin(
            &mut sink,
            "resp_fail".into(),
            "m".into(),
            1,
            EnvelopeEcho::default(),
        )
        .unwrap();
        stream
            .on_partition(&PartitionEvent::Visible("answer <tool_".into()))
            .unwrap();
        stream
            .fail(&crate::serve::items::ServeError::invalid_request(
                None, "boom",
            ))
            .unwrap();
        let failed = &sink.0.last().unwrap().1["response"];
        assert_eq!(failed["output"][0]["content"][0]["text"], "answer <tool_");
        assert!(failed["completed_at"].as_u64().unwrap() > 1);
    }

    #[test]
    fn failure_surfaces_complete_tool_syntax_without_emitting_a_call() {
        let mut sink = CollectEvents::default();
        let mut stream = ResponseStream::begin(
            &mut sink,
            "resp_fail_call".into(),
            "m".into(),
            1,
            EnvelopeEcho::default(),
        )
        .unwrap();
        stream.set_allowed_tools(vec!["write".into()]);
        stream
            .on_partition(&PartitionEvent::Visible(
                "<tool_call>\n<function=write>\n</function>\n</tool_call>".into(),
            ))
            .unwrap();
        stream
            .fail(&crate::serve::items::ServeError::invalid_request(
                None,
                "backend failed",
            ))
            .unwrap();
        assert!(sink.0.iter().all(|(kind, payload)| {
            !matches!(
                kind.as_str(),
                "response.function_call_arguments.delta" | "response.function_call_arguments.done"
            ) && payload["item"]["type"] != "function_call"
        }));
        let failed = &sink.0.last().unwrap().1["response"];
        assert_eq!(failed["output"][0]["status"], "incomplete");
        assert_eq!(
            failed["output"][0]["content"][0]["text"],
            "<tool_call>\n<function=write>\n</function>\n</tool_call>"
        );
    }

    #[test]
    fn reasoning_delta_has_exactly_one_ai_sdk_alias() {
        let (events, _) = drive(&["<think>one chunk</think>answer"], StopReason::Eos);
        assert_eq!(
            events
                .iter()
                .filter(|(kind, _)| kind == "response.reasoning_text.delta")
                .count(),
            1
        );
        assert!(
            events
                .iter()
                .all(|(kind, _)| kind != "response.reasoning_text.done")
        );
    }

    #[test]
    fn function_argument_events_include_the_call_id() {
        let (events, _) = drive(
            &["<tool_call>\n<function=ping>\n</function>\n</tool_call>"],
            StopReason::Eos,
        );
        let call_id = events
            .iter()
            .find(|(kind, _)| kind == "response.output_item.added")
            .unwrap()
            .1["item"]["call_id"]
            .clone();
        for kind in [
            "response.function_call_arguments.delta",
            "response.function_call_arguments.done",
        ] {
            let payload = &events.iter().find(|(event, _)| event == kind).unwrap().1;
            assert_eq!(payload["call_id"], call_id, "missing call_id on {kind}");
        }
    }

    #[test]
    fn terminal_completed_at_is_stamped_at_finish() {
        let (events, envelope) = drive(&["done"], StopReason::Eos);
        assert!(events[0].1["response"]["completed_at"].is_null());
        assert!(envelope["completed_at"].as_u64().unwrap() > 1_755_500_000);
    }

    #[test]
    fn normalized_tool_choice_and_strict_false_are_echoed() {
        let request = crate::serve::items::parse_request(&json!({
            "model":"m", "input":"q",
            "tools":[{"type":"function","name":"a","strict":false}],
            "tool_choice":{"type":"allowed_tools","tools":[
                {"type":"function","name":"a"}
            ]}
        }))
        .unwrap();
        let echo = envelope_echo(&request);
        assert_eq!(echo.tool_choice["mode"], "auto");
        assert_eq!(echo.tools[0]["strict"], false);
    }
}
