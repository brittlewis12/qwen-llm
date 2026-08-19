//! Response object construction and sequence-numbered streaming events.
//!
//! Every streamed event carries a monotonic `sequence_number` (SERVE.md,
//! review defect 4) and is delivered through [`EventWrite`], so tests
//! assert exact event order and payloads without any socket. SSE framing
//! (`event:`/`data:` lines, heartbeat comments, `[DONE]`) lives in
//! [`SseWriter`]; the HTTP slice owns the socket.
//!
//! Part-type note: reasoning item content parts use `reasoning_text`
//! (matching the OpenAI-lineage `response.reasoning_text.delta` events this
//! module emits). The gate-5 conformance run adjudicates this choice; it is
//! isolated behind `REASONING_PART_TYPE`.

use super::items::ServeRequest;
use super::partition::PartitionEvent;
use serde_json::{Value, json};
use std::io::{self, Write};
use std::time::{Duration, Instant};

pub(crate) const REASONING_PART_TYPE: &str = "reasoning_text";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopReason {
    Eos,
    TokenLimit,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct Usage {
    pub(crate) input_tokens: usize,
    pub(crate) output_tokens: usize,
}

impl Usage {
    fn to_json(&self) -> Value {
        json!({
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        })
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
    open: OpenItem,
    output_index: usize,
    reasoning_item_id: String,
    message_item_id: String,
    reasoning_text: String,
    visible_text: String,
    reasoning_opened: bool,
    message_opened: bool,
    last_activity: Instant,
}

impl<'a, W: EventWrite> ResponseStream<'a, W> {
    pub(crate) fn begin(
        writer: &'a mut W,
        response_id: String,
        model: String,
        created_at: u64,
    ) -> io::Result<Self> {
        let mut stream = Self {
            writer,
            sequence_number: 0,
            response_id,
            model,
            created_at,
            open: OpenItem::None,
            output_index: 0,
            reasoning_item_id: String::new(),
            message_item_id: String::new(),
            reasoning_text: String::new(),
            visible_text: String::new(),
            reasoning_opened: false,
            message_opened: false,
            last_activity: Instant::now(),
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
        let mut envelope = json!({
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "model": self.model,
            "status": status,
            "output": output,
            "store": false,
        });
        if let Some(usage) = usage {
            envelope["usage"] = usage.to_json();
        }
        if let Some(reason) = incomplete_reason {
            envelope["incomplete_details"] = json!({"reason": reason});
        }
        envelope
    }

    fn open_reasoning(&mut self) -> io::Result<()> {
        debug_assert_eq!(self.open, OpenItem::None);
        self.reasoning_opened = true;
        self.open = OpenItem::Reasoning;
        let item = json!({
            "id": self.reasoning_item_id,
            "type": "reasoning",
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
                "part": {"type": "output_text", "text": ""},
            }),
        )
    }

    fn close_reasoning(&mut self, status: &str) -> io::Result<()> {
        debug_assert_eq!(self.open, OpenItem::Reasoning);
        let item_id = self.reasoning_item_id.clone();
        let output_index = self.output_index;
        let text = self.reasoning_text.clone();
        self.emit(
            "response.reasoning_text.done",
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
        let part = json!({"type": "output_text", "text": self.visible_text});
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
            "content": [{"type": REASONING_PART_TYPE, "text": self.reasoning_text}],
        })
    }

    fn message_item_json(&self, status: &str) -> Value {
        json!({
            "id": self.message_item_id,
            "type": "message",
            "role": "assistant",
            "status": status,
            "content": [{"type": "output_text", "text": self.visible_text}],
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
        }
    }

    /// Close open items, emit the terminal response event, and return the
    /// final response object (shared with the non-stream path).
    pub(crate) fn finish(
        mut self,
        stop_reason: StopReason,
        usage: Usage,
        stats: Option<&ServeStats>,
    ) -> io::Result<Value> {
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
            let status = if stop_reason == StopReason::TokenLimit && !truncated_in_reasoning {
                "incomplete"
            } else {
                "completed"
            };
            output.push(self.message_item_json(status));
        }
        let (status, event_type, incomplete_reason) = match stop_reason {
            StopReason::Eos => ("completed", "response.completed", None),
            StopReason::TokenLimit => (
                "incomplete",
                "response.incomplete",
                Some("max_output_tokens"),
            ),
        };
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
    let mut stream =
        ResponseStream::begin(&mut sink, response_id, request.model.clone(), created_at)?;
    for event in partition_events {
        stream.on_partition(event)?;
    }
    stream.finish(stop_reason, usage, stats)
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
        )
        .unwrap();
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
                "response.reasoning_text.delta",
                "response.reasoning_text.done",
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
            },
            None,
        )
        .unwrap();
        assert_eq!(built, streamed);
    }
}
