//! GLM-5.3-Flash output with declared tools: the prompt-preopened reasoning
//! grammar (`<think>` … first `</think>`), then visible text routed through
//! `glm5_next_chat::ToolOutputStream`, which holds back tool markers and
//! publishes calls together once the turn's whole tool block parses. A tool
//! marker inside reasoning stays reasoning text.
//!
//! Memory: the block is capped at the request's byte bound and admitted
//! before it grows. Before any push that would begin or extend the block
//! beyond its capacity, the capacity grows by whole [`ADMISSION_STEP`]s, and
//! the process must have headroom for the block's whole outstanding peak at
//! the new capacity: [`tool_block_peak_bytes`] (text, parsed values with
//! their container overhead, and the published serializations) less what
//! the block already holds. The check repeats at every step against current
//! headroom, so earlier steps reserve nothing that later allocations could
//! take, and once more before the block is parsed. A failed check ends the
//! turn with `memory_admission_denied` (503) instead of growing unpriced.
use super::items::ServeError;
use super::output_partition::GenerationEnd;
use super::partition::PartitionEvent;
use super::partition_preopened::PreopenedPartition;
use qwen_llm::glm5_next_chat::{
    ToolDefinition, ToolOutputEnd, ToolOutputStream, tool_block_peak_bytes,
};

/// Block capacity admitted per step.
pub(crate) const ADMISSION_STEP: usize = 64 << 10;

pub(crate) struct Glm5NextToolsPartition {
    reasoning: PreopenedPartition,
    tools: ToolOutputStream,
    max_bytes: usize,
    failure: Option<ServeError>,
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
            max_bytes,
            failure: None,
            headroom,
        }
    }

    /// Headroom for the block's outstanding peak at `capacity`, given what
    /// it already holds.
    fn admit_outstanding(&self, capacity: usize) -> Result<(), ServeError> {
        let held = self.tools.block_capacity();
        let outstanding = tool_block_peak_bytes(capacity).saturating_sub(held);
        super::transport_memory::admit_resident_transport(outstanding as u64, (self.headroom)())
    }

    /// Before a push that makes the block `len` bytes: grow its capacity
    /// by whole steps (within the byte bound), admitted first.
    fn admit_block(&mut self, len: usize) -> Result<(), ServeError> {
        if len <= self.tools.block_capacity() || len > self.max_bytes {
            // Fits, or the stream refuses it over the bound without growing.
            return Ok(());
        }
        let capacity = len
            .div_ceil(ADMISSION_STEP)
            .saturating_mul(ADMISSION_STEP)
            .min(self.max_bytes);
        self.admit_outstanding(capacity)?;
        self.tools.reserve_block(capacity);
        Ok(())
    }

    fn route(&mut self, incoming: Vec<PartitionEvent>, events: &mut Vec<PartitionEvent>) {
        for event in incoming {
            if self.failure.is_some() {
                return;
            }
            if let PartitionEvent::Visible(text) = event {
                let len = self.tools.block_len_after(&text);
                if len > 0
                    && let Err(error) = self.admit_block(len)
                {
                    self.failure = Some(error);
                    return;
                }
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
        // Headroom may have shrunk since the last step: price the parse and
        // publication once more against it.
        let capacity = self.tools.block_capacity();
        if capacity > 0 {
            self.admit_outstanding(capacity)?;
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

    const STOP: GenerationEnd = GenerationEnd::StopToken(154_829);

    std::thread_local! {
        static HEADROOM: std::cell::Cell<u64> = const { std::cell::Cell::new(u64::MAX) };
    }

    fn scripted() -> Option<u64> {
        Some(HEADROOM.get())
    }

    fn peak(capacity: usize) -> u64 {
        tool_block_peak_bytes(capacity) as u64
    }

    /// Ample headroom publishes the call; headroom below the first step's
    /// outstanding peak fails the turn with memory_admission_denied before
    /// the block allocates; plain text with no block never asks.
    #[test]
    fn tool_block_growth_is_admitted_before_it_allocates() {
        let call = "</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>Paris</arg_value></tool_call>";
        let run = |headroom: u64, output: &str| {
            HEADROOM.set(headroom);
            let mut partition =
                Glm5NextToolsPartition::with_headroom(vec![weather()], 1 << 24, scripted);
            let mut events = Vec::new();
            partition.push(output.as_bytes(), &mut events);
            let held = partition.tools.block_capacity();
            (held, partition.finish(STOP, &mut events).map(|()| events))
        };
        let (held, events) = run(peak(ADMISSION_STEP), call);
        assert_eq!(held, ADMISSION_STEP, "capacity is the admitted step");
        assert!(
            events
                .unwrap()
                .iter()
                .any(|e| matches!(e, PartitionEvent::FunctionCall(_)))
        );
        let (held, error) = run(peak(ADMISSION_STEP) - 1, call);
        assert_eq!(held, 0, "a refused step allocates no block");
        let error = error.unwrap_err();
        assert_eq!(error.status, 503);
        assert_eq!(error.code, Some("memory_admission_denied"));
        // Text with no tool block buffers nothing and asks nothing.
        assert!(run(0x10, "</think>Plain answer.").1.is_ok());
    }

    /// The block grows over several steps against fixed headroom. Each step
    /// prices the whole outstanding peak at the new capacity, so headroom
    /// that covers two steps' peak (less the first step's text) admits the
    /// second step and refuses the third, before it allocates.
    #[test]
    fn every_step_prices_the_whole_outstanding_peak() {
        HEADROOM.set(peak(2 * ADMISSION_STEP) - ADMISSION_STEP as u64);
        let mut partition =
            Glm5NextToolsPartition::with_headroom(vec![weather()], 1 << 24, scripted);
        let mut events = Vec::new();
        partition.push(
            b"</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>",
            &mut events,
        );
        assert_eq!(partition.tools.block_capacity(), ADMISSION_STEP);
        let filler = "a".repeat(ADMISSION_STEP / 4);
        let mut steps = Vec::new();
        for _ in 0..12 {
            partition.push(filler.as_bytes(), &mut events);
            steps.push((
                partition.tools.block_capacity(),
                partition.failure.is_some(),
            ));
            if partition.failure.is_some() {
                break;
            }
        }
        let refused = steps.iter().position(|&(_, failed)| failed).unwrap();
        // Up to two steps fit; the push that needs a third is refused, and
        // the capacity never ran ahead of an admitted step.
        assert!(
            steps[..refused]
                .iter()
                .all(|&(c, _)| c <= 2 * ADMISSION_STEP)
        );
        assert_eq!(steps[refused].0, 2 * ADMISSION_STEP);
        assert!(partition.tools.buffered_bytes() <= 2 * ADMISSION_STEP);
        let error = partition.finish(STOP, &mut events).unwrap_err();
        assert_eq!(error.code, Some("memory_admission_denied"));
    }

    /// Headroom falls while the block grows: a step admitted earlier is not a
    /// reservation, so the next step is priced against what is left, and a
    /// drop after the last step fails the parse at finish.
    #[test]
    fn decreasing_headroom_is_rechecked_at_each_step_and_before_the_parse() {
        let head = b"</think><tool_call>get_weather<arg_key>city</arg_key><arg_value>";
        let tail = b"</arg_value></tool_call>";
        let filler = "a".repeat(ADMISSION_STEP);
        // Ample for the first step, then too little for the second.
        HEADROOM.set(u64::MAX);
        let mut partition =
            Glm5NextToolsPartition::with_headroom(vec![weather()], 1 << 24, scripted);
        let mut events = Vec::new();
        partition.push(head, &mut events);
        assert!(partition.failure.is_none());
        HEADROOM.set(peak(2 * ADMISSION_STEP) - ADMISSION_STEP as u64 - 1);
        partition.push(filler.as_bytes(), &mut events);
        assert!(
            partition.failure.is_some(),
            "the second step was not rechecked"
        );
        assert_eq!(partition.tools.block_capacity(), ADMISSION_STEP);
        // Every step admitted, then the parse is refused at finish.
        HEADROOM.set(u64::MAX);
        let mut partition =
            Glm5NextToolsPartition::with_headroom(vec![weather()], 1 << 24, scripted);
        partition.push(head, &mut events);
        partition.push(filler.as_bytes(), &mut events);
        partition.push(tail, &mut events);
        assert!(partition.failure.is_none());
        let capacity = partition.tools.block_capacity();
        HEADROOM.set(peak(capacity) - capacity as u64 - 1);
        let error = partition.finish(STOP, &mut events).unwrap_err();
        assert_eq!(error.code, Some("memory_admission_denied"));
        HEADROOM.set(u64::MAX);
    }

    /// A container-heavy argument is priced like any other block byte: the
    /// model's factor covers its parse (measured in qwen-llm's
    /// glm53_tool_block_peak), so admission depends only on the block's
    /// length and publishes the parsed tree.
    #[test]
    fn container_heavy_blocks_publish_within_the_admitted_model() {
        let items = vec!["[[1]]"; 20_000].join(",");
        let schema = ToolDefinition::from_value(&json!({"name": "f", "parameters": {
            "type": "object", "properties": {"a": {"type": "array"}}}}))
        .unwrap();
        let output = format!(
            "</think><tool_call>f<arg_key>a</arg_key><arg_value>[{items}]</arg_value></tool_call>"
        );
        let block = output.len() - "</think>".len();
        let capacity = block.div_ceil(ADMISSION_STEP) * ADMISSION_STEP;
        HEADROOM.set(peak(capacity));
        let mut partition = Glm5NextToolsPartition::with_headroom(vec![schema], 1 << 24, scripted);
        let mut events = Vec::new();
        for piece in output.as_bytes().chunks(509) {
            partition.push(piece, &mut events);
        }
        assert_eq!(partition.tools.block_capacity(), capacity);
        partition.finish(STOP, &mut events).unwrap();
        let PartitionEvent::FunctionCall(call) = events.last().unwrap() else {
            panic!("no call published");
        };
        assert_eq!(call.arguments["a"].as_array().unwrap().len(), 20_000);
        HEADROOM.set(u64::MAX);
    }
}
