//! No-tools IFM output grammar. Prompt-preopened reasoning never becomes an
//! answer merely because a budget or malformed termination cut it short.
use super::items::ServeError;
use super::output_partition::GenerationEnd;
use super::partition::{PartitionEvent, safe_emit_len};
use super::utf8::Utf8Assembler;
use qwen_llm::k2_horizon_chat::{CHAT_STOPS, Effort};

pub(crate) struct K2Partition {
    utf8: Utf8Assembler,
    open: String,
    close: String,
    at_start: bool,
    closed: bool,
    emitted_reasoning: bool,
    pending: String,
}

impl K2Partition {
    pub(crate) fn new(effort: Effort) -> Self {
        Self {
            utf8: Utf8Assembler::new(),
            open: format!("<{}>", effort.tag()),
            close: format!("</{}>", effort.tag()),
            at_start: true,
            closed: false,
            emitted_reasoning: false,
            pending: String::new(),
        }
    }
    pub(crate) fn closed(&self) -> bool {
        self.closed
    }
    pub(crate) fn push(&mut self, bytes: &[u8], events: &mut Vec<PartitionEvent>) {
        let text = self.utf8.push(bytes);
        self.text(&text, events);
    }
    fn reasoning(&mut self, text: String, events: &mut Vec<PartitionEvent>) {
        self.emitted_reasoning = true;
        events.push(PartitionEvent::Reasoning(text));
    }
    fn text(&mut self, text: &str, events: &mut Vec<PartitionEvent>) {
        self.pending.push_str(text);
        if self.at_start {
            if self.open.starts_with(&self.pending) && self.pending.len() < self.open.len() {
                return;
            }
            if self.pending.starts_with(&self.open) {
                self.pending.drain(..self.open.len());
            }
            self.at_start = false;
        }
        if !self.closed {
            if let Some(index) = self.pending.find(&self.close) {
                let reasoning = self.pending[..index].to_owned();
                self.pending.drain(..index + self.close.len());
                if !reasoning.is_empty() || !self.emitted_reasoning {
                    self.reasoning(reasoning, events);
                }
                events.push(PartitionEvent::ReasoningClosed);
                self.closed = true;
            } else {
                let safe = safe_emit_len(&self.pending, &self.close);
                if safe > 0 {
                    let text = self.pending[..safe].to_owned();
                    self.pending.drain(..safe);
                    self.reasoning(text, events);
                }
                return;
            }
        }
        if !self.pending.is_empty() {
            events.push(PartitionEvent::Visible(std::mem::take(&mut self.pending)));
        }
    }
    pub(crate) fn finish(
        mut self,
        end: GenerationEnd,
        events: &mut Vec<PartitionEvent>,
    ) -> Result<(), ServeError> {
        let text = self.utf8.finish();
        self.text(&text, events);
        if !self.closed {
            // Flush a truncated delimiter as reasoning, never as final text.
            let pending = std::mem::take(&mut self.pending);
            if !pending.is_empty() || !self.emitted_reasoning {
                self.reasoning(pending, events);
            }
            if !end.is_token_limit() {
                return Err(ServeError::server_error(
                    "invalid K2 model output: stop before reasoning close",
                ));
            }
        }
        if let GenerationEnd::StopToken(id) = end
            && !CHAT_STOPS.contains(&id)
        {
            return Err(ServeError::server_error(
                "invalid K2 model output: unsupported stop token",
            ));
        }
        Ok(())
    }
    pub(crate) fn abort(self, _: &mut Vec<PartitionEvent>) {
        // No completion or guessed interpretation of pending delimiter/UTF-8 bytes.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn collect(events: &[PartitionEvent]) -> (String, String, usize) {
        let (mut r, mut v, mut close) = (String::new(), String::new(), 0);
        for e in events {
            match e {
                PartitionEvent::Reasoning(t) => r.push_str(t),
                PartitionEvent::Visible(t) => v.push_str(t),
                PartitionEvent::ReasoningClosed => close += 1,
                _ => panic!("tool event"),
            }
        }
        (r, v, close)
    }
    #[test]
    fn k2_every_byte_split_preserves_utf8_and_only_matching_first_close() {
        for effort in [Effort::High, Effort::Medium, Effort::Low] {
            for opener in [String::new(), format!("<{}>", effort.tag())] {
                for reason in ["", "plan \u{1f389} <think>x</think><ifm|tool_calls>literal"] {
                    let visible =
                        "answer \u{2192}<ifm|think>literal</ifm|think><tool_call>text</tool_call>";
                    let all = format!("{opener}{reason}</{}>{visible}", effort.tag());
                    for split in 0..=all.len() {
                        let mut p = K2Partition::new(effort);
                        let mut events = Vec::new();
                        p.push(&all.as_bytes()[..split], &mut events);
                        p.push(&all.as_bytes()[split..], &mut events);
                        assert!(p.closed());
                        p.finish(GenerationEnd::StopToken(250019), &mut events)
                            .unwrap();
                        assert_eq!(collect(&events), (reason.into(), visible.into(), 1));
                        assert!(matches!(events.first(), Some(PartitionEvent::Reasoning(_))));
                    }
                }
            }
        }
    }
    #[test]
    fn k2_truncated_reasoning_is_incomplete_and_bad_stops_fail() {
        for text in [
            "",
            "p</ifm|thi",
            "<ifm|thi",
            "p</ifm|think_fast>",
            "<ifm|think>",
        ] {
            for end in [
                GenerationEnd::TokenLimit,
                GenerationEnd::StopToken(1),
                GenerationEnd::StopToken(250019),
            ] {
                let mut p = K2Partition::new(Effort::High);
                let mut events = Vec::new();
                for b in text.as_bytes() {
                    p.push(&[*b], &mut events);
                }
                assert_eq!(p.finish(end, &mut events).is_ok(), end.is_token_limit());
                assert!(collect(&events).1.is_empty());
                assert_eq!(collect(&events).2, 0);
            }
        }
        let mut p = K2Partition::new(Effort::High);
        let mut events = Vec::new();
        p.push(b"a</ifm|think>x", &mut events);
        assert!(p.finish(GenerationEnd::StopToken(42), &mut events).is_err());
    }
    #[test]
    fn k2_abort_discards_ambiguity_and_invalid_utf8_is_replaced() {
        let mut p = K2Partition::new(Effort::High);
        let mut events = Vec::new();
        p.push(b"plan</ifm|thi", &mut events);
        p.abort(&mut events);
        assert_eq!(collect(&events), ("plan".into(), "".into(), 0));
        let mut p = K2Partition::new(Effort::High);
        let mut events = Vec::new();
        p.push(b"\xff</ifm|think>\xf0\x9f", &mut events);
        p.finish(GenerationEnd::TokenLimit, &mut events).unwrap();
        assert_eq!(collect(&events), ("\u{fffd}".into(), "\u{fffd}".into(), 1));
    }
}
