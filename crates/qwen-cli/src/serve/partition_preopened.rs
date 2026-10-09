//! Output grammar for a generation whose prompt pre-opened reasoning (K2
//! Horizon, GLM-5.3-Flash): bytes are reasoning until the first released
//! close tag, then the answer. Reasoning never becomes an answer merely
//! because a budget or malformed termination cut it short.
use super::items::ServeError;
use super::output_partition::GenerationEnd;
use super::partition::{PartitionEvent, safe_emit_len};
use super::utf8::Utf8Assembler;

/// A family's pre-opened reasoning grammar.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct PreopenedGrammar {
    /// Names the family in output errors.
    pub(crate) family: &'static str,
    /// The opener the prompt already wrote: one leading repeat is dropped.
    pub(crate) open: String,
    /// Released reasoning terminators; the first one seen closes.
    pub(crate) closes: &'static [&'static str],
    /// Released stop tokens.
    pub(crate) stops: &'static [i32],
}

pub(crate) struct PreopenedPartition {
    grammar: PreopenedGrammar,
    utf8: Utf8Assembler,
    at_start: bool,
    closed: bool,
    emitted_reasoning: bool,
    pending: String,
}

impl PreopenedPartition {
    pub(crate) fn new(grammar: PreopenedGrammar) -> Self {
        Self {
            grammar,
            utf8: Utf8Assembler::new(),
            at_start: true,
            closed: false,
            emitted_reasoning: false,
            pending: String::new(),
        }
    }

    #[cfg(test)]
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
        let open = self.grammar.open.as_str();
        if self.at_start {
            if open.starts_with(&self.pending) && self.pending.len() < open.len() {
                return;
            }
            if self.pending.starts_with(open) {
                self.pending.drain(..open.len());
            }
            self.at_start = false;
        }
        if !self.closed {
            // Effort controls the prompt, not which released terminator the
            // model emits. Never infer closure from a tool marker or a stop.
            if let Some((index, close)) = self
                .grammar
                .closes
                .iter()
                .filter_map(|close| self.pending.find(close).map(|index| (index, close)))
                .min_by_key(|(index, _)| *index)
            {
                let reasoning = self.pending[..index].to_owned();
                self.pending.drain(..index + close.len());
                if !reasoning.is_empty() || !self.emitted_reasoning {
                    self.reasoning(reasoning, events);
                }
                events.push(PartitionEvent::ReasoningClosed);
                self.closed = true;
            } else {
                let safe = self
                    .grammar
                    .closes
                    .iter()
                    .map(|close| safe_emit_len(&self.pending, close))
                    .min()
                    .unwrap();
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
        let family = self.grammar.family;
        if !self.closed {
            // Flush a truncated delimiter as reasoning, never as final text.
            let pending = std::mem::take(&mut self.pending);
            if !pending.is_empty() || !self.emitted_reasoning {
                self.reasoning(pending, events);
            }
            if !end.is_token_limit() {
                return Err(ServeError::server_error(format!(
                    "invalid {family} model output: stop before reasoning close"
                )));
            }
        }
        if let GenerationEnd::StopToken(id) = end
            && !self.grammar.stops.contains(&id)
        {
            return Err(ServeError::server_error(format!(
                "invalid {family} model output: unsupported stop token"
            )));
        }
        Ok(())
    }

    pub(crate) fn abort(self, _: &mut Vec<PartitionEvent>) {
        // No completion or guessed interpretation of pending delimiter/UTF-8 bytes.
    }
}
