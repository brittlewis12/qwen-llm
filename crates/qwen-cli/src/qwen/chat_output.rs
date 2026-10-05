//! `qwen run` chat output: reasoning streams to stderr, the answer to stdout.

use crate::serve::partition::PartitionEvent;
use anyhow::{Result, bail};
use std::io::Write;

/// Writes partition events as they arrive; `visible` records whether any
/// answer text reached stdout. A no-tools grammar never yields a call.
pub(crate) fn write_chat_events(
    events: &[PartitionEvent],
    stdout: &mut impl Write,
    stderr: &mut impl Write,
    visible: &mut bool,
    family: &str,
) -> Result<()> {
    for event in events {
        match event {
            PartitionEvent::Reasoning(text) => stderr.write_all(text.as_bytes())?,
            PartitionEvent::Visible(text) => {
                stdout.write_all(text.as_bytes())?;
                *visible |= !text.is_empty();
            }
            PartitionEvent::ReasoningClosed => {}
            PartitionEvent::FunctionCall(_) => {
                bail!("{family} no-tools partition produced a tool call")
            }
        }
    }
    stdout.flush()?;
    stderr.flush()?;
    Ok(())
}
