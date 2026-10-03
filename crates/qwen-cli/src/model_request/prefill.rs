//! An unfinished assistant continuation is prompt content, not a history turn.

use crate::messages::{MessageRenderChannel, MessageRenderSpanKind};
use crate::open_responses::render::QwenGeneration;
use crate::prompt_template::QwenPromptTemplate;
use anyhow::{Result, bail, ensure};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AssistantPrefillChannel {
    Reasoning,
    Final,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct AssistantPrefill {
    pub(crate) channel: AssistantPrefillChannel,
    pub(crate) text: String,
}

impl AssistantPrefill {
    pub(crate) fn validate_qwen(&self) -> Result<()> {
        for marker in [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<think>",
            "</think>",
            "<tool_call>",
            "</tool_call>",
            "<tool_response>",
            "</tool_response>",
        ] {
            ensure!(
                !self.text.contains(marker),
                "assistant prefill contains structural protocol marker {marker:?}"
            );
        }
        Ok(())
    }
}

pub(crate) type TransitionSpan = (
    &'static str,
    MessageRenderSpanKind,
    Option<MessageRenderChannel>,
);

const CLOSE_REASONING: &[TransitionSpan] = &[
    (
        "\n",
        MessageRenderSpanKind::ContentSeparator,
        Some(MessageRenderChannel::Thinking),
    ),
    (
        "</think>",
        MessageRenderSpanKind::ThinkingChannelEndMarker,
        Some(MessageRenderChannel::Thinking),
    ),
    ("\n\n", MessageRenderSpanKind::ContentSeparator, None),
];

pub(crate) fn qwen_transition(
    protocol: QwenPromptTemplate,
    initial: QwenGeneration,
    prefill: &AssistantPrefill,
) -> Result<(&'static [TransitionSpan], QwenGeneration)> {
    ensure!(
        matches!(
            protocol,
            QwenPromptTemplate::Qwen36 | QwenPromptTemplate::Qwen38
        ),
        "assistant prefill is unsupported for this prompt protocol"
    );
    prefill.validate_qwen()?;
    match (prefill.channel, initial) {
        (AssistantPrefillChannel::Reasoning, QwenGeneration::PreOpen) => Ok((&[], initial)),
        (AssistantPrefillChannel::Reasoning, _) => {
            bail!("reasoning assistant prefill requires thinking mode")
        }
        (AssistantPrefillChannel::Final, QwenGeneration::PreOpen) => {
            Ok((CLOSE_REASONING, QwenGeneration::PreClosed))
        }
        (AssistantPrefillChannel::Final, QwenGeneration::PreClosed) => Ok((&[], initial)),
        _ => bail!("assistant prefill requires a verified channel boundary"),
    }
}
