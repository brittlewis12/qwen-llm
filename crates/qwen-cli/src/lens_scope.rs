//! Transport-neutral authored position scopes and validation.

use crate::lens_input::{LensRenderedSpan, is_known_lens_span};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub(crate) const MAX_SELECTOR_VALUES: usize = 4096;

pub(crate) const MAX_RENDERED_SELECTOR_TEXT_BYTES: usize = 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Scope {
    pub(crate) layers: Selector,
    #[serde(default)]
    pub(crate) prefill: Option<Selector>,
    #[serde(default)]
    pub(crate) decode: Option<Selector>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(crate) enum Selector {
    All,
    Values {
        values: Vec<u32>,
    },
    Range {
        start: u32,
        end: u32,
    },
    RenderedSpans {
        selectors: Vec<RenderedSpanSelector>,
    },
}

#[derive(Clone, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RenderedSpanSelector {
    pub(crate) span_kind: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) message_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) tool_call_index: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) channel: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) label: Option<String>,
    #[serde(default)]
    pub(crate) occurrence: RenderedSpanOccurrence,
    pub(crate) edge: RenderedSpanEdge,
}

impl Selector {
    pub(crate) fn validate(&self, name: &str, allow_rendered_spans: bool) -> Result<()> {
        match self {
            Self::All => Ok(()),
            Self::Values { values } => {
                ensure!(!values.is_empty(), "{name} values must not be empty");
                ensure!(
                    values.len() <= MAX_SELECTOR_VALUES,
                    "{name} has too many explicit values"
                );
                ensure!(
                    values.windows(2).all(|pair| pair[0] < pair[1]),
                    "{name} values must be sorted and unique"
                );
                Ok(())
            }
            Self::Range { start, end } => {
                ensure!(
                    start <= end,
                    "{name} range must be inclusive with start <= end"
                );
                ensure!(
                    u64::from(*end) - u64::from(*start) < MAX_SELECTOR_VALUES as u64,
                    "{name} range is too large"
                );
                Ok(())
            }
            Self::RenderedSpans { selectors } => {
                ensure!(
                    allow_rendered_spans,
                    "{name} does not support rendered-span selectors"
                );
                ensure!(
                    !selectors.is_empty() && selectors.len() <= MAX_SELECTOR_VALUES,
                    "{name} rendered_spans requires 1..={MAX_SELECTOR_VALUES} selectors"
                );
                ensure!(
                    selectors.iter().collect::<BTreeSet<_>>().len() == selectors.len(),
                    "{name} repeats an authored rendered-span selector"
                );
                for (index, selector) in selectors.iter().enumerate() {
                    selector.validate(&format!("{name}.selectors[{index}]"))?;
                }
                Ok(())
            }
        }
    }

    pub(crate) fn expand(&self, upper_bound: u32, name: &str) -> Result<Vec<u32>> {
        self.validate(name, false)?;
        let values = match self {
            Self::All => (0..upper_bound).collect(),
            Self::Values { values } => values.clone(),
            Self::Range { start, end } => (*start..=*end).collect(),
            Self::RenderedSpans { .. } => unreachable!("validation rejects unresolved selectors"),
        };
        ensure!(
            values.iter().all(|&value| value < upper_bound),
            "{name} contains a value outside 0..{upper_bound}"
        );
        Ok(values)
    }
}

impl RenderedSpanSelector {
    pub(crate) fn validate(&self, name: &str) -> Result<()> {
        ensure!(
            is_known_lens_span(&self.span_kind),
            "{name}.span_kind is not a known renderer span"
        );
        ensure!(
            self.message_index
                .is_none_or(|index| index < MAX_SELECTOR_VALUES)
                && self
                    .tool_call_index
                    .is_none_or(|index| index < MAX_SELECTOR_VALUES),
            "{name} message/tool-call index exceeds the selector bound"
        );
        ensure!(
            self.role
                .as_deref()
                .is_none_or(|role| { matches!(role, "system" | "user" | "assistant" | "tool") }),
            "{name}.role is unsupported"
        );
        ensure!(
            self.channel.as_deref().is_none_or(|channel| {
                matches!(channel, "thinking" | "tool_call" | "tool_result")
            }),
            "{name}.channel is unsupported"
        );
        for (field, value) in [
            ("role", self.role.as_deref()),
            ("channel", self.channel.as_deref()),
            ("label", self.label.as_deref()),
        ] {
            ensure!(
                value.is_none_or(|value| {
                    !value.is_empty() && value.len() <= MAX_RENDERED_SELECTOR_TEXT_BYTES
                }),
                "{name}.{field} is empty or too long"
            );
        }
        Ok(())
    }

    pub(crate) fn matches(&self, span: &LensRenderedSpan) -> bool {
        span.kind == self.span_kind
            && self
                .message_index
                .is_none_or(|value| span.message_index == Some(value))
            && self
                .tool_call_index
                .is_none_or(|value| span.tool_call_index == Some(value))
            && self
                .role
                .as_deref()
                .is_none_or(|value| span.role.as_deref() == Some(value))
            && self
                .channel
                .as_deref()
                .is_none_or(|value| span.channel.as_deref() == Some(value))
            && self
                .label
                .as_deref()
                .is_none_or(|value| span.label.as_deref() == Some(value))
    }
}

impl Scope {
    pub(crate) fn validate(&self, name: &str, plan_version: u32) -> Result<()> {
        self.layers.validate(&format!("{name}.layers"), false)?;
        ensure!(
            self.prefill.is_some() || self.decode.is_some(),
            "{name} must select prefill and/or decode"
        );
        if let Some(selector) = &self.prefill {
            selector.validate(&format!("{name}.prefill"), plan_version == 2)?;
        }
        if let Some(selector) = &self.decode {
            selector.validate(&format!("{name}.decode"), false)?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RenderedSpanOccurrence {
    #[default]
    Unique,
    First,
    Last,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RenderedSpanEdge {
    Start,
    End,
}
