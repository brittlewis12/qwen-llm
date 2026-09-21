//! Legacy option admission per family lane.
//!
//! One vocabulary of the lane-varying CLI options, one detector for "was
//! this option supplied" (explicit for defaulted values, never a literal
//! default comparison), and one table per family lane listing what it
//! honours. A lane rejects everything supplied outside its table in its
//! own envelope message; value-dependent rules (durable tuning needs
//! `--durable-prefix-cache`, DS4 preserve-thinking needs a thinking tier,
//! DS4 `--max-context-tokens` per input source) stay with the lane.

use anyhow::{Result, ensure};

use super::args::{Args, ExplicitCliOptions};
use super::jsonl::RequestErrorPolicy;
use crate::DeepSeekV4MultigroupSelectorArg;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum LegacyOption {
    PromptLookup,
    PrefillChunk,
    PrefixCacheMaxMib,
    CachePrefixTokens,
    CachePrefixAutoMinTokens,
    RequestStats,
    RequestTimings,
    RequestTimingWarmFollowup,
    ModelPrefetch,
    MessagesNoGenerationPrompt,
    MessagesPreserveThinking,
    MessagesStripThinking,
    RequestsJsonl,
    BatchSize,
    Concurrency,
    ExecutionMode,
    OnRequestErrorContinue,
    DurablePrefixCache,
    DurablePrefixCacheMaxMib,
    DurablePrefixCacheMaxEntryMib,
    DurablePrefixCacheMinTokens,
    SamplingAttribution,
    SampledStructural,
    TraceRequest,
    Reasoning,
    PreserveReasoning,
    DeepSeekV4Snapshot,
    DeepSeekV4MultigroupSelector,
    MaxContextTokens,
    RequestStatsJsonl,
}

impl LegacyOption {
    pub(crate) fn flag(self) -> &'static str {
        match self {
            Self::PromptLookup => "--prompt-lookup",
            Self::PrefillChunk => "--prefill-chunk",
            Self::PrefixCacheMaxMib => "--prefix-cache-max-mib",
            Self::CachePrefixTokens => "--cache-prefix-tokens",
            Self::CachePrefixAutoMinTokens => "--cache-prefix-auto-min-tokens",
            Self::RequestStats => "--request-stats",
            Self::RequestTimings => "--request-timings",
            Self::RequestTimingWarmFollowup => "--request-timing-warm-followup",
            Self::ModelPrefetch => "--model-prefetch",
            Self::MessagesNoGenerationPrompt => "--messages-no-generation-prompt",
            Self::MessagesPreserveThinking => "--messages-preserve-thinking",
            Self::MessagesStripThinking => "--messages-strip-thinking",
            Self::RequestsJsonl => "--requests-jsonl",
            Self::BatchSize => "--batch-size",
            Self::Concurrency => "--concurrency",
            Self::ExecutionMode => "--execution-mode",
            Self::OnRequestErrorContinue => "--on-request-error continue",
            Self::DurablePrefixCache => "--durable-prefix-cache",
            Self::DurablePrefixCacheMaxMib => "--durable-prefix-cache-max-mib",
            Self::DurablePrefixCacheMaxEntryMib => "--durable-prefix-cache-max-entry-mib",
            Self::DurablePrefixCacheMinTokens => "--durable-prefix-cache-min-tokens",
            Self::SamplingAttribution => "--sampling-attribution",
            Self::SampledStructural => "--sampled-structural",
            Self::TraceRequest => "--trace-request",
            Self::Reasoning => "--reasoning",
            Self::PreserveReasoning => "--preserve-reasoning",
            Self::DeepSeekV4Snapshot => "--deepseek-v4-snapshot",
            Self::DeepSeekV4MultigroupSelector => "--deepseek-v4-multigroup-selector",
            Self::MaxContextTokens => "--max-context-tokens",
            Self::RequestStatsJsonl => "--request-stats-jsonl",
        }
    }
}

/// Every lane-varying option the invocation supplied. Defaulted values
/// count only when given on the command line (`ExplicitCliOptions`);
/// `--on-request-error stop` and `--deepseek-v4-multigroup-selector auto`
/// are their defaults and count only when explicit.
pub(crate) fn supplied(args: &Args, explicit: ExplicitCliOptions) -> Vec<LegacyOption> {
    use LegacyOption as O;
    let candidates: [(O, bool); 30] = [
        (O::PromptLookup, args.prompt_lookup),
        (O::PrefillChunk, explicit.prefill_chunk),
        (O::PrefixCacheMaxMib, explicit.prefix_cache_max_mib),
        (O::CachePrefixTokens, args.cache_prefix_tokens.is_some()),
        (
            O::CachePrefixAutoMinTokens,
            explicit.cache_prefix_auto_min_tokens,
        ),
        (O::RequestStats, args.request_stats.is_some()),
        (O::RequestTimings, args.request_timings.is_some()),
        (
            O::RequestTimingWarmFollowup,
            args.request_timing_warm_followup,
        ),
        (O::ModelPrefetch, args.model_prefetch.is_some()),
        (
            O::MessagesNoGenerationPrompt,
            args.messages_no_generation_prompt,
        ),
        (O::MessagesPreserveThinking, args.messages_preserve_thinking),
        (O::MessagesStripThinking, args.messages_strip_thinking),
        (O::RequestsJsonl, args.requests_jsonl.is_some()),
        (O::BatchSize, args.batch_size.is_some()),
        (O::Concurrency, args.concurrency.is_some()),
        (O::ExecutionMode, args.execution_mode.is_some()),
        (
            O::OnRequestErrorContinue,
            args.on_request_error != RequestErrorPolicy::Stop,
        ),
        (O::DurablePrefixCache, args.durable_prefix_cache.is_some()),
        (
            O::DurablePrefixCacheMaxMib,
            explicit.durable_prefix_cache_max_mib,
        ),
        (
            O::DurablePrefixCacheMaxEntryMib,
            explicit.durable_prefix_cache_max_entry_mib,
        ),
        (
            O::DurablePrefixCacheMinTokens,
            explicit.durable_prefix_cache_min_tokens,
        ),
        (O::SamplingAttribution, args.sampling_attribution),
        (O::SampledStructural, args.sampled_structural),
        (O::TraceRequest, args.trace_request.is_some()),
        (O::Reasoning, args.reasoning.is_some()),
        (O::PreserveReasoning, args.preserve_reasoning),
        (O::DeepSeekV4Snapshot, args.deepseek_v4_snapshot.is_some()),
        (
            O::DeepSeekV4MultigroupSelector,
            explicit.deepseek_v4_multigroup_selector
                || args.deepseek_v4_multigroup_selector != DeepSeekV4MultigroupSelectorArg::Auto,
        ),
        (O::MaxContextTokens, args.max_context_tokens.is_some()),
        (O::RequestStatsJsonl, args.request_stats_jsonl.is_some()),
    ];
    candidates
        .into_iter()
        .filter_map(|(option, set)| set.then_some(option))
        .collect()
}

/// What one family lane honours, and how it names itself when refusing.
pub(crate) struct LaneAdmission {
    pub(crate) envelope: &'static str,
    pub(crate) supported: &'static [LegacyOption],
}

impl LaneAdmission {
    /// Flags of every supplied option outside the table. Lanes append their
    /// value-dependent refusals before `refuse`, so one error names all.
    pub(crate) fn unsupported(&self, supplied: &[LegacyOption]) -> Vec<&'static str> {
        supplied
            .iter()
            .filter(|option| !self.supported.contains(option))
            .map(|option| option.flag())
            .collect()
    }

    pub(crate) fn refuse(&self, unsupported: &[&str]) -> Result<()> {
        ensure!(
            unsupported.is_empty(),
            "{}; unsupported options: {}",
            self.envelope,
            unsupported.join(", ")
        );
        Ok(())
    }

    pub(crate) fn admit(&self, supplied: &[LegacyOption]) -> Result<()> {
        self.refuse(&self.unsupported(supplied))
    }
}

use LegacyOption as O;

pub(crate) const FLASH_NEXT_SINGLE_TURN: LaneAdmission = LaneAdmission {
    envelope: "Qwen3.8-Flash-Next currently supports request-shaped serial single-turn generation only",
    supported: &[O::MaxContextTokens, O::RequestStatsJsonl],
};

pub(crate) const MUSE_GLIMMER_SINGLE_TURN: LaneAdmission = LaneAdmission {
    envelope: "Muse Glimmer currently supports request-shaped serial text generation only",
    supported: &[O::MaxContextTokens, O::RequestStatsJsonl],
};

pub(crate) const K2_RAW_SINGLE_TURN: LaneAdmission = LaneAdmission {
    envelope: "K2 Horizon currently supports bounded raw single-turn generation only",
    supported: &[O::MaxContextTokens, O::RequestStatsJsonl],
};

/// Durable tuning admits here and binds against `--durable-prefix-cache`
/// in the lane; preserve-thinking binds against the reasoning tier.
pub(crate) const DEEPSEEK_V4_SINGLE_TURN: LaneAdmission = LaneAdmission {
    envelope: "DeepSeek V4 currently supports bounded raw or ordinary-message single-turn generation only",
    supported: &[
        O::DurablePrefixCache,
        O::DurablePrefixCacheMaxMib,
        O::DurablePrefixCacheMaxEntryMib,
        O::DurablePrefixCacheMinTokens,
        O::TraceRequest,
        O::MessagesPreserveThinking,
        O::MessagesStripThinking,
        O::Reasoning,
        O::PreserveReasoning,
        O::DeepSeekV4Snapshot,
        O::DeepSeekV4MultigroupSelector,
        O::RequestStatsJsonl,
    ],
};

/// `--max-context-tokens` admits here and binds per input source in the
/// lane (required from stdin, rejected for a file).
pub(crate) const DEEPSEEK_V4_BATCH: LaneAdmission = LaneAdmission {
    envelope: "DeepSeek V4 --requests-jsonl supports raw prompt requests only",
    supported: &[
        O::RequestsJsonl,
        O::Concurrency,
        O::ExecutionMode,
        O::TraceRequest,
        O::DeepSeekV4MultigroupSelector,
        O::MaxContextTokens,
    ],
};

/// `--cache-prefix-tokens` and durable tuning bind against
/// `--durable-prefix-cache` in the lane.
pub(crate) const ORDINARY_QWEN_SINGLE_TURN: LaneAdmission = LaneAdmission {
    envelope: "ordinary Qwen single-turn generation does not support every legacy option",
    supported: &[
        O::PromptLookup,
        O::PrefillChunk,
        O::PrefixCacheMaxMib,
        O::CachePrefixTokens,
        O::RequestTimings,
        O::RequestTimingWarmFollowup,
        O::MessagesNoGenerationPrompt,
        O::MessagesPreserveThinking,
        O::MessagesStripThinking,
        O::DurablePrefixCache,
        O::DurablePrefixCacheMaxMib,
        O::DurablePrefixCacheMaxEntryMib,
        O::DurablePrefixCacheMinTokens,
        O::SamplingAttribution,
        O::SampledStructural,
        O::TraceRequest,
        O::MaxContextTokens,
        O::RequestStatsJsonl,
    ],
};

pub(crate) const ORDINARY_QWEN_BATCH: LaneAdmission = LaneAdmission {
    envelope: "ordinary Qwen --requests-jsonl does not support every legacy option",
    supported: &[
        O::PromptLookup,
        O::PrefillChunk,
        O::PrefixCacheMaxMib,
        O::CachePrefixTokens,
        O::CachePrefixAutoMinTokens,
        O::RequestStats,
        O::ModelPrefetch,
        O::RequestsJsonl,
        O::BatchSize,
        O::Concurrency,
        O::ExecutionMode,
        O::OnRequestErrorContinue,
        O::TraceRequest,
        O::MaxContextTokens,
    ],
};
