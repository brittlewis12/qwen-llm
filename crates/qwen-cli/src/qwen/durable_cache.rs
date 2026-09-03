//! Durable prefix-cache store selection and capture policy.

use super::*;

pub(crate) fn prefix_cache_max_bytes(args: &Args) -> Result<u64> {
    mib_to_bytes(args.prefix_cache_max_mib, "prefix cache byte budget")
}

pub(crate) fn durable_checkpoint_store(
    args: &Args,
    staged_integrity: Option<StagedIntegrityMode>,
) -> Result<Option<DurableCheckpointStore>> {
    let Some(root) = args.durable_prefix_cache.as_ref() else {
        return Ok(None);
    };
    let budget = mib_to_bytes(
        args.durable_prefix_cache_max_mib,
        "durable prefix cache byte budget",
    )?;
    Ok(Some(match staged_integrity {
        Some(mode) => DurableCheckpointStore::with_staged_integrity(root, budget, mode),
        None => DurableCheckpointStore::new(root, budget),
    }))
}

pub(crate) fn configured_checkpoint_staged_integrity() -> Result<Option<StagedIntegrityMode>> {
    match std::env::var(CHECKPOINT_STAGED_INTEGRITY_ENV) {
        Ok(value) => parse_checkpoint_staged_integrity(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_checkpoint_staged_integrity(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            bail!("invalid {CHECKPOINT_STAGED_INTEGRITY_ENV}; expected decode or deferred-restore")
        }
    }
}

pub(crate) fn parse_checkpoint_staged_integrity(
    value: Option<&str>,
) -> Result<Option<StagedIntegrityMode>> {
    match value {
        None => Ok(None),
        Some(value) => StagedIntegrityMode::parse(value).map(Some).ok_or_else(|| {
            anyhow!(
                "invalid {CHECKPOINT_STAGED_INTEGRITY_ENV}; expected decode or deferred-restore"
            )
        }),
    }
}

pub(crate) fn durable_prefix_cache_max_entry_bytes(args: &Args) -> Result<u64> {
    mib_to_bytes(
        args.durable_prefix_cache_max_entry_mib,
        "durable prefix cache per-entry budget",
    )
}

pub(crate) fn mib_to_bytes(mib: u64, label: &str) -> Result<u64> {
    mib.checked_mul(1024 * 1024)
        .with_context(|| format!("{label} overflow"))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DurableCapturePolicy {
    Disabled,
    ExplicitPrompt(usize),
    AutomaticPrompt(usize),
    AutomaticCompleted,
}

impl DurableCapturePolicy {
    pub(crate) fn prompt_prefix_len(self) -> Option<usize> {
        match self {
            Self::ExplicitPrompt(len) | Self::AutomaticPrompt(len) => Some(len),
            Self::Disabled | Self::AutomaticCompleted => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct CompletedCheckpointBoundary {
    pub(crate) consumed_generated_tokens: usize,
    pub(crate) consumed_prefix_len: usize,
    pub(crate) pending_token: i32,
}

impl CompletedCheckpointBoundary {
    pub(crate) fn consumed_tokens(self, prompt_ids: &[i32], generated: &[i32]) -> Vec<i32> {
        let mut consumed = Vec::with_capacity(self.consumed_prefix_len);
        consumed.extend_from_slice(prompt_ids);
        consumed.extend_from_slice(&generated[..self.consumed_generated_tokens]);
        consumed
    }
}

pub(crate) fn derive_completed_checkpoint_boundary(
    prompt_len: usize,
    generated: &[i32],
    transitions: usize,
    sequence_position: usize,
) -> Result<CompletedCheckpointBoundary> {
    ensure!(!generated.is_empty(), "completed generation has no tokens");
    ensure!(
        transitions.checked_add(1) == Some(generated.len()),
        "completed generation transitions {} do not match token count {}",
        transitions,
        generated.len()
    );
    let consumed_prefix_len = prompt_len
        .checked_add(transitions)
        .context("completed checkpoint prefix length overflow")?;
    ensure!(
        sequence_position == consumed_prefix_len,
        "completed sequence position {} does not match consumed prefix length {}",
        sequence_position,
        consumed_prefix_len
    );
    Ok(CompletedCheckpointBoundary {
        consumed_generated_tokens: transitions,
        consumed_prefix_len,
        pending_token: generated[transitions],
    })
}

pub(crate) fn selected_single_turn_durable_policy(
    args: &Args,
    prompt_len: usize,
    completed_checkpoint_eligible: bool,
) -> DurableCapturePolicy {
    if let Some(configured) = args.cache_prefix_tokens {
        return if configured > 0 && prompt_len > 0 {
            DurableCapturePolicy::ExplicitPrompt(configured.min(prompt_len))
        } else {
            DurableCapturePolicy::Disabled
        };
    }
    if args.durable_prefix_cache_min_tokens == 0
        || prompt_len < args.durable_prefix_cache_min_tokens
    {
        return DurableCapturePolicy::Disabled;
    }
    if completed_checkpoint_eligible {
        DurableCapturePolicy::AutomaticCompleted
    } else {
        DurableCapturePolicy::AutomaticPrompt(prompt_len)
    }
}

pub(crate) fn selected_single_turn_durable_lookup_len(args: &Args, prompt_len: usize) -> usize {
    args.cache_prefix_tokens
        .filter(|&configured| configured > 0)
        .map_or(prompt_len, |configured| configured.min(prompt_len))
}

pub(crate) fn identity_cache_outcome_label(outcome: IdentityCacheOutcome) -> &'static str {
    match outcome {
        IdentityCacheOutcome::Hit => "hit",
        IdentityCacheOutcome::ComputedAndStored => "computed_stored",
        IdentityCacheOutcome::ComputedAndRepaired => "computed_repaired",
        IdentityCacheOutcome::ComputedUncached => "computed_uncached",
        IdentityCacheOutcome::DeclaredAndStored => "declared_stored",
        IdentityCacheOutcome::DeclaredUncached => "declared_uncached",
    }
}

pub(crate) fn publish_outcome_label(outcome: PublishOutcome) -> &'static str {
    match outcome {
        PublishOutcome::Published => "published",
        PublishOutcome::ExistingValid => "existing_valid",
        PublishOutcome::RepairedCorrupt => "repaired_corrupt",
    }
}
