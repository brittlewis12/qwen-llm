use super::*;

pub(super) const MIN_TOKENS: usize = 1024;
const MIN_USES: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Plan {
    pub(super) enabled: bool,
    pub(super) common_prefix_tokens: usize,
    pub(super) selected_prefix_tokens: usize,
    pub(super) planned_uses: usize,
    pub(super) planned_avoided_prefix_evaluations: usize,
    pub(super) planned_avoided_prompt_tokens: usize,
    pub(super) reason: &'static str,
}

pub(super) struct Prepared {
    pub(super) plan: Plan,
    pub(super) checkpoint: PreparedCheckpoint,
}

pub(super) fn enabled(env: &str, default_enabled: bool) -> Result<bool> {
    let value = std::env::var_os(env)
        .map(|value| {
            value
                .into_string()
                .map_err(|_| anyhow!("{env} must be valid UTF-8"))
        })
        .transpose()?;
    parse_enabled(env, value.as_deref(), default_enabled)
}

fn parse_enabled(env: &str, value: Option<&str>, default_enabled: bool) -> Result<bool> {
    let Some(value) = value else {
        return Ok(default_enabled);
    };
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => bail!("{env} must be a boolean"),
    }
}

pub(super) fn plan(
    prompts: &[&[i32]],
    prefill_chunk: PrefillChunkArg,
    use_count: usize,
    enabled: bool,
) -> Plan {
    let common_prefix_tokens = prompts
        .split_first()
        .map(|(first, rest)| {
            rest.iter().fold(first.len(), |common, prompt| {
                first[..common]
                    .iter()
                    .zip(*prompt)
                    .take_while(|(left, right)| left == right)
                    .count()
            })
        })
        .unwrap_or(0);
    let rejected = |reason| Plan {
        enabled,
        common_prefix_tokens,
        selected_prefix_tokens: 0,
        planned_uses: 0,
        planned_avoided_prefix_evaluations: 0,
        planned_avoided_prompt_tokens: 0,
        reason,
    };
    if !enabled {
        return rejected("disabled");
    }
    if use_count < MIN_USES {
        return rejected("below_minimum_uses");
    }
    let PrefillChunkArg::Fixed(chunk) = prefill_chunk else {
        return rejected("auto_chunk_unsupported");
    };
    let selected_prefix_tokens = common_prefix_tokens / chunk * chunk;
    if selected_prefix_tokens < MIN_TOKENS {
        return rejected("alignment_below_minimum");
    }
    let avoided_prefix_evaluations = use_count.saturating_sub(1);
    let Some(avoided_prompt_tokens) =
        selected_prefix_tokens.checked_mul(avoided_prefix_evaluations)
    else {
        return rejected("saved_token_count_overflow");
    };
    Plan {
        enabled,
        common_prefix_tokens,
        selected_prefix_tokens,
        planned_uses: use_count,
        planned_avoided_prefix_evaluations: avoided_prefix_evaluations,
        planned_avoided_prompt_tokens: avoided_prompt_tokens,
        reason: "selected_chunk_aligned",
    }
}

pub(super) fn prepare(
    loaded: &LoadedModel,
    root_prompt: &[i32],
    prefill_chunk: PrefillChunkArg,
    plan: Plan,
    max_capacity: usize,
    snapshot_required_bytes: u64,
) -> Result<(Prepared, f64, f64)> {
    let PrefillChunkArg::Fixed(requested_chunk) = prefill_chunk else {
        unreachable!("file-root planner rejects automatic chunks")
    };
    let prefix_len = plan.selected_prefix_tokens;
    let chunk = requested_chunk.min(prefix_len);
    let mut scratch = allocate_legacy_prefill_scratch(loaded, chunk, prefix_len)
        .context("allocate file-root prefill scratch")?;
    let mut sequence = loaded
        .create_sequence(SequenceConfig::new(max_capacity))
        .context("allocate file-root source sequence")?;
    let (prefix_logits, prefill_ms) = prefill_owned(
        loaded,
        &mut sequence,
        &mut scratch,
        &root_prompt[..prefix_len],
        0,
    )
    .context("prefill file-scoped shared root")?;
    shutdown::checkpoint()?;
    let snapshot_t0 = Instant::now();
    let checkpoint = loaded
        .prepare_checkpoint_boundary(
            &sequence,
            root_prompt[..prefix_len].to_vec(),
            None,
            Some(prefix_logits),
            None,
            0,
        )
        .context("capture file-scoped shared root")?;
    let snapshot_ms = snapshot_t0.elapsed().as_secs_f64() * 1e3;
    ensure!(
        checkpoint.snapshot_bytes() == snapshot_required_bytes,
        "file-root snapshot bytes {} != estimate {snapshot_required_bytes}",
        checkpoint.snapshot_bytes(),
    );
    Ok((Prepared { plan, checkpoint }, prefill_ms, snapshot_ms))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planner_requires_multiple_uses_and_aligns_the_root() {
        let base = vec![7; 1_600];
        let prompts = [
            base.as_slice(),
            &base[..1_500],
            &base[..1_400],
            &base[..1_300],
        ];
        assert_eq!(
            plan(&prompts, PrefillChunkArg::Fixed(512), 2, true),
            Plan {
                enabled: true,
                common_prefix_tokens: 1_300,
                selected_prefix_tokens: 1_024,
                planned_uses: 2,
                planned_avoided_prefix_evaluations: 1,
                planned_avoided_prompt_tokens: 1_024,
                reason: "selected_chunk_aligned",
            }
        );
        assert_eq!(
            plan(&prompts, PrefillChunkArg::Fixed(512), 1, true).reason,
            "below_minimum_uses"
        );
    }

    #[test]
    fn planner_fails_closed_for_controls_and_short_roots() {
        let prompts = [
            vec![7; 1_500],
            vec![7; 1_400],
            vec![7; 1_300],
            vec![7; 1_200],
        ];
        let refs = prompts.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert_eq!(
            plan(&refs, PrefillChunkArg::Fixed(512), 2, false).reason,
            "disabled"
        );
        assert_eq!(
            plan(&refs, PrefillChunkArg::Auto, 2, true).reason,
            "auto_chunk_unsupported"
        );
        let short = [vec![7; 1_000], vec![7; 1_000]];
        let short_refs = short.iter().map(Vec::as_slice).collect::<Vec<_>>();
        assert_eq!(
            plan(&short_refs, PrefillChunkArg::Fixed(512), 2, true).reason,
            "alignment_below_minimum"
        );
        assert!(parse_enabled("ROOT_ENV", None, true).unwrap());
        assert!(!parse_enabled("ROOT_ENV", Some("off"), true).unwrap());
        assert!(parse_enabled("ROOT_ENV", Some("maybe"), true).is_err());
    }
}
