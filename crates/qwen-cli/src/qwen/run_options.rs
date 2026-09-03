//! Env/arg-driven run options: model prefetch, GPU greedy argmax, DeepSeek V4 selector policy.

use super::*;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) enum DeepSeekV4PrefetchMode {
    Off,
    Always,
    #[default]
    Auto,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub(crate) enum QwenModelPrefetchArg {
    #[default]
    Auto,
    Off,
}

impl QwenModelPrefetchArg {
    pub(crate) fn policy(self) -> PrefetchPolicy {
        match self {
            Self::Auto => LoadedModelConfig::default().prefetch_policy,
            Self::Off => PrefetchPolicy::Off,
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
        }
    }
}

pub(crate) fn prefetch_policy_label(policy: PrefetchPolicy) -> &'static str {
    match policy {
        PrefetchPolicy::Off => "off",
        PrefetchPolicy::Always => "always",
        PrefetchPolicy::ColdOnly { .. } => "cold_only",
    }
}

impl DeepSeekV4PrefetchMode {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Always => "always",
            Self::Auto => "auto",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct DeepSeekV4PrefetchOutcome {
    pub(crate) mode: DeepSeekV4PrefetchMode,
    pub(crate) wall_ms: f64,
}

pub(crate) fn parse_deepseek_v4_prefetch_mode(
    value: Option<&OsStr>,
) -> Result<DeepSeekV4PrefetchMode> {
    let Some(value) = value else {
        return Ok(DeepSeekV4PrefetchMode::Auto);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{DEEPSEEK_V4_PREFETCH_ENV} is not valid UTF-8"))?;
    match value {
        "off" => Ok(DeepSeekV4PrefetchMode::Off),
        "always" => Ok(DeepSeekV4PrefetchMode::Always),
        "auto" => Ok(DeepSeekV4PrefetchMode::Auto),
        _ => bail!("{DEEPSEEK_V4_PREFETCH_ENV} must be one of auto|off|always, got {value:?}"),
    }
}

pub(crate) fn configured_deepseek_v4_prefetch_mode() -> Result<DeepSeekV4PrefetchMode> {
    parse_deepseek_v4_prefetch_mode(std::env::var_os(DEEPSEEK_V4_PREFETCH_ENV).as_deref())
}

#[cfg(feature = "dsv4-diagnostics")]
pub(crate) fn parse_deepseek_v4_temporal_window(value: Option<&OsStr>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(0);
    };
    let value = value
        .to_str()
        .with_context(|| format!("{DEEPSEEK_V4_TEMPORAL_WINDOW_ENV} is not valid UTF-8"))?;
    let window = value.parse::<usize>().with_context(|| {
        format!("{DEEPSEEK_V4_TEMPORAL_WINDOW_ENV}={value:?} is not an integer")
    })?;
    ensure!(
        window <= 65,
        "{DEEPSEEK_V4_TEMPORAL_WINDOW_ENV} must be in 0..=65, got {window}"
    );
    Ok(window)
}

#[cfg(feature = "dsv4-diagnostics")]
pub(crate) fn configured_deepseek_v4_temporal_window() -> Result<usize> {
    parse_deepseek_v4_temporal_window(std::env::var_os(DEEPSEEK_V4_TEMPORAL_WINDOW_ENV).as_deref())
}

pub(crate) fn deepseek_v4_prefetch_policy(mode: DeepSeekV4PrefetchMode) -> PrefetchPolicy {
    match mode {
        DeepSeekV4PrefetchMode::Off => PrefetchPolicy::Off,
        DeepSeekV4PrefetchMode::Always => PrefetchPolicy::Always,
        DeepSeekV4PrefetchMode::Auto => {
            PrefetchPolicy::cold_only(DEEPSEEK_V4_PREFETCH_AUTO_THRESHOLD)
                .expect("DeepSeek V4 auto-prefetch threshold is a valid fraction")
        }
    }
}

pub(crate) fn apply_deepseek_v4_prefetch(
    gguf: &GgufFile,
    mode: DeepSeekV4PrefetchMode,
) -> Result<DeepSeekV4PrefetchOutcome> {
    let process_before = PidSnapshot::now().ok();
    let defaults = LoadedModelConfig::default();
    let config = LoadedModelConfig {
        prefetch_policy: deepseek_v4_prefetch_policy(mode),
        prefetch_residency_probe: PrefetchResidencyProbe::Sampled,
        ..defaults
    };
    let report = prefetch_opened_gguf(gguf, &config);
    let process_delta = process_before
        .zip(PidSnapshot::now().ok())
        .map(|(before, after)| PidDelta::between(before, after));
    let bytes_returned = report.bytes_returned_total();
    if mode == DeepSeekV4PrefetchMode::Always {
        ensure!(
            report.shards_skipped() == 0,
            "explicit DeepSeek V4 prefetch skipped {} of {} shards",
            report.shards_skipped(),
            gguf.shard_count(),
        );
        ensure!(
            bytes_returned == gguf.total_mapped_len() as u64,
            "DeepSeek V4 prefetch returned {bytes_returned} bytes for {} mapped bytes",
            gguf.total_mapped_len(),
        );
    }
    let seconds = report.total_wall.as_secs_f64();
    let effective_bytes_per_sec = if seconds > 0.0 {
        bytes_returned as f64 / seconds
    } else {
        0.0
    };
    eprintln!(
        concat!(
            "deepseek_v4 prefetch: mode={} shards_prefetched={} shards_skipped={} workers={} chunk_bytes={} ",
            "bytes_returned={} physical_read_bytes={} wall_ms={:.1} effective_gbps={:.2}"
        ),
        mode.as_str(),
        report.shards_prefetched(),
        report.shards_skipped(),
        DEFAULT_WORKERS,
        DEFAULT_CHUNK_BYTES,
        bytes_returned,
        process_delta
            .map(|delta| delta.diskio_bytesread.to_string())
            .unwrap_or_else(|| "unavailable".to_string()),
        report.total_wall.as_secs_f64() * 1e3,
        effective_bytes_per_sec / 1e9,
    );
    Ok(DeepSeekV4PrefetchOutcome {
        mode,
        wall_ms: report.total_wall.as_secs_f64() * 1e3,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GreedyGpuArgmaxMode {
    DefaultOff,
    ForceEnabled,
    ExplicitRollback,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GreedyGpuDecision {
    pub(crate) enabled: bool,
    pub(crate) reason: &'static str,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, clap::ValueEnum)]
pub(crate) enum DeepSeekV4MultigroupSelectorArg {
    #[default]
    Auto,
    Off,
    QualifiedExperimental,
}

impl DeepSeekV4MultigroupSelectorArg {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Off => "off",
            Self::QualifiedExperimental => "qualified_experimental",
        }
    }
}

pub(crate) fn parse_greedy_gpu_argmax_mode(value: Option<&OsStr>) -> GreedyGpuArgmaxMode {
    match value {
        None => GreedyGpuArgmaxMode::DefaultOff,
        Some(value)
            if value
                .to_str()
                .is_some_and(qwen_llm::env_flag::env_value_truthy) =>
        {
            GreedyGpuArgmaxMode::ForceEnabled
        }
        Some(value)
            if value
                .to_str()
                .is_some_and(qwen_llm::env_flag::env_value_falsy) =>
        {
            GreedyGpuArgmaxMode::ExplicitRollback
        }
        Some(_) => GreedyGpuArgmaxMode::ExplicitRollback,
    }
}

pub(crate) fn configured_greedy_gpu_argmax_mode() -> GreedyGpuArgmaxMode {
    static MODE: std::sync::OnceLock<GreedyGpuArgmaxMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        parse_greedy_gpu_argmax_mode(std::env::var_os(GREEDY_GPU_ARGMAX_ENV).as_deref())
    })
}

pub(crate) fn resolve_greedy_gpu_decision(
    mode: GreedyGpuArgmaxMode,
    sampling: SamplingConfig,
    prompt_lookup: bool,
) -> GreedyGpuDecision {
    if mode == GreedyGpuArgmaxMode::ExplicitRollback {
        return GreedyGpuDecision {
            enabled: false,
            reason: "disabled_by_explicit_rollback",
        };
    }
    if sampling.temperature > 0.0 || prompt_lookup {
        return GreedyGpuDecision {
            enabled: false,
            reason: "ineligible_request",
        };
    }
    match mode {
        GreedyGpuArgmaxMode::ExplicitRollback => unreachable!("handled above"),
        GreedyGpuArgmaxMode::ForceEnabled => GreedyGpuDecision {
            enabled: true,
            reason: "force_enabled",
        },
        GreedyGpuArgmaxMode::DefaultOff => GreedyGpuDecision {
            enabled: false,
            reason: "default_off",
        },
    }
}
