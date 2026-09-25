//! Production option and chunking decisions for the non-Qwen families, shared
//! by `qwen run`, `qwen serve` and `qwen-bench` so every frontend executes and
//! measures the same path. Compiled into both binaries: explicit imports only,
//! no CLI types.

use anyhow::{Context, Result, bail, ensure};
use qwen_llm::deepseek_v4_metal::{
    DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS,
};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerRuntimeOptions;
use qwen_llm::qwen4exp_runtime::Qwen4ExpDecodeOptions;

pub(crate) const QWEN4EXP_HC_UP_MIX_ENV: &str = "QWEN4EXP_HC_UP_MIX";
pub(crate) const QWEN4EXP_GUARDED_TOPK_ENV: &str = "QWEN4EXP_GUARDED_TOPK";

/// `qwen run`'s Muse math levers (serve reads its own `QWEN_SERVE_MUSE_*`).
pub(crate) const MUSE_SPLIT_DECODE_ENV: &str = "QWEN_MUSE_SPLIT_DECODE";
pub(crate) const MUSE_MATRIX_PREFILL_ENV: &str = "QWEN_MUSE_MATRIX_PREFILL";

pub(crate) const DEEPSEEK_V4_PREFILL_CHUNK_ENV: &str = "QWEN_DSV4_PREFILL_CHUNK_TOKENS";

/// `QWEN4EXP_GUARDED_TOPK` / `QWEN4EXP_HC_UP_MIX` rollbacks, shared by the
/// run and serve lanes.
pub(crate) fn qwen4exp_decode_options_from_env() -> Result<Qwen4ExpDecodeOptions> {
    Ok(Qwen4ExpDecodeOptions {
        guarded_topk: parse_qwen4exp_decode_flag(
            std::env::var_os(QWEN4EXP_GUARDED_TOPK_ENV).as_deref(),
            QWEN4EXP_GUARDED_TOPK_ENV,
        )?,
        hc_up_mix: parse_qwen4exp_decode_flag(
            std::env::var_os(QWEN4EXP_HC_UP_MIX_ENV).as_deref(),
            QWEN4EXP_HC_UP_MIX_ENV,
        )?,
    })
}

pub(crate) fn parse_qwen4exp_decode_flag(
    value: Option<&std::ffi::OsStr>,
    name: &str,
) -> Result<bool> {
    match value {
        None => Ok(true),
        Some(value) if value == "0" => Ok(false),
        Some(value) if value == "1" => Ok(true),
        _ => bail!("{name} must be 0 or 1"),
    }
}

/// The production Muse math `qwen run` loads (split decode and matrix
/// prefill, both default on).
pub(crate) fn muse_runtime_options_from_env() -> Result<MuseGlimmerRuntimeOptions> {
    Ok(MuseGlimmerRuntimeOptions {
        split_decode: read_math_flag(MUSE_SPLIT_DECODE_ENV)?,
        matrix_prefill: read_math_flag(MUSE_MATRIX_PREFILL_ENV)?,
    })
}

pub(crate) fn read_math_flag(name: &str) -> Result<bool> {
    let value = std::env::var(name);
    match value {
        Ok(value) => parse_math_flag(name, Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_math_flag(name, None),
        Err(error) => Err(error).with_context(|| format!("read {name}")),
    }
}

pub(crate) fn parse_math_flag(name: &str, value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("1") => Ok(true),
        Some("0") => Ok(false),
        Some(value) => bail!("{name} must be 0 or 1, got {value:?}"),
    }
}

pub(crate) fn parse_deepseek_v4_prefill_chunk_tokens(value: Option<&str>) -> Result<usize> {
    let Some(value) = value else {
        return Ok(DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS);
    };
    let chunk_tokens = value
        .parse::<usize>()
        .with_context(|| format!("QWEN_DSV4_PREFILL_CHUNK_TOKENS={value:?} is not an integer"))?;
    ensure!(
        (1..=DEEPSEEK_V4_PREFILL_MAX_TOKENS).contains(&chunk_tokens),
        "QWEN_DSV4_PREFILL_CHUNK_TOKENS must be in 1..={DEEPSEEK_V4_PREFILL_MAX_TOKENS}, got {chunk_tokens}"
    );
    Ok(chunk_tokens)
}

pub(crate) fn deepseek_v4_prefill_chunk_tokens() -> Result<usize> {
    let value = std::env::var(DEEPSEEK_V4_PREFILL_CHUNK_ENV).ok();
    parse_deepseek_v4_prefill_chunk_tokens(value.as_deref())
}

pub(crate) fn deepseek_v4_prefill_chunk_ranges(
    prompt_tokens: usize,
    chunk_tokens: usize,
) -> Vec<std::ops::Range<usize>> {
    let mut ranges = Vec::new();
    let mut start = 0usize;
    while start < prompt_tokens {
        let remaining = prompt_tokens - start;
        let len = if remaining >= chunk_tokens {
            chunk_tokens
        } else if chunk_tokens == DEEPSEEK_V4_PREFILL_MAX_TOKENS && remaining >= 2_048 {
            2_048
        } else {
            remaining
        };
        ranges.push(start..start + len);
        start += len;
    }
    ranges
}
