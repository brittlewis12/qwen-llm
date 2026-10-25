//! Production option and chunking decisions for the non-Qwen families, shared
//! by `qwen run`, `qwen serve` and `qwen-bench` so every frontend executes and
//! measures the same ordinary path. Cold-only V4.1 execution selection is not
//! read by serving. Compiled into both binaries: explicit imports only, no CLI types.

use anyhow::{Context, Result, bail, ensure};
use qwen_llm::deepseek_v4_metal::{
    DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS, DEEPSEEK_V4_PREFILL_MAX_TOKENS,
};
use qwen_llm::muse_glimmer_runtime::MuseGlimmerRuntimeOptions;
use qwen_llm::qwen4exp_runtime::Qwen4ExpDecodeOptions;

/// Global engine ceiling, separate from admission-selected runtime geometry.
pub(crate) const V41_PREFILL_ROWS: usize = qwen_llm::deepseek_v41::forward::MAX_PREFILL_CHUNK_ROWS;
pub(crate) const V41_PREFILL_MODE: &str = "layer_major_batched+final_logits_copy";

pub(crate) fn v41_prefill_geometry(
    report: qwen_llm::deepseek_v41::forward::PrefillGeometry,
) -> serde_json::Value {
    serde_json::json!({
        "engine_rows_max": V41_PREFILL_ROWS, "selected_rows": report.selected_rows,
        "actual_chunks": report.chunk_count(),
        "chunk_ranges": report.chunk_ranges().map(|r| [r.start, r.end]).collect::<Vec<_>>(),
        "start": report.start, "logical_tokens": report.tokens,
        "max_chunk_rows": report.max_chunk_rows(),
    })
}

pub(crate) fn v41_execution_geometry(
    report: &qwen_llm::deepseek_v41::forward::PrefillReport,
) -> serde_json::Value {
    let mut value = v41_prefill_geometry(report.geometry);
    if report.execution == qwen_llm::deepseek_v41::forward::PrefillExecution::ShrinkingCed {
        let fields = value.as_object_mut().unwrap();
        let count = fields.remove("actual_chunks").unwrap();
        let ranges = fields.remove("chunk_ranges").unwrap();
        fields.insert("input_encoder_chunks".into(), count);
        fields.insert("input_encoder_chunk_ranges".into(), ranges);
        fields.insert(
            "scope".into(),
            "logical_input_encoder_not_decoder_work".into(),
        );
        fields.insert("schema_version".into(), 2.into());
    }
    value
}

pub(crate) const V41_PREFILL_EXECUTION_ENV: &str = "QWEN_V41_PREFILL_EXECUTION";

/// Cold run/bench only. Serving deliberately does not read this control.
pub(crate) fn v41_prefill_execution_from_env()
-> Result<qwen_llm::deepseek_v41::forward::PrefillExecution> {
    parse_v41_prefill_execution(std::env::var_os(V41_PREFILL_EXECUTION_ENV).as_deref())
}

pub(crate) fn parse_v41_prefill_execution(
    value: Option<&std::ffi::OsStr>,
) -> Result<qwen_llm::deepseek_v41::forward::PrefillExecution> {
    use qwen_llm::deepseek_v41::forward::PrefillExecution;
    match value {
        None => Ok(PrefillExecution::FullForward),
        Some(value) if value == "full_forward" => Ok(PrefillExecution::FullForward),
        Some(value) if value == "shrinking_ced" => Ok(PrefillExecution::ShrinkingCed),
        _ => bail!(
            "{V41_PREFILL_EXECUTION_ENV} must be full_forward or shrinking_ced, got {value:?}"
        ),
    }
}

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct V41PrefillDiagnostics {
    pub(crate) requested_execution: qwen_llm::deepseek_v41::forward::PrefillExecution,
    #[serde(flatten)]
    pub(crate) report: qwen_llm::deepseek_v41::forward::PrefillReport,
}

impl V41PrefillDiagnostics {
    pub(crate) fn new(
        requested_execution: qwen_llm::deepseek_v41::forward::PrefillExecution,
        report: qwen_llm::deepseek_v41::forward::PrefillReport,
    ) -> Option<Self> {
        use qwen_llm::deepseek_v41::forward::PrefillExecution::ShrinkingCed;
        (requested_execution == ShrinkingCed || report.execution == ShrinkingCed).then_some(Self {
            requested_execution,
            report,
        })
    }
}

pub(crate) fn v41_prefill_mode(
    execution: qwen_llm::deepseek_v41::forward::PrefillExecution,
) -> &'static str {
    match execution {
        qwen_llm::deepseek_v41::forward::PrefillExecution::FullForward => V41_PREFILL_MODE,
        qwen_llm::deepseek_v41::forward::PrefillExecution::ShrinkingCed => {
            "shrinking_ced+final_logits_copy"
        }
    }
}

/// Enables HC up-mix for eligible Flash-Next singleton Q8 up projections.
/// Defaults on; accepts `0` (off) or `1` (on); any other value, including
/// non-UTF-8, errors with
/// `QWEN4EXP_HC_UP_MIX must be 0 or 1`. Read per `qwen run` request, per
/// `qwen serve` backend construction, and per `qwen-bench` setup.
pub(crate) const QWEN4EXP_HC_UP_MIX_ENV: &str = "QWEN4EXP_HC_UP_MIX";
/// Enables guarded top-k for Flash-Next N512/K10 singleton selection.
/// Defaults on; accepts `0` (off) or `1` (on); any other value, including
/// non-UTF-8, errors with
/// `QWEN4EXP_GUARDED_TOPK must be 0 or 1`. Read per `qwen run` request, per
/// `qwen serve` backend construction, and per `qwen-bench` setup.
pub(crate) const QWEN4EXP_GUARDED_TOPK_ENV: &str = "QWEN4EXP_GUARDED_TOPK";

/// Enables Muse split decode for `qwen run`. Defaults on; accepts `0` (off)
/// or `1` (on); other UTF-8 values error with
/// `QWEN_MUSE_SPLIT_DECODE must be 0 or 1, got {value:?}`, and non-UTF-8
/// errors while reading with `read QWEN_MUSE_SPLIT_DECODE`. Read per request
/// and per `qwen-bench` setup. Serve reads `QWEN_SERVE_MUSE_SPLIT_DECODE`
/// during backend construction.
pub(crate) const MUSE_SPLIT_DECODE_ENV: &str = "QWEN_MUSE_SPLIT_DECODE";
/// Enables Muse matrix prefill for eligible `qwen run` prompts. Defaults on;
/// accepts `0` (off) or `1` (on); other UTF-8 values error with
/// `QWEN_MUSE_MATRIX_PREFILL must be 0 or 1, got {value:?}`, and non-UTF-8
/// errors while reading with `read QWEN_MUSE_MATRIX_PREFILL`. Read per request
/// and per `qwen-bench` setup. Serve reads `QWEN_SERVE_MUSE_MATRIX_PREFILL`
/// during backend construction.
pub(crate) const MUSE_MATRIX_PREFILL_ENV: &str = "QWEN_MUSE_MATRIX_PREFILL";

/// Sets the DeepSeek V4 prefill chunk size. Defaults to 4096 tokens; accepts
/// an integer in `1..=4096`. A non-integer errors with
/// `QWEN_DSV4_PREFILL_CHUNK_TOKENS={value:?} is not an integer`; an
/// out-of-range integer errors with
/// `QWEN_DSV4_PREFILL_CHUNK_TOKENS must be in 1..=4096, got {chunk_tokens}`;
/// this applies to successfully parsed `usize` values. Negative and larger
/// than-`usize::MAX` values error as not integers. A non-UTF-8 value is
/// treated as unset. Read during single-turn `qwen run` request setup, once
/// before a `qwen run` JSONL request stream, during `qwen serve` backend
/// preparation, and per `qwen-bench` setup.
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
    read_math_flag_result(name, std::env::var(name))
}

fn read_math_flag_result(
    name: &str,
    value: std::result::Result<String, std::env::VarError>,
) -> Result<bool> {
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
    deepseek_v4_prefill_chunk_tokens_from_read_result(std::env::var(DEEPSEEK_V4_PREFILL_CHUNK_ENV))
}

fn deepseek_v4_prefill_chunk_tokens_from_read_result(
    value: std::result::Result<String, std::env::VarError>,
) -> Result<usize> {
    let value = value.ok();
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn v41_cold_prefill_parser_defaults_and_valid_values() {
        use qwen_llm::deepseek_v41::forward::PrefillExecution;
        for (value, expected) in [
            (None, PrefillExecution::FullForward),
            (Some("full_forward"), PrefillExecution::FullForward),
            (Some("shrinking_ced"), PrefillExecution::ShrinkingCed),
        ] {
            assert_eq!(
                parse_v41_prefill_execution(value.map(OsStr::new)).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn v41_cold_prefill_parser_rejects_invalid_and_non_utf8_values() {
        for value in [
            "",
            "1",
            "0",
            "true",
            "ced",
            "ShrinkingCed",
            " shrinking_ced",
            "full_forward ",
        ] {
            let error = parse_v41_prefill_execution(Some(OsStr::new(value))).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("QWEN_V41_PREFILL_EXECUTION must be full_forward or shrinking_ced")
            );
        }
        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let value = std::ffi::OsString::from_vec(vec![0xff]);
            assert!(parse_v41_prefill_execution(Some(&value)).is_err());
        }
    }

    #[test]
    fn v41_prefill_diagnostics_preserve_actual_report_and_default_omission() {
        use qwen_llm::deepseek_v41::forward::{
            CedCounters, CedMemoryEstimate, PrefillExecution, PrefillReport,
        };
        let full = PrefillReport {
            geometry: qwen_llm::deepseek_v41::forward::PrefillGeometry { start: 0, tokens: 3, selected_rows: 4096 },
            execution: PrefillExecution::FullForward,
            fallback: None,
            counters: None,
            memory: None,
            observed_incremental_metal_bytes: None,
        };
        assert!(V41PrefillDiagnostics::new(PrefillExecution::FullForward, full.clone()).is_none());
        let fallback = PrefillReport {
            fallback: Some("unsupported structure".into()),
            ..full
        };
        let json = serde_json::to_value(
            V41PrefillDiagnostics::new(PrefillExecution::ShrinkingCed, fallback).unwrap(),
        )
        .unwrap();
        assert_eq!(json["requested_execution"], "shrinking_ced");
        assert_eq!(json["execution"], "full_forward");
        assert_eq!(json["fallback"], "unsupported structure");
        assert!(json["counters"].is_null());
        let ced = PrefillReport {
            geometry: qwen_llm::deepseek_v41::forward::PrefillGeometry { start: 0, tokens: 3, selected_rows: 4096 },
            execution: PrefillExecution::ShrinkingCed,
            fallback: None,
            counters: Some(CedCounters {
                completed_commands: 7,
                ..Default::default()
            }),
            memory: Some(CedMemoryEstimate {
                priced_retained_bytes: 123,
                ..Default::default()
            }),
            observed_incremental_metal_bytes: Some(456),
        };
        let json = serde_json::to_value(
            V41PrefillDiagnostics::new(PrefillExecution::ShrinkingCed, ced).unwrap(),
        )
        .unwrap();
        assert_eq!(json["execution"], "shrinking_ced");
        assert!(json["fallback"].is_null());
        assert_eq!(json["counters"]["completed_commands"], 7);
        assert_eq!(json["memory"]["priced_retained_bytes"], 123);
        assert_eq!(json["observed_incremental_metal_bytes"], 456);
        assert_eq!(
            v41_prefill_mode(PrefillExecution::FullForward),
            V41_PREFILL_MODE
        );
        assert_eq!(
            v41_prefill_mode(PrefillExecution::ShrinkingCed),
            "shrinking_ced+final_logits_copy"
        );
    }

    #[test]
    fn qwen4exp_decode_flags_follow_the_documented_contract() {
        for name in [QWEN4EXP_GUARDED_TOPK_ENV, QWEN4EXP_HC_UP_MIX_ENV] {
            for (value, expected) in [
                (None, true),
                (Some(OsStr::new("0")), false),
                (Some(OsStr::new("1")), true),
            ] {
                assert_eq!(parse_qwen4exp_decode_flag(value, name).unwrap(), expected);
            }
            for value in ["", "true", "false", " 1", "1 ", "2"] {
                let error = parse_qwen4exp_decode_flag(Some(OsStr::new(value)), name).unwrap_err();
                assert_eq!(error.to_string(), format!("{name} must be 0 or 1"));
            }
            #[cfg(unix)]
            {
                use std::os::unix::ffi::OsStringExt;
                let invalid = std::ffi::OsString::from_vec(vec![0xff]);
                let error = parse_qwen4exp_decode_flag(Some(&invalid), name).unwrap_err();
                assert_eq!(error.to_string(), format!("{name} must be 0 or 1"));
            }
        }
    }

    #[test]
    fn muse_math_flags_follow_the_documented_contract() {
        for name in [MUSE_SPLIT_DECODE_ENV, MUSE_MATRIX_PREFILL_ENV] {
            for (value, expected) in [(None, true), (Some("0"), false), (Some("1"), true)] {
                assert_eq!(parse_math_flag(name, value).unwrap(), expected);
            }
            for value in ["", "true", "false", " 1", "1 ", "2"] {
                let error = parse_math_flag(name, Some(value)).unwrap_err();
                assert_eq!(
                    error.to_string(),
                    format!("{name} must be 0 or 1, got {value:?}")
                );
            }
        }
    }

    #[cfg(unix)]
    #[test]
    fn muse_environment_reader_reports_non_utf8_values() {
        use std::os::unix::ffi::OsStringExt;

        for name in [MUSE_SPLIT_DECODE_ENV, MUSE_MATRIX_PREFILL_ENV] {
            let invalid = std::ffi::OsString::from_vec(vec![0xff]);
            let read_result = Err(std::env::VarError::NotUnicode(invalid));
            let error = read_math_flag_result(name, read_result).unwrap_err();
            assert_eq!(error.to_string(), format!("read {name}"));
        }
    }

    #[test]
    fn deepseek_v4_chunk_parser_and_non_utf8_wrapper_mapping_are_pinned() {
        for (value, expected) in [
            (None, DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS),
            (Some("1"), 1),
            (Some("2048"), 2_048),
            (Some("4096"), DEEPSEEK_V4_PREFILL_MAX_TOKENS),
        ] {
            assert_eq!(
                parse_deepseek_v4_prefill_chunk_tokens(value).unwrap(),
                expected
            );
        }
        for (value, expected_error) in [
            (
                "nope",
                "QWEN_DSV4_PREFILL_CHUNK_TOKENS=\"nope\" is not an integer",
            ),
            (
                "0",
                "QWEN_DSV4_PREFILL_CHUNK_TOKENS must be in 1..=4096, got 0",
            ),
            (
                "4097",
                "QWEN_DSV4_PREFILL_CHUNK_TOKENS must be in 1..=4096, got 4097",
            ),
            (
                "-1",
                "QWEN_DSV4_PREFILL_CHUNK_TOKENS=\"-1\" is not an integer",
            ),
        ] {
            let error = parse_deepseek_v4_prefill_chunk_tokens(Some(value)).unwrap_err();
            assert_eq!(error.to_string(), expected_error);
        }

        let overflow = format!("{}0", usize::MAX);
        let error = parse_deepseek_v4_prefill_chunk_tokens(Some(&overflow)).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("QWEN_DSV4_PREFILL_CHUNK_TOKENS={overflow:?} is not an integer")
        );

        #[cfg(unix)]
        {
            use std::os::unix::ffi::OsStringExt;
            let invalid = std::ffi::OsString::from_vec(vec![0xff]);
            assert_eq!(
                deepseek_v4_prefill_chunk_tokens_from_read_result(Err(
                    std::env::VarError::NotUnicode(invalid),
                ))
                .unwrap(),
                DEEPSEEK_V4_PREFILL_DEFAULT_TOKENS
            );
        }
    }
}
