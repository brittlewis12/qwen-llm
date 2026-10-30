//! Serve-only retained-weight coalescing; no execution or kernel selection.
use anyhow::{Context, Result, ensure};
use qwen_llm::deepseek_v41::gguf::{ResidentWeights, resident_storage_plan_with_target};
use qwen_llm::gguf::GgufFile;
use qwen_llm::metal::{MetalContext, host_page_size_bytes};
use std::sync::Arc;

/// Serve-only soft retained-weight coalescing target in MiB. Unset or 0 keeps
/// ordinary loading; 4096 requests 4 GiB windows. Oversized tensors remain
/// eligible up to the device limit. Read once at startup; no CED selection.
const WEIGHT_WINDOW_MIB_ENV: &str = "QWEN_SERVE_V41_WEIGHT_WINDOW_MIB";
const MIB: usize = 1 << 20;

/// Read once into `Prepared`; unset and zero preserve the ordinary loader.
pub(super) fn target_from_env() -> Result<Option<usize>> {
    match std::env::var(WEIGHT_WINDOW_MIB_ENV) {
        Ok(value) => parse_target(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_target(None),
        Err(error) => Err(error).with_context(|| format!("read {WEIGHT_WINDOW_MIB_ENV}")),
    }
}

fn parse_target(value: Option<&str>) -> Result<Option<usize>> {
    let Some(value) = value else {
        return Ok(None);
    };
    ensure!(
        !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()),
        "{WEIGHT_WINDOW_MIB_ENV} must be a non-negative integer MiB value, got {value:?}"
    );
    let mib = value
        .parse::<usize>()
        .with_context(|| format!("{WEIGHT_WINDOW_MIB_ENV} MiB value overflows usize: {value:?}"))?;
    let bytes = mib.checked_mul(MIB).with_context(|| {
        format!("{WEIGHT_WINDOW_MIB_ENV} MiB-to-byte conversion overflows usize: {value:?}")
    })?;
    Ok((bytes != 0).then_some(bytes))
}

pub(super) fn load(
    ctx: &MetalContext,
    gguf: &Arc<GgufFile>,
    target_bytes: Option<usize>,
) -> Result<ResidentWeights> {
    let Some(target_bytes) = target_bytes else {
        return crate::deepseek_v41::load(ctx, Arc::clone(gguf));
    };
    crate::shutdown::checkpoint()?;
    let storage = resident_storage_plan_with_target(
        gguf,
        host_page_size_bytes()?,
        ctx.max_buffer_length(),
        Some(target_bytes),
    )?;
    // The loader revalidates this geometry and performs its existing admission.
    let weights = ResidentWeights::load_with_storage_plan(ctx, Arc::clone(gguf), &storage)?;
    let effective_target_bytes =
        effective_target_bytes(storage.page_size, storage.max_buffer_length, target_bytes);
    let largest_window_bytes = storage
        .windows
        .iter()
        .map(|planned| planned.window.length)
        .max()
        .unwrap_or(0);
    tracing::info!(
        requested_target_bytes = target_bytes,
        effective_target_bytes,
        windows = storage.windows.len(),
        largest_window_bytes,
        exposed_mapping_bytes = storage.metal_view_bytes,
        private_shadow_bytes = storage.private_shadow_bytes,
        "serve: V4.1 retained-weight storage plan (mapping exposure, not physical residency; target is soft, not a singleton limit)"
    );
    Ok(weights)
}

// Match the planner's page rounding and device clamp, after geometry validation.
fn effective_target_bytes(page_size: usize, max_buffer_length: usize, target: usize) -> usize {
    (target / page_size * page_size)
        .max(page_size)
        .min(max_buffer_length / page_size * page_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unset_and_zero_disable() {
        for value in [None, Some("0"), Some("000")] {
            assert_eq!(parse_target(value).unwrap(), None);
        }
    }

    #[test]
    fn positive_mib_converts_to_bytes() {
        assert_eq!(parse_target(Some("1")).unwrap(), Some(MIB));
        assert_eq!(parse_target(Some("4096")).unwrap(), Some(4096 * MIB));
    }

    #[test]
    fn invalid_empty_negative_and_non_decimal_values_fail() {
        for value in [
            "", " ", "-1", "-0", "+1", "1.5", "1MiB", "0x10", "1_024", " 1", "1 ", "1\n",
        ] {
            let error = parse_target(Some(value)).unwrap_err().to_string();
            assert!(error.contains(WEIGHT_WINDOW_MIB_ENV), "{error}");
            assert!(error.contains("non-negative integer MiB"), "{error}");
        }
    }

    #[test]
    fn integer_overflow_fails() {
        let value = format!("{}0", usize::MAX);
        let error = parse_target(Some(&value)).unwrap_err().to_string();
        assert!(error.contains(WEIGHT_WINDOW_MIB_ENV), "{error}");
        assert!(error.contains("MiB value overflows"), "{error}");
    }

    #[test]
    fn byte_conversion_overflow_fails() {
        let value = (usize::MAX / MIB + 1).to_string();
        let error = parse_target(Some(&value)).unwrap_err().to_string();
        assert!(error.contains(WEIGHT_WINDOW_MIB_ENV), "{error}");
        assert!(
            error.contains("MiB-to-byte conversion overflows"),
            "{error}"
        );
    }

    #[test]
    fn largest_representable_mib_succeeds() {
        let mib = usize::MAX / MIB;
        assert_eq!(
            parse_target(Some(&mib.to_string())).unwrap(),
            Some(mib * MIB)
        );
    }

    #[test]
    fn effective_target_rounds_to_pages_and_clamps_to_device() {
        let page = 16_384;
        let max_buffer = 4 * MIB + page - 1;
        assert_eq!(effective_target_bytes(page, max_buffer, 1), page);
        assert_eq!(effective_target_bytes(page, max_buffer, MIB + 1), MIB);
        assert_eq!(effective_target_bytes(page, max_buffer, 8 * MIB), 4 * MIB);
    }

    #[test]
    fn sparse_cpu_plan_keeps_oversized_singletons_and_payload() {
        let gguf = crate::deepseek_v41_fixture::gguf_with_banks(
            3,
            "deepseek41.context_length",
            128,
            [qwen_llm::tensor::GgmlType::F32; 3],
        );
        let page = 16_384;
        let max_buffer = 16 * MIB;
        let ordinary = resident_storage_plan_with_target(&gguf, page, max_buffer, None).unwrap();
        let target = parse_target(Some("1")).unwrap().unwrap();
        let selected =
            resident_storage_plan_with_target(&gguf, page, max_buffer, Some(target)).unwrap();
        assert_eq!(selected.soft_coalescing_target, Some(target));
        assert_eq!(selected.bytes, ordinary.bytes);
        assert_eq!(effective_target_bytes(page, max_buffer, target), target);
        let small = resident_storage_plan_with_target(&gguf, page, max_buffer, Some(page)).unwrap();
        assert!(
            small
                .windows
                .iter()
                .any(|planned| planned.window.length > page)
        );
        assert_eq!(small.bytes, ordinary.bytes);
        assert!(small.windows.len() > ordinary.windows.len());
        assert_eq!(
            small.metal_view_bytes,
            small
                .windows
                .iter()
                .map(|planned| planned.window.length as u64)
                .sum::<u64>()
        );
        assert_eq!(
            small.private_shadow_bytes,
            small
                .windows
                .iter()
                .map(|planned| planned.mapping_kind.private_bytes())
                .sum::<u64>()
        );
    }
}
