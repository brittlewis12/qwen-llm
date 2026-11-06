use super::*;
use crate::metal::Pipeline;
use std::sync::OnceLock;

/// M5 Flash-Next packed QSA selector for N32 bands and >=8192 block capacity. Default 2 caches a <=512-key threshold bucket in threadgroup memory, exits resolved radix passes early and uses parallel stable scans. 1 selects scans/early exit without bucket caching; 0 restores the existing radix4 selector. IDs, ties and error statuses remain exact; no added device buffers. Unsupported PSOs/devices use the incumbent.
const SELECTOR_ENV: &str = "QWEN_FN_QSA_SELECTOR_COMPACT";

fn selector_mode() -> usize {
    static MODE: OnceLock<usize> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var(SELECTOR_ENV).ok().as_deref() {
        Some("0") => 0,
        Some("1") => 1,
        _ => 2,
    })
}

/// M5 Flash-Next selected QSA logits at N32 and depth >=32768. Default 8 reuses original F32 query vectors across eight selected keys per SIMD; 16/32 tune the key loop; 1 aliases 8; 0 restores the incumbent. Preserves the dimension/SIMD reduction order, with no operand narrowing or new device buffers.
const LOGITS_ENV: &str = "QWEN_FN_QSA_LOGITS_REUSE";

fn logits_slots() -> usize {
    static SLOTS: OnceLock<usize> = OnceLock::new();
    *SLOTS.get_or_init(|| match std::env::var(LOGITS_ENV).ok().as_deref() {
        Some("0") => 0,
        Some("16") => 16,
        Some("32") => 32,
        _ => 8,
    })
}

crate::env_flag!(
    /// M5 Flash-Next selected QSA value at N32 and depth >=32768: explicit F32 vector accumulators with the existing eight-way split and slot/reduction order. Original F32 masses and F16 values; no added GPU buffers. Set 0 for the existing split8 kernel.
    default_on value_vector_enabled,
    "QWEN_FN_QSA_VALUE_VECTOR"
);

fn optional(
    ctx: &MetalContext,
    name: &str,
    threads: usize,
    dynamic: usize,
) -> Result<Option<Pipeline>, MetalError> {
    match ctx.pipeline(name) {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= threads
                && p.staticThreadgroupMemoryLength()
                    .checked_add(dynamic)
                    .is_some_and(|v| v <= ctx.device.maxThreadgroupMemoryLength()) =>
        {
            Ok(Some(p))
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => Ok(None),
        Err(e) => Err(e),
    }
}

pub(super) fn selector_pipeline(
    ctx: &MetalContext,
    capacity: usize,
    budget: usize,
    count: usize,
    threads: usize,
) -> Result<Option<Pipeline>, MetalError> {
    let mode = selector_mode();
    if mode == 0
        || !ctx.tensor_ops_status().is_available()
        || count != 32
        || capacity < 8192
        || budget != 512
    {
        return Ok(None);
    }
    optional(
        ctx,
        if mode == 1 {
            "kernel_qwen4exp_qsa_select_scan_f32"
        } else {
            "kernel_qwen4exp_qsa_select_compact_f32"
        },
        threads,
        threads * 8,
    )
}

pub(super) fn logits_pipeline(
    ctx: &MetalContext,
    g: QwenSparseAttentionMetalGeometry,
    start: usize,
    count: usize,
) -> Result<Option<(Pipeline, usize)>, MetalError> {
    let slots = logits_slots();
    if slots == 0
        || !ctx.tensor_ops_status().is_available()
        || !selected_value::geometry_qualified(g)
        || count != 32
        || start < 32768
    {
        return Ok(None);
    }
    let name = match slots {
        16 => "kernel_qwen4exp_qsa_logits_reuse4_s16_f32",
        32 => "kernel_qwen4exp_qsa_logits_reuse4_s32_f32",
        _ => "kernel_qwen4exp_qsa_logits_reuse4_f32",
    };
    Ok(optional(ctx, name, PACKED_ATTENTION_THREADS, 0)?.map(|p| (p, slots * 4)))
}

pub(super) fn value_pipeline(ctx: &MetalContext) -> Result<Option<Pipeline>, MetalError> {
    if !value_vector_enabled() || !ctx.tensor_ops_status().is_available() {
        return Ok(None);
    }
    optional(ctx, "kernel_qwen4exp_qsa_selected_value_vec8_f32", 256, 0)
}
