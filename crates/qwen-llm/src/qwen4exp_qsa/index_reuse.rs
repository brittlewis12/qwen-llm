use super::*;
use std::sync::OnceLock;

#[cfg(test)]
mod tests;

/// Query reuse for M5 Flash-Next selected QSA index scoring at >=8192 visible blocks and full 32-query bands. Value 8 (default) reuses each original F16 key across eight F32 query rows; 4 selects the four-query tile; 0 restores the incumbent. Same lane/head reduction order, no narrowing or added GPU buffers. Other geometries/devices and unavailable PSOs retain the incumbent.
const INDEX_QUERY_TILE_ENV: &str = "QWEN_FN_QSA_INDEX_QUERY_TILE";

fn configured_tile() -> usize {
    static TILE: OnceLock<usize> = OnceLock::new();
    *TILE.get_or_init(
        || match std::env::var(INDEX_QUERY_TILE_ENV).ok().as_deref() {
            Some("0") => 0,
            Some("4") => 4,
            _ => 8,
        },
    )
}

/// M5 Flash-Next exact radix4 selector width for full 32-query packed bands with >=8192 block capacity. Default 128; 32/64 are tuning alternatives and 256 restores the incumbent. Integer selection, stable ties, status and chronological IDs are unchanged. Other geometries/devices retain 256 threads.
const SELECTOR_THREADS_ENV: &str = "QWEN_FN_QSA_SELECTOR_THREADS";

pub(super) fn selector_threads(ctx: &MetalContext, capacity: usize, count: usize) -> usize {
    if !ctx.tensor_ops_status().is_available() || capacity < 8192 || count != 32 {
        return ATTENTION_THREADS;
    }
    static THREADS: OnceLock<usize> = OnceLock::new();
    *THREADS.get_or_init(
        || match std::env::var(SELECTOR_THREADS_ENV).ok().as_deref() {
            Some("32") => 32,
            Some("64") => 64,
            Some("256") => 256,
            _ => 128,
        },
    )
}

pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    keys: &MetalTensor,
    scores: &MetalTensor,
    visible: &MetalTensor,
    start: usize,
    count: usize,
    ratio: usize,
    capacity: usize,
) -> Result<bool, MetalError> {
    let tile = configured_tile();
    let Some(end) = start.checked_add(count) else {
        return Ok(false);
    };
    if tile == 0
        || !ctx.tensor_ops_status().is_available()
        || count != 32
        || ratio != 4
        || end / ratio < 8192
        || end / ratio > capacity
    {
        return Ok(false);
    }
    let name = if tile == 4 {
        "kernel_qwen4exp_qsa_index_reuse4_f32"
    } else {
        "kernel_qwen4exp_qsa_index_reuse8_f32"
    };
    let pipeline = match ctx.pipeline(name) {
        Ok(p)
            if p.threadExecutionWidth() == 32
                && p.maxTotalThreadsPerThreadgroup() >= 256
                && p.staticThreadgroupMemoryLength() <= ctx.device.maxThreadgroupMemoryLength() =>
        {
            p
        }
        Ok(_) | Err(MetalError::NoFunction(_) | MetalError::Pipeline(_, _)) => return Ok(false),
        Err(e) => return Err(e),
    };
    use crate::metal::checks::{check_disjoint, check_tensor, require_serial};
    require_serial("qsa_index_reuse", enc)?;
    check_tensor(
        "qsa_index_reuse",
        queries,
        GgmlType::F32,
        &[512, count as u64],
        false,
        "queries",
    )?;
    check_tensor(
        "qsa_index_reuse",
        keys,
        GgmlType::F16,
        &[128, capacity as u64],
        false,
        "keys",
    )?;
    check_tensor(
        "qsa_index_reuse",
        scores,
        GgmlType::F32,
        &[capacity as u64, count as u64],
        true,
        "scores",
    )?;
    check_tensor(
        "qsa_index_reuse",
        visible,
        GgmlType::I32,
        &[count as u64],
        true,
        "visible",
    )?;
    check_disjoint(
        "qsa_index_reuse",
        scores,
        &[(queries, "queries"), (keys, "keys"), (visible, "visible")],
    )?;
    check_disjoint(
        "qsa_index_reuse",
        visible,
        &[(queries, "queries"), (keys, "keys")],
    )?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PackedScoreArgs {
            start_position: start as u32,
            query_count: count as u32,
            ratio: ratio as u32,
            block_capacity: capacity as u32,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, keys);
    enc.set_tensor(3, scores);
    enc.set_tensor(4, visible);
    enc.dispatch(
        MTLSize {
            width: (end / ratio).div_ceil(8),
            height: count.div_ceil(tile),
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(true)
}
