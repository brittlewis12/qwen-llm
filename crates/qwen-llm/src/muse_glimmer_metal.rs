//! Muse Glimmer-specific Metal kernels over the shared execution primitives.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, encode_attn_decode_f16kv_f32,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2_metal::{MTLBuffer, MTLComputePipelineState, MTLDevice, MTLSize};

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    query_head_count: usize,
    key_head_count: usize,
    head_dim: usize,
    position: u32,
    theta: f32,
) -> Result<(), MetalError> {
    if query_head_count == 0 || key_head_count == 0 || head_dim == 0 || !head_dim.is_multiple_of(2)
    {
        return bad_shape(
            "muse_glimmer_rope",
            format!(
                "head counts and even head_dim must be nonzero, got q={query_head_count} k={key_head_count} dim={head_dim}"
            ),
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        return bad_shape(
            "muse_glimmer_rope",
            format!("theta must be finite and positive, got {theta}"),
        );
    }
    validate_writable_f32(
        query,
        checked_elements(
            "muse_glimmer_rope",
            query_head_count,
            head_dim,
            "query element count",
        )?,
        "query",
        "muse_glimmer_rope",
    )?;
    validate_writable_f32(
        key,
        checked_elements(
            "muse_glimmer_rope",
            key_head_count,
            head_dim,
            "key element count",
        )?,
        "key",
        "muse_glimmer_rope",
    )?;
    if metal_tensor_ranges_overlap(query, key) {
        return bad_shape(
            "muse_glimmer_rope",
            "query and key storage ranges overlap".into(),
        );
    }
    if position == 0 {
        return Ok(());
    }

    let pairs_per_head = head_dim / 2;
    let query_pair_count = checked_elements(
        "muse_glimmer_rope",
        query_head_count,
        pairs_per_head,
        "query pair count",
    )?;
    let key_pair_count = checked_elements(
        "muse_glimmer_rope",
        key_head_count,
        pairs_per_head,
        "key pair count",
    )?;
    let pair_count = query_pair_count
        .checked_add(key_pair_count)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "muse_glimmer_rope",
            detail: "combined pair count overflow".into(),
        })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        query_pair_count: u32,
        pair_count: u32,
        head_dim: u32,
        position: u32,
        theta: f32,
    }

    let pso = ctx.pipeline("kernel_muse_glimmer_rope_adjacent_pair_in_place_f32")?;
    enc.note_write(query);
    enc.note_write(key);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            query_pair_count: checked_u32(
                "muse_glimmer_rope",
                query_pair_count,
                "query pair count",
            )?,
            pair_count: checked_u32("muse_glimmer_rope", pair_count, "pair count")?,
            head_dim: checked_u32("muse_glimmer_rope", head_dim, "head_dim")?,
            position,
            theta,
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, key);
    let threads = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_rope_adjacent_pair_rows_in_place_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    query_head_count: usize,
    key_head_count: usize,
    head_dim: usize,
    row_count: usize,
    base_position: usize,
    theta: f32,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_glimmer_rope_rows";
    if query_head_count == 0
        || key_head_count == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || row_count == 0
    {
        return bad_shape(
            KERNEL,
            format!(
                "head counts, even head_dim, and rows must be nonzero, got q={query_head_count} k={key_head_count} dim={head_dim} rows={row_count}"
            ),
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        return bad_shape(
            KERNEL,
            format!("theta must be finite and positive, got {theta}"),
        );
    }
    let last_position =
        base_position
            .checked_add(row_count - 1)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "position range overflow".into(),
            })?;
    let base_position_u32 = checked_u32(KERNEL, base_position, "base position")?;
    checked_u32(KERNEL, last_position, "last position")?;

    let query_width = checked_elements(KERNEL, query_head_count, head_dim, "query width")?;
    let key_width = checked_elements(KERNEL, key_head_count, head_dim, "key width")?;
    let query_elements = checked_elements(KERNEL, row_count, query_width, "query elements")?;
    let key_elements = checked_elements(KERNEL, row_count, key_width, "key elements")?;
    validate_writable_f32_shape(
        query,
        query_elements,
        &[query_width as u64, row_count as u64],
        "query",
        KERNEL,
    )?;
    validate_writable_f32_shape(
        key,
        key_elements,
        &[key_width as u64, row_count as u64],
        "key",
        KERNEL,
    )?;
    if metal_tensor_ranges_overlap(query, key) {
        return bad_shape(KERNEL, "query and key storage ranges overlap".into());
    }

    let pairs_per_head = head_dim / 2;
    let query_pairs_per_row = checked_elements(
        KERNEL,
        query_head_count,
        pairs_per_head,
        "query pairs per row",
    )?;
    let key_pairs_per_row =
        checked_elements(KERNEL, key_head_count, pairs_per_head, "key pairs per row")?;
    let query_pair_count =
        checked_elements(KERNEL, row_count, query_pairs_per_row, "query pair count")?;
    let key_pair_count = checked_elements(KERNEL, row_count, key_pairs_per_row, "key pair count")?;
    let pair_count = query_pair_count
        .checked_add(key_pair_count)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "combined pair count overflow".into(),
        })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        query_pair_count: u32,
        pair_count: u32,
        query_pairs_per_row: u32,
        key_pairs_per_row: u32,
        head_dim: u32,
        base_position: u32,
        theta: f32,
    }

    let pipeline = ctx.pipeline("kernel_muse_glimmer_rope_adjacent_pair_rows_in_place_f32")?;
    enc.note_write(query);
    enc.note_write(key);
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            query_pair_count: checked_u32(KERNEL, query_pair_count, "query pair count")?,
            pair_count: checked_u32(KERNEL, pair_count, "pair count")?,
            query_pairs_per_row: checked_u32(KERNEL, query_pairs_per_row, "query pairs per row")?,
            key_pairs_per_row: checked_u32(KERNEL, key_pairs_per_row, "key pairs per row")?,
            head_dim: checked_u32(KERNEL, head_dim, "head_dim")?,
            base_position: base_position_u32,
            theta,
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, key);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    query_head_count: usize,
    key_head_count: usize,
    head_dim: usize,
    row_count: usize,
    position_period: usize,
    theta: f32,
    inverse: bool,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_glimmer_rope_periodic";
    if query_head_count == 0
        || key_head_count == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || row_count == 0
        || position_period == 0
        || !row_count.is_multiple_of(position_period)
    {
        return bad_shape(
            KERNEL,
            format!(
                "head counts, even head_dim, rows, and a row-dividing position period must be nonzero, got q={query_head_count} k={key_head_count} dim={head_dim} rows={row_count} period={position_period}"
            ),
        );
    }
    if !theta.is_finite() || theta <= 0.0 {
        return bad_shape(
            KERNEL,
            format!("theta must be finite and positive, got {theta}"),
        );
    }

    let query_width = checked_elements(KERNEL, query_head_count, head_dim, "query width")?;
    let key_width = checked_elements(KERNEL, key_head_count, head_dim, "key width")?;
    let query_elements = checked_elements(KERNEL, row_count, query_width, "query elements")?;
    let key_elements = checked_elements(KERNEL, row_count, key_width, "key elements")?;
    validate_writable_f32_shape(
        query,
        query_elements,
        &[query_width as u64, row_count as u64],
        "query",
        KERNEL,
    )?;
    validate_writable_f32_shape(
        key,
        key_elements,
        &[key_width as u64, row_count as u64],
        "key",
        KERNEL,
    )?;
    if metal_tensor_ranges_overlap(query, key) {
        return bad_shape(KERNEL, "query and key storage ranges overlap".into());
    }

    let pairs_per_head = head_dim / 2;
    let query_pairs_per_row = checked_elements(
        KERNEL,
        query_head_count,
        pairs_per_head,
        "query pairs per row",
    )?;
    let key_pairs_per_row =
        checked_elements(KERNEL, key_head_count, pairs_per_head, "key pairs per row")?;
    let query_pair_count =
        checked_elements(KERNEL, row_count, query_pairs_per_row, "query pair count")?;
    let key_pair_count = checked_elements(KERNEL, row_count, key_pairs_per_row, "key pair count")?;
    let pair_count = query_pair_count
        .checked_add(key_pair_count)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "combined pair count overflow".into(),
        })?;

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        query_pair_count: u32,
        pair_count: u32,
        query_pairs_per_row: u32,
        key_pairs_per_row: u32,
        head_dim: u32,
        position_period: u32,
        inverse: u32,
        theta: f32,
    }

    let pso = ctx.pipeline("kernel_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32")?;
    enc.note_write(query);
    enc.note_write(key);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            query_pair_count: checked_u32(KERNEL, query_pair_count, "query pair count")?,
            pair_count: checked_u32(KERNEL, pair_count, "pair count")?,
            query_pairs_per_row: checked_u32(KERNEL, query_pairs_per_row, "query pairs per row")?,
            key_pairs_per_row: checked_u32(KERNEL, key_pairs_per_row, "key pairs per row")?,
            head_dim: checked_u32(KERNEL, head_dim, "head_dim")?,
            position_period: checked_u32(KERNEL, position_period, "position period")?,
            inverse: u32::from(inverse),
            theta,
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, key);
    let threads = pso.maxTotalThreadsPerThreadgroup().min(256);
    enc.dispatch(
        MTLSize {
            width: pair_count.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_muse_glimmer_logit_softcap_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    output: &MetalTensor,
    scale: f32,
    cap: f32,
) -> Result<(), MetalError> {
    let count = usize::try_from(input.n_elements()).map_err(|_| MetalError::BadShape {
        kernel: "muse_glimmer_logit_softcap",
        detail: "input element count exceeds usize".into(),
    })?;
    if count == 0 || output.n_elements() != input.n_elements() {
        return bad_shape(
            "muse_glimmer_logit_softcap",
            format!(
                "input/output must have the same nonzero element count, got {}/{}",
                input.n_elements(),
                output.n_elements()
            ),
        );
    }
    validate_readable_f32(input, count, "input", "muse_glimmer_logit_softcap")?;
    validate_writable_f32(output, count, "output", "muse_glimmer_logit_softcap")?;
    if !scale.is_finite() || !cap.is_finite() || cap <= 0.0 {
        return bad_shape(
            "muse_glimmer_logit_softcap",
            format!("scale must be finite and cap positive, got scale={scale} cap={cap}"),
        );
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        count: u32,
        scale: f32,
        cap: f32,
    }

    let pso = ctx.pipeline("kernel_muse_glimmer_logit_softcap_f32")?;
    enc.note_read(input);
    enc.note_write(output);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            count: checked_u32("muse_glimmer_logit_softcap", count, "logit count")?,
            scale,
            cap,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, output);
    let threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: count.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub const MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS: usize = 7_168;

#[path = "muse_split_attention.rs"]
pub(crate) mod split_attention;
#[cfg(test)]
pub(crate) use split_attention as split_attention_pilot;

#[cfg(test)]
thread_local! {
    static FORCE_ONLINE_ATTENTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

#[cfg(test)]
pub(crate) fn with_online_attention<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_ONLINE_ATTENTION.set(self.0);
        }
    }
    let _restore = Restore(FORCE_ONLINE_ATTENTION.replace(enabled));
    run()
}
const MUSE_GLIMMER_QUERY_HEAD_COUNT: usize = 32;
const MUSE_GLIMMER_KV_HEAD_COUNT: usize = 2;
const MUSE_GLIMMER_ATTENTION_HEAD_DIM: usize = 128;

#[cfg(test)]
thread_local! {
    static FORCE_PACKED_ONLINE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static FORCE_TILED_PREFILL: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static TILED_PREFILL_DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    static FORCE_MATRIX_PV: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static MATRIX_PV_DISPATCHES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn with_matrix_pv<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_MATRIX_PV.set(self.0);
        }
    }
    let _restore = Restore(FORCE_MATRIX_PV.replace(enabled));
    run()
}

#[cfg(test)]
pub(crate) fn tiled_prefill_dispatch_count() -> u64 {
    TILED_PREFILL_DISPATCHES.get()
}

#[cfg(test)]
pub(crate) fn with_tiled_prefill<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_TILED_PREFILL.set(self.0);
        }
    }
    let _restore = Restore(FORCE_TILED_PREFILL.replace(Some(enabled)));
    run()
}

#[cfg(test)]
pub(crate) fn with_packed_online<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            FORCE_PACKED_ONLINE.set(self.0);
        }
    }
    let _restore = Restore(FORCE_PACKED_ONLINE.replace(enabled));
    run()
}

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_attn_prefill_f16kv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    output: &MetalTensor,
    row_count: usize,
    base_position: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
) -> Result<(), MetalError> {
    encode_muse_glimmer_attn_prefill_with_online(
        ctx,
        enc,
        query,
        key_cache,
        value_cache,
        output,
        row_count,
        base_position,
        query_head_count,
        kv_head_count,
        head_dim,
        sliding_window,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_muse_glimmer_attn_prefill_with_online(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    output: &MetalTensor,
    row_count: usize,
    base_position: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    online: bool,
) -> Result<(), MetalError> {
    encode_muse_glimmer_attn_prefill_with_tiling(
        ctx,
        enc,
        query,
        key_cache,
        value_cache,
        output,
        row_count,
        base_position,
        query_head_count,
        kv_head_count,
        head_dim,
        sliding_window,
        online,
        false,
    )
}

pub(crate) fn tiled_prefill_work_eligible(row_count: usize, query_offset: u64) -> bool {
    row_count == 128 && query_offset.is_multiple_of(32)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_muse_glimmer_attn_prefill_with_tiling(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    output: &MetalTensor,
    row_count: usize,
    base_position: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    online: bool,
    tiled_policy: bool,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_glimmer_attn_prefill";
    if row_count == 0
        || query_head_count == 0
        || kv_head_count == 0
        || head_dim == 0
        || !query_head_count.is_multiple_of(kv_head_count)
        || sliding_window == Some(0)
    {
        return bad_shape(
            KERNEL,
            format!(
                "expected nonzero rows and GQA geometry, got rows={row_count} q={query_head_count} kv={kv_head_count} dim={head_dim} window={sliding_window:?}"
            ),
        );
    }
    let end_position =
        base_position
            .checked_add(row_count)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "position range overflow".into(),
            })?;
    let query_width = checked_elements(KERNEL, query_head_count, head_dim, "query width")?;
    let kv_width = checked_elements(KERNEL, kv_head_count, head_dim, "KV width")?;
    let query_elements = checked_elements(KERNEL, row_count, query_width, "query elements")?;
    let cache_elements = checked_elements(KERNEL, end_position, kv_width, "cache elements")?;
    let maximum_visible = sliding_window
        .map(|window| end_position.min(window))
        .unwrap_or(end_position);
    validate_readable_f32(query, query_elements, "query", KERNEL)?;
    validate_readable_f16(key_cache, cache_elements, "key cache", KERNEL)?;
    validate_readable_f16(value_cache, cache_elements, "value cache", KERNEL)?;
    validate_writable_f32(output, query_elements, "output", KERNEL)?;
    for (source, name) in [
        (query, "query"),
        (key_cache, "key cache"),
        (value_cache, "value cache"),
    ] {
        if metal_tensor_ranges_overlap(source, output) {
            return bad_shape(KERNEL, format!("{name} overlaps output storage"));
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        row_count: u32,
        base_position: u32,
        end_position: u32,
        kv_stride: u32,
        query_head_count: u32,
        kv_head_count: u32,
        head_dim: u32,
        sliding_window: u32,
        scale: f32,
    }
    #[cfg(test)]
    let online = online || FORCE_PACKED_ONLINE.get();
    if online {
        if (query_head_count, kv_head_count, head_dim) != (32, 2, 128) || row_count > 128 {
            return bad_shape(
                KERNEL,
                "packed online requires G16/H128 and at most128 rows".into(),
            );
        }
        for (tensor, alignment) in [(query, 16), (key_cache, 8), (value_cache, 8), (output, 16)] {
            if !tensor.offset.is_multiple_of(alignment) {
                return bad_shape(KERNEL, "unaligned packed online view".into());
            }
        }
        let tiled = tiled_policy && tiled_prefill_work_eligible(row_count, query.offset);
        #[cfg(test)]
        let tiled = FORCE_TILED_PREFILL.get().unwrap_or(tiled);
        if tiled && !query.offset.is_multiple_of(32) {
            return bad_shape(
                KERNEL,
                "tiled F32 matrix query requires32-byte alignment".into(),
            );
        }
        let threads = if tiled { 128 } else { 32 };
        #[cfg(test)]
        if tiled {
            TILED_PREFILL_DISPATCHES.set(TILED_PREFILL_DISPATCHES.get() + 1);
        }
        #[cfg(test)]
        let matrix_pv = tiled && FORCE_MATRIX_PV.get();
        #[cfg(not(test))]
        let matrix_pv = false;
        #[cfg(test)]
        if matrix_pv {
            MATRIX_PV_DISPATCHES.set(MATRIX_PV_DISPATCHES.get() + 1);
        }
        let pipeline = ctx.pipeline(if matrix_pv {
            "kernel_muse_prefill_matrix_pv_f32_h128"
        } else if tiled {
            "kernel_muse_prefill_tiled_f32_h128"
        } else {
            "kernel_muse_prefill_online_h128"
        })?;
        if pipeline.threadExecutionWidth() != 32
            || pipeline.maxTotalThreadsPerThreadgroup() < threads
        {
            return bad_shape(KERNEL, "packed online requires32-lane SIMDgroups".into());
        }
        if pipeline.staticThreadgroupMemoryLength() > ctx.device.maxThreadgroupMemoryLength() {
            return bad_shape(
                KERNEL,
                "packed attention exceeds threadgroup memory capacity".into(),
            );
        }
        enc.note_read(query);
        enc.note_read(key_cache);
        enc.note_read(value_cache);
        enc.note_write(output);
        enc.set_pipeline(&pipeline);
        enc.set_bytes(
            0,
            &Args {
                row_count: checked_u32(KERNEL, row_count, "row count")?,
                base_position: checked_u32(KERNEL, base_position, "base position")?,
                end_position: checked_u32(KERNEL, end_position, "end position")?,
                kv_stride: 256,
                query_head_count: 32,
                kv_head_count: 2,
                head_dim: 128,
                sliding_window: checked_u32(KERNEL, sliding_window.unwrap_or(0), "sliding window")?,
                scale: 128.0_f32.sqrt().recip(),
            },
        );
        enc.set_tensor(1, query);
        enc.set_tensor(2, key_cache);
        enc.set_tensor(3, value_cache);
        enc.set_tensor(4, output);
        enc.dispatch(
            MTLSize {
                width: if tiled { 2 } else { 32 },
                height: if tiled {
                    row_count.div_ceil(2)
                } else {
                    row_count
                },
                depth: 1,
            },
            MTLSize {
                width: threads,
                height: 1,
                depth: 1,
            },
        );
        return Ok(());
    }
    if maximum_visible > MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS {
        return bad_shape(
            KERNEL,
            format!(
                "materialized packed attention supports at most {MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS} visible positions, got {maximum_visible}"
            ),
        );
    }
    let pipeline = ctx.pipeline("kernel_muse_glimmer_attn_prefill_f16kv_f32")?;
    let materialized_pipeline = ctx.pipeline("kernel_attn_decode_f16kv")?;
    let threads = materialized_pipeline
        .maxTotalThreadsPerThreadgroup()
        .min(1024);
    if pipeline.threadExecutionWidth() != 32
        || materialized_pipeline.threadExecutionWidth() != 32
        || threads < 32
        || !threads.is_multiple_of(32)
        || pipeline.maxTotalThreadsPerThreadgroup() < threads
    {
        return bad_shape(
            KERNEL,
            format!(
                "packed attention requires the materialized kernel's positive multiple-of-32 thread geometry, got packed width/max={}/{} materialized width/threads={}/{}",
                pipeline.threadExecutionWidth(),
                pipeline.maxTotalThreadsPerThreadgroup(),
                materialized_pipeline.threadExecutionWidth(),
                threads,
            ),
        );
    }
    let simdgroups = threads / 32;
    let scores_bytes = maximum_visible
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "score scratch byte count overflow".into(),
        })?;
    let reduction_bytes = (simdgroups * std::mem::size_of::<f32>()).max(32);
    let dynamic_memory =
        scores_bytes
            .checked_add(reduction_bytes)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "threadgroup memory byte count overflow".into(),
            })?;
    let required_memory = pipeline
        .staticThreadgroupMemoryLength()
        .checked_add(dynamic_memory)
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "total threadgroup memory byte count overflow".into(),
        })?;
    if required_memory > ctx.device.maxThreadgroupMemoryLength() {
        return bad_shape(
            KERNEL,
            format!(
                "packed attention requires {required_memory} threadgroup bytes, device exposes {}",
                ctx.device.maxThreadgroupMemoryLength()
            ),
        );
    }
    enc.note_read(query);
    enc.note_read(key_cache);
    enc.note_read(value_cache);
    enc.note_write(output);
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            row_count: checked_u32(KERNEL, row_count, "row count")?,
            base_position: checked_u32(KERNEL, base_position, "base position")?,
            end_position: checked_u32(KERNEL, end_position, "end position")?,
            kv_stride: checked_u32(KERNEL, kv_width, "KV stride")?,
            query_head_count: checked_u32(KERNEL, query_head_count, "query head count")?,
            kv_head_count: checked_u32(KERNEL, kv_head_count, "KV head count")?,
            head_dim: checked_u32(KERNEL, head_dim, "head dim")?,
            sliding_window: checked_u32(KERNEL, sliding_window.unwrap_or(0), "sliding window")?,
            scale: (head_dim as f32).sqrt().recip(),
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, key_cache);
    enc.set_tensor(3, value_cache);
    enc.set_tensor(4, output);
    enc.set_threadgroup_memory(0, scores_bytes);
    enc.set_threadgroup_memory(1, reduction_bytes);
    enc.dispatch(
        MTLSize {
            width: query_head_count,
            height: row_count,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_attn_decode_f16kv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    output: &MetalTensor,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    visible_positions: usize,
) -> Result<(), MetalError> {
    if visible_positions == 0 {
        return bad_shape(
            "muse_glimmer_attn_decode",
            "visible position count must be nonzero".into(),
        );
    }
    #[cfg(test)]
    if split_attention_pilot::try_encode(
        ctx,
        enc,
        query,
        key_cache,
        value_cache,
        output,
        query_head_count,
        kv_head_count,
        head_dim,
        visible_positions,
    )? {
        return Ok(());
    }
    #[cfg(test)]
    if FORCE_ONLINE_ATTENTION.get() {
        return encode_muse_glimmer_attn_decode_online_f16kv_f32(
            ctx,
            enc,
            query,
            key_cache,
            value_cache,
            output,
            query_head_count,
            kv_head_count,
            head_dim,
            visible_positions,
        );
    }
    if visible_positions <= MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS {
        return encode_attn_decode_f16kv_f32(
            ctx,
            enc,
            query,
            key_cache,
            value_cache,
            output,
            query_head_count,
            kv_head_count,
            head_dim,
            visible_positions,
        );
    }
    encode_muse_glimmer_attn_decode_online_f16kv_f32(
        ctx,
        enc,
        query,
        key_cache,
        value_cache,
        output,
        query_head_count,
        kv_head_count,
        head_dim,
        visible_positions,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_muse_glimmer_attn_decode_online_f16kv_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key_cache: &MetalTensor,
    value_cache: &MetalTensor,
    output: &MetalTensor,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    visible_positions: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_glimmer_attn_decode_online_f16kv";
    if query_head_count != MUSE_GLIMMER_QUERY_HEAD_COUNT
        || kv_head_count != MUSE_GLIMMER_KV_HEAD_COUNT
        || head_dim != MUSE_GLIMMER_ATTENTION_HEAD_DIM
        || visible_positions == 0
    {
        return bad_shape(
            KERNEL,
            format!(
                "expected released 32Q/2KV/H128 geometry and at least one visible position, got q={query_head_count} kv={kv_head_count} dim={head_dim} positions={visible_positions}"
            ),
        );
    }
    let query_width = checked_elements(KERNEL, query_head_count, head_dim, "query width")?;
    let kv_width = checked_elements(KERNEL, kv_head_count, head_dim, "KV width")?;
    let cache_elements = checked_elements(
        KERNEL,
        visible_positions,
        kv_width,
        "visible cache elements",
    )?;
    validate_readable_f32(query, query_width, "query", KERNEL)?;
    validate_readable_f16(key_cache, cache_elements, "key cache", KERNEL)?;
    validate_readable_f16(value_cache, cache_elements, "value cache", KERNEL)?;
    validate_writable_f32(output, query_width, "output", KERNEL)?;
    for (tensor, alignment, name) in [
        (query, 16, "query"),
        (key_cache, 8, "key cache"),
        (value_cache, 8, "value cache"),
        (output, 16, "output"),
    ] {
        if !tensor.offset.is_multiple_of(alignment) {
            return bad_shape(
                KERNEL,
                format!(
                    "{name} offset {} is not {alignment}-byte aligned",
                    tensor.offset
                ),
            );
        }
    }
    for (source, name) in [
        (query, "query"),
        (key_cache, "key cache"),
        (value_cache, "value cache"),
    ] {
        if metal_tensor_ranges_overlap(source, output) {
            return bad_shape(KERNEL, format!("{name} overlaps output storage"));
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        visible_positions: u32,
        kv_stride: u32,
        scale: f32,
    }
    let pipeline = ctx.pipeline("kernel_muse_glimmer_attn_decode_online_f16kv_h128_f32")?;
    if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 32 {
        return bad_shape(
            KERNEL,
            "online attention requires one 32-thread SIMDgroup".into(),
        );
    }
    enc.note_read(query);
    enc.note_read(key_cache);
    enc.note_read(value_cache);
    enc.note_write(output);
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            visible_positions: checked_u32(KERNEL, visible_positions, "visible position count")?,
            kv_stride: checked_u32(KERNEL, kv_width, "KV stride")?,
            scale: (head_dim as f32).sqrt().recip(),
        },
    );
    enc.set_tensor(1, query);
    enc.set_tensor(2, key_cache);
    enc.set_tensor(3, value_cache);
    enc.set_tensor(4, output);
    enc.dispatch(
        MTLSize {
            width: query_head_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn encode_muse_glimmer_causal_gqa_vjp_bank_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    query: &MetalTensor,
    key: &MetalTensor,
    value: &MetalTensor,
    gate: &MetalTensor,
    attention_output: &MetalTensor,
    probabilities: &MetalTensor,
    grad_gated: &MetalTensor,
    grad_query: &MetalTensor,
    partial_grad_key: &MetalTensor,
    partial_grad_value: &MetalTensor,
    grad_key: &MetalTensor,
    grad_value: &MetalTensor,
    grad_gate: &MetalTensor,
    basis_count: usize,
    n_tokens: usize,
    query_head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
) -> Result<(), MetalError> {
    const KERNEL: &str = "muse_glimmer_causal_gqa_vjp_bank";
    if enc.is_concurrent() {
        return bad_shape(
            KERNEL,
            "dependent VJP dispatches require a serial encoder".into(),
        );
    }
    if basis_count == 0
        || n_tokens == 0
        || n_tokens > 16
        || query_head_count == 0
        || kv_head_count == 0
        || head_dim == 0
        || !query_head_count.is_multiple_of(kv_head_count)
        || !head_dim.is_multiple_of(32)
        || head_dim > 256
    {
        return bad_shape(
            KERNEL,
            format!(
                "expected B>0, T in 1..=16, divisible heads, and head_dim in 32..=256; got B={basis_count} T={n_tokens} q={query_head_count} kv={kv_head_count} dim={head_dim}"
            ),
        );
    }

    let query_width = checked_elements(KERNEL, query_head_count, head_dim, "query width")?;
    let kv_width = checked_elements(KERNEL, kv_head_count, head_dim, "KV width")?;
    let primal_query_elements =
        checked_elements(KERNEL, n_tokens, query_width, "primal query elements")?;
    let primal_kv_elements = checked_elements(KERNEL, n_tokens, kv_width, "primal KV elements")?;
    let bank_rows = checked_elements(KERNEL, basis_count, n_tokens, "bank rows")?;
    let bank_query_elements =
        checked_elements(KERNEL, bank_rows, query_width, "bank query elements")?;
    let bank_kv_elements = checked_elements(KERNEL, bank_rows, kv_width, "bank KV elements")?;
    let probability_rows =
        checked_elements(KERNEL, n_tokens, query_head_count, "probability rows")?;
    let probability_elements =
        checked_elements(KERNEL, probability_rows, n_tokens, "probability elements")?;
    let partial_heads = checked_elements(KERNEL, basis_count, query_head_count, "partial heads")?;
    let partial_rows = checked_elements(KERNEL, partial_heads, n_tokens, "partial rows")?;
    let partial_elements = checked_elements(KERNEL, partial_rows, head_dim, "partial elements")?;

    let primal_query_shape = [query_width as u64, n_tokens as u64];
    let primal_kv_shape = [kv_width as u64, n_tokens as u64];
    let probability_shape = [n_tokens as u64, query_head_count as u64, n_tokens as u64];
    let bank_query_shape = [query_width as u64, bank_rows as u64];
    let bank_kv_shape = [kv_width as u64, bank_rows as u64];
    let partial_shape = [
        head_dim as u64,
        n_tokens as u64,
        query_head_count as u64,
        basis_count as u64,
    ];

    for (tensor, elements, shape, name) in [
        (
            query,
            primal_query_elements,
            primal_query_shape.as_slice(),
            "query",
        ),
        (key, primal_kv_elements, primal_kv_shape.as_slice(), "key"),
        (
            value,
            primal_kv_elements,
            primal_kv_shape.as_slice(),
            "value",
        ),
        (
            gate,
            primal_query_elements,
            primal_query_shape.as_slice(),
            "gate",
        ),
        (
            attention_output,
            primal_query_elements,
            primal_query_shape.as_slice(),
            "attention output",
        ),
        (
            probabilities,
            probability_elements,
            probability_shape.as_slice(),
            "probabilities",
        ),
        (
            grad_gated,
            bank_query_elements,
            bank_query_shape.as_slice(),
            "gated cotangent bank",
        ),
    ] {
        validate_readable_f32_shape(tensor, elements, shape, name, KERNEL)?;
    }
    for (tensor, elements, shape, name) in [
        (
            grad_query,
            bank_query_elements,
            bank_query_shape.as_slice(),
            "query cotangent bank",
        ),
        (
            partial_grad_key,
            partial_elements,
            partial_shape.as_slice(),
            "partial key bank",
        ),
        (
            partial_grad_value,
            partial_elements,
            partial_shape.as_slice(),
            "partial value bank",
        ),
        (
            grad_key,
            bank_kv_elements,
            bank_kv_shape.as_slice(),
            "key cotangent bank",
        ),
        (
            grad_value,
            bank_kv_elements,
            bank_kv_shape.as_slice(),
            "value cotangent bank",
        ),
        (
            grad_gate,
            bank_query_elements,
            bank_query_shape.as_slice(),
            "gate cotangent bank",
        ),
    ] {
        validate_writable_f32_shape(tensor, elements, shape, name, KERNEL)?;
    }

    let reads = [
        query,
        key,
        value,
        gate,
        attention_output,
        probabilities,
        grad_gated,
    ];
    let writes = [
        grad_query,
        partial_grad_key,
        partial_grad_value,
        grad_key,
        grad_value,
        grad_gate,
    ];
    for (write_index, write) in writes.iter().enumerate() {
        if reads
            .iter()
            .any(|read| metal_tensor_ranges_overlap(write, read))
            || writes[..write_index]
                .iter()
                .any(|prior| metal_tensor_ranges_overlap(write, prior))
        {
            return bad_shape(KERNEL, "VJP output overlaps another tensor".into());
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        basis_count: u32,
        n_tokens: u32,
        query_heads: u32,
        kv_heads: u32,
        head_dim: u32,
        scale: f32,
    }
    let args = Args {
        basis_count: checked_u32(KERNEL, basis_count, "basis count")?,
        n_tokens: checked_u32(KERNEL, n_tokens, "token count")?,
        query_heads: checked_u32(KERNEL, query_head_count, "query head count")?,
        kv_heads: checked_u32(KERNEL, kv_head_count, "KV head count")?,
        head_dim: checked_u32(KERNEL, head_dim, "head dimension")?,
        scale: (head_dim as f32).sqrt().recip(),
    };
    let partial_tgm_elements =
        checked_elements(KERNEL, n_tokens, head_dim, "partial TGM elements")?
            .checked_mul(2)
            .ok_or_else(|| MetalError::BadShape {
                kernel: KERNEL,
                detail: "partial TGM element count overflow".into(),
            })?;
    let partial_tgm_bytes = partial_tgm_elements
        .checked_mul(std::mem::size_of::<f32>())
        .ok_or_else(|| MetalError::BadShape {
            kernel: KERNEL,
            detail: "partial TGM byte count overflow".into(),
        })?;

    let vjp = ctx.pipeline("kernel_muse_glimmer_causal_gqa_vjp_bank_f32")?;
    let reduce = ctx.pipeline("kernel_muse_glimmer_causal_gqa_vjp_reduce_kv_f32")?;
    for (name, pipeline) in [("VJP", &vjp), ("KV reduction", &reduce)] {
        if pipeline.threadExecutionWidth() != 32 || pipeline.maxTotalThreadsPerThreadgroup() < 32 {
            return bad_shape(KERNEL, format!("{name} requires one 32-thread SIMDgroup"));
        }
    }
    if ctx.device.maxThreadgroupMemoryLength() < partial_tgm_bytes {
        return bad_shape(
            KERNEL,
            format!("VJP requires {partial_tgm_bytes} bytes of threadgroup memory"),
        );
    }

    for tensor in reads {
        enc.note_read(tensor);
    }
    for tensor in writes {
        enc.note_write(tensor);
    }
    enc.set_pipeline(&vjp);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, query);
    enc.set_tensor(2, key);
    enc.set_tensor(3, value);
    enc.set_tensor(4, gate);
    enc.set_tensor(5, attention_output);
    enc.set_tensor(6, probabilities);
    enc.set_tensor(7, grad_gated);
    enc.set_tensor(8, grad_query);
    enc.set_tensor(9, partial_grad_key);
    enc.set_tensor(10, partial_grad_value);
    enc.set_tensor(11, grad_gate);
    enc.set_threadgroup_memory(0, partial_tgm_bytes);
    enc.dispatch(
        MTLSize {
            width: query_head_count,
            height: basis_count,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );

    enc.set_pipeline(&reduce);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, partial_grad_key);
    enc.set_tensor(2, partial_grad_value);
    enc.set_tensor(3, grad_key);
    enc.set_tensor(4, grad_value);
    enc.dispatch(
        MTLSize {
            width: kv_head_count,
            height: n_tokens,
            depth: basis_count,
        },
        MTLSize {
            width: 32,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn validate_readable_f32(
    tensor: &MetalTensor,
    elements: usize,
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    if tensor.dtype != GgmlType::F32 || tensor.n_elements() != elements as u64 {
        return bad_shape(
            kernel,
            format!(
                "{name} must be F32 with {elements} elements, got {:?} and {}",
                tensor.dtype,
                tensor.n_elements()
            ),
        );
    }
    if !tensor
        .offset
        .is_multiple_of(std::mem::align_of::<f32>() as u64)
    {
        return bad_shape(
            kernel,
            format!("{name} offset {} is not F32-aligned", tensor.offset),
        );
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: format!("{name} buffer range overflow"),
        })?;
    if end > tensor.buffer.length() as u64 {
        return bad_shape(
            kernel,
            format!(
                "{name} range ends at {end}, beyond buffer length {}",
                tensor.buffer.length()
            ),
        );
    }
    Ok(())
}

fn validate_readable_f16(
    tensor: &MetalTensor,
    elements: usize,
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    if tensor.dtype != GgmlType::F16 || tensor.n_elements() != elements as u64 {
        return bad_shape(
            kernel,
            format!(
                "{name} must be F16 with {elements} elements, got {:?} and {}",
                tensor.dtype,
                tensor.n_elements()
            ),
        );
    }
    if !tensor
        .offset
        .is_multiple_of(std::mem::align_of::<u16>() as u64)
    {
        return bad_shape(
            kernel,
            format!("{name} offset {} is not F16-aligned", tensor.offset),
        );
    }
    let end = tensor
        .offset
        .checked_add(tensor.n_bytes())
        .ok_or_else(|| MetalError::BadShape {
            kernel,
            detail: format!("{name} buffer range overflow"),
        })?;
    if end > tensor.buffer.length() as u64 {
        return bad_shape(
            kernel,
            format!(
                "{name} range ends at {end}, beyond buffer length {}",
                tensor.buffer.length()
            ),
        );
    }
    Ok(())
}

fn validate_writable_f32(
    tensor: &MetalTensor,
    elements: usize,
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    validate_readable_f32(tensor, elements, name, kernel)?;
    if !tensor.is_writable() {
        return bad_shape(kernel, format!("{name} must be writable"));
    }
    Ok(())
}

fn validate_readable_f32_shape(
    tensor: &MetalTensor,
    elements: usize,
    shape: &[u64],
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    validate_readable_f32(tensor, elements, name, kernel)?;
    if tensor.shape.as_slice() != shape {
        return bad_shape(
            kernel,
            format!("{name} shape {:?} differs from {shape:?}", tensor.shape),
        );
    }
    Ok(())
}

fn validate_writable_f32_shape(
    tensor: &MetalTensor,
    elements: usize,
    shape: &[u64],
    name: &str,
    kernel: &'static str,
) -> Result<(), MetalError> {
    validate_writable_f32(tensor, elements, name, kernel)?;
    if tensor.shape.as_slice() != shape {
        return bad_shape(
            kernel,
            format!("{name} shape {:?} differs from {shape:?}", tensor.shape),
        );
    }
    Ok(())
}

fn metal_tensor_ranges_overlap(left: &MetalTensor, right: &MetalTensor) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let Some(left_end) = left.offset.checked_add(left.n_bytes()) else {
        return true;
    };
    let Some(right_end) = right.offset.checked_add(right.n_bytes()) else {
        return true;
    };
    left.offset < right_end && right.offset < left_end
}

fn checked_elements(
    kernel: &'static str,
    left: usize,
    right: usize,
    label: &'static str,
) -> Result<usize, MetalError> {
    left.checked_mul(right).ok_or_else(|| MetalError::BadShape {
        kernel,
        detail: format!("{label} overflow"),
    })
}

fn checked_u32(kernel: &'static str, value: usize, label: &str) -> Result<u32, MetalError> {
    u32::try_from(value).map_err(|_| MetalError::BadShape {
        kernel,
        detail: format!("{label} {value} exceeds u32"),
    })
}

fn bad_shape<T>(kernel: &'static str, detail: String) -> Result<T, MetalError> {
    Err(MetalError::BadShape { kernel, detail })
}

#[cfg(test)]
mod tests {
    use super::*;
    use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

    fn tensor_from_f32(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
        MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    }

    fn tensor_from_f32_shape(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
    }

    fn tensor_from_f16(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
        let values = values
            .iter()
            .map(|&value| half::f16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&values),
            vec![values.len() as u64],
            GgmlType::F16,
        )
        .unwrap()
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        let mut values = vec![0.0; tensor.n_elements() as usize];
        unsafe {
            std::ptr::copy_nonoverlapping(
                tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<f32>()
                    .add(tensor.offset as usize / std::mem::size_of::<f32>()),
                values.as_mut_ptr(),
                values.len(),
            );
        }
        values
    }

    fn attention_fixture(position_count: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let query = (0..MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM)
            .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.002)
            .collect::<Vec<_>>();
        let cache_elements =
            position_count * MUSE_GLIMMER_KV_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM;
        let key = (0..cache_elements)
            .map(|index| ((index * 13 % 89) as f32 - 44.0) * 0.003)
            .collect::<Vec<_>>();
        let value = (0..cache_elements)
            .map(|index| ((index * 19 % 97) as f32 - 48.0) * 0.004)
            .collect::<Vec<_>>();
        (query, key, value)
    }

    #[test]
    fn muse_glimmer_attn_online_matches_materialized_inside_overlap() {
        const POSITIONS: usize = 257;
        let ctx = MetalContext::new().unwrap();
        let (query_values, key_values, value_values) = attention_fixture(POSITIONS);
        let query = tensor_from_f32(&ctx, &query_values);
        let key = tensor_from_f16(&ctx, &key_values);
        let value = tensor_from_f16(&ctx, &value_values);
        let materialized = tensor_from_f32(
            &ctx,
            &vec![f32::NAN; MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM],
        );
        let online = tensor_from_f32(
            &ctx,
            &vec![f32::NAN; MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM],
        );

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_attn_decode_f16kv_f32(
            &ctx,
            &encoder,
            &query,
            &key,
            &value,
            &materialized,
            MUSE_GLIMMER_QUERY_HEAD_COUNT,
            MUSE_GLIMMER_KV_HEAD_COUNT,
            MUSE_GLIMMER_ATTENTION_HEAD_DIM,
            POSITIONS,
        )
        .unwrap();
        encode_muse_glimmer_attn_decode_online_f16kv_f32(
            &ctx,
            &encoder,
            &query,
            &key,
            &value,
            &online,
            MUSE_GLIMMER_QUERY_HEAD_COUNT,
            MUSE_GLIMMER_KV_HEAD_COUNT,
            MUSE_GLIMMER_ATTENTION_HEAD_DIM,
            POSITIONS,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let materialized = read_f32(&materialized);
        let online = read_f32(&online);
        assert!(materialized.iter().all(|value| value.is_finite()));
        assert!(online.iter().all(|value| value.is_finite()));
        let max_abs = materialized
            .iter()
            .zip(&online)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        let dot = materialized
            .iter()
            .zip(&online)
            .map(|(left, right)| left * right)
            .sum::<f32>();
        let materialized_norm = materialized
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        let online_norm = online.iter().map(|value| value * value).sum::<f32>().sqrt();
        let cosine = dot / (materialized_norm * online_norm);
        assert!(max_abs <= 5e-4, "max_abs={max_abs}");
        assert!(cosine >= 0.999_999, "cosine={cosine}");
    }

    #[test]
    #[ignore = "serial Metal, existing online attention overlap screen"]
    fn muse_online_attention_overlap_wall_screen() {
        attention_overlap_screen(false);
    }

    #[test]
    #[ignore = "serial Metal, H128 split-position attention screen"]
    fn muse_split_attention_overlap_wall_screen() {
        attention_overlap_screen(true);
    }

    fn attention_overlap_screen(split: bool) {
        let ctx = MetalContext::new().unwrap();
        let before = ctx.current_allocated_size();
        let scratch = split.then(|| split_attention_pilot::allocate(&ctx));
        let scratch_bytes = ctx.current_allocated_size() - before;
        assert!(scratch_bytes <= 1024 * 1024);
        let positions = if split {
            vec![
                1, 2, 31, 32, 127, 128, 129, 257, 2048, 6229, 7168, 7169, 32769, 257,
            ]
        } else {
            vec![1, 257, 2048, 6229, 7168]
        };
        for positions in positions {
            if let Some(partial) = &scratch {
                unsafe {
                    std::ptr::write_bytes(
                        (partial.buffer.contents().as_ptr() as *mut u8)
                            .add(partial.offset as usize),
                        0xff,
                        partial.n_bytes() as usize,
                    );
                }
            }
            let (mut query_values, mut key_values, mut value_values) =
                attention_fixture(positions + 1);
            if split {
                query_values.iter_mut().for_each(|x| *x *= 10.0);
                key_values.iter_mut().for_each(|x| *x *= 8.0);
                value_values.iter_mut().for_each(|x| *x *= 5.0);
            }
            let mut padded_query = vec![42.125; 4];
            padded_query.extend_from_slice(&query_values);
            let query_storage = tensor_from_f32(&ctx, &padded_query);
            let query = query_storage.view_subrange(4, vec![4096]);
            let key_storage = tensor_from_f16(&ctx, &key_values);
            let value_storage = tensor_from_f16(&ctx, &value_values);
            let key = key_storage.view_subrange(256, vec![(positions * 256) as u64]);
            let value = value_storage.view_subrange(256, vec![(positions * 256) as u64]);
            let a_storage = tensor_from_f32(&ctx, &vec![42.125; 4104]);
            let b_storage = tensor_from_f32(&ctx, &vec![42.125; 4104]);
            let a = a_storage.view_subrange(4, vec![4096]);
            let b = b_storage.view_subrange(4, vec![4096]);
            if split && positions == 1 {
                for concurrent in [true, false] {
                    let command = ctx.queue.commandBuffer().unwrap();
                    let encoder = if concurrent {
                        KernelEncoder::begin_concurrent(&command)
                    } else {
                        KernelEncoder::begin(&command)
                    };
                    let error =
                        split_attention_pilot::with_scratch(scratch.as_ref().unwrap(), || {
                            encode_muse_glimmer_attn_decode_f16kv_f32(
                                &ctx,
                                &encoder,
                                &query,
                                &key,
                                &value,
                                &b,
                                32,
                                2,
                                128,
                                if concurrent { 1 } else { u32::MAX as usize },
                            )
                        })
                        .unwrap_err();
                    assert!(error.to_string().contains(if concurrent {
                        "serial encoder"
                    } else {
                        "bounded nonempty"
                    }));
                    encoder.end();
                }
            }
            let run = |online: bool, repetitions: usize| {
                let started = std::time::Instant::now();
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                let encode = || {
                    for _ in 0..repetitions {
                        encode_muse_glimmer_attn_decode_f16kv_f32(
                            &ctx,
                            &encoder,
                            &query,
                            &key,
                            &value,
                            if online { &b } else { &a },
                            32,
                            2,
                            128,
                            positions,
                        )
                        .unwrap();
                    }
                };
                if split && online {
                    split_attention_pilot::with_scratch(scratch.as_ref().unwrap(), encode);
                } else {
                    with_online_attention(online, encode);
                }
                encoder.end();
                command.commit();
                command.waitUntilCompleted();
                assert_eq!(
                    command.status(),
                    objc2_metal::MTLCommandBufferStatus::Completed
                );
                assert!(command.error().is_none());
                (
                    started.elapsed().as_secs_f64() * 1e3 / repetitions as f64,
                    (command.GPUEndTime() - command.GPUStartTime()) * 1e3 / repetitions as f64,
                )
            };
            run(false, 1);
            run(true, 1);
            let av = read_f32(&a);
            let bv = read_f32(&b);
            let max_abs = av
                .iter()
                .zip(&bv)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f32, f32::max);
            let dot: f64 = av
                .iter()
                .zip(&bv)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum();
            let aa: f64 = av.iter().map(|&a| f64::from(a).powi(2)).sum();
            let bb: f64 = bv.iter().map(|&b| f64::from(b).powi(2)).sum();
            let cosine = dot / (aa * bb).sqrt();
            assert!(
                max_abs <= 5e-4 && cosine >= 0.999_999,
                "positions={positions} max_abs={max_abs} cosine={cosine}"
            );
            for storage in [&a_storage, &b_storage] {
                let values = read_f32(storage);
                assert!(
                    values[..4]
                        .iter()
                        .chain(&values[4100..])
                        .all(|&x| x == 42.125)
                );
            }
            eprintln!(
                "MUSE_ATTENTION_JSON {}",
                serde_json::json!({"kind":"oracle", "positions":positions,
                "cosine":cosine,"max_abs":max_abs,"nonzero_views_and_output_guards":true,
                "candidate":if split {"split"} else {"online"}, "scratch_driver_bytes":scratch_bytes})
            );
            if matches!(positions, 2048 | 6229) {
                for (index, online) in [false, true, true, false, false, true, true, false]
                    .into_iter()
                    .enumerate()
                {
                    let (wall_ms, gpu_ms) = run(online, 8);
                    eprintln!(
                        "MUSE_ATTENTION_JSON {}",
                        serde_json::json!({"kind":if index<4 {"warmup"} else {"sample"},
                        "positions":positions, "arm":if online {"B"} else {"A"}, "wall_ms":wall_ms,"gpu_ms":gpu_ms,
                        "candidate":if split {"split"} else {"online"}})
                    );
                }
            }
        }
    }

    include!("muse_packed_online_pilot.rs");
    include!("muse_attention_context_tests.rs");
    include!("muse_tiled_prefill_tests.rs");

    #[test]
    fn muse_glimmer_packed_attention_matches_scalar_rows_bitwise() {
        const ROWS: usize = 5;
        const QUERY_WIDTH: usize = MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM;
        const KV_WIDTH: usize = MUSE_GLIMMER_KV_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM;

        for (base_position, sliding_window) in [(3, None), (2_046, Some(2_048)), (7_163, None)] {
            let position_count = base_position + ROWS;
            let ctx = MetalContext::new().unwrap();
            let query_values = (0..ROWS * QUERY_WIDTH)
                .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.002)
                .collect::<Vec<_>>();
            let cache_elements = position_count * KV_WIDTH;
            let key_values = (0..cache_elements)
                .map(|index| ((index * 13 % 89) as f32 - 44.0) * 0.003)
                .collect::<Vec<_>>();
            let value_values = (0..cache_elements)
                .map(|index| ((index * 19 % 97) as f32 - 48.0) * 0.004)
                .collect::<Vec<_>>();
            let query =
                tensor_from_f32_shape(&ctx, &query_values, vec![QUERY_WIDTH as u64, ROWS as u64]);
            let key = tensor_from_f16(&ctx, &key_values);
            let value = tensor_from_f16(&ctx, &value_values);
            let packed =
                MetalTensor::zeros_f32(&ctx, vec![QUERY_WIDTH as u64, ROWS as u64]).unwrap();
            let scalar =
                MetalTensor::zeros_f32(&ctx, vec![QUERY_WIDTH as u64, ROWS as u64]).unwrap();

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_muse_glimmer_attn_prefill_f16kv_f32(
                &ctx,
                &encoder,
                &query,
                &key,
                &value,
                &packed,
                ROWS,
                base_position,
                MUSE_GLIMMER_QUERY_HEAD_COUNT,
                MUSE_GLIMMER_KV_HEAD_COUNT,
                MUSE_GLIMMER_ATTENTION_HEAD_DIM,
                sliding_window,
            )
            .unwrap();
            for row in 0..ROWS {
                let end = base_position + row + 1;
                let start = sliding_window
                    .map(|window| end.saturating_sub(window))
                    .unwrap_or(0);
                let visible = end - start;
                let query_row =
                    query.view_subrange((row * QUERY_WIDTH) as u64, vec![QUERY_WIDTH as u64]);
                let output_row =
                    scalar.view_subrange((row * QUERY_WIDTH) as u64, vec![QUERY_WIDTH as u64]);
                let key_view =
                    key.view_subrange((start * KV_WIDTH) as u64, vec![(visible * KV_WIDTH) as u64]);
                let value_view = value
                    .view_subrange((start * KV_WIDTH) as u64, vec![(visible * KV_WIDTH) as u64]);
                encode_attn_decode_f16kv_f32(
                    &ctx,
                    &encoder,
                    &query_row,
                    &key_view,
                    &value_view,
                    &output_row,
                    MUSE_GLIMMER_QUERY_HEAD_COUNT,
                    MUSE_GLIMMER_KV_HEAD_COUNT,
                    MUSE_GLIMMER_ATTENTION_HEAD_DIM,
                    visible,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(
                command.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
            );

            assert_eq!(
                read_f32(&packed)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                read_f32(&scalar)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                "base={base_position} window={sliding_window:?}"
            );
        }
    }

    #[test]
    fn muse_glimmer_attn_online_crosses_materialized_limit_and_reads_tail() {
        const POSITIONS: usize = MUSE_GLIMMER_MATERIALIZED_ATTENTION_MAX_POSITIONS + 1;
        let ctx = MetalContext::new().unwrap();
        let query_values =
            vec![0.0; MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM];
        let key_values =
            vec![0.0; POSITIONS * MUSE_GLIMMER_KV_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM];
        let mut value_values = key_values.clone();
        let sentinels = [0, POSITIONS / 2, POSITIONS - 1];
        for kv_head in 0..MUSE_GLIMMER_KV_HEAD_COUNT {
            for dimension in 0..MUSE_GLIMMER_ATTENTION_HEAD_DIM {
                for (ordinal, &position) in sentinels.iter().enumerate() {
                    let index = (position * MUSE_GLIMMER_KV_HEAD_COUNT + kv_head)
                        * MUSE_GLIMMER_ATTENTION_HEAD_DIM
                        + dimension;
                    value_values[index] = (kv_head as f32 + 1.0) * (ordinal as f32 + 1.0)
                        + (dimension % 11) as f32 * 0.03125;
                }
            }
        }
        let query = tensor_from_f32(&ctx, &query_values);
        let key = tensor_from_f16(&ctx, &key_values);
        let value = tensor_from_f16(&ctx, &value_values);
        let output = tensor_from_f32(
            &ctx,
            &vec![f32::NAN; MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM],
        );

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_attn_decode_f16kv_f32(
            &ctx,
            &encoder,
            &query,
            &key,
            &value,
            &output,
            MUSE_GLIMMER_QUERY_HEAD_COUNT,
            MUSE_GLIMMER_KV_HEAD_COUNT,
            MUSE_GLIMMER_ATTENTION_HEAD_DIM,
            POSITIONS,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let output = read_f32(&output);
        assert!(output.iter().all(|value| value.is_finite()));
        for query_head in 0..MUSE_GLIMMER_QUERY_HEAD_COUNT {
            let kv_head = query_head / (MUSE_GLIMMER_QUERY_HEAD_COUNT / MUSE_GLIMMER_KV_HEAD_COUNT);
            for dimension in 0..MUSE_GLIMMER_ATTENTION_HEAD_DIM {
                let sum = sentinels
                    .iter()
                    .map(|&position| {
                        let index = (position * MUSE_GLIMMER_KV_HEAD_COUNT + kv_head)
                            * MUSE_GLIMMER_ATTENTION_HEAD_DIM
                            + dimension;
                        half::f16::from_f32(value_values[index]).to_f32()
                    })
                    .sum::<f32>();
                let expected = sum / POSITIONS as f32;
                let actual = output[query_head * MUSE_GLIMMER_ATTENTION_HEAD_DIM + dimension];
                assert!(
                    (actual - expected).abs() <= 2e-6,
                    "head={query_head} dim={dimension} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    #[ignore = "allocates 128 MiB of KV and scans the full released context"]
    fn muse_glimmer_attn_online_reaches_released_context() {
        const POSITIONS: usize = 131_072;
        let ctx = MetalContext::new().unwrap();
        let query = MetalTensor::zeros_f32(
            &ctx,
            vec![(MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM) as u64],
        )
        .unwrap();
        let cache_elements =
            POSITIONS * MUSE_GLIMMER_KV_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM;
        let key = MetalTensor::zeros_f16(&ctx, vec![cache_elements as u64]).unwrap();
        let value = MetalTensor::zeros_f16(&ctx, vec![cache_elements as u64]).unwrap();
        let sentinels = [0, POSITIONS / 2, POSITIONS - 1];
        unsafe {
            query
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .write_bytes(0, 4096 * 4);
            key.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .write_bytes(0, cache_elements * 2);
            value
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .write_bytes(0, cache_elements * 2);
            let values = value.buffer.contents().as_ptr().cast::<u16>();
            for kv_head in 0..MUSE_GLIMMER_KV_HEAD_COUNT {
                for dimension in 0..MUSE_GLIMMER_ATTENTION_HEAD_DIM {
                    for (ordinal, &position) in sentinels.iter().enumerate() {
                        let index = (position * MUSE_GLIMMER_KV_HEAD_COUNT + kv_head)
                            * MUSE_GLIMMER_ATTENTION_HEAD_DIM
                            + dimension;
                        values.add(index).write(
                            half::f16::from_f32(
                                (kv_head + 1) as f32 * (ordinal + 1) as f32
                                    + (dimension % 7) as f32 * 0.0625,
                            )
                            .to_bits(),
                        );
                    }
                }
            }
        }
        let output = tensor_from_f32(
            &ctx,
            &vec![f32::NAN; MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM],
        );

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_attn_decode_f16kv_f32(
            &ctx,
            &encoder,
            &query,
            &key,
            &value,
            &output,
            MUSE_GLIMMER_QUERY_HEAD_COUNT,
            MUSE_GLIMMER_KV_HEAD_COUNT,
            MUSE_GLIMMER_ATTENTION_HEAD_DIM,
            POSITIONS,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let output = read_f32(&output);
        assert!(output.iter().all(|value| value.is_finite()));
        for query_head in [0, 15, 16, 31] {
            let kv_head = query_head / (MUSE_GLIMMER_QUERY_HEAD_COUNT / MUSE_GLIMMER_KV_HEAD_COUNT);
            for dimension in 0..MUSE_GLIMMER_ATTENTION_HEAD_DIM {
                let expected = sentinels
                    .iter()
                    .enumerate()
                    .map(|(ordinal, _)| {
                        half::f16::from_f32(
                            (kv_head + 1) as f32 * (ordinal + 1) as f32
                                + (dimension % 7) as f32 * 0.0625,
                        )
                        .to_f32()
                    })
                    .sum::<f32>()
                    / POSITIONS as f32;
                let actual = output[query_head * MUSE_GLIMMER_ATTENTION_HEAD_DIM + dimension];
                assert!(
                    (actual - expected).abs() <= 2e-7,
                    "head={query_head} dim={dimension} actual={actual} expected={expected}"
                );
            }
        }
    }

    #[test]
    fn muse_glimmer_attn_rejects_zero_positions() {
        let ctx = MetalContext::new().unwrap();
        let query = MetalTensor::zeros_f32(
            &ctx,
            vec![(MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM) as u64],
        )
        .unwrap();
        let cache = MetalTensor::zeros_f16(&ctx, vec![1]).unwrap();
        let output = MetalTensor::zeros_f32(
            &ctx,
            vec![(MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM) as u64],
        )
        .unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let error = encode_muse_glimmer_attn_decode_f16kv_f32(
            &ctx,
            &encoder,
            &query,
            &cache,
            &cache,
            &output,
            MUSE_GLIMMER_QUERY_HEAD_COUNT,
            MUSE_GLIMMER_KV_HEAD_COUNT,
            MUSE_GLIMMER_ATTENTION_HEAD_DIM,
            0,
        )
        .unwrap_err();
        encoder.end();
        assert!(error.to_string().contains("nonzero"));
    }

    #[test]
    fn adjacent_rope_and_softcap_match_cpu_formulas() {
        let ctx = MetalContext::new().unwrap();
        let query_source = (0..16)
            .map(|value| value as f32 / 7.0 - 1.0)
            .collect::<Vec<_>>();
        let key_source = (0..8)
            .map(|value| value as f32 / 5.0 - 0.5)
            .collect::<Vec<_>>();
        let query = tensor_from_f32(&ctx, &query_source);
        let key = tensor_from_f32(&ctx, &key_source);
        let logits_source = [-100.0, -3.0, 0.0, 4.0, 100.0];
        let logits = tensor_from_f32(&ctx, &logits_source);

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
            &ctx, &encoder, &query, &key, 2, 1, 8, 11, 500_000.0,
        )
        .unwrap();
        encode_muse_glimmer_logit_softcap_f32(&ctx, &encoder, &logits, &logits, 0.196_116_13, 20.0)
            .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let mut expected_query = query_source;
        let mut expected_key = key_source;
        for values in [&mut expected_query[..], &mut expected_key[..]] {
            for head in values.chunks_exact_mut(8) {
                for pair in 0..4 {
                    let relative = pair * 2;
                    let angle = 11.0 * 500_000.0_f32.powf(-(relative as f32) / 8.0);
                    let (sine, cosine) = angle.sin_cos();
                    let first = head[relative];
                    let second = head[relative + 1];
                    head[relative] = first * cosine - second * sine;
                    head[relative + 1] = first * sine + second * cosine;
                }
            }
        }
        for (actual, expected) in read_f32(&query).into_iter().zip(expected_query) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        for (actual, expected) in read_f32(&key).into_iter().zip(expected_key) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        for (actual, raw) in read_f32(&logits).into_iter().zip(logits_source) {
            let expected = 20.0 * (raw * 0.196_116_13 / 20.0).tanh();
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn packed_adjacent_rope_matches_scalar_rows_bitwise() {
        const ROWS: usize = 5;
        const QUERY_WIDTH: usize = MUSE_GLIMMER_QUERY_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM;
        const KEY_WIDTH: usize = MUSE_GLIMMER_KV_HEAD_COUNT * MUSE_GLIMMER_ATTENTION_HEAD_DIM;
        const THETA: f32 = 500_000.0;

        for base_position in [0, 2_046] {
            let ctx = MetalContext::new().unwrap();
            let query_source = (0..ROWS * QUERY_WIDTH)
                .map(|index| ((index * 17 % 101) as f32 - 50.0) * 0.002)
                .collect::<Vec<_>>();
            let key_source = (0..ROWS * KEY_WIDTH)
                .map(|index| ((index * 13 % 89) as f32 - 44.0) * 0.003)
                .collect::<Vec<_>>();
            let packed_query =
                tensor_from_f32_shape(&ctx, &query_source, vec![QUERY_WIDTH as u64, ROWS as u64]);
            let packed_key =
                tensor_from_f32_shape(&ctx, &key_source, vec![KEY_WIDTH as u64, ROWS as u64]);
            let scalar_query =
                tensor_from_f32_shape(&ctx, &query_source, vec![QUERY_WIDTH as u64, ROWS as u64]);
            let scalar_key =
                tensor_from_f32_shape(&ctx, &key_source, vec![KEY_WIDTH as u64, ROWS as u64]);

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_muse_glimmer_rope_adjacent_pair_rows_in_place_f32(
                &ctx,
                &encoder,
                &packed_query,
                &packed_key,
                MUSE_GLIMMER_QUERY_HEAD_COUNT,
                MUSE_GLIMMER_KV_HEAD_COUNT,
                MUSE_GLIMMER_ATTENTION_HEAD_DIM,
                ROWS,
                base_position,
                THETA,
            )
            .unwrap();
            for row in 0..ROWS {
                encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
                    &ctx,
                    &encoder,
                    &scalar_query
                        .view_subrange((row * QUERY_WIDTH) as u64, vec![QUERY_WIDTH as u64]),
                    &scalar_key.view_subrange((row * KEY_WIDTH) as u64, vec![KEY_WIDTH as u64]),
                    MUSE_GLIMMER_QUERY_HEAD_COUNT,
                    MUSE_GLIMMER_KV_HEAD_COUNT,
                    MUSE_GLIMMER_ATTENTION_HEAD_DIM,
                    (base_position + row) as u32,
                    THETA,
                )
                .unwrap();
            }
            encoder.end();
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(
                command.status(),
                objc2_metal::MTLCommandBufferStatus::Completed
            );

            assert_eq!(
                read_f32(&packed_query)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                read_f32(&scalar_query)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "query base={base_position}"
            );
            assert_eq!(
                read_f32(&packed_key)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                read_f32(&scalar_key)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>(),
                "key base={base_position}"
            );
        }
    }

    #[test]
    fn periodic_adjacent_rope_matches_cpu_and_round_trips() {
        const B: usize = 3;
        const T: usize = 5;
        const QH: usize = 2;
        const KH: usize = 1;
        const D: usize = 8;
        const THETA: f32 = 500_000.0;
        let ctx = MetalContext::new().unwrap();
        let rows = B * T;
        let query_source = (0..rows * QH * D)
            .map(|index| (index % 97) as f32 * 0.013 - 0.4)
            .collect::<Vec<_>>();
        let key_source = (0..rows * KH * D)
            .map(|index| (index % 71) as f32 * -0.017 + 0.3)
            .collect::<Vec<_>>();
        let query = tensor_from_f32_shape(&ctx, &query_source, vec![(QH * D) as u64, rows as u64]);
        let key = tensor_from_f32_shape(&ctx, &key_source, vec![(KH * D) as u64, rows as u64]);
        let rotate_cpu = |values: &mut [f32], heads: usize, inverse: bool| {
            for row in 0..rows {
                let position = row % T;
                for head in 0..heads {
                    let base = (row * heads + head) * D;
                    for pair in 0..D / 2 {
                        let relative = pair * 2;
                        let angle = position as f32 * THETA.powf(-(relative as f32) / D as f32);
                        let (mut sine, cosine) = angle.sin_cos();
                        if inverse {
                            sine = -sine;
                        }
                        let first = values[base + relative];
                        let second = values[base + relative + 1];
                        values[base + relative] = first * cosine - second * sine;
                        values[base + relative + 1] = first * sine + second * cosine;
                    }
                }
            }
        };

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32(
            &ctx, &encoder, &query, &key, QH, KH, D, rows, T, THETA, false,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();

        let mut expected_query = query_source.clone();
        let mut expected_key = key_source.clone();
        rotate_cpu(&mut expected_query, QH, false);
        rotate_cpu(&mut expected_key, KH, false);
        for (actual, expected) in read_f32(&query).into_iter().zip(expected_query) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }
        for (actual, expected) in read_f32(&key).into_iter().zip(expected_key) {
            assert!((actual - expected).abs() < 2e-5, "{actual} != {expected}");
        }

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_muse_glimmer_rope_adjacent_pair_periodic_in_place_f32(
            &ctx, &encoder, &query, &key, QH, KH, D, rows, T, THETA, true,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        for (actual, expected) in read_f32(&query).into_iter().zip(query_source) {
            assert!((actual - expected).abs() < 4e-5, "{actual} != {expected}");
        }
        for (actual, expected) in read_f32(&key).into_iter().zip(key_source) {
            assert!((actual - expected).abs() < 4e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn causal_gqa_vjp_bank_rejects_zero_width_and_malformed_layout() {
        let ctx = MetalContext::new().unwrap();
        const B: usize = 1;
        const T: usize = 3;
        const QH: usize = 4;
        const KVH: usize = 2;
        const D: usize = 32;
        let query_width = QH * D;
        let kv_width = KVH * D;
        let query = MetalTensor::zeros_f32(&ctx, vec![query_width as u64, T as u64]).unwrap();
        let key = MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, T as u64]).unwrap();
        let value = MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, T as u64]).unwrap();
        let gate = MetalTensor::zeros_f32(&ctx, vec![query_width as u64, T as u64]).unwrap();
        let attention = MetalTensor::zeros_f32(&ctx, vec![query_width as u64, T as u64]).unwrap();
        let probabilities =
            MetalTensor::zeros_f32(&ctx, vec![T as u64, QH as u64, T as u64]).unwrap();
        let grad_gated =
            MetalTensor::zeros_f32(&ctx, vec![query_width as u64, (B * T) as u64]).unwrap();
        let grad_query =
            MetalTensor::zeros_f32(&ctx, vec![query_width as u64, (B * T) as u64]).unwrap();
        let partial_shape = vec![D as u64, T as u64, QH as u64, B as u64];
        let partial_key = MetalTensor::zeros_f32(&ctx, partial_shape.clone()).unwrap();
        let partial_value = MetalTensor::zeros_f32(&ctx, partial_shape).unwrap();
        let grad_key = MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, (B * T) as u64]).unwrap();
        let grad_value =
            MetalTensor::zeros_f32(&ctx, vec![kv_width as u64, (B * T) as u64]).unwrap();
        let grad_gate =
            MetalTensor::zeros_f32(&ctx, vec![query_width as u64, (B * T) as u64]).unwrap();

        let reject = |query: &MetalTensor, head_dim: usize| {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let error = encode_muse_glimmer_causal_gqa_vjp_bank_f32(
                &ctx,
                &encoder,
                query,
                &key,
                &value,
                &gate,
                &attention,
                &probabilities,
                &grad_gated,
                &grad_query,
                &partial_key,
                &partial_value,
                &grad_key,
                &grad_value,
                &grad_gate,
                B,
                T,
                QH,
                KVH,
                head_dim,
            )
            .unwrap_err();
            encoder.end();
            error.to_string()
        };
        assert!(reject(&query, 0).contains("head_dim"));
        let mut malformed_query = query.clone();
        malformed_query.shape = vec![(query_width * T) as u64];
        assert!(reject(&malformed_query, D).contains("query shape"));
    }
}
