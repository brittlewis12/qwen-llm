//! Shared-latent (MLA) attention over a linear F16 cache whose rows are both
//! key and value, as in GLM-5.3-Flash's absorbed NoPE MLA.
//!
//! For each of `rows` consecutive query rows at absolute positions
//! `start + r`, every head attends causally to cache rows `0..=start + r`:
//! `o[h] = softmax_j(q[h] . c_j * scale) . c`, accumulated in F32. Sinks enter
//! the softmax as one extra logit per head; pass [`LATENT_NO_SINK`] for none
//! (a large finite value: the metallib builds with fast math). Reuses DS4's
//! grouped online kernel (64 heads x 512, eight heads share staged rows) with
//! the ring window widened to the cache capacity and no compressed rows.

use super::checks::{bad_shape, check_alignment, check_disjoint, check_tensor, require_serial};
use super::*;

/// Sink logit that contributes negligible softmax mass whenever some visible
/// row scores far above it (always true for real scores, which are O(10));
/// finite because the metallib builds with fast math.
pub const LATENT_NO_SINK: f32 = -1.0e30;
pub const LATENT_HEADS: usize = 64;
pub const LATENT_WIDTH: usize = 512;

/// `queries` and `output` are F32 `[512, 64, rows]`; `cache` is F16 `[512,
/// capacity]` with rows `0..start + rows` written; `sinks` is F32 `[64]`.
#[allow(clippy::too_many_arguments)]
pub fn encode_latent_attention(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    queries: &MetalTensor,
    cache: &MetalTensor,
    sinks: &MetalTensor,
    output: &MetalTensor,
    start: usize,
    rows: usize,
    scale: f32,
) -> Result<(), MetalError> {
    const K: &str = "latent_attention";
    require_serial(K, enc)?;
    if rows == 0 || !scale.is_finite() || scale <= 0.0 {
        return Err(bad_shape(
            K,
            "rows must be positive and scale finite and positive",
        ));
    }
    let (heads, width) = (LATENT_HEADS as u64, LATENT_WIDTH as u64);
    let shape = [width, heads, rows as u64];
    check_tensor(K, queries, GgmlType::F32, &shape, false, "queries")?;
    check_tensor(K, output, GgmlType::F32, &shape, true, "output")?;
    check_alignment(K, queries, 16, "queries")?;
    check_alignment(K, output, 16, "output")?;
    if cache.dtype != GgmlType::F16 || cache.shape.len() != 2 || cache.shape[0] != width {
        return Err(bad_shape(
            K,
            format!(
                "cache must be F16 [512, capacity], got {:?} {:?}",
                cache.dtype, cache.shape
            ),
        ));
    }
    let capacity = cache.shape[1];
    check_tensor(K, cache, GgmlType::F16, &[width, capacity], false, "cache")?;
    check_alignment(K, cache, 16, "cache")?;
    let end = start
        .checked_add(rows)
        .filter(|&end| end as u64 <= capacity && u32::try_from(end).is_ok())
        .ok_or_else(|| {
            bad_shape(
                K,
                format!("positions {start}..{start}+{rows} exceed capacity {capacity}"),
            )
        })?;
    let _ = end;
    check_tensor(K, sinks, GgmlType::F32, &[heads], false, "sinks")?;
    check_disjoint(
        K,
        output,
        &[(queries, "queries"), (cache, "cache"), (sinks, "sinks")],
    )?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        n_tokens: u32,
        compression_ratio: u32,
        start_position: u32,
        window: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_grouped_online_dense_sink_attention_f16")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 256 {
        return Err(bad_shape(K, "needs 32-lane simdgroups and 256 threads"));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: heads as u32,
            head_dim: width as u32,
            n_tokens: rows as u32,
            compression_ratio: 0,
            start_position: start as u32,
            // A window covering the whole cache makes ring addressing linear.
            window: u32::try_from(capacity).map_err(|_| bad_shape(K, "capacity exceeds u32"))?,
            raw_cache_is_chunk: 0,
            scale,
        },
    );
    enc.set_tensor(1, queries);
    enc.set_tensor(2, cache);
    enc.set_tensor(3, cache);
    enc.set_tensor(4, cache);
    enc.set_tensor(5, sinks);
    enc.set_tensor(6, output);
    // 16 staged rows of 512 halves.
    enc.set_threadgroup_memory(0, 16 * LATENT_WIDTH * 2);
    enc.dispatch(
        MTLSize {
            width: rows,
            height: LATENT_HEADS / 8,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

/// Buffers for [`encode_online_selected_attention_f16`].
pub struct SelectedAttention<'a> {
    /// F32 `[512 * heads, token_count]`: query rows of every chunk token.
    pub queries: &'a MetalTensor,
    /// F16 raw-window cache (ring of `window` rows, or the chunk's rows when
    /// `raw_cache_is_chunk`); unused when `window == 0` (bind any F16 cache).
    pub raw_cache: &'a MetalTensor,
    /// F16 ring holding rows before the chunk (chunk layout only).
    pub raw_cache_before_chunk: &'a MetalTensor,
    /// F16 `[512, compressed_capacity]`: rows addressed by `selected_ids`.
    pub compressed_cache: &'a MetalTensor,
    /// I32 `[selected_slots, query_count]`; ids below 0 or at/after a query's
    /// visible count are skipped.
    pub selected_ids: &'a MetalTensor,
    /// I32 `[query_count]`: slots in use per query.
    pub selected_counts: &'a MetalTensor,
    /// I32 `[query_count]`: compressed rows visible to each query.
    pub visible_counts: &'a MetalTensor,
    /// F32 `[heads]`: one sink logit per head ([`LATENT_NO_SINK`]: none).
    pub sinks: &'a MetalTensor,
    /// F32 like `queries`; only the queried tokens' rows are written.
    pub output: &'a MetalTensor,
}

/// Geometry for [`encode_online_selected_attention_f16`].
#[derive(Clone, Copy, Debug)]
pub struct SelectedAttentionShape {
    pub head_count: usize,
    pub query_count: usize,
    /// First queried token within the chunk's `token_count` query rows.
    pub query_token_offset: usize,
    pub token_count: usize,
    pub chunk_start_position: usize,
    /// Most recent raw rows each query attends before its selected rows.
    pub window: usize,
    pub raw_cache_is_chunk: bool,
    pub selected_slots: usize,
    pub compressed_capacity: usize,
    pub scale: f32,
    /// Read rows straight from device memory instead of staging them.
    pub direct: bool,
}

/// The buffer and geometry contract shared by the serial and split
/// selected-attention encoders.
fn validate_selected_attention(
    b: &SelectedAttention<'_>,
    g: SelectedAttentionShape,
) -> Result<(), MetalError> {
    const K: &str = "online_selected_attention";
    let overflow = || bad_shape(K, "geometry exceeds 32-bit shader offsets");
    let width = LATENT_WIDTH;
    let query_width = g.head_count.checked_mul(width).ok_or_else(overflow)?;
    let query_end = g
        .query_token_offset
        .checked_add(g.query_count)
        .filter(|&end| end <= g.token_count)
        .ok_or_else(|| bad_shape(K, "queried tokens exceed the chunk"))?;
    if g.head_count == 0
        || g.query_count == 0
        || g.selected_slots == 0
        || g.compressed_capacity == 0
        || !g.scale.is_finite()
        || g.scale <= 0.0
    {
        return Err(bad_shape(K, "empty geometry or invalid scale"));
    }
    // Every shader offset below must fit 32 bits.
    let fits = |v: Option<usize>| v.and_then(|v| u32::try_from(v).ok()).is_some();
    if !fits(query_width.checked_mul(g.token_count))
        || !fits(g.compressed_capacity.checked_mul(width))
        || !fits(g.selected_slots.checked_mul(g.query_count))
        || !fits(g.chunk_start_position.checked_add(query_end))
        || !fits(g.window.checked_mul(width))
    {
        return Err(overflow());
    }
    let (q_shape, tokens) = (query_width as u64, g.token_count as u64);
    check_tensor(
        K,
        b.queries,
        GgmlType::F32,
        &[q_shape, tokens],
        false,
        "queries",
    )?;
    check_tensor(
        K,
        b.output,
        GgmlType::F32,
        &[q_shape, tokens],
        true,
        "output",
    )?;
    check_alignment(K, b.queries, 16, "queries")?;
    check_alignment(K, b.output, 16, "output")?;
    let rows = [width as u64, g.compressed_capacity as u64];
    check_tensor(
        K,
        b.compressed_cache,
        GgmlType::F16,
        &rows,
        false,
        "compressed cache",
    )?;
    check_alignment(K, b.compressed_cache, 8, "compressed cache")?;
    for (cache, name) in [
        (b.raw_cache, "raw cache"),
        (b.raw_cache_before_chunk, "raw cache before chunk"),
    ] {
        let elements = cache.n_elements();
        if cache.dtype != GgmlType::F16 || !elements.is_multiple_of(width as u64) {
            return Err(bad_shape(K, format!("{name} must be F16 rows of 512")));
        }
        check_tensor(K, cache, GgmlType::F16, &cache.shape.clone(), false, name)?;
        check_alignment(K, cache, 8, name)?;
    }
    // Raw rows the kernel can address: the ring (`position % window`), or in
    // chunk layout the chunk's rows plus the ring of rows before it.
    if g.window > 0 {
        let rows = |t: &MetalTensor| t.n_elements() / width as u64;
        let (window, tokens) = (g.window as u64, g.token_count as u64);
        let short = |name: &str, needed: u64, have: u64| {
            bad_shape(
                K,
                format!("{name} holds {have} rows; window {window} needs {needed}"),
            )
        };
        if g.raw_cache_is_chunk {
            if rows(b.raw_cache) < tokens {
                return Err(short("raw chunk cache", tokens, rows(b.raw_cache)));
            }
            if rows(b.raw_cache_before_chunk) < window {
                let have = rows(b.raw_cache_before_chunk);
                return Err(short("raw cache before chunk", window, have));
            }
        } else if rows(b.raw_cache) < window {
            return Err(short("raw ring", window, rows(b.raw_cache)));
        }
    }
    let (slots, queries) = (g.selected_slots as u64, g.query_count as u64);
    check_tensor(
        K,
        b.selected_ids,
        GgmlType::I32,
        &[slots, queries],
        false,
        "selected ids",
    )?;
    check_tensor(
        K,
        b.selected_counts,
        GgmlType::I32,
        &[queries],
        false,
        "selected counts",
    )?;
    check_tensor(
        K,
        b.visible_counts,
        GgmlType::I32,
        &[queries],
        false,
        "visible counts",
    )?;
    check_tensor(
        K,
        b.sinks,
        GgmlType::F32,
        &[g.head_count as u64],
        false,
        "sinks",
    )?;
    check_disjoint(
        K,
        b.output,
        &[
            (b.queries, "queries"),
            (b.raw_cache, "raw cache"),
            (b.raw_cache_before_chunk, "raw cache before chunk"),
            (b.compressed_cache, "compressed cache"),
            (b.selected_ids, "selected ids"),
            (b.selected_counts, "selected counts"),
            (b.visible_counts, "visible counts"),
            (b.sinks, "sinks"),
        ],
    )?;
    Ok(())
}

/// Online-softmax attention of each query over its raw window (the newest
/// `window` positions) and then its selected compressed rows, in slot order,
/// one simdgroup per (query, head), head width 512, accumulated in F32. The
/// raw-cache layout is the caller's contract (DS4 validates it before
/// delegating); GLM passes `window = 0` and its latent cache as the
/// compressed cache.
pub fn encode_online_selected_attention_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &SelectedAttention<'_>,
    g: SelectedAttentionShape,
) -> Result<(), MetalError> {
    const K: &str = "online_selected_attention";
    require_serial(K, enc)?;
    validate_selected_attention(b, g)?;
    let width = LATENT_WIDTH;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        query_count: u32,
        query_token_offset: u32,
        chunk_start_position: u32,
        window: u32,
        selected_slots: u32,
        compressed_capacity: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
    }
    let (kernel, staged_bytes) = if g.direct {
        (
            "kernel_deepseek_v4_online_packed_selected_sink_attention_f16_direct",
            0,
        )
    } else {
        (
            "kernel_deepseek_v4_online_packed_selected_sink_attention_f16",
            width * 2,
        )
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < 32
        || ctx.device.maxThreadgroupMemoryLength() < staged_bytes
    {
        return Err(bad_shape(
            K,
            "needs one 32-lane simdgroup per (query, head)",
        ));
    }
    let u = |v: usize, name: &str| super::checks::to_u32(K, v, name);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            head_count: u(g.head_count, "heads")?,
            head_dim: u(width, "head dim")?,
            query_count: u(g.query_count, "queries")?,
            query_token_offset: u(g.query_token_offset, "query offset")?,
            chunk_start_position: u(g.chunk_start_position, "chunk start")?,
            window: u(g.window, "window")?,
            selected_slots: u(g.selected_slots, "selected slots")?,
            compressed_capacity: u(g.compressed_capacity, "compressed capacity")?,
            raw_cache_is_chunk: u32::from(g.raw_cache_is_chunk),
            scale: g.scale,
        },
    );
    enc.set_tensor(1, b.queries);
    enc.set_tensor(2, b.raw_cache);
    enc.set_tensor(3, b.raw_cache_before_chunk);
    enc.set_tensor(4, b.compressed_cache);
    enc.set_tensor(5, b.selected_ids);
    enc.set_tensor(6, b.selected_counts);
    enc.set_tensor(7, b.visible_counts);
    enc.set_tensor(8, b.sinks);
    enc.set_tensor(9, b.output);
    if staged_bytes != 0 {
        enc.set_threadgroup_memory(0, staged_bytes);
    }
    enc.dispatch(
        MTLSize {
            width: g.query_count,
            height: g.head_count,
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

/// Rows per split of [`encode_online_selected_attention_split_f16`].
pub const SELECTED_SPLIT_ROWS: usize = 128;

/// Splits per (query, head): its raw window plus selected slots, in ranges
/// of [`SELECTED_SPLIT_ROWS`].
pub fn selected_attention_splits(window: usize, selected_slots: usize) -> usize {
    window
        .saturating_add(selected_slots)
        .div_ceil(SELECTED_SPLIT_ROWS)
        .max(1)
}

/// Scratch for [`encode_online_selected_attention_split_f16`]: F32
/// `[512, units]` unnormalized accumulators and F32 `[2, units]`
/// (maximum, denominator) pairs, `units = queries * heads * splits`.
pub struct SelectedAttentionPartials<'a> {
    pub values: &'a MetalTensor,
    pub stats: &'a MetalTensor,
}

/// [`encode_online_selected_attention_f16`] with each query's rows (raw
/// window, then selected slots) cut into ranges of [`SELECTED_SPLIT_ROWS`]:
/// one simdgroup per (query, head, split) runs the online update over its
/// range, then one per (query, head) folds the splits in order into the
/// sink's initial state. Many more simdgroups for a single query (decode)
/// at the cost of a different summation order than the serial kernel
/// (numerical, not bitwise). The partition depends only on each query's
/// own geometry, so a query's result does not depend on how many queries
/// share the dispatch. Direct row reads only.
pub fn encode_online_selected_attention_split_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &SelectedAttention<'_>,
    g: SelectedAttentionShape,
    partials: &SelectedAttentionPartials<'_>,
) -> Result<(), MetalError> {
    const K: &str = "online_selected_attention_split";
    require_serial(K, enc)?;
    validate_selected_attention(b, g)?;
    if !g.direct {
        return Err(bad_shape(K, "the split kernel reads rows directly"));
    }
    let width = LATENT_WIDTH;
    let splits = selected_attention_splits(g.window, g.selected_slots);
    let units = g
        .query_count
        .checked_mul(g.head_count)
        .and_then(|v| v.checked_mul(splits))
        .filter(|&v| {
            v.checked_mul(width)
                .and_then(|v| u32::try_from(v).ok())
                .is_some()
        })
        .ok_or_else(|| bad_shape(K, "partials exceed 32-bit shader offsets"))?;
    check_tensor(
        K,
        partials.values,
        GgmlType::F32,
        &[width as u64, units as u64],
        true,
        "partial values",
    )?;
    check_tensor(
        K,
        partials.stats,
        GgmlType::F32,
        &[2, units as u64],
        true,
        "partial stats",
    )?;
    check_alignment(K, partials.values, 16, "partial values")?;
    check_alignment(K, partials.stats, 8, "partial stats")?;
    for (partial, name) in [
        (partials.values, "partial values"),
        (partials.stats, "partial stats"),
    ] {
        check_disjoint(
            K,
            partial,
            &[
                (b.queries, "queries"),
                (b.raw_cache, "raw cache"),
                (b.raw_cache_before_chunk, "raw cache before chunk"),
                (b.compressed_cache, "compressed cache"),
                (b.selected_ids, "selected ids"),
                (b.selected_counts, "selected counts"),
                (b.visible_counts, "visible counts"),
                (b.sinks, "sinks"),
                (b.output, "output"),
            ],
        )
        .map_err(|e| bad_shape(K, format!("{name}: {e}")))?;
    }
    check_disjoint(K, partials.values, &[(partials.stats, "partial stats")])?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        head_count: u32,
        head_dim: u32,
        query_count: u32,
        query_token_offset: u32,
        chunk_start_position: u32,
        window: u32,
        selected_slots: u32,
        compressed_capacity: u32,
        raw_cache_is_chunk: u32,
        scale: f32,
        split_rows: u32,
        splits: u32,
    }
    let u = |v: usize, name: &str| super::checks::to_u32(K, v, name);
    let args = Args {
        head_count: u(g.head_count, "heads")?,
        head_dim: u(width, "head dim")?,
        query_count: u(g.query_count, "queries")?,
        query_token_offset: u(g.query_token_offset, "query offset")?,
        chunk_start_position: u(g.chunk_start_position, "chunk start")?,
        window: u(g.window, "window")?,
        selected_slots: u(g.selected_slots, "selected slots")?,
        compressed_capacity: u(g.compressed_capacity, "compressed capacity")?,
        raw_cache_is_chunk: u32::from(g.raw_cache_is_chunk),
        scale: g.scale,
        split_rows: u(SELECTED_SPLIT_ROWS, "split rows")?,
        splits: u(splits, "splits")?,
    };
    let simdgroup = MTLSize {
        width: 32,
        height: 1,
        depth: 1,
    };
    let partial = ctx.pipeline("kernel_online_selected_attention_f16_split_partial")?;
    let merge = ctx.pipeline("kernel_online_selected_attention_f16_split_merge")?;
    for pso in [&partial, &merge] {
        if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 32 {
            return Err(bad_shape(K, "needs one 32-lane simdgroup per work unit"));
        }
    }
    enc.set_pipeline(&partial);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, b.queries);
    enc.set_tensor(2, b.raw_cache);
    enc.set_tensor(3, b.raw_cache_before_chunk);
    enc.set_tensor(4, b.compressed_cache);
    enc.set_tensor(5, b.selected_ids);
    enc.set_tensor(6, b.selected_counts);
    enc.set_tensor(7, b.visible_counts);
    enc.set_tensor(8, partials.values);
    enc.set_tensor(9, partials.stats);
    enc.dispatch(
        MTLSize {
            width: g.query_count,
            height: g.head_count,
            depth: splits,
        },
        simdgroup,
    );
    enc.set_pipeline(&merge);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, partials.values);
    enc.set_tensor(2, partials.stats);
    enc.set_tensor(3, b.sinks);
    enc.set_tensor(4, b.output);
    enc.dispatch(
        MTLSize {
            width: g.query_count,
            height: g.head_count,
            depth: 1,
        },
        simdgroup,
    );
    Ok(())
}

/// Group-axis Q8_0 mat-mat over `rows` tokens with F32 activations and F32
/// simdgroup accumulation: for each token and group `g`,
/// `output[g * n_out..][..n_out] = W_g * input[g * n_in..][..n_in]`, with
/// weights `[n_in, n_out, groups]` (one contiguous matrix per group) and
/// token-major activations `[n_in * groups, rows]` / `[n_out * groups, rows]`.
/// Requires `n_in % 64 == 0`, `n_out % 16 == 0` and `rows % 128 == 0`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_q8_0_grouped_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
) -> Result<(), MetalError> {
    const K: &str = "mat_mat_q8_0_grouped";
    require_serial(K, enc)?;
    if weight.dtype != GgmlType::Q8_0
        || groups == 0
        || rows == 0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !rows.is_multiple_of(128)
    {
        return Err(bad_shape(
            K,
            format!(
                "needs Q8_0, n_in % 64, n_out % 16 and rows % 128; got {:?} {n_in} {n_out} {rows}",
                weight.dtype
            ),
        ));
    }
    let overflow = || bad_shape(K, "dimension product overflows");
    let in_width = n_in.checked_mul(groups).ok_or_else(overflow)?;
    let out_width = n_out.checked_mul(groups).ok_or_else(overflow)?;
    // `groups` contiguous [n_in, n_out] Q8_0 matrices, viewed per group
    // ([n_in, n_out, groups]) or flat ([n_in, n_out * groups]), inside the
    // weight's buffer. The kernel reads each block's half scale through a
    // typed pointer, so the base must be 2-byte aligned.
    let (block, block_bytes) = GgmlType::Q8_0.storage_layout().ok_or_else(overflow)?;
    let shapes = [
        [n_in as u64, n_out as u64, groups as u64].to_vec(),
        [n_in as u64, out_width as u64].to_vec(),
    ];
    if !shapes.contains(&weight.shape) {
        return Err(bad_shape(
            K,
            format!(
                "weight must be {:?} or {:?}, got {:?}",
                shapes[0], shapes[1], weight.shape
            ),
        ));
    }
    let bytes = (n_in as u64 / block)
        .checked_mul(block_bytes)
        .and_then(|row| row.checked_mul(out_width as u64))
        .ok_or_else(overflow)?;
    let end = weight.offset.checked_add(bytes);
    if weight.n_bytes() != bytes || end.is_none_or(|end| end > weight.buffer.length() as u64) {
        return Err(bad_shape(
            K,
            format!("weight is not {groups} contiguous Q8_0 matrices inside its buffer"),
        ));
    }
    check_alignment(K, weight, 2, "weight")?;
    check_tensor(
        K,
        input,
        GgmlType::F32,
        &[in_width as u64, rows as u64],
        false,
        "input",
    )?;
    check_tensor(
        K,
        output,
        GgmlType::F32,
        &[out_width as u64, rows as u64],
        true,
        "output",
    )?;
    check_disjoint(K, output, &[(input, "input"), (weight, "weight")])?;
    let pso = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c16k64_grouped")?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 128 {
        return Err(bad_shape(K, "needs four 32-lane simdgroups"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        groups: u32,
        nb01: u32,
        stride_b: u32,
        stride_c: u32,
    }
    let u = |v: usize, name: &str| super::checks::to_u32(K, v, name);
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            m: u(n_out, "n_out")?,
            n: u(rows, "rows")?,
            k: u(n_in, "n_in")?,
            groups: u(groups, "groups")?,
            nb01: u(n_in / 32 * 34, "row bytes")?,
            stride_b: u(in_width, "input width")?,
            stride_c: u(out_width, "output width")?,
        },
    );
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: rows / 128,
            height: n_out / 16,
            depth: groups,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::test_support::{
        max_abs_diff_finite, offset_tensor, synthetic_q8_0_bank, tensor_f32_at_offset,
    };
    use super::*;

    const H: usize = LATENT_HEADS;
    const W: usize = LATENT_WIDTH;

    fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        offset_tensor(
            ctx,
            32,
            bytemuck::cast_slice(values),
            32,
            shape,
            GgmlType::F32,
        )
    }

    fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode(&encoder);
        encoder.end();
        command.commit();
        wait_completed(&command).expect("command buffer failed");
    }

    fn reference(
        q: &[f32],
        cache: &[half::f16],
        start: usize,
        rows: usize,
        scale: f32,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; W * H * rows];
        for r in 0..rows {
            let visible = start + r + 1;
            for h in 0..H {
                let qh = &q[(r * H + h) * W..][..W];
                let scores: Vec<f64> = (0..visible)
                    .map(|j| {
                        qh.iter()
                            .zip(&cache[j * W..(j + 1) * W])
                            .map(|(q, c)| *q as f64 * c.to_f64())
                            .sum::<f64>()
                            * scale as f64
                    })
                    .collect();
                let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                let total: f64 = weights.iter().sum();
                for d in 0..W {
                    let o: f64 = (0..visible)
                        .map(|j| weights[j] * cache[j * W + d].to_f64())
                        .sum();
                    out[(r * H + h) * W + d] = (o / total) as f32;
                }
            }
        }
        out
    }

    /// Visible lengths covering staging tails (16-row tiles) and the dense
    /// frontier, distinct heads, poisoned rows past the visible range, and a
    /// packed causal block.
    #[test]
    fn latent_attention_matches_reference_without_sinks() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const CAPACITY: usize = 2064;
        let scale = 1.0 / 16.0;
        let mut cache: Vec<half::f16> = (0..CAPACITY * W)
            .map(|i| half::f16::from_f32((((i * 37 + i / 513) % 199) as f32 - 99.0) * 0.011))
            .collect();
        let sinks = f32_tensor(&ctx, &[LATENT_NO_SINK; H], vec![H as u64]);
        for (start, rows) in [
            (0, 1),
            (14, 1),
            (15, 1),
            (16, 1),
            (99, 1),
            (2050, 1),
            (12, 3),
        ] {
            // Rows past the last visible position are NaN; reading them would poison.
            let visible_end = start + rows;
            for value in &mut cache[visible_end * W..] {
                *value = half::f16::NAN;
            }
            let q: Vec<f32> = (0..W * H * rows)
                .map(|i| (((i * 29 + (i / W) * 131) % 97) as f32 - 48.0) * 0.02)
                .collect();
            let cache_t = offset_tensor(
                &ctx,
                64,
                bytemuck::cast_slice(&cache),
                64,
                vec![W as u64, CAPACITY as u64],
                GgmlType::F16,
            );
            let q_t = f32_tensor(&ctx, &q, vec![W as u64, H as u64, rows as u64]);
            let out_t = f32_tensor(
                &ctx,
                &vec![0.0; W * H * rows],
                vec![W as u64, H as u64, rows as u64],
            );
            run(&ctx, |enc| {
                encode_latent_attention(
                    &ctx, enc, &q_t, &cache_t, &sinks, &out_t, start, rows, scale,
                )
                .unwrap();
            });
            let expected = reference(&q, &cache, start, rows, scale);
            let actual = tensor_f32_at_offset(&out_t);
            let worst = max_abs_diff_finite("latent attention", &actual, &expected);
            assert!(
                actual.iter().all(|v| v.is_finite()),
                "start {start}: non-finite"
            );
            assert!(worst <= 2e-4, "start {start} rows {rows}: max abs {worst}");
            // Restore poisoned rows with finite data for the next case.
            for (i, value) in cache.iter_mut().enumerate().skip(visible_end * W) {
                *value = half::f16::from_f32((((i * 37 + i / 513) % 199) as f32 - 99.0) * 0.011);
            }
        }
        // One visible row returns that row (rounded to F16) for every head.
        let one = reference(&vec![1.0; W * H], &cache, 0, 1, scale);
        assert!(one.chunks(W).all(|h| {
            h.iter()
                .zip(&cache[..W])
                .all(|(o, c)| (o - c.to_f32()).abs() < 1e-6)
        }));
    }

    /// GLM-5.3 absorption shapes through the existing grouped Q8_0 GEMV:
    /// `attn_k_b` (256 -> 512 per head) and `attn_v_b` (512 -> 256 per head).
    #[test]
    fn grouped_q8_absorption_matches_reference_at_glm_shapes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        for (n_in, n_out) in [(256usize, 512usize), (512, 256)] {
            let (bytes, decoded) = synthetic_q8_0_bank(n_in, n_out * H);
            let weight = offset_tensor(
                &ctx,
                256,
                &bytes,
                64,
                vec![n_in as u64, n_out as u64, H as u64],
                GgmlType::Q8_0,
            );
            let x: Vec<f32> = (0..n_in * H)
                .map(|i| (((i * 17 + (i / n_in) * 53) % 89) as f32 - 44.0) * 0.03)
                .collect();
            let x_t = f32_tensor(&ctx, &x, vec![n_in as u64, H as u64]);
            let y_t = f32_tensor(&ctx, &vec![0.0; n_out * H], vec![n_out as u64, H as u64]);
            run(&ctx, |enc| {
                encode_mat_vec_q8_0_grouped_f32(&ctx, enc, &weight, &x_t, &y_t, n_in, n_out, H)
                    .unwrap();
            });
            let actual = tensor_f32_at_offset(&y_t);
            for h in 0..H {
                for r in 0..n_out {
                    let row = &decoded[(h * n_out + r) * n_in..][..n_in];
                    let expected: f64 = row
                        .iter()
                        .zip(&x[h * n_in..(h + 1) * n_in])
                        .map(|(w, v)| *w as f64 * *v as f64)
                        .sum();
                    let got = actual[h * n_out + r] as f64;
                    assert!(
                        (got - expected).abs() <= 1e-4 * (1.0 + expected.abs()),
                        "{n_in}->{n_out} head {h} row {r}: {got} vs {expected}"
                    );
                }
            }
        }
    }

    /// Grouped Q8_0 mat-mat over 128 rows equals the per-row grouped GEMV at
    /// both GLM absorption shapes (F32 accumulation, different order).
    #[test]
    fn grouped_q8_mat_mat_matches_per_row_gemv_at_glm_shapes() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 128;
        for (n_in, n_out) in [(256usize, 512usize), (512, 256)] {
            let (bytes, _) = synthetic_q8_0_bank(n_in, n_out * H);
            let weight = offset_tensor(
                &ctx,
                256,
                &bytes,
                64,
                vec![n_in as u64, n_out as u64, H as u64],
                GgmlType::Q8_0,
            );
            let x: Vec<f32> = (0..n_in * H * ROWS)
                .map(|i| (((i * 17 + (i / n_in) * 53) % 89) as f32 - 44.0) * 0.03)
                .collect();
            let x_t = f32_tensor(&ctx, &x, vec![(n_in * H) as u64, ROWS as u64]);
            let y_t = f32_tensor(
                &ctx,
                &vec![0.0; n_out * H * ROWS],
                vec![(n_out * H) as u64, ROWS as u64],
            );
            run(&ctx, |enc| {
                encode_mat_mat_q8_0_grouped_f32(
                    &ctx, enc, &weight, &x_t, &y_t, n_in, n_out, H, ROWS,
                )
                .unwrap();
            });
            let grouped = tensor_f32_at_offset(&y_t);
            for row in [0usize, 1, 63, 127] {
                let xr = x_t.view_subrange((row * n_in * H) as u64, vec![(n_in * H) as u64]);
                let yr = f32_tensor(&ctx, &vec![0.0; n_out * H], vec![(n_out * H) as u64]);
                run(&ctx, |enc| {
                    encode_mat_vec_q8_0_grouped_f32(&ctx, enc, &weight, &xr, &yr, n_in, n_out, H)
                        .unwrap();
                });
                let gemv = tensor_f32_at_offset(&yr);
                let got = &grouped[row * n_out * H..(row + 1) * n_out * H];
                let scale = gemv.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let worst = max_abs_diff_finite("grouped vs gemv", got, &gemv);
                assert!(
                    worst <= 1e-5 * scale.max(1.0),
                    "{n_in}->{n_out} row {row}: {worst} (scale {scale})"
                );
            }
        }
    }

    /// The grouped Q8_0 weight contract: per-group or flat (DS4) views of
    /// the same bytes give identical results; any other shape, a view past
    /// its buffer, a misaligned base and overflowing dimensions are refused.
    #[test]
    fn grouped_q8_mat_mat_refuses_invalid_weight_views() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const ROWS: usize = 128;
        let (n_in, n_out) = (256usize, 512usize);
        let (bytes, _) = synthetic_q8_0_bank(n_in, n_out * H);
        let view = |prefix: usize, suffix: usize, shape: Vec<u64>| {
            offset_tensor(&ctx, prefix, &bytes, suffix, shape, GgmlType::Q8_0)
        };
        let grouped = view(256, 64, vec![n_in as u64, n_out as u64, H as u64]);
        let flat = view(256, 64, vec![n_in as u64, (n_out * H) as u64]);
        let x: Vec<f32> = (0..n_in * H * ROWS)
            .map(|i| ((i * 29 % 97) as f32 - 48.0) * 0.02)
            .collect();
        let x_t = f32_tensor(&ctx, &x, vec![(n_in * H) as u64, ROWS as u64]);
        let output = || {
            f32_tensor(
                &ctx,
                &vec![0.0; n_out * H * ROWS],
                vec![(n_out * H) as u64, ROWS as u64],
            )
        };
        let (y_grouped, y_flat) = (output(), output());
        run(&ctx, |enc| {
            for (weight, y) in [(&grouped, &y_grouped), (&flat, &y_flat)] {
                encode_mat_mat_q8_0_grouped_f32(&ctx, enc, weight, &x_t, y, n_in, n_out, H, ROWS)
                    .unwrap();
            }
        });
        let (a, b) = (
            tensor_f32_at_offset(&y_grouped),
            tensor_f32_at_offset(&y_flat),
        );
        assert_eq!(max_abs_diff_finite("grouped vs flat", &a, &b), 0.0);
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));

        let y = output();
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let refuse = |weight: &MetalTensor, groups: usize, needle: &str| {
            let err = encode_mat_mat_q8_0_grouped_f32(
                &ctx, &enc, weight, &x_t, &y, n_in, n_out, groups, ROWS,
            )
            .unwrap_err()
            .to_string();
            assert!(err.contains(needle), "{err}");
        };
        refuse(
            &view(256, 64, vec![(n_in * H) as u64, n_out as u64]),
            H,
            "weight must be",
        );
        let past_end = MetalTensor {
            offset: grouped.offset + 128,
            ..grouped.clone()
        };
        refuse(&past_end, H, "inside its buffer");
        refuse(
            &view(257, 64, vec![n_in as u64, n_out as u64, H as u64]),
            H,
            "not 2-byte aligned",
        );
        refuse(&grouped, usize::MAX / 2, "overflow");
        enc.end();
    }

    /// CPU softmax over exactly `rows` (no sink) for one query token.
    fn selected_reference(q: &[f32], cache: &[half::f16], rows: &[usize], scale: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; W * H];
        for h in 0..H {
            let qh = &q[h * W..][..W];
            let scores: Vec<f64> = rows
                .iter()
                .map(|&j| {
                    qh.iter()
                        .zip(&cache[j * W..(j + 1) * W])
                        .map(|(q, c)| *q as f64 * c.to_f64())
                        .sum::<f64>()
                        * scale as f64
                })
                .collect();
            let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..W {
                let o: f64 = rows
                    .iter()
                    .zip(&weights)
                    .map(|(&j, w)| w * cache[j * W + d].to_f64())
                    .sum();
                out[h * W + d] = (o / total) as f32;
            }
        }
        out
    }

    /// Selected latent attention (online direct kernel, window 0, no sink)
    /// equals a CPU softmax over exactly the selected rows: 2051 ascending
    /// rows of a larger visible prefix, a distinctive excluded row that would
    /// dominate if attended, -1 padding with full and partial counts, a
    /// nonzero query offset into a larger chunk; and it equals dense latent
    /// attention when the selection is the whole visible prefix.
    #[test]
    fn selected_latent_attention_matches_reference_and_dense_equivalence() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const CAP: usize = 2_300;
        const SLOTS: usize = 2_051;
        const TOKENS: usize = 5;
        const START: usize = 2_100;
        const OFFSET: usize = 2;
        const QUERIES: usize = 3;
        let scale = 1.0 / 16.0;
        let mut state = 0x1234_5678u32;
        let mut noise = |scale: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
        };
        let q: Vec<f32> = (0..W * H * TOKENS).map(|_| noise(1.0)).collect();
        let mut cache: Vec<half::f16> = (0..W * CAP)
            .map(|_| half::f16::from_f32(noise(0.5)))
            .collect();
        // Row 1500 aligns with head 0 of the first queried token.
        const EXCLUDED: usize = 1_500;
        for d in 0..W {
            let sign = q[(OFFSET * H) * W + d].signum();
            cache[EXCLUDED * W + d] = half::f16::from_f32(0.9 * sign);
        }
        let mut ids = vec![-1i32; SLOTS * QUERIES];
        let mut counts = vec![0i32; QUERIES];
        let mut visible = vec![0i32; QUERIES];
        let mut selections = Vec::new();
        for query in 0..QUERIES {
            let l = START + OFFSET + query + 1;
            visible[query] = l as i32;
            // Keep 2051 rows (2040 for the last query), dropping EXCLUDED
            // and an even spread of others.
            let keep = if query == QUERIES - 1 { 2_040 } else { SLOTS };
            let mut rows: Vec<usize> = (0..l).filter(|&r| r != EXCLUDED).collect();
            let drop = rows.len() - keep;
            let stride = rows.len() / drop;
            rows = rows
                .into_iter()
                .enumerate()
                .filter(|(i, _)| !(i % stride == 0 && i / stride < drop))
                .map(|(_, r)| r)
                .collect();
            assert_eq!(rows.len(), keep);
            for (slot, &r) in rows.iter().enumerate() {
                ids[query * SLOTS + slot] = r as i32;
            }
            // The last query keeps its -1 padding inside the counted slots.
            counts[query] = SLOTS as i32;
            selections.push(rows);
        }
        let i32_t = |v: &[i32], shape: Vec<u64>| {
            offset_tensor(&ctx, 16, bytemuck::cast_slice(v), 16, shape, GgmlType::I32)
        };
        let q_t = f32_tensor(&ctx, &q, vec![(W * H) as u64, TOKENS as u64]);
        let cache_t = offset_tensor(
            &ctx,
            64,
            bytemuck::cast_slice(&cache),
            64,
            vec![W as u64, CAP as u64],
            GgmlType::F16,
        );
        let ids_t = i32_t(&ids, vec![SLOTS as u64, QUERIES as u64]);
        let counts_t = i32_t(&counts, vec![QUERIES as u64]);
        let visible_t = i32_t(&visible, vec![QUERIES as u64]);
        let sinks_t = f32_tensor(&ctx, &[LATENT_NO_SINK; H], vec![H as u64]);
        let out_t = f32_tensor(
            &ctx,
            &vec![7.0; W * H * TOKENS],
            vec![(W * H) as u64, TOKENS as u64],
        );
        run(&ctx, |enc| {
            encode_online_selected_attention_f16(
                &ctx,
                enc,
                &SelectedAttention {
                    queries: &q_t,
                    raw_cache: &cache_t,
                    raw_cache_before_chunk: &cache_t,
                    compressed_cache: &cache_t,
                    selected_ids: &ids_t,
                    selected_counts: &counts_t,
                    visible_counts: &visible_t,
                    sinks: &sinks_t,
                    output: &out_t,
                },
                SelectedAttentionShape {
                    head_count: H,
                    query_count: QUERIES,
                    query_token_offset: OFFSET,
                    token_count: TOKENS,
                    chunk_start_position: START,
                    window: 0,
                    raw_cache_is_chunk: false,
                    selected_slots: SLOTS,
                    compressed_capacity: CAP,
                    scale,
                    direct: true,
                },
            )
            .unwrap();
        });
        let out = tensor_f32_at_offset(&out_t);
        assert!(
            out[..OFFSET * W * H].iter().all(|&v| v == 7.0),
            "unqueried tokens written"
        );
        for (query, rows) in selections.iter().enumerate() {
            let token = OFFSET + query;
            let qt = &q[token * W * H..(token + 1) * W * H];
            let want = selected_reference(qt, &cache, rows, scale);
            let got = &out[token * W * H..(token + 1) * W * H];
            let worst = max_abs_diff_finite("selected attention", got, &want);
            assert!(worst <= 2e-5, "query {query}: max |diff| {worst}");
            if query == 0 {
                // The fixture discriminates: attending EXCLUDED changes head 0.
                let mut with = rows.clone();
                with.push(EXCLUDED);
                let wrong = selected_reference(qt, &cache, &with, scale);
                let moved = max_abs_diff_finite("excluded row", &wrong[..W], &got[..W]);
                assert!(moved > 0.1, "excluded row is not distinctive ({moved})");
            }
        }

        // Dense equivalence: selecting the whole visible prefix equals dense
        // latent attention (below the sparse frontier).
        const P: usize = 2_000;
        let q1 = f32_tensor(&ctx, &q[..W * H], vec![W as u64, H as u64, 1]);
        let dense_t = f32_tensor(&ctx, &vec![0.0; W * H], vec![W as u64, H as u64, 1]);
        let mut all = vec![-1i32; SLOTS];
        for (slot, id) in all.iter_mut().enumerate().take(P + 1) {
            *id = slot as i32;
        }
        let all_t = i32_t(&all, vec![SLOTS as u64, 1]);
        let one = |v: i32| i32_t(&[v], vec![1]);
        let (count1, visible1) = (one((P + 1) as i32), one((P + 1) as i32));
        let q1_flat = f32_tensor(&ctx, &q[..W * H], vec![(W * H) as u64, 1]);
        let sel_t = f32_tensor(&ctx, &vec![0.0; W * H], vec![(W * H) as u64, 1]);
        run(&ctx, |enc| {
            encode_latent_attention(&ctx, enc, &q1, &cache_t, &sinks_t, &dense_t, P, 1, scale)
                .unwrap();
            encode_online_selected_attention_f16(
                &ctx,
                enc,
                &SelectedAttention {
                    queries: &q1_flat,
                    raw_cache: &cache_t,
                    raw_cache_before_chunk: &cache_t,
                    compressed_cache: &cache_t,
                    selected_ids: &all_t,
                    selected_counts: &count1,
                    visible_counts: &visible1,
                    sinks: &sinks_t,
                    output: &sel_t,
                },
                SelectedAttentionShape {
                    head_count: H,
                    query_count: 1,
                    query_token_offset: 0,
                    token_count: 1,
                    chunk_start_position: P,
                    window: 0,
                    raw_cache_is_chunk: false,
                    selected_slots: SLOTS,
                    compressed_capacity: CAP,
                    scale,
                    direct: true,
                },
            )
            .unwrap();
        });
        let (dense, selected) = (tensor_f32_at_offset(&dense_t), tensor_f32_at_offset(&sel_t));
        let worst = max_abs_diff_finite("dense vs full selection", &selected, &dense);
        assert!(worst <= 1e-5, "dense vs full selection: {worst}");
    }

    /// The split kernel equals the serial kernel within float tolerance on
    /// a DS4-shaped geometry (raw ring window, finite sinks, padded and
    /// out-of-range selections) and a GLM-shaped one (window 0, no sink,
    /// 2,051 slots); and a query's split result is bitwise independent of
    /// how many queries share the dispatch.
    #[test]
    fn split_selected_attention_matches_serial_and_is_dispatch_independent() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        let mut state = 0x9e37_79b9u32;
        let mut noise = |scale: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
        };
        const CAP: usize = 2_400;
        const TOKENS: usize = 4;
        for (window, slots, sink, start) in [
            (128usize, 300usize, 0.75f32, 2_200usize),
            (0, 2_051, LATENT_NO_SINK, 2_300),
        ] {
            let q: Vec<f32> = (0..W * H * TOKENS).map(|_| noise(1.0)).collect();
            let cache: Vec<half::f16> = (0..W * CAP)
                .map(|_| half::f16::from_f32(noise(0.5)))
                .collect();
            let ring: Vec<half::f16> = (0..W * window.max(1))
                .map(|_| half::f16::from_f32(noise(0.5)))
                .collect();
            let mut ids = vec![-1i32; slots * TOKENS];
            let mut counts = [0i32; TOKENS];
            let mut visible = [0i32; TOKENS];
            for t in 0..TOKENS {
                let l = start + t + 1;
                visible[t] = l as i32;
                counts[t] = (slots - 7 * t) as i32;
                for slot in 0..slots {
                    // Mostly valid ids; some beyond visibility or negative.
                    ids[t * slots + slot] = match slot % 97 {
                        0 => -1,
                        1 => (l + 3) as i32,
                        _ => ((slot * 7 + t) % l) as i32,
                    };
                }
            }
            let i32_t = |v: &[i32], shape: Vec<u64>| {
                offset_tensor(&ctx, 16, bytemuck::cast_slice(v), 16, shape, GgmlType::I32)
            };
            let f16_t = |v: &[half::f16], rows: usize| {
                offset_tensor(
                    &ctx,
                    64,
                    bytemuck::cast_slice(v),
                    64,
                    vec![W as u64, rows as u64],
                    GgmlType::F16,
                )
            };
            let q_t = f32_tensor(&ctx, &q, vec![(W * H) as u64, TOKENS as u64]);
            let cache_t = f16_t(&cache, CAP);
            let ring_t = f16_t(&ring, window.max(1));
            let sinks_t = f32_tensor(&ctx, &[sink; H], vec![H as u64]);
            let splits = selected_attention_splits(window, slots);
            let run_kind = |split: bool, first: usize, count: usize| -> Vec<f32> {
                let out_t = f32_tensor(
                    &ctx,
                    &vec![3.0; W * H * TOKENS],
                    vec![(W * H) as u64, TOKENS as u64],
                );
                let ids_t = i32_t(
                    &ids[first * slots..(first + count) * slots],
                    vec![slots as u64, count as u64],
                );
                let counts_t = i32_t(&counts[first..first + count], vec![count as u64]);
                let visible_t = i32_t(&visible[first..first + count], vec![count as u64]);
                let units = (count * H * splits) as u64;
                let values_t =
                    f32_tensor(&ctx, &vec![0.0; W * units as usize], vec![W as u64, units]);
                let stats_t = f32_tensor(&ctx, &vec![0.0; 2 * units as usize], vec![2, units]);
                let buffers = SelectedAttention {
                    queries: &q_t,
                    raw_cache: &ring_t,
                    raw_cache_before_chunk: &ring_t,
                    compressed_cache: &cache_t,
                    selected_ids: &ids_t,
                    selected_counts: &counts_t,
                    visible_counts: &visible_t,
                    sinks: &sinks_t,
                    output: &out_t,
                };
                let shape = SelectedAttentionShape {
                    head_count: H,
                    query_count: count,
                    query_token_offset: first,
                    token_count: TOKENS,
                    chunk_start_position: start,
                    window,
                    raw_cache_is_chunk: false,
                    selected_slots: slots,
                    compressed_capacity: CAP,
                    scale: 1.0 / 16.0,
                    direct: true,
                };
                run(&ctx, |enc| {
                    if split {
                        let partials = SelectedAttentionPartials {
                            values: &values_t,
                            stats: &stats_t,
                        };
                        encode_online_selected_attention_split_f16(
                            &ctx, enc, &buffers, shape, &partials,
                        )
                        .unwrap();
                    } else {
                        encode_online_selected_attention_f16(&ctx, enc, &buffers, shape).unwrap();
                    }
                });
                tensor_f32_at_offset(&out_t)
            };
            let serial = run_kind(false, 0, TOKENS);
            let split = run_kind(true, 0, TOKENS);
            let worst = max_abs_diff_finite("split vs serial", &split, &serial);
            assert!(worst <= 2e-6, "window {window}: split vs serial {worst}");
            assert_ne!(
                split.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                vec![3.0f32.to_bits(); split.len()]
            );
            for t in 0..TOKENS {
                let alone = run_kind(true, t, 1);
                let row = t * W * H..(t + 1) * W * H;
                assert!(
                    alone[row.clone()]
                        .iter()
                        .zip(&split[row])
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "window {window}: token {t} depends on the dispatch"
                );
            }
        }
    }

    /// Directed edge cases of the split kernel against an independent f64
    /// softmax (the sink as one extra logit with a zero value row) and
    /// against the serial kernel: row totals of 127, 128 and 129 and across
    /// three splits with a raw window, zero counts, an all-rejected interior
    /// split and an empty trailing split, all-equal scores, a dominant sink
    /// and wide score separation. Partials start NaN-poisoned and are reused
    /// by consecutive dispatches of different query counts in one serial
    /// encoder (the packed sub-batch pattern, through sub-views); a lone
    /// dispatch leaves the other output rows untouched and reproduces its
    /// row bitwise.
    #[test]
    fn split_selected_attention_directed_cases_match_an_independent_softmax() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        struct Case {
            name: &'static str,
            window: usize,
            slots: usize,
            sink: f32,
            query_scale: f32,
            counts: [i32; TOKENS],
            rejected: [std::ops::Range<usize>; TOKENS],
            tolerance: f64,
        }
        const CAP: usize = 2_400;
        const TOKENS: usize = 3;
        const START: usize = 2_000;
        const SCALE: f32 = 1.0 / 16.0;
        let cases = [
            Case {
                name: "totals 127/128/129",
                window: 0,
                slots: 129,
                sink: LATENT_NO_SINK,
                query_scale: 1.0,
                counts: [127, 128, 129],
                rejected: [0..0, 0..0, 0..0],
                tolerance: 2e-5,
            },
            Case {
                name: "raw window across splits",
                window: 100,
                slots: 157,
                sink: 0.75,
                query_scale: 1.0,
                counts: [157, 28, 29],
                rejected: [0..0, 0..0, 0..0],
                tolerance: 2e-5,
            },
            Case {
                name: "zero, rejected interior, empty trailing",
                window: 0,
                slots: 300,
                sink: LATENT_NO_SINK,
                query_scale: 1.0,
                counts: [0, 300, 200],
                rejected: [0..0, 128..256, 0..0],
                tolerance: 2e-5,
            },
            Case {
                name: "all-equal scores",
                window: 0,
                slots: 300,
                sink: LATENT_NO_SINK,
                query_scale: 0.0,
                counts: [300, 255, 129],
                rejected: [0..0, 0..0, 0..0],
                tolerance: 2e-5,
            },
            Case {
                name: "dominant sink",
                window: 0,
                slots: 300,
                sink: 40.0,
                query_scale: 1.0,
                counts: [300, 300, 300],
                rejected: [0..0, 0..0, 0..0],
                tolerance: 2e-5,
            },
            Case {
                name: "wide separation",
                window: 0,
                slots: 300,
                sink: LATENT_NO_SINK,
                query_scale: 40.0,
                counts: [300, 300, 300],
                rejected: [0..0, 0..0, 0..0],
                tolerance: 1e-4,
            },
        ];
        let mut state = 0x2545_f491u32;
        let mut noise = |scale: f32| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            ((state >> 8) as f32 / 16_777_216.0 - 0.5) * 2.0 * scale
        };
        for case in &cases {
            let name = case.name;
            let (window, slots) = (case.window, case.slots);
            let q: Vec<f32> = (0..W * H * TOKENS)
                .map(|_| noise(case.query_scale))
                .collect();
            let cache: Vec<half::f16> = (0..W * CAP)
                .map(|_| half::f16::from_f32(noise(0.5)))
                .collect();
            let ring: Vec<half::f16> = (0..W * window.max(1))
                .map(|_| half::f16::from_f32(noise(0.5)))
                .collect();
            let visible: [i32; TOKENS] = std::array::from_fn(|t| (START + t + 1) as i32);
            let mut ids = vec![0i32; slots * TOKENS];
            for t in 0..TOKENS {
                let l = visible[t] as usize;
                for slot in 0..slots {
                    ids[t * slots + slot] = if case.rejected[t].contains(&slot) {
                        // Negative, then beyond visibility (inside capacity).
                        if slot % 2 == 0 { -1 } else { (l + 3) as i32 }
                    } else {
                        ((slot * 13 + t * 5) % l) as i32
                    };
                }
            }

            // Independent softmax over (sink, raw window rows, admitted ids).
            let oracle = |t: usize| -> Vec<f64> {
                let visible_end = START + t + 1;
                let raw_count = visible_end.min(window);
                let mut rows: Vec<&[half::f16]> = (visible_end - raw_count..visible_end)
                    .map(|p| &ring[(p % window) * W..][..W])
                    .collect();
                let count = (case.counts[t].max(0) as usize).min(slots);
                for &id in &ids[t * slots..][..count] {
                    if id >= 0 && (id as usize) < visible_end && (id as usize) < CAP {
                        rows.push(&cache[id as usize * W..][..W]);
                    }
                }
                let mut out = vec![0.0f64; W * H];
                for h in 0..H {
                    let qh = &q[(t * H + h) * W..][..W];
                    let scores: Vec<f64> = rows
                        .iter()
                        .map(|row| {
                            qh.iter()
                                .zip(row.iter())
                                .map(|(q, k)| *q as f64 * k.to_f64())
                                .sum::<f64>()
                                * SCALE as f64
                        })
                        .collect();
                    let top = scores.iter().fold(case.sink as f64, |m, &s| m.max(s));
                    let mut denominator = (case.sink as f64 - top).exp();
                    let o = &mut out[h * W..][..W];
                    for (row, s) in rows.iter().zip(&scores) {
                        let weight = (s - top).exp();
                        denominator += weight;
                        for (o, k) in o.iter_mut().zip(row.iter()) {
                            *o += weight * k.to_f64();
                        }
                    }
                    o.iter_mut().for_each(|v| *v /= denominator);
                }
                out
            };

            let i32_t = |v: &[i32], shape: Vec<u64>| {
                offset_tensor(&ctx, 16, bytemuck::cast_slice(v), 16, shape, GgmlType::I32)
            };
            let f16_t = |v: &[half::f16], rows: usize| {
                offset_tensor(
                    &ctx,
                    64,
                    bytemuck::cast_slice(v),
                    64,
                    vec![W as u64, rows as u64],
                    GgmlType::F16,
                )
            };
            let q_t = f32_tensor(&ctx, &q, vec![(W * H) as u64, TOKENS as u64]);
            let cache_t = f16_t(&cache, CAP);
            let ring_t = f16_t(&ring, window.max(1));
            let sinks_t = f32_tensor(&ctx, &[case.sink; H], vec![H as u64]);
            let ids_t = i32_t(&ids, vec![slots as u64, TOKENS as u64]);
            let counts_t = i32_t(&case.counts, vec![TOKENS as u64]);
            let visible_t = i32_t(&visible, vec![TOKENS as u64]);
            let splits = selected_attention_splits(window, slots);
            let all_units = (TOKENS * H * splits) as u64;
            let values_t = f32_tensor(
                &ctx,
                &vec![f32::NAN; W * all_units as usize],
                vec![W as u64, all_units],
            );
            let stats_t = f32_tensor(
                &ctx,
                &vec![f32::NAN; 2 * all_units as usize],
                vec![2, all_units],
            );
            let output = || {
                f32_tensor(
                    &ctx,
                    &vec![3.0; W * H * TOKENS],
                    vec![(W * H) as u64, TOKENS as u64],
                )
            };
            // Encode (first, count) dispatches in order into one serial encoder.
            let encode = |out_t: &MetalTensor, split: bool, batches: &[(usize, usize)]| {
                run(&ctx, |enc| {
                    for &(first, count) in batches {
                        let ids_v = ids_t.view_subrange(
                            (first * slots) as u64,
                            vec![slots as u64, count as u64],
                        );
                        let counts_v = counts_t.view_subrange(first as u64, vec![count as u64]);
                        let visible_v = visible_t.view_subrange(first as u64, vec![count as u64]);
                        let buffers = SelectedAttention {
                            queries: &q_t,
                            raw_cache: &ring_t,
                            raw_cache_before_chunk: &ring_t,
                            compressed_cache: &cache_t,
                            selected_ids: &ids_v,
                            selected_counts: &counts_v,
                            visible_counts: &visible_v,
                            sinks: &sinks_t,
                            output: out_t,
                        };
                        let shape = SelectedAttentionShape {
                            head_count: H,
                            query_count: count,
                            query_token_offset: first,
                            token_count: TOKENS,
                            chunk_start_position: START,
                            window,
                            raw_cache_is_chunk: false,
                            selected_slots: slots,
                            compressed_capacity: CAP,
                            scale: SCALE,
                            direct: true,
                        };
                        if split {
                            let units = (count * H * splits) as u64;
                            let partials = SelectedAttentionPartials {
                                values: &values_t.view_subrange(0, vec![W as u64, units]),
                                stats: &stats_t.view_subrange(0, vec![2, units]),
                            };
                            encode_online_selected_attention_split_f16(
                                &ctx, enc, &buffers, shape, &partials,
                            )
                            .unwrap();
                        } else {
                            encode_online_selected_attention_f16(&ctx, enc, &buffers, shape)
                                .unwrap();
                        }
                    }
                });
                tensor_f32_at_offset(out_t)
            };
            let split = encode(&output(), true, &[(0, 2), (2, 1)]);
            let serial = encode(&output(), false, &[(0, TOKENS)]);
            for t in 0..TOKENS {
                let expected = oracle(t);
                let row = t * W * H..(t + 1) * W * H;
                for h in 0..H {
                    let head = |v: &[f32]| v[row.start + h * W..][..W].to_vec();
                    let (got, reference) = (head(&split), head(&serial));
                    let want = &expected[h * W..][..W];
                    let magnitude = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                    let bound = case.tolerance * magnitude;
                    let worst = |v: &[f32]| {
                        v.iter()
                            .zip(want)
                            .map(|(a, e)| (*a as f64 - e).abs())
                            .fold(0.0f64, f64::max)
                    };
                    assert!(
                        got.iter().all(|v| v.is_finite()),
                        "{name}: token {t} head {h}: non-finite split output"
                    );
                    assert!(
                        worst(&got) <= bound,
                        "{name}: token {t} head {h}: split vs oracle {} > {bound}",
                        worst(&got)
                    );
                    assert!(
                        worst(&reference) <= bound,
                        "{name}: token {t} head {h}: serial vs oracle {} > {bound}",
                        worst(&reference)
                    );
                    if magnitude == 0.0 {
                        assert!(got.iter().all(|v| *v == 0.0), "{name}: token {t}: not zero");
                    }
                }
            }
            // A lone middle query leaves the other rows untouched and
            // reproduces its row bitwise.
            let alone = encode(&output(), true, &[(1, 1)]);
            for t in 0..TOKENS {
                let row = t * W * H..(t + 1) * W * H;
                let expected: Vec<u32> = if t == 1 {
                    split[row.clone()].iter().map(|v| v.to_bits()).collect()
                } else {
                    vec![3.0f32.to_bits(); W * H]
                };
                assert_eq!(
                    alone[row].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    expected,
                    "{name}: lone dispatch, token {t}"
                );
            }
        }
    }

    /// The encoder refuses raw caches too short for the window or chunk it
    /// is asked to read (no work is submitted).
    #[test]
    fn online_selected_attention_refuses_short_raw_caches() {
        let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
            return;
        };
        const TOKENS: usize = 2;
        let f16_rows = |rows: usize| {
            offset_tensor(
                &ctx,
                64,
                &vec![0u8; rows * W * 2],
                64,
                vec![W as u64, rows as u64],
                GgmlType::F16,
            )
        };
        let i32_t = |v: &[i32], shape: Vec<u64>| {
            offset_tensor(&ctx, 16, bytemuck::cast_slice(v), 16, shape, GgmlType::I32)
        };
        let q = f32_tensor(
            &ctx,
            &vec![0.0; W * H * TOKENS],
            vec![(W * H) as u64, TOKENS as u64],
        );
        let out = f32_tensor(
            &ctx,
            &vec![0.0; W * H * TOKENS],
            vec![(W * H) as u64, TOKENS as u64],
        );
        let compressed = f16_rows(16);
        let ids = i32_t(&[0; 4 * TOKENS], vec![4, TOKENS as u64]);
        let counts = i32_t(&[1; TOKENS], vec![TOKENS as u64]);
        let visible = i32_t(&[1; TOKENS], vec![TOKENS as u64]);
        let sinks = f32_tensor(&ctx, &[LATENT_NO_SINK; H], vec![H as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        let attempt = |raw: &MetalTensor, before: &MetalTensor, window: usize, chunk: bool| {
            encode_online_selected_attention_f16(
                &ctx,
                &enc,
                &SelectedAttention {
                    queries: &q,
                    raw_cache: raw,
                    raw_cache_before_chunk: before,
                    compressed_cache: &compressed,
                    selected_ids: &ids,
                    selected_counts: &counts,
                    visible_counts: &visible,
                    sinks: &sinks,
                    output: &out,
                },
                SelectedAttentionShape {
                    head_count: H,
                    query_count: TOKENS,
                    query_token_offset: 0,
                    token_count: TOKENS,
                    chunk_start_position: 200,
                    window,
                    raw_cache_is_chunk: chunk,
                    selected_slots: 4,
                    compressed_capacity: 16,
                    scale: 0.0625,
                    direct: true,
                },
            )
            .map_err(|e| e.to_string())
        };
        let (one, ring, chunk_rows) = (f16_rows(1), f16_rows(128), f16_rows(TOKENS));
        let err = attempt(&one, &one, 128, false).unwrap_err();
        assert!(err.contains("raw ring holds 1 rows"), "{err}");
        let err = attempt(&one, &ring, 128, true).unwrap_err();
        assert!(err.contains("raw chunk cache holds 1 rows"), "{err}");
        let err = attempt(&chunk_rows, &one, 128, true).unwrap_err();
        assert!(err.contains("raw cache before chunk holds 1 rows"), "{err}");
        // Window 0 never reads raw rows; ring 128 and chunk layouts that fit
        // are accepted.
        attempt(&one, &one, 0, false).unwrap();
        attempt(&ring, &one, 128, false).unwrap();
        attempt(&chunk_rows, &ring, 128, true).unwrap();
        enc.end();
    }
}
