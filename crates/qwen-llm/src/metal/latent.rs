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

const GROUPED_TAIL_KERNEL: &str = "kernel_mat_mat_q8_0_f32_r2c16k64_grouped_tail";

fn grouped_tail_capacity(
    width: usize,
    threads: usize,
    static_bytes: usize,
    device_bytes: usize,
) -> Result<(), MetalError> {
    if width != 32
        || threads < 128
        || static_bytes
            .checked_add(4096)
            .is_none_or(|bytes| bytes > device_bytes)
    {
        return Err(bad_shape(
            GROUPED_TAIL_KERNEL,
            "needs SIMD32, TG128 and 4096 dynamic TGM bytes",
        ));
    }
    Ok(())
}

/// Grouped Q8_0/F32 absorption for complete eight-token fragments, without
/// padded activations or outputs. Same layouts as `encode_mat_mat_q8_0_grouped_f32`;
/// requires positive K % 64 == 0, M % 16 == 0, groups > 0 and rows in 8..=128,
/// divisible by eight. The 128-row case is a diagnostic control. Arithmetic
/// retains F32 dequantization, activations and accumulation of the full tile.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_q8_0_grouped_tail_f32(
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
    const K: &str = GROUPED_TAIL_KERNEL;
    require_serial(K, enc)?;
    if n_in == 0
        || n_out == 0
        || groups == 0
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !(8..=128).contains(&rows)
        || !rows.is_multiple_of(8)
    {
        return Err(bad_shape(
            K,
            "needs positive K % 64, M % 16, groups and rows % 8 in 8..=128",
        ));
    }
    let product = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| bad_shape(K, "dimension product overflows"))
    };
    let in_width = product(n_in, groups)?;
    let out_width = product(n_out, groups)?;
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
    let args = Args {
        m: u(n_out, "M")?,
        n: u(rows, "N")?,
        k: u(n_in, "K")?,
        groups: u(groups, "groups")?,
        nb01: u(product(n_in / 32, 34)?, "row bytes")?,
        stride_b: u(in_width, "input width")?,
        stride_c: u(out_width, "output width")?,
    };
    if weight.shape != [n_in as u64, n_out as u64, groups as u64]
        && weight.shape != [n_in as u64, out_width as u64]
    {
        return Err(bad_shape(
            K,
            "weight must be [K, M, groups] or [K, M * groups]",
        ));
    }
    if input.shape != [in_width as u64, rows as u64]
        || output.shape != [out_width as u64, rows as u64]
    {
        return Err(bad_shape(
            K,
            "activations must be token-major [width * groups, rows]",
        ));
    }
    for (name, tensor, elements, dtype, writable, alignment) in [
        (
            "weight",
            weight,
            product(n_in, out_width)?,
            GgmlType::Q8_0,
            false,
            2,
        ),
        (
            "input",
            input,
            product(in_width, rows)?,
            GgmlType::F32,
            false,
            4,
        ),
        (
            "output",
            output,
            product(out_width, rows)?,
            GgmlType::F32,
            true,
            4,
        ),
    ] {
        super::moe::validate_moe_decode_tensor(
            K,
            name,
            tensor,
            elements,
            &[dtype],
            writable,
            alignment,
        )?;
    }
    check_disjoint(K, output, &[(input, "input"), (weight, "weight")])?;
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device
        || [weight, input, output]
            .iter()
            .any(|t| t.buffer.device().registryID() != device)
    {
        return Err(bad_shape(
            K,
            "encoder and bindings must belong to the context device",
        ));
    }
    let pso = ctx.pipeline(GROUPED_TAIL_KERNEL)?;
    grouped_tail_capacity(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        pso.staticThreadgroupMemoryLength(),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4096);
    enc.dispatch(
        MTLSize {
            width: 1,
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

/// Native grouped projection formats with canonical decoders in `quant_tiles.h`.
/// This is a storage-format capability list, independent of model family.
pub const GROUPED_PROJECTION_DTYPES: &[GgmlType] = &[
    GgmlType::Q2_0,
    GgmlType::Q2_K,
    GgmlType::Q3_K,
    GgmlType::Q4_K,
    GgmlType::Q5_K,
    GgmlType::Q6_K,
    GgmlType::Q8_0,
    GgmlType::Q4_0,
    GgmlType::Q4_1,
    GgmlType::IQ2_S,
    GgmlType::IQ3_XXS,
    GgmlType::IQ3_S,
    GgmlType::IQ4_NL,
    GgmlType::IQ4_XS,
    GgmlType::F32,
    GgmlType::F16,
    GgmlType::BF16,
];

/// Whether the native grouped F32-operand projection supports this storage type.
pub fn grouped_projection_supported(dtype: GgmlType) -> bool {
    GROUPED_PROJECTION_DTYPES.contains(&dtype)
}

/// Scalar baseline pipeline name, independent of specialized Q8 and Q6_K MMA dispatch.
pub fn grouped_projection_pipeline_name(dtype: GgmlType) -> Option<String> {
    let suffix = match dtype {
        GgmlType::Q2_0 => "q2_0",
        GgmlType::Q2_K => "q2_K",
        GgmlType::Q3_K => "q3_K",
        GgmlType::Q4_K => "q4_K",
        GgmlType::Q5_K => "q5_K",
        GgmlType::Q6_K => "q6_K",
        GgmlType::Q8_0 => "q8_0",
        GgmlType::Q4_0 => "q4_0",
        GgmlType::Q4_1 => "q4_1",
        GgmlType::IQ2_S => "iq2_s",
        GgmlType::IQ3_XXS => "iq3_xxs",
        GgmlType::IQ3_S => "iq3_s",
        GgmlType::IQ4_NL => "iq4_nl",
        GgmlType::IQ4_XS => "iq4_xs",
        GgmlType::F32 => "f32",
        GgmlType::F16 => "f16",
        GgmlType::BF16 => "bf16",
        _ => return None,
    };
    Some(format!("kernel_latent_grouped_{suffix}_f32"))
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GroupedProjectionArgs {
    k: u32,
    m: u32,
    groups: u32,
    rows: u32,
    row_bytes: u32,
    input_stride: u32,
    output_stride: u32,
}

#[cfg(test)]
const GROUPED_Q6_SCALAR8_KERNEL: &str = "kernel_latent_grouped_q6_K_f32_scalar8";
const GROUPED_Q6_MMA_KERNEL: &str = "kernel_latent_grouped_q6_K_f32_r2c16k64";
const GROUPED_F32_MMA_KERNEL: &str = "kernel_latent_grouped_f32_f32_r2c16k64";
const GROUPED_MMA_SCRATCH_BYTES: usize = 16 * 64 * std::mem::size_of::<f32>();

// This checks storage/tile geometry, not a model-family or head-count whitelist.
fn grouped_projection_mma_eligible(dtype: GgmlType, args: &GroupedProjectionArgs) -> bool {
    dtype == GgmlType::Q6_K
        && args.k > 0
        && args.k.is_multiple_of(256)
        && args.m > 0
        && args.m.is_multiple_of(16)
        && args.groups > 0
        && args.rows >= 128
}

// `allow_mma` has already resolved the test baseline policy. The separate F32
// permission must remain explicit even when that override forces MMA on.
fn grouped_projection_mma_kernel(
    dtype: GgmlType,
    args: &GroupedProjectionArgs,
    allow_mma: bool,
    allow_f32_mma: bool,
) -> Option<&'static str> {
    if !allow_mma {
        return None;
    }
    if grouped_projection_mma_eligible(dtype, args) {
        return Some(GROUPED_Q6_MMA_KERNEL);
    }
    (allow_f32_mma
        && dtype == GgmlType::F32
        && args.k > 0
        && args.k.is_multiple_of(64)
        && args.m > 0
        && args.m.is_multiple_of(8)
        && args.groups > 0
        && args.rows >= 128)
        .then_some(GROUPED_F32_MMA_KERNEL)
}

#[cfg(test)]
thread_local! {
    static GROUPED_PROJECTION_SCALAR_BASELINE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    static GROUPED_PROJECTION_SCALAR8: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Select the pre-MMA scalar native baseline during a synchronous diagnostic.
/// `true` installs `Some(true)`, forcing old scalar4 (disables MMA and scalar8).
/// `false` installs `Some(false)`, permitting MMA regardless of caller policy,
/// subject to unchanged geometry/capacity checks and the scalar8 test scope.
/// It never grants the separate F32-weight MMA permission.
/// Outside an override (`None`),
/// caller policy applies and non-MMA work uses scalar4 unless the separate
/// test-only scalar8 experiment is explicitly enabled.
/// Q8 specialization and decode behavior are unaffected. State belongs
/// to the calling thread, supports nesting, and is restored during unwinding.
/// The closure must encode on this thread; the setting does not propagate to
/// worker threads or remain active for work deferred until after it returns.
#[cfg(test)]
pub(crate) fn with_grouped_projection_scalar_baseline_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GROUPED_PROJECTION_SCALAR_BASELINE.with(|state| state.set(self.0));
        }
    }
    let _restore =
        Restore(GROUPED_PROJECTION_SCALAR_BASELINE.with(|state| state.replace(Some(enabled))));
    f()
}

/// Test-only scalar8 experiment for whole-call Q6_K scalar work with >=8 rows.
/// Enabling this scope suppresses MMA, including a forced-MMA baseline scope,
/// so the scalar8/4 experiment stays isolated when caller policy permits MMA.
/// The scalar-baseline override `Some(true)` always wins and forces scalar4;
/// MMA remainders also remain scalar4. `false` disables the experiment within
/// a nested scope. Thread-local, restores nesting/unwinding, no environment
/// toggle, and no scalar8 host selection is compiled into production builds.
#[cfg(test)]
pub(crate) fn with_grouped_projection_scalar8_override<R>(
    enabled: bool,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            GROUPED_PROJECTION_SCALAR8.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(GROUPED_PROJECTION_SCALAR8.with(|state| state.replace(enabled)));
    f()
}

#[cfg(test)]
fn grouped_projection_scalar8_enabled() -> bool {
    GROUPED_PROJECTION_SCALAR8.with(|state| state.get())
}

fn grouped_projection_scalar_baseline() -> Option<bool> {
    #[cfg(test)]
    {
        GROUPED_PROJECTION_SCALAR_BASELINE.with(|state| state.get())
    }
    #[cfg(not(test))]
    {
        None
    }
}

// Some(false) preserves same-binary MMA diagnostics even for callers which
// deliberately disallow MMA in production (for example DenseF32 or Exact).
fn grouped_projection_policy(allow_mma: bool, baseline: Option<bool>) -> (bool, bool) {
    let allow_mma = baseline.map_or(allow_mma, |scalar4| !scalar4);
    #[cfg(test)]
    let allow_mma = allow_mma && !grouped_projection_scalar8_enabled();
    (allow_mma, baseline == Some(true))
}

fn grouped_projection_scalar_preflight(
    candidate: Result<Pipeline, MetalError>,
    device_bytes: usize,
) -> Result<Pipeline, MetalError> {
    let pso = candidate?;
    if pso.threadExecutionWidth() != 32
        || pso.maxTotalThreadsPerThreadgroup() < 128
        || pso.staticThreadgroupMemoryLength() > device_bytes
    {
        return Err(bad_shape(
            "grouped_projection",
            "needs four 32-lane simdgroups",
        ));
    }
    Ok(pso)
}

fn grouped_projection_mma_capacity(
    simd_width: usize,
    max_threads: usize,
    static_bytes: usize,
    device_bytes: usize,
) -> Result<(), MetalError> {
    if simd_width != 32
        || max_threads < 128
        || static_bytes
            .checked_add(GROUPED_MMA_SCRATCH_BYTES)
            .is_none_or(|bytes| bytes > device_bytes)
    {
        return Err(bad_shape(
            "grouped_projection",
            "MMA needs four 32-lane simdgroups and 4096 bytes plus static threadgroup memory",
        ));
    }
    Ok(())
}

// All failures here select whole-call scalar; the caller preflights that
// fallback before dispatching anything. Keep
// pipeline lookup separate so missing/failed candidates can be tested without
// compiling or dispatching GPU work.
fn grouped_projection_mma_preflight(
    candidate: Result<Pipeline, MetalError>,
    device_bytes: usize,
) -> Option<Pipeline> {
    let pso = candidate.ok()?;
    grouped_projection_mma_capacity(
        pso.threadExecutionWidth(),
        pso.maxTotalThreadsPerThreadgroup(),
        pso.staticThreadgroupMemoryLength(),
        device_bytes,
    )
    .ok()?;
    Some(pso)
}

#[allow(clippy::too_many_arguments)]
fn validate_grouped_projection(
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
) -> Result<GroupedProjectionArgs, MetalError> {
    const K: &str = "grouped_projection";
    if !grouped_projection_supported(weight.dtype) {
        return Err(bad_shape(
            K,
            format!("unsupported native type {:?}", weight.dtype),
        ));
    }
    let (block, bytes) = weight
        .dtype
        .storage_layout()
        .ok_or_else(|| bad_shape(K, "unknown layout"))?;
    if n_in == 0
        || n_out == 0
        || groups == 0
        || rows == 0
        || !n_in.is_multiple_of(32)
        || !(n_in as u64).is_multiple_of(block)
    {
        return Err(bad_shape(
            K,
            "positive K/M/groups/rows required; K must align to 32 and the storage block",
        ));
    }
    let product = |a: usize, b: usize| {
        a.checked_mul(b)
            .ok_or_else(|| bad_shape(K, "dimension product overflows"))
    };
    let input_stride = product(n_in, groups)?;
    let output_stride = product(n_out, groups)?;
    if weight.shape != [n_in as u64, n_out as u64, groups as u64]
        && weight.shape != [n_in as u64, output_stride as u64]
    {
        return Err(bad_shape(
            K,
            "weight must be contiguous [K, M, groups] or [K, M * groups]",
        ));
    }
    if input.shape != [input_stride as u64, rows as u64]
        || output.shape != [output_stride as u64, rows as u64]
    {
        return Err(bad_shape(
            K,
            "input/output must be token-major [K * groups, rows] / [M * groups, rows]",
        ));
    }
    for (name, tensor, elements, dtype, writable, alignment) in [
        (
            "weight",
            weight,
            product(n_in, output_stride)?,
            weight.dtype,
            false,
            if weight.dtype == GgmlType::F32 { 4 } else { 2 },
        ),
        (
            "input",
            input,
            product(input_stride, rows)?,
            GgmlType::F32,
            false,
            4,
        ),
        (
            "output",
            output,
            product(output_stride, rows)?,
            GgmlType::F32,
            true,
            4,
        ),
    ] {
        super::moe::validate_moe_decode_tensor(
            K,
            name,
            tensor,
            elements,
            &[dtype],
            writable,
            alignment,
        )?;
    }
    check_disjoint(K, output, &[(input, "input"), (weight, "weight")])?;
    let u = |v, name| super::checks::to_u32(K, v, name);
    Ok(GroupedProjectionArgs {
        k: u(n_in, "K")?,
        m: u(n_out, "M")?,
        groups: u(groups, "groups")?,
        rows: u(rows, "rows")?,
        row_bytes: u(
            product(n_in / block as usize, bytes as usize)?,
            "weight row bytes",
        )?,
        input_stride: u(input_stride, "input stride")?,
        output_stride: u(output_stride, "output stride")?,
    })
}

/// Compressed grouped projection for decode or packed callers (Exact operands).
/// Computes `Y[token, g, m] = sum_k W[g, m, k] * X[token, g, k]`.
///
/// * Weights: native contiguous `[K, M, groups]` (or identical bytes viewed as
///   `[K, M * groups]`), K fastest, no group padding or transposition.
/// * Input/output: contiguous token-major F32 `[K * groups, rows]` and
///   `[M * groups, rows]`; group slices within each token have strides K/M.
/// * K is positive and divisible by 32 and the native storage block size;
///   M, groups and rows are positive, with no tile divisibility requirement.
/// * Dimensions, token strides and weight row bytes must fit u32; total
///   extents must fit usize and their buffers. Address arithmetic is 64-bit.
/// * Nonzero buffer offsets are supported: weight alignment is two bytes
///   (four for F32), activation/output alignment is four bytes. Output must
///   be writable and disjoint from input and weights. All bindings and the
///   serial encoder must belong to the context's device.
///
/// No caller scratch, repacking or expanded weight bank. Generic kernels
/// decode 16 weights per lane into F32 registers and reuse them for up to four
/// tokens (eight only for the explicitly enabled test-only Q6_K experiment).
/// F32/BF16 weights and F32 activations are never narrowed to half;
/// accumulation is F32. "Exact" means operand precision, not bitwise agreement
/// with another reduction order or strict IEEE behavior under Metal fast math.
/// Q8 decode, full 128-token tiles and eligible 8-token tails retain the
/// existing Q8 kernels unchanged (internal threadgroup scratch only).
///
/// Q6_K uses F32 MMA for the complete 128-row prefix when `K % 256 == 0`,
/// `M % 16 == 0`, and `rows >= 128`, for any positive group count. The remaining
/// `rows % 128` tokens use one scalar dispatch on bounded tail views. Thus
/// 129/255 rows dispatch 128 MMA rows plus 1/127 scalar rows; 512 rows use one
/// MMA dispatch. Whole bindings are validated before creating either view,
/// and both pipelines are preflighted before the first dispatch. An unavailable
/// MMA pipeline or insufficient MMA capacity falls back to whole-call scalar.
/// No padding or extra device allocation is required. Each MMA tile
/// stages 16x64 decoded F32 weights in 4096 transient threadgroup bytes and
/// reads F32 activations directly. The shared native decoder and operand
/// precision are unchanged; MMA changes the F32 reduction order within the
/// existing Exact allowance. No half staging, repacking, or persistent scratch.
/// Ineligible Q6_K calls and the test-scoped baseline override retain the
/// whole-call scalar4 path. Scalar8 is retained only as an explicit test-only
/// experiment; production fallback and MMA remainders always use scalar4.
/// Other formats and decode are unchanged.
/// This public encoder delegates with `allow_mma = true`.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_mat_grouped_f32(
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
    encode_mat_mat_grouped_f32_with_policy(
        ctx, enc, weight, input, output, n_in, n_out, groups, rows, true,
    )
}

/// [`encode_mat_mat_grouped_f32`] with caller-controlled MMA permission.
/// `allow_mma = false` selects scalar4; it does not change operands, shapes,
/// or Q8 dispatch. Scalar8 is available only through its explicit test scope.
/// `true` permits the existing Q6_K MMA prefix plus scalar4 remainder.
/// Test override `Some(true)` forces old scalar4; `Some(false)` permits MMA
/// regardless of this argument; `None` follows this argument. Missing/unsupported
/// scalar8 candidates fall back to scalar4 before any dispatch. Whole-call
/// scalar bindings are borrowed without cloning shapes or allocating views.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mat_mat_grouped_f32_with_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
    allow_mma: bool,
) -> Result<(), MetalError> {
    encode_mat_mat_grouped_f32_with_f32_mma_policy(
        ctx, enc, weight, input, output, n_in, n_out, groups, rows, allow_mma, false,
    )
}

/// Explicit opt-in F32-weight MMA candidate with the same bindings/validation
/// as [`encode_mat_mat_grouped_f32`]. Both effective `allow_mma` and explicit
/// `allow_f32_mma` are required for F32; existing wrappers pass the latter false.
/// The baseline test override may change the former, but cannot grant the
/// latter. A forced scalar baseline (or scalar8 test scope) suppresses MMA.
/// Q6_K eligibility/arithmetic and Q8 selection are unchanged by the new flag.
///
/// F32 tiles require positive K divisible by 64, positive M divisible by 8,
/// positive groups, and rows >=128. Dispatch covers ceil(M/16) output tiles
/// and the complete 128-token prefix; a scalar4 dispatch handles any token
/// remainder. The final tile zeros missing weight rows before decode/read and
/// skips missing 8-output stores, so M=8,24,40,... need no padding or repacking.
/// Uses the same 4096-byte transient scratch and fresh F32 partial per K64 as
/// Q6_K. F32 operands are retained; reduction order can differ from scalar4
/// within the existing Exact operand-precision contract (no bitwise promise).
/// Whole bindings and both pipelines are checked before dispatch; unavailable
/// MMA falls back to whole-call scalar4 without extra views or device scratch.
#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_mat_mat_grouped_f32_with_f32_mma_policy(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
    rows: usize,
    allow_mma: bool,
    allow_f32_mma: bool,
) -> Result<(), MetalError> {
    const K: &str = "grouped_projection";
    require_serial(K, enc)?;
    let args = validate_grouped_projection(weight, input, output, n_in, n_out, groups, rows)?;
    let device = ctx.device.registryID();
    if enc.parent_command_buffer().device().registryID() != device
        || [weight, input, output]
            .iter()
            .any(|t| t.buffer.device().registryID() != device)
    {
        return Err(bad_shape(
            K,
            "encoder and bindings must belong to the context device",
        ));
    }
    if weight.dtype == GgmlType::Q8_0 {
        if rows == 1 {
            return encode_mat_vec_q8_0_grouped_f32(
                ctx, enc, weight, input, output, n_in, n_out, groups,
            );
        }
        if n_in.is_multiple_of(64) && n_out.is_multiple_of(16) {
            if rows.is_multiple_of(128) {
                return encode_mat_mat_q8_0_grouped_f32(
                    ctx, enc, weight, input, output, n_in, n_out, groups, rows,
                );
            }
            if rows <= 128 && rows.is_multiple_of(8) {
                return encode_mat_mat_q8_0_grouped_tail_f32(
                    ctx, enc, weight, input, output, n_in, n_out, groups, rows,
                );
            }
        }
    }
    let (allow_mma, _force_scalar4) =
        grouped_projection_policy(allow_mma, grouped_projection_scalar_baseline());
    // Resolve both candidates before creating views or encoding any dispatch.
    let mma = if let Some(kernel) =
        grouped_projection_mma_kernel(weight.dtype, &args, allow_mma, allow_f32_mma)
    {
        grouped_projection_mma_preflight(
            ctx.pipeline(kernel),
            ctx.device.maxThreadgroupMemoryLength(),
        )
    } else {
        None
    };
    // Production fallback is always scalar4. Scalar8 host selection and its
    // pipeline lookup are compiled only for explicitly scoped experiments.
    let kernel = grouped_projection_pipeline_name(weight.dtype).expect("validated native dtype");
    let pso = grouped_projection_scalar_preflight(
        ctx.pipeline(&kernel),
        ctx.device.maxThreadgroupMemoryLength(),
    )?;
    let scalar_tokens = 4;
    #[cfg(test)]
    let (pso, scalar_tokens) = {
        let scalar8 = if grouped_projection_scalar8_enabled()
            && mma.is_none()
            && !_force_scalar4
            && weight.dtype == GgmlType::Q6_K
            && rows >= 8
        {
            grouped_projection_scalar_preflight(
                ctx.pipeline(GROUPED_Q6_SCALAR8_KERNEL),
                ctx.device.maxThreadgroupMemoryLength(),
            )
            .ok()
        } else {
            None
        };
        match scalar8 {
            Some(candidate) => (candidate, 8),
            None => (pso, scalar_tokens),
        }
    };
    let (args, scalar_tail) = if let Some(mma) = mma {
        let prefix_rows = rows / 128 * 128;
        let tail_rows = rows - prefix_rows;
        // view_subrange checks the logical bounds and preserves buffer offsets
        // and provenance. Full validation above proved all extent products and
        // ranges fit; these are subranges of those same contiguous F32 tensors.
        let view = |tensor: &MetalTensor, stride: u32, start: usize, count: usize| {
            tensor.view_subrange(
                start as u64 * u64::from(stride),
                vec![u64::from(stride), count as u64],
            )
        };
        let prefix_input = view(input, args.input_stride, 0, prefix_rows);
        let prefix_output = view(output, args.output_stride, 0, prefix_rows);
        let tail = (tail_rows > 0).then(|| {
            (
                view(input, args.input_stride, prefix_rows, tail_rows),
                view(output, args.output_stride, prefix_rows, tail_rows),
            )
        });
        let prefix_args = GroupedProjectionArgs {
            rows: prefix_rows as u32,
            ..args
        };
        enc.set_pipeline(&mma);
        enc.set_bytes(0, &prefix_args);
        enc.set_tensor(1, weight);
        enc.set_tensor(2, &prefix_input);
        enc.set_tensor(3, &prefix_output);
        enc.set_threadgroup_memory(0, GROUPED_MMA_SCRATCH_BYTES);
        enc.dispatch(
            MTLSize {
                width: prefix_rows / 128,
                height: n_out.div_ceil(16),
                depth: groups,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        let Some((tail_input, tail_output)) = tail else {
            return Ok(());
        };
        (
            GroupedProjectionArgs {
                rows: tail_rows as u32,
                ..args
            },
            Some((tail_input, tail_output)),
        )
    } else {
        (args, None)
    };
    let (scalar_input, scalar_output) = scalar_tail
        .as_ref()
        .map(|(input, output)| (input, output))
        .unwrap_or((input, output));
    // No fallible work remains after the prefix. The unchanged scalar kernel
    // indexes tokens relative to these bounded tail views (or the whole call).
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, scalar_input);
    enc.set_tensor(3, scalar_output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
            height: (args.rows as usize).div_ceil(scalar_tokens),
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

/// Single-token wrapper for [`encode_mat_mat_grouped_f32`]. Input/output may
/// be `[K * groups, 1]` / `[M * groups, 1]`, flat vectors, or `[K, groups]` /
/// `[M, groups]` views of the same contiguous bytes. No copying or scratch.
#[allow(clippy::too_many_arguments)]
pub fn encode_mat_vec_grouped_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    groups: usize,
) -> Result<(), MetalError> {
    let view = |tensor: &MetalTensor, width: usize| -> Result<MetalTensor, MetalError> {
        let total = width
            .checked_mul(groups)
            .ok_or_else(|| bad_shape("grouped_projection", "dimension product overflows"))?
            as u64;
        if tensor.shape != [total, 1]
            && tensor.shape != [total]
            && tensor.shape != [width as u64, groups as u64]
        {
            return Err(bad_shape(
                "grouped_projection",
                "decode activation must be [width * groups, 1], [width * groups] or [width, groups]",
            ));
        }
        Ok(MetalTensor {
            shape: vec![total, 1],
            ..tensor.clone()
        })
    };
    encode_mat_mat_grouped_f32(
        ctx,
        enc,
        weight,
        &view(input, n_in)?,
        &view(output, n_out)?,
        n_in,
        n_out,
        groups,
        1,
    )
}

#[cfg(test)]
#[path = "latent/grouped_native_tests.rs"]
mod grouped_native_tests;

#[cfg(test)]
#[path = "latent/grouped_tail_tests.rs"]
mod grouped_tail_tests;

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
                        crate::compare::assert_max_abs_diff_f64(
                            concat!(file!(), ":", line!()),
                            v.iter(),
                            want,
                        )
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
