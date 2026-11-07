//! Selected-latent attention with Half-staged Q/P and native F16 K/V.
//! Callers explicitly permit narrowing; Exact/F32 owners keep their encoders.
use super::*;

const KERNEL: &str = "selected_attention_tensor";
const HEADS: usize = 64;
const KEY_TILE: usize = 32;
const VALUE_TILE: usize = 64;

fn score_layout(slots: usize, queries: usize) -> Result<[u64; 3], MetalError> {
    let bad = || bad_shape(KERNEL, "score slab exceeds 32-bit offsets");
    let stride = slots
        .div_ceil(KEY_TILE)
        .checked_mul(KEY_TILE)
        .ok_or_else(bad)?;
    stride
        .checked_mul(HEADS)
        .and_then(|v| v.checked_mul(queries))
        .and_then(|v| u32::try_from(v).ok())
        .ok_or_else(bad)?;
    Ok([stride as u64, HEADS as u64, queries as u64])
}

fn capacity_supported(
    width: usize,
    threads: usize,
    required: usize,
    static_bytes: usize,
    limit: usize,
) -> bool {
    width == 32 && threads >= required && static_bytes <= limit
}

/// Three-pass attention over selected F16 latent rows (K=V, width 512,
/// 64 heads). Q and probabilities narrow to Half; matmul accumulation,
/// softmax and output remain F32. Raw windows are unsupported. Invalid ids
/// and causal visibility use the singleton encoder's rules, including sinks.
///
/// `scores` is writable, 16-byte-aligned F32 `[ceil(slots/32)*32,64,queries]`,
/// disjoint from every input and output. It may reuse the caller's partial
/// scratch. No buffer is allocated. Returns false for unsupported layout/tier
/// or PSO capacity, before touching the encoder. Binding and pipeline errors
/// are returned, rather than hidden by fallback.
pub fn try_encode_selected_attention_tensor_f16(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    b: &SelectedAttention<'_>,
    g: SelectedAttentionShape,
    scores: &MetalTensor,
) -> Result<bool, MetalError> {
    require_serial(KERNEL, enc)?;
    if g.head_count != HEADS
        || g.window != 0
        || !g.direct
        || !ctx.tensor_ops_status().is_available()
    {
        return Ok(false);
    }
    validate_selected_attention(b, g)?;
    let shape = score_layout(g.selected_slots, g.query_count)?;
    check_tensor(KERNEL, scores, GgmlType::F32, &shape, true, "scores")?;
    check_alignment(KERNEL, scores, 16, "scores")?;
    check_disjoint(
        KERNEL,
        scores,
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
    )?;
    let psos = [
        ctx.tensor_pipeline("kernel_glm_latent_score_tensor")?,
        ctx.tensor_pipeline("kernel_glm_latent_softmax")?,
        ctx.tensor_pipeline("kernel_glm_latent_pv_tensor")?,
    ];
    let threads = [128, 32, 128];
    for (pso, &required) in psos.iter().zip(&threads) {
        if !capacity_supported(
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            required,
            pso.staticThreadgroupMemoryLength(),
            ctx.device.maxThreadgroupMemoryLength(),
        ) {
            return Ok(false);
        }
    }
    // Shared validation bounds every argument and queried token below to u32.
    let a = [
        g.query_count as u32,
        g.selected_slots as u32,
        shape[0] as u32,
        g.query_token_offset as u32,
    ];
    let c = [g.compressed_capacity as u32, HEADS as u32, 0, 0];
    let widths = [
        g.selected_slots.div_ceil(KEY_TILE),
        HEADS,
        LATENT_WIDTH / VALUE_TILE,
    ];
    for input in [
        b.queries,
        b.compressed_cache,
        b.selected_ids,
        b.selected_counts,
        b.visible_counts,
        b.sinks,
    ] {
        enc.note_read(input);
    }
    enc.note_write(scores);
    enc.note_write(b.output);
    for (i, pso) in psos.iter().enumerate() {
        enc.set_pipeline(pso);
        enc.set_bytes(0, &a);
        enc.set_bytes(1, &c);
        enc.set_bytes(2, &g.scale);
        for (slot, tensor) in [
            (3, b.queries),
            (4, b.compressed_cache),
            (5, b.selected_ids),
            (6, b.selected_counts),
            (7, b.visible_counts),
            (8, scores),
            (9, b.output),
            (10, b.sinks),
        ] {
            enc.set_tensor(slot, tensor);
        }
        enc.dispatch(
            MTLSize {
                width: widths[i],
                height: g.query_count,
                depth: 1,
            },
            MTLSize {
                width: threads[i],
                height: 1,
                depth: 1,
            },
        );
    }
    Ok(true)
}

#[cfg(test)]
mod policy_tests {
    use super::*;
    #[test]
    fn score_slab_fits_existing_glm_partial_scratch() {
        let slots = 2051;
        let partials = crate::glm5_next::memory::PACKED_SPLIT_QUERIES as usize
            * HEADS
            * selected_attention_splits(0, slots)
            * LATENT_WIDTH;
        for queries in 1..=64 {
            let dims = score_layout(slots, queries).unwrap();
            assert_eq!(dims, [2080, 64, queries as u64]);
            assert!(dims.into_iter().product::<u64>() <= partials as u64);
        }
        assert_eq!(score_layout(1, 1).unwrap(), [32, 64, 1]);
        assert_eq!(score_layout(33, 3).unwrap(), [64, 64, 3]);
        assert!(score_layout(usize::MAX, 64).is_err());
        assert!(score_layout(u32::MAX as usize, 64).is_err());
        assert!(score_layout(2051, usize::MAX).is_err());
    }
    #[test]
    fn requires_simd_width_threads_and_static_memory_capacity() {
        assert!(capacity_supported(32, 128, 128, 32768, 32768));
        assert!(!capacity_supported(16, 128, 128, 32768, 32768));
        assert!(!capacity_supported(32, 127, 128, 32768, 32768));
        assert!(!capacity_supported(32, 128, 128, 32769, 32768));
    }
}

#[cfg(test)]
#[path = "tensor_tests.rs"]
mod tests;
