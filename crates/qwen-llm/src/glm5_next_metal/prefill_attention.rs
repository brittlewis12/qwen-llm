//! Qualified sparse-attention scheduling; reuses admitted split scratch.
use super::packed::{Arith, FastPrecision, PackedLineage};
use crate::metal::{
    self, KernelEncoder, MetalContext, MetalError, MetalTensor, SelectedAttention,
    SelectedAttentionShape,
};

crate::env_flag!(
    /// Default-on M5 Fast/Half QK/softmax/PV tensor attention at 512/1024/2048 command rows and 32..=64 sparse queries per batch. Reuses the split attention partial buffer. Zero restores split attention; Exact/F32, the global tensor rollback, and the 512-row short-prefill rollback retain their routes. No new GPU buffer or cache format.
    default_on requested,
    "QWEN_GLM53_PREFILL_ATTN"
);

fn eligible(arith: Arith, rows: usize, heads: usize, slots: usize, queries: usize) -> bool {
    arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && heads == 64
        && slots == 2051
        && (32..=64).contains(&queries)
        && (matches!(rows, 1024 | 2048) || (rows == 512 && super::short_prefill::enabled()))
}

pub(super) fn try_encode(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    arith: Arith,
    rows: usize,
    b: &SelectedAttention<'_>,
    g: SelectedAttentionShape,
    partials: &MetalTensor,
) -> Result<bool, MetalError> {
    if !requested()
        || !eligible(arith, rows, g.head_count, g.selected_slots, g.query_count)
        || !metal::moe_grouped_tensor_selected(ctx)
    {
        return Ok(false);
    }
    let stride = g.selected_slots.div_ceil(32) * 32;
    let elements = stride * g.head_count * g.query_count;
    if partials.n_elements() < elements as u64 {
        return Ok(false);
    }
    let scores = partials.view_subrange(
        0,
        vec![stride as u64, g.head_count as u64, g.query_count as u64],
    );
    metal::try_encode_selected_attention_tensor_f16(ctx, enc, b, g, &scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserves_exact_f32_small_batches_and_release_geometry() {
        let half = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
        };
        for rows in [1024, 2048] {
            for queries in [32, 48, 61, 64] {
                assert!(eligible(half, rows, 64, 2051, queries));
            }
        }
        assert_eq!(
            eligible(half, 512, 64, 2051, 64),
            super::super::short_prefill::enabled()
        );
        for queries in [0, 1, 16, 31, 65, 128] {
            assert!(!eligible(half, 2048, 64, 2051, queries));
        }
        for rows in [0, 1, 128, 256, 511, 513, 1023, 1025, 2047, 2049, 4096] {
            assert!(!eligible(half, rows, 64, 2051, 64));
        }
        for (heads, slots) in [(32, 2051), (128, 2051), (64, 2048), (64, 2052)] {
            assert!(!eligible(half, 2048, heads, slots, 64));
        }
        for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
            for precision in FastPrecision::ALL {
                let a = Arith { lineage, precision };
                if a != half {
                    assert!(!eligible(a, 2048, 64, 2051, 64));
                }
            }
        }
    }
}
