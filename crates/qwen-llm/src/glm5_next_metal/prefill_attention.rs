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

crate::env_flag!(
    /// Default-on M5 Fast/Half selected attention with F32 queries/probabilities and exact F16 cache promotion at irregular 128..=4096 rows, 32..=64 queries. Reuses admitted partial scratch; preserves legacy Half row gates, Exact/F32, ANY_ROWS, SHORT_PREFILL, the enclosing attention and tensor-tier controls. Zero restores split attention at these widths.
    default_on f32_requested,
    "QWEN_GLM53_PREFILL_ATTN_F32"
);

fn eligible_f32(arith: Arith, rows: usize, heads: usize, slots: usize, queries: usize) -> bool {
    arith.lineage == PackedLineage::Fast
        && arith.precision == FastPrecision::Half
        && heads == 64
        && slots == 2051
        && (32..=64).contains(&queries)
        && super::row_policy::expanded(rows)
        && (rows >= 1024 || super::short_prefill::enabled())
        && !eligible(arith, rows, heads, slots, queries)
}

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
    if !requested() || !metal::moe_grouped_tensor_selected(ctx) {
        return Ok(false);
    }
    let half = eligible(arith, rows, g.head_count, g.selected_slots, g.query_count);
    if !half
        && !(f32_requested()
            && eligible_f32(arith, rows, g.head_count, g.selected_slots, g.query_count))
    {
        return Ok(false);
    }
    let alignment = if half { 32 } else { 128 };
    let stride = g.selected_slots.div_ceil(alignment) * alignment;
    let elements = stride * g.head_count * g.query_count;
    if partials.n_elements() < elements as u64 {
        return Ok(false);
    }
    let scores = partials.view_subrange(
        0,
        vec![stride as u64, g.head_count as u64, g.query_count as u64],
    );
    if half {
        metal::try_encode_selected_attention_tensor_f16(ctx, enc, b, g, &scores)
    } else {
        metal::try_encode_selected_attention_tensor_f32(ctx, enc, b, g, &scores)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn f32_attention_retains_modes_legacy_rows_bounds_and_enclosing_policy() {
        let _rows = super::super::row_policy::Scope::new(true);
        let half = Arith {
            lineage: PackedLineage::Fast,
            precision: FastPrecision::Half,
        };
        for rows in [
            128, 129, 400, 511, 513, 1023, 1025, 2000, 2047, 2049, 4095, 4096,
        ] {
            for queries in [32, 48, 63, 64] {
                for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
                    for precision in FastPrecision::ALL {
                        let arith = Arith { lineage, precision };
                        assert_eq!(
                            eligible_f32(arith, rows, 64, 2051, queries),
                            arith == half
                                && (rows >= 1024 || super::super::short_prefill::enabled())
                        );
                    }
                }
            }
        }
        for rows in [0, 1, 127, 1024, 2048, 4097, usize::MAX] {
            assert!(!eligible_f32(half, rows, 64, 2051, 64));
        }
        assert!(!eligible_f32(half, 512, 64, 2051, 64));
        for queries in [0, 1, 16, 31, 65, usize::MAX] {
            assert!(!eligible_f32(half, 2000, 64, 2051, queries));
        }
        for (heads, slots) in [(32, 2051), (128, 2051), (64, 2048), (64, 2052)] {
            assert!(!eligible_f32(half, 2000, heads, slots, 64));
        }
        let _off = super::super::row_policy::Scope::new(false);
        assert!(!eligible_f32(half, 400, 64, 2051, 64));
        assert!(!eligible_f32(half, 2000, 64, 2051, 64));
        assert!(eligible(half, 2048, 64, 2051, 64));
    }

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
