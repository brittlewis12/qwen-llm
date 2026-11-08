use super::*;

#[test]
fn independent_batch_mode_never_substitutes_exact_or_f32_operands() {
    for lineage in [None, Some(PackedLineage::Exact), Some(PackedLineage::Fast)] {
        for precision in [
            None,
            Some(FastPrecision::Half),
            Some(FastPrecision::DenseF32),
            Some(FastPrecision::F32),
        ] {
            assert_eq!(
                batch_mode(lineage, precision),
                lineage == Some(PackedLineage::Fast) && precision == Some(FastPrecision::Half)
            );
        }
    }
}

#[test]
fn independent_batch_projection_checks_format_and_inner_tile_width() {
    for dtype in [
        GgmlType::Q6_K,
        GgmlType::Q8_0,
        GgmlType::F32,
        GgmlType::IQ2_S,
    ] {
        for k in [0, 32, 64, 128, 255, 256, 4096] {
            assert_eq!(
                projection_geometry(dtype, &[k, 288]),
                matches!(dtype, GgmlType::Q6_K | GgmlType::Q8_0)
                    && k > 0
                    && k.is_multiple_of(if dtype == GgmlType::Q8_0 { 64 } else { 256 })
            );
        }
    }
    for dims in [vec![], vec![256], vec![256, 0], vec![256, 128, 2]] {
        assert!(!projection_geometry(GgmlType::Q6_K, &dims));
    }
}

#[path = "batch_model.rs"]
mod model;
