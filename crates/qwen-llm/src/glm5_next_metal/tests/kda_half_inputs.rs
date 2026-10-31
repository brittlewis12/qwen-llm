use super::*;
use crate::glm5_next_metal::packed::{Arith, FastPrecision, PackedLineage, Stage, StageMode};

#[test]
fn cpu_kda_half_scope_restores_nested_panics_and_other_threads() {
    let runtime = memory::kda_half_inputs_enabled();
    assert_eq!(
        runtime,
        crate::env_flag::read_default_off("QWEN_GLM53_KDA_HALF_INPUTS")
    );
    let _off = memory::KdaHalfInputsScope::new(false);
    assert!(!memory::kda_half_inputs_enabled());
    let unwind = std::panic::catch_unwind(|| {
        let _on = memory::KdaHalfInputsScope::new(true);
        assert!(memory::kda_half_inputs_enabled());
        {
            let _nested = memory::KdaHalfInputsScope::new(false);
            assert!(!memory::kda_half_inputs_enabled());
        }
        assert!(memory::kda_half_inputs_enabled());
        panic!("restore candidate only");
    });
    assert!(unwind.is_err());
    assert!(!memory::kda_half_inputs_enabled());
    assert_eq!(
        std::thread::spawn(memory::kda_half_inputs_enabled)
            .join()
            .unwrap(),
        runtime
    );
}

#[test]
fn cpu_kda_half_selection_respects_resolved_mode_and_actual_packet_rows() {
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            let arith = Arith { lineage, precision };
            for dtype in [
                GgmlType::Q8_0,
                GgmlType::Q6_K,
                GgmlType::F32,
                GgmlType::F16,
                GgmlType::Q4_K,
            ] {
                for width in [8192, 128, 64] {
                    let mode =
                        packed::projection_mode(arith, Stage::KdaProjection, dtype, 4096, width);
                    for rows in [0, 1, 8, 512, 1023, 1024, 1025, 2047, 2048] {
                        assert_eq!(
                            tensor_prefill::eligible(arith, mode, dtype, 4096, width, rows)
                                .unwrap(),
                            lineage == PackedLineage::Fast
                                && precision == FastPrecision::Half
                                && mode == StageMode::Fast
                                && matches!(dtype, GgmlType::Q8_0 | GgmlType::Q6_K)
                                && rows >= 1024
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_kda_half_physical_extent_checks_original_backing_not_packet() {
    assert!(backing_extent(&[4096, 2048], 256, 256 + (16 << 20), 4096, 1025).is_ok());
    for (shape, offset, physical, rows) in [
        (vec![4096, 2048], 256, 256 + (16 << 20) - 1, 1024),
        (vec![4096, 2048], 2, 32 << 20, 1024),
        (vec![4096, 1024], 0, 32 << 20, 1025),
        (vec![4096, 2048], u64::MAX - 15, u64::MAX, 1024),
        (vec![4096, u64::MAX], 0, u64::MAX, 1024),
        (vec![4096], 0, 32 << 20, 1024),
    ] {
        assert!(backing_extent(&shape, offset, physical, 4096, rows).is_err());
    }
}
