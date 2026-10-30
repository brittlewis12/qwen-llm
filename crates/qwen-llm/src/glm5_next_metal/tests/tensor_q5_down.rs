use super::*;
use crate::glm5_next_metal::packed::grouped_down_policy;
use crate::metal::GroupedDownPolicy;

const HALF: Arith = Arith {
    lineage: PackedLineage::Fast,
    precision: FastPrecision::Half,
};

#[test]
fn cpu_q5_down_policy_bounds_arithmetic_dtype_geometry_topk_and_committed_rows() {
    let _on = Scope::new(true);
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            for mode in [StageMode::Fast, StageMode::FastF32, StageMode::Exact] {
                for dtype in [
                    GgmlType::Q5_K,
                    GgmlType::Q4_K,
                    GgmlType::IQ3_S,
                    GgmlType::Q8_0,
                    GgmlType::F16,
                    GgmlType::F32,
                    GgmlType::Unknown,
                ] {
                    for (k, m, e, top) in [
                        (2048, 4096, 288, 8),
                        (2047, 4096, 288, 8),
                        (2049, 4096, 288, 8),
                        (2048, 4095, 288, 8),
                        (2048, 4097, 288, 8),
                        (2048, 4096, 287, 8),
                        (2048, 4096, 289, 8),
                        (2048, 4096, 32, 8),
                        (2048, 4096, 288, 0),
                        (2048, 4096, 288, 7),
                        (2048, 4096, 288, 9),
                        (2048, 4096, 288, 16),
                    ] {
                        for rows in [
                            0,
                            1,
                            32,
                            512,
                            1023,
                            1024,
                            1025,
                            2047,
                            2048,
                            2049,
                            4096,
                            usize::MAX,
                        ] {
                            let arith = Arith { lineage, precision };
                            let expected = arith == HALF
                                && mode == StageMode::Fast
                                && dtype == GgmlType::Q5_K
                                && (k, m, e, top) == (2048, 4096, 288, 8)
                                && matches!(rows, 1024 | 2048);
                            assert_eq!(
                                selected(arith, mode, dtype, (k, m, e, top), rows),
                                expected
                            );
                            let policy =
                                grouped_down_policy(arith, mode, dtype, k, m, e, top, rows);
                            assert_eq!(policy == GroupedDownPolicy::Q5KTensor, expected);
                            if dtype == GgmlType::Q5_K && !expected {
                                assert_eq!(policy, GroupedDownPolicy::Incumbent);
                            }
                        }
                    }
                }
            }
        }
    }
    let _off = Scope::new(false);
    for rows in [1024, 2048] {
        assert_eq!(
            grouped_down_policy(
                HALF,
                StageMode::Fast,
                GgmlType::Q5_K,
                2048,
                4096,
                288,
                8,
                rows
            ),
            GroupedDownPolicy::Incumbent
        );
        assert_eq!(
            grouped_down_policy(
                HALF,
                StageMode::Fast,
                GgmlType::IQ3_S,
                2048,
                4096,
                288,
                8,
                rows
            ),
            GroupedDownPolicy::Iq3STensor
        );
    }
}

#[test]
fn cpu_q5_down_default_on_scope_is_nested_thread_local_and_unwind_safe() {
    let runtime = crate::env_flag::read_default_on("QWEN_GLM53_TENSOR_Q5_DOWN");
    assert_eq!(requested(), runtime);
    assert_eq!(enabled(), runtime);
    assert_eq!(
        selected(
            HALF,
            StageMode::Fast,
            GgmlType::Q5_K,
            (2048, 4096, 288, 8),
            2048
        ),
        runtime
    );
    {
        let _on = Scope::new(true);
        assert!(enabled());
        {
            let _off = Scope::new(false);
            assert!(!enabled());
        }
        assert!(enabled());
        assert!(
            std::panic::catch_unwind(|| {
                let _off = Scope::new(false);
                assert!(!enabled());
                panic!("restore Q5 down policy");
            })
            .is_err()
        );
        assert!(enabled());
        std::thread::spawn(|| {
            let runtime = crate::env_flag::read_default_on("QWEN_GLM53_TENSOR_Q5_DOWN");
            assert_eq!(enabled(), runtime);
            {
                let _on = Scope::new(true);
                assert!(enabled());
            }
            assert_eq!(enabled(), runtime);
        })
        .join()
        .unwrap();
        assert!(enabled());
    }
    assert_eq!(enabled(), runtime);
}
