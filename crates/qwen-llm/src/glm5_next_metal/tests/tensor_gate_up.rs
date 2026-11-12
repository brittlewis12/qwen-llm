use super::*;
use crate::glm5_next_metal::packed;
use crate::metal::{GroupedDownPolicy, expert_tensor_gate_up as adapter};

const HALF: Arith = Arith {
    lineage: PackedLineage::Fast,
    precision: FastPrecision::Half,
};

#[test]
fn cpu_short_gate_preserves_exact_f32_and_geometry_contracts() {
    let _small = crate::glm5_next_metal::row_policy::SmallScope::new(false);
    let _scope = memory::TensorGateUpScope::new(true);
    let shape = (4096, 2048, 288, 8);
    for dtype in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
        assert_eq!(
            short_selected(HALF, StageMode::Fast, dtype, dtype, shape, 512),
            crate::glm5_next_metal::short_prefill::enabled() && iq_tensor_gate_up_enabled()
        );
        for rows in [0, 1, 8, 511, 513, 1024, 2048] {
            assert!(!short_selected(
                HALF,
                StageMode::Fast,
                dtype,
                dtype,
                shape,
                rows
            ));
        }
        for mode in [StageMode::Exact, StageMode::FastF32] {
            assert!(!short_selected(HALF, mode, dtype, dtype, shape, 512));
        }
        for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
            for precision in FastPrecision::ALL {
                let arith = Arith { lineage, precision };
                if arith != HALF {
                    assert!(!short_selected(
                        arith,
                        StageMode::Fast,
                        dtype,
                        dtype,
                        shape,
                        512
                    ));
                }
            }
        }
    }
    for (gate, up) in [
        (GgmlType::Q4_K, GgmlType::Q4_K),
        (GgmlType::IQ2_S, GgmlType::IQ3_S),
        (GgmlType::F32, GgmlType::F32),
    ] {
        assert!(!short_selected(HALF, StageMode::Fast, gate, up, shape, 512));
    }
    for other in [
        (4096, 2048, 288, 4),
        (4096, 2048, 256, 8),
        (2048, 2048, 288, 8),
    ] {
        assert!(!short_selected(
            HALF,
            StageMode::Fast,
            GgmlType::IQ2_S,
            GgmlType::IQ2_S,
            other,
            512
        ));
    }
    let _off = memory::TensorGateUpScope::new(false);
    assert!(!short_selected(
        HALF,
        StageMode::Fast,
        GgmlType::IQ2_S,
        GgmlType::IQ2_S,
        shape,
        512
    ));
}

#[test]
fn cpu_gate_up_selection_protects_modes_dtypes_rows_and_production_geometry() {
    let _scope = memory::TensorGateUpScope::new(true);
    for lineage in [PackedLineage::Fast, PackedLineage::Exact] {
        for precision in FastPrecision::ALL {
            let arith = Arith { lineage, precision };
            for mode in [StageMode::Fast, StageMode::FastF32, StageMode::Exact] {
                for gate in [
                    GgmlType::Q4_K,
                    GgmlType::Q5_K,
                    GgmlType::IQ2_S,
                    GgmlType::IQ3_S,
                    GgmlType::Q6_K,
                    GgmlType::F32,
                ] {
                    for up in [
                        GgmlType::Q4_K,
                        GgmlType::Q5_K,
                        GgmlType::IQ2_S,
                        GgmlType::IQ3_S,
                    ] {
                        for rows in [
                            0,
                            1,
                            20,
                            32,
                            33,
                            97,
                            161,
                            512,
                            1023,
                            1024,
                            1025,
                            2047,
                            2048,
                            2049,
                            usize::MAX,
                        ] {
                            assert_eq!(
                                selected(arith, mode, gate, up, (4096, 2048, 288, 8), rows),
                                arith == HALF
                                    && mode == StageMode::Fast
                                    && gate == up
                                    && (matches!(gate, GgmlType::Q4_K | GgmlType::Q5_K)
                                        || (iq_tensor_gate_up_enabled()
                                            && matches!(gate, GgmlType::IQ2_S | GgmlType::IQ3_S)))
                                    && (matches!(rows, 1024 | 2048)
                                        || (matches!(gate, GgmlType::IQ2_S | GgmlType::IQ3_S)
                                            && (crate::glm5_next_metal::row_policy::iq_expanded(
                                                rows,
                                            ) || crate::glm5_next_metal::row_policy::iq_small_expanded(
                                                rows,
                                            ))))
                            );
                        }
                    }
                }
            }
        }
    }
    for shape in [
        (4096, 2048, 16, 8),
        (4096, 2048, 32, 8),
        (2048, 2048, 288, 8),
        (4096, 4096, 288, 8),
        (4096, 2048, 288, 4),
    ] {
        assert!(!selected(
            HALF,
            StageMode::Fast,
            GgmlType::Q4_K,
            GgmlType::Q4_K,
            shape,
            2048
        ));
    }
    let _q5off = crate::glm5_next_metal::tensor_q5_down::Scope::new(false);
    assert_eq!(
        packed::grouped_down_policy(
            HALF,
            StageMode::Fast,
            GgmlType::Q5_K,
            2048,
            4096,
            288,
            8,
            2048
        ),
        GroupedDownPolicy::Incumbent
    );
}

#[test]
fn cpu_gate_up_default_on_scope_is_nested_thread_local_and_unwind_safe() {
    let global = memory::tensor_gate_up_enabled();
    assert_eq!(
        global,
        crate::env_flag::read_default_on("QWEN_GLM53_TENSOR_GATE_UP")
    );
    let _off = memory::TensorGateUpScope::new(false);
    assert!(!memory::tensor_gate_up_enabled());
    {
        let _on = memory::TensorGateUpScope::new(true);
        assert!(memory::tensor_gate_up_enabled());
        let _off = memory::TensorGateUpScope::new(false);
        assert!(!memory::tensor_gate_up_enabled());
    }
    assert!(!memory::tensor_gate_up_enabled());
    assert!(
        std::panic::catch_unwind(|| {
            let _on = memory::TensorGateUpScope::new(true);
            panic!("restore gate/up scope");
        })
        .is_err()
    );
    assert!(!memory::tensor_gate_up_enabled());
    assert_eq!(
        std::thread::spawn(memory::tensor_gate_up_enabled)
            .join()
            .unwrap(),
        global
    );
    // Allocation captures the global policy, never the prefill-only TLS arm.
    assert_eq!(
        memory::ScratchOptions::for_tensor_tier(false, true).tensor_gate_up,
        global
    );
}

#[test]
fn cpu_gate_up_default_rollback_never_enables_kda_or_non_tensor_ownership() {
    let global = crate::env_flag::read_default_on("QWEN_GLM53_TENSOR_GATE_UP");
    assert_eq!(memory::tensor_gate_up_enabled(), global);
    assert_eq!(
        memory::kda_half_inputs_enabled(),
        crate::env_flag::read_default_on("QWEN_GLM53_KDA_HALF_INPUTS")
    );
    let _kda_off = memory::KdaHalfInputsScope::new(false);
    for enabled in [false, true] {
        let _routed = memory::TensorGateUpScope::new(enabled);
        assert!(!memory::kda_half_inputs_enabled());
        assert_eq!(
            selected(
                HALF,
                StageMode::Fast,
                GgmlType::Q4_K,
                GgmlType::Q4_K,
                (4096, 2048, 288, 8),
                1024
            ),
            enabled
        );
        let options = memory::ScratchOptions::for_tensor_tier(false, true);
        assert_eq!(options.tensor_gate_up, global);
        assert!(!options.kda_half_inputs);
    }
    assert_eq!(
        memory::ScratchOptions::for_tensor_tier(false, false),
        memory::ScratchOptions::default()
    );
}

#[test]
fn cpu_gate_up_geometry_and_capacity_are_checked() {
    for e in [16, 32, 288] {
        assert!(adapter::geometry(4096, 2048, e, 8, 2048).is_ok());
    }
    for (h, f, e, top, rows) in [
        (0, 256, 16, 8, 32),
        (255, 256, 16, 8, 32),
        (256, 255, 16, 8, 32),
        (256, 256, 0, 8, 32),
        (256, 256, 16, 17, 32),
        (256, 256, 16, 8, 0),
        (usize::MAX, 256, 16, 8, 32),
        (4096, 2048, 288, 8, usize::MAX),
        (4096, 2048, 288, 8, i32::MAX as usize),
    ] {
        assert!(adapter::geometry(h, f, e, top, rows).is_err());
    }
    assert!(adapter::capacity(32, 128, 0, 8192, 8192, 128));
    for (w, t, s, d, l, r) in [
        (16, 128, 0, 8192, 8192, 128),
        (32, 127, 0, 8192, 8192, 128),
        (32, 128, 1, 8192, 8192, 128),
        (32, 128, usize::MAX, 8192, usize::MAX, 128),
        (32, 128, 0, 0, 8192, 256),
    ] {
        assert!(!adapter::capacity(w, t, s, d, l, r));
    }
}

#[test]
fn cpu_explicit_adapter_admits_each_down_policy_by_dtype() {
    use adapter::{DownRoute, down_route};
    for policy in [
        GroupedDownPolicy::Incumbent,
        GroupedDownPolicy::Iq3SSmallCounts,
        GroupedDownPolicy::Iq3STensor,
        GroupedDownPolicy::Iq3STensorOrSmallCounts,
        GroupedDownPolicy::Q5KTensor,
    ] {
        for dtype in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::IQ2_S,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
            GgmlType::Q8_0,
        ] {
            let expected = match (dtype, policy) {
                (GgmlType::Q5_K, GroupedDownPolicy::Q5KTensor) => Some(DownRoute::Tensor),
                (GgmlType::IQ3_S, GroupedDownPolicy::Iq3STensor) => Some(DownRoute::Tensor),
                (GgmlType::IQ3_S, GroupedDownPolicy::Iq3STensorOrSmallCounts) => {
                    Some(DownRoute::TensorOrSmallCounts)
                }
                (
                    GgmlType::IQ3_S | GgmlType::IQ4_XS | GgmlType::Q6_K,
                    GroupedDownPolicy::Incumbent,
                ) => Some(DownRoute::Incumbent),
                _ => None,
            };
            assert_eq!(down_route(dtype, policy), expected, "{dtype:?}/{policy:?}");
        }
    }
    // The grouped adapter accepts this pairing, so the explicit adapter must
    // not decline it. This is the regression that silently took short appends
    // through the F32-input grouped gate/up.
    assert_eq!(
        down_route(GgmlType::IQ3_S, GroupedDownPolicy::Iq3STensorOrSmallCounts),
        Some(DownRoute::TensorOrSmallCounts)
    );
}

#[test]
fn cpu_glm_small_append_policy_maps_to_the_explicit_adapter() {
    use crate::glm5_next_metal::row_policy;
    use adapter::{DownRoute, down_route};
    for rows in [128, 161, 256, 384, 511, 512] {
        let policy = packed::grouped_down_policy(
            HALF,
            StageMode::Fast,
            GgmlType::IQ3_S,
            2048,
            4096,
            288,
            8,
            rows,
        );
        assert!(
            down_route(GgmlType::IQ3_S, policy).is_some(),
            "explicit adapter declined IQ3_S/{policy:?} at {rows} rows"
        );
        if rows == 512 || row_policy::small_expanded(rows) {
            assert_eq!(policy, GroupedDownPolicy::Iq3STensorOrSmallCounts);
            assert_eq!(
                down_route(GgmlType::IQ3_S, policy),
                Some(DownRoute::TensorOrSmallCounts)
            );
        }
    }
    // The 32-row route was already explicit.
    let policy = packed::grouped_down_policy(
        HALF,
        StageMode::Fast,
        GgmlType::IQ3_S,
        2048,
        4096,
        288,
        8,
        32,
    );
    assert_eq!(policy, GroupedDownPolicy::Iq3STensor);
    assert_eq!(down_route(GgmlType::IQ3_S, policy), Some(DownRoute::Tensor));
    // IQ4_XS-down exception blocks keep the incumbent down policy and remain
    // on the explicit composition.
    let policy = packed::grouped_down_policy(
        HALF,
        StageMode::Fast,
        GgmlType::IQ4_XS,
        2048,
        4096,
        288,
        8,
        161,
    );
    assert_eq!(policy, GroupedDownPolicy::Incumbent);
    assert_eq!(
        down_route(GgmlType::IQ4_XS, policy),
        Some(DownRoute::Incumbent)
    );
}
