use super::*;
use crate::metal::admit_iq1_expert_fixture;

#[test]
fn iq1_native_cpu_codec_real_k_and_route_contract() {
    let _lease = production_lease();
    let ctx = metal_test_context().expect("IQ1 expert GPU proof requires Metal");
    admit_iq1_expert_fixture(&ctx);
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        for k in [256, 2304, 5120] {
            for status in [ROUTE_STATUS_READY, ROUTE_STATUS_PENDING, -7, 2] {
                gate_up_case(&ctx, dtype, dtype, k, 7, status);
            }
        }
    }
}

#[test]
fn iq1_native_rows_are_scalar_lineage_not_compact_packed() {
    let _lease = production_lease();
    let ctx = metal_test_context().expect("IQ1 expert GPU proof requires Metal");
    admit_iq1_expert_fixture(&ctx);
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        assert!(all_slots_gate_up_native_rows_supported(dtype, dtype));
        assert!(all_slots_down_native_rows_supported(dtype));
        for (k, m) in [(256, 5), (2304, 7), (5120, 9)] {
            for n in [1, 5] {
                for down in [false, true] {
                    let scalar = rows::fixture(&ctx, dtype, down, false, n, k, m, 3, None);
                    let batched = rows::fixture(&ctx, dtype, down, false, n, k, m, 3, Some(n));
                    assert_eq!(scalar, batched, "{dtype:?} down={down} K={k} N={n}");
                }
            }
        }
    }
}

#[test]
fn iq1_native_high_expert_ids_down_and_gate_cpu_codec() {
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        large_banks_case(dtype, &[256, 2304, 5120]);
    }
}

#[test]
fn iq1_canonical_decode_is_shared_without_dense_math_changes() {
    let header = include_str!("../../../../../../kernels/iq1_decode.metalh");
    let dense = include_str!("../../../../../../kernels/iq1.metal");
    let tiles = include_str!("../../../../../../kernels/quant_tiles.h");
    for src in [SOURCE, dense, tiles] {
        assert!(src.contains("#include \"iq1_decode.metalh\""));
        assert!(!src.contains("inline constant char * iq1_grid8"));
    }
    assert!(header.contains("inline float iq1_value("));
    assert!(header.contains("inline void iq1_dequant16("));
    assert!(!header.contains("device const uint *"));
    assert!(!header.contains("device const ulong *"));
    assert!(!header.contains("device const float *"));
}

#[test]
fn iq1_native_scalar_and_rows_atomic_preflight_refusals() {
    for (g, u, d) in [
        (GgmlType::IQ1_S, GgmlType::IQ1_M, GgmlType::IQ1_S),
        (GgmlType::IQ1_M, GgmlType::IQ1_S, GgmlType::IQ1_M),
        (GgmlType::IQ1_M, GgmlType::IQ1_M, GgmlType::IQ2_XXS),
        (GgmlType::IQ2_XS, GgmlType::IQ2_XS, GgmlType::IQ1_M),
    ] {
        native_validation_case(g, u);
        rows::preflight_case(g, u, d);
    }
}

#[test]
fn iq1_frozen_compiled_identity() {
    use sha2::{Digest, Sha256};
    let sources: &[(&str, &[u8])] = &[
        (
            "iq1_decode",
            include_bytes!("../../../../../../kernels/iq1_decode.metalh"),
        ),
        (
            "iq1_grid",
            include_bytes!("../../../../../../kernels/iq1_grid.metalh"),
        ),
        (
            "iq1_dense",
            include_bytes!("../../../../../../kernels/iq1.metal"),
        ),
        (
            "expert_native_metal",
            include_bytes!("../../../../../../kernels/expert_native.metal"),
        ),
        (
            "quant_tiles",
            include_bytes!("../../../../../../kernels/quant_tiles.h"),
        ),
        ("moe", include_bytes!("../../../../../../kernels/moe.metal")),
        (
            "tensor_moe",
            include_bytes!("../../../../../../kernels/tensor/moe_grouped_slots_tensor.metal"),
        ),
        ("native_rs", include_bytes!("../../expert_native.rs")),
        ("grouped_rs", include_bytes!("../../moe_grouped_generic.rs")),
        (
            "tensor_rs",
            include_bytes!("../../moe_grouped_generic/tensor.rs"),
        ),
        ("build_rs", include_bytes!("../../../../build.rs")),
        ("product_metallib", crate::KERNELS_METALLIB),
        ("tensor_metallib", crate::KERNELS_TENSOR_METALLIB),
    ];
    for (name, bytes) in sources {
        eprintln!("IQ1_IDENTITY {name} sha256={:x}", Sha256::digest(bytes));
    }
    eprintln!(
        "IQ1_IDENTITY tensor_tier={}",
        env!("QWEN_TENSOR_METALLIB_TIERS")
    );
    eprintln!(
        "IQ1_IDENTITY compiler={}",
        env!("QWEN_METALLIB_COMPILER_IDENTITY")
    );
    eprintln!("IQ1_IDENTITY sdk={}", env!("QWEN_METALLIB_SDK_VERSION"));
}
