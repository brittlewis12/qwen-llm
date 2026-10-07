//! CPU-only admission tests: no Metal device or archived weights are opened.

use super::*;

#[test]
fn gsq_projection_storage_admission() {
    for (dtype, block, bytes) in [
        (GgmlType::F32, 1, 4),
        (GgmlType::BF16, 1, 2),
        (GgmlType::Q8_0, 32, 34),
        (GgmlType::Q4_K, 256, 144),
        (GgmlType::Q5_K, 256, 176),
        (GgmlType::Q6_K, 256, 210),
        (GgmlType::IQ4_XS, 256, 136),
        (GgmlType::IQ4_NL, 32, 18),
    ] {
        let width = 2 * block;
        let storage = 6 * bytes;
        let offset = if dtype == GgmlType::F32 { 4 } else { 2 };
        assert!(projection_dtype_supported(dtype));
        validate_projection_range(dtype, width, 3, offset, offset + storage).unwrap();
        assert!(validate_projection_range(dtype, width, 3, offset, offset + storage - 1).is_err());
        assert!(validate_projection_range(dtype, width, 3, 1, storage + 1).is_err());
        assert!(validate_projection_range(dtype, width, 3, u64::MAX - 3, u64::MAX).is_err());
        if block > 1 {
            // Total elements align, but each row does not.
            assert!(validate_projection_range(dtype, block / 2, 2, 0, bytes).is_err());
        }
        for tokens in [1, 2, 4, 8, 16, 64, 2048] {
            validate_projection_addressing(dtype, width, 3, tokens).unwrap();
        }
    }
    // HC low-rank BF16 rows need no quantization block padding.
    validate_projection_range(GgmlType::BF16, 7, 3, 2, 44).unwrap();
}

#[test]
fn gsq_projection_addressing_rejects_overflow_and_unsupported_types() {
    for &dtype in PROJECTION_DTYPES {
        for (n_in, n_out, tokens) in [
            (0, 1, 1),
            (256, 0, 1),
            (256, 1, 0),
            (65_536, 65_536, 1),
            (65_536, 1, 65_536),
            (256, 65_536, 65_536),
            (usize::MAX, 2, 1),
        ] {
            assert!(validate_projection_addressing(dtype, n_in, n_out, tokens).is_err());
        }
    }
    for dtype in [GgmlType::F16, GgmlType::IQ3_S, GgmlType::I32] {
        assert!(!projection_dtype_supported(dtype));
        assert!(projection_kernel_names(dtype, false, true).is_none());
        assert!(projection_kernel_names(dtype, true, true).is_none());
        assert!(validate_projection_range(dtype, 256, 1, 0, 4096).is_err());
    }
}

#[test]
fn gsq_projection_preflights_cover_native_dispatch_variants() {
    for &dtype in PROJECTION_DTYPES {
        for packed in [false, true] {
            assert!(
                !projection_kernel_names(dtype, packed, true)
                    .unwrap()
                    .is_empty()
            );
        }
    }
    assert!(projection_kernel_names(GgmlType::BF16, false, false).is_none());
    assert!(projection_kernel_names(GgmlType::BF16, true, false).is_none());
    assert!(
        projection_kernel_names(GgmlType::Q8_0, true, false)
            .unwrap()
            .iter()
            .all(|name| !name.contains("fewrow"))
    );
    for (dtype, packed, expected) in [
        (GgmlType::F32, false, "kernel_mat_vec_f32_f32_lcpp_r2"),
        (GgmlType::Q8_0, false, "kernel_mat_vec_q8_0_f32_lcpp"),
        (GgmlType::Q8_0, true, "kernel_mat_mat_q8_0_f32"),
        (GgmlType::Q8_0, true, "kernel_mat_mat_q8_0_f32_n16"),
        (
            GgmlType::Q8_0,
            true,
            "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
        ),
        (GgmlType::Q4_K, true, "kernel_mat_vec_q4_K_nc2_rp4_f32"),
        (
            GgmlType::Q4_K,
            true,
            "kernel_mat_mat_q4_K_mma8v_r2c1k64_vec4_f32",
        ),
        (GgmlType::Q5_K, true, "kernel_mat_mat_q5_K_f32_n64"),
        (GgmlType::Q6_K, true, "kernel_mat_mat_q6_K_f32_n64"),
        (GgmlType::IQ4_XS, false, "kernel_mat_vec_iq4_xs_f32_fast"),
        (GgmlType::IQ4_XS, true, "kernel_mat_mat_iq4_xs_f32_mm"),
        (GgmlType::IQ4_NL, false, "kernel_mat_vec_iq4_nl_f32_fast"),
        (GgmlType::IQ4_NL, true, "kernel_mat_mat_iq4_nl_f32_mm"),
        (GgmlType::BF16, true, "kernel_mat_mat_bf16_f32"),
        (GgmlType::BF16, true, "kernel_mat_mat_bf16_bfloat_act_f32"),
    ] {
        assert!(
            projection_kernel_names(dtype, packed, true)
                .unwrap()
                .contains(&expected)
        );
    }
}

#[test]
fn gsq_token_embedding_preflight_retains_q8_and_admits_iq4_xs() {
    assert_eq!(
        token_embedding_kernel_name(GgmlType::Q8_0),
        Some("kernel_get_rows_q8_0_f32")
    );
    assert_eq!(
        token_embedding_kernel_name(GgmlType::IQ4_XS),
        Some("kernel_get_rows_iq4_xs_f32")
    );
    assert_eq!(token_embedding_kernel_name(GgmlType::IQ3_S), None);
}
