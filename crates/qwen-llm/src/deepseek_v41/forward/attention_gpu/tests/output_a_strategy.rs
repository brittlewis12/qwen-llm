use super::*;

mod fixture;
mod fixture_contracts;

fn config(groups: usize) -> Config {
    Config {
        groups,
        heads: groups * 2,
        head_dim: 32,
        rope_dim: 32,
        output_rank: 7,
        ..Config::default()
    }
}

#[test]
fn native_group_count_limit_preserves_general_fallback() {
    for groups in [1, 8, 16, 17, 32, 64] {
        let c = config(groups);
        crate::deepseek_v41::execution::validate(&c).unwrap();
        for dtype in [GgmlType::F32, GgmlType::F16, GgmlType::BF16, GgmlType::Q8_0] {
            assert_eq!(output_a_native_layout(&c, dtype, 0).is_some(), groups <= 16);
        }
    }
}

#[test]
fn native_layout_checks_dtype_alignment_and_quantized_rows() {
    let c = config(8);
    let layout = output_a_native_layout(&c, GgmlType::Q8_0, 16).unwrap();
    assert_eq!(
        (layout.group_input, layout.group_bytes, layout.bank_bytes),
        (64, 2 * 34 * 7, 2 * 34 * 7 * 8)
    );
    assert!(output_a_native_layout(&c, GgmlType::Q3_K, 0).is_none());
    assert!(output_a_native_layout(&c, GgmlType::I32, 0).is_none());
    for offset in [1, 4, 8, u64::MAX - 15] {
        assert!(output_a_native_layout(&c, GgmlType::F32, offset).is_none());
    }
    let c = Config { head_dim: 128, ..c };
    assert!(output_a_native_layout(&c, GgmlType::Q3_K, 0).is_some());
}

#[test]
fn native_layout_rejects_malformed_and_overflowing_limits_without_panics() {
    let c = config(8);
    for bad in [
        Config {
            groups: 0,
            ..c.clone()
        },
        Config {
            heads: 0,
            ..c.clone()
        },
        Config {
            heads: 17,
            ..c.clone()
        },
        Config {
            head_dim: 0,
            ..c.clone()
        },
        Config {
            output_rank: 0,
            ..c.clone()
        },
        Config {
            head_dim: usize::MAX,
            ..c.clone()
        },
        Config {
            output_rank: usize::MAX,
            ..c.clone()
        },
        Config {
            heads: u32::MAX as usize,
            groups: 1,
            ..c.clone()
        },
        Config {
            heads: 16,
            head_dim: u32::MAX as usize,
            output_rank: u32::MAX as usize,
            ..c
        },
    ] {
        assert!(output_a_native_layout(&bad, GgmlType::F32, 0).is_none());
    }
}

#[test]
#[ignore = "bounded Metal qualification for the parent M5 lane; never a CPU regression"]
fn grouped_decode_output_a_native_and_general_fallback_match() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    for groups in fixture::GROUPS {
        for dtype in fixture::DTYPES {
            fixture::run_case(&ctx, groups, dtype);
        }
    }
}
