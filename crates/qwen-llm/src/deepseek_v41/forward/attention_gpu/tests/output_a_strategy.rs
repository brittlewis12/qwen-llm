use super::*;

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
    let _lease = metal::acquire_metal_benchmark_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    let schedule = Schedule {
        layers: 1,
        kv_source_layers: vec![],
        index_source_layers: vec![],
        compression_ratios: vec![0],
        candidate_source_layer: None,
    };
    for groups in [1, 8, 16, 17] {
        let c = Config {
            hidden: 32,
            intermediate: 32,
            query_rank: 32,
            output_rank: 32,
            index_heads: 1,
            index_dim: 32,
            index_topk: 2,
            window: 2,
            rope_dim: 0,
            candidate_block: 2,
            candidate_topk: 1,
            norm_eps: 1e-6,
            ..config(groups)
        };
        let reference = scheduled_weights(&ctx, &c, &schedule, 0, GgmlType::F32);
        let mut weights = scheduled_weights(&ctx, &c, &schedule, 0, GgmlType::F32);
        let values: Vec<_> = reference
            .output_a
            .iter()
            .flat_map(|p| f32_slice(&p.tensor).unwrap().iter().copied())
            .collect();
        let width = c.heads / c.groups * c.head_dim;
        let bank = upload(
            &ctx,
            &values,
            vec![width as u64, c.output_rank as u64, groups as u64],
        )
        .unwrap();
        for (group, projection) in weights.output_a.iter_mut().enumerate() {
            projection.tensor = bank.view_bytes(
                (group * width * c.output_rank * 4) as u64,
                vec![width as u64, c.output_rank as u64],
            );
        }
        assert_eq!(
            Workspace::output_a_bank(&weights, &c).is_some(),
            groups <= 16
        );
        let mut actual = Workspace::new_with_schedule(&ctx, &c, 2, &[&weights], &schedule).unwrap();
        let mut expected =
            Workspace::new_with_schedule(&ctx, &c, 2, &[&reference], &schedule).unwrap();
        for position in 0..2 {
            let x: Vec<_> = (0..c.hidden)
                .map(|i| (i as f32 - 15.0 + position as f32) / 31.0)
                .collect();
            let input = upload(&ctx, &x, vec![c.hidden as u64]).unwrap();
            metal::dispatch_census_begin();
            execute(&ctx, |enc| {
                actual
                    .encode(&ctx, enc, 0, position, &weights, &input)
                    .unwrap();
            });
            let census = metal::dispatch_census_take();
            let native = census
                .iter()
                .filter(|r| r.kernel.starts_with("kernel_expert_native_"))
                .count();
            assert_eq!(native, usize::from(groups <= 16));
            execute(&ctx, |enc| {
                expected
                    .encode(&ctx, enc, 0, position, &reference, &input)
                    .unwrap();
            });
            close(
                f32_slice(&actual.low).unwrap(),
                f32_slice(&expected.low).unwrap(),
            );
            close(
                f32_slice(&actual.output).unwrap(),
                f32_slice(&expected.output).unwrap(),
            );
        }
    }
}
