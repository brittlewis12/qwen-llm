//! Focused row-lineage and controlled batched arithmetic tests.
use super::*;

#[test]
fn original_scalar_tail_matches_the_actual_full_chunk_schedule() {
    for n in 0..=3 * DEFAULT_CHUNK_ROWS + 2 {
        let last = n % DEFAULT_CHUNK_ROWS;
        assert_eq!(original_scalar_tail(n), (last == 1).then(|| n - 1));
        if let Some(position) = original_scalar_tail(n) {
            assert_eq!(position + 1, n);
            assert!(position.is_multiple_of(DEFAULT_CHUNK_ROWS));
        }
    }
    assert_eq!(original_scalar_tail(0), None);
    assert_eq!(original_scalar_tail(1), Some(0));
    assert_eq!(original_scalar_tail(2048), None);
    assert_eq!(original_scalar_tail(2049), Some(2048));
    assert_eq!(original_scalar_tail(4097), Some(4096));
    let n = usize::MAX - usize::MAX % DEFAULT_CHUNK_ROWS + 1;
    assert_eq!(original_scalar_tail(n), Some(n - 1));
}

#[test]
fn packed_lineage_changes_only_nonempty_singleton_dense_eligibility() {
    for dtype in [
        GgmlType::F32,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q6_K,
        GgmlType::IQ2_XXS,
        GgmlType::IQ1_S,
        GgmlType::I32,
    ] {
        for input in [31usize, 32, 256, 4096] {
            let geometry = dtype != GgmlType::F32
                && crate::metal_forward::mat_mat_dispatch_supported(dtype)
                && input.is_multiple_of(32);
            for rows in [0, 1, 2, 33, 129, 2048] {
                for (policy, eligible) in [
                    (RowsArithmetic::Ordinary, rows > 1),
                    (RowsArithmetic::Packed, rows > 0),
                ] {
                    assert_eq!(
                        mma_linear_rows_selected(dtype, input, rows, policy),
                        geometry && eligible,
                        "{dtype:?} K={input} N={rows} {policy:?}"
                    );
                    with_f32_linears(|| {
                        assert!(!mma_linear_rows_selected(dtype, input, rows, policy));
                    });
                    with_rowwise_linears(|| {
                        assert!(!mma_linear_rows_selected(dtype, input, rows, policy));
                    });
                }
            }
        }
    }
}

#[test]
fn batched_f32_scope_nests_is_thread_local_and_restores_on_unwind() {
    let selected = || mma_linear_rows_selected(GgmlType::BF16, 32, 1, RowsArithmetic::Packed);
    assert!(selected());
    with_batched_f32(|| {
        assert!(!selected());
        with_batched_f32(|| assert!(!selected()));
        assert!(!selected());
        assert!(std::thread::spawn(selected).join().unwrap());
    });
    assert!(selected());
    with_rowwise_linears(|| {
        with_batched_f32(|| assert!(ROWWISE_LINEAR_ROWS.with(|flag| flag.get())));
        assert!(!selected());
    });
    assert!(std::panic::catch_unwind(|| with_batched_f32(|| panic!("restore"))).is_err());
    assert!(selected());
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn singleton_dense_lineage_uses_real_views_and_ordinary_wrapper_is_unchanged() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("arithmetic GPU lease");
    let ctx = MetalContext::new().unwrap();
    let data: Vec<_> = (0..32 * 17)
        .map(|i| half::bf16::from_f32(((i * 19 % 101) as f32 - 50.0) / 103.0).to_bits())
        .collect();
    let linear = Linear {
        ctx: &ctx,
        tensor: MetalTensor::from_bytes(
            &ctx,
            bytemuck::cast_slice(&data),
            vec![32, 17],
            GgmlType::BF16,
        )
        .unwrap(),
        native_control: None,
        input: 32,
        output: 17,
    };
    for n in [1, 33] {
        let x = upload(
            &ctx,
            &(0..32 * n)
                .map(|i| ((i * 31 % 113) as f32 - 56.0) / 71.0)
                .collect::<Vec<_>>(),
            vec![32, n as u64],
        )
        .unwrap();
        let backing = upload(&ctx, &vec![-777.0; 17 * n + 8], vec![(17 * n + 8) as u64]).unwrap();
        let y = backing.view_subrange(4, vec![17, n as u64]);
        let encode = |policy: Option<RowsArithmetic>, f32: bool| {
            let run = || {
                metal::dispatch_census_begin();
                let result = submit(&ctx, "dense_arithmetic", [n, 0, 0, 0], |enc| match policy {
                    None => linear.encode_rows(enc, &x, &y, n),
                    Some(policy) => linear.encode_rows_with_arithmetic(enc, &x, &y, n, policy),
                });
                let census = metal::dispatch_census_take();
                result.unwrap();
                assert!(f32_slice(&y).unwrap().iter().all(|v| v.is_finite()));
                let values = f32_slice(&backing).unwrap();
                assert!(
                    values[..4]
                        .iter()
                        .chain(&values[4 + 17 * n..])
                        .all(|v| *v == -777.0)
                );
                (
                    census,
                    f32_slice(&y)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                )
            };
            if f32 { with_batched_f32(run) } else { run() }
        };
        let (wrapper, expected) = encode(None, false);
        let (ordinary, actual) = encode(Some(RowsArithmetic::Ordinary), false);
        assert_eq!(actual, expected);
        assert_eq!(
            ordinary.iter().map(|d| &d.kernel).collect::<Vec<_>>(),
            wrapper.iter().map(|d| &d.kernel).collect::<Vec<_>>()
        );
        if n == 1 {
            assert_eq!(
                wrapper[0].kernel,
                metal::grouped_projection_pipeline_name(GgmlType::BF16).unwrap()
            );
        }
        let (packed, _) = encode(Some(RowsArithmetic::Packed), false);
        assert_eq!(packed.len(), 1);
        if metal::moe_grouped_tensor_selected(&ctx) {
            assert_eq!(
                packed[0].kernel,
                metal::mat_mat_tensor_pipeline_name(GgmlType::BF16, metal::TensorActivation::F32)
                    .unwrap()
            );
        } else {
            assert!(packed[0].kernel.starts_with("kernel_mat_mat_bf16"));
        }
        let (f32, _) = encode(Some(RowsArithmetic::Packed), true);
        assert_eq!(f32.len(), 1);
        assert_eq!(
            f32[0].kernel,
            metal::grouped_projection_pipeline_name(GgmlType::BF16).unwrap()
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&command);
        assert!(
            linear
                .encode_rows_with_arithmetic(&enc, &x, &y, 0, RowsArithmetic::Packed)
                .is_err()
        );
        assert!(
            linear
                .encode_rows_with_arithmetic(&enc, &x, &y, n + 1, RowsArithmetic::Packed)
                .is_err()
        );
        enc.end();
    }
}
