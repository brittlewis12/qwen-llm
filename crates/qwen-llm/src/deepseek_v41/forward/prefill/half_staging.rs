use super::*;

#[test]
fn half_staging_default_geometry_is_202_mib() {
    let c = Config::default();
    let attention = attention_half_scratch_elements(&c, 2048).unwrap();
    let [input, inner] = ffn_half_scratch_elements(&c, 2048).unwrap();
    let bytes = [attention, input, inner].map(|n| HalfScratch::logical_bytes(n).unwrap());
    assert_eq!(bytes, [128u64, 20, 54].map(|mib| mib * 1024 * 1024));
    assert_eq!(bytes.into_iter().sum::<u64>(), 202 * 1024 * 1024);
    assert_eq!(attention_half_scratch_elements(&c, 0).unwrap(), 0);
    assert_eq!(ffn_half_scratch_elements(&c, 0).unwrap(), [0, 0]);
    assert_eq!(
        ffn_half_scratch_elements(&Config { topk: 0, ..c }, 3).unwrap(),
        [3 * 5120, 3 * 2304]
    );
}

#[test]
fn half_staging_geometry_checks_products_and_bytes() {
    let mut c = Config {
        hidden: 3,
        query_rank: 5,
        heads: 2,
        head_dim: 7,
        groups: 3,
        output_rank: 11,
        intermediate: 13,
        topk: 2,
        ..Config::default()
    };
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 33 * 17);
    assert_eq!(
        ffn_half_scratch_elements(&c, 17).unwrap(),
        [3 * 17, 26 * 17]
    );
    c.query_rank = 37;
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 37 * 17);
    c.hidden = 41;
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 41 * 17);
    c.heads = 7;
    assert_eq!(attention_half_scratch_elements(&c, 17).unwrap(), 49 * 17);
    assert!(attention_half_scratch_elements(&c, usize::MAX).is_err());
    for invalid in [
        Config {
            heads: usize::MAX,
            head_dim: 2,
            ..c.clone()
        },
        Config {
            groups: usize::MAX,
            output_rank: 2,
            ..c.clone()
        },
        Config {
            query_rank: usize::MAX,
            ..c.clone()
        },
    ] {
        assert!(attention_half_scratch_elements(&invalid, 2).is_err());
    }
    for invalid in [
        Config {
            hidden: usize::MAX,
            ..c.clone()
        },
        Config {
            intermediate: usize::MAX,
            ..c.clone()
        },
        Config {
            topk: usize::MAX,
            ..c
        },
    ] {
        assert!(ffn_half_scratch_elements(&invalid, 2).is_err());
    }
    assert_eq!(HalfScratch::logical_bytes(0).unwrap(), 0);
    assert!(HalfScratch::logical_bytes(usize::MAX).is_err());
    assert!(half_scratch_extent(&[usize::MAX / 2 + 1, 1]).is_err());
}

fn require_debug_lease_environment() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn half_staging_allocation_ledger_counts_unused_override_buffers() {
    require_debug_lease_environment();
    let _lease = metal::acquire_metal_benchmark_lease().expect("half staging GPU lease");
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        hidden: 33,
        query_rank: 41,
        heads: 3,
        head_dim: 7,
        groups: 2,
        output_rank: 23,
        intermediate: 17,
        topk: 3,
        ..Config::default()
    };
    assert!(HalfScratch::new(&ctx, 0).unwrap().is_none());
    assert_eq!(half_scratch_budget(&ctx, &c, 0).unwrap(), 0);
    for rows in [1, 33, 129] {
        let elements = [46 * rows, 33 * rows, 51 * rows];
        assert_eq!(
            attention_half_scratch_elements(&c, rows).unwrap(),
            elements[0]
        );
        assert_eq!(
            ffn_half_scratch_elements(&c, rows).unwrap(),
            [elements[1], elements[2]]
        );
        let budget = half_scratch_budget(&ctx, &c, rows).unwrap();
        let expected: u64 = if HalfScratch::selected(&ctx) {
            elements
                .into_iter()
                .map(|n| {
                    ctx.price_shared_buffer_upper((n * 2) as u64)
                        .unwrap()
                        .priced_upper_bytes
                })
                .sum()
        } else {
            0
        };
        assert_eq!(budget, expected);
        with_batched_f32(|| {
            assert_eq!(half_scratch_budget(&ctx, &c, rows).unwrap(), budget);
            let before = ctx.memory_signals().current_allocated_bytes;
            let scratch = elements.map(|n| HalfScratch::new(&ctx, n).unwrap());
            for (buffer, n) in scratch.iter().zip(elements) {
                assert_eq!(buffer.is_some(), HalfScratch::selected(&ctx));
                if let Some(buffer) = buffer {
                    assert_eq!(buffer.tensor.dtype, GgmlType::F16);
                    assert_eq!(buffer.tensor.n_elements(), n as u64);
                    assert_eq!(buffer.tensor.n_bytes(), (2 * n) as u64);
                }
            }
            let allocated = ctx
                .memory_signals()
                .current_allocated_bytes
                .saturating_sub(before);
            assert!(
                allocated <= budget,
                "{allocated} allocated exceeds {budget} priced bytes"
            );
            drop(scratch);
        });
    }
}

fn projection(ctx: &MetalContext, dtype: GgmlType) -> Linear<'_> {
    let (input, output) = (64, 65);
    let bytes = match dtype {
        GgmlType::BF16 => (0..input * output)
            .flat_map(|i| {
                half::bf16::from_f32(((i * 19 % 101) as f32 - 50.0) / 8192.0).to_le_bytes()
            })
            .collect::<Vec<_>>(),
        GgmlType::Q8_0 => (0..input * output / 32)
            .flat_map(|block| {
                let mut bytes = half::f16::from_f32(1.0 / 8192.0).to_le_bytes().to_vec();
                bytes.extend((0..32).map(|i| ((i * 17 + block * 3) % 255) as u8));
                bytes
            })
            .collect(),
        _ => unreachable!(),
    };
    Linear {
        ctx,
        tensor: MetalTensor::from_bytes(ctx, &bytes, vec![input as u64, output as u64], dtype)
            .unwrap(),
        native_control: None,
        input,
        output,
    }
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    f32_slice(t).unwrap().iter().map(|v| v.to_bits()).collect()
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn half_staging_conversion_and_dense_operand_bits_match() {
    require_debug_lease_environment();
    let _lease = metal::acquire_metal_benchmark_lease().expect("half staging GPU lease");
    let ctx = MetalContext::new().unwrap();
    if !HalfScratch::selected(&ctx) {
        return;
    }
    for dtype in [GgmlType::BF16, GgmlType::Q8_0] {
        let linear = projection(&ctx, dtype);
        for rows in [1, 33, 129] {
            let elements = linear.input * rows;
            let mut values: Vec<_> = (0..elements)
                .map(|i| ((i * 31 % 113) as f32 - 56.0) / 71.0)
                .collect();
            values[..6].copy_from_slice(&[
                -0.0,
                0.0,
                2.0f32.powi(-24),
                2.0f32.powi(-25),
                1.0 + 2.0f32.powi(-11),
                -1.0 - 2.0f32.powi(-11),
            ]);
            let backing = upload(
                &ctx,
                &[vec![-777.0; 4], values.clone()].concat(),
                vec![(elements + 4) as u64],
            )
            .unwrap();
            let x = backing.view_subrange(4, vec![linear.input as u64, rows as u64]);
            let half = HalfScratch::new(&ctx, elements).unwrap().unwrap();
            submit(&ctx, "half_staging_convert", [rows, 0, 0, 0], |enc| {
                half.convert(&ctx, enc, &x, elements).map(|_| ())
            })
            .unwrap();
            let half_bits = unsafe {
                std::slice::from_raw_parts(
                    half.tensor
                        .buffer
                        .contents()
                        .as_ptr()
                        .cast::<u16>()
                        .add((half.tensor.offset / 2) as usize),
                    elements,
                )
            };
            assert_eq!(
                half_bits,
                values
                    .iter()
                    .map(|&v| half::f16::from_f32(v).to_bits())
                    .collect::<Vec<_>>()
            );
            let baseline = allocate(&ctx, linear.output, rows, GgmlType::F32).unwrap();
            let guarded = upload(
                &ctx,
                &vec![-777.0; linear.output * rows + 8],
                vec![(linear.output * rows + 8) as u64],
            )
            .unwrap();
            let staged = guarded.view_subrange(4, vec![linear.output as u64, rows as u64]);
            for arithmetic in [RowsArithmetic::Ordinary, RowsArithmetic::Packed] {
                assert_eq!(
                    linear.tensor_rows_selected_with_arithmetic(rows, arithmetic),
                    arithmetic.packed_operands(rows)
                );
                submit(&ctx, "half_staging_dense", [rows, 0, 0, 0], |enc| {
                    linear.encode_rows_with_arithmetic(enc, &x, &baseline, rows, arithmetic)?;
                    linear.encode_rows_with_arithmetic_and_half(
                        enc,
                        &x,
                        Some(&half.tensor),
                        &staged,
                        rows,
                        arithmetic,
                    )
                })
                .unwrap();
                assert_eq!(
                    bits(&baseline),
                    bits(&staged),
                    "{dtype:?} {rows} {arithmetic:?}"
                );
                assert!(f32_slice(&staged).unwrap().iter().all(|v| v.is_finite()));
                let all = f32_slice(&guarded).unwrap();
                assert!(
                    all[..4]
                        .iter()
                        .chain(&all[all.len() - 4..])
                        .all(|&v| v == -777.0)
                );
                with_batched_f32(|| {
                    assert!(!linear.tensor_rows_selected_with_arithmetic(rows, arithmetic));
                    submit(&ctx, "half_staging_f32_override", [rows, 0, 0, 0], |enc| {
                        linear.encode_rows_with_arithmetic(enc, &x, &baseline, rows, arithmetic)?;
                        linear.encode_rows_with_arithmetic_and_half(
                            enc,
                            &x,
                            Some(&x),
                            &staged,
                            rows,
                            arithmetic,
                        )
                    })
                    .unwrap();
                    assert_eq!(bits(&baseline), bits(&staged));
                });
            }
            let command = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&command);
            assert!(half.convert(&ctx, &enc, &x, 0).is_err());
            assert!(half.convert(&ctx, &enc, &x, elements + 1).is_err());
            assert!(half.convert(&ctx, &enc, &half.tensor, elements).is_err());
            enc.end();
        }
    }
}
