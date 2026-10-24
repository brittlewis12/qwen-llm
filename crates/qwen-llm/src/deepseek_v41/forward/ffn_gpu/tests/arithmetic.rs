use super::*;

fn bank(ctx: &MetalContext, dtype: GgmlType, seed: usize) -> MetalTensor {
    let (block, size) = dtype.storage_layout().unwrap();
    let mut bytes = Vec::new();
    for ordinal in 0..256 * 256 * 8 / block as usize {
        let mut row: Vec<_> = (0..size as usize)
            .map(|i| (ordinal * 73 + i * 29 + seed * 11) as u8)
            .collect();
        if dtype == GgmlType::MXFP4 {
            row[0] = 116 + ((ordinal + seed) % 5) as u8;
        } else {
            row[..2].copy_from_slice(&half::f16::from_f32(1.0 / 65536.0).to_le_bytes());
        }
        bytes.extend(row);
    }
    MetalTensor::from_bytes(ctx, &bytes, vec![256, 256, 8], dtype).unwrap()
}

fn dense(linear: &mut Linear<'_>, input: usize, output: usize, dtype: GgmlType) {
    let values: Vec<_> = (0..input * output)
        .map(|i| ((i * 17 % 31) as f32 - 15.0) / 8192.0)
        .collect();
    linear.input = input;
    linear.output = output;
    linear.tensor = if dtype == GgmlType::BF16 {
        let values: Vec<_> = values
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_bits())
            .collect();
        MetalTensor::from_bytes(
            linear.ctx,
            bytemuck::cast_slice(&values),
            vec![input as u64, output as u64],
            dtype,
        )
        .unwrap()
    } else {
        upload(linear.ctx, &values, vec![input as u64, output as u64]).unwrap()
    };
    linear.native_control = None;
}

#[test]
#[ignore = "requires explicit Metal benchmark lease and debug layer"]
fn packed_singleton_ffn_preserves_operands_and_all_f32_controls_win() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("arithmetic GPU lease");
    let ctx = MetalContext::new().unwrap();
    let banks = [3, 7, 11].map(|seed| bank(&ctx, GgmlType::IQ3_XXS, seed));
    let fallback = [3, 7, 11].map(|seed| bank(&ctx, GgmlType::MXFP4, seed));
    let mut unsupported = banks[1].clone();
    unsupported.dtype = GgmlType::I32;
    let weights = toy::Weights::new(&ctx);
    let mut model = weights.model(&ctx);
    model.layers.truncate(1);
    model.config.hidden = 256;
    model.config.intermediate = 256;
    let layer = &mut model.layers[0];
    layer.banks = [&banks[0], &banks[1], &banks[2]];
    layer.ffn_hc.norm = vec![1.0; 256];
    dense(&mut layer.ffn_hc.projection, 1024, 24, GgmlType::BF16);
    dense(&mut layer.router, 256, 8, GgmlType::F32);
    for linear in &mut layer.shared {
        dense(linear, 256, 256, GgmlType::BF16);
    }
    layer.banks[1] = &unsupported;
    assert!(!half_grouped_selected_with_arithmetic(
        &model.config,
        layer,
        1,
        RowsArithmetic::Packed
    ));
    layer.banks[1] = &banks[1];
    let c = &model.config;
    let layer = &model.layers[0];
    let workspace = Workspace::new(&model).unwrap();
    for n in [0, 1, 33, 129, 2048] {
        assert_eq!(half_grouped_selected(c, layer, n), n > 1);
        assert_eq!(
            half_grouped_selected_with_arithmetic(c, layer, n, RowsArithmetic::Packed),
            n > 0
        );
        tests::with_grouped_rows(|| {
            assert!(!half_grouped_selected_with_arithmetic(
                c,
                layer,
                n,
                RowsArithmetic::Packed
            ));
            tests::with_native_rows(|| {
                assert!(!half_grouped_selected_with_arithmetic(
                    c,
                    layer,
                    n,
                    RowsArithmetic::Packed
                ));
                assert!(!grouped_selected(c, layer, n));
            });
        });
        tests::with_native_rows(|| {
            tests::with_grouped_rows(|| {
                assert!(!half_grouped_selected_with_arithmetic(
                    c,
                    layer,
                    n,
                    RowsArithmetic::Packed
                ));
                assert!(!grouped_selected(c, layer, n));
            });
        });
    }
    let mut invalid = c.clone();
    invalid.hidden -= 1;
    assert!(!half_grouped_selected_with_arithmetic(
        &invalid,
        layer,
        1,
        RowsArithmetic::Packed
    ));
    invalid = c.clone();
    invalid.intermediate -= 1;
    assert!(!half_grouped_selected_with_arithmetic(
        &invalid,
        layer,
        129,
        RowsArithmetic::Packed
    ));
    struct RestoreRowwise(bool);
    impl Drop for RestoreRowwise {
        fn drop(&mut self) {
            prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.set(self.0));
        }
    }
    {
        let _restore = RestoreRowwise(prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.replace(true)));
        assert!(!half_grouped_selected_with_arithmetic(
            c,
            layer,
            1,
            RowsArithmetic::Packed
        ));
        assert!(!grouped_selected(c, layer, 129));
        assert!(!native_rows_supported(c, layer));
    }
    for n in [1, 33, 129] {
        assert_eq!(
            Rows::new(&ctx, c, n, 1).unwrap().arithmetic,
            RowsArithmetic::Ordinary
        );
        let input = upload(
            &ctx,
            &(0..1024 * n)
                .map(|i| ((i * 19 % 101) as f32 - 50.0) / 17.0)
                .collect::<Vec<_>>(),
            vec![256, 4, n as u64],
        )
        .unwrap();
        let incoming = upload(&ctx, &[0.7, 0.1, 0.3, 0.9].repeat(n), vec![4, n as u64]).unwrap();
        let output = prefill::allocate(&ctx, 1024, n, GgmlType::F32)
            .unwrap()
            .view_bytes(0, vec![256, 4, n as u64]);
        for arithmetic in [RowsArithmetic::Ordinary, RowsArithmetic::Packed] {
            let scratch = Rows::with_arithmetic(&ctx, c, n, 1, arithmetic).unwrap();
            assert_eq!(scratch.buffers[8].n_elements(), (n * c.topk) as u64);
            assert_eq!(scratch.buffers[11].n_elements(), (n * c.experts) as u64);
            let encode = || {
                metal::dispatch_census_begin();
                let result = submit(&ctx, "ffn_arithmetic", [n, 0, 0, 0], |enc| {
                    workspace
                        .encode_rows(&ctx, enc, c, layer, 0, &input, &incoming, &output, &scratch)
                });
                let census = metal::dispatch_census_take();
                result.unwrap();
                scratch.validate_status().unwrap();
                assert!(f32_slice(&output).unwrap().iter().all(|v| v.is_finite()));
                census
            };
            let ordinary = encode();
            let half = arithmetic.packed_operands(n);
            assert_eq!(
                ordinary
                    .iter()
                    .any(|d| d.kernel.ends_with("_f16act_grouped_slots_tensor")
                        || d.kernel.ends_with("_f32_grouped_slots_generic")),
                half
            );
            assert_eq!(
                ordinary
                    .iter()
                    .filter(|d| d.kernel.starts_with("kernel_mat_mat_bf16"))
                    .count(),
                if half { 4 } else { 0 }
            );
            let controlled = prefill::with_batched_f32(encode);
            assert!(
                !controlled
                    .iter()
                    .any(|d| d.kernel.starts_with("kernel_mat_mat_")
                        || d.kernel.ends_with("_grouped_slots_tensor")
                        || d.kernel.ends_with("_f32_grouped_slots_generic"))
            );
            let grouped: Vec<_> = controlled
                .iter()
                .filter(|d| d.kernel.ends_with("_f32x_grouped_slots"))
                .collect();
            assert_eq!(grouped.len(), 2);
            assert!(
                !controlled
                    .iter()
                    .any(|d| d.kernel.starts_with("kernel_expert_native_"))
            );
            let native_override = tests::with_native_rows(|| prefill::with_batched_f32(encode));
            let native: Vec<_> = native_override
                .iter()
                .filter(|d| d.kernel.starts_with("kernel_expert_native_"))
                .collect();
            assert_eq!(native.len(), 3);
            assert!(native.iter().all(|d| d.grid_depth == n as u64));
            assert_eq!(
                controlled
                    .iter()
                    .filter(|d| Some(&d.kernel)
                        == metal::grouped_projection_pipeline_name(GgmlType::BF16).as_ref())
                    .count(),
                4
            );
            assert_eq!(
                controlled
                    .iter()
                    .filter(|d| d.kernel == "kernel_deepseek_v41_route")
                    .count(),
                1
            );
        }
    }
    model.layers[0].banks = [&fallback[0], &fallback[1], &fallback[2]];
    let layer = &model.layers[0];
    assert!(!native_rows_supported(c, layer));
    let n = 33;
    let scratch = Rows::with_arithmetic(&ctx, c, n, 1, RowsArithmetic::Packed).unwrap();
    let input = upload(&ctx, &vec![0.25; 1024 * n], vec![256, 4, n as u64]).unwrap();
    let incoming = upload(&ctx, &[0.7, 0.1, 0.3, 0.9].repeat(n), vec![4, n as u64]).unwrap();
    let output = prefill::allocate(&ctx, 1024, n, GgmlType::F32)
        .unwrap()
        .view_bytes(0, vec![256, 4, n as u64]);
    prefill::with_batched_f32(|| {
        metal::dispatch_census_begin();
        let result = submit(&ctx, "ffn_f32_grouped_fallback", [n, 0, 0, 0], |enc| {
            workspace.encode_rows(&ctx, enc, c, layer, 0, &input, &incoming, &output, &scratch)
        });
        let census = metal::dispatch_census_take();
        result.unwrap();
        scratch.validate_status().unwrap();
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel.ends_with("_f32x_grouped_slots"))
                .count(),
            2
        );
        assert!(
            !census
                .iter()
                .any(|d| d.kernel.starts_with("kernel_expert_native_"))
        );
        assert_eq!(
            census
                .iter()
                .filter(|d| d.kernel == "kernel_deepseek_v41_route")
                .count(),
            1
        );
    });
}

// The toy owner retains all 40 layers even though this fixture binds only one.
const WIDE_FFN_RESIDENT_BOUND: u64 = 32 * 1024 * 1024;

fn wide_ffn_rows_bound(c: &Config, n: usize) -> u64 {
    let price = |bytes: usize| (bytes as u64).div_ceil(16384) * 16384 + 1024;
    let scratch: u64 = Rows::widths(c, 1)
        .into_iter()
        .map(|w| price(w * n * 4))
        .sum();
    let half: u64 = prefill::ffn_half_scratch_elements(c, n)
        .unwrap()
        .into_iter()
        .map(|e| price(e * 2))
        .sum();
    scratch + half + 2 * price(4 * c.hidden * n * 4) + price(4 * n * 4)
}

#[test]
fn wide_ffn_fixture_has_bounded_full_and_decoder_backing() {
    let mut c = toy::config();
    c.hidden = 256;
    c.intermediate = 256;
    let bound =
        wide_ffn_rows_bound(&c, 4096) + wide_ffn_rows_bound(&c, 2048) + WIDE_FFN_RESIDENT_BOUND;
    assert!(bound < 256 * 1024 * 1024, "{bound}");
    assert_eq!(c.topk, 6);
    for n in [1, 33, 2048] {
        assert!(wide_ffn_rows_bound(&c, n) <= wide_ffn_rows_bound(&c, 2048));
    }
}

#[test]
#[ignore = "M5 tensor tier required; self-leased, exact Full-wide vs bounded decoder FFN/status"]
fn wide_full_vs_compacted_ffn_maya_bitwise_and_status() {
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    let _lease = metal::acquire_metal_benchmark_lease().expect("wide FFN compatibility lease");
    let ctx = MetalContext::new().unwrap();
    assert!(
        metal::moe_grouped_tensor_selected(&ctx),
        "M5 tensor tier required, no fallback proof"
    );
    assert!(std::env::var_os("QWEN_V41_EXPERT_WIDE").is_none_or(|v| v != "0"));
    for dtype in [
        GgmlType::IQ2_XXS,
        GgmlType::IQ2_S,
        GgmlType::IQ3_XXS,
        GgmlType::IQ3_S,
    ] {
        let banks = [3, 7, 11].map(|seed| bank(&ctx, dtype, seed));
        let weights = toy::Weights::new(&ctx);
        let mut model = weights.model(&ctx);
        model.layers.truncate(1);
        model.config.hidden = 256;
        model.config.intermediate = 256;
        let layer = &mut model.layers[0];
        layer.banks = [&banks[0], &banks[1], &banks[2]];
        layer.ffn_hc.norm = vec![1.0; 256];
        dense(&mut layer.ffn_hc.projection, 1024, 24, GgmlType::BF16);
        dense(&mut layer.router, 256, 8, GgmlType::F32);
        for linear in &mut layer.shared {
            dense(linear, 256, 256, GgmlType::BF16);
        }
        let c = &model.config;
        let layer = &model.layers[0];
        let workspace = Workspace::new(&model).unwrap();
        let resident = ctx.memory_signals().current_allocated_bytes;
        assert!(
            resident <= WIDE_FFN_RESIDENT_BOUND,
            "fixture resident buffers {resident}"
        );
        for n in [4096, 2048] {
            let row_bytes = Rows::widths(c, 1)
                .into_iter()
                .chain([4 * c.hidden, 4 * c.hidden, 4])
                .map(|w| (w * n * 4) as u64);
            let half_bytes = prefill::ffn_half_scratch_elements(c, n)
                .unwrap()
                .into_iter()
                .map(|e| (e * 2) as u64);
            for logical in row_bytes.chain(half_bytes) {
                assert!(
                    ctx.price_shared_buffer_upper(logical)
                        .unwrap()
                        .priced_upper_bytes
                        <= logical.div_ceil(16384) * 16384 + 1024
                );
            }
        }
        let bound = resident + wide_ffn_rows_bound(c, 4096) + wide_ffn_rows_bound(c, 2048);
        assert!(
            bound < 256 * 1024 * 1024,
            "fixture planned Metal buffers {bound}"
        );
        let x: Vec<_> = (0..1024 * 4096)
            .map(|i| ((i * 19 % 1009) as f32 - 504.0) / 171.0)
            .collect();
        let p = [0.7, 0.1, 0.3, 0.9];
        let input = upload(&ctx, &x, vec![256, 4, 4096]).unwrap();
        let incoming = upload(&ctx, &p.repeat(4096), vec![4, 4096]).unwrap();
        let full_out = prefill::allocate(&ctx, 1024, 4096, GgmlType::F32)
            .unwrap()
            .view_bytes(0, vec![256, 4, 4096]);
        let full = Rows::new(&ctx, c, 4096, 1).unwrap();
        let encode =
            |input: &MetalTensor, incoming: &MetalTensor, out: &MetalTensor, scratch: &Rows| {
                metal::dispatch_census_begin();
                let result = objc2::rc::autoreleasepool(|_| {
                    submit(&ctx, "wide_ffn_compat", [scratch.hc.rows, 0, 0, 0], |enc| {
                        workspace.encode_rows(&ctx, enc, c, layer, 0, input, incoming, out, scratch)
                    })
                });
                let census = metal::dispatch_census_take();
                result.unwrap();
                scratch.validate_status().unwrap();
                assert!(f32_slice(out).unwrap().iter().all(|v| v.is_finite()));
                assert!(f32_slice(out).unwrap().iter().any(|&v| v != 0.0));
                assert_eq!(
                    census
                        .iter()
                        .filter(|d| d.kernel.starts_with("kernel_moe_wide_probe_"))
                        .count(),
                    if scratch.hc.rows >= 4096 { 2 } else { 0 }
                );
                assert_eq!(
                    census
                        .iter()
                        .filter(|d| d.kernel.starts_with("kernel_moe_down_")
                            && d.kernel.ends_with("_f16act_grouped_slots_tensor"))
                        .count(),
                    usize::from(scratch.arithmetic.packed_operands(scratch.hc.rows))
                );
            };
        encode(&input, &incoming, &full_out, &full);
        for n in [1, 33, 2048] {
            let rows: Vec<_> = (0..n).map(|i| (4050 + i * 4051) % 4096).collect();
            let compact_x: Vec<_> = rows
                .iter()
                .flat_map(|&r| x[r * 1024..(r + 1) * 1024].iter().copied())
                .collect();
            let compact_x = upload(&ctx, &compact_x, vec![256, 4, n as u64]).unwrap();
            let compact_p = upload(&ctx, &p.repeat(n), vec![4, n as u64]).unwrap();
            let out = prefill::allocate(&ctx, 1024, n, GgmlType::F32)
                .unwrap()
                .view_bytes(0, vec![256, 4, n as u64]);
            let scratch = Rows::with_arithmetic(&ctx, c, n, 1, RowsArithmetic::Packed).unwrap();
            encode(&compact_x, &compact_p, &out, &scratch);
            for (reference, actual, width) in [
                (&full_out, &out, 1024),
                (&full.hc.pre, &scratch.hc.pre, 4),
                (&full.hc.normalized, &scratch.hc.normalized, 256),
                (&full.buffers[8], &scratch.buffers[8], c.topk),
                (&full.buffers[9], &scratch.buffers[9], c.topk),
                (
                    &full.buffers[12],
                    &scratch.buffers[12],
                    c.intermediate * c.topk,
                ),
            ] {
                // Route IDs are I32; compare their raw words too, not floats.
                let mut reference_view = reference.clone();
                let mut actual_view = actual.clone();
                assert!(matches!(
                    reference_view.dtype,
                    GgmlType::F32 | GgmlType::I32
                ));
                assert_eq!(reference_view.dtype, actual_view.dtype);
                reference_view.dtype = GgmlType::F32;
                actual_view.dtype = GgmlType::F32;
                let reference = f32_slice(&reference_view).unwrap();
                let actual = f32_slice(&actual_view).unwrap();
                for (i, &row) in rows.iter().enumerate() {
                    for j in 0..width {
                        assert_eq!(
                            actual[i * width + j].to_bits(),
                            reference[row * width + j].to_bits(),
                            "{dtype:?} n={n} original={row} width={width} j={j}"
                        );
                    }
                }
            }
            if n == 1 {
                let scalar = Rows::new(&ctx, c, 1, 1).unwrap();
                assert!(!scalar.arithmetic.packed_operands(1));
                encode(&compact_x, &compact_p, &out, &scalar);
            }
        }
        eprintln!("strict wide Full/compacted FFN {dtype:?} passed");
    }
}
