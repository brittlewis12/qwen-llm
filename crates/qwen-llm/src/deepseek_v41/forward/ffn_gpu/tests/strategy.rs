use super::*;

#[test]
fn grouped_override_is_scoped_nested_thread_local_and_unwind_safe() {
    with_grouped_rows(|| {
        with_grouped_rows(|| assert!(FORCE_GROUPED_ROWS.with(|flag| flag.get())));
        assert!(FORCE_GROUPED_ROWS.with(|flag| flag.get()));
        assert!(
            !std::thread::spawn(|| FORCE_GROUPED_ROWS.with(|flag| flag.get()))
                .join()
                .unwrap()
        );
    });
    assert!(!FORCE_GROUPED_ROWS.with(|flag| flag.get()));
    assert!(std::panic::catch_unwind(|| with_grouped_rows(|| panic!("restore"))).is_err());
    assert!(!FORCE_GROUPED_ROWS.with(|flag| flag.get()));
}

fn with_rowwise_rows<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.set(self.0));
        }
    }
    let _restore = Restore(prefill::ROWWISE_LINEAR_ROWS.with(|flag| flag.replace(true)));
    f()
}

/// Which routed-expert route a chunk takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    /// Half-staged generic grouped tiles (the production default above one row).
    Half,
    /// F32-operand grouped tiles.
    F32Grouped,
    /// Per-row native GEMV with a row axis.
    Native,
}

fn assert_census(census: &[metal::DispatchCensusRow], rows: usize, route: Route) {
    let routed: Vec<_> = census
        .iter()
        .filter(|r| {
            r.kernel.starts_with("kernel_expert_native_")
                || r.kernel.starts_with("kernel_moe_")
                || matches!(
                    r.kernel.as_str(),
                    "kernel_deepseek_v41_route"
                        | "kernel_deepseek_v41_weight"
                        | "kernel_deepseek_v41_sum"
                )
        })
        .collect();
    assert_eq!(routed[0].kernel, "kernel_deepseek_v41_route");
    assert_eq!(routed[0].grid_width, rows as u64);
    assert_eq!(routed.last().unwrap().kernel, "kernel_deepseek_v41_sum");
    match route {
        Route::Half => {
            // Fused same-type gate/up is one dispatch; other pairs run gate
            // into the inner scratch, then up with the clamped epilogue.
            assert!(matches!(routed.len(), 6 | 7), "{routed:?}");
            assert!(routed[1].kernel.contains("bucket"));
            let weight = routed.len() - 3;
            assert_eq!(routed[weight].kernel, "kernel_deepseek_v41_weight");
            for r in &routed[2..weight] {
                assert!(
                    r.kernel.starts_with("kernel_moe_") && !r.kernel.contains("_f32x_"),
                    "{r:?}"
                );
            }
            assert!(
                routed[weight + 1]
                    .kernel
                    .ends_with("_f32_grouped_slots_generic"),
                "{routed:?}"
            );
        }
        Route::F32Grouped => {
            assert_eq!(routed.len(), 6, "{routed:?}");
            assert_eq!(routed[3].kernel, "kernel_deepseek_v41_weight");
            assert!(routed[1].kernel.contains("bucket"));
            assert!(routed[2].kernel.starts_with("kernel_moe_swiglu_clamped_"));
            for i in [2, 4] {
                assert!(routed[i].kernel.ends_with("_f32x_grouped_slots"));
            }
        }
        Route::Native => {
            assert_eq!(routed.len(), 6, "{routed:?}");
            assert_eq!(routed[3].kernel, "kernel_deepseek_v41_weight");
            for i in [1, 2, 4] {
                assert!(routed[i].kernel.starts_with("kernel_expert_native_"));
                assert_eq!(routed[i].grid_depth, rows as u64);
            }
        }
    }
}

fn bits(t: &MetalTensor) -> Vec<u32> {
    f32_slice(t).unwrap().iter().map(|v| v.to_bits()).collect()
}

#[test]
fn selected_strategy_threshold_legacy_fallback_and_entrypoints() {
    let _lease = production_lease();
    let ctx = MetalContext::new().unwrap();
    const H: usize = 256;
    const E: usize = 8;
    for (gate, up, down, grouped_cap, native_cap, threshold) in [
        (
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            true,
            true,
            true,
        ),
        (
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            true,
            true,
            true,
        ),
        (
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::MXFP4,
            true,
            true,
            true,
        ),
        (
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            true,
            true,
            true,
        ),
        (
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::MXFP4,
            true,
            true,
            true,
        ),
        (
            GgmlType::MXFP4,
            GgmlType::MXFP4,
            GgmlType::MXFP4,
            true,
            false,
            false,
        ),
        (
            GgmlType::IQ2_S,
            GgmlType::IQ2_S,
            GgmlType::MXFP4,
            true,
            false,
            false,
        ),
        (
            GgmlType::IQ3_S,
            GgmlType::IQ3_S,
            GgmlType::IQ3_S,
            true,
            false,
            false,
        ),
        (
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ2_S,
            true,
            false,
            false,
        ),
        (
            GgmlType::Q3_K,
            GgmlType::Q3_K,
            GgmlType::IQ2_S,
            true,
            false,
            false,
        ),
        (
            GgmlType::IQ2_XS,
            GgmlType::IQ2_XS,
            GgmlType::MXFP4,
            false,
            true,
            false,
        ),
        (
            GgmlType::IQ2_XXS,
            GgmlType::IQ2_XXS,
            GgmlType::IQ2_XXS,
            false,
            true,
            false,
        ),
        (
            GgmlType::IQ3_XXS,
            GgmlType::Q3_K,
            GgmlType::MXFP4,
            false,
            true,
            false,
        ),
    ] {
        let banks = [
            native_f32::bank(&ctx, gate, H, H, E, 3),
            native_f32::bank(&ctx, up, H, H, E, 7),
            native_f32::bank(&ctx, down, H, H, E, 11),
        ];
        let weights = toy::Weights::new(&ctx);
        let mut model = weights.model(&ctx);
        model.layers.truncate(1);
        model.config.hidden = H;
        model.config.intermediate = H;
        model.config.experts = E;
        let layer = &mut model.layers[0];
        layer.banks = [&banks[0], &banks[1], &banks[2]];
        layer.bias = (0..E).map(|i| i as f32 * 0.01).collect();
        layer.ffn_hc.norm = vec![1.0; H];
        let reset = |linear: &mut Linear<'_>, input, output, seed| {
            linear.input = input;
            linear.output = output;
            let x: Vec<_> = (0..input * output)
                .map(|i| ((i * 17 + seed * 13) % 31) as f32 / 8192.0 - 0.001)
                .collect();
            linear.tensor = upload(&ctx, &x, vec![input as u64, output as u64]).unwrap();
        };
        reset(&mut layer.ffn_hc.projection, 4 * H, 24, 1);
        reset(&mut layer.router, H, E, 3);
        for (i, linear) in layer.shared.iter_mut().enumerate() {
            reset(linear, H, H, i + 5);
        }
        let c = &model.config;
        let layer = &model.layers[0];
        assert_eq!(grouped_supported(c, layer), grouped_cap);
        assert_eq!(native_rows_supported(c, layer), native_cap);
        // Half-staged generic layouts exist for every listed storage type
        // except IQ2_XS/IQ2_XXS (native rows).
        let half_cap = ![gate, up, down]
            .iter()
            .any(|dtype| matches!(dtype, GgmlType::IQ2_XS | GgmlType::IQ2_XXS));
        assert_eq!(half_grouped_supported(c, layer), half_cap);
        let ws = Workspace::new(&model).unwrap();
        for n in [1, 3, 7, 31, 33, 65, 127, 128, 129, 257] {
            let grouped = grouped_cap && (!threshold || n >= 128);
            assert_eq!(grouped_selected(c, layer, n), grouped);
            let half = half_cap && n > 1;
            assert_eq!(half_grouped_selected(c, layer, n), half);
            let route = if half {
                Route::Half
            } else if grouped {
                Route::F32Grouped
            } else {
                Route::Native
            };
            let x: Vec<_> = (0..4 * H * n)
                .map(|i| ((i * 19 + i / (4 * H) * 31) % 101) as f32 / 17.0 - 3.0)
                .collect();
            let input = upload(&ctx, &x, vec![H as u64, 4, n as u64]).unwrap();
            let incoming =
                upload(&ctx, &[0.7, 0.1, 0.3, 0.9].repeat(n), vec![4, n as u64]).unwrap();
            let output = MetalTensor::zeros_f32(&ctx, vec![H as u64, 4, n as u64]).unwrap();
            let scratch = Rows::new(&ctx, c, n, 1).unwrap();
            let encode = || {
                metal::dispatch_census_begin();
                submit(&ctx, "selected_strategy_full_ffn", [0; 4], |enc| {
                    ws.encode_rows(&ctx, enc, c, layer, 0, &input, &incoming, &output, &scratch)
                })
                .unwrap();
                scratch.validate_status().unwrap();
                assert!(f32_slice(&output).unwrap().iter().all(|v| v.is_finite()));
                metal::dispatch_census_take()
            };
            assert_census(&encode(), n, route);
            let expected: Vec<_> = [9, 12, 13, 14, 5].map(|i| bits(&scratch.buffers[i])).into();
            metal::dispatch_census_begin();
            submit(&ctx, "selected_strategy_routed_ffn", [0; 4], |enc| {
                ws.encode_routed_rows(&ctx, enc, c, layer, 0, &scratch)
            })
            .unwrap();
            assert_census(&metal::dispatch_census_take(), n, route);
            for (i, expected) in [9, 12, 13, 14, 5].into_iter().zip(expected) {
                assert_eq!(bits(&scratch.buffers[i]), expected);
            }
            if n == 3 {
                with_grouped_rows(|| {
                    assert_eq!(grouped_selected(c, layer, n), grouped_cap);
                    assert!(!half_grouped_selected(c, layer, n));
                    assert_census(
                        &encode(),
                        n,
                        if grouped_cap {
                            Route::F32Grouped
                        } else {
                            Route::Native
                        },
                    );
                    if native_cap {
                        with_native_rows(|| {
                            assert!(!grouped_selected(c, layer, n));
                            assert_census(&encode(), n, Route::Native);
                        });
                    }
                    with_rowwise_rows(|| {
                        assert!(!grouped_selected(c, layer, n));
                        assert!(!native_rows_supported(c, layer));
                        let census = encode();
                        assert_eq!(
                            census
                                .iter()
                                .filter(|r| r.kernel == "kernel_deepseek_v41_route")
                                .count(),
                            n
                        );
                        assert_eq!(
                            census
                                .iter()
                                .filter(|r| r.kernel.starts_with("kernel_expert_native_"))
                                .count(),
                            3 * n
                        );
                        assert!(!census.iter().any(|r| r.kernel.starts_with("kernel_moe_")));
                    });
                });
                assert_eq!(grouped_selected(c, layer, n), grouped);
            }
        }
    }
}
