use super::*;

const EXPERTS: usize = 3;
const TOKENS: usize = 35;
const TOP: usize = 2;
const SENTINEL: f32 = -777.0;
const CLAMP: f32 = 0.037;

fn half_values(values: &[f32]) -> Vec<f32> {
    values
        .iter()
        .map(|&v| half::f16::from_f32(v).to_f32())
        .collect()
}

fn assigned_projections(output: &[f32], expected: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(output.len(), expected.len());
    let mut actual = Vec::new();
    let mut reference = Vec::new();
    for (&a, &b) in output.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        if b == SENTINEL {
            assert_eq!(a, SENTINEL, "invalid/unassigned slot must not be written");
        } else {
            actual.push(a);
            reference.push(b);
        }
    }
    (actual, reference)
}

#[test]
fn iq1_tensor_agreement_rejects_active_error_despite_large_sentinels() {
    let mut expected = vec![SENTINEL; 128];
    expected[3] = 0.25;
    expected[91] = -0.125;
    let (reference, _) = assigned_projections(&expected, &expected);
    assert_eq!(reference, [0.25, -0.125]);
    assert_tile_agreement("assigned positive control", &reference, &reference);

    let mut changed = expected.clone();
    changed[3] += 0.25;
    let (active, _) = assigned_projections(&changed, &expected);
    assert!(
        std::panic::catch_unwind(|| {
            assert_tile_agreement("assigned error negative control", &active, &reference);
        })
        .is_err()
    );

    changed = expected.clone();
    changed[0] = 0.0;
    assert!(std::panic::catch_unwind(|| assigned_projections(&changed, &expected)).is_err());
}

struct Fixture {
    k: usize,
    m: usize,
    gate: MetalTensor,
    up: MetalTensor,
    x: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    expected: Vec<f32>,
    down: bool,
}

impl Fixture {
    fn new(
        ctx: &MetalContext,
        dtype: GgmlType,
        k: usize,
        m: usize,
        down: bool,
        bad_routes: bool,
    ) -> Self {
        admit_iq1_expert_fixture(ctx);
        let gb = synthetic_bank(dtype, k, m, EXPERTS, 0x1234);
        let ub = synthetic_bank(dtype, k, m, EXPERTS, 0x5678);
        let mut rng = Lcg(0xdeadbeef);
        let x: Vec<_> = (0..k * TOKENS * if down { TOP } else { 1 })
            .map(|_| rng.signed(0.53))
            .collect();
        let counts = [
            35,
            if bad_routes { -7 } else { 17 },
            if bad_routes { 36 } else { 0 },
        ];
        let mut ids = vec![-1i32; EXPERTS * TOKENS];
        for (i, id) in ids.iter_mut().take(TOKENS).enumerate() {
            *id = i as i32;
        }
        for i in 0..17 {
            ids[TOKENS + i] = (35 + i) as i32;
        }
        if bad_routes {
            ids[0] = -1;
            ids[1] = (TOKENS * TOP) as i32;
            ids[2] = i32::MAX;
            ids[3] = i32::MIN;
            // These unique valid slots must stay untouched because their
            // expert's count exceeds its ID-row capacity, not because IDs fail.
            for (i, id) in ids[2 * TOKENS..].iter_mut().enumerate() {
                *id = (TOKENS + i) as i32;
            }
        }
        let hx = half_values(&x);
        let mut expected = vec![SENTINEL; TOKENS * TOP * m];
        for (e, &count) in counts.iter().enumerate() {
            if count <= 0 || count as usize > TOKENS {
                continue;
            }
            let g = half_values(&cpu_expert(dtype, &gb, k, m, e));
            let u = half_values(&cpu_expert(dtype, &ub, k, m, e));
            for &slot in &ids[e * TOKENS..e * TOKENS + count as usize] {
                if slot < 0 || slot as usize >= TOKENS * TOP {
                    continue;
                }
                let slot = slot as usize;
                let token = if down { slot } else { slot / TOP };
                let input = &hx[token * k..][..k];
                let gv = matvec(&g, k, input);
                let value = if down {
                    gv
                } else {
                    swiglu_epilogue(&gv, &matvec(&u, k, input), Some(CLAMP))
                };
                expected[slot * m..][..m].copy_from_slice(&value);
            }
        }
        Self {
            k,
            m,
            down,
            expected,
            gate: offset_tensor(
                ctx,
                18,
                &gb,
                30,
                vec![k as u64, m as u64, EXPERTS as u64],
                dtype,
            ),
            up: offset_tensor(
                ctx,
                34,
                &ub,
                14,
                vec![k as u64, m as u64, EXPERTS as u64],
                dtype,
            ),
            x: offset_tensor(
                ctx,
                32,
                bytemuck::cast_slice(&x),
                16,
                vec![x.len() as u64],
                GgmlType::F32,
            ),
            counts: offset_tensor(
                ctx,
                16,
                bytemuck::cast_slice(&counts),
                16,
                vec![EXPERTS as u64],
                GgmlType::I32,
            ),
            ids: offset_tensor(
                ctx,
                16,
                bytemuck::cast_slice(&ids),
                16,
                vec![(EXPERTS * TOKENS) as u64],
                GgmlType::I32,
            ),
        }
    }

    fn run(&self, ctx: &MetalContext, tensor: bool, half: bool, wide: bool) -> Vec<f32> {
        let f16_x;
        let x = if half {
            f16_x = f16_tensor(ctx, &tensor_f32_at_offset(&self.x));
            &f16_x
        } else {
            &self.x
        };
        let out = offset_tensor(
            ctx,
            32,
            bytemuck::cast_slice(&vec![SENTINEL; self.expected.len()]),
            32,
            vec![self.expected.len() as u64],
            GgmlType::F32,
        );
        let inputs = [&self.gate, &self.up, x, &self.counts, &self.ids];
        let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
        dispatch_census_begin();
        one_shot(ctx, |enc| {
            if wide {
                if self.down {
                    encode_moe_down_wide_probe(
                        ctx,
                        enc,
                        &self.gate,
                        x,
                        &self.counts,
                        &self.ids,
                        &out,
                        self.k,
                        self.m,
                        EXPERTS,
                        TOKENS,
                    )
                } else {
                    encode_moe_swiglu_wide_probe(
                        ctx,
                        enc,
                        &self.gate,
                        &self.up,
                        x,
                        &self.counts,
                        &self.ids,
                        &out,
                        self.k,
                        self.m,
                        EXPERTS,
                        TOP,
                        TOKENS,
                        CLAMP,
                    )
                }
            } else if tensor {
                if self.down {
                    encode_moe_down_f32_grouped_slots_tensor(
                        ctx,
                        enc,
                        &self.gate,
                        x,
                        &self.counts,
                        &self.ids,
                        &out,
                        self.k,
                        self.m,
                        EXPERTS,
                        TOKENS,
                    )
                } else {
                    encode_moe_swiglu_f32_grouped_slots_tensor(
                        ctx,
                        enc,
                        &self.gate,
                        &self.up,
                        x,
                        &self.counts,
                        &self.ids,
                        &out,
                        self.k,
                        self.m,
                        EXPERTS,
                        TOP,
                        TOKENS,
                        Some(CLAMP),
                    )
                }
            } else if self.down {
                encode_moe_down_f32_grouped_slots_generic(
                    ctx,
                    enc,
                    &self.gate,
                    x,
                    &self.counts,
                    &self.ids,
                    &out,
                    self.k,
                    self.m,
                    EXPERTS,
                    TOKENS,
                )
            } else {
                encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                    ctx,
                    enc,
                    &self.gate,
                    &self.up,
                    x,
                    &self.counts,
                    &self.ids,
                    &out,
                    self.k,
                    self.m,
                    EXPERTS,
                    TOP,
                    TOKENS,
                    CLAMP,
                )
            }
        })
        .expect("executable IQ1 expert projection");
        let census = dispatch_census_take();
        assert_eq!(census.len(), if self.down { 1 } else { 2 });
        let suffix = moe_grouped_generic_layout(self.gate.dtype).unwrap().suffix;
        for record in &census {
            assert!(record.kernel.contains(suffix), "{record:?}");
            assert_eq!(record.threads_width, 128);
            assert!(record.kernel.contains(if wide {
                "wide_probe"
            } else if tensor {
                "_tensor"
            } else if self.down {
                "_generic"
            } else {
                "grouped_slots"
            }));
        }
        let got = tensor_f32_at_offset(&out);
        let (actual, reference) = assigned_projections(&got, &self.expected);
        assert_close("IQ1 canonical half operands / F32 sum", &actual, &reference);
        assert_offset_guards(&out, 32, 32);
        for (t, bytes) in inputs.iter().zip(before) {
            assert_eq!(tensor_backing_bytes(t), bytes);
        }
        actual
    }
}

#[test]
fn iq1_grouped_half_codec_real_k_tails_unaligned_offsets_and_invalid_routes() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("IQ1 expert GPU proof requires Metal");
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        for (k, m) in [(256, 5), (2304, 67), (5120, 65)] {
            for down in [false, true] {
                for bad_routes in [false, true] {
                    let f = Fixture::new(&ctx, dtype, k, m, down, bad_routes);
                    let active = f.run(&ctx, false, false, false);
                    assert_eq!(
                        active.len(),
                        f.expected.iter().filter(|&&v| v != SENTINEL).count()
                    );
                }
            }
        }
    }
}

#[test]
fn iq1_shared_count_guards_precede_ids_and_barriers() {
    let f32x = include_str!("../../../../../../kernels/moe_grouped_f32.metal");
    let n32 = include_str!("../../../../../../kernels/moe_grouped_n32.metal");
    for (source, entry, capacity) in [
        (
            MOE_METAL,
            "inline void moe_grouped_slots_mm_generic_body(",
            "N",
        ),
        (
            MOE_METAL,
            "inline void moe_swiglu_grouped_slots_n16_body(",
            "n_tokens",
        ),
        (
            TENSOR_METAL,
            "inline void moe_grouped_slots_mm_tensor_body(",
            "N",
        ),
        (
            TENSOR_METAL,
            "kernel void kernel_moe_wide_probe_f16act(",
            "N",
        ),
        (f32x, "kernel void kernel_moe_down_grouped_slots_f32x(", "N"),
        (
            f32x,
            "kernel void kernel_moe_swiglu_clamped_grouped_slots_f32x(",
            "n_tokens",
        ),
        (
            n32,
            "kernel void kernel_moe_swiglu_clamped_grouped_slots_n32(",
            "n_tokens",
        ),
    ] {
        let body = source.split_once(entry).unwrap().1;
        let count = body.find("const int count =").unwrap();
        let guard_start = count + body[count..].find("if (").unwrap();
        let guard_end = guard_start + body[guard_start..].find("return;").unwrap() + 7;
        let guard: String = body[guard_start..guard_end].split_whitespace().collect();
        let expected: String = format!(
            "if (count < 0 || uint(count) > args.{capacity} || count < int(args.min_count) || count > int(args.max_count) || r1 >= count) return;"
        ).split_whitespace().collect();
        assert_eq!(
            guard, expected,
            "{entry}: only threadgroup-uniform predicates"
        );
        assert!(body[..count].contains("const int r1 = tgpig.x *"));
        assert!(
            guard_end < body.find("ids[").unwrap(),
            "{entry}: ID load precedes rejection"
        );
        assert!(
            guard_end < body.find("threadgroup_barrier(").unwrap(),
            "{entry}: barrier precedes rejection"
        );
    }
}

#[test]
fn iq1_shared_generic_roles_reject_capacity_and_count_ranges_with_valid_ids() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("count-bound GPU proof requires Metal");
    for dtype in [
        GgmlType::IQ1_S,
        GgmlType::IQ1_M,
        GgmlType::IQ2_S,
        GgmlType::IQ3_S,
    ] {
        let f = Fixture::new(&ctx, dtype, 256, 5, false, true);
        for (count, min, max) in [
            (36, 0, i32::MAX as u32),
            (i32::MAX, 0, i32::MAX as u32),
            (-7, 0, i32::MAX as u32),
            (35, 0, 34),
            (35, 36, i32::MAX as u32),
        ] {
            let counts = i32_typed_tensor(&ctx, &[0, 0, count]);
            let out = offset_tensor(
                &ctx,
                32,
                bytemuck::cast_slice(&vec![SENTINEL; f.expected.len()]),
                32,
                vec![f.expected.len() as u64],
                GgmlType::F32,
            );
            let roles = if MOE_GROUPED_CLAMPED_SWIGLU_DTYPES.contains(&dtype) {
                4
            } else {
                3
            };
            for role in 0..roles {
                one_shot(&ctx, |enc| match role {
                    0 | 1 => encode_generic_mm(
                        &ctx,
                        enc,
                        if role == 0 {
                            MoeGroupedGenericRole::Down
                        } else {
                            MoeGroupedGenericRole::UpSiluMul
                        },
                        "count_bound_test",
                        &f.gate,
                        &f.x,
                        &counts,
                        &f.ids,
                        &out,
                        f.k,
                        f.m,
                        EXPERTS,
                        TOKENS,
                        TOP,
                        min,
                        max,
                    ),
                    _ => encode_fused_swiglu(
                        &ctx,
                        enc,
                        "count_bound_test",
                        &f.gate,
                        &f.up,
                        &f.x,
                        &counts,
                        &f.ids,
                        &out,
                        f.k,
                        f.m,
                        EXPERTS,
                        TOP,
                        TOKENS,
                        min,
                        max,
                        if role == 3 { Some(CLAMP) } else { None },
                    ),
                })
                .unwrap();
                assert!(
                    tensor_f32_at_offset(&out).iter().all(|&v| v == SENTINEL),
                    "{dtype:?} role={role} count={count} range={min}..={max}"
                );
                assert_offset_guards(&out, 32, 32);
            }
        }
    }
}

#[test]
#[ignore = "requires actual M5 tensor execution; unavailable hardware is not a pass"]
fn iq1_m5_tensor_half_codec_real_k_tails_and_wide() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("M5 proof requires Metal");
    assert!(
        moe_grouped_tensor_available(&ctx),
        "requires M5: {:?}",
        ctx.tensor_ops_status()
    );
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        for (k, m) in [(256, 5), (2304, 67), (5120, 65)] {
            for down in [false, true] {
                let f = Fixture::new(&ctx, dtype, k, m, down, true);
                let generic_assigned = f.run(&ctx, false, false, false);
                for (half, wide) in [(false, false), (true, false), (true, true)] {
                    let tensor_assigned = f.run(&ctx, true, half, wide);
                    assert_tile_agreement(
                        "IQ1 tensor vs canonical generic",
                        &tensor_assigned,
                        &generic_assigned,
                    );
                }
            }
        }
    }
}

#[test]
fn iq1_actual_product_pso_preflight() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("IQ1 PSO proof requires Metal");
    eprintln!(
        "IQ1_DEVICE name={} tensor_status={:?}",
        ctx.device.name(),
        ctx.tensor_ops_status()
    );
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        for role in MoeGroupedGenericRole::ALL {
            let name = moe_grouped_generic_pipeline_name(role, dtype).unwrap();
            assert_eq!(ctx.pipeline(&name).unwrap().threadExecutionWidth(), 32);
        }
        let recipe = moe_grouped_clamped_recipe(dtype, dtype).unwrap();
        let MoeGroupedClampedRecipe::GateThenUpClamped { gate, up } = recipe else {
            panic!("IQ1 keeps two-pass recipe")
        };
        ctx.pipeline(&gate).unwrap();
        ctx.pipeline(&up).unwrap();
    }
}

#[test]
fn iq1_grouped_clamped_refuses_invalid_bindings_without_dispatch() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("IQ1 expert GPU proof requires Metal");
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        let f = Fixture::new(&ctx, dtype, 256, 5, false, false);
        let output = offset_tensor(
            &ctx,
            32,
            bytemuck::cast_slice(&f.expected),
            32,
            vec![f.expected.len() as u64],
            GgmlType::F32,
        );
        let original = [f.gate, f.up, f.x, f.counts, f.ids, output];
        let before: Vec<_> = original.iter().map(tensor_backing_bytes).collect();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let encode = |t: &[MetalTensor; 6], k, n, c| {
            encode_moe_swiglu_clamped_f32_grouped_slots_generic(
                &ctx, &enc, &t[0], &t[1], &t[2], &t[3], &t[4], &t[5], k, 5, EXPERTS, TOP, n, c,
            )
        };
        dispatch_census_begin();
        for which in 0..6 {
            for fault in 0..4 {
                let mut t = original.clone();
                match fault {
                    0 => t[which].offset += 1,
                    1 => t[which].offset = t[which].buffer.length() as u64,
                    2 => t[which].offset = u64::MAX - 3,
                    _ => t[which].dtype = GgmlType::Unknown,
                }
                assert!(
                    encode(&t, 256, TOKENS, CLAMP).is_err(),
                    "binding {which} fault {fault}"
                );
            }
        }
        for which in 0..5 {
            let mut t = original.clone();
            let shared = ctx.buffer_from(&vec![0u8; 65536]).unwrap();
            t[which].buffer = shared.clone();
            t[which].offset = 32;
            t[5].buffer = shared;
            t[5].offset = 32;
            assert!(
                encode(&t, 256, TOKENS, CLAMP)
                    .unwrap_err()
                    .to_string()
                    .contains("aliases")
            );
        }
        for c in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            assert!(encode(&original, 256, TOKENS, c).is_err());
        }
        for (k, n) in [(255, TOKENS), (256, 0), (256, usize::MAX)] {
            assert!(encode(&original, k, n, CLAMP).is_err());
        }
        let mut t = original.clone();
        t[5].provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(encode(&t, 256, TOKENS, CLAMP).is_err());
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        assert!(dispatch_census_take().is_empty());
        for (t, b) in original.iter().zip(before) {
            assert_eq!(tensor_backing_bytes(t), b);
        }
    }
}

#[test]
fn iq1_grouped_release_mixed_down_is_independent() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("IQ1 expert GPU proof requires Metal");
    for (layer, gate_up, down) in [
        (7, GgmlType::IQ1_M, GgmlType::IQ2_XXS),
        (17, GgmlType::IQ2_XS, GgmlType::IQ1_M),
    ] {
        eprintln!(
            "IQ1_RELEASE_STORAGE layer={layer} gate={gate_up:?} up={gate_up:?} down={down:?}"
        );
        Fixture::new(&ctx, gate_up, 5120, 65, false, false).run(&ctx, false, false, false);
        Fixture::new(&ctx, down, 2304, 67, true, false).run(&ctx, false, false, false);
    }
}

#[test]
#[ignore = "requires actual M5; PSO preflight is not M4 execution evidence"]
fn iq1_m5_all_tensor_psos_and_release_mixed_down() {
    let _lease = crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease");
    let ctx = metal_test_context().expect("M5 proof requires Metal");
    assert!(
        moe_grouped_tensor_available(&ctx),
        "requires M5: {:?}",
        ctx.tensor_ops_status()
    );
    for dtype in [GgmlType::IQ1_S, GgmlType::IQ1_M] {
        for activation in [TensorActivation::F32, TensorActivation::F16] {
            for role in MoeGroupedTensorRole::ALL {
                ctx.tensor_pipeline(
                    &moe_grouped_tensor_pipeline_name(role, dtype, activation).unwrap(),
                )
                .unwrap();
            }
            ctx.tensor_pipeline(&mat_mat_tensor_pipeline_name(dtype, activation).unwrap())
                .unwrap();
        }
        ctx.tensor_pipeline(&wide_probe_pipeline_name(dtype).unwrap())
            .unwrap();
    }
    for (gate_up, down) in [
        (GgmlType::IQ1_M, GgmlType::IQ2_XXS),
        (GgmlType::IQ2_XS, GgmlType::IQ1_M),
    ] {
        for (dtype, k, m, is_down) in [(gate_up, 5120, 65, false), (down, 2304, 67, true)] {
            let f = Fixture::new(&ctx, dtype, k, m, is_down, false);
            let generic_assigned = f.run(&ctx, false, false, false);
            for (half, wide) in [(false, false), (true, false), (true, true)] {
                let got_assigned = f.run(&ctx, true, half, wide);
                assert_tile_agreement("released mixed storage", &got_assigned, &generic_assigned);
            }
        }
    }
}
