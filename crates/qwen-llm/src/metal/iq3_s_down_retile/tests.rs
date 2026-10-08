use super::*;
use crate::metal::test_support::{
    assert_offset_guards, metal_test_context, offset_tensor, tensor_backing_bytes,
    tensor_f32_at_offset,
};
use crate::tensor::TensorDesc;

const EXPERTS: usize = 288;
const SLOTS: usize = 160;
const SENTINEL: f32 = -777.0;
const BUCKETS: [(usize, usize); 8] = [
    (0, 0),
    (1, 1),
    (17, 15),
    (32, 16),
    (128, 17),
    (255, 31),
    (286, 32),
    (287, 33),
];

struct Fixture {
    k: usize,
    m: usize,
    n: usize,
    weight: MetalTensor,
    inner: MetalTensor,
    counts: MetalTensor,
    ids: MetalTensor,
    expected: Vec<f64>,
    slot_counts: Vec<usize>,
}

impl Fixture {
    fn new(ctx: &MetalContext, k: usize, m: usize, n: usize) -> Self {
        let mut state = 0xDEADBEEFu32;
        let mut bytes = vec![0; k / 256 * m * EXPERTS * 110];
        for (b, block) in bytes.chunks_exact_mut(110).enumerate() {
            for byte in &mut block[2..] {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                *byte = (state >> 24) as u8;
            }
            let d = if b % 13 == 0 {
                0.0
            } else {
                (b % 7 + 1) as f32 / 4096.0
            };
            let d = if b % 3 == 0 { -d } else { d };
            block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        }
        let x: Vec<f32> = (0..k * SLOTS)
            .map(|i| ((i * 17 + i / 23) % 107) as f32 / 51.0 - 1.0)
            .collect();
        let mut counts = vec![0i32; EXPERTS];
        let mut ids = vec![i32::MAX; n * EXPERTS];
        let mut expected = vec![f64::from(SENTINEL); m * SLOTS];
        let mut slot_counts = vec![0; SLOTS];
        let stride = k / 256 * m * 110;
        let mut ordinal = 0;
        for (expert, count) in BUCKETS {
            counts[expert] = count as i32;
            if count == 0 {
                continue;
            }
            let desc = TensorDesc {
                name: format!("iq3_s_retile_expert_{expert}"),
                shape: vec![k as u64, m as u64],
                dtype: GgmlType::IQ3_S,
                shard_idx: 0,
                data_offset: 0,
                n_bytes: stride as u64,
            };
            // Independent GGML codec/table, followed by the kernel's half staging
            // contract and f64 dot accumulation (no incumbent output as oracle).
            let decoded =
                crate::codec::dequant_to_f32(&desc, &bytes[expert * stride..(expert + 1) * stride])
                    .unwrap();
            let decoded: Vec<f64> = decoded
                .into_iter()
                .map(|w| half::f16::from_f32(w).to_f64())
                .collect();
            for j in 0..count {
                let slot = (ordinal * 37 + 11) % SLOTS;
                ordinal += 1;
                let id = match (count, j) {
                    (15, 3) => -1,
                    (17, 0) => SLOTS as i32,
                    (33, 32) => i32::MAX,
                    _ => slot as i32,
                };
                // Reverse each list so addressing cannot accidentally follow token order.
                ids[expert * n + count - 1 - j] = id;
                if id < 0 || id >= SLOTS as i32 {
                    continue;
                }
                assert_eq!(slot_counts[slot], 0);
                slot_counts[slot] = count;
                for row in 0..m {
                    expected[slot * m + row] = decoded[row * k..(row + 1) * k]
                        .iter()
                        .zip(&x[slot * k..(slot + 1) * k])
                        .map(|(&w, &x)| w * half::f16::from_f32(x).to_f64())
                        .sum();
                }
            }
        }
        Self {
            k,
            m,
            n,
            weight: offset_tensor(
                ctx,
                18,
                &bytes,
                22,
                vec![k as u64, m as u64, EXPERTS as u64],
                GgmlType::IQ3_S,
            ),
            inner: offset_tensor(
                ctx,
                16,
                bytemuck::cast_slice(&x),
                20,
                vec![k as u64, SLOTS as u64],
                GgmlType::F32,
            ),
            counts: offset_tensor(
                ctx,
                12,
                bytemuck::cast_slice(&counts),
                20,
                vec![EXPERTS as u64],
                GgmlType::I32,
            ),
            ids: offset_tensor(
                ctx,
                12,
                bytemuck::cast_slice(&ids),
                20,
                vec![n as u64, EXPERTS as u64],
                GgmlType::I32,
            ),
            expected,
            slot_counts,
        }
    }

    fn output(&self, ctx: &MetalContext) -> MetalTensor {
        offset_tensor(
            ctx,
            20,
            bytemuck::cast_slice(&vec![SENTINEL; self.m * SLOTS]),
            28,
            vec![self.m as u64, SLOTS as u64],
            GgmlType::F32,
        )
    }

    fn encode(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        variant: DownRetile,
        out: &MetalTensor,
    ) -> Result<(), MetalError> {
        encode_variant(
            ctx,
            enc,
            variant,
            &self.weight,
            &self.inner,
            &self.counts,
            &self.ids,
            out,
            self.k,
            self.m,
            EXPERTS,
            self.n,
        )
    }

    fn assert_output(&self, out: &MetalTensor, active: impl Fn(usize) -> bool) {
        let actual = tensor_f32_at_offset(out);
        for (i, &a) in actual.iter().enumerate() {
            let count = self.slot_counts[i / self.m];
            assert!(a.is_finite(), "non-finite result at {i}");
            if count == 0 || !active(count) {
                assert_eq!(
                    a.to_bits(),
                    SENTINEL.to_bits(),
                    "unexpected write at {i}, count={count}"
                );
            } else {
                let e = self.expected[i];
                assert!(e.is_finite());
                assert!(
                    (f64::from(a) - e).abs() <= 3e-5 * (1.0 + e.abs()),
                    "K={} M={} N={} index={i} count={count}: {a} vs {e}",
                    self.k,
                    self.m,
                    self.n
                );
            }
        }
        assert_offset_guards(out, 20, 28);
    }
}

#[test]
fn iq3_s_retile_variants_match_codec_f64_tails_offsets_and_sparse_buckets() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    for (k, m, n) in [(256, 1, 40), (512, 129, 128), (2048, 131, 512)] {
        let fixture = Fixture::new(&ctx, k, m, n);
        let inputs = [
            &fixture.weight,
            &fixture.inner,
            &fixture.counts,
            &fixture.ids,
        ];
        let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
        for variant in [
            DownRetile::Incumbent,
            DownRetile::Blanket,
            DownRetile::SmallCounts,
        ] {
            let out = fixture.output(&ctx);
            dispatch_census_begin();
            let result = one_shot(&ctx, |enc| fixture.encode(&ctx, enc, variant, &out));
            let census = dispatch_census_take();
            result.unwrap();
            fixture.assert_output(&out, |_| true);
            let expected = match variant {
                DownRetile::Incumbent => vec![(INCUMBENT, n.div_ceil(32), m.div_ceil(64))],
                DownRetile::Blanket => vec![(RETILE, n.div_ceil(16), m.div_ceil(128))],
                DownRetile::SmallCounts => vec![
                    (RETILE, 1, m.div_ceil(128)),
                    (INCUMBENT, n.div_ceil(32), m.div_ceil(64)),
                ],
            };
            assert_eq!(census.len(), expected.len());
            for (row, (name, width, height)) in census.iter().zip(expected) {
                assert_eq!(row.kernel, name);
                assert_eq!(
                    (row.grid_width, row.grid_height, row.grid_depth),
                    (width as u64, height as u64, EXPERTS as u64)
                );
                assert_eq!(
                    (row.threads_width, row.threads_height, row.threads_depth),
                    (128, 1, 1)
                );
                assert!(!row.encoder_concurrent);
            }
            for (input, before) in inputs.iter().zip(&before) {
                assert_eq!(tensor_backing_bytes(input), *before, "input mutated");
            }
        }
    }
}

#[test]
fn iq3_s_retile_small_count_ranges_are_disjoint() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 129, 128);
    let out = f.output(&ctx);
    let mut args = checked_args(
        &f.weight, &f.inner, &f.counts, &f.ids, &out, f.k, f.m, EXPERTS, f.n,
    )
    .unwrap();
    args.min_count = 1;
    args.max_count = 16;
    assert_eq!(args.n, 128); // Bucket stride must not shrink to the single panel.
    one_shot(&ctx, |enc| {
        enc.set_pipeline(&pipeline(&ctx, RETILE, 9216)?);
        enc.set_bytes(0, &args);
        for (index, t) in [&f.weight, &f.inner, &f.counts, &f.ids, &out]
            .into_iter()
            .enumerate()
        {
            enc.set_tensor(index + 1, t);
        }
        enc.set_threadgroup_memory(0, 9216);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 2,
                depth: EXPERTS,
            },
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    })
    .unwrap();
    f.assert_output(&out, |count| count <= 16);
    let small = tensor_f32_at_offset(&out);
    one_shot(&ctx, |enc| {
        encode_moe_down_f32_grouped_slots_generic_range(
            &ctx,
            enc,
            &f.weight,
            &f.inner,
            &f.counts,
            &f.ids,
            &out,
            f.k,
            f.m,
            EXPERTS,
            f.n,
            17,
            i32::MAX as u32,
        )
    })
    .unwrap();
    f.assert_output(&out, |_| true);
    let all = tensor_f32_at_offset(&out);
    for slot in 0..SLOTS {
        if f.slot_counts[slot] <= 16 {
            for row in 0..f.m {
                let i = slot * f.m + row;
                assert_eq!(
                    all[i].to_bits(),
                    small[i].to_bits(),
                    "large range touched small/unused slot"
                );
            }
        }
    }
}

#[test]
fn iq3_s_retile_rejects_unsafe_bindings_without_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 1, 40);
    let out = f.output(&ctx);
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&cmd);
    dispatch_census_begin();
    let ((), substitutions) = with_production(|| {
        // Preparation itself cannot write outputs or increment the counter.
        assert!(
            prepare_small_counts(
                &ctx, &enc, &f.weight, &f.inner, &f.counts, &f.ids, &out, f.k, f.m, EXPERTS, f.n,
            )
            .unwrap()
            .is_some()
        );
        assert!(
            prepare_small_counts(
                &ctx, &enc, &f.weight, &f.inner, &f.counts, &f.ids, &out, 2048, 4096, EXPERTS, 128,
            )
            .is_err()
        );
    });
    assert_eq!(substitutions, 0);
    for variant in [
        DownRetile::Incumbent,
        DownRetile::Blanket,
        DownRetile::SmallCounts,
    ] {
        for which in 0..5 {
            for fault in 0..5 {
                let mut tensors = [
                    f.weight.clone(),
                    f.inner.clone(),
                    f.counts.clone(),
                    f.ids.clone(),
                    out.clone(),
                ];
                let t = &mut tensors[which];
                match fault {
                    0 => t.offset += 1,
                    1 => t.offset = t.buffer.length() as u64,
                    2 => t.offset = u64::MAX - 15,
                    3 => t.dtype = GgmlType::F16,
                    _ => t.shape = vec![1],
                }
                // A one-element M=1 output is itself valid, but cannot match the input slot count.
                assert!(
                    encode_variant(
                        &ctx,
                        &enc,
                        variant,
                        &tensors[0],
                        &tensors[1],
                        &tensors[2],
                        &tensors[3],
                        &tensors[4],
                        f.k,
                        f.m,
                        EXPERTS,
                        f.n
                    )
                    .is_err()
                );
            }
        }
        let mut readonly = out.clone();
        readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(f.encode(&ctx, &enc, variant, &readonly).is_err());
        let alias = f.inner.view_subrange(0, out.shape.clone());
        assert!(f.encode(&ctx, &enc, variant, &alias).is_err());
        let mut misaligned = f.inner.clone();
        misaligned.offset += 4;
        assert!(
            encode_variant(
                &ctx,
                &enc,
                variant,
                &f.weight,
                &misaligned,
                &f.counts,
                &f.ids,
                &out,
                f.k,
                f.m,
                EXPERTS,
                f.n
            )
            .is_err()
        );
        for (k, m, e, n) in [
            (128, 1, EXPERTS, 40),
            (256, 0, EXPERTS, 40),
            (256, 1, 0, 40),
            (256, 1, EXPERTS, 0),
            (256, 1, EXPERTS, i32::MAX as usize + 1),
            (256, i32::MAX as usize, i32::MAX as usize, 40),
        ] {
            assert!(
                encode_variant(
                    &ctx, &enc, variant, &f.weight, &f.inner, &f.counts, &f.ids, &out, k, m, e, n
                )
                .is_err()
            );
        }
    }
    enc.end();
    assert!(dispatch_census_take().is_empty());
    let cmd = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin_concurrent(&cmd);
    assert!(f.encode(&ctx, &enc, DownRetile::SmallCounts, &out).is_err());
    enc.end();
}

#[test]
fn iq3_s_retile_scope_restores_on_nesting_unwind_and_other_threads() {
    assert_eq!(STATE.with(Cell::get), (None, 0));
    let (value, count) = with_variant(DownRetile::Blanket, || {
        STATE.with(|s| s.set((Some(DownRetile::Blanket), 3)));
        assert_eq!(
            with_variant(DownRetile::SmallCounts, || STATE.with(Cell::get)),
            ((Some(DownRetile::SmallCounts), 0), 0)
        );
        assert_eq!(with_production(|| STATE.with(Cell::get)), ((None, 0), 0));
        assert_eq!(STATE.with(Cell::get), (Some(DownRetile::Blanket), 3));
        let _ = std::panic::catch_unwind(|| with_production(|| panic!("restore production probe")));
        assert_eq!(STATE.with(Cell::get), (Some(DownRetile::Blanket), 3));
        let _ = std::panic::catch_unwind(|| {
            with_variant(DownRetile::SmallCounts, || panic!("restore variant probe"))
        });
        assert_eq!(STATE.with(Cell::get), (Some(DownRetile::Blanket), 3));
        assert_eq!(
            std::thread::spawn(|| STATE.with(Cell::get)).join().unwrap(),
            (None, 0)
        );
        7
    });
    assert_eq!((value, count), (7, 3));
    assert_eq!(STATE.with(Cell::get), (None, 0));
    assert_eq!(
        with_variant(DownRetile::Incumbent, || {
            assert_eq!(with_production(|| STATE.with(Cell::get)), ((None, 0), 0));
            STATE.with(Cell::get)
        }),
        ((Some(DownRetile::Incumbent), 0), 0)
    );
}

#[test]
fn iq3_s_retile_pipeline_limits_and_argument_abi() {
    assert_eq!(std::mem::size_of::<GenericMmArgs>(), 36);
    check_capacity(32, 128, 0, 9216, 9216).unwrap();
    for (width, threads, stat, limit) in [
        (16, 128, 0, 32768),
        (32, 127, 0, 32768),
        (32, 128, 1, 9216),
        (32, 128, usize::MAX, usize::MAX),
    ] {
        assert!(check_capacity(width, threads, stat, 9216, limit).is_err());
    }
}

fn with_preflight_fault<R>(fault: Option<(&'static str, bool)>, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<(&'static str, bool)>);
    impl Drop for Restore {
        fn drop(&mut self) {
            PREFLIGHT_FAULT.with(|state| state.set(self.0));
        }
    }
    let _restore = Restore(PREFLIGHT_FAULT.with(|state| state.replace(fault)));
    f()
}

#[test]
fn iq3_s_retile_production_preflight_falls_back_without_partial_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 129, 128);
    let inputs = [&f.weight, &f.inner, &f.counts, &f.ids];
    let before: Vec<_> = inputs.iter().map(|t| tensor_backing_bytes(t)).collect();
    for fault in [
        None,
        Some((RETILE, true)),
        Some((RETILE, false)),
        Some((INCUMBENT, true)),
        Some((INCUMBENT, false)),
    ] {
        let out = f.output(&ctx);
        let (result, substitutions) = with_production(|| {
            with_preflight_fault(fault, || {
                one_shot(&ctx, |enc| {
                    dispatch_census_begin();
                    let mut readonly = out.clone();
                    readonly.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
                    assert!(
                        prepare_small_counts(
                            &ctx, enc, &f.weight, &f.inner, &f.counts, &f.ids, &readonly, f.k, f.m,
                            EXPERTS, f.n,
                        )
                        .is_err(),
                        "capability fallback masked an invalid binding"
                    );
                    let prepared = prepare_small_counts(
                        &ctx, enc, &f.weight, &f.inner, &f.counts, &f.ids, &out, f.k, f.m, EXPERTS,
                        f.n,
                    )?;
                    assert!(
                        dispatch_census_take().is_empty(),
                        "preflight dispatched work"
                    );
                    assert_eq!(prepared.is_some(), fault.is_none());
                    dispatch_census_begin();
                    if let Some(down) = prepared {
                        down.encode()
                    } else {
                        encode_moe_down_f32_grouped_slots_generic_range(
                            &ctx,
                            enc,
                            &f.weight,
                            &f.inner,
                            &f.counts,
                            &f.ids,
                            &out,
                            f.k,
                            f.m,
                            EXPERTS,
                            f.n,
                            0,
                            i32::MAX as u32,
                        )
                    }
                })
            })
        });
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(substitutions, usize::from(fault.is_none()));
        let kernels: Vec<_> = census.iter().map(|r| r.kernel.as_str()).collect();
        assert_eq!(
            kernels,
            if fault.is_none() {
                vec![RETILE, INCUMBENT]
            } else {
                vec![INCUMBENT]
            }
        );
        if fault.is_none() {
            assert_eq!(
                (
                    census[0].grid_width,
                    census[0].grid_height,
                    census[0].grid_depth
                ),
                (1, 2, EXPERTS as u64)
            );
        }
        f.assert_output(&out, |_| true);
        for (tensor, bytes) in inputs.iter().zip(&before) {
            assert_eq!(&tensor_backing_bytes(tensor), bytes);
        }
    }
}

#[test]
fn iq3_s_retile_production_and_explicit_scopes_count_encoded_substitutions() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let f = Fixture::new(&ctx, 256, 1, 40);
    for variant in [
        None,
        Some(DownRetile::Incumbent),
        Some(DownRetile::SmallCounts),
        Some(DownRetile::Blanket),
    ] {
        let out = f.output(&ctx);
        dispatch_census_begin();
        let encode = || {
            one_shot(&ctx, |enc| {
                let prepared = prepare_small_counts(
                    &ctx, enc, &f.weight, &f.inner, &f.counts, &f.ids, &out, f.k, f.m, EXPERTS, f.n,
                )?;
                if let Some(down) = prepared {
                    down.encode()
                } else {
                    f.encode(&ctx, enc, DownRetile::Incumbent, &out)
                }
            })
        };
        let (result, substitutions) = match variant {
            Some(variant) => with_variant(variant, encode),
            None => with_production(encode),
        };
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(
            substitutions,
            usize::from(variant != Some(DownRetile::Incumbent))
        );
        let kernels: Vec<_> = census.iter().map(|r| r.kernel.as_str()).collect();
        assert_eq!(
            kernels,
            match variant {
                Some(DownRetile::Incumbent) => vec![INCUMBENT],
                Some(DownRetile::Blanket) => vec![RETILE],
                _ => vec![RETILE, INCUMBENT],
            }
        );
        f.assert_output(&out, |_| true);
    }
}

#[test]
fn iq3_s_retile_shared_composition_requires_explicit_policy_and_honors_fallback() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const WIDTH: usize = 256;
    const ROWS: usize = 33;
    const E: usize = 2;
    let bank = offset_tensor(
        &ctx,
        16,
        &vec![0; WIDTH * E * 110],
        16,
        vec![WIDTH as u64, WIDTH as u64, E as u64],
        GgmlType::IQ3_S,
    );
    let f32_tensor = |shape: Vec<u64>, fill: f32| {
        let count = shape.iter().product::<u64>() as usize;
        offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&vec![fill; count]),
            16,
            shape,
            GgmlType::F32,
        )
    };
    let i32_tensor = |shape: Vec<u64>, values: &[i32]| {
        offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(values),
            16,
            shape,
            GgmlType::I32,
        )
    };
    let input = f32_tensor(vec![WIDTH as u64, ROWS as u64], 1.0);
    let ids = i32_tensor(
        vec![1, ROWS as u64],
        &(0..ROWS).map(|i| (i % E) as i32).collect::<Vec<_>>(),
    );
    let weights = f32_tensor(vec![1, ROWS as u64], 1.0);
    let counts = i32_tensor(vec![E as u64], &[0; E]);
    let slots = i32_tensor(vec![(E * ROWS) as u64], &[0; E * ROWS]);
    let inner = f32_tensor(vec![WIDTH as u64, ROWS as u64], SENTINEL);
    let slot_out = f32_tensor(vec![WIDTH as u64, ROWS as u64], SENTINEL);
    for (explicit, variant, fault, expected) in [
        (false, None, None, 0),
        (false, Some(DownRetile::SmallCounts), None, 0),
        (true, Some(DownRetile::Incumbent), None, 0),
        (true, None, None, 1),
        (true, None, Some((RETILE, true)), 0),
        (true, None, Some((RETILE, false)), 0),
    ] {
        let output = f32_tensor(vec![WIDTH as u64, ROWS as u64], SENTINEL);
        let b = GroupedExperts {
            gate_bank: &bank,
            up_bank: &bank,
            down_bank: &bank,
            input: &input,
            ids: &ids,
            weights: &weights,
            counts: &counts,
            slots: &slots,
            inner: &inner,
            slot_out: &slot_out,
            output: &output,
        };
        dispatch_census_begin();
        let encode = || {
            with_preflight_fault(fault, || {
                one_shot(&ctx, |enc| {
                    if explicit {
                        encode_grouped_routed_experts_with_down_policy(
                            &ctx,
                            enc,
                            &b,
                            WIDTH,
                            WIDTH,
                            E,
                            1,
                            ROWS,
                            10.0,
                            GroupedDownPolicy::Iq3SSmallCounts,
                        )
                    } else {
                        encode_grouped_routed_experts(&ctx, enc, &b, WIDTH, WIDTH, E, 1, ROWS, 10.0)
                    }
                })
            })
        };
        let (result, substitutions) = match variant {
            Some(variant) => with_variant(variant, encode),
            None => with_production(encode),
        };
        let census = dispatch_census_take();
        result.unwrap();
        assert_eq!(substitutions, expected);
        assert_eq!(
            census.iter().filter(|r| r.kernel == RETILE).count(),
            expected
        );
        assert_eq!(census.iter().filter(|r| r.kernel == INCUMBENT).count(), 1);
        assert!(
            tensor_f32_at_offset(&output)
                .iter()
                .all(|v| v.is_finite() && *v == 0.0)
        );
        assert_offset_guards(&output, 16, 16);
    }
}
