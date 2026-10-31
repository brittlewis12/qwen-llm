//! Tiny native compositions, using the established guarded-buffer/lease helpers.
use super::*;
use crate::metal::expert_tensor_gate_up as gate_adapter;

const H: usize = 256;
const F: usize = 256;
const E: usize = 32;

struct Tiny {
    tensors: [MetalTensor; 11],
    half: MetalTensor,
    compact: [Vec<f32>; 3],
    rows: usize,
}

impl Tiny {
    fn new(ctx: &MetalContext, gate: GgmlType, down: GgmlType, rows: usize) -> Self {
        let _allocation = ctx.begin_allocation_transaction();
        let kinds = [gate, gate, down];
        let bank_bytes = kinds.map(|dtype| {
            let (block, bytes) = dtype.storage_layout().unwrap();
            H / block as usize * bytes as usize * F * E
        });
        let logical = [
            bank_bytes[0],
            bank_bytes[1],
            bank_bytes[2],
            H * rows * 4,
            TOP * rows * 4,
            TOP * rows * 4,
            E * 4,
            E * rows * 4,
            F * TOP * rows * 4,
            H * TOP * rows * 4,
            H * rows * 4,
            H * (rows + 32) * 2,
        ];
        let gpu: u64 = logical
            .iter()
            .map(|&n| {
                ctx.price_shared_buffer_upper((n + 2 * GUARD) as u64)
                    .unwrap()
                    .priced_upper_bytes
            })
            .sum();
        // Upload + guarded upload, snapshots/readbacks and compact sampled
        // oracles. No full decoded weight banks exist on either CPU or GPU.
        let cpu = logical.iter().sum::<usize>() as u64 * 6 + (32 << 20);
        let peak = gpu
            .checked_add(cpu)
            .unwrap()
            .checked_add(crate::glm5_next::memory::DYNAMIC_RESERVE_BYTES)
            .unwrap();
        assert!(peak <= 1 << 30, "conservative tiny peak {peak}");
        let admission = evaluate_metal_memory_admission(
            gpu + cpu,
            crate::glm5_next::memory::DYNAMIC_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        assert!(admission.is_admitted(), "{admission:?}");
        let mut compact = std::array::from_fn(|_| Vec::new());
        let banks: [MetalTensor; 3] = std::array::from_fn(|index| {
            let dtype = kinds[index];
            let mut native = native_compact(dtype, H);
            if dtype == GgmlType::Q6_K {
                for (i, block) in native.chunks_exact_mut(210).enumerate() {
                    block[208..210].copy_from_slice(
                        &half::f16::from_f32((i + 1) as f32 * 0.00003).to_le_bytes(),
                    );
                }
            }
            let desc = TensorDesc {
                name: "tiny_native_bank".into(),
                shape: vec![H as u64, WR as u64],
                dtype,
                shard_idx: 0,
                data_offset: 0,
                n_bytes: native.len() as u64,
            };
            compact[index] = crate::codec::dequant_to_f32(&desc, &native).unwrap();
            assert!(compact[index].iter().all(|v| v.is_finite()));
            let stride = native.len() / WR;
            let mut payload = Vec::with_capacity(bank_bytes[index]);
            for expert in 0..E {
                for row in 0..F {
                    let r = Self::bank_row(index, expert, row);
                    payload.extend_from_slice(&native[r * stride..(r + 1) * stride]);
                }
            }
            let mut t = guarded(ctx, &payload, vec![H as u64, F as u64, E as u64], dtype);
            t.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
            t
        });
        let [gate, up, down] = banks;
        let input: Vec<f32> = (0..H * rows)
            .map(|i| (((i * 37 + i / H * 19) % 211) as f32 - 105.0) * 0.03713)
            .collect();
        let tails = [15, 16, 17, 31, 32, 33, 63, 64, 65];
        let routes: Vec<i32> = (0..rows * TOP)
            .map(|s| {
                let lane = s % TOP;
                if lane == 1 {
                    let mut end = 0;
                    for (index, count) in tails.iter().enumerate() {
                        end += count;
                        if s / TOP < end {
                            return (E - 1 - index) as i32;
                        }
                    }
                }
                lane as i32
            })
            .collect();
        let weights: Vec<f32> = (0..TOP * rows).map(Fixture::route_weight).collect();
        let tensors = [
            gate,
            up,
            down,
            guarded(
                ctx,
                bytemuck::cast_slice(&input),
                vec![H as u64, rows as u64],
                GgmlType::F32,
            ),
            guarded(
                ctx,
                bytemuck::cast_slice(&routes),
                vec![TOP as u64, rows as u64],
                GgmlType::I32,
            ),
            guarded(
                ctx,
                bytemuck::cast_slice(&weights),
                vec![TOP as u64, rows as u64],
                GgmlType::F32,
            ),
            guarded(ctx, &vec![0xff; logical[6]], vec![E as u64], GgmlType::I32),
            guarded(
                ctx,
                &vec![0xff; logical[7]],
                vec![(E * rows) as u64],
                GgmlType::I32,
            ),
            guarded(
                ctx,
                &vec![0xff; logical[8]],
                vec![F as u64, (TOP * rows) as u64],
                GgmlType::F32,
            ),
            guarded(
                ctx,
                &vec![0xff; logical[9]],
                vec![H as u64, (TOP * rows) as u64],
                GgmlType::F32,
            ),
            guarded(
                ctx,
                &vec![0xff; logical[10]],
                vec![H as u64, rows as u64],
                GgmlType::F32,
            ),
        ];
        let half = guarded(
            ctx,
            &vec![0xff; logical[11]],
            vec![H as u64, (rows + 32) as u64],
            GgmlType::F16,
        );
        Self {
            tensors,
            half,
            compact,
            rows,
        }
    }

    fn bank_row(index: usize, expert: usize, row: usize) -> usize {
        (expert * 3 + row + index * 2) % WR
    }

    fn bindings(t: &[MetalTensor; 11]) -> GroupedExperts<'_> {
        let [
            gate_bank,
            up_bank,
            down_bank,
            input,
            ids,
            weights,
            counts,
            slots,
            inner,
            slot_out,
            output,
        ] = t;
        GroupedExperts {
            gate_bank,
            up_bank,
            down_bank,
            input,
            ids,
            weights,
            counts,
            slots,
            inner,
            slot_out,
            output,
        }
    }

    fn policy(&self) -> GroupedDownPolicy {
        if self.tensors[2].dtype == GgmlType::Q5_K {
            GroupedDownPolicy::Q5KTensor
        } else {
            GroupedDownPolicy::Incumbent
        }
    }

    fn encode(&self, ctx: &MetalContext, enc: &KernelEncoder, lane: usize, limit: f32) {
        let b = Self::bindings(&self.tensors);
        if lane == 2 {
            gate_adapter::prepare(
                ctx,
                enc,
                &b,
                &self.half,
                H,
                F,
                E,
                TOP,
                self.rows,
                limit,
                self.policy(),
            )
            .unwrap()
            .expect("M5 candidate")
            .encode()
            .unwrap();
            return;
        }
        if lane == 0 {
            encode_grouped_routed_experts_with_down_policy(
                ctx,
                enc,
                &b,
                H,
                F,
                E,
                TOP,
                self.rows,
                limit,
                self.policy(),
            )
            .unwrap();
            return;
        }
        // F32-input tensor is the bitwise reference for copy-inclusive F16,
        // NOT for the incumbent generic reduction order.
        encode_moe_route_bucket_slots_f32(
            ctx,
            enc,
            &b.ids.view_subrange(0, vec![(TOP * self.rows) as u64]),
            b.counts,
            b.slots,
            E,
            self.rows,
            TOP,
        )
        .unwrap();
        encode_moe_swiglu_f32_grouped_slots_tensor(
            ctx,
            enc,
            b.gate_bank,
            b.up_bank,
            b.input,
            b.counts,
            b.slots,
            b.inner,
            H,
            F,
            E,
            TOP,
            self.rows,
            Some(limit),
        )
        .unwrap();
        if self.policy() == GroupedDownPolicy::Q5KTensor {
            adapter::prepare(
                ctx,
                enc,
                b.down_bank,
                b.inner,
                b.counts,
                b.slots,
                b.slot_out,
                F,
                H,
                E,
                TOP,
                self.rows,
            )
            .unwrap()
            .unwrap()
            .encode()
            .unwrap();
        } else {
            encode_moe_down_f32_grouped_slots_generic(
                ctx,
                enc,
                b.down_bank,
                b.inner,
                b.counts,
                b.slots,
                b.slot_out,
                F,
                H,
                E,
                self.rows,
            )
            .unwrap();
        }
        encode_moe_weighted_sum_packed_f32(
            ctx, enc, b.slot_out, b.weights, b.output, H, TOP, self.rows,
        )
        .unwrap();
    }

    fn oracle(&self, limit: f32) {
        let input = read(&self.tensors[3]);
        let routes: &[i32] =
            bytemuck::cast_slice(&backing(&self.tensors[4])[GUARD..GUARD + TOP * self.rows * 4]);
        let inner = read(&self.tensors[8]);
        let slots = read(&self.tensors[9]);
        let out = read(&self.tensors[10]);
        let half = |v: f32| half::f16::from_f32(v).to_f64();
        let mut samples: [Vec<(f64, f64, bool)>; 2] = std::array::from_fn(|_| Vec::new());
        let mut clamped = [false; 3];
        for token in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, self.rows - 1] {
            for lane in 0..TOP {
                let slot = token * TOP + lane;
                let expert = routes[slot] as usize;
                for row in [0, 1, 63, 64, F - 1] {
                    for use_half in [false, true] {
                        let cv = |v: f32| if use_half { half(v) } else { f64::from(v) };
                        let dot = |index| {
                            let wr = Self::bank_row(index, expert, row);
                            (0..H)
                                .map(|i| {
                                    cv(self.compact[index][wr * H + i]) * cv(input[token * H + i])
                                })
                                .sum::<f64>()
                        };
                        let gate = dot(0);
                        let up = dot(1);
                        clamped[0] |= gate > f64::from(limit);
                        clamped[1] |= up > f64::from(limit);
                        clamped[2] |= up < -f64::from(limit);
                        let g = gate.min(f64::from(limit));
                        let u = up.clamp(-f64::from(limit), f64::from(limit));
                        let expected = g / (1.0 + (-g).exp()) * u;
                        let actual = f64::from(inner[slot * F + row]);
                        samples[0].push((actual, expected, use_half));
                        let wr = Self::bank_row(2, expert, row);
                        let expected: f64 = (0..F)
                            .map(|i| cv(self.compact[2][wr * F + i]) * cv(inner[slot * F + i]))
                            .sum();
                        let actual = f64::from(slots[slot * H + row]);
                        samples[1].push((actual, expected, use_half));
                    }
                }
            }
            for row in [0, 1, 63, 64, H - 1] {
                let expected: f64 = (0..TOP)
                    .map(|lane| {
                        let s = token * TOP + lane;
                        f64::from(slots[s * H + row]) * f64::from(Fixture::route_weight(s))
                    })
                    .sum();
                assert!(
                    (f64::from(out[token * H + row]) - expected).abs()
                        <= 2e-6 * (1.0 + expected.abs())
                );
            }
        }
        if limit == 0.0001 {
            assert_eq!(
                clamped, [true; 3],
                "both sides of the asymmetric epilogue exercised"
            );
        }
        for stage in samples {
            let scale = stage
                .iter()
                .map(|(_, expected, _)| expected.abs())
                .fold(0.0f64, f64::max);
            assert!(scale > 0.0);
            for (actual, expected, use_half) in stage {
                let tolerance = if use_half { 2e-3 } else { 1e-2 };
                assert!(
                    actual.is_finite() && (actual - expected).abs() <= tolerance * scale + 1e-5,
                    "sample {actual} vs {expected}, Half={use_half}, scale={scale}"
                );
            }
        }
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease; conservative total tiny peak <=1GiB"]
fn m5_gate_up_native_compositions_tensor_f32_f16_bitwise_and_sampled_f64() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    for (gate, down) in [
        (GgmlType::Q4_K, GgmlType::Q5_K),
        (GgmlType::Q4_K, GgmlType::Q6_K),
        (GgmlType::Q5_K, GgmlType::Q6_K),
    ] {
        for rows in [129, 1024, 2048] {
            objc2::rc::autoreleasepool(|_| {
                let f = Tiny::new(&ctx, gate, down, rows);
                let before: Vec<_> = f.tensors[..6].iter().map(fingerprint).collect();
                for limit in [0.0001, 0.03, 7.0] {
                    let mut tensor_reference = None;
                    for lane in 0..3 {
                        for t in &f.tensors[6..] {
                            poison(t);
                        }
                        poison(&f.half);
                        let census = run(&ctx, |enc| f.encode(&ctx, enc, lane, limit));
                        f.oracle(limit);
                        let result: Vec<Vec<u32>> = f.tensors[8..]
                            .iter()
                            .map(|t| {
                                read(t)
                                    .iter()
                                    .map(|v| {
                                        assert!(v.is_finite());
                                        v.to_bits()
                                    })
                                    .collect()
                            })
                            .collect();
                        if lane == 1 {
                            tensor_reference = Some(result);
                        } else if lane == 2 {
                            assert_eq!(tensor_reference.as_ref().unwrap(), &result);
                            assert_eq!(census.len(), 6);
                            assert_eq!(census[0].kernel, "kernel_scatter_offset_f32_to_f16");
                            assert_eq!(census[1].kernel, "kernel_moe_route_bucket_slots_f32");
                            assert!(census[2].kernel.contains("f16act_grouped_slots_tensor"));
                            assert!(census[3].kernel.contains("up_clamped"));
                            assert!(census[4].kernel.contains(if down == GgmlType::Q5_K {
                                "f32_grouped_slots_tensor"
                            } else {
                                "f32_grouped_slots_generic"
                            }));
                            assert_eq!(census[5].kernel, "kernel_moe_weighted_sum_packed_f32");
                            for row in &census[2..4] {
                                assert_eq!(row.grid_width, rows.div_ceil(32) as u64);
                                assert_eq!(row.threads_width, 128);
                            }
                            let counts: &[i32] =
                                bytemuck::cast_slice(&backing(&f.tensors[6])[GUARD..GUARD + E * 4]);
                            assert!(counts.contains(&0) && counts.contains(&(rows as i32)));
                            if rows >= 1024 {
                                for count in [15, 16, 17, 31, 32, 33, 63, 64, 65] {
                                    assert!(counts.contains(&count), "bucket edge {count}");
                                }
                            }
                            let converted: &[u16] = bytemuck::cast_slice(
                                &backing(&f.half)[GUARD..GUARD + H * rows * 2],
                            );
                            for (&actual, original) in converted.iter().zip(read(&f.tensors[3])) {
                                assert_eq!(actual, half::f16::from_f32(original).to_bits());
                            }
                            assert!(
                                backing(&f.half)
                                    [GUARD + H * rows * 2..GUARD + f.half.n_bytes() as usize]
                                    .iter()
                                    .all(|&b| b == 0xff),
                                "inactive admitted Half rows were written"
                            );
                        }
                        for t in &f.tensors {
                            guards(t);
                        }
                        guards(&f.half);
                    }
                }
                for (t, hash) in f.tensors[..6].iter().zip(before) {
                    assert_eq!(fingerprint(t), hash);
                }
            });
        }
    }
}

#[test]
#[ignore = "actual M5, validation ON, production lease; all refusals precede mutation"]
fn m5_gate_up_whole_composition_preflight_failures() {
    let _lease = validation_lease();
    let ctx = MetalContext::new().unwrap();
    require_m5(&ctx);
    let f = Tiny::new(&ctx, GgmlType::Q4_K, GgmlType::Q5_K, 129);
    let prepare =
        |enc: &KernelEncoder, t: &[MetalTensor; 11], half: &MetalTensor, limit, policy| {
            let b = Tiny::bindings(t);
            gate_adapter::prepare(&ctx, enc, &b, half, H, F, E, TOP, f.rows, limit, policy)
                .map(|p| p.is_some())
        };
    let before: Vec<_> = f.tensors.iter().map(fingerprint).collect();
    let half_before = fingerprint(&f.half);
    for index in 0..6 {
        let _missing = gate_adapter::FaultScope::new(gate_adapter::Fault::Missing(index));
        assert_no_mutation(&ctx, |enc| {
            assert!(matches!(
                prepare(enc, &f.tensors, &f.half, 7.0, f.policy()),
                Err(MetalError::NoFunction(_))
            ))
        });
    }
    for index in 0..6 {
        let _capacity = gate_adapter::FaultScope::new(gate_adapter::Fault::Capacity(index));
        assert_no_mutation(&ctx, |enc| {
            assert!(!prepare(enc, &f.tensors, &f.half, 7.0, f.policy()).unwrap())
        });
    }
    {
        let _both = gate_adapter::FaultScope::new(gate_adapter::Fault::CapacityAndMissing);
        assert_no_mutation(&ctx, |enc| {
            assert!(matches!(
                prepare(enc, &f.tensors, &f.half, 7.0, f.policy()),
                Err(MetalError::NoFunction(_))
            ))
        });
    }
    {
        let _capacity = gate_adapter::FaultScope::new(gate_adapter::Fault::Capacity(1));
        let _missing_down = adapter::FaultScope::new(adapter::PreflightFault::MissingPipeline);
        assert_no_mutation(&ctx, |enc| {
            assert!(matches!(
                prepare(enc, &f.tensors, &f.half, 7.0, f.policy()),
                Err(MetalError::NoFunction(_))
            ))
        });
    }
    for index in 0..11 {
        for fault in 0..9 {
            let mut bad = f.tensors.clone();
            match fault {
                0 => bad[index].shape = vec![u64::MAX, 2],
                1 => bad[index].offset = bad[index].buffer.length() as u64,
                2 => bad[index].offset += 1,
                3 if index >= 6 => {
                    bad[index].provenance = MetalTensorProvenance::OwnedWeightReadOnly
                }
                4 if index >= 6 => {
                    bad[index] = MetalTensor {
                        shape: bad[index].shape.clone(),
                        dtype: bad[index].dtype,
                        ..f.tensors[3].clone()
                    }
                }
                5 if index >= 6 => {
                    bad[index] = MetalTensor {
                        shape: bad[index].shape.clone(),
                        dtype: bad[index].dtype,
                        ..f.tensors[if index == 8 { 9 } else { 8 }].clone()
                    }
                }
                6 => bad[index].offset = u64::MAX - 3,
                7 => bad[index].shape[0] -= 1,
                8 if index >= 3 => {
                    bad[index].dtype = if bad[index].dtype == GgmlType::I32 {
                        GgmlType::F32
                    } else {
                        GgmlType::I32
                    }
                }
                _ => continue,
            }
            let _capacity = gate_adapter::FaultScope::new(gate_adapter::Fault::Capacity(1));
            assert_no_mutation(&ctx, |enc| {
                assert!(
                    prepare(enc, &bad, &f.half, 7.0, f.policy()).is_err(),
                    "binding {index} fault {fault}"
                )
            });
        }
    }
    for limit in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert_no_mutation(&ctx, |enc| {
            assert!(prepare(enc, &f.tensors, &f.half, limit, f.policy()).is_err())
        });
    }
    for fault in 0..6 {
        let mut half = f.half.clone();
        match fault {
            0 => half.shape = vec![u64::MAX, 2],
            1 => half.offset = half.buffer.length() as u64,
            2 => half.offset += 2,
            3 => half.provenance = MetalTensorProvenance::OwnedWeightReadOnly,
            4 => {
                half = MetalTensor {
                    shape: f.half.shape.clone(),
                    dtype: GgmlType::F16,
                    ..f.tensors[3].clone()
                }
            }
            _ => half.shape[1] = (f.rows - 1) as u64,
        }
        assert_no_mutation(&ctx, |enc| {
            assert!(prepare(enc, &f.tensors, &half, 7.0, f.policy()).is_err())
        });
    }
    // Q5 rollback is an explicit whole-candidate refusal, not an override.
    assert_no_mutation(&ctx, |enc| {
        assert!(!prepare(enc, &f.tensors, &f.half, 7.0, GroupedDownPolicy::Incumbent).unwrap())
    });
    objc2::rc::autoreleasepool(|_| {
        let command = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin_concurrent(&command);
        assert!(prepare(&enc, &f.tensors, &f.half, 7.0, f.policy()).is_err());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.end();
    });
    objc2::rc::autoreleasepool(|_| {
        let command = ctx.queue.commandBuffer().unwrap();
        let mut enc = KernelEncoder::begin(&command);
        enc.ended = true;
        assert!(prepare(&enc, &f.tensors, &f.half, 7.0, f.policy()).is_err());
        assert!(enc.bound_pipeline_name.borrow().is_none());
        enc.ended = false;
        enc.end();
    });
    for (t, hash) in f.tensors.iter().zip(before) {
        assert_eq!(fingerprint(t), hash);
    }
    assert_eq!(fingerprint(&f.half), half_before);
    let baseline_census = run(&ctx, |enc| f.encode(&ctx, enc, 0, 7.0));
    let baseline: Vec<_> = f.tensors[8..].iter().map(read).collect();
    {
        let _capacity = gate_adapter::FaultScope::new(gate_adapter::Fault::Capacity(1));
        let fallback = run(&ctx, |enc| {
            assert!(!prepare(enc, &f.tensors, &f.half, 7.0, f.policy()).unwrap());
            f.encode(&ctx, enc, 0, 7.0);
        });
        assert_eq!(
            fallback.iter().map(|r| &r.kernel).collect::<Vec<_>>(),
            baseline_census
                .iter()
                .map(|r| &r.kernel)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            baseline,
            f.tensors[8..].iter().map(read).collect::<Vec<_>>()
        );
        assert_eq!(
            fingerprint(&f.half),
            half_before,
            "fallback must not convert"
        );
    }
    let routes = &f.tensors[4];
    unsafe {
        let ids = routes
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(routes.offset as usize)
            .cast::<i32>();
        ids.write(-1);
        ids.add(TOP * f.rows - 1).write(E as i32);
    }
    for t in &f.tensors[6..] {
        poison(t);
    }
    run(&ctx, |enc| f.encode(&ctx, enc, 2, 7.0));
    for t in &f.tensors[8..10] {
        let values = read(t);
        for slot in [0, TOP * f.rows - 1] {
            assert!(
                values[slot * H..(slot + 1) * H]
                    .iter()
                    .all(|v| v.to_bits() == u32::MAX),
                "dropped slot must remain untouched"
            );
        }
        guards(t);
    }
}
