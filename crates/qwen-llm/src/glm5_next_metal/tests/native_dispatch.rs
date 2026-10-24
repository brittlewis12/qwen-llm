//! Synthetic GLM dispatch checks; no checkpoint or full-model allocation.
use super::*;
use crate::deepseek_v4_oracle::{hyper_connection_pre, rms_norm};
use crate::glm5_next::{BlockKind, FfnKind};
use crate::metal::{Mhc4PreInputs, Mhc4PreOutputs, Mhc4PrePartials};

fn config() -> Glm5NextConfig {
    Glm5NextConfig {
        architecture: crate::glm5_next::ARCHITECTURE_NAME.into(),
        stored_block_count: 46,
        nextn_layer_count: 1,
        context_length: 8192,
        hidden_size: 4096,
        dense_ffn_size: 12288,
        head_count: 64,
        vocab_size: crate::glm5_next::RELEASE_VOCAB_SIZE,
        rms_epsilon: 1e-5,
        layer_norm_epsilon: 1e-6,
        expert_count: 288,
        expert_used_count: 8,
        expert_ffn_size: 2048,
        shared_expert_count: 1,
        shared_expert_ffn_size: 2048,
        leading_dense_block_count: 3,
        expert_weights_scale: 2.5,
        swiglu_clamp: 10.0,
        q_lora_rank: 1536,
        kv_lora_rank: 512,
        mla_key_head_dim: 256,
        mla_value_head_dim: 256,
        kda_head_dim: 128,
        kda_conv_kernel: 4,
        kda_gate_lower_bound: -5.0,
        indexer_head_count: 32,
        indexer_head_dim: 128,
        indexer_top_k: 2048,
        indexer_pool: 4,
        hc_streams: 4,
        hc_sinkhorn_iterations: 20,
        hc_epsilon: 1e-6,
        blocks: (0..45)
            .map(|i| BlockKind {
                mixer: if i % 4 == 3 {
                    MixerKind::Mla
                } else {
                    MixerKind::Kda
                },
                ffn: if i < 3 { FfnKind::Dense } else { FfnKind::Moe },
            })
            .collect(),
    }
}

#[test]
fn generic_hc_ledger_prices_decode_and_bounded_packed_scratch() {
    let c = config();
    let price = memory::apple_16k_price;
    let specs = memory::generic_hc_scratch_specs(&c);
    assert_eq!(specs.len(), 2);
    assert_eq!(
        specs.iter().map(|s| s.bytes().unwrap()).sum::<u64>(),
        32 * u64::from(c.hidden_size)
    );
    let decode_extra = memory::priced(&specs, &price).unwrap();
    for rows in [0, 1, 3, 32, 512] {
        let packed_specs = memory::packed_generic_hc_scratch_specs(&c, rows);
        assert_eq!(packed_specs.len(), usize::from(rows > 0));
        assert_eq!(
            packed_specs.iter().map(|s| s.bytes().unwrap()).sum::<u64>(),
            16 * u64::from(c.hidden_size) * rows
        );
        if rows > 0 {
            assert_eq!(packed_specs[0].shape, [4 * u64::from(c.hidden_size), rows]);
        }
        let packed_extra = memory::priced(&packed_specs, &price).unwrap();
        if rows == 512 {
            assert_eq!(packed_extra, 32 * 1024 * 1024);
        }
        let old = Glm5NextMemoryLedger::new(&c, 1234, 512, rows, &price).unwrap();
        let q8 =
            Glm5NextMemoryLedger::new_with_generic_hc(&c, 1234, 512, rows, false, &price).unwrap();
        let native =
            Glm5NextMemoryLedger::new_with_generic_hc(&c, 1234, 512, rows, true, &price).unwrap();
        assert_eq!(old.terms(), q8.terms());
        let extra = decode_extra + packed_extra;
        assert_eq!(native.peak_bytes() - q8.peak_bytes(), extra);
        assert_eq!(
            native.session_buffer_bytes() - q8.session_buffer_bytes(),
            extra
        );
        let term = |ledger: &Glm5NextMemoryLedger, name| {
            ledger
                .terms()
                .into_iter()
                .find(|(n, _)| *n == name)
                .unwrap()
                .1
        };
        assert_eq!(
            term(&native, "decode_scratch") - term(&q8, "decode_scratch"),
            decode_extra
        );
        assert_eq!(
            term(&native, "packed_scratch") - term(&q8, "packed_scratch"),
            packed_extra
        );
    }
    assert!(
        memory::packed_generic_hc_scratch_specs(&c, u64::MAX)[0]
            .bytes()
            .is_none()
    );
}

#[test]
fn packed_hc_allocation_is_optional_and_uses_exact_chunk_capacity() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let mut c = config();
    c.hidden_size = 64;
    c.head_count = 2;
    c.kda_head_dim = 32;
    c.q_lora_rank = 64;
    c.kv_lora_rank = 32;
    c.mla_key_head_dim = 32;
    c.mla_value_head_dim = 32;
    c.dense_ffn_size = 128;
    c.expert_count = 3;
    c.expert_used_count = 2;
    c.expert_ffn_size = 128;
    c.shared_expert_ffn_size = 128;
    c.vocab_size = 64;
    for needed in [false, true] {
        let p = PackedScratch::new(&ctx, &c, 17, 32, needed).unwrap();
        assert_eq!(p.hc_normalized.is_some(), needed);
        if let Some(normalized) = &p.hc_normalized {
            assert_eq!(normalized.shape, [256, 17]);
            assert_eq!(normalized.dtype, GgmlType::F32);
            assert_eq!(
                normalized.n_elements() * 4,
                16 * u64::from(c.hidden_size) * 17
            );
            for rows in [17usize, 3, 1] {
                let active = rows_view(normalized, rows);
                assert_eq!(active.shape, [256, rows as u64]);
                assert_eq!(active.offset, normalized.offset);
            }
        }
    }
}

#[test]
fn generic_hc_capacity_advice_includes_actual_chunk_capacity() {
    let c = config();
    let price = memory::apple_16k_price;
    for rows in [0u64, 1, 512] {
        let smallest = rows.max(1);
        let minimum =
            Glm5NextMemoryLedger::new_with_generic_hc(&c, 0, smallest, rows, true, &price).unwrap();
        assert_eq!(
            Glm5NextMemoryLedger::max_capacity_with_generic_hc(
                &c,
                0,
                rows,
                minimum.peak_bytes() - 1,
                true,
                &price,
            )
            .unwrap(),
            None
        );
        let cap = Glm5NextMemoryLedger::max_capacity_with_generic_hc(
            &c,
            0,
            rows,
            minimum.peak_bytes(),
            true,
            &price,
        )
        .unwrap()
        .unwrap();
        let fits = |n| {
            Glm5NextMemoryLedger::new_with_generic_hc(&c, 0, n, rows, true, &price)
                .unwrap()
                .peak_bytes()
        };
        assert!(fits(cap) <= minimum.peak_bytes());
        assert!(cap == u64::from(c.context_length) || fits(cap + 1) > minimum.peak_bytes());
        let q8_budget = Glm5NextMemoryLedger::new(&c, 0, smallest, rows, &price)
            .unwrap()
            .peak_bytes();
        assert_eq!(
            Glm5NextMemoryLedger::max_capacity_with_generic_hc(
                &c, 0, rows, q8_budget, true, &price,
            )
            .unwrap(),
            None
        );
    }
}

#[test]
fn host_limited_capacity_advice_fits_combined_admission_budget() {
    use crate::metal::{HostMemoryBudget, MetalMemoryAdmissionReason, MetalMemorySignals};

    let c = config();
    let price = memory::apple_16k_price;
    let retained = 1234;
    let rows = 512;
    let peak = |capacity| {
        Glm5NextMemoryLedger::new_with_generic_hc(&c, retained, capacity, rows, true, &price)
            .unwrap()
            .peak_bytes()
    };
    let requested = u64::from(c.context_length);
    let required = peak(requested);
    let host_remaining = peak(1024);
    assert!(host_remaining < required);
    let physical = required * 10;
    let signals = MetalMemorySignals {
        recommended_max_bytes: required * 2,
        current_allocated_bytes: 0,
        process_limit_remaining_bytes: Some(required * 2),
        host_budget: Some(HostMemoryBudget {
            physical_bytes: physical,
            conservative_available_bytes: host_remaining,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    };
    assert!(evaluate_metal_memory_admission(required, 0, signals, true).is_admitted());
    // Exercise CPU availability without treating it as a second Metal ceiling.
    let denied =
        crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(0, required, 0, signals, true);
    assert_eq!(
        denied.reason,
        MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    assert_eq!(signals.host_remaining_bytes(), Some(host_remaining));
    assert!(denied.working_set_headroom_bytes.unwrap() > required);
    assert!(signals.process_limit_remaining_bytes.unwrap() > required);
    let budget = admission_budget(&denied);
    assert_eq!(budget, host_remaining);
    let fitting = Glm5NextMemoryLedger::max_capacity_with_generic_hc(
        &c, retained, rows, budget, true, &price,
    )
    .unwrap()
    .unwrap();
    assert!((1024..requested).contains(&fitting));
    assert!(
        crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            0,
            peak(fitting),
            0,
            signals,
            true,
        )
        .is_admitted()
    );
    assert_eq!(
        crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
            0,
            peak(fitting + 1),
            0,
            signals,
            true,
        )
        .reason,
        MetalMemoryAdmissionReason::HostAvailabilityInsufficient
    );
    assert_eq!(
        Glm5NextMemoryLedger::max_capacity_with_generic_hc(
            &c,
            retained,
            rows,
            denied.working_set_headroom_bytes.unwrap(),
            true,
            &price,
        )
        .unwrap(),
        Some(requested)
    );
}

#[test]
fn capacity_advice_budget_is_zero_without_valid_host_headroom() {
    use crate::metal::{HostMemoryBudget, MetalMemorySignals};

    for host_budget in [
        None,
        Some(HostMemoryBudget {
            physical_bytes: 0,
            conservative_available_bytes: 0,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
        Some(HostMemoryBudget {
            physical_bytes: 1000,
            conservative_available_bytes: 1001,
            external_gap_bytes: 0,
            private_shadow_bytes: 0,
        }),
    ] {
        let signals = MetalMemorySignals {
            recommended_max_bytes: 1000,
            current_allocated_bytes: 0,
            process_limit_remaining_bytes: Some(1000),
            host_budget,
        };
        let admission = evaluate_metal_memory_admission(100, 0, signals, true);
        assert_eq!(signals.host_remaining_bytes(), None);
        assert!(!admission.is_admitted());
        assert_eq!(admission_budget(&admission), 0);
    }
}

fn tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
}

fn empty(ctx: &MetalContext, shape: Vec<u64>) -> MetalTensor {
    tensor(
        ctx,
        &vec![f32::NAN; shape.iter().product::<u64>() as usize],
        shape,
    )
}

fn run(ctx: &MetalContext, encode: impl FnOnce(&KernelEncoder)) {
    let command = ctx.queue.commandBuffer().unwrap();
    let enc = KernelEncoder::begin(&command);
    encode(&enc);
    enc.end();
    command.commit();
    wait_completed(&command).unwrap();
}

// Seed compressed payloads, pin each scale finite, then use the independent
// linked GGML decoder for the CPU oracle (never a runtime weight expansion).
fn bank(ctx: &MetalContext, dtype: GgmlType, shape: Vec<u64>) -> (MetalTensor, Vec<f32>) {
    let n = shape.iter().product::<u64>() as usize;
    let mut bytes = Vec::new();
    if matches!(dtype, GgmlType::F32 | GgmlType::F16 | GgmlType::BF16) {
        for i in 0..n {
            let v = ((i * 23 + 7) % 97) as f32 * 0.00317 - 0.14;
            match dtype {
                GgmlType::F32 => bytes.extend(v.to_le_bytes()),
                GgmlType::F16 => bytes.extend(half::f16::from_f32(v).to_le_bytes()),
                GgmlType::BF16 => bytes.extend(half::bf16::from_f32(v).to_le_bytes()),
                _ => unreachable!(),
            }
        }
    } else {
        let (block, size) = dtype.storage_layout().unwrap();
        for ordinal in 0..n / block as usize {
            let mut b: Vec<u8> = (0..size as usize)
                .map(|i| (ordinal * 73 + i * 29 + ordinal / 7 * 11) as u8)
                .collect();
            if dtype == GgmlType::MXFP4 {
                b[0] = 122;
            } else {
                let offsets: &[usize] = match dtype {
                    GgmlType::Q2_K => &[80, 82],
                    GgmlType::Q3_K => &[108],
                    GgmlType::Q6_K => &[208],
                    GgmlType::Q4_K | GgmlType::Q5_K | GgmlType::Q4_1 => &[0, 2],
                    _ => &[0],
                };
                for (j, &offset) in offsets.iter().enumerate() {
                    let scale = (1 + (ordinal + j) % 7) as f32 * 0.00013;
                    b[offset..offset + 2]
                        .copy_from_slice(&half::f16::from_f32(scale).to_le_bytes());
                }
            }
            bytes.extend(b);
        }
    }
    let desc = TensorDesc {
        name: "native_dispatch".into(),
        shape: shape.clone(),
        dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: bytes.len() as u64,
    };
    let decoded = crate::codec::dequant_to_f32(&desc, &bytes).unwrap();
    (
        MetalTensor::from_bytes(ctx, &bytes, shape, dtype).unwrap(),
        decoded,
    )
}

fn close(label: &str, got: &[f32], want: &[f32], tolerance: f32) {
    assert_eq!(got.len(), want.len());
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= tolerance * (1.0 + b.abs()),
            "{label}[{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn native_hc_runtime_dispatch_matches_oracle_and_serial_rows() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let h = 64usize;
    let mut c = config();
    c.hidden_size = h as u32;
    let scratch = GenericHcScratch::new(&ctx, &c).unwrap();
    const MAX_ROWS: usize = 17;
    let mut buffers = SpecBuffers::allocate(
        &ctx,
        "test_packed_hc",
        &memory::packed_generic_hc_scratch_specs(&c, MAX_ROWS as u64),
    )
    .unwrap();
    let normalized = buffers.take("hc_normalized").unwrap();
    buffers.finish().unwrap();
    let scale = [0.8, -0.45, 1.1];
    let base: Vec<f32> = (0..24).map(|i| (i as f32 - 11.5) * 0.03).collect();
    let norm: Vec<f32> = (0..h).map(|i| 0.5 + (i % 11) as f32 * 0.09).collect();
    let scale_t = tensor(&ctx, &scale, vec![3]);
    let base_t = tensor(&ctx, &base, vec![24]);
    let norm_t = tensor(&ctx, &norm, vec![h as u64]);
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q6_K,
        GgmlType::IQ4_NL,
        GgmlType::MXFP4,
    ] {
        let (mix, decoded) = bank(&ctx, dtype, vec![(4 * h) as u64, 24]);
        for rows in [16usize, 3, 1] {
            write_f32(&normalized, &vec![f32::NAN; 4 * h * MAX_ROWS]).unwrap();
            let residual: Vec<f32> = (0..4 * h * rows)
                .map(|i| ((i * 7 + i / (4 * h) * 3) % 79) as f32 * 0.01713 - 0.6)
                .collect();
            let residual_t = tensor(&ctx, &residual, vec![h as u64, 4, rows as u64]);
            let make = || {
                [
                    empty(&ctx, vec![24, rows as u64]),
                    empty(&ctx, vec![4, rows as u64]),
                    empty(&ctx, vec![4, rows as u64]),
                    empty(&ctx, vec![4, 4, rows as u64]),
                    empty(&ctx, vec![h as u64, rows as u64]),
                    empty(&ctx, vec![h as u64, rows as u64]),
                ]
            };
            let packed = make();
            let serial = make();
            let dots = empty(&ctx, vec![24, memory::hc_chunks(&c), rows as u64]);
            let sumsq = empty(&ctx, vec![memory::hc_chunks(&c), rows as u64]);
            let encode = |enc: &KernelEncoder,
                          out: &[MetalTensor; 6],
                          start: usize,
                          count: usize,
                          packed_scratch: bool| {
                let slice = |t: &MetalTensor| {
                    let mut shape = t.shape.clone();
                    let width = shape[..shape.len() - 1].iter().product::<u64>();
                    *shape.last_mut().unwrap() = count as u64;
                    t.view_subrange(start as u64 * width, shape)
                };
                let normalized_view = if packed_scratch {
                    rows_view(&normalized, count)
                } else {
                    scratch.normalized.clone()
                };
                let generic = crate::metal::Mhc4PreGenericScratch {
                    normalized: &normalized_view,
                    ones: &scratch.ones,
                };
                encode_hc_pre_dispatch(
                    &ctx,
                    enc,
                    h,
                    count,
                    hc_pre_eps(&c),
                    &Mhc4PreInputs {
                        residual: &slice(&residual_t),
                        mix: &mix,
                        scale: &scale_t,
                        base: &base_t,
                        norm_weight: &norm_t,
                    },
                    &Mhc4PrePartials {
                        dots: &slice(&dots),
                        sumsq: &slice(&sumsq),
                    },
                    (dtype != GgmlType::Q8_0).then_some(&generic),
                    &Mhc4PreOutputs {
                        mixes: &slice(&out[0]),
                        pre: &slice(&out[1]),
                        post: &slice(&out[2]),
                        comb: &slice(&out[3]),
                        collapsed: &slice(&out[4]),
                        normed: &slice(&out[5]),
                    },
                )
                .unwrap();
            };
            let mut kernels = Vec::new();
            run(&ctx, |enc| {
                crate::metal::dispatch_census_begin();
                encode(enc, &packed, 0, rows, true);
                kernels = crate::metal::dispatch_census_take();
                for row in 0..rows {
                    encode(enc, &serial, row, 1, false);
                }
            });
            if dtype != GgmlType::Q8_0 && rows > 1 {
                if crate::metal::grouped_projection_supported(dtype) {
                    assert_eq!(
                        kernels
                            .iter()
                            .filter(|r| r.kernel.starts_with("kernel_latent_grouped_"))
                            .count(),
                        1,
                        "packed HC must encode one grouped projection: {dtype:?}/{rows}: {kernels:?}"
                    );
                    assert!(
                        !kernels.iter().any(|r| r.kernel.contains("mat_vec")),
                        "{kernels:?}"
                    );
                } else {
                    assert_eq!(
                        kernels
                            .iter()
                            .filter(|r| r.kernel.contains("mat_vec"))
                            .count(),
                        rows,
                        "GEMV-only projection fallback: {kernels:?}"
                    );
                }
            }
            let untouched_start = if dtype != GgmlType::Q8_0 {
                rows * 4 * h
            } else {
                0
            };
            assert!(
                read_f32(&normalized).unwrap()[untouched_start..]
                    .iter()
                    .all(|v| v.is_nan()),
                "generic HC wrote beyond its active normalized prefix"
            );
            for (p, s) in packed.iter().zip(&serial) {
                let p = read_f32(p).unwrap();
                let s = read_f32(s).unwrap();
                if dtype == GgmlType::Q8_0 {
                    assert_eq!(
                        p.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        s.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        "{dtype:?}/{rows}"
                    );
                } else {
                    // Exact refers to F32 operands; grouped reduction may
                    // differ from single-row GEMV without narrowing to half.
                    close(
                        &format!("packed vs decode HC {dtype:?}/{rows}"),
                        &p,
                        &s,
                        7e-5,
                    );
                }
            }
            for row in 0..rows {
                let cpu = hyper_connection_pre(
                    &residual[row * 4 * h..(row + 1) * 4 * h],
                    h,
                    4,
                    &decoded,
                    &scale,
                    &base,
                    c.rms_epsilon,
                    20,
                    c.hc_epsilon,
                )
                .unwrap();
                let normed = rms_norm(&cpu.input, Some(&norm), c.rms_epsilon).unwrap();
                for (t, expected) in packed.iter().zip([
                    &cpu.mixes,
                    &cpu.controls.pre,
                    &cpu.controls.post,
                    &cpu.controls.combination,
                    &cpu.input,
                    &normed,
                ]) {
                    let actual = read_f32(t).unwrap();
                    let width = expected.len();
                    close(
                        &format!("HC {dtype:?}/{rows}/{row}"),
                        &actual[row * width..(row + 1) * width],
                        expected,
                        7e-5,
                    );
                }
            }
        }
    }
}

#[test]
fn native_latent_runtime_decode_and_packed_keep_f32_operands() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, m, groups) = (256usize, 32usize, 2usize);
    for &dtype in crate::metal::GROUPED_PROJECTION_DTYPES {
        let (weight, decoded) = bank(&ctx, dtype, vec![k as u64, m as u64, groups as u64]);
        for rows in [1usize, 5, 129] {
            let values: Vec<f32> = (0..k * groups * rows)
                .map(|i| ((i * 13 + i / k) % 89) as f32 * 0.01719 - 0.7)
                .collect();
            let input = tensor(&ctx, &values, vec![k as u64, groups as u64, rows as u64]);
            let serial = empty(&ctx, vec![m as u64, groups as u64, rows as u64]);
            run(&ctx, |enc| {
                for row in 0..rows {
                    grouped_matvec(
                        &ctx,
                        enc,
                        &weight,
                        &input.view_subrange(
                            (row * k * groups) as u64,
                            vec![k as u64, groups as u64, 1],
                        ),
                        &serial.view_subrange(
                            (row * m * groups) as u64,
                            vec![m as u64, groups as u64, 1],
                        ),
                        k,
                        m,
                        groups,
                    )
                    .unwrap();
                }
            });
            let mut expected = Vec::new();
            for row in 0..rows {
                for g in 0..groups {
                    for out in 0..m {
                        expected.push(
                            (0..k)
                                .map(|i| {
                                    decoded[(g * m + out) * k + i] as f64
                                        * values[(row * groups + g) * k + i] as f64
                                })
                                .sum::<f64>() as f32,
                        );
                    }
                }
            }
            close(
                &format!("latent decode {dtype:?}"),
                &read_f32(&serial).unwrap(),
                &expected,
                4e-5,
            );
            for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
                let output = empty(&ctx, vec![m as u64, groups as u64, rows as u64]);
                run(&ctx, |enc| {
                    absorb_rows(
                        &ctx,
                        enc,
                        Arith {
                            lineage,
                            precision: FastPrecision::Half,
                        },
                        &weight,
                        &input,
                        &output,
                        k,
                        m,
                        groups,
                        rows,
                    )
                    .unwrap()
                });
                close(
                    &format!("latent {dtype:?}/{lineage:?}/{rows}"),
                    &read_f32(&output).unwrap(),
                    &expected,
                    4e-5,
                );
            }
        }
    }
}

#[test]
fn router_native_dtypes_use_indexer_projection_policy() {
    // F32 alone is eligible for the measured E8P32 geometry.
    assert!(router_e8p32_selected(
        PackedLineage::Fast,
        4096,
        288,
        GgmlType::F32,
        128,
        || true
    ));
    for dtype in [
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q6_K,
    ] {
        assert!(!router_e8p32_selected(
            PackedLineage::Fast,
            4096,
            288,
            dtype,
            128,
            || panic!("non-F32 router must not probe E8P32")
        ));
    }
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, m, rows) = (256usize, 32usize, 9usize);
    let x: Vec<f32> = (0..k * rows)
        .map(|i| (i % 71) as f32 * 0.01317 - 0.4)
        .collect();
    // Q8 F32 tiles require backing padded to 32 tokens.
    let input = MetalTensor::zeros_f32(&ctx, vec![k as u64, 32]).unwrap();
    write_f32(&input.view_subrange(0, vec![k as u64, rows as u64]), &x).unwrap();
    for dtype in [
        GgmlType::F32,
        GgmlType::F16,
        GgmlType::BF16,
        GgmlType::Q8_0,
        GgmlType::Q6_K,
    ] {
        let (weight, decoded) = bank(&ctx, dtype, vec![k as u64, m as u64]);
        for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
            for precision in FastPrecision::ALL {
                let router = empty(&ctx, vec![m as u64, rows as u64]);
                let indexer = empty(&ctx, vec![m as u64, rows as u64]);
                let arith = Arith { lineage, precision };
                run(&ctx, |enc| {
                    for (stage, out) in [
                        (Stage::Router, &router),
                        (Stage::IndexerProjection, &indexer),
                    ] {
                        matmat(
                            &ctx,
                            enc,
                            arith,
                            stage,
                            &weight,
                            &rows_view(&input, rows),
                            out,
                            k,
                            m,
                            rows,
                        )
                        .unwrap();
                    }
                });
                let actual = read_f32(&router).unwrap();
                assert_eq!(
                    actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    read_f32(&indexer)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    "{dtype:?}/{lineage:?}/{precision:?}"
                );
                if lineage == PackedLineage::Exact {
                    let mut expected = Vec::new();
                    for row in 0..rows {
                        for out in 0..m {
                            expected.push(
                                (0..k)
                                    .map(|i| decoded[out * k + i] as f64 * x[row * k + i] as f64)
                                    .sum::<f64>() as f32,
                            );
                        }
                    }
                    close("exact router", &actual, &expected, 4e-5);
                }
            }
        }
    }
}

#[test]
fn native_latent_exact_does_not_narrow_large_f32_operands() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, m, groups, rows) = (32usize, 3usize, 2usize, 5usize);
    for dtype in [GgmlType::F32, GgmlType::BF16, GgmlType::F16] {
        let source: Vec<f32> = (0..k * m * groups)
            .map(|i| {
                if i % k == 0 && dtype != GgmlType::F16 {
                    70001.75
                } else {
                    0.001731 * (i % 11) as f32
                }
            })
            .collect();
        let mut decoded = Vec::new();
        let mut bytes = Vec::new();
        for v in source {
            match dtype {
                GgmlType::F32 => {
                    bytes.extend(v.to_le_bytes());
                    decoded.push(v);
                }
                GgmlType::BF16 => {
                    let v = half::bf16::from_f32(v);
                    bytes.extend(v.to_le_bytes());
                    decoded.push(v.to_f32());
                }
                _ => {
                    let v = half::f16::from_f32(v);
                    bytes.extend(v.to_le_bytes());
                    decoded.push(v.to_f32());
                }
            }
        }
        let weight =
            MetalTensor::from_bytes(&ctx, &bytes, vec![k as u64, m as u64, groups as u64], dtype)
                .unwrap();
        let x: Vec<f32> = (0..k * groups * rows)
            .map(|i| 65535.875 + (i % 3) as f32 * 0.125)
            .collect();
        let input = tensor(&ctx, &x, vec![(k * groups) as u64, rows as u64]);
        let output = empty(&ctx, vec![(m * groups) as u64, rows as u64]);
        run(&ctx, |enc| {
            absorb_rows(
                &ctx,
                enc,
                Arith {
                    lineage: PackedLineage::Exact,
                    precision: FastPrecision::Half,
                },
                &weight,
                &input,
                &output,
                k,
                m,
                groups,
                rows,
            )
            .unwrap()
        });
        let mut expected = Vec::new();
        for row in 0..rows {
            for g in 0..groups {
                for out in 0..m {
                    expected.push(
                        (0..k)
                            .map(|i| {
                                decoded[(g * m + out) * k + i] as f64
                                    * x[(row * groups + g) * k + i] as f64
                            })
                            .sum::<f64>() as f32,
                    );
                }
            }
        }
        close(
            "F32 operand range",
            &read_f32(&output).unwrap(),
            &expected,
            3e-6,
        );
    }
}

#[test]
fn hc_scratch_selection_checks_both_sites_and_only_mix_weights() {
    use crate::glm5_next::{DenseFfnTensors, HyperConnectionTensors, KdaTensors};
    let hc = || HyperConnectionTensors {
        mix: GgmlType::Q8_0,
        base: GgmlType::F32,
        scale: GgmlType::F32,
    };
    let mut blocks = vec![Glm5NextBlock {
        attention_hc: hc(),
        ffn_hc: hc(),
        attention_norm: GgmlType::F32,
        ffn_norm: GgmlType::F32,
        mixer: MixerTensors::Kda(KdaTensors {
            query: GgmlType::F16,
            key: GgmlType::BF16,
            value: GgmlType::Q6_K,
            query_conv: GgmlType::F32,
            key_conv: GgmlType::F32,
            value_conv: GgmlType::F32,
            decay_a: GgmlType::F16,
            decay_b: GgmlType::F16,
            decay_bias: GgmlType::F32,
            neg_exp_a_log: GgmlType::F32,
            beta: GgmlType::F16,
            gate_a: GgmlType::F16,
            gate_b: GgmlType::F16,
            output_norm: GgmlType::F32,
            output: GgmlType::F16,
        }),
        ffn: FfnTensors::Dense(DenseFfnTensors {
            gate: GgmlType::F16,
            up: GgmlType::F16,
            down: GgmlType::F16,
        }),
    }];
    assert!(!needs_generic_hc(&blocks, |t| *t));
    blocks[0].ffn_hc.mix = GgmlType::BF16;
    assert!(needs_generic_hc(&blocks, |t| *t));
    blocks[0].ffn_hc.mix = GgmlType::Q8_0;
    blocks[0].attention_hc.mix = GgmlType::MXFP4;
    assert!(needs_generic_hc(&blocks, |t| *t));
    assert!(!needs_generic_hc::<GgmlType>(&[], |t| *t));
}

#[test]
fn packed_fallback_policy_preserves_available_fast_paths() {
    use crate::metal::{grouped_clamped_gate_up_supported, grouped_expert_down_supported};
    for precision in FastPrecision::ALL {
        let arith = Arith {
            lineage: PackedLineage::Fast,
            precision,
        };
        for stage in Stage::ALL {
            assert_eq!(
                projection_mode(arith, stage, GgmlType::MXFP4, 256, 32),
                StageMode::Exact
            );
            for dtype in [
                GgmlType::F32,
                GgmlType::BF16,
                GgmlType::Q8_0,
                GgmlType::Q6_K,
            ] {
                assert_eq!(
                    projection_mode(arith, stage, dtype, 256, 32),
                    if dtype == GgmlType::BF16 && stage_mode(arith, stage) == StageMode::FastF32 {
                        StageMode::Exact
                    } else {
                        stage_mode(arith, stage)
                    }
                );
            }
        }
    }
    for gate in [
        GgmlType::IQ2_S,
        GgmlType::IQ3_S,
        GgmlType::IQ2_XS,
        GgmlType::IQ3_XXS,
        GgmlType::Q3_K,
    ] {
        for down in [
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
            GgmlType::Q4_K,
            GgmlType::MXFP4,
        ] {
            assert_eq!(
                expert_row_fallback(PackedLineage::Fast, gate, gate, down),
                !grouped_clamped_gate_up_supported(gate, gate)
                    || !grouped_expert_down_supported(down)
            );
            assert!(expert_row_fallback(PackedLineage::Exact, gate, gate, down));
        }
    }
    // MXFP4 down has a generic grouped layout now, so this pair is grouped.
    assert!(!expert_row_fallback(
        PackedLineage::Fast,
        GgmlType::IQ2_S,
        GgmlType::IQ2_S,
        GgmlType::MXFP4
    ));
    assert!(!expert_row_fallback(
        PackedLineage::Fast,
        GgmlType::IQ2_S,
        GgmlType::IQ2_S,
        GgmlType::IQ4_XS
    ));
    assert!(!expert_row_fallback(
        PackedLineage::Fast,
        GgmlType::IQ3_S,
        GgmlType::IQ3_S,
        GgmlType::IQ3_S
    ));
    // Mixed pairs follow the actual grouped capability, not dtype equality.
    assert_eq!(
        expert_row_fallback(
            PackedLineage::Fast,
            GgmlType::IQ2_S,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
        ),
        !grouped_clamped_gate_up_supported(GgmlType::IQ2_S, GgmlType::IQ3_S)
    );
}

#[test]
fn packed_mxfp4_dense_baseline_matches_exact_rows() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, m) = (256usize, 32usize);
    let (weight, decoded) = bank(&ctx, GgmlType::MXFP4, vec![k as u64, m as u64]);
    for rows in [1usize, 3, 9] {
        let values: Vec<f32> = (0..k * rows)
            .map(|i| (i % 79) as f32 * 0.01713 - 0.6)
            .collect();
        let input = tensor(&ctx, &values, vec![k as u64, rows as u64]);
        let serial = empty(&ctx, vec![m as u64, rows as u64]);
        run(&ctx, |enc| {
            for row in 0..rows {
                matvec(
                    &ctx,
                    enc,
                    &weight,
                    &input.view_subrange((row * k) as u64, vec![k as u64]),
                    &serial.view_subrange((row * m) as u64, vec![m as u64]),
                    k,
                    m,
                )
                .unwrap();
            }
        });
        let serial = read_f32(&serial).unwrap();
        let mut expected = Vec::new();
        for row in 0..rows {
            for out in 0..m {
                expected.push(
                    (0..k)
                        .map(|i| decoded[out * k + i] as f64 * values[row * k + i] as f64)
                        .sum::<f64>() as f32,
                );
            }
        }
        close("MXFP4 row oracle", &serial, &expected, 4e-5);
        for precision in FastPrecision::ALL {
            for stage in [
                Stage::DenseFfn,
                Stage::Router,
                Stage::SharedExpert,
                Stage::KdaProjection,
                Stage::KdaExpand,
                Stage::MlaProjection,
                Stage::IndexerProjection,
            ] {
                let output = empty(&ctx, vec![m as u64, rows as u64]);
                let arith = Arith {
                    lineage: PackedLineage::Fast,
                    precision,
                };
                run(&ctx, |enc| {
                    if stage == Stage::KdaExpand {
                        expand_rows(
                            &ctx, enc, arith, stage, &weight, &input, &output, k, m, rows,
                        )
                        .unwrap();
                    } else {
                        matmat(
                            &ctx, enc, arith, stage, &weight, &input, &output, k, m, rows,
                        )
                        .unwrap();
                    }
                });
                assert_eq!(
                    read_f32(&output)
                        .unwrap()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>(),
                    serial.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{stage:?}/{precision:?}/{rows}"
                );
            }
        }
    }
}

#[test]
fn packed_expert_baseline_reuses_decode_scratch_and_matches_oracle() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (h, f, e, k, rows) = (256usize, 256usize, 3usize, 2usize, 3usize);
    let clamp = 0.02;
    let values: Vec<f32> = (0..h * rows)
        .map(|i| (i % 71) as f32 * 0.01713 - 0.5)
        .collect();
    let input = tensor(&ctx, &values, vec![h as u64, rows as u64]);
    let ids = [2i32, 0, 1, 2, 0, 1];
    let route_weights = [0.7f32, 0.3, 0.2, 0.8, 0.6, 0.4];
    let ids_t = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&ids),
        vec![k as u64, rows as u64],
        GgmlType::I32,
    )
    .unwrap();
    let weights_t = tensor(&ctx, &route_weights, vec![k as u64, rows as u64]);
    let status = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[ROUTE_STATUS_READY; 3]),
        vec![rows as u64],
        GgmlType::I32,
    )
    .unwrap();
    // Exactly one row's scratch, reused across rows and combinations.
    let inner = empty(&ctx, vec![f as u64, k as u64]);
    let slot_out = empty(&ctx, vec![h as u64, k as u64]);
    for (gate_dtype, down_dtype) in [
        (GgmlType::IQ2_XS, GgmlType::IQ4_XS),
        (GgmlType::IQ3_XXS, GgmlType::IQ4_XS),
        (GgmlType::Q3_K, GgmlType::IQ4_XS),
        (GgmlType::IQ2_S, GgmlType::MXFP4),
        (GgmlType::Q3_K, GgmlType::MXFP4),
    ] {
        let (gate, gate_cpu) = bank(&ctx, gate_dtype, vec![h as u64, f as u64, e as u64]);
        let (up, up_cpu) = bank(&ctx, gate_dtype, vec![h as u64, f as u64, e as u64]);
        let (down, down_cpu) = bank(&ctx, down_dtype, vec![f as u64, h as u64, e as u64]);
        let output = empty(&ctx, vec![h as u64, rows as u64]);
        let serial = empty(&ctx, vec![h as u64, rows as u64]);
        run(&ctx, |enc| {
            encode_expert_rows(
                &ctx,
                enc,
                &ExpertRowBuffers {
                    gate: &gate,
                    up: &up,
                    down: &down,
                    input: &input,
                    ids: &ids_t,
                    weights: &weights_t,
                    status: &status,
                    inner: &inner,
                    slot_out: &slot_out,
                    output: &output,
                },
                h,
                f,
                e,
                k,
                rows,
                clamp,
            )
            .unwrap();
            for row in 0..rows {
                let ids = ids_t.view_subrange((row * k) as u64, vec![k as u64]);
                let st = status.view_subrange(row as u64, vec![1]);
                encode_all_slots_gate_up_swiglu(
                    &ctx,
                    enc,
                    &gate,
                    &up,
                    &input.view_subrange((row * h) as u64, vec![h as u64]),
                    &ids,
                    &st,
                    &inner,
                    h,
                    f,
                    e,
                    k,
                    clamp,
                )
                .unwrap();
                encode_all_slots_down(&ctx, enc, &down, &inner, &ids, &st, &slot_out, f, h, e, k)
                    .unwrap();
                encode_moe_weighted_sum_f32(
                    &ctx,
                    enc,
                    &slot_out,
                    &weights_t.view_subrange((row * k) as u64, vec![k as u64]),
                    &serial.view_subrange((row * h) as u64, vec![h as u64]),
                    h,
                    k,
                )
                .unwrap();
            }
        });
        let got = read_f32(&output).unwrap();
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            read_f32(&serial)
                .unwrap()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
        let mut expected = vec![0.0f32; h * rows];
        for row in 0..rows {
            for slot in 0..k {
                let expert = ids[row * k + slot] as usize;
                let dot = |bank: &[f32], out: usize| {
                    (0..h)
                        .map(|i| {
                            bank[(expert * f + out) * h + i] as f64 * values[row * h + i] as f64
                        })
                        .sum::<f64>() as f32
                };
                let g: Vec<f32> = (0..f).map(|out| dot(&gate_cpu, out)).collect();
                let u: Vec<f32> = (0..f).map(|out| dot(&up_cpu, out)).collect();
                let act = crate::deepseek_v4_oracle::clamped_swiglu(&g, &u, clamp).unwrap();
                for out in 0..h {
                    let value = (0..f)
                        .map(|i| down_cpu[(expert * h + out) * f + i] as f64 * act[i] as f64)
                        .sum::<f64>() as f32;
                    expected[row * h + out] += route_weights[row * k + slot] * value;
                }
            }
        }
        let scale = expected
            .iter()
            .copied()
            .map(f32::abs)
            .fold(0.0f32, f32::max);
        assert!(scale > 1e-7, "synthetic experts must have a nonzero result");
        assert!(
            got.iter().all(|value| value.is_finite()),
            "expert baseline {gate_dtype:?}/{down_dtype:?}: GPU output contains non-finite values"
        );
        let worst = crate::compare::assert_max_abs_diff_f32(
            &format!("expert baseline {gate_dtype:?}/{down_dtype:?}"),
            got.iter(),
            &expected,
        );
        assert!(
            got.iter().all(|v| v.is_finite()) && worst <= 1e-3 * scale,
            "expert baseline {gate_dtype:?}/{down_dtype:?}: max error {worst}, scale {scale}"
        );
    }
}

#[test]
fn f32_precision_fallback_policy_checks_dtype_geometry_and_expert_combination() {
    let dense = Stage::DenseFfn;
    for precision in [FastPrecision::DenseF32, FastPrecision::F32] {
        let arith = Arith {
            lineage: PackedLineage::Fast,
            precision,
        };
        for dtype in [GgmlType::Q2_K, GgmlType::BF16, GgmlType::F16] {
            assert_eq!(
                projection_mode(arith, dense, dtype, 256, 32),
                StageMode::Exact
            );
        }
        assert_eq!(
            projection_mode(arith, dense, GgmlType::Q8_0, 256, 32),
            StageMode::FastF32
        );
        assert_eq!(
            projection_mode(arith, dense, GgmlType::Q6_K, 256, 32),
            StageMode::FastF32
        );
        assert_eq!(
            projection_mode(arith, dense, GgmlType::F32, 256, 32),
            StageMode::FastF32
        );
        assert_eq!(
            projection_mode(arith, dense, GgmlType::Q8_0, 32, 32),
            StageMode::Exact
        );
        assert_eq!(
            projection_mode(arith, dense, GgmlType::Q8_0, 256, 17),
            StageMode::Exact
        );
        for gate in [GgmlType::IQ2_S, GgmlType::IQ3_S] {
            for down in [GgmlType::Q4_K, GgmlType::IQ3_XXS, GgmlType::IQ4_XS] {
                assert_eq!(
                    expert_mode(arith, gate, gate, down),
                    if precision == FastPrecision::F32 {
                        StageMode::FastF32
                    } else {
                        StageMode::Fast
                    }
                );
            }
        }
    }
}

#[test]
fn disabling_f32_tiles_preserves_requested_operand_precision() {
    let _scope = F32Stages::with_tiles(&[Stage::DenseFfn, Stage::RoutedExperts], &[]);
    let arith = Arith {
        lineage: PackedLineage::Fast,
        precision: FastPrecision::F32,
    };
    for dtype in [GgmlType::Q8_0, GgmlType::Q6_K] {
        assert_eq!(
            projection_mode(arith, Stage::DenseFfn, dtype, 256, 32),
            StageMode::Exact
        );
    }
    for down in [GgmlType::IQ4_XS, GgmlType::IQ3_XXS, GgmlType::Q4_K] {
        assert_eq!(
            expert_mode(arith, GgmlType::IQ2_S, GgmlType::IQ2_S, down),
            StageMode::Exact
        );
    }
}

#[test]
fn q2_k_dense_half_and_f32_requests_dispatch_different_operand_paths() {
    use crate::metal::{dispatch_census_begin, dispatch_census_take};
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, m, rows) = (256usize, 32usize, 9usize);
    let (weight, _) = bank(&ctx, GgmlType::Q2_K, vec![k as u64, m as u64]);
    let values: Vec<f32> = (0..k * rows)
        .map(|i| (i % 71) as f32 * 0.01713 - 0.5)
        .collect();
    let input = tensor(&ctx, &values, vec![k as u64, rows as u64]);
    let exact = empty(&ctx, vec![m as u64, rows as u64]);
    run(&ctx, |enc| {
        matmat(
            &ctx,
            enc,
            Arith {
                lineage: PackedLineage::Exact,
                precision: FastPrecision::Half,
            },
            Stage::DenseFfn,
            &weight,
            &input,
            &exact,
            k,
            m,
            rows,
        )
        .unwrap()
    });
    let expected = read_f32(&exact).unwrap();
    for precision in FastPrecision::ALL {
        let output = empty(&ctx, vec![m as u64, rows as u64]);
        let census = F32Census::begin();
        let mut kernels = Vec::new();
        run(&ctx, |enc| {
            dispatch_census_begin();
            matmat(
                &ctx,
                enc,
                Arith {
                    lineage: PackedLineage::Fast,
                    precision,
                },
                Stage::DenseFfn,
                &weight,
                &input,
                &output,
                k,
                m,
                rows,
            )
            .unwrap();
            kernels = dispatch_census_take();
        });
        let counts = census.take();
        if precision == FastPrecision::Half {
            assert!(
                kernels.iter().any(|r| r.kernel.contains("mat_mat_q2_K")),
                "{kernels:?}"
            );
            assert!(
                !kernels.iter().any(|r| r.kernel.contains("mat_vec_q2_K")),
                "{kernels:?}"
            );
            assert_eq!(
                counts.get(&(Stage::DenseFfn, "Q2_K".into(), CensusPath::Half)),
                Some(&1)
            );
        } else {
            assert_eq!(kernels.len(), rows, "{kernels:?}");
            assert!(
                kernels.iter().all(|r| r.kernel.contains("mat_vec_q2_K")),
                "{kernels:?}"
            );
            assert_eq!(
                counts.get(&(Stage::DenseFfn, "Q2_K".into(), CensusPath::RowBaseline)),
                Some(&1)
            );
            assert_eq!(
                read_f32(&output)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn expert_f32_capability_tracks_primitive_down_and_retains_fused_pair_contract() {
    use crate::metal::{
        grouped_routed_experts_f32x_supported, moe_down_f32x_grouped_slots_supported,
    };
    use GgmlType::*;
    let down_types: Vec<_> = (0..=255).map(GgmlType::from_raw).collect();
    // The loop checks the relation for every type; only require it is not vacuous.
    assert!(
        down_types
            .iter()
            .any(|&t| moe_down_f32x_grouped_slots_supported(t))
    );
    for &down in &down_types {
        for (gate, up) in [
            (IQ2_S, IQ2_S),
            (IQ3_S, IQ3_S),
            (IQ2_S, IQ3_S),
            (IQ3_S, IQ2_S),
            (Q4_K, Q4_K),
            (F32, F32),
            (IQ2_XS, IQ2_XS),
        ] {
            let supported = gate == up
                && matches!(gate, IQ2_S | IQ3_S)
                && moe_down_f32x_grouped_slots_supported(down);
            assert_eq!(
                grouped_routed_experts_f32x_supported(gate, up, down),
                supported
            );
            assert_eq!(
                expert_mode(
                    Arith {
                        lineage: PackedLineage::Fast,
                        precision: FastPrecision::F32
                    },
                    gate,
                    up,
                    down
                ),
                if supported {
                    StageMode::FastF32
                } else {
                    StageMode::Exact
                },
                "{gate:?}/{up:?}/{down:?}"
            );
            assert_eq!(
                expert_mode(
                    Arith {
                        lineage: PackedLineage::Exact,
                        precision: FastPrecision::F32
                    },
                    gate,
                    up,
                    down
                ),
                StageMode::Exact
            );
        }
    }
}

#[test]
fn expert_half_and_f32_requests_dispatch_grouped_or_all_slot_paths() {
    use crate::metal::{
        dispatch_census_begin, dispatch_census_take, grouped_clamped_gate_up_supported,
        grouped_expert_down_supported, moe_down_f32x_grouped_slots_pipeline_name,
    };
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (h, f, e, k, rows) = (256usize, 256usize, 3usize, 2usize, 3usize);
    let clamp = 0.2;
    let values: Vec<f32> = (0..h * rows)
        .map(|i| (i % 71) as f32 * 0.01713 - 0.5)
        .collect();
    let route_ids = [2i32, 0, 1, 2, 0, 1];
    let route_weights = [0.7f32, 0.3, 0.2, 0.8, 0.6, 0.4];
    let input = tensor(&ctx, &values, vec![h as u64, rows as u64]);
    let ids = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&route_ids),
        vec![k as u64, rows as u64],
        GgmlType::I32,
    )
    .unwrap();
    let weights = tensor(&ctx, &route_weights, vec![k as u64, rows as u64]);
    let status = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[ROUTE_STATUS_READY; 3]),
        vec![rows as u64],
        GgmlType::I32,
    )
    .unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![e as u64]).unwrap();
    let slots = MetalTensor::zeros_i32(&ctx, vec![(e * rows) as u64]).unwrap();
    let inner = empty(&ctx, vec![f as u64, (k * rows) as u64]);
    let slot_out = empty(&ctx, vec![h as u64, (k * rows) as u64]);
    let row_inner = empty(&ctx, vec![f as u64, k as u64]);
    let row_out = empty(&ctx, vec![h as u64, k as u64]);
    let (gate, gate_cpu) = bank(&ctx, GgmlType::IQ2_S, vec![h as u64, f as u64, e as u64]);
    let (up, up_cpu) = bank(&ctx, GgmlType::IQ2_S, vec![h as u64, f as u64, e as u64]);
    for down_dtype in [
        GgmlType::Q4_K,
        GgmlType::IQ3_XXS,
        GgmlType::Q2_0,
        GgmlType::BF16,
        GgmlType::IQ2_XS,
        GgmlType::MXFP4,
    ] {
        let (down, down_cpu) = bank(&ctx, down_dtype, vec![f as u64, h as u64, e as u64]);
        // Independent GGML dequantization plus scalar F64 dot products; the
        // oracle rounds each projection to F32 before clamped SwiGLU/summing.
        let mut expected = vec![0.0f32; h * rows];
        for row in 0..rows {
            for slot in 0..k {
                let expert = route_ids[row * k + slot] as usize;
                let dot = |w: &[f32], out: usize| {
                    (0..h)
                        .map(|i| w[(expert * f + out) * h + i] as f64 * values[row * h + i] as f64)
                        .sum::<f64>() as f32
                };
                let g: Vec<f32> = (0..f).map(|o| dot(&gate_cpu, o)).collect();
                let u: Vec<f32> = (0..f).map(|o| dot(&up_cpu, o)).collect();
                let act = crate::deepseek_v4_oracle::clamped_swiglu(&g, &u, clamp).unwrap();
                for o in 0..h {
                    let v = (0..f)
                        .map(|i| down_cpu[(expert * h + o) * f + i] as f64 * act[i] as f64)
                        .sum::<f64>() as f32;
                    expected[row * h + o] += route_weights[row * k + slot] * v;
                }
            }
        }
        let mut exact_bits = None;
        for (lineage, precision, disable_tile) in [
            (PackedLineage::Exact, FastPrecision::Half, false),
            (PackedLineage::Fast, FastPrecision::Half, false),
            (PackedLineage::Fast, FastPrecision::DenseF32, false),
            (PackedLineage::Fast, FastPrecision::F32, false),
            (PackedLineage::Fast, FastPrecision::F32, true),
        ] {
            let _tiles = disable_tile.then(|| F32Stages::with_tiles(&[Stage::RoutedExperts], &[]));
            let f32_pipeline = moe_down_f32x_grouped_slots_pipeline_name(down_dtype);
            let expected_mode = if lineage == PackedLineage::Exact {
                StageMode::Exact
            } else if precision == FastPrecision::F32 {
                if !disable_tile && f32_pipeline.is_some() {
                    StageMode::FastF32
                } else {
                    StageMode::Exact
                }
            } else if grouped_clamped_gate_up_supported(gate.dtype, up.dtype)
                && grouped_expert_down_supported(down_dtype)
            {
                StageMode::Fast
            } else {
                StageMode::Exact
            };
            let output = empty(&ctx, vec![h as u64, rows as u64]);
            let buffers = GroupedExperts {
                gate_bank: &gate,
                up_bank: &up,
                down_bank: &down,
                input: &input,
                ids: &ids,
                weights: &weights,
                counts: &counts,
                slots: &slots,
                inner: &inner,
                slot_out: &slot_out,
                output: &output,
            };
            let census = F32Census::begin();
            let mut kernels = Vec::new();
            run(&ctx, |enc| {
                dispatch_census_begin();
                let mode = encode_experts(
                    &ctx,
                    enc,
                    Arith { lineage, precision },
                    &buffers,
                    &status,
                    &row_inner,
                    &row_out,
                    h,
                    f,
                    e,
                    k,
                    rows,
                    clamp,
                )
                .unwrap();
                assert_eq!(
                    mode, expected_mode,
                    "{down_dtype:?}/{precision:?} tiles disabled={disable_tile}"
                );
                kernels = dispatch_census_take();
            });
            let census_counts = census.take();
            if lineage == PackedLineage::Fast {
                let path = match expected_mode {
                    StageMode::FastF32 => CensusPath::F32Tile,
                    StageMode::Fast => CensusPath::Half,
                    StageMode::Exact => CensusPath::RowBaseline,
                };
                assert_eq!(
                    census_counts.get(&(Stage::RoutedExperts, "experts".into(), path)),
                    Some(&1)
                );
            }
            match expected_mode {
                StageMode::FastF32 => {
                    assert_eq!(
                        kernels
                            .iter()
                            .filter(|r| r.kernel == f32_pipeline.unwrap())
                            .count(),
                        1,
                        "{kernels:?}"
                    );
                    assert_eq!(
                        kernels
                            .iter()
                            .filter(|r| r.kernel
                                == "kernel_moe_swiglu_clamped_iq2_s_f32x_grouped_slots")
                            .count(),
                        1,
                        "{kernels:?}"
                    );
                    assert!(
                        !kernels.iter().any(|r| r.kernel.contains("all_slots_")
                            || (r.kernel.contains("grouped_slots") && !r.kernel.contains("f32x"))),
                        "{kernels:?}"
                    );
                }
                StageMode::Fast => {
                    assert!(
                        kernels.iter().any(|r| r.kernel.contains("grouped_slots")),
                        "{kernels:?}"
                    );
                    assert!(
                        !kernels
                            .iter()
                            .any(|r| r.kernel.contains("all_slots_") || r.kernel.contains("f32x")),
                        "{kernels:?}"
                    );
                }
                StageMode::Exact => {
                    assert_eq!(
                        kernels
                            .iter()
                            .filter(|r| r.kernel.contains("all_slots_swiglu_iq2_s"))
                            .count(),
                        rows,
                        "{kernels:?}"
                    );
                    assert!(
                        !kernels.iter().any(|r| r.kernel.contains("grouped_slots")),
                        "{kernels:?}"
                    );
                }
            }
            let got = read_f32(&output).unwrap();
            let scale = expected
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max);
            assert!(scale > 1e-7 && expected.iter().chain(&got).all(|v| v.is_finite()));
            let worst = crate::compare::assert_max_abs_diff_f32(
                &format!("expert dispatch {down_dtype:?}/{precision:?}/{expected_mode:?}"),
                got.iter(),
                &expected,
            );
            assert!(
                worst
                    <= (if expected_mode == StageMode::Fast {
                        5e-3
                    } else {
                        1e-3
                    }) * scale,
                "{down_dtype:?}/{precision:?} {expected_mode:?}: error {worst}, scale {scale}"
            );
            if expected_mode == StageMode::Exact {
                let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                if let Some(reference) = &exact_bits {
                    assert_eq!(&bits, reference);
                } else {
                    exact_bits = Some(bits);
                }
            }
        }
    }
}

#[test]
fn mixed_expert_precision_policy_requires_an_actual_pair_tile() {
    use crate::metal::{grouped_clamped_gate_up_supported, grouped_expert_down_supported};
    for (gate, up, down) in [
        (GgmlType::IQ2_S, GgmlType::IQ3_S, GgmlType::IQ4_XS),
        (GgmlType::IQ3_S, GgmlType::IQ2_S, GgmlType::IQ3_S),
        (GgmlType::Q4_K, GgmlType::F16, GgmlType::Q6_K),
        (GgmlType::IQ2_XS, GgmlType::Q8_0, GgmlType::F32),
    ] {
        for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
            for precision in [
                FastPrecision::Half,
                FastPrecision::F32,
                FastPrecision::DenseF32,
            ] {
                let grouped = lineage == PackedLineage::Fast
                    && precision != FastPrecision::F32
                    && grouped_clamped_gate_up_supported(gate, up)
                    && grouped_expert_down_supported(down);
                assert_eq!(
                    expert_mode(Arith { lineage, precision }, gate, up, down),
                    if grouped {
                        StageMode::Fast
                    } else {
                        StageMode::Exact
                    },
                    "{gate:?}/{up:?}/{down:?} {lineage:?}/{precision:?}"
                );
            }
        }
    }
}

#[test]
fn mixed_native_experts_dispatch_by_capability_and_match_oracle() {
    use crate::metal::{
        dispatch_census_begin, dispatch_census_take, grouped_clamped_gate_up_supported,
        grouped_expert_down_supported,
    };
    use GgmlType::*;
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (h, f, e, k, rows) = (256usize, 256usize, 3usize, 2usize, 3usize);
    let clamp = 0.2;
    let x: Vec<f32> = (0..h * rows)
        .map(|i| (i % 71) as f32 * 0.01713 - 0.5)
        .collect();
    let input = tensor(&ctx, &x, vec![h as u64, rows as u64]);
    let route_ids = [2i32, 0, 1, 2, 0, 1];
    let route_weights = [0.7f32, 0.3, 0.2, 0.8, 0.6, 0.4];
    let ids = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&route_ids),
        vec![k as u64, rows as u64],
        I32,
    )
    .unwrap();
    let weights = tensor(&ctx, &route_weights, vec![k as u64, rows as u64]);
    let status = MetalTensor::from_bytes(
        &ctx,
        bytemuck::cast_slice(&[ROUTE_STATUS_READY; 3]),
        vec![rows as u64],
        I32,
    )
    .unwrap();
    let counts = MetalTensor::zeros_i32(&ctx, vec![e as u64]).unwrap();
    let slots = MetalTensor::zeros_i32(&ctx, vec![(e * rows) as u64]).unwrap();
    let inner = empty(&ctx, vec![f as u64, (k * rows) as u64]);
    let slot_out = empty(&ctx, vec![h as u64, (k * rows) as u64]);
    let row_inner = empty(&ctx, vec![f as u64, k as u64]);
    let row_out = empty(&ctx, vec![h as u64, k as u64]);
    for (gate_type, up_type, down_type, native_passes_per_row) in [
        (Q4_K, F16, Q6_K, 3),
        (IQ2_S, IQ3_S, Q8_0, 3),
        (BF16, Q2_K, IQ2_S, 3),
        (IQ2_XS, Q8_0, F32, 3),
        (Q8_0, Q8_0, BF16, 3),
        (Q5_K, Q4_0, Q4_1, 3),
        (F32, BF16, IQ4_NL, 3),
        // Down has an F32 tile, but mixed gate/up must still use row decode
        // for F32 requests; Fast/Half packets have a native half path.
        (IQ2_S, IQ3_S, IQ4_XS, 2),
    ] {
        let (gate, gcpu) = bank(&ctx, gate_type, vec![h as u64, f as u64, e as u64]);
        let (up, ucpu) = bank(&ctx, up_type, vec![h as u64, f as u64, e as u64]);
        let (down, dcpu) = bank(&ctx, down_type, vec![f as u64, h as u64, e as u64]);
        let mut expected = vec![0.0f32; h * rows];
        for row in 0..rows {
            for slot in 0..k {
                let expert = route_ids[row * k + slot] as usize;
                let dot = |w: &[f32], out: usize| {
                    (0..h)
                        .map(|i| w[(expert * f + out) * h + i] as f64 * x[row * h + i] as f64)
                        .sum::<f64>() as f32
                };
                let g: Vec<f32> = (0..f).map(|o| dot(&gcpu, o)).collect();
                let u: Vec<f32> = (0..f).map(|o| dot(&ucpu, o)).collect();
                let act = crate::deepseek_v4_oracle::clamped_swiglu(&g, &u, clamp).unwrap();
                for o in 0..h {
                    let v = (0..f)
                        .map(|i| dcpu[(expert * h + o) * f + i] as f64 * act[i] as f64)
                        .sum::<f64>() as f32;
                    expected[row * h + o] += route_weights[row * k + slot] * v;
                }
            }
        }
        let mut decode_bits = None;
        for (lineage, precision) in [
            (PackedLineage::Exact, FastPrecision::Half),
            (PackedLineage::Fast, FastPrecision::F32),
            (PackedLineage::Fast, FastPrecision::Half),
            (PackedLineage::Fast, FastPrecision::DenseF32),
        ] {
            let output = empty(&ctx, vec![h as u64, rows as u64]);
            let b = GroupedExperts {
                gate_bank: &gate,
                up_bank: &up,
                down_bank: &down,
                input: &input,
                ids: &ids,
                weights: &weights,
                counts: &counts,
                slots: &slots,
                inner: &inner,
                slot_out: &slot_out,
                output: &output,
            };
            let half_mode = lineage == PackedLineage::Fast
                && precision != FastPrecision::F32
                && grouped_clamped_gate_up_supported(gate_type, up_type)
                && grouped_expert_down_supported(down_type);
            let packet_half = half_mode
                && precision == FastPrecision::Half
                && (1..=8).contains(&rows)
                && matches!(gate_type, IQ2_S | IQ3_S)
                && matches!(up_type, IQ2_S | IQ3_S)
                && matches!(down_type, IQ3_S | IQ4_XS);
            let grouped = half_mode && !packet_half;
            let mut kernels = Vec::new();
            run(&ctx, |enc| {
                dispatch_census_begin();
                let mode = encode_experts(
                    &ctx,
                    enc,
                    Arith { lineage, precision },
                    &b,
                    &status,
                    &row_inner,
                    &row_out,
                    h,
                    f,
                    e,
                    k,
                    rows,
                    clamp,
                )
                .unwrap();
                assert_eq!(
                    mode,
                    if half_mode {
                        StageMode::Fast
                    } else {
                        StageMode::Exact
                    }
                );
                kernels = dispatch_census_take();
            });
            if packet_half {
                let native = |dtype: GgmlType| {
                    format!(
                        "kernel_expert_native_{}_f32",
                        format!("{dtype:?}").to_lowercase()
                    )
                };
                assert_eq!(
                    kernels.iter().map(|r| r.kernel.clone()).collect::<Vec<_>>(),
                    vec![
                        native(gate_type),
                        native(up_type),
                        native(down_type),
                        "kernel_moe_weighted_sum_packed_f32".into(),
                    ],
                    "{kernels:?}"
                );
            } else if grouped {
                assert!(
                    kernels.iter().any(|r| r.kernel.contains("grouped_slots")),
                    "{kernels:?}"
                );
                assert!(
                    !kernels
                        .iter()
                        .any(|r| r.kernel.starts_with("kernel_expert_native_")),
                    "{kernels:?}"
                );
            } else {
                // Native gate/up takes two passes. Down remains optimized when
                // available and otherwise takes one native pass; scratch is reused.
                assert_eq!(
                    kernels
                        .iter()
                        .filter(|r| r.kernel.starts_with("kernel_expert_native_"))
                        .count(),
                    rows * native_passes_per_row,
                    "{kernels:?}"
                );
                assert!(
                    !kernels.iter().any(|r| r.kernel.contains("grouped_slots")),
                    "{kernels:?}"
                );
            }
            let got = read_f32(&output).unwrap();
            let scale = expected
                .iter()
                .copied()
                .map(f32::abs)
                .fold(0.0f32, f32::max);
            assert!(scale > 1e-7 && expected.iter().chain(&got).all(|v| v.is_finite()));
            let worst = crate::compare::assert_max_abs_diff_f32(
                &format!(
                    "expert dispatch {gate_type:?}/{up_type:?}/{down_type:?}/{lineage:?}/{precision:?}/grouped={grouped}/packet_half={packet_half}"
                ),
                got.iter(),
                &expected,
            );
            assert!(
                worst <= (if half_mode { 5e-3 } else { 2e-4 }) * scale,
                "{gate_type:?}/{up_type:?}/{down_type:?} {lineage:?}/{precision:?}: error {worst}, scale {scale}"
            );
            if !half_mode {
                let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                if let Some(reference) = &decode_bits {
                    assert_eq!(&bits, reference);
                } else {
                    decode_bits = Some(bits);
                }
            }
        }
    }
}

#[test]
fn latent_mma_policy_is_fast_half_or_f32() {
    for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
        for precision in FastPrecision::ALL {
            assert_eq!(
                latent_mma_allowed(Arith { lineage, precision }),
                lineage == PackedLineage::Fast
                    && matches!(precision, FastPrecision::Half | FastPrecision::F32)
            );
        }
    }
    for precision in [FastPrecision::Half, FastPrecision::F32] {
        let arith = Arith {
            lineage: PackedLineage::Fast,
            precision,
        };
        {
            let _exact = ExactStages::set(&[Stage::MlaAbsorb]);
            assert!(!latent_mma_allowed(arith));
        }
        assert!(latent_mma_allowed(arith));
    }
}

#[test]
fn q6_latent_production_dispatch_modes_ragged_and_geometry_fallback() {
    use crate::metal::{
        dispatch_census_begin, dispatch_census_take,
        with_grouped_projection_scalar_baseline_override,
    };
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, groups) = (256usize, 2usize);
    for m in [32usize, 17] {
        // m=17 cannot use the MMA output tile.
        let scalar4 = "kernel_latent_grouped_q6_K_f32";
        let mma = "kernel_latent_grouped_q6_K_f32_r2c16k64";
        let (weight, decoded) = bank(
            &ctx,
            GgmlType::Q6_K,
            vec![k as u64, m as u64, groups as u64],
        );
        for rows in [1usize, 7, 8, 127, 128, 129, 136] {
            let values: Vec<f32> = (0..k * groups * rows)
                .map(|i| ((i * 13 + i / k) % 89) as f32 * 0.01719 - 0.7)
                .collect();
            let input = tensor(&ctx, &values, vec![(k * groups) as u64, rows as u64]);
            let baseline = empty(&ctx, vec![(m * groups) as u64, rows as u64]);
            run(&ctx, |enc| {
                dispatch_census_begin();
                absorb_rows(
                    &ctx,
                    enc,
                    Arith {
                        lineage: PackedLineage::Exact,
                        precision: FastPrecision::Half,
                    },
                    &weight,
                    &input,
                    &baseline,
                    k,
                    m,
                    groups,
                    rows,
                )
                .unwrap();
                let kernels = dispatch_census_take();
                assert_eq!(kernels.len(), 1);
                assert_eq!(kernels[0].kernel, scalar4);
            });
            let baseline = read_f32(&baseline).unwrap();
            let mut oracle = Vec::new();
            for row in 0..rows {
                for g in 0..groups {
                    for out in 0..m {
                        oracle.push(
                            (0..k)
                                .map(|i| {
                                    decoded[(g * m + out) * k + i] as f64
                                        * values[(row * groups + g) * k + i] as f64
                                })
                                .sum::<f64>() as f32,
                        );
                    }
                }
            }
            for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
                for precision in FastPrecision::ALL {
                    let arith = Arith { lineage, precision };
                    let output = empty(&ctx, vec![(m * groups) as u64, rows as u64]);
                    let mut kernels = Vec::new();
                    run(&ctx, |enc| {
                        dispatch_census_begin();
                        absorb_rows(
                            &ctx, enc, arith, &weight, &input, &output, k, m, groups, rows,
                        )
                        .unwrap();
                        kernels = dispatch_census_take();
                    });
                    let takes_mma = lineage == PackedLineage::Fast
                        && matches!(precision, FastPrecision::Half | FastPrecision::F32)
                        && rows >= 128
                        && m.is_multiple_of(16);
                    if takes_mma {
                        assert_eq!(
                            kernels.iter().filter(|r| r.kernel == mma).count(),
                            1,
                            "{kernels:?}"
                        );
                        let tail: Vec<_> = kernels.iter().filter(|r| r.kernel == scalar4).collect();
                        assert_eq!(tail.len(), usize::from(rows % 128 != 0));
                        assert!(
                            tail.iter()
                                .all(|r| r.grid_height == (rows % 128).div_ceil(4) as u64)
                        );
                        assert_eq!(kernels.len(), 1 + tail.len());
                    } else {
                        assert_eq!(kernels.len(), 1, "{kernels:?}");
                        let r = &kernels[0];
                        assert_eq!(
                            r.kernel, scalar4,
                            "production scalar4 dispatch: {kernels:?}"
                        );
                        assert_eq!(r.grid_height, rows.div_ceil(4) as u64);
                    }
                    let got = read_f32(&output).unwrap();
                    close(
                        &format!("Q6 latent {lineage:?}/{precision:?}/{rows}"),
                        &got,
                        &oracle,
                        4e-5,
                    );
                    if !takes_mma {
                        assert_eq!(
                            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            baseline.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                            "production/Exact scalar4 arithmetic {lineage:?}/{precision:?}/{rows}"
                        );
                    }
                }
            }
            if rows >= 128 && m.is_multiple_of(16) {
                // Forced diagnostics still bypass DenseF32 caller policy.
                let output = empty(&ctx, vec![(m * groups) as u64, rows as u64]);
                run(&ctx, |enc| {
                    with_grouped_projection_scalar_baseline_override(false, || {
                        dispatch_census_begin();
                        absorb_rows(
                            &ctx,
                            enc,
                            Arith {
                                lineage: PackedLineage::Fast,
                                precision: FastPrecision::DenseF32,
                            },
                            &weight,
                            &input,
                            &output,
                            k,
                            m,
                            groups,
                            rows,
                        )
                        .unwrap();
                        assert!(dispatch_census_take().iter().any(|r| r.kernel == mma));
                    })
                });
            }
        }
    }
}

#[test]
fn q8_latent_dispatch_is_unchanged_across_fast_precisions() {
    use crate::metal::{dispatch_census_begin, dispatch_census_take};
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let (k, m, groups, rows) = (64usize, 16usize, 2usize, 129usize);
    let (weight, _) = bank(
        &ctx,
        GgmlType::Q8_0,
        vec![k as u64, m as u64, groups as u64],
    );
    let values: Vec<f32> = (0..k * groups * rows)
        .map(|i| (i % 23) as f32 * 0.017 - 0.15)
        .collect();
    let input = tensor(&ctx, &values, vec![(k * groups) as u64, rows as u64]);
    for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
        let mut reference = None;
        for precision in FastPrecision::ALL {
            let output = empty(&ctx, vec![(m * groups) as u64, rows as u64]);
            let mut signature = Vec::new();
            run(&ctx, |enc| {
                dispatch_census_begin();
                absorb_rows(
                    &ctx,
                    enc,
                    Arith { lineage, precision },
                    &weight,
                    &input,
                    &output,
                    k,
                    m,
                    groups,
                    rows,
                )
                .unwrap();
                signature = dispatch_census_take()
                    .into_iter()
                    .map(|r| (r.kernel, r.grid_width, r.grid_height, r.grid_depth))
                    .collect::<Vec<_>>();
            });
            let got = read_f32(&output).unwrap();
            assert!(got.iter().all(|v| v.is_finite()));
            let bits = got.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            if let Some((want_signature, want_bits)) = &reference {
                assert_eq!(&signature, want_signature);
                assert_eq!(&bits, want_bits);
            } else {
                reference = Some((signature, bits));
            }
        }
    }
}

#[test]
fn f32_latent_production_mma_preserves_exact_and_dense_policies() {
    use crate::metal::{dispatch_census_begin, dispatch_census_take};
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let groups = 64usize;
    for (k, m) in [(256usize, 512usize), (512, 256)] {
        let (weight, decoded) = bank(&ctx, GgmlType::F32, vec![k as u64, m as u64, groups as u64]);
        for rows in [64usize, 65] {
            let values: Vec<f32> = (0..k * groups * rows)
                .map(|i| ((i * 13 + i / k) % 89) as f32 * 0.01719 - 0.7)
                .collect();
            let input = tensor(&ctx, &values, vec![(k * groups) as u64, rows as u64]);
            let output = empty(&ctx, vec![(m * groups) as u64, rows as u64]);
            let mut baseline: Option<Vec<f32>> = None;
            for lineage in [PackedLineage::Exact, PackedLineage::Fast] {
                for precision in FastPrecision::ALL {
                    let arith = Arith { lineage, precision };
                    dispatch_census_begin();
                    run(&ctx, |enc| {
                        absorb_rows(
                            &ctx, enc, arith, &weight, &input, &output, k, m, groups, rows,
                        )
                        .unwrap()
                    });
                    let census = dispatch_census_take();
                    let mma = lineage == PackedLineage::Fast
                        && matches!(precision, FastPrecision::Half | FastPrecision::F32);
                    let scalar = "kernel_latent_grouped_f32_f32";
                    let expected = if mma {
                        let mut names = vec!["kernel_latent_grouped_f32_f32_m16n64k32"];
                        if rows == 65 {
                            names.push(scalar);
                        }
                        names
                    } else {
                        vec![scalar]
                    };
                    assert_eq!(
                        census.iter().map(|r| r.kernel.as_str()).collect::<Vec<_>>(),
                        expected
                    );
                    let got = read_f32(&output).unwrap();
                    if let Some(ref reference) = baseline {
                        close("F32 latent scalar/MMA", &got, reference, 2e-5);
                        if !mma {
                            assert_eq!(&got, reference);
                        }
                    } else {
                        baseline = Some(got.clone());
                    }
                    for row in [0, 63, rows - 1] {
                        for g in [0, 7, groups - 1] {
                            for out in [0, 15, m - 1] {
                                let expected: f64 = (0..k)
                                    .map(|i| {
                                        f64::from(decoded[(g * m + out) * k + i])
                                            * f64::from(values[(row * groups + g) * k + i])
                                    })
                                    .sum();
                                let actual = f64::from(got[(row * groups + g) * m + out]);
                                assert!(
                                    (actual - expected).abs() <= 2e-5 * expected.abs().max(1.0)
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
