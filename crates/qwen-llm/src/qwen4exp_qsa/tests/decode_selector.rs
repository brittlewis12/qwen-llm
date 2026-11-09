use super::*;

#[test]
fn singleton_selector_scope_keeps_packed_and_other_budgets() {
    for cap in [513, 2048, 2112, 8192, 32768] {
        assert!(prefill_reuse::decode_selector_shape(true, cap, 512, 1));
        assert!(!prefill_reuse::decode_selector_shape(false, cap, 512, 1));
        assert!(!prefill_reuse::decode_selector_shape(true, cap, 512, 32));
        assert!(!prefill_reuse::decode_selector_shape(true, cap, 17, 1));
    }
    assert!(!prefill_reuse::decode_selector_shape(true, 512, 512, 1));
}

fn guarded(ctx: &MetalContext, count: usize, dtype: GgmlType) -> (MetalTensor, MetalTensor) {
    let storage = MetalTensor::zeros_dtype(ctx, vec![(count + 8) as u64], dtype).unwrap();
    let view = storage.view_subrange(4, vec![count as u64]);
    (storage, view)
}

#[test]
fn singleton_compact_selector_preserves_ids_status_and_guards() {
    let _lease = crate::metal::acquire_metal_test_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    for cap in [513usize, 2048, 2112, 8192, 32768] {
        for mode in 0..7 {
            let values: Vec<f32> = (0..cap)
                .map(|i| match mode {
                    // Distinct pseudorandom scores drive every radix pass.
                    5 => {
                        let mut s = (i as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                        s ^= s >> 29;
                        s = s.wrapping_mul(0xbf58_476d_1ce4_e5b9);
                        ((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 7.
                    }
                    // A wide band of exact ties straddles the top-512 threshold.
                    6 => {
                        if i % 3 == 0 {
                            0.25
                        } else {
                            (i % 977) as f32 / 977. - 0.5
                        }
                    }
                    0 => ((i * 37 % 1007) as f32 - 503.) / 37.,
                    1 => -2.,
                    2 => match i % 4 {
                        0 => 0.,
                        1 => -0.,
                        2 => f32::from_bits(1),
                        _ => -f32::from_bits(3),
                    },
                    _ => f32::from_bits(1f32.to_bits() + (i % 1024) as u32),
                })
                .collect();
            for visible in [
                cap as i32,
                cap as i32 - 1,
                513,
                512,
                7,
                0,
                -1,
                cap as i32 + 1,
            ] {
                let mut values = values.clone();
                if mode == 3 {
                    values[5] = f32::NAN;
                }
                if mode == 4 {
                    values[cap - 1] = f32::INFINITY;
                }
                let (scores_store, scores) = guarded(&ctx, cap, GgmlType::F32);
                write_f32_tensor(&scores_store, &vec![-777.; cap + 8]);
                write_f32_tensor(&scores, &values);
                let (vis_store, vis) = guarded(&ctx, 1, GgmlType::I32);
                write_i32_tensor(&vis_store, &[-777; 9]);
                write_i32_tensor(&vis, &[visible]);
                let frozen = read_tensor_bytes(&scores_store);
                // Every wide variant whose bound covers the row, at its PSO width.
                let wide: Vec<_> = prefill_reuse::WIDE_SELECTOR_KEYS
                    .into_iter()
                    .filter_map(|keys| {
                        let name = format!("kernel_qwen4exp_qsa_select_decode{keys}_f32");
                        let pso = ctx.pipeline(&name).unwrap();
                        let threads = pso.maxTotalThreadsPerThreadgroup().min(1024) / 32 * 32;
                        (keys * threads >= cap).then_some((pso, threads))
                    })
                    .collect();
                assert!(!wide.is_empty(), "cap={cap}");
                let kernels: Vec<_> = [
                    "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
                    "kernel_qwen4exp_qsa_select_scan_f32",
                ]
                .into_iter()
                .map(|name| (ctx.pipeline(name).unwrap(), 256))
                .chain(wide)
                .collect();
                let outputs: Vec<_> = (0..kernels.len())
                    .map(|_| {
                        let ids = guarded(&ctx, 512, GgmlType::I32);
                        let count = guarded(&ctx, 1, GgmlType::I32);
                        let status = guarded(&ctx, 1, GgmlType::I32);
                        for (storage, _) in [&ids, &count, &status] {
                            write_i32_tensor(storage, &vec![-777; storage.n_elements() as usize]);
                        }
                        (ids, count, status)
                    })
                    .collect();
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                for ((pso, threads), (ids, count, status)) in kernels.iter().zip(&outputs) {
                    enc.set_pipeline(pso);
                    enc.set_bytes(0, &[cap as u32, 512, 1, 0]);
                    for (slot, tensor) in [&scores, &vis, &ids.1, &count.1, &status.1]
                        .into_iter()
                        .enumerate()
                    {
                        enc.set_tensor(slot + 1, tensor);
                    }
                    enc.set_threadgroup_memory(0, 1024);
                    enc.set_threadgroup_memory(1, 1024);
                    enc.dispatch(
                        MTLSize {
                            width: 1,
                            height: 1,
                            depth: 1,
                        },
                        MTLSize {
                            width: *threads,
                            height: 1,
                            depth: 1,
                        },
                    );
                }
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
                let expected = &outputs[0];
                for (variant, actual) in outputs.iter().enumerate().skip(1) {
                    for (old, new) in [
                        (&expected.0, &actual.0),
                        (&expected.1, &actual.1),
                        (&expected.2, &actual.2),
                    ] {
                        assert_eq!(
                            read_tensor_bytes(&old.0),
                            read_tensor_bytes(&new.0),
                            "cap={cap} mode={mode} visible={visible} variant={variant}"
                        );
                        let values = read_i32(&new.0);
                        assert_eq!(&values[..4], &[-777; 4]);
                        assert_eq!(&values[values.len() - 4..], &[-777; 4]);
                    }
                }
                let actual = &outputs[outputs.len() - 1];
                let valid = visible > 0 && visible as usize <= cap;
                let n = if valid { visible as usize } else { 0 };
                let error = if !valid {
                    1
                } else if values[..n].iter().any(|v| !v.is_finite()) {
                    2
                } else {
                    0
                };
                assert_eq!(read_i32(&actual.2.1), [error]);
                assert_eq!(read_i32(&actual.1.1), [n.min(512) as i32]);
                assert_eq!(frozen, read_tensor_bytes(&scores_store));
            }
        }
    }
}
