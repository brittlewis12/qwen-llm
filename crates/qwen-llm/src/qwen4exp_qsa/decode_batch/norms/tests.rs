use super::*;
use objc2_metal::MTLCommandQueue;
fn data(t: &MetalTensor) -> Vec<f32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            t.n_elements() as usize,
        )
        .to_vec()
    }
}
fn write(t: &MetalTensor, v: &[f32]) {
    assert_eq!(v.len(), t.n_elements() as usize);
    unsafe {
        std::ptr::copy_nonoverlapping(
            v.as_ptr(),
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<f32>(),
            v.len(),
        );
    }
}
fn guarded(ctx: &MetalContext, n: usize) -> (MetalTensor, MetalTensor) {
    let s = MetalTensor::zeros_f32(ctx, vec![(n + 16) as u64]).unwrap();
    write(&s, &vec![-777.; n + 16]);
    let v = s.view_subrange(8, vec![n as u64]);
    (s, v)
}
#[test]
fn independent_norms_preserve_positions_guards_and_gate_layout() {
    let _lease = crate::metal::acquire_metal_test_lease().unwrap();
    let ctx = MetalContext::new().unwrap();
    for dim in [128, 256] {
        for b in [2, 3, 4] {
            for gated in [false, true] {
                for heads in [1, 4, 20] {
                    let n = heads * dim;
                    let w = MetalTensor::zeros_f32(&ctx, vec![dim as u64]).unwrap();
                    let weights: Vec<_> = (0..dim).map(|i| 0.75 + i as f32 / 256.).collect();
                    write(&w, &weights);
                    let args: Vec<_> = (0..b)
                        .map(|i| Args {
                            rope: RopeArgs {
                                head_count: heads as u32,
                                head_dim: dim as u32,
                                rotary_dim: 64,
                                position: [0, 8192, 32768, 128000][i],
                                theta: 10000.,
                                eps: 1e-6,
                            },
                            gated: u32::from(gated),
                        })
                        .collect();
                    let inputs: Vec<_> = (0..b)
                        .map(|r| {
                            let (s, v) = guarded(&ctx, n * if gated { 2 } else { 1 });
                            let values: Vec<_> = (0..v.n_elements() as usize)
                                .map(|i| {
                                    if i < dim {
                                        0.
                                    } else {
                                        ((i * 37 + r * 11) % 503) as f32 / 71. - 3.
                                    }
                                })
                                .collect();
                            write(&v, &values);
                            (s, v, values)
                        })
                        .collect();
                    let outputs: Vec<_> = (0..b).map(|_| guarded(&ctx, n)).collect();
                    let gates: Vec<_> = (0..b).map(|_| guarded(&ctx, n)).collect();
                    let serial: Vec<_> = (0..b).map(|_| guarded(&ctx, n)).collect();
                    let serial_gates: Vec<_> = (0..b).map(|_| guarded(&ctx, n)).collect();
                    let cmd = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&cmd);
                    encode_grid(
                        &ctx,
                        &enc,
                        &w,
                        &inputs.iter().map(|s| &s.1).collect::<Vec<_>>(),
                        &outputs.iter().map(|s| &s.1).collect::<Vec<_>>(),
                        &gates.iter().map(|s| &s.1).collect::<Vec<_>>(),
                        &args,
                    )
                    .unwrap();
                    for r in 0..b {
                        enc.set_pipeline(
                            &ctx.pipeline(if gated {
                                "kernel_qwen4exp_qsa_qgate_norm_rope_f32"
                            } else {
                                "kernel_qwen4exp_qsa_norm_rope_f32"
                            })
                            .unwrap(),
                        );
                        enc.set_bytes(0, &args[r].rope);
                        enc.set_tensor(1, &inputs[r].1);
                        enc.set_tensor(2, &w);
                        enc.set_tensor(3, &serial[r].1);
                        if gated {
                            enc.set_tensor(4, &serial_gates[r].1);
                        }
                        enc.dispatch(
                            MTLSize {
                                width: n.div_ceil(256),
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 256,
                                height: 1,
                                depth: 1,
                            },
                        );
                    }
                    enc.end();
                    cmd.commit();
                    crate::metal::wait_completed(&cmd).unwrap();
                    let mut max = 0f64;
                    let (mut f64_ss, mut scale_ss) = (0., 0.);
                    for r in 0..b {
                        let y = data(&outputs[r].1);
                        let reference = data(&serial[r].1);
                        let scale = reference.iter().map(|v| v.abs() as f64).fold(1., f64::max);
                        for (&a, &z) in y.iter().zip(&reference) {
                            assert!(a.is_finite());
                            max = max.max((a as f64 - z as f64).abs() / scale);
                        }
                        assert!(max < 5e-6, "relative error {max}");
                        if gated {
                            assert_eq!(data(&gates[r].1), data(&serial_gates[r].1));
                        } else {
                            assert!(data(&gates[r].0).iter().all(|v| *v == -777.));
                        }
                        for s in [&outputs[r].0, &gates[r].0] {
                            let v = data(s);
                            assert_eq!(&v[..8], &[-777.; 8]);
                            assert_eq!(&v[v.len() - 8..], &[-777.; 8]);
                        }
                        assert_eq!(data(&inputs[r].1), inputs[r].2, "read-only input");
                        if r == 0 {
                            for h in 0..heads {
                                let row = &inputs[r].2[h * dim * if gated { 2 } else { 1 }..];
                                let ss =
                                    row[..dim].iter().map(|v| (*v as f64).powi(2)).sum::<f64>();
                                let norm = 1. / (ss / dim as f64 + 1e-6f32 as f64).sqrt();
                                for i in 0..dim {
                                    let z = row[i] as f64 * weights[i] as f64 * norm;
                                    f64_ss += (y[h * dim + i] as f64 - z).powi(2);
                                    scale_ss += z * z;
                                }
                            }
                        }
                    }
                    let normalized = (f64_ss / scale_ss.max(1e-30)).sqrt();
                    assert!(normalized < 2e-6, "F64 normalized error {normalized}");
                    eprintln!(
                        "qsa_norm b={b} gated={gated} heads={heads} max_scaled={max:e} f64_normalized={normalized:e}"
                    );
                }
            }
        }
    }
}
