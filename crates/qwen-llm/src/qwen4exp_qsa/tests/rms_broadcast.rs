use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    heads: u32,
    dim: u32,
    rotary: u32,
    position: u32,
    theta: f32,
    eps: f32,
}

struct Case {
    heads: usize,
    dim: usize,
    gated: bool,
    input: MetalTensor,
    weight: MetalTensor,
    storage: MetalTensor,
    output: MetalTensor,
    gate: MetalTensor,
}

#[test]
#[ignore = "serial Metal; exact-order singleton RMS broadcast complete 12-layer component screen"]
fn rms_broadcast_screen() {
    let ctx = MetalContext::new().unwrap();
    let config = Qwen4ExpConfig::flash_next_reference();
    let geometry = QwenSparseAttentionMetalGeometry::from_config(&config, 3, 2184).unwrap();
    let mut cases = Vec::new();
    for layer in 0..12 {
        for (heads, dim, gated) in [(24, 256, true), (2, 256, false), (4, 128, false)] {
            let count = heads * dim;
            let input_count = count * if gated { 2 } else { 1 };
            let mut input = vec![0.0; input_count + 8];
            for (i, x) in input[4..input_count + 4].iter_mut().enumerate() {
                *x = (((i * 17 + layer * 29) % 101) as f32 - 50.0) * 0.02;
                if layer == 0 {
                    *x = 0.0;
                }
                if layer == 1 {
                    *x *= 1e-6;
                }
                if layer == 2 {
                    *x *= 100.0;
                }
            }
            let input = weight(&ctx, &input, vec![input.len() as u64])
                .view_subrange(4, vec![input_count as u64]);
            let weights = (0..dim)
                .map(|i| 0.8 + ((i * 7 + layer) % 31) as f32 * 0.01)
                .collect::<Vec<_>>();
            let weight = weight(&ctx, &weights, vec![dim as u64]);
            let storage = MetalTensor::zeros_f32(&ctx, vec![(count * 2 + 12) as u64]).unwrap();
            write_f32_tensor(&storage, &vec![-777.0; count * 2 + 12]);
            let output = storage.view_subrange(4, vec![count as u64]);
            let gate = storage.view_subrange((count + 8) as u64, vec![count as u64]);
            cases.push(Case {
                heads,
                dim,
                gated,
                input,
                weight,
                storage,
                output,
                gate,
            });
        }
    }
    let frozen: Vec<_> = cases
        .iter()
        .map(|c| (read_tensor_bytes(&c.input), read_tensor_bytes(&c.weight)))
        .collect();
    for position in [0, 2179, 32767, 131071] {
        let run = |candidate: bool, repeats: usize| {
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            for _ in 0..repeats {
                for case in &cases {
                    let kernel = match (candidate, case.gated) {
                        (false, false) => "kernel_qwen4exp_qsa_norm_rope_f32",
                        (false, true) => "kernel_qwen4exp_qsa_qgate_norm_rope_f32",
                        (true, false) => "kernel_qwen4exp_qsa_rms_broadcast_f32",
                        (true, true) => "kernel_qwen4exp_qsa_qgate_rms_broadcast_f32",
                    };
                    let pso = ctx.pipeline(kernel).unwrap();
                    enc.set_pipeline(&pso);
                    enc.set_bytes(
                        0,
                        &Args {
                            heads: case.heads as u32,
                            dim: case.dim as u32,
                            rotary: geometry.rotary_dim as u32,
                            position,
                            theta: geometry.theta,
                            eps: geometry.eps,
                        },
                    );
                    enc.set_tensor(1, &case.input);
                    enc.set_tensor(2, &case.weight);
                    enc.set_tensor(3, &case.output);
                    if case.gated {
                        enc.set_tensor(4, &case.gate);
                    }
                    if candidate {
                        enc.dispatch(
                            MTLSize {
                                width: case.heads,
                                height: 1,
                                depth: 1,
                            },
                            MTLSize {
                                width: 32,
                                height: 1,
                                depth: 1,
                            },
                        );
                    } else {
                        dispatch_1d(&enc, &pso, case.heads * case.dim);
                    }
                }
            }
            enc.end();
            cmd.commit();
            cmd.waitUntilCompleted();
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
            assert!(cmd.error().is_none());
            let gpu = (cmd.GPUEndTime() - cmd.GPUStartTime()) * 1e3 / repeats as f64;
            assert!(gpu.is_finite() && gpu > 0.0);
            gpu
        };
        for case in &cases {
            write_f32_tensor(&case.output, &vec![f32::NAN; case.heads * case.dim]);
            if case.gated {
                write_f32_tensor(&case.gate, &vec![f32::NAN; case.heads * case.dim]);
            }
        }
        run(false, 1);
        let reference: Vec<_> = cases
            .iter()
            .map(|c| read_tensor_bytes(&c.storage))
            .collect();
        for case in &cases {
            assert!(read_f32(&case.output).iter().all(|x| x.is_finite()));
            if case.gated {
                let input = read_f32(&case.input);
                let gate = read_f32(&case.gate);
                for head in 0..case.heads {
                    assert_eq!(
                        &gate[head * case.dim..(head + 1) * case.dim],
                        &input[(head * 2 + 1) * case.dim..(head * 2 + 2) * case.dim],
                    );
                }
            }
            write_f32_tensor(&case.output, &vec![f32::NAN; case.heads * case.dim]);
            if case.gated {
                write_f32_tensor(&case.gate, &vec![f32::NAN; case.heads * case.dim]);
            }
        }
        run(true, 1);
        for (case, expected) in cases.iter().zip(&reference) {
            assert_eq!(
                &read_tensor_bytes(&case.storage),
                expected,
                "RMS output/gate bits position={position}"
            );
            let values = read_f32(&case.storage);
            let count = case.heads * case.dim;
            for region in [
                &values[..4],
                &values[count + 4..count + 8],
                &values[count * 2 + 8..],
            ] {
                assert_eq!(region, &[-777.0; 4]);
            }
        }
        eprintln!("rms_broadcast numerical PASS position={position} calls=36");
        if position == 2179 {
            let checked_run = |candidate| {
                let time = run(candidate, 12);
                for (case, expected) in cases.iter().zip(&reference) {
                    assert_eq!(
                        &read_tensor_bytes(&case.storage),
                        expected,
                        "RMS repeated replay"
                    );
                }
                time
            };
            for candidate in [false, true, true, false] {
                checked_run(candidate);
            }
            let times = [
                checked_run(false),
                checked_run(true),
                checked_run(true),
                checked_run(false),
            ];
            let a = (times[0] + times[3]) * 0.5;
            let b = (times[1] + times[2]) * 0.5;
            let spread = (times[0] - times[3]).abs() / a;
            let useful = a - b >= 0.5 && times[0] - times[1] >= 0.5 && times[3] - times[2] >= 0.5;
            let verdict = if spread > 0.05 {
                "INCONCLUSIVE"
            } else if useful {
                "USEFUL"
            } else {
                "HOLD"
            };
            eprintln!(
                "rms_broadcast 12layer_gpu_abba_ms={times:?} saved_ms={} spread={spread} verdict={verdict}",
                a - b
            );
        }
    }
    for (case, (input, weight)) in cases.iter().zip(&frozen) {
        assert_eq!(&read_tensor_bytes(&case.input), input);
        assert_eq!(&read_tensor_bytes(&case.weight), weight);
    }
}
