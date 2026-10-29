use super::*;

fn poison(tensor: &MetalTensor) {
    let elements = f32_slice(tensor).unwrap().len();
    // Every prior command is terminal; no borrowed slice survives this write.
    unsafe {
        std::slice::from_raw_parts_mut(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            elements,
        )
        .fill(f32::NAN);
    }
}

fn score(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder, cached: bool) {
    f.bind(enc);
    enc.set_tensor(2, &f.query_half);
    tensor_pipeline(
        ctx,
        enc,
        if cached {
            "score_tensor64_qregister"
        } else {
            "score_tensor64_qhalf"
        },
    )
    .unwrap();
    grid(enc, 1, f.n, 1, 128);
}

fn sink(f: &Fixture, ctx: &MetalContext, enc: &KernelEncoder, cached: bool) {
    score(f, ctx, enc, cached);
    pipeline(ctx, enc, "softmax_rows").unwrap();
    grid(enc, H, f.n, 1, 32);
    tensor_pipeline(ctx, enc, "pv_tensor64").unwrap();
    grid(enc, 1, f.n, 1, 128);
    let pso = pipeline(ctx, enc, "normalize_rows").unwrap();
    dispatch(enc, &pso, f.n * H * M);
}

#[test]
#[ignore = "resident Q-register-cache score/sink timing; no debug or shader validation"]
fn attention_q_register_cache_probe() {
    for name in [
        "MTL_DEBUG_LAYER",
        "MTL_SHADER_VALIDATION",
        "QWEN_REQUIRE_METAL_TESTS",
    ] {
        assert!(std::env::var_os(name).is_none(), "unset {name} for timing");
    }
    let signals = super::pv_dim2::CancellationGuard::install();
    let lease = metal::acquire_metal_benchmark_lease().unwrap();
    {
        let ctx = MetalContext::new().expect("Q-register timing requires Metal");
        assert!(
            metal::moe_grouped_tensor_available(&ctx),
            "tensor tier required"
        );
        for (k, n, ragged, first) in production_cases() {
            signals.check();
            let f = Fixture::new(&ctx, k, n, ragged, first);
            execute(&ctx, |enc| score(&f, &ctx, enc, false));
            let expected = f32_slice(&f.scores).unwrap().to_vec();
            poison(&f.scores);
            execute(&ctx, |enc| score(&f, &ctx, enc, true));
            let actual = f32_slice(&f.scores).unwrap();
            let mut compared = 0;
            for q in 0..n {
                for h in 0..H {
                    let base = (q * H + h) * SCORE_STRIDE;
                    for i in 0..f.count(q) {
                        assert!(expected[base + i].is_finite());
                        assert_eq!(expected[base + i].to_bits(), actual[base + i].to_bits());
                        compared += 1;
                    }
                }
            }
            execute(&ctx, |enc| sink(&f, &ctx, enc, false));
            let expected_out = f32_slice(&f.output).unwrap().to_vec();
            let expected_status = ids(&f.status, 1);
            poison(&f.output);
            execute(&ctx, |enc| sink(&f, &ctx, enc, true));
            let actual_out = f32_slice(&f.output).unwrap();
            let mut base_nonfinite = 0;
            for (&a, &b) in expected_out.iter().zip(actual_out) {
                if a.is_finite() {
                    assert_eq!(a.to_bits(), b.to_bits());
                } else {
                    base_nonfinite += 1;
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
            assert_eq!(ids(&f.status, 1), expected_status);
            f.check_guards();
            eprintln!(
                "qregister_check K={k} n={n} ragged={ragged} score_elements={compared} output_elements={} base_nonfinite={base_nonfinite} mismatches=0",
                expected_out.len()
            );
            for (scope, encode) in [
                (
                    "score",
                    score as fn(&Fixture, &MetalContext, &KernelEncoder, bool),
                ),
                ("sink", sink),
            ] {
                let timed = |cached| {
                    signals.check();
                    let command = ctx.queue.commandBuffer().unwrap();
                    let enc = KernelEncoder::begin(&command);
                    for _ in 0..4 {
                        encode(&f, &ctx, &enc, cached);
                    }
                    enc.end();
                    metal::commit_and_wait(&command).unwrap();
                    let start = command.GPUStartTime();
                    let end = command.GPUEndTime();
                    assert!(start.is_finite() && end.is_finite() && end > start && start > 0.0);
                    (end - start) * 1e6 / 4.0
                };
                timed(false);
                timed(true);
                let mut samples = [Vec::new(), Vec::new()];
                for repeat in 0..3 {
                    for arm in [0, 1, 1, 0] {
                        let us = timed(arm == 1);
                        samples[arm].push(us);
                        eprintln!(
                            "qregister_sample K={k} n={n} ragged={ragged} scope={scope} repeat={repeat} arm={arm} gpu_us={us:.3}"
                        );
                    }
                }
                for values in &mut samples {
                    values.sort_by(f64::total_cmp);
                }
                let base = (samples[0][2] + samples[0][3]) * 0.5;
                let cached = (samples[1][2] + samples[1][3]) * 0.5;
                eprintln!(
                    "qregister_median K={k} n={n} ragged={ragged} scope={scope} qhalf_gpu_us={base:.3} qregister_gpu_us={cached:.3} speedup={:.4}",
                    base / cached
                );
            }
            f.check_guards();
        }
    }
    drop(signals);
    drop(lease);
}
