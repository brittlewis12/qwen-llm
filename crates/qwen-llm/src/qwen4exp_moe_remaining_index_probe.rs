use super::*;

fn dispatch(
    ctx: &MetalContext,
    tensors: &[&MetalTensor],
    gate: bool,
    wide: bool,
    experts: usize,
    n: usize,
    m: usize,
    k: usize,
) {
    let base = if gate {
        "kernel_moe_swiglu_iq4_xs_f32_grouped_slots_n16"
    } else {
        "kernel_moe_down_q8_0_f32_grouped_slots"
    };
    let name = if wide {
        format!("{base}_wide_probe")
    } else {
        base.into()
    };
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encoder.set_pipeline(&ctx.pipeline(&name).unwrap());
    let args = if gate {
        vec![
            m as u32,
            k as u32,
            experts as u32,
            if n == 2048 { 10 } else { 2 },
            n as u32,
            (k / 256) as u32,
            k as u32,
            0,
            n as u32,
        ]
    } else {
        vec![m as u32, n as u32, k as u32, (k / 32 * 34) as u32, k as u32]
    };
    encoder.set_bytes_slice(0, &args);
    for (i, tensor) in tensors.iter().enumerate() {
        encoder.set_tensor(i + 1, tensor);
    }
    encoder.set_threadgroup_memory(0, if gate { 16384 } else { 8192 });
    let grid = MTLSize {
        width: n.div_ceil(if gate { 16 } else { 32 }),
        height: m.div_ceil(64),
        depth: experts,
    };
    eprintln!("remaining_index kernel={name} grid={grid:?} TG=128 k={k}");
    encoder.dispatch(
        grid,
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none(), "{:?}", command.error());
}

fn empty(gate: bool, wide: bool, experts: usize) {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let ctx = MetalContext::new().expect("real Metal required");
    let dummy = f32_tensor(&ctx, &[123.0; 64]);
    let counts = guarded(
        &ctx,
        bytemuck::cast_slice(&vec![0i32; experts]),
        vec![experts as u64],
        GgmlType::I32,
    );
    let output = f32_tensor(&ctx, &[f32::NAN; 64]);
    let frozen = [&dummy, &counts, &output].map(bytes);
    let tensors = if gate {
        vec![&dummy, &dummy, &dummy, &counts, &dummy, &output]
    } else {
        vec![&dummy, &dummy, &counts, &dummy, &output]
    };
    dispatch(
        &ctx,
        &tensors,
        gate,
        wide,
        experts,
        2048,
        if gate { 640 } else { 2560 },
        if gate { 2560 } else { 640 },
    );
    for (tensor, original) in [&dummy, &counts, &output].into_iter().zip(frozen) {
        assert_eq!(bytes(tensor), original);
    }
}

#[test]
#[ignore = "serial production lease; actual-artifact IQ4_XS empty-route validation"]
fn iq4_xs_original_empty_512() {
    empty(true, false, 512);
}

#[test]
#[ignore = "serial production lease; actual-artifact Q8 down empty-route validation"]
fn q8_down_original_empty_512() {
    empty(false, false, 512);
}

#[test]
#[ignore = "serial production lease; original IQ4_XS boundary control"]
fn qualified_iq4_xs_narrow_511() {
    empty(true, false, 511);
}

#[test]
#[ignore = "serial production lease; widened IQ4_XS boundary"]
fn qualified_iq4_xs_wide_512() {
    empty(true, true, 512);
}

#[test]
#[ignore = "serial production lease; original Q8 down boundary control"]
fn qualified_q8_down_narrow_511() {
    empty(false, false, 511);
}

#[test]
#[ignore = "serial production lease; widened Q8 down boundary"]
fn qualified_q8_down_wide_512() {
    empty(false, true, 512);
}

fn populated(gate: bool) {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!(
        "qwen4exp-remaining-index-{}-gate{gate}",
        std::process::id()
    ));
    std::fs::create_dir(&artifact).unwrap();
    let ctx = MetalContext::new().expect("real Metal required");
    let k = if gate { 256 } else { 64 };
    let mut state = 0x4d595df4d0f33173;
    let mut bank = || {
        let data = if gate {
            let mut data = Vec::new();
            for _ in 0..2 * 70 {
                data.extend_from_slice(
                    &half::f16::from_f32(sample(&mut state) * 0.0005)
                        .to_bits()
                        .to_le_bytes(),
                );
                for _ in 0..134 {
                    data.push(random(&mut state) as u8);
                }
            }
            data
        } else {
            q8(&mut state, 2 * 70 * k / 32, 0.002)
        };
        guarded(
            &ctx,
            &data,
            vec![k as u64, 70, 2],
            if gate {
                GgmlType::IQ4_XS
            } else {
                GgmlType::Q8_0
            },
        )
    };
    let weights = bank();
    let up = if gate { bank() } else { weights.clone() };
    let input = f32_tensor(
        &ctx,
        &(0..(if gate { 33 } else { 66 }) * k)
            .map(|_| sample(&mut state))
            .collect::<Vec<_>>(),
    );
    let counts = guarded(
        &ctx,
        bytemuck::cast_slice(&[33i32, 9]),
        vec![2],
        GgmlType::I32,
    );
    let mut ids = vec![-1i32; 66];
    for i in 0..33 {
        ids[i] = ((32 - i) * 2) as i32;
    }
    for i in 0..9 {
        ids[33 + i] = ((8 - i) * 2 + 1) as i32;
    }
    let ids = guarded(&ctx, bytemuck::cast_slice(&ids), vec![66], GgmlType::I32);
    let output = f32_tensor(&ctx, &vec![f32::NAN; 66 * 70]);
    let mut inputs = if gate {
        vec![&weights, &up, &input, &counts, &ids]
    } else {
        vec![&weights, &input, &counts, &ids]
    };
    let frozen: Vec<_> = inputs.iter().map(|t| bytes(t)).collect();
    inputs.push(&output);
    let mut observations = Vec::new();
    for (wide, name) in [(false, "narrow"), (true, "wide")] {
        write(&output, &vec![f32::NAN; 66 * 70]);
        dispatch(&ctx, &inputs, gate, wide, 2, 33, 70, k);
        let values = read(&output);
        std::fs::write(
            artifact.join(format!("{name}-66x70.f32le")),
            bytemuck::cast_slice(&values),
        )
        .unwrap();
        observations.push(values);
    }
    eprintln!("remaining_index artifacts={}", artifact.display());
    assert_bits(&observations[0], &observations[1]);
    let mut nonzero = 0;
    for slot in 0..66 {
        for &value in &observations[1][slot * 70..(slot + 1) * 70] {
            if slot % 2 == 0 || slot < 18 {
                assert!(value.is_finite());
                nonzero += usize::from(value != 0.0);
            } else {
                assert!(value.is_nan());
            }
        }
    }
    assert!(nonzero > 0);
    guards(&output);
    inputs.pop();
    for (tensor, original) in inputs.into_iter().zip(frozen) {
        assert_eq!(bytes(tensor), original);
    }
    eprintln!(
        "remaining_index populated gate={gate} bitwise PASS active_slots=42 untouched_slots=24 nonzero={nonzero}"
    );
}

#[test]
#[ignore = "serial production lease; IQ4_XS narrow/wide populated equivalence"]
fn qualified_iq4_xs_populated() {
    populated(true);
}

#[test]
#[ignore = "serial production lease; Q8 down narrow/wide populated equivalence"]
fn qualified_q8_down_populated() {
    populated(false);
}
