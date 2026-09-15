use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    m: u32,
    n: u32,
    k: u32,
    nb01: u32,
    stride_b: u32,
}

fn dispatch(
    ctx: &MetalContext,
    tensors: &[&MetalTensor; 5],
    args: Args,
    experts: usize,
    wide: bool,
) {
    let name = if wide {
        "kernel_moe_down_iq4_nl_f32_grouped_slots_wide_probe"
    } else {
        "kernel_moe_down_iq4_nl_f32_grouped_slots"
    };
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encoder.set_pipeline(&ctx.pipeline(name).unwrap());
    encoder.set_bytes(0, &args);
    for (index, tensor) in tensors.iter().enumerate() {
        encoder.set_tensor(index + 1, tensor);
    }
    encoder.set_threadgroup_memory(0, 8192);
    let grid = MTLSize {
        width: (args.n as usize).div_ceil(32),
        height: (args.m as usize).div_ceil(64),
        depth: experts,
    };
    eprintln!("moe_down_index kernel={name} grid={grid:?} TG=128 shared_bytes=8192");
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

fn empty(experts: usize, wide: bool) {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let ctx = MetalContext::new().expect("real Metal required");
    let placeholder = f32_tensor(&ctx, &[123.0; 64]);
    let counts = guarded(
        &ctx,
        bytemuck::cast_slice(&vec![0i32; experts]),
        vec![experts as u64],
        GgmlType::I32,
    );
    let output = f32_tensor(&ctx, &[f32::NAN; 64]);
    let before = [&placeholder, &counts, &output].map(bytes);
    dispatch(
        &ctx,
        &[&placeholder, &placeholder, &counts, &placeholder, &output],
        Args {
            m: 2560,
            n: 2048,
            k: 640,
            nb01: 20,
            stride_b: 640,
        },
        experts,
        wide,
    );
    for (tensor, original) in [&placeholder, &counts, &output].into_iter().zip(before) {
        assert_eq!(bytes(tensor), original);
    }
}

#[test]
#[ignore = "serial production lease; isolated narrow builtin control"]
fn empty_narrow_511() {
    empty(511, false);
}

#[test]
#[ignore = "serial production lease; expected API-validation abort, diagnostic only"]
fn empty_narrow_512() {
    empty(512, false);
}

#[test]
#[ignore = "serial production lease; isolated widened builtin boundary"]
fn empty_wide_512() {
    empty(512, true);
}

#[test]
#[ignore = "serial production lease; nonempty narrow/wide IQ4 down equivalence"]
fn populated_down_index_equivalence() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!("qwen4exp-moe-down-index-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    let ctx = MetalContext::new().expect("real Metal required");
    let mut state = 0x4d595df4d0f33173;
    let mut data = Vec::new();
    for _ in 0..2 * 70 * 2 {
        let scale = half::f16::from_f32(sample(&mut state) * 0.01);
        data.extend_from_slice(&scale.to_bits().to_le_bytes());
        for _ in 0..16 {
            data.push(random(&mut state) as u8);
        }
    }
    let weights = guarded(&ctx, &data, vec![64, 70, 2], GgmlType::IQ4_NL);
    let input = f32_tensor(
        &ctx,
        &(0..66 * 64).map(|_| sample(&mut state)).collect::<Vec<_>>(),
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
    let inputs = [&weights, &input, &counts, &ids];
    let frozen = inputs.map(bytes);
    let args = Args {
        m: 70,
        n: 33,
        k: 64,
        nb01: 2,
        stride_b: 64,
    };
    let mut observations = Vec::new();
    for (wide, name) in [(false, "narrow"), (true, "wide")] {
        write(&output, &vec![f32::NAN; 66 * 70]);
        dispatch(
            &ctx,
            &[&weights, &input, &counts, &ids, &output],
            args,
            2,
            wide,
        );
        let values = read(&output);
        std::fs::write(
            artifact.join(format!("{name}-66x70.f32le")),
            bytemuck::cast_slice(&values),
        )
        .unwrap();
        observations.push(values);
    }
    eprintln!("moe_down_index artifacts={}", artifact.display());
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
    for (tensor, original) in inputs.into_iter().zip(frozen) {
        assert_eq!(bytes(tensor), original);
    }
    eprintln!(
        "moe_down_index populated bitwise PASS active_slots=42 untouched_slots=24 nonzero={nonzero}"
    );
}
