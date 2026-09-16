use super::*;
use sha2::{Digest, Sha256};

fn dispatch(ctx: &MetalContext, tensors: &[&MetalTensor; 5], experts: usize, populated: bool) {
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    if populated {
        crate::metal::encode_moe_down_iq4_nl_f32_grouped_slots_m128_n16(
            ctx, &encoder, tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], 64, 130,
            experts, 17,
        )
        .unwrap();
    } else {
        let name = "kernel_moe_down_iq4_nl_f32_grouped_slots_m128_n16";
        encoder.set_pipeline(&ctx.pipeline(name).unwrap());
        encoder.set_bytes_slice(0, &[2560u32, 527, 640, 20, 640]);
        for (index, tensor) in tensors.iter().enumerate() {
            encoder.set_tensor(index + 1, tensor);
        }
        encoder.set_threadgroup_memory(0, 9216);
        let grid = MTLSize {
            width: 33,
            height: 20,
            depth: experts,
        };
        eprintln!("m128_index kernel={name} grid={grid:?} TG=128 shared_bytes=9216");
        encoder.dispatch(
            grid,
            MTLSize {
                width: 128,
                height: 1,
                depth: 1,
            },
        );
    }
    encoder.end();
    command.commit();
    command.waitUntilCompleted();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
    assert!(command.error().is_none(), "{:?}", command.error());
}

fn empty(experts: usize) {
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
    dispatch(
        &ctx,
        &[&dummy, &dummy, &counts, &dummy, &output],
        experts,
        false,
    );
    for (tensor, original) in [&dummy, &counts, &output].into_iter().zip(frozen) {
        assert_eq!(bytes(tensor), original);
    }
}

#[test]
#[ignore = "serial production lease; M128 IQ4_NL API-validation boundary control"]
fn empty_experts_511() {
    empty(511);
}

#[test]
#[ignore = "serial production lease; M128 IQ4_NL API-validation boundary"]
fn empty_experts_512() {
    empty(512);
}

#[test]
#[ignore = "serial production lease; M128 IQ4_NL populated checked-host capture"]
fn populated_m128_index_regression() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let parent = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/profiles");
    std::fs::create_dir_all(&parent).unwrap();
    let artifact = parent.join(format!("qwen4exp-m128-index-{}", std::process::id()));
    std::fs::create_dir(&artifact).unwrap();
    let ctx = MetalContext::new().expect("real Metal required");
    let mut state = 0x4d595df4d0f33173;
    let mut data = Vec::new();
    for _ in 0..2 * 130 * 2 {
        data.extend_from_slice(
            &half::f16::from_f32(sample(&mut state) * 0.01)
                .to_bits()
                .to_le_bytes(),
        );
        for _ in 0..16 {
            data.push(random(&mut state) as u8);
        }
    }
    let weights = guarded(&ctx, &data, vec![64, 130, 2], GgmlType::IQ4_NL);
    let input = f32_tensor(
        &ctx,
        &(0..34 * 64).map(|_| sample(&mut state)).collect::<Vec<_>>(),
    );
    let counts = guarded(
        &ctx,
        bytemuck::cast_slice(&[17i32, 9]),
        vec![2],
        GgmlType::I32,
    );
    let mut ids = vec![-1i32; 34];
    for i in 0..17 {
        ids[i] = ((16 - i) * 2) as i32;
    }
    for i in 0..9 {
        ids[17 + i] = ((8 - i) * 2 + 1) as i32;
    }
    let ids = guarded(&ctx, bytemuck::cast_slice(&ids), vec![34], GgmlType::I32);
    let output = f32_tensor(&ctx, &vec![f32::NAN; 34 * 130]);
    let inputs = [&weights, &input, &counts, &ids];
    let frozen = inputs.map(bytes);
    let mut observations = Vec::new();
    for name in ["product", "replay"] {
        write(&output, &vec![f32::NAN; 34 * 130]);
        dispatch(&ctx, &[&weights, &input, &counts, &ids, &output], 2, true);
        let values = read(&output);
        std::fs::write(
            artifact.join(format!("{name}-34x130.f32le")),
            bytemuck::cast_slice(&values),
        )
        .unwrap();
        guards(&output);
        for (tensor, original) in inputs.into_iter().zip(&frozen) {
            assert_eq!(&bytes(tensor), original);
        }
        observations.push(values);
    }
    eprintln!(
        "m128_index artifacts={} sha256={:x}",
        artifact.display(),
        Sha256::digest(bytemuck::cast_slice(&observations[0]))
    );
    assert_bits(&observations[0], &observations[1]);
    // Original narrow-signature M4 Max capture, before the builtin repair.
    assert_eq!(
        format!(
            "{:x}",
            Sha256::digest(bytemuck::cast_slice(&observations[0]))
        ),
        "a81d1e5299d92066cbbd46aed742ea17e6c220c2f2510dd80c151ec4ff22dd10"
    );
    let mut nonzero = 0;
    for slot in 0..34 {
        for &value in &observations[1][slot * 130..(slot + 1) * 130] {
            if slot % 2 == 0 || slot < 18 {
                assert!(value.is_finite());
                nonzero += usize::from(value != 0.0);
            } else {
                assert!(value.is_nan());
            }
        }
    }
    assert!(nonzero > 0);
    eprintln!(
        "m128_index populated bitwise PASS active_slots=26 untouched_slots=8 nonzero={nonzero}"
    );
}
