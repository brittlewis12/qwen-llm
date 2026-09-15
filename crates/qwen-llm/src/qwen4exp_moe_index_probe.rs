use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Args {
    ffn: u32,
    hidden: u32,
    experts: u32,
    topk: u32,
    tokens: u32,
    nb01: u32,
    stride_b: u32,
    min_count: u32,
    max_count: u32,
}

fn dispatch(ctx: &MetalContext, tensors: &[&MetalTensor; 6], args: Args, wide: bool) {
    let name = if wide {
        "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16_wide_probe"
    } else {
        "kernel_moe_swiglu_iq3_xxs_f32_grouped_slots_n16"
    };
    let command = ctx.queue.commandBuffer().unwrap();
    let encoder = KernelEncoder::begin(&command);
    encoder.set_pipeline(&ctx.pipeline(name).unwrap());
    encoder.set_bytes(0, &args);
    for (i, t) in tensors.iter().enumerate() {
        encoder.set_tensor(i + 1, t);
    }
    encoder.set_threadgroup_memory(0, 16384);
    eprintln!(
        "moe_index kernel={name} experts={} tokens={} ffn={} TG=128",
        args.experts, args.tokens, args.ffn
    );
    encoder.dispatch(
        MTLSize {
            width: (args.tokens as usize).div_ceil(16),
            height: (args.ffn as usize).div_ceil(64),
            depth: args.experts as usize,
        },
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

fn empty(experts: u32, wide: bool) {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let ctx = MetalContext::new().expect("real Metal required");
    let placeholder = f32_tensor(&ctx, &[123.0; 64]);
    let counts = f32_tensor(&ctx, &vec![0.0; experts as usize]);
    let before = bytes(&placeholder);
    let counts_before = bytes(&counts);
    let output = f32_tensor(&ctx, &[f32::NAN; 64]);
    let output_before = bytes(&output);
    dispatch(
        &ctx,
        &[
            &placeholder,
            &placeholder,
            &placeholder,
            &counts,
            &placeholder,
            &output,
        ],
        Args {
            ffn: 640,
            hidden: 2560,
            experts,
            topk: 10,
            tokens: 2048,
            nb01: 980,
            stride_b: 2560,
            min_count: 0,
            max_count: i32::MAX as u32,
        },
        wide,
    );
    assert_eq!(bytes(&placeholder), before);
    assert_eq!(bytes(&counts), counts_before);
    assert_eq!(bytes(&output), output_before);
}

#[test]
#[ignore = "serial production lease; isolated potentially aborting validation boundary"]
fn empty_narrow_511() {
    empty(511, false);
}

#[test]
#[ignore = "serial production lease; expected API validation abort, diagnostic only"]
fn empty_narrow_512() {
    empty(512, false);
}

#[test]
#[ignore = "serial production lease; isolated wide builtin validation probe"]
fn empty_wide_512() {
    empty(512, true);
}

#[test]
#[ignore = "serial production lease; narrow versus wide nonempty index equivalence"]
fn populated_index_equivalence() {
    let _lease =
        crate::metal::acquire_metal_benchmark_lease().expect("production GPU lease required");
    let ctx = MetalContext::new().expect("real Metal required");
    let mut state = 0x4d595df4d0f33173;
    let mut bank = || {
        let mut data = Vec::new();
        for _ in 0..128 {
            data.extend_from_slice(&half::f16::from_f32(1.0 / 1024.0).to_bits().to_le_bytes());
            for _ in 0..96 {
                data.push(random(&mut state) as u8);
            }
        }
        guarded(&ctx, &data, vec![256, 64, 2], GgmlType::IQ3_XXS)
    };
    let gate = bank();
    let up = bank();
    let x = f32_tensor(
        &ctx,
        &(0..17 * 256)
            .map(|i| (i % 19) as f32 * 0.02 - 0.18)
            .collect::<Vec<_>>(),
    );
    let counts = guarded(
        &ctx,
        bytemuck::cast_slice(&[17i32, 9]),
        vec![2],
        GgmlType::F32,
    );
    let mut ids = vec![-1i32; 34];
    for i in 0..17 {
        ids[i] = ((16 - i) * 2) as i32;
    }
    for i in 0..9 {
        ids[17 + i] = ((8 - i) * 2 + 1) as i32;
    }
    let ids = guarded(&ctx, bytemuck::cast_slice(&ids), vec![34], GgmlType::F32);
    let output = f32_tensor(&ctx, &vec![f32::NAN; 34 * 64]);
    let inputs = [&gate, &up, &x, &counts, &ids];
    let frozen = inputs.map(bytes);
    let args = Args {
        ffn: 64,
        hidden: 256,
        experts: 2,
        topk: 2,
        tokens: 17,
        nb01: 98,
        stride_b: 256,
        min_count: 0,
        max_count: i32::MAX as u32,
    };
    dispatch(&ctx, &[&gate, &up, &x, &counts, &ids, &output], args, false);
    let baseline = read(&output);
    write(&output, &vec![f32::NAN; 34 * 64]);
    dispatch(&ctx, &[&gate, &up, &x, &counts, &ids, &output], args, true);
    let candidate = read(&output);
    assert_bits(&candidate, &baseline);
    let mut nonzero = 0;
    for slot in 0..34 {
        for &value in &candidate[slot * 64..(slot + 1) * 64] {
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
    for (t, before) in inputs.iter().zip(frozen) {
        assert_eq!(bytes(t), before);
    }
    eprintln!("moe_index populated bitwise PASS active_slots=26 nonzero={nonzero}");
}
