use super::*;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue};
fn guarded(ctx: &MetalContext, bytes: &[u8], dtype: GgmlType, shape: Vec<u64>) -> MetalTensor {
    let mut data = vec![0xa5; bytes.len() + 128];
    data[64..64 + bytes.len()].copy_from_slice(bytes);
    let size = if dtype == GgmlType::F16 { 2 } else { 4 };
    MetalTensor::from_bytes(ctx, &data, vec![(data.len() / size) as u64], dtype)
        .unwrap()
        .view_bytes(64, shape)
}
fn bits(t: &MetalTensor) -> Vec<u32> {
    unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(t.offset as usize)
                .cast::<u32>(),
            t.n_bytes() as usize / 4,
        )
        .to_vec()
    }
}
fn guards(t: &MetalTensor) {
    let b = unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    };
    assert!(
        b[..64]
            .iter()
            .chain(&b[64 + t.n_bytes() as usize..64 + t.n_bytes() as usize + 64])
            .all(|&v| v == 0xa5)
    );
}
fn next(s: &mut u64) -> f32 {
    *s ^= *s << 13;
    *s ^= *s >> 7;
    *s ^= *s << 17;
    (*s % 20001) as f32 / 10000.0 - 1.0
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct AttentionArgs {
    start: u32,
    count: u32,
    qh: u32,
    kvh: u32,
    dim: u32,
    budget: u32,
    ratio: u32,
    stride: u32,
    cap: u32,
    scale: f32,
}
fn size(x: usize, y: usize, z: usize) -> MTLSize {
    MTLSize {
        width: x,
        height: y,
        depth: z,
    }
}
fn attention(ctx: &MetalContext) {
    const Q: usize = 32;
    for start in [2051usize, 32768, 131072] {
        let c = start + Q + 5;
        let mut seed = 0x12345678fedcba98;
        let q: Vec<f32> = (0..256 * 24 * Q).map(|_| next(&mut seed) * 0.5).collect();
        let k: Vec<half::f16> = (0..256 * 2 * c)
            .map(|_| half::f16::from_f32(next(&mut seed)))
            .collect();
        let mut ids: Vec<i32> = (0..2051 * Q).map(|i| (i % c) as i32).collect();
        for i in [17, 299, 2100, 12999] {
            ids[i] = -1;
        }
        ids[500] = c as i32;
        ids[501] = (start + 17) as i32;
        let mut counts = vec![512i32; Q];
        counts[1] = 511;
        let mut status = vec![0i32; Q];
        status[2] = 1;
        let qt = guarded(
            ctx,
            bytemuck::cast_slice(&q),
            GgmlType::F32,
            vec![256, 24, Q as u64],
        );
        let kt = guarded(
            ctx,
            bytemuck::cast_slice(&k),
            GgmlType::F16,
            vec![256, 2, c as u64],
        );
        let it = guarded(
            ctx,
            bytemuck::cast_slice(&ids),
            GgmlType::I32,
            vec![2051, Q as u64],
        );
        let ct = guarded(
            ctx,
            bytemuck::cast_slice(&counts),
            GgmlType::I32,
            vec![Q as u64],
        );
        let st = guarded(
            ctx,
            bytemuck::cast_slice(&status),
            GgmlType::I32,
            vec![Q as u64],
        );
        let outs: Vec<_> = (0..2)
            .map(|_| {
                guarded(
                    ctx,
                    bytemuck::cast_slice(&vec![0x7fc00001u32; 2051 * 24 * Q]),
                    GgmlType::F32,
                    vec![2051, 24, Q as u64],
                )
            })
            .collect();
        let a = AttentionArgs {
            start: start as u32,
            count: Q as u32,
            qh: 24,
            kvh: 2,
            dim: 256,
            budget: 512,
            ratio: 4,
            stride: 2051,
            cap: c as u32,
            scale: 1.0 / 16.0,
        };
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        for (i, name) in [
            "kernel_qwen4exp_qsa_attention_logits_packed_gqa4_f16",
            "kernel_qwen4exp_qsa_logits_reuse4_f32",
        ]
        .iter()
        .enumerate()
        {
            enc.set_pipeline(&ctx.pipeline(name).unwrap());
            enc.set_bytes(0, &a);
            for (b, t) in [&qt, &kt, &it, &ct, &st, &outs[i]].iter().enumerate() {
                enc.set_tensor((b + 1) as usize, t);
            }
            enc.dispatch(size(3 * 2051usize.div_ceil(32), 2, Q), size(128, 1, 1));
        }
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        for i in 1..outs.len() {
            check_bits(
                &outs[0],
                &outs[i],
                &format!("logits start{start} variant{i}"),
            );
        }
        for t in &outs {
            guards(t);
        }
        eprintln!("logits depth{start}: all bits/guards match");
        let gates: Vec<f32> = (0..12288 * Q).map(|_| next(&mut seed) * 0.5).collect();
        let masses: Vec<f32> = (0..2051 * 24 * Q).map(|_| next(&mut seed).abs()).collect();
        let den: Vec<f32> = (0..24 * Q)
            .map(|i| if i % 37 == 0 { 0.0 } else { 1024.0 })
            .collect();
        let gt = guarded(
            ctx,
            bytemuck::cast_slice(&gates),
            GgmlType::F32,
            vec![12288, Q as u64],
        );
        let mt = guarded(
            ctx,
            bytemuck::cast_slice(&masses),
            GgmlType::F32,
            vec![2051, 24, Q as u64],
        );
        let dt = guarded(
            ctx,
            bytemuck::cast_slice(&den),
            GgmlType::F32,
            vec![24, Q as u64],
        );
        let outputs: Vec<_> = (0..2)
            .map(|_| {
                guarded(
                    ctx,
                    bytemuck::cast_slice(&vec![0x7fc00001u32; 6144 * Q]),
                    GgmlType::F32,
                    vec![6144, Q as u64],
                )
            })
            .collect();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        for (i, name) in [
            "kernel_qwen4exp_qsa_selected_value_split8_f32",
            "kernel_qwen4exp_qsa_selected_value_vec8_f32",
        ]
        .iter()
        .enumerate()
        {
            enc.set_pipeline(&ctx.pipeline(name).unwrap());
            enc.set_bytes(0, &a);
            for (b, t) in [&gt, &kt, &it, &ct, &st, &mt, &outputs[i], &dt]
                .iter()
                .enumerate()
            {
                enc.set_tensor(b + 1, t);
            }
            enc.dispatch(size(8, 2, Q), size(256, 1, 1));
        }
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        check_bits(&outputs[0], &outputs[1], &format!("value start{start}"));
        for t in &outputs {
            guards(t);
        }
        for t in [&qt, &kt, &it, &ct, &st, &gt, &mt, &dt] {
            guards(t);
        }
        eprintln!("value depth{start}: all bits/guards match");
    }
}
fn check_bits(a: &MetalTensor, b: &MetalTensor, label: &str) {
    let aa = bits(a);
    let bb = bits(b);
    let mismatches: Vec<_> = aa
        .iter()
        .zip(&bb)
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(i, (&a, &b))| (i, f32::from_bits(a), f32::from_bits(b)))
        .collect();
    assert!(
        mismatches.is_empty(),
        "{label}: {} mismatches; first {:?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(5)]
    );
}

#[test]
fn reused_attention_matches_incumbent_bits_and_guards_at_depth() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !simd_available(&ctx) {
        return;
    }
    let tensor = ctx.tensor_ops_status().is_available();
    let pipelines = super::super::selected_value::Pipelines::prepare(&ctx)
        .unwrap()
        .unwrap();
    let split_on = crate::env_flag::read_default_on("QWEN_FN_QSA_PREFILL_SPLIT_VALUE");
    let vector_on = crate::env_flag::read_default_on("QWEN_FN_QSA_VALUE_VECTOR");
    for start in [2051, 32767, 32768] {
        assert_eq!(
            pipelines.uses_split(32, start),
            split_on && (tensor || (vector_on && start >= 32768))
        );
        assert_eq!(
            pipelines.uses_vector(32, start),
            split_on
                && vector_on
                && (start >= 32768
                    || (tensor
                        && crate::env_flag::read_default_on("QWEN_FN_QSA_SHALLOW_VALUE_VECTOR")))
        );
        assert!(!pipelines.uses_vector(31, start));
        assert!(!pipelines.uses_split(31, start));
    }
    attention(&ctx);
}
