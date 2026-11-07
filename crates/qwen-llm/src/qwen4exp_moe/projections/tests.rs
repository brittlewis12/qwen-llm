use super::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

fn guarded(ctx: &MetalContext, bytes: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let (block, size) = dtype.storage_layout().unwrap();
    let pad = if block == 1 { 64 } else { size as usize * 32 };
    let mut data = vec![0xa5; pad * 2 + bytes.len()];
    data[pad..pad + bytes.len()].copy_from_slice(bytes);
    MetalTensor::from_bytes(ctx, &data, vec![data.len() as u64 / size * block], dtype)
        .unwrap()
        .view_bytes(pad as u64, shape)
}
fn guard(t: &MetalTensor) {
    let p = t.offset as usize;
    let b = unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    };
    assert!(
        b[..p]
            .iter()
            .chain(&b[p + t.n_bytes() as usize..])
            .all(|&v| v == 0xa5)
    );
}
fn read(t: &MetalTensor) -> Vec<f32> {
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
fn quant(values: &[f32], dtype: GgmlType) -> Vec<u8> {
    let (block, size) = dtype.storage_layout().unwrap();
    let mut bytes = vec![0; values.len() / block as usize * size as usize];
    unsafe {
        llama_cpp_sys_2::ggml_quantize_init(dtype as u32);
        assert_eq!(
            llama_cpp_sys_2::ggml_quantize_chunk(
                dtype as u32,
                values.as_ptr(),
                bytes.as_mut_ptr().cast(),
                0,
                640,
                2560,
                std::ptr::null()
            ),
            bytes.len()
        );
    }
    bytes
}

#[test]
fn mixed_projection_packet_matches_all_quant_pairs_and_preserves_scratch_guards() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !enabled()
        || !gate_wide()
        || !crate::metal::mat_vec_f32_cooperative_enabled()
        || !ctx.tensor_ops_status().is_available()
    {
        return;
    }
    let mut seed = 0xf23746cad198u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 20001) as f32 / 10000. - 1.
    };
    let matrix: Vec<_> = (0..2560 * 640).map(|_| next() * 0.03).collect();
    let f = |v: &[f32], shape| guarded(&ctx, bytemuck::cast_slice(v), shape, GgmlType::F32);
    let rw: Vec<_> = (0..2560 * 512).map(|_| next() * 0.03).collect();
    let sw: Vec<_> = (0..2560).map(|_| next() * 0.03).collect();
    let router = f(&rw, vec![2560, 512]);
    let scalar = f(&sw, vec![2560]);
    let dummy = f(&[0.], vec![1]);
    let up_matrix: Vec<_> = (0..2560 * 640).map(|_| next() * 0.03).collect();
    let quantized: Vec<_> = [
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::IQ4_XS,
    ]
    .into_iter()
    .map(|d| guarded(&ctx, &quant(&matrix, d), vec![2560, 640], d))
    .collect();
    let up_quantized: Vec<_> = [
        GgmlType::Q4_K,
        GgmlType::Q5_K,
        GgmlType::Q6_K,
        GgmlType::IQ4_XS,
    ]
    .into_iter()
    .map(|d| guarded(&ctx, &quant(&up_matrix, d), vec![2560, 640], d))
    .collect();
    let g = Qwen4ExpMoeMetalGeometry::new(2560, 512, 10, 640, 640).unwrap();
    for cancellation in [false, true] {
        let xv: Vec<_> = (0..2560)
            .map(|i| {
                if cancellation {
                    if i % 2 == 0 { 1. } else { -1. }
                } else {
                    next() * 3.
                }
            })
            .collect();
        let input = f(&xv, vec![2560]);
        for gate in &quantized {
            for up in &up_quantized {
                let w = Qwen4ExpMoeMetalWeights {
                    geometry: g,
                    router: &router,
                    routed_gate: &dummy,
                    routed_up: &dummy,
                    routed_down: &dummy,
                    shared_router: &scalar,
                    shared_gate: gate,
                    shared_up: up,
                    shared_down: &dummy,
                };
                if pipeline(&ctx, w).unwrap().is_none() {
                    continue;
                }
                let base_gate = f(&vec![0.; 640], vec![640]);
                let base_up = f(&vec![0.; 640], vec![640]);
                let base_router = f(&vec![0.; 512], vec![512]);
                let base_scale = f(&[0.], vec![1]);
                let marker = f32::from_bits(0x7fc01234);
                let output = f(&vec![marker; 2560], vec![2560]);
                let logits = f(&vec![marker; 512], vec![512]);
                let scale = f(&[marker], vec![1]);
                let b = Qwen4ExpMoeSingletonBuffers {
                    guarded_topk: true,
                    router_logits: &logits,
                    topk_ids: &dummy,
                    topk_weights: &dummy,
                    shared_scale: &scale,
                    route_status: &dummy,
                    routed_inner: &dummy,
                    routed_expert_output: &dummy,
                    shared_inner: &dummy,
                    shared_output: &output,
                    output: &dummy,
                };
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                encode_mat_vec_dispatch(&ctx, &enc, gate, &input, &base_gate, 2560, 640).unwrap();
                encode_mat_vec_dispatch(&ctx, &enc, up, &input, &base_up, 2560, 640).unwrap();
                encode_mat_vec_dispatch(&ctx, &enc, &router, &input, &base_router, 2560, 512)
                    .unwrap();
                encode_dot_sigmoid_wide_f32(&ctx, &enc, &scalar, &input, &base_scale, 2560)
                    .unwrap();
                crate::metal::dispatch_census_begin();
                assert!(try_encode(&ctx, &enc, &input, w, b).unwrap());
                let census = crate::metal::dispatch_census_take();
                assert_eq!(census.len(), 1);
                assert_eq!(census[0].threads_width, 128);
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
                for (base, actual) in [
                    (&base_gate, output.view_subrange(0, vec![640])),
                    (&base_up, output.view_subrange(640, vec![640])),
                    (&base_router, logits.clone()),
                    (&base_scale, scale.clone()),
                ] {
                    for (a, b) in read(&actual).into_iter().zip(read(base)) {
                        assert!(
                            a.is_finite()
                                && b.is_finite()
                                && (a - b).abs() <= 1e-6 + 1e-5 * b.abs(),
                            "{:?}/{:?}: {a} vs {b}",
                            gate.dtype,
                            up.dtype
                        );
                    }
                }
                assert!(
                    read(&output)[1280..]
                        .iter()
                        .all(|v| v.to_bits() == marker.to_bits())
                );
                for t in [gate, up, &router, &scalar, &input, &output, &logits, &scale] {
                    guard(t);
                }
                let cmd = ctx.queue.commandBuffer().unwrap();
                let enc = KernelEncoder::begin(&cmd);
                let mut unaligned = input.clone();
                unaligned.offset += 4;
                crate::metal::dispatch_census_begin();
                assert!(!try_encode(&ctx, &enc, &unaligned, w, b).unwrap());
                assert!(crate::metal::dispatch_census_take().is_empty());
                enc.end();
                cmd.commit();
                crate::metal::wait_completed(&cmd).unwrap();
            }
        }
    }
}
