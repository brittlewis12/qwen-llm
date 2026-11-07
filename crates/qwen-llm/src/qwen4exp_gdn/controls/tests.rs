use super::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};

fn guarded(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
    let mut bytes = vec![0xa5; values.len() * 4 + 128];
    bytes[64..64 + values.len() * 4].copy_from_slice(bytemuck::cast_slice(values));
    MetalTensor::from_bytes(ctx, &bytes, vec![(bytes.len() / 4) as u64], GgmlType::F32)
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
            t.n_elements() as usize,
        )
        .to_vec()
    }
}
fn guard(t: &MetalTensor) {
    let bytes = unsafe {
        std::slice::from_raw_parts(t.buffer.contents().as_ptr().cast::<u8>(), t.buffer.length())
    };
    assert!(
        bytes[..64]
            .iter()
            .chain(&bytes[64 + t.n_bytes() as usize..])
            .all(|&b| b == 0xa5)
    );
}

#[test]
fn fused_controls_match_separate_projections_and_decay_with_offsets_and_guards() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    let g = GatedDeltaNetMetalGeometry::new(2560, 16, 48, 128, 4, 1e-6).unwrap();
    if pipeline(&ctx, g).unwrap().is_none() {
        return;
    }
    let mut seed = 0x673fe8ab12456u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed % 20001) as f32 / 10000. - 1.
    };
    for cancellation in [false, true] {
        let x: Vec<_> = (0..2560)
            .map(|i| {
                if cancellation {
                    if i % 2 == 0 { 1. } else { -1. }
                } else {
                    next() * 3.
                }
            })
            .collect();
        let bw: Vec<_> = (0..2560 * 48).map(|_| next() * 0.03).collect();
        let aw: Vec<_> = (0..2560 * 48).map(|_| next() * 0.03).collect();
        let dt: Vec<_> = (0..48).map(|i| [-80., 0., 80.][i % 3]).collect();
        let a: Vec<_> = (0..48).map(|_| -next().abs()).collect();
        let f = |v: &[f32], shape| guarded(&ctx, v, shape);
        let input = f(&x, vec![2560]);
        let beta = f(&bw, vec![2560, 48]);
        let alpha = f(&aw, vec![2560, 48]);
        let dt = f(&dt, vec![48]);
        let a = f(&a, vec![48]);
        let dummy = f(&[0.], vec![1]);
        let weights = GatedDeltaNetMetalWeights {
            geometry: g,
            qkv: &dummy,
            gate: &dummy,
            beta: &beta,
            alpha: &alpha,
            a: &a,
            dt_bias: &dt,
            conv: &dummy,
            norm: &dummy,
            output: &dummy,
        };
        let base_beta = f(&vec![0.; 48], vec![48]);
        let base_alpha = f(&vec![0.; 48], vec![48]);
        let base_decay = f(&vec![0.; 48], vec![48]);
        let mut workspace = GatedDeltaNetMetalWorkspace::new(&ctx, g).unwrap();
        workspace.beta = f(&vec![f32::NAN; 48], vec![48]);
        workspace.alpha = f(&vec![f32::NAN; 48], vec![48]);
        workspace.decay = f(&vec![f32::NAN; 48], vec![48]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_vec_f32_sigmoid(&ctx, &enc, &beta, &input, &base_beta, 2560, 48).unwrap();
        encode_mat_vec_dispatch(&ctx, &enc, &alpha, &input, &base_alpha, 2560, 48).unwrap();
        encode_gdn_decay_chain_f32(&ctx, &enc, &base_alpha, &dt, &a, &base_decay).unwrap();
        crate::metal::dispatch_census_begin();
        assert!(try_encode(&ctx, &enc, &input, weights, &workspace).unwrap());
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].kernel, KERNEL);
        assert_eq!(census[0].grid_width, 36);
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        for (base, candidate) in [
            (&base_alpha, &workspace.alpha),
            (&base_beta, &workspace.beta),
            (&base_decay, &workspace.decay),
        ] {
            assert_eq!(bits(base), bits(candidate));
            guard(candidate);
        }
        for tensor in [&input, &beta, &alpha, &dt, &a] {
            guard(tensor);
        }
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let mut unaligned = input.clone();
        unaligned.offset += 4;
        crate::metal::dispatch_census_begin();
        assert!(!try_encode(&ctx, &enc, &unaligned, weights, &workspace).unwrap());
        assert!(crate::metal::dispatch_census_take().is_empty());
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        let small = GatedDeltaNetMetalGeometry::new(128, 1, 1, 128, 4, 1e-6).unwrap();
        assert!(pipeline(&ctx, small).unwrap().is_none());
    }
}
