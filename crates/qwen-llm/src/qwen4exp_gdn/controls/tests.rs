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
    if pipeline(&ctx, g, false).unwrap().is_none() {
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
        let cv: Vec<_> = (0..10240 * 4).map(|_| next() * 0.1).collect();
        let current: Vec<_> = (0..10240).map(|_| next() * 3.).collect();
        let past: Vec<_> = (0..10240 * 3).map(|_| next() * 3.).collect();
        let conv = f(&cv, vec![4, 10240]);
        let dummy = f(&[0.], vec![1]);
        let weights = GatedDeltaNetMetalWeights {
            geometry: g,
            qkv: &dummy,
            gate: &dummy,
            beta: &beta,
            alpha: &alpha,
            a: &a,
            dt_bias: &dt,
            conv: &conv,
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
        workspace.qkv = f(&current, vec![10240]);
        workspace.conv_state = f(&past, vec![3 * 10240]);
        workspace.qkv_conv = f(&vec![f32::NAN; 10240], vec![10240]);
        workspace.query_norm = f(&vec![f32::NAN; 2048], vec![2048]);
        workspace.key_norm = f(&vec![f32::NAN; 2048], vec![2048]);
        let base_state = f(&past, vec![3 * 10240]);
        let base_conv = f(&vec![0.; 10240], vec![10240]);
        let base_q = f(&vec![0.; 2048], vec![2048]);
        let base_k = f(&vec![0.; 2048], vec![2048]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode_ssm_conv_silu_f32(
            &ctx,
            &enc,
            &workspace.qkv,
            &base_state,
            &conv,
            &base_conv,
            10240,
        )
        .unwrap();
        encode_l2_norm_pair_batched_f32(
            &ctx,
            &enc,
            &base_conv.view_subrange(0, vec![2048]),
            &base_q,
            &base_conv.view_subrange(2048, vec![2048]),
            &base_k,
            16,
            128,
            g.eps,
        )
        .unwrap();
        encode_mat_vec_f32_sigmoid(&ctx, &enc, &beta, &input, &base_beta, 2560, 48).unwrap();
        encode_mat_vec_dispatch(&ctx, &enc, &alpha, &input, &base_alpha, 2560, 48).unwrap();
        encode_gdn_decay_chain_f32(&ctx, &enc, &base_alpha, &dt, &a, &base_decay).unwrap();
        crate::metal::dispatch_census_begin();
        let fusion = try_encode(&ctx, &enc, &input, weights, &workspace).unwrap();
        assert_ne!(fusion, Fusion::Incumbent);
        let prepared = fusion == Fusion::ControlsAndPrep;
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 1);
        assert_eq!(census[0].kernel, if prepared { PREP } else { KERNEL });
        assert_eq!(census[0].grid_width, if prepared { 56 } else { 36 });
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
        if prepared {
            for (base, candidate) in [
                (&base_state, &workspace.conv_state),
                (&base_conv, &workspace.qkv_conv),
                (&base_q, &workspace.query_norm),
                (&base_k, &workspace.key_norm),
            ] {
                assert_eq!(bits(base), bits(candidate));
                guard(candidate);
            }
        }
        for tensor in [&input, &beta, &alpha, &dt, &a, &conv, &workspace.qkv] {
            guard(tensor);
        }
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        let mut unaligned = input.clone();
        unaligned.offset += 4;
        crate::metal::dispatch_census_begin();
        assert_eq!(
            try_encode(&ctx, &enc, &unaligned, weights, &workspace).unwrap(),
            Fusion::Incumbent
        );
        assert!(crate::metal::dispatch_census_take().is_empty());
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        let small = GatedDeltaNetMetalGeometry::new(128, 1, 1, 128, 4, 1e-6).unwrap();
        assert!(pipeline(&ctx, small, false).unwrap().is_none());
    }
}
