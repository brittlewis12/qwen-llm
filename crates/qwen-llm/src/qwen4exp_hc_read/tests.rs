use super::*;
use objc2_metal::{MTLCommandBuffer, MTLCommandQueue};
fn guarded(ctx: &MetalContext, bytes: &[u8], shape: Vec<u64>, dtype: GgmlType) -> MetalTensor {
    let size = if dtype == GgmlType::BF16 { 2 } else { 4 };
    let mut b = vec![0xa5; bytes.len() + 128];
    b[64..64 + bytes.len()].copy_from_slice(bytes);
    MetalTensor::from_bytes(ctx, &b, vec![(b.len() / size) as u64], dtype)
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
#[test]
fn fused_hc_matches_incumbent_bits_with_offsets_and_keeps_fallbacks() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    if !enabled() || !incumbent_tg_enabled() {
        return;
    }
    let mut state = 0x734acdf012345678u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 20001) as f32 / 10000. - 1.
    };
    for cancellation in [false, true] {
        let x: Vec<f32> = (0..10240)
            .map(|i| {
                if cancellation {
                    if i % 2 == 0 { 1. } else { -1. }
                } else {
                    next() * 3.
                }
            })
            .collect();
        let w: Vec<half::bf16> = (0..10240 * 320)
            .map(|_| half::bf16::from_f32(next() * 0.03))
            .collect();
        let iw: Vec<f32> = (0..10240 * 4)
            .map(|_| half::bf16::from_f32(next() * 0.03).to_f32())
            .collect();
        let out: Vec<f32> = (0..2560).map(|_| next()).collect();
        let f = |v: &[f32], s| guarded(&ctx, bytemuck::cast_slice(v), s, GgmlType::F32);
        let input = f(&x, vec![10240]);
        let down = guarded(
            &ctx,
            bytemuck::cast_slice(&w),
            vec![10240, 320],
            GgmlType::BF16,
        );
        let inject = f(&iw, vec![10240, 4]);
        let block = f(&out, vec![2560]);
        let base_low = f(&vec![0.; 320], vec![320]);
        let base_raw = f(&vec![0.; 4], vec![4]);
        let base_res = f(&x, vec![2560, 4]);
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        encode_mat_vec_dispatch(&ctx, &enc, &down, &input, &base_low, 10240, 320).unwrap();
        encode_hc_low_activation(&ctx, &enc, &base_low, 4).unwrap();
        encode_hc_combine(
            &ctx, &enc, &inject, &input, &block, &base_res, &base_raw, 4, 2560,
        )
        .unwrap();
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        for prepare in [false, true] {
            let mut scratch = GatedResidualMetalScratch::new(&ctx, 4, 2560, 320).unwrap();
            scratch.normalized = f(&x, vec![10240]);
            scratch.low = f(&vec![0.; 320], vec![320]);
            scratch.injection = f(&vec![f32::from_bits(0x7fc12345); 4], vec![4]);
            let residual = f(&x, vec![2560, 4]);
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::dispatch_census_begin();
            let result =
                try_encode(&ctx, &enc, &down, prepare.then_some(&inject), &scratch).unwrap();
            assert!(result.is_some());
            let prepared = result.unwrap();
            let census = crate::metal::dispatch_census_take();
            assert_eq!(census.len(), 1);
            assert_eq!(census[0].grid_width, if prepared { 324 } else { 320 });
            assert_eq!(census[0].threads_width, 128);
            if prepare {
                if prepared {
                    encode_hc_injection(&ctx, &enc, &block, &scratch.injection, &residual, 4, 2560)
                        .unwrap();
                } else {
                    encode_hc_combine(
                        &ctx,
                        &enc,
                        &inject,
                        &input,
                        &block,
                        &residual,
                        &scratch.injection,
                        4,
                        2560,
                    )
                    .unwrap();
                }
            }
            enc.end();
            cmd.commit();
            crate::metal::wait_completed(&cmd).unwrap();
            assert_eq!(cmd.status(), MTLCommandBufferStatus::Completed);
            assert_eq!(bits(&scratch.low), bits(&base_low));
            if prepare {
                assert_eq!(bits(&scratch.injection), bits(&base_raw));
                assert_eq!(bits(&residual), bits(&base_res));
            } else {
                assert_eq!(bits(&scratch.injection), vec![0x7fc12345; 4]);
            }
            for t in [
                &scratch.normalized,
                &scratch.low,
                &scratch.injection,
                &residual,
            ] {
                guard(t);
            }
            let cmd = ctx.queue.commandBuffer().unwrap();
            let enc = KernelEncoder::begin(&cmd);
            crate::metal::dispatch_census_begin();
            let mut unaligned = down.clone();
            unaligned.offset += 2;
            assert!(
                try_encode(&ctx, &enc, &unaligned, None, &scratch)
                    .unwrap()
                    .is_none()
            );
            let saved = scratch.low_rank;
            scratch.low_rank = 321;
            assert!(
                try_encode(&ctx, &enc, &down, None, &scratch)
                    .unwrap()
                    .is_none()
            );
            scratch.low_rank = saved;
            assert!(crate::metal::dispatch_census_take().is_empty());
            enc.end();
        }
        for t in [
            &input, &down, &inject, &block, &base_low, &base_raw, &base_res,
        ] {
            guard(t);
        }
    }
}
