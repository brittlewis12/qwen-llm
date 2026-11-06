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

#[test]
fn scalar_query_reuse_preserves_bits_invalid_ids_and_padding() {
    let Some(ctx) = crate::test_fixtures::metal_context_or_skip() else {
        return;
    };
    const CAP: usize = 4099;
    const STRIDE: usize = 2051;
    let mut seed = 0xabc123456789u64;
    let q: Vec<f32> = (0..24 * 256).map(|_| next(&mut seed)).collect();
    let k: Vec<half::f16> = (0..CAP * 2 * 256)
        .map(|_| half::f16::from_f32(next(&mut seed)))
        .collect();
    let mut ids: Vec<i32> = (0..STRIDE)
        .map(|i| ((i * 29 + 1701) % CAP) as i32)
        .collect();
    ids[17] = -1;
    ids[19] = CAP as i32;
    ids[31] = i32::MAX;
    ids[255] = (CAP - 1) as i32;
    let qt = guarded(&ctx, bytemuck::cast_slice(&q), GgmlType::F32, vec![256, 24]);
    let kt = guarded(
        &ctx,
        bytemuck::cast_slice(&k),
        GgmlType::F16,
        vec![256, 2, CAP as u64],
    );
    let it = guarded(
        &ctx,
        bytemuck::cast_slice(&ids),
        GgmlType::I32,
        vec![STRIDE as u64],
    );
    for count in [256usize, 257, 2048, 2051] {
        let outputs: Vec<_> = (0..2)
            .map(|_| {
                guarded(
                    &ctx,
                    bytemuck::cast_slice(&vec![0x7fc00001u32; 24 * STRIDE]),
                    GgmlType::F32,
                    vec![STRIDE as u64, 24],
                )
            })
            .collect();
        let cmd = ctx.queue.commandBuffer().unwrap();
        let enc = KernelEncoder::begin(&cmd);
        for (i, name) in [
            "kernel_qwen4exp_qsa_attention_logits_f16",
            "kernel_qwen4exp_qsa_decode_logits_reuse_f32",
        ]
        .into_iter()
        .enumerate()
        {
            enc.set_pipeline(&ctx.pipeline(name).unwrap());
            enc.set_bytes(
                0,
                &super::super::AttentionArgs {
                    query_heads: 24,
                    kv_heads: 2,
                    head_dim: 256,
                    id_count: count as u32,
                    row_stride: STRIDE as u32,
                    cache_capacity: CAP as u32,
                    scale: 0.0625,
                },
            );
            enc.set_tensor(1, &qt);
            enc.set_tensor(2, &kt);
            enc.set_tensor(3, &it);
            enc.set_tensor(4, &outputs[i]);
            enc.dispatch(
                MTLSize {
                    width: if i == 0 {
                        (24 * count).div_ceil(8)
                    } else {
                        3 * count.div_ceil(32)
                    },
                    height: if i == 0 { 1 } else { 2 },
                    depth: 1,
                },
                MTLSize {
                    width: if i == 0 { 256 } else { 128 },
                    height: 1,
                    depth: 1,
                },
            );
        }
        enc.end();
        cmd.commit();
        crate::metal::wait_completed(&cmd).unwrap();
        assert_eq!(bits(&outputs[0]), bits(&outputs[1]), "id count {count}");
        let b = bits(&outputs[1]);
        for head in 0..24 {
            for slot in count..STRIDE {
                assert_eq!(b[head * STRIDE + slot], 0x7fc00001);
            }
            for slot in [17, 19, 31] {
                assert_eq!(b[head * STRIDE + slot], f32::NEG_INFINITY.to_bits());
            }
            assert!(f32::from_bits(b[head * STRIDE + 255]).is_finite());
        }
        for t in &outputs {
            guards(t);
        }
    }
    for t in [&qt, &kt, &it] {
        guards(t);
    }
}
