//! A3B G8 coverage; the G6 encoder and its original tests remain unchanged.
use super::*;

#[test]
fn a3b_layout_and_geometry_bounds() {
    for (m, pos) in [(1, 64), (513, 131648), (2048, 262144)] {
        let a = tensor_flash_qwen_layout(m, pos - m, pos, 16, 2, 256).unwrap();
        assert_eq!(a.tiles, (m * 8).div_ceil(32));
        assert_eq!(a.q_elements, (m * 16 * 256) as u64);
        assert_eq!(a.kv_elements, (pos * 512) as u64);
        assert_eq!(a.mask_bytes, a.tiles * (pos / 64));
    }
    for (q, k, d) in [
        (0, 0, 256),
        (16, 0, 256),
        (16, 4, 256),
        (24, 2, 256),
        (16, 2, 128),
    ] {
        assert!(tensor_flash_qwen_layout(512, 32768, 33280, q, k, d).is_err());
    }
    assert!(tensor_flash_qwen_layout(512, 32768, 33281, 16, 2, 256).is_err());
    assert!(tensor_flash_qwen_layout(512, usize::MAX, 33280, 16, 2, 256).is_err());
}

fn numerical(ctx: &MetalContext, m: usize, pos: usize, uniform: bool) {
    let mut state = 0x93512ab8641u64;
    let mut random = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state >> 40) as f32 / 16_777_216. * 2. - 1.) * 1.7320508
    };
    let q: Vec<f32> = (0..m * 4096)
        .map(|_| if uniform { 0. } else { random() })
        .collect();
    let k: Vec<u16> = (0..pos * 512)
        .map(|_| half::f16::from_f32(random()).to_bits())
        .collect();
    let v: Vec<u16> = (0..pos * 512)
        .map(|_| half::f16::from_f32(random()).to_bits())
        .collect();
    let qt = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&q),
        vec![q.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let kt = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&k),
        vec![k.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    let vt = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&v),
        vec![v.len() as u64],
        GgmlType::F16,
    )
    .unwrap();
    const GUARD: usize = 16;
    const CANARY: u32 = 0x4a729ed1;
    let mut poison = vec![CANARY; q.len() + 2 * GUARD];
    poison[GUARD..GUARD + q.len()].fill(f32::NAN.to_bits());
    let backing = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&poison),
        vec![poison.len() as u64],
        GgmlType::F32,
    )
    .unwrap();
    let out = backing.view_subrange(GUARD as u64, vec![q.len() as u64]);
    let layout = tensor_flash_qwen_layout(m, pos - m, pos, 16, 2, 256).unwrap();
    let masks = MetalTensor::zeros_f32(ctx, vec![layout.mask_bytes.div_ceil(4) as u64]).unwrap();
    one_shot(ctx, |e| {
        encode_attn_tensor_flash_qwen(
            ctx,
            e,
            &qt,
            &kt,
            &vt,
            &masks,
            &out,
            m,
            pos - m,
            pos,
            16,
            2,
            256,
        )
    })
    .unwrap();
    let raw = read_back_f32(&backing.buffer, poison.len());
    assert!(
        raw[..GUARD]
            .iter()
            .chain(&raw[GUARD + q.len()..])
            .all(|v| v.to_bits() == CANARY)
    );
    let actual = &raw[GUARD..GUARD + q.len()];
    assert!(actual.iter().all(|v| v.is_finite()));
    let transposed = MetalTensor::zeros_f16(ctx, vec![(pos * 512) as u64]).unwrap();
    let scores = MetalTensor::zeros_f16(ctx, vec![(64 * 16 * pos) as u64]).unwrap();
    let ml = MetalTensor::zeros_f32(ctx, vec![attn_matrix_ml_elems(64, 16, pos) as u64]).unwrap();
    let inc = MetalTensor::zeros_f32(ctx, vec![q.len() as u64]).unwrap();
    one_shot(ctx, |e| {
        encode_attn_matrix_transpose_v_f16(ctx, e, &vt, &transposed, 0, pos, pos, 512, pos, 2, 256)
    })
    .unwrap();
    for first in (0..m).step_by(64) {
        let n = (m - first).min(64);
        let qv = qt.view_subrange((first * 4096) as u64, vec![(n * 4096) as u64]);
        let ov = inc.view_subrange((first * 4096) as u64, vec![(n * 4096) as u64]);
        let sv = scores.view_subrange(0, vec![(n * 16 * pos) as u64]);
        let mv = ml.view_subrange(0, vec![attn_matrix_ml_elems(n, 16, pos) as u64]);
        one_shot(ctx, |e| {
            encode_attn_matrix_kq_online_f32(
                ctx,
                e,
                &qv,
                &kt,
                &sv,
                &mv,
                n,
                pos - m + first,
                pos,
                512,
                16,
                2,
                8,
                256,
                true,
            )?;
            encode_attn_matrix_kqv_norm_f32(
                ctx,
                e,
                &sv,
                &mv,
                &transposed,
                &ov,
                n,
                pos - m + first,
                pos,
                pos,
                16,
                2,
                8,
                256,
                true,
            )
        })
        .unwrap();
    }
    let incumbent = read_back_f32(&inc.buffer, q.len());
    compare_vectors("A3B complete incumbent", actual, &incumbent);
    let mut av = Vec::new();
    let mut bv = Vec::new();
    let mut reference = Vec::new();
    for row in [0, m / 2, m - 1] {
        for h in [0, 7, 8, 15] {
            let end = pos - m + row + 1;
            let scores: Vec<f64> = (0..end)
                .map(|p| {
                    (0..256)
                        .map(|c| {
                            f64::from(q[(row * 16 + h) * 256 + c])
                                * f64::from(
                                    half::f16::from_bits(k[p * 512 + (h / 8) * 256 + c]).to_f32(),
                                )
                        })
                        .sum::<f64>()
                        / 16.
                })
                .collect();
            let peak = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - peak).exp()).collect();
            let sum = weights.iter().sum::<f64>();
            for c in 0..256 {
                reference.push(
                    weights
                        .iter()
                        .enumerate()
                        .map(|(p, w)| {
                            w * f64::from(
                                half::f16::from_bits(v[p * 512 + h / 8 * 256 + c]).to_f32(),
                            )
                        })
                        .sum::<f64>()
                        / sum,
                );
                av.push(actual[(row * 16 + h) * 256 + c]);
                bv.push(incumbent[(row * 16 + h) * 256 + c]);
            }
        }
    }
    let metrics = |v: &[f32]| {
        let mut max = 0f64;
        let mut e = 0.;
        let mut r = 0.;
        for (&a, &b) in v.iter().zip(&reference) {
            assert!(a.is_finite() && b.is_finite());
            let d = f64::from(a) - b;
            max = max.max(d.abs());
            e += d * d;
            r += b * b;
        }
        (max, (e / r).sqrt())
    };
    let (a, b) = (metrics(&av), metrics(&bv));
    eprintln!(
        "A3B attention F64 M{m} KV{pos} uniform={uniform} flash={a:?} incumbent={b:?} ratios={:?}",
        (a.0 / b.0, a.1 / b.1)
    );
    assert!(
        a.0 <= 1.5 * b.0 && a.1 <= 1.5 * b.1,
        "A3B F64 error exceeds 1.5x incumbent: flash={a:?} incumbent={b:?}"
    );
    // Keep the landed flash family's complete-vector and strict F64 contract.
    let ref32: Vec<f32> = reference.iter().map(|x| *x as f32).collect();
    compare_vectors("A3B sampled F64", &av, &ref32);
}

#[test]
fn a3b_small_and_partial_queries_f64() {
    gpu(true, |c| {
        for (m, p) in [(1, 64), (33, 128), (129, 192)] {
            numerical(c, m, p, false)
        }
    });
}
#[test]
fn a3b_append_512_depth32768_f64() {
    gpu(true, |c| numerical(c, 512, 33280, false));
}
#[test]
fn a3b_append_512_depth131072_f64() {
    gpu(true, |c| numerical(c, 512, 131584, false));
}
#[test]
fn a3b_query_tail_and_context_limit_f64() {
    gpu(true, |c| {
        numerical(c, 513, 131648, false);
        numerical(c, 128, 262144, false)
    });
}
#[test]
fn a3b_long_uniform_float_accumulator() {
    gpu(true, |c| numerical(c, 65, 131072, true));
}
