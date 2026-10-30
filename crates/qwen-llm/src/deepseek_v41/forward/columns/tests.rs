//! Batched column projections (`columns::encode_linear`,
//! `columns::encode_linear_pair`) against each sequence's own
//! `Linear::encode` / `encode_linear_pair`: Q5_K at the V4.1 attention q_b
//! and output_b shapes and a head-width one, and the Q8_0 query-rank /
//! KV-latent pair, through the skinny kernels (F32 rounding); F32 bitwise;
//! both column-copy forms. Synthetic random weights, not a real artifact.
use super::*;

fn xorshift(seed: u64) -> impl FnMut() -> u64 {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    }
}

fn q5_k_linear(ctx: &MetalContext, input: usize, output: usize, seed: u64) -> Linear<'_> {
    let mut next = xorshift(seed);
    let mut bytes: Vec<u8> = (0..input / 256 * output * 176)
        .map(|_| next() as u8)
        .collect();
    for block in bytes.as_chunks_mut::<176>().0 {
        for offset in [0, 2] {
            let v = 0.0001 + (next() % 1000) as f32 * 0.000004;
            block[offset..offset + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
        }
    }
    Linear {
        ctx,
        tensor: MetalTensor::from_bytes(
            ctx,
            &bytes,
            vec![input as u64, output as u64],
            GgmlType::Q5_K,
        )
        .unwrap(),
        native_control: native_control(ctx, GgmlType::Q5_K).unwrap(),
        input,
        output,

        half_weight: None,
    }
}

fn f32_linear(ctx: &MetalContext, input: usize, output: usize, seed: u64) -> Linear<'_> {
    let mut next = xorshift(seed);
    let values: Vec<f32> = (0..input * output)
        .map(|_| ((next() >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 0.05)
        .collect();
    Linear {
        ctx,
        tensor: upload(ctx, &values, vec![input as u64, output as u64]).unwrap(),
        native_control: native_control(ctx, GgmlType::F32).unwrap(),
        input,
        output,

        half_weight: None,
    }
}

fn vector(ctx: &MetalContext, n: usize, seed: u64) -> MetalTensor {
    let mut next = xorshift(seed);
    let values: Vec<f32> = (0..n)
        .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        .collect();
    upload(ctx, &values, vec![n as u64]).unwrap()
}

/// Per-sequence `Linear::encode` and `encode_linear` on the same inputs;
/// returns (per-sequence, batched) outputs.
fn run_both(
    ctx: &MetalContext,
    linear: &Linear<'_>,
    cols: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let inputs: Vec<_> = (0..cols)
        .map(|b| vector(ctx, linear.input, seed * 31 + b as u64))
        .collect();
    let nan = |n: usize| upload(ctx, &vec![f32::NAN; n], vec![n as u64]).unwrap();
    let want: Vec<_> = (0..cols).map(|_| nan(linear.output)).collect();
    let got: Vec<_> = (0..cols).map(|_| nan(linear.output)).collect();
    let scratch = allocate(ctx, &[linear.input, linear.output], cols, "test columns").unwrap();
    submit(ctx, "columns_reference", [0; 4], |enc| {
        for (x, y) in inputs.iter().zip(&want) {
            linear.encode(enc, x, y)?;
        }
        Ok(())
    })
    .unwrap();
    submit(ctx, "columns_batched", [0; 4], |enc| {
        encode_linear(
            ctx,
            enc,
            linear,
            &inputs.iter().collect::<Vec<_>>(),
            &got.iter().collect::<Vec<_>>(),
            (&scratch[0], &scratch[1]),
        )
    })
    .unwrap();
    let read = |t: &MetalTensor| f32_slice(t).unwrap().to_vec();
    (
        want.iter().map(read).collect(),
        got.iter().map(read).collect(),
    )
}

#[test]
fn batched_q5_k_attention_projections_match_per_sequence_to_f32_rounding() {
    let ctx = MetalContext::new().unwrap();
    let c = Config::default();
    let projections = [
        (
            "q_b",
            q5_k_linear(&ctx, c.query_rank, c.heads * c.head_dim, 1),
        ),
        (
            "output_b",
            q5_k_linear(&ctx, c.groups * c.output_rank, c.hidden, 2),
        ),
        ("head width", q5_k_linear(&ctx, c.hidden, 12288, 3)),
    ];
    let mut worst = 0f32;
    for per_column in [false, true] {
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(Some(per_column)));
        for (name, linear) in &projections {
            for cols in [2usize, 3, 8, 9] {
                let (want, got) = run_both(&ctx, linear, cols, cols as u64);
                for (b, (want, got)) in want.iter().zip(&got).enumerate() {
                    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                    assert!(scale > 0.0 && scale.is_finite(), "{name} column {b}");
                    for (i, (w, g)) in want.iter().zip(got).enumerate() {
                        let err = (w - g).abs();
                        assert!(
                            err <= 1e-5 * scale + 1e-6,
                            "{name} B {cols} column {b} element {i}: {g} vs {w} (scale {scale})"
                        );
                        worst = worst.max(err / scale);
                    }
                }
            }
        }
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(None));
    }
    eprintln!("[columns] Q5_K q_b/output_b max |batched - per-sequence| / scale {worst:.2e}");
}

#[test]
fn batched_f32_projection_is_bitwise_per_sequence() {
    let ctx = MetalContext::new().unwrap();
    let linear = f32_linear(&ctx, 96, 40, 3);
    for per_column in [false, true] {
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(Some(per_column)));
        for cols in [2usize, 8, 11] {
            let (want, got) = run_both(&ctx, &linear, cols, 50 + cols as u64);
            assert_eq!(want, got, "B {cols} per-column {per_column}");
        }
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(None));
    }
}

fn q8_0_linear(ctx: &MetalContext, input: usize, output: usize, seed: u64) -> Linear<'_> {
    let mut next = xorshift(seed);
    let mut bytes: Vec<u8> = (0..input / 32 * output * 34)
        .map(|_| next() as u8)
        .collect();
    for block in bytes.as_chunks_mut::<34>().0 {
        let d = 0.0002 + (next() % 1000) as f32 * 0.000004;
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_le_bytes());
    }
    Linear {
        ctx,
        tensor: MetalTensor::from_bytes(
            ctx,
            &bytes,
            vec![input as u64, output as u64],
            GgmlType::Q8_0,
        )
        .unwrap(),
        native_control: native_control(ctx, GgmlType::Q8_0).unwrap(),
        input,
        output,

        half_weight: None,
    }
}

/// Per-sequence `encode_linear_pair` and the batched pair on the same
/// inputs; returns (per-sequence, batched) outputs of a then b.
fn run_pair(
    ctx: &MetalContext,
    (a, b): (&Linear<'_>, &Linear<'_>),
    cols: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let inputs: Vec<_> = (0..cols)
        .map(|c| vector(ctx, a.input, seed * 31 + c as u64))
        .collect();
    let nan = |n: usize| upload(ctx, &vec![f32::NAN; n], vec![n as u64]).unwrap();
    let outs = |n: usize| (0..cols).map(|_| nan(n)).collect::<Vec<_>>();
    let (want_a, want_b, got_a, got_b) = (
        outs(a.output),
        outs(b.output),
        outs(a.output),
        outs(b.output),
    );
    let scratch = allocate(ctx, &[a.input, a.output, b.output], cols, "test columns").unwrap();
    submit(ctx, "pair_reference", [0; 4], |enc| {
        for ((x, ya), yb) in inputs.iter().zip(&want_a).zip(&want_b) {
            super::super::encode_linear_pair(enc, (a, ya), (b, yb), x)?;
        }
        Ok(())
    })
    .unwrap();
    fn refs(v: &[MetalTensor]) -> Vec<&MetalTensor> {
        v.iter().collect()
    }
    submit(ctx, "pair_batched", [0; 4], |enc| {
        encode_linear_pair(
            ctx,
            enc,
            (a, &refs(&got_a)),
            (b, &refs(&got_b)),
            &refs(&inputs),
            (&scratch[0], &scratch[1], &scratch[2]),
        )
    })
    .unwrap();
    let read = |v: &[MetalTensor]| {
        v.iter()
            .map(|t| f32_slice(t).unwrap().to_vec())
            .collect::<Vec<_>>()
    };
    let join = |x: Vec<Vec<f32>>, y: Vec<Vec<f32>>| x.into_iter().chain(y).collect();
    (
        join(read(&want_a), read(&want_b)),
        join(read(&got_a), read(&got_b)),
    )
}

#[test]
fn batched_q8_0_rank_pair_matches_per_sequence_to_f32_rounding() {
    let ctx = MetalContext::new().unwrap();
    let c = Config::default();
    let a = q8_0_linear(&ctx, c.hidden, c.query_rank, 7);
    let b = q8_0_linear(&ctx, c.hidden, c.head_dim, 8);
    let mut worst = 0f32;
    for per_column in [false, true] {
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(Some(per_column)));
        for cols in [2usize, 3, 8, 9] {
            let (want, got) = run_pair(&ctx, (&a, &b), cols, cols as u64);
            for (k, (want, got)) in want.iter().zip(&got).enumerate() {
                let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
                assert!(scale > 0.0 && scale.is_finite(), "output {k}");
                for (i, (w, g)) in want.iter().zip(got).enumerate() {
                    let err = (w - g).abs();
                    assert!(
                        err <= 1e-5 * scale + 1e-6,
                        "B {cols} output {k} element {i}: {g} vs {w} (scale {scale})"
                    );
                    worst = worst.max(err / scale);
                }
            }
        }
        PER_COLUMN_COPY_OVERRIDE.with(|o| o.set(None));
    }
    eprintln!("[columns] Q8_0 rank pair max |batched - per-sequence| / scale {worst:.2e}");
}

#[test]
fn batched_f32_pair_is_bitwise_per_sequence() {
    let ctx = MetalContext::new().unwrap();
    let (a, b) = (f32_linear(&ctx, 64, 24, 9), f32_linear(&ctx, 64, 10, 10));
    for cols in [2usize, 9] {
        let (want, got) = run_pair(&ctx, (&a, &b), cols, 60 + cols as u64);
        assert_eq!(want, got, "B {cols}");
    }
}
