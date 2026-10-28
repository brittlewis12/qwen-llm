//! The batched shared expert (`encode_shared_columns`) against each
//! sequence's own shared expert (`encode_shared_vector`, the decode path's
//! arithmetic) at V4.1 widths: Q5_K weights go through the skinny kernel and
//! must agree to F32 rounding; F32 weights fall back column by column and
//! must agree bitwise. Synthetic random weights, not a real artifact.
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

/// Random Q5_K `[input, output]` bytes with small positive `d`/`dmin`.
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
    }
}

fn random_vector(ctx: &MetalContext, n: usize, seed: u64) -> MetalTensor {
    let mut next = xorshift(seed);
    let values: Vec<f32> = (0..n)
        .map(|_| (next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
        .collect();
    upload(ctx, &values, vec![n as u64]).unwrap()
}

fn nan_vector(ctx: &MetalContext, n: usize) -> MetalTensor {
    upload(ctx, &vec![f32::NAN; n], vec![n as u64]).unwrap()
}

/// Per-sequence shared experts and the column batch on the same inputs;
/// returns (per-sequence outputs, batched outputs, dispatched kernels of the
/// batch).
fn run_both(
    ctx: &MetalContext,
    c: &Config,
    shared: &[Linear<'_>; 3],
    cols: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<String>) {
    let inputs: Vec<_> = (0..cols)
        .map(|b| random_vector(ctx, c.hidden, seed * 31 + b as u64))
        .collect();
    let scratch: Vec<[MetalTensor; 4]> = (0..cols)
        .map(|_| {
            [c.intermediate, c.intermediate, c.intermediate, c.hidden].map(|n| nan_vector(ctx, n))
        })
        .collect();
    let batched: Vec<_> = (0..cols).map(|_| nan_vector(ctx, c.hidden)).collect();
    let columns = SharedColumns::new(ctx, c, cols).unwrap();
    submit(ctx, "shared_reference", [0; 4], |enc| {
        for (x, [g, u, a, o]) in inputs.iter().zip(&scratch) {
            encode_shared_vector(ctx, enc, shared, c.clamp, x, [g, u, a, o])?;
        }
        Ok(())
    })
    .unwrap();
    metal::dispatch_census_begin();
    submit(ctx, "shared_columns", [0; 4], |enc| {
        encode_shared_columns(
            ctx,
            enc,
            c,
            shared,
            0,
            &inputs.iter().collect::<Vec<_>>(),
            &batched.iter().collect::<Vec<_>>(),
            &columns,
        )
    })
    .unwrap();
    let kernels = metal::dispatch_census_take()
        .into_iter()
        .map(|r| r.kernel)
        .collect();
    let read = |t: &MetalTensor| f32_slice(t).unwrap().to_vec();
    (
        scratch.iter().map(|s| read(&s[3])).collect(),
        batched.iter().map(read).collect(),
        kernels,
    )
}

#[test]
fn batched_q5_k_shared_expert_matches_per_sequence_to_f32_rounding() {
    let ctx = MetalContext::new().unwrap();
    let c = Config::default();
    let shared = [
        q5_k_linear(&ctx, c.hidden, c.intermediate, 1),
        q5_k_linear(&ctx, c.hidden, c.intermediate, 2),
        q5_k_linear(&ctx, c.intermediate, c.hidden, 3),
    ];
    let mut worst = 0f32;
    for cols in [2usize, 3, 4, 8] {
        let (want, got, kernels) = run_both(&ctx, &c, &shared, cols, cols as u64);
        let count = |name: &str| kernels.iter().filter(|k| *k == name).count();
        assert_eq!(count("kernel_dense_skinny_mma_q5_K_f32"), 3, "{kernels:?}");
        assert_eq!(
            count("kernel_deepseek_v41_gather_columns"),
            1,
            "{kernels:?}"
        );
        assert_eq!(
            count("kernel_deepseek_v41_scatter_columns"),
            1,
            "{kernels:?}"
        );
        assert_eq!(kernels.len(), 6, "{kernels:?}");
        for (b, (want, got)) in want.iter().zip(&got).enumerate() {
            let scale = want.iter().fold(0f32, |m, v| m.max(v.abs()));
            assert!(scale > 0.0 && scale.is_finite(), "column {b} scale {scale}");
            for (i, (w, g)) in want.iter().zip(got).enumerate() {
                let err = (w - g).abs();
                assert!(
                    err <= 1e-5 * scale + 1e-6,
                    "B {cols} column {b} element {i}: batched {g} vs per-sequence {w} (scale {scale})"
                );
                worst = worst.max(err / scale);
            }
        }
    }
    eprintln!("[shared-columns] Q5_K max |batched - per-sequence| / output scale {worst:.2e}");
}

#[test]
fn batched_f32_shared_expert_is_bitwise_per_sequence() {
    let ctx = MetalContext::new().unwrap();
    let c = Config {
        hidden: 64,
        intermediate: 96,
        ..Config::default()
    };
    let shared = [
        f32_linear(&ctx, c.hidden, c.intermediate, 4),
        f32_linear(&ctx, c.hidden, c.intermediate, 5),
        f32_linear(&ctx, c.intermediate, c.hidden, 6),
    ];
    for cols in [2usize, 9] {
        let (want, got, kernels) = run_both(&ctx, &c, &shared, cols, 40 + cols as u64);
        assert!(!kernels.iter().any(|k| k.contains("skinny")), "{kernels:?}");
        assert_eq!(want, got, "B {cols}");
    }
}
