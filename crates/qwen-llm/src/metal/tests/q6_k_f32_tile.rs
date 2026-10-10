//! `kernel_mat_mat_q6_K_f32_mm64x32`, the F32-operand Q6_K prompt tile:
//! its dequantization against an independent ggml-order decoder, its error
//! against an F64 reference (beside the half-staged tile's), the dispatch
//! independence GLM's Fast chunk identities rely on, guards, and refusals.

use super::*;

/// Deterministic xorshift stream.
struct Stream(u64);

impl Stream {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
    }
}

/// One Q6_K block in ggml's layout from 6-bit quants (`q[i]` in 0..64 for
/// element `i`), per-16 scales and the super-block scale.
fn q6_k_block(d: f32, scales: &[i8; 16], q: &[u8; 256]) -> [u8; 210] {
    let mut block = [0u8; 210];
    for n in 0..2 {
        for l in 0..32 {
            let [q1, q2, q3, q4] = [0, 32, 64, 96].map(|o| q[128 * n + l + o]);
            assert!(q1.max(q2).max(q3).max(q4) < 64);
            block[64 * n + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
            block[64 * n + l + 32] = (q2 & 0xF) | ((q4 & 0xF) << 4);
            block[128 + 32 * n + l] =
                (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
        }
    }
    for (i, &s) in scales.iter().enumerate() {
        block[192 + i] = s as u8;
    }
    block[208..].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    block
}

/// ggml's `dequantize_row_q6_K` for one block, written from its loop
/// structure (not from the kernel's sub-tile indexing).
fn ggml_dequantize_q6_k(block: &[u8; 210]) -> [f32; 256] {
    let d = half::f16::from_bits(u16::from_le_bytes([block[208], block[209]])).to_f32();
    let mut y = [0.0f32; 256];
    for n in 0..2 {
        let (ql, qh, sc) = (
            &block[64 * n..],
            &block[128 + 32 * n..],
            &block[192 + 8 * n..],
        );
        for l in 0..32 {
            let is = l / 16;
            let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            let scale = |i: usize| f32::from(sc[i] as i8);
            y[128 * n + l] = d * scale(is) * q1 as f32;
            y[128 * n + l + 32] = d * scale(is + 2) * q2 as f32;
            y[128 * n + l + 64] = d * scale(is + 4) * q3 as f32;
            y[128 * n + l + 96] = d * scale(is + 6) * q4 as f32;
        }
    }
    y
}

/// A Q6_K `[n_in, n_out]` weight (bytes, ggml-decoded F32 row-major) whose
/// blocks cover signed and extreme scales and every quant, 31-33 included.
fn q6_k_weight(n_in: usize, n_out: usize, seed: u64) -> (Vec<u8>, Vec<f32>) {
    let mut stream = Stream(seed);
    let (mut bytes, mut decoded) = (Vec::new(), Vec::new());
    for row in 0..n_out {
        for b in 0..n_in / 256 {
            let ordinal = row * (n_in / 256) + b;
            let d = [1.0 / 64.0, 3.0e-3, 0.25, 7.5e-4][ordinal % 4];
            let scales: [i8; 16] = std::array::from_fn(|i| match (ordinal + i) % 6 {
                0 => -128,
                1 => 127,
                2 => 0,
                3 => -1,
                _ => (stream.next() % 255) as i8,
            });
            let q: [u8; 256] = std::array::from_fn(|i| match (ordinal * 7 + i) % 9 {
                0 => 31,
                1 => 32,
                2 => 33,
                _ => (stream.next() % 64) as u8,
            });
            let block = q6_k_block(d, &scales, &q);
            let values = ggml_dequantize_q6_k(&block);
            // The decoder reads back what the layout encoded.
            let stored_d = half::f16::from_f32(d).to_f32();
            for i in 0..256 {
                let want = stored_d * f32::from(scales[i / 16]) * (f32::from(q[i]) - 32.0);
                assert_eq!(values[i].to_bits(), want.to_bits(), "layout element {i}");
            }
            bytes.extend_from_slice(&block);
            decoded.extend_from_slice(&values);
        }
    }
    (bytes, decoded)
}

/// Runs `encode` into a guarded output of `n_out x n_tokens` and returns it.
fn run_tile(
    ctx: &MetalContext,
    n_out: usize,
    n_tokens: usize,
    encode: impl FnOnce(&KernelEncoder, &MetalTensor) -> Result<(), MetalError>,
) -> Vec<f32> {
    let output = offset_tensor(
        ctx,
        48,
        &vec![0x7Fu8; n_out * n_tokens * size_of::<f32>()],
        52,
        vec![n_out as u64, n_tokens as u64],
        GgmlType::F32,
    );
    let command = ctx.queue.commandBuffer().expect("command buffer");
    let encoder = KernelEncoder::begin(&command);
    let result = encode(&encoder, &output);
    encoder.end();
    result.expect("encode");
    command.commit();
    crate::metal::wait_completed(&command).expect("command buffer completed");
    assert!(command.error().is_none(), "{:?}", command.error());
    assert_offset_guards(&output, 48, 52);
    tensor_f32_at_offset(&output)
}

fn f32_tensor(ctx: &MetalContext, prefix: usize, values: &[f32], n_in: usize) -> MetalTensor {
    offset_tensor(
        ctx,
        prefix,
        bytemuck::cast_slice(values),
        36,
        vec![n_in as u64, (values.len() / n_in) as u64],
        GgmlType::F32,
    )
}

/// Basis-vector inputs read the dequantized weights back exactly: output
/// `[t, r]` of token `e_t` is `W[r, t]` (one nonzero product, exact sums of
/// zeros), so every weight equals ggml's decode, including a partial
/// 64-row tile (80 outputs) and two super-blocks per row.
#[test]
fn q6_f32_dequant_matches_independent_reference() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (n_in, n_out) = (512usize, 80usize);
    let (bytes, decoded) = q6_k_weight(n_in, n_out, 0x9E37_79B9_7F4A_7C15);
    let weight = offset_tensor(
        &ctx,
        32,
        &bytes,
        24,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q6_K,
    );
    let mut basis = vec![0.0f32; n_in * n_in];
    for t in 0..n_in {
        basis[t * n_in + t] = 1.0;
    }
    let x = f32_tensor(&ctx, 64, &basis, n_in);
    let got = run_tile(&ctx, n_out, n_in, |enc, y| {
        crate::metal::encode_mat_mat_q6_k_f32_mm64x32(&ctx, enc, &weight, &x, y, n_in, n_out, n_in)
    });
    for t in 0..n_in {
        for r in 0..n_out {
            let (g, w) = (got[t * n_out + r], decoded[r * n_in + t]);
            assert!(g == w, "weight [{r}, {t}]: tile {g} vs ggml {w}");
        }
    }
}

/// Against an F64 product of the decoded weights, the F32 tile's error per
/// output stays within `gamma_K * sum |x w|` (`gamma_K = K u / (1 - K u)`,
/// `u = 2^-24`: the classical bound for an F32 dot product with
/// IEEE-rounded products and sums in any order, used here as this fixture's
/// ceiling, since the hardware matrix unit's internal rounding is not
/// specified), and its relative RMS is far below the half-staged tile's on
/// the same heavy-tailed activations (outlier channels at 50x).
#[test]
fn q6_f32_tile_tracks_f64_and_beats_half_tile() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (n_in, n_out, n_tokens) = (1024usize, 128usize, 64usize);
    let (bytes, decoded) = q6_k_weight(n_in, n_out, 0xD1B5_4A32_D192_ED03);
    let weight = offset_tensor(
        &ctx,
        32,
        &bytes,
        24,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q6_K,
    );
    let mut stream = Stream(0x2545_F491_4F6C_DD1D);
    let inputs: Vec<f32> = (0..n_tokens * n_in)
        .map(|i| {
            let outlier = [5, 77, 300, 901].contains(&(i % n_in));
            stream.unit() * if outlier { 50.0 } else { 1.0 }
        })
        .collect();
    let x = f32_tensor(&ctx, 64, &inputs, n_in);
    let f32_tile = run_tile(&ctx, n_out, n_tokens, |enc, y| {
        crate::metal::encode_mat_mat_q6_k_f32_mm64x32(
            &ctx, enc, &weight, &x, y, n_in, n_out, n_tokens,
        )
    });
    let half_tile = run_tile(&ctx, n_out, n_tokens, |enc, y| {
        crate::metal::encode_mat_mat_q6_k_f32(&ctx, enc, &weight, &x, y, n_in, n_out, n_tokens)
    });
    let ku = n_in as f64 * 2f64.powi(-24);
    let bound = ku / (1.0 - ku);
    let (mut f32_sq, mut half_sq, mut ref_sq, mut worst_scaled) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for t in 0..n_tokens {
        for r in 0..n_out {
            let (mut exact, mut magnitude) = (0.0f64, 0.0f64);
            for k in 0..n_in {
                let p = f64::from(inputs[t * n_in + k]) * f64::from(decoded[r * n_in + k]);
                exact += p;
                magnitude += p.abs();
            }
            let (f, h) = (
                f64::from(f32_tile[t * n_out + r]),
                f64::from(half_tile[t * n_out + r]),
            );
            assert!(f.is_finite() && h.is_finite(), "[{t}, {r}] finite");
            worst_scaled = worst_scaled.max((f - exact).abs() / magnitude.max(f64::MIN_POSITIVE));
            f32_sq += (f - exact).powi(2);
            half_sq += (h - exact).powi(2);
            ref_sq += exact.powi(2);
        }
    }
    let (f32_rms, half_rms) = ((f32_sq / ref_sq).sqrt(), (half_sq / ref_sq).sqrt());
    eprintln!(
        "Q6_K K={n_in}: F32 tile relative RMS {f32_rms:.3e}, worst scaled error {worst_scaled:.3e} (bound {bound:.3e}); half tile relative RMS {half_rms:.3e}"
    );
    assert!(
        worst_scaled <= bound,
        "worst scaled error {worst_scaled:e} > {bound:e}"
    );
    assert!(
        f32_rms * 50.0 < half_rms,
        "F32 {f32_rms:e} vs half {half_rms:e}"
    );
}

/// Each token's outputs are bitwise those of the same token in a 512-token
/// dispatch, whatever the token count, the token's place in its 32-token
/// tile, and whatever surrounds the view in its buffer (poisoned rows before
/// and after are never read); outputs outside the view are never written.
#[test]
fn q6_f32_token_outputs_do_not_depend_on_the_dispatch() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (n_in, n_out, full) = (256usize, 80usize, 512usize);
    let (bytes, _) = q6_k_weight(n_in, n_out, 0x0123_4567_89AB_CDEF);
    let weight = offset_tensor(
        &ctx,
        32,
        &bytes,
        24,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q6_K,
    );
    let mut stream = Stream(0xA076_1D64_78BD_642F);
    let values: Vec<f32> = (0..full * n_in).map(|_| stream.unit()).collect();
    let all = f32_tensor(&ctx, 64, &values, n_in);
    let reference = run_tile(&ctx, n_out, full, |enc, y| {
        crate::metal::encode_mat_mat_q6_k_f32_mm64x32(
            &ctx, enc, &weight, &all, y, n_in, n_out, full,
        )
    });
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    for n_tokens in [1usize, 31, 32, 33, 97, 128, 509, 512] {
        for first in [0usize, 3, 37] {
            if first + n_tokens > full {
                continue;
            }
            for poison in [f32::NAN, f32::INFINITY, 1.0e30] {
                // `before` poisoned rows, the tokens [first, first + n), then 40 poisoned rows.
                let before = 5;
                let mut data = vec![poison; before * n_in];
                data.extend_from_slice(&values[first * n_in..(first + n_tokens) * n_in]);
                data.resize(data.len() + 40 * n_in, poison);
                let backing = f32_tensor(&ctx, 16, &data, n_in);
                let x = backing
                    .view_subrange((before * n_in) as u64, vec![n_in as u64, n_tokens as u64]);
                let got = run_tile(&ctx, n_out, n_tokens, |enc, y| {
                    crate::metal::encode_mat_mat_q6_k_f32_mm64x32(
                        &ctx, enc, &weight, &x, y, n_in, n_out, n_tokens,
                    )
                });
                assert_eq!(
                    bits(&got),
                    bits(&reference[first * n_out..(first + n_tokens) * n_out]),
                    "N={n_tokens} first={first} poison={poison}"
                );
            }
        }
    }
}

/// The encoder refuses, before encoding, every binding the kernel cannot
/// serve: wrong dtypes, empty or non-256 geometry, size mismatches,
/// misaligned activations, a read-only or aliasing output.
#[test]
fn q6_f32_encoder_refuses_bad_bindings() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let (n_in, n_out, n) = (256usize, 64usize, 8usize);
    let (bytes, _) = q6_k_weight(n_in, n_out, 7);
    let weight = offset_tensor(
        &ctx,
        32,
        &bytes,
        24,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q6_K,
    );
    let x = f32_tensor(&ctx, 64, &vec![0.5f32; n * n_in], n_in);
    let y = offset_tensor(
        &ctx,
        16,
        &vec![0u8; n * n_out * 4],
        16,
        vec![n_out as u64, n as u64],
        GgmlType::F32,
    );
    let misaligned = f32_tensor(&ctx, 4, &vec![0.5f32; n * n_in], n_in);
    let read_only = MetalTensor {
        provenance: MetalTensorProvenance::OwnedWeightReadOnly,
        ..y.clone()
    };
    let aliasing = x.view_subrange(0, vec![n_out as u64, n as u64]);
    let (q8_bytes, _) = synthetic_q8_0_bank(n_in, n_out);
    let q8_weight = offset_tensor(
        &ctx,
        32,
        &q8_bytes,
        24,
        vec![n_in as u64, n_out as u64],
        GgmlType::Q8_0,
    );
    let cases: Vec<(&str, &MetalTensor, &MetalTensor, &MetalTensor, [usize; 3])> = vec![
        ("weight dtype", &q8_weight, &x, &y, [n_in, n_out, n]),
        ("x dtype", &weight, &weight, &y, [n_in, n_out, n]),
        ("zero tokens", &weight, &x, &y, [n_in, n_out, 0]),
        ("zero outputs", &weight, &x, &y, [n_in, 0, n]),
        ("K not 256", &weight, &x, &y, [128, n_out, n]),
        ("size mismatch", &weight, &x, &y, [n_in, n_out, n - 1]),
        ("misaligned x", &weight, &misaligned, &y, [n_in, n_out, n]),
        ("read-only y", &weight, &x, &read_only, [n_in, n_out, n]),
        ("y aliases x", &weight, &x, &aliasing, [n_in, n_out, n]),
    ];
    for (label, w, input, output, [k, m, t]) in cases {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        let result = crate::metal::encode_mat_mat_q6_k_f32_mm64x32(
            &ctx, &encoder, w, input, output, k, m, t,
        );
        encoder.end();
        assert!(
            matches!(result, Err(MetalError::BadShape { .. })),
            "{label}: {result:?}"
        );
    }
}
