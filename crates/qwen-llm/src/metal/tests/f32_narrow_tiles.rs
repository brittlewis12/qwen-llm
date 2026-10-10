//! The short-span F32-operand kernels (`kernel_mat_mat_q6_K_f32_r8c8`,
//! `kernel_mat_mat_q8_0_f32_r2c1k64`, map #12 accuracy lane) against their
//! wide tiles: each token's outputs bitwise equal, whatever the span's length
//! (1-8, and past 8), its start inside the wide tile, partial output tiles,
//! and poisoned neighbouring rows; guards intact; refusals.

use super::q6_k_f32_tile::{Stream, q6_k_weight};
use super::*;

const POISONS: [f32; 3] = [f32::NAN, f32::INFINITY, 1.0e30];

/// Runs `encode` into a guarded output of `n_out x n_tokens` and returns it.
fn run_guarded(
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

/// Activations with outlier channels (50x) and, on every third token,
/// alternate channels negated and scaled by 0.999 (sign-mixed rows; not
/// constructed to cancel against particular weights).
fn activations(tokens: usize, n_in: usize, seed: u64) -> Vec<f32> {
    let mut stream = Stream(seed);
    (0..tokens * n_in)
        .map(|i| {
            let channel = i % n_in;
            let value = stream.unit();
            if [5, 77, 300].contains(&(channel % 512)) {
                value * 50.0
            } else if (i / n_in).is_multiple_of(3) && channel % 2 == 1 {
                -value * 0.999
            } else {
                value
            }
        })
        .collect()
}

/// `values[first..first + n]` (token rows) between `before` and `after`
/// poisoned rows, viewed as `[n_in, n]`.
fn poisoned_view(
    ctx: &MetalContext,
    values: &[f32],
    n_in: usize,
    first: usize,
    n: usize,
    before: usize,
    after: usize,
    poison: f32,
) -> MetalTensor {
    let mut data = vec![poison; before * n_in];
    data.extend_from_slice(&values[first * n_in..(first + n) * n_in]);
    data.resize(data.len() + after * n_in, poison);
    let backing = offset_tensor(
        ctx,
        16,
        bytemuck::cast_slice(&data),
        16,
        vec![(data.len()) as u64],
        GgmlType::F32,
    );
    backing.view_subrange((before * n_in) as u64, vec![n_in as u64, n as u64])
}

fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Q6_K: for spans of 1-8 tokens (and 9, 17) starting anywhere in a 32-token
/// tile, the narrow kernel's outputs equal the wide tile's for the same
/// tokens in a 512-token dispatch, bit for bit; K 256 to 12,288 (the dense
/// FFN down width); output counts 1, 7, 9, 64, 79 and 80 (partly valid
/// SIMD groups, whose rows are clamped and stores masked, and wholly idle
/// groups).
#[test]
fn q6_f32_narrow_matches_the_wide_tile_bitwise() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const FULL: usize = 512;
    for (n_in, n_out) in [
        (256usize, 1usize),
        (256, 7),
        (512, 9),
        (256, 80),
        (2048, 64),
        (4096, 80),
        (12288, 79),
    ] {
        let (bytes, _) = q6_k_weight(n_in, n_out, 0x51ED_270B_27D0_9F3B ^ n_in as u64);
        let weight = offset_tensor(
            &ctx,
            32,
            &bytes,
            24,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q6_K,
        );
        let values = activations(FULL, n_in, 0x9E37_79B9 ^ n_in as u64);
        let all = offset_tensor(
            &ctx,
            64,
            bytemuck::cast_slice(&values),
            36,
            vec![n_in as u64, FULL as u64],
            GgmlType::F32,
        );
        let wide = run_guarded(&ctx, n_out, FULL, |enc, y| {
            crate::metal::encode_mat_mat_q6_k_f32_mm64x32(
                &ctx, enc, &weight, &all, y, n_in, n_out, FULL,
            )
        });
        for n in (1usize..=9).chain([17]) {
            for first in [0usize, 3, 8, 29, 31, 503 - 9] {
                for poison in POISONS {
                    let x = poisoned_view(&ctx, &values, n_in, first, n, 5, 3, poison);
                    let narrow = run_guarded(&ctx, n_out, n, |enc, y| {
                        crate::metal::encode_mat_mat_q6_k_f32_r8c8(
                            &ctx, enc, &weight, &x, y, n_in, n_out, n,
                        )
                    });
                    assert_eq!(
                        bits(&narrow),
                        bits(&wide[first * n_out..(first + n) * n_out]),
                        "K={n_in} M={n_out} N={n} first={first} poison={poison}"
                    );
                }
            }
        }
    }
}

/// Q8_0: the one-column kernel's outputs equal R2C4K64's for the same
/// tokens in a 512-token dispatch, bit for bit, for spans of 1-9 and 17
/// tokens at any start; the 8-row padded backing may hold poison.
#[test]
fn q8_f32_narrow_matches_the_wide_tile_bitwise() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    const FULL: usize = 512;
    for (n_in, n_out) in [(64usize, 16usize), (1024, 48), (4096, 32)] {
        let (bytes, _) = synthetic_q8_0_bank(n_in, n_out);
        let weight = offset_tensor(
            &ctx,
            32,
            &bytes,
            24,
            vec![n_in as u64, n_out as u64],
            GgmlType::Q8_0,
        );
        let values = activations(FULL, n_in, 0xC2B2_AE35 ^ n_in as u64);
        let all = offset_tensor(
            &ctx,
            64,
            bytemuck::cast_slice(&values),
            36,
            vec![n_in as u64, FULL as u64],
            GgmlType::F32,
        );
        let wide = run_guarded(&ctx, n_out, FULL, |enc, y| {
            crate::metal::encode_mat_mat_q8_0_f32_r2c4k64(
                &ctx, enc, &weight, &all, y, n_in, n_out, FULL,
            )
        });
        for n in (1usize..=9).chain([17]) {
            for first in [0usize, 3, 8, 29, 31, 503 - 9] {
                for poison in POISONS {
                    // The narrow kernel reads whole 8-token column tiles.
                    let after = n.div_ceil(8) * 8 - n;
                    let x = poisoned_view(&ctx, &values, n_in, first, n, 5, after, poison);
                    let narrow = run_guarded(&ctx, n_out, n, |enc, y| {
                        crate::metal::encode_mat_mat_q8_0_f32_r2c1k64(
                            &ctx, enc, &weight, &x, y, n_in, n_out, n,
                        )
                    });
                    assert_eq!(
                        bits(&narrow),
                        bits(&wide[first * n_out..(first + n) * n_out]),
                        "K={n_in} M={n_out} N={n} first={first} poison={poison}"
                    );
                }
            }
        }
    }
}

/// The narrow encoders refuse, before encoding: Q8_0 input without 8-row
/// padded backing, a misaligned or truncated weight, a truncated or
/// read-only output, an input offset past its buffer, an output over the
/// input's padding rows; Q6_K with a misaligned input, a non-256 K, or an
/// output aliasing its input.
#[test]
fn f32_narrow_encoders_refuse_bad_bindings() {
    let Some(ctx) = metal_test_context() else {
        return;
    };
    let refused =
        |result: Result<(), MetalError>| matches!(result, Err(MetalError::BadShape { .. }));
    let encode = |f: &dyn Fn(&KernelEncoder) -> Result<(), MetalError>| {
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        let result = f(&encoder);
        encoder.end();
        result
    };
    let (q8_bytes, _) = synthetic_q8_0_bank(64, 16);
    let q8 = offset_tensor(&ctx, 32, &q8_bytes, 24, vec![64, 16], GgmlType::Q8_0);
    let tight = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&vec![0.5f32; 64 * 3]),
        0,
        vec![64, 3],
        GgmlType::F32,
    );
    let y = offset_tensor(&ctx, 16, &[0u8; 16 * 3 * 4], 16, vec![16, 3], GgmlType::F32);
    assert!(refused(encode(&|enc| {
        crate::metal::encode_mat_mat_q8_0_f32_r2c1k64(&ctx, enc, &q8, &tight, &y, 64, 16, 3)
    })));
    // An 8-row backing with the valid control, then one fault at a time.
    let padded = offset_tensor(
        &ctx,
        16,
        bytemuck::cast_slice(&vec![0.5f32; 64 * 8 + 16 * 3]),
        0,
        vec![(64 * 8 + 16 * 3) as u64],
        GgmlType::F32,
    );
    let x8 = padded.view_subrange(0, vec![64, 3]);
    let q8_narrow = |w: &MetalTensor, x: &MetalTensor, y: &MetalTensor| {
        encode(&|enc| crate::metal::encode_mat_mat_q8_0_f32_r2c1k64(&ctx, enc, w, x, y, 64, 16, 3))
    };
    assert!(q8_narrow(&q8, &x8, &y).is_ok(), "valid Q8 control");
    let q8_misaligned = offset_tensor(&ctx, 1, &q8_bytes, 24, vec![64, 16], GgmlType::Q8_0);
    assert!(
        refused(q8_narrow(&q8_misaligned, &x8, &y)),
        "misaligned weight"
    );
    let q8_truncated = offset_tensor(
        &ctx,
        32,
        &q8_bytes[..q8_bytes.len() / 2],
        0,
        vec![64, 16],
        GgmlType::Q8_0,
    );
    assert!(
        refused(q8_narrow(&q8_truncated, &x8, &y)),
        "truncated weight"
    );
    let y_truncated = offset_tensor(&ctx, 16, &[0u8; 16 * 4], 0, vec![16, 3], GgmlType::F32);
    assert!(
        refused(q8_narrow(&q8, &x8, &y_truncated)),
        "truncated output"
    );
    let y_read_only = MetalTensor {
        provenance: MetalTensorProvenance::OwnedWeightReadOnly,
        ..y.clone()
    };
    assert!(
        refused(q8_narrow(&q8, &x8, &y_read_only)),
        "read-only output"
    );
    let x_past_end = MetalTensor {
        offset: padded.buffer.length() as u64 + 16,
        ..x8.clone()
    };
    assert!(
        refused(q8_narrow(&q8, &x_past_end, &y)),
        "input offset past its buffer"
    );
    let y_on_padding = padded.view_subrange(64 * 3, vec![16, 3]);
    assert!(
        refused(q8_narrow(&q8, &x8, &y_on_padding)),
        "output over the padding rows"
    );

    let (q6_bytes, _) = q6_k_weight(256, 64, 3);
    let q6 = offset_tensor(&ctx, 32, &q6_bytes, 24, vec![256, 64], GgmlType::Q6_K);
    let x = offset_tensor(
        &ctx,
        64,
        bytemuck::cast_slice(&vec![0.5f32; 256 * 4]),
        16,
        vec![256, 4],
        GgmlType::F32,
    );
    let misaligned = offset_tensor(
        &ctx,
        4,
        bytemuck::cast_slice(&vec![0.5f32; 256 * 4]),
        16,
        vec![256, 4],
        GgmlType::F32,
    );
    let y6 = offset_tensor(
        &ctx,
        16,
        &vec![0u8; 64 * 4 * 4],
        16,
        vec![64, 4],
        GgmlType::F32,
    );
    let aliasing = x.view_subrange(0, vec![64, 4]);
    let narrow = |x: &MetalTensor, y: &MetalTensor, n_in: usize| {
        encode(&|enc| crate::metal::encode_mat_mat_q6_k_f32_r8c8(&ctx, enc, &q6, x, y, n_in, 64, 4))
    };
    assert!(narrow(&x, &y6, 256).is_ok(), "valid control");
    assert!(refused(narrow(&misaligned, &y6, 256)), "misaligned x");
    assert!(refused(narrow(&x, &y6, 128)), "K not 256");
    assert!(refused(narrow(&x, &aliasing, 256)), "y aliases x");
}
