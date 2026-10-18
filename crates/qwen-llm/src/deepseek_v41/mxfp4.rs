//! Model-free HF1017 MXFP4 storage and NativeF32 CPU oracle.
//!
//! Source layout evidence: antirez/ds4 commit
//! bd66c402070042bf0a79ad6ece8242de4c93680c,
//! `gguf-tools/deepseek41_quantize.py` lines 160-193. HF source weights
//! are I8 `[out, in/2]` (treated here as raw bytes), scales are F8_E8M0
//! `[out, in/32]`; adjacent E2M1 values occupy low-even/high-odd nibbles.
//! The source HF fixtures kernel handles scaled E2M1 values. This is not
//! an FP8 equivalence claim or a reference derived from a new GPU shader.
//!
//! GGML type 39 stores 32 values in 17 bytes: scale, then 16 packed bytes
//! whose low nibbles hold positions 0..15 and high nibbles hold 16..31.
//! Repacking only permutes codes, retaining scales and negative-zero codes.
//! GGML decoding canonicalizes code 8 to positive zero: decoded zero bits
//! therefore differ from the source oracle, despite lossless packed storage.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    InvalidShape,
    SizeOverflow,
    SizeMismatch,
    InvalidCode,
    ReservedScale,
    NonFinite,
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MXFP4: {self:?}")
    }
}

impl std::error::Error for Error {}

/// Decode one source E2M1 code times 2^(scale-127), preserving signed zero.
/// Exponent 255 is reserved even for zero; finite scales can still overflow.
pub fn decode_source(code: u8, scale: u8) -> Result<f32, Error> {
    if code > 15 {
        return Err(Error::InvalidCode);
    }
    if scale == 255 {
        return Err(Error::ReservedScale);
    }
    let exponent = (code >> 1) & 3;
    let mantissa = code & 1;
    let magnitude = if exponent == 0 {
        f64::from(mantissa) * 0.5
    } else {
        (1.0 + f64::from(mantissa) * 0.5) * 2.0_f64.powi(i32::from(exponent) - 1)
    };
    // F64 keeps scale zero/subnormal and high-scale products exact until the
    // final F32 conversion, including the overflow check.
    let signed = if code & 8 != 0 { -magnitude } else { magnitude };
    let value = (signed * 2.0_f64.powi(i32::from(scale) - 127)) as f32;
    if !value.is_finite() {
        return Err(Error::NonFinite);
    }
    Ok(value)
}

/// Exact packed byte counts (source codes, source scales, GGML destination).
/// Empty dimensions are rejected; input width must consist of full blocks.
pub fn storage_sizes(rows: usize, columns: usize) -> Result<(usize, usize, usize), Error> {
    if rows == 0 || columns == 0 || columns % 32 != 0 {
        return Err(Error::InvalidShape);
    }
    let elements = rows.checked_mul(columns).ok_or(Error::SizeOverflow)?;
    let blocks = elements / 32;
    let destination = blocks.checked_mul(17).ok_or(Error::SizeOverflow)?;
    Ok((elements / 2, blocks, destination))
}

fn code_at(packed: &[u8], position: usize) -> u8 {
    (packed[position / 2] >> (4 * (position % 2))) & 15
}

fn validate_source(
    rows: usize,
    columns: usize,
    codes: &[u8],
    scales: &[u8],
) -> Result<usize, Error> {
    let (code_bytes, scale_bytes, destination) = storage_sizes(rows, columns)?;
    if codes.len() != code_bytes || scales.len() != scale_bytes {
        return Err(Error::SizeMismatch);
    }
    for (block, &scale) in codes.chunks_exact(16).zip(scales) {
        if scale == 255 {
            return Err(Error::ReservedScale);
        }
        // Magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6. Only the two
        // highest finite scales can overflow F32; the sign is irrelevant.
        if scale >= 253 {
            let limit = if scale == 253 { 6 } else { 4 };
            for &byte in block {
                if byte & 7 >= limit || (byte >> 4) & 7 >= limit {
                    return Err(Error::NonFinite);
                }
            }
        }
    }
    Ok(destination)
}

/// Row-major lossless source-to-GGML permutation into exact caller storage.
/// No allocation or requantization. Every error leaves all output untouched,
/// including a reserved scale or overflowing value in the final block.
pub fn repack_into(
    rows: usize,
    columns: usize,
    codes: &[u8],
    scales: &[u8],
    output: &mut [u8],
) -> Result<(), Error> {
    let (_, _, destination) = storage_sizes(rows, columns)?;
    if output.len() != destination {
        return Err(Error::SizeMismatch);
    }
    validate_source(rows, columns, codes, scales)?;
    for ((source, &scale), target) in codes
        .chunks_exact(16)
        .zip(scales)
        .zip(output.chunks_exact_mut(17))
    {
        target[0] = scale;
        for j in 0..16 {
            target[j + 1] = code_at(source, j) | (code_at(source, j + 16) << 4);
        }
    }
    Ok(())
}

/// Tiny NativeF32 row-dot oracle, using source decoding and sequential F32
/// multiply then add (not fused). No expanded persistent bank is created.
/// Future parallel GPU comparisons require tolerances, not bitwise equality.
/// Nonfinite inputs, products, or intermediate sums are rejected.
pub fn row_dot_native_f32(codes: &[u8], scales: &[u8], input: &[f32]) -> Result<f32, Error> {
    validate_source(1, input.len(), codes, scales)?;
    let mut sum = 0.0_f32;
    for (position, &x) in input.iter().enumerate() {
        if !x.is_finite() {
            return Err(Error::NonFinite);
        }
        let weight = decode_source(code_at(codes, position), scales[position / 32])?;
        let product = weight * x;
        sum += product;
        if !product.is_finite() || !sum.is_finite() {
            return Err(Error::NonFinite);
        }
    }
    Ok(sum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{GgmlType, TensorDesc};

    #[test]
    fn exhaustive_codes_and_scales() {
        // Independent exact half-unit magnitudes, used only in tests.
        let halves = [0_u32, 1, 2, 3, 4, 6, 8, 12];
        for scale in 0..=255_u8 {
            for code in 0..16_u8 {
                let actual = decode_source(code, scale);
                if scale == 255 {
                    assert_eq!(actual, Err(Error::ReservedScale));
                    continue;
                }
                let magnitude =
                    f64::from(halves[(code & 7) as usize]) * 2.0_f64.powi(i32::from(scale) - 128);
                let expected = (if code & 8 == 0 { magnitude } else { -magnitude }) as f32;
                if expected.is_finite() {
                    assert_eq!(actual.unwrap().to_bits(), expected.to_bits());
                } else {
                    assert_eq!(actual, Err(Error::NonFinite));
                }
            }
        }
        assert_eq!(decode_source(1, 0).unwrap().to_bits(), 0x0020_0000);
        assert_eq!(decode_source(1, 1).unwrap().to_bits(), 0x0040_0000);
        assert_eq!(decode_source(8, 254).unwrap().to_bits(), 0x8000_0000);
        assert_eq!(decode_source(2, 254).unwrap(), 2.0_f32.powi(127));
        assert_eq!(decode_source(4, 254), Err(Error::NonFinite));
        assert_eq!(decode_source(16, 127), Err(Error::InvalidCode));
    }

    #[test]
    fn exhaustive_packed_validation_matches_decoder_with_canaries() {
        for scale in 0..=255_u8 {
            for byte in 0..=255_u8 {
                let codes = [byte; 16];
                let expected = (0..32).try_for_each(|position| {
                    decode_source(code_at(&codes, position), scale).map(|_| ())
                });
                assert_eq!(
                    validate_source(1, 32, &codes, &[scale]),
                    expected.map(|()| 17)
                );
                let mut output = [0xa5; 19];
                assert_eq!(
                    repack_into(1, 32, &codes, &[scale], &mut output[1..18]),
                    expected,
                    "byte={byte} scale={scale}"
                );
                if expected.is_err() {
                    assert_eq!(output, [0xa5; 19]);
                } else {
                    assert_eq!((output[0], output[18]), (0xa5, 0xa5));
                    assert_eq!(output[1], scale);
                    for position in 0..32 {
                        let packed = output[2 + position % 16];
                        assert_eq!(
                            (packed >> (4 * (position / 16))) & 15,
                            code_at(&codes, position)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn validation_preserves_first_error_and_atomicity() {
        for (scales, expected) in [
            ([253, 255], Error::NonFinite),
            ([255, 253], Error::ReservedScale),
            ([252, 255], Error::ReservedScale),
            ([252, 254], Error::NonFinite),
        ] {
            let codes = [0x77; 32];
            let mut output = [0xa5; 36];
            assert_eq!(
                repack_into(1, 64, &codes, &scales, &mut output[1..35]),
                Err(expected)
            );
            assert_eq!(output, [0xa5; 36]);
            assert_eq!(
                repack_into(1, 64, &codes, &scales, &mut output),
                Err(Error::SizeMismatch)
            );
            assert_eq!(output, [0xa5; 36]);
        }
    }

    #[test]
    fn every_position_routes_uniquely_across_rows_and_blocks() {
        // One marked nibble at a time distinguishes all 128 positions, even
        // though only sixteen distinct codes exist.
        for position in 0..128 {
            let mut source = [0_u8; 64];
            source[position / 2] = 13 << (4 * (position % 2));
            let scales = [125, 126, 127, 128];
            let mut output = [0_u8; 68];
            repack_into(2, 64, &source, &scales, &mut output).unwrap();
            let mut expected = [0_u8; 68];
            for block in 0..4 {
                expected[block * 17] = scales[block];
            }
            let local = position % 32;
            expected[(position / 32) * 17 + 1 + local % 16] = 13 << (4 * (local / 16));
            assert_eq!(output, expected);
        }
    }

    #[test]
    fn linked_ggml_independent_decode() {
        assert_eq!(GgmlType::MXFP4 as i32, 39);
        // Homogeneous blocks admit every finite code/scale pair, including
        // high scales which would overflow for other codes.
        for scale in 0..255_u8 {
            for code in 0..16_u8 {
                let Ok(expected) = decode_source(code, scale) else {
                    continue;
                };
                let mut output = [0; 17];
                repack_into(1, 32, &[code | (code << 4); 16], &[scale], &mut output).unwrap();
                let desc = TensorDesc {
                    name: "synthetic-mxfp4".into(),
                    shape: vec![32, 1],
                    dtype: GgmlType::MXFP4,
                    shard_idx: 0,
                    data_offset: 0,
                    n_bytes: 17,
                };
                let decoded = crate::codec::dequant_to_f32(&desc, &output).unwrap();
                for value in decoded {
                    if code == 8 {
                        assert_eq!(expected.to_bits(), 0x8000_0000);
                        assert_eq!(value.to_bits(), 0);
                        assert_eq!(value, expected);
                    } else {
                        assert_eq!(
                            value.to_bits(),
                            expected.to_bits(),
                            "code={code} scale={scale}"
                        );
                    }
                }
                assert_eq!(output[1..], [code | (code << 4); 16]);
            }
        }
    }

    #[test]
    fn linked_ggml_mixed_rows() {
        let source: Vec<u8> = (0..64)
            .map(|i| ((i * 7 % 16) | (((i * 3 + 1) % 16) << 4)) as u8)
            .collect();
        let scales = [0, 1, 127, 251];
        let mut packed = [0; 68];
        repack_into(2, 64, &source, &scales, &mut packed).unwrap();
        let desc = TensorDesc {
            name: "mixed-rows-mxfp4".into(),
            shape: vec![64, 2],
            dtype: GgmlType::MXFP4,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: 68,
        };
        let decoded = crate::codec::dequant_to_f32(&desc, &packed).unwrap();
        for (i, actual) in decoded.iter().enumerate() {
            let code = code_at(&source, i);
            let expected = decode_source(code, scales[i / 32]).unwrap();
            if code == 8 {
                assert_eq!(actual.to_bits(), 0);
                assert_eq!(expected.to_bits(), 0x8000_0000);
            } else {
                assert_eq!(actual.to_bits(), expected.to_bits());
            }
        }
    }

    #[test]
    fn errors_are_atomic_with_canaries() {
        let mut output = [0xa5; 70];
        let mut codes = [0x22; 64];
        for scales in [[127, 127, 127, 255], [127, 127, 127, 254]] {
            codes[63] = 0x42;
            assert!(repack_into(2, 64, &codes, &scales, &mut output[1..69]).is_err());
            assert_eq!(output, [0xa5; 70]);
        }
        for (rows, columns) in [(0, 32), (1, 0), (1, 31), (usize::MAX, 32)] {
            assert!(repack_into(rows, columns, &codes, &[127; 4], &mut output[1..69]).is_err());
            assert_eq!(output, [0xa5; 70]);
        }
        assert_eq!(
            storage_sizes(usize::MAX / 32 + 1, 32),
            Err(Error::SizeOverflow)
        );
        assert!(repack_into(2, 64, &codes[..63], &[127; 4], &mut output[1..69]).is_err());
        assert!(repack_into(2, 64, &codes, &[127; 3], &mut output[1..69]).is_err());
        assert!(repack_into(2, 64, &codes, &[127; 4], &mut output).is_err());
        assert_eq!(output, [0xa5; 70]);
        repack_into(2, 64, &codes, &[127; 4], &mut output[1..69]).unwrap();
        assert_eq!((output[0], output[69]), (0xa5, 0xa5));
    }

    #[test]
    fn tiny_native_f32_dot() {
        let codes = [0x32; 32]; // alternating 1 and 1.5
        let mut input = [1.0; 64];
        assert_eq!(
            row_dot_native_f32(&codes, &[127, 128], &input).unwrap(),
            120.0
        );
        input[0] = 16_777_216.0;
        input[1] = 1.0;
        input[2] = -16_777_216.0;
        let mut expected = 0.0_f32;
        for (i, x) in input.iter().enumerate() {
            let weight = (if i % 2 == 0 { 1.0 } else { 1.5 }) * (if i < 32 { 1.0 } else { 2.0 });
            expected += weight * x;
        }
        assert_eq!(
            row_dot_native_f32(&codes, &[127, 128], &input).unwrap(),
            expected
        );
        input[63] = f32::NAN;
        assert_eq!(
            row_dot_native_f32(&codes, &[127, 128], &input),
            Err(Error::NonFinite)
        );
    }
}
