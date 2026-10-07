//! Tooling-only CPU dequantization via llama.cpp's `ggml_get_type_traits`
//! seam. This is the same path ResponseForest uses to walk every quant
//! format on disk.
//!
//! **Not for the GPU hot path.** The hot path consumes K-quant blocks in
//! place via lifted Metal kernels (`kernel_mul_mv_q4_K_f32_impl` etc.).
//! See `docs/PLAN.md` weight-format role table.
//!
//! Used for:
//! * GGUF metadata sanity checks (read a tensor, dump its histogram).
//! * Re-pack-on-load (v2): produce f32 source bytes, then re-pack into a
//!   tighter Metal tile layout.
//! * Diagnostic tools that want fp32 host memory.
//! * Pieces we may want to keep in fp32 (e.g. embedding rows for diagnostic
//!   runs).

use crate::tensor::{GgmlType, TensorDesc};

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("no type traits for ggml type {0}")]
    NoTraits(i32),
    #[error("no to_float dequantization function for ggml type {0}")]
    NoToFloat(i32),
    #[error("linked GGML codec storage layout disagrees with {0:?}")]
    TraitsLayoutMismatch(GgmlType),
    #[error("byte length {got} does not match expected {expected} for shape × dtype")]
    SizeMismatch { got: usize, expected: usize },
    #[error("tensor size overflows host usize for {name:?}")]
    SizeOverflow { name: String },
    #[error(
        "tensor {name:?} row width {row_elements} is not divisible by {block_elements} elements for {dtype:?}"
    )]
    InvalidBlockGeometry {
        name: String,
        dtype: GgmlType,
        row_elements: u64,
        block_elements: u64,
    },
    #[error("failed to allocate {bytes} output bytes for tensor {name:?}")]
    AllocationFailed { name: String, bytes: usize },
    #[error("output length {got} does not match expected {expected} elements")]
    OutputLengthMismatch { got: usize, expected: usize },
}

#[derive(Clone, Copy)]
struct DequantPlan {
    elements: usize,
    source_bytes: usize,
}

const MAX_CODEC_CALL_ELEMENTS: usize = 65_536;

fn codec_chunk_layout(dtype: GgmlType) -> Result<(usize, usize), CodecError> {
    let (elements, bytes) = dtype
        .storage_layout()
        .ok_or(CodecError::NoTraits(dtype as i32))?;
    let blocks = MAX_CODEC_CALL_ELEMENTS / elements as usize;
    if blocks == 0 {
        return Err(CodecError::NoTraits(dtype as i32));
    }
    Ok((blocks * elements as usize, blocks * bytes as usize))
}

fn try_uninit_f32(desc: &TensorDesc, n: usize) -> Result<Vec<f32>, CodecError> {
    let bytes =
        n.checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CodecError::SizeOverflow {
                name: desc.name.clone(),
            })?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(n)
        .map_err(|_| CodecError::AllocationFailed {
            name: desc.name.clone(),
            bytes,
        })?;
    Ok(output)
}

fn validate_dequant(desc: &TensorDesc, bytes: &[u8]) -> Result<DequantPlan, CodecError> {
    let n_u64 = desc
        .checked_n_elements()
        .ok_or_else(|| CodecError::SizeOverflow {
            name: desc.name.clone(),
        })?;
    let n = usize::try_from(n_u64).map_err(|_| CodecError::SizeOverflow {
        name: desc.name.clone(),
    })?;

    let (block_elements, block_bytes) = desc
        .dtype
        .storage_layout()
        .ok_or(CodecError::NoTraits(desc.dtype as i32))?;
    let row_elements = desc.shape.first().copied().unwrap_or(n_u64);
    if block_elements == 0 || !row_elements.is_multiple_of(block_elements) {
        return Err(CodecError::InvalidBlockGeometry {
            name: desc.name.clone(),
            dtype: desc.dtype,
            row_elements,
            block_elements,
        });
    }
    let layout_bytes_u64 = n_u64
        .checked_div(block_elements)
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| CodecError::SizeOverflow {
            name: desc.name.clone(),
        })?;
    let layout_bytes = usize::try_from(layout_bytes_u64).map_err(|_| CodecError::SizeOverflow {
        name: desc.name.clone(),
    })?;

    // Universal length guard: `to_float(src, dst, n)` reads `desc.n_bytes`
    // from `src` with no FFI-side bounds check. Callers that hand us a
    // sub-slice (e.g. forward.rs splitting a packed tensor) are the
    // realistic mismatch source; mmap-wide callers will pass-through.
    let expected = usize::try_from(desc.n_bytes).map_err(|_| CodecError::SizeOverflow {
        name: desc.name.clone(),
    })?;
    if expected != layout_bytes {
        return Err(CodecError::SizeMismatch {
            got: expected,
            expected: layout_bytes,
        });
    }
    if bytes.len() != expected {
        return Err(CodecError::SizeMismatch {
            got: bytes.len(),
            expected,
        });
    }
    Ok(DequantPlan {
        elements: n,
        source_bytes: expected,
    })
}

/// Validate layout and codec availability without allocating or decoding weights.
pub fn validate_dequantization(desc: &TensorDesc, bytes: &[u8]) -> Result<(), CodecError> {
    validate_dequant(desc, bytes)?;
    if desc.dtype != GgmlType::F32 && codec_traits(desc.dtype)?.to_float.is_none() {
        return Err(CodecError::NoToFloat(desc.dtype as i32));
    }
    Ok(())
}

fn codec_traits(dtype: GgmlType) -> Result<&'static llama_cpp_sys_2::ggml_type_traits, CodecError> {
    let raw = dtype as i32;
    // GGML asserts the enum bound before indexing its table; it is not a
    // nullable lookup for unknown or newer wire types.
    if raw < 0 || raw as u32 >= llama_cpp_sys_2::GGML_TYPE_COUNT {
        return Err(CodecError::NoTraits(raw));
    }
    // SAFETY: the checked tag indexes GGML's immutable static trait table.
    let traits = unsafe { llama_cpp_sys_2::ggml_get_type_traits(raw as u32).as_ref() }
        .ok_or(CodecError::NoTraits(raw))?;
    if dtype.storage_layout() != Some((traits.blck_size as u64, traits.type_size as u64)) {
        return Err(CodecError::TraitsLayoutMismatch(dtype));
    }
    Ok(traits)
}

fn dequant_validated_into(
    desc: &TensorDesc,
    bytes: &[u8],
    plan: DequantPlan,
    output: &mut [std::mem::MaybeUninit<f32>],
) -> Result<(), CodecError> {
    if output.len() != plan.elements {
        return Err(CodecError::OutputLengthMismatch {
            got: output.len(),
            expected: plan.elements,
        });
    }

    // Fast path: F32 — no codec call needed.
    if desc.dtype == GgmlType::F32 {
        let copy_bytes = plan
            .elements
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| CodecError::SizeOverflow {
                name: desc.name.clone(),
            })?;
        if plan.source_bytes != copy_bytes {
            return Err(CodecError::SizeMismatch {
                got: plan.source_bytes,
                expected: copy_bytes,
            });
        }
        // SAFETY: validation proves equal complete source and destination byte
        // spans. Byte-wise copy does not require the source to be f32-aligned.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                output.as_mut_ptr().cast::<u8>(),
                copy_bytes,
            );
        }
        return Ok(());
    }

    let raw_dtype = desc.dtype as i32;
    let to_float = codec_traits(desc.dtype)?
        .to_float
        .ok_or(CodecError::NoToFloat(raw_dtype))?;

    // GGML accepts i64 lengths, but several codecs use signed int indexing.
    // Bound individual calls, not the tensor, and keep whole storage blocks.
    let (chunk_elements, chunk_bytes) = codec_chunk_layout(desc.dtype)?;
    // &[u8] promises no alignment. GGML dereferences typed block pointers;
    // an explicitly aligned, bounded staging buffer handles such callers.
    let mut staging = Vec::<u64>::new();
    if bytes.as_ptr().align_offset(std::mem::align_of::<u64>()) != 0
        || !chunk_bytes.is_multiple_of(std::mem::align_of::<u64>())
    {
        let words = chunk_bytes.min(bytes.len()).div_ceil(8);
        staging
            .try_reserve_exact(words)
            .map_err(|_| CodecError::AllocationFailed {
                name: desc.name.clone(),
                bytes: words * 8,
            })?;
        staging.resize(words, 0);
    }
    for (source, destination) in bytes
        .chunks(chunk_bytes)
        .zip(output.chunks_mut(chunk_elements))
    {
        let source = if staging.is_empty() {
            source
        } else {
            let aligned = bytemuck::cast_slice_mut::<u64, u8>(&mut staging);
            aligned[..source.len()].copy_from_slice(source);
            &aligned[..source.len()]
        };
        // SAFETY: both spans are complete validated blocks, source alignment
        // is at least eight (sufficient for the admitted GGML block structs),
        // and each call's element count and signed indexing fit i32.
        unsafe {
            to_float(
                source.as_ptr().cast(),
                destination.as_mut_ptr().cast::<f32>(),
                destination.len() as i64,
            );
        }
    }
    Ok(())
}

/// Fill an exactly-sized uninitialized F32 destination from a tensor payload.
///
/// On success every destination element is initialized. All validation and
/// fallible work happens before the producer writes, so an error exposes no
/// partially initialized output.
pub(crate) fn dequant_to_f32_into(
    desc: &TensorDesc,
    bytes: &[u8],
    output: &mut [std::mem::MaybeUninit<f32>],
) -> Result<(), CodecError> {
    let plan = validate_dequant(desc, bytes)?;
    dequant_validated_into(desc, bytes, plan, output)
}

/// Overwrite an initialized F32 destination from an exactly-sized tensor
/// payload without allocating an intermediate vector.
pub(crate) fn dequant_to_f32_in_place(
    desc: &TensorDesc,
    bytes: &[u8],
    output: &mut [f32],
) -> Result<(), CodecError> {
    let plan = validate_dequant(desc, bytes)?;
    let output = unsafe {
        std::slice::from_raw_parts_mut(
            output.as_mut_ptr().cast::<std::mem::MaybeUninit<f32>>(),
            output.len(),
        )
    };
    dequant_validated_into(desc, bytes, plan, output)
}

/// Dequantize the raw `bytes` of a `desc` tensor into a fresh `Vec<f32>`.
///
/// This calls `ggml_get_type_traits(dtype).to_float(bytes, dst, n)`, which
/// uses the pinned GGML codecs, including Q1_0, Q2_0, NVFP4 and ternary
/// storage. Codec availability does not imply native Metal execution support.
pub fn dequant_to_f32(desc: &TensorDesc, bytes: &[u8]) -> Result<Vec<f32>, CodecError> {
    let plan = validate_dequant(desc, bytes)?;
    let mut out = try_uninit_f32(desc, plan.elements)?;
    let output = unsafe {
        std::slice::from_raw_parts_mut(
            out.as_mut_ptr().cast::<std::mem::MaybeUninit<f32>>(),
            plan.elements,
        )
    };
    dequant_validated_into(desc, bytes, plan, output)?;
    // SAFETY: the producer returned successfully after initializing every slot.
    unsafe {
        out.set_len(plan.elements);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(name: &str, shape: Vec<u64>, dtype: GgmlType, n_bytes: usize) -> TensorDesc {
        TensorDesc {
            name: name.to_string(),
            shape,
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: n_bytes as u64,
        }
    }

    #[test]
    fn extended_quant_codecs_match_linked_storage_layouts() {
        for dtype in [
            GgmlType::TQ1_0,
            GgmlType::TQ2_0,
            GgmlType::NVFP4,
            GgmlType::Q1_0,
            GgmlType::Q2_0,
        ] {
            let (elements, bytes) = dtype.storage_layout().unwrap();
            let tensor = desc("extended", vec![elements], dtype, bytes as usize);
            let payload = vec![0; bytes as usize];
            validate_dequantization(&tensor, &payload).unwrap();
            let values = dequant_to_f32(&tensor, &payload).unwrap();
            assert_eq!(values.len(), elements as usize);
            assert!(values.iter().all(|v| v.is_finite() && *v == 0.0), "{dtype}");
        }
        assert!(matches!(
            codec_traits(GgmlType::Unknown),
            Err(CodecError::NoTraits(-1))
        ));
    }

    #[test]
    fn q2_0_codec_matches_independent_packing_at_production_row_width() {
        // 640 inputs = ten blocks, not a whole number of K-quant blocks.
        let scales = [0.5f32, -2.0, 0.0, -0.0, 0.03125];
        let mut payload = Vec::new();
        let mut expected = Vec::new();
        for block in 0..1030 {
            let scale = scales[block % scales.len()];
            payload.extend_from_slice(&half::f16::from_f32(scale).to_le_bytes());
            for byte in 0..16 {
                // Vary all four code positions independently, across rows too.
                let packed = (block * 73 + byte * 29) as u8;
                payload.push(packed);
                for divisor in [1u16, 4, 16, 64] {
                    let code = (u16::from(packed) / divisor) % 4;
                    expected.push((code as f32 - 1.0) * scale);
                }
            }
        }
        let tensor = desc("q2_0", vec![640, 103], GgmlType::Q2_0, payload.len());
        // Cross a codec-call boundary from a deliberately odd source address.
        let mut storage = vec![0u64; (payload.len() + 1).div_ceil(8)];
        let unaligned = &mut bytemuck::cast_slice_mut::<u64, u8>(&mut storage)[1..=payload.len()];
        unaligned.copy_from_slice(&payload);
        let actual = dequant_to_f32(&tensor, unaligned).unwrap();
        assert_eq!(actual.len(), expected.len());
        for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(actual.to_bits(), expected.to_bits(), "element {index}");
        }
        let mut in_place = vec![f32::NAN; expected.len()];
        let mut aligned_storage = vec![0u64; payload.len().div_ceil(8)];
        let aligned =
            &mut bytemuck::cast_slice_mut::<u64, u8>(&mut aligned_storage)[..payload.len()];
        aligned.copy_from_slice(&payload);
        dequant_to_f32_in_place(&tensor, aligned, &mut in_place).unwrap();
        assert!(
            in_place
                .iter()
                .zip(&actual)
                .all(|(a, b)| a.to_bits() == b.to_bits())
        );
        assert!(dequant_to_f32(&tensor, &payload[..payload.len() - 1]).is_err());
        let bad_row = desc("q2_bad_row", vec![32, 2], GgmlType::Q2_0, 18);
        assert!(matches!(
            dequant_to_f32(&bad_row, &[0; 18]),
            Err(CodecError::InvalidBlockGeometry { .. })
        ));
    }

    #[test]
    fn codec_calls_are_bounded_aligned_whole_blocks() {
        for raw in 0..llama_cpp_sys_2::GGML_TYPE_COUNT {
            let dtype = GgmlType::from_raw(raw);
            if dtype == GgmlType::Unknown {
                continue;
            }
            let (block, bytes) = dtype.storage_layout().unwrap();
            let (elements, source_bytes) = codec_chunk_layout(dtype).unwrap();
            assert!(elements > 0 && elements <= MAX_CODEC_CALL_ELEMENTS);
            assert!(elements <= i32::MAX as usize);
            assert_eq!(elements as u64 % block, 0);
            assert_eq!(source_bytes as u64, elements as u64 / block * bytes);
            assert_eq!(source_bytes % 8, 0);
        }
    }

    #[test]
    fn q1_0_codec_preserves_bit_order_and_signed_scale() {
        let mut payload = Vec::new();
        payload.extend_from_slice(&half::f16::from_f32(-0.5).to_le_bytes());
        payload.extend(0u8..16);
        let tensor = desc("q1_0", vec![128], GgmlType::Q1_0, 18);
        let actual = dequant_to_f32(&tensor, &payload).unwrap();
        for (i, value) in actual.iter().enumerate() {
            let expected = if ((i / 8) >> (i % 8)) & 1 == 1 {
                -0.5
            } else {
                0.5
            };
            assert_eq!(*value, expected, "element {i}");
        }
    }

    #[test]
    fn nvfp4_codec_uses_four_scales_and_subblock_nibble_order() {
        // UE4M3 scales 1, 2, 1/512, and the GGML zero sentinel.
        let mut payload = vec![0x38, 0x40, 0x01, 0x7f];
        for _ in 0..4 {
            payload.extend((0u8..8).map(|q| q | ((q + 8) << 4)));
        }
        let tensor = desc("nvfp4", vec![64], GgmlType::NVFP4, 36);
        let actual = dequant_to_f32(&tensor, &payload).unwrap();
        let positive = [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (sub, scale) in [1.0, 2.0, 1.0 / 512.0, 0.0].into_iter().enumerate() {
            for (i, value) in positive.into_iter().enumerate() {
                assert_eq!(actual[sub * 16 + i], value * scale);
                assert_eq!(actual[sub * 16 + 8 + i], -value * scale);
            }
        }
    }

    #[test]
    fn f32_copy_commits_exact_output() {
        let values = [1.25f32, -0.0, f32::INFINITY, f32::from_bits(0x7fc0_1234)];
        let tensor = desc("f32", vec![values.len() as u64], GgmlType::F32, 16);
        let decoded = dequant_to_f32(&tensor, bytemuck::cast_slice(&values)).unwrap();
        assert_eq!(decoded.len(), values.len());
        assert!(
            decoded
                .iter()
                .zip(values)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }

    #[test]
    fn q8_0_dequant_fills_every_reserved_element() {
        let scale = half::f16::from_f32(0.25).to_bits().to_le_bytes();
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&scale);
        for (index, byte) in block[2..].iter_mut().enumerate() {
            *byte = (index as i8 - 16) as u8;
        }
        let tensor = desc("q8", vec![32], GgmlType::Q8_0, block.len());
        let decoded = dequant_to_f32(&tensor, &block).unwrap();
        assert_eq!(decoded.len(), 32);
        for (index, value) in decoded.iter().enumerate() {
            assert_eq!(value.to_bits(), ((index as f32 - 16.0) * 0.25).to_bits());
        }
    }

    #[test]
    fn q8_0_dequant_into_initializes_exact_destination() {
        let scale = half::f16::from_f32(0.5).to_bits().to_le_bytes();
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&scale);
        for (index, byte) in block[2..].iter_mut().enumerate() {
            *byte = (index as i8 - 8) as u8;
        }
        let tensor = desc("q8-into", vec![32], GgmlType::Q8_0, block.len());
        let mut output = [std::mem::MaybeUninit::<f32>::uninit(); 32];
        dequant_to_f32_into(&tensor, &block, &mut output).unwrap();
        let output = unsafe { &*(&output as *const _ as *const [f32; 32]) };
        for (index, value) in output.iter().enumerate() {
            assert_eq!(value.to_bits(), ((index as f32 - 8.0) * 0.5).to_bits());
        }

        let mut short = [std::mem::MaybeUninit::<f32>::uninit(); 31];
        assert!(matches!(
            dequant_to_f32_into(&tensor, &block, &mut short),
            Err(CodecError::OutputLengthMismatch {
                got: 31,
                expected: 32
            })
        ));
    }

    #[test]
    fn q8_0_dequant_in_place_overwrites_exact_destination() {
        let scale = half::f16::from_f32(0.125).to_bits().to_le_bytes();
        let mut block = [0u8; 34];
        block[..2].copy_from_slice(&scale);
        for (index, byte) in block[2..].iter_mut().enumerate() {
            *byte = (index as i8 - 4) as u8;
        }
        let tensor = desc("q8-in-place", vec![32], GgmlType::Q8_0, block.len());
        let mut output = [f32::NAN; 32];
        dequant_to_f32_in_place(&tensor, &block, &mut output).unwrap();
        for (index, value) in output.iter().enumerate() {
            assert_eq!(value.to_bits(), ((index as f32 - 4.0) * 0.125).to_bits());
        }

        let mut short = [0.0f32; 31];
        assert!(matches!(
            dequant_to_f32_in_place(&tensor, &block, &mut short),
            Err(CodecError::OutputLengthMismatch {
                got: 31,
                expected: 32
            })
        ));
    }

    #[test]
    fn q4_k_known_block_matches_reference_formula() {
        let scales = [1u8, 3, 5, 7, 9, 17, 33, 49];
        let mins = [2u8, 4, 6, 8, 10, 18, 34, 50];
        let mut words = [0u32; 36];
        let block = bytemuck::cast_slice_mut::<u32, u8>(&mut words);
        block[..2].copy_from_slice(&half::f16::from_f32(0.5).to_bits().to_le_bytes());
        block[2..4].copy_from_slice(&half::f16::from_f32(0.25).to_bits().to_le_bytes());
        for j in 0..4 {
            block[4 + j] = scales[j] | ((scales[j + 4] >> 4) << 6);
            block[8 + j] = mins[j] | ((mins[j + 4] >> 4) << 6);
            block[12 + j] = (scales[j + 4] & 0x0f) | ((mins[j + 4] & 0x0f) << 4);
        }
        let mut expected = [0.0f32; 256];
        for chunk in 0..4 {
            let even_group = chunk * 2;
            let odd_group = even_group + 1;
            for lane in 0..32 {
                let low = ((chunk * 3 + lane) & 0x0f) as u8;
                let high = ((15 + chunk * 5 - (lane & 0x0f)) & 0x0f) as u8;
                block[16 + chunk * 32 + lane] = low | (high << 4);
                expected[chunk * 64 + lane] =
                    0.5 * scales[even_group] as f32 * low as f32 - 0.25 * mins[even_group] as f32;
                expected[chunk * 64 + 32 + lane] =
                    0.5 * scales[odd_group] as f32 * high as f32 - 0.25 * mins[odd_group] as f32;
            }
        }

        let tensor = desc("q4-k-known", vec![256], GgmlType::Q4_K, block.len());
        let mut actual = [f32::NAN; 256];
        dequant_to_f32_in_place(&tensor, block, &mut actual).unwrap();
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.to_bits() == expected.to_bits())
        );
    }

    #[test]
    fn quantized_rows_must_be_block_aligned() {
        let tensor = desc("bad-row", vec![16, 2], GgmlType::Q8_0, 34);
        let error = dequant_to_f32(&tensor, &[0u8; 34]).unwrap_err();
        assert!(matches!(
            error,
            CodecError::InvalidBlockGeometry {
                row_elements: 16,
                block_elements: 32,
                ..
            }
        ));
    }

    #[test]
    fn descriptor_bytes_must_match_storage_layout() {
        let tensor = desc("bad-size", vec![32], GgmlType::Q8_0, 35);
        let error = dequant_to_f32(&tensor, &[0u8; 35]).unwrap_err();
        assert!(matches!(
            error,
            CodecError::SizeMismatch {
                got: 35,
                expected: 34
            }
        ));
    }
}
