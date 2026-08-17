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

    // SAFETY: `ggml_get_type_traits` is read-only and idempotent. The
    // returned pointer is a static-lifetime table per the ggml API; null
    // means "no traits registered for this type."
    let raw_dtype = desc.dtype as i32;
    let traits = unsafe { llama_cpp_sys_2::ggml_get_type_traits(raw_dtype as u32) };
    if traits.is_null() {
        return Err(CodecError::NoTraits(raw_dtype));
    }
    // SAFETY: traits is non-null per the check above. `to_float` is an
    // optional function pointer.
    let to_float = unsafe { (*traits).to_float }.ok_or(CodecError::NoToFloat(raw_dtype))?;

    let n_i64 = i64::try_from(plan.elements).map_err(|_| CodecError::SizeOverflow {
        name: desc.name.clone(),
    })?;
    // SAFETY: `to_float(src, dst, n_elements)` reads `desc.n_bytes` from
    // `bytes` and writes every output f32. The checked storage
    // geometry and universal length guard prove that both spans are complete.
    unsafe {
        to_float(
            bytes.as_ptr() as *const std::ffi::c_void,
            output.as_mut_ptr().cast::<f32>(),
            n_i64,
        );
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

/// Dequantize the raw `bytes` of a `desc` tensor into a fresh `Vec<f32>`.
///
/// This calls `ggml_get_type_traits(dtype).to_float(bytes, dst, n)`, which
/// covers F32 / F16 / BF16 / Q*_0 / Q*_1 / Q*_K / IQ* / MXFP4 — everything
/// llama.cpp ships.
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
