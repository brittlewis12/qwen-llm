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
}

/// Dequantize the raw `bytes` of a `desc` tensor into a fresh `Vec<f32>`.
///
/// This calls `ggml_get_type_traits(dtype).to_float(bytes, dst, n)`, which
/// covers F32 / F16 / BF16 / Q*_0 / Q*_1 / Q*_K / IQ* / MXFP4 — everything
/// llama.cpp ships.
pub fn dequant_to_f32(desc: &TensorDesc, bytes: &[u8]) -> Result<Vec<f32>, CodecError> {
    let n_u64 = desc
        .checked_n_elements()
        .ok_or_else(|| CodecError::SizeOverflow {
            name: desc.name.clone(),
        })?;
    let n = usize::try_from(n_u64).map_err(|_| CodecError::SizeOverflow {
        name: desc.name.clone(),
    })?;

    // Universal length guard: `to_float(src, dst, n)` reads `desc.n_bytes`
    // from `src` with no FFI-side bounds check. Callers that hand us a
    // sub-slice (e.g. forward.rs splitting a packed tensor) are the
    // realistic mismatch source; mmap-wide callers will pass-through.
    let expected = usize::try_from(desc.n_bytes).map_err(|_| CodecError::SizeOverflow {
        name: desc.name.clone(),
    })?;
    if bytes.len() != expected {
        return Err(CodecError::SizeMismatch {
            got: bytes.len(),
            expected,
        });
    }

    // Fast path: F32 — no codec call needed.
    if desc.dtype == GgmlType::F32 {
        // SAFETY: alignment of f32 is 4; mmap pages are page-aligned, but
        // `bytes` may not be — copy out via byte-wise read.
        let mut out = Vec::<f32>::with_capacity(n);
        let copy_bytes =
            n.checked_mul(std::mem::size_of::<f32>())
                .ok_or_else(|| CodecError::SizeOverflow {
                    name: desc.name.clone(),
                })?;
        if expected != copy_bytes {
            return Err(CodecError::SizeMismatch {
                got: expected,
                expected: copy_bytes,
            });
        }
        // SAFETY: we just allocated `n` slots; size guard above proves
        // `bytes.len() == n * size_of::<f32>()` for F32 dtype.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, copy_bytes);
            out.set_len(n);
        }
        return Ok(out);
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

    let mut out = vec![0.0f32; n];
    // SAFETY: `to_float(src, dst, n_elements)` reads `desc.n_bytes` from
    // `bytes` and writes `n` f32s to `out`. The universal length guard
    // above ensures `bytes.len() == desc.n_bytes` for this (shape, dtype).
    unsafe {
        to_float(
            bytes.as_ptr() as *const std::ffi::c_void,
            out.as_mut_ptr(),
            n as i64,
        );
    }
    Ok(out)
}
