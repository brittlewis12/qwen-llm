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
}

/// Dequantize the raw `bytes` of a `desc` tensor into a fresh `Vec<f32>`.
///
/// This calls `ggml_get_type_traits(dtype).to_float(bytes, dst, n)`, which
/// covers F32 / F16 / BF16 / Q*_0 / Q*_1 / Q*_K / IQ* / MXFP4 — everything
/// llama.cpp ships.
pub fn dequant_to_f32(desc: &TensorDesc, bytes: &[u8]) -> Result<Vec<f32>, CodecError> {
    let n = desc.n_elements() as usize;

    // Fast path: F32 — no codec call needed.
    if desc.dtype == GgmlType::F32 {
        if bytes.len() != n * std::mem::size_of::<f32>() {
            return Err(CodecError::SizeMismatch {
                got: bytes.len(),
                expected: n * std::mem::size_of::<f32>(),
            });
        }
        // SAFETY: alignment of f32 is 4; mmap pages are page-aligned, but
        // `bytes` may not be — copy out via byte-wise read.
        let mut out = Vec::<f32>::with_capacity(n);
        // SAFETY: we just allocated `n` slots.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                out.as_mut_ptr() as *mut u8,
                n * std::mem::size_of::<f32>(),
            );
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
    // `bytes` and writes `n` f32s to `out`. `bytes.len()` is checked at
    // mmap slice time to equal `desc.n_bytes`.
    unsafe {
        to_float(
            bytes.as_ptr() as *const std::ffi::c_void,
            out.as_mut_ptr(),
            n as i64,
        );
    }
    Ok(out)
}
