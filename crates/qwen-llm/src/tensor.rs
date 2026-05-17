//! Backend-agnostic tensor descriptor.
//!
//! A `TensorDesc` is a view into one mmap'd GGUF shard: name + shape + ggml
//! type tag + shard index + byte offset. Nothing is copied at load time; the
//! file shard *is* the resident model storage.

use std::fmt;

/// Mirrors `enum ggml_type` in `ggml.h`. Kept as a numeric tag so the wider
/// codebase doesn't need `llama-cpp-sys-2` in its public API.
///
/// Variant names track the canonical ggml spelling (`Q4_K`, `IQ4_XS`) on
/// purpose — keep diffability with llama.cpp source higher-priority than
/// Rust's `UpperCamelCase` lint.
#[allow(non_camel_case_types)]
#[repr(i32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    BF16 = 30,
    MXFP4 = 39,
    Unknown = -1,
}

impl GgmlType {
    pub fn from_raw(raw: u32) -> Self {
        match raw as i32 {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2_K,
            11 => Self::Q3_K,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            15 => Self::Q8_K,
            16 => Self::IQ2_XXS,
            17 => Self::IQ2_XS,
            18 => Self::IQ3_XXS,
            19 => Self::IQ1_S,
            20 => Self::IQ4_NL,
            21 => Self::IQ3_S,
            22 => Self::IQ2_S,
            23 => Self::IQ4_XS,
            30 => Self::BF16,
            39 => Self::MXFP4,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for GgmlType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

/// One tensor's descriptor in the GGUF file.
///
/// `shard_idx` selects the mmap'd GGUF shard inside `GgufFile`. `data_offset`
/// is the absolute byte offset from the start of that shard's mmap.
/// `n_bytes` is `ggml_nbytes()` for the (shape, type) pair.
#[derive(Debug, Clone)]
pub struct TensorDesc {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub shard_idx: usize,
    pub data_offset: u64,
    pub n_bytes: u64,
}

impl TensorDesc {
    pub fn n_elements(&self) -> u64 {
        self.shape.iter().product()
    }
}
