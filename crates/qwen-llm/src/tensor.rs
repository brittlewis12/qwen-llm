//! Backend-agnostic tensor descriptor.
//!
//! A `TensorDesc` is a view into one mmap'd GGUF shard: name + shape + ggml
//! type tag + shard index + byte offset. It describes on-disk GGUF storage;
//! GPU backends may still copy those bytes into backend-native buffers.

use std::fmt;

/// Checked product of a tensor shape.
///
/// Use this instead of `shape.iter().product()` for any file-derived or
/// metadata-derived shape. The standard iterator product silently wraps in
/// release builds for integer types.
pub fn checked_shape_elements(shape: &[u64]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1_u64, |acc, &dim| acc.checked_mul(dim))
}

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

pub(crate) fn ggml_type_layout_raw(raw: u32) -> Option<(u64, u64)> {
    const K: u64 = 256;
    Some(match raw {
        0 => (1, 4),
        1 => (1, 2),
        2 => (32, 2 + 32 / 2),
        3 => (32, 2 + 2 + 32 / 2),
        4 | 5 => (0, 0),
        6 => (32, 2 + 4 + 32 / 2),
        7 => (32, 2 + 2 + 4 + 32 / 2),
        8 => (32, 2 + 32),
        9 => (32, 4 + 4 + 32),
        10 => (K, K / 16 + K / 4 + 2 + 2),
        11 => (K, K / 8 + K / 4 + 12 + 2),
        12 => (K, 2 + 2 + 12 + K / 2),
        13 => (K, 2 + 2 + 12 + K / 8 + K / 2),
        14 => (K, K / 2 + K / 4 + K / 16 + 2),
        15 => (K, 4 + K + K / 16 * 2),
        16 => (K, 2 + K / 8 * 2),
        17 => (K, 2 + K / 8 * 2 + K / 32),
        18 => (K, 2 + 3 * (K / 8)),
        19 => (K, 2 + K / 8 + K / 16),
        20 => (32, 2 + 16),
        21 => (K, 2 + 13 * (K / 32) + K / 64),
        22 => (K, 2 + K / 4 + K / 16),
        23 => (K, 2 + 2 + K / 64 + K / 2),
        24 => (1, 1),
        25 => (1, 2),
        26 => (1, 4),
        27 => (1, 8),
        28 => (1, 8),
        29 => (K, K / 8 + K / 16 + K / 32),
        30 => (1, 2),
        31..=33 => (0, 0),
        34 => (K, 2 + K / 64 + (K - 4 * (K / 64)) / 5),
        35 => (K, 2 + K / 4),
        36..=38 => (0, 0),
        39 => (32, 17),
        _ => return None,
    })
}

pub(crate) fn ggml_type_layout(dtype: GgmlType) -> Option<(u64, u64)> {
    match dtype {
        GgmlType::Unknown => None,
        _ => ggml_type_layout_raw(dtype as u32),
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
    pub fn checked_n_elements(&self) -> Option<u64> {
        checked_shape_elements(&self.shape)
    }

    pub fn n_elements(&self) -> u64 {
        self.checked_n_elements()
            .expect("TensorDesc shape element count overflow")
    }
}

#[cfg(test)]
mod tests {
    use super::{GgmlType, ggml_type_layout, ggml_type_layout_raw};

    #[test]
    fn ggml_scalar_and_extended_layouts_match_authoritative_values() {
        let cases = [
            (24, (1, 1)),
            (25, (1, 2)),
            (26, (1, 4)),
            (27, (1, 8)),
            (28, (1, 8)),
            (29, (256, 56)),
            (30, (1, 2)),
            (34, (256, 2 + 4 + (256 - 16) / 5)),
            (35, (256, 66)),
            (39, (32, 17)),
        ];
        for (raw, layout) in cases {
            assert_eq!(ggml_type_layout_raw(raw), Some(layout), "raw type {raw}");
        }
    }

    #[test]
    fn ggml_enum_layouts_match_raw_table() {
        assert_eq!(ggml_type_layout(GgmlType::BF16), ggml_type_layout_raw(30));
        assert_eq!(ggml_type_layout(GgmlType::MXFP4), ggml_type_layout_raw(39));
        assert_eq!(ggml_type_layout(GgmlType::F32), ggml_type_layout_raw(0));
        assert_eq!(ggml_type_layout(GgmlType::Unknown), None);
    }
}
