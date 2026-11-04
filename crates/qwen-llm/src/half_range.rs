//! Load-time range guard for weights that enter half-staged kernels.
//!
//! Several production prefill paths narrow decoded weights to F16 inside the
//! kernel (`qt_dequantize_*`, the llama.cpp-style `_mm` dense tiles, grouped
//! IQ/K-quant tiles, GLM half rows, the V4.1 Q5_K -> F16 cache, and F32/BF16
//! generic-tile banks). A decoded value at or above 2^15 is outside the
//! conservative guard band even though half itself only overflows at 65520;
//! the extra binade keeps the check independent of rounding mode.
//!
//! This module computes one bound per tensor from the GGUF mapping:
//!
//! * MXFP4: the E8M0 exponent byte, as a histogram. The largest code is 12,
//!   so a block bound is `12 * 2^(e-128)`.
//! * K-quant / legacy / I-quant: the per-block scale fields (`d`, `dmin`, `m`)
//!   with the format's maximum code magnitude. Only the header fields are
//!   touched; the quant payload is not decoded.
//! * F32 / F16 / BF16: the full tensor max (these are used as weight banks by
//!   the generic half tile, so there is no header shortcut).
//! * RMSNorm weights: `sqrt(n) * max|gamma|`, because a norm output feeding a
//!   half-staged site is bounded by that quantity.
//!
//! The scan is cached per artifact identity (shard device/inode/size/mtime/
//! ctime plus the full tensor descriptor table). A cache hit costs one small
//! JSON read. The first scan is CPU-only and parallel over tensors; the loader
//! has already mapped the shards, so no second copy is made.
//!
//! Guard policy: an unsafe tensor is allowed only when the load path already
//! selects an F32-operand lane for it (currently DS4 MXFP4 routed down). Every
//! other unsafe tensor fails with [`HalfRangeError::UnsafeTensor`] naming the
//! tensor, dtype, bound and limit.

use crate::gguf::{GgufError, GgufFile};
use crate::tensor::{GgmlType, TensorDesc};
use half::f16;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Decoded weights entering a half-staged path must stay below this bound.
pub const HALF_DECODED_MAX: f32 = 32_768.0;
/// RMSNorm output bound (`sqrt(n) * max|gamma|`) for half-staged consumers.
pub const RMS_NORM_MAX: f32 = 2_048.0;
/// Bump when the bound formulas or cache schema change.
pub const CENSUS_VERSION: u32 = 1;

/// Contiguous chunk for parallel in-tensor scans (1 MiB).
const SCAN_CHUNK_BYTES: usize = 1 << 20;

const CACHE_SCHEMA: &[u8] = b"qwen-half-range-census-v1\0";
const CACHE_DIR_ENV: &str = "QWEN_HALF_RANGE_CACHE_DIR";
const CACHE_ENV: &str = "QWEN_HALF_RANGE_CACHE";

/// How a tensor bound was obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeMethod {
    Mxfp4Exponent,
    BlockScale,
    FullPrecision,
}

/// Per-tensor decoded-magnitude bound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorRange {
    pub name: String,
    pub dtype: String,
    pub method: RangeMethod,
    /// Conservative bound on `max |decoded weight|`.
    pub max_abs: f32,
    pub blocks: u64,
    /// Nonzero MXFP4 exponent bytes (only for [`RangeMethod::Mxfp4Exponent`]).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mxfp4_e_hist: Option<BTreeMap<u8, u64>>,
    /// Number of non-finite scale fields or full-precision values.
    pub non_finite: u64,
}

/// Per-RMSNorm bound.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RmsNormRange {
    pub name: String,
    pub n: u64,
    pub max_abs: f32,
    /// `sqrt(n) * max_abs`.
    pub bound: f32,
}

/// One reason a tensor or norm cannot enter a half-staged path.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationKind {
    DecodedRange,
    NonFinite,
    RmsRange,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HalfRangeViolation {
    pub name: String,
    pub dtype: String,
    pub kind: ViolationKind,
    pub bound: f32,
    pub limit: f32,
    pub detail: String,
}

/// Cached result of one artifact census.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HalfRangeCensus {
    pub version: u32,
    pub tensors: BTreeMap<String, TensorRange>,
    pub norms: BTreeMap<String, RmsNormRange>,
    pub violations: Vec<HalfRangeViolation>,
    pub scanned_bytes: u64,
    #[serde(default)]
    pub scan_wall_ms: u64,
    #[serde(default)]
    pub cache_hit: bool,
}

impl HalfRangeCensus {
    pub fn tensor(&self, name: &str) -> Option<&TensorRange> {
        self.tensors.get(name)
    }

    pub fn scan_wall(&self) -> Duration {
        Duration::from_millis(self.scan_wall_ms)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum HalfRangeError {
    #[error("half-range census: {0}")]
    Gguf(#[from] GgufError),
    #[error(
        "half-range guard: tensor {name:?} ({dtype}) decoded max bound {bound:.1} is not below {limit:.0}; \
         this tensor enters a half-staged path and no F32-operand reroute is wired for it"
    )]
    UnsafeTensor {
        name: String,
        dtype: String,
        bound: f32,
        limit: f32,
    },
    #[error(
        "half-range guard: tensor {name:?} ({dtype}) has {count} non-finite decoded values/scales; \
         refusing to stage it as half"
    )]
    NonFiniteTensor {
        name: String,
        dtype: String,
        count: u64,
    },
    #[error(
        "half-range guard: RMSNorm {name:?} has sqrt({n})*max|gamma| = {bound:.1}, not below {limit:.0}; \
         its output feeds a half-staged path"
    )]
    UnsafeNorm {
        name: String,
        n: u64,
        bound: f32,
        limit: f32,
    },
}

/// Family-specific exceptions for tensors that already select an F32 lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HalfRangeScope {
    /// No automatic reroute; every unsafe tensor refuses.
    Generic,
    /// DS4: MXFP4 routed down is dispatched to `kernel_mat_mat_mxfp4_f32_*`
    /// or per-row F32 GEMV by default, so it never enters the half tile.
    DeepSeekV4,
}

impl HalfRangeScope {
    fn allows_f32_reroute(self, range: &TensorRange) -> bool {
        match self {
            Self::Generic => false,
            Self::DeepSeekV4 => {
                range.dtype == GgmlType::MXFP4.wire_name() && is_routed_down_name(&range.name)
            }
        }
    }
}

fn is_routed_down_name(name: &str) -> bool {
    name.ends_with(".down_exps.weight")
        || name.ends_with(".ffn_down_exps.weight")
        || name.ends_with(".w2.weight")
        || name.ends_with(".down.weight")
}

/// Compute (or load from cache) the census and refuse unsafe tensors.
pub fn audit_or_refuse(
    gguf: &GgufFile,
    scope: HalfRangeScope,
) -> Result<HalfRangeCensus, HalfRangeError> {
    let census = cached_census(gguf)?;
    for violation in &census.violations {
        match violation.kind {
            ViolationKind::RmsRange => {
                return Err(HalfRangeError::UnsafeNorm {
                    name: violation.name.clone(),
                    n: census.norms.get(&violation.name).map(|n| n.n).unwrap_or(0),
                    bound: violation.bound,
                    limit: violation.limit,
                });
            }
            ViolationKind::NonFinite => {
                return Err(HalfRangeError::NonFiniteTensor {
                    name: violation.name.clone(),
                    dtype: violation.dtype.clone(),
                    count: census
                        .tensors
                        .get(&violation.name)
                        .map(|t| t.non_finite)
                        .unwrap_or(0),
                });
            }
            ViolationKind::DecodedRange => {
                if let Some(range) = census.tensors.get(&violation.name)
                    && scope.allows_f32_reroute(range)
                {
                    continue;
                }
                return Err(HalfRangeError::UnsafeTensor {
                    name: violation.name.clone(),
                    dtype: violation.dtype.clone(),
                    bound: violation.bound,
                    limit: violation.limit,
                });
            }
        }
    }
    Ok(census)
}

/// Cached census for diagnostics and cost measurement.
pub fn cached_census(gguf: &GgufFile) -> Result<HalfRangeCensus, GgufError> {
    let key = cache_key(gguf)?;
    let started = Instant::now();
    if let Some(root) = cache_root()
        && let Some(mut cached) = read_cache(&root, &key)
    {
        cached.cache_hit = true;
        cached.scan_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
        report_census(&cached);
        return Ok(cached);
    }
    let mut census = census_uncached(gguf)?;
    census.scan_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    census.cache_hit = false;
    if let Some(root) = cache_root() {
        write_cache(&root, &key, &census);
    }
    report_census(&census);
    Ok(census)
}

fn report_census(census: &HalfRangeCensus) {
    if std::env::var_os("QWEN_HALF_RANGE_REPORT").is_none() {
        return;
    }
    eprintln!(
        "half-range-census tensors={} norms={} bytes={} scan_ms={} cache_hit={} violations={}",
        census.tensors.len(),
        census.norms.len(),
        census.scanned_bytes,
        census.scan_wall_ms,
        census.cache_hit,
        census.violations.len()
    );
}

/// Uncached census. Used by tests and by `QWEN_HALF_RANGE_CACHE=0` runs.
pub fn census_uncached(gguf: &GgufFile) -> Result<HalfRangeCensus, GgufError> {
    // Read each shard in offset order. Tensor descriptors are not stored in
    // file order across a split model, and a random-order parallel scan turns
    // a cold artifact into a seek storm. Parallelism is per shard; each shard
    // is a sequential pass.
    for shard in &gguf.shards {
        let _ = shard.advise(memmap2::Advice::Sequential);
    }
    let mut by_shard: Vec<Vec<&TensorDesc>> = vec![Vec::new(); gguf.shards.len()];
    for desc in &gguf.tensors {
        let Some(group) = by_shard.get_mut(desc.shard_idx) else {
            return Err(GgufError::Decode(format!(
                "tensor {:?}: shard index {} is out of range",
                desc.name, desc.shard_idx
            )));
        };
        group.push(desc);
    }
    for group in &mut by_shard {
        group.sort_by_key(|desc| desc.data_offset);
    }
    let per_shard: Vec<Result<Vec<ScanOutput>, GgufError>> = by_shard
        .par_iter()
        .map(|descs| {
            let mut outputs = Vec::with_capacity(descs.len());
            for desc in descs {
                outputs.push(scan_tensor(gguf, desc)?);
            }
            Ok(outputs)
        })
        .collect();

    let mut tensors = BTreeMap::new();
    let mut norms = BTreeMap::new();
    let mut violations = Vec::new();
    let mut scanned_bytes = 0u64;

    for result in per_shard {
        for output in result? {
            scanned_bytes = scanned_bytes.saturating_add(output.scanned_bytes);
            if let Some(range) = output.range {
                if range.non_finite > 0 {
                    violations.push(HalfRangeViolation {
                        name: range.name.clone(),
                        dtype: range.dtype.clone(),
                        kind: ViolationKind::NonFinite,
                        bound: range.max_abs,
                        limit: HALF_DECODED_MAX,
                        detail: format!("{} non-finite decoded values/scales", range.non_finite),
                    });
                } else if !range.max_abs.is_finite() || range.max_abs >= HALF_DECODED_MAX {
                    violations.push(HalfRangeViolation {
                        name: range.name.clone(),
                        dtype: range.dtype.clone(),
                        kind: ViolationKind::DecodedRange,
                        bound: range.max_abs,
                        limit: HALF_DECODED_MAX,
                        detail: "decoded max bound is not below 2^15".to_string(),
                    });
                }
                tensors.insert(range.name.clone(), range);
            }
            if let Some(norm) = output.norm {
                if !norm.bound.is_finite() || norm.bound >= RMS_NORM_MAX {
                    violations.push(HalfRangeViolation {
                        name: norm.name.clone(),
                        dtype: "RMSNorm".to_string(),
                        kind: ViolationKind::RmsRange,
                        bound: norm.bound,
                        limit: RMS_NORM_MAX,
                        detail: format!("sqrt({})*max|gamma| is not below 2^11", norm.n),
                    });
                }
                norms.insert(norm.name.clone(), norm);
            }
        }
    }

    Ok(HalfRangeCensus {
        version: CENSUS_VERSION,
        tensors,
        norms,
        violations,
        scanned_bytes,
        scan_wall_ms: 0,
        cache_hit: false,
    })
}

struct ScanOutput {
    range: Option<TensorRange>,
    norm: Option<RmsNormRange>,
    scanned_bytes: u64,
}

fn scan_tensor(gguf: &GgufFile, desc: &TensorDesc) -> Result<ScanOutput, GgufError> {
    let is_norm = is_rms_norm_weight(desc);
    if !is_half_staged_dtype(desc.dtype) && !is_norm {
        return Ok(ScanOutput {
            range: None,
            norm: None,
            scanned_bytes: 0,
        });
    }
    let bytes = gguf.try_slice(desc)?;
    let scanned_bytes = bytes.len() as u64;
    let range = scan_bytes(desc, bytes)?;
    let norm = if is_norm {
        scan_norm(desc, bytes)?
    } else {
        None
    };
    Ok(ScanOutput {
        range,
        norm,
        scanned_bytes,
    })
}

/// Pure byte-level scan, exposed for unit tests.
pub(crate) fn scan_bytes(
    desc: &TensorDesc,
    bytes: &[u8],
) -> Result<Option<TensorRange>, GgufError> {
    if !is_half_staged_dtype(desc.dtype) {
        return Ok(None);
    }
    if bytes.len() as u64 != desc.n_bytes {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: mapped slice is {} bytes but descriptor declares {}",
            desc.name,
            bytes.len(),
            desc.n_bytes
        )));
    }
    let dtype_name = desc.dtype.wire_name();
    match desc.dtype {
        GgufType::F32 => Ok(Some(full_precision_range(desc, dtype_name, f32_max(bytes)))),
        GgufType::F16 => Ok(Some(full_precision_range(desc, dtype_name, f16_max(bytes)))),
        GgufType::BF16 => Ok(Some(full_precision_range(
            desc,
            dtype_name,
            bf16_max(bytes),
        ))),
        GgufType::MXFP4 => Ok(Some(scan_mxfp4(desc, bytes)?)),
        _ => Ok(Some(scan_block_scales(desc, bytes)?)),
    }
}

fn full_precision_range(
    desc: &TensorDesc,
    dtype: &str,
    (max_abs, non_finite): (f32, u64),
) -> TensorRange {
    TensorRange {
        name: desc.name.clone(),
        dtype: dtype.to_string(),
        method: RangeMethod::FullPrecision,
        max_abs,
        blocks: desc.n_elements(),
        mxfp4_e_hist: None,
        non_finite,
    }
}

fn scan_norm(desc: &TensorDesc, bytes: &[u8]) -> Result<Option<RmsNormRange>, GgufError> {
    let (max_abs, non_finite) = match desc.dtype {
        GgufType::F32 => f32_max(bytes),
        GgufType::F16 => f16_max(bytes),
        GgufType::BF16 => bf16_max(bytes),
        _ => return Ok(None),
    };
    let n = desc.n_elements();
    let bound = (n as f64).sqrt() as f32 * max_abs;
    let non_finite = if non_finite > 0 {
        // Surface as an out-of-range norm so the named error path reports it.
        f32::INFINITY
    } else {
        bound
    };
    Ok(Some(RmsNormRange {
        name: desc.name.clone(),
        n,
        max_abs,
        bound: non_finite,
    }))
}

fn scan_mxfp4(desc: &TensorDesc, bytes: &[u8]) -> Result<TensorRange, GgufError> {
    let block_bytes = 17usize;
    if bytes.len() % block_bytes != 0 {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: MXFP4 byte length {} is not a multiple of {block_bytes}",
            desc.name,
            bytes.len()
        )));
    }
    let blocks = (bytes.len() / block_bytes) as u64;
    // Parallel contiguous chunks keep each worker's reads sequential while
    // using every core. Only the exponent byte is touched.
    let chunk_bytes = (SCAN_CHUNK_BYTES / block_bytes).max(1) * block_bytes;
    let (hist, max_abs, non_finite) = bytes
        .par_chunks(chunk_bytes)
        .map(|chunk| {
            let mut hist = [0u64; 256];
            let mut max_abs = 0.0f32;
            let mut non_finite = 0u64;
            for block in chunk.chunks_exact(block_bytes) {
                let e = block[0];
                hist[e as usize] = hist[e as usize].saturating_add(1);
                let bound = mxfp4_block_bound(e);
                if !bound.is_finite() {
                    non_finite = non_finite.saturating_add(1);
                }
                if bound > max_abs {
                    max_abs = bound;
                }
            }
            (hist, max_abs, non_finite)
        })
        .reduce(
            || ([0u64; 256], 0.0f32, 0u64),
            |mut a, b| {
                for (slot, count) in a.0.iter_mut().zip(b.0) {
                    *slot = slot.saturating_add(count);
                }
                (a.0, a.1.max(b.1), a.2.saturating_add(b.2))
            },
        );
    let mut nonzero = BTreeMap::new();
    for (e, &count) in hist.iter().enumerate() {
        if count != 0 {
            nonzero.insert(e as u8, count);
        }
    }
    Ok(TensorRange {
        name: desc.name.clone(),
        dtype: desc.dtype.wire_name().to_string(),
        method: RangeMethod::Mxfp4Exponent,
        max_abs,
        blocks,
        mxfp4_e_hist: Some(nonzero),
        non_finite,
    })
}

fn mxfp4_block_bound(e: u8) -> f32 {
    // Matches kernels/quant_tiles.h qt_dequantize_mxfp4: the code table is
    // doubled (max 12) and the scale is 2^(e-128), with e=0/1 mapping to the
    // two smallest F32 subnormals.
    let scale = match e {
        0 => f32::from_bits(0x0020_0000),
        1 => f32::from_bits(0x0040_0000),
        _ => f32::from_bits(((e as u32) - 1) << 23),
    };
    12.0 * scale
}

fn scan_block_scales(desc: &TensorDesc, bytes: &[u8]) -> Result<TensorRange, GgufError> {
    let (block_elems, block_bytes) = desc
        .dtype
        .storage_layout()
        .ok_or_else(|| GgufError::Decode(format!("tensor {:?}: unknown layout", desc.name)))?;
    if block_bytes == 0 || block_elems == 0 {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: invalid block geometry",
            desc.name
        )));
    }
    if bytes.len() as u64 % block_bytes != 0 {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: byte length {} is not a multiple of block size {block_bytes}",
            desc.name,
            bytes.len()
        )));
    }
    let block_bytes = block_bytes as usize;
    let blocks = (bytes.len() / block_bytes) as u64;
    let chunk_bytes = (SCAN_CHUNK_BYTES / block_bytes).max(1) * block_bytes;
    let (max_abs, non_finite) = bytes
        .par_chunks(chunk_bytes)
        .map(|chunk| scan_block_chunk(desc.dtype, chunk))
        .reduce(
            || (0.0f32, 0u64),
            |a, b| (a.0.max(b.0), a.1.saturating_add(b.1)),
        );
    Ok(TensorRange {
        name: desc.name.clone(),
        dtype: desc.dtype.wire_name().to_string(),
        method: RangeMethod::BlockScale,
        max_abs,
        blocks,
        mxfp4_e_hist: None,
        non_finite,
    })
}

fn scan_block_chunk(dtype: GgmlType, bytes: &[u8]) -> (f32, u64) {
    let block_bytes = dtype.storage_layout().map(|(_, b)| b as usize).unwrap_or(0);
    let mut max_abs = 0.0f32;
    let mut non_finite = 0u64;
    for block in bytes.chunks_exact(block_bytes) {
        let bound = match dtype {
            GgufType::Q2_K => {
                let d = half_at(block, 80);
                let dmin = half_at(block, 82);
                45.0 * d.abs() + 15.0 * dmin.abs()
            }
            GgufType::Q3_K => 128.0 * half_at(block, 108).abs(),
            GgufType::Q4_K => {
                let d = half_at(block, 0);
                let dmin = half_at(block, 2);
                945.0 * d.abs() + 63.0 * dmin.abs()
            }
            GgufType::Q5_K => {
                let d = half_at(block, 0);
                let dmin = half_at(block, 2);
                1_953.0 * d.abs() + 63.0 * dmin.abs()
            }
            GgufType::Q6_K => 4_096.0 * half_at(block, 208).abs(),
            GgufType::Q8_0 => 128.0 * half_at(block, 0).abs(),
            GgufType::Q4_0 => 8.0 * half_at(block, 0).abs(),
            GgufType::Q4_1 => {
                let d = half_at(block, 0);
                let m = half_at(block, 2);
                15.0 * d.abs() + m.abs()
            }
            GgufType::IQ4_NL => 127.0 * half_at(block, 0).abs(),
            GgufType::IQ2_S | GgufType::IQ2_XS | GgufType::IQ2_XXS => {
                166.625 * half_at(block, 0).abs()
            }
            GgufType::IQ3_XXS => 480.5 * half_at(block, 0).abs(),
            GgufType::IQ3_S => 465.0 * half_at(block, 0).abs(),
            GgufType::IQ4_XS => 4_064.0 * half_at(block, 0).abs(),
            GgufType::IQ1_S => 16.875 * half_at(block, 0).abs(),
            GgufType::IQ1_M => 16.875 * iq1_m_scale(block).abs(),
            GgufType::Q2_0 => 2.0 * half_at(block, 0).abs(),
            _ => f32::INFINITY,
        };
        if !bound.is_finite() {
            non_finite = non_finite.saturating_add(1);
        }
        if bound > max_abs {
            max_abs = bound;
        }
    }
    (max_abs, non_finite)
}

fn half_at(block: &[u8], offset: usize) -> f32 {
    if offset + 2 > block.len() {
        return f32::INFINITY;
    }
    let bits = u16::from_le_bytes([block[offset], block[offset + 1]]);
    f16::from_bits(bits).to_f32()
}

fn iq1_m_scale(block: &[u8]) -> f32 {
    // kernels/iq1_decode.metalh: the scale is the high nibble of each of four
    // little-endian ushort words at byte 48.
    if block.len() < 56 {
        return f32::INFINITY;
    }
    let sc: [u16; 4] =
        std::array::from_fn(|i| u16::from_le_bytes([block[48 + 2 * i], block[49 + 2 * i]]));
    let bits = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    f16::from_bits(bits).to_f32()
}

fn f32_max(bytes: &[u8]) -> (f32, u64) {
    let (max_bits, non_finite) = bytes
        .par_chunks(SCAN_CHUNK_BYTES)
        .map(|chunk| {
            let mut max_bits = 0u32;
            let mut non_finite = 0u64;
            for bytes in chunk.chunks_exact(4) {
                let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let abs = bits & 0x7fff_ffff;
                if abs & 0x7f80_0000 == 0x7f80_0000 {
                    non_finite = non_finite.saturating_add(1);
                } else if abs > max_bits {
                    max_bits = abs;
                }
            }
            (max_bits, non_finite)
        })
        .reduce(
            || (0u32, 0u64),
            |a, b| (a.0.max(b.0), a.1.saturating_add(b.1)),
        );
    (f32::from_bits(max_bits), non_finite)
}

fn f16_max(bytes: &[u8]) -> (f32, u64) {
    let (max_bits, non_finite) = bytes
        .par_chunks(SCAN_CHUNK_BYTES)
        .map(|chunk| {
            let mut max_bits = 0u16;
            let mut non_finite = 0u64;
            for bytes in chunk.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let abs = bits & 0x7fff;
                if abs & 0x7c00 == 0x7c00 {
                    non_finite = non_finite.saturating_add(1);
                } else if abs > max_bits {
                    max_bits = abs;
                }
            }
            (max_bits, non_finite)
        })
        .reduce(
            || (0u16, 0u64),
            |a, b| (a.0.max(b.0), a.1.saturating_add(b.1)),
        );
    (f16::from_bits(max_bits).to_f32(), non_finite)
}

fn bf16_max(bytes: &[u8]) -> (f32, u64) {
    let (max_bits, non_finite) = bytes
        .par_chunks(SCAN_CHUNK_BYTES)
        .map(|chunk| {
            let mut max_bits = 0u16;
            let mut non_finite = 0u64;
            for bytes in chunk.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let abs = bits & 0x7fff;
                if abs & 0x7f80 == 0x7f80 {
                    non_finite = non_finite.saturating_add(1);
                } else if abs > max_bits {
                    max_bits = abs;
                }
            }
            (max_bits, non_finite)
        })
        .reduce(
            || (0u16, 0u64),
            |a, b| (a.0.max(b.0), a.1.saturating_add(b.1)),
        );
    (f32::from_bits((max_bits as u32) << 16), non_finite)
}

fn is_rms_norm_weight(desc: &TensorDesc) -> bool {
    let name = desc.name.as_str();
    name.ends_with("norm.weight") || name.ends_with("norm.weight.0")
}

fn is_half_staged_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::MXFP4
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::IQ4_NL
            | GgmlType::IQ2_S
            | GgmlType::IQ2_XS
            | GgmlType::IQ2_XXS
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::IQ4_XS
            | GgmlType::IQ1_S
            | GgmlType::IQ1_M
            | GgmlType::Q2_0
    )
}

// GgufType is only a local alias so the match arms stay readable.
use GgmlType as GgufType;

fn cache_key(gguf: &GgufFile) -> Result<[u8; 32], GgufError> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CACHE_SCHEMA);
    for shard in &gguf.shards {
        let stamp = shard.source_stamp;
        hasher.update(&stamp.dev.to_le_bytes());
        hasher.update(&stamp.ino.to_le_bytes());
        hasher.update(&stamp.size.to_le_bytes());
        hasher.update(&stamp.mtime_sec.to_le_bytes());
        hasher.update(&stamp.mtime_nsec.to_le_bytes());
        hasher.update(&stamp.ctime_sec.to_le_bytes());
        hasher.update(&stamp.ctime_nsec.to_le_bytes());
        hasher.update(&shard.tensor_data_start.to_le_bytes());
        hasher.update(&shard.alignment.to_le_bytes());
    }
    for tensor in &gguf.tensors {
        hasher.update(tensor.name.as_bytes());
        hasher.update(b"\0");
        hasher.update(&(tensor.dtype as i32).to_le_bytes());
        hasher.update(&(tensor.shard_idx as u64).to_le_bytes());
        hasher.update(&tensor.data_offset.to_le_bytes());
        hasher.update(&tensor.n_bytes.to_le_bytes());
        for &dim in &tensor.shape {
            hasher.update(&dim.to_le_bytes());
        }
    }
    Ok(*hasher.finalize().as_bytes())
}

fn cache_root() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(CACHE_DIR_ENV) {
        return Some(PathBuf::from(dir));
    }
    if std::env::var(CACHE_ENV).as_deref() == Ok("0") {
        return None;
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/qwen-llm/half-range"))
}

fn cache_path(root: &Path, key: &[u8; 32]) -> PathBuf {
    root.join(format!("{}.json", hex(key)))
}

fn read_cache(root: &Path, key: &[u8; 32]) -> Option<HalfRangeCensus> {
    let bytes = std::fs::read(cache_path(root, key)).ok()?;
    let census: HalfRangeCensus = serde_json::from_slice(&bytes).ok()?;
    (census.version == CENSUS_VERSION).then_some(census)
}

fn write_cache(root: &Path, key: &[u8; 32], census: &HalfRangeCensus) {
    let path = cache_path(root, key);
    let Ok(bytes) = serde_json::to_vec(census) else {
        return;
    };
    if std::fs::create_dir_all(root).is_err() {
        return;
    }
    let tmp = path.with_extension(format!("tmp.{}", std::process::id()));
    let Ok(mut file) = std::fs::File::create(&tmp) else {
        return;
    };
    if file.write_all(&bytes).is_err() {
        let _ = std::fs::remove_file(&tmp);
        return;
    }
    let _ = file.sync_all();
    if std::fs::rename(&tmp, &path).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

fn hex(bytes: &[u8; 32]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(64);
    for &byte in bytes {
        out.push(DIGITS[(byte >> 4) as usize] as char);
        out.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(name: &str, dtype: GgmlType, shape: Vec<u64>, n_bytes: u64) -> TensorDesc {
        TensorDesc {
            name: name.to_string(),
            shape,
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes,
        }
    }

    fn half_bytes(value: f32) -> [u8; 2] {
        f16::from_f32(value).to_bits().to_le_bytes()
    }

    #[test]
    fn mxfp4_exponent_bound_matches_quant_tile_table() {
        assert_eq!(mxfp4_block_bound(127), 6.0);
        assert_eq!(mxfp4_block_bound(128), 12.0);
        assert!(mxfp4_block_bound(141) >= HALF_DECODED_MAX);
        assert!(mxfp4_block_bound(139) < HALF_DECODED_MAX);
    }

    #[test]
    fn mxfp4_histogram_and_bound_are_conservative() {
        let mut bytes = Vec::new();
        for e in [117u8, 117, 127] {
            let mut block = [0u8; 17];
            block[0] = e;
            bytes.extend_from_slice(&block);
        }
        let d = desc(
            "blk.0.ffn_down_exps.weight",
            GgmlType::MXFP4,
            vec![32 * 3],
            51,
        );
        let range = scan_bytes(&d, &bytes).unwrap().unwrap();
        assert_eq!(range.blocks, 3);
        assert_eq!(range.max_abs, 6.0);
        let hist = range.mxfp4_e_hist.unwrap();
        assert_eq!(hist.get(&117), Some(&2));
        assert_eq!(hist.get(&127), Some(&1));
    }

    #[test]
    fn q8_0_scale_bound_uses_int8_max() {
        let mut block = Vec::new();
        block.extend_from_slice(&half_bytes(0.25));
        block.extend_from_slice(&[0u8; 32]);
        let d = desc("blk.0.ffn_down.weight", GgmlType::Q8_0, vec![32], 34);
        let range = scan_bytes(&d, &block).unwrap().unwrap();
        assert_eq!(range.max_abs, 32.0);
        assert_eq!(range.method, RangeMethod::BlockScale);
    }

    #[test]
    fn q4_k_scale_bound_uses_both_fields() {
        let mut block = vec![0u8; 144];
        block[0..2].copy_from_slice(&half_bytes(0.5));
        block[2..4].copy_from_slice(&half_bytes(0.25));
        let d = desc("blk.0.ffn_down.weight", GgmlType::Q4_K, vec![256], 144);
        let range = scan_bytes(&d, &block).unwrap().unwrap();
        assert_eq!(range.max_abs, 945.0 * 0.5 + 63.0 * 0.25);
    }

    #[test]
    fn f16_full_max_keeps_largest_absolute_value() {
        let mut bytes = Vec::new();
        for value in [1.0f32, -3.5, 2.0, 0.0] {
            bytes.extend_from_slice(&half_bytes(value));
        }
        let d = desc("blk.0.attn_norm.weight", GgmlType::F16, vec![4], 8);
        let range = scan_bytes(&d, &bytes).unwrap().unwrap();
        assert_eq!(range.max_abs, 3.5);
        assert_eq!(range.method, RangeMethod::FullPrecision);
    }

    #[test]
    fn non_finite_scale_is_counted() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0x00, 0x7c]); // +inf half
        block.extend_from_slice(&[0u8; 32]);
        let d = desc("blk.0.ffn_down.weight", GgmlType::Q8_0, vec![32], 34);
        let range = scan_bytes(&d, &block).unwrap().unwrap();
        assert_eq!(range.non_finite, 1);
    }

    #[test]
    fn unsafe_tensor_names_the_tensor() {
        let mut block = Vec::new();
        block.extend_from_slice(&half_bytes(300.0));
        block.extend_from_slice(&[0u8; 32]);
        let d = desc("blk.0.ffn_down.weight", GgmlType::Q8_0, vec![32], 34);
        let range = scan_bytes(&d, &block).unwrap().unwrap();
        assert!(range.max_abs >= HALF_DECODED_MAX);
        assert_eq!(range.name, "blk.0.ffn_down.weight");
    }

    #[test]
    #[ignore = "artifact cached census; set QWEN_HALF_RANGE_MODEL to a GGUF"]
    fn artifact_cached_census_report() {
        let Some(path) = std::env::var_os("QWEN_HALF_RANGE_MODEL") else {
            return;
        };
        let gguf = GgufFile::open(path).expect("open GGUF");
        let started = Instant::now();
        let census = cached_census(&gguf).expect("cached census");
        let wall = started.elapsed();
        eprintln!(
            "half-range-census cached tensors={} norms={} bytes={} wall_ms={} cache_hit={} violations={}",
            census.tensors.len(),
            census.norms.len(),
            census.scanned_bytes,
            wall.as_millis(),
            census.cache_hit,
            census.violations.len()
        );
    }

    #[test]
    #[ignore = "artifact census; set QWEN_HALF_RANGE_MODEL to a GGUF"]
    fn artifact_census_report() {
        let Some(path) = std::env::var_os("QWEN_HALF_RANGE_MODEL") else {
            return;
        };
        let gguf = GgufFile::open(path).expect("open GGUF");
        if std::env::var_os("QWEN_HALF_RANGE_PREFETCH").is_some() {
            for shard in &gguf.shards {
                let _ = crate::prefetch::prefetch_file(
                    &shard.path,
                    crate::prefetch::DEFAULT_WORKERS,
                    crate::prefetch::DEFAULT_CHUNK_BYTES,
                );
            }
        }
        let started = Instant::now();
        let census = census_uncached(&gguf).expect("census");
        let wall = started.elapsed();
        eprintln!(
            "half-range-census uncached tensors={} norms={} bytes={} wall_ms={} violations={}",
            census.tensors.len(),
            census.norms.len(),
            census.scanned_bytes,
            wall.as_millis(),
            census.violations.len()
        );
        for violation in &census.violations {
            eprintln!("half-range-census violation {violation:?}");
        }
        let mut worst = census
            .tensors
            .values()
            .map(|range| (range.max_abs, range.name.clone(), range.dtype.clone()))
            .collect::<Vec<_>>();
        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (bound, name, dtype) in worst.into_iter().take(8) {
            eprintln!("half-range-census worst bound={bound:.6e} dtype={dtype} tensor={name}");
        }
        let mut worst_norm = census
            .norms
            .values()
            .map(|norm| (norm.bound, norm.name.clone(), norm.n, norm.max_abs))
            .collect::<Vec<_>>();
        worst_norm.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (bound, name, n, max_abs) in worst_norm.into_iter().take(5) {
            eprintln!(
                "half-range-census worst-norm bound={bound:.6e} n={n} max_abs={max_abs:.6} tensor={name}"
            );
        }
    }

    #[test]
    fn ds4_scope_allows_mxfp4_down_reroute_only() {
        let down = TensorRange {
            name: "blk.3.ffn_down_exps.weight".to_string(),
            dtype: "MXFP4".to_string(),
            method: RangeMethod::Mxfp4Exponent,
            max_abs: 100_000.0,
            blocks: 1,
            mxfp4_e_hist: None,
            non_finite: 0,
        };
        assert!(HalfRangeScope::DeepSeekV4.allows_f32_reroute(&down));
        let gate = TensorRange {
            name: "blk.3.ffn_gate_exps.weight".to_string(),
            ..down.clone()
        };
        assert!(!HalfRangeScope::DeepSeekV4.allows_f32_reroute(&gate));
        assert!(!HalfRangeScope::Generic.allows_f32_reroute(&down));
    }
}
