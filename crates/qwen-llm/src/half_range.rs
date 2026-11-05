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
//! JSON read. The first scan reads each shard sequentially in offset order
//! through a reusable 16 MiB pread buffer; shards are scanned in parallel.
//! Only scale/header bytes are inspected, but the payload pages are read once.
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
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Decoded weights entering a half-staged path must stay below this bound.
pub const HALF_DECODED_MAX: f32 = 32_768.0;
/// RMSNorm output bound (`sqrt(n) * max|gamma|`) for half-staged consumers.
pub const RMS_NORM_MAX: f32 = 2_048.0;
/// Bump when the bound formulas or cache schema change.
pub const CENSUS_VERSION: u32 = 2;

/// Bounded pread chunk for the sequential per-shard census reader.
const SCAN_READ_BYTES: usize = 16 * 1024 * 1024;

const CACHE_SCHEMA: &[u8] = b"qwen-half-range-census-v2\0";
/// Overrides the directory for the per-artifact half-range census cache.
const CACHE_DIR_ENV: &str = "QWEN_HALF_RANGE_CACHE_DIR";
/// Set to 0 to disable the half-range census cache for this process.
const CACHE_ENV: &str = "QWEN_HALF_RANGE_CACHE";
/// Set to 1 to print one half-range census timing line per load.
const REPORT_ENV: &str = "QWEN_HALF_RANGE_REPORT";

/// How a tensor bound was obtained.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RangeMethod {
    Mxfp4Exponent,
    BlockScale,
    ExactDecode,
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
    fn recognizes_existing_f32_dispatch(self, range: &TensorRange) -> bool {
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

crate::env_flag!(
    /// Enables the load-time decoded-range guard for half-staged weights.
    /// Default off; set to 1/true/TRUE/yes/YES to enable. Disabled returns
    /// before cache identity, filesystem access, or scanning.
    default_off half_range_guard_enabled,
    "QWEN_HALF_RANGE_GUARD"
);

#[cfg(test)]
thread_local! {
    static GUARD_OVERRIDE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Test shim: the env flag is latched once per process, so loader tests need a
/// scoped override.
#[cfg(test)]
pub(crate) fn with_guard_override<R>(value: bool, f: impl FnOnce() -> R) -> R {
    struct Restore(Option<bool>);
    impl Drop for Restore {
        fn drop(&mut self) {
            GUARD_OVERRIDE.with(|slot| slot.set(self.0));
        }
    }
    let previous = GUARD_OVERRIDE.with(|slot| slot.replace(Some(value)));
    let _restore = Restore(previous);
    f()
}

fn guard_enabled() -> bool {
    #[cfg(test)]
    if let Some(value) = GUARD_OVERRIDE.with(|slot| slot.get()) {
        return value;
    }
    half_range_guard_enabled()
}

#[cfg(test)]
pub(crate) fn guard_enabled_for_test() -> bool {
    guard_enabled()
}

fn empty_census() -> HalfRangeCensus {
    HalfRangeCensus {
        version: CENSUS_VERSION,
        tensors: BTreeMap::new(),
        norms: BTreeMap::new(),
        violations: Vec::new(),
        scanned_bytes: 0,
        scan_wall_ms: 0,
        cache_hit: false,
    }
}

#[cfg(test)]
pub(crate) fn disabled_census_for_test() -> HalfRangeCensus {
    empty_census()
}

/// Compute (or load from cache) the census and refuse unsafe tensors.
pub fn audit_or_refuse(
    gguf: &GgufFile,
    scope: HalfRangeScope,
) -> Result<HalfRangeCensus, HalfRangeError> {
    if !guard_enabled() {
        return Ok(empty_census());
    }
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
                    && scope.recognizes_existing_f32_dispatch(range)
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
    cached_census_with_root_impl(gguf, cache_root().as_deref())
}

#[cfg(test)]
pub(crate) fn cached_census_with_root(
    gguf: &GgufFile,
    root: Option<&Path>,
) -> Result<HalfRangeCensus, GgufError> {
    cached_census_with_root_impl(gguf, root)
}

fn cached_census_with_root_impl(
    gguf: &GgufFile,
    root: Option<&Path>,
) -> Result<HalfRangeCensus, GgufError> {
    // Opening-time stamps are not enough: an already-open GgufFile can outlive
    // an in-place rewrite. Revalidate the retained descriptors before the cache
    // lookup and again after scanning before publishing/returning success.
    let before = gguf.revalidate_retained_shard_stamps()?;
    let key = cache_key(gguf)?;
    let key_lock = key_lock(&key);
    let _single_flight = key_lock
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let started = Instant::now();

    if let Some(root) = root
        && let Some(mut cached) = read_cache(root, &key)
    {
        let after = gguf.revalidate_retained_shard_stamps()?;
        if after != before {
            return Err(scan_changed_error());
        }
        if cache_covers(gguf, &cached) {
            cached.violations = violations_from(&cached.tensors, &cached.norms);
            cached.cache_hit = true;
            cached.scan_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
            report_census(&cached);
            return Ok(cached);
        }
    }

    let mut census = census_uncached(gguf)?;
    let after = gguf.revalidate_retained_shard_stamps()?;
    if after != before {
        return Err(scan_changed_error());
    }
    census.violations = violations_from(&census.tensors, &census.norms);
    census.scan_wall_ms = u64::try_from(started.elapsed().as_millis()).unwrap_or(u64::MAX);
    census.cache_hit = false;
    if let Some(root) = root {
        write_cache(root, &key, &census);
    }
    report_census(&census);
    Ok(census)
}

fn scan_changed_error() -> GgufError {
    GgufError::Decode("retained GGUF shard changed during half-range census".into())
}

fn report_enabled() -> bool {
    std::env::var(REPORT_ENV)
        .as_deref()
        .is_ok_and(crate::env_flag::env_value_truthy)
}

fn report_census(census: &HalfRangeCensus) {
    if !report_enabled() {
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

fn violations_from(
    tensors: &BTreeMap<String, TensorRange>,
    norms: &BTreeMap<String, RmsNormRange>,
) -> Vec<HalfRangeViolation> {
    let mut violations = Vec::new();
    for range in tensors.values() {
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
    }
    for norm in norms.values() {
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
    }
    violations.sort_by(|left, right| left.name.cmp(&right.name));
    violations
}

/// Uncached census. Used by tests and by `QWEN_HALF_RANGE_CACHE=0` runs.
pub fn census_uncached(gguf: &GgufFile) -> Result<HalfRangeCensus, GgufError> {
    // Tensor descriptors are not stored in file order across a split model.
    // Sort each shard by offset, then read it with bounded pread chunks. One
    // sequential reader per shard avoids both the random-order seek storm and
    // the mmap page-fault storm a many-threaded fault path causes.
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
        .enumerate()
        .map(|(shard_idx, descs)| scan_shard(gguf, shard_idx, descs))
        .collect();

    let mut tensors = BTreeMap::new();
    let mut norms = BTreeMap::new();
    let mut scanned_bytes = 0u64;

    for result in per_shard {
        for output in result? {
            scanned_bytes = scanned_bytes.saturating_add(output.scanned_bytes);
            if let Some(range) = output.range {
                tensors.insert(range.name.clone(), range);
            }
            if let Some(norm) = output.norm {
                norms.insert(norm.name.clone(), norm);
            }
        }
    }
    let violations = violations_from(&tensors, &norms);

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

fn scan_shard(
    gguf: &GgufFile,
    shard_idx: usize,
    descs: &[&TensorDesc],
) -> Result<Vec<ScanOutput>, GgufError> {
    let shard = &gguf.shards[shard_idx];
    let mut buffer = vec![0u8; SCAN_READ_BYTES];
    let mut outputs = Vec::with_capacity(descs.len());
    for desc in descs {
        outputs.push(scan_tensor_pread(gguf, shard, desc, &mut buffer)?);
    }
    Ok(outputs)
}

fn scan_tensor_pread(
    gguf: &GgufFile,
    shard: &crate::gguf::GgufShard,
    desc: &TensorDesc,
    buffer: &mut [u8],
) -> Result<ScanOutput, GgufError> {
    let is_norm = is_rms_norm_weight(desc);
    let half_staged = requires_half_bound(desc);
    if !half_staged && !is_norm {
        return Ok(ScanOutput {
            range: None,
            norm: None,
            scanned_bytes: 0,
        });
    }
    let (block_elems, block_bytes) = match desc.dtype.storage_layout() {
        Some(layout) => layout,
        None => {
            return Err(GgufError::Decode(format!(
                "tensor {:?}: unknown layout",
                desc.name
            )));
        }
    };
    if block_bytes == 0 || block_elems == 0 {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: invalid block geometry",
            desc.name
        )));
    }
    if desc.n_bytes % block_bytes != 0 {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: byte length {} is not a multiple of block size {block_bytes}",
            desc.name, desc.n_bytes
        )));
    }
    let block_bytes = block_bytes as usize;
    let chunk_bytes = (SCAN_READ_BYTES / block_bytes).max(1) * block_bytes;
    let mut max_abs = 0.0f32;
    let mut non_finite = 0u64;
    let mut blocks = 0u64;
    let mut hist = [0u64; 256];

    let mut offset = 0u64;
    while offset < desc.n_bytes {
        let want = (chunk_bytes as u64).min(desc.n_bytes - offset) as usize;
        read_exact_at(
            shard.file.as_ref(),
            &mut buffer[..want],
            desc.data_offset + offset,
        )?;
        let chunk = &buffer[..want];
        if half_staged {
            match desc.dtype {
                GgufType::MXFP4 => {
                    let (part, m, nf) = scan_mxfp4_chunk(chunk);
                    for (slot, count) in hist.iter_mut().zip(part) {
                        *slot = slot.saturating_add(count);
                    }
                    max_abs = max_abs.max(m);
                    non_finite = non_finite.saturating_add(nf);
                    blocks = blocks.saturating_add((chunk.len() / 17) as u64);
                }
                GgufType::F32 | GgufType::F16 | GgufType::BF16 => {
                    let (m, nf) = scan_full_chunk(desc.dtype, chunk);
                    max_abs = max_abs.max(m);
                    non_finite = non_finite.saturating_add(nf);
                    blocks = blocks.saturating_add((chunk.len() / block_bytes) as u64);
                }
                _ => {
                    let (m, nf) = scan_block_chunk(desc.dtype, chunk);
                    max_abs = max_abs.max(m);
                    non_finite = non_finite.saturating_add(nf);
                    blocks = blocks.saturating_add((chunk.len() / block_bytes) as u64);
                }
            }
        } else {
            let (m, nf) = scan_full_chunk(desc.dtype, chunk);
            max_abs = max_abs.max(m);
            non_finite = non_finite.saturating_add(nf);
        }
        offset += want as u64;
    }

    let mut method = match desc.dtype {
        GgufType::MXFP4 => RangeMethod::Mxfp4Exponent,
        GgufType::F32 | GgufType::F16 | GgufType::BF16 => RangeMethod::FullPrecision,
        _ => RangeMethod::BlockScale,
    };
    if half_staged
        && (max_abs >= HALF_DECODED_MAX || non_finite > 0)
        && !matches!(desc.dtype, GgufType::F32 | GgufType::F16 | GgufType::BF16)
        && let Some((exact_max, exact_non_finite)) = exact_decode_range(gguf, desc)?
    {
        // The coarse bound is deliberately conservative (max code/subscale).
        // Only on a coarse refusal do we pay for an exact decode. This admits
        // e.g. an MXFP4 exponent 140 block whose only codes are small, and a
        // K/IQ block whose maximum subscale is zero.
        max_abs = exact_max;
        non_finite = exact_non_finite;
        method = RangeMethod::ExactDecode;
    }

    let range = if half_staged {
        let mut nonzero = BTreeMap::new();
        for (e, &count) in hist.iter().enumerate() {
            if count != 0 {
                nonzero.insert(e as u8, count);
            }
        }
        Some(TensorRange {
            name: desc.name.clone(),
            dtype: desc.dtype.wire_name().to_string(),
            method,
            max_abs,
            blocks,
            mxfp4_e_hist: (desc.dtype == GgufType::MXFP4).then_some(nonzero),
            non_finite,
        })
    } else {
        None
    };
    let norm = if is_norm {
        let n = desc.n_elements();
        let bound = (n as f64).sqrt() as f32 * max_abs;
        let bound = if non_finite > 0 { f32::INFINITY } else { bound };
        Some(RmsNormRange {
            name: desc.name.clone(),
            n,
            max_abs,
            bound,
        })
    } else {
        None
    };
    Ok(ScanOutput {
        range,
        norm,
        scanned_bytes: desc.n_bytes,
    })
}

fn exact_decode_range(gguf: &GgufFile, desc: &TensorDesc) -> Result<Option<(f32, u64)>, GgufError> {
    let bytes = gguf.try_slice(desc)?;
    Ok(exact_max_from_bytes(desc, bytes))
}

fn exact_max_from_bytes(desc: &TensorDesc, bytes: &[u8]) -> Option<(f32, u64)> {
    let values = crate::codec::dequant_to_f32(desc, bytes).ok()?;
    let mut max_abs = 0.0f32;
    let mut non_finite = 0u64;
    for value in values {
        if !value.is_finite() {
            non_finite = non_finite.saturating_add(1);
        } else if value.abs() > max_abs {
            max_abs = value.abs();
        }
    }
    Some((max_abs, non_finite))
}

fn read_exact_at(
    file: &std::fs::File,
    mut buffer: &mut [u8],
    mut offset: u64,
) -> Result<(), GgufError> {
    use std::os::unix::fs::FileExt;
    while !buffer.is_empty() {
        match file.read_at(buffer, offset) {
            Ok(0) => {
                return Err(GgufError::Io(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    format!("short read at offset {offset}"),
                )));
            }
            Ok(n) => {
                buffer = &mut buffer[n..];
                offset += n as u64;
            }
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
            Err(error) => return Err(GgufError::Io(error)),
        }
    }
    Ok(())
}

/// Pure byte-level scan, exposed for unit tests.
#[cfg(test)]
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

#[cfg(test)]
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

#[cfg(test)]
fn scan_mxfp4(desc: &TensorDesc, bytes: &[u8]) -> Result<TensorRange, GgufError> {
    let block_bytes = 17usize;
    if bytes.len() % block_bytes != 0 {
        return Err(GgufError::Decode(format!(
            "tensor {:?}: MXFP4 byte length {} is not a multiple of {block_bytes}",
            desc.name,
            bytes.len()
        )));
    }
    let (hist, max_abs, non_finite) = scan_mxfp4_chunk(bytes);
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
        blocks: (bytes.len() / block_bytes) as u64,
        mxfp4_e_hist: Some(nonzero),
        non_finite,
    })
}

fn scan_mxfp4_chunk(bytes: &[u8]) -> ([u64; 256], f32, u64) {
    let mut hist = [0u64; 256];
    let mut max_abs = 0.0f32;
    let mut non_finite = 0u64;
    for block in bytes.chunks_exact(17) {
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

#[cfg(test)]
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
    let (max_abs, non_finite) = scan_block_chunk(desc.dtype, bytes);
    Ok(TensorRange {
        name: desc.name.clone(),
        dtype: desc.dtype.wire_name().to_string(),
        method: RangeMethod::BlockScale,
        max_abs,
        blocks: (bytes.len() as u64 / block_bytes) as u64,
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

fn scan_full_chunk(dtype: GgmlType, bytes: &[u8]) -> (f32, u64) {
    match dtype {
        GgufType::F32 => {
            let mut max_bits = 0u32;
            let mut non_finite = 0u64;
            for bytes in bytes.chunks_exact(4) {
                let bits = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                let abs = bits & 0x7fff_ffff;
                if abs & 0x7f80_0000 == 0x7f80_0000 {
                    non_finite = non_finite.saturating_add(1);
                } else if abs > max_bits {
                    max_bits = abs;
                }
            }
            (f32::from_bits(max_bits), non_finite)
        }
        GgufType::F16 => {
            let mut max_bits = 0u16;
            let mut non_finite = 0u64;
            for bytes in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let abs = bits & 0x7fff;
                if abs & 0x7c00 == 0x7c00 {
                    non_finite = non_finite.saturating_add(1);
                } else if abs > max_bits {
                    max_bits = abs;
                }
            }
            (f16::from_bits(max_bits).to_f32(), non_finite)
        }
        GgufType::BF16 => {
            let mut max_bits = 0u16;
            let mut non_finite = 0u64;
            for bytes in bytes.chunks_exact(2) {
                let bits = u16::from_le_bytes([bytes[0], bytes[1]]);
                let abs = bits & 0x7fff;
                if abs & 0x7f80 == 0x7f80 {
                    non_finite = non_finite.saturating_add(1);
                } else if abs > max_bits {
                    max_bits = abs;
                }
            }
            (f32::from_bits((max_bits as u32) << 16), non_finite)
        }
        _ => (f32::INFINITY, 1),
    }
}

#[cfg(test)]
fn f32_max(bytes: &[u8]) -> (f32, u64) {
    scan_full_chunk(GgufType::F32, bytes)
}

#[cfg(test)]
fn f16_max(bytes: &[u8]) -> (f32, u64) {
    scan_full_chunk(GgufType::F16, bytes)
}

#[cfg(test)]
fn bf16_max(bytes: &[u8]) -> (f32, u64) {
    scan_full_chunk(GgufType::BF16, bytes)
}

fn is_rms_norm_weight_name(name: &str) -> bool {
    name.ends_with("norm.weight")
        || name.ends_with("norm.weight.0")
        || name.ends_with("_norm_key.weight")
        || name.ends_with("_norm_query.weight")
        || name.ends_with("_norm_conv.weight")
}

fn is_rms_norm_weight(desc: &TensorDesc) -> bool {
    is_rms_norm_weight_name(&desc.name)
}

/// Weight role used to decide whether a tensor can enter a half-staged path.
/// Names are the GGUF binding vocabulary; unknown names stay conservative.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TensorRole {
    Expert,
    Dense,
    Embedding,
    Norm,
    Auxiliary,
    Unknown,
}

fn tensor_role(name: &str) -> TensorRole {
    if name.contains("_exps") {
        return TensorRole::Expert;
    }
    if is_rms_norm_weight_name(name) {
        return TensorRole::Norm;
    }
    if name.ends_with(".bias")
        || name.ends_with("ssm_a")
        || name.ends_with("ssm_conv1d.weight")
        || name.ends_with("conv1d.weight")
        || name.ends_with("a_log")
    {
        return TensorRole::Auxiliary;
    }
    if name == "token_embd.weight" || name == "output.weight" || name.contains("token_embd") {
        return TensorRole::Embedding;
    }
    if name.contains("ffn_")
        || name.contains("attn_")
        || name.contains("ssm_")
        || name.contains("indexer.")
        || name == "output.weight"
    {
        return TensorRole::Dense;
    }
    TensorRole::Unknown
}

fn dense_half_staged_dtype(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::IQ4_NL
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::IQ4_XS
    )
}

fn requires_half_bound(desc: &TensorDesc) -> bool {
    match tensor_role(&desc.name) {
        TensorRole::Expert => is_half_staged_dtype(desc.dtype),
        TensorRole::Dense | TensorRole::Embedding => dense_half_staged_dtype(desc.dtype),
        TensorRole::Norm | TensorRole::Auxiliary => false,
        TensorRole::Unknown => is_half_staged_dtype(desc.dtype),
    }
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
    // Disable wins over a directory override.
    if std::env::var(CACHE_ENV).as_deref() == Ok("0") {
        return None;
    }
    if let Some(dir) = std::env::var_os(CACHE_DIR_ENV) {
        return Some(PathBuf::from(dir));
    }
    std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache/qwen-llm/half-range"))
}

fn cache_path(root: &Path, key: &[u8; 32]) -> PathBuf {
    root.join(format!("{}.json", hex(key)))
}

fn cache_covers(gguf: &GgufFile, census: &HalfRangeCensus) -> bool {
    gguf.tensors.iter().all(|desc| {
        let required = requires_half_bound(desc);
        let is_norm = is_rms_norm_weight(desc);
        (!required || census.tensors.contains_key(&desc.name))
            && (!is_norm || census.norms.contains_key(&desc.name))
    })
}

fn read_cache(root: &Path, key: &[u8; 32]) -> Option<HalfRangeCensus> {
    let bytes = std::fs::read(cache_path(root, key)).ok()?;
    let census: HalfRangeCensus = serde_json::from_slice(&bytes).ok()?;
    (census.version == CENSUS_VERSION).then_some(census)
}

static KEY_LOCKS: OnceLock<Mutex<BTreeMap<[u8; 32], Arc<Mutex<()>>>>> = OnceLock::new();
static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn key_lock(key: &[u8; 32]) -> Arc<Mutex<()>> {
    let locks = KEY_LOCKS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut guard = locks
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard
        .entry(*key)
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone()
}

fn private_cache_dir(root: &Path) {
    let mut builder = std::fs::DirBuilder::new();
    builder.recursive(true).mode(0o700);
    let _ = builder.create(root);
    let _ = std::fs::set_permissions(root, std::fs::Permissions::from_mode(0o700));
}

fn write_cache(root: &Path, key: &[u8; 32], census: &HalfRangeCensus) {
    let path = cache_path(root, key);
    let Ok(bytes) = serde_json::to_vec(census) else {
        return;
    };
    private_cache_dir(root);
    for attempt in 0..8 {
        let tmp = path.with_extension(format!(
            "tmp.{}.{}.{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed),
            attempt
        ));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&tmp)
        {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return,
        };
        if file.write_all(&bytes).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        let _ = file.sync_all();
        if std::fs::rename(&tmp, &path).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return;
        }
        return;
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
        assert!(HalfRangeScope::DeepSeekV4.recognizes_existing_f32_dispatch(&down));
        let gate = TensorRange {
            name: "blk.3.ffn_gate_exps.weight".to_string(),
            ..down.clone()
        };
        assert!(!HalfRangeScope::DeepSeekV4.recognizes_existing_f32_dispatch(&gate));
        assert!(!HalfRangeScope::Generic.recognizes_existing_f32_dispatch(&down));
    }

    #[test]
    fn role_requirements_skip_f32_dense_and_auxiliary_but_keep_expert_banks() {
        let f32_dense = desc("blk.0.ffn_down.weight", GgmlType::F32, vec![32], 128);
        assert!(!requires_half_bound(&f32_dense));
        let f32_expert = desc("blk.0.ffn_down_exps.weight", GgmlType::F32, vec![32], 128);
        assert!(requires_half_bound(&f32_expert));
        let q8_dense = desc("blk.0.ffn_down.weight", GgmlType::Q8_0, vec![32], 34);
        assert!(requires_half_bound(&q8_dense));
        let mxfp4_dense = desc("blk.0.ffn_down.weight", GgmlType::MXFP4, vec![32], 17);
        assert!(!requires_half_bound(&mxfp4_dense));
        let mxfp4_expert = desc("blk.0.ffn_down_exps.weight", GgmlType::MXFP4, vec![32], 17);
        assert!(requires_half_bound(&mxfp4_expert));
        let bias = desc("blk.0.ssm_dt.bias", GgmlType::F32, vec![32], 128);
        assert!(!requires_half_bound(&bias));
        let norm = desc("blk.0.attn_norm.weight", GgmlType::F32, vec![32], 128);
        assert!(!requires_half_bound(&norm));
        assert!(is_rms_norm_weight(&norm));
    }

    #[test]
    fn coarse_bounds_cover_exact_decoder_for_half_formats() {
        let formats = [
            GgmlType::Q2_K,
            GgmlType::Q3_K,
            GgmlType::Q4_K,
            GgmlType::Q5_K,
            GgmlType::Q6_K,
            GgmlType::Q8_0,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::IQ4_NL,
            GgmlType::IQ2_S,
            GgmlType::IQ2_XS,
            GgmlType::IQ2_XXS,
            GgmlType::IQ3_XXS,
            GgmlType::IQ3_S,
            GgmlType::IQ4_XS,
            GgmlType::IQ1_S,
            GgmlType::IQ1_M,
            GgmlType::Q2_0,
            GgmlType::MXFP4,
        ];
        let mut state = 0x1234_5678u32;
        for dtype in formats {
            let (block_elems, block_bytes) = dtype.storage_layout().unwrap();
            let blocks = 2u64;
            let mut bytes = vec![0u8; (block_bytes * blocks) as usize];
            for byte in &mut bytes {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                *byte = (state >> 24) as u8;
            }
            let d = desc(
                "blk.0.ffn_down.weight",
                dtype,
                vec![block_elems * blocks],
                block_bytes * blocks,
            );
            let coarse = scan_bytes(&d, &bytes).unwrap().unwrap();
            if let Some((exact_max, _)) = exact_max_from_bytes(&d, &bytes) {
                if coarse.max_abs.is_finite() && exact_max.is_finite() {
                    assert!(
                        coarse.max_abs >= exact_max,
                        "{dtype:?}: coarse bound {} < exact max {}",
                        coarse.max_abs,
                        exact_max
                    );
                }
            }
        }
    }

    #[test]
    fn exact_decode_admits_zero_scale_q8_block() {
        let mut block = Vec::new();
        block.extend_from_slice(&half_bytes(300.0));
        block.extend_from_slice(&[0u8; 32]);
        let d = desc("blk.0.ffn_down.weight", GgmlType::Q8_0, vec![32], 34);
        let coarse = scan_bytes(&d, &block).unwrap().unwrap();
        assert!(coarse.max_abs >= HALF_DECODED_MAX);
        let (exact_max, non_finite) = exact_max_from_bytes(&d, &block).unwrap();
        assert_eq!(exact_max, 0.0);
        assert_eq!(non_finite, 0);
    }

    #[test]
    fn exact_decode_keeps_nonfinite_scale_unsafe() {
        let mut block = Vec::new();
        block.extend_from_slice(&[0x00, 0x7c]);
        block.extend_from_slice(&[0u8; 32]);
        let d = desc("blk.0.ffn_down.weight", GgmlType::Q8_0, vec![32], 34);
        let coarse = scan_bytes(&d, &block).unwrap().unwrap();
        assert_eq!(coarse.non_finite, 1);
        if let Some((exact_max, exact_non_finite)) = exact_max_from_bytes(&d, &block) {
            assert!(exact_non_finite > 0 || !exact_max.is_finite());
        }
    }
}
