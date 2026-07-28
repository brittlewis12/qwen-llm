//! Stable ownership boundary for loaded-model execution.
//!
//! The Metal modules expose the low-level execution pieces directly:
//! [`MetalContext`] owns device resources, [`MetalModel`] owns resident
//! weights, and [`MetalSession`] owns mutable per-sequence state. This module
//! gives callers semantic names for those lifetimes without changing the
//! underlying execution path.

use crate::cache_probe::{available_memory_bytes, probe_fd_residency};
use crate::checkpoint_codec::SNAPSHOT_RECORD_FIXED_BYTES;
use crate::checkpoint_identity::{
    CheckpointCompatibilityReport, CheckpointIdentityCache, CheckpointIdentityError,
    checkpoint_compatibility,
};
use crate::checkpoint_store::{
    CheckpointStoreError, DurableCheckpointStore, PublishReport, StoreContext,
};
use crate::gguf::{GgufError, GgufFile, GgufShard};
use crate::loader::{LoadError, Model};
use crate::metal::{MetalContext, MetalError};
use crate::metal_forward::{
    MetalForward, MetalLoadPrefetchAdvice, MetalModel, MetalModelLoadOptions, MetalSession,
    MfError, SessionSnapshot, SnapshotIdentity, SnapshotValidationError,
};
use crate::model::Arch;
use crate::prefetch::{DEFAULT_CHUNK_BYTES, DEFAULT_WORKERS, prefetch_fd};
use crate::prefix_cache::{DEFAULT_MAX_BYTES, PrefixCache, PrefixCacheStats};
use crate::tokenizer::{TokError, Tokenizer};
use parking_lot::Mutex;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("metal runtime: {0}")]
    Metal(#[from] MetalError),
    #[error("gguf load: {0}")]
    Gguf(#[from] GgufError),
    #[error("model bind: {0}")]
    Load(#[from] LoadError),
    #[error("metal model load: {0}")]
    MetalModel(#[from] MfError),
    #[error("tokenizer load: {0}")]
    Tokenizer(#[from] TokError),
    #[error("sequence max_context_tokens must be >= 1")]
    EmptySequenceCapacity,
    #[error(
        "sequence position mismatch: caller position {caller_position}, tracked position {tracked_position}"
    )]
    SequencePositionMismatch {
        caller_position: usize,
        tracked_position: usize,
    },
    #[error(
        "sequence capacity exceeded: position {position} + {n_tokens} tokens > max_context_tokens {max_context_tokens}"
    )]
    SequenceCapacityExceeded {
        position: usize,
        n_tokens: usize,
        max_context_tokens: usize,
    },
    #[error("prefix snapshot identity mismatch: expected {expected:?}, snapshot {snapshot:?}")]
    SnapshotIdentityMismatch {
        expected: SnapshotIdentity,
        snapshot: SnapshotIdentity,
    },
    #[error("prefix snapshot logits length {got} != vocab size {expected}")]
    PrefixLogitsLengthMismatch { got: usize, expected: usize },
    #[error("prefix snapshot validation: {0}")]
    SnapshotValidation(#[from] SnapshotValidationError),
    #[error("sequence belongs to a different loaded model")]
    SequenceModelMismatch,
    #[error("checkpoint compatibility identity: {0}")]
    CheckpointIdentity(#[from] CheckpointIdentityError),
    #[error("durable checkpoint store: {0}")]
    CheckpointStore(#[from] CheckpointStoreError),
    #[error("checkpoint size estimate overflow")]
    CheckpointSizeOverflow,
}

struct RuntimeInner {
    ctx: MetalContext,
}

#[derive(Debug)]
struct ModelOwnerToken;

fn ensure_same_model_owner(
    expected: &Arc<ModelOwnerToken>,
    actual: &Arc<ModelOwnerToken>,
) -> Result<(), RuntimeError> {
    if !Arc::ptr_eq(expected, actual) {
        return Err(RuntimeError::SequenceModelMismatch);
    }
    Ok(())
}

/// Backend/device execution environment.
///
/// `Runtime` owns the reusable Metal device resources. It deliberately does
/// not own models or sequence state, so benchmarks and future applications can
/// make cold-load, warm-model, and warm-session boundaries explicit.
#[derive(Clone)]
pub struct Runtime {
    inner: Arc<RuntimeInner>,
}

impl Runtime {
    /// Create a runtime backed by the system default Metal device.
    pub fn metal() -> Result<Self, RuntimeError> {
        Ok(Self {
            inner: Arc::new(RuntimeInner {
                ctx: MetalContext::new()?,
            }),
        })
    }

    /// Alias for callers that prefer constructor-style naming.
    pub fn new_metal() -> Result<Self, RuntimeError> {
        Self::metal()
    }

    /// Access the underlying Metal context for low-level paths.
    pub fn context(&self) -> &MetalContext {
        &self.inner.ctx
    }

    /// Human-readable device description.
    pub fn describe(&self) -> String {
        self.context().describe()
    }

    /// Open a GGUF, bind its model metadata, and load resident Metal weights.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<LoadedModel, RuntimeError> {
        self.load_model_with_config(path, LoadedModelConfig::default())
    }

    pub fn load_model_with_config(
        &self,
        path: impl AsRef<Path>,
        config: LoadedModelConfig,
    ) -> Result<LoadedModel, RuntimeError> {
        self.load_model_with_intent(path, config, ModelLoadIntent::ForceOnly)
    }

    /// Load for a disposable single-turn request, permitting authenticated
    /// cold-load policies that remain disabled for reusable model instances.
    pub fn load_model_for_disposable_single_turn_with_config(
        &self,
        path: impl AsRef<Path>,
        config: LoadedModelConfig,
    ) -> Result<LoadedModel, RuntimeError> {
        self.load_model_with_intent(path, config, ModelLoadIntent::DisposableSingleTurn)
    }

    fn load_model_with_intent(
        &self,
        path: impl AsRef<Path>,
        config: LoadedModelConfig,
        intent: ModelLoadIntent,
    ) -> Result<LoadedModel, RuntimeError> {
        let path = path.as_ref();
        let gguf = GgufFile::open(path)?;
        let bound = Model::from_gguf(&gguf)?;
        let prepared = MetalModel::prepare_load_with_options(
            self.context(),
            &gguf,
            &bound,
            intent.metal_options(),
        )?;
        let prefetch_outcome = apply_prefetch_policy(&gguf, &config, prepared.prefetch_advice());
        let identity_shards = snapshot_shard_identity_inputs(&gguf);
        let metal_model = MetalModel::load_prepared(prepared)?;
        Ok(LoadedModel {
            runtime: self.clone(),
            path: path.to_path_buf(),
            gguf,
            metal_model,
            identity_shards,
            identity_parts: OnceLock::new(),
            prefix_cache: Mutex::new(PrefixCache::with_max_bytes(config.prefix_cache_max_bytes)),
            owner: Arc::new(ModelOwnerToken),
            prefetch_outcome,
        })
    }
}

/// Runs the prefetch policy against the freshly-opened GGUF and returns
/// a per-shard outcome record for observability. On any I/O error we
/// log via `tracing::warn` and mark the shard as `skipped: true` — a
/// prefetch failure is a latency regression at worst, never a
/// correctness issue, so it should not abort the load.
fn apply_prefetch_policy(
    gguf: &GgufFile,
    config: &LoadedModelConfig,
    advice: MetalLoadPrefetchAdvice,
) -> PrefetchOutcome {
    let started = Instant::now();
    let action = effective_prefetch_action(config.prefetch_policy, advice);

    if let PrefetchAction::Suppressed {
        reason: PrefetchSuppressionReason::AuthenticatedDisposableAutoA3bDirectPread,
    } = action
    {
        eprintln!(concat!(
            "[runtime-prefetch] schema=1 configured=cold-only action=suppressed ",
            "reason=authenticated-disposable-auto-a3b-direct-pread ",
            "profile=a3b-q4km-v1 population=pread"
        ));
        return PrefetchOutcome::suppressed(config.prefetch_policy, action);
    }

    let mut shards: Vec<ShardPrefetch> = Vec::with_capacity(gguf.shards.len());

    if matches!(config.prefetch_policy, PrefetchPolicy::Off) {
        return PrefetchOutcome {
            policy: config.prefetch_policy,
            action,
            shards,
            total_wall: started.elapsed(),
        };
    }

    let workers = if config.prefetch_workers == 0 {
        DEFAULT_WORKERS
    } else {
        config.prefetch_workers
    };
    let chunk = if config.prefetch_chunk_bytes == 0 {
        DEFAULT_CHUNK_BYTES
    } else {
        config.prefetch_chunk_bytes
    };

    for shard in &gguf.shards {
        shards.push(apply_shard_prefetch(shard, config, workers, chunk));
    }

    PrefetchOutcome {
        policy: config.prefetch_policy,
        action,
        shards,
        total_wall: started.elapsed(),
    }
}
/// Per-shard prefetch. Uses the shard's retained file description
/// (no path reopen, no TOCTOU).
///
/// **Fail-open policy on probe failure.** Both the residency probe and
/// the memory-headroom probe are diagnostic; a failure returns fresh
/// unknown state, not a veto. Codex jam session 019f8b66 worked
/// through the expected value: skipping-on-probe-failure trades
/// certain small waste (a fully warm reread measured at ~170 ms) for potentially
/// forgoing large wins (~18 s on 27B cold). The failed probe's error
/// is recorded in `skipped_reason` for diagnosis, and the prefetch
/// is attempted anyway. If the actual prefetch also fails, that
/// path records the real I/O error.
///
/// **Memory-headroom is telemetry-only in this build.** A hard skip
/// gate needs a controlled pressure experiment first (A3B under three
/// admission regimes) to identify the actual regression boundary.
/// Until then we compute the conservative bound, log if it would fail,
/// but let the prefetch proceed. On Britt's 128 GiB workstation the
/// naive `missing + shard + margin` formula would silently prevent
/// the measured 5× first-byte wins because `available_memory` on an
/// active desktop is often < 30 GiB.
fn apply_shard_prefetch(
    shard: &GgufShard,
    config: &LoadedModelConfig,
    workers: usize,
    chunk: usize,
) -> ShardPrefetch {
    let mut record = ShardPrefetch {
        path: shard.path.clone(),
        bytes_returned: 0,
        wall: Duration::ZERO,
        pre_resident_fraction: 0.0,
        skipped: false,
        skipped_reason: None,
    };

    // Explicit-Always callers opt out of the residency probe entirely.
    // ColdOnly needs the residency to decide, but a probe failure
    // resolves fail-open (proceed with prefetch) rather than skip.
    let need_residency = !matches!(config.prefetch_policy, PrefetchPolicy::Always);
    let residency = if need_residency {
        match probe_fd_residency(shard.file.as_ref()) {
            Ok(r) => {
                record.pre_resident_fraction = r.resident_fraction();
                Some(r)
            }
            Err(e) => {
                tracing::warn!(
                    shard = %shard.path.display(),
                    error = %e,
                    "prefetch: residency probe failed; attempting anyway (fail-open)"
                );
                // Record the failure for observability, but proceed.
                record.skipped_reason = Some(format!(
                    "residency probe failed (fail-open, prefetch attempted): {e}"
                ));
                None
            }
        }
    } else {
        None
    };

    let should_prefetch = match config.prefetch_policy {
        PrefetchPolicy::Off => false,
        PrefetchPolicy::Always => true,
        PrefetchPolicy::ColdOnly { threshold } => match residency {
            Some(r) => r.resident_fraction() < threshold.value(),
            // Probe failed under ColdOnly: fail-open (attempt prefetch).
            None => true,
        },
    };

    if !should_prefetch {
        record.skipped = true;
        record.skipped_reason = Some(format!(
            "residency {:.1}% >= threshold ({:?})",
            record.pre_resident_fraction * 100.0,
            config.prefetch_policy
        ));
        tracing::info!(
            shard = %shard.path.display(),
            resident_pct = record.pre_resident_fraction * 100.0,
            "prefetch: skipped (already warm)"
        );
        return record;
    }

    // Memory-headroom telemetry (see fn-doc for why it does not gate).
    // The `+ shard_size` term models the copied-storage case (Metal
    // allocates fresh dest bytes while we still hold the source in
    // page cache); it is over-cautious for retained storage, which is
    // the common dense-model path. Refactoring the loader to bind
    // Model and build a MetalLoadPlan before prefetch would let us
    // pick the right formula per storage plan; captured as a follow-up.
    let shard_size = shard.mmap_len() as u64;
    let missing_bytes = ((1.0 - record.pre_resident_fraction) * shard_size as f64) as u64;
    let margin = config.effective_prefetch_min_headroom_bytes();
    let conservative_bound = missing_bytes
        .saturating_add(shard_size)
        .saturating_add(margin);
    match available_memory_bytes() {
        Ok(available) => {
            if available < conservative_bound {
                tracing::warn!(
                    shard = %shard.path.display(),
                    available_mib = available / (1 << 20),
                    conservative_bound_mib = conservative_bound / (1 << 20),
                    missing_mib = missing_bytes / (1 << 20),
                    shard_mib = shard_size / (1 << 20),
                    margin_mib = margin / (1 << 20),
                    "prefetch: available memory below conservative bound; proceeding anyway (telemetry-only until validated)"
                );
            } else {
                tracing::debug!(
                    shard = %shard.path.display(),
                    available_mib = available / (1 << 20),
                    conservative_bound_mib = conservative_bound / (1 << 20),
                    "prefetch: memory headroom adequate for conservative bound"
                );
            }
        }
        Err(e) => {
            tracing::warn!(
                shard = %shard.path.display(),
                error = %e,
                "prefetch: headroom probe failed (fail-open; proceeding)"
            );
        }
    }

    let phase = Instant::now();
    match prefetch_fd(shard.file.as_ref(), workers, chunk) {
        Ok(report) => {
            record.bytes_returned = report.bytes;
            record.wall = phase.elapsed();
            tracing::info!(
                shard = %shard.path.display(),
                bytes = report.bytes,
                wall_ms = record.wall.as_secs_f64() * 1e3,
                effective_gbps = report.effective_bytes_per_sec() / 1e9,
                "prefetch: shard warmed"
            );
        }
        Err(e) => {
            record.skipped = true;
            record.skipped_reason = Some(format!("prefetch io: {e}"));
            record.wall = phase.elapsed();
            tracing::warn!(
                shard = %shard.path.display(),
                error = %e,
                "prefetch: shard failed; falling back to demand paging"
            );
        }
    }
    record
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_owner_tokens_reject_cross_model_state() {
        let first = Arc::new(ModelOwnerToken);
        let same = Arc::clone(&first);
        let second = Arc::new(ModelOwnerToken);
        assert!(ensure_same_model_owner(&first, &same).is_ok());
        assert!(matches!(
            ensure_same_model_owner(&first, &second),
            Err(RuntimeError::SequenceModelMismatch)
        ));
    }

    #[test]
    fn model_load_intent_scopes_parallel_copy_auto_admission() {
        assert!(
            !ModelLoadIntent::ForceOnly
                .metal_options()
                .auto_parallel_copy_a3b
        );
        assert!(
            ModelLoadIntent::DisposableSingleTurn
                .metal_options()
                .auto_parallel_copy_a3b
        );
    }

    #[test]
    fn prefetch_action_selector_table_is_fail_closed() {
        let cold_only = PrefetchPolicy::cold_only(DEFAULT_COLD_ONLY_THRESHOLD).unwrap();
        let suppress = MetalLoadPrefetchAdvice::SuppressColdOnlyAuthenticatedA3bDirectPread;
        let preserve = MetalLoadPrefetchAdvice::PreserveConfiguredPolicy;
        let suppressed = PrefetchAction::Suppressed {
            reason: PrefetchSuppressionReason::AuthenticatedDisposableAutoA3bDirectPread,
        };

        let cases = [
            ("cold-only-authenticated", cold_only, suppress, suppressed),
            (
                "cold-only-preserve",
                cold_only,
                preserve,
                PrefetchAction::ConfiguredPolicy,
            ),
            (
                "off-authenticated",
                PrefetchPolicy::Off,
                suppress,
                PrefetchAction::ConfiguredPolicy,
            ),
            (
                "always-authenticated",
                PrefetchPolicy::Always,
                suppress,
                PrefetchAction::ConfiguredPolicy,
            ),
        ];
        for (name, policy, advice, expected) in cases {
            assert_eq!(
                effective_prefetch_action(policy, advice),
                expected,
                "{name}"
            );
        }

        let outcome = PrefetchOutcome::suppressed(cold_only, suppressed);
        assert_eq!(outcome.policy, cold_only);
        assert_eq!(outcome.action, suppressed);
        assert!(outcome.shards.is_empty());
        assert_eq!(outcome.bytes_returned_total(), 0);
        assert_eq!(outcome.shards_prefetched(), 0);
        assert_eq!(outcome.shards_skipped(), 0);
        assert_eq!(outcome.total_wall, Duration::ZERO);
    }

    #[test]
    fn residency_threshold_accepts_bounds() {
        assert!(ResidencyThreshold::new(0.0).is_ok());
        assert!(ResidencyThreshold::new(1.0).is_ok());
        assert!(ResidencyThreshold::new(0.5).is_ok());
    }

    #[test]
    fn residency_threshold_rejects_nan_inf_and_out_of_range() {
        assert!(ResidencyThreshold::new(f64::NAN).is_err());
        assert!(ResidencyThreshold::new(f64::INFINITY).is_err());
        assert!(ResidencyThreshold::new(f64::NEG_INFINITY).is_err());
        assert!(ResidencyThreshold::new(-0.001).is_err());
        assert!(ResidencyThreshold::new(1.001).is_err());
        assert!(ResidencyThreshold::new(-1e100).is_err());
    }

    #[test]
    fn residency_threshold_equality_is_reflexive_and_transitive() {
        // Reflexivity is the property that makes `Eq` sound on a
        // f64-containing type. NaN would break this; the constructor
        // excludes NaN.
        let a = ResidencyThreshold::new(0.42).unwrap();
        let b = ResidencyThreshold::new(0.42).unwrap();
        let c = ResidencyThreshold::new(0.42).unwrap();
        assert_eq!(a, a);
        assert_eq!(a, b);
        assert_eq!(b, c);
        // Order sanity: different values are unequal.
        let d = ResidencyThreshold::new(0.43).unwrap();
        assert_ne!(a, d);
    }

    #[test]
    fn prefetch_policy_cold_only_convenience_constructor() {
        let ok = PrefetchPolicy::cold_only(0.5).unwrap();
        assert!(matches!(ok, PrefetchPolicy::ColdOnly { .. }));
        assert!(PrefetchPolicy::cold_only(f64::NAN).is_err());
        assert!(PrefetchPolicy::cold_only(1.1).is_err());
    }

    #[test]
    fn prefetch_policy_default_is_off_when_constructed_directly() {
        // Zero-value default of the enum type stays Off — callers who
        // construct `PrefetchPolicy` from scratch (rather than via
        // `LoadedModelConfig::default`) get no prefetch behaviour.
        assert_eq!(PrefetchPolicy::default(), PrefetchPolicy::Off);
    }

    #[test]
    fn loaded_model_config_default_prefetches_cold_only() {
        // `Runtime::load_model` (no config arg) uses
        // `LoadedModelConfig::default`, which since the pread-warmup
        // spike promotes cache warming to ColdOnly. Prior behaviour
        // (Off) is still available via explicit config override.
        // If a caller regresses on warm-load latency after this change,
        // see the ColdOnly gate in `apply_shard_prefetch` and the
        // fallback path in `runtime_load_spike` for the diagnostic
        // signal.
        let config = LoadedModelConfig::default();
        match config.prefetch_policy {
            PrefetchPolicy::ColdOnly { threshold } => {
                assert_eq!(threshold.value(), DEFAULT_COLD_ONLY_THRESHOLD);
            }
            other => panic!("expected ColdOnly, got {other:?}"),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadedModelConfig {
    pub prefix_cache_max_bytes: u64,
    /// Policy for warming the OS unified buffer cache during model load.
    /// [`LoadedModelConfig::default`] selects residency-gated `ColdOnly`;
    /// [`PrefetchPolicy::default`] itself remains `Off`.
    pub prefetch_policy: PrefetchPolicy,
    /// Number of parallel `pread` workers per shard when prefetching.
    /// Zero uses [`crate::prefetch::DEFAULT_WORKERS`] (4).
    pub prefetch_workers: usize,
    /// Per-worker scratch buffer size in bytes. Zero uses
    /// [`crate::prefetch::DEFAULT_CHUNK_BYTES`] (16 MiB).
    pub prefetch_chunk_bytes: usize,
    /// Additional telemetry margin above theoretical prefetch + destination
    /// sizes. Zero uses [`DEFAULT_PREFETCH_MIN_HEADROOM_BYTES`] (1 GiB).
    pub prefetch_min_headroom_bytes: u64,
}

impl LoadedModelConfig {
    pub(crate) fn effective_prefetch_min_headroom_bytes(&self) -> u64 {
        if self.prefetch_min_headroom_bytes == 0 {
            DEFAULT_PREFETCH_MIN_HEADROOM_BYTES
        } else {
            self.prefetch_min_headroom_bytes
        }
    }
}

/// Telemetry margin above `missing_bytes + shard_size`. The current build logs
/// a failed conservative bound but does not veto prefetch. One GiB covers kernel
/// bookkeeping and unrelated allocations in that diagnostic estimate.
pub const DEFAULT_PREFETCH_MIN_HEADROOM_BYTES: u64 = 1024 * 1024 * 1024;

/// Default cache-residency threshold for [`PrefetchPolicy::ColdOnly`]
/// used by [`LoadedModelConfig::default`].
///
/// Value `0.9` says: prefetch if the shard is less than 90% resident.
///
/// Reasoning (analytical, verified against measured cold/warm endpoints;
/// intermediate-residency sweep is a follow-up):
///
/// * Cold-drive prefetch throughput: ~6.7 GB/s (4-worker parallel pread
///   on Apple Silicon internal SSD; measured on 27B and A3B).
/// * Metal demand-page throughput: ~0.5–0.7 GB/s (single-thread mmap
///   demand paging; measured on the same drive).
/// * Ratio: prefetch is ~10× faster per byte than demand paging.
/// * Therefore prefetching a fraction `f = (1 - resident_fraction)` of
///   cold bytes wins vs. paying `f × 10× per byte` for Metal to
///   demand-page them, as long as `f > 0.1`, i.e. the file is < 90%
///   resident.
///
/// The prior default was `0.5`, chosen without this analysis. `0.5`
/// would skip prefetch on files at 60–89% resident even though prefetch
/// still wins there. Numerical values above ~0.9 have small effective
/// difference because at very-warm the prefetch itself is a short RAM
/// copy (measured Wa vs Wo on 27B: only +170 ms overhead when file was
/// 100% resident with `Always`), so the loss from an unnecessary
/// prefetch is bounded.
pub const DEFAULT_COLD_ONLY_THRESHOLD: f64 = 0.9;

/// A validated cache-residency threshold in `[0.0, 1.0]`.
///
/// Rejects NaN, negative, subnormal-but-negative, and >1 values at
/// construction. Wrapping the raw `f64` (rather than exposing it in
/// [`PrefetchPolicy`]) lets the enum soundly derive `Eq`, `PartialEq`,
/// and `Hash` even though `f64` normally cannot: the newtype's manual
/// impls are safe because construction excludes NaN.
#[derive(Clone, Copy, Debug)]
pub struct ResidencyThreshold(f64);

/// Error returned by [`ResidencyThreshold::new`] when the caller-supplied
/// value is not a finite fraction in `[0.0, 1.0]`.
#[derive(Debug, thiserror::Error)]
#[error("residency threshold must be finite in [0.0, 1.0]; got {0}")]
pub struct InvalidResidencyThreshold(pub f64);

impl ResidencyThreshold {
    /// Construct a threshold from a fraction in `[0.0, 1.0]`. Rejects
    /// non-finite (NaN or infinity) and out-of-range values.
    pub fn new(value: f64) -> Result<Self, InvalidResidencyThreshold> {
        if !value.is_finite() || value < 0.0 || value > 1.0 {
            return Err(InvalidResidencyThreshold(value));
        }
        Ok(Self(value))
    }

    /// Extract the underlying fraction. Guaranteed to be finite and
    /// in `[0.0, 1.0]`.
    pub const fn value(self) -> f64 {
        self.0
    }
}

// Safe: `new` rejects NaN, so `to_bits` equality coincides with `==`
// on the stored value AND reflexivity holds. No infinities or NaNs to
// worry about.
impl PartialEq for ResidencyThreshold {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for ResidencyThreshold {}
impl std::hash::Hash for ResidencyThreshold {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

/// Cache-warming policy applied by [`Runtime::load_model_with_config`]
/// between GGUF validation and Metal weight resident-load.
///
/// Rationale: on cold storage, `MetalModel::load_with_options` is
/// bounded by mmap demand-paging (~0.5 GB/s on macOS APFS). Parallel
/// `pread` warmup unlocks the drive's true ceiling (measured 6-7 GB/s
/// on Apple Silicon internal SSD in the load_spike experiment).
///
/// See `docs/PLAN.md` v0.591 for prior evidence that unconditional
/// whole-shard prefault costs 1.8-1.9s and degrades warm-cache paths.
/// `ColdOnly` gates on `mincore` residency to preserve warm-load
/// performance; this is the recommended default when the loader
/// callsite doesn't otherwise know cache state.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PrefetchPolicy {
    /// Do not prefetch. Loader behaves exactly as before this option
    /// existed.
    Off,
    /// Prefetch regardless of current residency.
    Always,
    /// Prefetch only when `mincore`-reported unified-cache residency
    /// is below `threshold`. Use [`PrefetchPolicy::cold_only`] to
    /// construct with a validated fraction.
    ColdOnly { threshold: ResidencyThreshold },
}

impl PrefetchPolicy {
    /// Convenience constructor for [`PrefetchPolicy::ColdOnly`] that
    /// validates `threshold` at the call site rather than forcing
    /// callers to hand-construct a [`ResidencyThreshold`].
    pub fn cold_only(threshold: f64) -> Result<Self, InvalidResidencyThreshold> {
        Ok(Self::ColdOnly {
            threshold: ResidencyThreshold::new(threshold)?,
        })
    }
}

impl Default for PrefetchPolicy {
    fn default() -> Self {
        Self::Off
    }
}

/// Per-shard observability record of what the prefetch policy did on
/// one `load_model_*` call.
#[derive(Clone, Debug)]
pub struct ShardPrefetch {
    pub path: PathBuf,
    /// Bytes returned by `pread` calls. Warm cache hits count here too;
    /// use `pid_metrics::PidDelta` for physical disk bytes.
    pub bytes_returned: u64,
    /// Wall-clock time of the prefetch or the residency-probe if
    /// skipped.
    pub wall: Duration,
    /// mincore-reported residency of the shard *before* the arm.
    pub pre_resident_fraction: f64,
    pub skipped: bool,
    pub skipped_reason: Option<String>,
}

/// Result of running the configured [`PrefetchPolicy`] during a
/// `Runtime::load_model_*` call.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefetchSuppressionReason {
    AuthenticatedDisposableAutoA3bDirectPread,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PrefetchAction {
    ConfiguredPolicy,
    Suppressed { reason: PrefetchSuppressionReason },
}

fn effective_prefetch_action(
    policy: PrefetchPolicy,
    advice: MetalLoadPrefetchAdvice,
) -> PrefetchAction {
    match (policy, advice) {
        (
            PrefetchPolicy::ColdOnly { .. },
            MetalLoadPrefetchAdvice::SuppressColdOnlyAuthenticatedA3bDirectPread,
        ) => PrefetchAction::Suppressed {
            reason: PrefetchSuppressionReason::AuthenticatedDisposableAutoA3bDirectPread,
        },
        _ => PrefetchAction::ConfiguredPolicy,
    }
}

#[derive(Clone, Debug)]
pub struct PrefetchOutcome {
    pub policy: PrefetchPolicy,
    pub action: PrefetchAction,
    pub shards: Vec<ShardPrefetch>,
    /// Total wall time of the prefetch phase across all shards
    /// (including residency probes for skipped shards), or zero when
    /// prefetch is suppressed before that phase begins.
    pub total_wall: Duration,
}

impl PrefetchOutcome {
    fn suppressed(policy: PrefetchPolicy, action: PrefetchAction) -> Self {
        debug_assert!(matches!(action, PrefetchAction::Suppressed { .. }));
        Self {
            policy,
            action,
            shards: Vec::new(),
            total_wall: Duration::ZERO,
        }
    }

    pub fn bytes_returned_total(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes_returned).sum()
    }
    pub fn shards_prefetched(&self) -> usize {
        self.shards.iter().filter(|s| !s.skipped).count()
    }
    pub fn shards_skipped(&self) -> usize {
        self.shards.iter().filter(|s| s.skipped).count()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ModelLoadIntent {
    ForceOnly,
    DisposableSingleTurn,
}

impl ModelLoadIntent {
    fn metal_options(self) -> MetalModelLoadOptions {
        MetalModelLoadOptions {
            auto_parallel_copy_a3b: self == Self::DisposableSingleTurn,
        }
    }
}

impl Default for LoadedModelConfig {
    /// The default `prefetch_policy` is [`PrefetchPolicy::ColdOnly`]
    /// with [`DEFAULT_COLD_ONLY_THRESHOLD`]. Rationale: measured 5x
    /// first-byte win on cold storage of dense 27B, 1.5x on A3B MoE,
    /// with correctly-gated skip on warm cache (`Wc` arm actually 60 ms
    /// faster than baseline `Wo` at first-byte). Callers that don't
    /// want cache warming (bench harnesses that specifically want to
    /// measure the mmap demand-paging path, or callers targeting
    /// memory-tight systems where we don't yet have a validated
    /// admission gate) can opt out via
    /// `LoadedModelConfig { prefetch_policy: PrefetchPolicy::Off, .. }`.
    fn default() -> Self {
        Self {
            prefix_cache_max_bytes: DEFAULT_MAX_BYTES,
            prefetch_policy: PrefetchPolicy::cold_only(DEFAULT_COLD_ONLY_THRESHOLD)
                .expect("DEFAULT_COLD_ONLY_THRESHOLD is a valid fraction"),
            prefetch_workers: 0,
            prefetch_chunk_bytes: 0,
            prefetch_min_headroom_bytes: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PrefixCacheInsert {
    pub snapshot_bytes: u64,
    pub stats: PrefixCacheStats,
}

#[derive(Clone, Debug)]
pub struct PrefixCacheRestore {
    pub matched_prefix_len: usize,
    pub restored_prefix_len: usize,
    pub exact: bool,
    pub exact_final_logits: Option<Vec<f32>>,
    /// Cache-index accounting captured at lookup, before the unlocked restore.
    pub stats_at_lookup: PrefixCacheStats,
}

/// An immutable CPU snapshot which may retain hundreds of MiB until dropped.
/// RAM insertion shares its arenas; it does not transfer their ownership.
pub struct PreparedCheckpoint {
    owner: Arc<ModelOwnerToken>,
    snapshot: Arc<SessionSnapshot>,
    max_context_tokens: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CheckpointBoundarySizeEstimate {
    pub snapshot_bytes: u64,
    pub record_bytes: u64,
}

impl PreparedCheckpoint {
    pub fn snapshot_bytes(&self) -> u64 {
        self.snapshot.n_bytes()
    }

    pub fn matched_prefix_len(&self) -> usize {
        self.snapshot.matched_prefix_len()
    }

    pub fn restored_prefix_len(&self) -> usize {
        self.snapshot.prefix_len()
    }

    pub fn has_pending_token(&self) -> bool {
        self.snapshot.pending_token.is_some()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DurableLookupTelemetry {
    pub matched_prefix_len: usize,
    pub restored_prefix_len: usize,
    pub exact: bool,
    pub candidates_examined: usize,
    pub corrupt_entries_removed: usize,
    pub touched: bool,
}

pub struct DurablePrefixRestore {
    pub matched_prefix_len: usize,
    pub restored_prefix_len: usize,
    pub exact: bool,
    pub exact_final_logits: Option<Vec<f32>>,
    checkpoint: PreparedCheckpoint,
}

pub struct DurableRestoreAttempt {
    pub compatibility: CheckpointCompatibilityReport,
    pub lookup: DurableLookupTelemetry,
    pub hit: Option<DurablePrefixRestore>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DurablePublishReport {
    pub compatibility: CheckpointCompatibilityReport,
    pub store: PublishReport,
}

/// One model loaded into a [`Runtime`].
///
/// This is the expensive model-load boundary: the GGUF is mmap'd and the Metal
/// weights are resident. Per-request mutable state is created separately as a
/// [`Sequence`].
pub struct LoadedModel {
    runtime: Runtime,
    path: PathBuf,
    gguf: GgufFile,
    metal_model: MetalModel,
    identity_shards: Vec<SnapshotShardIdentityInput>,
    identity_parts: OnceLock<(u64, u64)>,
    prefix_cache: Mutex<PrefixCache>,
    owner: Arc<ModelOwnerToken>,
    prefetch_outcome: PrefetchOutcome,
}

impl LoadedModel {
    /// Path used to load this model.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Underlying GGUF view, retained for metadata and tokenizer access.
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    /// What the configured [`PrefetchPolicy`] did during this load.
    /// Always present; `PrefetchPolicy::Off` yields an outcome with an
    /// empty `shards` vec and zero `total_wall`.
    pub fn prefetch_outcome(&self) -> &PrefetchOutcome {
        &self.prefetch_outcome
    }

    /// Resident Metal weights.
    pub fn metal_model(&self) -> &MetalModel {
        &self.metal_model
    }

    /// Runtime backing this loaded model.
    pub fn runtime(&self) -> &Runtime {
        &self.runtime
    }

    /// Metal context backing this loaded model.
    pub fn context(&self) -> &MetalContext {
        self.runtime.context()
    }

    /// Architecture metadata copied into the resident Metal model.
    pub fn arch(&self) -> Arch {
        self.metal_model.arch
    }

    /// Create a tokenizer from the same GGUF metadata as this loaded model.
    ///
    /// Tokenizer construction remains explicit for phase 1 so synthetic
    /// benchmark rows do not pay tokenizer setup unless they need it.
    pub fn tokenizer(&self) -> Result<Tokenizer, RuntimeError> {
        Ok(Tokenizer::from_gguf(&self.gguf)?)
    }

    /// Create fresh per-sequence state with the requested context capacity.
    pub fn create_sequence(&self, config: SequenceConfig) -> Result<Sequence, RuntimeError> {
        if config.max_context_tokens == 0 {
            return Err(RuntimeError::EmptySequenceCapacity);
        }
        let state =
            MetalSession::fresh(self.context(), &self.metal_model, config.max_context_tokens)?;
        Ok(Sequence {
            max_context_tokens: config.max_context_tokens,
            position: 0,
            state,
            owner: Arc::clone(&self.owner),
        })
    }

    /// Borrowed execution driver over this loaded model.
    pub fn forward(&self) -> MetalForward<'_> {
        MetalForward::new(self.context(), &self.metal_model)
    }

    /// Resolve the cheap metadata compatibility identity used by in-process
    /// policy and RAM snapshot matching. This is not a content digest; durable
    /// checkpoints must use [`Self::checkpoint_compatibility`] instead.
    pub fn metadata_compatibility_ids(&self) -> (u64, u64) {
        *self.identity_parts.get_or_init(|| {
            snapshot_identity_parts(&self.gguf, self.metal_model.arch, &self.identity_shards)
        })
    }

    pub fn snapshot_identity(&self, sequence: &Sequence) -> Result<SnapshotIdentity, RuntimeError> {
        self.ensure_owns(sequence)?;
        let (model_id, tokenizer_id) = self.metadata_compatibility_ids();
        Ok(sequence
            .metal_session()
            .snapshot_identity(model_id, tokenizer_id))
    }

    /// Resolve the strong durable-checkpoint identity lazily. A cache hit only
    /// validates retained shard file descriptions and reads one tiny entry;
    /// complete shard bytes are hashed only on a cache miss.
    pub fn checkpoint_compatibility(
        &self,
        sequence: &Sequence,
        cache: &CheckpointIdentityCache,
    ) -> Result<CheckpointCompatibilityReport, RuntimeError> {
        self.ensure_owns(sequence)?;
        Ok(checkpoint_compatibility(
            &self.gguf,
            sequence.snapshot_abi(),
            cache,
        )?)
    }

    fn ensure_owns(&self, sequence: &Sequence) -> Result<(), RuntimeError> {
        ensure_same_model_owner(&self.owner, &sequence.owner)
    }

    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        self.prefix_cache.lock().stats()
    }

    pub fn set_prefix_cache_max_bytes(&self, max_bytes: u64) -> PrefixCacheStats {
        let mut cache = self.prefix_cache.lock();
        cache.set_max_bytes(max_bytes);
        cache.stats()
    }

    pub fn clear_prefix_cache(&self) -> PrefixCacheStats {
        let mut cache = self.prefix_cache.lock();
        cache.clear();
        cache.stats()
    }

    pub fn cache_sequence_prefix(
        &self,
        sequence: &Sequence,
        prefix_tokens: Vec<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> Result<PrefixCacheInsert, RuntimeError> {
        self.cache_sequence_boundary(sequence, prefix_tokens, None, final_logits)
    }

    /// Cache a canonical request boundary. `prefix_tokens` must be the tokens
    /// already consumed into `sequence`; `pending_token` is an optional final
    /// token that was emitted but deliberately not transitioned.
    pub fn cache_sequence_boundary(
        &self,
        sequence: &Sequence,
        prefix_tokens: Vec<i32>,
        pending_token: Option<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> Result<PrefixCacheInsert, RuntimeError> {
        let prepared =
            self.prepare_checkpoint_boundary(sequence, prefix_tokens, pending_token, final_logits)?;
        self.cache_prepared_checkpoint(&prepared)
    }

    /// Capture one immutable sequence boundary for RAM insertion, deferred
    /// durable publication, or both. The snapshot remains independent of later
    /// sequence mutation.
    pub fn prepare_checkpoint_boundary(
        &self,
        sequence: &Sequence,
        prefix_tokens: Vec<i32>,
        pending_token: Option<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> Result<PreparedCheckpoint, RuntimeError> {
        self.ensure_owns(sequence)?;
        sequence.check_position(prefix_tokens.len())?;
        if let Some(logits) = final_logits.as_ref() {
            let expected = self.metal_model.arch.vocab_size as usize;
            if logits.len() != expected {
                return Err(RuntimeError::PrefixLogitsLengthMismatch {
                    got: logits.len(),
                    expected,
                });
            }
        }
        let identity = self.snapshot_identity(sequence)?;
        let mut snap = sequence.snapshot(identity.clone(), prefix_tokens, final_logits)?;
        snap.pending_token = pending_token;
        snap.validate_for_restore(
            &identity,
            sequence.max_context_tokens(),
            Some(self.metal_model.arch.vocab_size as usize),
        )?;
        Ok(PreparedCheckpoint {
            owner: Arc::clone(&self.owner),
            snapshot: Arc::new(snap),
            max_context_tokens: sequence.max_context_tokens(),
        })
    }

    /// Estimate the CPU payload retained by a prepared boundary before copying
    /// any Metal state. Encoded records add a small fixed header and digest.
    pub fn estimate_checkpoint_boundary_sizes(
        &self,
        sequence: &Sequence,
        prefix_len: usize,
        has_pending_token: bool,
        has_final_logits: bool,
    ) -> Result<CheckpointBoundarySizeEstimate, RuntimeError> {
        self.ensure_owns(sequence)?;
        sequence.check_position(prefix_len)?;
        let abi = sequence.snapshot_abi();
        let prefix_len = prefix_len as u128;
        let n_attn = abi.n_attn_layers as u128;
        let n_gdn = abi.n_gdn_layers as u128;
        let kv_bytes = 2u128 * n_attn * prefix_len * abi.kv_bytes_per_token as u128;
        let gdn_bytes = 4u128
            * n_gdn
            * (abi.gdn_conv_elements_per_layer as u128 + abi.gdn_state_elements_per_layer as u128);
        let token_bytes = 4u128 * prefix_len + 4u128 * u128::from(has_pending_token);
        let position_bytes = 8u128 * n_attn;
        let logits_bytes = if has_final_logits {
            4u128 * self.metal_model.arch.vocab_size as u128
        } else {
            0
        };
        let snapshot_bytes =
            u64::try_from(kv_bytes + gdn_bytes + token_bytes + position_bytes + logits_bytes)
                .map_err(|_| RuntimeError::CheckpointSizeOverflow)?;
        let pending_memory_bytes = 4 * u64::from(has_pending_token);
        let record_bytes = snapshot_bytes
            .checked_sub(pending_memory_bytes)
            .and_then(|bytes| bytes.checked_add(SNAPSHOT_RECORD_FIXED_BYTES))
            .ok_or(RuntimeError::CheckpointSizeOverflow)?;
        Ok(CheckpointBoundarySizeEstimate {
            snapshot_bytes,
            record_bytes,
        })
    }

    /// Index a prepared boundary in the process-local cache without cloning its
    /// large state arenas.
    pub fn cache_prepared_checkpoint(
        &self,
        prepared: &PreparedCheckpoint,
    ) -> Result<PrefixCacheInsert, RuntimeError> {
        ensure_same_model_owner(&self.owner, &prepared.owner)?;
        let snapshot_bytes = prepared.snapshot.n_bytes();
        let mut cache = self.prefix_cache.lock();
        cache.insert_shared(Arc::clone(&prepared.snapshot));
        Ok(PrefixCacheInsert {
            snapshot_bytes,
            stats: cache.stats(),
        })
    }

    /// Publish an already captured boundary. Callers can defer this until after
    /// response generation so codec and filesystem durability do not enter TTFT.
    /// `max_record_bytes` bounds one encoded record independently of the store's
    /// aggregate managed-byte budget.
    pub fn publish_prepared_checkpoint(
        &self,
        store: &DurableCheckpointStore,
        prepared: &PreparedCheckpoint,
        max_record_bytes: u64,
    ) -> Result<DurablePublishReport, RuntimeError> {
        ensure_same_model_owner(&self.owner, &prepared.owner)?;
        let compatibility = checkpoint_compatibility(
            &self.gguf,
            prepared.snapshot.identity.abi(),
            &store.identity_cache(),
        )?;
        let store_report = store.publish(
            StoreContext {
                compatibility_id: &compatibility.compatibility_id,
                identity: &prepared.snapshot.identity,
                vocab_size: self.metal_model.arch.vocab_size as usize,
                max_context_tokens: prepared.max_context_tokens,
                max_record_bytes,
            },
            &prepared.snapshot,
        )?;
        Ok(DurablePublishReport {
            compatibility,
            store: store_report,
        })
    }

    /// Restore the longest compatible durable prefix into a fresh sequence.
    /// Clean misses retain identity and lookup telemetry for cold-start policy.
    /// `max_record_bytes` is a per-record allocation bound, not a disk budget.
    pub fn restore_durable_prefix(
        &self,
        store: &DurableCheckpointStore,
        sequence: &mut Sequence,
        request_tokens: &[i32],
        max_record_bytes: u64,
    ) -> Result<DurableRestoreAttempt, RuntimeError> {
        self.ensure_owns(sequence)?;
        sequence.check_position(0)?;
        sequence.ensure_can_append(request_tokens.len())?;
        let identity = self.snapshot_identity(sequence)?;
        let compatibility = self.checkpoint_compatibility(sequence, &store.identity_cache())?;
        let mut lookup = store.lookup(
            StoreContext {
                compatibility_id: &compatibility.compatibility_id,
                identity: &identity,
                vocab_size: self.metal_model.arch.vocab_size as usize,
                max_context_tokens: sequence.max_context_tokens(),
                max_record_bytes,
            },
            request_tokens,
        )?;
        let telemetry = DurableLookupTelemetry {
            matched_prefix_len: lookup.matched_prefix_len,
            restored_prefix_len: lookup.restored_prefix_len,
            exact: lookup.exact,
            candidates_examined: lookup.candidates_examined,
            corrupt_entries_removed: lookup.corrupt_entries_removed,
            touched: lookup.touched,
        };
        let hit = if let Some(snapshot) = lookup.snapshot.take() {
            snapshot.validate_for_restore(
                &identity,
                sequence.max_context_tokens(),
                Some(self.metal_model.arch.vocab_size as usize),
            )?;
            sequence.restore_from_snapshot(&snapshot, &identity)?;
            let snapshot = Arc::new(snapshot);
            let exact_final_logits =
                if lookup.exact && lookup.restored_prefix_len == lookup.matched_prefix_len {
                    snapshot.final_logits.clone()
                } else {
                    None
                };
            Some(DurablePrefixRestore {
                matched_prefix_len: lookup.matched_prefix_len,
                restored_prefix_len: lookup.restored_prefix_len,
                exact: lookup.exact,
                exact_final_logits,
                checkpoint: PreparedCheckpoint {
                    owner: Arc::clone(&self.owner),
                    snapshot,
                    max_context_tokens: sequence.max_context_tokens(),
                },
            })
        } else {
            None
        };
        Ok(DurableRestoreAttempt {
            compatibility,
            lookup: telemetry,
            hit,
        })
    }

    /// Promote a durable restore into the process-local cache without another
    /// state capture or arena clone.
    pub fn cache_durable_restore(
        &self,
        restored: &DurablePrefixRestore,
    ) -> Result<PrefixCacheInsert, RuntimeError> {
        self.cache_prepared_checkpoint(&restored.checkpoint)
    }

    pub fn restore_cached_prefix(
        &self,
        sequence: &mut Sequence,
        request_tokens: &[i32],
    ) -> Result<Option<PrefixCacheRestore>, RuntimeError> {
        self.ensure_owns(sequence)?;
        sequence.check_position(0)?;
        sequence.ensure_can_append(request_tokens.len())?;
        let identity = self.snapshot_identity(sequence)?;
        let (hit, stats) = {
            let mut cache = self.prefix_cache.lock();
            let Some(hit) = cache.lookup_longest_for_completion(&identity, request_tokens) else {
                return Ok(None);
            };
            (hit, cache.stats())
        };
        hit.snapshot.validate_for_restore(
            &identity,
            sequence.max_context_tokens(),
            Some(self.metal_model.arch.vocab_size as usize),
        )?;
        sequence.restore_from_snapshot(&hit.snapshot, &identity)?;
        let exact_final_logits = if hit.exact && hit.restored_prefix_len == hit.matched_prefix_len {
            hit.snapshot.final_logits.clone()
        } else {
            None
        };
        Ok(Some(PrefixCacheRestore {
            matched_prefix_len: hit.matched_prefix_len,
            restored_prefix_len: hit.restored_prefix_len,
            exact: hit.exact,
            exact_final_logits,
            stats_at_lookup: stats,
        }))
    }
}

/// Allocation/configuration for one [`Sequence`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SequenceConfig {
    /// Maximum token positions the sequence state can hold.
    pub max_context_tokens: usize,
}

impl SequenceConfig {
    pub fn new(max_context_tokens: usize) -> Self {
        Self { max_context_tokens }
    }

    pub fn try_new(max_context_tokens: usize) -> Result<Self, RuntimeError> {
        if max_context_tokens == 0 {
            return Err(RuntimeError::EmptySequenceCapacity);
        }
        Ok(Self { max_context_tokens })
    }
}

/// User-facing mutable state for one generated sequence.
///
/// Internally this is the current `MetalSession`: KV cache, GDN state, scratch,
/// ids, logits, and related per-sequence buffers.
pub struct Sequence {
    max_context_tokens: usize,
    position: usize,
    state: MetalSession,
    owner: Arc<ModelOwnerToken>,
}

impl Sequence {
    pub fn snapshot_abi(&self) -> crate::metal_forward::SnapshotAbi {
        self.state.snapshot_abi()
    }

    pub fn max_context_tokens(&self) -> usize {
        self.max_context_tokens
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn remaining_context_tokens(&self) -> usize {
        self.max_context_tokens.saturating_sub(self.position)
    }

    pub fn check_position(&self, caller_position: usize) -> Result<(), RuntimeError> {
        if caller_position != self.position {
            return Err(RuntimeError::SequencePositionMismatch {
                caller_position,
                tracked_position: self.position,
            });
        }
        Ok(())
    }

    pub fn ensure_can_append(&self, n_tokens: usize) -> Result<(), RuntimeError> {
        let end =
            self.position
                .checked_add(n_tokens)
                .ok_or(RuntimeError::SequenceCapacityExceeded {
                    position: self.position,
                    n_tokens,
                    max_context_tokens: self.max_context_tokens,
                })?;
        if end > self.max_context_tokens {
            return Err(RuntimeError::SequenceCapacityExceeded {
                position: self.position,
                n_tokens,
                max_context_tokens: self.max_context_tokens,
            });
        }
        Ok(())
    }

    pub fn advance_by(&mut self, n_tokens: usize) -> Result<(), RuntimeError> {
        self.ensure_can_append(n_tokens)?;
        self.position += n_tokens;
        Ok(())
    }

    fn snapshot(
        &self,
        identity: SnapshotIdentity,
        prefix_tokens: Vec<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> Result<SessionSnapshot, RuntimeError> {
        Ok(self.state.snapshot(identity, prefix_tokens, final_logits)?)
    }

    fn restore_from_snapshot(
        &mut self,
        snapshot: &SessionSnapshot,
        expected_identity: &SnapshotIdentity,
    ) -> Result<(), RuntimeError> {
        if &snapshot.identity != expected_identity {
            return Err(RuntimeError::SnapshotIdentityMismatch {
                expected: expected_identity.clone(),
                snapshot: snapshot.identity.clone(),
            });
        }
        self.check_position(0)?;
        self.ensure_can_append(snapshot.prefix_len())?;
        self.state.restore_from(snapshot, expected_identity)?;
        self.position = snapshot.prefix_len();
        Ok(())
    }

    /// Low-level bridge for the existing Metal execution functions.
    pub fn metal_session(&self) -> &MetalSession {
        &self.state
    }

    /// Low-level mutable bridge outside the safe runtime provenance contract.
    ///
    /// # Safety
    ///
    /// Every state-producing operation must use the same `LoadedModel` that
    /// created this sequence. Mixing another model's executor can create state
    /// that must never be snapshotted or published under this sequence owner.
    pub unsafe fn metal_session_mut(&mut self) -> &mut MetalSession {
        &mut self.state
    }
}

const HASH_OFFSET: u64 = 0xcbf29ce484222325;
const HASH_PRIME: u64 = 0x100000001b3;

struct SnapshotShardIdentityInput {
    path: String,
    mapped_len: u64,
    file_len: Option<u64>,
    modified_nanos: Option<u64>,
}

fn hash_u64(h: &mut u64, value: u64) {
    for byte in value.to_le_bytes() {
        *h ^= byte as u64;
        *h = h.wrapping_mul(HASH_PRIME);
    }
}

fn hash_str(h: &mut u64, value: &str) {
    hash_u64(h, value.len() as u64);
    for byte in value.as_bytes() {
        *h ^= *byte as u64;
        *h = h.wrapping_mul(HASH_PRIME);
    }
}

fn hash_value(h: &mut u64, value: &Value) {
    match value {
        Value::Null => hash_str(h, "null"),
        Value::Bool(v) => {
            hash_str(h, "bool");
            hash_u64(h, u64::from(*v));
        }
        Value::Number(v) => {
            hash_str(h, "number");
            hash_str(h, &v.to_string());
        }
        Value::String(v) => {
            hash_str(h, "string");
            hash_str(h, v);
        }
        Value::Array(xs) => {
            hash_str(h, "array");
            hash_u64(h, xs.len() as u64);
            for x in xs {
                hash_value(h, x);
            }
        }
        Value::Object(map) => {
            hash_str(h, "object");
            let mut keys: Vec<_> = map.keys().collect();
            keys.sort();
            for key in keys {
                hash_str(h, key);
                hash_value(h, &map[key]);
            }
        }
    }
}

fn snapshot_shard_identity_inputs(gguf: &GgufFile) -> Vec<SnapshotShardIdentityInput> {
    gguf.shards
        .iter()
        .map(|shard| {
            let metadata = std::fs::metadata(&shard.path).ok();
            SnapshotShardIdentityInput {
                path: shard.path.display().to_string(),
                mapped_len: shard.mmap.len() as u64,
                file_len: metadata.as_ref().map(std::fs::Metadata::len),
                modified_nanos: metadata
                    .and_then(|value| value.modified().ok())
                    .and_then(|value| value.duration_since(std::time::UNIX_EPOCH).ok())
                    .map(|value| value.as_nanos() as u64),
            }
        })
        .collect()
}

fn snapshot_identity_parts(
    gguf: &GgufFile,
    arch: Arch,
    shards: &[SnapshotShardIdentityInput],
) -> (u64, u64) {
    let mut model_hash = HASH_OFFSET;
    let mut tokenizer_hash = HASH_OFFSET;

    hash_str(&mut model_hash, "qwen-llm-model-v1");
    hash_str(&mut model_hash, &format!("{arch:?}"));
    hash_u64(&mut model_hash, gguf.shard_count() as u64);
    hash_u64(&mut model_hash, gguf.total_mapped_len() as u64);
    for shard in shards {
        hash_str(&mut model_hash, &shard.path);
        hash_u64(&mut model_hash, shard.mapped_len);
        if let Some(file_len) = shard.file_len {
            hash_u64(&mut model_hash, file_len);
        }
        if let Some(modified_nanos) = shard.modified_nanos {
            hash_u64(&mut model_hash, modified_nanos);
        }
    }
    for tensor in &gguf.tensors {
        hash_str(&mut model_hash, &tensor.name);
        hash_u64(&mut model_hash, tensor.dtype as i32 as u32 as u64);
        hash_u64(&mut model_hash, tensor.shard_idx as u64);
        hash_u64(&mut model_hash, tensor.data_offset);
        hash_u64(&mut model_hash, tensor.n_bytes);
        hash_u64(&mut model_hash, tensor.shape.len() as u64);
        for &dim in &tensor.shape {
            hash_u64(&mut model_hash, dim);
        }
    }

    hash_str(&mut tokenizer_hash, "qwen-llm-tokenizer-v1");
    for (key, value) in gguf.model.metadata() {
        if key.starts_with("tokenizer.") {
            hash_str(&mut tokenizer_hash, key);
            hash_value(&mut tokenizer_hash, value);
        } else {
            hash_str(&mut model_hash, key);
            hash_value(&mut model_hash, value);
        }
    }

    (model_hash, tokenizer_hash)
}
