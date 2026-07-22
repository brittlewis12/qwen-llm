//! Stable ownership boundary for loaded-model execution.
//!
//! The Metal modules expose the low-level execution pieces directly:
//! [`MetalContext`] owns device resources, [`MetalModel`] owns resident
//! weights, and [`MetalSession`] owns mutable per-sequence state. This module
//! gives callers semantic names for those lifetimes without changing the
//! underlying execution path.

use crate::cache_probe::probe_file_residency;
use crate::checkpoint_codec::SNAPSHOT_RECORD_FIXED_BYTES;
use crate::checkpoint_identity::{
    CheckpointCompatibilityReport, CheckpointIdentityCache, CheckpointIdentityError,
    checkpoint_compatibility,
};
use crate::checkpoint_store::{
    CheckpointStoreError, DurableCheckpointStore, PublishReport, StoreContext,
};
use crate::gguf::{GgufError, GgufFile};
use crate::loader::{LoadError, Model};
use crate::metal::{MetalContext, MetalError};
use crate::metal_forward::{
    MetalForward, MetalModel, MetalModelLoadOptions, MetalSession, MfError, SessionSnapshot,
    SnapshotIdentity, SnapshotValidationError,
};
use crate::model::Arch;
use crate::prefetch::{DEFAULT_CHUNK_BYTES, DEFAULT_WORKERS, prefetch_file};
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
        let prefetch_outcome = apply_prefetch_policy(&gguf, &config);
        let bound = Model::from_gguf(&gguf)?;
        let identity_shards = snapshot_shard_identity_inputs(&gguf);
        let metal_model =
            MetalModel::load_with_options(self.context(), &gguf, &bound, intent.metal_options())?;
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
fn apply_prefetch_policy(gguf: &GgufFile, config: &LoadedModelConfig) -> PrefetchOutcome {
    let started = Instant::now();
    let mut shards: Vec<ShardPrefetch> = Vec::with_capacity(gguf.shards.len());

    if matches!(config.prefetch_policy, PrefetchPolicy::Off) {
        return PrefetchOutcome {
            policy: config.prefetch_policy,
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
        let mut record = ShardPrefetch {
            path: shard.path.clone(),
            bytes_returned: 0,
            wall: Duration::ZERO,
            pre_resident_fraction: 0.0,
            skipped: false,
            skipped_reason: None,
        };

        // Residency probe. Failure here is non-fatal: we treat unknown
        // residency as "cold" and let the policy decide.
        let residency = match probe_file_residency(&shard.path) {
            Ok(r) => Some(r),
            Err(e) => {
                tracing::warn!(
                    shard = %shard.path.display(),
                    error = %e,
                    "prefetch: residency probe failed; assuming cold"
                );
                None
            }
        };
        if let Some(r) = residency {
            record.pre_resident_fraction = r.resident_fraction();
        }

        let should_prefetch = match config.prefetch_policy {
            PrefetchPolicy::Off => false,
            PrefetchPolicy::Always => true,
            PrefetchPolicy::ColdOnly { threshold } => match residency {
                Some(r) => r.resident_fraction() < threshold,
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
            shards.push(record);
            continue;
        }

        let phase = Instant::now();
        match prefetch_file(&shard.path, workers, chunk) {
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
        shards.push(record);
    }

    PrefetchOutcome {
        policy: config.prefetch_policy,
        shards,
        total_wall: started.elapsed(),
    }
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
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct LoadedModelConfig {
    pub prefix_cache_max_bytes: u64,
    /// Policy for warming the OS unified buffer cache during model
    /// load. Defaults to [`PrefetchPolicy::Off`] to preserve exact
    /// prior behaviour.
    pub prefetch_policy: PrefetchPolicy,
    /// Number of parallel `pread` workers per shard when prefetching.
    /// Zero uses [`crate::prefetch::DEFAULT_WORKERS`] (4).
    pub prefetch_workers: usize,
    /// Per-worker scratch buffer size in bytes. Zero uses
    /// [`crate::prefetch::DEFAULT_CHUNK_BYTES`] (16 MiB).
    pub prefetch_chunk_bytes: usize,
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PrefetchPolicy {
    /// Do not prefetch. Loader behaves exactly as before this option
    /// existed.
    Off,
    /// Prefetch regardless of current residency.
    Always,
    /// Prefetch only when `mincore`-reported unified-cache residency
    /// is below `threshold` (0.0..=1.0). A pragmatic default is
    /// 0.5 — half the pages already resident means the win is
    /// diminishing and the prefetch cost is not.
    ColdOnly { threshold: f64 },
}

impl Eq for PrefetchPolicy {}

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
#[derive(Clone, Debug)]
pub struct PrefetchOutcome {
    pub policy: PrefetchPolicy,
    pub shards: Vec<ShardPrefetch>,
    /// Total wall time of the prefetch phase across all shards
    /// (including residency probes for skipped shards).
    pub total_wall: Duration,
}

impl PrefetchOutcome {
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
    fn default() -> Self {
        Self {
            prefix_cache_max_bytes: DEFAULT_MAX_BYTES,
            prefetch_policy: PrefetchPolicy::Off,
            prefetch_workers: 0,
            prefetch_chunk_bytes: 0,
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

    pub fn snapshot_identity(&self, sequence: &Sequence) -> Result<SnapshotIdentity, RuntimeError> {
        self.ensure_owns(sequence)?;
        let &(model_id, tokenizer_id) = self.identity_parts.get_or_init(|| {
            snapshot_identity_parts(&self.gguf, self.metal_model.arch, &self.identity_shards)
        });
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
