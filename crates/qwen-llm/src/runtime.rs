//! Stable ownership boundary for loaded-model execution.
//!
//! The Metal modules expose the low-level execution pieces directly:
//! [`MetalContext`] owns device resources, [`MetalModel`] owns resident
//! weights, and [`MetalSession`] owns mutable per-sequence state. This module
//! gives callers semantic names for those lifetimes without changing the
//! underlying execution path.

use crate::gguf::{GgufError, GgufFile};
use crate::loader::{LoadError, Model};
use crate::metal::{MetalContext, MetalError};
use crate::metal_forward::{
    MetalForward, MetalModel, MetalModelLoadOptions, MetalSession, MfError, SessionSnapshot,
    SnapshotIdentity, SnapshotValidationError,
};
use crate::model::Arch;
use crate::prefix_cache::{DEFAULT_MAX_BYTES, PrefixCache, PrefixCacheStats};
use crate::tokenizer::{TokError, Tokenizer};
use parking_lot::Mutex;
use serde_json::Value;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

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
}

struct RuntimeInner {
    ctx: MetalContext,
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
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    pub exact: bool,
    pub exact_final_logits: Option<Vec<f32>>,
    /// Cache-index accounting captured at lookup, before the unlocked restore.
    pub stats_at_lookup: PrefixCacheStats,
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
        })
    }

    /// Borrowed execution driver over this loaded model.
    pub fn forward(&self) -> MetalForward<'_> {
        MetalForward::new(self.context(), &self.metal_model)
    }

    pub fn snapshot_identity(&self, sequence: &Sequence) -> SnapshotIdentity {
        let &(model_id, tokenizer_id) = self.identity_parts.get_or_init(|| {
            snapshot_identity_parts(&self.gguf, self.metal_model.arch, &self.identity_shards)
        });
        sequence
            .metal_session()
            .snapshot_identity(model_id, tokenizer_id)
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
        let snap = sequence.snapshot(
            self.snapshot_identity(sequence),
            prefix_tokens,
            final_logits,
        )?;
        let snapshot_bytes = snap.n_bytes();
        let mut cache = self.prefix_cache.lock();
        cache.insert(snap);
        Ok(PrefixCacheInsert {
            snapshot_bytes,
            stats: cache.stats(),
        })
    }

    pub fn restore_cached_prefix(
        &self,
        sequence: &mut Sequence,
        request_tokens: &[i32],
    ) -> Result<Option<PrefixCacheRestore>, RuntimeError> {
        sequence.check_position(0)?;
        sequence.ensure_can_append(request_tokens.len())?;
        let identity = self.snapshot_identity(sequence);
        let (hit, stats) = {
            let mut cache = self.prefix_cache.lock();
            let Some(hit) = cache.lookup_longest_with_exact_logits(&identity, request_tokens)
            else {
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
        let exact_final_logits = if hit.exact {
            hit.snapshot.final_logits.clone()
        } else {
            None
        };
        Ok(Some(PrefixCacheRestore {
            matched_prefix_len: hit.matched_prefix_len,
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
}

impl Sequence {
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

    pub fn snapshot(
        &self,
        identity: SnapshotIdentity,
        prefix_tokens: Vec<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> Result<SessionSnapshot, RuntimeError> {
        Ok(self.state.snapshot(identity, prefix_tokens, final_logits)?)
    }

    pub fn restore_from_snapshot(
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

    /// Low-level mutable bridge for the existing Metal execution functions.
    pub fn metal_session_mut(&mut self) -> &mut MetalSession {
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
