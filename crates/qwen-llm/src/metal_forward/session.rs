//! MetalSession state, snapshots, and sequence positions.

use super::*;

/// Identity tag for a snapshot. Checked at restore to refuse silent
/// corruption from model drift / dtype change / layout-version bump.
///
/// `model_id` is a caller-supplied compatibility fingerprint (typically a hash
/// of GGUF metadata + tensor descriptors for in-process caches). Persistent or
/// cross-process caches need a stronger content identity. `layout_version` is a
/// manual counter bumped whenever the `MetalSession` field layout changes in a
/// way that would invalidate prior snapshots.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotIdentity {
    pub model_id: u64,
    pub tokenizer_id: u64,
    pub layout_version: u32,
    pub n_attn_layers: u32,
    pub n_gdn_layers: u32,
    pub kv_dim_elements: u32,
    pub kv_bytes_per_token: u32,
    pub kv_storage_kind: SnapshotKvStorageKind,
    pub gdn_state_elements_per_layer: u32,
    pub gdn_conv_elements_per_layer: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SnapshotAbi {
    pub layout_version: u32,
    pub n_attn_layers: u32,
    pub n_gdn_layers: u32,
    pub kv_dim_elements: u32,
    pub kv_bytes_per_token: u32,
    pub kv_storage_kind: SnapshotKvStorageKind,
    pub gdn_state_elements_per_layer: u32,
    pub gdn_conv_elements_per_layer: u32,
}

impl SnapshotIdentity {
    pub fn abi(&self) -> SnapshotAbi {
        SnapshotAbi {
            layout_version: self.layout_version,
            n_attn_layers: self.n_attn_layers,
            n_gdn_layers: self.n_gdn_layers,
            kv_dim_elements: self.kv_dim_elements,
            kv_bytes_per_token: self.kv_bytes_per_token,
            kv_storage_kind: self.kv_storage_kind,
            gdn_state_elements_per_layer: self.gdn_state_elements_per_layer,
            gdn_conv_elements_per_layer: self.gdn_conv_elements_per_layer,
        }
    }
}

/// Bump this when MetalSession's per-layer state shape changes.
pub const SNAPSHOT_LAYOUT_VERSION: u32 = 4;

/// Captured state at the end of prefilling `prefix_tokens` through a
/// fresh session. Restoring into a fresh session and running additional
/// tokens is bit-equivalent to cold prefill of the full sequence
/// (validated by `h2_prefix_cache_correctness_spike`).
#[derive(Clone, Debug)]
pub struct SessionSnapshot {
    pub identity: SnapshotIdentity,
    /// Tokens consumed into the captured KV/GDN state.
    pub prefix_tokens: Vec<i32>,
    /// A terminal token selected and emitted from `final_logits`, but not yet
    /// consumed into model state. Prefix matching includes this token; restore
    /// resumes from it so a completed request never pays an unused transition.
    pub pending_token: Option<i32>,
    /// `kv_n_pos[attn_layer]` after prefill. Same value across layers
    /// for our forward pass (single-stream); kept per-layer for safety.
    pub kv_n_pos: Vec<usize>,
    /// Packed K cache slice: `n_attn × prefix_len × kv_bytes_per_token` bytes.
    /// Sized exactly to the prefix; doesn't carry the unused tail of the
    /// session's full-capacity KV buffer.
    pub kv_k_arena: Vec<u8>,
    /// Packed V cache slice (same shape).
    pub kv_v_arena: Vec<u8>,
    /// Packed GDN conv buffers: `n_gdn × gdn_conv_elements_per_layer × 4 bytes` (F32).
    pub gdn_conv_arena: Vec<u8>,
    /// Packed GDN recurrent state: `n_gdn × gdn_state_elements_per_layer × 4 bytes` (F32).
    pub gdn_state_arena: Vec<u8>,
    /// Logits at the last prefix token (vocab_size F32). Lets a
    /// subsequent exact-hit (request == cached prefix) sample directly
    /// without a forward pass. None if not stored at snapshot time.
    pub final_logits: Option<Vec<f32>>,
    /// Optional drafter capture tail: the last
    /// `min(prefix_len, DFLASH_CAPTURE_WINDOW)` target columns as F32,
    /// `capture_tail_features` elements per column, column-major over
    /// positions. Lets a restored request seed a windowed DFlash drafter
    /// without replaying the matched prefix. None for legacy snapshots and
    /// when no drafter head is loaded.
    pub capture_tail: Option<Vec<f32>>,
}

#[derive(Clone, Debug, thiserror::Error, Eq, PartialEq)]
pub enum SnapshotValidationError {
    #[error("snapshot identity mismatch: expected {expected:?}, got {actual:?}")]
    IdentityMismatch {
        expected: SnapshotIdentity,
        actual: SnapshotIdentity,
    },
    #[error("snapshot prefix length {prefix_len} exceeds capacity {capacity}")]
    PrefixCapacity { prefix_len: usize, capacity: usize },
    #[error("snapshot {section} length arithmetic overflow")]
    LengthOverflow { section: &'static str },
    #[error("snapshot {section} length {actual} != expected {expected}")]
    SectionLength {
        section: &'static str,
        actual: usize,
        expected: usize,
    },
    #[error("snapshot {section} allocation of {bytes} bytes failed")]
    AllocationFailed { section: &'static str, bytes: usize },
    #[error(
        "snapshot {section} tensor at layer {layer} has {available} bytes available, needs {required}"
    )]
    TensorBounds {
        section: &'static str,
        layer: usize,
        available: u64,
        required: usize,
    },
    #[error("snapshot KV position at layer {layer} is {actual} != prefix length {expected}")]
    KvPosition {
        layer: usize,
        actual: usize,
        expected: usize,
    },
    #[error("snapshot token {token} at index {index} is outside vocabulary size {vocab_size}")]
    TokenOutOfRange {
        index: usize,
        token: i32,
        vocab_size: usize,
    },
}

impl SessionSnapshot {
    /// Total in-memory cost of this snapshot, in bytes.
    pub fn n_bytes(&self) -> u64 {
        (self.kv_k_arena.len()
            + self.kv_v_arena.len()
            + self.gdn_conv_arena.len()
            + self.gdn_state_arena.len()
            + self.final_logits.as_ref().map_or(0, |v| v.len() * 4)
            + self.capture_tail.as_ref().map_or(0, |v| v.len() * 4)
            + self.prefix_tokens.len() * 4
            + self.pending_token.map_or(0, |_| 4)
            + self.kv_n_pos.len() * 8) as u64
    }

    /// Number of tokens consumed up to this snapshot.
    pub fn prefix_len(&self) -> usize {
        self.prefix_tokens.len()
    }

    /// Number of canonical prefix tokens represented by this checkpoint.
    pub fn matched_prefix_len(&self) -> usize {
        self.prefix_len() + usize::from(self.pending_token.is_some())
    }

    /// Validate every shape and offset premise used by restore before any
    /// session buffer is mutated. Persistent readers must call this after
    /// decoding their bounded sections and before handing the snapshot to
    /// Metal; the production restore path also calls it defensively.
    pub fn validate_for_restore(
        &self,
        expected_identity: &SnapshotIdentity,
        max_context_tokens: usize,
        expected_vocab_size: Option<usize>,
    ) -> Result<(), SnapshotValidationError> {
        if &self.identity != expected_identity {
            return Err(SnapshotValidationError::IdentityMismatch {
                expected: expected_identity.clone(),
                actual: self.identity.clone(),
            });
        }

        let prefix_len = self.prefix_len();
        if prefix_len > max_context_tokens {
            return Err(SnapshotValidationError::PrefixCapacity {
                prefix_len,
                capacity: max_context_tokens,
            });
        }

        let n_attn = self.identity.n_attn_layers as usize;
        let n_gdn = self.identity.n_gdn_layers as usize;
        require_snapshot_len("kv_n_pos", self.kv_n_pos.len(), n_attn)?;
        for (layer, &actual) in self.kv_n_pos.iter().enumerate() {
            if actual != prefix_len {
                return Err(SnapshotValidationError::KvPosition {
                    layer,
                    actual,
                    expected: prefix_len,
                });
            }
        }

        let kv_per_layer = checked_snapshot_product(
            "kv_arena",
            &[prefix_len, self.identity.kv_bytes_per_token as usize],
        )?;
        let kv_total = checked_snapshot_product("kv_arena", &[n_attn, kv_per_layer])?;
        require_snapshot_len("kv_k_arena", self.kv_k_arena.len(), kv_total)?;
        require_snapshot_len("kv_v_arena", self.kv_v_arena.len(), kv_total)?;

        let gdn_conv_total = checked_snapshot_product(
            "gdn_conv_arena",
            &[
                n_gdn,
                self.identity.gdn_conv_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        require_snapshot_len("gdn_conv_arena", self.gdn_conv_arena.len(), gdn_conv_total)?;
        let gdn_state_total = checked_snapshot_product(
            "gdn_state_arena",
            &[
                n_gdn,
                self.identity.gdn_state_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        require_snapshot_len(
            "gdn_state_arena",
            self.gdn_state_arena.len(),
            gdn_state_total,
        )?;

        if let Some(vocab_size) = expected_vocab_size {
            for (index, &token) in self.prefix_tokens.iter().enumerate() {
                if token < 0 || token as usize >= vocab_size {
                    return Err(SnapshotValidationError::TokenOutOfRange {
                        index,
                        token,
                        vocab_size,
                    });
                }
            }
            if let Some(token) = self.pending_token
                && (token < 0 || token as usize >= vocab_size)
            {
                return Err(SnapshotValidationError::TokenOutOfRange {
                    index: prefix_len,
                    token,
                    vocab_size,
                });
            }
            if let Some(logits) = self.final_logits.as_ref() {
                require_snapshot_len("final_logits", logits.len(), vocab_size)?;
            }
        }
        Ok(())
    }
}

pub(super) fn checked_snapshot_product(
    section: &'static str,
    factors: &[usize],
) -> Result<usize, SnapshotValidationError> {
    factors.iter().try_fold(1usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or(SnapshotValidationError::LengthOverflow { section })
    })
}

pub(super) fn require_snapshot_len(
    section: &'static str,
    actual: usize,
    expected: usize,
) -> Result<(), SnapshotValidationError> {
    if actual != expected {
        return Err(SnapshotValidationError::SectionLength {
            section,
            actual,
            expected,
        });
    }
    Ok(())
}

pub(super) fn allocate_snapshot_arena(
    section: &'static str,
    bytes: usize,
) -> Result<Vec<u8>, SnapshotValidationError> {
    let mut arena = Vec::new();
    arena
        .try_reserve_exact(bytes)
        .map_err(|_| SnapshotValidationError::AllocationFailed { section, bytes })?;
    // SAFETY: the caller writes every byte before exposing the arena. u8 has
    // no validity invariant and no destructor.
    unsafe {
        arena.set_len(bytes);
    }
    Ok(arena)
}

pub(super) fn validate_snapshot_tensor_span(
    section: &'static str,
    layer: usize,
    tensor: &MetalTensor,
    required: usize,
) -> Result<(), SnapshotValidationError> {
    let available = (tensor.buffer.length() as u64).saturating_sub(tensor.offset);
    if tensor.offset > tensor.buffer.length() as u64 || required as u64 > available {
        return Err(SnapshotValidationError::TensorBounds {
            section,
            layer,
            available,
            required,
        });
    }
    Ok(())
}

impl MetalSession {
    pub fn snapshot_abi(&self) -> SnapshotAbi {
        SnapshotAbi {
            layout_version: SNAPSHOT_LAYOUT_VERSION,
            n_attn_layers: self.kv_k.len() as u32,
            n_gdn_layers: self.gdn_state.len() as u32,
            kv_dim_elements: (self.kv_k.first().map(|t| t.n_elements()).unwrap_or(0)
                / self.kv_capacity as u64) as u32,
            kv_bytes_per_token: self
                .kv_k
                .first()
                .map(|t| t.n_bytes() / self.kv_capacity as u64)
                .unwrap_or(0) as u32,
            kv_storage_kind: match self.kv_k.first().map(|tensor| tensor.dtype) {
                None => SnapshotKvStorageKind::None,
                Some(GgmlType::F16) => SnapshotKvStorageKind::F16,
                Some(GgmlType::Q8_0) => SnapshotKvStorageKind::Q8_0,
                Some(_) => unreachable!("unsupported snapshot KV storage"),
            },
            gdn_state_elements_per_layer: self
                .gdn_state
                .first()
                .map(|t| t.n_elements())
                .unwrap_or(0) as u32,
            gdn_conv_elements_per_layer: self.gdn_conv.first().map(|t| t.n_elements()).unwrap_or(0)
                as u32,
        }
    }

    /// Compute the identity tag for snapshots produced by this session
    /// shape under the given model. Stable across runs of the same
    /// (model, tokenizer, layout) tuple.
    pub fn snapshot_identity(&self, model_id: u64, tokenizer_id: u64) -> SnapshotIdentity {
        let abi = self.snapshot_abi();
        SnapshotIdentity {
            model_id,
            tokenizer_id,
            layout_version: abi.layout_version,
            n_attn_layers: abi.n_attn_layers,
            n_gdn_layers: abi.n_gdn_layers,
            kv_dim_elements: abi.kv_dim_elements,
            kv_bytes_per_token: abi.kv_bytes_per_token,
            kv_storage_kind: abi.kv_storage_kind,
            gdn_state_elements_per_layer: abi.gdn_state_elements_per_layer,
            gdn_conv_elements_per_layer: abi.gdn_conv_elements_per_layer,
        }
    }

    /// Build a `SessionSnapshot` from the current session state.
    ///
    /// `prefix_tokens` is the token sequence that was just consumed
    /// (used as the cache key). `final_logits` is the logits at the
    /// last consumed token (stored for exact-hit lookups; pass `None`
    /// to skip).
    ///
    /// The caller must ensure all prior forward-pass work on this
    /// session has completed (i.e., the last `single_token` call has
    /// returned, which implies its command buffer was waited on).
    /// Reads bytes from MTLBuffer.contents() directly via memcpy --
    /// safe on shared-storage UMA per Apple docs once writes are
    /// scheduled and complete.
    pub fn snapshot(
        &self,
        identity: SnapshotIdentity,
        prefix_tokens: Vec<i32>,
        final_logits: Option<Vec<f32>>,
    ) -> Result<SessionSnapshot, MfError> {
        self.ensure_usable()?;
        let prefix_len = prefix_tokens.len();
        let n_attn = self.kv_k.len();
        let n_gdn = self.gdn_state.len();
        require_snapshot_len("kv_v_layers", self.kv_v.len(), n_attn)?;
        require_snapshot_len("gdn_conv_layers", self.gdn_conv.len(), n_gdn)?;
        let expected_identity = self.snapshot_identity(identity.model_id, identity.tokenizer_id);
        let vocab_size = usize::try_from(self.logits.n_elements()).map_err(|_| {
            SnapshotValidationError::LengthOverflow {
                section: "final_logits",
            }
        })?;
        if identity != expected_identity {
            return Err(SnapshotValidationError::IdentityMismatch {
                expected: expected_identity,
                actual: identity,
            }
            .into());
        }
        if prefix_len > self.kv_capacity {
            return Err(SnapshotValidationError::PrefixCapacity {
                prefix_len,
                capacity: self.kv_capacity,
            }
            .into());
        }
        require_snapshot_len("kv_n_pos", self.kv_n_pos.len(), n_attn)?;
        for (layer, &actual) in self.kv_n_pos.iter().enumerate() {
            if actual != prefix_len {
                return Err(SnapshotValidationError::KvPosition {
                    layer,
                    actual,
                    expected: prefix_len,
                }
                .into());
            }
        }
        for (index, &token) in prefix_tokens.iter().enumerate() {
            if token < 0 || token as usize >= vocab_size {
                return Err(SnapshotValidationError::TokenOutOfRange {
                    index,
                    token,
                    vocab_size,
                }
                .into());
            }
        }
        if let Some(logits) = final_logits.as_ref() {
            require_snapshot_len("final_logits", logits.len(), vocab_size)?;
        }
        // KV: per-layer slice is exactly prefix_len rows of the active KV dtype.
        let kv_slice_bytes = checked_snapshot_product(
            "kv_arena",
            &[prefix_len, identity.kv_bytes_per_token as usize],
        )?;
        let kv_arena_bytes = checked_snapshot_product("kv_arena", &[n_attn, kv_slice_bytes])?;
        for (layer, (k, v)) in self.kv_k.iter().zip(&self.kv_v).enumerate() {
            validate_snapshot_tensor_span("kv_k", layer, k, kv_slice_bytes)?;
            validate_snapshot_tensor_span("kv_v", layer, v, kv_slice_bytes)?;
        }

        // Allocate each arena UNINITIALIZED (no zero-init), then memcpy
        // directly from MTLBuffer.contents() into slices. ONE write per byte.
        // Zero-init costs almost as much as the actual copy at this scale
        // (158 MB of writes), so skipping it ~halves wall time.
        // Safety: the entire allocation is overwritten by read_tensor_into
        // before any read; no uninitialized bytes ever escape.
        let mut kv_k_arena = allocate_snapshot_arena("kv_k_arena", kv_arena_bytes)?;
        let mut kv_v_arena = allocate_snapshot_arena("kv_v_arena", kv_arena_bytes)?;
        for i in 0..n_attn {
            let off = i * kv_slice_bytes;
            read_tensor_into(&mut kv_k_arena[off..off + kv_slice_bytes], &self.kv_k[i]);
            read_tensor_into(&mut kv_v_arena[off..off + kv_slice_bytes], &self.kv_v[i]);
        }

        // GDN: each layer's full buffer (size doesn't depend on prefix_len).
        let gdn_conv_per = checked_snapshot_product(
            "gdn_conv_arena",
            &[
                identity.gdn_conv_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        let gdn_state_per = checked_snapshot_product(
            "gdn_state_arena",
            &[
                identity.gdn_state_elements_per_layer as usize,
                std::mem::size_of::<f32>(),
            ],
        )?;
        let gdn_conv_total = checked_snapshot_product("gdn_conv_arena", &[n_gdn, gdn_conv_per])?;
        let gdn_state_total = checked_snapshot_product("gdn_state_arena", &[n_gdn, gdn_state_per])?;
        for (layer, (conv, state)) in self.gdn_conv.iter().zip(&self.gdn_state).enumerate() {
            validate_snapshot_tensor_span("gdn_conv", layer, conv, gdn_conv_per)?;
            validate_snapshot_tensor_span("gdn_state", layer, state, gdn_state_per)?;
        }
        let mut gdn_conv_arena = allocate_snapshot_arena("gdn_conv_arena", gdn_conv_total)?;
        let mut gdn_state_arena = allocate_snapshot_arena("gdn_state_arena", gdn_state_total)?;
        for i in 0..n_gdn {
            let off_c = i * gdn_conv_per;
            let off_s = i * gdn_state_per;
            read_tensor_into(
                &mut gdn_conv_arena[off_c..off_c + gdn_conv_per],
                &self.gdn_conv[i],
            );
            read_tensor_into(
                &mut gdn_state_arena[off_s..off_s + gdn_state_per],
                &self.gdn_state[i],
            );
        }

        Ok(SessionSnapshot {
            identity,
            prefix_tokens,
            pending_token: None,
            kv_n_pos: self.kv_n_pos.clone(),
            kv_k_arena,
            kv_v_arena,
            gdn_conv_arena,
            gdn_state_arena,
            final_logits,
            capture_tail: None,
        })
    }

    /// Restore a session to the state captured in `snap`. The session
    /// MUST have been freshly created with the same architecture as
    /// the one that produced `snap` (validated by identity check).
    /// Returns Err on identity mismatch (to avoid silent corruption).
    ///
    /// The caller must ensure no in-flight GPU work is reading these
    /// session buffers (i.e., this should be called after
    /// `MetalSession::fresh` and before the first `single_token`).
    pub fn restore_from(
        &mut self,
        snap: &SessionSnapshot,
        expected_identity: &SnapshotIdentity,
    ) -> Result<(), MfError> {
        let want =
            self.snapshot_identity(expected_identity.model_id, expected_identity.tokenizer_id);
        if &want != expected_identity {
            return Err(SnapshotValidationError::IdentityMismatch {
                expected: want,
                actual: expected_identity.clone(),
            }
            .into());
        }
        let vocab_size = usize::try_from(self.logits.n_elements()).map_err(|_| {
            SnapshotValidationError::LengthOverflow {
                section: "final_logits",
            }
        })?;
        snap.validate_for_restore(expected_identity, self.kv_capacity, Some(vocab_size))?;
        let n_attn = self.kv_k.len();
        let n_gdn = self.gdn_state.len();
        require_snapshot_len("kv_v_layers", self.kv_v.len(), n_attn)?;
        require_snapshot_len("gdn_conv_layers", self.gdn_conv.len(), n_gdn)?;
        let prefix_len = snap.prefix_len();
        let kv_slice_bytes = prefix_len * snap.identity.kv_bytes_per_token as usize;

        for (layer, (k, v)) in self.kv_k.iter().zip(&self.kv_v).enumerate() {
            validate_snapshot_tensor_span("kv_k", layer, k, kv_slice_bytes)?;
            validate_snapshot_tensor_span("kv_v", layer, v, kv_slice_bytes)?;
        }
        let gdn_conv_per = (snap.identity.gdn_conv_elements_per_layer as usize) * 4;
        let gdn_state_per = (snap.identity.gdn_state_elements_per_layer as usize) * 4;
        for (layer, (conv, state)) in self.gdn_conv.iter().zip(&self.gdn_state).enumerate() {
            validate_snapshot_tensor_span("gdn_conv", layer, conv, gdn_conv_per)?;
            validate_snapshot_tensor_span("gdn_state", layer, state, gdn_state_per)?;
        }

        for i in 0..n_attn {
            let off = i * kv_slice_bytes;
            write_tensor_bytes(&self.kv_k[i], &snap.kv_k_arena[off..off + kv_slice_bytes]);
            write_tensor_bytes(&self.kv_v[i], &snap.kv_v_arena[off..off + kv_slice_bytes]);
        }
        self.kv_n_pos.copy_from_slice(&snap.kv_n_pos);

        for i in 0..n_gdn {
            let off_c = i * gdn_conv_per;
            let off_s = i * gdn_state_per;
            write_tensor_bytes(
                &self.gdn_conv[i],
                &snap.gdn_conv_arena[off_c..off_c + gdn_conv_per],
            );
            write_tensor_bytes(
                &self.gdn_state[i],
                &snap.gdn_state_arena[off_s..off_s + gdn_state_per],
            );
        }
        self.poison_reason = None;
        Ok(())
    }
}
