use super::*;

mod codec;
mod file;

pub use codec::{
    DeepSeekV4EncodedSnapshot, DeepSeekV4SnapshotCodecConstraints, DeepSeekV4SnapshotCodecError,
    causal_snapshot_record_bytes, decode_causal_snapshot, encode_causal_snapshot,
};
pub use file::{
    DeepSeekV4SnapshotFileError, DeepSeekV4SnapshotFileOutcome, DeepSeekV4SnapshotFileReport,
    load_causal_snapshot_file, publish_causal_snapshot_file,
};

const CAUSAL_SNAPSHOT_ABI_VERSION: u32 = 1;
// Numerics v1 includes LlamaCppB10222F16HadamardV1 scoring, tie-breaking,
// cache-order compaction, and the F16 Hadamard index-key representation.
const CAUSAL_SNAPSHOT_NUMERICS_VERSION: u32 = 1;
const CAUSAL_SNAPSHOT_ENCODING_VERSION: u32 = 1;
const COMPATIBILITY_DOMAIN: &[u8] = b"qwen-dsv4-metal-causal-compatibility-v1\0";
const PREFIX_DOMAIN: &[u8] = b"qwen-dsv4-metal-causal-prefix-v1\0";
const STATE_DOMAIN: &[u8] = b"qwen-dsv4-metal-causal-state-v1\0";

/// Caller-asserted digest of complete ordered model contents.
///
/// The session binds this value at construction but does not prove how the
/// caller calculated it. Durable users must derive it from every GGUF shard,
/// not from paths, timestamps, or tensor geometry alone.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeepSeekV4ModelContentId([u8; 32]);

impl DeepSeekV4ModelContentId {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeepSeekV4CompatibilityDigest([u8; 32]);

impl DeepSeekV4CompatibilityDigest {
    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4SnapshotObservation {
    Unavailable,
    Available,
}

/// Exact causal state captured at a completed token boundary.
///
/// Arenas are private and canonical. Raw rows are chronological rather than
/// physical ring order; compressor state preserves exact F32 bits; published
/// rows preserve exact F16 bits. Restoring never republishes source logits or
/// final hidden state.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DeepSeekV4CausalSnapshot {
    model_content_id: DeepSeekV4ModelContentId,
    compatibility_digest: DeepSeekV4CompatibilityDigest,
    next_position: u32,
    prefix_tokens: Box<[u32]>,
    prefix_digest: [u8; 32],
    source_observation: DeepSeekV4SnapshotObservation,
    raw_f16_bits: Box<[u16]>,
    compressor_f32_bits: Box<[u32]>,
    published_f16_bits: Box<[u16]>,
    causal_digest: [u8; 32],
}

impl DeepSeekV4CausalSnapshot {
    pub fn model_content_id(&self) -> DeepSeekV4ModelContentId {
        self.model_content_id
    }

    pub fn compatibility_digest(&self) -> DeepSeekV4CompatibilityDigest {
        self.compatibility_digest
    }

    pub fn next_position(&self) -> u32 {
        self.next_position
    }

    pub fn prefix_tokens(&self) -> &[u32] {
        &self.prefix_tokens
    }

    pub fn prefix_digest(&self) -> &[u8; 32] {
        &self.prefix_digest
    }

    pub fn source_observation(&self) -> DeepSeekV4SnapshotObservation {
        self.source_observation
    }

    pub fn causal_digest(&self) -> &[u8; 32] {
        &self.causal_digest
    }

    /// Bytes in the canonical token and state arenas, excluding Rust object and
    /// allocator metadata.
    pub fn payload_bytes(&self) -> u64 {
        ((self.prefix_tokens.len() + self.compressor_f32_bits.len()) * size_of::<u32>()
            + (self.raw_f16_bits.len() + self.published_f16_bits.len()) * size_of::<u16>())
            as u64
    }

    #[cfg(test)]
    fn refresh_digests(&mut self) {
        self.prefix_digest = prefix_digest(&self.prefix_tokens);
        self.causal_digest = causal_digest(self);
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SnapshotGeometry {
    raw_start_position: u32,
    raw_rows: usize,
    raw_elements: usize,
    compressor_elements: usize,
    published_elements: usize,
    published_capacity_elements: usize,
}

struct RestoreImages {
    raw_f16_bits: Vec<u16>,
}

impl DeepSeekV4Session {
    pub fn snapshot_compatibility_digest(
        &self,
    ) -> Result<DeepSeekV4CompatibilityDigest, DeepSeekV4MetalError> {
        let model_content_id = self.snapshot_model_content_id.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 session has no bound model-content identity".into(),
            )
        })?;
        Ok(snapshot_compatibility_digest(
            model_content_id,
            self.residency.config(),
        ))
    }

    pub fn capture_causal_snapshot(
        &self,
    ) -> Result<DeepSeekV4CausalSnapshot, DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        {
            if self.fp4_selection_mode.is_counterfactual() {
                return invalid("FP4 selection-counterfactual sessions cannot export snapshot v1");
            }
            self.decision_diagnostics
                .ensure_no_active_capture("capture a causal snapshot")?;
            self.fp4_shadow_diagnostics
                .ensure_no_active_capture("capture a causal snapshot")?;
        }
        let model_content_id = self.snapshot_model_content_id.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 session has no bound model-content identity".into(),
            )
        })?;
        capture_causal_state(
            self.residency.config(),
            self.capacity,
            self.phase,
            &self.committed_tokens,
            model_content_id,
            &self.raw_cache,
            &self.compressor_frontiers,
        )
    }

    /// Replace this ready session's causal state with a compatible snapshot.
    ///
    /// Every expected failure is checked before destination mutation. Once
    /// copying starts, the session is poisoned until every arena and transcript
    /// is installed; a panic cannot expose partially restored state as ready.
    pub fn restore_causal_snapshot(
        &mut self,
        snapshot: &DeepSeekV4CausalSnapshot,
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        {
            if self.fp4_selection_mode.is_counterfactual() {
                return invalid("FP4 selection-counterfactual sessions cannot restore snapshot v1");
            }
            self.decision_diagnostics
                .ensure_no_active_capture("restore a causal snapshot")?;
            self.fp4_shadow_diagnostics
                .ensure_no_active_capture("restore a causal snapshot")?;
        }
        let model_content_id = self.snapshot_model_content_id.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 session has no bound model-content identity".into(),
            )
        })?;
        restore_causal_state(
            self.residency.config(),
            self.capacity,
            &mut self.phase,
            &mut self.committed_tokens,
            model_content_id,
            &self.raw_cache,
            &self.compressor_frontiers,
            snapshot,
        )?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.compressor_frontiers.disable_fp4_shadow_lineage()?;
            self.fp4_shadow_diagnostics.invalidate_lineage();
        }
        Ok(())
    }

    /// Hashes the current causal arenas for a diagnostics-only FP4 selection
    /// counterfactual without authorizing snapshot-v1 export or restore.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn fp4_counterfactual_state_digest(
        &self,
    ) -> Result<DeepSeekV4Fp4CounterfactualStateDigest, DeepSeekV4MetalError> {
        if !self.fp4_selection_mode.is_counterfactual() {
            return invalid("session is not an FP4 selection counterfactual");
        }
        self.decision_diagnostics
            .ensure_no_active_capture("digest counterfactual state")?;
        self.fp4_shadow_diagnostics
            .ensure_no_active_capture("digest counterfactual state")?;
        let model_content_id = self.snapshot_model_content_id.ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 session has no bound model-content identity".into(),
            )
        })?;
        let state = capture_causal_state(
            self.residency.config(),
            self.capacity,
            self.phase,
            &self.committed_tokens,
            model_content_id,
            &self.raw_cache,
            &self.compressor_frontiers,
        )?;
        let state_digest = *state.causal_digest();
        let (selection_trace_digest, consumed_layer_count) = self.fp4_counterfactual_trace.digest();
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"qwen-dsv4-fp4-selection-counterfactual-v1\0");
        hasher.update(&state_digest);
        hasher.update(&selection_trace_digest);
        hasher.update(&consumed_layer_count.to_le_bytes());
        Ok(DeepSeekV4Fp4CounterfactualStateDigest {
            prefix_digest: *state.prefix_digest(),
            state_digest,
            selection_trace_digest,
            consumed_layer_count,
            counterfactual_domain_digest: *hasher.finalize().as_bytes(),
        })
    }
}

fn capture_causal_state(
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
    phase: DeepSeekV4SessionPhase,
    committed_tokens: &[u32],
    model_content_id: DeepSeekV4ModelContentId,
    raw_cache: &MetalTensor,
    frontiers: &DeepSeekV4CompressorFrontiers,
) -> Result<DeepSeekV4CausalSnapshot, DeepSeekV4MetalError> {
    let next_position = phase.ready_position()?;
    validate_prefix(committed_tokens, next_position, config.vocab_size as usize)?;
    let geometry = snapshot_geometry(config, capacity, next_position)?;
    validate_persistent_tensors(config, capacity, raw_cache, frontiers)?;

    let mut raw_f16_bits = try_bits(geometry.raw_elements, "snapshot raw F16 arena")?;
    capture_raw_rows(raw_cache, config, geometry, &mut raw_f16_bits);
    let mut compressor_f32_bits = try_bits(
        geometry.compressor_elements,
        "snapshot compressor F32 arena",
    )?;
    let mut published_f16_bits =
        try_bits(geometry.published_elements, "snapshot published F16 arena")?;
    capture_frontiers(
        frontiers,
        config,
        next_position,
        &mut compressor_f32_bits,
        &mut published_f16_bits,
    );
    debug_assert_eq!(raw_f16_bits.len(), geometry.raw_elements);
    debug_assert_eq!(compressor_f32_bits.len(), geometry.compressor_elements);
    debug_assert_eq!(published_f16_bits.len(), geometry.published_elements);

    let mut prefix_tokens = try_bits(committed_tokens.len(), "snapshot committed tokens")?;
    prefix_tokens.extend_from_slice(committed_tokens);
    let prefix_tokens = prefix_tokens.into_boxed_slice();
    let prefix_digest = prefix_digest(&prefix_tokens);
    let compatibility_digest = snapshot_compatibility_digest(model_content_id, config);
    let source_observation = if phase.observation_valid() {
        DeepSeekV4SnapshotObservation::Available
    } else {
        DeepSeekV4SnapshotObservation::Unavailable
    };
    let mut snapshot = DeepSeekV4CausalSnapshot {
        model_content_id,
        compatibility_digest,
        next_position,
        prefix_tokens,
        prefix_digest,
        source_observation,
        raw_f16_bits: raw_f16_bits.into_boxed_slice(),
        compressor_f32_bits: compressor_f32_bits.into_boxed_slice(),
        published_f16_bits: published_f16_bits.into_boxed_slice(),
        causal_digest: [0; 32],
    };
    snapshot.causal_digest = causal_digest(&snapshot);
    validate_snapshot(&snapshot, config, capacity, model_content_id)?;
    Ok(snapshot)
}

fn restore_causal_state(
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
    phase: &mut DeepSeekV4SessionPhase,
    committed_tokens: &mut Vec<u32>,
    model_content_id: DeepSeekV4ModelContentId,
    raw_cache: &MetalTensor,
    frontiers: &DeepSeekV4CompressorFrontiers,
    snapshot: &DeepSeekV4CausalSnapshot,
) -> Result<(), DeepSeekV4MetalError> {
    let replaced_position = phase.ready_position()?;
    validate_prefix(
        committed_tokens,
        replaced_position,
        config.vocab_size as usize,
    )?;
    validate_snapshot(snapshot, config, capacity, model_content_id)?;
    validate_persistent_tensors(config, capacity, raw_cache, frontiers)?;
    if committed_tokens.capacity() < snapshot.prefix_tokens.len() {
        return invalid(format!(
            "DeepSeek V4 transcript capacity {} cannot restore {} tokens",
            committed_tokens.capacity(),
            snapshot.prefix_tokens.len()
        ));
    }
    let images = build_restore_images(snapshot, config, capacity)?;

    let begun_position = phase.begin_restore()?;
    debug_assert_eq!(begun_position, replaced_position);
    write_f16_bits(raw_cache, &images.raw_f16_bits);
    write_frontiers(
        frontiers,
        config,
        snapshot.next_position,
        &snapshot.compressor_f32_bits,
        &snapshot.published_f16_bits,
    );
    #[cfg(feature = "dsv4-diagnostics")]
    frontiers.invalidate_fp4_shadow_lineage()?;
    committed_tokens.clear();
    committed_tokens.extend_from_slice(&snapshot.prefix_tokens);
    phase.complete_restore(replaced_position, snapshot.next_position)
}

fn snapshot_geometry(
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
    next_position: u32,
) -> Result<SnapshotGeometry, DeepSeekV4MetalError> {
    capacity.validate_next_position(next_position)?;
    if next_position > config.context_length {
        return invalid(format!(
            "DeepSeek V4 snapshot position {next_position} exceeds context length {}",
            config.context_length
        ));
    }
    if config.attention_kinds.len() != DEEPSEEK_V4_LAYER_COUNT {
        return invalid(format!(
            "DeepSeek V4 snapshot expected {DEEPSEEK_V4_LAYER_COUNT} layers, got {}",
            config.attention_kinds.len()
        ));
    }
    let raw_rows = (next_position as usize).min(DEEPSEEK_V4_LOCAL_WINDOW);
    let raw_start_position = next_position - raw_rows as u32;
    let raw_elements = checked_product(
        &[
            DEEPSEEK_V4_LAYER_COUNT,
            raw_rows,
            config.key_length as usize,
        ],
        "snapshot raw arena",
    )?;
    let mut compressor_elements = 0usize;
    let mut published_elements = 0usize;
    let mut published_capacity_elements = 0usize;
    for kind in config.attention_kinds.iter().copied() {
        match kind {
            AttentionKind::SlidingWindow => {}
            AttentionKind::CompressedSparse => {
                for head_dim in [
                    config.key_length as usize,
                    config.indexer_key_length as usize,
                ] {
                    let (width, rows, state_elements) = compressor_frontier_geometry(4, head_dim)?;
                    debug_assert_eq!(state_elements, width * rows);
                    compressor_elements = checked_add(
                        compressor_elements,
                        checked_mul(state_elements, 2, "snapshot compressor state pair")?,
                        "snapshot compressor arena",
                    )?;
                    let count = next_position as usize / 4;
                    if count > capacity.csa_physical_rows() {
                        return invalid(format!(
                            "DeepSeek V4 snapshot CSA row count {count} exceeds history capacity {}",
                            capacity.csa_physical_rows()
                        ));
                    }
                    published_elements = checked_add(
                        published_elements,
                        checked_mul(count, head_dim, "snapshot CSA published rows")?,
                        "snapshot published arena",
                    )?;
                    published_capacity_elements = checked_add(
                        published_capacity_elements,
                        checked_mul(
                            capacity.csa_physical_rows(),
                            head_dim,
                            "snapshot CSA publication capacity",
                        )?,
                        "snapshot publication capacity",
                    )?;
                }
            }
            AttentionKind::HeavilyCompressed => {
                let head_dim = config.key_length as usize;
                let (_, _, state_elements) = compressor_frontier_geometry(128, head_dim)?;
                compressor_elements = checked_add(
                    compressor_elements,
                    checked_mul(state_elements, 2, "snapshot HCA state pair")?,
                    "snapshot compressor arena",
                )?;
                let count = next_position as usize / 128;
                if count > capacity.hca_physical_rows() {
                    return invalid(format!(
                        "DeepSeek V4 snapshot HCA row count {count} exceeds history capacity {}",
                        capacity.hca_physical_rows()
                    ));
                }
                published_elements = checked_add(
                    published_elements,
                    checked_mul(count, head_dim, "snapshot HCA published rows")?,
                    "snapshot published arena",
                )?;
                published_capacity_elements = checked_add(
                    published_capacity_elements,
                    checked_mul(
                        capacity.hca_physical_rows(),
                        head_dim,
                        "snapshot HCA publication capacity",
                    )?,
                    "snapshot publication capacity",
                )?;
            }
        }
    }
    Ok(SnapshotGeometry {
        raw_start_position,
        raw_rows,
        raw_elements,
        compressor_elements,
        published_elements,
        published_capacity_elements,
    })
}

fn validate_snapshot(
    snapshot: &DeepSeekV4CausalSnapshot,
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
    expected_model_content_id: DeepSeekV4ModelContentId,
) -> Result<(), DeepSeekV4MetalError> {
    if snapshot.model_content_id != expected_model_content_id {
        return invalid("DeepSeek V4 snapshot model-content identity mismatch");
    }
    let expected_compatibility = snapshot_compatibility_digest(expected_model_content_id, config);
    if snapshot.compatibility_digest != expected_compatibility {
        return invalid("DeepSeek V4 snapshot compatibility digest mismatch");
    }
    validate_prefix(
        &snapshot.prefix_tokens,
        snapshot.next_position,
        config.vocab_size as usize,
    )?;
    if snapshot.prefix_digest != prefix_digest(&snapshot.prefix_tokens) {
        return invalid("DeepSeek V4 snapshot prefix digest mismatch");
    }
    let geometry = snapshot_geometry(config, capacity, snapshot.next_position)?;
    require_len(
        "snapshot raw F16 arena",
        snapshot.raw_f16_bits.len(),
        geometry.raw_elements,
    )?;
    require_len(
        "snapshot compressor F32 arena",
        snapshot.compressor_f32_bits.len(),
        geometry.compressor_elements,
    )?;
    require_len(
        "snapshot published F16 arena",
        snapshot.published_f16_bits.len(),
        geometry.published_elements,
    )?;
    validate_compressor_arena(
        config,
        snapshot.next_position,
        &snapshot.compressor_f32_bits,
    )?;
    if snapshot.causal_digest != causal_digest(snapshot) {
        return invalid("DeepSeek V4 snapshot causal digest mismatch");
    }
    Ok(())
}

fn validate_prefix(
    tokens: &[u32],
    next_position: u32,
    vocab_size: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if tokens.len() != next_position as usize {
        return invalid(format!(
            "DeepSeek V4 committed-token transcript has length {}, expected {next_position}",
            tokens.len()
        ));
    }
    if let Some((index, token)) = tokens
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token)| *token as usize >= vocab_size)
    {
        return invalid(format!(
            "DeepSeek V4 committed token {index} ID {token} is outside vocabulary {vocab_size}"
        ));
    }
    Ok(())
}

fn validate_persistent_tensors(
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
    raw_cache: &MetalTensor,
    frontiers: &DeepSeekV4CompressorFrontiers,
) -> Result<(), DeepSeekV4MetalError> {
    validate_f16(
        raw_cache,
        &[
            config.key_length as u64,
            DEEPSEEK_V4_LOCAL_WINDOW as u64,
            config.layer_count as u64,
        ],
        true,
        "DeepSeek V4 snapshot raw cache",
    )?;
    if frontiers.layers.len() != config.attention_kinds.len() {
        return invalid(format!(
            "DeepSeek V4 snapshot frontier layer count {} differs from schedule {}",
            frontiers.layers.len(),
            config.attention_kinds.len()
        ));
    }
    for (layer, (kind, layer_frontiers)) in config
        .attention_kinds
        .iter()
        .copied()
        .zip(&frontiers.layers)
        .enumerate()
    {
        match (kind, layer_frontiers) {
            (AttentionKind::SlidingWindow, DeepSeekV4LayerCompressorFrontiers::SlidingWindow) => {}
            (
                AttentionKind::CompressedSparse,
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer },
            ) => {
                validate_frontier(
                    attention,
                    4,
                    config.key_length as usize,
                    DeepSeekV4CompressorPublication::Attention,
                    capacity.csa_physical_rows(),
                    layer,
                )?;
                validate_frontier(
                    indexer,
                    4,
                    config.indexer_key_length as usize,
                    DeepSeekV4CompressorPublication::IndexerHadamard,
                    capacity.csa_physical_rows(),
                    layer,
                )?;
            }
            (
                AttentionKind::HeavilyCompressed,
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention },
            ) => validate_frontier(
                attention,
                128,
                config.key_length as usize,
                DeepSeekV4CompressorPublication::Attention,
                capacity.hca_physical_rows(),
                layer,
            )?,
            _ => {
                return invalid(format!(
                    "DeepSeek V4 snapshot frontier kind differs from layer {layer} schedule"
                ));
            }
        }
    }
    Ok(())
}

fn validate_frontier(
    frontier: &DeepSeekV4CompressorFrontier,
    ratio: usize,
    head_dim: usize,
    publication: DeepSeekV4CompressorPublication,
    capacity_rows: usize,
    layer: usize,
) -> Result<(), DeepSeekV4MetalError> {
    let (width, rows, _) = compressor_frontier_geometry(ratio, head_dim)?;
    if frontier.ratio != ratio
        || frontier.head_dim != head_dim
        || frontier.width != width
        || frontier.rows != rows
        || frontier.capacity_rows != capacity_rows
        || frontier.publication != publication
    {
        return invalid(format!(
            "DeepSeek V4 snapshot frontier geometry differs at layer {layer}"
        ));
    }
    validate_f32(
        &frontier.kv_state,
        &[width as u64, rows as u64],
        true,
        "snapshot compressor KV state",
    )?;
    validate_f32(
        &frontier.score_state,
        &[width as u64, rows as u64],
        true,
        "snapshot compressor score state",
    )?;
    validate_f16(
        &frontier.published,
        &[head_dim as u64, capacity_rows as u64],
        true,
        "snapshot compressor published rows",
    )
}

fn capture_raw_rows(
    raw_cache: &MetalTensor,
    config: &DeepSeekV4Config,
    geometry: SnapshotGeometry,
    arena: &mut Vec<u16>,
) {
    let width = config.key_length as usize;
    let source = tensor_u16(raw_cache);
    for layer in 0..config.layer_count as usize {
        for logical_position in
            geometry.raw_start_position..geometry.raw_start_position + geometry.raw_rows as u32
        {
            let slot = logical_position as usize % DEEPSEEK_V4_LOCAL_WINDOW;
            let start = (layer * DEEPSEEK_V4_LOCAL_WINDOW + slot) * width;
            arena.extend_from_slice(&source[start..start + width]);
        }
    }
}

fn capture_frontiers(
    frontiers: &DeepSeekV4CompressorFrontiers,
    config: &DeepSeekV4Config,
    next_position: u32,
    compressor_arena: &mut Vec<u32>,
    published_arena: &mut Vec<u16>,
) {
    for layer in &frontiers.layers {
        match layer {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => {}
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                capture_frontier(
                    attention,
                    next_position as usize / 4,
                    compressor_arena,
                    published_arena,
                );
                capture_frontier(
                    indexer,
                    next_position as usize / 4,
                    compressor_arena,
                    published_arena,
                );
            }
            DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => {
                capture_frontier(
                    attention,
                    next_position as usize / 128,
                    compressor_arena,
                    published_arena,
                );
            }
        }
    }
    debug_assert_eq!(frontiers.layers.len(), config.attention_kinds.len());
}

fn capture_frontier(
    frontier: &DeepSeekV4CompressorFrontier,
    published_count: usize,
    compressor_arena: &mut Vec<u32>,
    published_arena: &mut Vec<u16>,
) {
    compressor_arena.extend_from_slice(tensor_u32(&frontier.kv_state));
    compressor_arena.extend_from_slice(tensor_u32(&frontier.score_state));
    let published = tensor_u16(&frontier.published);
    let elements = published_count * frontier.head_dim;
    published_arena.extend_from_slice(&published[..elements]);
}

fn build_restore_images(
    snapshot: &DeepSeekV4CausalSnapshot,
    config: &DeepSeekV4Config,
    capacity: DeepSeekV4SessionCapacity,
) -> Result<RestoreImages, DeepSeekV4MetalError> {
    let geometry = snapshot_geometry(config, capacity, snapshot.next_position)?;
    let raw_capacity = checked_product(
        &[
            config.layer_count as usize,
            DEEPSEEK_V4_LOCAL_WINDOW,
            config.key_length as usize,
        ],
        "snapshot raw restore image",
    )?;
    let mut raw_f16_bits = try_zeroed_bits(raw_capacity, "snapshot raw restore image")?;
    let width = config.key_length as usize;
    let mut source_cursor = 0usize;
    for layer in 0..config.layer_count as usize {
        for logical_position in
            geometry.raw_start_position..geometry.raw_start_position + geometry.raw_rows as u32
        {
            let slot = logical_position as usize % DEEPSEEK_V4_LOCAL_WINDOW;
            let destination = (layer * DEEPSEEK_V4_LOCAL_WINDOW + slot) * width;
            raw_f16_bits[destination..destination + width]
                .copy_from_slice(&snapshot.raw_f16_bits[source_cursor..source_cursor + width]);
            source_cursor += width;
        }
    }
    debug_assert_eq!(source_cursor, snapshot.raw_f16_bits.len());

    Ok(RestoreImages { raw_f16_bits })
}

fn write_frontiers(
    frontiers: &DeepSeekV4CompressorFrontiers,
    config: &DeepSeekV4Config,
    next_position: u32,
    compressor_arena: &[u32],
    published_arena: &[u16],
) {
    let mut compressor_cursor = 0usize;
    let mut published_cursor = 0usize;
    for layer in &frontiers.layers {
        match layer {
            DeepSeekV4LayerCompressorFrontiers::SlidingWindow => {}
            DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                write_frontier(
                    attention,
                    compressor_arena,
                    &mut compressor_cursor,
                    published_arena,
                    &mut published_cursor,
                    next_position as usize / 4,
                );
                write_frontier(
                    indexer,
                    compressor_arena,
                    &mut compressor_cursor,
                    published_arena,
                    &mut published_cursor,
                    next_position as usize / 4,
                );
            }
            DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => write_frontier(
                attention,
                compressor_arena,
                &mut compressor_cursor,
                published_arena,
                &mut published_cursor,
                next_position as usize / 128,
            ),
        }
    }
    debug_assert_eq!(frontiers.layers.len(), config.attention_kinds.len());
    debug_assert_eq!(compressor_cursor, compressor_arena.len());
    debug_assert_eq!(published_cursor, published_arena.len());
}

fn write_frontier(
    frontier: &DeepSeekV4CompressorFrontier,
    compressor_arena: &[u32],
    compressor_cursor: &mut usize,
    published_arena: &[u16],
    published_cursor: &mut usize,
    published_count: usize,
) {
    let state_elements = frontier.width * frontier.rows;
    write_u32_bits(
        &frontier.kv_state,
        &compressor_arena[*compressor_cursor..*compressor_cursor + state_elements],
    );
    *compressor_cursor += state_elements;
    write_u32_bits(
        &frontier.score_state,
        &compressor_arena[*compressor_cursor..*compressor_cursor + state_elements],
    );
    *compressor_cursor += state_elements;
    let published_elements = published_count * frontier.head_dim;
    zero_f16_bits(&frontier.published);
    write_f16_prefix(
        &frontier.published,
        &published_arena[*published_cursor..*published_cursor + published_elements],
    );
    *published_cursor += published_elements;
}

fn validate_compressor_arena(
    config: &DeepSeekV4Config,
    next_position: u32,
    arena: &[u32],
) -> Result<(), DeepSeekV4MetalError> {
    let mut cursor = 0usize;
    for (layer, kind) in config.attention_kinds.iter().copied().enumerate() {
        match kind {
            AttentionKind::SlidingWindow => {}
            AttentionKind::CompressedSparse => {
                for (name, head_dim) in [
                    ("CSA attention", config.key_length as usize),
                    ("CSA indexer", config.indexer_key_length as usize),
                ] {
                    validate_frontier_bits(
                        layer,
                        name,
                        4,
                        head_dim,
                        next_position,
                        arena,
                        &mut cursor,
                    )?;
                }
            }
            AttentionKind::HeavilyCompressed => validate_frontier_bits(
                layer,
                "HCA attention",
                128,
                config.key_length as usize,
                next_position,
                arena,
                &mut cursor,
            )?,
        }
    }
    require_len("snapshot compressor cursor", cursor, arena.len())
}

#[allow(clippy::too_many_arguments)]
fn validate_frontier_bits(
    layer: usize,
    name: &str,
    ratio: usize,
    head_dim: usize,
    next_position: u32,
    arena: &[u32],
    cursor: &mut usize,
) -> Result<(), DeepSeekV4MetalError> {
    let (width, rows, state_elements) = compressor_frontier_geometry(ratio, head_dim)?;
    let kv = &arena[*cursor..*cursor + state_elements];
    *cursor += state_elements;
    let scores = &arena[*cursor..*cursor + state_elements];
    *cursor += state_elements;
    for row in 0..rows {
        let written = if ratio == 4 {
            let complete_blocks = next_position as usize / 4;
            if row < 4 {
                complete_blocks > 0
            } else {
                complete_blocks > 0 || row - 4 < next_position as usize % 4
            }
        } else {
            next_position as usize >= ratio || row < next_position as usize % ratio
        };
        let range = row * width..(row + 1) * width;
        for (dimension, (&kv_bits, &score_bits)) in
            kv[range.clone()].iter().zip(&scores[range]).enumerate()
        {
            let kv_value = f32::from_bits(kv_bits);
            let score_value = f32::from_bits(score_bits);
            if written {
                if !kv_value.is_finite() || !score_value.is_finite() {
                    return invalid(format!(
                        "DeepSeek V4 snapshot {name} layer {layer} row {row} dimension {dimension} is not finite"
                    ));
                }
            } else if kv_bits != 0 || score_bits != f32::NEG_INFINITY.to_bits() {
                return invalid(format!(
                    "DeepSeek V4 snapshot {name} layer {layer} row {row} dimension {dimension} is initialized outside its phase"
                ));
            }
        }
    }
    if ratio == 4 && next_position >= 4 {
        let overwritten = next_position as usize % 4;
        for row in overwritten..4 {
            let lower = row * width..(row + 1) * width;
            let upper = (row + 4) * width..(row + 5) * width;
            if kv[lower.clone()] != kv[upper.clone()] || scores[lower] != scores[upper] {
                return invalid(format!(
                    "DeepSeek V4 snapshot {name} layer {layer} ratio-4 overlap row {row} differs across banks"
                ));
            }
        }
    }
    Ok(())
}

fn snapshot_compatibility_digest(
    model_content_id: DeepSeekV4ModelContentId,
    config: &DeepSeekV4Config,
) -> DeepSeekV4CompatibilityDigest {
    let mut hasher = blake3::Hasher::new();
    hasher.update(COMPATIBILITY_DOMAIN);
    hasher.update(model_content_id.as_bytes());
    hash_named_u32(&mut hasher, b"causal_abi", CAUSAL_SNAPSHOT_ABI_VERSION);
    hash_named_u32(
        &mut hasher,
        b"numerics_abi",
        CAUSAL_SNAPSHOT_NUMERICS_VERSION,
    );
    hash_named_u32(
        &mut hasher,
        b"encoding_abi",
        CAUSAL_SNAPSHOT_ENCODING_VERSION,
    );
    hash_named_u32(&mut hasher, b"raw_storage", 1);
    hash_named_u32(&mut hasher, b"frontier_storage", 2);
    hash_named_u32(&mut hasher, b"published_storage", 1);
    hash_named_u32(&mut hasher, b"raw_order", 1);
    hash_named_u32(&mut hasher, b"frontier_order", 1);
    hash_named_u32(&mut hasher, b"published_order", 1);
    hash_named_u32(&mut hasher, b"rope_pairing", 1);
    hash_named_u32(&mut hasher, b"ratio4_lane_layout", 1);
    hash_named_u32(&mut hasher, b"same_token_publication", 1);
    hash_named_u32(&mut hasher, b"indexer_hadamard", 1);
    hash_named_u64(
        &mut hasher,
        b"compressed_slab_rows",
        DEEPSEEK_V4_COMPRESSED_HISTORY_SLAB_ROWS as u64,
    );
    hash_config(&mut hasher, config);
    DeepSeekV4CompatibilityDigest(*hasher.finalize().as_bytes())
}

fn hash_config(hasher: &mut blake3::Hasher, config: &DeepSeekV4Config) {
    let DeepSeekV4Config {
        layer_count,
        context_length,
        hidden_size,
        vocab_size,
        attention_head_count,
        kv_head_count,
        key_length,
        value_length,
        rope_dimension_count,
        rope_freq_base,
        attention_rms_epsilon,
        rope_scaling_type,
        rope_scaling_factor,
        rope_original_context_length,
        rope_yarn_beta_fast,
        rope_yarn_beta_slow,
        q_lora_rank,
        sliding_window,
        expert_count,
        expert_used_count,
        expert_feed_forward_length,
        shared_expert_count,
        expert_weights_scale,
        expert_weights_norm,
        expert_gating_func,
        indexer_head_count,
        indexer_key_length,
        indexer_top_k,
        output_group_count,
        output_lora_rank,
        attention_kinds,
        compress_ratio_tail,
        compress_rope_freq_base,
        hyper_connection_count,
        sinkhorn_iterations,
        hyper_connection_epsilon,
        hash_layer_count,
        swiglu_clamp_experts,
        swiglu_clamp_shared,
        tokenizer_model,
        tokenizer_pre,
        bos_token_id,
        eos_token_id,
        padding_token_id,
        tokenizer_token_type_count,
        tokenizer_merge_count,
        add_bos_token,
        add_eos_token,
    } = config;
    for (name, value) in [
        (b"layer_count".as_slice(), *layer_count),
        (b"context_length", *context_length),
        (b"hidden_size", *hidden_size),
        (b"vocab_size", *vocab_size),
        (b"attention_head_count", *attention_head_count),
        (b"kv_head_count", *kv_head_count),
        (b"key_length", *key_length),
        (b"value_length", *value_length),
        (b"rope_dimension_count", *rope_dimension_count),
        (
            b"rope_original_context_length",
            *rope_original_context_length,
        ),
        (b"q_lora_rank", *q_lora_rank),
        (b"sliding_window", *sliding_window),
        (b"expert_count", *expert_count),
        (b"expert_used_count", *expert_used_count),
        (b"expert_feed_forward_length", *expert_feed_forward_length),
        (b"shared_expert_count", *shared_expert_count),
        (b"expert_gating_func", *expert_gating_func),
        (b"indexer_head_count", *indexer_head_count),
        (b"indexer_key_length", *indexer_key_length),
        (b"indexer_top_k", *indexer_top_k),
        (b"output_group_count", *output_group_count),
        (b"output_lora_rank", *output_lora_rank),
        (b"hyper_connection_count", *hyper_connection_count),
        (b"sinkhorn_iterations", *sinkhorn_iterations),
        (b"hash_layer_count", *hash_layer_count),
    ] {
        hash_named_u32(hasher, name, value);
    }
    for (name, value) in [
        (b"rope_freq_base".as_slice(), *rope_freq_base),
        (b"attention_rms_epsilon", *attention_rms_epsilon),
        (b"rope_scaling_factor", *rope_scaling_factor),
        (b"rope_yarn_beta_fast", *rope_yarn_beta_fast),
        (b"rope_yarn_beta_slow", *rope_yarn_beta_slow),
        (b"expert_weights_scale", *expert_weights_scale),
        (b"compress_rope_freq_base", *compress_rope_freq_base),
        (b"hyper_connection_epsilon", *hyper_connection_epsilon),
    ] {
        hash_named_u32(hasher, name, value.to_bits());
    }
    hash_named_bytes(hasher, b"rope_scaling_type", rope_scaling_type.as_bytes());
    hash_named_bool(hasher, b"expert_weights_norm", *expert_weights_norm);
    hash_named_u64(hasher, b"attention_kinds_len", attention_kinds.len() as u64);
    for kind in attention_kinds {
        hash_u32(hasher, kind.ratio());
    }
    hash_named_u64(
        hasher,
        b"compress_ratio_tail_len",
        compress_ratio_tail.len() as u64,
    );
    for value in compress_ratio_tail {
        hash_u32(hasher, *value);
    }
    for (name, values) in [
        (b"swiglu_clamp_experts".as_slice(), swiglu_clamp_experts),
        (b"swiglu_clamp_shared", swiglu_clamp_shared),
    ] {
        hash_named_u64(hasher, name, values.len() as u64);
        for value in values {
            hash_u32(hasher, value.to_bits());
        }
    }
    hash_named_bytes(hasher, b"tokenizer_model", tokenizer_model.as_bytes());
    hash_named_bytes(hasher, b"tokenizer_pre", tokenizer_pre.as_bytes());
    for (name, value) in [
        (b"bos_token_id".as_slice(), *bos_token_id),
        (b"eos_token_id", *eos_token_id),
        (b"padding_token_id", *padding_token_id),
    ] {
        hash_named_option_u32(hasher, name, value);
    }
    hash_named_u64(
        hasher,
        b"tokenizer_token_type_count",
        *tokenizer_token_type_count as u64,
    );
    hash_named_u64(
        hasher,
        b"tokenizer_merge_count",
        *tokenizer_merge_count as u64,
    );
    hash_named_bool(hasher, b"add_bos_token", *add_bos_token);
    hash_named_bool(hasher, b"add_eos_token", *add_eos_token);
}

fn prefix_digest(tokens: &[u32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(PREFIX_DOMAIN);
    hash_u64(&mut hasher, tokens.len() as u64);
    for token in tokens {
        hash_u32(&mut hasher, *token);
    }
    *hasher.finalize().as_bytes()
}

fn causal_digest(snapshot: &DeepSeekV4CausalSnapshot) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(STATE_DOMAIN);
    hasher.update(snapshot.model_content_id.as_bytes());
    hasher.update(snapshot.compatibility_digest.as_bytes());
    hash_u32(&mut hasher, snapshot.next_position);
    hasher.update(&snapshot.prefix_digest);
    hash_u16_arena(&mut hasher, &snapshot.raw_f16_bits);
    hash_u32_arena(&mut hasher, &snapshot.compressor_f32_bits);
    hash_u16_arena(&mut hasher, &snapshot.published_f16_bits);
    *hasher.finalize().as_bytes()
}

fn hash_u16_arena(hasher: &mut blake3::Hasher, values: &[u16]) {
    hash_u64(hasher, values.len() as u64);
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn hash_u32_arena(hasher: &mut blake3::Hasher, values: &[u32]) {
    hash_u64(hasher, values.len() as u64);
    for value in values {
        hasher.update(&value.to_le_bytes());
    }
}

fn hash_named_u32(hasher: &mut blake3::Hasher, name: &[u8], value: u32) {
    hash_named_bytes(hasher, name, &value.to_le_bytes());
}

fn hash_named_u64(hasher: &mut blake3::Hasher, name: &[u8], value: u64) {
    hash_named_bytes(hasher, name, &value.to_le_bytes());
}

fn hash_named_bool(hasher: &mut blake3::Hasher, name: &[u8], value: bool) {
    hash_named_bytes(hasher, name, &[u8::from(value)]);
}

fn hash_named_option_u32(hasher: &mut blake3::Hasher, name: &[u8], value: Option<u32>) {
    hash_named_bytes(hasher, name, &[u8::from(value.is_some())]);
    if let Some(value) = value {
        hash_u32(hasher, value);
    }
}

fn hash_named_bytes(hasher: &mut blake3::Hasher, name: &[u8], value: &[u8]) {
    hash_u64(hasher, name.len() as u64);
    hasher.update(name);
    hash_u64(hasher, value.len() as u64);
    hasher.update(value);
}

fn hash_u32(hasher: &mut blake3::Hasher, value: u32) {
    hasher.update(&value.to_le_bytes());
}

fn hash_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_le_bytes());
}

fn try_bits<T>(capacity: usize, name: &str) -> Result<Vec<T>, DeepSeekV4MetalError> {
    let mut values = Vec::new();
    values.try_reserve_exact(capacity).map_err(|error| {
        DeepSeekV4MetalError::Invalid(format!("allocate {name} ({capacity} elements): {error}"))
    })?;
    Ok(values)
}

fn try_zeroed_bits<T: Clone + Default>(
    len: usize,
    name: &str,
) -> Result<Vec<T>, DeepSeekV4MetalError> {
    let mut values = try_bits(len, name)?;
    values.resize(len, T::default());
    Ok(values)
}

fn checked_product(factors: &[usize], name: &str) -> Result<usize, DeepSeekV4MetalError> {
    factors.iter().try_fold(1usize, |product, factor| {
        checked_mul(product, *factor, name)
    })
}

fn checked_add(a: usize, b: usize, name: &str) -> Result<usize, DeepSeekV4MetalError> {
    a.checked_add(b)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} overflows usize")))
}

fn require_len(name: &str, actual: usize, expected: usize) -> Result<(), DeepSeekV4MetalError> {
    if actual != expected {
        return invalid(format!(
            "{name} length {actual} differs from expected {expected}"
        ));
    }
    Ok(())
}

fn tensor_u16(tensor: &MetalTensor) -> &[u16] {
    // Persistent session tensors are shared-storage allocations. Callers
    // validate dtype, alignment, logical length, and backing-buffer bounds
    // before reaching this raw host view.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u16>(),
            tensor.n_elements() as usize,
        )
    }
}

fn tensor_u32(tensor: &MetalTensor) -> &[u32] {
    // See tensor_u16: this is restricted to validated shared session state.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<u32>(),
            tensor.n_elements() as usize,
        )
    }
}

fn zero_f16_bits(tensor: &MetalTensor) {
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>();
        std::ptr::write_bytes(destination, 0, tensor.n_elements() as usize);
    }
}

fn write_f16_prefix(tensor: &MetalTensor, bits: &[u16]) {
    assert!(
        bits.len() <= tensor.n_elements() as usize,
        "F16 snapshot prefix must fit its validated tensor"
    );
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>();
        std::ptr::copy_nonoverlapping(bits.as_ptr(), destination, bits.len());
    }
}

fn write_f16_bits(tensor: &MetalTensor, bits: &[u16]) {
    assert_eq!(
        tensor.n_elements() as usize,
        bits.len(),
        "F16 snapshot write length must match its validated tensor"
    );
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u16>();
        std::ptr::copy_nonoverlapping(bits.as_ptr(), destination, bits.len());
    }
}

fn write_u32_bits(tensor: &MetalTensor, bits: &[u32]) {
    assert_eq!(
        tensor.n_elements() as usize,
        bits.len(),
        "F32 snapshot write length must match its validated tensor"
    );
    unsafe {
        let destination = tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(tensor.offset as usize)
            .cast::<u32>();
        std::ptr::copy_nonoverlapping(bits.as_ptr(), destination, bits.len());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SyntheticState {
        capacity: DeepSeekV4SessionCapacity,
        raw_cache: MetalTensor,
        frontiers: DeepSeekV4CompressorFrontiers,
        phase: DeepSeekV4SessionPhase,
        committed_tokens: Vec<u32>,
    }

    fn model_id(byte: u8) -> DeepSeekV4ModelContentId {
        DeepSeekV4ModelContentId::new([byte; 32])
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }

    fn finite_bits(seed: u32, row: usize, dimension: usize, score: bool) -> u32 {
        let tag = seed as usize + row * 131 + dimension * 17 + usize::from(score) * 43;
        let value = (tag % 997) as f32 * 0.000_37 + if score { -0.41 } else { 0.13 };
        value.to_bits()
    }

    fn write_seeded_row(kv: &mut [u32], scores: &mut [u32], width: usize, row: usize, seed: u32) {
        for dimension in 0..width {
            let index = row * width + dimension;
            kv[index] = finite_bits(seed, row, dimension, false);
            scores[index] = finite_bits(seed, row, dimension, true);
        }
    }

    fn seed_frontier(frontier: &DeepSeekV4CompressorFrontier, next_position: u32, seed: u32) {
        let elements = frontier.width * frontier.rows;
        let mut kv = vec![0u32; elements];
        let mut scores = vec![f32::NEG_INFINITY.to_bits(); elements];
        if frontier.ratio == 4 {
            let complete_blocks = next_position as usize / 4;
            let partial = next_position as usize % 4;
            if complete_blocks == 0 {
                for row in 0..partial {
                    write_seeded_row(&mut kv, &mut scores, frontier.width, 4 + row, seed + 11);
                }
            } else {
                for row in 0..4 {
                    write_seeded_row(&mut kv, &mut scores, frontier.width, row, seed + 23);
                    let lower = row * frontier.width..(row + 1) * frontier.width;
                    let upper = (row + 4) * frontier.width..(row + 5) * frontier.width;
                    kv.copy_within(lower.clone(), upper.start);
                    scores.copy_within(lower, upper.start);
                }
                for row in 0..partial {
                    write_seeded_row(&mut kv, &mut scores, frontier.width, 4 + row, seed + 47);
                }
            }
        } else {
            let written = if next_position >= frontier.ratio as u32 {
                frontier.rows
            } else {
                next_position as usize % frontier.ratio
            };
            for row in 0..written {
                write_seeded_row(&mut kv, &mut scores, frontier.width, row, seed + 59);
            }
        }
        write_u32_bits(&frontier.kv_state, &kv);
        write_u32_bits(&frontier.score_state, &scores);

        let mut published = vec![0x7e00u16; frontier.head_dim * frontier.capacity_rows];
        let count = next_position as usize / frontier.ratio;
        for row in 0..count {
            for dimension in 0..frontier.head_dim {
                let value = f32::from_bits(finite_bits(seed + 71, row, dimension, false));
                published[row * frontier.head_dim + dimension] =
                    half::f16::from_f32(value).to_bits();
            }
        }
        write_f16_bits(&frontier.published, &published);
    }

    fn synthetic_state(
        ctx: &MetalContext,
        config: &DeepSeekV4Config,
        next_position: u32,
        observation: DeepSeekV4SnapshotObservation,
        seed: u32,
    ) -> SyntheticState {
        let capacity =
            DeepSeekV4SessionCapacity::for_forward_limit(3_073, config.context_length).unwrap();
        synthetic_state_with_capacity(ctx, config, capacity, next_position, observation, seed)
    }

    fn synthetic_state_with_capacity(
        ctx: &MetalContext,
        config: &DeepSeekV4Config,
        capacity: DeepSeekV4SessionCapacity,
        next_position: u32,
        observation: DeepSeekV4SnapshotObservation,
        seed: u32,
    ) -> SyntheticState {
        capacity.validate_next_position(next_position).unwrap();
        let raw_cache = MetalTensor::zeros_f16(
            ctx,
            vec![
                config.key_length as u64,
                DEEPSEEK_V4_LOCAL_WINDOW as u64,
                config.layer_count as u64,
            ],
        )
        .unwrap();
        let raw = (0..raw_cache.n_elements() as usize)
            .map(|index| {
                let value = (seed as usize + index * 19) % 521;
                half::f16::from_f32(value as f32 * 0.001_3 - 0.2).to_bits()
            })
            .collect::<Vec<_>>();
        write_f16_bits(&raw_cache, &raw);
        let frontiers = DeepSeekV4CompressorFrontiers::new(ctx, config, capacity).unwrap();
        for (layer, layer_frontiers) in frontiers.layers.iter().enumerate() {
            match layer_frontiers {
                DeepSeekV4LayerCompressorFrontiers::SlidingWindow => {}
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                    seed_frontier(attention, next_position, seed + layer as u32 * 5);
                    seed_frontier(indexer, next_position, seed + layer as u32 * 5 + 1);
                }
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => {
                    seed_frontier(attention, next_position, seed + layer as u32 * 5 + 2);
                }
            }
        }
        let phase = match observation {
            DeepSeekV4SnapshotObservation::Unavailable => {
                DeepSeekV4SessionPhase::ReadyWithoutObservation { next_position }
            }
            DeepSeekV4SnapshotObservation::Available => {
                DeepSeekV4SessionPhase::ReadyWithObservation { next_position }
            }
        };
        let mut committed_tokens = Vec::new();
        committed_tokens
            .try_reserve_exact(capacity.forward_limit())
            .unwrap();
        committed_tokens.extend((0..next_position).map(|position| position % config.vocab_size));
        SyntheticState {
            capacity,
            raw_cache,
            frontiers,
            phase,
            committed_tokens,
        }
    }

    fn capture_synthetic(
        config: &DeepSeekV4Config,
        state: &SyntheticState,
        id: DeepSeekV4ModelContentId,
    ) -> DeepSeekV4CausalSnapshot {
        capture_causal_state(
            config,
            state.capacity,
            state.phase,
            &state.committed_tokens,
            id,
            &state.raw_cache,
            &state.frontiers,
        )
        .unwrap()
    }

    #[test]
    fn snapshot_geometry_covers_every_promoted_boundary() {
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            config.context_length,
        )
        .unwrap();
        for (position, raw_rows, csa_rows, hca_rows) in [
            (0, 0, 0, 0),
            (1, 1, 0, 0),
            (3, 3, 0, 0),
            (4, 4, 1, 0),
            (127, 127, 31, 0),
            (128, 128, 32, 1),
            (129, 128, 32, 1),
            (255, 128, 63, 1),
            (256, 128, 64, 2),
            (1_024, 128, 256, 8),
            (1_025, 128, 256, 8),
            (1_027, 128, 256, 8),
            (1_028, 128, 257, 8),
            (2_047, 128, 511, 15),
            (2_048, 128, 512, 16),
            (2_049, 128, 512, 16),
            (2_051, 128, 512, 16),
            (2_052, 128, 513, 16),
            (2_053, 128, 513, 16),
            (2_175, 128, 543, 16),
            (2_176, 128, 544, 17),
            (2_177, 128, 544, 17),
            (3_071, 128, 767, 23),
            (3_072, 128, 768, 24),
            (3_073, 128, 768, 24),
            (3_075, 128, 768, 24),
            (3_076, 128, 769, 24),
            (32_768, 128, 8_192, 256),
            (65_535, 128, 16_383, 511),
            (65_536, 128, 16_384, 512),
            (65_663, 128, 16_415, 512),
            (65_664, 128, 16_416, 513),
            (1_000_000, 128, 250_000, 7_812),
            (1_048_575, 128, 262_143, 8_191),
            (1_048_576, 128, 262_144, 8_192),
        ] {
            let geometry = snapshot_geometry(&config, capacity, position).unwrap();
            assert_eq!(geometry.raw_rows, raw_rows, "position {position}");
            let expected_published = 21 * csa_rows * (512 + 128) + 20 * hca_rows * 512;
            assert_eq!(
                geometry.published_elements, expected_published,
                "position {position}"
            );
        }
        let terminal = snapshot_geometry(&config, capacity, 1_048_576).unwrap();
        assert_eq!(terminal.raw_elements, 2_818_048);
        assert_eq!(terminal.compressor_elements, 3_051_520);
        assert_eq!(terminal.published_elements, 3_607_101_440);
        assert_eq!(terminal.published_capacity_elements, 3_607_101_440);
        assert!(snapshot_geometry(&config, capacity, 1_048_577).is_err());
    }

    #[test]
    fn compatibility_digest_is_canonical_and_config_sensitive() {
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let digest = snapshot_compatibility_digest(model_id(0x5a), &config);
        assert_eq!(
            hex(digest.as_bytes()),
            "a09708848ad712566ead829a594f0850b3b1fa5e0a2344859c012e797d965896"
        );
        assert_ne!(
            digest,
            snapshot_compatibility_digest(model_id(0xa5), &config)
        );
        let mut drift = config.clone();
        drift.rope_freq_base = f32::from_bits(drift.rope_freq_base.to_bits() + 1);
        assert_ne!(
            digest,
            snapshot_compatibility_digest(model_id(0x5a), &drift)
        );
        let mut tokenizer_drift = config;
        tokenizer_drift.tokenizer_pre.push('x');
        assert_ne!(
            digest,
            snapshot_compatibility_digest(model_id(0x5a), &tokenizer_drift)
        );
    }

    #[test]
    fn causal_snapshot_roundtrip_preserves_bits_and_revokes_observation() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        for &position in &[
            0, 4, 127, 128, 129, 1_025, 1_028, 2_049, 2_052, 2_053, 2_176, 2_177, 3_072, 3_073,
        ] {
            let source = synthetic_state(
                &ctx,
                &config,
                position,
                DeepSeekV4SnapshotObservation::Available,
                17 + position,
            );
            let snapshot = capture_synthetic(&config, &source, model_id(0x5a));
            assert_eq!(
                snapshot.source_observation(),
                DeepSeekV4SnapshotObservation::Available
            );
            assert_eq!(snapshot.next_position(), position);

            let mut destination = synthetic_state(
                &ctx,
                &config,
                1,
                DeepSeekV4SnapshotObservation::Available,
                901,
            );
            restore_causal_state(
                &config,
                destination.capacity,
                &mut destination.phase,
                &mut destination.committed_tokens,
                model_id(0x5a),
                &destination.raw_cache,
                &destination.frontiers,
                &snapshot,
            )
            .unwrap();
            assert_eq!(
                destination.phase,
                DeepSeekV4SessionPhase::ReadyWithoutObservation {
                    next_position: position
                }
            );
            assert_eq!(destination.committed_tokens, snapshot.prefix_tokens());
            let restored = capture_synthetic(&config, &destination, model_id(0x5a));
            assert_eq!(
                restored.source_observation(),
                DeepSeekV4SnapshotObservation::Unavailable
            );
            assert_eq!(restored.prefix_tokens, snapshot.prefix_tokens);
            assert_eq!(restored.raw_f16_bits, snapshot.raw_f16_bits);
            assert_eq!(restored.compressor_f32_bits, snapshot.compressor_f32_bits);
            assert_eq!(restored.published_f16_bits, snapshot.published_f16_bits);
            assert_eq!(restored.causal_digest, snapshot.causal_digest);

            if position == 4 {
                let raw = tensor_u16(&destination.raw_cache);
                let width = config.key_length as usize;
                for layer in 0..config.layer_count as usize {
                    let unused = (layer * DEEPSEEK_V4_LOCAL_WINDOW + 4) * width;
                    assert!(
                        raw[unused..(layer + 1) * DEEPSEEK_V4_LOCAL_WINDOW * width]
                            .iter()
                            .all(|&bits| bits == 0)
                    );
                }
                for layer in &destination.frontiers.layers {
                    match layer {
                        DeepSeekV4LayerCompressorFrontiers::SlidingWindow => {}
                        DeepSeekV4LayerCompressorFrontiers::CompressedSparse {
                            attention,
                            indexer,
                        } => {
                            assert!(
                                tensor_u16(&attention.published)[attention.head_dim..]
                                    .iter()
                                    .all(|&bits| bits == 0)
                            );
                            assert!(
                                tensor_u16(&indexer.published)[indexer.head_dim..]
                                    .iter()
                                    .all(|&bits| bits == 0)
                            );
                        }
                        DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => {
                            assert!(
                                tensor_u16(&attention.published)
                                    .iter()
                                    .all(|&bits| bits == 0)
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn snapshot_v1_restores_canonically_across_physical_capacities() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let mut config = crate::deepseek_v4::flash_0731_config_fixture();
        config.key_length = 64;
        config.value_length = 64;
        config.attention_kinds = vec![AttentionKind::SlidingWindow; DEEPSEEK_V4_LAYER_COUNT];
        config.attention_kinds[2] = AttentionKind::CompressedSparse;
        config.attention_kinds[3] = AttentionKind::HeavilyCompressed;
        let source_capacity =
            DeepSeekV4SessionCapacity::for_forward_limit(3_073, config.context_length).unwrap();
        let destination_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            config.context_length,
        )
        .unwrap();
        let source = synthetic_state_with_capacity(
            &ctx,
            &config,
            source_capacity,
            3_072,
            DeepSeekV4SnapshotObservation::Unavailable,
            117,
        );
        let snapshot = capture_synthetic(&config, &source, model_id(0x5a));
        let mut destination = synthetic_state_with_capacity(
            &ctx,
            &config,
            destination_capacity,
            1,
            DeepSeekV4SnapshotObservation::Available,
            913,
        );
        restore_causal_state(
            &config,
            destination_capacity,
            &mut destination.phase,
            &mut destination.committed_tokens,
            model_id(0x5a),
            &destination.raw_cache,
            &destination.frontiers,
            &snapshot,
        )
        .unwrap();
        let restored = capture_synthetic(&config, &destination, model_id(0x5a));
        assert_eq!(restored.prefix_tokens, snapshot.prefix_tokens);
        assert_eq!(restored.raw_f16_bits, snapshot.raw_f16_bits);
        assert_eq!(restored.compressor_f32_bits, snapshot.compressor_f32_bits);
        assert_eq!(restored.published_f16_bits, snapshot.published_f16_bits);
        assert_eq!(restored.causal_digest, snapshot.causal_digest);
        assert_eq!(restored.compatibility_digest, snapshot.compatibility_digest);

        let encode = |snapshot: &DeepSeekV4CausalSnapshot,
                      session_capacity: DeepSeekV4SessionCapacity| {
            let mut bytes = Vec::new();
            encode_causal_snapshot(
                &mut bytes,
                snapshot,
                DeepSeekV4SnapshotCodecConstraints {
                    config: &config,
                    session_capacity,
                    expected_model_content_id: model_id(0x5a),
                    max_record_bytes: 1024 * 1024 * 1024,
                },
            )
            .unwrap();
            bytes
        };
        assert_eq!(
            encode(&snapshot, source_capacity),
            encode(&restored, destination_capacity)
        );

        for layer in &destination.frontiers.layers {
            match layer {
                DeepSeekV4LayerCompressorFrontiers::SlidingWindow => {}
                DeepSeekV4LayerCompressorFrontiers::CompressedSparse { attention, indexer } => {
                    let visible = 768;
                    assert!(
                        tensor_u16(&attention.published)[visible * attention.head_dim..]
                            .iter()
                            .all(|&bits| bits == 0)
                    );
                    assert!(
                        tensor_u16(&indexer.published)[visible * indexer.head_dim..]
                            .iter()
                            .all(|&bits| bits == 0)
                    );
                }
                DeepSeekV4LayerCompressorFrontiers::HeavilyCompressed { attention } => {
                    let visible = 24;
                    assert!(
                        tensor_u16(&attention.published)[visible * attention.head_dim..]
                            .iter()
                            .all(|&bits| bits == 0)
                    );
                }
            }
        }
    }

    #[test]
    fn larger_capacity_snapshot_rejects_smaller_destination_atomically() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let mut config = crate::deepseek_v4::flash_0731_config_fixture();
        config.key_length = 64;
        config.value_length = 64;
        config.attention_kinds = vec![AttentionKind::SlidingWindow; DEEPSEEK_V4_LAYER_COUNT];
        config.attention_kinds[2] = AttentionKind::CompressedSparse;
        config.attention_kinds[3] = AttentionKind::HeavilyCompressed;
        let source_capacity = DeepSeekV4SessionCapacity::for_forward_limit(
            DEEPSEEK_V4_PROMOTED_FORWARD_CAPACITY,
            config.context_length,
        )
        .unwrap();
        let destination_capacity =
            DeepSeekV4SessionCapacity::for_forward_limit(3_075, config.context_length).unwrap();
        assert_eq!(source_capacity.csa_physical_rows(), 262_144);
        assert_eq!(destination_capacity.csa_physical_rows(), 768);

        let source = synthetic_state_with_capacity(
            &ctx,
            &config,
            source_capacity,
            3_076,
            DeepSeekV4SnapshotObservation::Unavailable,
            211,
        );
        let snapshot = capture_synthetic(&config, &source, model_id(0x5a));
        let mut destination = synthetic_state_with_capacity(
            &ctx,
            &config,
            destination_capacity,
            1,
            DeepSeekV4SnapshotObservation::Available,
            977,
        );
        let before = capture_synthetic(&config, &destination, model_id(0x5a));
        let error = restore_causal_state(
            &config,
            destination_capacity,
            &mut destination.phase,
            &mut destination.committed_tokens,
            model_id(0x5a),
            &destination.raw_cache,
            &destination.frontiers,
            &snapshot,
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("state position 3076 exceeds session capacity 3075")
        );
        assert_eq!(
            capture_synthetic(&config, &destination, model_id(0x5a)),
            before
        );
    }

    #[test]
    fn snapshot_rejection_is_preflight_atomic_and_poison_is_ineligible() {
        let Ok(ctx) = MetalContext::new() else {
            return;
        };
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let source = synthetic_state(
            &ctx,
            &config,
            4,
            DeepSeekV4SnapshotObservation::Available,
            33,
        );
        let snapshot = capture_synthetic(&config, &source, model_id(0x5a));
        let mut destination = synthetic_state(
            &ctx,
            &config,
            1,
            DeepSeekV4SnapshotObservation::Available,
            77,
        );
        let before = capture_synthetic(&config, &destination, model_id(0x5a));

        let mut truncated = snapshot.clone();
        truncated.raw_f16_bits = truncated.raw_f16_bits[..truncated.raw_f16_bits.len() - 1]
            .to_vec()
            .into_boxed_slice();
        truncated.refresh_digests();
        let error = restore_causal_state(
            &config,
            destination.capacity,
            &mut destination.phase,
            &mut destination.committed_tokens,
            model_id(0x5a),
            &destination.raw_cache,
            &destination.frontiers,
            &truncated,
        )
        .unwrap_err();
        assert!(error.to_string().contains("raw F16 arena length"));
        assert_eq!(
            capture_synthetic(&config, &destination, model_id(0x5a)),
            before
        );

        let mut nonfinite = snapshot.clone();
        nonfinite.compressor_f32_bits[0] = f32::NAN.to_bits();
        nonfinite.refresh_digests();
        let error = restore_causal_state(
            &config,
            destination.capacity,
            &mut destination.phase,
            &mut destination.committed_tokens,
            model_id(0x5a),
            &destination.raw_cache,
            &destination.frontiers,
            &nonfinite,
        )
        .unwrap_err();
        assert!(error.to_string().contains("not finite"));
        assert_eq!(
            capture_synthetic(&config, &destination, model_id(0x5a)),
            before
        );

        let error = restore_causal_state(
            &config,
            destination.capacity,
            &mut destination.phase,
            &mut destination.committed_tokens,
            model_id(0xa5),
            &destination.raw_cache,
            &destination.frontiers,
            &snapshot,
        )
        .unwrap_err();
        assert!(error.to_string().contains("model-content identity"));
        assert_eq!(
            capture_synthetic(&config, &destination, model_id(0x5a)),
            before
        );

        destination.phase = DeepSeekV4SessionPhase::Poisoned { next_position: 1 };
        assert!(
            capture_causal_state(
                &config,
                destination.capacity,
                destination.phase,
                &destination.committed_tokens,
                model_id(0x5a),
                &destination.raw_cache,
                &destination.frontiers,
            )
            .is_err()
        );
        assert!(
            restore_causal_state(
                &config,
                destination.capacity,
                &mut destination.phase,
                &mut destination.committed_tokens,
                model_id(0x5a),
                &destination.raw_cache,
                &destination.frontiers,
                &snapshot,
            )
            .is_err()
        );
        assert_eq!(
            destination.phase,
            DeepSeekV4SessionPhase::Poisoned { next_position: 1 }
        );
    }
}
