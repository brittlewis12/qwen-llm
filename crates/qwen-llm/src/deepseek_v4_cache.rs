//! Typed sequence-cache semantics for DeepSeek V4.
//!
//! This module is deliberately disconnected from the Qwen runtime and from
//! Metal generation dispatch. It models the state and visibility contract that
//! a later DeepSeek V4 session must preserve, including whole-token atomicity.
//!
//! A transaction stages layers strictly in model order. Its views include the
//! current raw row and any compressed row completed by that token; commit
//! publishes all layers together, while drop or error publishes none. Completed
//! histories grow lazily in fixed-row F32 oracle slabs. Full snapshots deep
//! copy those histories and are correctness artifacts, not a durable or
//! long-context production encoding.

use crate::deepseek_v4::{AttentionKind, DeepSeekV4Config, DeepSeekV4Error};
use crate::deepseek_v4_oracle::{
    CompressorState, DeepSeekV4OracleError, RopeParameters,
    attention_fp8_nope_bf16_rope_roundtrip_in_place, indexer_qat_roundtrip_in_place,
};

const HISTORY_CHUNK_ROWS: usize = 256;

pub type CacheResult<T> = Result<T, DeepSeekV4CacheError>;

#[derive(Debug, thiserror::Error)]
pub enum DeepSeekV4CacheError {
    #[error("invalid {name}: {detail}")]
    Invalid { name: &'static str, detail: String },
    #[error("{name} length mismatch: expected {expected}, got {actual}")]
    Length {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("allocation failed for {name}: {detail}")]
    Allocation { name: &'static str, detail: String },
    #[error("DeepSeek V4 cache profile is incompatible: {0}")]
    Profile(#[source] DeepSeekV4Error),
    #[error(transparent)]
    Oracle(#[from] DeepSeekV4OracleError),
}

impl From<DeepSeekV4Error> for DeepSeekV4CacheError {
    fn from(error: DeepSeekV4Error) -> Self {
        Self::Profile(error)
    }
}

/// Storage semantics used by the CPU cache spike.
///
/// Attention rows remain as decoded F32 values after the official mixed
/// FP8-NoPE/BF16-RoPE quantize/dequantize round trip. Indexer rows remain as
/// decoded F32 values after Hadamard-128 and MXFP4 QAT. Compressor frontiers
/// remain F32. This is an explicit numerical-oracle format, not the packed
/// production cache ABI planned for the Metal performance phase.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DeepSeekV4CacheStorageFormat {
    F32QuantizationOracleV1,
}

/// Strong checkpoint-and-numerics identity supplied by the eventual DS4
/// session. Geometry alone cannot make weight-derived cache state compatible.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct DeepSeekV4CacheIdentity([u8; 32]);

impl DeepSeekV4CacheIdentity {
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeepSeekV4CacheLayout {
    context_length: u32,
    local_window: usize,
    attention_dim: usize,
    rotary_dim: usize,
    indexer_dim: usize,
    attention_kinds: Vec<AttentionKind>,
    compressor_rms_epsilon: f32,
    compressor_rope: RopeParameters,
}

impl DeepSeekV4CacheLayout {
    pub fn from_config(config: &DeepSeekV4Config) -> CacheResult<Self> {
        let layer_count = usize::try_from(config.layer_count)
            .map_err(|_| invalid_error("cache layer count", "does not fit usize"))?;
        let local_window = usize::try_from(config.sliding_window)
            .map_err(|_| invalid_error("cache local window", "does not fit usize"))?;
        let attention_dim = usize::try_from(config.key_length)
            .map_err(|_| invalid_error("cache attention dimension", "does not fit usize"))?;
        let rotary_dim = usize::try_from(config.rope_dimension_count)
            .map_err(|_| invalid_error("cache rotary dimension", "does not fit usize"))?;
        let indexer_dim = usize::try_from(config.indexer_key_length)
            .map_err(|_| invalid_error("cache indexer dimension", "does not fit usize"))?;
        if config.kv_head_count != 1 {
            return invalid(
                "cache KV heads",
                "shared-KV cache requires exactly one head",
            );
        }
        if config.key_length != config.value_length {
            return invalid(
                "cache shared-KV width",
                "key and value dimensions must be equal",
            );
        }
        if config.attention_kinds.len() != layer_count {
            return length(
                "cache layer schedule",
                layer_count,
                config.attention_kinds.len(),
            );
        }
        if config.rope_scaling_type != "yarn" {
            return invalid(
                "cache compressor RoPE scaling",
                "the current DS4 cache oracle requires YaRN",
            );
        }
        let mut attention_kinds = Vec::new();
        try_reserve_exact(
            &mut attention_kinds,
            config.attention_kinds.len(),
            "cache layer schedule",
        )?;
        attention_kinds.extend_from_slice(&config.attention_kinds);
        let layout = Self {
            context_length: config.context_length,
            local_window,
            attention_dim,
            rotary_dim,
            indexer_dim,
            attention_kinds,
            compressor_rms_epsilon: config.attention_rms_epsilon,
            compressor_rope: RopeParameters::yarn(
                rotary_dim,
                config.compress_rope_freq_base,
                config.rope_scaling_factor,
                config.rope_original_context_length,
                config.rope_yarn_beta_fast,
                config.rope_yarn_beta_slow,
            ),
        };
        layout.validate()?;
        Ok(layout)
    }

    pub fn flash_0731(config: &DeepSeekV4Config) -> CacheResult<Self> {
        config.validate_flash_0731_profile()?;
        Self::from_config(config)
    }

    pub fn context_length(&self) -> u32 {
        self.context_length
    }

    pub fn local_window(&self) -> usize {
        self.local_window
    }

    pub fn attention_dim(&self) -> usize {
        self.attention_dim
    }

    pub fn rotary_dim(&self) -> usize {
        self.rotary_dim
    }

    pub fn indexer_dim(&self) -> usize {
        self.indexer_dim
    }

    pub fn layer_count(&self) -> usize {
        self.attention_kinds.len()
    }

    pub fn attention_kinds(&self) -> &[AttentionKind] {
        &self.attention_kinds
    }

    pub fn compressor_rms_epsilon(&self) -> f32 {
        self.compressor_rms_epsilon
    }

    pub fn compressor_rope(&self) -> RopeParameters {
        self.compressor_rope
    }

    pub fn layer_kind(&self, layer: usize) -> CacheResult<AttentionKind> {
        self.attention_kinds
            .get(layer)
            .copied()
            .ok_or_else(|| invalid_error("cache layer", &format!("index {layer} is out of range")))
    }

    pub fn max_attention_rows(&self, layer: usize) -> CacheResult<usize> {
        let ratio = self.layer_kind(layer)?.ratio();
        if ratio == 0 {
            return Ok(0);
        }
        usize::try_from(self.context_length / ratio)
            .map_err(|_| invalid_error("compressed cache capacity", "does not fit usize"))
    }

    pub fn max_indexer_rows(&self, layer: usize) -> CacheResult<usize> {
        Ok(
            if self.layer_kind(layer)? == AttentionKind::CompressedSparse {
                self.max_attention_rows(layer)?
            } else {
                0
            },
        )
    }

    fn validate(&self) -> CacheResult<()> {
        if self.context_length == 0 {
            return invalid("cache context length", "must be nonzero");
        }
        if self.local_window == 0 {
            return invalid("cache local window", "must be nonzero");
        }
        if u64::try_from(self.local_window)
            .map_err(|_| invalid_error("cache local window", "does not fit u64"))?
            > u64::from(self.context_length)
        {
            return invalid("cache local window", "must not exceed context length");
        }
        if self.attention_dim == 0 {
            return invalid("cache attention dimension", "must be nonzero");
        }
        if self.rotary_dim == 0
            || self.rotary_dim > self.attention_dim
            || !self.rotary_dim.is_multiple_of(2)
        {
            return invalid(
                "cache rotary dimension",
                "must be even, nonzero, and no larger than the attention dimension",
            );
        }
        let nope_dim = self.attention_dim - self.rotary_dim;
        if !nope_dim.is_multiple_of(64) {
            return invalid(
                "cache NoPE dimension",
                "must be divisible by the 64-value FP8 block width",
            );
        }
        if self.indexer_dim != 128 {
            return invalid(
                "cache indexer dimension",
                "the current Hadamard/MXFP4 oracle requires exactly 128 values",
            );
        }
        if self
            .attention_kinds
            .contains(&AttentionKind::CompressedSparse)
            && self.rotary_dim > self.indexer_dim
        {
            return invalid(
                "cache indexer rotary dimension",
                "must not exceed the CSA indexer cache dimension",
            );
        }
        if self.attention_kinds.is_empty() {
            return invalid("cache layer schedule", "must be nonempty");
        }
        if !self.compressor_rms_epsilon.is_finite() || self.compressor_rms_epsilon <= 0.0 {
            return invalid(
                "cache compressor RMSNorm epsilon",
                "must be finite and positive",
            );
        }
        if self.compressor_rope.rotary_dim != self.rotary_dim
            || !self.compressor_rope.theta.is_finite()
            || self.compressor_rope.theta <= 1.0
            || !self.compressor_rope.scaling_factor.is_finite()
            || self.compressor_rope.scaling_factor < 1.0
            || (self.compressor_rope.scaling_factor > 1.0
                && (self.compressor_rope.original_context_length == 0
                    || !self.compressor_rope.beta_fast.is_finite()
                    || self.compressor_rope.beta_fast <= 0.0
                    || !self.compressor_rope.beta_slow.is_finite()
                    || self.compressor_rope.beta_slow <= 0.0))
        {
            return invalid(
                "cache compressor RoPE",
                "parameters are inconsistent with the cache rotary geometry",
            );
        }
        let _ = self
            .local_window
            .checked_mul(self.attention_dim)
            .ok_or_else(|| invalid_error("raw cache dimensions", "calculation overflowed"))?;
        for (layer, kind) in self.attention_kinds.iter().copied().enumerate() {
            if let Some(rows) = self.context_length.checked_div(kind.ratio()) {
                let rows = usize::try_from(rows).map_err(|_| {
                    invalid_error("compressed cache capacity", "does not fit usize")
                })?;
                let _ = rows.checked_mul(self.attention_dim).ok_or_else(|| {
                    invalid_error(
                        "compressed cache dimensions",
                        &format!("layer {layer} calculation overflowed"),
                    )
                })?;
                if kind == AttentionKind::CompressedSparse {
                    let _ = rows.checked_mul(self.indexer_dim).ok_or_else(|| {
                        invalid_error(
                            "indexer cache dimensions",
                            &format!("layer {layer} calculation overflowed"),
                        )
                    })?;
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4AttentionRowRef<'a> {
    start_position: u32,
    values: &'a [f32],
}

impl<'a> DeepSeekV4AttentionRowRef<'a> {
    pub fn start_position(self) -> u32 {
        self.start_position
    }

    pub fn values(self) -> &'a [f32] {
        self.values
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeepSeekV4IndexerRowRef<'a> {
    start_position: u32,
    values: &'a [f32],
}

impl<'a> DeepSeekV4IndexerRowRef<'a> {
    pub fn start_position(self) -> u32 {
        self.start_position
    }

    pub fn values(self) -> &'a [f32] {
        self.values
    }
}

pub struct DeepSeekV4LayerView<'a> {
    kind: AttentionKind,
    raw_rows: Vec<DeepSeekV4AttentionRowRef<'a>>,
    compressed_rows: Option<&'a AttentionHistory>,
    staged_compressed_row: Option<&'a AttentionCacheRow>,
    indexer_rows: Option<&'a IndexerHistory>,
    staged_indexer_row: Option<&'a IndexerCacheRow>,
}

impl<'a> DeepSeekV4LayerView<'a> {
    pub fn kind(&self) -> AttentionKind {
        self.kind
    }

    pub fn raw_rows(&self) -> &[DeepSeekV4AttentionRowRef<'a>] {
        &self.raw_rows
    }

    pub fn compressed_rows(&self) -> DeepSeekV4AttentionRows<'a> {
        DeepSeekV4AttentionRows::new(self.compressed_rows, self.staged_compressed_row)
    }

    pub fn indexer_rows(&self) -> DeepSeekV4IndexerRows<'a> {
        DeepSeekV4IndexerRows::new(self.indexer_rows, self.staged_indexer_row)
    }
}

#[derive(Clone)]
pub struct DeepSeekV4AttentionRows<'a> {
    history: Option<&'a AttentionHistory>,
    index: usize,
    staged: Option<&'a AttentionCacheRow>,
}

impl<'a> DeepSeekV4AttentionRows<'a> {
    fn new(history: Option<&'a AttentionHistory>, staged: Option<&'a AttentionCacheRow>) -> Self {
        Self {
            history,
            index: 0,
            staged,
        }
    }
}

impl<'a> Iterator for DeepSeekV4AttentionRows<'a> {
    type Item = DeepSeekV4AttentionRowRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(history) = self.history
            && self.index < history.row_count()
        {
            let index = self.index;
            self.index += 1;
            return Some(DeepSeekV4AttentionRowRef {
                start_position: index as u32 * history.0.ratio,
                values: history.0.row(index),
            });
        }
        self.staged.take().map(AttentionCacheRow::as_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for DeepSeekV4AttentionRows<'_> {
    fn len(&self) -> usize {
        self.history.map_or(0, AttentionHistory::row_count) - self.index
            + usize::from(self.staged.is_some())
    }
}

#[derive(Clone)]
pub struct DeepSeekV4IndexerRows<'a> {
    history: Option<&'a IndexerHistory>,
    index: usize,
    staged: Option<&'a IndexerCacheRow>,
}

impl<'a> DeepSeekV4IndexerRows<'a> {
    fn new(history: Option<&'a IndexerHistory>, staged: Option<&'a IndexerCacheRow>) -> Self {
        Self {
            history,
            index: 0,
            staged,
        }
    }
}

impl<'a> Iterator for DeepSeekV4IndexerRows<'a> {
    type Item = DeepSeekV4IndexerRowRef<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(history) = self.history
            && self.index < history.row_count()
        {
            let index = self.index;
            self.index += 1;
            return Some(DeepSeekV4IndexerRowRef {
                start_position: index as u32 * history.0.ratio,
                values: history.0.row(index),
            });
        }
        self.staged.take().map(IndexerCacheRow::as_ref)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for DeepSeekV4IndexerRows<'_> {
    fn len(&self) -> usize {
        self.history.map_or(0, IndexerHistory::row_count) - self.index
            + usize::from(self.staged.is_some())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DeepSeekV4LayerCacheStats {
    pub kind: AttentionKind,
    pub raw_rows: usize,
    pub compressed_rows: usize,
    pub indexer_rows: usize,
    pub max_compressed_rows: usize,
    pub max_indexer_rows: usize,
}

#[derive(Debug, PartialEq)]
struct AttentionCacheRow {
    start_position: u32,
    values: Vec<f32>,
}

impl AttentionCacheRow {
    fn as_ref(&self) -> DeepSeekV4AttentionRowRef<'_> {
        DeepSeekV4AttentionRowRef {
            start_position: self.start_position,
            values: &self.values,
        }
    }
}

#[derive(Debug, PartialEq)]
struct IndexerCacheRow {
    start_position: u32,
    values: Vec<f32>,
}

impl IndexerCacheRow {
    fn as_ref(&self) -> DeepSeekV4IndexerRowRef<'_> {
        DeepSeekV4IndexerRowRef {
            start_position: self.start_position,
            values: &self.values,
        }
    }
}

#[derive(Debug, PartialEq)]
struct DenseHistory {
    ratio: u32,
    width: usize,
    row_count: usize,
    chunks: Vec<Vec<f32>>,
}

impl DenseHistory {
    fn new(ratio: u32, width: usize) -> Self {
        Self {
            ratio,
            width,
            row_count: 0,
            chunks: Vec::new(),
        }
    }

    fn row_count(&self) -> usize {
        self.row_count
    }

    fn row(&self, index: usize) -> &[f32] {
        let chunk = index / HISTORY_CHUNK_ROWS;
        let row = index % HISTORY_CHUNK_ROWS;
        let start = row * self.width;
        &self.chunks[chunk][start..start + self.width]
    }

    fn prepare_append(&mut self, name: &'static str) -> CacheResult<Option<Vec<f32>>> {
        if !self.row_count.is_multiple_of(HISTORY_CHUNK_ROWS) {
            let chunk = self
                .chunks
                .last_mut()
                .ok_or_else(|| invalid_error(name, "partial history has no backing slab"))?;
            try_reserve(chunk, self.width, name)?;
            return Ok(None);
        }
        try_reserve(&mut self.chunks, 1, name)?;
        let chunk_elements = self
            .width
            .checked_mul(HISTORY_CHUNK_ROWS)
            .ok_or_else(|| invalid_error(name, "chunk dimensions overflowed"))?;
        let mut chunk = Vec::new();
        try_reserve_exact(&mut chunk, chunk_elements, name)?;
        Ok(Some(chunk))
    }

    fn append(&mut self, start_position: u32, values: &[f32], prepared_chunk: Option<Vec<f32>>) {
        debug_assert_eq!(values.len(), self.width);
        debug_assert_eq!(
            start_position,
            self.row_count() as u32 * self.ratio,
            "history rows must remain append-only and position-derived"
        );
        if self.row_count.is_multiple_of(HISTORY_CHUNK_ROWS) {
            let chunk = prepared_chunk.expect("new history slab was preflighted before commit");
            debug_assert!(chunk.is_empty());
            debug_assert!(chunk.capacity() >= self.width * HISTORY_CHUNK_ROWS);
            self.chunks.push(chunk);
        } else {
            debug_assert!(prepared_chunk.is_none());
        }
        let chunk = self
            .chunks
            .last_mut()
            .expect("history contains a slab after append preflight");
        debug_assert!(chunk.capacity() - chunk.len() >= self.width);
        chunk.extend_from_slice(values);
        self.row_count += 1;
    }

    fn validate(
        &self,
        expected_ratio: u32,
        expected_width: usize,
        next_position: u64,
        maximum_rows: usize,
        name: &'static str,
    ) -> CacheResult<()> {
        if self.ratio != expected_ratio {
            return invalid(
                name,
                &format!("expected ratio {expected_ratio}, got {}", self.ratio),
            );
        }
        if self.width != expected_width || self.width == 0 {
            return invalid(
                name,
                &format!("expected row width {expected_width}, got {}", self.width),
            );
        }
        let expected_rows = usize::try_from(next_position / u64::from(expected_ratio))
            .map_err(|_| invalid_error(name, "row count does not fit usize"))?;
        if self.row_count() != expected_rows {
            return length(name, expected_rows, self.row_count());
        }
        if self.row_count() > maximum_rows {
            return invalid(name, "row count exceeds configured capacity");
        }
        let expected_chunks = self.row_count.div_ceil(HISTORY_CHUNK_ROWS);
        if self.chunks.len() != expected_chunks {
            return length(name, expected_chunks, self.chunks.len());
        }
        let full_chunk_len = self
            .width
            .checked_mul(HISTORY_CHUNK_ROWS)
            .ok_or_else(|| invalid_error(name, "chunk dimensions overflowed"))?;
        for (index, chunk) in self.chunks.iter().enumerate() {
            let rows = if index + 1 < expected_chunks
                || self.row_count.is_multiple_of(HISTORY_CHUNK_ROWS)
            {
                HISTORY_CHUNK_ROWS
            } else {
                self.row_count % HISTORY_CHUNK_ROWS
            };
            let expected_len = rows
                .checked_mul(self.width)
                .ok_or_else(|| invalid_error(name, "chunk length overflowed"))?;
            if chunk.len() != expected_len || chunk.len() > full_chunk_len {
                return invalid(
                    name,
                    &format!(
                        "chunk {index} expected {expected_len} values, got {}",
                        chunk.len()
                    ),
                );
            }
            require_finite(name, chunk)?;
        }
        Ok(())
    }

    fn fallible_clone(&self, name: &'static str) -> CacheResult<Self> {
        let mut chunks = Vec::new();
        try_reserve_exact(&mut chunks, self.chunks.len(), name)?;
        let chunk_elements = self
            .width
            .checked_mul(HISTORY_CHUNK_ROWS)
            .ok_or_else(|| invalid_error(name, "chunk dimensions overflowed"))?;
        for source in &self.chunks {
            let mut chunk = Vec::new();
            try_reserve_exact(&mut chunk, chunk_elements, name)?;
            chunk.extend_from_slice(source);
            chunks.push(chunk);
        }
        Ok(Self {
            ratio: self.ratio,
            width: self.width,
            row_count: self.row_count,
            chunks,
        })
    }
}

#[derive(Debug, PartialEq)]
struct AttentionHistory(DenseHistory);

impl AttentionHistory {
    fn new(ratio: u32, width: usize) -> Self {
        Self(DenseHistory::new(ratio, width))
    }

    fn row_count(&self) -> usize {
        self.0.row_count()
    }

    fn prepare_append(&mut self, name: &'static str) -> CacheResult<Option<Vec<f32>>> {
        self.0.prepare_append(name)
    }

    fn append(&mut self, row: AttentionCacheRow, prepared_chunk: Option<Vec<f32>>) {
        self.0
            .append(row.start_position, &row.values, prepared_chunk);
    }
}

#[derive(Debug, PartialEq)]
struct IndexerHistory(DenseHistory);

impl IndexerHistory {
    fn new(ratio: u32, width: usize) -> Self {
        Self(DenseHistory::new(ratio, width))
    }

    fn row_count(&self) -> usize {
        self.0.row_count()
    }

    fn prepare_append(&mut self, name: &'static str) -> CacheResult<Option<Vec<f32>>> {
        self.0.prepare_append(name)
    }

    fn append(&mut self, row: IndexerCacheRow, prepared_chunk: Option<Vec<f32>>) {
        self.0
            .append(row.start_position, &row.values, prepared_chunk);
    }
}

#[derive(Debug, PartialEq)]
struct RawAttentionRing {
    slots: Vec<Option<AttentionCacheRow>>,
}

impl RawAttentionRing {
    fn new(window: usize) -> CacheResult<Self> {
        let mut slots = Vec::new();
        try_reserve_exact(&mut slots, window, "raw cache ring")?;
        slots.resize_with(window, || None);
        Ok(Self { slots })
    }

    fn row(&self, position: u64) -> Option<&AttentionCacheRow> {
        let slot = (position % self.slots.len() as u64) as usize;
        self.slots[slot]
            .as_ref()
            .filter(|row| u64::from(row.start_position) == position)
    }

    fn insert(&mut self, row: AttentionCacheRow) {
        let slot = (u64::from(row.start_position) % self.slots.len() as u64) as usize;
        self.slots[slot] = Some(row);
    }

    fn visible_count(&self, next_position: u64) -> usize {
        usize::try_from(next_position.min(self.slots.len() as u64)).unwrap_or(self.slots.len())
    }

    fn validate(&self, next_position: u64, width: usize) -> CacheResult<()> {
        if self.slots.is_empty() {
            return invalid("raw cache ring", "must be nonempty");
        }
        let start = next_position.saturating_sub(self.slots.len() as u64);
        let expected_count = usize::try_from(next_position - start)
            .map_err(|_| invalid_error("raw cache row count", "does not fit usize"))?;
        let mut actual_count = 0usize;
        for (slot, row) in self.slots.iter().enumerate() {
            let Some(row) = row else {
                continue;
            };
            actual_count += 1;
            let position = u64::from(row.start_position);
            if position < start || position >= next_position {
                return invalid(
                    "raw cache ring",
                    &format!("slot {slot} contains invisible position {position}"),
                );
            }
            if position % self.slots.len() as u64 != slot as u64 {
                return invalid(
                    "raw cache ring",
                    &format!("position {position} is stored in the wrong slot {slot}"),
                );
            }
            validate_attention_row(row, width)?;
        }
        if actual_count != expected_count {
            return length("raw cache visible rows", expected_count, actual_count);
        }
        for position in start..next_position {
            if self.row(position).is_none() {
                return invalid(
                    "raw cache ring",
                    &format!("missing visible position {position}"),
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
enum DeepSeekV4LayerCache {
    SlidingWindow {
        raw: RawAttentionRing,
    },
    CompressedSparse {
        raw: RawAttentionRing,
        attention_state: CompressorState,
        attention_rows: AttentionHistory,
        indexer_state: CompressorState,
        indexer_rows: IndexerHistory,
    },
    HeavilyCompressed {
        raw: RawAttentionRing,
        attention_state: CompressorState,
        attention_rows: AttentionHistory,
    },
}

impl DeepSeekV4LayerCache {
    fn new(kind: AttentionKind, layout: &DeepSeekV4CacheLayout) -> CacheResult<Self> {
        let raw = RawAttentionRing::new(layout.local_window)?;
        Ok(match kind {
            AttentionKind::SlidingWindow => Self::SlidingWindow { raw },
            AttentionKind::CompressedSparse => Self::CompressedSparse {
                raw,
                attention_state: CompressorState::new(4, layout.attention_dim)?,
                attention_rows: AttentionHistory::new(4, layout.attention_dim),
                indexer_state: CompressorState::new(4, layout.indexer_dim)?,
                indexer_rows: IndexerHistory::new(4, layout.indexer_dim),
            },
            AttentionKind::HeavilyCompressed => Self::HeavilyCompressed {
                raw,
                attention_state: CompressorState::new(128, layout.attention_dim)?,
                attention_rows: AttentionHistory::new(128, layout.attention_dim),
            },
        })
    }

    fn kind(&self) -> AttentionKind {
        match self {
            Self::SlidingWindow { .. } => AttentionKind::SlidingWindow,
            Self::CompressedSparse { .. } => AttentionKind::CompressedSparse,
            Self::HeavilyCompressed { .. } => AttentionKind::HeavilyCompressed,
        }
    }

    fn raw(&self) -> &RawAttentionRing {
        match self {
            Self::SlidingWindow { raw }
            | Self::CompressedSparse { raw, .. }
            | Self::HeavilyCompressed { raw, .. } => raw,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DeepSeekV4CompressorUpdate<'a> {
    /// Width is `head_dim` for ratio 128 and `2 * head_dim` for ratio 4.
    pub projected_kv: &'a [f32],
    /// Same width as `projected_kv`; APE is added by the cache-bound profile.
    pub projected_scores: &'a [f32],
    /// Row-major `[ratio, projection_width]` positional score table.
    pub ape: &'a [f32],
    /// RMSNorm weight with one value per emitted cache dimension.
    pub norm_weight: &'a [f32],
}

#[derive(Clone, Copy, Debug)]
pub enum DeepSeekV4LayerUpdate<'a> {
    SlidingWindow {
        /// Normalized shared-KV row after forward local RoPE and before cache
        /// quantization.
        raw_kv: &'a [f32],
    },
    CompressedSparse {
        raw_kv: &'a [f32],
        attention: DeepSeekV4CompressorUpdate<'a>,
        indexer: DeepSeekV4CompressorUpdate<'a>,
    },
    HeavilyCompressed {
        raw_kv: &'a [f32],
        attention: DeepSeekV4CompressorUpdate<'a>,
    },
}

#[derive(Debug, PartialEq)]
pub struct DeepSeekV4Cache {
    identity: DeepSeekV4CacheIdentity,
    layout: DeepSeekV4CacheLayout,
    storage_format: DeepSeekV4CacheStorageFormat,
    next_position: u64,
    layers: Vec<DeepSeekV4LayerCache>,
}

impl DeepSeekV4Cache {
    pub fn from_config(
        config: &DeepSeekV4Config,
        identity: DeepSeekV4CacheIdentity,
    ) -> CacheResult<Self> {
        Self::new(DeepSeekV4CacheLayout::from_config(config)?, identity)
    }

    pub fn flash_0731(
        config: &DeepSeekV4Config,
        identity: DeepSeekV4CacheIdentity,
    ) -> CacheResult<Self> {
        Self::new(DeepSeekV4CacheLayout::flash_0731(config)?, identity)
    }

    pub fn new(
        layout: DeepSeekV4CacheLayout,
        identity: DeepSeekV4CacheIdentity,
    ) -> CacheResult<Self> {
        layout.validate()?;
        let mut layers = Vec::new();
        try_reserve_exact(
            &mut layers,
            layout.layer_count(),
            "DeepSeek V4 cache layers",
        )?;
        for kind in layout.attention_kinds.iter().copied() {
            layers.push(DeepSeekV4LayerCache::new(kind, &layout)?);
        }
        Ok(Self {
            identity,
            layout,
            storage_format: DeepSeekV4CacheStorageFormat::F32QuantizationOracleV1,
            next_position: 0,
            layers,
        })
    }

    pub fn layout(&self) -> &DeepSeekV4CacheLayout {
        &self.layout
    }

    pub fn identity(&self) -> DeepSeekV4CacheIdentity {
        self.identity
    }

    pub fn storage_format(&self) -> DeepSeekV4CacheStorageFormat {
        self.storage_format
    }

    pub fn next_position(&self) -> u64 {
        self.next_position
    }

    pub fn layer_stats(&self, layer: usize) -> CacheResult<DeepSeekV4LayerCacheStats> {
        let cache = self.layers.get(layer).ok_or_else(|| {
            invalid_error("cache layer", &format!("index {layer} is out of range"))
        })?;
        let (compressed_rows, indexer_rows) = match cache {
            DeepSeekV4LayerCache::SlidingWindow { .. } => (0, 0),
            DeepSeekV4LayerCache::CompressedSparse {
                attention_rows,
                indexer_rows,
                ..
            } => (attention_rows.row_count(), indexer_rows.row_count()),
            DeepSeekV4LayerCache::HeavilyCompressed { attention_rows, .. } => {
                (attention_rows.row_count(), 0)
            }
        };
        Ok(DeepSeekV4LayerCacheStats {
            kind: cache.kind(),
            raw_rows: cache.raw().visible_count(self.next_position),
            compressed_rows,
            indexer_rows,
            max_compressed_rows: self.layout.max_attention_rows(layer)?,
            max_indexer_rows: self.layout.max_indexer_rows(layer)?,
        })
    }

    pub fn layer_view(&self, layer: usize) -> CacheResult<DeepSeekV4LayerView<'_>> {
        let cache = self.layers.get(layer).ok_or_else(|| {
            invalid_error("cache layer", &format!("index {layer} is out of range"))
        })?;
        build_layer_view(cache, self.next_position, self.layout.local_window, None)
    }

    /// Starts a detached whole-token transaction.
    ///
    /// Call `stage_layer` exactly once for each layer in ascending order. A
    /// staged `layer_view` has same-token visibility. Dropping the transaction
    /// or returning from a staging error leaves logical cache state unchanged.
    pub fn begin_token(&mut self, position: u32) -> CacheResult<DeepSeekV4CacheTransaction<'_>> {
        if u64::from(position) != self.next_position {
            return invalid(
                "cache token position",
                &format!("expected {}, got {position}", self.next_position),
            );
        }
        if position >= self.layout.context_length {
            return invalid(
                "cache token position",
                &format!(
                    "position {position} reaches context limit {}",
                    self.layout.context_length
                ),
            );
        }
        let mut staged = Vec::new();
        try_reserve_exact(
            &mut staged,
            self.layers.len(),
            "DeepSeek V4 token transaction",
        )?;
        Ok(DeepSeekV4CacheTransaction {
            cache: self,
            position,
            staged,
        })
    }

    /// Captures a full in-memory correctness snapshot.
    ///
    /// This fallibly deep-copies all visible F32 oracle histories. It is not the
    /// future packed Metal checkpoint ABI and should not be used for 1M-context
    /// memory projections.
    pub fn snapshot(&self) -> CacheResult<DeepSeekV4CacheSnapshot> {
        let mut layers = Vec::new();
        try_reserve_exact(&mut layers, self.layers.len(), "cache snapshot layers")?;
        for layer in &self.layers {
            layers.push(LayerSnapshot::capture(layer)?);
        }
        Ok(DeepSeekV4CacheSnapshot {
            identity: self.identity,
            layout: self.layout.clone(),
            storage_format: self.storage_format,
            next_position: self.next_position,
            layers,
        })
    }

    /// Reconstructs a cache only when the caller's expected compatibility
    /// contract matches the snapshot's checkpoint identity, layout, and
    /// storage format.
    pub fn from_snapshot(
        snapshot: DeepSeekV4CacheSnapshot,
        expected_identity: DeepSeekV4CacheIdentity,
        expected_layout: &DeepSeekV4CacheLayout,
        expected_storage_format: DeepSeekV4CacheStorageFormat,
    ) -> CacheResult<Self> {
        if snapshot.identity != expected_identity {
            return invalid(
                "cache snapshot identity",
                "does not match the expected checkpoint and numerics",
            );
        }
        if snapshot.layout != *expected_layout {
            return invalid(
                "cache snapshot layout",
                "does not match the expected cache layout",
            );
        }
        if snapshot.storage_format != expected_storage_format {
            return invalid(
                "cache snapshot storage format",
                "does not match the expected storage format",
            );
        }
        Self::from_snapshot_contents(snapshot)
    }

    fn from_snapshot_contents(snapshot: DeepSeekV4CacheSnapshot) -> CacheResult<Self> {
        snapshot.layout.validate()?;
        if snapshot.next_position > u64::from(snapshot.layout.context_length) {
            return invalid(
                "cache snapshot position",
                "exceeds the configured context length",
            );
        }
        if snapshot.layers.len() != snapshot.layout.layer_count() {
            return length(
                "cache snapshot layers",
                snapshot.layout.layer_count(),
                snapshot.layers.len(),
            );
        }

        let mut layers = Vec::new();
        try_reserve_exact(&mut layers, snapshot.layers.len(), "restored cache layers")?;
        for (layer_index, (kind, layer)) in snapshot
            .layout
            .attention_kinds
            .iter()
            .copied()
            .zip(snapshot.layers)
            .enumerate()
        {
            layers.push(layer.restore(
                layer_index,
                kind,
                &snapshot.layout,
                snapshot.next_position,
            )?);
        }
        Ok(Self {
            identity: snapshot.identity,
            layout: snapshot.layout,
            storage_format: snapshot.storage_format,
            next_position: snapshot.next_position,
            layers,
        })
    }

    pub fn restore(&mut self, snapshot: DeepSeekV4CacheSnapshot) -> CacheResult<()> {
        if snapshot.identity != self.identity {
            return invalid(
                "cache snapshot identity",
                "does not match the destination checkpoint and numerics",
            );
        }
        if snapshot.layout != self.layout {
            return invalid(
                "cache snapshot layout",
                "does not match the destination cache",
            );
        }
        if snapshot.storage_format != self.storage_format {
            return invalid(
                "cache snapshot storage format",
                "does not match the destination cache",
            );
        }
        let candidate = Self::from_snapshot_contents(snapshot)?;
        *self = candidate;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
enum StagedLayer {
    SlidingWindow {
        raw: AttentionCacheRow,
    },
    CompressedSparse {
        raw: AttentionCacheRow,
        attention_state: CompressorState,
        attention_row: Option<AttentionCacheRow>,
        attention_chunk: Option<Vec<f32>>,
        indexer_state: CompressorState,
        indexer_row: Option<IndexerCacheRow>,
        indexer_chunk: Option<Vec<f32>>,
    },
    HeavilyCompressed {
        raw: AttentionCacheRow,
        attention_state: CompressorState,
        attention_row: Option<AttentionCacheRow>,
        attention_chunk: Option<Vec<f32>>,
    },
}

impl StagedLayer {
    fn kind(&self) -> AttentionKind {
        match self {
            Self::SlidingWindow { .. } => AttentionKind::SlidingWindow,
            Self::CompressedSparse { .. } => AttentionKind::CompressedSparse,
            Self::HeavilyCompressed { .. } => AttentionKind::HeavilyCompressed,
        }
    }

    fn raw(&self) -> &AttentionCacheRow {
        match self {
            Self::SlidingWindow { raw }
            | Self::CompressedSparse { raw, .. }
            | Self::HeavilyCompressed { raw, .. } => raw,
        }
    }
}

pub struct DeepSeekV4CacheTransaction<'a> {
    cache: &'a mut DeepSeekV4Cache,
    position: u32,
    staged: Vec<StagedLayer>,
}

impl<'a> DeepSeekV4CacheTransaction<'a> {
    pub fn position(&self) -> u32 {
        self.position
    }

    pub fn next_layer(&self) -> usize {
        self.staged.len()
    }

    pub fn stage_layer(
        &mut self,
        layer: usize,
        update: DeepSeekV4LayerUpdate<'_>,
    ) -> CacheResult<()> {
        if layer != self.staged.len() {
            return invalid(
                "cache transaction layer order",
                &format!("expected layer {}, got {layer}", self.staged.len()),
            );
        }
        let base = self.cache.layers.get(layer).ok_or_else(|| {
            invalid_error("cache layer", &format!("index {layer} is out of range"))
        })?;
        let expected_kind = base.kind();
        let actual_kind = match update {
            DeepSeekV4LayerUpdate::SlidingWindow { .. } => AttentionKind::SlidingWindow,
            DeepSeekV4LayerUpdate::CompressedSparse { .. } => AttentionKind::CompressedSparse,
            DeepSeekV4LayerUpdate::HeavilyCompressed { .. } => AttentionKind::HeavilyCompressed,
        };
        if actual_kind != expected_kind {
            return invalid(
                "cache transaction layer kind",
                &format!("layer {layer} expects {expected_kind:?}, got {actual_kind:?}"),
            );
        }

        let staged = match (base, update) {
            (
                DeepSeekV4LayerCache::SlidingWindow { .. },
                DeepSeekV4LayerUpdate::SlidingWindow { raw_kv },
            ) => StagedLayer::SlidingWindow {
                raw: store_attention_row(
                    self.position,
                    raw_kv,
                    &self.cache.layout,
                    self.cache.storage_format,
                )?,
            },
            (
                DeepSeekV4LayerCache::CompressedSparse {
                    attention_state,
                    attention_rows,
                    indexer_state,
                    indexer_rows,
                    ..
                },
                DeepSeekV4LayerUpdate::CompressedSparse {
                    raw_kv,
                    attention,
                    indexer,
                },
            ) => {
                validate_history_frontier(attention_rows.row_count(), self.position, 4, layer)?;
                validate_history_frontier(indexer_rows.row_count(), self.position, 4, layer)?;
                let (next_attention_state, emitted_attention) = attention_state
                    .prepare_projected_push(
                        self.position,
                        attention.projected_kv,
                        attention.projected_scores,
                        attention.ape,
                        attention.norm_weight,
                        self.cache.layout.compressor_rms_epsilon,
                        self.cache.layout.compressor_rope,
                    )?;
                let attention_row = emitted_attention
                    .map(|row| {
                        store_attention_row(
                            row.start_position,
                            &row.value,
                            &self.cache.layout,
                            self.cache.storage_format,
                        )
                    })
                    .transpose()?;
                let (next_indexer_state, emitted_indexer) = indexer_state.prepare_projected_push(
                    self.position,
                    indexer.projected_kv,
                    indexer.projected_scores,
                    indexer.ape,
                    indexer.norm_weight,
                    self.cache.layout.compressor_rms_epsilon,
                    self.cache.layout.compressor_rope,
                )?;
                let indexer_row = emitted_indexer
                    .map(|row| {
                        store_indexer_row(
                            row.start_position,
                            &row.value,
                            self.cache.layout.indexer_dim,
                            self.cache.storage_format,
                        )
                    })
                    .transpose()?;
                match (&attention_row, &indexer_row) {
                    (Some(attention), Some(indexer))
                        if attention.start_position == indexer.start_position => {}
                    (None, None) => {}
                    _ => {
                        return invalid(
                            "CSA compressor alignment",
                            "attention and indexer compressors emitted different groups",
                        );
                    }
                }
                validate_staged_capacity(
                    attention_rows.row_count(),
                    attention_row.is_some(),
                    self.cache.layout.max_attention_rows(layer)?,
                    "CSA attention cache",
                )?;
                validate_staged_capacity(
                    indexer_rows.row_count(),
                    indexer_row.is_some(),
                    self.cache.layout.max_indexer_rows(layer)?,
                    "CSA indexer cache",
                )?;
                StagedLayer::CompressedSparse {
                    raw: store_attention_row(
                        self.position,
                        raw_kv,
                        &self.cache.layout,
                        self.cache.storage_format,
                    )?,
                    attention_state: next_attention_state,
                    attention_row,
                    attention_chunk: None,
                    indexer_state: next_indexer_state,
                    indexer_row,
                    indexer_chunk: None,
                }
            }
            (
                DeepSeekV4LayerCache::HeavilyCompressed {
                    attention_state,
                    attention_rows,
                    ..
                },
                DeepSeekV4LayerUpdate::HeavilyCompressed { raw_kv, attention },
            ) => {
                validate_history_frontier(attention_rows.row_count(), self.position, 128, layer)?;
                let (next_attention_state, emitted_attention) = attention_state
                    .prepare_projected_push(
                        self.position,
                        attention.projected_kv,
                        attention.projected_scores,
                        attention.ape,
                        attention.norm_weight,
                        self.cache.layout.compressor_rms_epsilon,
                        self.cache.layout.compressor_rope,
                    )?;
                let attention_row = emitted_attention
                    .map(|row| {
                        store_attention_row(
                            row.start_position,
                            &row.value,
                            &self.cache.layout,
                            self.cache.storage_format,
                        )
                    })
                    .transpose()?;
                validate_staged_capacity(
                    attention_rows.row_count(),
                    attention_row.is_some(),
                    self.cache.layout.max_attention_rows(layer)?,
                    "HCA attention cache",
                )?;
                StagedLayer::HeavilyCompressed {
                    raw: store_attention_row(
                        self.position,
                        raw_kv,
                        &self.cache.layout,
                        self.cache.storage_format,
                    )?,
                    attention_state: next_attention_state,
                    attention_row,
                    attention_chunk: None,
                }
            }
            _ => {
                return invalid(
                    "cache transaction layer kind",
                    "validated layer kind changed during staging",
                );
            }
        };
        self.staged.push(staged);
        Ok(())
    }

    pub fn layer_view(&self, layer: usize) -> CacheResult<DeepSeekV4LayerView<'_>> {
        let staged = self.staged.get(layer).ok_or_else(|| {
            invalid_error(
                "cache transaction layer view",
                &format!("layer {layer} has not been staged"),
            )
        })?;
        let base = &self.cache.layers[layer];
        if base.kind() != staged.kind() {
            return invalid(
                "cache transaction layer view",
                "base and staged layer kinds differ",
            );
        }
        build_layer_view(
            base,
            u64::from(self.position) + 1,
            self.cache.layout.local_window,
            Some(staged),
        )
    }

    pub fn commit(mut self) -> CacheResult<()> {
        if self.staged.len() != self.cache.layers.len() {
            return length(
                "staged cache layers",
                self.cache.layers.len(),
                self.staged.len(),
            );
        }
        for (base, staged) in self.cache.layers.iter_mut().zip(&mut self.staged) {
            match (base, staged) {
                (DeepSeekV4LayerCache::SlidingWindow { .. }, StagedLayer::SlidingWindow { .. }) => {
                }
                (
                    DeepSeekV4LayerCache::CompressedSparse {
                        attention_rows,
                        indexer_rows,
                        ..
                    },
                    StagedLayer::CompressedSparse {
                        attention_row,
                        attention_chunk,
                        indexer_row,
                        indexer_chunk,
                        ..
                    },
                ) => {
                    if attention_row.is_some() {
                        *attention_chunk =
                            attention_rows.prepare_append("CSA attention history")?;
                    }
                    if indexer_row.is_some() {
                        *indexer_chunk = indexer_rows.prepare_append("CSA indexer history")?;
                    }
                }
                (
                    DeepSeekV4LayerCache::HeavilyCompressed { attention_rows, .. },
                    StagedLayer::HeavilyCompressed {
                        attention_row,
                        attention_chunk,
                        ..
                    },
                ) => {
                    if attention_row.is_some() {
                        *attention_chunk =
                            attention_rows.prepare_append("HCA attention history")?;
                    }
                }
                _ => {
                    return invalid(
                        "cache transaction commit",
                        "base and staged layer kinds differ",
                    );
                }
            }
        }

        let staged = std::mem::take(&mut self.staged);
        for (base, staged) in self.cache.layers.iter_mut().zip(staged) {
            match (base, staged) {
                (
                    DeepSeekV4LayerCache::SlidingWindow { raw },
                    StagedLayer::SlidingWindow { raw: next_raw },
                ) => raw.insert(next_raw),
                (
                    DeepSeekV4LayerCache::CompressedSparse {
                        raw,
                        attention_state,
                        attention_rows,
                        indexer_state,
                        indexer_rows,
                    },
                    StagedLayer::CompressedSparse {
                        raw: next_raw,
                        attention_state: next_attention_state,
                        attention_row,
                        attention_chunk,
                        indexer_state: next_indexer_state,
                        indexer_row,
                        indexer_chunk,
                    },
                ) => {
                    raw.insert(next_raw);
                    *attention_state = next_attention_state;
                    if let Some(row) = attention_row {
                        attention_rows.append(row, attention_chunk);
                    }
                    *indexer_state = next_indexer_state;
                    if let Some(row) = indexer_row {
                        indexer_rows.append(row, indexer_chunk);
                    }
                }
                (
                    DeepSeekV4LayerCache::HeavilyCompressed {
                        raw,
                        attention_state,
                        attention_rows,
                    },
                    StagedLayer::HeavilyCompressed {
                        raw: next_raw,
                        attention_state: next_attention_state,
                        attention_row,
                        attention_chunk,
                    },
                ) => {
                    raw.insert(next_raw);
                    *attention_state = next_attention_state;
                    if let Some(row) = attention_row {
                        attention_rows.append(row, attention_chunk);
                    }
                }
                _ => unreachable!("transaction variants were validated before commit"),
            }
        }
        self.cache.next_position = u64::from(self.position) + 1;
        Ok(())
    }
}

#[derive(Debug, PartialEq)]
pub struct DeepSeekV4CacheSnapshot {
    identity: DeepSeekV4CacheIdentity,
    layout: DeepSeekV4CacheLayout,
    storage_format: DeepSeekV4CacheStorageFormat,
    next_position: u64,
    layers: Vec<LayerSnapshot>,
}

impl DeepSeekV4CacheSnapshot {
    pub fn identity(&self) -> DeepSeekV4CacheIdentity {
        self.identity
    }

    pub fn layout(&self) -> &DeepSeekV4CacheLayout {
        &self.layout
    }

    pub fn storage_format(&self) -> DeepSeekV4CacheStorageFormat {
        self.storage_format
    }

    pub fn next_position(&self) -> u64 {
        self.next_position
    }
}

#[derive(Debug, PartialEq)]
struct CompressorSnapshot {
    ratio: usize,
    head_dim: usize,
    next_position: u64,
    kv: Vec<f32>,
    scores: Vec<f32>,
}

impl CompressorSnapshot {
    fn capture(state: &CompressorState) -> CacheResult<Self> {
        Ok(Self {
            ratio: state.ratio(),
            head_dim: state.head_dim(),
            next_position: state.next_position(),
            kv: copy_slice(state.kv_state(), "compressor KV snapshot")?,
            scores: copy_slice(state.score_state(), "compressor score snapshot")?,
        })
    }

    fn restore(
        self,
        expected_ratio: usize,
        expected_head_dim: usize,
        expected_position: u64,
    ) -> CacheResult<CompressorState> {
        if self.ratio != expected_ratio {
            return invalid(
                "compressor snapshot ratio",
                &format!("expected {expected_ratio}, got {}", self.ratio),
            );
        }
        if self.head_dim != expected_head_dim {
            return invalid(
                "compressor snapshot head dimension",
                &format!("expected {expected_head_dim}, got {}", self.head_dim),
            );
        }
        if self.next_position != expected_position {
            return invalid(
                "compressor snapshot position",
                &format!("expected {expected_position}, got {}", self.next_position),
            );
        }
        Ok(CompressorState::from_snapshot(
            self.ratio,
            self.head_dim,
            self.next_position,
            self.kv,
            self.scores,
        )?)
    }
}

#[derive(Debug, PartialEq)]
struct RawRingSnapshot {
    slots: Vec<Option<AttentionCacheRow>>,
}

impl RawRingSnapshot {
    fn capture(ring: &RawAttentionRing) -> CacheResult<Self> {
        let mut slots = Vec::new();
        try_reserve_exact(&mut slots, ring.slots.len(), "raw ring snapshot")?;
        for row in &ring.slots {
            slots.push(row.as_ref().map(clone_attention_row).transpose()?);
        }
        Ok(Self { slots })
    }

    fn restore(
        self,
        window: usize,
        next_position: u64,
        width: usize,
    ) -> CacheResult<RawAttentionRing> {
        if self.slots.len() != window {
            return length("raw ring snapshot slots", window, self.slots.len());
        }
        let ring = RawAttentionRing { slots: self.slots };
        ring.validate(next_position, width)?;
        Ok(ring)
    }
}

#[derive(Debug, PartialEq)]
enum LayerSnapshot {
    SlidingWindow {
        raw: RawRingSnapshot,
    },
    CompressedSparse {
        raw: RawRingSnapshot,
        attention_state: CompressorSnapshot,
        attention_rows: AttentionHistory,
        indexer_state: CompressorSnapshot,
        indexer_rows: IndexerHistory,
    },
    HeavilyCompressed {
        raw: RawRingSnapshot,
        attention_state: CompressorSnapshot,
        attention_rows: AttentionHistory,
    },
}

impl LayerSnapshot {
    fn capture(layer: &DeepSeekV4LayerCache) -> CacheResult<Self> {
        Ok(match layer {
            DeepSeekV4LayerCache::SlidingWindow { raw } => Self::SlidingWindow {
                raw: RawRingSnapshot::capture(raw)?,
            },
            DeepSeekV4LayerCache::CompressedSparse {
                raw,
                attention_state,
                attention_rows,
                indexer_state,
                indexer_rows,
            } => Self::CompressedSparse {
                raw: RawRingSnapshot::capture(raw)?,
                attention_state: CompressorSnapshot::capture(attention_state)?,
                attention_rows: AttentionHistory(
                    attention_rows
                        .0
                        .fallible_clone("attention history snapshot")?,
                ),
                indexer_state: CompressorSnapshot::capture(indexer_state)?,
                indexer_rows: IndexerHistory(
                    indexer_rows.0.fallible_clone("indexer history snapshot")?,
                ),
            },
            DeepSeekV4LayerCache::HeavilyCompressed {
                raw,
                attention_state,
                attention_rows,
            } => Self::HeavilyCompressed {
                raw: RawRingSnapshot::capture(raw)?,
                attention_state: CompressorSnapshot::capture(attention_state)?,
                attention_rows: AttentionHistory(
                    attention_rows
                        .0
                        .fallible_clone("attention history snapshot")?,
                ),
            },
        })
    }

    fn restore(
        self,
        layer: usize,
        expected_kind: AttentionKind,
        layout: &DeepSeekV4CacheLayout,
        next_position: u64,
    ) -> CacheResult<DeepSeekV4LayerCache> {
        match (expected_kind, self) {
            (AttentionKind::SlidingWindow, Self::SlidingWindow { raw }) => {
                Ok(DeepSeekV4LayerCache::SlidingWindow {
                    raw: raw.restore(layout.local_window, next_position, layout.attention_dim)?,
                })
            }
            (
                AttentionKind::CompressedSparse,
                Self::CompressedSparse {
                    raw,
                    attention_state,
                    attention_rows,
                    indexer_state,
                    indexer_rows,
                },
            ) => {
                attention_rows.0.validate(
                    4,
                    layout.attention_dim,
                    next_position,
                    layout.max_attention_rows(layer)?,
                    "attention history rows",
                )?;
                indexer_rows.0.validate(
                    4,
                    layout.indexer_dim,
                    next_position,
                    layout.max_indexer_rows(layer)?,
                    "indexer history rows",
                )?;
                if attention_rows.row_count() != indexer_rows.row_count() {
                    return invalid(
                        "CSA snapshot history alignment",
                        "attention and indexer row counts differ",
                    );
                }
                Ok(DeepSeekV4LayerCache::CompressedSparse {
                    raw: raw.restore(layout.local_window, next_position, layout.attention_dim)?,
                    attention_state: attention_state.restore(
                        4,
                        layout.attention_dim,
                        next_position,
                    )?,
                    attention_rows,
                    indexer_state: indexer_state.restore(4, layout.indexer_dim, next_position)?,
                    indexer_rows,
                })
            }
            (
                AttentionKind::HeavilyCompressed,
                Self::HeavilyCompressed {
                    raw,
                    attention_state,
                    attention_rows,
                },
            ) => {
                attention_rows.0.validate(
                    128,
                    layout.attention_dim,
                    next_position,
                    layout.max_attention_rows(layer)?,
                    "attention history rows",
                )?;
                Ok(DeepSeekV4LayerCache::HeavilyCompressed {
                    raw: raw.restore(layout.local_window, next_position, layout.attention_dim)?,
                    attention_state: attention_state.restore(
                        128,
                        layout.attention_dim,
                        next_position,
                    )?,
                    attention_rows,
                })
            }
            (expected, actual) => invalid(
                "cache snapshot layer kind",
                &format!(
                    "layer {layer} expects {expected:?}, got {:?}",
                    actual.kind()
                ),
            ),
        }
    }

    fn kind(&self) -> AttentionKind {
        match self {
            Self::SlidingWindow { .. } => AttentionKind::SlidingWindow,
            Self::CompressedSparse { .. } => AttentionKind::CompressedSparse,
            Self::HeavilyCompressed { .. } => AttentionKind::HeavilyCompressed,
        }
    }
}

fn build_layer_view<'a>(
    base: &'a DeepSeekV4LayerCache,
    visible_end: u64,
    local_window: usize,
    staged: Option<&'a StagedLayer>,
) -> CacheResult<DeepSeekV4LayerView<'a>> {
    let raw_start = visible_end.saturating_sub(local_window as u64);
    let raw_count = usize::try_from(visible_end - raw_start)
        .map_err(|_| invalid_error("raw cache view", "row count does not fit usize"))?;
    let mut raw_rows = Vec::new();
    try_reserve_exact(&mut raw_rows, raw_count, "raw cache view")?;
    for position in raw_start..visible_end {
        let staged_row = staged
            .map(StagedLayer::raw)
            .filter(|row| u64::from(row.start_position) == position);
        let row = if let Some(row) = staged_row {
            row
        } else {
            base.raw().row(position).ok_or_else(|| {
                invalid_error(
                    "raw cache view",
                    &format!("missing visible position {position}"),
                )
            })?
        };
        raw_rows.push(row.as_ref());
    }

    let (compressed_rows, staged_compressed_row, indexer_rows, staged_indexer_row) =
        match (base, staged) {
            (DeepSeekV4LayerCache::SlidingWindow { .. }, None)
            | (
                DeepSeekV4LayerCache::SlidingWindow { .. },
                Some(StagedLayer::SlidingWindow { .. }),
            ) => (None, None, None, None),
            (
                DeepSeekV4LayerCache::CompressedSparse {
                    attention_rows,
                    indexer_rows: base_indexer_rows,
                    ..
                },
                staged,
            ) => {
                let staged_attention = match staged {
                    Some(StagedLayer::CompressedSparse { attention_row, .. }) => {
                        attention_row.as_ref()
                    }
                    None => None,
                    _ => {
                        return invalid("cache layer view", "base and staged layer kinds differ");
                    }
                };
                let staged_indexer = match staged {
                    Some(StagedLayer::CompressedSparse { indexer_row, .. }) => indexer_row.as_ref(),
                    None => None,
                    _ => {
                        return invalid("cache layer view", "base and staged layer kinds differ");
                    }
                };
                (
                    Some(attention_rows),
                    staged_attention,
                    Some(base_indexer_rows),
                    staged_indexer,
                )
            }
            (DeepSeekV4LayerCache::HeavilyCompressed { attention_rows, .. }, staged) => {
                let staged_attention = match staged {
                    Some(StagedLayer::HeavilyCompressed { attention_row, .. }) => {
                        attention_row.as_ref()
                    }
                    None => None,
                    _ => {
                        return invalid("cache layer view", "base and staged layer kinds differ");
                    }
                };
                (Some(attention_rows), staged_attention, None, None)
            }
            _ => {
                return invalid("cache layer view", "base and staged layer kinds differ");
            }
        };
    Ok(DeepSeekV4LayerView {
        kind: base.kind(),
        raw_rows,
        compressed_rows,
        staged_compressed_row,
        indexer_rows,
        staged_indexer_row,
    })
}

fn store_attention_row(
    start_position: u32,
    values: &[f32],
    layout: &DeepSeekV4CacheLayout,
    format: DeepSeekV4CacheStorageFormat,
) -> CacheResult<AttentionCacheRow> {
    if values.len() != layout.attention_dim {
        return length("attention cache row", layout.attention_dim, values.len());
    }
    let mut values = copy_slice(values, "attention cache row")?;
    match format {
        DeepSeekV4CacheStorageFormat::F32QuantizationOracleV1 => {
            attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut values, layout.rotary_dim)?;
        }
    }
    Ok(AttentionCacheRow {
        start_position,
        values,
    })
}

fn store_indexer_row(
    start_position: u32,
    values: &[f32],
    expected_dim: usize,
    format: DeepSeekV4CacheStorageFormat,
) -> CacheResult<IndexerCacheRow> {
    if values.len() != expected_dim {
        return length("indexer cache row", expected_dim, values.len());
    }
    let mut values = copy_slice(values, "indexer cache row")?;
    match format {
        DeepSeekV4CacheStorageFormat::F32QuantizationOracleV1 => {
            indexer_qat_roundtrip_in_place(&mut values)?;
        }
    }
    Ok(IndexerCacheRow {
        start_position,
        values,
    })
}

fn validate_history_frontier(
    actual_rows: usize,
    position: u32,
    ratio: u32,
    layer: usize,
) -> CacheResult<()> {
    let expected = usize::try_from(position / ratio)
        .map_err(|_| invalid_error("compressed cache frontier", "does not fit usize"))?;
    if actual_rows != expected {
        return invalid(
            "compressed cache frontier",
            &format!(
                "layer {layer} ratio {ratio} expected {expected} rows before position {position}, got {actual_rows}"
            ),
        );
    }
    Ok(())
}

fn validate_staged_capacity(
    existing: usize,
    emitted: bool,
    maximum: usize,
    name: &'static str,
) -> CacheResult<()> {
    let following = existing
        .checked_add(usize::from(emitted))
        .ok_or_else(|| invalid_error(name, "row count overflowed"))?;
    if following > maximum {
        return invalid(name, "would exceed the configured context capacity");
    }
    Ok(())
}

fn validate_attention_row(row: &AttentionCacheRow, width: usize) -> CacheResult<()> {
    if row.values.len() != width {
        return length("attention cache row", width, row.values.len());
    }
    require_finite("attention cache row", &row.values)
}

fn clone_attention_row(row: &AttentionCacheRow) -> CacheResult<AttentionCacheRow> {
    Ok(AttentionCacheRow {
        start_position: row.start_position,
        values: copy_slice(&row.values, "attention row snapshot")?,
    })
}

fn copy_slice(values: &[f32], name: &'static str) -> CacheResult<Vec<f32>> {
    let mut output = Vec::new();
    try_reserve_exact(&mut output, values.len(), name)?;
    output.extend_from_slice(values);
    Ok(output)
}

fn try_reserve<T>(values: &mut Vec<T>, additional: usize, name: &'static str) -> CacheResult<()> {
    values
        .try_reserve(additional)
        .map_err(|error| DeepSeekV4CacheError::Allocation {
            name,
            detail: error.to_string(),
        })
}

fn try_reserve_exact<T>(
    values: &mut Vec<T>,
    additional: usize,
    name: &'static str,
) -> CacheResult<()> {
    values
        .try_reserve_exact(additional)
        .map_err(|error| DeepSeekV4CacheError::Allocation {
            name,
            detail: error.to_string(),
        })
}

fn require_finite(name: &'static str, values: &[f32]) -> CacheResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return invalid(name, "must contain only finite values");
    }
    Ok(())
}

fn length<T>(name: &'static str, expected: usize, actual: usize) -> CacheResult<T> {
    Err(DeepSeekV4CacheError::Length {
        name,
        expected,
        actual,
    })
}

fn invalid<T>(name: &'static str, detail: &str) -> CacheResult<T> {
    Err(invalid_error(name, detail))
}

fn invalid_error(name: &'static str, detail: &str) -> DeepSeekV4CacheError {
    DeepSeekV4CacheError::Invalid {
        name,
        detail: detail.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, PartialEq)]
    struct LayerObservation {
        raw_positions: Vec<u32>,
        compressed_positions: Vec<u32>,
        indexer_positions: Vec<u32>,
    }

    impl LayerObservation {
        fn capture(view: &DeepSeekV4LayerView<'_>) -> Self {
            Self {
                raw_positions: view
                    .raw_rows()
                    .iter()
                    .map(|row| row.start_position())
                    .collect(),
                compressed_positions: view
                    .compressed_rows()
                    .map(|row| row.start_position())
                    .collect(),
                indexer_positions: view
                    .indexer_rows()
                    .map(|row| row.start_position())
                    .collect(),
            }
        }
    }

    struct TestVectors {
        csa_attention_kv: Vec<f32>,
        csa_attention_scores: Vec<f32>,
        csa_attention_ape: Vec<f32>,
        csa_indexer_kv: Vec<f32>,
        csa_indexer_scores: Vec<f32>,
        csa_indexer_ape: Vec<f32>,
        hca_kv: Vec<f32>,
        hca_scores: Vec<f32>,
        hca_ape: Vec<f32>,
        norm: Vec<f32>,
    }

    impl TestVectors {
        fn new() -> Self {
            Self {
                csa_attention_kv: values(256, 0.003, -0.2),
                csa_attention_scores: values(256, 0.0007, -0.05),
                csa_attention_ape: values(4 * 256, 0.0001, -0.01),
                csa_indexer_kv: values(256, -0.002, 0.3),
                csa_indexer_scores: values(256, 0.0005, -0.03),
                csa_indexer_ape: values(4 * 256, -0.00008, 0.02),
                hca_kv: values(128, 0.004, -0.25),
                hca_scores: values(128, -0.0006, 0.04),
                hca_ape: values(128 * 128, 0.00001, -0.005),
                norm: vec![1.0; 128],
            }
        }

        fn csa_attention(&self) -> DeepSeekV4CompressorUpdate<'_> {
            DeepSeekV4CompressorUpdate {
                projected_kv: &self.csa_attention_kv,
                projected_scores: &self.csa_attention_scores,
                ape: &self.csa_attention_ape,
                norm_weight: &self.norm,
            }
        }

        fn csa_indexer(&self) -> DeepSeekV4CompressorUpdate<'_> {
            DeepSeekV4CompressorUpdate {
                projected_kv: &self.csa_indexer_kv,
                projected_scores: &self.csa_indexer_scores,
                ape: &self.csa_indexer_ape,
                norm_weight: &self.norm,
            }
        }

        fn hca(&self) -> DeepSeekV4CompressorUpdate<'_> {
            DeepSeekV4CompressorUpdate {
                projected_kv: &self.hca_kv,
                projected_scores: &self.hca_scores,
                ape: &self.hca_ape,
                norm_weight: &self.norm,
            }
        }
    }

    fn values(len: usize, scale: f32, offset: f32) -> Vec<f32> {
        (0..len)
            .map(|index| offset + (index % 97) as f32 * scale)
            .collect()
    }

    fn raw_values(position: u32, layer: usize) -> Vec<f32> {
        (0..128)
            .map(|index| {
                position as f32 * 0.001 + layer as f32 * 0.02 + index as f32 * 0.0003 - 0.1
            })
            .collect()
    }

    fn test_layout(
        attention_kinds: Vec<AttentionKind>,
        local_window: usize,
        context_length: u32,
    ) -> DeepSeekV4CacheLayout {
        let layout = DeepSeekV4CacheLayout {
            context_length,
            local_window,
            attention_dim: 128,
            rotary_dim: 64,
            indexer_dim: 128,
            attention_kinds,
            compressor_rms_epsilon: 1e-6,
            compressor_rope: RopeParameters::local(64, 160_000.0),
        };
        layout.validate().unwrap();
        layout
    }

    fn three_layer_cache(local_window: usize, context_length: u32) -> DeepSeekV4Cache {
        DeepSeekV4Cache::new(
            test_layout(
                vec![
                    AttentionKind::SlidingWindow,
                    AttentionKind::CompressedSparse,
                    AttentionKind::HeavilyCompressed,
                ],
                local_window,
                context_length,
            ),
            test_identity(),
        )
        .unwrap()
    }

    fn test_identity() -> DeepSeekV4CacheIdentity {
        DeepSeekV4CacheIdentity::new([0x5a; 32])
    }

    fn restore_copy(cache: &DeepSeekV4Cache) -> DeepSeekV4Cache {
        DeepSeekV4Cache::from_snapshot(
            cache.snapshot().unwrap(),
            cache.identity(),
            cache.layout(),
            cache.storage_format(),
        )
        .unwrap()
    }

    fn push_three_layers(
        cache: &mut DeepSeekV4Cache,
        position: u32,
        vectors: &TestVectors,
    ) -> [LayerObservation; 3] {
        let raw0 = raw_values(position, 0);
        let raw1 = raw_values(position, 1);
        let raw2 = raw_values(position, 2);
        let mut transaction = cache.begin_token(position).unwrap();
        transaction
            .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw0 })
            .unwrap();
        let local = LayerObservation::capture(&transaction.layer_view(0).unwrap());
        transaction
            .stage_layer(
                1,
                DeepSeekV4LayerUpdate::CompressedSparse {
                    raw_kv: &raw1,
                    attention: vectors.csa_attention(),
                    indexer: vectors.csa_indexer(),
                },
            )
            .unwrap();
        let csa = LayerObservation::capture(&transaction.layer_view(1).unwrap());
        transaction
            .stage_layer(
                2,
                DeepSeekV4LayerUpdate::HeavilyCompressed {
                    raw_kv: &raw2,
                    attention: vectors.hca(),
                },
            )
            .unwrap();
        let hca = LayerObservation::capture(&transaction.layer_view(2).unwrap());
        transaction.commit().unwrap();
        [local, csa, hca]
    }

    fn push_single_csa(
        cache: &mut DeepSeekV4Cache,
        position: u32,
        raw: &[f32],
        vectors: &TestVectors,
    ) {
        let mut transaction = cache.begin_token(position).unwrap();
        transaction
            .stage_layer(
                0,
                DeepSeekV4LayerUpdate::CompressedSparse {
                    raw_kv: raw,
                    attention: vectors.csa_attention(),
                    indexer: vectors.csa_indexer(),
                },
            )
            .unwrap();
        transaction.commit().unwrap();
    }

    #[test]
    fn frozen_schedule_has_exact_lazy_capacity_plan() {
        let layout = DeepSeekV4CacheLayout {
            context_length: 1_048_576,
            local_window: 128,
            attention_dim: 512,
            rotary_dim: 64,
            indexer_dim: 128,
            attention_kinds: (0usize..43)
                .map(|layer| {
                    if layer < 2 {
                        AttentionKind::SlidingWindow
                    } else if layer.is_multiple_of(2) {
                        AttentionKind::CompressedSparse
                    } else {
                        AttentionKind::HeavilyCompressed
                    }
                })
                .collect(),
            compressor_rms_epsilon: 1e-6,
            compressor_rope: RopeParameters::yarn(64, 160_000.0, 16.0, 65_536, 32.0, 1.0),
        };
        layout.validate().unwrap();
        let cache = DeepSeekV4Cache::new(layout, test_identity()).unwrap();
        let counts =
            cache
                .layout()
                .attention_kinds()
                .iter()
                .fold((0, 0, 0), |(local, csa, hca), kind| match kind {
                    AttentionKind::SlidingWindow => (local + 1, csa, hca),
                    AttentionKind::CompressedSparse => (local, csa + 1, hca),
                    AttentionKind::HeavilyCompressed => (local, csa, hca + 1),
                });
        assert_eq!(counts, (2, 21, 20));
        assert_eq!(cache.layer_stats(2).unwrap().max_compressed_rows, 262_144);
        assert_eq!(cache.layer_stats(2).unwrap().max_indexer_rows, 262_144);
        assert_eq!(cache.layer_stats(3).unwrap().max_compressed_rows, 8_192);
        for layer in 0..43 {
            let stats = cache.layer_stats(layer).unwrap();
            assert_eq!(stats.raw_rows, 0);
            assert_eq!(stats.compressed_rows, 0);
            assert_eq!(stats.indexer_rows, 0);
        }
    }

    #[test]
    fn history_growth_uses_fixed_slabs_without_relocating_prior_rows() {
        let mut history = AttentionHistory::new(4, 2);
        for row in 0..HISTORY_CHUNK_ROWS {
            let chunk = history.prepare_append("test history").unwrap();
            history.append(
                AttentionCacheRow {
                    start_position: row as u32 * 4,
                    values: vec![row as f32, -(row as f32)],
                },
                chunk,
            );
        }
        assert_eq!(history.0.chunks.len(), 1);
        let first_slab = history.0.chunks[0].as_ptr();

        let chunk = history.prepare_append("test history").unwrap();
        assert!(chunk.is_some());
        history.append(
            AttentionCacheRow {
                start_position: HISTORY_CHUNK_ROWS as u32 * 4,
                values: vec![256.0, -256.0],
            },
            chunk,
        );
        assert_eq!(history.0.chunks.len(), 2);
        assert_eq!(history.0.chunks[0].as_ptr(), first_slab);
        assert_eq!(history.0.chunks[0].len(), HISTORY_CHUNK_ROWS * 2);
        assert_eq!(history.0.chunks[1].len(), 2);
        history
            .0
            .validate(4, 2, 1_028, 257, "test history")
            .unwrap();
        assert_eq!(history.0.row(255), [255.0, -255.0]);
        assert_eq!(history.0.row(256), [256.0, -256.0]);
    }

    #[test]
    fn multi_slab_cache_snapshot_restores_and_continues_exactly() {
        let vectors = TestVectors::new();
        let raw = raw_values(0, 0);
        let mut cache = DeepSeekV4Cache::new(
            test_layout(vec![AttentionKind::CompressedSparse], 4, 1_032),
            test_identity(),
        )
        .unwrap();
        for position in 0..1_028 {
            push_single_csa(&mut cache, position, &raw, &vectors);
        }
        assert_eq!(cache.layer_stats(0).unwrap().compressed_rows, 257);
        let mut restored = restore_copy(&cache);
        assert_eq!(restored, cache);

        for row in [255, 256] {
            let left = cache
                .layer_view(0)
                .unwrap()
                .compressed_rows()
                .nth(row)
                .unwrap();
            let right = restored
                .layer_view(0)
                .unwrap()
                .compressed_rows()
                .nth(row)
                .unwrap();
            assert_eq!(left.start_position(), row as u32 * 4);
            assert_eq!(left, right);
        }

        for position in 1_028..1_032 {
            push_single_csa(&mut cache, position, &raw, &vectors);
            push_single_csa(&mut restored, position, &raw, &vectors);
            assert_eq!(restored, cache);
        }
        assert_eq!(cache.layer_stats(0).unwrap().compressed_rows, 258);
    }

    #[test]
    fn flash_constructor_binds_profile_numerics_and_checkpoint_identity() {
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let cache = DeepSeekV4Cache::flash_0731(&config, test_identity()).unwrap();
        assert_eq!(cache.identity(), test_identity());
        assert_eq!(cache.layout().layer_count(), 43);
        assert_eq!(cache.layout().compressor_rms_epsilon(), 1e-6);
        assert_eq!(
            cache.layout().compressor_rope(),
            RopeParameters::yarn(64, 160_000.0, 16.0, 65_536, 32.0, 1.0)
        );

        let mut drift = config;
        drift.compress_rope_freq_base = 80_000.0;
        assert!(DeepSeekV4CacheLayout::flash_0731(&drift).is_err());
        let generic = DeepSeekV4CacheLayout::from_config(&drift).unwrap();
        assert_eq!(generic.compressor_rope().theta, 80_000.0);
    }

    #[test]
    fn flash_cache_executes_bound_yarn_across_two_csa_boundaries() {
        let config = crate::deepseek_v4::flash_0731_config_fixture();
        let mut cache = DeepSeekV4Cache::flash_0731(&config, test_identity()).unwrap();
        let schedule = cache.layout().attention_kinds().to_vec();
        let raw = values(512, 0.0003, -0.1);
        let attention_kv = values(1_024, 0.0002, -0.08);
        let attention_scores = values(1_024, 0.00003, -0.01);
        let attention_ape = values(4 * 1_024, 0.000002, -0.001);
        let attention_norm = vec![1.0; 512];
        let indexer_kv = values(256, -0.0004, 0.05);
        let indexer_scores = values(256, 0.00002, -0.002);
        let indexer_ape = values(4 * 256, -0.000001, 0.001);
        let indexer_norm = vec![1.0; 128];
        let hca_kv = values(512, 0.0001, -0.03);
        let hca_scores = values(512, -0.00001, 0.002);
        let hca_ape = values(128 * 512, 0.0000001, -0.0001);
        let mut expected = CompressorState::new(4, 512).unwrap();

        for position in 0..8 {
            let emitted = expected
                .push_projected(
                    position,
                    &attention_kv,
                    &attention_scores,
                    &attention_ape,
                    &attention_norm,
                    cache.layout().compressor_rms_epsilon(),
                    cache.layout().compressor_rope(),
                )
                .unwrap()
                .map(|mut row| {
                    attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut row.value, 64).unwrap();
                    row
                });
            let expected_rows = ((position + 1) / 4) as usize;
            let attention = DeepSeekV4CompressorUpdate {
                projected_kv: &attention_kv,
                projected_scores: &attention_scores,
                ape: &attention_ape,
                norm_weight: &attention_norm,
            };
            let indexer = DeepSeekV4CompressorUpdate {
                projected_kv: &indexer_kv,
                projected_scores: &indexer_scores,
                ape: &indexer_ape,
                norm_weight: &indexer_norm,
            };
            let hca = DeepSeekV4CompressorUpdate {
                projected_kv: &hca_kv,
                projected_scores: &hca_scores,
                ape: &hca_ape,
                norm_weight: &attention_norm,
            };
            let mut transaction = cache.begin_token(position).unwrap();
            for (layer, kind) in schedule.iter().copied().enumerate() {
                let update = match kind {
                    AttentionKind::SlidingWindow => {
                        DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw }
                    }
                    AttentionKind::CompressedSparse => DeepSeekV4LayerUpdate::CompressedSparse {
                        raw_kv: &raw,
                        attention,
                        indexer,
                    },
                    AttentionKind::HeavilyCompressed => DeepSeekV4LayerUpdate::HeavilyCompressed {
                        raw_kv: &raw,
                        attention: hca,
                    },
                };
                transaction.stage_layer(layer, update).unwrap();
                if layer == 2
                    && let Some(row) = &emitted
                {
                    let mut rows = transaction.layer_view(layer).unwrap().compressed_rows();
                    assert_eq!(rows.len(), expected_rows);
                    let actual = rows.nth(expected_rows - 1).unwrap();
                    assert_eq!(actual.start_position(), row.start_position);
                    assert_eq!(actual.values(), row.value);
                }
            }
            transaction.commit().unwrap();
            if let Some(row) = &emitted {
                let mut rows = cache.layer_view(2).unwrap().compressed_rows();
                assert_eq!(rows.len(), expected_rows);
                let actual = rows.nth(expected_rows - 1).unwrap();
                assert_eq!(actual.start_position(), row.start_position);
                assert_eq!(actual.values(), row.value);
            }
        }
    }

    #[test]
    fn transaction_view_has_same_token_boundaries_and_raw_wrap() {
        let vectors = TestVectors::new();
        let mut cache = three_layer_cache(4, 260);
        for position in 0..=256 {
            let [local, csa, hca] = push_three_layers(&mut cache, position, &vectors);
            let raw_start = (position + 1).saturating_sub(4);
            let expected_raw = (raw_start..=position).collect::<Vec<_>>();
            assert_eq!(local.raw_positions, expected_raw);
            assert_eq!(csa.raw_positions, expected_raw);
            assert_eq!(hca.raw_positions, expected_raw);
            assert_eq!(
                csa.compressed_positions,
                (0..((position + 1) / 4))
                    .map(|row| row * 4)
                    .collect::<Vec<_>>()
            );
            assert_eq!(csa.indexer_positions, csa.compressed_positions);
            assert_eq!(
                hca.compressed_positions,
                (0..((position + 1) / 128))
                    .map(|row| row * 128)
                    .collect::<Vec<_>>()
            );
            if position == 3 {
                assert_eq!(csa.compressed_positions, [0]);
            }
            if position == 127 {
                assert_eq!(hca.compressed_positions, [0]);
            }
            if matches!(position, 3 | 4 | 7 | 8 | 126 | 127 | 128 | 254 | 255 | 256) {
                let restored = restore_copy(&cache);
                assert_eq!(restored, cache);
            }
        }
        assert_eq!(cache.next_position(), 257);
        assert_eq!(cache.layer_stats(0).unwrap().raw_rows, 4);
        assert_eq!(cache.layer_stats(1).unwrap().compressed_rows, 64);
        assert_eq!(cache.layer_stats(1).unwrap().indexer_rows, 64);
        assert_eq!(cache.layer_stats(2).unwrap().compressed_rows, 2);
    }

    #[test]
    fn storage_format_matches_attention_and_indexer_roundtrips() {
        let vectors = TestVectors::new();
        let mut cache = DeepSeekV4Cache::new(
            test_layout(vec![AttentionKind::CompressedSparse], 4, 8),
            test_identity(),
        )
        .unwrap();
        let mut expected_attention = CompressorState::new(4, 128).unwrap();
        let mut expected_indexer = CompressorState::new(4, 128).unwrap();

        for position in 0..8 {
            let raw = raw_values(position, 0);
            let mut expected_raw = raw.clone();
            attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut expected_raw, 64).unwrap();
            let emitted_attention = expected_attention
                .push_projected(
                    position,
                    &vectors.csa_attention_kv,
                    &vectors.csa_attention_scores,
                    &vectors.csa_attention_ape,
                    &vectors.norm,
                    1e-6,
                    RopeParameters::local(64, 160_000.0),
                )
                .unwrap();
            let emitted_indexer = expected_indexer
                .push_projected(
                    position,
                    &vectors.csa_indexer_kv,
                    &vectors.csa_indexer_scores,
                    &vectors.csa_indexer_ape,
                    &vectors.norm,
                    1e-6,
                    RopeParameters::local(64, 160_000.0),
                )
                .unwrap();
            let expected_attention_row = emitted_attention.map(|mut row| {
                attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut row.value, 64).unwrap();
                row
            });
            let expected_indexer_row = emitted_indexer.map(|mut row| {
                indexer_qat_roundtrip_in_place(&mut row.value).unwrap();
                row
            });
            let expected_rows = ((position + 1) / 4) as usize;
            let mut transaction = cache.begin_token(position).unwrap();
            transaction
                .stage_layer(
                    0,
                    DeepSeekV4LayerUpdate::CompressedSparse {
                        raw_kv: &raw,
                        attention: vectors.csa_attention(),
                        indexer: vectors.csa_indexer(),
                    },
                )
                .unwrap();
            let view = transaction.layer_view(0).unwrap();
            assert_eq!(view.raw_rows().last().unwrap().values(), expected_raw);
            match &expected_attention_row {
                Some(row) => {
                    let mut rows = view.compressed_rows();
                    assert_eq!(rows.len(), expected_rows);
                    let actual = rows.nth(expected_rows - 1).unwrap();
                    assert_eq!(actual.start_position(), row.start_position);
                    assert_eq!(actual.values(), row.value);
                }
                None => assert_eq!(view.compressed_rows().len(), expected_rows),
            }
            match &expected_indexer_row {
                Some(row) => {
                    let mut rows = view.indexer_rows();
                    assert_eq!(rows.len(), expected_rows);
                    let actual = rows.nth(expected_rows - 1).unwrap();
                    assert_eq!(actual.start_position(), row.start_position);
                    assert_eq!(actual.values(), row.value);
                }
                None => assert_eq!(view.indexer_rows().len(), expected_rows),
            }
            drop(view);
            transaction.commit().unwrap();
            let committed = cache.layer_view(0).unwrap();
            if let Some(row) = &expected_attention_row {
                let actual = committed.compressed_rows().nth(expected_rows - 1).unwrap();
                assert_eq!(actual.start_position(), row.start_position);
                assert_eq!(actual.values(), row.value);
            }
            if let Some(row) = &expected_indexer_row {
                let actual = committed.indexer_rows().nth(expected_rows - 1).unwrap();
                assert_eq!(actual.start_position(), row.start_position);
                assert_eq!(actual.values(), row.value);
            }
            if position == 3 {
                cache = restore_copy(&cache);
            }
        }
        assert_eq!(
            cache.storage_format(),
            DeepSeekV4CacheStorageFormat::F32QuantizationOracleV1
        );
        assert_eq!(
            cache.snapshot().unwrap().storage_format(),
            DeepSeekV4CacheStorageFormat::F32QuantizationOracleV1
        );

        let mut hca_cache = DeepSeekV4Cache::new(
            test_layout(vec![AttentionKind::HeavilyCompressed], 4, 256),
            test_identity(),
        )
        .unwrap();
        let mut expected_hca = CompressorState::new(128, 128).unwrap();
        for position in 0..256 {
            let raw = raw_values(position, 0);
            let emitted = expected_hca
                .push_projected(
                    position,
                    &vectors.hca_kv,
                    &vectors.hca_scores,
                    &vectors.hca_ape,
                    &vectors.norm,
                    1e-6,
                    RopeParameters::local(64, 160_000.0),
                )
                .unwrap();
            let expected_row = emitted.map(|mut row| {
                attention_fp8_nope_bf16_rope_roundtrip_in_place(&mut row.value, 64).unwrap();
                row
            });
            let expected_rows = ((position + 1) / 128) as usize;
            let mut transaction = hca_cache.begin_token(position).unwrap();
            transaction
                .stage_layer(
                    0,
                    DeepSeekV4LayerUpdate::HeavilyCompressed {
                        raw_kv: &raw,
                        attention: vectors.hca(),
                    },
                )
                .unwrap();
            let view = transaction.layer_view(0).unwrap();
            match &expected_row {
                Some(row) => {
                    let mut rows = view.compressed_rows();
                    assert_eq!(rows.len(), expected_rows);
                    let actual = rows.nth(expected_rows - 1).unwrap();
                    assert_eq!(actual.start_position(), row.start_position);
                    assert_eq!(actual.values(), row.value);
                }
                None => assert_eq!(view.compressed_rows().len(), expected_rows),
            }
            drop(view);
            transaction.commit().unwrap();
            if let Some(row) = &expected_row {
                let committed = hca_cache.layer_view(0).unwrap();
                let actual = committed.compressed_rows().nth(expected_rows - 1).unwrap();
                assert_eq!(actual.start_position(), row.start_position);
                assert_eq!(actual.values(), row.value);
            }
            if position == 127 {
                hca_cache = restore_copy(&hca_cache);
            }
        }
        assert_eq!(hca_cache.next_position(), 256);
        let mut restored_hca = restore_copy(&hca_cache);
        assert_eq!(restored_hca, hca_cache);
        assert_eq!(
            restored_hca.layer_view(0).unwrap().compressed_rows().len(),
            2
        );
        assert!(restored_hca.begin_token(256).is_err());
    }

    #[test]
    fn failed_and_incomplete_transactions_leave_semantic_state_unchanged() {
        let vectors = TestVectors::new();
        let mut cache = three_layer_cache(4, 16);
        push_three_layers(&mut cache, 0, &vectors);
        let baseline = cache.snapshot().unwrap();

        {
            let raw = raw_values(1, 0);
            let raw_csa = raw_values(1, 1);
            let mut bad_attention = vectors.csa_attention();
            bad_attention.projected_kv = &[];
            let mut transaction = cache.begin_token(1).unwrap();
            transaction
                .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw })
                .unwrap();
            assert!(
                transaction
                    .stage_layer(
                        1,
                        DeepSeekV4LayerUpdate::CompressedSparse {
                            raw_kv: &raw_csa,
                            attention: bad_attention,
                            indexer: vectors.csa_indexer(),
                        },
                    )
                    .is_err()
            );
        }
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        {
            let raw = raw_values(1, 0);
            let mut transaction = cache.begin_token(1).unwrap();
            assert!(
                transaction
                    .stage_layer(1, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw },)
                    .is_err()
            );
            transaction
                .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw })
                .unwrap();
            assert!(transaction.commit().is_err());
        }
        assert_eq!(cache.snapshot().unwrap(), baseline);
        push_three_layers(&mut cache, 1, &vectors);
        assert_eq!(cache.next_position(), 2);
    }

    #[test]
    fn late_failures_roll_back_boundary_emissions_and_ring_overwrites() {
        let vectors = TestVectors::new();
        let mut cache = three_layer_cache(4, 132);
        for position in 0..3 {
            push_three_layers(&mut cache, position, &vectors);
        }

        let baseline = cache.snapshot().unwrap();
        {
            let raw0 = raw_values(3, 0);
            let raw1 = raw_values(3, 1);
            let mut invalid_indexer = vectors.csa_indexer();
            let nonfinite = vec![f32::NAN; 256];
            invalid_indexer.projected_scores = &nonfinite;
            let mut transaction = cache.begin_token(3).unwrap();
            transaction
                .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw0 })
                .unwrap();
            assert!(
                transaction
                    .stage_layer(
                        1,
                        DeepSeekV4LayerUpdate::CompressedSparse {
                            raw_kv: &raw1,
                            attention: vectors.csa_attention(),
                            indexer: invalid_indexer,
                        },
                    )
                    .is_err()
            );
        }
        assert_eq!(cache.snapshot().unwrap(), baseline);
        let observations = push_three_layers(&mut cache, 3, &vectors);
        assert_eq!(observations[1].compressed_positions, [0]);
        assert_eq!(observations[1].indexer_positions, [0]);

        let baseline = cache.snapshot().unwrap();
        {
            let raw0 = raw_values(4, 0);
            let raw1 = raw_values(4, 1);
            let raw2 = raw_values(4, 2);
            let mut invalid_hca = vectors.hca();
            invalid_hca.projected_kv = &[];
            let mut transaction = cache.begin_token(4).unwrap();
            transaction
                .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw0 })
                .unwrap();
            transaction
                .stage_layer(
                    1,
                    DeepSeekV4LayerUpdate::CompressedSparse {
                        raw_kv: &raw1,
                        attention: vectors.csa_attention(),
                        indexer: vectors.csa_indexer(),
                    },
                )
                .unwrap();
            assert!(
                transaction
                    .stage_layer(
                        2,
                        DeepSeekV4LayerUpdate::HeavilyCompressed {
                            raw_kv: &raw2,
                            attention: invalid_hca,
                        },
                    )
                    .is_err()
            );
        }
        assert_eq!(cache.snapshot().unwrap(), baseline);
        assert_eq!(
            LayerObservation::capture(&cache.layer_view(0).unwrap()).raw_positions,
            [0, 1, 2, 3]
        );
        let observations = push_three_layers(&mut cache, 4, &vectors);
        assert_eq!(observations[0].raw_positions, [1, 2, 3, 4]);

        for position in 5..127 {
            push_three_layers(&mut cache, position, &vectors);
        }
        let baseline = cache.snapshot().unwrap();
        {
            let raw0 = raw_values(127, 0);
            let raw1 = raw_values(127, 1);
            let raw2 = raw_values(127, 2);
            let mut invalid_hca = vectors.hca();
            let nonfinite = vec![f32::INFINITY; 128];
            invalid_hca.projected_scores = &nonfinite;
            let mut transaction = cache.begin_token(127).unwrap();
            transaction
                .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw0 })
                .unwrap();
            transaction
                .stage_layer(
                    1,
                    DeepSeekV4LayerUpdate::CompressedSparse {
                        raw_kv: &raw1,
                        attention: vectors.csa_attention(),
                        indexer: vectors.csa_indexer(),
                    },
                )
                .unwrap();
            assert!(
                transaction
                    .stage_layer(
                        2,
                        DeepSeekV4LayerUpdate::HeavilyCompressed {
                            raw_kv: &raw2,
                            attention: invalid_hca,
                        },
                    )
                    .is_err()
            );
        }
        assert_eq!(cache.snapshot().unwrap(), baseline);
        let observations = push_three_layers(&mut cache, 127, &vectors);
        assert_eq!(observations[2].compressed_positions, [0]);
    }

    #[test]
    fn cache_rejects_nonfinite_rows_and_invalid_bound_profile_parameters() {
        let mut cache = three_layer_cache(4, 8);
        let baseline = cache.snapshot().unwrap();
        let mut raw = raw_values(0, 0);
        raw[17] = f32::NAN;
        let mut transaction = cache.begin_token(0).unwrap();
        assert!(
            transaction
                .stage_layer(0, DeepSeekV4LayerUpdate::SlidingWindow { raw_kv: &raw })
                .is_err()
        );
        drop(transaction);
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let mut bad_epsilon = test_layout(vec![AttentionKind::SlidingWindow], 4, 8);
        bad_epsilon.compressor_rms_epsilon = 0.0;
        assert!(DeepSeekV4Cache::new(bad_epsilon, test_identity()).is_err());

        let mut bad_rope = test_layout(vec![AttentionKind::SlidingWindow], 4, 8);
        bad_rope.compressor_rope.theta = f32::NAN;
        assert!(DeepSeekV4Cache::new(bad_rope, test_identity()).is_err());

        let mut oversized_indexer_rope = test_layout(vec![AttentionKind::CompressedSparse], 4, 8);
        oversized_indexer_rope.attention_dim = 256;
        oversized_indexer_rope.rotary_dim = 192;
        oversized_indexer_rope.compressor_rope = RopeParameters::local(192, 160_000.0);
        assert!(DeepSeekV4Cache::new(oversized_indexer_rope, test_identity()).is_err());
    }

    #[test]
    fn snapshot_roundtrip_restores_ring_and_compressor_phases() {
        let vectors = TestVectors::new();
        let mut cache = three_layer_cache(4, 260);
        let empty = restore_copy(&cache);
        assert_eq!(empty, cache);
        for position in 0..=128 {
            push_three_layers(&mut cache, position, &vectors);
        }
        let snapshot = cache.snapshot().unwrap();
        assert_eq!(snapshot.next_position(), 129);
        let mut restored = DeepSeekV4Cache::from_snapshot(
            snapshot,
            cache.identity(),
            cache.layout(),
            cache.storage_format(),
        )
        .unwrap();
        assert_eq!(restored, cache);

        for position in 129..=133 {
            let left = push_three_layers(&mut cache, position, &vectors);
            let right = push_three_layers(&mut restored, position, &vectors);
            assert_eq!(left, right);
            assert_eq!(restored, cache);
        }
    }

    #[test]
    fn snapshot_corruption_is_rejected_before_destination_mutation() {
        let vectors = TestVectors::new();
        let mut cache = three_layer_cache(4, 16);
        for position in 0..4 {
            push_three_layers(&mut cache, position, &vectors);
        }

        assert!(
            DeepSeekV4Cache::from_snapshot(
                cache.snapshot().unwrap(),
                DeepSeekV4CacheIdentity::new([0xa5; 32]),
                cache.layout(),
                cache.storage_format(),
            )
            .is_err()
        );

        let baseline = cache.snapshot().unwrap();
        let mut wrong_identity = cache.snapshot().unwrap();
        wrong_identity.identity = DeepSeekV4CacheIdentity::new([0xa5; 32]);
        assert!(cache.restore(wrong_identity).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut wrong_position = cache.snapshot().unwrap();
        wrong_position.next_position += 1;
        assert!(cache.restore(wrong_position).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut wrong_raw_slot = cache.snapshot().unwrap();
        let LayerSnapshot::SlidingWindow { raw } = &mut wrong_raw_slot.layers[0] else {
            panic!("expected local snapshot");
        };
        raw.slots[0].as_mut().unwrap().start_position = 1;
        assert!(cache.restore(wrong_raw_slot).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut misaligned = cache.snapshot().unwrap();
        let LayerSnapshot::CompressedSparse { indexer_rows, .. } = &mut misaligned.layers[1] else {
            panic!("expected CSA snapshot");
        };
        indexer_rows.0.row_count = 0;
        indexer_rows.0.chunks.clear();
        assert!(cache.restore(misaligned).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut nonfinite_history = cache.snapshot().unwrap();
        let LayerSnapshot::CompressedSparse { attention_rows, .. } =
            &mut nonfinite_history.layers[1]
        else {
            panic!("expected CSA snapshot");
        };
        attention_rows.0.chunks[0][0] = f32::NAN;
        assert!(cache.restore(nonfinite_history).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut nonfinite_score = cache.snapshot().unwrap();
        let LayerSnapshot::CompressedSparse {
            attention_state, ..
        } = &mut nonfinite_score.layers[1]
        else {
            panic!("expected CSA snapshot");
        };
        attention_state.scores[0] = f32::NAN;
        assert!(cache.restore(nonfinite_score).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut bad_phase = cache.snapshot().unwrap();
        let LayerSnapshot::CompressedSparse {
            attention_state, ..
        } = &mut bad_phase.layers[1]
        else {
            panic!("expected CSA snapshot");
        };
        attention_state.kv[0] += 1.0;
        assert!(cache.restore(bad_phase).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);

        let baseline = cache.snapshot().unwrap();
        let mut wrong_layout = cache.snapshot().unwrap();
        wrong_layout.layout.local_window += 1;
        assert!(cache.restore(wrong_layout).is_err());
        assert_eq!(cache.snapshot().unwrap(), baseline);
    }

    #[test]
    fn cache_rejects_position_gaps_duplicates_and_context_overrun() {
        let vectors = TestVectors::new();
        let mut cache = three_layer_cache(4, 4);
        assert!(cache.begin_token(1).is_err());
        push_three_layers(&mut cache, 0, &vectors);
        assert!(cache.begin_token(0).is_err());
        assert!(cache.begin_token(2).is_err());
        for position in 1..4 {
            push_three_layers(&mut cache, position, &vectors);
        }
        assert_eq!(cache.next_position(), 4);
        assert!(cache.begin_token(4).is_err());
        let mut restored = restore_copy(&cache);
        assert_eq!(restored, cache);
        assert!(restored.begin_token(4).is_err());
    }
}
