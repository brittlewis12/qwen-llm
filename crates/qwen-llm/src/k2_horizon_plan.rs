//! Host-only plans for bounded K2 cache storage and causal visibility.
//!
//! These plans allocate no buffers and admit no runtime. The materialized-score
//! ceiling is a source constraint of the candidate attention kernel, NOT a
//! numerically qualified K2 capacity. An eventual encoder must additionally check
//! physical buffer ranges, dtype, writability, aliasing, and actual cache state.

use crate::k2_horizon::{K2HorizonConfig, K2HorizonError, K2KvStorage};
use std::ops::Range;

/// `metal::encode_attn_decode_f16kv_f32` limits score scratch to 28 KiB.
pub const MATERIALIZED_POSITION_CEILING: u32 = 7168;
pub(crate) const PACKED_CHUNK_TOKENS: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum PlanError {
    #[error(transparent)]
    Profile(#[from] K2HorizonError),
    #[error("invalid K2 short-context plan: {0}")]
    Invalid(&'static str),
}

type Result<T> = std::result::Result<T, PlanError>;

/// Canonical arena order: [layer, K-or-V, position, KV head, channel].
/// Physical byte ranges are storage-specific; logical positions are not bytes.
#[derive(Clone, Debug)]
pub struct K2ShortContextPlan {
    config: K2HorizonConfig,
    capacity: u32,
    start_position: u32,
    row_bytes: u64,
    plane_bytes: u64,
    layer_bytes: u64,
    arena_bytes: u64,
    storage: K2KvStorage,
}

impl K2ShortContextPlan {
    pub fn new(config: K2HorizonConfig, start_position: u32, capacity: u32) -> Result<Self> {
        Self::with_storage(config, start_position, capacity, K2KvStorage::F16)
    }

    pub(crate) fn with_storage(
        config: K2HorizonConfig,
        start_position: u32,
        capacity: u32,
        storage: K2KvStorage,
    ) -> Result<Self> {
        config.validate_7b()?;
        if capacity == 0 || capacity > MATERIALIZED_POSITION_CEILING {
            return Err(PlanError::Invalid(
                "capacity exceeds candidate materialized-score limit",
            ));
        }
        if start_position
            .checked_add(capacity)
            .is_none_or(|end| end > config.context_length)
        {
            return Err(PlanError::Invalid(
                "absolute request extent exceeds checkpoint context",
            ));
        }
        // The candidate paired-RoPE wrapper is stricter than structural binding.
        if config.rope_theta <= 1.0 {
            return Err(PlanError::Invalid(
                "candidate paired RoPE requires theta > 1",
            ));
        }
        let row_bytes =
            storage.row_bytes(u64::from(config.kv_head_count) * u64::from(config.key_head_dim))?;
        let plane_bytes = row_bytes * u64::from(capacity);
        let layer_bytes = plane_bytes * 2;
        let arena_bytes = config.kv_storage_bytes(u64::from(capacity), storage)?;
        Ok(Self {
            config,
            capacity,
            start_position,
            row_bytes,
            plane_bytes,
            layer_bytes,
            arena_bytes,
            storage,
        })
    }

    pub fn declared_context(&self) -> u32 {
        self.config.context_length
    }
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
    pub fn start_position(&self) -> u32 {
        self.start_position
    }
    pub fn arena_bytes(&self) -> u64 {
        self.arena_bytes
    }
    pub fn row_bytes(&self) -> u64 {
        self.row_bytes
    }

    pub(crate) fn storage(&self) -> K2KvStorage {
        self.storage
    }

    /// Logical K/V only, with no padding, weights, scratch, or allocator reserve.
    pub fn layer_planes(&self, layer: u32) -> Result<KvRanges> {
        if layer >= self.config.layer_count {
            return Err(PlanError::Invalid("layer outside profile"));
        }
        let start = u64::from(layer) * self.layer_bytes;
        Ok(KvRanges {
            key: start..start + self.plane_bytes,
            value: start + self.plane_bytes..start + self.layer_bytes,
        })
    }

    /// A nonzero base is an isolated positioned request, never silent eviction
    /// of earlier cached positions. The caller supplies an actually committed
    /// prefix; this pure plan neither proves residency nor changes that prefix.
    pub fn append(
        &self,
        committed: u32,
        absolute_position: u32,
        tokens: u32,
    ) -> Result<AppendPlan<'_>> {
        let end = committed
            .checked_add(tokens)
            .ok_or(PlanError::Invalid("prefix plus append overflow"))?;
        if tokens == 0 || committed > self.capacity || end > self.capacity {
            return Err(PlanError::Invalid(
                "empty append or request capacity exceeded",
            ));
        }
        if absolute_position != self.start_position + committed {
            return Err(PlanError::Invalid("noncontiguous absolute append position"));
        }
        Ok(AppendPlan {
            request: self,
            committed,
            tokens,
        })
    }

    /// Four independent direct-gamma RMS calls, with corresponding slices of
    /// x/y/gamma. This deliberately avoids shared-gamma per-head norm kernels.
    pub fn norm_groups(&self) -> [NormGroup; 4] {
        let width = u64::from(self.config.hidden_size / self.config.norm_groups);
        std::array::from_fn(|group| {
            let start = group as u64 * width;
            NormGroup {
                elements: start..start + width,
                epsilon: self.config.rms_epsilon,
            }
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvRanges {
    pub key: Range<u64>,
    pub value: Range<u64>,
}

#[derive(Debug)]
pub struct NormGroup {
    /// Element offsets into each full hidden-width F32 x, y, and gamma view.
    pub elements: Range<u64>,
    pub epsilon: f32,
}

pub struct AppendPlan<'a> {
    request: &'a K2ShortContextPlan,
    committed: u32,
    tokens: u32,
}

impl AppendPlan<'_> {
    pub fn final_prefix(&self) -> u32 {
        self.committed + self.tokens
    }

    /// Serial causal schedule; this does not claim packed attention support.
    pub fn token(&self, index: u32) -> Result<TokenPlan<'_>> {
        if index >= self.tokens {
            return Err(PlanError::Invalid("token outside append"));
        }
        Ok(TokenPlan {
            request: self.request,
            cache_index: self.committed + index,
        })
    }
}

pub struct TokenPlan<'a> {
    request: &'a K2ShortContextPlan,
    cache_index: u32,
}

impl TokenPlan<'_> {
    pub(crate) fn storage(&self) -> K2KvStorage {
        self.request.storage()
    }

    pub fn arena_bytes(&self) -> u64 {
        self.request.arena_bytes()
    }

    pub fn absolute_position(&self) -> u32 {
        self.request.start_position + self.cache_index
    }
    pub fn visible_positions(&self) -> u32 {
        self.cache_index + 1
    }

    pub fn rope(&self) -> FullNeoxRope {
        FullNeoxRope {
            position: self.absolute_position(),
            theta: self.request.config.rope_theta,
        }
    }

    /// The row to store after RoPE (K) / projection (V), before attention reads.
    pub fn write_ranges(&self, layer: u32) -> Result<KvRanges> {
        let planes = self.request.layer_planes(layer)?;
        let row = u64::from(self.cache_index) * self.request.row_bytes;
        let range =
            |plane: Range<u64>| plane.start + row..plane.start + row + self.request.row_bytes;
        Ok(KvRanges {
            key: range(planes.key),
            value: range(planes.value),
        })
    }

    /// Includes the current stored row but excludes future rows even if an
    /// append has reserved or eventually written them. No rolling/truncation.
    pub fn read_ranges(&self, layer: u32) -> Result<KvRanges> {
        let planes = self.request.layer_planes(layer)?;
        let bytes = u64::from(self.visible_positions()) * self.request.row_bytes;
        Ok(KvRanges {
            key: planes.key.start..planes.key.start + bytes,
            value: planes.value.start..planes.value.start + bytes,
        })
    }

    /// Dynamic threadgroup scores, rounded to Metal's 16-byte requirement.
    /// The candidate's separate reduction allocation is at most 128 bytes
    /// (1024 threads / 32 lanes * sizeof(float)). Neither is retained KV.
    pub fn score_scratch_bytes(&self) -> u32 {
        (self.visible_positions() * 4).next_multiple_of(16)
    }
}

/// The only RoPE form emitted by this bridge: no partial/interleaved option.
pub struct FullNeoxRope {
    position: u32,
    theta: f32,
}

impl FullNeoxRope {
    pub const QUERY_HEADS: usize = 32;
    pub const KV_HEADS: usize = 8;
    pub const HEAD_DIM: usize = 128;
    pub const ROTARY_DIM: usize = 128;
    pub fn position(&self) -> u32 {
        self.position
    }
    pub fn theta(&self) -> f32 {
        self.theta
    }
}

#[cfg(test)]
mod tests;
