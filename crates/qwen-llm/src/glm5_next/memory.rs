//! Allocation ledger for one GLM-5.3 session.
//!
//! The trunk leaves about 2.5 GiB of the 112 GiB Metal working set, so the
//! ledger prices exactly what the session allocates: the native session builds
//! every buffer from the [`BufferSpec`] lists below, and the ledger sums the
//! same lists (rounded to the Metal allocation granule) before allocation.
//! Retained weights come from the real retained-window plan.
//!
//! Layout: F32 recurrent/conv state; an append-only F16 cache of MLA latents
//! and completed pooled indexer keys plus a 4-slot pending ring; decode scratch
//! and optional packed-prefill scratch, both resident for the session's life;
//! per-MoE-block route records for each; last-position logits only. Sparse
//! selection buffers are added here when P4 allocates them.

use super::{FfnKind, Glm5NextConfig, Glm5NextError, MixerKind, Result};

/// Command buffers, argument tables and allocator slack.
pub const DYNAMIC_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
/// Metal shared-buffer allocation granule priced per buffer.
pub const ALLOCATION_GRANULE: u64 = 16 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BufferType {
    F32,
    F16,
    I32,
}

impl BufferType {
    pub fn bytes(self) -> u64 {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 => 2,
        }
    }
}

/// One session buffer: name, element type, shape and initial fill.
#[derive(Clone, Debug, PartialEq)]
pub struct BufferSpec {
    pub name: &'static str,
    pub dtype: BufferType,
    pub shape: Vec<u64>,
    pub fill: f32,
}

impl BufferSpec {
    fn zeros(name: &'static str, dtype: BufferType, shape: &[u64]) -> Self {
        Self {
            name,
            dtype,
            shape: shape.to_vec(),
            fill: 0.0,
        }
    }

    fn filled(name: &'static str, shape: &[u64], fill: f32) -> Self {
        Self {
            name,
            dtype: BufferType::F32,
            shape: shape.to_vec(),
            fill,
        }
    }

    pub fn bytes(&self) -> u64 {
        self.shape.iter().product::<u64>() * self.dtype.bytes()
    }

    /// Bytes charged against the working set (rounded to the granule).
    pub fn priced_bytes(&self) -> u64 {
        self.bytes().max(1).div_ceil(ALLOCATION_GRANULE) * ALLOCATION_GRANULE
    }
}

fn priced(specs: &[BufferSpec]) -> u64 {
    specs.iter().map(BufferSpec::priced_bytes).sum()
}

use BufferType::{F16, F32, I32};

/// Decode scratch shared by every block, including the logits and constants.
pub fn decode_scratch_specs(c: &Glm5NextConfig) -> Vec<BufferSpec> {
    let h = c.hidden_size as u64;
    let w = c.kda_width() as u64;
    let heads = c.head_count as u64;
    let d = c.kda_head_dim as u64;
    let kv = c.kv_lora_rank as u64;
    let k = c.expert_used_count as u64;
    let z = BufferSpec::zeros;
    vec![
        z("token", I32, &[1]),
        z("embedding", F32, &[h]),
        z("residual_a", F32, &[h, 4]),
        z("residual_b", F32, &[h, 4]),
        BufferSpec::filled("ones", &[c.hc_width() as u64], 1.0),
        BufferSpec::filled("quarter", &[4], 0.25),
        BufferSpec::filled("no_sink", &[heads], crate::metal::LATENT_NO_SINK),
        z("normalized", F32, &[c.hc_width() as u64]),
        z("mixes", F32, &[c.hc_mix_count() as u64]),
        z("pre", F32, &[4]),
        z("post", F32, &[4]),
        z("comb", F32, &[4, 4]),
        z("collapsed", F32, &[h]),
        z("normed", F32, &[h]),
        z("block_out", F32, &[h]),
        z("q", F32, &[w]),
        z("k", F32, &[w]),
        z("v", F32, &[w]),
        z("rank_a", F32, &[d]),
        z("raw_gate", F32, &[w]),
        z("raw_beta", F32, &[heads]),
        z("rank_b", F32, &[d]),
        z("output_gate", F32, &[w]),
        z("kda_out", F32, &[w]),
        z("query_a", F32, &[c.q_lora_rank as u64]),
        z("query_r", F32, &[c.q_lora_rank as u64]),
        z("query", F32, &[c.mla_width() as u64]),
        z("latent_raw", F32, &[kv]),
        z("latent", F32, &[kv]),
        z("query_latent", F32, &[kv, heads, 1]),
        z("output_latent", F32, &[kv, heads, 1]),
        z("heads_out", F32, &[c.mla_width() as u64]),
        z("index_key", F32, &[c.indexer_head_dim as u64]),
        z("index_gate", F32, &[c.indexer_head_dim as u64]),
        z("dense_gate", F32, &[c.dense_ffn_size as u64]),
        z("dense_up", F32, &[c.dense_ffn_size as u64]),
        z("router", F32, &[c.expert_count as u64]),
        z("expert_inner", F32, &[c.expert_ffn_size as u64, k]),
        z("expert_out", F32, &[h, k]),
        z("routed", F32, &[h]),
        z("shared_gate", F32, &[c.shared_expert_ffn_size as u64]),
        z("shared_up", F32, &[c.shared_expert_ffn_size as u64]),
        z("shared", F32, &[h]),
        z("final_hidden", F32, &[h]),
        z("final_normed", F32, &[h]),
        z("logits", F32, &[c.vocab_size as u64]),
    ]
}

/// Packed-prefill scratch for chunks of up to `rows` tokens.
pub fn packed_scratch_specs(c: &Glm5NextConfig, rows: u64) -> Vec<BufferSpec> {
    let r = rows;
    let h = c.hidden_size as u64;
    let w = c.kda_width() as u64;
    let heads = c.head_count as u64;
    let d = c.kda_head_dim as u64;
    let kv = c.kv_lora_rank as u64;
    let k = c.expert_used_count as u64;
    let e = c.expert_count as u64;
    let z = BufferSpec::zeros;
    vec![
        z("token", I32, &[r]),
        z("embedding", F32, &[h, r]),
        z("residual_a", F32, &[h, 4, r]),
        z("residual_b", F32, &[h, 4, r]),
        z("normalized", F32, &[c.hc_width() as u64, r]),
        z("mixes", F32, &[c.hc_mix_count() as u64, r]),
        z("pre", F32, &[4, r]),
        z("post", F32, &[4, r]),
        z("comb", F32, &[4, 4, r]),
        z("collapsed", F32, &[h, r]),
        z("normed", F32, &[h, r]),
        z("block_out", F32, &[h, r]),
        z("q", F32, &[w, r]),
        z("k", F32, &[w, r]),
        z("v", F32, &[w, r]),
        z("rank_a", F32, &[d, r]),
        z("raw_gate", F32, &[w, r]),
        z("raw_beta", F32, &[heads, r]),
        z("rank_b", F32, &[d, r]),
        z("output_gate", F32, &[w, r]),
        z("kda_out", F32, &[w, r]),
        z("query_a", F32, &[c.q_lora_rank as u64, r]),
        z("query_r", F32, &[c.q_lora_rank as u64, r]),
        z("query", F32, &[c.mla_width() as u64, r]),
        z("latent_raw", F32, &[kv, r]),
        z("latent", F32, &[kv, r]),
        z("query_latent", F32, &[kv, heads, r]),
        z("output_latent", F32, &[kv, heads, r]),
        z("heads_out", F32, &[c.mla_width() as u64, r]),
        z("index_key", F32, &[c.indexer_head_dim as u64, r]),
        z("index_gate", F32, &[c.indexer_head_dim as u64, r]),
        z("dense_gate", F32, &[c.dense_ffn_size as u64, r]),
        z("dense_up", F32, &[c.dense_ffn_size as u64, r]),
        z("router", F32, &[e, r]),
        z("counts", I32, &[e]),
        z("slots", I32, &[e * r]),
        z("inner", F32, &[c.expert_ffn_size as u64, k * r]),
        z("slot_out", F32, &[h, k * r]),
        z("routed", F32, &[h, r]),
        z("shared_gate", F32, &[c.shared_expert_ffn_size as u64, r]),
        z("shared_up", F32, &[c.shared_expert_ffn_size as u64, r]),
        z("shared", F32, &[h, r]),
    ]
}

/// One MoE block's route record for `rows` tokens (1 for decode): ids and
/// weights `[top_k, rows]` (decode: `[top_k]`), status `[rows]`.
pub fn route_specs(c: &Glm5NextConfig, rows: Option<u64>) -> Vec<BufferSpec> {
    let k = c.expert_used_count as u64;
    let (ids, status) = match rows {
        None => (vec![k], vec![1]),
        Some(r) => (vec![k, r], vec![r]),
    };
    vec![
        BufferSpec::zeros("ids", I32, &ids),
        BufferSpec::zeros("weights", F32, &ids),
        BufferSpec::zeros("status", I32, &status),
    ]
}

/// KDA block state: conv tails `[width, 3, 3]` and S `[128, 128, heads]`.
pub fn kda_state_specs(c: &Glm5NextConfig) -> Vec<BufferSpec> {
    let w = c.kda_width() as u64;
    let d = c.kda_head_dim as u64;
    vec![
        BufferSpec::zeros("conv", F32, &[w, 3, 3]),
        BufferSpec::zeros("state", F32, &[d, d, c.head_count as u64]),
    ]
}

/// MLA block state for `capacity` positions: F16 latent rows, pending
/// key|gate ring and completed pooled keys.
pub fn mla_state_specs(c: &Glm5NextConfig, capacity: u64) -> Vec<BufferSpec> {
    let pools = capacity.div_ceil(c.indexer_pool as u64).max(1);
    let id = c.indexer_head_dim as u64;
    vec![
        BufferSpec::zeros("latent", F16, &[c.kv_lora_rank as u64, capacity]),
        BufferSpec::zeros("pending", F16, &[id, 2, c.indexer_pool as u64]),
        BufferSpec::zeros("pooled", F16, &[id, pools]),
    ]
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm5NextMemoryLedger {
    capacity: u64,
    prefill_rows: u64,
    retained_weight_bytes: u64,
    kda_state: u64,
    mla_state: u64,
    decode_scratch: u64,
    decode_routes: u64,
    packed_scratch: u64,
    packed_routes: u64,
    reserve: u64,
    session_state: u64,
    peaks: Glm5NextPhasePeaks,
}

/// Cumulative Metal bytes: weights alone, and weights plus the whole session
/// (every session buffer lives as long as the session) and the reserve.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm5NextPhasePeaks {
    pub resident: u64,
    pub session: u64,
}

impl Glm5NextMemoryLedger {
    /// `prefill_rows == 0` prices a decode-only session.
    pub fn new(
        config: &Glm5NextConfig,
        retained_weight_bytes: u64,
        capacity: u64,
        prefill_rows: u64,
    ) -> Result<Self> {
        config.validate_release()?;
        if capacity == 0 || capacity > u64::from(config.context_length) {
            return Err(Glm5NextError::InvalidMetadata {
                key: "capacity".into(),
                detail: format!(
                    "{capacity} is outside the checkpoint context 1..={}",
                    config.context_length
                ),
            });
        }
        let c = config;
        let overflow = || Glm5NextError::Overflow("memory ledger");
        let kda = c.block_count(MixerKind::Kda) as u64;
        let mla = c.block_count(MixerKind::Mla) as u64;
        let moe = c.blocks.iter().filter(|b| b.ffn == FfnKind::Moe).count() as u64;
        let times = |n: u64, bytes: u64| n.checked_mul(bytes).ok_or_else(overflow);
        let kda_state = times(kda, priced(&kda_state_specs(c)))?;
        let mla_state = times(mla, priced(&mla_state_specs(c, capacity)))?;
        let decode_scratch = priced(&decode_scratch_specs(c));
        let decode_routes = times(moe, priced(&route_specs(c, None)))?;
        let (packed_scratch, packed_routes) = if prefill_rows == 0 {
            (0, 0)
        } else {
            (
                priced(&packed_scratch_specs(c, prefill_rows)),
                times(moe, priced(&route_specs(c, Some(prefill_rows))))?,
            )
        };
        let sum = |terms: &[u64]| -> Result<u64> {
            terms
                .iter()
                .try_fold(0u64, |acc, &t| acc.checked_add(t))
                .ok_or_else(overflow)
        };
        let session_state = sum(&[kda_state, mla_state])?;
        let session = sum(&[
            retained_weight_bytes,
            session_state,
            decode_scratch,
            decode_routes,
            packed_scratch,
            packed_routes,
            DYNAMIC_RESERVE_BYTES,
        ])?;
        Ok(Self {
            capacity,
            prefill_rows,
            retained_weight_bytes,
            kda_state,
            mla_state,
            decode_scratch,
            decode_routes,
            packed_scratch,
            packed_routes,
            reserve: DYNAMIC_RESERVE_BYTES,
            session_state,
            peaks: Glm5NextPhasePeaks {
                resident: retained_weight_bytes,
                session,
            },
        })
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn prefill_rows(&self) -> u64 {
        self.prefill_rows
    }

    /// Named terms in bytes, for reports and allocation checks.
    pub fn terms(&self) -> [(&'static str, u64); 10] {
        [
            ("retained_weights", self.retained_weight_bytes),
            ("kda_state", self.kda_state),
            ("mla_state", self.mla_state),
            ("decode_scratch", self.decode_scratch),
            ("decode_routes", self.decode_routes),
            ("packed_scratch", self.packed_scratch),
            ("packed_routes", self.packed_routes),
            ("reserve", self.reserve),
            ("session_state", self.session_state),
            (
                "session_buffers",
                self.peaks.session - self.retained_weight_bytes - self.reserve,
            ),
        ]
    }

    /// Recurrent and cache state owned by the session.
    pub fn session_state_bytes(&self) -> u64 {
        self.session_state
    }

    /// Every session buffer (state, scratch, routes), without the reserve.
    pub fn session_buffer_bytes(&self) -> u64 {
        self.peaks.session - self.retained_weight_bytes - self.reserve
    }

    pub fn phase_peaks(&self) -> Glm5NextPhasePeaks {
        self.peaks
    }

    /// What admission compares with the device budget.
    pub fn peak_bytes(&self) -> u64 {
        self.peaks.session
    }

    /// Largest capacity whose peak fits `budget_bytes`, or `None` if even one
    /// position does not fit.
    pub fn max_capacity(
        config: &Glm5NextConfig,
        retained_weight_bytes: u64,
        prefill_rows: u64,
        budget_bytes: u64,
    ) -> Result<Option<u64>> {
        let fits = |capacity: u64| -> Result<bool> {
            Ok(
                Self::new(config, retained_weight_bytes, capacity, prefill_rows)?.peak_bytes()
                    <= budget_bytes,
            )
        };
        if !fits(1)? {
            return Ok(None);
        }
        let (mut lo, mut hi) = (1u64, u64::from(config.context_length));
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if fits(mid)? {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Ok(Some(lo))
    }
}
