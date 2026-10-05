//! Allocation ledger for one GLM-5.3 session.
//!
//! The trunk leaves about 2.5 GiB of the 112 GiB Metal working set, so the
//! ledger prices exactly what the session allocates: the native session builds
//! every buffer from the [`BufferSpec`] lists below, and the ledger sums the
//! same lists through a [`BufferPricer`] before allocation. Live sessions price
//! with the device; planning without a device uses the named
//! [`apple_16k_price`] profile. All size arithmetic is checked. Retained
//! weights come from the real retained-window plan.
//!
//! Layout: F32 recurrent/conv state; an append-only F16 cache of MLA latents
//! and completed pooled indexer keys plus a 4-slot pending ring; decode scratch
//! and optional packed-prefill scratch, both resident for the session's life;
//! per-MoE-block route records for each; last-position logits only; and,
//! for sessions that reach the sparse frontier, single-token and packed
//! (microbatched) sparse-selection scratch.

use super::{FfnKind, Glm5NextConfig, Glm5NextError, MixerKind, Result};

/// Command buffers, argument tables and allocator slack.
pub const DYNAMIC_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
/// Page granule of the device-free Apple silicon pricing profile.
pub const APPLE_16K_GRANULE: u64 = 16 * 1024;

/// Allocation price of one buffer of `logical_bytes`, or `None` when the
/// buffer cannot be allocated (or its price is unrepresentable).
pub type BufferPricer<'a> = &'a dyn Fn(u64) -> Option<u64>;

/// Device-free planning profile for Apple silicon with 16 KiB pages and
/// shared buffers priced at page granularity (M4 Max measured within it).
/// Live admission prices with the device instead
/// (`MetalContext::price_shared_buffer_upper`).
pub fn apple_16k_price(logical_bytes: u64) -> Option<u64> {
    logical_bytes
        .max(1)
        .checked_next_multiple_of(APPLE_16K_GRANULE)
}

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

    /// Logical bytes, or `None` if the shape's byte size overflows.
    pub fn bytes(&self) -> Option<u64> {
        self.shape
            .iter()
            .try_fold(self.dtype.bytes(), |acc, &dim| acc.checked_mul(dim))
    }

    /// Price under `price`, with overflow and unallocatable sizes as errors.
    pub fn priced_bytes(&self, price: BufferPricer<'_>) -> Result<u64> {
        let bytes = self.bytes().ok_or(Glm5NextError::Overflow("buffer size"))?;
        price(bytes).ok_or_else(|| Glm5NextError::Unsupported {
            key: format!("buffer {}", self.name),
            detail: format!("{bytes} bytes cannot be allocated on this device"),
        })
    }
}

/// Checked sum of the priced specs.
pub fn priced(specs: &[BufferSpec], price: BufferPricer<'_>) -> Result<u64> {
    specs.iter().try_fold(0u64, |acc, spec| {
        acc.checked_add(spec.priced_bytes(price)?)
            .ok_or(Glm5NextError::Overflow("buffer prices"))
    })
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

/// Packed-prefill scratch for chunks of up to `rows` tokens. Products of
/// dimensions saturate, so an absurd `rows` fails [`BufferSpec::bytes`]
/// instead of wrapping to a small shape.
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
        z("slots", I32, &[e.saturating_mul(r)]),
        z(
            "inner",
            F32,
            &[c.expert_ffn_size as u64, k.saturating_mul(r)],
        ),
        z("slot_out", F32, &[h, k.saturating_mul(r)]),
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

/// Sparse-selection scratch for single-token decode, needed only when the
/// session can reach the sparse frontier: indexer query (F32 projection and
/// its F16 rounding), scaled head weights, one score per pool, selected pool
/// ids, expanded latent rows, per-token visibility, and one sticky selector
/// status per MLA block (each block selects into its own slot).
pub fn sparse_decode_specs(c: &Glm5NextConfig, capacity: u64) -> Vec<BufferSpec> {
    let pools = capacity.div_ceil(c.indexer_pool as u64).max(1);
    let (ih, id) = (c.indexer_head_count as u64, c.indexer_head_dim as u64);
    let mla = c.block_count(MixerKind::Mla) as u64;
    let units = split_attention_units(c, 1);
    let z = BufferSpec::zeros;
    vec![
        z("index_query", F32, &[id * ih]),
        z("index_query_f16", F16, &[id, ih, 1]),
        z("index_weights", F32, &[ih, 1]),
        z("scores", F32, &[pools, 1]),
        z("pool_ids", I32, &[c.selected_pool_count() as u64, 1]),
        z("pool_counts", I32, &[1]),
        z("visible_pools", I32, &[1]),
        z("visible_rows", I32, &[1]),
        z("row_ids", I32, &[c.selection_width() as u64, 1]),
        z("row_counts", I32, &[1]),
        z("select_status", I32, &[mla.max(1)]),
        z("attention_partials", F32, &[c.kv_lora_rank as u64, units]),
        z("attention_partial_stats", F32, &[2, units]),
    ]
}

/// Queries per split selected-attention dispatch in packed prefill: bounds
/// the partial scratch (one 512-wide accumulator per query, head and
/// split) without changing any query's result.
pub const PACKED_SPLIT_QUERIES: u64 = 16;

/// (query, head, split) work units of the split selected attention for
/// `queries` queries over the release selection width (window 0).
pub fn split_attention_units(c: &Glm5NextConfig, queries: u64) -> u64 {
    let splits = crate::metal::selected_attention_splits(0, c.selection_width() as usize) as u64;
    queries * c.head_count as u64 * splits
}

/// Sparse rows of one packed chunk run in microbatches of at most this many
/// queries, reusing the score, selection and expansion scratch.
pub const PACKED_SPARSE_QUERIES: u64 = 64;

/// Packed-prefill sparse-selection scratch for chunks of up to `rows` tokens
/// in a session of `capacity` positions: chunk-wide indexer queries, head
/// weights and per-row visibility; microbatch scores, selected pools and
/// expanded rows for [`PACKED_SPARSE_QUERIES`] queries; and one sticky
/// selector status per (MLA block, chunk row).
pub fn packed_sparse_specs(c: &Glm5NextConfig, capacity: u64, rows: u64) -> Vec<BufferSpec> {
    let pools = capacity.div_ceil(c.indexer_pool as u64).max(1);
    let (ih, id) = (c.indexer_head_count as u64, c.indexer_head_dim as u64);
    let mla = c.block_count(MixerKind::Mla) as u64;
    let s = rows.min(PACKED_SPARSE_QUERIES);
    let units = split_attention_units(c, s.min(PACKED_SPLIT_QUERIES));
    let z = BufferSpec::zeros;
    vec![
        z("index_query", F32, &[id * ih, rows]),
        z("index_query_f16", F16, &[id, ih, rows]),
        z("index_weights", F32, &[ih, rows]),
        z("visible_pools", I32, &[rows]),
        z("visible_rows", I32, &[rows]),
        z("scores", F32, &[pools, s]),
        z("pool_ids", I32, &[c.selected_pool_count() as u64, s]),
        z("pool_counts", I32, &[s]),
        z("row_ids", I32, &[c.selection_width() as u64, s]),
        z("row_counts", I32, &[s]),
        z("select_status", I32, &[mla.max(1).saturating_mul(rows)]),
        z("attention_partials", F32, &[c.kv_lora_rank as u64, units]),
        z("attention_partial_stats", F32, &[2, units]),
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
    sparse_decode: u64,
    packed_scratch: u64,
    packed_sparse: u64,
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
    /// Prices a session of `capacity` positions whose packed chunks hold up
    /// to `prefill_rows` tokens (`0`: decode-only; at most `capacity`), with
    /// every buffer priced by `price`.
    pub fn new(
        config: &Glm5NextConfig,
        retained_weight_bytes: u64,
        capacity: u64,
        prefill_rows: u64,
        price: BufferPricer<'_>,
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
        if prefill_rows > capacity {
            return Err(Glm5NextError::InvalidMetadata {
                key: "prefill_rows".into(),
                detail: format!("{prefill_rows} exceeds the session capacity {capacity}"),
            });
        }
        let c = config;
        let overflow = || Glm5NextError::Overflow("memory ledger");
        let kda = c.block_count(MixerKind::Kda) as u64;
        let mla = c.block_count(MixerKind::Mla) as u64;
        let moe = c.blocks.iter().filter(|b| b.ffn == FfnKind::Moe).count() as u64;
        let times = |n: u64, bytes: u64| n.checked_mul(bytes).ok_or_else(overflow);
        let kda_state = times(kda, priced(&kda_state_specs(c), price)?)?;
        let mla_state = times(mla, priced(&mla_state_specs(c, capacity), price)?)?;
        let decode_scratch = priced(&decode_scratch_specs(c), price)?;
        let decode_routes = times(moe, priced(&route_specs(c, None), price)?)?;
        let sparse_decode = if capacity >= u64::from(c.sparse_frontier()) {
            priced(&sparse_decode_specs(c, capacity), price)?
        } else {
            0
        };
        let packed_sparse = if prefill_rows > 0 && capacity >= u64::from(c.sparse_frontier()) {
            priced(&packed_sparse_specs(c, capacity, prefill_rows), price)?
        } else {
            0
        };
        let (packed_scratch, packed_routes) = if prefill_rows == 0 {
            (0, 0)
        } else {
            (
                priced(&packed_scratch_specs(c, prefill_rows), price)?,
                times(moe, priced(&route_specs(c, Some(prefill_rows)), price)?)?,
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
            sparse_decode,
            packed_scratch,
            packed_sparse,
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
            sparse_decode,
            packed_scratch,
            packed_sparse,
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
    pub fn terms(&self) -> [(&'static str, u64); 12] {
        [
            ("retained_weights", self.retained_weight_bytes),
            ("kda_state", self.kda_state),
            ("mla_state", self.mla_state),
            ("decode_scratch", self.decode_scratch),
            ("decode_routes", self.decode_routes),
            ("sparse_decode", self.sparse_decode),
            ("packed_scratch", self.packed_scratch),
            ("packed_sparse", self.packed_sparse),
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

    /// Largest capacity (at least `prefill_rows`, and at least 1) whose peak
    /// fits `budget_bytes`, or `None` if the smallest one does not fit.
    pub fn max_capacity(
        config: &Glm5NextConfig,
        retained_weight_bytes: u64,
        prefill_rows: u64,
        budget_bytes: u64,
        price: BufferPricer<'_>,
    ) -> Result<Option<u64>> {
        let fits = |capacity: u64| -> Result<bool> {
            let ledger = Self::new(config, retained_weight_bytes, capacity, prefill_rows, price)?;
            Ok(ledger.peak_bytes() <= budget_bytes)
        };
        let smallest = prefill_rows.max(1);
        if !fits(smallest)? {
            return Ok(None);
        }
        let (mut lo, mut hi) = (smallest, u64::from(config.context_length));
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
