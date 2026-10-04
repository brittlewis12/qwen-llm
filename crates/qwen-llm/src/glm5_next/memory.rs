//! Allocation ledger for one GLM-5.3 session.
//!
//! The trunk leaves about 2.5 GiB of the 112 GiB Metal working set. These terms
//! are planning bounds the native session must allocate within; the session
//! packet replaces the aggregate activation term with named, priced requests
//! and lifetimes, and verifies actual allocations against them. Retained
//! weights come from the real retained-window plan, not the tensor census.
//!
//! Layout assumptions: F32 recurrent/conv state; an append-only F16 cache of
//! MLA latents and completed pooled indexer keys (historical key/gate rows are
//! not retained); last-position logits only; one scratch region shared across
//! blocks and sized for the larger of decode and prefill. Prefill attention is
//! online per row (no split partials); decode may split over context. No
//! `[rows, context]` score matrix exists except the indexer's per-pool scores.

use super::{Glm5NextConfig, Glm5NextError, MixerKind, Result};

/// Command buffers, argument tables and allocator slack.
pub const DYNAMIC_RESERVE_BYTES: u64 = 512 * 1024 * 1024;
/// Context splits budgeted for single-token latent attention.
pub const DECODE_ATTENTION_SPLITS: u64 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm5NextMemoryLedger {
    capacity: u64,
    prefill_rows: u64,
    retained_weight_bytes: u64,
    recurrent_state: u64,
    conv_state: u64,
    latent_cache: u64,
    pooled_keys: u64,
    pending_pool: u64,
    decode_activations: u64,
    decode_selection: u64,
    decode_routing: u64,
    decode_attention_partials: u64,
    prefill_activations: u64,
    prefill_selection: u64,
    prefill_routing: u64,
    logits: u64,
    reserve: u64,
    session_state: u64,
    peaks: Glm5NextPhasePeaks,
}

/// Cumulative Metal bytes at each lifetime phase (reserve included after load).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm5NextPhasePeaks {
    pub resident: u64,
    pub session: u64,
    pub prefill: u64,
    pub decode: u64,
}

impl Glm5NextMemoryLedger {
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
        if prefill_rows == 0 {
            return Err(Glm5NextError::InvalidMetadata {
                key: "prefill_rows".into(),
                detail: "must be positive".into(),
            });
        }
        let c = config;
        let kda = c.block_count(MixerKind::Kda) as u64;
        let mla = c.block_count(MixerKind::Mla) as u64;
        let heads = u64::from(c.head_count);
        let head_dim = u64::from(c.kda_head_dim);
        let pool = u64::from(c.indexer_pool);
        let index_dim = u64::from(c.indexer_head_dim);
        let kv = u64::from(c.kv_lora_rank);
        let pools = capacity.div_ceil(pool);
        let row = row_activation_floats(c);
        let selection_row = pools + u64::from(c.selection_width());
        // Route ids/weights/status plus per-expert slot counts and maps.
        let routing_row = 2 * u64::from(c.expert_count) + 2 * u64::from(c.expert_used_count) + 1;
        let m = |terms: &[u64]| -> Result<u64> {
            terms
                .iter()
                .try_fold(1u64, |acc, &t| acc.checked_mul(t))
                .ok_or(Glm5NextError::Overflow("memory ledger"))
        };
        let mut ledger = Self {
            capacity,
            prefill_rows,
            retained_weight_bytes,
            recurrent_state: m(&[kda, heads, head_dim, head_dim, 4])?,
            conv_state: m(&[
                kda,
                3,
                u64::from(c.kda_conv_kernel) - 1,
                u64::from(c.kda_width()),
                4,
            ])?,
            latent_cache: m(&[mla, capacity, kv, 2])?,
            pooled_keys: m(&[mla, pools, index_dim, 2])?,
            pending_pool: m(&[mla, pool, 2, index_dim, 2])?,
            decode_activations: m(&[row, 4])?,
            decode_selection: m(&[selection_row, 4])?,
            decode_routing: m(&[routing_row, 4])?,
            decode_attention_partials: m(&[heads, DECODE_ATTENTION_SPLITS, kv + 2, 4])?,
            prefill_activations: m(&[row, prefill_rows, 4])?,
            prefill_selection: m(&[selection_row, prefill_rows, 4])?,
            prefill_routing: m(&[routing_row, prefill_rows, 4])?,
            logits: m(&[u64::from(c.vocab_size), 4])?,
            reserve: DYNAMIC_RESERVE_BYTES,
            session_state: 0,
            peaks: Glm5NextPhasePeaks {
                resident: 0,
                session: 0,
                prefill: 0,
                decode: 0,
            },
        };
        let sum = |terms: &[u64]| -> Result<u64> {
            terms
                .iter()
                .try_fold(0u64, |acc, &t| acc.checked_add(t))
                .ok_or(Glm5NextError::Overflow("memory ledger"))
        };
        let l = &ledger;
        let session_state = sum(&[
            l.recurrent_state,
            l.conv_state,
            l.latent_cache,
            l.pooled_keys,
            l.pending_pool,
        ])?;
        let session = sum(&[retained_weight_bytes, session_state, l.reserve])?;
        let prefill = sum(&[
            session,
            l.prefill_activations,
            l.prefill_selection,
            l.prefill_routing,
            l.logits,
        ])?;
        let decode = sum(&[
            session,
            l.decode_activations,
            l.decode_selection,
            l.decode_routing,
            l.decode_attention_partials,
            l.logits,
        ])?;
        ledger.session_state = session_state;
        ledger.peaks = Glm5NextPhasePeaks {
            resident: retained_weight_bytes,
            session,
            prefill,
            decode,
        };
        Ok(ledger)
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn prefill_rows(&self) -> u64 {
        self.prefill_rows
    }

    /// Named terms in bytes, for reports and the session's allocation checks.
    pub fn terms(&self) -> [(&'static str, u64); 16] {
        [
            ("retained_weights", self.retained_weight_bytes),
            ("recurrent_state", self.recurrent_state),
            ("conv_state", self.conv_state),
            ("latent_cache", self.latent_cache),
            ("pooled_keys", self.pooled_keys),
            ("pending_pool", self.pending_pool),
            ("decode_activations", self.decode_activations),
            ("decode_selection", self.decode_selection),
            ("decode_routing", self.decode_routing),
            ("decode_attention_partials", self.decode_attention_partials),
            ("prefill_activations", self.prefill_activations),
            ("prefill_selection", self.prefill_selection),
            ("prefill_routing", self.prefill_routing),
            ("logits", self.logits),
            ("reserve", self.reserve),
            ("session_state", self.session_state),
        ]
    }

    /// Bytes owned by a live session independent of the phase.
    pub fn session_state_bytes(&self) -> u64 {
        self.session_state
    }

    pub fn phase_peaks(&self) -> Glm5NextPhasePeaks {
        self.peaks
    }

    /// The largest phase peak; one scratch region serves both phases, so this
    /// is what admission compares with the device budget.
    pub fn peak_bytes(&self) -> u64 {
        self.peaks.prefill.max(self.peaks.decode)
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

/// F32 activations live at once for one packed row. Sub-blocks run in order and
/// share scratch across blocks; mixer and FFN scratch are counted separately
/// (not aliased) so the bound stays conservative.
pub(crate) fn row_activation_floats(c: &Glm5NextConfig) -> u64 {
    let h = u64::from(c.hidden_size);
    let hc = 3 * u64::from(c.hc_width()) + 2 * h + 2 * u64::from(c.hc_mix_count());
    let kda_width = u64::from(c.kda_width());
    let rank = u64::from(c.kda_head_dim);
    // q/k/v before and after conv, decay, output gate and readout.
    let kda = 9 * kda_width + 2 * rank + u64::from(c.head_count);
    let heads = u64::from(c.head_count);
    let kv = u64::from(c.kv_lora_rank);
    let mla_width = u64::from(c.mla_width());
    let indexer = u64::from(c.indexer_head_count) * u64::from(c.indexer_head_dim)
        + 2 * u64::from(c.indexer_head_dim)
        + u64::from(c.indexer_head_count);
    // qr, q, absorbed q, latent, latent output, expanded output, indexer.
    let mla = u64::from(c.q_lora_rank) + 2 * mla_width + 2 * heads * kv + kv + indexer;
    let mixer = kda.max(mla) + h;
    let used = u64::from(c.expert_used_count);
    let moe = u64::from(c.expert_count)
        + 2 * used
        + used * (2 * u64::from(c.expert_ffn_size) + h)
        + 2 * u64::from(c.shared_expert_ffn_size)
        + h;
    let dense = 2 * u64::from(c.dense_ffn_size) + h;
    hc + mixer + moe.max(dense)
}
