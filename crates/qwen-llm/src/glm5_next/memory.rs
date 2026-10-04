//! Allocation ledger for one GLM-5.3 session.
//!
//! The trunk leaves about 2.5 GiB of the 112 GiB Metal working set, so this
//! ledger is the allocation contract rather than an estimate to revisit later:
//! the native session derives its buffers from these terms and must verify
//! that actual allocations stay within them. Retained weights come from the
//! real retained-window plan, not the tensor census.
//!
//! Layout assumptions: F32 recurrent/conv state; an append-only F16 cache of
//! MLA latents and completed pooled indexer keys (historical key/gate rows are
//! not retained); last-position logits only; scratch shared across blocks and
//! sized per packed row. Attention and selection use online/top-k kernels with
//! no `[rows, context]` score matrix except the indexer's per-pool scores.

use super::{Glm5NextConfig, Glm5NextError, MixerKind, Result};

/// Command buffers, argument tables and allocator slack.
pub const DYNAMIC_RESERVE_BYTES: u64 = 512 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Glm5NextMemoryLedger {
    pub capacity: u64,
    pub prefill_rows: u64,
    pub retained_weight_bytes: u64,
    /// KDA S matrices: blocks x heads x d x d, F32.
    pub recurrent_state_bytes: u64,
    /// KDA q/k/v conv tails: blocks x 3 x (kernel - 1) x width, F32.
    pub conv_state_bytes: u64,
    /// MLA latent rows: blocks x capacity x kv_lora_rank, F16.
    pub latent_cache_bytes: u64,
    /// Completed pooled indexer keys: blocks x pools x indexer dim, F16.
    pub pooled_key_bytes: u64,
    /// Incomplete pool key|gate rows: blocks x pool x 2 x indexer dim, F16.
    pub pending_pool_bytes: u64,
    /// Activations for one row, shared across blocks.
    pub decode_scratch_bytes: u64,
    /// Activations for `prefill_rows` packed rows, shared across blocks.
    pub prefill_scratch_bytes: u64,
    /// Indexer pool scores and selected ids for one row.
    pub decode_selection_bytes: u64,
    /// Indexer pool scores and selected ids for `prefill_rows` rows.
    pub prefill_selection_bytes: u64,
    /// Last-position F32 logits.
    pub logits_bytes: u64,
    pub reserve_bytes: u64,
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
        let head_dim = u64::from(c.kda_head_dim);
        let pool = u64::from(c.indexer_pool);
        let index_dim = u64::from(c.indexer_head_dim);
        let pools = capacity.div_ceil(pool);
        let row = row_activation_floats(c);
        let selection_row = pools + u64::from(c.selection_width());
        let m = |terms: &[u64]| -> Result<u64> {
            terms
                .iter()
                .try_fold(1u64, |acc, &t| acc.checked_mul(t))
                .ok_or(Glm5NextError::Overflow("memory ledger"))
        };
        Ok(Self {
            capacity,
            prefill_rows,
            retained_weight_bytes,
            recurrent_state_bytes: m(&[kda, u64::from(c.head_count), head_dim, head_dim, 4])?,
            conv_state_bytes: m(&[
                kda,
                3,
                u64::from(c.kda_conv_kernel) - 1,
                u64::from(c.kda_width()),
                4,
            ])?,
            latent_cache_bytes: m(&[mla, capacity, u64::from(c.kv_lora_rank), 2])?,
            pooled_key_bytes: m(&[mla, pools, index_dim, 2])?,
            pending_pool_bytes: m(&[mla, pool, 2, index_dim, 2])?,
            decode_scratch_bytes: m(&[row, 4])?,
            prefill_scratch_bytes: m(&[row, prefill_rows, 4])?,
            decode_selection_bytes: m(&[selection_row, 4])?,
            prefill_selection_bytes: m(&[selection_row, prefill_rows, 4])?,
            logits_bytes: m(&[u64::from(c.vocab_size), 4])?,
            reserve_bytes: DYNAMIC_RESERVE_BYTES,
        })
    }

    /// Bytes owned by a live session independent of the phase.
    pub fn session_state_bytes(&self) -> u64 {
        self.recurrent_state_bytes
            + self.conv_state_bytes
            + self.latent_cache_bytes
            + self.pooled_key_bytes
            + self.pending_pool_bytes
    }

    pub fn phase_peaks(&self) -> Glm5NextPhasePeaks {
        let session = self.retained_weight_bytes + self.session_state_bytes() + self.reserve_bytes;
        Glm5NextPhasePeaks {
            resident: self.retained_weight_bytes,
            session,
            prefill: session
                + self.prefill_scratch_bytes
                + self.prefill_selection_bytes
                + self.logits_bytes,
            decode: session
                + self.decode_scratch_bytes
                + self.decode_selection_bytes
                + self.logits_bytes,
        }
    }

    /// The largest phase peak; admission compares this with the device budget.
    pub fn peak_bytes(&self) -> u64 {
        let p = self.phase_peaks();
        p.prefill.max(p.decode)
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
/// (not aliased) so the ledger stays conservative.
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
