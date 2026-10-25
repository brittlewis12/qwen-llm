//! Prefer compatible live state (no restore copy), then the longest usable RAM
//! snapshot, then cold. This is deliberately not longest-prefix across both
//! sources. Snapshot retention/expiry never governs the live session.
use super::super::snapshot_cache::SnapshotCache;
use super::*;
use qwen_llm::deepseek_v41::forward::{Snapshot, State};

mod operations;
#[cfg(test)]
mod tests;

pub(super) struct Live<S> {
    pub state: S,
    pub tokens: Vec<u32>,
    pub logits: Vec<f32>,
}

pub(super) struct Restored<S> {
    pub live: Live<S>,
    pub source: &'static str,
}

pub(super) trait Reuse<S> {
    /// Non-allocating request preparation. Retire incompatible idle ownership
    /// before output admission; do not grow, restore, or consume compatible live.
    fn prepare(&mut self, _: &[u32]) {}
    fn evict_for(&mut self, _: u64) -> qwen_llm::snapshot_policy::Evicted {
        Default::default()
    }
    /// Allocating phase, after `prepare` and output admission. Each source must
    /// admit its own additional state/scratch peak while retaining `reserve`.
    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
        reserve: u64,
    ) -> Result<Option<Restored<S>>, ServeError>;
    fn prefill_report(
        &self,
        _: &S,
        start: usize,
        tokens: usize,
    ) -> qwen_llm::deepseek_v41::forward::PrefillReport {
        qwen_llm::deepseek_v41::forward::PrefillReport {
            start,
            tokens,
            selected_rows: crate::family_options::V41_PREFILL_ROWS,
        }
    }
    fn checkpoint(&mut self, state: &S, ids: &[u32], logits: &[f32], reserve: u64);
    fn commit(&mut self, live: Live<S>);
    fn capacity(&self, state: &S, required: usize) -> usize;
}

pub(super) struct Checkpoint {
    snapshot: Snapshot,
    logits: Vec<f32>,
}

pub(super) struct Store {
    live: Option<Live<State>>,
    cache: SnapshotCache<Checkpoint>,
}

impl Store {
    pub fn new(bytes: u64, policy: qwen_llm::snapshot_policy::SnapshotPolicyConfig) -> Self {
        Self {
            live: None,
            cache: SnapshotCache::new(bytes, policy),
        }
    }

    pub fn sweep(&mut self) {
        super::super::log_expired_snapshots("deepseek_v41", &self.cache.sweep());
    }
}

pub(super) struct Session<'a, 'b> {
    pub forward: &'a Forward<'b>,
    pub store: &'a mut Store,
    pub ctx: &'a MetalContext,
}

impl Reuse<State> for Session<'_, '_> {
    fn prepare(&mut self, ids: &[u32]) {
        operations::discard_incompatible(&mut self.store.live, ids, |state| {
            (state.position(), !state.is_poisoned())
        });
    }

    fn evict_for(&mut self, bytes: u64) -> qwen_llm::snapshot_policy::Evicted {
        self.store.cache.evict_for(bytes)
    }

    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
        reserve: u64,
    ) -> Result<Option<Restored<State>>, ServeError> {
        self.store.cache.sweep();
        let forward = self.forward;
        let suffix = ids.len() - self.store.live.as_ref().map_or(0, |live| live.tokens.len());
        operations::live_then_snapshot(
            self.store.live.take(),
            |state| forward.ensure_capacity_for_request(state, required, suffix, reserve),
            || self.take_snapshot(ids, required, reserve),
        )
    }

    fn prefill_report(
        &self,
        state: &State,
        _: usize,
        _: usize,
    ) -> qwen_llm::deepseek_v41::forward::PrefillReport {
        state
            .last_prefill_report()
            .expect("successful native prefill reports geometry")
    }

    fn checkpoint(&mut self, state: &State, ids: &[u32], logits: &[f32], reserve: u64) {
        self.capture_checkpoint(state, ids, logits, reserve);
    }

    fn commit(&mut self, live: Live<State>) {
        self.store.live = Some(live);
    }
    fn capacity(&self, state: &State, _: usize) -> usize {
        state.capacity()
    }
}

impl Session<'_, '_> {
    fn take_snapshot(
        &mut self,
        ids: &[u32],
        required: usize,
        reserve: u64,
    ) -> Result<Option<Restored<State>>, ServeError> {
        let hit = operations::snapshot_candidate(&mut self.store.cache, ids, |checkpoint| {
            !checkpoint.logits.is_empty()
        });
        let Some(hit) = hit else { return Ok(None) };
        let suffix = ids.len() - hit.prefix_len;
        let Some((state, hit)) = operations::restore_candidate(
            &mut self.store.cache,
            hit,
            || {
                let plan = self.forward.plan_request(required, suffix, reserve)?;
                self.forward.new_state_from_plan(&plan)
            },
            |state, checkpoint| self.forward.restore_snapshot(state, &checkpoint.snapshot),
        )?
        else {
            return Ok(None);
        };
        Ok(Some(Restored {
            live: Live {
                state,
                tokens: ids[..hit.prefix_len].to_vec(),
                logits: hit.value.logits.clone(),
            },
            source: "snapshot_ram",
        }))
    }

    fn capture_checkpoint(&mut self, state: &State, ids: &[u32], logits: &[f32], reserve: u64) {
        let Some(logits_bytes) = (logits.len() as u64).checked_mul(size_of::<f32>() as u64) else {
            return;
        };
        operations::capture_checkpoint(
            &mut self.store.cache,
            ids,
            logits_bytes,
            reserve,
            || self.ctx.memory_signals(),
            || self.forward.snapshot_payload_bytes(state),
            |expected_bytes| {
                let snapshot = self.forward.capture_snapshot(state)?;
                if snapshot.position() != ids.len() || snapshot.payload_bytes() != expected_bytes {
                    return Err(qwen_llm::deepseek_v41::forward::StateError::Invalid(
                        "snapshot extent changed during capture".into(),
                    ));
                }
                Ok(Checkpoint {
                    snapshot,
                    logits: logits.to_vec(),
                })
            },
        );
    }
}

#[cfg(test)]
pub(super) struct NoReuse;
#[cfg(test)]
impl<S> Reuse<S> for NoReuse {
    fn take(&mut self, _: &[u32], _: usize, _: u64) -> Result<Option<Restored<S>>, ServeError> {
        Ok(None)
    }
    fn checkpoint(&mut self, _: &S, _: &[u32], _: &[f32], _: u64) {}
    fn commit(&mut self, _: Live<S>) {}
    fn capacity(&self, _: &S, required: usize) -> usize {
        required
    }
}
