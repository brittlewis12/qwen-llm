//! Prefer compatible live state (no restore copy), then the longest usable RAM
//! snapshot, then cold. This is deliberately not longest-prefix across both
//! sources. Snapshot retention/expiry never governs the live session.
use super::super::snapshot_cache::SnapshotCache;
use super::*;
use qwen_llm::deepseek_v41::forward::{PrefillExecution, Snapshot, State};

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
    ) -> qwen_llm::deepseek_v41::forward::PrefillGeometry {
        qwen_llm::deepseek_v41::forward::PrefillGeometry {
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
    live: operations::LivePool<State>,
    cache: SnapshotCache<Checkpoint>,
}

impl Store {
    /// One idle live state (the single live session) and a snapshot cache.
    pub fn new(bytes: u64, policy: qwen_llm::snapshot_policy::SnapshotPolicyConfig) -> Self {
        Self {
            live: operations::LivePool::new(1, None),
            cache: SnapshotCache::new(bytes, policy),
        }
    }

    /// Keep up to `states` idle live states within an optional byte budget.
    pub fn with_live_budget(mut self, states: usize, bytes: Option<u64>) -> Self {
        self.live = operations::LivePool::new(states, bytes);
        self
    }

    pub fn sweep(&mut self) {
        super::super::log_expired_snapshots("deepseek_v41", &self.cache.sweep());
    }
}

pub(super) struct Session<'a, 'b> {
    pub forward: &'a Forward<'b>,
    pub store: &'a mut Store,
    pub ctx: &'a MetalContext,
    /// The prompt being admitted, between `prepare` and `take`: memory relief
    /// never releases an idle state that this request could take.
    pub protect: Vec<u32>,
}

impl<'a, 'b> Session<'a, 'b> {
    pub fn new(forward: &'a Forward<'b>, store: &'a mut Store, ctx: &'a MetalContext) -> Self {
        Self {
            forward,
            store,
            ctx,
            protect: Vec::new(),
        }
    }
}

fn reuse_eligible(ready: bool, origin: PrefillExecution) -> bool {
    ready && origin == PrefillExecution::FullForward
}

fn describe(state: &State) -> (usize, bool) {
    (
        state.position(),
        reuse_eligible(state.is_ready(), state.prefill_origin()),
    )
}

impl Reuse<State> for Session<'_, '_> {
    fn prepare(&mut self, ids: &[u32]) {
        self.store.live.prepare(ids, describe);
        self.protect = ids.to_vec();
    }

    /// Snapshots first, then idle live states the admitting request could not
    /// take. With one idle state that is never one (prepare kept only a
    /// compatible state), so relief is snapshots-only as before.
    fn evict_for(&mut self, bytes: u64) -> qwen_llm::snapshot_policy::Evicted {
        let mut evicted = self.store.cache.evict_for(bytes);
        let remaining = bytes.saturating_sub(evicted.bytes);
        if remaining > 0 {
            let (states, released) =
                self.store
                    .live
                    .release_idle(remaining, &self.protect, describe);
            if states > 0 {
                tracing::info!(target: "qwen_diag",
                    "serve: deepseek_v41 released {states} idle live states ({released} bytes) for memory relief");
                evicted.bytes = evicted.bytes.saturating_add(released);
                evicted
                    .ids
                    .extend((0..states as u64).map(qwen_llm::snapshot_policy::EntryId::external));
            }
        }
        evicted
    }

    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
        reserve: u64,
    ) -> Result<Option<Restored<State>>, ServeError> {
        self.store.cache.sweep();
        self.protect.clear();
        let forward = self.forward;
        let live = self.store.live.take(ids, describe);
        let suffix = ids.len() - live.as_ref().map_or(0, |live| live.tokens.len());
        operations::live_then_snapshot(
            live,
            |state| forward.ensure_capacity_for_request(state, required, suffix, reserve),
            || self.take_snapshot(ids, required, reserve),
        )
    }

    fn prefill_report(
        &self,
        state: &State,
        _: usize,
        _: usize,
    ) -> qwen_llm::deepseek_v41::forward::PrefillGeometry {
        state
            .last_prefill_report()
            .expect("successful native prefill reports geometry")
    }

    fn checkpoint(&mut self, state: &State, ids: &[u32], logits: &[f32], reserve: u64) {
        self.capture_checkpoint(state, ids, logits, reserve);
    }

    /// Only ready FullForward states are retained (never CED-origin or
    /// poisoned ones); the pool's budgets then apply.
    fn commit(&mut self, live: Live<State>) {
        if !reuse_eligible(live.state.is_ready(), live.state.prefill_origin()) {
            return;
        }
        let bytes = self
            .forward
            .estimate_request_memory_with_gpu_step(live.state.capacity())
            .map_or(u64::MAX, |estimate| estimate.total_bytes);
        self.store.live.commit(live, bytes);
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
                && reuse_eligible(true, checkpoint.snapshot.prefill_origin())
        })
        .filter(|hit| reuse_eligible(true, hit.value.snapshot.prefill_origin()));
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
        if !reuse_eligible(state.is_ready(), state.prefill_origin()) {
            return Ok(None);
        }
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
        if !reuse_eligible(state.is_ready(), state.prefill_origin()) {
            return;
        }
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
                if snapshot.position() != ids.len()
                    || snapshot.payload_bytes() != expected_bytes
                    || !reuse_eligible(true, snapshot.prefill_origin())
                {
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
