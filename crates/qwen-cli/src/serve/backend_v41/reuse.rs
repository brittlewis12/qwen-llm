use super::super::{decode_loop, snapshot_cache::SnapshotCache};
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
    fn take(&mut self, ids: &[u32], required: usize) -> Result<Option<Restored<S>>, ServeError>;
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
    pub headroom: Headroom,
}

impl Reuse<State> for Session<'_, '_> {
    fn take(
        &mut self,
        ids: &[u32],
        required: usize,
    ) -> Result<Option<Restored<State>>, ServeError> {
        self.store.cache.sweep();
        if let Some(live) = self.store.live.take() {
            let matched = decode_loop::extending_prefix(
                &live.tokens,
                ids,
                live.state.position(),
                !live.state.is_poisoned(),
            );
            let equal = !live.state.is_poisoned()
                && live.tokens == ids
                && live.state.position() == ids.len();
            if matched > 0 || equal {
                return operations::grow_live(live, |state| {
                    self.forward.ensure_capacity(state, required)
                })
                .map(|live| {
                    live.map(|live| Restored {
                        live,
                        source: "live_session",
                    })
                });
            }
        }
        let hit = self
            .store
            .cache
            .peek_exact_in(&(), ids)
            .filter(|hit| !hit.value.logits.is_empty())
            .or_else(|| self.store.cache.peek_best_prefix_in(&(), ids));
        let Some(hit) = hit else { return Ok(None) };
        let Some((state, hit)) = operations::restore_candidate(
            &mut self.store.cache,
            hit,
            || self.forward.new_state_with_gpu_step(required),
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

    fn checkpoint(&mut self, state: &State, ids: &[u32], logits: &[f32], reserve: u64) {
        let Some(logits_bytes) = (logits.len() as u64).checked_mul(size_of::<f32>() as u64) else {
            return;
        };
        operations::capture_checkpoint(
            &mut self.store.cache,
            ids,
            logits_bytes,
            reserve,
            self.headroom,
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

    fn commit(&mut self, live: Live<State>) {
        self.store.live = Some(live);
    }
    fn capacity(&self, state: &State, _: usize) -> usize {
        state.capacity()
    }
}

#[cfg(test)]
pub(super) struct NoReuse;
#[cfg(test)]
impl<S> Reuse<S> for NoReuse {
    fn take(&mut self, _: &[u32], _: usize) -> Result<Option<Restored<S>>, ServeError> {
        Ok(None)
    }
    fn checkpoint(&mut self, _: &S, _: &[u32], _: &[f32], _: u64) {}
    fn commit(&mut self, _: Live<S>) {}
    fn capacity(&self, _: &S, required: usize) -> usize {
        required
    }
}
