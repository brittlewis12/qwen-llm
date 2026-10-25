//! In-process, same-loaded-model exact causal snapshots. No transient activations.
use super::*;

pub struct Snapshot {
    identity: Arc<()>,
    addresses: engram::AddressState,
    prefill_origin: PrefillExecution,
    payload: Vec<f32>,
}

impl Snapshot {
    pub fn position(&self) -> usize {
        self.addresses.position() as usize
    }

    pub fn prefill_origin(&self) -> PrefillExecution {
        self.prefill_origin
    }

    /// Causal tensor bytes; fixed-size address and identity bookkeeping excluded.
    pub fn payload_bytes(&self) -> usize {
        self.payload.len() * std::mem::size_of::<f32>()
    }
}

fn allocate_with_capacity_fallback<T>(
    preferred: usize,
    required: usize,
    mut allocate: impl FnMut(usize) -> std::result::Result<T, StateError>,
) -> std::result::Result<T, StateError> {
    match allocate(preferred) {
        Err(StateError::MemoryAdmissionDenied(_)) if preferred > required => allocate(required),
        result => result,
    }
}

impl Forward<'_> {
    fn validate_snapshot_state(&self, state: &State) -> Result<()> {
        op::require(state.is_ready(), "snapshot requires ready state")?;
        op::require(
            Arc::ptr_eq(&self.identity, &state.identity),
            "state belongs to another model",
        )?;
        op::require(
            state.position() <= state.max_context,
            "invalid state position",
        )?;
        state
            .gpu_step
            .as_ref()
            .ok_or("snapshot requires GPU state")?
            .attention()
            .validate_snapshot_position(state.position())
    }

    fn snapshot_regions<'s>(
        &self,
        state: &'s State,
    ) -> std::result::Result<(Vec<(&'s MetalTensor, usize)>, usize), StateError> {
        self.validate_snapshot_state(state)?;
        let regions = state
            .gpu_step
            .as_ref()
            .unwrap()
            .attention()
            .causal_regions(state.position())?;
        let count = regions
            .iter()
            .try_fold(0usize, |sum, (_, n)| sum.checked_add(*n))
            .ok_or_else(|| "snapshot extent overflow".to_string())?;
        let bytes = count
            .checked_mul(std::mem::size_of::<f32>())
            .filter(|&n| n <= isize::MAX as usize)
            .ok_or_else(|| "snapshot byte overflow".to_string())?;
        Ok((regions, bytes))
    }

    /// Exact causal tensor payload size without copying or allocating the payload.
    /// Like `Snapshot::payload_bytes`, excludes fixed-size bookkeeping.
    pub fn snapshot_payload_bytes(&self, state: &State) -> std::result::Result<usize, StateError> {
        self.snapshot_regions(state).map(|(_, bytes)| bytes)
    }

    pub fn capture_snapshot(&self, state: &State) -> std::result::Result<Snapshot, StateError> {
        let (regions, bytes) = self.snapshot_regions(state)?;
        let count = bytes / std::mem::size_of::<f32>();
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            0,
            bytes as u64 + 4096,
            0,
            self.ctx.memory_signals(),
            true,
        );
        if let Some(reason) = admission.refusal() {
            return Err(StateError::MemoryAdmissionDenied(reason));
        }
        let mut payload = Vec::new();
        payload
            .try_reserve_exact(count)
            .map_err(|e| format!("snapshot allocation: {e}"))?;
        for (tensor, count) in regions {
            payload.extend_from_slice(&f32_slice(tensor)?[..count]);
        }
        Ok(Snapshot {
            identity: Arc::clone(&self.identity),
            addresses: state.addresses,
            prefill_origin: state.prefill_origin,
            payload,
        })
    }

    pub fn restore_snapshot(
        &self,
        state: &mut State,
        snapshot: &Snapshot,
    ) -> std::result::Result<(), StateError> {
        self.validate_snapshot_state(state)?;
        op::require(
            Arc::ptr_eq(&self.identity, &snapshot.identity),
            "snapshot belongs to another model",
        )?;
        op::require(
            snapshot.position() <= state.max_context,
            "snapshot exceeds state capacity",
        )?;
        let attention = state.gpu_step.as_mut().unwrap().attention_mut();
        attention.validate_causal_restore(snapshot.position(), snapshot.payload.len())?;
        state.phase = StatePhase::Poisoned;
        attention.restore_causal(snapshot.position(), &snapshot.payload)?;
        state.addresses = snapshot.addresses;
        state.prefill_origin = snapshot.prefill_origin;
        state.phase = StatePhase::Ready;
        Ok(())
    }

    pub fn ensure_capacity(
        &self,
        state: &mut State,
        required: usize,
    ) -> std::result::Result<(), StateError> {
        self.validate_snapshot_state(state)?;
        let limit = usize::try_from(self.engram_hash.max_position)
            .map_err(|e| format!("context limit: {e}"))?;
        op::require(required <= limit, "required capacity exceeds context limit")?;
        if required <= state.max_context {
            return Ok(());
        }
        let capacity = required.max(state.max_context.saturating_mul(2)).min(limit);
        let snapshot = self.capture_snapshot(state)?;
        let mut replacement =
            allocate_with_capacity_fallback(capacity, required, |n| self.new_state(n))?;
        self.restore_snapshot(&mut replacement, &snapshot)?;
        *state = replacement;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
