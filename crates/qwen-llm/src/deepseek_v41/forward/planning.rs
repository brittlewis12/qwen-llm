//! Admission selects geometry before encoding, never as a retry of mutated KV.
//! Chunk width can change arithmetic/routing; there is no bitwise-equivalence contract.
use super::*;
use crate::metal::{MetalMemoryAdmissionReason as Reason, MetalMemorySignals};

#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
/// Logical causal tokens, not padded tensor/expert storage widths.
pub struct PrefillGeometry {
    pub start: usize,
    pub tokens: usize,
    pub selected_rows: usize,
}

impl PrefillGeometry {
    pub fn chunk_count(self) -> usize {
        self.tokens.div_ceil(self.selected_rows)
    }

    pub fn max_chunk_rows(self) -> usize {
        self.tokens.min(self.selected_rows)
    }

    /// Absolute causal ranges, not outer API calls or padded tensor widths.
    pub fn chunk_ranges(self) -> impl Iterator<Item = std::ops::Range<usize>> {
        (0..self.tokens)
            .step_by(self.selected_rows)
            .map(move |offset| {
                self.start + offset
                    ..self.start + offset + (self.tokens - offset).min(self.selected_rows)
            })
    }
}

/// One cold request: exact state capacity plus the additional prefill high water
/// and CPU-only caller reserve. A plan is not a reservation; allocation rechecks
/// current signals without changing the selected geometry.
pub struct RequestPlan {
    capacity: usize,
    report: PrefillGeometry,
    memory: RequestMemoryEstimate,
    cpu_reserve: u64,
    identity: Arc<()>,
}

impl RequestPlan {
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn prefill(&self) -> PrefillGeometry {
        self.report
    }
    pub fn memory(&self) -> RequestMemoryEstimate {
        self.memory
    }
    pub fn caller_cpu_reserve(&self) -> u64 {
        self.cpu_reserve
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Cost {
    pub metal: u64,
    pub cpu: u64,
}

impl Cost {
    fn plus(self, other: Self) -> Result<Self> {
        Ok(Self {
            metal: self
                .metal
                .checked_add(other.metal)
                .ok_or("request Metal cost overflow")?,
            cpu: self
                .cpu
                .checked_add(other.cpu)
                .ok_or("request CPU cost overflow")?,
        })
    }

    pub(super) fn admit(self, signals: MetalMemorySignals) -> std::result::Result<(), StateError> {
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            self.metal, self.cpu, 0, signals, true,
        );
        admission.refusal().map_or(Ok(()), |denied| {
            Err(StateError::MemoryAdmissionDenied(denied))
        })
    }
}

fn pressure(error: &StateError) -> bool {
    error.memory_admission().is_some_and(|denied| {
        matches!(
            denied.reason(),
            Reason::WorkingSetInsufficient
                | Reason::ProcessInsufficient
                | Reason::BothInsufficient
                | Reason::HostInsufficient
                | Reason::HostAvailabilityInsufficient
        )
    })
}

/// Pure candidate pricing/admission: no state, tensor allocation or encoding.
/// Preserve the resident heuristic as a preference, not a capacity promise.
pub(super) fn select(
    preferred: usize,
    fixed: Option<usize>,
    tokens: usize,
    base: Cost,
    signals: MetalMemorySignals,
    mut price: impl FnMut(usize) -> Result<Cost>,
) -> std::result::Result<(usize, Cost), StateError> {
    let first = fixed.unwrap_or(preferred);
    let candidates = [first, prefill::DEFAULT_CHUNK_ROWS];
    for (index, rows) in candidates.into_iter().enumerate() {
        let cost = base.plus(price(tokens.min(rows))?)?;
        match cost.admit(signals) {
            Ok(()) => return Ok((rows, cost)),
            Err(error)
                if index == 0
                    && fixed.is_none()
                    && first > prefill::DEFAULT_CHUNK_ROWS
                    && pressure(&error) => {}
            Err(error) => return Err(error),
        }
    }
    unreachable!("last candidate returns its result")
}

impl Forward<'_> {
    /// Price actual-tier scratch with the state's full capacity, not suffix
    /// length as a substitute for attention workspace extents.
    pub(super) fn prefill_cost(&self, capacity: usize, rows: usize) -> Result<Cost> {
        let (metal, cpu) = prefill::full_scratch_budget_with(
            &self.config,
            self.layers.len(),
            &gpu_step::engram_projections(self),
            capacity,
            rows,
            metal::moe_grouped_tensor_selected(self.ctx),
            &|bytes| {
                self.ctx
                    .price_shared_buffer_upper(bytes)
                    .map(|p| p.priced_upper_bytes)
                    .map_err(|e| e.to_string())
            },
        )?;
        Ok(Cost { metal, cpu })
    }

    pub fn plan_request(
        &self,
        capacity: usize,
        prefill_tokens: usize,
        caller_cpu_reserve: u64,
    ) -> std::result::Result<RequestPlan, StateError> {
        op::require(
            prefill_tokens <= capacity,
            "prefill exceeds requested capacity",
        )?;
        let mut memory = self.estimate_request_memory_with_gpu_step(capacity)?;
        let base = Cost {
            metal: memory.metal_bytes,
            cpu: memory
                .cpu_bytes
                .checked_add(caller_cpu_reserve)
                .ok_or("request reserve overflow")?,
        };
        let (selected_rows, cost) = select(
            self.prefill_rows,
            None,
            prefill_tokens,
            base,
            self.ctx.memory_signals(),
            |rows| self.prefill_cost(capacity, rows),
        )?;
        memory.cpu_transient_bytes = memory
            .cpu_transient_bytes
            .checked_add(cost.cpu - base.cpu)
            .ok_or("request CPU cost overflow")?;
        memory.cpu_bytes = cost.cpu - caller_cpu_reserve;
        memory.metal_bytes = cost.metal;
        memory.total_bytes = memory
            .cpu_bytes
            .checked_add(memory.metal_bytes)
            .ok_or("request cost overflow")?;
        Ok(RequestPlan {
            capacity,
            report: PrefillGeometry {
                start: 0,
                tokens: prefill_tokens,
                selected_rows,
            },
            memory,
            cpu_reserve: caller_cpu_reserve,
            identity: self.identity.clone(),
        })
    }

    pub fn new_state_from_plan(
        &self,
        plan: &RequestPlan,
    ) -> std::result::Result<State, StateError> {
        op::require(
            Arc::ptr_eq(&self.identity, &plan.identity),
            "plan belongs to another model",
        )?;
        Cost {
            metal: plan.memory.metal_bytes,
            cpu: plan
                .memory
                .cpu_bytes
                .checked_add(plan.cpu_reserve)
                .ok_or("request reserve overflow")?,
        }
        .admit(self.ctx.memory_signals())?;
        let mut state = self.new_state_with_gpu_step(plan.capacity)?;
        state.prefill_rows = Some(plan.report.selected_rows);
        state.caller_cpu_reserve = plan.cpu_reserve;
        Ok(state)
    }

    pub(super) fn select_prefill(
        &self,
        state: &State,
        tokens: usize,
        caller_cpu_reserve: u64,
        fixed_rows: Option<usize>,
    ) -> std::result::Result<PrefillGeometry, StateError> {
        let (selected_rows, _) = select(
            state.prefill_rows.unwrap_or(self.prefill_rows),
            fixed_rows,
            tokens,
            Cost {
                metal: 0,
                cpu: caller_cpu_reserve,
            },
            self.ctx.memory_signals(),
            |rows| self.prefill_cost(state.capacity(), rows),
        )?;
        Ok(PrefillGeometry {
            start: state.position(),
            tokens,
            selected_rows,
        })
    }
}
