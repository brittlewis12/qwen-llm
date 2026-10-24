//! Metadata-only cold request admission. This is a budget, not an OS reservation;
//! loader, state allocation and prefill admission remain independently required.
use super::*;
use crate::deepseek_v41::gguf::{self, LoadError, ResidentStoragePlan};
use crate::gguf::GgufFile;

#[derive(Clone, Debug)]
pub struct ColdPrefillPlan {
    pub full_chunk_rows: usize,
    pub resident_storage: ResidentStoragePlan,
    pub request: RequestMemoryEstimate,
    pub full_incremental_metal_bytes: u64,
    pub full_incremental_cpu_bytes: u64,
    pub ced: Option<CedMemoryEstimate>,
    pub runtime_metal_bytes: u64,
    pub runtime_cpu_bytes: u64,
    /// Conservative binding vectors, metadata copies and descriptor bookkeeping.
    pub binding_cpu_bytes: u64,
    /// Retained per-linear native controls, priced as separate Metal resources.
    /// Excludes shared resident weight views and per-request runtime buffers.
    pub binding_metal_bytes: u64,
    pub caller_cpu_bytes: u64,
    /// Full cold peak, including the resident/binding and runtime phases.
    pub required_metal_bytes: u64,
    pub required_cpu_bytes: u64,
    /// Resident views plus retained binding Metal resources.
    pub load_metal_bytes: u64,
    /// New full private shadows, binding CPU and caller CPU envelopes.
    pub load_cpu_bytes: u64,
    /// Runtime CPU peak plus the full binding and caller envelopes, even when
    /// some of those bytes already affect the observed host availability.
    pub runtime_gate_cpu_bytes: u64,
}

#[derive(Clone, Debug)]
pub struct ColdLoadAssessment {
    pub peak_capacity: metal::MetalMemoryCapacity,
    pub load_admission: metal::MetalMemoryAdmission,
    /// Diagnostic only: a future peak can fail today's availability snapshot.
    pub whole_peak_now: metal::MetalMemoryAdmission,
}

#[derive(Clone, Debug)]
pub struct ColdPrefillPreflight {
    pub plan: ColdPrefillPlan,
    pub admission: metal::MetalMemoryAdmission,
}

impl std::ops::Deref for ColdPrefillPreflight {
    type Target = ColdPrefillPlan;

    fn deref(&self) -> &Self::Target {
        &self.plan
    }
}

impl ColdPrefillPlan {
    /// Allow only the load/binding phase, after checking the complete future
    /// peak's capacity. Neither result reserves memory for a later phase.
    pub fn assess_load(
        &self,
        signals: metal::MetalMemorySignals,
    ) -> std::result::Result<ColdLoadAssessment, LoadError> {
        let peak_capacity = metal::evaluate_metal_memory_capacity_with_cpu_bytes(
            self.required_metal_bytes,
            self.required_cpu_bytes,
            signals,
        )?;
        let load_admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            self.load_metal_bytes,
            self.load_cpu_bytes,
            0,
            signals,
            true,
        );
        if let Some(denied) = load_admission.refusal() {
            return Err(denied.into());
        }
        Ok(ColdLoadAssessment {
            peak_capacity,
            load_admission,
            whole_peak_now: self.whole_peak_admission(signals),
        })
    }

    /// Refresh after binding and before each fresh state, once the previous
    /// state's resources and autoreleases have drained. Resident imports,
    /// binding buffers and private shadows must already be in the live signals.
    /// Normal state/prefill checks and production fallback remain required.
    pub fn assess_runtime(
        &self,
        signals: metal::MetalMemorySignals,
    ) -> std::result::Result<metal::MetalMemoryAdmission, LoadError> {
        let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
            self.runtime_metal_bytes,
            self.runtime_gate_cpu_bytes,
            0,
            signals,
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(denied.into());
        }
        Ok(admission)
    }

    fn whole_peak_admission(
        &self,
        signals: metal::MetalMemorySignals,
    ) -> metal::MetalMemoryAdmission {
        metal::evaluate_metal_memory_admission_with_cpu_bytes(
            self.required_metal_bytes,
            self.required_cpu_bytes,
            0,
            signals,
            true,
        )
    }

    fn into_strict(
        self,
        signals: metal::MetalMemorySignals,
    ) -> std::result::Result<ColdPrefillPreflight, LoadError> {
        let admission = self.whole_peak_admission(signals);
        if let Some(denied) = admission.refusal() {
            return Err(denied.into());
        }
        Ok(ColdPrefillPreflight {
            plan: self,
            admission,
        })
    }
}

/// Strictly admit the entire cold peak against current availability before
/// weight import. This deliberately retains the stronger, non-staged contract.
/// Policies (including repetitions) run sequentially, never as concurrent states.
/// `ced` is absent when not requested or when runtime would fall back to Full.
/// No payload is imported, no RSS execution observation is made, and no memory
/// is reserved. The opened GGUF already retains its original shared mappings.
pub fn preflight_cold_prefill(
    ctx: &MetalContext,
    gguf: &GgufFile,
    prompt_rows: usize,
    capacity: usize,
    executions: &[PrefillExecution],
    caller_cpu_bytes: u64,
) -> std::result::Result<ColdPrefillPreflight, LoadError> {
    plan_cold_prefill(
        ctx,
        gguf,
        prompt_rows,
        capacity,
        executions,
        caller_cpu_bytes,
    )?
    .into_strict(ctx.memory_signals())
}

/// Compute budgets without admitting any allocation or importing payloads.
/// Staged callers must assess load and then reassess runtime with fresh signals.
pub fn plan_cold_prefill(
    ctx: &MetalContext,
    gguf: &GgufFile,
    prompt_rows: usize,
    capacity: usize,
    executions: &[PrefillExecution],
    caller_cpu_bytes: u64,
) -> std::result::Result<ColdPrefillPlan, LoadError> {
    let resident_storage = gguf::resident_storage_plan(
        gguf,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
    )?;
    plan_with_pricing(
        gguf,
        resident_storage,
        prompt_rows,
        capacity,
        executions,
        caller_cpu_bytes,
        metal::moe_grouped_tensor_selected(ctx),
        &|bytes| {
            if bytes == 0 {
                return Ok(0);
            }
            ctx.price_shared_buffer_upper(bytes)
                .map(|p| p.priced_upper_bytes)
                .map_err(|e| e.to_string())
        },
    )
    .map_err(|error| error.with_signals(ctx.memory_signals()))
}

#[derive(Debug)]
enum PlanError {
    Load(LoadError),
    Overflow,
}

impl From<LoadError> for PlanError {
    fn from(error: LoadError) -> Self {
        Self::Load(error)
    }
}

impl PlanError {
    fn with_signals(self, signals: metal::MetalMemorySignals) -> LoadError {
        match self {
            Self::Load(error) => error,
            Self::Overflow => metal::MemoryAdmissionDenied::new(
                metal::MetalMemoryAdmissionReason::RequiredBytesOverflow,
                None,
                signals,
                signals.recommended_max_bytes.checked_sub(
                    signals
                        .effective_device_usage()
                        .unwrap_or(signals.current_allocated_bytes),
                ),
            )
            .expect("overflow is a refusing reason")
            .into(),
        }
    }
}

fn sum(values: &[u64]) -> Result<u64> {
    values.iter().try_fold(0u64, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| "cold prefill budget overflow".into())
    })
}

fn product(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b)
        .ok_or_else(|| "cold prefill budget overflow".into())
}

pub(super) fn runtime_from_parts(
    c: &Config,
    schedule: &gguf::Schedule,
    hash: &engram::HashConfig,
    projections: &[(usize, usize)],
    prompt_rows: usize,
    capacity: usize,
    executions: &[PrefillExecution],
    full_chunk_rows: usize,
    half_selected: bool,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<(RequestMemoryEstimate, (u64, u64), Option<CedMemoryEstimate>)> {
    schedule.validate().map_err(|e| e.to_string())?;
    hash.validate()
        .map_err(|e| format!("invalid Engram hash: {e:?}"))?;
    op::require(
        !executions.is_empty(),
        "empty prefill execution policy list",
    )?;
    op::require(
        prompt_rows > 0 && prompt_rows <= capacity,
        "invalid cold prompt/capacity",
    )?;
    op::require(
        capacity as u64 <= hash.max_position,
        "cold prefill exceeds model context",
    )?;
    op::require(
        hash.layers.iter().all(|h| h.layer < schedule.layers),
        "Engram layer outside schedule",
    )?;
    op::require(
        projections.len() == hash.layers.len(),
        "Engram projection/hash count mismatch",
    )?;
    let request = memory::estimate_gpu_step(c, capacity, schedule, hash.layers.len())?;
    op::require(
        (1..=prefill::MAX_CHUNK_ROWS).contains(&full_chunk_rows),
        "cold Full chunk geometry",
    )?;
    let rows = prompt_rows.min(full_chunk_rows);
    let full = prefill::full_scratch_budget_with(
        c,
        schedule.layers,
        projections,
        capacity,
        rows,
        half_selected,
        price,
    )?;
    let ced = if executions.contains(&PrefillExecution::ShrinkingCed) {
        match ced_plan::CedPlan::shrinking(
            c,
            schedule,
            hash,
            &ced_plan::CedRequest {
                full_chunk_rows,
                ..ced_plan::CedRequest::cold(prompt_rows, capacity)
            },
        ) {
            Ok(plan) => Some(ced::estimate_from_parts(
                c,
                schedule.layers,
                projections,
                hash.layers.len(),
                capacity,
                &plan,
                half_selected,
                price,
            )?),
            // These valid cold requests take Full in the runtime too.
            Err(
                ced_plan::CedPlanError::UnsupportedSchedule(_)
                | ced_plan::CedPlanError::DecoderOwner(_)
                | ced_plan::CedPlanError::DecoderEngram(_),
            ) => None,
            Err(error) => return Err(error.to_string()),
        }
    } else {
        None
    };
    Ok((request, full, ced))
}

#[cfg(test)]
pub(in crate::deepseek_v41) fn preflight_with_pricing(
    gguf: &GgufFile,
    resident_storage: ResidentStoragePlan,
    prompt_rows: usize,
    capacity: usize,
    executions: &[PrefillExecution],
    caller_cpu_bytes: u64,
    half_selected: bool,
    price: &impl Fn(u64) -> Result<u64>,
    signals: metal::MetalMemorySignals,
) -> std::result::Result<ColdPrefillPreflight, LoadError> {
    plan_with_pricing(
        gguf,
        resident_storage,
        prompt_rows,
        capacity,
        executions,
        caller_cpu_bytes,
        half_selected,
        price,
    )
    .map_err(|error| error.with_signals(signals))?
    .into_strict(signals)
}

fn plan_with_pricing(
    gguf: &GgufFile,
    resident_storage: ResidentStoragePlan,
    prompt_rows: usize,
    capacity: usize,
    executions: &[PrefillExecution],
    caller_cpu_bytes: u64,
    half_selected: bool,
    price: &impl Fn(u64) -> Result<u64>,
) -> std::result::Result<ColdPrefillPlan, PlanError> {
    let p = gguf::plan(gguf)?;
    let full_chunk_rows = super::prefill_rows_for_resident(p.bytes.resident_bytes());
    let calculate = || -> Result<_> {
        let output = p
            .config
            .hidden
            .checked_mul(5)
            .ok_or("Engram projection width overflow")?;
        // Plan validates every active Engram projection against this geometry.
        let projections = vec![(6144, output); p.engram_hash.layers.len()];
        let (request, full, ced) = runtime_from_parts(
            &p.config,
            &p.schedule,
            &p.engram_hash,
            &projections,
            prompt_rows,
            capacity,
            executions,
            full_chunk_rows,
            half_selected,
            price,
        )?;
        let metadata_bytes = serde_json::to_vec(gguf.model.metadata())
            .map_err(|e| e.to_string())?
            .len() as u64;
        let binding_cpu_bytes = sum(&[
            product(metadata_bytes, 8)?,
            product(p.config.vocab as u64, 32)?,
            binding_vectors_bytes(&p.config, p.schedule.layers)?,
            128 * 1024 * 1024,
            product(resident_storage.windows.len() as u64, 4096)?,
            product(gguf.tensors.len() as u64, 1024)?,
        ])?;
        let binding_metal_bytes = binding_metal_bytes(p.binding_linears.values().copied(), price)?;
        Ok((request, full, ced, binding_cpu_bytes, binding_metal_bytes))
    };
    let (request, full, ced, binding_cpu_bytes, binding_metal_bytes) =
        calculate().map_err(LoadError::Invalid)?;
    let mut combined = combine_plan(
        resident_storage,
        request,
        full,
        ced,
        binding_cpu_bytes,
        binding_metal_bytes,
        caller_cpu_bytes,
    )?;
    combined.full_chunk_rows = full_chunk_rows;
    Ok(combined)
}

fn binding_metal_bytes(
    dtypes: impl IntoIterator<Item = GgmlType>,
    price: &impl Fn(u64) -> Result<u64>,
) -> Result<u64> {
    dtypes.into_iter().try_fold(0, |total, dtype| {
        let bytes = binding::native_control_bytes(dtype)?;
        if bytes == 0 {
            return Ok(total);
        }
        // Each call allocates its own buffer; rounding the aggregate underprices it.
        sum(&[total, price(bytes)?])
    })
}

fn binding_vectors_bytes(c: &Config, layers: usize) -> Result<u64> {
    // Up to 64 F32 control/vector widths per layer, including separate Engram
    // gates and grouped projection descriptors. Matrices remain shared views.
    let width = [
        c.hidden,
        c.query_rank,
        c.head_dim,
        c.index_dim,
        c.experts,
        c.heads,
        c.groups,
        c.rope_dim,
    ]
    .into_iter()
    .max()
    .unwrap_or(0);
    product(layers as u64, product(width as u64, 64 * 4)?)
}

fn combine_plan(
    resident_storage: ResidentStoragePlan,
    request: RequestMemoryEstimate,
    full: (u64, u64),
    ced: Option<CedMemoryEstimate>,
    binding_cpu_bytes: u64,
    binding_metal_bytes: u64,
    caller_cpu_bytes: u64,
) -> std::result::Result<ColdPrefillPlan, PlanError> {
    let sum = |values: &[u64]| sum(values).map_err(|_| PlanError::Overflow);
    let runtime_metal_bytes = sum(&[
        request.metal_bytes,
        full.0
            .max(ced.as_ref().map_or(0, |c| c.incremental_metal_bytes)),
    ])?;
    let runtime_cpu_bytes = sum(&[
        request.cpu_bytes,
        full.1
            .max(ced.as_ref().map_or(0, |c| c.incremental_cpu_bytes)),
    ])?;
    let required_metal_bytes = sum(&[
        resident_storage.metal_view_bytes,
        binding_metal_bytes,
        runtime_metal_bytes,
    ])?;
    let required_cpu_bytes = sum(&[
        resident_storage.private_shadow_bytes,
        binding_cpu_bytes,
        runtime_cpu_bytes,
        caller_cpu_bytes,
    ])?;
    let load_metal_bytes = sum(&[resident_storage.metal_view_bytes, binding_metal_bytes])?;
    let load_cpu_bytes = sum(&[
        resident_storage.private_shadow_bytes,
        binding_cpu_bytes,
        caller_cpu_bytes,
    ])?;
    let runtime_gate_cpu_bytes = sum(&[runtime_cpu_bytes, binding_cpu_bytes, caller_cpu_bytes])?;
    // Check the combined runtime gate too: the individual CPU/Metal totals may
    // be representable while their sum is not.
    sum(&[runtime_metal_bytes, runtime_gate_cpu_bytes])?;
    sum(&[required_metal_bytes, required_cpu_bytes])?;
    Ok(ColdPrefillPlan {
        full_chunk_rows: prefill::DEFAULT_CHUNK_ROWS,
        resident_storage,
        request,
        full_incremental_metal_bytes: full.0,
        full_incremental_cpu_bytes: full.1,
        ced,
        runtime_metal_bytes,
        runtime_cpu_bytes,
        binding_cpu_bytes,
        binding_metal_bytes,
        caller_cpu_bytes,
        required_metal_bytes,
        required_cpu_bytes,
        load_metal_bytes,
        load_cpu_bytes,
        runtime_gate_cpu_bytes,
    })
}

#[cfg(test)]
mod tests;
