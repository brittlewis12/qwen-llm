//! Metadata-only cold request admission. This is a budget, not an OS reservation;
//! loader, state allocation and prefill admission remain independently required.
use super::*;
use crate::deepseek_v41::gguf::{self, LoadError, ResidentStoragePlan};
use crate::gguf::GgufFile;

#[derive(Clone, Debug)]
pub struct ColdPrefillPreflight {
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
    pub required_metal_bytes: u64,
    pub required_cpu_bytes: u64,
    pub admission: metal::MetalMemoryAdmission,
}

/// Admit one cold sequence before weight import, using the actual GGUF geometry.
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
    let resident_storage = gguf::resident_storage_plan(
        gguf,
        metal::host_page_size_bytes()?,
        ctx.max_buffer_length(),
    )?;
    preflight_with_pricing(
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
        ctx.memory_signals(),
    )
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
    let rows = prompt_rows.min(prefill::DEFAULT_CHUNK_ROWS);
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
            &ced_plan::CedRequest::cold(prompt_rows, capacity),
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
    let p = gguf::plan(gguf)?;
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
    let result = combine(
        resident_storage,
        request,
        full,
        ced,
        binding_cpu_bytes,
        binding_metal_bytes,
        caller_cpu_bytes,
        signals,
    )?;
    if let Some(reason) = result.admission.refusal() {
        return Err(reason.into());
    }
    Ok(result)
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

fn combine(
    resident_storage: ResidentStoragePlan,
    request: RequestMemoryEstimate,
    full: (u64, u64),
    ced: Option<CedMemoryEstimate>,
    binding_cpu_bytes: u64,
    binding_metal_bytes: u64,
    caller_cpu_bytes: u64,
    signals: metal::MetalMemorySignals,
) -> std::result::Result<ColdPrefillPreflight, LoadError> {
    let sum = |values: &[u64]| {
        sum(values).map_err(|_| {
            LoadError::MemoryAdmissionDenied(
                metal::MemoryAdmissionDenied::new(
                    metal::MetalMemoryAdmissionReason::RequiredBytesOverflow,
                    None,
                    signals,
                    signals
                        .recommended_max_bytes
                        .checked_sub(signals.current_allocated_bytes),
                )
                .expect("overflow is a refusing reason"),
            )
        })
    };
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
    let admission = metal::evaluate_metal_memory_admission_with_cpu_bytes(
        required_metal_bytes,
        required_cpu_bytes,
        0,
        signals,
        true,
    );
    Ok(ColdPrefillPreflight {
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
        admission,
    })
}

#[cfg(test)]
mod tests;
