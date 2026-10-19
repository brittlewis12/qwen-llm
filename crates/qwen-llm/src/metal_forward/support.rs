//! Validation and shared helpers.

use super::*;

pub(super) fn alloc_shape_error(detail: &'static str) -> MetalError {
    MetalError::BadShape {
        kernel: "session_alloc",
        detail: detail.into(),
    }
}

pub(crate) fn checked_u64_mul(a: u64, b: u64, detail: &'static str) -> Result<u64, MetalError> {
    a.checked_mul(b).ok_or_else(|| alloc_shape_error(detail))
}

pub(crate) fn checked_u64_add(a: u64, b: u64, detail: &'static str) -> Result<u64, MetalError> {
    a.checked_add(b).ok_or_else(|| alloc_shape_error(detail))
}

pub(crate) fn checked_u64_mul3(
    a: u64,
    b: u64,
    c: u64,
    detail: &'static str,
) -> Result<u64, MetalError> {
    checked_u64_mul(checked_u64_mul(a, b, detail)?, c, detail)
}

pub(crate) fn checked_u64_mul4(
    a: u64,
    b: u64,
    c: u64,
    d: u64,
    detail: &'static str,
) -> Result<u64, MetalError> {
    checked_u64_mul(checked_u64_mul3(a, b, c, detail)?, d, detail)
}

pub(crate) fn checked_u64_double(a: u64, detail: &'static str) -> Result<u64, MetalError> {
    checked_u64_mul(a, 2, detail)
}

pub(crate) fn checked_u64_div_exact(
    numerator: u64,
    denominator: u64,
    detail: &'static str,
) -> Result<u64, MetalError> {
    if denominator == 0 {
        return Err(alloc_shape_error(detail));
    }
    if !numerator.is_multiple_of(denominator) {
        return Err(alloc_shape_error(detail));
    }
    Ok(numerator / denominator)
}

pub(super) fn parse_native_quant_embedding_mode(value: Option<&str>) -> NativeQuantEmbeddingMode {
    match value {
        None => NativeQuantEmbeddingMode::Auto,
        Some(value) if crate::env_flag::env_value_truthy(value) => NativeQuantEmbeddingMode::Forced,
        Some(value) if crate::env_flag::env_value_falsy(value) => {
            NativeQuantEmbeddingMode::Disabled
        }
        Some(_) => NativeQuantEmbeddingMode::Invalid,
    }
}

pub(super) fn native_quant_embedding_mode() -> NativeQuantEmbeddingMode {
    static MODE: std::sync::OnceLock<NativeQuantEmbeddingMode> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("QWEN_NATIVE_QUANT_EMBED") {
            Ok(value) => parse_native_quant_embedding_mode(Some(&value)),
            Err(std::env::VarError::NotPresent) => NativeQuantEmbeddingMode::Auto,
            Err(std::env::VarError::NotUnicode(_)) => NativeQuantEmbeddingMode::Invalid,
        };
        if mode == NativeQuantEmbeddingMode::Invalid {
            // Not routed through `qwen_diag`: no plan script parses this
            // specific `[metal-load] invalid …` line (unlike sibling
            // `[metal-load]` load-time lines). The default `Full` formatter
            // gives operators the visible `WARN` badge they need to notice
            // a misconfigured environment variable.
            tracing::warn!(
                "[metal-load] invalid QWEN_NATIVE_QUANT_EMBED value; disabling native embeddings",
            );
        }
        mode
    })
}

pub(super) fn native_quant_embedding_supported(dtype: GgmlType, shape: &[u64]) -> bool {
    shape.len() == 2
        && shape[0] > 0
        && shape[1] > 0
        && ((matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K) && shape[0].is_multiple_of(256))
            || (dtype == GgmlType::IQ1_M
                && shape[0].is_multiple_of(256)
                && i32::try_from(shape[0]).is_ok()
                && i32::try_from(shape[1]).is_ok())
            || (matches!(dtype, GgmlType::Q8_0 | GgmlType::IQ4_NL) && shape[0].is_multiple_of(32)))
}

pub(super) fn native_quant_embedding_default_promoted(
    arch: &crate::model::Arch,
    tied_embeddings: bool,
    dtype: GgmlType,
    shape: &[u64],
) -> bool {
    if tied_embeddings || shape != [arch.hidden_size as u64, arch.vocab_size as u64] {
        return false;
    }
    let common = arch.vocab_size == 248_320
        && arch.full_attention_interval == 4
        && arch.attn_head_dim == 256
        && arch.rope_theta == 10_000_000.0
        && arch.partial_rotary_factor == 0.25
        && arch.gdn_n_k_heads == 16
        && arch.gdn_head_dim == 128
        && arch.gdn_conv_kernel == 4;
    common
        && ((matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K | GgmlType::IQ1_M)
            && arch.kind == ArchKind::Dense
            && arch.n_layer == 64
            && arch.hidden_size == 5120
            && arch.intermediate_size == 17408
            && arch.n_q_heads == 24
            && arch.n_kv_heads == 4
            && arch.gdn_n_v_heads == 48
            && arch.expert_count == 0
            && arch.expert_used_count == 0
            && arch.expert_feed_forward_length == 0
            && arch.expert_shared_feed_forward_length == 0)
            || (dtype == GgmlType::Q8_0
                && arch.kind == ArchKind::Moe
                && arch.n_layer == 40
                && arch.hidden_size == 2048
                && arch.intermediate_size == 0
                && arch.n_q_heads == 16
                && arch.n_kv_heads == 2
                && arch.gdn_n_v_heads == 32
                && arch.expert_count == 256
                && arch.expert_used_count == 8
                && arch.expert_feed_forward_length == 512
                && arch.expert_shared_feed_forward_length == 512))
}

pub(super) fn resolve_native_quant_embedding(
    mode: NativeQuantEmbeddingMode,
    supported: bool,
    promoted: bool,
) -> NativeQuantEmbeddingSelection {
    if !supported {
        return NativeQuantEmbeddingSelection::Unsupported;
    }
    match mode {
        NativeQuantEmbeddingMode::Auto if promoted => NativeQuantEmbeddingSelection::AutoPromoted,
        NativeQuantEmbeddingMode::Forced => NativeQuantEmbeddingSelection::Forced,
        NativeQuantEmbeddingMode::Auto => NativeQuantEmbeddingSelection::AutoUnpromoted,
        NativeQuantEmbeddingMode::Disabled => NativeQuantEmbeddingSelection::RollbackDisabled,
        NativeQuantEmbeddingMode::Invalid => NativeQuantEmbeddingSelection::InvalidDisabled,
    }
}

pub(super) fn auto_parallel_copy_enabled(
    admission_enabled: bool,
    parallel_mode: GgufParallelCopyMode,
    explicit_override_present: bool,
) -> bool {
    admission_enabled && parallel_mode == GgufParallelCopyMode::Auto && !explicit_override_present
}

pub(super) fn auto_parallel_copy_override_present(
    mut is_present: impl FnMut(&str) -> bool,
) -> bool {
    PARALLEL_COPY_AUTO_OVERRIDE_ENVS
        .iter()
        .copied()
        .any(&mut is_present)
}

pub(super) fn validate_parallel_copy_policy(
    parallel_mode: GgufParallelCopyMode,
    no_copy_mode: GgufNoCopyMode,
    prefault_present: bool,
    native_embedding_present: bool,
    router_f16: Option<&str>,
) -> Result<(), MfError> {
    if !parallel_mode.is_forced() {
        return Ok(());
    }
    if no_copy_mode == GgufNoCopyMode::Forced {
        return Err(MfError::LoadPolicy(
            "parallel copy and retained no-copy are mutually exclusive".to_string(),
        ));
    }
    if prefault_present {
        return Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY_PREFAULT is invalid with parallel copy".to_string(),
        ));
    }
    if native_embedding_present {
        return Err(MfError::LoadPolicy(
            "parallel copy requires production-auto native embedding selection".to_string(),
        ));
    }
    match router_f16 {
        None => Ok(()),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(()),
        Some(value) if crate::env_flag::env_value_truthy(value) => Err(MfError::LoadPolicy(
            "QWEN_MOE_ROUTER_F16 is invalid with parallel copy".to_string(),
        )),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_MOE_ROUTER_F16 value {value:?} with parallel copy"
        ))),
    }
}

pub(super) fn hash_layout_bytes(hash: &mut u64, bytes: &[u8]) {
    const PRIME: u64 = 0x100000001b3;
    for byte in bytes {
        *hash ^= u64::from(*byte);
        *hash = hash.wrapping_mul(PRIME);
    }
}

pub(super) fn hash_layout_u64(hash: &mut u64, value: u64) {
    hash_layout_bytes(hash, &value.to_le_bytes());
}

pub(super) fn emit_native_quant_embedding_policy(
    model: &Model<'_>,
    embedding_selection: NativeQuantEmbeddingSelection,
) {
    emit_metal_load_line(format_args!(
        "[metal-load] native quantized token embedding policy: {} ({:?} {:?})",
        embedding_selection.label(),
        model.token_embd.dtype,
        model.token_embd.shape,
    ));
}

pub(super) fn checked_tail_range(
    tensor: &MetalTensor,
    logical_bytes: usize,
    label: &str,
) -> Result<(usize, usize), MfError> {
    let start = usize::try_from(tensor.offset)
        .map_err(|_| lm_head_tail_error(format!("{label} offset does not fit usize")))?;
    let end = start
        .checked_add(logical_bytes)
        .ok_or_else(|| lm_head_tail_error(format!("{label} endpoint overflow")))?;
    if end > tensor.buffer.length() {
        return Err(lm_head_tail_error(format!(
            "{label} range [{start}..{end}) exceeds buffer length {}",
            tensor.buffer.length()
        )));
    }
    Ok((start, end))
}

pub(super) fn tail_ranges_overlap(
    left: &MetalTensor,
    left_range: (usize, usize),
    right: &MetalTensor,
    right_range: (usize, usize),
) -> bool {
    Retained::as_ptr(&left.buffer) == Retained::as_ptr(&right.buffer)
        && left_range.0 < right_range.1
        && right_range.0 < left_range.1
}

pub(super) fn parallel_copy_timeval_us(value: libc::timeval) -> Result<i64, MfError> {
    value
        .tv_sec
        .checked_mul(1_000_000)
        .and_then(|seconds| seconds.checked_add(i64::from(value.tv_usec)))
        .ok_or_else(|| MfError::LoadPolicy("parallel-copy CPU time overflow".to_string()))
}

pub(super) fn parallel_copy_i64_delta(
    after: i64,
    before: i64,
    label: &str,
) -> Result<u64, MfError> {
    let delta = after
        .checked_sub(before)
        .ok_or_else(|| MfError::LoadPolicy(format!("parallel-copy {label} counter underflow")))?;
    u64::try_from(delta)
        .map_err(|_| MfError::LoadPolicy(format!("parallel-copy {label} counter regressed")))
}

pub(super) fn parallel_copy_u64_delta(
    after: u64,
    before: u64,
    label: &str,
) -> Result<u64, MfError> {
    after
        .checked_sub(before)
        .ok_or_else(|| MfError::LoadPolicy(format!("parallel-copy {label} counter regressed")))
}

pub(super) fn finish_parallel_copy_accounting(
    before: ParallelCopyUsage,
    after: ParallelCopyUsage,
    proc_before: ParallelCopyProcUsage,
    proc_after: ParallelCopyProcUsage,
) -> Result<ParallelCopyEndpointAccounting, MfError> {
    let user_cpu_us = parallel_copy_i64_delta(after.user_time_us, before.user_time_us, "user CPU")?;
    let system_cpu_us =
        parallel_copy_i64_delta(after.system_time_us, before.system_time_us, "system CPU")?;
    let total_cpu_us = user_cpu_us
        .checked_add(system_cpu_us)
        .ok_or_else(|| MfError::LoadPolicy("parallel-copy total CPU overflow".to_string()))?;
    Ok(ParallelCopyEndpointAccounting {
        user_cpu_us,
        system_cpu_us,
        total_cpu_us,
        timer_minor_faults: parallel_copy_i64_delta(
            after.minor_faults,
            before.minor_faults,
            "minor fault",
        )?,
        timer_major_faults: parallel_copy_i64_delta(
            after.major_faults,
            before.major_faults,
            "major fault",
        )?,
        instructions_delta_raw: parallel_copy_u64_delta(
            proc_after.instructions,
            proc_before.instructions,
            "instruction",
        )?,
        cycles_delta_raw: parallel_copy_u64_delta(proc_after.cycles, proc_before.cycles, "cycle")?,
    })
}

pub(super) fn validate_parallel_population(
    _plan: &ParallelCopyLoadPlan,
    _population: ParallelPopulationMethod,
) -> Result<(), MfError> {
    Ok(())
}

pub(super) fn validate_parallel_destination_length(
    plan: &ParallelCopyLoadPlan,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<(), MfError> {
    validate_parallel_population(plan, population)?;
    if destination_length == ParallelDestinationLength::PageRounded
        && population != ParallelPopulationMethod::MmapCopy
    {
        return Err(MfError::LoadPolicy(
            "page-rounded destinations require mmap-copy population".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn parallel_destination_resource_length(
    logical_bytes: u64,
    destination_length: ParallelDestinationLength,
    page_size: usize,
    max_buffer_length: usize,
) -> Result<usize, MfError> {
    let logical = usize::try_from(logical_bytes).map_err(|_| {
        MfError::LoadPolicy("parallel-copy resource length does not fit usize".to_string())
    })?;
    if logical == 0 {
        return Err(MfError::LoadPolicy(
            "parallel-copy resource length is zero".to_string(),
        ));
    }
    let allocated = match destination_length {
        ParallelDestinationLength::LogicalExact => logical,
        ParallelDestinationLength::PageRounded => {
            if page_size == 0 {
                return Err(MfError::LoadPolicy("host page size is zero".to_string()));
            }
            logical
                .checked_add(page_size - 1)
                .map(|value| value / page_size * page_size)
                .ok_or_else(|| {
                    MfError::LoadPolicy("page-rounded resource length overflow".to_string())
                })?
        }
    };
    if allocated > max_buffer_length {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy resource length {allocated} exceeds max buffer {max_buffer_length}"
        )));
    }
    Ok(allocated)
}

pub(super) fn parallel_destination_accounting(
    expected: &[ModelWeightStorageIdentity],
    destination_length: ParallelDestinationLength,
    page_size: usize,
    max_buffer_length: usize,
) -> Result<(u64, usize), MfError> {
    let mut allocated_bytes = 0u64;
    let mut padded_resources = 0usize;
    for identity in expected {
        let allocated = parallel_destination_resource_length(
            identity.source_bytes,
            destination_length,
            page_size,
            max_buffer_length,
        )?;
        allocated_bytes = allocated_bytes
            .checked_add(allocated as u64)
            .ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy allocated byte overflow".to_string())
            })?;
        padded_resources += usize::from(allocated as u64 != identity.source_bytes);
    }
    Ok((allocated_bytes, padded_resources))
}

pub(super) fn parallel_copy_marker_label(
    _plan: &ParallelCopyLoadPlan,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<&'static str, MfError> {
    Ok(match (population, destination_length) {
        (ParallelPopulationMethod::MmapCopy, ParallelDestinationLength::LogicalExact) => {
            "[metal-gguf-parallel-copied]"
        }
        (ParallelPopulationMethod::Pread, ParallelDestinationLength::LogicalExact) => {
            "[metal-gguf-parallel-pread]"
        }
        (ParallelPopulationMethod::MmapCopy, ParallelDestinationLength::PageRounded) => {
            "[metal-gguf-parallel-page-rounded]"
        }
        (ParallelPopulationMethod::Pread, ParallelDestinationLength::PageRounded) => {
            return Err(MfError::LoadPolicy(
                "page-rounded pread is not supported".to_string(),
            ));
        }
    })
}

pub(super) fn emit_parallel_copy_marker(
    _ctx: &MetalContext,
    plan: &ParallelCopyLoadPlan,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
    timing: ParallelCopyTiming,
    accounting: Option<ParallelCopyEndpointAccounting>,
) -> Result<(), MfError> {
    let marker = parallel_copy_marker_label(plan, population, destination_length)?;
    let page_size = plan.page_size;
    let alignment = plan.alignment;
    let max_buffer_length = plan.max_buffer_length;
    let cuts = plan
        .cuts
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let tasks = plan
        .task_counts
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let worker_bytes = plan
        .worker_bytes
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let accounting = accounting.unwrap_or(ParallelCopyEndpointAccounting {
        user_cpu_us: 0,
        system_cpu_us: 0,
        total_cpu_us: 0,
        timer_minor_faults: 0,
        timer_major_faults: 0,
        instructions_delta_raw: 0,
        cycles_delta_raw: 0,
    });
    emit_metal_load_line(format_args!(
        concat!(
            "{} schema=3 plan={} resources={} source_bytes={} destination_bytes={} ",
            "padding_bytes={} padded_resources={} fallback_bytes={} workers={} cuts={} tasks={} ",
            "worker_bytes={} create=shared,default_cache,default observed=shared,default_cache,tracked ",
            "page={:?} alignment={} max_buffer={} allocation_us={} source_us={} copy_us={} ",
            "binding_us={} ready_us={} user_cpu_us={} system_cpu_us={} total_cpu_us={} ",
            "timer_minor_faults={} timer_major_faults={} instructions_delta_raw={} cycles_delta_raw={}"
        ),
        marker,
        "descriptor-plan-v1",
        plan.request_count,
        plan.source_bytes,
        plan.destination_bytes,
        plan.padding_bytes,
        plan.padded_resources,
        plan.fallback_bytes,
        plan.task_counts.len(),
        cuts,
        tasks,
        worker_bytes,
        page_size,
        alignment,
        max_buffer_length,
        timing.allocation_us,
        timing.source_us,
        timing.copy_us,
        timing.binding_us,
        timing.ready_us,
        accounting.user_cpu_us,
        accounting.system_cpu_us,
        accounting.total_cpu_us,
        accounting.timer_minor_faults,
        accounting.timer_major_faults,
        accounting.instructions_delta_raw,
        accounting.cycles_delta_raw,
    ));
    Ok(())
}

pub(super) fn frozen_parallel_copy_order(
    plan: &ParallelCopyLoadPlan,
    expected: &[ModelWeightStorageIdentity],
) -> Result<Vec<usize>, MfError> {
    if expected.len() != plan.request_count {
        return Err(MfError::LoadPolicy(
            "parallel-copy request count changed after planning".to_string(),
        ));
    }
    let schedule = plan_parallel_schedule(expected, plan.task_counts.len())?;
    if schedule.cuts != plan.cuts
        || schedule.worker_tasks != plan.task_counts
        || schedule.worker_bytes != plan.worker_bytes
        || expected.iter().try_fold(0u64, |total, identity| {
            total.checked_add(identity.source_bytes)
        }) != Some(plan.source_bytes)
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy load plan changed before realization".to_string(),
        ));
    }
    Ok(schedule.sorted_request_indices)
}

#[derive(Debug, Eq, PartialEq)]
pub(super) struct PlannedParallelSchedule {
    pub sorted_request_indices: Vec<usize>,
    pub cuts: Vec<usize>,
    pub worker_bytes: Vec<u64>,
    pub worker_tasks: Vec<usize>,
}

pub(super) fn plan_parallel_schedule(
    expected: &[ModelWeightStorageIdentity],
    worker_count: usize,
) -> Result<PlannedParallelSchedule, MfError> {
    if worker_count == 0 || expected.len() < worker_count {
        return Err(MfError::LoadPolicy(
            "parallel-copy plan needs at least one request per worker".to_string(),
        ));
    }
    if expected.iter().any(|identity| {
        identity.kind != ModelWeightStorageKind::Direct
            || identity.source_bytes == 0
            || identity.resident_bytes != identity.source_bytes
    }) {
        return Err(MfError::LoadPolicy(
            "parallel-copy plan requires nonempty direct requests".to_string(),
        ));
    }
    let mut sorted_request_indices = (0..expected.len()).collect::<Vec<_>>();
    sorted_request_indices.sort_by_key(|&index| {
        let identity = &expected[index];
        (identity.shard_idx, identity.data_offset, index)
    });
    let total_bytes = sorted_request_indices
        .iter()
        .try_fold(0u64, |total, &index| {
            total
                .checked_add(expected[index].source_bytes)
                .ok_or_else(|| {
                    MfError::LoadPolicy("parallel-copy plan byte count overflow".to_string())
                })
        })?;
    let mut cuts = Vec::with_capacity(worker_count - 1);
    let mut worker_bytes = Vec::with_capacity(worker_count);
    let mut worker_tasks = Vec::with_capacity(worker_count);
    let mut start = 0usize;
    let mut prefix_bytes = 0u64;
    for worker in 0..worker_count {
        let workers_left = worker_count - worker;
        if workers_left == 1 {
            let bytes = sorted_request_indices[start..]
                .iter()
                .try_fold(0u64, |sum, &index| {
                    sum.checked_add(expected[index].source_bytes)
                        .ok_or_else(|| {
                            MfError::LoadPolicy("parallel-copy partition byte overflow".to_string())
                        })
                })?;
            worker_bytes.push(bytes);
            worker_tasks.push(sorted_request_indices.len() - start);
            break;
        }
        let target = (u128::from(total_bytes) * (worker as u128 + 1) / worker_count as u128) as u64;
        let min_end = start + 1;
        let max_end = sorted_request_indices.len() - (workers_left - 1);
        let mut candidate_prefix = prefix_bytes;
        let mut best_end = min_end;
        let mut best_prefix = 0u64;
        let mut best_distance = u64::MAX;
        for end in min_end..=max_end {
            candidate_prefix = candidate_prefix
                .checked_add(expected[sorted_request_indices[end - 1]].source_bytes)
                .ok_or_else(|| MfError::LoadPolicy("parallel-copy prefix overflow".to_string()))?;
            let distance = candidate_prefix.abs_diff(target);
            if distance < best_distance {
                best_end = end;
                best_prefix = candidate_prefix;
                best_distance = distance;
            }
        }
        worker_bytes.push(best_prefix - prefix_bytes);
        worker_tasks.push(best_end - start);
        cuts.push(best_end);
        start = best_end;
        prefix_bytes = best_prefix;
    }
    Ok(PlannedParallelSchedule {
        sorted_request_indices,
        cuts,
        worker_bytes,
        worker_tasks,
    })
}

pub(super) unsafe fn exclusive_buffer_bytes_mut(
    buffer: &mut Buffer,
    logical_length: usize,
) -> &mut [u8] {
    // SAFETY: the caller proves this logical prefix is nonempty and within the
    // CPU-accessible resource, pairwise disjoint from every other destination,
    // disjoint from all immutable sources, and exclusively borrowed until the
    // slice dies. Any physical padding remains uninitialized and inaccessible.
    unsafe {
        std::slice::from_raw_parts_mut(buffer.contents().as_ptr().cast::<u8>(), logical_length)
    }
}

pub(super) fn build_parallel_copy_plan(
    ctx: &MetalContext,
    _gguf: &GgufFile,
    _model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    _embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<ParallelCopyLoadPlan, MfError> {
    if !ctx.device.hasUnifiedMemory() {
        return Err(MfError::LoadPolicy(
            "parallel population requires unified memory".to_string(),
        ));
    }
    let page_size = host_page_size_bytes()?;
    if page_size == 0 {
        return Err(MfError::LoadPolicy("host page size is zero".to_string()));
    }
    let identities = expected
        .iter()
        .map(expected_model_weight_identity)
        .collect::<Vec<_>>();
    if identities.iter().any(|identity| {
        usize::try_from(identity.source_bytes).map_or(true, |bytes| bytes > ctx.max_buffer_length())
    }) {
        return Err(MfError::LoadPolicy(
            "a parallel destination exceeds the device max buffer length".to_string(),
        ));
    }
    if identities.is_empty() {
        return Err(MfError::LoadPolicy(
            "parallel plan has no direct storage requests".to_string(),
        ));
    }
    let shard_lengths = _gguf.shard_mapped_lengths();
    let mut source_bytes = 0u64;
    let mut destination_bytes = 0u64;
    for request in expected {
        let Some(shard_length) = shard_lengths.get(request.desc.shard_idx) else {
            return Err(MfError::LoadPolicy(
                "parallel request refers to a missing GGUF shard".to_string(),
            ));
        };
        let source_end = request
            .desc
            .data_offset
            .checked_add(request.desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("parallel source range overflow".to_string()))?;
        if request.kind != ModelWeightStorageKind::Direct
            || request.desc.n_bytes == 0
            || request.resident_bytes != request.desc.n_bytes
            || source_end > *shard_length as u64
        {
            return Err(MfError::LoadPolicy(
                "parallel plan requires in-bounds direct tensors".to_string(),
            ));
        }
        let priced = ctx
            .shared_buffer_size_and_align(request.resident_bytes)
            .map_err(MfError::Metal)?;
        if priced.size > ctx.max_buffer_length() as u64 {
            return Err(MfError::LoadPolicy(format!(
                "parallel destination {} exceeds max buffer {}",
                priced.size,
                ctx.max_buffer_length()
            )));
        }
        source_bytes = source_bytes
            .checked_add(request.desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("parallel source byte overflow".to_string()))?;
        destination_bytes = destination_bytes
            .checked_add(priced.size)
            .ok_or_else(|| MfError::LoadPolicy("parallel destination byte overflow".to_string()))?;
    }
    let worker_count = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(PARALLEL_COPY_MAX_WORKERS)
        .min(identities.len());
    let schedule = plan_parallel_schedule(&identities, worker_count)?;
    Ok(ParallelCopyLoadPlan {
        request_count: identities.len(),
        source_bytes,
        page_size,
        destination_bytes,
        padding_bytes: 0,
        padded_resources: 0,
        fallback_bytes: 0,
        max_buffer_length: ctx.max_buffer_length(),
        alignment: GGUF_NO_COPY_ALIGNMENT,
        cuts: schedule.cuts,
        task_counts: schedule.worker_tasks,
        worker_bytes: schedule.worker_bytes,
    })
}

pub(super) fn parallel_copy_auto_host_supported(facts: &crate::metal::DeviceFacts) -> bool {
    facts.unified_memory
}

pub(super) enum AutoParallelCopyLoadPlanSelection {
    NoMatch,
    HostPlanUnsupported,
    Selected(ParallelCopyLoadPlan),
}

pub(super) fn select_auto_parallel_copy_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<AutoParallelCopyLoadPlanSelection, MfError> {
    if !parallel_copy_auto_host_supported(ctx.device_facts()) {
        return Ok(AutoParallelCopyLoadPlanSelection::NoMatch);
    }
    match build_parallel_copy_plan(ctx, gguf, model, expected, embedding_selection) {
        Ok(plan) => Ok(AutoParallelCopyLoadPlanSelection::Selected(plan)),
        Err(_) => Ok(AutoParallelCopyLoadPlanSelection::HostPlanUnsupported),
    }
}

pub(super) fn auto_parallel_copy_population() -> Option<ParallelPopulationMethod> {
    Some(ParallelPopulationMethod::Pread)
}

/// Helper: scatter `n` floats from `src[0..n]` into `dst[off..off+n]`.
/// Inverse of `copy_offset` (which gathers). Used to write into the KV
/// cache slot for the current position, and (via the metal_mtp module)
/// to assemble the `[e_normed, h_normed]` concat for the eh_proj input.
/// Copy `n_rows` contiguous F32 source rows of `row_len` elements into
/// `dst` rows at `dst_base + row * dst_stride` (v0.432). One dispatch
/// replaces a per-row `encode_scatter_offset_f32` loop in the DFlash
/// prefill hidden-capture tap (chunk_p dispatches per capture layer per
/// chunk -> 1).
#[allow(clippy::too_many_arguments)]
pub fn encode_copy_rows_dst_strided_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    n_rows: usize,
    row_len: usize,
    dst_stride: usize,
    dst_base: usize,
) -> Result<(), MetalError> {
    if n_rows == 0 || row_len == 0 {
        return Err(MetalError::BadShape {
            kernel: "copy_rows_dst_strided",
            detail: "n_rows/row_len must be nonzero".to_string(),
        });
    }
    let total = (n_rows * row_len) as u64;
    if src.n_elements() < total {
        return Err(MetalError::BadShape {
            kernel: "copy_rows_dst_strided",
            detail: format!("src has {} elements, needs >= {total}", src.n_elements()),
        });
    }
    let dst_need = dst_base as u64 + (n_rows as u64 - 1) * dst_stride as u64 + row_len as u64;
    if dst.n_elements() < dst_need {
        return Err(MetalError::BadShape {
            kernel: "copy_rows_dst_strided",
            detail: format!("dst has {} elements, needs >= {dst_need}", dst.n_elements()),
        });
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_rows: u32,
        row_len: u32,
        dst_stride: u32,
        dst_base: u32,
    }
    let pso = ctx.pipeline("kernel_copy_rows_dst_strided_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n_rows: n_rows as u32,
            row_len: row_len as u32,
            dst_stride: dst_stride as u32,
            dst_base: dst_base as u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);
    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: (total as usize).div_ceil(tg_threads),
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub fn encode_scatter_offset_f32(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    src: &MetalTensor,
    dst: &MetalTensor,
    dst_off: usize,
    n: usize,
) -> Result<(), MetalError> {
    if src.dtype != GgmlType::F32
        || dst.dtype != GgmlType::F32
        || !dst.is_writable()
        || src.n_elements() as usize != n
    {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: format!(
                "expected F32 src with n={n} and writable F32 dst, got src={:?}/{} dst={:?} writable={}",
                src.dtype,
                src.n_elements(),
                dst.dtype,
                dst.is_writable()
            ),
        });
    }
    let dst_end = dst_off.checked_add(n).ok_or_else(|| MetalError::BadShape {
        kernel: "scatter_offset",
        detail: format!("dst_off={dst_off} + n={n} overflows usize"),
    })?;
    if dst_end as u64 > dst.n_elements() {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: format!("dst_off+n={dst_end} > dst.n={}", dst.n_elements()),
        });
    }
    let element_bytes = std::mem::size_of::<f32>() as u64;
    let copy_bytes = (n as u64)
        .checked_mul(element_bytes)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "scatter_offset",
            detail: "copy byte count overflow".into(),
        })?;
    let dst_byte_offset = (dst_off as u64)
        .checked_mul(element_bytes)
        .and_then(|offset| dst.offset.checked_add(offset))
        .ok_or_else(|| MetalError::BadShape {
            kernel: "scatter_offset",
            detail: "destination byte offset overflow".into(),
        })?;
    let src_end = src
        .offset
        .checked_add(copy_bytes)
        .ok_or_else(|| MetalError::BadShape {
            kernel: "scatter_offset",
            detail: "source endpoint overflow".into(),
        })?;
    let dst_byte_end =
        dst_byte_offset
            .checked_add(copy_bytes)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "scatter_offset",
                detail: "destination endpoint overflow".into(),
            })?;
    if !src.offset.is_multiple_of(element_bytes)
        || !dst_byte_offset.is_multiple_of(element_bytes)
        || src_end > src.buffer.length() as u64
        || dst_byte_end > dst.buffer.length() as u64
        || capture_ranges_overlap(
            src,
            (src.offset, src_end),
            dst,
            (dst_byte_offset, dst_byte_end),
        )
    {
        return Err(MetalError::BadShape {
            kernel: "scatter_offset",
            detail: "copy ranges are unaligned, out of bounds, or overlapping".into(),
        });
    }
    let n_u32 = u32::try_from(n).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset",
        detail: format!("n={n} does not fit u32 kernel args"),
    })?;
    let dst_off_u32 = u32::try_from(dst_off).map_err(|_| MetalError::BadShape {
        kernel: "scatter_offset",
        detail: format!("dst_off={dst_off} does not fit u32 kernel args"),
    })?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        dst_off: u32,
    }
    let pso = ctx.pipeline("kernel_scatter_offset_f32")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(
        0,
        &Args {
            n: n_u32,
            dst_off: dst_off_u32,
        },
    );
    enc.set_tensor(1, src);
    enc.set_tensor(2, dst);

    let tg_threads = pso.maxTotalThreadsPerThreadgroup().min(1024);
    let n_tg = n.div_ceil(tg_threads);
    enc.dispatch(
        objc2_metal::MTLSize {
            width: n_tg,
            height: 1,
            depth: 1,
        },
        objc2_metal::MTLSize {
            width: tg_threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

impl DecodeStageRecorder {
    pub(super) fn new(ctx: &MetalContext, sample_count: usize) -> Result<Self, MfError> {
        Ok(Self {
            samples: ctx.timestamp_sample_buffer(sample_count)?,
            next_sample: 0,
            records: Vec::new(),
        })
    }

    pub(super) fn begin(
        &mut self,
        cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
        meta: DecodeStageMeta,
        concurrent: bool,
    ) -> Result<KernelEncoder, MfError> {
        let start_sample = self.next_sample;
        let end_sample = start_sample + 1;
        if end_sample >= self.samples.sample_count() {
            return Err(MfError::Metal(MetalError::Counter(format!(
                "decode stage timestamp buffer exhausted at sample {end_sample}"
            ))));
        }
        self.next_sample += 2;
        self.records.push(DecodeStageRecord {
            meta,
            concurrent,
            start_sample,
            end_sample,
        });
        Ok(KernelEncoder::begin_sampled(
            cmd,
            &self.samples,
            start_sample,
            end_sample,
            concurrent,
        ))
    }

    pub(super) fn resolve(
        self,
        ctx: &MetalContext,
        token: TokenProfile,
    ) -> Result<DecodeStageProfile, MfError> {
        let timestamps = ctx.resolve_timestamp_samples(&self.samples, self.next_sample)?;
        let sampled_span_ticks = match (self.records.first(), self.records.last()) {
            (Some(first), Some(last)) => {
                timestamps[last.end_sample].saturating_sub(timestamps[first.start_sample])
            }
            _ => 0,
        };
        let scale_ms_per_tick = if sampled_span_ticks > 0 {
            token.gpu_kernel_ms / sampled_span_ticks as f64
        } else {
            0.0
        };
        let stages = self
            .records
            .into_iter()
            .map(|record| {
                let start_timestamp = timestamps[record.start_sample];
                let end_timestamp = timestamps[record.end_sample];
                let duration_ticks = end_timestamp.saturating_sub(start_timestamp);
                let duration_ms_scaled = duration_ticks as f64 * scale_ms_per_tick;
                let fraction_of_gpu = if token.gpu_kernel_ms > 0.0 {
                    duration_ms_scaled / token.gpu_kernel_ms
                } else {
                    0.0
                };
                DecodeStageTiming {
                    family: record.meta.family.to_string(),
                    block_kind: record.meta.block_kind.to_string(),
                    block_index: record.meta.block_index,
                    local_index: record.meta.local_index,
                    concurrent: record.concurrent,
                    start_sample: record.start_sample,
                    end_sample: record.end_sample,
                    start_timestamp,
                    end_timestamp,
                    duration_ticks,
                    duration_ms_scaled,
                    fraction_of_gpu,
                }
            })
            .collect();
        let raw_span_ms_assuming_ns = sampled_span_ticks as f64 * 1e-6;
        let raw_coverage_assuming_ns = if token.gpu_kernel_ms > 0.0 {
            raw_span_ms_assuming_ns / token.gpu_kernel_ms
        } else {
            0.0
        };
        Ok(DecodeStageProfile {
            token,
            stages,
            sampled_span_ticks,
            raw_span_ms_assuming_ns,
            raw_coverage_assuming_ns,
        })
    }
}

pub(super) fn begin_decode_stage(
    cmd: &Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    recorder: Option<&mut DecodeStageRecorder>,
    meta: Option<DecodeStageMeta>,
    concurrent: bool,
) -> Result<KernelEncoder, MfError> {
    if let Some(meta) = &meta {
        // bench-only dispatch census label; no-op unless a census is active
        qwen_llm_dispatch_census_set_family(meta.family);
    }
    if let (Some(recorder), Some(meta)) = (recorder, meta) {
        recorder.begin(cmd, meta, concurrent)
    } else if concurrent {
        Ok(KernelEncoder::begin_concurrent(cmd))
    } else {
        Ok(KernelEncoder::begin(cmd))
    }
}

/// Copy raw bytes FROM a shared-storage MetalTensor's MTLBuffer INTO an
/// existing destination slice (single memcpy, no intermediate alloc).
/// Caller must ensure any prior GPU write is complete (i.e., the command
/// buffer that wrote this tensor was committed AND waitUntilCompleted'd).
pub(super) fn read_tensor_into(dst: &mut [u8], t: &MetalTensor) {
    // Always-on bounds check: this guards an unchecked memcpy. A debug-only
    // assert would let release builds silently read past the MTLBuffer end.
    let end = (dst.len() as u64)
        .checked_add(t.offset)
        .expect("read offset+len overflow");
    assert!(
        end <= t.buffer.length() as u64,
        "read OOB: offset={} + n={} > buffer.len={}",
        t.offset,
        dst.len(),
        t.buffer.length()
    );
    unsafe {
        let src = (t.buffer.contents().as_ptr() as *const u8).add(t.offset as usize);
        std::ptr::copy_nonoverlapping(src, dst.as_mut_ptr(), dst.len());
    }
}

/// Write raw bytes into a shared-storage MetalTensor's MTLBuffer at offset.
/// Caller must ensure any in-flight GPU read of this tensor has completed
/// before calling. Subsequent GPU work will see the written bytes.
pub(super) fn write_tensor_bytes(t: &MetalTensor, bytes: &[u8]) {
    // Always-on bounds check: see read_tensor_into above.
    let end = (bytes.len() as u64)
        .checked_add(t.offset)
        .expect("write offset+len overflow");
    assert!(
        end <= t.buffer.length() as u64,
        "write OOB: offset={} + n={} > buffer.len={}",
        t.offset,
        bytes.len(),
        t.buffer.length()
    );
    unsafe {
        let dst = (t.buffer.contents().as_ptr() as *mut u8).add(t.offset as usize);
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst, bytes.len());
    }
}
