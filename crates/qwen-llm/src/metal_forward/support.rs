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
            // Not routed through `qwen_diag`: no profile script parses this
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
        && ((matches!(dtype, GgmlType::Q4_K | GgmlType::Q6_K)
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

pub(super) fn auto_parallel_copy_a3b_enabled(
    admission_enabled: bool,
    parallel_mode: GgufParallelCopyMode,
    explicit_override_present: bool,
) -> bool {
    admission_enabled && parallel_mode == GgufParallelCopyMode::Auto && !explicit_override_present
}

pub(super) fn auto_parallel_copy_a3b_override_present(
    mut is_present: impl FnMut(&str) -> bool,
) -> bool {
    A3B_PARALLEL_COPY_AUTO_OVERRIDE_ENVS
        .iter()
        .copied()
        .any(&mut is_present)
}

pub(super) fn validate_parallel_copy_policy(
    parallel_mode: GgufParallelCopyMode,
    no_copy_mode: GgufNoCopyMode,
    owned_mode: GgufOwnedArenaMode,
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
    if owned_mode == GgufOwnedArenaMode::Forced {
        return Err(MfError::LoadPolicy(
            "parallel copy and owned arena are mutually exclusive".to_string(),
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
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
) -> Result<(), MfError> {
    if profile.id == ParallelCopyProfileId::A10bQ4xlV1
        && population != ParallelPopulationMethod::Pread
    {
        return Err(MfError::LoadPolicy(
            "A10B parallel population requires direct pread".to_string(),
        ));
    }
    if population == ParallelPopulationMethod::Pread && !profile.supports_direct_pread {
        return Err(MfError::LoadPolicy(format!(
            "parallel pread rejects unauthenticated profile {}",
            profile.id.label()
        )));
    }
    Ok(())
}

pub(super) fn validate_parallel_destination_length(
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<(), MfError> {
    validate_parallel_population(profile, population)?;
    if destination_length == ParallelDestinationLength::PageRounded16K
        && (profile.id != ParallelCopyProfileId::A3bQ4kmV1
            || population != ParallelPopulationMethod::MmapCopy)
    {
        return Err(MfError::LoadPolicy(
            "page-rounded parallel copy requires authenticated A3B mmap population".to_string(),
        ));
    }
    Ok(())
}

pub(super) fn parallel_destination_resource_length(
    logical_bytes: u64,
    destination_length: ParallelDestinationLength,
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
        ParallelDestinationLength::PageRounded16K => logical
            .checked_add(16_383)
            .map(|value| value & !16_383)
            .ok_or_else(|| {
                MfError::LoadPolicy("page-rounded resource length overflow".to_string())
            })?,
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
    max_buffer_length: usize,
) -> Result<(u64, usize), MfError> {
    let mut allocated_bytes = 0u64;
    let mut padded_resources = 0usize;
    for identity in expected {
        let allocated = parallel_destination_resource_length(
            identity.source_bytes,
            destination_length,
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
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<&'static str, MfError> {
    validate_parallel_destination_length(profile, population, destination_length)?;
    Ok(match (population, destination_length) {
        (ParallelPopulationMethod::MmapCopy, ParallelDestinationLength::LogicalExact) => {
            "[metal-gguf-parallel-copied]"
        }
        (ParallelPopulationMethod::Pread, ParallelDestinationLength::LogicalExact) => {
            "[metal-gguf-parallel-pread]"
        }
        (ParallelPopulationMethod::MmapCopy, ParallelDestinationLength::PageRounded16K) => {
            "[metal-gguf-parallel-page-rounded]"
        }
        (ParallelPopulationMethod::Pread, ParallelDestinationLength::PageRounded16K) => {
            unreachable!("page-rounded pread is rejected above")
        }
    })
}

pub(super) fn emit_parallel_copy_marker(
    profile: &ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
    timing: ParallelCopyTiming,
    accounting: Option<ParallelCopyEndpointAccounting>,
) -> Result<(), MfError> {
    let marker = parallel_copy_marker_label(profile, population, destination_length)?;
    match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => {
            if accounting.is_some() {
                return Err(MfError::LoadPolicy(
                    "A3B parallel-copy marker received dense accounting".to_string(),
                ));
            }
            match destination_length {
                ParallelDestinationLength::LogicalExact => {
                    emit_metal_load_line(format_args!(
                        concat!(
                            "{} schema=1 resources=733 bytes=22123538944 ",
                            "workers=4 cuts=155,359,539 tasks=155,204,180,194 ",
                            "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
                            "first_offsets=10990048,5543736288,11006052064,16601574368 ",
                            "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
                            "create=shared,default_cache,default observed=shared,default_cache,tracked ",
                            "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
                            "layout=0x5ae645df5cf7d568 ",
                            "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
                            "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af ",
                            "allocation_us={} source_us={} copy_us={} binding_us={} ready_us={}"
                        ),
                        marker,
                        timing.allocation_us,
                        timing.source_us,
                        timing.copy_us,
                        timing.binding_us,
                        timing.ready_us,
                    ));
                }
                ParallelDestinationLength::PageRounded16K => {
                    emit_metal_load_line(format_args!(
                        concat!(
                            "{} schema=2 resources=733 logical_bytes=22123538944 ",
                            "allocated_bytes=22126297088 padding_bytes=2758144 ",
                            "padded_resources=232 workers=4 cuts=155,359,539 ",
                            "tasks=155,204,180,194 ",
                            "worker_bytes=5532746240,5462315776,5595522304,5532954624 ",
                            "first_offsets=10990048,5543736288,11006052064,16601574368 ",
                            "last_offsets=5392741344,11004937952,16450579424,22134520800 ",
                            "create=shared,default_cache,default observed=shared,default_cache,tracked ",
                            "page=16384 alignment=32 max_buffer=77309411328 mapped=22134528992 ",
                            "layout=0x5ae645df5cf7d568 ",
                            "inventory=f57153febec22463c7789b892d4d084041d722483a93191c81c40ab86be7d9e5 ",
                            "plan=fa2685e223ad8ea6271c6061041fe8d996b4e6cc70e060588b750732577c92af ",
                            "allocation_us={} source_us={} copy_us={} binding_us={} ready_us={}"
                        ),
                        marker,
                        timing.allocation_us,
                        timing.source_us,
                        timing.copy_us,
                        timing.binding_us,
                        timing.ready_us,
                    ));
                }
            }
        }
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            let accounting = accounting.ok_or_else(|| {
                MfError::LoadPolicy(
                    "profile parallel-copy marker is missing endpoint accounting".to_string(),
                )
            })?;
            let mapped_bytes =
                profile
                    .shard_mapped_lengths
                    .iter()
                    .try_fold(0u64, |total, &length| {
                        total.checked_add(length as u64).ok_or_else(|| {
                            MfError::LoadPolicy("parallel-copy mapped byte overflow".to_string())
                        })
                    })?;
            let boundary = profile.boundaries;
            emit_metal_load_line(format_args!(
                concat!(
                    "{} schema=2 profile={} ",
                    "resources={} bytes={} workers=4 cuts={},{},{} ",
                    "tasks={},{},{},{} worker_bytes={},{},{},{} ",
                    "w0_first={},{},{},{},{} w0_last={},{},{},{},{} ",
                    "w1_first={},{},{},{},{} w1_last={},{},{},{},{} ",
                    "w2_first={},{},{},{},{} w2_last={},{},{},{},{} ",
                    "w3_first={},{},{},{},{} w3_last={},{},{},{},{} ",
                    "create=shared,default_cache,default ",
                    "observed=shared,default_cache,tracked ",
                    "page=16384 alignment=32 max_buffer=77309411328 ",
                    "mapped={} layout={:#018x} ",
                    "inventory={} allocation_us={} source_us={} copy_us={} ",
                    "binding_us={} ready_us={} user_cpu_us={} system_cpu_us={} ",
                    "total_cpu_us={} timer_minor_faults={} timer_major_faults={} ",
                    "instructions_delta_raw={} cycles_delta_raw={}"
                ),
                marker,
                profile.id.label(),
                profile.request_count,
                profile.source_bytes,
                profile.cuts[0],
                profile.cuts[1],
                profile.cuts[2],
                profile.task_counts[0],
                profile.task_counts[1],
                profile.task_counts[2],
                profile.task_counts[3],
                profile.worker_bytes[0],
                profile.worker_bytes[1],
                profile.worker_bytes[2],
                profile.worker_bytes[3],
                boundary[0].first.request_index,
                boundary[0].first.name,
                boundary[0].first.shard_idx,
                boundary[0].first.source_offset,
                boundary[0].first.source_bytes,
                boundary[0].last.request_index,
                boundary[0].last.name,
                boundary[0].last.shard_idx,
                boundary[0].last.source_offset,
                boundary[0].last.source_bytes,
                boundary[1].first.request_index,
                boundary[1].first.name,
                boundary[1].first.shard_idx,
                boundary[1].first.source_offset,
                boundary[1].first.source_bytes,
                boundary[1].last.request_index,
                boundary[1].last.name,
                boundary[1].last.shard_idx,
                boundary[1].last.source_offset,
                boundary[1].last.source_bytes,
                boundary[2].first.request_index,
                boundary[2].first.name,
                boundary[2].first.shard_idx,
                boundary[2].first.source_offset,
                boundary[2].first.source_bytes,
                boundary[2].last.request_index,
                boundary[2].last.name,
                boundary[2].last.shard_idx,
                boundary[2].last.source_offset,
                boundary[2].last.source_bytes,
                boundary[3].first.request_index,
                boundary[3].first.name,
                boundary[3].first.shard_idx,
                boundary[3].first.source_offset,
                boundary[3].first.source_bytes,
                boundary[3].last.request_index,
                boundary[3].last.name,
                boundary[3].last.shard_idx,
                boundary[3].last.source_offset,
                boundary[3].last.source_bytes,
                mapped_bytes,
                profile.descriptor_layout_digest,
                profile.inventory_digest,
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
        }
    }
    Ok(())
}

pub(super) fn frozen_parallel_copy_order(
    profile: &ParallelCopyProfile,
    expected: &[ModelWeightStorageIdentity],
) -> Result<Vec<usize>, MfError> {
    if expected.len() != profile.request_count
        || expected.iter().any(|identity| {
            identity.kind != ModelWeightStorageKind::Direct
                || identity.source_bytes == 0
                || identity.resident_bytes != identity.source_bytes
        })
    {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy {} request inventory is not all-direct and nonempty",
            profile.id.label()
        )));
    }

    let mut sorted_request_indices = (0..expected.len()).collect::<Vec<_>>();
    sorted_request_indices.sort_by_key(|&request_index| {
        let identity = &expected[request_index];
        (identity.shard_idx, identity.data_offset, request_index)
    });
    let mut permutation_check = sorted_request_indices.clone();
    permutation_check.sort_unstable();
    if permutation_check.iter().copied().ne(0..expected.len()) {
        return Err(MfError::LoadPolicy(
            "parallel-copy schedule is not a complete permutation".to_string(),
        ));
    }

    let boundaries = [
        0,
        profile.cuts[0],
        profile.cuts[1],
        profile.cuts[2],
        expected.len(),
    ];
    if boundaries[0] != 0
        || boundaries[GGUF_OWNED_WORKERS] != expected.len()
        || boundaries.windows(2).any(|pair| pair[0] >= pair[1])
        || boundaries
            .windows(2)
            .map(|pair| pair[1] - pair[0])
            .ne(profile.task_counts)
        || profile.task_counts.iter().sum::<usize>() != expected.len()
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy frozen boundaries drifted".to_string(),
        ));
    }
    let mut total_bytes = 0u64;
    for worker in 0..GGUF_OWNED_WORKERS {
        let partition = &sorted_request_indices[boundaries[worker]..boundaries[worker + 1]];
        let worker_bytes = partition.iter().try_fold(0u64, |total, &request_index| {
            total
                .checked_add(expected[request_index].source_bytes)
                .ok_or_else(|| {
                    MfError::LoadPolicy("parallel-copy partition byte overflow".to_string())
                })
        })?;
        let first = &expected[partition[0]];
        let first_index = partition[0];
        let last_index = *partition.last().expect("partition is nonempty");
        let last = &expected[last_index];
        let frozen = profile.boundaries[worker];
        let first_matches = first_index == frozen.first.request_index
            && first.name == frozen.first.name
            && first.shard_idx == frozen.first.shard_idx
            && first.data_offset == frozen.first.source_offset
            && first.source_bytes == frozen.first.source_bytes;
        let last_matches = last_index == frozen.last.request_index
            && last.name == frozen.last.name
            && last.shard_idx == frozen.last.shard_idx
            && last.data_offset == frozen.last.source_offset
            && last.source_bytes == frozen.last.source_bytes;
        if partition.len() != profile.task_counts[worker]
            || worker_bytes != profile.worker_bytes[worker]
            || !first_matches
            || !last_matches
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy frozen partition {worker} drifted"
            )));
        }
        total_bytes = total_bytes.checked_add(worker_bytes).ok_or_else(|| {
            MfError::LoadPolicy("parallel-copy schedule byte overflow".to_string())
        })?;
    }
    if total_bytes != profile.source_bytes {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy schedule bytes drifted: {total_bytes}"
        )));
    }
    Ok(sorted_request_indices)
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

pub(super) fn owned_arena_four_worker_boundaries(
    length: usize,
    page_size: usize,
) -> Result<[usize; 5], MfError> {
    if page_size == 0 || !length.is_multiple_of(page_size) {
        return Err(MfError::LoadPolicy(
            "owned arena range is not page aligned".to_string(),
        ));
    }
    let pages = length / page_size;
    if pages < GGUF_OWNED_WORKERS {
        return Err(MfError::LoadPolicy(
            "owned arena range has fewer than four pages".to_string(),
        ));
    }
    let boundaries =
        std::array::from_fn(|worker| page_size * (worker * pages / GGUF_OWNED_WORKERS));
    if boundaries[0] != 0 || boundaries[GGUF_OWNED_WORKERS] != length {
        return Err(MfError::LoadPolicy(
            "owned arena worker boundaries do not cover the range".to_string(),
        ));
    }
    if boundaries.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MfError::LoadPolicy(
            "owned arena worker boundaries overlap or are empty".to_string(),
        ));
    }
    Ok(boundaries)
}

pub(super) fn copy_owned_arena_four_workers(
    source: &[u8],
    destination: &Buffer,
    page_size: usize,
) -> Result<(), MfError> {
    if source.len() != destination.length() {
        return Err(MfError::LoadPolicy(format!(
            "owned arena source {} differs from destination {}",
            source.len(),
            destination.length()
        )));
    }
    let boundaries = owned_arena_four_worker_boundaries(source.len(), page_size)?;
    // SAFETY: the anonymous shared buffer is retained by the caller, has exactly
    // source.len() bytes, and no typed views exist until all scoped workers join.
    let destination = unsafe {
        std::slice::from_raw_parts_mut(destination.contents().as_ptr().cast::<u8>(), source.len())
    };
    std::thread::scope(|scope| {
        let mut source_tail = source;
        let mut destination_tail = destination;
        let mut previous = 0usize;
        for &boundary in boundaries.iter().skip(1) {
            let length = boundary - previous;
            let (source_chunk, next_source) = source_tail.split_at(length);
            let (destination_chunk, next_destination) = destination_tail.split_at_mut(length);
            scope.spawn(move || destination_chunk.copy_from_slice(source_chunk));
            source_tail = next_source;
            destination_tail = next_destination;
            previous = boundary;
        }
    });
    Ok(())
}

pub(super) fn copy_owned_arena_serial(source: &[u8], destination: &Buffer) -> Result<(), MfError> {
    if source.len() != destination.length() {
        return Err(MfError::LoadPolicy(format!(
            "owned fallback source {} differs from destination {}",
            source.len(),
            destination.length()
        )));
    }
    // SAFETY: the fallback buffer is retained, exact-sized, and has no live views.
    let destination = unsafe {
        std::slice::from_raw_parts_mut(destination.contents().as_ptr().cast::<u8>(), source.len())
    };
    destination.copy_from_slice(source);
    Ok(())
}

pub(super) fn create_a10b_parallel_residency_set(
    ctx: &MetalContext,
    direct_storage: &DirectStorage,
) -> Result<Option<MetalModelResidencySetGuard>, MfError> {
    let DirectStorage::ForcedParallelCopied(storage) = direct_storage else {
        return Ok(None);
    };
    if storage.profile.id != ParallelCopyProfileId::A10bQ4xlV1 {
        return Ok(None);
    }
    if storage.resources.len() != storage.profile.request_count {
        return Err(MfError::LoadPolicy(format!(
            "A10B residency resource count drifted: {}/{}",
            storage.resources.len(),
            storage.profile.request_count
        )));
    }

    let descriptor = MTLResidencySetDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str("qwen-a10b-parallel-pread")));
    // SAFETY: initialCapacity is advisory and equals the authenticated resource count.
    unsafe { descriptor.setInitialCapacity(storage.resources.len()) };
    let set = ctx
        .device
        .newResidencySetWithDescriptor_error(&descriptor)
        .map_err(|error| {
            let error: Retained<NSError> = error;
            MfError::LoadPolicy(format!(
                "A10B residency set creation failed: {}",
                error.localizedDescription()
            ))
        })?;
    let mut expected_allocated_bytes = 0u64;
    for buffer in &storage.resources {
        let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(&**buffer);
        let allocated_bytes = u64::try_from(allocation.allocatedSize()).map_err(|_| {
            MfError::LoadPolicy("A10B residency allocation size does not fit u64".to_string())
        })?;
        expected_allocated_bytes = expected_allocated_bytes
            .checked_add(allocated_bytes)
            .ok_or_else(|| {
                MfError::LoadPolicy("A10B residency allocated byte overflow".to_string())
            })?;
        set.addAllocation(allocation);
    }
    set.commit();
    if set.allocationCount() != storage.resources.len()
        || set.allocatedSize() != expected_allocated_bytes
    {
        let actual_count = set.allocationCount();
        let actual_bytes = set.allocatedSize();
        set.endResidency();
        return Err(MfError::LoadPolicy(format!(
            concat!(
                "A10B residency set commitment drifted: allocations={}/{} ",
                "bytes={}/{}"
            ),
            actual_count,
            storage.resources.len(),
            actual_bytes,
            expected_allocated_bytes,
        )));
    }
    let started = std::time::Instant::now();
    set.requestResidency();
    let residency_ms = started.elapsed().as_secs_f64() * 1e3;
    ctx.queue.addResidencySet(&set);
    emit_metal_load_line(format_args!(
        concat!(
            "[metal-gguf-parallel-residency] profile={} allocations={} ",
            "allocated_bytes={} request_ms={:.3} rollback=QWEN_GGUF_PARALLEL_COPY=0"
        ),
        storage.profile.id.label(),
        set.allocationCount(),
        set.allocatedSize(),
        residency_ms,
    ));
    Ok(Some(MetalModelResidencySetGuard {
        queue: ctx.queue.clone(),
        set,
    }))
}

pub(super) fn matches_no_copy_27b_sentinel(gguf: &GgufFile, model: &Model<'_>) -> bool {
    gguf.shard_count() == 1
        && gguf.total_mapped_len() == GGUF_NO_COPY_27B_MAPPED_BYTES
        && gguf.tensors.len() == GGUF_NO_COPY_27B_DESCRIPTOR_COUNT
        && gguf_descriptor_layout_digest(gguf) == GGUF_NO_COPY_27B_LAYOUT_DIGEST
        && !model.tied_embeddings
        && model.mtp.is_none()
        && native_quant_embedding_default_promoted(
            &model.arch,
            model.tied_embeddings,
            model.token_embd.dtype,
            &model.token_embd.shape,
        )
}

pub(super) fn matches_owned_a3b_arch(model: &Model<'_>) -> bool {
    let arch = model.arch;
    arch.kind == ArchKind::Moe
        && arch.n_layer == 40
        && arch.hidden_size == 2048
        && arch.intermediate_size == 0
        && arch.vocab_size == 248_320
        && arch.full_attention_interval == 4
        && arch.n_q_heads == 16
        && arch.n_kv_heads == 2
        && arch.attn_head_dim == 256
        && arch.rope_theta == 10_000_000.0
        && arch.partial_rotary_factor == 0.25
        && arch.gdn_n_v_heads == 32
        && arch.gdn_n_k_heads == 16
        && arch.gdn_head_dim == 128
        && arch.gdn_conv_kernel == 4
        && arch.expert_count == 256
        && arch.expert_used_count == 8
        && arch.expert_feed_forward_length == 512
        && arch.expert_shared_feed_forward_length == 512
        && arch.mtp_n_hidden_layers == 0
}

pub(super) fn select_unique_parallel_copy_profile<F>(
    profiles: &[&'static ParallelCopyProfile],
    mut matches: F,
) -> Result<&'static ParallelCopyProfile, MfError>
where
    F: FnMut(&ParallelCopyProfile) -> Result<bool, MfError>,
{
    let mut ids = HashSet::with_capacity(profiles.len());
    for profile in profiles {
        if !ids.insert(profile.id) {
            return Err(MfError::LoadPolicy(format!(
                "duplicate parallel-copy profile id {}",
                profile.id.label()
            )));
        }
    }
    let mut selected = None;
    for profile in profiles {
        if !matches(profile)? {
            continue;
        }
        if selected.is_some() {
            return Err(MfError::LoadPolicy(
                "parallel-copy profile match is ambiguous".to_string(),
            ));
        }
        selected = Some(*profile);
    }
    selected.ok_or_else(|| {
        MfError::LoadPolicy("forced parallel copy rejects unsupported geometry".to_string())
    })
}

pub(super) fn parallel_copy_profile_matches(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    profile: &ParallelCopyProfile,
) -> Result<bool, MfError> {
    if !ctx.device.hasUnifiedMemory() {
        return Ok(false);
    }
    if let ParallelCopyDeviceConstraint::ExactUnified(name) = profile.device_constraint
        && ctx.device.name().to_string() != name
    {
        return Ok(false);
    }
    let architecture = gguf.architecture();
    if profile
        .architecture_label
        .is_some_and(|expected| architecture.as_deref() != Some(expected))
    {
        return Ok(false);
    }
    let shard_lengths = gguf.shard_mapped_lengths();
    let embedding_qualified = match profile.id {
        ParallelCopyProfileId::A10bQ4xlV1 => {
            embedding_selection == NativeQuantEmbeddingSelection::Forced
        }
        ParallelCopyProfileId::A3bQ4kmV1 | ParallelCopyProfileId::Dense27bQ4kmV1 => {
            embedding_selection == NativeQuantEmbeddingSelection::AutoPromoted
                && native_quant_embedding_default_promoted(
                    &model.arch,
                    model.tied_embeddings,
                    model.token_embd.dtype,
                    &model.token_embd.shape,
                )
        }
    };
    if shard_lengths.as_slice() != profile.shard_mapped_lengths
        || gguf_descriptor_layout_digest(gguf) != profile.descriptor_layout_digest
        || model.arch != profile.arch
        || model.tied_embeddings != profile.tied_embeddings
        || model.mtp.is_some() != profile.mtp_present
        || model.token_embd.dtype != profile.embedding_dtype
        || model.token_embd.shape.as_slice() != profile.embedding_shape
        || !embedding_qualified
        || host_page_size_bytes()? != 16_384
        || ctx.max_buffer_length() != 77_309_411_328
        || expected.len() != profile.request_count
        || model_weight_storage_inventory_digest(expected) != profile.inventory_digest
    {
        return Ok(false);
    }
    let mut source_bytes = 0u64;
    for request in expected {
        let Some(shard_len) = profile.shard_mapped_lengths.get(request.desc.shard_idx) else {
            return Ok(false);
        };
        let Some(source_end) = request.desc.data_offset.checked_add(request.desc.n_bytes) else {
            return Ok(false);
        };
        if request.kind != ModelWeightStorageKind::Direct
            || request.desc.n_bytes == 0
            || request.resident_bytes != request.desc.n_bytes
            || source_end > *shard_len as u64
            || request.desc.n_bytes > ctx.max_buffer_length() as u64
            || usize::try_from(request.desc.n_bytes).is_err()
        {
            return Ok(false);
        }
        source_bytes = source_bytes
            .checked_add(request.desc.n_bytes)
            .ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy profile source byte overflow".to_string())
            })?;
    }
    Ok(source_bytes == profile.source_bytes)
}

pub(super) fn select_parallel_copy_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<&'static ParallelCopyProfile, MfError> {
    select_unique_parallel_copy_profile(&PARALLEL_COPY_PROFILES, |profile| {
        parallel_copy_profile_matches(ctx, gguf, model, expected, embedding_selection, profile)
    })
}

pub(super) fn host_physical_memory_bytes() -> Option<u64> {
    let mut bytes = 0u64;
    let mut size = std::mem::size_of::<u64>();
    let result = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            (&mut bytes as *mut u64).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && size == std::mem::size_of::<u64>()).then_some(bytes)
}

pub(super) fn a3b_parallel_copy_auto_host_supported(
    unified_memory: bool,
    device_name: &str,
    physical_memory_bytes: Option<u64>,
) -> bool {
    unified_memory
        && device_name == A3B_PARALLEL_COPY_AUTO_DEVICE
        && physical_memory_bytes.is_some_and(|bytes| bytes >= A3B_PARALLEL_COPY_AUTO_MIN_MEMORY)
}

pub(super) fn select_auto_parallel_copy_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<Option<&'static ParallelCopyProfile>, MfError> {
    if !a3b_parallel_copy_auto_host_supported(
        ctx.device.hasUnifiedMemory(),
        &ctx.device.name().to_string(),
        host_physical_memory_bytes(),
    ) {
        return Ok(None);
    }
    parallel_copy_profile_matches(
        ctx,
        gguf,
        model,
        expected,
        embedding_selection,
        &A3B_PARALLEL_COPY_PROFILE,
    )
    .map(|matched| matched.then_some(&A3B_PARALLEL_COPY_PROFILE))
}

pub(super) fn auto_parallel_copy_population(
    profile: ParallelCopyProfileId,
) -> Option<ParallelPopulationMethod> {
    match profile {
        ParallelCopyProfileId::A3bQ4kmV1 => Some(ParallelPopulationMethod::Pread),
        ParallelCopyProfileId::A10bQ4xlV1 => None,
        ParallelCopyProfileId::Dense27bQ4kmV1 => None,
    }
}

pub(super) fn forced_a10b_parallel_pread_embedding_qualified(
    parallel_mode: GgufParallelCopyMode,
    gguf: &GgufFile,
    model: &Model<'_>,
) -> bool {
    if parallel_mode != GgufParallelCopyMode::ForcedPread {
        return false;
    }
    let profile = &A10B_PARALLEL_PREAD_PROFILE;
    let architecture = gguf.architecture();
    architecture.as_deref() == profile.architecture_label
        && gguf.shard_mapped_lengths().as_slice() == profile.shard_mapped_lengths
        && gguf_descriptor_layout_digest(gguf) == profile.descriptor_layout_digest
        && model.arch == profile.arch
        && model.tied_embeddings == profile.tied_embeddings
        && model.mtp.is_some() == profile.mtp_present
        && model.token_embd.dtype == profile.embedding_dtype
        && model.token_embd.shape.as_slice() == profile.embedding_shape
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
