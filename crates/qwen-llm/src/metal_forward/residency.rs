//! Weight residency, retained storage, and loader plumbing.

use super::*;

/// Single source of truth for which weight dtypes the loader keeps in
/// their native form (vs. dequant'ing to F32). Used by both `load_weight`
/// in `MetalModel::load` and the byte-ledger diagnostic, so they stay
/// in sync. If you add a new native quant kernel, list its dtype here.
///
/// Q8_0 added v0.73b.1 — DFlash drafter switches from F32-dequant
/// resident (~7.4 GB) to native Q8_0 (~1.85 GB). F16/BF16 stay native once
/// their primitive mat-vec/mat-mat/get_rows kernels are available.
pub fn weight_dtype_kept_native(dtype: GgmlType) -> bool {
    matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::Q2_K
            | GgmlType::Q3_K
            | GgmlType::IQ2_S
            | GgmlType::IQ3_XXS
            | GgmlType::IQ3_S
            | GgmlType::Q4_0
            | GgmlType::Q4_1
            | GgmlType::Q4_K
            | GgmlType::Q5_K
            | GgmlType::Q6_K
            | GgmlType::Q8_0
            | GgmlType::IQ4_NL
            | GgmlType::IQ4_XS
    )
}

pub(super) fn parse_gguf_owned_arena_mode(
    value: Option<&str>,
) -> Result<GgufOwnedArenaMode, MfError> {
    match value {
        None => Ok(GgufOwnedArenaMode::Disabled),
        Some(value) if crate::env_flag::env_value_truthy(value) => Ok(GgufOwnedArenaMode::Forced),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(GgufOwnedArenaMode::Disabled),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_OWNED_ARENA value {value:?}"
        ))),
    }
}

pub(super) fn gguf_owned_arena_mode() -> Result<GgufOwnedArenaMode, MfError> {
    match std::env::var("QWEN_GGUF_OWNED_ARENA") {
        Ok(value) => parse_gguf_owned_arena_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_owned_arena_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_OWNED_ARENA is not valid Unicode".to_string(),
        )),
    }
}

pub(super) fn parse_gguf_parallel_copy_mode(
    value: Option<&str>,
) -> Result<GgufParallelCopyMode, MfError> {
    match value {
        None => Ok(GgufParallelCopyMode::Auto),
        Some(value) if value.eq_ignore_ascii_case("pread") => Ok(GgufParallelCopyMode::ForcedPread),
        Some(value) if value.eq_ignore_ascii_case("page-rounded-copy") => {
            Ok(GgufParallelCopyMode::ForcedPageRoundedCopy)
        }
        Some(value) if crate::env_flag::env_value_truthy(value) => {
            Ok(GgufParallelCopyMode::ForcedCopy)
        }
        Some(value) if crate::env_flag::env_value_falsy(value) => {
            Ok(GgufParallelCopyMode::Disabled)
        }
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_PARALLEL_COPY value {value:?}"
        ))),
    }
}

pub(super) fn gguf_parallel_copy_mode() -> Result<GgufParallelCopyMode, MfError> {
    match std::env::var("QWEN_GGUF_PARALLEL_COPY") {
        Ok(value) => parse_gguf_parallel_copy_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_parallel_copy_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_PARALLEL_COPY is not valid Unicode".to_string(),
        )),
    }
}

pub(super) fn auto_retained_single_pass_enabled(
    admission_enabled: bool,
    unified_memory: bool,
    no_copy_mode: GgufNoCopyMode,
    owned_mode: GgufOwnedArenaMode,
    parallel_mode: GgufParallelCopyMode,
    prefault_mode: GgufNoCopyPrefaultMode,
    explicit_override_present: bool,
) -> bool {
    admission_enabled
        && unified_memory
        && no_copy_mode == GgufNoCopyMode::Disabled
        && owned_mode == GgufOwnedArenaMode::Disabled
        && parallel_mode == GgufParallelCopyMode::Auto
        && prefault_mode == GgufNoCopyPrefaultMode::Default
        && !explicit_override_present
}

pub(super) fn parse_gguf_no_copy_mode(value: Option<&str>) -> Result<GgufNoCopyMode, MfError> {
    match value {
        None => Ok(GgufNoCopyMode::Disabled),
        Some(value) if crate::env_flag::env_value_truthy(value) => Ok(GgufNoCopyMode::Forced),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(GgufNoCopyMode::Disabled),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_NO_COPY value {value:?}"
        ))),
    }
}

pub(super) fn gguf_no_copy_mode() -> Result<GgufNoCopyMode, MfError> {
    match std::env::var("QWEN_GGUF_NO_COPY") {
        Ok(value) => parse_gguf_no_copy_mode(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_no_copy_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY is not valid Unicode".to_string(),
        )),
    }
}

pub(super) fn parse_gguf_no_copy_prefault(
    value: Option<&str>,
) -> Result<GgufNoCopyPrefaultMode, MfError> {
    match value {
        None => Ok(GgufNoCopyPrefaultMode::Default),
        Some(value) if crate::env_flag::env_value_truthy(value) => {
            Ok(GgufNoCopyPrefaultMode::Enabled)
        }
        Some(value) if crate::env_flag::env_value_falsy(value) => {
            Ok(GgufNoCopyPrefaultMode::Disabled)
        }
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_NO_COPY_PREFAULT value {value:?}"
        ))),
    }
}

pub(super) fn gguf_no_copy_prefault_mode() -> Result<GgufNoCopyPrefaultMode, MfError> {
    match std::env::var("QWEN_GGUF_NO_COPY_PREFAULT") {
        Ok(value) => parse_gguf_no_copy_prefault(Some(&value)),
        Err(std::env::VarError::NotPresent) => parse_gguf_no_copy_prefault(None),
        Err(std::env::VarError::NotUnicode(_)) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_NO_COPY_PREFAULT is not valid Unicode".to_string(),
        )),
    }
}

pub fn gguf_descriptor_layout_digest(gguf: &GgufFile) -> u64 {
    let mut hash = 0xcbf29ce484222325;
    hash_layout_bytes(&mut hash, b"qwen-gguf-layout-v1");
    hash_layout_u64(&mut hash, gguf.shard_count() as u64);
    for shard in &gguf.shards {
        hash_layout_u64(&mut hash, shard.mmap.len() as u64);
    }
    hash_layout_u64(&mut hash, gguf.tensors.len() as u64);
    for desc in &gguf.tensors {
        hash_layout_u64(&mut hash, desc.name.len() as u64);
        hash_layout_bytes(&mut hash, desc.name.as_bytes());
        hash_layout_u64(&mut hash, desc.dtype as i32 as u32 as u64);
        hash_layout_u64(&mut hash, desc.shard_idx as u64);
        hash_layout_u64(&mut hash, desc.data_offset);
        hash_layout_u64(&mut hash, desc.n_bytes);
        hash_layout_u64(&mut hash, desc.shape.len() as u64);
        for dim in &desc.shape {
            hash_layout_u64(&mut hash, *dim);
        }
    }
    hash
}

/// Shared emitter for load-time diagnostic lines (all the
/// `[metal-load]`, `[metal-load-ledger]`, `[metal-gguf-parallel-*]` etc.
/// prefixes). Historically these went straight to `stderr` via
/// `eprintln!`; they now go through `tracing::info!` at the `qwen_diag`
/// target so that
///
/// * the CLI's `DiagAwareFormat` emits them as bare bodies for the
///   downstream Python profile scripts (byte-for-byte compatible), and
/// * `RUST_LOG=qwen_diag=off` can suppress them without silencing real
///   warnings/errors from other targets.
///
/// Test builds also push each formatted line into a per-thread capture
/// buffer that `capture_metal_load_lines` drains for assertions.
pub(super) fn emit_metal_load_line(arguments: std::fmt::Arguments<'_>) {
    #[cfg(not(test))]
    tracing::info!(target: "qwen_diag", "{arguments}");

    #[cfg(test)]
    {
        let line = arguments.to_string();
        tracing::info!(target: "qwen_diag", "{}", line);
        METAL_LOAD_TEST_LINES.with(|lines| {
            if let Some(lines) = lines.borrow_mut().as_mut() {
                lines.push(line);
            }
        });
    }
}

pub(super) fn prepare_auto_retained_selection(
    parallel_auto_selected: bool,
    eligible: bool,
    planner: impl FnOnce() -> Result<RetainedStoragePlan, MfError>,
) -> PreparedAutoRetainedSelection {
    if parallel_auto_selected || !eligible {
        return PreparedAutoRetainedSelection::NotEligible;
    }
    match planner() {
        Ok(plan) => PreparedAutoRetainedSelection::Selected(plan),
        Err(error) => PreparedAutoRetainedSelection::NoMatch(error.to_string()),
    }
}

pub(super) fn storage_digest_records(records: impl IntoIterator<Item = String>) -> String {
    let mut hasher = Sha256::new();
    for record in records {
        hasher.update((record.len() as u64).to_be_bytes());
        hasher.update(record.as_bytes());
    }
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

pub fn model_weight_storage_inventory_digest(requests: &[ModelWeightStorageRequest<'_>]) -> String {
    storage_digest_records(requests.iter().map(|request| {
        format!(
            "{}\0{}\0{}\0{}\0{:?}\0{:?}\0{:?}\0{}",
            request.desc.name,
            request.desc.shard_idx,
            request.desc.data_offset,
            request.desc.n_bytes,
            request.desc.dtype,
            request.desc.shape,
            request.kind,
            request.resident_bytes
        )
    }))
}

pub fn retained_storage_plan_digest(plan: &RetainedStoragePlan) -> String {
    let mut records = vec![format!(
        "header\0{}\0{}\0{}\0{}",
        plan.page_size, plan.max_buffer_length, plan.usable_window_length, plan.required_alignment
    )];
    records.extend(plan.windows.iter().enumerate().map(|(index, window)| {
        format!(
            "window\0{index}\0{}\0{}\0{}",
            window.shard_idx, window.mmap_offset, window.length
        )
    }));
    records.extend(plan.entries.iter().map(|entry| {
        format!(
            "entry\0{}\0{}\0{}\0{}\0{}\0{:?}",
            entry.request_index,
            entry.name,
            entry.shard_idx,
            entry.data_offset,
            entry.n_bytes,
            entry.disposition
        )
    }));
    storage_digest_records(records)
}

pub(super) fn expected_model_weight_identity(
    request: &ModelWeightStorageRequest<'_>,
) -> ModelWeightStorageIdentity {
    ModelWeightStorageIdentity {
        name: request.desc.name.clone(),
        shard_idx: request.desc.shard_idx,
        data_offset: request.desc.data_offset,
        source_bytes: request.desc.n_bytes,
        dtype: request.desc.dtype,
        shape: request.desc.shape.clone(),
        kind: request.kind,
        resident_bytes: request.resident_bytes,
    }
}

pub(super) fn validate_model_weight_request_sequence(
    actual: &[ModelWeightStorageIdentity],
    expected: &[ModelWeightStorageRequest<'_>],
) -> Result<(), MfError> {
    if actual.len() != expected.len() {
        return Err(MfError::LoadPolicy(format!(
            "model storage request count drift: actual={} expected={}",
            actual.len(),
            expected.len()
        )));
    }
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        let expected = expected_model_weight_identity(expected);
        if *actual != expected {
            return Err(MfError::LoadPolicy(format!(
                "model storage request drift at index {index}: actual={actual:?} expected={expected:?}"
            )));
        }
    }
    Ok(())
}

pub(super) fn push_model_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    desc: &'a TensorDesc,
    kind: ModelWeightStorageKind,
) -> Result<(), MfError> {
    let elements = desc.checked_n_elements().ok_or_else(|| {
        MfError::LoadPolicy(format!("tensor {:?} element count overflow", desc.name))
    })?;
    let resident_bytes = match kind {
        ModelWeightStorageKind::Direct => desc.n_bytes,
        ModelWeightStorageKind::ConvertedF32 => elements
            .checked_mul(4)
            .ok_or_else(|| MfError::LoadPolicy("F32 conversion size overflow".to_string()))?,
        ModelWeightStorageKind::ConvertedF16 => elements
            .checked_mul(2)
            .ok_or_else(|| MfError::LoadPolicy("F16 conversion size overflow".to_string()))?,
    };
    requests.push(ModelWeightStorageRequest {
        desc,
        kind,
        resident_bytes,
    });
    Ok(())
}

pub(super) fn push_f32_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    desc: &'a TensorDesc,
) -> Result<(), MfError> {
    let kind = if desc.dtype == GgmlType::F32 {
        ModelWeightStorageKind::Direct
    } else {
        ModelWeightStorageKind::ConvertedF32
    };
    push_model_weight_request(requests, desc, kind)
}

pub(super) fn push_native_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    desc: &'a TensorDesc,
) -> Result<(), MfError> {
    if weight_dtype_kept_native(desc.dtype) {
        push_model_weight_request(requests, desc, ModelWeightStorageKind::Direct)
    } else {
        push_f32_weight_request(requests, desc)
    }
}

pub fn native_quant_embedding_storage_supported(model: &Model<'_>) -> bool {
    native_quant_embedding_supported(model.token_embd.dtype, &model.token_embd.shape)
}

pub fn production_native_quant_embedding_storage_enabled(model: &Model<'_>) -> bool {
    native_quant_embedding_storage_supported(model)
        && native_quant_embedding_default_promoted(
            &model.arch,
            model.tied_embeddings,
            model.token_embd.dtype,
            &model.token_embd.shape,
        )
}

pub fn model_weight_storage_requests<'a>(
    model: &Model<'a>,
    native_quant_embedding: bool,
    router_f16: bool,
) -> Result<Vec<ModelWeightStorageRequest<'a>>, MfError> {
    if native_quant_embedding && !native_quant_embedding_storage_supported(model) {
        return Err(MfError::LoadPolicy(format!(
            "native token embedding is unsupported for {:?} {:?}",
            model.token_embd.dtype, model.token_embd.shape
        )));
    }
    let mut requests = Vec::new();
    let embedding_direct = matches!(
        model.token_embd.dtype,
        GgmlType::F32 | GgmlType::F16 | GgmlType::BF16
    ) || native_quant_embedding;
    if embedding_direct {
        push_model_weight_request(
            &mut requests,
            model.token_embd,
            ModelWeightStorageKind::Direct,
        )?;
    } else {
        push_f32_weight_request(&mut requests, model.token_embd)?;
    }
    push_f32_weight_request(&mut requests, model.output_norm)?;
    push_native_weight_request(&mut requests, model.lm_head)?;

    for block in &model.blocks {
        match block {
            Block::Gdn(gdn) => {
                push_f32_weight_request(&mut requests, gdn.attn_norm)?;
                push_f32_weight_request(&mut requests, gdn.post_attention_norm)?;
                push_native_weight_request(&mut requests, gdn.ffn_gate)?;
                push_native_weight_request(&mut requests, gdn.ffn_up)?;
                push_native_weight_request(&mut requests, gdn.ffn_down)?;
                push_native_weight_request(&mut requests, gdn.in_proj_qkv)?;
                push_native_weight_request(&mut requests, gdn.in_proj_z)?;
                push_native_weight_request(&mut requests, gdn.beta_proj)?;
                push_native_weight_request(&mut requests, gdn.alpha_proj)?;
                push_f32_weight_request(&mut requests, gdn.a_log)?;
                push_f32_weight_request(&mut requests, gdn.dt_bias)?;
                push_f32_weight_request(&mut requests, gdn.conv1d)?;
                push_f32_weight_request(&mut requests, gdn.norm)?;
                push_native_weight_request(&mut requests, gdn.out_proj)?;
                if let Some(moe) = gdn.ffn_moe.as_ref() {
                    push_moe_weight_requests(&mut requests, moe, router_f16)?;
                }
            }
            Block::Attn(attn) => {
                push_native_weight_request(&mut requests, attn.q)?;
                push_native_weight_request(&mut requests, attn.k)?;
                push_native_weight_request(&mut requests, attn.v)?;
                push_f32_weight_request(&mut requests, attn.attn_norm)?;
                push_f32_weight_request(&mut requests, attn.post_attention_norm)?;
                push_native_weight_request(&mut requests, attn.ffn_gate)?;
                push_native_weight_request(&mut requests, attn.ffn_up)?;
                push_native_weight_request(&mut requests, attn.ffn_down)?;
                push_native_weight_request(&mut requests, attn.o)?;
                push_f32_weight_request(&mut requests, attn.q_norm)?;
                push_f32_weight_request(&mut requests, attn.k_norm)?;
                if let Some(moe) = attn.ffn_moe.as_ref() {
                    push_moe_weight_requests(&mut requests, moe, router_f16)?;
                }
            }
        }
    }
    Ok(requests)
}

pub fn mtp_weight_source_descriptors<'a>(model: &Model<'a>) -> Vec<&'a TensorDesc> {
    let Some(mtp) = model.mtp.as_ref() else {
        return Vec::new();
    };
    let attn = &mtp.attn;
    let mut descriptors = vec![
        attn.q,
        attn.k,
        attn.v,
        attn.attn_norm,
        attn.post_attention_norm,
        attn.ffn_gate,
        attn.ffn_up,
        attn.ffn_down,
        attn.o,
        attn.q_norm,
        attn.k_norm,
    ];
    if let Some(moe) = attn.ffn_moe.as_ref() {
        descriptors.extend([
            moe.gate_inp,
            moe.gate_exps,
            moe.up_exps,
            moe.down_exps,
            moe.gate_inp_shexp,
        ]);
    }
    descriptors.extend([mtp.eh_proj, mtp.enorm, mtp.hnorm, mtp.shared_head_norm]);
    descriptors
}

pub(super) fn validate_parallel_copied_topology(
    profile: &ParallelCopyProfile,
    destination_length: ParallelDestinationLength,
    expected: &[ModelWeightStorageIdentity],
    resources: &[Buffer],
    tensors: &[MetalTensor],
) -> Result<(), MfError> {
    if expected.len() != profile.request_count
        || resources.len() != expected.len()
        || tensors.len() != expected.len()
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy topology count drifted".to_string(),
        ));
    }
    let mut resource_identities = HashSet::with_capacity(resources.len());
    let mut resource_bytes = 0u64;
    for (index, ((identity, resource), tensor)) in
        expected.iter().zip(resources).zip(tensors).enumerate()
    {
        let resource_identity = Retained::as_ptr(resource) as *const () as usize;
        let expected_resource_length = parallel_destination_resource_length(
            identity.source_bytes,
            destination_length,
            usize::MAX,
        )?;
        resource_bytes = resource_bytes
            .checked_add(resource.length() as u64)
            .ok_or_else(|| MfError::LoadPolicy("parallel-copy byte overflow".to_string()))?;
        if !resource_identities.insert(resource_identity)
            || resource.length() != expected_resource_length
            || resource.storageMode() != MTLStorageMode::Shared
            || resource.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || resource.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
            || Retained::as_ptr(&tensor.buffer) != Retained::as_ptr(resource)
            || tensor.offset != 0
            || tensor.n_bytes() != identity.resident_bytes
            || tensor.dtype != identity.dtype
            || tensor.shape != identity.shape
            || tensor.provenance() != MetalTensorProvenance::OwnedWeightReadOnly
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy resource or tensor {index} drifted"
            )));
        }
    }
    let (expected_resource_bytes, padded_resources) =
        parallel_destination_accounting(expected, destination_length, usize::MAX)?;
    let accounting_matches = match destination_length {
        ParallelDestinationLength::LogicalExact => {
            expected_resource_bytes == profile.source_bytes && padded_resources == 0
        }
        ParallelDestinationLength::PageRounded16K => {
            expected_resource_bytes == GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES
                && expected_resource_bytes.checked_sub(profile.source_bytes)
                    == Some(GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES)
                && padded_resources == GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES
        }
    };
    if resource_bytes != expected_resource_bytes || !accounting_matches {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy resource accounting drifted: actual={resource_bytes} expected={expected_resource_bytes} padded={padded_resources}"
        )));
    }
    Ok(())
}

pub(super) fn authenticated_a3b_storage_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<RetainedStoragePlan, MfError> {
    if !ctx.device.hasUnifiedMemory() {
        return Err(MfError::LoadPolicy(
            "forced A3B storage requires unified memory".to_string(),
        ));
    }
    let expected_source_bytes = expected.iter().try_fold(0u64, |total, request| {
        total
            .checked_add(request.desc.n_bytes)
            .ok_or_else(|| MfError::LoadPolicy("A3B source byte overflow".to_string()))
    })?;
    if gguf.shard_count() != 1
        || gguf.total_mapped_len() != GGUF_OWNED_A3B_MAPPED_BYTES
        || gguf_descriptor_layout_digest(gguf) != GGUF_OWNED_A3B_LAYOUT_DIGEST
        || !matches_owned_a3b_arch(model)
        || model.tied_embeddings
        || model.mtp.is_some()
        || embedding_selection != NativeQuantEmbeddingSelection::AutoPromoted
        || expected.len() != GGUF_OWNED_A3B_REQUESTS
        || expected_source_bytes != GGUF_OWNED_A3B_SOURCE_BYTES
        || expected
            .iter()
            .any(|request| request.kind != ModelWeightStorageKind::Direct)
        || model_weight_storage_inventory_digest(expected) != GGUF_OWNED_A3B_INVENTORY_DIGEST
    {
        return Err(MfError::LoadPolicy(
            "forced A3B storage rejects non-sentinel layout".to_string(),
        ));
    }

    let direct = expected
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let page_size = host_page_size_bytes()?;
    let plan = plan_retained_storage(
        &gguf.shard_mapped_lengths(),
        &direct,
        page_size,
        ctx.max_buffer_length(),
        GGUF_NO_COPY_ALIGNMENT,
    )?;
    let window_bytes = plan.windows.iter().try_fold(0u64, |total, window| {
        total
            .checked_add(window.length as u64)
            .ok_or_else(|| MfError::LoadPolicy("A3B window byte overflow".to_string()))
    })?;
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let alias_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let fallback_entries = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .collect::<Vec<_>>();
    if page_size != 16_384
        || ctx.max_buffer_length() != 77_309_411_328
        || retained_storage_plan_digest(&plan) != GGUF_OWNED_A3B_PLAN_DIGEST
        || plan.windows.len() != 1
        || window_bytes != GGUF_OWNED_A3B_WINDOW_BYTES
        || view_count != GGUF_OWNED_A3B_VIEWS
        || plan.unique_view_bytes != GGUF_OWNED_A3B_VIEW_BYTES
        || plan.logical_view_bytes != GGUF_OWNED_A3B_VIEW_BYTES
        || alias_count != 0
        || plan.alias_bytes != 0
        || fallback_entries.len() != 1
        || plan.unique_fallback_bytes != GGUF_OWNED_A3B_FALLBACK_BYTES
        || window_bytes - plan.unique_view_bytes != GGUF_OWNED_A3B_GAP_BYTES
        || !matches!(
            fallback_entries[0].disposition,
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage
            }
        )
    {
        return Err(MfError::LoadPolicy(
            "forced A3B storage planner geometry drifted".to_string(),
        ));
    }
    Ok(plan)
}

pub(super) fn planned_owned_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<PlannedOwnedStorage, MfError> {
    let plan = authenticated_a3b_storage_plan(ctx, gguf, model, expected, embedding_selection)?;
    let direct = expected
        .iter()
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    let fallback_entries = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .collect::<Vec<_>>();
    let window_bytes = plan.windows[0].length as u64;

    let ready_started = std::time::Instant::now();
    let allocation_started = std::time::Instant::now();
    let resources = vec![
        ctx.buffer_uninit(plan.windows[0].length)?,
        ctx.buffer_uninit(GGUF_OWNED_A3B_FALLBACK_BYTES as usize)?,
    ];
    let allocation_ms = allocation_started.elapsed().as_secs_f64() * 1e3;
    let copy_started = std::time::Instant::now();
    let window = &plan.windows[0];
    let window_source = gguf
        .try_shard_range(window.shard_idx, window.mmap_offset, window.length)
        .map_err(|error| MfError::LoadPolicy(format!("owned window source: {error}")))?;
    copy_owned_arena_four_workers(window_source, &resources[0], plan.page_size)?;
    let fallback = fallback_entries[0];
    let fallback_desc = direct.get(fallback.request_index).ok_or_else(|| {
        MfError::LoadPolicy("owned fallback request index is out of bounds".to_string())
    })?;
    copy_owned_arena_serial(gguf.slice(fallback_desc), &resources[1])?;
    let copy_ms = copy_started.elapsed().as_secs_f64() * 1e3;
    let ready_ms = ready_started.elapsed().as_secs_f64() * 1e3;
    let physical_bytes = resources.iter().try_fold(0u64, |total, buffer| {
        total
            .checked_add(buffer.length() as u64)
            .ok_or_else(|| MfError::LoadPolicy("owned physical byte overflow".to_string()))
    })?;
    if resources.len() != 2
        || physical_bytes != GGUF_OWNED_A3B_PHYSICAL_BYTES
        || resources
            .iter()
            .any(|buffer| buffer.storageMode() != MTLStorageMode::Shared)
    {
        return Err(MfError::LoadPolicy(
            "forced owned arena physical realization drifted".to_string(),
        ));
    }
    // Migrated from `eprintln!` to `qwen_diag`; consumed by v0598 (owned
    // arena pilot) which anchors `line.startswith("[metal-gguf-owned]")`.
    // The CLI subscriber's bare-body renderer keeps the byte format so
    // that downstream match logic continues to work without changes.
    tracing::info!(
        target: "qwen_diag",
        concat!(
            "[metal-gguf-owned] windows=1 window_bytes={} gaps={} fallback=1/{} ",
            "resources=2/{} workers={} page={} alignment={} allocation_ms={:.3} ",
            "copy_ms={:.3} ready_ms={:.3}",
        ),
        window_bytes,
        GGUF_OWNED_A3B_GAP_BYTES,
        GGUF_OWNED_A3B_FALLBACK_BYTES,
        physical_bytes,
        GGUF_OWNED_WORKERS,
        plan.page_size,
        plan.required_alignment,
        allocation_ms,
        copy_ms,
        ready_ms,
    );
    let mut fallback_resources = HashMap::new();
    fallback_resources.insert(fallback.request_index, 1);
    Ok(PlannedOwnedStorage {
        realized: vec![None; plan.entries.len()],
        plan,
        resources,
        fallback_resources,
        cursor: 0,
    })
}

pub(super) fn planned_parallel_copied_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<PlannedParallelCopiedStorage, MfError> {
    let profile = select_parallel_copy_profile(ctx, gguf, model, expected, embedding_selection)?;
    let prepared = prepare_parallel_copied_profile(
        ctx,
        gguf,
        model,
        expected,
        embedding_selection,
        profile,
        population,
        destination_length,
    )?;
    realize_parallel_copied_profile(ctx, gguf, expected, prepared)
}

pub(super) fn validate_a10b_parallel_memory_admission(
    ctx: &MetalContext,
    profile: &ParallelCopyProfile,
    phase: &str,
) -> Result<(), MfError> {
    if profile.id != ParallelCopyProfileId::A10bQ4xlV1 {
        return Ok(());
    }
    let admission = evaluate_metal_memory_admission(
        A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES,
        0,
        ctx.memory_signals(),
        true,
    );
    emit_metal_load_line(format_args!(
        concat!(
            "[metal-gguf-parallel-memory] profile={} phase={} admitted={} reason={} ",
            "required={} recommended={} current={} process_remaining={:?}"
        ),
        profile.id.label(),
        phase,
        admission.admitted,
        admission.reason.as_str(),
        A10B_PARALLEL_PREAD_REQUIRED_HEADROOM_BYTES,
        admission.signals.recommended_max_bytes,
        admission.signals.current_allocated_bytes,
        admission.signals.process_limit_remaining_bytes,
    ));
    if !admission.admitted {
        return Err(MfError::LoadPolicy(format!(
            "A10B parallel pread memory admission failed during {phase}: {}",
            admission.reason.as_str()
        )));
    }
    Ok(())
}

pub(super) fn prepare_parallel_copied_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    profile: &'static ParallelCopyProfile,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<PreparedParallelCopiedProfile, MfError> {
    validate_parallel_destination_length(profile, population, destination_length)?;
    validate_a10b_parallel_memory_admission(ctx, profile, "prepare")?;
    let proof = match profile.authentication {
        ParallelCopyAuthentication::A3bRetainedPlan => {
            authenticated_a3b_storage_plan(ctx, gguf, model, expected, embedding_selection)?;
            PreparedParallelCopyProof::A3bRetainedPlan
        }
        ParallelCopyAuthentication::A10bPlannerFree => PreparedParallelCopyProof::A10bPlannerFree,
        ParallelCopyAuthentication::DensePlannerFree => PreparedParallelCopyProof::DensePlannerFree,
    };
    let expected_identities = expected
        .iter()
        .map(expected_model_weight_identity)
        .collect::<Vec<_>>();
    let sorted_request_indices = frozen_parallel_copy_order(profile, &expected_identities)?;
    Ok(PreparedParallelCopiedProfile {
        profile,
        population,
        destination_length,
        expected_identities,
        sorted_request_indices,
        _proof: proof,
    })
}

pub(super) fn realize_parallel_copied_profile(
    ctx: &MetalContext,
    gguf: &GgufFile,
    expected: &[ModelWeightStorageRequest<'_>],
    prepared: PreparedParallelCopiedProfile,
) -> Result<PlannedParallelCopiedStorage, MfError> {
    let PreparedParallelCopiedProfile {
        profile,
        population,
        destination_length,
        expected_identities,
        sorted_request_indices,
        _proof,
    } = prepared;
    validate_model_weight_request_sequence(&expected_identities, expected)?;
    validate_a10b_parallel_memory_admission(ctx, profile, "realize")?;
    let source_stamps = if profile.id == ParallelCopyProfileId::A10bQ4xlV1 {
        Some(gguf.revalidate_retained_shard_stamps().map_err(|error| {
            MfError::LoadPolicy(format!("A10B pread source preflight failed: {error}"))
        })?)
    } else {
        None
    };
    let usage_before = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_usage()?)
        }
    };
    let proc_before = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_proc_usage()?)
        }
    };
    let ready_started = std::time::Instant::now();
    let resources_result = expected
        .iter()
        .map(|request| {
            let length = parallel_destination_resource_length(
                request.desc.n_bytes,
                destination_length,
                ctx.max_buffer_length(),
            )?;
            ctx.buffer_uninit(length).map_err(MfError::from)
        })
        .collect::<Result<Vec<_>, _>>();
    let mut resources = resources_result?;
    let allocation_finished = std::time::Instant::now();

    let sources = match population {
        ParallelPopulationMethod::MmapCopy => Some(
            expected
                .iter()
                .enumerate()
                .map(|(index, request)| {
                    gguf.try_slice(request.desc).map_err(|error| {
                        MfError::LoadPolicy(format!(
                            "parallel-copy source resolution failed at {index}: {error}"
                        ))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?,
        ),
        ParallelPopulationMethod::Pread => None,
    };

    let mut resource_identities = HashSet::with_capacity(resources.len());
    let mut destination_ranges = Vec::with_capacity(resources.len());
    let mut destination_bytes = 0u64;
    for (index, (resource, request)) in resources.iter().zip(expected).enumerate() {
        let identity = Retained::as_ptr(resource) as *const () as usize;
        let start = resource.contents().as_ptr().cast::<u8>() as usize;
        let end = start.checked_add(resource.length()).ok_or_else(|| {
            MfError::LoadPolicy("parallel-copy destination range overflow".to_string())
        })?;
        destination_bytes = destination_bytes
            .checked_add(resource.length() as u64)
            .ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy destination byte overflow".to_string())
            })?;
        let expected_resource_length = parallel_destination_resource_length(
            request.desc.n_bytes,
            destination_length,
            ctx.max_buffer_length(),
        )?;
        if !resource_identities.insert(identity)
            || resource.length() != expected_resource_length
            || resource.length() == 0
            || start == 0
            || resource.storageMode() != MTLStorageMode::Shared
            || resource.cpuCacheMode() != MTLCPUCacheMode::DefaultCache
            || resource.hazardTrackingMode() != MTLHazardTrackingMode::Tracked
        {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy destination resource {index} drifted"
            )));
        }
        destination_ranges.push((start, end));
    }
    let (expected_destination_bytes, padded_resources) = parallel_destination_accounting(
        &expected_identities,
        destination_length,
        ctx.max_buffer_length(),
    )?;
    let destination_accounting_matches = match destination_length {
        ParallelDestinationLength::LogicalExact => {
            expected_destination_bytes == profile.source_bytes && padded_resources == 0
        }
        ParallelDestinationLength::PageRounded16K => {
            expected_destination_bytes == GGUF_PAGE_ROUNDED_A3B_ALLOCATED_BYTES
                && expected_destination_bytes.checked_sub(profile.source_bytes)
                    == Some(GGUF_PAGE_ROUNDED_A3B_PADDING_BYTES)
                && padded_resources == GGUF_PAGE_ROUNDED_A3B_PADDED_RESOURCES
        }
    };
    if destination_bytes != expected_destination_bytes || !destination_accounting_matches {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy destination accounting drifted: actual={destination_bytes} expected={expected_destination_bytes} padded={padded_resources}"
        )));
    }
    let mut ranges_by_address = destination_ranges.clone();
    ranges_by_address.sort_unstable();
    if ranges_by_address
        .windows(2)
        .any(|pair| pair[0].1 > pair[1].0)
    {
        return Err(MfError::LoadPolicy(
            "parallel-copy destination resources overlap".to_string(),
        ));
    }
    if let Some(sources) = sources.as_ref() {
        for (index, source) in sources.iter().enumerate() {
            let source_start = source.as_ptr() as usize;
            let source_end = source_start.checked_add(source.len()).ok_or_else(|| {
                MfError::LoadPolicy("parallel-copy source range overflow".to_string())
            })?;
            if source.is_empty()
                || source_start == 0
                || source.len() as u64 != expected[index].desc.n_bytes
                || source.len() > resources[index].length()
                || destination_ranges
                    .iter()
                    .any(|&(start, end)| source_start < end && start < source_end)
            {
                return Err(MfError::LoadPolicy(format!(
                    "parallel-copy source range {index} is invalid or overlaps a destination"
                )));
            }
        }
    }

    let mut tasks_by_request = resources
        .iter_mut()
        .zip(expected)
        .enumerate()
        .map(|(request_index, (resource, request))| {
            // SAFETY: every prerequisite in exclusive_buffer_bytes_mut's
            // contract was established for the complete resource set above.
            let logical_length = usize::try_from(request.desc.n_bytes).map_err(|_| {
                MfError::LoadPolicy(
                    "parallel-copy logical destination length does not fit usize".to_string(),
                )
            })?;
            let destination = unsafe { exclusive_buffer_bytes_mut(resource, logical_length) };
            Ok(Some(match population {
                ParallelPopulationMethod::MmapCopy => {
                    let source = *sources
                        .as_ref()
                        .expect("mmap-copy population must resolve sources")
                        .get(request_index)
                        .expect("source inventory matches destination inventory");
                    ParallelPopulationTask::MmapCopy(ParallelCopyTask {
                        source,
                        destination,
                    })
                }
                ParallelPopulationMethod::Pread => {
                    ParallelPopulationTask::Pread(ParallelPreadTask {
                        shard_idx: request.desc.shard_idx,
                        source_offset: request.desc.data_offset,
                        destination,
                    })
                }
            }))
        })
        .collect::<Result<Vec<_>, MfError>>()?;
    let mut tasks = Vec::with_capacity(tasks_by_request.len());
    for &request_index in &sorted_request_indices {
        tasks.push(tasks_by_request[request_index].take().ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "parallel-copy request {request_index} was assigned more than once"
            ))
        })?);
    }
    if tasks_by_request.iter().any(Option::is_some) {
        return Err(MfError::LoadPolicy(
            "parallel-copy task union is incomplete".to_string(),
        ));
    }
    let source_finished = std::time::Instant::now();

    let copy_result = std::thread::scope(|scope| {
        let mut task_tail = tasks.as_mut_slice();
        let mut handles = Vec::with_capacity(GGUF_OWNED_WORKERS);
        let mut spawn_error = None;
        let mut partition_error = None;
        let mut assigned_tasks = 0usize;
        for worker in 0..GGUF_OWNED_WORKERS {
            let count = profile.task_counts[worker];
            if count == 0 || count > task_tail.len() {
                partition_error = Some(worker);
                break;
            }
            let (worker_tasks, remaining) = task_tail.split_at_mut(count);
            match std::thread::Builder::new().spawn_scoped(scope, move || {
                for task in worker_tasks {
                    match task {
                        ParallelPopulationTask::MmapCopy(task) => {
                            task.destination.copy_from_slice(task.source);
                        }
                        ParallelPopulationTask::Pread(task) => {
                            gguf.read_shard_exact_at(
                                task.shard_idx,
                                task.source_offset,
                                task.destination,
                            )
                            .map_err(|error| {
                                MfError::LoadPolicy(format!(
                                    "parallel-pread worker {worker} read failed: {error}"
                                ))
                            })?;
                        }
                    }
                }
                Ok::<(), MfError>(())
            }) {
                Ok(handle) => handles.push((worker, handle)),
                Err(error) => {
                    spawn_error = Some((worker, error));
                    break;
                }
            }
            task_tail = remaining;
            assigned_tasks += count;
        }

        let mut worker_error = None;
        let mut panicked_worker = None;
        for (worker, handle) in handles {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(error)) if worker_error.is_none() => {
                    worker_error = Some((worker, error));
                }
                Ok(Err(_)) => {}
                Err(_) if panicked_worker.is_none() => panicked_worker = Some(worker),
                Err(_) => {}
            }
        }
        if let Some((worker, error)) = spawn_error {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy worker {worker} spawn failed: {error}"
            )));
        }
        if let Some(worker) = partition_error {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy worker {worker} partition is invalid"
            )));
        }
        if let Some((worker, error)) = worker_error {
            return Err(MfError::LoadPolicy(format!(
                "parallel population worker {worker} failed: {error}"
            )));
        }
        if let Some(worker) = panicked_worker {
            return Err(MfError::LoadPolicy(format!(
                "parallel-copy worker {worker} panicked"
            )));
        }
        if assigned_tasks != profile.request_count {
            return Err(MfError::LoadPolicy(
                "parallel-copy workers did not consume every task".to_string(),
            ));
        }
        Ok(())
    });
    copy_result?;
    if let Some(before) = source_stamps {
        let after = gguf.revalidate_retained_shard_stamps().map_err(|error| {
            MfError::LoadPolicy(format!("A10B pread source postflight failed: {error}"))
        })?;
        if after != before {
            return Err(MfError::LoadPolicy(
                "A10B retained source stamps changed during population".to_string(),
            ));
        }
    }
    let copy_finished = std::time::Instant::now();
    drop(tasks);
    drop(tasks_by_request);
    drop(sources);

    let tensors = resources
        .iter()
        .zip(expected)
        .map(|(resource, request)| {
            MetalTensor::owned_weight_view(
                resource.clone(),
                0,
                request.desc.shape.clone(),
                request.desc.dtype,
                GGUF_NO_COPY_ALIGNMENT,
            )
            .map_err(MfError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate_parallel_copied_topology(
        profile,
        destination_length,
        &expected_identities,
        &resources,
        &tensors,
    )?;
    let binding_finished = std::time::Instant::now();
    let usage_after = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_usage()?)
        }
    };
    let proc_after = match profile.marker_contract {
        ParallelCopyMarkerContract::A3b => None,
        ParallelCopyMarkerContract::A10bSchema2 | ParallelCopyMarkerContract::DenseSchema2 => {
            Some(capture_parallel_copy_proc_usage()?)
        }
    };

    let allocation_us = allocation_finished
        .duration_since(ready_started)
        .as_micros() as u64;
    let source_us = source_finished
        .duration_since(allocation_finished)
        .as_micros() as u64;
    let copy_us = copy_finished.duration_since(source_finished).as_micros() as u64;
    let binding_us = binding_finished.duration_since(copy_finished).as_micros() as u64;
    let ready_us = binding_finished.duration_since(ready_started).as_micros() as u64;
    let phase_us = allocation_us
        .checked_add(source_us)
        .and_then(|total| total.checked_add(copy_us))
        .and_then(|total| total.checked_add(binding_us))
        .ok_or_else(|| MfError::LoadPolicy("parallel-copy phase time overflow".to_string()))?;
    if ready_us == 0 || ready_us.abs_diff(phase_us) > 4 {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy timing reconciliation drifted: ready={ready_us} phases={phase_us}"
        )));
    }

    let accounting = match (usage_before, usage_after, proc_before, proc_after) {
        (None, None, None, None) => None,
        (Some(before), Some(after), Some(proc_before), Some(proc_after)) => Some(
            finish_parallel_copy_accounting(before, after, proc_before, proc_after)?,
        ),
        _ => {
            return Err(MfError::LoadPolicy(
                "parallel-copy endpoint accounting capture is incomplete".to_string(),
            ));
        }
    };
    emit_parallel_copy_marker(
        profile,
        population,
        destination_length,
        ParallelCopyTiming {
            allocation_us,
            source_us,
            copy_us,
            binding_us,
            ready_us,
        },
        accounting,
    )?;

    Ok(PlannedParallelCopiedStorage {
        profile,
        destination_length,
        expected: expected_identities,
        sorted_request_indices,
        resources,
        tensors,
        cursor: 0,
    })
}

pub(super) fn retained_storage_plan_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    expected: &[ModelWeightStorageRequest<'_>],
) -> Result<RetainedStoragePlan, MfError> {
    let direct = expected
        .iter()
        .filter(|request| request.kind == ModelWeightStorageKind::Direct)
        .map(|request| request.desc)
        .collect::<Vec<_>>();
    if direct.is_empty() {
        return Err(MfError::LoadPolicy(
            "forced retained storage requires at least one direct tensor".to_string(),
        ));
    }
    let page_size = host_page_size_bytes()?;
    let max_buffer_length = ctx.device.maxBufferLength();
    let plan = plan_retained_storage(
        &gguf.shard_mapped_lengths(),
        &direct,
        page_size,
        max_buffer_length,
        GGUF_NO_COPY_ALIGNMENT,
    )?;
    for entry in &plan.entries {
        if let RetainedStorageDisposition::CopyFallback { reason } = entry.disposition
            && reason != RetainedStorageFallback::FinalPartialPage
        {
            return Err(MfError::LoadPolicy(format!(
                "forced retained storage rejects {:?} fallback for {:?}",
                reason, entry.name
            )));
        }
    }

    Ok(plan)
}

pub(super) fn realize_retained_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    plan: RetainedStoragePlan,
    prefault_enabled: bool,
) -> Result<PlannedRetainedStorage, MfError> {
    let mut windows = Vec::with_capacity(plan.windows.len());
    let mut window_bytes = 0u64;
    let mut prefault_pages = 0usize;
    let mut prefault_bytes = 0usize;
    let mut prefault_ms = 0.0;
    let mut prefault_checksum = 0u64;
    for window in &plan.windows {
        let mmap = gguf.retained_shard_mmap(window.shard_idx).ok_or_else(|| {
            MfError::LoadPolicy(format!(
                "retained storage window references missing shard {}",
                window.shard_idx
            ))
        })?;
        let mmap_offset = usize::try_from(window.mmap_offset).map_err(|_| {
            MfError::LoadPolicy(format!(
                "retained storage window offset {} does not fit usize",
                window.mmap_offset
            ))
        })?;
        let backing = ctx.gguf_no_copy_window(
            mmap,
            window.shard_idx,
            mmap_offset,
            window.length,
            GGUF_NO_COPY_ALIGNMENT,
        )?;
        if backing.mmap_offset() != mmap_offset
            || backing.exposed_len() != window.length
            || backing.required_alignment() != GGUF_NO_COPY_ALIGNMENT
        {
            return Err(MfError::LoadPolicy(format!(
                "retained storage window realization drift at shard {} offset {}",
                window.shard_idx, window.mmap_offset
            )));
        }
        window_bytes = window_bytes
            .checked_add(window.length as u64)
            .ok_or_else(|| MfError::LoadPolicy("retained window bytes overflow".to_string()))?;
        if prefault_enabled {
            let report = backing.prefault_read();
            let expected_pages = backing.exposed_len() / backing.page_size();
            if report.page_count != expected_pages || report.covered_bytes != backing.exposed_len()
            {
                return Err(MfError::LoadPolicy(format!(
                    "retained prefault mismatch: pages={}/{} covered={}/{}",
                    report.page_count,
                    expected_pages,
                    report.covered_bytes,
                    backing.exposed_len(),
                )));
            }
            prefault_pages = prefault_pages
                .checked_add(report.page_count)
                .ok_or_else(|| {
                    MfError::LoadPolicy("retained prefault page count overflow".to_string())
                })?;
            prefault_bytes = prefault_bytes
                .checked_add(report.covered_bytes)
                .ok_or_else(|| {
                    MfError::LoadPolicy("retained prefault byte count overflow".to_string())
                })?;
            prefault_ms += report.wall_ms;
            prefault_checksum = prefault_checksum.rotate_left(7) ^ report.checksum;
        }
        windows.push(backing);
    }
    let view_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::View { .. }))
        .count();
    let alias_count = plan
        .entries
        .iter()
        .filter(|entry| matches!(entry.disposition, RetainedStorageDisposition::Alias { .. }))
        .count();
    let fallback_count = plan
        .entries
        .iter()
        .filter(|entry| {
            matches!(
                entry.disposition,
                RetainedStorageDisposition::CopyFallback { .. }
            )
        })
        .count();
    // Migrated from `eprintln!` to `qwen_diag`; consumed by v0595/v0596
    // and the dense-27b parallel-pread scripts (v0605 etc.) that anchor
    // `line.startswith("[metal-gguf-retained]")` and parse the
    // `source=/direct_copy=` fields.
    tracing::info!(
        target: "qwen_diag",
        concat!(
            "[metal-gguf-retained] windows={} window_bytes={} direct={} view={}/{} ",
            "alias={}/{} fallback={}/{} page={} max_buffer={} alignment={} ",
            "prefault={} prefault_pages={} prefault_bytes={} prefault_ms={:.3} ",
            "checksum={:#018x}",
        ),
        plan.windows.len(),
        window_bytes,
        plan.entries.len(),
        view_count,
        plan.unique_view_bytes,
        alias_count,
        plan.alias_bytes,
        fallback_count,
        plan.unique_fallback_bytes,
        plan.page_size,
        plan.max_buffer_length,
        plan.required_alignment,
        if prefault_enabled {
            "enabled"
        } else {
            "disabled"
        },
        prefault_pages,
        prefault_bytes,
        prefault_ms,
        prefault_checksum,
    );
    Ok(PlannedRetainedStorage {
        realized: vec![None; plan.entries.len()],
        plan,
        windows,
        cursor: 0,
    })
}

pub(super) fn planned_retained_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    expected: &[ModelWeightStorageRequest<'_>],
    prefault_enabled: bool,
) -> Result<PlannedRetainedStorage, MfError> {
    let plan = retained_storage_plan_for_load(ctx, gguf, expected)?;
    realize_retained_storage_for_load(ctx, gguf, plan, prefault_enabled)
}

pub(super) fn direct_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    mode: GgufNoCopyMode,
    prefault_enabled: bool,
    owned_mode: GgufOwnedArenaMode,
    parallel_mode: GgufParallelCopyMode,
    prepared_auto: PreparedAutoSelection,
    prepared_auto_retained: PreparedAutoRetainedSelection,
    embedding_selection: NativeQuantEmbeddingSelection,
) -> Result<(DirectStorage, bool), MfError> {
    let exact_sentinel = matches_no_copy_27b_sentinel(gguf, model);
    if let Some((population, destination_length)) = parallel_mode.forced_configuration() {
        let storage = planned_parallel_copied_storage_for_load(
            ctx,
            gguf,
            model,
            expected,
            embedding_selection,
            population,
            destination_length,
        )?;
        return Ok((DirectStorage::ForcedParallelCopied(storage), false));
    }
    if owned_mode == GgufOwnedArenaMode::Forced {
        if mode != GgufNoCopyMode::Disabled {
            return Err(MfError::LoadPolicy(
                "owned arena and retained no-copy are mutually exclusive".to_string(),
            ));
        }
        let storage =
            planned_owned_storage_for_load(ctx, gguf, model, expected, embedding_selection)?;
        return Ok((DirectStorage::ForcedOwned(storage), false));
    }
    if let PreparedAutoSelection::Selected(prepared) = prepared_auto {
        let storage = realize_parallel_copied_profile(ctx, gguf, expected, prepared)?;
        return Ok((DirectStorage::ForcedParallelCopied(storage), false));
    }
    if let PreparedAutoRetainedSelection::Selected(plan) = prepared_auto_retained {
        let storage = realize_retained_storage_for_load(ctx, gguf, plan, false)?;
        return Ok((DirectStorage::ForcedPlanned(storage), false));
    }
    if mode == GgufNoCopyMode::Disabled {
        return Ok((DirectStorage::Copied, exact_sentinel));
    }
    if !ctx.device.hasUnifiedMemory() {
        return Err(MfError::LoadPolicy(
            "forced no-copy requires a unified-memory Metal device".to_string(),
        ));
    }
    if exact_sentinel
        && expected
            .first()
            .is_none_or(|request| request.kind != ModelWeightStorageKind::Direct)
    {
        return Err(MfError::LoadPolicy(
            "exact 27B no-copy requires native token embedding residency".to_string(),
        ));
    }
    if !exact_sentinel {
        let storage = planned_retained_storage_for_load(ctx, gguf, expected, prefault_enabled)?;
        return Ok((DirectStorage::ForcedPlanned(storage), false));
    }
    let mmap = gguf.retained_shard_mmap(0).ok_or_else(|| {
        MfError::LoadPolicy("exact no-copy sentinel is missing shard 0".to_string())
    })?;
    let backing = ctx.gguf_no_copy_backing(mmap, 0, GGUF_NO_COPY_ALIGNMENT)?;
    let expected_pages = backing.exposed_len() / backing.page_size();
    if backing.required_alignment() != GGUF_NO_COPY_ALIGNMENT {
        return Err(MfError::LoadPolicy(format!(
            "forced no-copy alignment mismatch: {}/{}",
            backing.required_alignment(),
            GGUF_NO_COPY_ALIGNMENT,
        )));
    }
    let prefault = if prefault_enabled {
        let report = backing.prefault_read();
        if report.page_count != expected_pages || report.covered_bytes != backing.exposed_len() {
            return Err(MfError::LoadPolicy(format!(
                "forced no-copy prefault mismatch: pages={}/{} covered={}/{}",
                report.page_count,
                expected_pages,
                report.covered_bytes,
                backing.exposed_len(),
            )));
        }
        Some(report)
    } else {
        None
    };
    // Migrated from `eprintln!` to `qwen_diag`; consumed by v0593 (no-copy
    // demand-paged) which uses `re.match(r"^\[metal-...")` anchored regex
    // on the line body. Bare-body rendering keeps the anchor intact.
    tracing::info!(
        target: "qwen_diag",
        concat!(
            "[metal-gguf-no-copy] mapped={} exposed={} suffix={} page={} pages={} ",
            "alignment={} prefault={} prefault_pages={} prefault_ms={:.3} ",
            "checksum={:#018x}",
        ),
        backing.mapped_len(),
        backing.exposed_len(),
        backing.mapped_len() - backing.exposed_len(),
        backing.page_size(),
        expected_pages,
        backing.required_alignment(),
        if prefault_enabled {
            "enabled"
        } else {
            "disabled"
        },
        prefault.map_or(0, |report| report.page_count),
        prefault.map_or(0.0, |report| report.wall_ms),
        prefault.map_or(0, |report| report.checksum),
    );
    Ok((DirectStorage::ForcedExact27B(backing), true))
}
