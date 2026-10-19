//! Weight residency, retained storage, and loader plumbing.

use super::*;

/// Legacy common native set, including resident expert banks. Dense-only
/// additions belong in `weight_role_dtype_supported`, not this whitelist.
/// Base-model planning and loading use `weight_storage_kind_with_policy`.
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

/// The consuming graph assigns the role; tensor names/rank never grant a role.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum WeightRole {
    DenseProjection,
    MoeProjection,
    OutputHead,
    ExpertBank,
}

pub(crate) fn projection_weight_role(kind: crate::model::ArchKind) -> WeightRole {
    match kind {
        crate::model::ArchKind::Dense => WeightRole::DenseProjection,
        crate::model::ArchKind::Moe => WeightRole::MoeProjection,
    }
}

/// Executable native storage capability, independent of diagnostic rollback.
/// Embeddings and routers retain their separate policies. The IQ additions
/// are confined to dense-model projections; whole-model qualification is separate.
pub(crate) fn weight_role_dtype_supported(role: WeightRole, dtype: GgmlType) -> bool {
    weight_dtype_kept_native(dtype)
        || (role == WeightRole::DenseProjection
            && matches!(
                dtype,
                GgmlType::IQ2_XS | GgmlType::IQ2_XXS | GgmlType::IQ1_S | GgmlType::IQ1_M
            ))
}

#[cfg(test)]
thread_local! {
    static DENSE_IQ_NATIVE_STORAGE: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// Select storage policy when preparing a load on this thread. Preparation
/// freezes the choice: later scopes cannot change its plan or realization.
/// Nested scopes/unwinding restore the previous selection. Existing tensors
/// and kernel eligibility are unaffected.
#[cfg(test)]
pub(crate) fn with_dense_iq_native_storage<R>(enabled: bool, run: impl FnOnce() -> R) -> R {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            DENSE_IQ_NATIVE_STORAGE.with(|slot| slot.set(self.0));
        }
    }
    let _restore = Restore(DENSE_IQ_NATIVE_STORAGE.with(|slot| slot.replace(enabled)));
    run()
}

pub(super) fn dense_iq_native_storage_enabled() -> bool {
    #[cfg(test)]
    return DENSE_IQ_NATIVE_STORAGE.with(std::cell::Cell::get);
    #[cfg(not(test))]
    true
}

/// Storage decision using the current diagnostic selection. Prepared loads use
/// the explicit-policy helper below. Newly admitted IQ matrices must fit
/// their primitive's dimensions, ulong weight addressing and exact GGML blocks.
/// Invalid native geometry fails before allocation instead of inflating to F32.
pub(crate) fn weight_storage_kind(
    role: WeightRole,
    desc: &TensorDesc,
) -> Result<ModelWeightStorageKind, MfError> {
    weight_storage_kind_with_policy(role, desc, dense_iq_native_storage_enabled())
}

/// Checked storage decision from an already resolved policy. Planning and
/// realization must use the same frozen boolean, never reread thread-local
/// state during materialization.
pub(crate) fn weight_storage_kind_with_policy(
    role: WeightRole,
    desc: &TensorDesc,
    dense_iq_native: bool,
) -> Result<ModelWeightStorageKind, MfError> {
    let native = weight_role_dtype_supported(role, desc.dtype)
        && !(role == WeightRole::DenseProjection
            && matches!(
                desc.dtype,
                GgmlType::IQ2_XS | GgmlType::IQ2_XXS | GgmlType::IQ1_S | GgmlType::IQ1_M
            )
            && !dense_iq_native);
    if native && !weight_dtype_kept_native(desc.dtype) {
        validate_native_iq_matrix_storage(desc)?;
    }
    Ok(if native {
        ModelWeightStorageKind::Direct
    } else {
        ModelWeightStorageKind::ConvertedF32
    })
}

/// Physical matrix storage contract shared by newly native projections and
/// IQ1_M embeddings. Selection policy is resolved separately by each role.
pub(super) fn validate_native_iq_matrix_storage(desc: &TensorDesc) -> Result<(), MfError> {
    let invalid = || {
        MfError::LoadPolicy(format!(
            "native IQ {:?} has invalid {:?} matrix storage: shape={:?} bytes={}",
            desc.name, desc.dtype, desc.shape, desc.n_bytes,
        ))
    };
    let [k, m] = desc.shape.as_slice() else {
        return Err(invalid());
    };
    let (block, bytes) = desc.dtype.storage_layout().ok_or_else(invalid)?;
    let elements = k.checked_mul(*m).ok_or_else(invalid)?;
    // All families widen weight row offsets before multiplying. IQ1/XXS
    // additionally requires positive i32 dimensions and a u32 row stride;
    // XS uses u32 dimensions and a ulong row stride. No u32 bank-size cap.
    let row_bytes = (k / block).checked_mul(bytes).ok_or_else(invalid)?;
    let expected = row_bytes.checked_mul(*m).ok_or_else(invalid)?;
    if *k == 0
        || *m == 0
        || !k.is_multiple_of(block)
        || u32::try_from(*k).is_err()
        || u32::try_from(*m).is_err()
        || (matches!(
            desc.dtype,
            GgmlType::IQ2_XXS | GgmlType::IQ1_S | GgmlType::IQ1_M
        ) && (i32::try_from(*k).is_err()
            || i32::try_from(*m).is_err()
            || u32::try_from(row_bytes).is_err()))
        || usize::try_from(elements).is_err()
        || expected != desc.n_bytes
        || usize::try_from(expected).is_err()
        || desc.data_offset.checked_add(expected).is_none()
    {
        return Err(invalid());
    }
    Ok(())
}

/// Complete descriptor plans for A (old storage) and B (native IQ2_XS/IQ2_XXS/IQ1_S/IQ1_M).
/// No payload reads, allocations of model weights, or Metal context required.
#[cfg(test)]
pub(crate) fn dense_iq_native_storage_comparison_plans<'a>(
    model: &Model<'a>,
    native_quant_embedding: bool,
    router_f16: bool,
) -> Result<[Vec<ModelWeightStorageRequest<'a>>; 2], MfError> {
    let old = model_weight_storage_requests_with_policy(
        model,
        native_quant_embedding,
        router_f16,
        false,
    )?;
    let native =
        model_weight_storage_requests_with_policy(model, native_quant_embedding, router_f16, true)?;
    Ok([old, native])
}

pub(super) fn parse_gguf_owned_arena_mode(value: Option<&str>) -> Result<(), MfError> {
    match value {
        None => Ok(()),
        Some(value) if crate::env_flag::env_value_truthy(value) => Err(MfError::LoadPolicy(
            "QWEN_GGUF_OWNED_ARENA was archived after the v0.598 loaded-performance result"
                .to_string(),
        )),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(()),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_OWNED_ARENA value {value:?}"
        ))),
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

pub(super) fn auto_retained_single_pass_enabled(
    admission_enabled: bool,
    unified_memory: bool,
    no_copy_mode: GgufNoCopyMode,
    parallel_mode: GgufParallelCopyMode,
    prefault_mode: GgufNoCopyPrefaultMode,
    explicit_override_present: bool,
) -> bool {
    admission_enabled
        && unified_memory
        && matches!(
            no_copy_mode,
            GgufNoCopyMode::Disabled | GgufNoCopyMode::Auto
        )
        && parallel_mode == GgufParallelCopyMode::Auto
        && prefault_mode == GgufNoCopyPrefaultMode::Default
        && !explicit_override_present
}

pub(super) fn parse_gguf_no_copy_mode(value: Option<&str>) -> Result<GgufNoCopyMode, MfError> {
    match value {
        None => Ok(GgufNoCopyMode::Auto),
        Some(value) if crate::env_flag::env_value_truthy(value) => Ok(GgufNoCopyMode::Forced),
        Some(value) if crate::env_flag::env_value_falsy(value) => Ok(GgufNoCopyMode::Disabled),
        Some(value) => Err(MfError::LoadPolicy(format!(
            "invalid QWEN_GGUF_NO_COPY value {value:?}"
        ))),
    }
}

pub const GGUF_COPY_HEADROOM_MARGIN_PERCENT: u64 = 15;
pub const GGUF_COPY_SESSION_RESERVE_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GgufNoCopyDecision {
    pub map_weights: bool,
    pub capacity_would_map: bool,
    pub tight_copy: bool,
    pub copied_bytes: u64,
    pub auxiliary_weight_bytes: u64,
    pub working_set_bytes: u64,
    pub reason: &'static str,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GgufStorageRequest {
    Automatic,
    Copy(&'static str),
    Map(&'static str),
}

pub fn choose_gguf_no_copy_mode(
    copied_bytes: u64,
    session_reserve_bytes: Option<u64>,
    recommended_working_set_bytes: u64,
    current_allocated_bytes: u64,
) -> GgufNoCopyDecision {
    decide_gguf_storage_mode(
        copied_bytes,
        0,
        session_reserve_bytes,
        recommended_working_set_bytes,
        current_allocated_bytes,
        true,
        GgufStorageRequest::Automatic,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn decide_gguf_storage_mode(
    copied_bytes: u64,
    auxiliary_weight_bytes: u64,
    session_reserve_bytes: Option<u64>,
    recommended_working_set_bytes: u64,
    current_allocated_bytes: u64,
    unified_memory: bool,
    request: GgufStorageRequest,
    mapping_unsupported_reason: Option<&'static str>,
) -> GgufNoCopyDecision {
    decide_gguf_storage_mode_with_signals(
        copied_bytes,
        auxiliary_weight_bytes,
        session_reserve_bytes,
        crate::metal::MetalMemorySignals {
            recommended_max_bytes: recommended_working_set_bytes,
            current_allocated_bytes,
            process_limit_remaining_bytes: Some(u64::MAX),
        },
        unified_memory,
        request,
        mapping_unsupported_reason,
    )
}

pub fn decide_gguf_storage_mode_with_signals(
    copied_bytes: u64,
    auxiliary_weight_bytes: u64,
    session_reserve_bytes: Option<u64>,
    signals: crate::metal::MetalMemorySignals,
    unified_memory: bool,
    request: GgufStorageRequest,
    mapping_unsupported_reason: Option<&'static str>,
) -> GgufNoCopyDecision {
    let reserve = session_reserve_bytes.unwrap_or(GGUF_COPY_SESSION_RESERVE_BYTES);
    // Performance heuristic fitted to measurements: 76% copied decodes faster; 97% copied loads 4–6x slower with heavy paging, not an admission guarantee.
    let recommended_working_set_bytes = signals.recommended_max_bytes;
    let margin =
        recommended_working_set_bytes.saturating_mul(GGUF_COPY_HEADROOM_MARGIN_PERCENT) / 100;
    let admission = crate::metal::evaluate_metal_memory_admission_with_cpu_bytes(
        copied_bytes,
        auxiliary_weight_bytes,
        reserve.saturating_add(margin),
        signals,
        true,
    );
    let capacity_would_map = !admission.admitted;
    let (map_weights, reason) = match request {
        GgufStorageRequest::Map(reason) => (true, reason),
        GgufStorageRequest::Copy(reason) => (false, reason),
        GgufStorageRequest::Automatic if !unified_memory => {
            (false, "device_does_not_support_unified_mapping")
        }
        GgufStorageRequest::Automatic
            if capacity_would_map && mapping_unsupported_reason.is_some() =>
        {
            (false, mapping_unsupported_reason.unwrap())
        }
        GgufStorageRequest::Automatic if capacity_would_map => match admission.reason {
            crate::metal::MetalMemoryAdmissionReason::ProcessInsufficient => {
                (true, "copy_exceeds_process_headroom")
            }
            crate::metal::MetalMemoryAdmissionReason::BothInsufficient => {
                (true, "copy_exceeds_working_set_and_process_headroom")
            }
            _ => (true, "copy_exceeds_working_set_headroom"),
        },
        GgufStorageRequest::Automatic => (false, "copy_fits_with_headroom"),
    };
    GgufNoCopyDecision {
        map_weights,
        capacity_would_map,
        tight_copy: !map_weights && capacity_would_map,
        copied_bytes,
        auxiliary_weight_bytes,
        working_set_bytes: recommended_working_set_bytes,
        reason: if !capacity_would_map
            && (recommended_working_set_bytes == 0
                || admission.working_set_headroom_bytes.is_none())
            && request == GgufStorageRequest::Automatic
        {
            "working_set_signal_invalid"
        } else {
            reason
        },
    }
}

pub fn retained_mapping_unsupported_reason(
    shard_mapped_lengths: &[usize],
    direct_requests: &[&TensorDesc],
    page_size: usize,
    max_buffer_length: usize,
) -> Option<&'static str> {
    if direct_requests.is_empty() {
        return Some("automatic_mapping_unsupported_no_direct_weights");
    }
    let plan = match plan_retained_storage(
        shard_mapped_lengths,
        direct_requests,
        page_size,
        max_buffer_length,
        GGUF_NO_COPY_ALIGNMENT,
    ) {
        Ok(plan) => plan,
        Err(_) => return Some("automatic_mapping_unsupported_retained_geometry"),
    };
    plan.entries
        .iter()
        .find_map(|entry| match entry.disposition {
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::FinalPartialPage,
            }
            | RetainedStorageDisposition::View { .. }
            | RetainedStorageDisposition::Alias { .. } => None,
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::BindingMisalignment,
            } => Some("automatic_mapping_unsupported_binding_misalignment"),
            RetainedStorageDisposition::CopyFallback { .. } => {
                Some("automatic_mapping_unsupported_retained_geometry")
            }
        })
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
///   downstream Python plan scripts (byte-for-byte compatible), and
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

pub(super) fn gguf_storage_log_line(
    mode: &str,
    copied_bytes: u64,
    working_set_bytes: u64,
    reason: &str,
) -> String {
    format!(
        "[metal-gguf-storage] mode={mode} copied_bytes={copied_bytes} working_set_bytes={working_set_bytes} reason={reason}"
    )
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

/// Digest the complete host-specific plan for benchmark report validation.
/// Runtime qualification uses artifact and inventory digests instead.
pub fn retained_storage_host_report_digest(plan: &RetainedStoragePlan) -> String {
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

#[cfg(test)]
pub(super) fn push_native_weight_request<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    role: WeightRole,
    desc: &'a TensorDesc,
) -> Result<(), MfError> {
    push_native_weight_request_with_policy(requests, role, desc, dense_iq_native_storage_enabled())
}

pub(super) fn push_native_weight_request_with_policy<'a>(
    requests: &mut Vec<ModelWeightStorageRequest<'a>>,
    role: WeightRole,
    desc: &'a TensorDesc,
    dense_iq_native: bool,
) -> Result<(), MfError> {
    push_model_weight_request(
        requests,
        desc,
        weight_storage_kind_with_policy(role, desc, dense_iq_native)?,
    )
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
    model_weight_storage_requests_with_policy(
        model,
        native_quant_embedding,
        router_f16,
        dense_iq_native_storage_enabled(),
    )
}

pub(crate) fn model_weight_storage_requests_with_policy<'a>(
    model: &Model<'a>,
    native_quant_embedding: bool,
    router_f16: bool,
    dense_iq_native: bool,
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
    if native_quant_embedding && model.token_embd.dtype == GgmlType::IQ1_M {
        validate_native_iq_matrix_storage(model.token_embd)?;
    }
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
    push_native_weight_request_with_policy(
        &mut requests,
        WeightRole::OutputHead,
        model.lm_head,
        dense_iq_native,
    )?;

    let projection_role = projection_weight_role(model.arch.kind);
    for block in &model.blocks {
        match block {
            Block::Gdn(gdn) => {
                push_f32_weight_request(&mut requests, gdn.attn_norm)?;
                push_f32_weight_request(&mut requests, gdn.post_attention_norm)?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.ffn_gate,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.ffn_up,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.ffn_down,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.in_proj_qkv,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.in_proj_z,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.beta_proj,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.alpha_proj,
                    dense_iq_native,
                )?;
                push_f32_weight_request(&mut requests, gdn.a_log)?;
                push_f32_weight_request(&mut requests, gdn.dt_bias)?;
                push_f32_weight_request(&mut requests, gdn.conv1d)?;
                push_f32_weight_request(&mut requests, gdn.norm)?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    gdn.out_proj,
                    dense_iq_native,
                )?;
                if let Some(moe) = gdn.ffn_moe.as_ref() {
                    push_moe_weight_requests(&mut requests, moe, router_f16, dense_iq_native)?;
                }
            }
            Block::Attn(attn) => {
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.q,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.k,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.v,
                    dense_iq_native,
                )?;
                push_f32_weight_request(&mut requests, attn.attn_norm)?;
                push_f32_weight_request(&mut requests, attn.post_attention_norm)?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.ffn_gate,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.ffn_up,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.ffn_down,
                    dense_iq_native,
                )?;
                push_native_weight_request_with_policy(
                    &mut requests,
                    projection_role,
                    attn.o,
                    dense_iq_native,
                )?;
                push_f32_weight_request(&mut requests, attn.q_norm)?;
                push_f32_weight_request(&mut requests, attn.k_norm)?;
                if let Some(moe) = attn.ffn_moe.as_ref() {
                    push_moe_weight_requests(&mut requests, moe, router_f16, dense_iq_native)?;
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
    plan: &ParallelCopyLoadPlan,
    destination_length: ParallelDestinationLength,
    expected: &[ModelWeightStorageIdentity],
    resources: &[Buffer],
    tensors: &[MetalTensor],
) -> Result<(), MfError> {
    if expected.len() != plan.request_count
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
            plan.page_size,
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
        parallel_destination_accounting(expected, destination_length, plan.page_size, usize::MAX)?;
    let accounting_matches = expected_resource_bytes >= plan.source_bytes
        && (destination_length == ParallelDestinationLength::PageRounded
            || (expected_resource_bytes == plan.source_bytes && padded_resources == 0));
    if resource_bytes != expected_resource_bytes || !accounting_matches {
        return Err(MfError::LoadPolicy(format!(
            "parallel-copy resource accounting drifted: actual={resource_bytes} expected={expected_resource_bytes} padded={padded_resources}"
        )));
    }
    Ok(())
}

pub(super) fn planned_parallel_storage_for_load(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Model<'_>,
    expected: &[ModelWeightStorageRequest<'_>],
    embedding_selection: NativeQuantEmbeddingSelection,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<PlannedParallelCopiedStorage, MfError> {
    let plan = build_parallel_copy_plan(ctx, gguf, model, expected, embedding_selection)?;
    let prepared = prepare_parallel_copy_plan(ctx, expected, plan, population, destination_length)?;
    realize_parallel_copy_plan(ctx, gguf, expected, prepared)
}

pub(super) fn validate_parallel_schedule_memory_admission(
    ctx: &MetalContext,
    plan: &ParallelCopyLoadPlan,
    phase: &str,
) -> Result<(), MfError> {
    let required_headroom = plan
        .destination_bytes
        .checked_add(plan.fallback_bytes)
        .and_then(|bytes| bytes.checked_add(GGUF_COPY_SESSION_RESERVE_BYTES))
        .ok_or_else(|| MfError::LoadPolicy("parallel schedule headroom overflow".to_string()))?;
    let admission =
        evaluate_metal_memory_admission(required_headroom, 0, ctx.memory_signals(), true);
    emit_metal_load_line(format_args!(
        concat!(
            "[metal-gguf-parallel-memory] plan={} phase={} admitted={} reason={} ",
            "required={} recommended={} current={} process_remaining={:?}"
        ),
        "descriptor-plan-v1",
        phase,
        admission.admitted,
        admission.reason.as_str(),
        required_headroom,
        admission.signals.recommended_max_bytes,
        admission.signals.current_allocated_bytes,
        admission.signals.process_limit_remaining_bytes,
    ));
    if !admission.admitted {
        return Err(MfError::LoadPolicy(format!(
            "parallel population memory admission failed during {phase}: {}",
            admission.reason.as_str()
        )));
    }
    Ok(())
}

pub(super) fn prepare_parallel_copy_plan(
    ctx: &MetalContext,
    expected: &[ModelWeightStorageRequest<'_>],
    mut plan: ParallelCopyLoadPlan,
    population: ParallelPopulationMethod,
    destination_length: ParallelDestinationLength,
) -> Result<PreparedParallelCopyPlan, MfError> {
    validate_parallel_destination_length(&plan, population, destination_length)?;
    let expected_identities = expected
        .iter()
        .map(expected_model_weight_identity)
        .collect::<Vec<_>>();
    let mut destination_bytes = 0u64;
    let mut padded_resources = 0usize;
    for identity in &expected_identities {
        let logical_length = parallel_destination_resource_length(
            identity.source_bytes,
            destination_length,
            plan.page_size,
            ctx.max_buffer_length(),
        )?;
        let priced = ctx
            .shared_buffer_size_and_align(logical_length as u64)
            .map_err(MfError::Metal)?;
        if priced.size > ctx.max_buffer_length() as u64 {
            return Err(MfError::LoadPolicy(format!(
                "parallel destination {} exceeds max buffer {}",
                priced.size,
                ctx.max_buffer_length()
            )));
        }
        destination_bytes = destination_bytes.checked_add(priced.size).ok_or_else(|| {
            MfError::LoadPolicy("parallel destination bytes overflow".to_string())
        })?;
        padded_resources += usize::from(logical_length as u64 != identity.source_bytes);
    }
    plan.destination_bytes = destination_bytes;
    plan.padding_bytes = destination_bytes.saturating_sub(plan.source_bytes);
    plan.padded_resources = padded_resources;
    plan.max_buffer_length = ctx.max_buffer_length();
    validate_parallel_schedule_memory_admission(ctx, &plan, "prepare")?;
    let sorted_request_indices = frozen_parallel_copy_order(&plan, &expected_identities)?;
    Ok(PreparedParallelCopyPlan {
        plan,
        population,
        destination_length,
        expected_identities,
        sorted_request_indices,
    })
}

pub(super) fn realize_parallel_copy_plan(
    ctx: &MetalContext,
    gguf: &GgufFile,
    expected: &[ModelWeightStorageRequest<'_>],
    prepared: PreparedParallelCopyPlan,
) -> Result<PlannedParallelCopiedStorage, MfError> {
    let PreparedParallelCopyPlan {
        plan,
        population,
        destination_length,
        expected_identities,
        sorted_request_indices,
    } = prepared;
    validate_model_weight_request_sequence(&expected_identities, expected)?;
    validate_parallel_schedule_memory_admission(ctx, &plan, "realize")?;
    let source_stamps = if population == ParallelPopulationMethod::Pread {
        Some(gguf.revalidate_retained_shard_stamps().map_err(|error| {
            MfError::LoadPolicy(format!("parallel pread source preflight failed: {error}"))
        })?)
    } else {
        None
    };
    let usage_before = Some(capture_parallel_copy_usage()?);
    let proc_before = Some(capture_parallel_copy_proc_usage()?);
    let ready_started = std::time::Instant::now();
    let resources_result = expected
        .iter()
        .map(|request| {
            let length = parallel_destination_resource_length(
                request.desc.n_bytes,
                destination_length,
                plan.page_size,
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
            plan.page_size,
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
        plan.page_size,
        ctx.max_buffer_length(),
    )?;
    let destination_accounting_matches = expected_destination_bytes >= plan.source_bytes
        && (destination_length == ParallelDestinationLength::PageRounded
            || (expected_destination_bytes == plan.source_bytes && padded_resources == 0));
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
        let mut handles = Vec::with_capacity(plan.task_counts.len());
        let mut spawn_error = None;
        let mut partition_error = None;
        let mut assigned_tasks = 0usize;
        for worker in 0..plan.task_counts.len() {
            let count = plan.task_counts[worker];
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
        if assigned_tasks != plan.request_count {
            return Err(MfError::LoadPolicy(
                "parallel-copy workers did not consume every task".to_string(),
            ));
        }
        Ok(())
    });
    copy_result?;
    if let Some(before) = source_stamps {
        let after = gguf.revalidate_retained_shard_stamps().map_err(|error| {
            MfError::LoadPolicy(format!("parallel pread source postflight failed: {error}"))
        })?;
        if after != before {
            return Err(MfError::LoadPolicy(
                "parallel pread source stamps changed during population".to_string(),
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
        &plan,
        destination_length,
        &expected_identities,
        &resources,
        &tensors,
    )?;
    let binding_finished = std::time::Instant::now();
    let usage_after = Some(capture_parallel_copy_usage()?);
    let proc_after = Some(capture_parallel_copy_proc_usage()?);

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
        ctx,
        &plan,
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
        plan,
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
    configuration: DirectStorageConfiguration,
) -> Result<DirectStorage, MfError> {
    let DirectStorageConfiguration {
        no_copy_mode: mode,
        prefault_enabled,
        parallel_mode,
        auto_parallel: prepared_auto,
        auto_retained: prepared_auto_retained,
        embedding_selection,
    } = configuration;
    if let Some((population, destination_length)) = parallel_mode.forced_configuration() {
        let storage = planned_parallel_storage_for_load(
            ctx,
            gguf,
            model,
            expected,
            embedding_selection,
            population,
            destination_length,
        )?;
        return Ok(DirectStorage::ForcedParallelCopied(storage));
    }
    if let PreparedAutoSelection::Selected(prepared) = prepared_auto {
        let storage = realize_parallel_copy_plan(ctx, gguf, expected, prepared)?;
        return Ok(DirectStorage::ForcedParallelCopied(storage));
    }
    if let PreparedAutoRetainedSelection::Selected(plan) = prepared_auto_retained {
        let storage = realize_retained_storage_for_load(ctx, gguf, plan, false)?;
        return Ok(DirectStorage::ForcedPlanned(storage));
    }
    if mode == GgufNoCopyMode::Disabled {
        return Ok(DirectStorage::Copied);
    }
    if !ctx.device.hasUnifiedMemory() {
        return Err(MfError::LoadPolicy(
            "forced no-copy requires a unified-memory Metal device".to_string(),
        ));
    }
    let storage = planned_retained_storage_for_load(ctx, gguf, expected, prefault_enabled)?;
    Ok(DirectStorage::ForcedPlanned(storage))
}
