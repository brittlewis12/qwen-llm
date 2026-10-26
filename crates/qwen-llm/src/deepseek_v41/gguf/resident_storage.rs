use super::*;

// Shared selection for metadata and loading; the generic Metal wrapper stays None.
pub(super) const SOFT_COALESCING_TARGET: Option<usize> = None;

/// Mapping ownership selected by the same geometry used during load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentMappingKind {
    /// Reuse the original read-only shared shard mapping, with absolute offsets.
    OriginalShared,
    /// Read-only shared window including the zero-filled partial EOF page.
    SharedEof,
    /// Private COW window clearing disk Engram bytes on its boundary pages.
    PrivateBoundary { bytes: u64 },
}

impl ResidentMappingKind {
    /// Full private span potentially shadowed on import, not just CPU-cleared bytes.
    pub fn private_bytes(self) -> u64 {
        match self {
            Self::PrivateBoundary { bytes } => bytes,
            Self::OriginalShared | Self::SharedEof => 0,
        }
    }
}

impl From<storage::MappingKind> for ResidentMappingKind {
    fn from(kind: storage::MappingKind) -> Self {
        match kind {
            storage::MappingKind::OriginalShared => Self::OriginalShared,
            storage::MappingKind::SharedEof => Self::SharedEof,
            storage::MappingKind::PrivateBoundary { bytes } => Self::PrivateBoundary { bytes },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentStorageWindow {
    pub window: metal::RetainedStorageWindow,
    pub mapping_kind: ResidentMappingKind,
}

/// Storage geometry, not a prediction of physical residency or runtime fit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentStoragePlan {
    pub page_size: usize,
    pub max_buffer_length: usize,
    pub soft_coalescing_target: Option<usize>,
    /// Serialized resident payload and disk-only Engram bytes, without page padding.
    pub bytes: ByteCounters,
    /// Ordered exactly as the loader imports them; boundary pages may repeat.
    pub windows: Vec<ResidentStorageWindow>,
    /// Sum of entire Metal window lengths, including padding and repeated pages.
    pub metal_view_bytes: u64,
    /// Sum of entire private window lengths; shared EOF windows contribute zero.
    /// This is additional exposure, not a subset to subtract from Metal view bytes.
    pub private_shadow_bytes: u64,
    pub shared_eof_window_count: usize,
    pub private_window_count: usize,
}

/// Plan resident storage without importing weights or probing hardware.
///
/// Uses metadata/descriptors and mapped lengths, plus bounded CPU Engram control
/// payload reads when serialized as tensors. A declared companion is opened and
/// retained by the plan, but no lookup-table payload is read or imported.
/// No prefetch or Metal context is needed.
/// The supplied geometry must match the eventual loader's host/device geometry.
/// Opening `GgufFile` itself retains shared shard mappings and reads headers.
/// Files must remain immutable; load still revalidates retained shard stamps.
/// Runtime allocations, driver overhead, and memory admission are not included.
pub fn resident_storage_plan(
    g: &GgufFile,
    page_size: usize,
    max_buffer_length: usize,
) -> Result<ResidentStoragePlan, LoadError> {
    let p = plan(g)?;
    for_plan(g, &p, page_size, max_buffer_length)
}

pub(super) fn for_plan(
    g: &GgufFile,
    p: &Plan,
    page_size: usize,
    max_buffer_length: usize,
) -> Result<ResidentStoragePlan, LoadError> {
    for_plan_with_target(g, p, page_size, max_buffer_length, SOFT_COALESCING_TARGET)
}

pub(super) fn for_plan_with_target(
    g: &GgufFile,
    p: &Plan,
    page_size: usize,
    max_buffer_length: usize,
    soft_target: Option<usize>,
) -> Result<ResidentStoragePlan, LoadError> {
    let lengths = storage::padded_lengths(g, page_size)?;
    // Do not coalesce across disk-only or unused tensors (including MTP).
    let mut segments = BTreeMap::<(usize, usize), Vec<&TensorDesc>>::new();
    for t in g.tensors.iter().filter(|t| resident_tensor(t, p.layers)) {
        let shard = g
            .shards
            .get(t.shard_idx)
            .ok_or_else(|| invalid("missing resident tensor shard"))?;
        if add(t.data_offset, t.n_bytes)? > shard.mmap_len() as u64 {
            return Err(invalid("resident tensor extends beyond raw EOF"));
        }
        let segment = g
            .tensors
            .iter()
            .filter(|other| {
                !resident_tensor(other, p.layers)
                    && other.shard_idx == t.shard_idx
                    && other.data_offset < t.data_offset
            })
            .count();
        segments.entry((t.shard_idx, segment)).or_default().push(t);
    }
    let mut result = ResidentStoragePlan {
        page_size,
        max_buffer_length,
        soft_coalescing_target: soft_target,
        bytes: p.bytes,
        windows: Vec::new(),
        metal_view_bytes: 0,
        private_shadow_bytes: 0,
        shared_eof_window_count: 0,
        private_window_count: 0,
    };
    for segment in segments.values() {
        let retained = metal::plan_retained_storage_with_soft_target(
            &lengths,
            segment,
            page_size,
            max_buffer_length,
            16,
            soft_target,
        )?;
        if retained.unique_fallback_bytes != 0 {
            return Err(invalid(
                "tensor cannot be directly bound: device maximum buffer length or binding alignment",
            ));
        }
        for window in retained.windows {
            let kind = storage::mapping_kind(g, &window, page_size)?;
            result.metal_view_bytes = add(result.metal_view_bytes, window.length as u64)?;
            result.private_shadow_bytes = add(result.private_shadow_bytes, kind.private_bytes())?;
            match kind {
                storage::MappingKind::SharedEof => result.shared_eof_window_count += 1,
                storage::MappingKind::PrivateBoundary { .. } => result.private_window_count += 1,
                storage::MappingKind::OriginalShared => (),
            }
            result.windows.push(ResidentStorageWindow {
                window,
                mapping_kind: kind.into(),
            });
        }
    }
    Ok(result)
}

#[derive(Clone, Debug)]
pub struct ResidentTensorExtent {
    pub tensor: TensorDesc,
    pub page_rounded_singleton_bytes: u64,
}

/// Bounded descriptor audit using the loader's private resident predicate.
#[derive(Clone, Debug)]
pub struct ResidentTensorAudit {
    pub count: usize,
    pub payload_bytes: u64,
    pub largest_payload_bytes: u64,
    pub largest_page_rounded_singleton_bytes: u64,
    pub above_soft_target_count: usize,
    pub above_soft_target_payload_bytes: u64,
    pub top: Vec<ResidentTensorExtent>,
}

pub struct ResidentStorageAudit {
    pub selected_soft_coalescing_target: Option<usize>,
    pub control: ResidentStoragePlan,
    pub candidate: ResidentStoragePlan,
    pub tensors: ResidentTensorAudit,
}

/// Fixed None/4 GiB comparison, not a runtime selection or admission API.
/// Like resident_storage_plan, this may read bounded CPU control payloads and
/// open a declared companion's headers, but never reads weight/table payloads
/// or imports Metal mappings.
pub fn resident_storage_audit(
    g: &GgufFile,
    page_size: usize,
    max_buffer_length: usize,
) -> Result<ResidentStorageAudit, LoadError> {
    let p = plan(g)?;
    let target = 4_294_967_296;
    let control = for_plan_with_target(g, &p, page_size, max_buffer_length, None)?;
    let candidate = for_plan_with_target(g, &p, page_size, max_buffer_length, Some(target))?;
    let effective = (target / page_size * page_size)
        .max(page_size)
        .min(max_buffer_length / page_size * page_size) as u64;
    let tensors = audit_tensors(g, &p, page_size, effective)?;
    Ok(ResidentStorageAudit {
        selected_soft_coalescing_target: SOFT_COALESCING_TARGET,
        control,
        candidate,
        tensors,
    })
}

pub(super) fn audit_tensors(
    g: &GgufFile,
    p: &Plan,
    page_size: usize,
    effective_soft_target: u64,
) -> Result<ResidentTensorAudit, LoadError> {
    let mut tensors = ResidentTensorAudit {
        count: 0,
        payload_bytes: 0,
        largest_payload_bytes: 0,
        largest_page_rounded_singleton_bytes: 0,
        above_soft_target_count: 0,
        above_soft_target_payload_bytes: 0,
        top: Vec::with_capacity(11),
    };
    for t in g.tensors.iter().filter(|t| resident_tensor(t, p.layers)) {
        let start = t.data_offset / page_size as u64 * page_size as u64;
        let end = add(t.data_offset, t.n_bytes)?;
        let rounded_end = add(end, page_size as u64 - 1)? / page_size as u64 * page_size as u64;
        let singleton = rounded_end - start;
        tensors.count += 1;
        tensors.payload_bytes = add(tensors.payload_bytes, t.n_bytes)?;
        tensors.largest_payload_bytes = tensors.largest_payload_bytes.max(t.n_bytes);
        tensors.largest_page_rounded_singleton_bytes =
            tensors.largest_page_rounded_singleton_bytes.max(singleton);
        if singleton > effective_soft_target {
            tensors.above_soft_target_count += 1;
            tensors.above_soft_target_payload_bytes =
                add(tensors.above_soft_target_payload_bytes, t.n_bytes)?;
        }
        tensors.top.push(ResidentTensorExtent {
            tensor: t.clone(),
            page_rounded_singleton_bytes: singleton,
        });
        tensors.top.sort_by(|a, b| {
            b.tensor
                .n_bytes
                .cmp(&a.tensor.n_bytes)
                .then_with(|| a.tensor.name.cmp(&b.tensor.name))
                .then_with(|| a.tensor.shard_idx.cmp(&b.tensor.shard_idx))
                .then_with(|| a.tensor.data_offset.cmp(&b.tensor.data_offset))
        });
        tensors.top.truncate(10);
    }
    Ok(tensors)
}
