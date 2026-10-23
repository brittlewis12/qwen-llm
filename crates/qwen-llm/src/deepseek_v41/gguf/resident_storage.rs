use super::*;

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
/// Uses only the already-opened GGUF's metadata/descriptors and mapped lengths.
/// No tensor payload reads, new mappings, prefetch, or Metal context are needed.
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
        bytes: p.bytes,
        windows: Vec::new(),
        metal_view_bytes: 0,
        private_shadow_bytes: 0,
        shared_eof_window_count: 0,
        private_window_count: 0,
    };
    for segment in segments.values() {
        let retained =
            metal::plan_retained_storage(&lengths, segment, page_size, max_buffer_length, 16)?;
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
