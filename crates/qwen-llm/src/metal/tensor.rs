//! MetalTensor: typed buffer views, GGUF backing, retained storage plans.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GgufBackingEligibility {
    Eligible,
    FinalPartialPage,
    WrongShard,
    BindingMisalignment,
    OutsideBacking,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct GgufBackingGeometry {
    pub(crate) shard_idx: usize,
    pub(crate) mapped_len: usize,
    pub(crate) mmap_offset: usize,
    pub(crate) exposed_len: usize,
    pub(crate) page_size: usize,
    pub(crate) required_alignment: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedStorageFallback {
    MissingShard,
    BindingMisalignment,
    OutsideShard,
    FinalPartialPage,
    TensorExceedsWindow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RetainedStorageDisposition {
    View {
        window_index: usize,
        buffer_offset: u64,
    },
    Alias {
        source_request_index: usize,
    },
    CopyFallback {
        reason: RetainedStorageFallback,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedStorageWindow {
    pub shard_idx: usize,
    pub mmap_offset: u64,
    pub length: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedStorageEntry {
    pub request_index: usize,
    pub name: String,
    pub shard_idx: usize,
    pub data_offset: u64,
    pub n_bytes: u64,
    pub disposition: RetainedStorageDisposition,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RetainedStoragePlan {
    pub page_size: usize,
    pub max_buffer_length: usize,
    pub usable_window_length: usize,
    pub required_alignment: usize,
    pub windows: Vec<RetainedStorageWindow>,
    pub entries: Vec<RetainedStorageEntry>,
    pub unique_view_bytes: u64,
    pub logical_view_bytes: u64,
    pub unique_fallback_bytes: u64,
    pub alias_bytes: u64,
}

#[derive(Clone)]
pub(crate) struct CanonicalRetainedRequest<'a> {
    pub(crate) desc: &'a TensorDesc,
    pub(crate) source_request_index: usize,
    pub(crate) disposition: Option<RetainedStorageDisposition>,
}

pub(crate) fn checked_page_floor(value: usize, page_size: usize) -> usize {
    value / page_size * page_size
}

pub(crate) fn checked_page_ceil(value: usize, page_size: usize) -> Result<usize, MetalError> {
    value
        .checked_add(page_size - 1)
        .map(|end| end / page_size * page_size)
        .ok_or_else(|| MetalError::GgufNoCopy("page-rounded endpoint overflow".to_string()))
}

pub fn plan_retained_storage(
    shard_mapped_lengths: &[usize],
    requests: &[&TensorDesc],
    page_size: usize,
    max_buffer_length: usize,
    required_alignment: usize,
) -> Result<RetainedStoragePlan, MetalError> {
    if !page_size.is_power_of_two() {
        return Err(MetalError::GgufNoCopy(format!(
            "host page size {page_size} is not a power of two"
        )));
    }
    if !required_alignment.is_power_of_two() {
        return Err(MetalError::GgufNoCopy(format!(
            "required binding alignment {required_alignment} is not a power of two"
        )));
    }
    if !page_size.is_multiple_of(required_alignment) {
        return Err(MetalError::GgufNoCopy(format!(
            "required binding alignment {required_alignment} does not divide page size {page_size}"
        )));
    }
    let usable_window_length = max_buffer_length / page_size * page_size;
    if usable_window_length == 0 {
        return Err(MetalError::GgufNoCopy(format!(
            "Metal maxBufferLength {max_buffer_length} exposes no complete {page_size}-byte page"
        )));
    }

    let mut canonical = Vec::<CanonicalRetainedRequest<'_>>::new();
    let mut by_source = HashMap::<(usize, u64, u64), usize>::new();
    let mut request_to_canonical = Vec::with_capacity(requests.len());
    for (request_index, desc) in requests.iter().copied().enumerate() {
        let (_, expected) = checked_ggml_shape_bytes(&desc.shape, desc.dtype)?;
        let declared = usize::try_from(desc.n_bytes).map_err(|_| {
            MetalError::GgufNoCopy(format!(
                "tensor {:?} n_bytes {} does not fit usize",
                desc.name, desc.n_bytes
            ))
        })?;
        if expected != declared {
            return Err(MetalError::GgufNoCopy(format!(
                "tensor {:?} shape/dtype expects {expected} bytes, descriptor declares {declared}",
                desc.name
            )));
        }
        if declared == 0 {
            return Err(MetalError::GgufNoCopy(format!(
                "tensor {:?} has an empty storage range",
                desc.name
            )));
        }
        let key = (desc.shard_idx, desc.data_offset, desc.n_bytes);
        if let Some(&canonical_index) = by_source.get(&key) {
            let existing = &canonical[canonical_index];
            if existing.desc.dtype != desc.dtype || existing.desc.shape != desc.shape {
                return Err(MetalError::GgufNoCopy(format!(
                    "aliased tensor {:?} is not representation-compatible with {:?}",
                    desc.name, existing.desc.name
                )));
            }
            request_to_canonical.push(canonical_index);
            continue;
        }
        let canonical_index = canonical.len();
        by_source.insert(key, canonical_index);
        canonical.push(CanonicalRetainedRequest {
            desc,
            source_request_index: request_index,
            disposition: None,
        });
        request_to_canonical.push(canonical_index);
    }

    let mut sorted = (0..canonical.len()).collect::<Vec<_>>();
    sorted.sort_unstable_by_key(|&index| {
        let desc = canonical[index].desc;
        (desc.shard_idx, desc.data_offset, desc.n_bytes)
    });

    let mut windows = Vec::<RetainedStorageWindow>::new();
    let mut last_window_by_shard = HashMap::<usize, usize>::new();
    let mut previous_end_by_shard = HashMap::<usize, (usize, String)>::new();
    for canonical_index in sorted {
        let desc = canonical[canonical_index].desc;
        let Some(&mapped_len) = shard_mapped_lengths.get(desc.shard_idx) else {
            canonical[canonical_index].disposition =
                Some(RetainedStorageDisposition::CopyFallback {
                    reason: RetainedStorageFallback::MissingShard,
                });
            continue;
        };
        let start = usize::try_from(desc.data_offset).map_err(|_| {
            MetalError::GgufNoCopy(format!(
                "tensor {:?} offset {} does not fit usize",
                desc.name, desc.data_offset
            ))
        })?;
        let length = usize::try_from(desc.n_bytes).map_err(|_| {
            MetalError::GgufNoCopy(format!(
                "tensor {:?} n_bytes {} does not fit usize",
                desc.name, desc.n_bytes
            ))
        })?;
        let end = start.checked_add(length).ok_or_else(|| {
            MetalError::GgufNoCopy(format!("tensor {:?} range overflows usize", desc.name))
        })?;
        if let Some((previous_end, previous_name)) = previous_end_by_shard.get(&desc.shard_idx)
            && start < *previous_end
        {
            return Err(MetalError::GgufNoCopy(format!(
                "tensor {:?} range starts at {start} before prior tensor {:?} ends at {previous_end}",
                desc.name, previous_name
            )));
        }
        previous_end_by_shard.insert(desc.shard_idx, (end, desc.name.clone()));
        let candidate_window_start = checked_page_floor(start, page_size);
        let candidate_buffer_offset = start - candidate_window_start;
        if !candidate_buffer_offset.is_multiple_of(required_alignment) {
            canonical[canonical_index].disposition =
                Some(RetainedStorageDisposition::CopyFallback {
                    reason: RetainedStorageFallback::BindingMisalignment,
                });
            continue;
        }
        if end > mapped_len {
            canonical[canonical_index].disposition =
                Some(RetainedStorageDisposition::CopyFallback {
                    reason: RetainedStorageFallback::OutsideShard,
                });
            continue;
        }
        let rounded_end = checked_page_ceil(end, page_size)?;
        let candidate_window_length = rounded_end
            .checked_sub(candidate_window_start)
            .ok_or_else(|| MetalError::GgufNoCopy("planned window range underflow".to_string()))?;
        if candidate_window_length > usable_window_length {
            canonical[canonical_index].disposition =
                Some(RetainedStorageDisposition::CopyFallback {
                    reason: RetainedStorageFallback::TensorExceedsWindow,
                });
            continue;
        }
        let exposed_len = mapped_len / page_size * page_size;
        if end > exposed_len {
            canonical[canonical_index].disposition =
                Some(RetainedStorageDisposition::CopyFallback {
                    reason: RetainedStorageFallback::FinalPartialPage,
                });
            continue;
        }

        let mut selected_window = None;
        if let Some(&window_index) = last_window_by_shard.get(&desc.shard_idx) {
            let window = &windows[window_index];
            let window_start = usize::try_from(window.mmap_offset).map_err(|_| {
                MetalError::GgufNoCopy("planned window offset does not fit usize".to_string())
            })?;
            let required_length = rounded_end.checked_sub(window_start);
            if start >= window_start
                && required_length.is_some_and(|bytes| bytes <= usable_window_length)
            {
                selected_window = Some(window_index);
            }
        }
        let window_index = if let Some(window_index) = selected_window {
            let window_start =
                usize::try_from(windows[window_index].mmap_offset).map_err(|_| {
                    MetalError::GgufNoCopy("planned window offset does not fit usize".to_string())
                })?;
            windows[window_index].length = rounded_end - window_start;
            window_index
        } else {
            let window_index = windows.len();
            windows.push(RetainedStorageWindow {
                shard_idx: desc.shard_idx,
                mmap_offset: candidate_window_start as u64,
                length: candidate_window_length,
            });
            last_window_by_shard.insert(desc.shard_idx, window_index);
            window_index
        };
        let window_start = windows[window_index].mmap_offset;
        let buffer_offset = desc.data_offset.checked_sub(window_start).ok_or_else(|| {
            MetalError::GgufNoCopy("tensor offset precedes planned window".to_string())
        })?;
        debug_assert_eq!(buffer_offset % required_alignment as u64, 0);
        canonical[canonical_index].disposition = Some(RetainedStorageDisposition::View {
            window_index,
            buffer_offset,
        });
    }

    let mut entries = Vec::with_capacity(requests.len());
    let mut unique_view_bytes = 0u64;
    let mut logical_view_bytes = 0u64;
    let mut unique_fallback_bytes = 0u64;
    let mut alias_bytes = 0u64;
    for (request_index, desc) in requests.iter().copied().enumerate() {
        let canonical_request = &canonical[request_to_canonical[request_index]];
        let source_disposition = canonical_request
            .disposition
            .expect("every canonical retained request must be classified");
        let disposition = if request_index == canonical_request.source_request_index {
            match source_disposition {
                RetainedStorageDisposition::View { .. } => {
                    unique_view_bytes =
                        unique_view_bytes.checked_add(desc.n_bytes).ok_or_else(|| {
                            MetalError::GgufNoCopy(
                                "unique view byte accounting overflow".to_string(),
                            )
                        })?;
                    logical_view_bytes =
                        logical_view_bytes
                            .checked_add(desc.n_bytes)
                            .ok_or_else(|| {
                                MetalError::GgufNoCopy(
                                    "logical view byte accounting overflow".to_string(),
                                )
                            })?;
                }
                RetainedStorageDisposition::CopyFallback { .. } => {
                    unique_fallback_bytes = unique_fallback_bytes
                        .checked_add(desc.n_bytes)
                        .ok_or_else(|| {
                            MetalError::GgufNoCopy(
                                "unique fallback byte accounting overflow".to_string(),
                            )
                        })?;
                }
                RetainedStorageDisposition::Alias { .. } => unreachable!(),
            }
            source_disposition
        } else {
            alias_bytes = alias_bytes.checked_add(desc.n_bytes).ok_or_else(|| {
                MetalError::GgufNoCopy("alias byte accounting overflow".to_string())
            })?;
            if matches!(source_disposition, RetainedStorageDisposition::View { .. }) {
                logical_view_bytes =
                    logical_view_bytes
                        .checked_add(desc.n_bytes)
                        .ok_or_else(|| {
                            MetalError::GgufNoCopy(
                                "logical view byte accounting overflow".to_string(),
                            )
                        })?;
            }
            RetainedStorageDisposition::Alias {
                source_request_index: canonical_request.source_request_index,
            }
        };
        entries.push(RetainedStorageEntry {
            request_index,
            name: desc.name.clone(),
            shard_idx: desc.shard_idx,
            data_offset: desc.data_offset,
            n_bytes: desc.n_bytes,
            disposition,
        });
    }

    Ok(RetainedStoragePlan {
        page_size,
        max_buffer_length,
        usable_window_length,
        required_alignment,
        windows,
        entries,
        unique_view_bytes,
        logical_view_bytes,
        unique_fallback_bytes,
        alias_bytes,
    })
}

impl GgufBackingGeometry {
    pub(crate) fn new(
        shard_idx: usize,
        mapped_len: usize,
        page_size: usize,
        required_alignment: usize,
    ) -> Result<Self, MetalError> {
        if !page_size.is_power_of_two() {
            return Err(MetalError::GgufNoCopy(format!(
                "host page size {page_size} is not a power of two"
            )));
        }
        let exposed_len = mapped_len / page_size * page_size;
        Self::new_window(
            shard_idx,
            mapped_len,
            0,
            exposed_len,
            page_size,
            required_alignment,
        )
    }

    pub(crate) fn new_window(
        shard_idx: usize,
        mapped_len: usize,
        mmap_offset: usize,
        exposed_len: usize,
        page_size: usize,
        required_alignment: usize,
    ) -> Result<Self, MetalError> {
        if !page_size.is_power_of_two() {
            return Err(MetalError::GgufNoCopy(format!(
                "host page size {page_size} is not a power of two"
            )));
        }
        if !required_alignment.is_power_of_two() {
            return Err(MetalError::GgufNoCopy(format!(
                "required binding alignment {required_alignment} is not a power of two"
            )));
        }
        if !page_size.is_multiple_of(required_alignment) {
            return Err(MetalError::GgufNoCopy(format!(
                "required binding alignment {required_alignment} does not divide \
                 page size {page_size}"
            )));
        }
        if !mmap_offset.is_multiple_of(page_size) {
            return Err(MetalError::GgufNoCopy(format!(
                "window offset {mmap_offset} is not page aligned to {page_size}"
            )));
        }
        if exposed_len == 0 {
            return Err(MetalError::GgufNoCopy(format!(
                "window length must contain at least one complete {page_size}-byte page"
            )));
        }
        if !exposed_len.is_multiple_of(page_size) {
            return Err(MetalError::GgufNoCopy(format!(
                "window length {exposed_len} is not page aligned to {page_size}"
            )));
        }
        let window_end = mmap_offset.checked_add(exposed_len).ok_or_else(|| {
            MetalError::GgufNoCopy("window offset and length overflow usize".to_string())
        })?;
        if window_end > mapped_len {
            return Err(MetalError::GgufNoCopy(format!(
                "window range {mmap_offset}..{window_end} exceeds mapped length {mapped_len}"
            )));
        }
        Ok(Self {
            shard_idx,
            mapped_len,
            mmap_offset,
            exposed_len,
            page_size,
            required_alignment,
        })
    }

    pub(crate) fn mapped_len(self) -> usize {
        self.mapped_len
    }

    pub(crate) fn mmap_offset(self) -> usize {
        self.mmap_offset
    }

    pub(crate) fn exposed_len(self) -> usize {
        self.exposed_len
    }

    pub(crate) fn page_size(self) -> usize {
        self.page_size
    }

    pub(crate) fn required_alignment(self) -> usize {
        self.required_alignment
    }

    pub(crate) fn classify(self, desc: &TensorDesc) -> Result<GgufBackingEligibility, MetalError> {
        classify_gguf_backing(desc, self)
    }
}

#[derive(Clone)]
pub(crate) struct MetalGgufBacking {
    pub(crate) buffer: Buffer,
    pub(crate) geometry: GgufBackingGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GgufPrefaultReport {
    pub page_count: usize,
    pub covered_bytes: usize,
    pub checksum: u64,
    pub wall_ms: f64,
}

impl MetalGgufBacking {
    pub(crate) fn mapped_len(&self) -> usize {
        self.geometry.mapped_len()
    }

    pub(crate) fn exposed_len(&self) -> usize {
        self.geometry.exposed_len()
    }

    pub(crate) fn mmap_offset(&self) -> usize {
        self.geometry.mmap_offset()
    }

    pub(crate) fn page_size(&self) -> usize {
        self.geometry.page_size()
    }

    pub(crate) fn required_alignment(&self) -> usize {
        self.geometry.required_alignment()
    }

    pub(crate) fn prefault_read(&self) -> GgufPrefaultReport {
        let started = std::time::Instant::now();
        let base = self.buffer.contents().as_ptr().cast::<u8>();
        let mut checksum = 0u64;
        let mut page_count = 0usize;
        for offset in (0..self.exposed_len()).step_by(self.page_size()) {
            // SAFETY: every offset is within the complete-page prefix owned by
            // this read-only MTLBuffer. Volatile reads prevent elision.
            let byte = unsafe { std::ptr::read_volatile(base.add(offset)) };
            checksum = checksum.rotate_left(5) ^ u64::from(byte);
            page_count += 1;
        }
        GgufPrefaultReport {
            page_count,
            covered_bytes: self.exposed_len(),
            checksum,
            wall_ms: started.elapsed().as_secs_f64() * 1e3,
        }
    }

    pub(crate) fn classify(&self, desc: &TensorDesc) -> Result<GgufBackingEligibility, MetalError> {
        self.geometry.classify(desc)
    }

    pub(crate) fn tensor(
        &self,
        desc: &TensorDesc,
    ) -> Result<(GgufBackingEligibility, Option<MetalTensor>), MetalError> {
        let eligibility = self.classify(desc)?;
        if eligibility != GgufBackingEligibility::Eligible {
            return Ok((eligibility, None));
        }
        Ok((
            eligibility,
            Some(MetalTensor {
                buffer: self.buffer.clone(),
                offset: desc.data_offset - self.geometry.mmap_offset() as u64,
                shape: desc.shape.clone(),
                dtype: desc.dtype,
                provenance: MetalTensorProvenance::RetainedGgufReadOnly,
            }),
        ))
    }
}

#[doc(hidden)]
pub struct DiagnosticGgufBlitSourceWindow {
    pub(crate) backing: MetalGgufBacking,
    pub(crate) probe: DiagnosticGgufBlitReleaseProbe,
}

impl DiagnosticGgufBlitSourceWindow {
    #[doc(hidden)]
    pub fn release_probe(&self) -> DiagnosticGgufBlitReleaseProbe {
        self.probe.clone()
    }

    #[doc(hidden)]
    pub fn shard_idx(&self) -> usize {
        self.backing.geometry.shard_idx
    }

    #[doc(hidden)]
    pub fn mmap_offset(&self) -> u64 {
        self.backing.geometry.mmap_offset() as u64
    }

    #[doc(hidden)]
    pub fn exposed_len(&self) -> usize {
        self.backing.exposed_len()
    }

    #[doc(hidden)]
    pub fn encode_copy_to(
        &self,
        encoder: &BlitEncoder,
        shard_idx: usize,
        absolute_shard_offset: u64,
        destination: &Buffer,
        destination_offset: u64,
        length: u64,
    ) -> Result<(), MetalError> {
        if shard_idx != self.shard_idx() {
            return Err(MetalError::GgufNoCopy(format!(
                "diagnostic blit source shard mismatch: expected {}, got {shard_idx}",
                self.shard_idx()
            )));
        }
        let source_offset = absolute_shard_offset
            .checked_sub(self.mmap_offset())
            .ok_or_else(|| {
                MetalError::GgufNoCopy(format!(
                    "diagnostic blit source offset {absolute_shard_offset} precedes window {}",
                    self.mmap_offset()
                ))
            })?;
        let source_end = source_offset.checked_add(length).ok_or_else(|| {
            MetalError::GgufNoCopy("diagnostic blit source endpoint overflow".to_string())
        })?;
        let destination_end = destination_offset.checked_add(length).ok_or_else(|| {
            MetalError::GgufNoCopy("diagnostic blit destination endpoint overflow".to_string())
        })?;
        if source_end > self.exposed_len() as u64 {
            return Err(MetalError::GgufNoCopy(format!(
                "diagnostic blit source endpoint {source_end} exceeds window length {}",
                self.exposed_len()
            )));
        }
        if destination_end > destination.length() as u64 {
            return Err(MetalError::GgufNoCopy(format!(
                "diagnostic blit destination endpoint {destination_end} exceeds length {}",
                destination.length()
            )));
        }
        if Retained::as_ptr(&self.backing.buffer) == Retained::as_ptr(destination) {
            return Err(MetalError::GgufNoCopy(
                "diagnostic blit source and destination are the same resource".to_string(),
            ));
        }
        usize::try_from(source_offset).map_err(|_| {
            MetalError::GgufNoCopy("diagnostic blit source offset does not fit usize".to_string())
        })?;
        usize::try_from(destination_offset).map_err(|_| {
            MetalError::GgufNoCopy(
                "diagnostic blit destination offset does not fit usize".to_string(),
            )
        })?;
        usize::try_from(length).map_err(|_| {
            MetalError::GgufNoCopy("diagnostic blit length does not fit usize".to_string())
        })?;
        encoder.copy_buffer(
            &self.backing.buffer,
            source_offset,
            destination,
            destination_offset,
            length,
        );
        Ok(())
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct DiagnosticGgufBlitReleaseProbe {
    pub(crate) weak: Weak<ProtocolObject<dyn MTLBuffer>>,
    pub(crate) deallocator_calls: Arc<AtomicUsize>,
    pub(crate) deallocator_mismatches: Arc<AtomicUsize>,
}

#[doc(hidden)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DiagnosticGgufBlitReleaseReport {
    pub source_alive: bool,
    pub deallocator_calls: usize,
    pub deallocator_mismatches: usize,
}

impl DiagnosticGgufBlitReleaseProbe {
    #[doc(hidden)]
    pub fn report(&self) -> DiagnosticGgufBlitReleaseReport {
        let deallocator_calls = self.deallocator_calls.load(Ordering::Acquire);
        DiagnosticGgufBlitReleaseReport {
            source_alive: self.weak.load().is_some(),
            deallocator_calls,
            deallocator_mismatches: self.deallocator_mismatches.load(Ordering::Relaxed),
        }
    }
}

pub(crate) fn classify_gguf_backing(
    desc: &TensorDesc,
    geometry: GgufBackingGeometry,
) -> Result<GgufBackingEligibility, MetalError> {
    let (_, expected) = checked_ggml_shape_bytes(&desc.shape, desc.dtype)?;
    let declared = usize::try_from(desc.n_bytes).map_err(|_| {
        MetalError::GgufNoCopy(format!(
            "tensor {:?} n_bytes {} does not fit usize",
            desc.name, desc.n_bytes
        ))
    })?;
    if expected != declared {
        return Err(MetalError::GgufNoCopy(format!(
            "tensor {:?} shape/dtype expects {expected} bytes, descriptor declares {declared}",
            desc.name
        )));
    }
    if desc.shard_idx != geometry.shard_idx {
        return Ok(GgufBackingEligibility::WrongShard);
    }
    let start = usize::try_from(desc.data_offset).map_err(|_| {
        MetalError::GgufNoCopy(format!(
            "tensor {:?} offset {} does not fit usize",
            desc.name, desc.data_offset
        ))
    })?;
    let end = start.checked_add(declared).ok_or_else(|| {
        MetalError::GgufNoCopy(format!("tensor {:?} range overflows usize", desc.name))
    })?;
    if end > geometry.mapped_len {
        return Ok(GgufBackingEligibility::OutsideBacking);
    }
    let window_end = geometry
        .mmap_offset
        .checked_add(geometry.exposed_len)
        .expect("validated GGUF window end must not overflow");
    if start < geometry.mmap_offset || start >= window_end {
        return Ok(GgufBackingEligibility::OutsideBacking);
    }
    if end > window_end {
        let complete_page_end = geometry.mapped_len / geometry.page_size * geometry.page_size;
        if window_end == complete_page_end {
            return Ok(GgufBackingEligibility::FinalPartialPage);
        }
        return Ok(GgufBackingEligibility::OutsideBacking);
    }
    let buffer_offset = start - geometry.mmap_offset;
    if !buffer_offset.is_multiple_of(geometry.required_alignment) {
        return Ok(GgufBackingEligibility::BindingMisalignment);
    }
    Ok(GgufBackingEligibility::Eligible)
}

/// Constructor-path provenance used by typed write APIs as a fail-closed
/// defense. This is not a complete mutability capability: `buffer` remains
/// public for low-level encoders and host access, so those paths must preserve
/// the model-weight read-only contract independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MetalTensorProvenance {
    OwnedWritable,
    OwnedWeightReadOnly,
    RetainedGgufReadOnly,
}

/// A typed, shape-aware view into an `MTLBuffer`. The buffer is owned via
/// `Retained` (cloned into multiple `MetalTensor`s if you want sub-views;
/// they share the same underlying storage).
///
/// `dtype` is the on-disk ggml type for weight tensors (Q4_K, Q6_K, F32,
/// etc.); for activation/scratch buffers it's typically `F32`.
///
/// `offset` is in bytes from the buffer base. Copied tensors use zero; retained
/// GGUF views and packed arenas use nonzero offsets.
#[derive(Clone)]
pub struct MetalTensor {
    pub buffer: Buffer,
    pub offset: u64,
    pub shape: Vec<u64>,
    pub dtype: GgmlType,
    pub(crate) provenance: MetalTensorProvenance,
}

impl MetalTensor {
    pub fn provenance(&self) -> MetalTensorProvenance {
        self.provenance
    }

    pub fn is_writable(&self) -> bool {
        self.provenance == MetalTensorProvenance::OwnedWritable
    }

    pub(crate) fn assert_writable(&self, operation: &str) {
        assert!(
            self.is_writable(),
            "{operation} cannot write a read-only weight tensor"
        );
    }

    /// Total element count.
    pub fn n_elements(&self) -> u64 {
        checked_shape_elements(&self.shape).expect("MetalTensor shape element count overflow")
    }

    /// Total byte length (consults the dtype's per-block layout via
    /// checked host-side arithmetic).
    pub fn n_bytes(&self) -> u64 {
        let (_n, bytes) = checked_ggml_shape_bytes(&self.shape, self.dtype)
            .expect("MetalTensor byte-size computation overflow");
        bytes as u64
    }

    /// Build a tensor from raw bytes (e.g. the GGUF mmap slice for a
    /// weight tensor). Copies into a fresh `MTLBuffer` with shared storage.
    ///
    /// Validates that `bytes.len()` matches the byte size implied by
    /// `(shape, dtype)`. Without this check, a mismatched caller can
    /// create a short buffer with a large logical shape, and every
    /// downstream kernel or view silently operates past the buffer end.
    pub fn from_bytes(
        ctx: &MetalContext,
        bytes: &[u8],
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, MetalError> {
        let (_n, expected) = checked_ggml_shape_bytes(&shape, dtype)?;
        if bytes.len() != expected {
            return Err(MetalError::BadShape {
                kernel: "from_bytes",
                detail: format!(
                    "bytes.len()={} but shape={shape:?} dtype={dtype:?} expects {expected} bytes",
                    bytes.len()
                ),
            });
        }
        let buffer = ctx.buffer_from(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Build from a `[TensorDesc]` + the GGUF mmap. This is the production
    /// path for loading model weights.
    pub fn from_gguf_tensor(
        ctx: &MetalContext,
        desc: &TensorDesc,
        bytes: &[u8],
    ) -> Result<Self, MetalError> {
        Self::from_bytes(ctx, bytes, desc.shape.clone(), desc.dtype)
    }

    pub(crate) fn copied_gguf_weight(
        ctx: &MetalContext,
        desc: &TensorDesc,
        bytes: &[u8],
    ) -> Result<Self, MetalError> {
        let (_, expected) = checked_ggml_shape_bytes(&desc.shape, desc.dtype)?;
        if bytes.len() != expected || desc.n_bytes != expected as u64 {
            return Err(MetalError::BadShape {
                kernel: "copied_gguf_weight",
                detail: format!(
                    "tensor {:?} has bytes.len()={} n_bytes={} but shape={:?} dtype={:?} expects {expected} bytes",
                    desc.name,
                    bytes.len(),
                    desc.n_bytes,
                    desc.shape,
                    desc.dtype
                ),
            });
        }
        let buffer = ctx.buffer_from(bytes)?;
        Self::owned_weight_view(buffer, 0, desc.shape.clone(), desc.dtype, 32)
    }

    pub(crate) fn owned_weight_view(
        buffer: Buffer,
        offset: u64,
        shape: Vec<u64>,
        dtype: GgmlType,
        required_alignment: usize,
    ) -> Result<Self, MetalError> {
        if !required_alignment.is_power_of_two() {
            return Err(MetalError::BadShape {
                kernel: "owned_weight_view",
                detail: format!("alignment {required_alignment} is not a power of two"),
            });
        }
        let offset_usize = usize::try_from(offset).map_err(|_| MetalError::BadShape {
            kernel: "owned_weight_view",
            detail: format!("offset {offset} does not fit usize"),
        })?;
        if offset_usize % required_alignment != 0 {
            return Err(MetalError::BadShape {
                kernel: "owned_weight_view",
                detail: format!("offset {offset_usize} is not aligned to {required_alignment}"),
            });
        }
        let (_, n_bytes) = checked_ggml_shape_bytes(&shape, dtype)?;
        let end = offset_usize
            .checked_add(n_bytes)
            .ok_or_else(|| MetalError::BadShape {
                kernel: "owned_weight_view",
                detail: "view endpoint overflow".to_string(),
            })?;
        if end > buffer.length() {
            return Err(MetalError::BadShape {
                kernel: "owned_weight_view",
                detail: format!(
                    "view [{offset_usize}..{end}) exceeds buffer length {}",
                    buffer.length()
                ),
            });
        }
        Ok(Self {
            buffer,
            offset,
            shape,
            dtype,
            provenance: MetalTensorProvenance::OwnedWeightReadOnly,
        })
    }

    /// Build a read-only Q6_K row-bank view with the fixed alignment required
    /// by the grammar-row lm-head floor. This deliberately exposes neither a
    /// caller-selected dtype nor a caller-selected alignment.
    #[doc(hidden)]
    pub fn q6_k_row_bank_weight_view(
        buffer: Buffer,
        offset: u64,
        n_in: usize,
        n_out: usize,
    ) -> Result<Self, MetalError> {
        if n_in == 0 || n_out == 0 || !n_in.is_multiple_of(256) {
            return Err(MetalError::BadShape {
                kernel: "q6_k_row_bank_weight_view",
                detail: format!(
                    "expected nonzero Q6_K rows with n_in divisible by 256, got n_in={n_in} n_out={n_out}"
                ),
            });
        }
        let n_in_u32 = u32::try_from(n_in).map_err(|_| MetalError::BadShape {
            kernel: "q6_k_row_bank_weight_view",
            detail: format!("n_in={n_in} does not fit the Q6_K kernel argument"),
        })?;
        let n_out_u32 = u32::try_from(n_out).map_err(|_| MetalError::BadShape {
            kernel: "q6_k_row_bank_weight_view",
            detail: format!("n_out={n_out} does not fit the Q6_K kernel argument"),
        })?;
        Self::owned_weight_view(
            buffer,
            offset,
            vec![u64::from(n_in_u32), u64::from(n_out_u32)],
            GgmlType::Q6_K,
            32,
        )
    }

    /// Allocate an F32 activation/scratch tensor of the given shape,
    /// uninitialized.
    pub fn zeros_f32(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_shape_bytes(&shape, std::mem::size_of::<f32>())?;
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::F32,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Allocate an I32 tensor for token IDs and integer kernel outputs.
    pub fn zeros_i32(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_shape_bytes(&shape, std::mem::size_of::<i32>())?;
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::I32,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Allocate an F16 (half-precision) tensor. Used for the KV cache
    /// when we want to halve attention bandwidth at long context. The
    /// scatter kernel converts F32 → F16 on append; the attn_decode
    /// kernel reads F16 and casts to F32 in the dot product.
    pub fn zeros_f16(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_shape_bytes(&shape, 2)?; // half = 2 bytes
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::F16,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Allocate a Q8_0 tensor. Used for experimental KV-Q8 cache storage.
    /// Logical shape is still expressed in ELEMENTS; byte size follows ggml's
    /// Q8_0 block layout via `n_bytes()`.
    pub fn zeros_q8_0(ctx: &MetalContext, shape: Vec<u64>) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_ggml_shape_bytes(&shape, GgmlType::Q8_0)?;
        let buffer = ctx.buffer_uninit(bytes)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype: GgmlType::Q8_0,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Allocate a zero-initialized tensor of any GGML dtype.
    ///
    /// This is mainly for synthetic benchmarks that need valid quantized
    /// buffers without loading a model tensor from disk.
    pub fn zeros_dtype(
        ctx: &MetalContext,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, MetalError> {
        let (_n, bytes) = checked_ggml_shape_bytes(&shape, dtype)?;
        let zeros = vec![0u8; bytes.max(1)];
        let buffer = ctx.buffer_from(&zeros)?;
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Byte-zero owned shared storage without a second, tensor-sized host buffer.
    /// The caller must admit the Metal allocation before calling this helper.
    pub(crate) fn zeros_dtype_unstaged(
        ctx: &MetalContext,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> Result<Self, MetalError> {
        let (_, bytes) = checked_ggml_shape_bytes(&shape, dtype)?;
        let extent = bytes.max(1);
        if extent > isize::MAX as usize || extent > ctx.max_buffer_length() {
            return Err(MetalError::BadShape {
                kernel: "zeros_dtype_unstaged",
                detail: "byte extent exceeds CPU or device buffer bounds".into(),
            });
        }
        let buffer = ctx.buffer_uninit(extent)?;
        if buffer.storageMode() != MTLStorageMode::Shared || buffer.length() < extent {
            return Err(MetalError::BadShape {
                kernel: "zeros_dtype_unstaged",
                detail: "allocation must provide the full shared byte extent".into(),
            });
        }
        // Fresh owned storage, not yet published or submitted to any command.
        // Zero quantized payload bytes exactly as the staged helper does; this
        // makes no additional claim about a dtype's decoded numerical zero.
        unsafe { std::ptr::write_bytes(buffer.contents().as_ptr().cast::<u8>(), 0, extent) };
        Ok(Self {
            buffer,
            offset: 0,
            shape,
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        })
    }

    /// Build a zero-copy sub-view of this tensor: same underlying MTLBuffer,
    /// shifted by `elem_offset` elements (of the tensor's dtype), with a
    /// new logical `shape`. The resulting view shares storage and aliases
    /// the parent — use only when you know the parent isn't being read by
    /// concurrent dispatches.
    ///
    /// Per Jeff & Sanjay (avoid copies / use indices instead of pointers):
    /// turns aliasing-via-explicit-copy into aliasing-via-offset, eliminating
    /// the entire copy_offset dispatch. Used to slice GDN's fused QKV
    /// conv-output buffer into Q / K / V subranges with zero kernels.
    ///
    /// Constraint: only valid for non-quantized dtypes (F32, F16, I32) where
    /// elem_offset translates trivially to byte offset. For quantized
    /// types (Q4_K, Q5_K, Q6_K) the byte offset would need to align to
    /// the super-block boundary, which `view_subrange` does not check.
    pub fn view_subrange(&self, elem_offset: u64, shape: Vec<u64>) -> Self {
        let elem_size: u64 = match self.dtype {
            GgmlType::F32 | GgmlType::I32 => 4,
            GgmlType::F16 => 2,
            other => panic!(
                "view_subrange only supports F32/F16/I32 (no super-block alignment), got {other:?}"
            ),
        };
        let n_view = checked_shape_elements(&shape).expect("view_subrange shape overflow");
        let parent_n = self.n_elements();
        // Always-on bounds check: a release-build OOB view is silent GPU UB
        // (kernels happily read/write past the buffer end). All callers
        // here are computing static slot indices, not user input, so a
        // failure is a programming bug, not a recoverable condition.
        assert!(
            elem_offset
                .checked_add(n_view)
                .is_some_and(|end| end <= parent_n),
            "view_subrange OOB: elem_offset={elem_offset} + n_view={n_view} > parent_n={parent_n}"
        );
        let byte_delta = elem_offset
            .checked_mul(elem_size)
            .expect("view_subrange byte offset overflow");
        let offset = self
            .offset
            .checked_add(byte_delta)
            .expect("view_subrange absolute offset overflow");
        Self {
            buffer: self.buffer.clone(),
            offset,
            shape,
            dtype: self.dtype,
            provenance: self.provenance,
        }
    }

    /// Build a zero-copy sub-view by byte offset. Intended for quantized
    /// expert banks where each expert slice is already laid out as a
    /// contiguous 2D tensor in the parent buffer.
    pub fn view_bytes(&self, byte_offset: u64, shape: Vec<u64>) -> Self {
        // Always-on bounds check: see view_subrange for rationale.
        let (_view_n, view_bytes_usize) = checked_ggml_shape_bytes(&shape, self.dtype)
            .expect("view_bytes byte-size computation overflow");
        let view_bytes = view_bytes_usize as u64;
        let parent_bytes = self.n_bytes();
        assert!(
            byte_offset
                .checked_add(view_bytes)
                .is_some_and(|end| end <= parent_bytes),
            "view_bytes OOB: byte_offset={byte_offset} + view_bytes={view_bytes} > parent_bytes={parent_bytes}"
        );
        let offset = self
            .offset
            .checked_add(byte_offset)
            .expect("view_bytes absolute offset overflow");
        Self {
            buffer: self.buffer.clone(),
            offset,
            shape,
            dtype: self.dtype,
            provenance: self.provenance,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metal::test_support::*;

    #[test]
    fn allocation_census_records_realized_storage() {
        let ctx = MetalContext::new().expect("create Metal context");
        allocation_census_begin();
        let first = ctx
            .buffer_uninit(17)
            .expect("allocate uninitialized buffer");
        let second = ctx.buffer_from(&[1u32, 2]).expect("allocate copied buffer");
        let rows = allocation_census_take();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].requested_bytes, 17);
        assert_eq!(rows[0].buffer_length, first.length() as u64);
        assert_eq!(rows[0].storage_mode, "shared");
        assert_eq!(rows[1].requested_bytes, 8);
        assert_eq!(rows[1].buffer_length, second.length() as u64);
        assert_eq!(rows[1].storage_mode, "shared");
        assert_eq!(diagnostics_observer_active_counts(), [0, 0, 0]);
    }

    #[test]
    fn gguf_backing_classification_is_typed_and_fail_closed() {
        let geometry = GgufBackingGeometry::new(0, 160, 64, 32).unwrap();
        assert_eq!(geometry.mapped_len(), 160);
        assert_eq!(geometry.mmap_offset(), 0);
        assert_eq!(geometry.exposed_len(), 128);
        assert_eq!(geometry.page_size(), 64);
        assert_eq!(geometry.required_alignment(), 32);
        assert_eq!(
            geometry.classify(&f32_desc("ok", 0, 32, 8)).unwrap(),
            GgufBackingEligibility::Eligible
        );
        assert_eq!(
            geometry.classify(&f32_desc("tail", 0, 96, 9)).unwrap(),
            GgufBackingEligibility::FinalPartialPage
        );
        assert_eq!(
            geometry.classify(&f32_desc("shard", 1, 32, 8)).unwrap(),
            GgufBackingEligibility::WrongShard
        );
        assert_eq!(
            geometry.classify(&f32_desc("align", 0, 36, 8)).unwrap(),
            GgufBackingEligibility::BindingMisalignment
        );
        assert_eq!(
            geometry.classify(&f32_desc("outside", 0, 160, 8)).unwrap(),
            GgufBackingEligibility::OutsideBacking
        );

        let mut malformed = f32_desc("malformed", 0, 32, 8);
        malformed.n_bytes -= 1;
        assert!(geometry.classify(&malformed).is_err());
        assert!(GgufBackingGeometry::new(0, 160, 0, 32).is_err());
        assert!(GgufBackingGeometry::new(0, 160, 64, 0).is_err());
        assert!(GgufBackingGeometry::new(0, 32, 64, 32).is_err());

        let window = GgufBackingGeometry::new_window(0, 256, 64, 128, 64, 32).unwrap();
        assert_eq!(window.mapped_len(), 256);
        assert_eq!(window.mmap_offset(), 64);
        assert_eq!(window.exposed_len(), 128);
        assert_eq!(
            window.classify(&f32_desc("before", 0, 32, 8)).unwrap(),
            GgufBackingEligibility::OutsideBacking
        );
        assert_eq!(
            window.classify(&f32_desc("inside", 0, 96, 8)).unwrap(),
            GgufBackingEligibility::Eligible
        );
        assert_eq!(
            window.classify(&f32_desc("after", 0, 192, 8)).unwrap(),
            GgufBackingEligibility::OutsideBacking
        );
        assert_eq!(
            window
                .classify(&f32_desc("after-misaligned", 0, 196, 8))
                .unwrap(),
            GgufBackingEligibility::OutsideBacking
        );
        let earlier = GgufBackingGeometry::new_window(0, 160, 0, 64, 64, 32).unwrap();
        assert_eq!(
            earlier
                .classify(&f32_desc("outside-earlier", 0, 96, 9))
                .unwrap(),
            GgufBackingEligibility::OutsideBacking
        );
        let terminal = GgufBackingGeometry::new_window(0, 160, 64, 64, 64, 32).unwrap();
        assert_eq!(
            terminal
                .classify(&f32_desc("terminal-tail", 0, 96, 9))
                .unwrap(),
            GgufBackingEligibility::FinalPartialPage
        );
        assert!(GgufBackingGeometry::new_window(0, 256, 32, 64, 64, 32).is_err());
        assert!(GgufBackingGeometry::new_window(0, 256, 64, 96, 64, 32).is_err());
        assert!(GgufBackingGeometry::new_window(0, 256, 192, 128, 64, 32).is_err());
    }

    #[test]
    fn retained_storage_plan_is_order_independent_and_window_bounded() {
        let a = f32_desc("a", 0, 32, 8);
        let b = f32_desc("b", 0, 64, 16);
        let c = f32_desc("c", 0, 128, 8);
        let requests = [&c, &a, &b];
        let plan = plan_retained_storage(&[192], &requests, 64, 130, 32).unwrap();
        assert_retained_plan_invariants(&plan, &requests, &[192]);

        assert_eq!(plan.usable_window_length, 128);
        assert_eq!(plan.windows.len(), 2);
        assert_eq!(
            plan.windows,
            vec![
                RetainedStorageWindow {
                    shard_idx: 0,
                    mmap_offset: 0,
                    length: 128,
                },
                RetainedStorageWindow {
                    shard_idx: 0,
                    mmap_offset: 128,
                    length: 64,
                },
            ]
        );
        assert_eq!(
            plan.entries[0].disposition,
            RetainedStorageDisposition::View {
                window_index: 1,
                buffer_offset: 0,
            }
        );
        assert_eq!(
            plan.entries[1].disposition,
            RetainedStorageDisposition::View {
                window_index: 0,
                buffer_offset: 32,
            }
        );
        assert_eq!(
            plan.entries[2].disposition,
            RetainedStorageDisposition::View {
                window_index: 0,
                buffer_offset: 64,
            }
        );
        assert_eq!(plan.unique_view_bytes, 128);
        assert_eq!(plan.logical_view_bytes, 128);
        assert_eq!(plan.unique_fallback_bytes, 0);
        assert_eq!(plan.alias_bytes, 0);

        let permutation = [&b, &c, &a];
        let permuted = plan_retained_storage(&[192], &permutation, 64, 130, 32).unwrap();
        assert_retained_plan_invariants(&permuted, &permutation, &[192]);
        assert_eq!(plan.windows, permuted.windows);
        for name in ["a", "b", "c"] {
            let original = plan
                .entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap();
            let permuted = permuted
                .entries
                .iter()
                .find(|entry| entry.name == name)
                .unwrap();
            assert_eq!(original.disposition, permuted.disposition);
        }

        let crossing_a = f32_desc("crossing-a", 0, 0, 24);
        let crossing_b = f32_desc("crossing-b", 0, 96, 16);
        let crossing =
            plan_retained_storage(&[192], &[&crossing_a, &crossing_b], 64, 128, 32).unwrap();
        assert_retained_plan_invariants(&crossing, &[&crossing_a, &crossing_b], &[192]);
        assert_eq!(crossing.windows.len(), 2);
        assert_eq!(crossing.windows[0].mmap_offset, 0);
        assert_eq!(crossing.windows[0].length, 128);
        assert_eq!(crossing.windows[1].mmap_offset, 64);
        assert_eq!(crossing.windows[1].length, 128);
        assert_eq!(
            crossing.entries[1].disposition,
            RetainedStorageDisposition::View {
                window_index: 1,
                buffer_offset: 32,
            }
        );
    }

    #[test]
    fn retained_storage_plan_classifies_fallbacks_and_aliases() {
        let view = f32_desc("view", 1, 64, 8);
        let missing = f32_desc("missing", 3, 0, 8);
        let tail = f32_desc("tail", 0, 96, 16);
        let outside = f32_desc("outside", 0, 160, 8);
        let misaligned = f32_desc("misaligned", 0, 36, 8);
        let too_large = f32_desc("too-large", 2, 32, 32);
        let requests = [
            &view,
            &view,
            &missing,
            &tail,
            &outside,
            &misaligned,
            &too_large,
        ];
        let plan = plan_retained_storage(&[160, 256, 256], &requests, 64, 128, 32).unwrap();
        assert_retained_plan_invariants(&plan, &requests, &[160, 256, 256]);

        assert_eq!(
            plan.entries[0].disposition,
            RetainedStorageDisposition::View {
                window_index: 0,
                buffer_offset: 0,
            }
        );
        assert_eq!(
            plan.entries[1].disposition,
            RetainedStorageDisposition::Alias {
                source_request_index: 0,
            }
        );
        let reasons = plan
            .entries
            .iter()
            .filter_map(|entry| match entry.disposition {
                RetainedStorageDisposition::CopyFallback { reason } => Some(reason),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            reasons,
            vec![
                RetainedStorageFallback::MissingShard,
                RetainedStorageFallback::FinalPartialPage,
                RetainedStorageFallback::OutsideShard,
                RetainedStorageFallback::BindingMisalignment,
                RetainedStorageFallback::TensorExceedsWindow,
            ]
        );
        assert_eq!(plan.unique_view_bytes, 32);
        assert_eq!(plan.logical_view_bytes, 64);
        assert_eq!(plan.unique_fallback_bytes, 288);
        assert_eq!(plan.alias_bytes, 32);

        let tail_alias = plan_retained_storage(&[160], &[&tail, &tail], 64, 128, 32).unwrap();
        assert_eq!(tail_alias.unique_fallback_bytes, 64);
        assert_eq!(tail_alias.alias_bytes, 64);
        assert_eq!(
            tail_alias.entries[1].disposition,
            RetainedStorageDisposition::Alias {
                source_request_index: 0,
            }
        );

        let shard_zero = f32_desc("shard-zero", 0, 32, 8);
        let shard_one = f32_desc("shard-one", 1, 64, 8);
        let multi_requests = [&shard_one, &shard_zero];
        let multi = plan_retained_storage(&[128, 192], &multi_requests, 64, 128, 32).unwrap();
        assert_retained_plan_invariants(&multi, &multi_requests, &[128, 192]);
        assert_eq!(multi.windows.len(), 2);
        assert_eq!(multi.windows[0].shard_idx, 0);
        assert_eq!(multi.windows[1].shard_idx, 1);
    }

    #[test]
    fn retained_storage_plan_rejects_malformed_inputs() {
        let valid = f32_desc("valid", 0, 64, 16);
        assert!(plan_retained_storage(&[128], &[&valid], 0, 128, 32).is_err());
        assert!(plan_retained_storage(&[128], &[&valid], 64, 63, 32).is_err());
        assert!(plan_retained_storage(&[128], &[&valid], 64, 128, 0).is_err());
        assert!(plan_retained_storage(&[128], &[&valid], 64, 128, 128).is_err());

        let mut malformed = valid.clone();
        malformed.n_bytes -= 1;
        assert!(plan_retained_storage(&[128], &[&malformed], 64, 128, 32).is_err());

        let empty = f32_desc("empty", 0, 0, 0);
        assert!(plan_retained_storage(&[128], &[&empty], 64, 128, 32).is_err());

        let alias_a = f32_desc("alias-a", 0, 64, 8);
        let mut alias_b = alias_a.clone();
        alias_b.name = "alias-b".to_string();
        alias_b.shape = vec![2, 4];
        assert!(plan_retained_storage(&[128], &[&alias_a, &alias_b], 64, 128, 32).is_err());

        let overlap_a = f32_desc("overlap-a", 0, 32, 16);
        let overlap_b = f32_desc("overlap-b", 0, 64, 8);
        assert!(plan_retained_storage(&[128], &[&overlap_a, &overlap_b], 64, 128, 32).is_err());

        let tail_misaligned = f32_desc("tail-misaligned", 0, 100, 8);
        let tail_misaligned_plan =
            plan_retained_storage(&[160], &[&tail_misaligned], 64, 128, 32).unwrap();
        assert_eq!(
            tail_misaligned_plan.entries[0].disposition,
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::BindingMisalignment,
            }
        );

        let tail_oversized = f32_desc("tail-oversized", 0, 64, 24);
        let tail_oversized_plan =
            plan_retained_storage(&[160], &[&tail_oversized], 64, 64, 32).unwrap();
        assert_eq!(
            tail_oversized_plan.entries[0].disposition,
            RetainedStorageDisposition::CopyFallback {
                reason: RetainedStorageFallback::TensorExceedsWindow,
            }
        );
    }

    #[test]
    #[ignore = "requires QWEN_GGUF_NO_COPY_MODEL local single-shard fixture"]
    fn gguf_no_copy_real_model_coverage_probe() {
        let path = std::env::var("QWEN_GGUF_NO_COPY_MODEL")
            .expect("set QWEN_GGUF_NO_COPY_MODEL to a local GGUF");
        let gguf = crate::gguf::GgufFile::open(&path).expect("open GGUF");
        assert_eq!(gguf.shard_count(), 1, "coverage probe requires one shard");
        let page_size = host_page_size().expect("host page size");
        let geometry = GgufBackingGeometry::new(0, gguf.total_mapped_len(), page_size, 32)
            .expect("GGUF backing geometry");
        let mut eligible_bytes = 0u64;
        let mut crossing = Vec::new();
        let mut dtype_geometry = std::collections::BTreeMap::new();
        for desc in &gguf.tensors {
            let alignment = desc.data_offset & desc.data_offset.wrapping_neg();
            let entry = dtype_geometry
                .entry(format!("{:?}", desc.dtype))
                .or_insert((0usize, 0u64, u64::MAX));
            entry.0 += 1;
            entry.1 = entry.1.saturating_add(desc.n_bytes);
            entry.2 = entry.2.min(alignment);
            match geometry.classify(desc).expect("classify tensor") {
                GgufBackingEligibility::Eligible => {
                    eligible_bytes = eligible_bytes.saturating_add(desc.n_bytes);
                }
                GgufBackingEligibility::FinalPartialPage => {
                    crossing.push((desc.name.clone(), desc.n_bytes));
                }
                other => panic!("unexpected ineligibility for {}: {other:?}", desc.name),
            }
        }
        let total_bytes: u64 = gguf.tensors.iter().map(|desc| desc.n_bytes).sum();
        let coverage = eligible_bytes as f64 / total_bytes.max(1) as f64;
        eprintln!(
            concat!(
                "[gguf-no-copy-coverage] model={} mapped={} exposed={} page={} ",
                "suffix={} tensors={} total={} eligible={} coverage={:.8} crossing={:?}"
            ),
            path,
            geometry.mapped_len(),
            geometry.exposed_len(),
            geometry.page_size(),
            geometry.mapped_len() - geometry.exposed_len(),
            gguf.tensors.len(),
            total_bytes,
            eligible_bytes,
            coverage,
            crossing,
        );
        eprintln!("[gguf-no-copy-dtypes] {dtype_geometry:?}");
        assert!(
            coverage >= 0.99,
            "no-copy coverage {coverage:.6} is below 99%"
        );
    }
}
