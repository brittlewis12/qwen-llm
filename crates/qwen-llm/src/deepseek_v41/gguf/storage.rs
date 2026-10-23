use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MappingKind {
    OriginalShared,
    SharedEof,
    PrivateBoundary { bytes: u64 },
}

impl MappingKind {
    /// Charge the entire private span, not just the pages cleared on the CPU.
    pub(super) fn private_bytes(self) -> u64 {
        match self {
            Self::PrivateBoundary { bytes } => bytes,
            Self::OriginalShared | Self::SharedEof => 0,
        }
    }
}

#[derive(Debug)]
pub(super) struct MappingPlan {
    pub(super) kind: MappingKind,
    clear: Vec<std::ops::Range<usize>>,
}

pub(super) fn padded_lengths(g: &GgufFile, page: usize) -> Result<Vec<usize>, LoadError> {
    if !page.is_power_of_two() {
        return Err(invalid("invalid mapping page size"));
    }
    g.shard_mapped_lengths()
        .into_iter()
        .map(|n| {
            n.checked_next_multiple_of(page)
                .ok_or_else(|| invalid("page-rounded mapping size overflow"))
        })
        .collect()
}

pub(super) fn mapping_kind(
    g: &GgufFile,
    w: &metal::RetainedStorageWindow,
    page: usize,
) -> Result<MappingKind, LoadError> {
    Ok(mapping_plan(g, w, page)?.kind)
}

fn mapping_plan(
    g: &GgufFile,
    w: &metal::RetainedStorageWindow,
    page: usize,
) -> Result<MappingPlan, LoadError> {
    let shard = g
        .shards
        .get(w.shard_idx)
        .ok_or_else(|| invalid("missing shard"))?;
    plan_mapping(shard.mmap_len() as u64, &g.tensors, w, page)
}

/// Metadata-only geometry; file_len is the raw EOF, never a padded extent.
pub(super) fn plan_mapping(
    file_len: u64,
    tensors: &[TensorDesc],
    w: &metal::RetainedStorageWindow,
    page: usize,
) -> Result<MappingPlan, LoadError> {
    if !page.is_power_of_two() {
        return Err(invalid("invalid mapping page size"));
    }
    if w.length == 0
        || w.length > isize::MAX as usize
        || w.mmap_offset % page as u64 != 0
        || w.length % page != 0
    {
        return Err(invalid("invalid boundary window alignment or length"));
    }
    let end = add(w.mmap_offset, w.length as u64)?;
    let rounded = file_len
        .checked_next_multiple_of(page as u64)
        .ok_or_else(|| invalid("mapping extent overflow"))?;
    if w.mmap_offset >= file_len || end > rounded {
        return Err(invalid("mapping window extends beyond EOF page"));
    }
    let mut clear = Vec::new();
    for table in tensors
        .iter()
        .filter(|t| engram::is_table(&t.name) && t.shard_idx == w.shard_idx)
    {
        let start = table.data_offset.max(w.mmap_offset);
        let stop = add(table.data_offset, table.n_bytes)?.min(end);
        if start < stop {
            let local_start = usize::try_from(start - w.mmap_offset)
                .map_err(|_| invalid("mapping offset overflow"))?;
            let local_end = usize::try_from(stop - w.mmap_offset)
                .map_err(|_| invalid("mapping offset overflow"))?;
            if !(local_end <= page || local_start >= w.length.saturating_sub(page)) {
                return Err(invalid("disk Engram occurs inside a resident window"));
            }
            clear.push(local_start..local_end);
        }
    }
    let kind = if !clear.is_empty() {
        MappingKind::PrivateBoundary {
            bytes: w.length as u64,
        }
    } else if end <= file_len {
        MappingKind::OriginalShared
    } else {
        MappingKind::SharedEof
    };
    Ok(MappingPlan { kind, clear })
}

/// Return a retained mapping and whether its offsets are window-relative.
/// Only Engram boundary windows use private mappings and bounded CPU clears.
pub(super) fn mapping(
    g: &GgufFile,
    w: &metal::RetainedStorageWindow,
    page: usize,
) -> Result<(Arc<memmap2::Mmap>, bool), LoadError> {
    let plan = mapping_plan(g, w, page)?;
    let shard = &g.shards[w.shard_idx];
    if plan.kind == MappingKind::OriginalShared {
        return Ok((shard.mmap.clone(), false));
    }
    let mut options = memmap2::MmapOptions::new();
    options.offset(w.mmap_offset).len(w.length);
    // SAFETY: the retained descriptor is the parsed file, the offset is page
    // aligned, and the mapping ends no later than the page containing EOF.
    // POSIX defines the partial EOF page as zero-filled. Never map a page
    // wholly beyond EOF. As with the shared loader, callers keep files immutable.
    let mmap = unsafe {
        match plan.kind {
            MappingKind::SharedEof => options.map(&*shard.file)?,
            MappingKind::PrivateBoundary { .. } => {
                let mut mmap = options.map_copy(&*shard.file)?;
                for range in plan.clear {
                    mmap[range].fill(0);
                }
                mmap.make_read_only()?
            }
            MappingKind::OriginalShared => unreachable!(),
        }
    };
    Ok((Arc::new(mmap), true))
}
