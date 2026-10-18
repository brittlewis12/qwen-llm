use super::*;

pub(super) fn padded_lengths(g: &GgufFile, page: usize) -> Result<Vec<usize>, LoadError> {
    g.shard_mapped_lengths()
        .into_iter()
        .map(|n| {
            n.checked_next_multiple_of(page)
                .ok_or_else(|| invalid("page-rounded mapping size overflow"))
        })
        .collect()
}

/// Return a retained mapping and whether its offsets are window-relative.
/// Private remaps preserve file-backed interior pages. Only boundary pages
/// containing disk-only bytes are dirtied; no tensor-sized staging is created.
pub(super) fn mapping(
    g: &GgufFile,
    w: &metal::RetainedStorageWindow,
    page: usize,
) -> Result<(Arc<memmap2::Mmap>, bool), LoadError> {
    let shard = g
        .shards
        .get(w.shard_idx)
        .ok_or_else(|| invalid("missing shard"))?;
    let end = add(w.mmap_offset, w.length as u64)?;
    let mut clear = Vec::new();
    for table in g
        .tensors
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
    if clear.is_empty() && end <= shard.mmap_len() as u64 {
        return Ok((shard.mmap.clone(), false));
    }
    let rounded = shard
        .mmap_len()
        .checked_next_multiple_of(page)
        .ok_or_else(|| invalid("mapping extent overflow"))?;
    if end > rounded as u64
        || w.length == 0
        || w.mmap_offset % page as u64 != 0
        || w.length % page != 0
    {
        return Err(invalid("invalid private boundary window"));
    }
    // SAFETY: the retained descriptor is the parsed file, the offset is page
    // aligned, and the mapping ends no later than the page containing EOF.
    // POSIX defines the partial EOF page as zero-filled. Never map a page
    // wholly beyond EOF. As with the shared loader, callers keep files immutable.
    let mut mmap = unsafe {
        memmap2::MmapOptions::new()
            .offset(w.mmap_offset)
            .len(w.length)
            .map_copy(&*shard.file)?
    };
    for range in clear {
        mmap[range].fill(0);
    }
    Ok((Arc::new(mmap.make_read_only()?), true))
}
