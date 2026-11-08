//! Joined, bounded CPU copies into an immutable snapshot arena.
use super::*;
use std::mem::MaybeUninit;

const MAX_WORKERS: usize = 8;
const PARALLEL_MIN_BYTES: usize = 256 << 20;

pub(super) fn copy_regions(
    sources: &[&[u8]],
    workers: usize,
) -> Result<Vec<u8>, Qwen4ExpTextSessionError> {
    if !(1..=MAX_WORKERS).contains(&workers) {
        return invalid("snapshot copy requires 1..=8 workers");
    }
    let bytes = sources
        .iter()
        .try_fold(0usize, |n, s| n.checked_add(s.len()))
        .ok_or_else(|| Qwen4ExpTextSessionError::Invalid("snapshot copy length overflow".into()))?;
    let reserve_at = std::time::Instant::now();
    let mut arena = Vec::new();
    arena.try_reserve_exact(bytes).map_err(|error| {
        Qwen4ExpTextSessionError::Invalid(format!(
            "snapshot arena of {bytes} bytes refused: {error}"
        ))
    })?;
    let reserve_ms = reserve_at.elapsed().as_secs_f64() * 1e3;
    let copy_at = std::time::Instant::now();
    if workers == 1 {
        for source in sources {
            arena.extend_from_slice(source);
        }
    } else {
        copy_parallel(sources, &mut arena.spare_capacity_mut()[..bytes], workers)?;
        // SAFETY: every byte of the disjoint destination slices has been
        // initialized, and every copy thread joined successfully. On error
        // the Vec retains length zero and no partial snapshot is published.
        unsafe { arena.set_len(bytes) };
    }
    tracing::info!(target: "qwen_diag", "qwen4exp snapshot copy: bytes={bytes} workers={workers} reserve_ms={reserve_ms:.3} copy_ms={:.3}", copy_at.elapsed().as_secs_f64() * 1e3);
    Ok(arena)
}

pub(super) fn parallel_workers(bytes: usize) -> usize {
    if bytes < PARALLEL_MIN_BYTES {
        1
    } else {
        std::thread::available_parallelism().map_or(1, |n| n.get().min(MAX_WORKERS))
    }
}

fn copy_parallel(
    sources: &[&[u8]],
    destination: &mut [MaybeUninit<u8>],
    workers: usize,
) -> Result<(), Qwen4ExpTextSessionError> {
    if destination.is_empty() {
        return Ok(());
    }
    // Use page-sized multiples for the partition sizes.
    let chunk_bytes = destination.len().div_ceil(workers).div_ceil(65536) * 65536;
    std::thread::scope(|scope| {
        let mut handles = Vec::with_capacity(workers);
        let mut failure = None;
        for (index, chunk) in destination.chunks_mut(chunk_bytes).enumerate() {
            match std::thread::Builder::new()
                .name("fn-snapshot-copy".into())
                .spawn_scoped(scope, move || {
                    copy_chunk(sources, index * chunk_bytes, chunk)
                }) {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    failure = Some(format!("snapshot copy worker refused: {error}"));
                    break;
                }
            }
        }
        for handle in handles {
            if handle.join().is_err() {
                failure = Some("snapshot copy worker panicked".into());
            }
        }
        match failure {
            Some(error) => invalid(error),
            None => Ok(()),
        }
    })
}

fn copy_chunk(sources: &[&[u8]], begin: usize, destination: &mut [MaybeUninit<u8>]) {
    let end = begin + destination.len();
    let mut source_begin = 0;
    for source in sources {
        let source_end = source_begin + source.len();
        let low = begin.max(source_begin);
        let high = end.min(source_end);
        if low < high {
            // SAFETY: intersections are inside both slices. Sources are
            // immutable, released producer views, and destination chunks are
            // disjoint slices of a different, newly allocated arena.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source.as_ptr().add(low - source_begin),
                    destination.as_mut_ptr().cast::<u8>().add(low - begin),
                    high - low,
                );
            }
        }
        source_begin = source_end;
        if source_begin >= end {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_copy_preserves_region_boundaries_and_immutable_payload() {
        // Cross multiple worker partitions and empty/odd-sized source regions.
        let sizes = [0, 17, 180003, 1, 97009, 0, 511337, 3];
        let mut inputs: Vec<Vec<u8>> = sizes
            .iter()
            .enumerate()
            .map(|(region, &n)| {
                (0..n)
                    .map(|i| ((i * 37 + region * 13) % 251) as u8)
                    .collect()
            })
            .collect();
        let expected: Vec<u8> = inputs.iter().flatten().copied().collect();
        let sources: Vec<_> = inputs.iter().map(Vec::as_slice).collect();
        let copies: Vec<_> = [1, 2, 3, 8]
            .into_iter()
            .map(|workers| copy_regions(&sources, workers).unwrap())
            .collect();
        for input in &mut inputs {
            input.fill(0xee);
        }
        for copy in copies {
            assert_eq!(copy, expected);
        }
    }

    #[test]
    fn empty_and_short_copy_have_no_unwritten_tail() {
        for workers in [1, 8] {
            assert!(copy_regions(&[], workers).unwrap().is_empty());
            assert_eq!(
                copy_regions(&[&[], &[0, 255, 7], &[]], workers).unwrap(),
                [0, 255, 7]
            );
        }
        assert!(copy_regions(&[&[1]], 0).is_err());
        assert!(copy_regions(&[&[1]], 9).is_err());
        assert_eq!(parallel_workers(PARALLEL_MIN_BYTES - 1), 1);
        assert!((1..=8).contains(&parallel_workers(PARALLEL_MIN_BYTES)));
    }
}
