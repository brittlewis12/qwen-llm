//! Parallel-pread cache warmer for mmap-backed GGUF shards.
//!
//! # Why
//!
//! On macOS APFS + Apple Silicon, `mmap` sequential reads deliver
//! roughly ~0.8 GB/s cold, while the same drive reaches ~6 GB/s under
//! direct large-block reads (measured on the internal 2 TB SSD via `fio`
//! posixaio, 1 MiB, QD8; see `ssd-bench/` in Britt's random workspace).
//! The gap is driven by macOS demand paging: mmap traverses the file
//! four pages at a time, each miss a synchronous fault. Populating the
//! unified buffer cache in parallel via `pread(2)` gets us closer to the
//! drive ceiling before the mmap consumer ever touches a page.
//!
//! Since macOS's unified buffer cache is shared between mmap and
//! `read(2)`/`pread(2)`, pages pulled in via `pread` become cache hits
//! for the existing `Arc<Mmap>` view. No change to the tensor storage
//! model, MTLBuffer construction, page alignment, or lifetime.
//!
//! # Spike scope
//!
//! * Warms the whole file, not just live tensor ranges. If this doesn't
//!   move end-to-end wall time, coalesced-range prefetch won't either.
//! * Runs against a single file; split shards should call it per shard
//!   (later: one global pool for all shards).
//! * Not wired into `open_one_shard`. Callers invoke `prefetch_file`
//!   explicitly. This is deliberately opt-in so it can be measured
//!   against status quo and MADV_WILLNEED as separate arms.
//!
//! # Design
//!
//! * `std::os::unix::fs::FileExt::read_at` is Rust stdlib's safe wrapper
//!   for `pread(2)`. `File: Sync` and `read_at(&self, ...)` takes
//!   `&self`, so N threads share the same descriptor without locking.
//!   No `unsafe`, no `libc`.
//! * Each worker owns a *contiguous* stripe of the file to preserve
//!   APFS read-ahead within its range.
//! * Per-worker scratch is a single reused `Vec<u8>` (default 16 MiB).
//!   Total anonymous memory ≈ `workers * chunk_bytes`, not file size.
//! * Read loop handles short reads and `Interrupted` per POSIX
//!   contract. Unexpected EOF (file truncated mid-warmup) surfaces as
//!   an `io::Error`.

use std::fs::File;
use std::io;
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Per-worker scratch buffer size. 16 MiB amortises the pread syscall
/// cost well past the interesting range (~40 syscalls/s per worker at
/// 6 GB/s) and keeps four-worker anonymous memory to 64 MiB.
pub const DEFAULT_CHUNK_BYTES: usize = 16 * 1024 * 1024;

/// Concurrent readers. Four matches what fio measurement and the v0.597
/// arena materialiser use to reach the internal-SSD ceiling on macOS;
/// higher values rarely help on external TB3 / USB 3.1 storage.
pub const DEFAULT_WORKERS: usize = 4;

/// Outcome of a prefetch call.
#[derive(Debug, Clone, Copy)]
pub struct PrefetchReport {
    /// Bytes *returned* by successful `pread` calls across all workers.
    /// IMPORTANT: this is not the same as bytes actually read from disk
    /// — a warm cache serves `pread` from RAM just as happily as a cold
    /// path serves it from block storage, and both increment this
    /// counter identically. For the physical-disk-read figure, snapshot
    /// [`crate::pid_metrics::PidSnapshot::now`] before and after the
    /// prefetch call and read `PidDelta::diskio_bytesread`.
    pub bytes: u64,
    /// Wall-clock time for the parallel warmup phase.
    pub wall: Duration,
    /// Workers used.
    pub workers: usize,
    /// Per-worker scratch buffer size in bytes.
    pub chunk_bytes: usize,
}

impl PrefetchReport {
    /// Bytes returned by `pread` divided by wall time. Reflects
    /// effective user-space throughput including any cache-hit
    /// portion, not physical disk throughput. See the field-level doc
    /// on `bytes`.
    pub fn effective_bytes_per_sec(&self) -> f64 {
        let s = self.wall.as_secs_f64();
        if s <= 0.0 { 0.0 } else { self.bytes as f64 / s }
    }
}

/// Pull the file's pages into the OS unified buffer cache using
/// `workers` parallel readers with `chunk_bytes` scratch each.
///
/// The bytes themselves are discarded — the goal is to populate the OS
/// page cache so that subsequent `mmap` access (via the existing
/// `Arc<Mmap>`) is a cache hit. Callers keep their existing mmap-backed
/// views; nothing about the tensor storage model changes.
///
/// Errors surface any per-worker `pread` failure with the offending
/// worker's context.
pub fn prefetch_file(
    path: impl AsRef<Path>,
    workers: usize,
    chunk_bytes: usize,
) -> io::Result<PrefetchReport> {
    if workers == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "workers must be > 0",
        ));
    }
    if chunk_bytes == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "chunk_bytes must be > 0",
        ));
    }

    let file = File::open(path.as_ref())?;
    let len = file.metadata()?.len();
    if len == 0 {
        return Ok(PrefetchReport {
            bytes: 0,
            wall: Duration::ZERO,
            workers,
            chunk_bytes,
        });
    }

    let file = &file;
    let bytes_read = AtomicU64::new(0);
    let bytes_read_ref = &bytes_read;

    // Contiguous stripe per worker preserves APFS read-ahead within
    // each worker's range. div_ceil so the final worker may get a
    // slightly smaller tail rather than any bytes going unread.
    let stripe = len.div_ceil(workers as u64);

    let started = Instant::now();
    std::thread::scope(|s| -> io::Result<()> {
        let mut handles = Vec::with_capacity(workers);
        for w in 0..workers {
            let start = (w as u64) * stripe;
            if start >= len {
                break;
            }
            let end = (start + stripe).min(len);
            handles.push(s.spawn(move || -> io::Result<()> {
                let mut scratch = vec![0u8; chunk_bytes];
                let mut offset = start;
                while offset < end {
                    let want = (chunk_bytes as u64).min(end - offset) as usize;
                    let mut got = 0;
                    while got < want {
                        match file.read_at(&mut scratch[got..want], offset + got as u64) {
                            Ok(0) => {
                                return Err(io::Error::new(
                                    io::ErrorKind::UnexpectedEof,
                                    format!(
                                        "worker {w}: unexpected EOF at offset {}",
                                        offset + got as u64
                                    ),
                                ));
                            }
                            Ok(n) => got += n,
                            Err(ref e) if e.kind() == io::ErrorKind::Interrupted => {}
                            Err(e) => return Err(e),
                        }
                    }
                    bytes_read_ref.fetch_add(want as u64, Ordering::Relaxed);
                    offset += want as u64;
                }
                Ok(())
            }));
        }
        for h in handles {
            match h.join() {
                Ok(inner) => inner?,
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
        Ok(())
    })?;

    Ok(PrefetchReport {
        bytes: bytes_read.load(Ordering::Relaxed),
        wall: started.elapsed(),
        workers,
        chunk_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    /// Tiny bespoke fixture holder: avoids pulling in `tempfile` as a
    /// dev-dep just for these tests. Drops the file on scope exit.
    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn make_fixture(bytes: usize) -> Fixture {
        let pid = std::process::id();
        let name = format!(
            "qwen-prefetch-test-{}-{}-{}.bin",
            pid,
            bytes,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&path).expect("create fixture");
        let chunk = (0u8..=255).collect::<Vec<_>>();
        let mut written = 0;
        while written < bytes {
            let n = (bytes - written).min(chunk.len());
            f.write_all(&chunk[..n]).expect("write fixture");
            written += n;
        }
        f.sync_all().expect("sync fixture");
        Fixture(path)
    }

    #[test]
    fn empty_workers_rejected() {
        let f = make_fixture(1024);
        assert!(prefetch_file(&f.0, 0, 4096).is_err());
    }

    #[test]
    fn empty_chunk_rejected() {
        let f = make_fixture(1024);
        assert!(prefetch_file(&f.0, 1, 0).is_err());
    }

    #[test]
    fn reads_full_file_across_workers() {
        // Small fixture, small chunk, many workers => exercises the
        // stripe partition and read loop.
        let size = 1 << 20; // 1 MiB
        let f = make_fixture(size);
        let report = prefetch_file(&f.0, 4, 64 * 1024).expect("prefetch");
        assert_eq!(report.bytes, size as u64);
        assert_eq!(report.workers, 4);
        assert_eq!(report.chunk_bytes, 64 * 1024);
    }

    #[test]
    fn missing_file_surfaces_io_error() {
        let err = prefetch_file("/nonexistent/path/definitely", 1, 4096).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }
}
