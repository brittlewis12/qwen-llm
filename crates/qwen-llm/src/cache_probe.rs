//! Targeted per-file eviction and observability for the OS unified
//! buffer cache (macOS / XNU).
//!
//! # Why
//!
//! On macOS the standard `sudo purge` is global — it drops the entire
//! unified buffer cache, which is disruptive to unrelated work in
//! flight (editors, compilers, other benchmarks). For iterating on
//! cold-load measurements we want a *targeted* mechanism that evicts
//! only the pages backing one specific file.
//!
//! # What works
//!
//! `msync(addr, len, MS_SYNC | MS_INVALIDATE)` on a `MAP_SHARED |
//! PROT_READ` mapping of the target file. On XNU this routes through
//! `VM_SYNC_INVALIDATE` into the external vnode pager and evicts the
//! backing VM object's clean file pages — so subsequent `mincore(2)`
//! reports 0 resident pages for the file, even in a fresh mapping.
//!
//! # What does not work
//!
//! * `MADV_DONTNEED` / `MADV_FREE` — deactivate but do not evict.
//! * `MAP_NOCACHE` — reclamation-priority hint only; not cache-bypass.
//! * `F_NOCACHE` / `F_GLOBAL_NOCACHE` — affect fd-based reads only;
//!   mmap faults still populate the unified cache.
//! * `MADV_DONTNEED` + memory pressure — collateral eviction of
//!   unrelated files. Not targeted.
//!
//! # Observability
//!
//! `mincore(2)` on macOS queries the underlying VM object, not just
//! this process's page table, so it reports unified-buffer-cache
//! residency — not merely "faulted into this process". This is what
//! we want: we can verify eviction happened without opening the
//! target file for real use.

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Residency report for a page range of a file, as observed via
/// `mincore(2)`.
#[derive(Debug, Clone, Copy)]
pub struct ResidencyReport {
    pub total_pages: usize,
    pub resident_pages: usize,
}

impl ResidencyReport {
    pub fn resident_fraction(&self) -> f64 {
        if self.total_pages == 0 {
            0.0
        } else {
            self.resident_pages as f64 / self.total_pages as f64
        }
    }
    pub fn resident_bytes(&self) -> u64 {
        self.resident_pages as u64 * host_page_size() as u64
    }
}

fn host_page_size() -> usize {
    // Apple Silicon is 16 KiB; Intel Macs 4 KiB. Query at runtime.
    let ps = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if ps > 0 { ps as usize } else { 4096 }
}

/// mmap `path` read-only shared, invalidate its unified-buffer-cache
/// pages, verify residency post-eviction, then drop the mapping.
///
/// This is designed to be called *between* measurement arms to reset
/// the cache state of one target file without touching global cache.
///
/// Returns the residency reports *before* and *after* the msync call,
/// so the caller can decide whether the eviction was effective. On
/// XNU this reliably drops residency to zero for clean, exclusively-
/// mapped files; if another process holds a mapping, its residency
/// will be re-materialised on that process's next access. Do not call
/// this while another process is actively using the same file.
pub fn invalidate_file_cache(path: impl AsRef<Path>) -> io::Result<InvalidateReport> {
    let path = path.as_ref();
    let file = File::open(path)?;
    let len = file.metadata()?.len() as usize;
    if len == 0 {
        return Ok(InvalidateReport {
            before: ResidencyReport {
                total_pages: 0,
                resident_pages: 0,
            },
            after: ResidencyReport {
                total_pages: 0,
                resident_pages: 0,
            },
        });
    }

    // Disposable read-only shared mapping. memmap2 uses MAP_SHARED via
    // Mmap::map so we could use it here too, but for a scoped probe we
    // want a plain call that we drop before starting the arm.
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    // Guard so we always munmap even on early return.
    struct Guard {
        addr: *mut libc::c_void,
        len: usize,
    }
    impl Drop for Guard {
        fn drop(&mut self) {
            unsafe {
                libc::munmap(self.addr, self.len);
            }
        }
    }
    let _guard = Guard { addr, len };

    let before = mincore_at(addr, len)?;

    let ret = unsafe { libc::msync(addr, len, libc::MS_SYNC | libc::MS_INVALIDATE) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }

    let after = mincore_at(addr, len)?;

    Ok(InvalidateReport { before, after })
}

/// Snapshot of the residency of the file backing an open mmap, without
/// modifying anything.
///
/// Prefer [`probe_fd_residency`] when a caller already holds an open
/// file description; that avoids reopening by path and its TOCTOU
/// window.
pub fn probe_file_residency(path: impl AsRef<Path>) -> io::Result<ResidencyReport> {
    let file = File::open(path.as_ref())?;
    probe_fd_residency(&file)
}

/// Same as [`probe_file_residency`] but takes a borrowed [`File`] so
/// callers with a retained descriptor (e.g. `GgufShard.file`) avoid
/// the path reopen.
pub fn probe_fd_residency(file: &File) -> io::Result<ResidencyReport> {
    let len = file.metadata()?.len() as usize;
    if len == 0 {
        return Ok(ResidencyReport {
            total_pages: 0,
            resident_pages: 0,
        });
    }
    let addr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ,
            libc::MAP_SHARED,
            file.as_raw_fd(),
            0,
        )
    };
    if addr == libc::MAP_FAILED {
        return Err(io::Error::last_os_error());
    }
    let report = mincore_at(addr, len);
    unsafe {
        libc::munmap(addr, len);
    }
    report
}

fn mincore_at(addr: *mut libc::c_void, len: usize) -> io::Result<ResidencyReport> {
    let page = host_page_size();
    let n_pages = len.div_ceil(page);
    let mut vec = vec![0i8; n_pages];
    // SAFETY: addr, len is a valid mapping; vec is n_pages long.
    // mincore's third arg is `*mut char` in POSIX (`c_char` in libc,
    // which is signed on macOS aarch64).
    let ret = unsafe { libc::mincore(addr, len, vec.as_mut_ptr()) };
    if ret != 0 {
        return Err(io::Error::last_os_error());
    }
    // MINCORE_INCORE = 0x1
    let resident = vec.iter().filter(|&&b| (b & 0x1) != 0).count();
    Ok(ResidencyReport {
        total_pages: n_pages,
        resident_pages: resident,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct InvalidateReport {
    pub before: ResidencyReport,
    pub after: ResidencyReport,
}

/// Estimate of memory the kernel could hand out without pressuring
/// currently-active work. Sums the vm_stat categories that XNU treats
/// as cheaply reclaimable: `free`, `inactive`, `speculative`, and
/// `purgeable`. Excludes wired, active, and compressor pages.
///
/// Used by the prefetch policy for headroom telemetry. The current policy logs
/// when its conservative bound would fail but does not skip warmup; a hard gate
/// awaits a controlled pressure-regime experiment.
///
/// Errors surface any `host_statistics64` failure.
pub fn available_memory_bytes() -> io::Result<u64> {
    let host = unsafe { libc::mach_host_self() };
    // SAFETY: host is a valid mach port; we pass a properly sized and
    // aligned buffer for the requested flavor; count is initialized to
    // the required size in units of integer_t.
    let mut stats = std::mem::MaybeUninit::<libc::vm_statistics64>::zeroed();
    let mut count: libc::mach_msg_type_number_t = libc::HOST_VM_INFO64_COUNT;
    let ret = unsafe {
        libc::host_statistics64(
            host,
            libc::HOST_VM_INFO64,
            stats.as_mut_ptr().cast(),
            &mut count,
        )
    };
    if ret != libc::KERN_SUCCESS {
        return Err(io::Error::other(format!(
            "host_statistics64(HOST_VM_INFO64) returned kern={ret}"
        )));
    }
    let stats = unsafe { stats.assume_init() };
    let page = host_page_size() as u64;
    let reclaimable = stats.free_count as u64
        + stats.inactive_count as u64
        + stats.speculative_count as u64
        + stats.purgeable_count as u64;
    Ok(reclaimable * page)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::path::PathBuf;

    struct Fixture(PathBuf);
    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn make_fixture(bytes: usize, tag: &str) -> Fixture {
        let name = format!(
            "qwen-cache-probe-{}-{}-{}.bin",
            std::process::id(),
            tag,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        let mut f = std::fs::File::create(&path).expect("create fixture");
        let chunk = (0u8..=255).collect::<Vec<_>>();
        let mut w = 0;
        while w < bytes {
            let n = (bytes - w).min(chunk.len());
            f.write_all(&chunk[..n]).expect("write");
            w += n;
        }
        f.sync_all().expect("sync");
        Fixture(path)
    }

    #[test]
    fn available_memory_is_positive_and_bounded_by_total() {
        // Sanity: the call succeeds on macOS, returns a positive value,
        // and is less than or equal to total physical memory (from the
        // hw.memsize sysctl).
        let avail = available_memory_bytes().expect("available_memory_bytes");
        assert!(avail > 0, "expected positive available memory, got {avail}");

        let total_pages = unsafe { libc::sysconf(libc::_SC_PHYS_PAGES) } as u64;
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as u64;
        let total = total_pages * page;
        assert!(
            avail <= total,
            "available {avail} exceeds total physical {total}"
        );
    }

    #[test]
    fn invalidate_drops_residency() {
        // Small file (4 MiB) so we're not hammering disk.
        let f = make_fixture(4 * 1024 * 1024, "invalidate");
        // Warm the cache by reading through.
        {
            use std::io::Read;
            let mut file = std::fs::File::open(&f.0).expect("open");
            let mut buf = Vec::new();
            file.read_to_end(&mut buf).expect("read");
            assert_eq!(buf.len(), 4 * 1024 * 1024);
        }
        let before = probe_file_residency(&f.0).expect("probe before");
        assert!(
            before.resident_pages > 0,
            "file should be resident after read"
        );

        let report = invalidate_file_cache(&f.0).expect("invalidate");
        // Codex's XNU trace: MS_INVALIDATE drops clean file pages to 0.
        // This might fail if another process holds a mapping, but for a
        // scoped temp file it should be reliable.
        assert_eq!(
            report.after.resident_pages, 0,
            "invalidate should evict all pages"
        );
    }
}
