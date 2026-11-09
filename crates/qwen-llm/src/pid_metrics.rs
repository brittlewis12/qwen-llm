//! Per-process I/O counters via `proc_pid_rusage(RUSAGE_INFO_V2)`.
//!
//! Why this module exists: `PrefetchReport.bytes` (in [`crate::prefetch`])
//! counts bytes *returned* by `pread`, not bytes actually read from
//! disk — warm cached reads increment it exactly like cold reads. To
//! know how many bytes each measurement arm actually pulled from
//! storage, we need the kernel's own counter.
//!
//! macOS exposes this via `proc_pid_rusage(pid, RUSAGE_INFO_V2, &out)`:
//!
//! * `ri_pageins`           — major page-ins served by the pager
//! * `ri_diskio_bytesread`  — physical bytes read from block devices
//! * `ri_diskio_byteswritten` — physical bytes written
//! * `ri_resident_size`     — process resident-set size
//! * `ri_phys_footprint`    — Apple's physical-footprint measure
//!
//! Snapshot at the start and end of each phase; the delta is the cost
//! attributable to that phase (whole process, including worker
//! threads). No sudo needed. Cheap enough to call around every
//! interesting boundary.

use std::io;

/// Snapshot of per-process I/O and memory counters at one instant.
///
/// Fields are the raw kernel counters, all monotonically increasing
/// (except memory-footprint fields which are current values). Compute
/// deltas by subtracting two snapshots.
#[derive(Debug, Clone, Copy)]
pub struct PidSnapshot {
    pub pageins: u64,
    pub diskio_bytesread: u64,
    pub diskio_byteswritten: u64,
    pub resident_size: u64,
    pub phys_footprint: u64,
}

impl PidSnapshot {
    /// Take a snapshot of the calling process's rusage.
    pub fn now() -> io::Result<Self> {
        // proc_pid_rusage writes to a rusage_info_v* struct whose
        // shape matches the requested flavor. rusage_info_v2 is the
        // minimum flavor exposing `ri_diskio_bytesread`; later
        // flavors extend it with strictly more fields at the same
        // offsets. libc gives us the struct type directly.
        //
        // SAFETY: pid is our own pid (always valid); the buffer we
        // pass is exactly the size the requested flavor expects;
        // libc encodes the correct FFI signature.
        let mut info = std::mem::MaybeUninit::<libc::rusage_info_v2>::zeroed();
        let ret = unsafe {
            libc::proc_pid_rusage(
                libc::getpid(),
                libc::RUSAGE_INFO_V2,
                info.as_mut_ptr().cast(),
            )
        };
        if ret != 0 {
            return Err(io::Error::last_os_error());
        }
        let info = unsafe { info.assume_init() };
        Ok(Self {
            pageins: info.ri_pageins,
            diskio_bytesread: info.ri_diskio_bytesread,
            diskio_byteswritten: info.ri_diskio_byteswritten,
            resident_size: info.ri_resident_size,
            phys_footprint: info.ri_phys_footprint,
        })
    }
}

/// Difference between two [`PidSnapshot`]s, as (later - earlier).
#[derive(Debug, Clone, Copy)]
pub struct PidDelta {
    /// Major page-ins during the interval.
    pub pageins: u64,
    /// Physical bytes read from block devices during the interval.
    pub diskio_bytesread: u64,
    /// Physical bytes written to block devices during the interval.
    pub diskio_byteswritten: u64,
    /// Change in resident-set size (may be negative; recorded as i64).
    pub resident_size_delta: i64,
    /// Change in phys_footprint.
    pub phys_footprint_delta: i64,
}

impl PidDelta {
    pub fn between(before: PidSnapshot, after: PidSnapshot) -> Self {
        Self {
            pageins: after.pageins.saturating_sub(before.pageins),
            diskio_bytesread: after
                .diskio_bytesread
                .saturating_sub(before.diskio_bytesread),
            diskio_byteswritten: after
                .diskio_byteswritten
                .saturating_sub(before.diskio_byteswritten),
            resident_size_delta: after.resident_size as i64 - before.resident_size as i64,
            phys_footprint_delta: after.phys_footprint as i64 - before.phys_footprint as i64,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    #[test]
    fn snapshot_now_succeeds() {
        let s = PidSnapshot::now().expect("proc_pid_rusage should succeed");
        // Sanity: the process has been alive, so pageins is a small-ish
        // nonzero-or-zero number. We can't assert much; just prove the
        // call works.
        let _ = s.pageins;
        let _ = s.diskio_bytesread;
    }

    #[test]
    fn reading_a_temp_file_shows_up_in_delta() {
        // Write a temp file cold-ish and read it; we expect either a
        // pageins bump or a diskio_bytesread bump (or both). The file
        // may be entirely in write-back cache and not touch disk, so
        // we can't assert either strictly, but the API must succeed
        // and produce a coherent delta.
        let bytes = 64 * 1024 * 1024;
        let directory = crate::test_fixtures::TestDir::new("pid-metrics");
        let path = directory.path().join("data.bin");
        {
            let mut f = std::fs::File::create(&path).expect("create");
            let chunk = vec![0xABu8; 1 << 20];
            for _ in 0..(bytes / chunk.len()) {
                use std::io::Write;
                f.write_all(&chunk).expect("write");
            }
            f.sync_all().expect("sync");
        }
        let before = PidSnapshot::now().expect("before");
        {
            let mut f = std::fs::File::open(&path).expect("open");
            let mut buf = Vec::new();
            f.read_to_end(&mut buf).expect("read");
            assert_eq!(buf.len(), bytes);
        }
        let after = PidSnapshot::now().expect("after");
        let _delta = PidDelta::between(before, after);
        // These are process-wide physical I/O counters: delayed writeback and
        // other threads may contribute writes during a read-only interval.
        assert!(after.diskio_byteswritten >= before.diskio_byteswritten);
        assert!(after.diskio_bytesread >= before.diskio_bytesread);
        let _ = std::fs::remove_file(&path);
    }
}

#[cfg(test)]
mod delta_tests {
    use super::*;

    #[test]
    fn deltas_use_owned_snapshots_and_saturate_counters() {
        let before = PidSnapshot {
            pageins: 8,
            diskio_bytesread: 1024,
            diskio_byteswritten: 4096,
            resident_size: 100,
            phys_footprint: 200,
        };
        let after = PidSnapshot {
            pageins: 10,
            diskio_bytesread: 1536,
            diskio_byteswritten: 8192,
            resident_size: 80,
            phys_footprint: 230,
        };
        let delta = PidDelta::between(before, after);
        assert_eq!(
            (
                delta.pageins,
                delta.diskio_bytesread,
                delta.diskio_byteswritten
            ),
            (2, 512, 4096)
        );
        assert_eq!(
            (delta.resident_size_delta, delta.phys_footprint_delta),
            (-20, 30)
        );
        let reversed = PidDelta::between(after, before);
        assert_eq!(
            (
                reversed.pageins,
                reversed.diskio_bytesread,
                reversed.diskio_byteswritten
            ),
            (0, 0, 0)
        );
    }
}
