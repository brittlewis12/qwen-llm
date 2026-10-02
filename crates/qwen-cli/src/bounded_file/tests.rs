use super::*;
use std::ffi::CString;
use std::io::{Seek, SeekFrom, Write};
use std::os::fd::AsRawFd;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{DirBuilderExt, symlink};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

struct Fixture(PathBuf);

impl Fixture {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let path = std::env::temp_dir().join(format!(
            "qwen-bounded-file-{}-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
            NEXT.fetch_add(1, Ordering::Relaxed),
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&path)
            .unwrap();
        Self(path)
    }

    fn file(&self, name: &str, bytes: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn exact_and_bounded_reads_accept_empty_files_and_inclusive_boundary() {
    let fixture = Fixture::new();
    let empty = fixture.file("empty", b"");
    assert!(read_regular_file_bounded(&empty, 0).unwrap().is_empty());
    assert!(read_regular_file_exact(&empty, 0).unwrap().is_empty());
    let path = fixture.file("data", b"payload");
    assert_eq!(read_regular_file_bounded(&path, 7).unwrap(), b"payload");
    assert_eq!(read_regular_file_exact(&path, 7).unwrap(), b"payload");
    assert_eq!(read_regular_file_bounded(&path, 8).unwrap(), b"payload");
    assert!(
        read_regular_file_bounded(&path, 6)
            .unwrap_err()
            .to_string()
            .contains("exceeds limit 6")
    );
    for length in [0, 6, 8] {
        assert!(
            read_regular_file_exact(&path, length)
                .unwrap_err()
                .to_string()
                .contains("!= expected")
        );
    }
}

#[test]
fn leaf_symlinks_directories_and_special_files_are_refused() {
    let fixture = Fixture::new();
    fixture.file("target", b"data");
    let link = fixture.0.join("link");
    symlink("target", &link).unwrap();
    let dangling = fixture.0.join("dangling");
    symlink("missing", &dangling).unwrap();
    let fifo = fixture.0.join("fifo");
    let name = CString::new(fifo.as_os_str().as_bytes()).unwrap();
    // The NUL-terminated pathname is valid for this call and inside our fixture.
    assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
    for path in [&link, &dangling, &fixture.0, &fifo] {
        let error = open_regular_file(path).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("must be a regular non-symlink file"),
            "{error:#}"
        );
    }
}

#[test]
fn opened_descriptor_is_nonblocking_and_reports_its_own_length() {
    let fixture = Fixture::new();
    let path = fixture.file("data", b"payload");
    let (file, length) = open_regular_file(&path).unwrap();
    assert_eq!(length, 7);
    // file owns the descriptor for the duration of this non-mutating query.
    let flags = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_GETFL) };
    assert_ne!(flags, -1);
    assert_ne!(flags & libc::O_NONBLOCK, 0);
    assert_eq!(
        read_opened_file_exact(file, &path, length).unwrap(),
        b"payload"
    );
}

#[test]
fn opened_reads_reject_shrink_and_growth_without_timing_assumptions() {
    let fixture = Fixture::new();
    let short = fixture.file("short", b"payload");
    let (file, length) = open_regular_file(&short).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&short)
        .unwrap()
        .set_len(3)
        .unwrap();
    let error = read_opened_file_exact(file, &short, length).unwrap_err();
    assert!(error.to_string().contains("read exact contents"));

    let long = fixture.file("long", b"payload");
    let (file, length) = open_regular_file(&long).unwrap();
    OpenOptions::new()
        .append(true)
        .open(&long)
        .unwrap()
        .write_all(b"!")
        .unwrap();
    let error = read_opened_file_exact(file, &long, length).unwrap_err();
    assert!(error.to_string().contains("grew while it was being read"));
}

#[test]
fn path_replacement_does_not_reopen_or_retarget_the_descriptor() {
    let fixture = Fixture::new();
    let path = fixture.file("data", b"original");
    let (file, length) = open_regular_file(&path).unwrap();
    let replacement = fixture.file("replacement", b"changed length");
    std::fs::rename(replacement, &path).unwrap();
    assert_eq!(
        read_opened_file_exact(file, &path, length).unwrap(),
        b"original"
    );
    assert_eq!(
        read_regular_file_bounded(&path, 20).unwrap(),
        b"changed length"
    );
}

#[test]
fn opened_reads_keep_current_offset_and_leave_same_length_integrity_to_callers() {
    let fixture = Fixture::new();
    let path = fixture.file("data", b"prefix:payload");
    let (mut file, _) = open_regular_file(&path).unwrap();
    file.seek(SeekFrom::Start(7)).unwrap();
    assert_eq!(read_opened_file_exact(file, &path, 7).unwrap(), b"payload");

    let (file, length) = open_regular_file(&path).unwrap();
    OpenOptions::new()
        .write(true)
        .open(&path)
        .unwrap()
        .write_all(b"PREFIX:PAYLOAD")
        .unwrap();
    assert_eq!(
        read_opened_file_exact(file, &path, length).unwrap(),
        b"PREFIX:PAYLOAD"
    );
}

#[test]
fn opened_read_allocation_failure_is_an_error() {
    let fixture = Fixture::new();
    let path = fixture.file("data", b"");
    let (file, _) = open_regular_file(&path).unwrap();
    let error = read_opened_file_exact(file, &path, usize::MAX).unwrap_err();
    assert!(error.to_string().contains("allocate"));
}
