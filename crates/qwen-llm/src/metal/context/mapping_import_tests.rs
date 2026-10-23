//! Model-free blit and compute import witness. Run alone, in release mode,
//! under an external watchdog. The raw counters overlap and are not distinct
//! resident-byte totals; fixture creation warms the file cache, and other process
//! activity adds noise. The compute PSO is warmed before fixture creation;
//! context/PSO and driver bookkeeping are not planned fixture resource costs.
//! Phases: 0 before mmap/import, 1 after import, 2 after all checked GPU blits,
//! 3 after all checked GPU compute copies, 4 after dropping case resources and
//! draining their autorelease pool. The historical test name is kept for scripts.

use super::*;
use mach2::task_info::{TASK_EVENTS_INFO, TASK_EVENTS_INFO_COUNT, task_events_info};
use memmap2::MmapOptions;
use std::os::unix::fs::{DirBuilderExt, FileExt};

const PAYLOAD_BYTES: usize = 8 * 1024 * 1024;
const FILE_BYTES: usize = PAYLOAD_BYTES + 37;
const MAX_PAGE_BYTES: usize = 16 * 1024;
const MAX_SPAN_BYTES: usize = PAYLOAD_BYTES + MAX_PAGE_BYTES;
const COPY_BYTES: usize = 64 * 1024;
const GUARD_BYTES: usize = 64;
const READBACK_BYTES: usize = COPY_BYTES + 2 * GUARD_BYTES;
const MAX_READBACK_UPPER_BYTES: usize = COPY_BYTES + MAX_PAGE_BYTES;
const RESERVE_BYTES: usize = 1024 * 1024;
// Conservatively budget three full spans (file cache, mapping/COW, possible
// import charge), not a claim that those charges represent independent pages.
// This bounds planned test resources, not opaque context/driver bookkeeping.
const PLANNED_LIVE_BYTES: usize =
    3 * MAX_SPAN_BYTES + MAX_READBACK_UPPER_BYTES + COPY_BYTES + RESERVE_BYTES;
const _: () = assert!(PLANNED_LIVE_BYTES == 26_411_008);
const _: () = assert!(PLANNED_LIVE_BYTES < 32 * 1024 * 1024);

#[derive(Clone, Copy)]
enum MapKind {
    SharedAligned,
    SharedEof,
    OldPrivateEof,
}

impl MapKind {
    fn name(self) -> &'static str {
        match self {
            Self::SharedAligned => "shared_aligned",
            Self::SharedEof => "shared_eof",
            Self::OldPrivateEof => "old_private_eof",
        }
    }

    fn span(self, page_bytes: usize) -> usize {
        match self {
            Self::SharedAligned => PAYLOAD_BYTES,
            Self::SharedEof | Self::OldPrivateEof => FILE_BYTES.next_multiple_of(page_bytes),
        }
    }
}

fn fixture_byte(offset: usize) -> u8 {
    if offset >= PAYLOAD_BYTES {
        // The 37 non-padding EOF bytes are source canaries, not zero fill.
        return 0xc1u8.wrapping_add((offset - PAYLOAD_BYTES) as u8);
    }
    let mixed = (offset as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
    (mixed ^ (mixed >> 17) ^ (mixed >> 37)) as u8
}

#[derive(PartialEq, Eq)]
struct FileIdentity {
    device: u64,
    inode: u64,
    length: u64,
    links: u64,
    mode: u32,
    modified: (i64, i64),
    changed: (i64, i64),
}

impl FileIdentity {
    fn read(metadata: &std::fs::Metadata) -> Self {
        assert!(metadata.is_file(), "fixture must remain a regular file");
        Self {
            device: metadata.dev(),
            inode: metadata.ino(),
            length: metadata.len(),
            links: metadata.nlink(),
            mode: metadata.mode(),
            modified: (metadata.mtime(), metadata.mtime_nsec()),
            changed: (metadata.ctime(), metadata.ctime_nsec()),
        }
    }
}

struct ImmutableFixture {
    directory: PathBuf,
    file: File,
    identity: FileIdentity,
}

impl ImmutableFixture {
    fn new() -> Self {
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("fixture clock")
            .as_nanos();
        let directory = std::env::temp_dir().join(format!(
            "qwen-mapping-import-{}-{nonce}",
            std::process::id()
        ));
        std::fs::DirBuilder::new()
            .mode(0o700)
            .create(&directory)
            .expect("create private fixture directory");
        let path = directory.join("immutable.bin");
        let mut writer = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&path)
            .expect("create fixture");
        let mut chunk = [0u8; COPY_BYTES];
        for start in (0..FILE_BYTES).step_by(COPY_BYTES) {
            let length = (FILE_BYTES - start).min(COPY_BYTES);
            for (index, byte) in chunk[..length].iter_mut().enumerate() {
                *byte = fixture_byte(start + index);
            }
            writer.write_all(&chunk[..length]).expect("write fixture");
        }
        writer.sync_all().expect("sync fixture");
        writer
            .set_permissions(std::fs::Permissions::from_mode(0o400))
            .expect("make fixture read only");
        drop(writer);
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
            .open(&path)
            .expect("open read-only fixture");
        let identity = FileIdentity::read(&file.metadata().expect("fixture metadata"));
        assert_eq!(identity.length, FILE_BYTES as u64);
        assert_eq!(identity.links, 1);
        assert_eq!(identity.mode & 0o777, 0o400);
        let fixture = Self {
            directory,
            file,
            identity,
        };
        fixture.assert_identity();
        fixture
    }

    fn assert_identity(&self) {
        let opened = FileIdentity::read(&self.file.metadata().expect("open fixture metadata"));
        let linked = FileIdentity::read(
            &std::fs::symlink_metadata(self.directory.join("immutable.bin"))
                .expect("linked fixture metadata"),
        );
        assert!(opened == self.identity, "open fixture identity changed");
        assert!(linked == self.identity, "linked fixture identity changed");
    }

    fn assert_unchanged(&self) {
        self.assert_identity();
        let mut chunk = [0u8; COPY_BYTES];
        for start in (0..FILE_BYTES).step_by(COPY_BYTES) {
            let length = (FILE_BYTES - start).min(COPY_BYTES);
            self.file
                .read_exact_at(&mut chunk[..length], start as u64)
                .expect("read original file after unmapping");
            for (index, &byte) in chunk[..length].iter().enumerate() {
                assert_eq!(byte, fixture_byte(start + index), "source file changed");
            }
        }
        self.assert_identity();
    }
}

impl Drop for ImmutableFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.directory.join("immutable.bin"));
        let _ = std::fs::remove_dir(&self.directory);
    }
}

#[derive(Clone, Copy, serde::Serialize)]
struct PhaseSnapshot {
    phase: u8,
    map_kind: &'static str,
    span_bytes: usize,
    planned_live_bytes: usize,
    device_current_allocated_size: u64,
    source_buffer_allocated_size: Option<u64>,
    readback_buffer_allocated_size: Option<u64>,
    task_events_result: i32,
    task_events_count: u32,
    faults: Option<u32>,
    pageins: Option<u32>,
    cow_faults: Option<u32>,
    task_vm_result: i32,
    task_vm_count: u32,
    resident_bytes: Option<u64>,
    phys_footprint_bytes: Option<u64>,
}

fn returned_field(result: i32, count: u32, capacity: u32, offset: usize, bytes: usize) -> bool {
    result == 0
        && count <= capacity
        && (count as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .zip(offset.checked_add(bytes))
            .is_some_and(|(returned, end)| returned >= end)
}

fn mapping_import_snapshot(
    ctx: &MetalContext,
    kind: MapKind,
    span: usize,
    phase: u8,
    source: Option<&Buffer>,
    readback: Option<&Buffer>,
) {
    let device_current_allocated_size = ctx.current_allocated_size();
    let source_buffer_allocated_size = source.map(|buffer| buffer.allocatedSize() as u64);
    let readback_buffer_allocated_size = readback.map(|buffer| buffer.allocatedSize() as u64);
    let mut events = task_events_info::default();
    let mut events_count = TASK_EVENTS_INFO_COUNT;
    // SAFETY: events is writable for the supplied number of natural_t words.
    let events_result = unsafe {
        task_info(
            mach_task_self_,
            TASK_EVENTS_INFO as i32,
            std::ptr::addr_of_mut!(events).cast::<i32>(),
            &mut events_count,
        )
    };
    let events_available = events_result == 0 && events_count == TASK_EVENTS_INFO_COUNT;
    let mut vm = std::mem::MaybeUninit::<TaskVmInfo>::zeroed();
    let vm_capacity = u32::try_from(std::mem::size_of::<TaskVmInfo>() / 4).expect("VM word count");
    let mut vm_count = vm_capacity;
    // SAFETY: vm is writable for vm_capacity natural_t words. Each field below
    // is read unaligned only if the returned revision includes the entire field.
    let vm_result = unsafe {
        task_info(
            mach_task_self_,
            super::TASK_VM_INFO,
            vm.as_mut_ptr().cast::<i32>(),
            &mut vm_count,
        )
    };
    let resident_bytes = returned_field(
        vm_result,
        vm_count,
        vm_capacity,
        std::mem::offset_of!(TaskVmInfo, resident_size),
        std::mem::size_of::<u64>(),
    )
    .then(|| unsafe { std::ptr::addr_of!((*vm.as_ptr()).resident_size).read_unaligned() });
    let phys_footprint_bytes = returned_field(
        vm_result,
        vm_count,
        vm_capacity,
        std::mem::offset_of!(TaskVmInfo, phys_footprint),
        std::mem::size_of::<u64>(),
    )
    .then(|| unsafe { std::ptr::addr_of!((*vm.as_ptr()).phys_footprint).read_unaligned() });
    let snapshot = PhaseSnapshot {
        phase,
        map_kind: kind.name(),
        span_bytes: span,
        planned_live_bytes: PLANNED_LIVE_BYTES,
        device_current_allocated_size,
        source_buffer_allocated_size,
        readback_buffer_allocated_size,
        task_events_result: events_result,
        task_events_count: events_count,
        // TASK_EVENTS_INFO exposes wrapping 32-bit counters, not byte counts.
        faults: events_available.then_some(events.faults as u32),
        pageins: events_available.then_some(events.pageins as u32),
        cow_faults: events_available.then_some(events.cow_faults as u32),
        task_vm_result: vm_result,
        task_vm_count: vm_count,
        resident_bytes,
        phys_footprint_bytes,
    };
    eprintln!(
        "{}",
        serde_json::to_string(&snapshot).expect("numeric phase JSON")
    );
}

fn blit_check_mapping_import(ctx: &MetalContext, source: &Buffer, readback: &Buffer, span: usize) {
    assert_eq!(source.length(), span);
    assert_eq!(readback.length(), READBACK_BYTES);
    for start in (0..span).step_by(COPY_BYTES) {
        let length = (span - start).min(COPY_BYTES);
        assert!(start.is_multiple_of(4) && length.is_multiple_of(4));
        objc2::rc::autoreleasepool(|_| {
            // SAFETY: this is the owned shared destination, idle between checked
            // completions. No CPU access ever reads or writes the source mmap.
            unsafe {
                readback
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .write_bytes(0xa5, READBACK_BYTES)
            };
            let command = ctx.queue.commandBuffer().expect("import blit command");
            assert!(command.retainedReferences());
            let blit = BlitEncoder::try_begin(&command).expect("import blit encoder");
            blit.copy_buffer(
                source,
                start as u64,
                readback,
                GUARD_BYTES as u64,
                length as u64,
            );
            blit.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("import blit completed");
            check_mapping_import_readback(readback, start, length);
        });
    }
}

fn compute_check_mapping_import(
    ctx: &MetalContext,
    source: &Buffer,
    readback: &Buffer,
    span: usize,
) {
    assert_eq!(source.length(), span);
    assert_eq!(readback.length(), READBACK_BYTES);
    assert!(span.is_multiple_of(4) && READBACK_BYTES.is_multiple_of(4));
    let source = MetalTensor {
        buffer: source.clone(),
        offset: 0,
        shape: vec![(span / 4) as u64],
        dtype: GgmlType::I32,
        provenance: MetalTensorProvenance::RetainedGgufReadOnly,
    };
    let readback_tensor = MetalTensor {
        buffer: readback.clone(),
        offset: 0,
        shape: vec![(READBACK_BYTES / 4) as u64],
        dtype: GgmlType::I32,
        provenance: MetalTensorProvenance::OwnedWritable,
    };
    assert_eq!(
        source.provenance(),
        MetalTensorProvenance::RetainedGgufReadOnly
    );
    assert!(!source.is_writable());
    for start in (0..span).step_by(COPY_BYTES) {
        let length = (span - start).min(COPY_BYTES);
        assert!(start.is_multiple_of(4) && length.is_multiple_of(4));
        objc2::rc::autoreleasepool(|_| {
            let destination =
                readback_tensor.view_subrange((GUARD_BYTES / 4) as u64, vec![(length / 4) as u64]);
            assert_eq!(destination.offset, GUARD_BYTES as u64);
            assert_eq!(destination.n_bytes(), length as u64);
            assert_eq!(
                destination.provenance(),
                MetalTensorProvenance::OwnedWritable
            );
            assert!(destination.is_writable());
            // SAFETY: the same owned shared destination is idle after the prior
            // checked completion. The retained source is never CPU-accessed.
            unsafe {
                readback
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .write_bytes(0xa5, READBACK_BYTES)
            };
            let command = ctx.queue.commandBuffer().expect("import compute command");
            assert!(command.retainedReferences());
            let encoder = KernelEncoder::begin(&command);
            encoder.note_read(&source);
            encoder.note_write(&destination);
            // The production helper binds both tensors through set_tensor;
            // integer copies preserve every raw bit, including float NaN bits.
            encode_copy_offset_i32(ctx, &encoder, &source, start / 4, &destination, length / 4)
                .expect("read-only import compute copy");
            encoder.end();
            command.commit();
            crate::metal::wait_completed(&command).expect("import compute completed");
            check_mapping_import_readback(readback, start, length);
        });
    }
}

fn check_mapping_import_readback(readback: &Buffer, start: usize, length: usize) {
    // SAFETY: callers check GPU completion before reading this shared destination.
    let bytes = unsafe {
        std::slice::from_raw_parts(readback.contents().as_ptr().cast::<u8>(), READBACK_BYTES)
    };
    assert!(
        bytes[..GUARD_BYTES].iter().all(|&byte| byte == 0xa5),
        "leading canary"
    );
    assert!(
        bytes[GUARD_BYTES + length..]
            .iter()
            .all(|&byte| byte == 0xa5),
        "trailing canary"
    );
    for (index, &byte) in bytes[GUARD_BYTES..GUARD_BYTES + length].iter().enumerate() {
        let offset = start + index;
        let expected = if offset < FILE_BYTES {
            fixture_byte(offset)
        } else {
            0
        };
        assert_eq!(
            byte, expected,
            "GPU copy payload/EOF padding at byte {offset}"
        );
    }
}

fn run_mapping_import_case(
    ctx: &MetalContext,
    fixture: &ImmutableFixture,
    kind: MapKind,
    page: usize,
) {
    fixture.assert_identity();
    check_host_wired_memory().expect("ordinary wired-memory gate before each case");
    let span = kind.span(page);
    assert!(span <= MAX_SPAN_BYTES);
    assert!(span <= ctx.max_buffer_length());
    if !matches!(kind, MapKind::SharedAligned) {
        assert!(span > FILE_BYTES && span - FILE_BYTES < page);
        assert_eq!(span / page, FILE_BYTES / page + 1);
    }
    let _transaction = ctx.begin_allocation_transaction();
    admit_mapping_import(ctx);
    let (mmap_weak, source_weak, readback_weak, calls, mismatches, alive_at_release) =
        objc2::rc::autoreleasepool(|_| {
            let readback = ctx
                .buffer_uninit(READBACK_BYTES)
                .expect("bounded readback buffer");
            // SAFETY: exclusively owned, idle shared destination. Fault it in
            // before the import baseline, rather than during the first blit.
            unsafe {
                readback
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .write_bytes(0xa5, READBACK_BYTES)
            };
            let readback_weak = Weak::from_retained(&readback);
            mapping_import_snapshot(ctx, kind, span, 0, None, Some(&readback));
            // SAFETY: the private fixture has no writable descriptors or other
            // writers. The rounded EOF mapping ends in the final file page, not
            // a page wholly beyond EOF. It is never truncated while mapped.
            let mmap = Arc::new(unsafe {
                let mut options = MmapOptions::new();
                options.len(span);
                match kind {
                    MapKind::SharedAligned | MapKind::SharedEof => options
                        .map(&fixture.file)
                        .expect("read-only MAP_SHARED fixture"),
                    MapKind::OldPrivateEof => options
                        .map_copy(&fixture.file)
                        .expect("old MAP_PRIVATE fixture")
                        .make_read_only()
                        .expect("protect old private fixture"),
                }
            });
            assert_eq!(mmap.len(), span);
            let mmap_weak = Arc::downgrade(&mmap);
            let expected_pointer = mmap.as_ptr() as usize;
            let calls = Arc::new(AtomicUsize::new(0));
            let mismatches = Arc::new(AtomicUsize::new(0));
            let alive_at_release = Arc::new(AtomicUsize::new(0));
            let observer_calls = Arc::clone(&calls);
            let observer_mismatches = Arc::clone(&mismatches);
            let observer_alive = Arc::clone(&alive_at_release);
            let observer_mapping = mmap_weak.clone();
            let backing = ctx
                .gguf_no_copy_window_with_observer(
                    Arc::clone(&mmap),
                    0,
                    0,
                    span,
                    4,
                    move |pointer, length| {
                        if pointer.as_ptr() as usize != expected_pointer || length != span {
                            observer_mismatches.fetch_add(1, Ordering::Relaxed);
                        }
                        if observer_mapping.strong_count() > 0 {
                            observer_alive.fetch_add(1, Ordering::Relaxed);
                        }
                        observer_calls.fetch_add(1, Ordering::Release);
                    },
                )
                .expect("actual no-copy window import");
            assert_eq!(backing.mapped_len(), span);
            assert_eq!(backing.exposed_len(), span);
            assert_eq!(backing.mmap_offset(), 0);
            assert_eq!(backing.page_size(), page);
            let source_weak = Weak::from_retained(&backing.buffer);
            drop(mmap);
            assert!(mmap_weak.upgrade().is_some(), "import must retain mapping");
            assert_eq!(calls.load(Ordering::Acquire), 0);
            mapping_import_snapshot(ctx, kind, span, 1, Some(&backing.buffer), Some(&readback));
            blit_check_mapping_import(ctx, &backing.buffer, &readback, span);
            mapping_import_snapshot(ctx, kind, span, 2, Some(&backing.buffer), Some(&readback));
            assert!(
                mmap_weak.upgrade().is_some(),
                "mapping must survive GPU reads"
            );
            assert_eq!(calls.load(Ordering::Acquire), 0);
            compute_check_mapping_import(ctx, &backing.buffer, &readback, span);
            mapping_import_snapshot(ctx, kind, span, 3, Some(&backing.buffer), Some(&readback));
            assert!(
                mmap_weak.upgrade().is_some(),
                "mapping must survive GPU compute reads"
            );
            assert_eq!(calls.load(Ordering::Acquire), 0);
            drop(backing);
            drop(readback);
            (
                mmap_weak,
                source_weak,
                readback_weak,
                calls,
                mismatches,
                alive_at_release,
            )
        });
    objc2::rc::autoreleasepool(|_| mapping_import_snapshot(ctx, kind, span, 4, None, None));
    assert!(
        source_weak.load().is_none(),
        "source buffer survived pool drain"
    );
    assert!(
        readback_weak.load().is_none(),
        "readback buffer survived pool drain"
    );
    assert_eq!(
        calls.load(Ordering::Acquire),
        1,
        "exactly one deallocator call"
    );
    assert_eq!(
        mismatches.load(Ordering::Relaxed),
        0,
        "deallocator pointer/length"
    );
    assert_eq!(
        alive_at_release.load(Ordering::Relaxed),
        1,
        "mapping alive in deallocator"
    );
    assert!(
        mmap_weak.upgrade().is_none(),
        "mapping survived final buffer release"
    );
    fixture.assert_unchanged();
}

fn admit_mapping_import(ctx: &MetalContext) {
    let readback_upper = ctx
        .price_shared_buffer_upper(READBACK_BYTES as u64)
        .expect("price bounded shared readback");
    assert!(readback_upper.priced_upper_bytes <= MAX_READBACK_UPPER_BYTES as u64);
    // The ordinary product policy permits a reported zero process budget, but
    // never a missing signal or inadequate recommended working-set headroom.
    let admission = evaluate_metal_memory_admission(
        (PLANNED_LIVE_BYTES - RESERVE_BYTES) as u64,
        RESERVE_BYTES as u64,
        ctx.memory_signals(),
        true,
    );
    assert!(
        admission.is_admitted(),
        "bounded import admission: {admission:?}"
    );
}

#[test]
#[ignore = "release-only, self-leased Metal mmap blit/compute import witness; external watchdog required"]
fn metal_mmap_import_blit_witness_release() {
    assert!(
        !cfg!(debug_assertions),
        "run this bounded witness in release mode"
    );
    assert_eq!(std::env::var("MTL_DEBUG_LAYER").as_deref(), Ok("1"));
    assert_eq!(std::env::var("QWEN_METAL_LEASE_WAIT").as_deref(), Ok("1"));
    assert!(
        !crate::env_flag::read_default_off(METAL_PROCESS_LEASE_SKIP_WIRED_GATE_ENV),
        "no wired gate bypass"
    );
    let _lease = acquire_metal_benchmark_lease().expect("production lease and ordinary wired gate");
    let page = host_page_size().expect("host page size");
    assert!(page.is_power_of_two() && page <= MAX_PAGE_BYTES && page > 37);
    assert!(PAYLOAD_BYTES.is_multiple_of(page));
    objc2::rc::autoreleasepool(|_| {
        let ctx = MetalContext::new().expect("leased Metal context");
        {
            let _transaction = ctx.begin_allocation_transaction();
            admit_mapping_import(&ctx);
        }
        objc2::rc::autoreleasepool(|_| {
            ctx.pipeline("kernel_copy_offset_i32")
                .expect("warm raw-bit compute copy PSO before import baselines");
        });
        let fixture = ImmutableFixture::new();
        for kind in [
            MapKind::SharedAligned,
            MapKind::SharedEof,
            MapKind::OldPrivateEof,
        ] {
            run_mapping_import_case(&ctx, &fixture, kind, page);
        }
    });
}
