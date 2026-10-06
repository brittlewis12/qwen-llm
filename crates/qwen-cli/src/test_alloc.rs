//! Test-only global allocator for peak-memory measurements of real serve
//! paths ([`measure`]). An allocation made by a thread inside `measure` is
//! tagged in a small header, so its free is counted on whichever thread
//! performs it (trace records are freed by the trace writer, for example).
//! Sizes are charged in 16-byte malloc quanta, as macOS hands them out.
//! Untagged allocations (every other test) pay one thread-local read and
//! the header.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicIsize, AtomicUsize, Ordering};

/// Header bytes before every allocation: the tag (the measurement
/// generation that allocated it, or 0); keeps 16-byte payload alignment.
const HEADER: usize = 16;

/// The current measurement; frees of allocations tagged by an earlier one
/// do not count against it.
static GENERATION: AtomicUsize = AtomicUsize::new(0);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);
static MEASURING: Mutex<()> = Mutex::new(());

thread_local! {
    static PARTICIPANT: Cell<bool> = const { Cell::new(false) };
}

fn charged(size: usize) -> isize {
    size.max(1).next_multiple_of(16) as isize
}

fn participating() -> bool {
    PARTICIPANT.try_with(Cell::get).unwrap_or(false)
}

fn offset(layout: Layout) -> usize {
    HEADER.max(layout.align())
}

pub(crate) struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let offset = offset(layout);
        let Ok(outer) = Layout::from_size_align(layout.size() + offset, layout.align().max(HEADER))
        else {
            return std::ptr::null_mut();
        };
        let base = unsafe { System.alloc(outer) };
        if base.is_null() {
            return base;
        }
        let payload = unsafe { base.add(offset) };
        let tag = if participating() {
            GENERATION.load(Ordering::SeqCst)
        } else {
            0
        };
        unsafe { (payload.sub(HEADER) as *mut usize).write(tag) };
        if tag != 0 {
            let live =
                LIVE.fetch_add(charged(layout.size()), Ordering::SeqCst) + charged(layout.size());
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        payload
    }

    unsafe fn dealloc(&self, payload: *mut u8, layout: Layout) {
        let offset = offset(layout);
        let tag = unsafe { (payload.sub(HEADER) as *const usize).read() };
        if tag != 0 && tag == GENERATION.load(Ordering::SeqCst) {
            LIVE.fetch_sub(charged(layout.size()), Ordering::SeqCst);
        }
        let outer =
            Layout::from_size_align(layout.size() + offset, layout.align().max(HEADER)).unwrap();
        unsafe { System.dealloc(payload.sub(offset), outer) };
    }

    unsafe fn realloc(&self, payload: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Allocate, copy, free: both blocks are live at once, as in the
        // worst case of a moving realloc.
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else {
            return std::ptr::null_mut();
        };
        let moved = unsafe { self.alloc(new_layout) };
        if !moved.is_null() {
            unsafe {
                std::ptr::copy_nonoverlapping(payload, moved, layout.size().min(new_size));
                self.dealloc(payload, layout);
            }
        }
        moved
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Run `f` on this thread and return its result with the peak of bytes
/// that this thread allocated while measuring and that were still live
/// (on any thread), relative to the start. Measurements are serialized.
pub(crate) fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let _serial = MEASURING.lock().unwrap_or_else(|error| error.into_inner());
    GENERATION.fetch_add(1, Ordering::SeqCst);
    LIVE.store(0, Ordering::SeqCst);
    PEAK.store(0, Ordering::SeqCst);
    PARTICIPANT.set(true);
    let out = f();
    PARTICIPANT.set(false);
    (out, PEAK.load(Ordering::SeqCst).max(0) as usize)
}
