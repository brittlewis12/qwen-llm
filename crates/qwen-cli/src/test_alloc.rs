//! Test-only global allocator for peak-memory measurements of real serve
//! paths ([`measure`]). An allocation made by a thread inside `measure` is
//! tagged with the measurement's generation in a small header, so its free
//! (or reallocation) is counted on whichever thread performs it: trace
//! records are freed by the trace writer, for example. Sizes are charged
//! in 16-byte malloc quanta, as macOS hands them out. Untagged allocations
//! (every other test) pay one thread-local read and the header.
//!
//! The generation and the live byte count share one atomic word, so a
//! free tagged by an earlier measurement can never be subtracted from a
//! later one, however it interleaves with the reset.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

/// Header bytes before every allocation: the tag (the measurement
/// generation that owns it, or 0); keeps 16-byte payload alignment.
const HEADER: usize = 16;
/// Low bits of [`STATE`]: live bytes of the current generation.
const LIVE_BITS: u32 = 40;
const LIVE_MASK: u64 = (1 << LIVE_BITS) - 1;
/// Generations wrap within the high bits; 0 means untagged.
const GENERATION_MASK: u64 = (1 << (64 - LIVE_BITS)) - 1;

/// `generation << LIVE_BITS | live`.
static STATE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static MEASURING: Mutex<()> = Mutex::new(());

thread_local! {
    static PARTICIPANT: Cell<bool> = const { Cell::new(false) };
}

fn charged(size: usize) -> u64 {
    size.max(1).next_multiple_of(16) as u64
}

fn current_generation() -> u64 {
    STATE.load(Ordering::SeqCst) >> LIVE_BITS
}

/// Apply `delta` to the live count if `generation` is still current;
/// returns the new live count.
fn account(generation: u64, delta: i64) -> Option<u64> {
    let mut state = STATE.load(Ordering::SeqCst);
    loop {
        if state >> LIVE_BITS != generation {
            return None;
        }
        let live = ((state & LIVE_MASK) as i64 + delta).max(0) as u64 & LIVE_MASK;
        let next = (generation << LIVE_BITS) | live;
        match STATE.compare_exchange_weak(state, next, Ordering::SeqCst, Ordering::SeqCst) {
            Ok(_) => return Some(live),
            Err(observed) => state = observed,
        }
    }
}

fn offset(layout: Layout) -> usize {
    HEADER.max(layout.align())
}

fn outer(layout: Layout) -> Option<Layout> {
    Layout::from_size_align(layout.size() + offset(layout), layout.align().max(HEADER)).ok()
}

unsafe fn tag_of(payload: *mut u8) -> u64 {
    unsafe { (payload.sub(HEADER) as *const u64).read() }
}

pub(crate) struct Counting;

impl Counting {
    /// Allocate with an explicit tag (0: untagged).
    unsafe fn alloc_tagged(&self, layout: Layout, tag: u64) -> *mut u8 {
        let Some(outer) = outer(layout) else {
            return std::ptr::null_mut();
        };
        let base = unsafe { System.alloc(outer) };
        if base.is_null() {
            return base;
        }
        let payload = unsafe { base.add(offset(layout)) };
        unsafe { (payload.sub(HEADER) as *mut u64).write(tag) };
        if tag != 0
            && let Some(live) = account(tag, charged(layout.size()) as i64)
        {
            PEAK.fetch_max(live, Ordering::SeqCst);
        }
        payload
    }
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let participating = PARTICIPANT.try_with(Cell::get).unwrap_or(false);
        let tag = if participating {
            current_generation()
        } else {
            0
        };
        unsafe { self.alloc_tagged(layout, tag) }
    }

    unsafe fn dealloc(&self, payload: *mut u8, layout: Layout) {
        let tag = unsafe { tag_of(payload) };
        if tag != 0 {
            account(tag, -(charged(layout.size()) as i64));
        }
        let outer = outer(layout).expect("layout allocated before");
        unsafe { System.dealloc(payload.sub(offset(layout)), outer) };
    }

    unsafe fn realloc(&self, payload: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // Allocate, copy, free: both blocks are live at once, as in the
        // worst case of a moving realloc. The new block keeps the old
        // block's owner, whichever thread reallocates it.
        let Ok(new_layout) = Layout::from_size_align(new_size, layout.align()) else {
            return std::ptr::null_mut();
        };
        let tag = unsafe { tag_of(payload) };
        let moved = unsafe { self.alloc_tagged(new_layout, tag) };
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

/// Marks this thread a participant until dropped (also on unwind).
struct Participant;

impl Participant {
    fn enter() -> Self {
        PARTICIPANT.set(true);
        Self
    }
}

impl Drop for Participant {
    fn drop(&mut self) {
        PARTICIPANT.set(false);
    }
}

/// Run `f` on this thread and return its result with the peak of bytes
/// that this thread allocated while measuring and that were still live
/// (on any thread), relative to the start. Measurements are serialized.
pub(crate) fn measure<T>(f: impl FnOnce() -> T) -> (T, usize) {
    let _serial = MEASURING.lock().unwrap_or_else(|error| error.into_inner());
    // A new generation with zero live bytes, in one store: frees tagged by
    // earlier generations no longer match.
    let next = (current_generation() % GENERATION_MASK) + 1;
    STATE.store(next << LIVE_BITS, Ordering::SeqCst);
    PEAK.store(0, Ordering::SeqCst);
    let out = {
        let _participant = Participant::enter();
        f()
    };
    (out, PEAK.load(Ordering::SeqCst) as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block allocated while measuring and freed on another thread is
    /// subtracted there; reallocating it on another thread keeps it owned.
    #[test]
    fn ownership_follows_the_block_across_threads() {
        let ((), peak) = measure(|| {
            let mut block = vec![0u8; 1 << 20];
            std::thread::scope(|scope| {
                scope.spawn(|| {
                    block.resize(2 << 20, 0);
                });
            });
            // Still owned after the foreign realloc: freeing it here must
            // bring the live count back to (about) zero.
            drop(block);
            assert!(STATE.load(Ordering::SeqCst) & LIVE_MASK < 4096);
            let sent = vec![0u8; 1 << 20];
            std::thread::spawn(move || drop(sent)).join().unwrap();
            assert!(STATE.load(Ordering::SeqCst) & LIVE_MASK < 4096);
        });
        // The realloc held both blocks at once.
        assert!(peak >= (3 << 20), "{peak}");
    }

    /// A free tagged by an earlier measurement never reduces a later one,
    /// and a panic inside `measure` leaves the thread a non-participant.
    #[test]
    fn earlier_generations_and_panics_do_not_leak_into_later_measurements() {
        let (kept, _) = measure(|| vec![0u8; 1 << 20]);
        let ((), peak) = measure(|| {
            drop(kept);
            let _small = vec![0u8; 4096];
        });
        assert!(
            peak >= 4096,
            "an earlier generation's free was subtracted: {peak}"
        );
        let unwound = std::panic::catch_unwind(|| measure(|| panic!("inside measure")));
        assert!(unwound.is_err());
        assert!(!PARTICIPANT.get(), "participant flag survived a panic");
    }
}
