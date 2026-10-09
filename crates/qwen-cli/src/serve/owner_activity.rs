//! Admission lifetime accounting; only the model owner runs maintenance callbacks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State {
    active: usize,
    completed: usize,
    /// Of `completed`, how many ended in a server-side failure.
    failed: usize,
    closed: bool,
}

#[derive(Clone, Default)]
pub(super) struct Admission(Arc<Mutex<State>>);

#[derive(Default)]
pub(super) struct OwnerActivity {
    admission: Admission,
}

pub(super) struct ActivityGuard {
    admission: Admission,
    finished: bool,
    completes: bool,
    /// The connection answered a server-side failure (5xx) after
    /// generation, which the owner did not see (a partition failure).
    server_failed: AtomicBool,
}

impl Admission {
    pub(super) fn try_admit(&self) -> Option<ActivityGuard> {
        self.try_admit_kind(true)
    }

    pub(super) fn try_prepare(&self) -> Option<ActivityGuard> {
        self.try_admit_kind(false)
    }

    fn try_admit_kind(&self, completes: bool) -> Option<ActivityGuard> {
        let mut state = self.0.lock().ok()?;
        if state.closed {
            return None;
        }
        state.active = state.active.checked_add(1)?;
        Some(ActivityGuard {
            admission: self.clone(),
            finished: false,
            completes,
            server_failed: AtomicBool::new(false),
        })
    }

    pub(super) fn close(&self) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.closed = true;
    }
}

impl ActivityGuard {
    pub(super) fn mark_work(&mut self) {
        self.completes = true;
    }
    /// Report a server-side failure to the owner with this completion.
    pub(super) fn mark_server_failure(&self) {
        self.server_failed.store(true, Ordering::Release);
    }
    fn finish(&mut self, completed: bool) {
        if self.finished {
            return;
        }
        // Drop must also work during unwinding. A poisoned mutex permanently
        // refuses admission and idle, but outstanding completions remain drainable.
        let mut state = self
            .admission
            .0
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.active -= 1;
        if completed {
            state.completed += 1;
            if self.server_failed.load(Ordering::Acquire) {
                state.failed += 1;
            }
        }
        self.finished = true;
    }
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        self.finish(self.completes);
    }
}

impl OwnerActivity {
    pub(super) fn admission(&self) -> Admission {
        self.admission.clone()
    }

    #[cfg(test)]
    pub(super) fn drain_finished(&mut self, mut callback: impl FnMut()) {
        self.drain_finished_with_failures(|_| callback());
    }

    /// [`Self::drain_finished`], telling the callback whether each
    /// completion ended in a server-side failure the owner did not see.
    /// Counts are aggregate, so within one batch the failures come first
    /// (not in completion order): any failure in a batch is conservative
    /// for the whole batch (idle residency closes and the batch's
    /// successes cannot reopen it).
    pub(super) fn drain_finished_with_failures(&mut self, mut callback: impl FnMut(bool)) {
        let (completed, failed) = {
            let mut state = self
                .admission
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            (
                std::mem::take(&mut state.completed),
                std::mem::take(&mut state.failed),
            )
        };
        for index in 0..completed {
            callback(index < failed);
        }
    }

    /// Serializes maintenance with admission. The callback must not access this
    /// accounting or wait on workers that do; admissions wait for it to return.
    pub(super) fn idle_if_quiet(&mut self, callback: impl FnOnce()) {
        let Ok(state) = self.admission.0.lock() else {
            return;
        };
        if !state.closed && state.active == 0 && state.completed == 0 {
            callback();
        }
    }

    pub(super) fn is_settled(&self) -> bool {
        self.admission
            .0
            .lock()
            .is_ok_and(|state| state.active == 0 && state.completed == 0)
    }
}

#[cfg(test)]
mod tests;
