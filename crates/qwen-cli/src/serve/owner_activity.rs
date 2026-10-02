//! Admission lifetime accounting; only the model owner runs maintenance callbacks.

use std::sync::{Arc, Mutex};

#[derive(Default)]
struct State {
    active: usize,
    completed: usize,
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
}

impl Admission {
    pub(super) fn try_admit(&self) -> Option<ActivityGuard> {
        let mut state = self.0.lock().ok()?;
        if state.closed {
            return None;
        }
        state.active = state.active.checked_add(1)?;
        Some(ActivityGuard {
            admission: self.clone(),
            finished: false,
        })
    }

    pub(super) fn close(&self) {
        let mut state = self.0.lock().unwrap_or_else(|error| error.into_inner());
        state.closed = true;
    }
}

impl ActivityGuard {
    /// Only explicitly classified read-only diagnostics bypass completion.
    /// Ordinary HTTP handling, including errors and GET /v1/models, does not.
    pub(super) fn release_read_only(mut self) {
        self.finish(false);
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
        }
        self.finished = true;
    }
}

impl Drop for ActivityGuard {
    fn drop(&mut self) {
        self.finish(true);
    }
}

impl OwnerActivity {
    pub(super) fn admission(&self) -> Admission {
        self.admission.clone()
    }

    pub(super) fn drain_finished(&mut self, mut callback: impl FnMut()) {
        let completed = {
            let mut state = self
                .admission
                .0
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            std::mem::take(&mut state.completed)
        };
        for _ in 0..completed {
            callback();
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
