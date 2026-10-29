//! Test-fixture process-wide cooperative INT/TERM, never command cancellation.
//! Declare the guard before contexts/resources; restore only after their teardown.
use std::io;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Mutex, MutexGuard};

static OWNER: Mutex<()> = Mutex::new(());
static STOP: AtomicBool = AtomicBool::new(false);
static ACTIVE: AtomicBool = AtomicBool::new(false);
static RESTORE_ERROR: AtomicI32 = AtomicI32::new(0);

#[cfg(not(target_has_atomic = "8"))]
compile_error!("cooperative signal handler requires lock-free byte atomics");

extern "C" fn request_stop(_: libc::c_int) {
    STOP.store(true, Ordering::Relaxed);
}

pub(crate) fn flag() -> &'static AtomicBool {
    &STOP
}

pub(crate) fn check() -> io::Result<()> {
    if ACTIVE.load(Ordering::Relaxed) && STOP.load(Ordering::Relaxed) {
        Err(io::Error::new(
            io::ErrorKind::Interrupted,
            "cooperative INT/TERM stop requested; no new GPU work",
        ))
    } else {
        Ok(())
    }
}

fn blocked() -> io::Result<libc::sigset_t> {
    unsafe {
        let mut set = std::mem::zeroed();
        if libc::sigemptyset(&mut set) != 0
            || libc::sigaddset(&mut set, libc::SIGINT) != 0
            || libc::sigaddset(&mut set, libc::SIGTERM) != 0
        {
            return Err(io::Error::last_os_error());
        }
        let mut old = std::mem::zeroed();
        let code = libc::pthread_sigmask(libc::SIG_BLOCK, &set, &mut old);
        if code != 0 {
            return Err(io::Error::from_raw_os_error(code));
        }
        Ok(old)
    }
}

fn restore_mask(mask: &libc::sigset_t) -> io::Result<()> {
    let code = unsafe { libc::pthread_sigmask(libc::SIG_SETMASK, mask, std::ptr::null_mut()) };
    if code == 0 {
        Ok(())
    } else {
        Err(io::Error::from_raw_os_error(code))
    }
}

fn latch_error(error: &io::Error) {
    RESTORE_ERROR.store(error.raw_os_error().unwrap_or(libc::EIO), Ordering::Relaxed);
}

#[must_use = "hold until every submitted command is checked complete and contexts are torn down"]
pub(crate) struct CancellationGuard {
    _owner: MutexGuard<'static, ()>,
    old_int: libc::sigaction,
    old_term: libc::sigaction,
    int_installed: bool,
    term_installed: bool,
}

impl CancellationGuard {
    pub(crate) fn install() -> io::Result<Self> {
        let owner = OWNER
            .try_lock()
            .map_err(|_| io::Error::other("cooperative signal guard already owned or poisoned"))?;
        if RESTORE_ERROR.load(Ordering::Relaxed) != 0 {
            return Err(io::Error::other(
                "earlier signal restoration failed; new fixture refused",
            ));
        }
        let mask = blocked()?;
        let mut guard = Self {
            _owner: owner,
            old_int: unsafe { std::mem::zeroed() },
            old_term: unsafe { std::mem::zeroed() },
            int_installed: false,
            term_installed: false,
        };
        STOP.store(false, Ordering::Relaxed);
        let result = (|| -> io::Result<()> {
            let mut action: libc::sigaction = unsafe { std::mem::zeroed() };
            action.sa_sigaction = request_stop as *const () as usize;
            action.sa_flags = libc::SA_RESTART;
            unsafe {
                if libc::sigemptyset(&mut action.sa_mask) != 0
                    || libc::sigaddset(&mut action.sa_mask, libc::SIGINT) != 0
                    || libc::sigaddset(&mut action.sa_mask, libc::SIGTERM) != 0
                {
                    return Err(io::Error::last_os_error());
                }
                if libc::sigaction(libc::SIGINT, &action, &mut guard.old_int) != 0 {
                    return Err(io::Error::last_os_error());
                }
                guard.int_installed = true;
                ACTIVE.store(true, Ordering::Relaxed);
                if libc::sigaction(libc::SIGTERM, &action, &mut guard.old_term) != 0 {
                    return Err(io::Error::last_os_error());
                }
                guard.term_installed = true;
            }
            Ok(())
        })();
        // Roll back partial installation while this thread still masks signals.
        let rollback = if result.is_err() {
            guard.restore_actions()
        } else {
            Ok(())
        };
        let unmask = restore_mask(&mask).inspect_err(latch_error);
        rollback.inspect_err(latch_error)?;
        unmask?;
        result?;
        Ok(guard)
    }

    fn restore_actions(&mut self) -> io::Result<()> {
        let mut error = None;
        for (signal, old, installed) in [
            (libc::SIGTERM, &self.old_term, &mut self.term_installed),
            (libc::SIGINT, &self.old_int, &mut self.int_installed),
        ] {
            if *installed {
                if unsafe { libc::sigaction(signal, old, std::ptr::null_mut()) } != 0 {
                    error = Some(io::Error::last_os_error());
                } else {
                    *installed = false;
                }
            }
        }
        if !self.int_installed && !self.term_installed {
            ACTIVE.store(false, Ordering::Relaxed);
        }
        error.map_or(Ok(()), Err)
    }

    /// Only call once all checked waits and context teardown have finished.
    pub(crate) fn restore(&mut self) -> io::Result<()> {
        let mask = blocked().inspect_err(latch_error)?;
        let actions = self.restore_actions();
        let unmask = restore_mask(&mask);
        actions.and(unmask).inspect_err(latch_error)
    }
}

impl Drop for CancellationGuard {
    fn drop(&mut self) {
        if (self.int_installed || self.term_installed)
            && let Err(error) = self.restore()
        {
            // Never panic during unwind. A failed restore blocks future fixtures;
            // normal paths explicitly check restore() and publish its error.
            RESTORE_ERROR.store(error.raw_os_error().unwrap_or(libc::EIO), Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Command, Stdio};
    use std::time::{Duration, Instant};

    const CHILD: &str = "V41_TEST_COOPERATIVE_SIGNAL_CHILD";
    const SELECTOR: &str = "metal::test_cancellation::tests::signals_wait_for_release_cpu";

    fn actions() -> [libc::sigaction; 2] {
        [libc::SIGINT, libc::SIGTERM].map(|sig| unsafe {
            let mut old = std::mem::zeroed();
            assert_eq!(libc::sigaction(sig, std::ptr::null(), &mut old), 0);
            old
        })
    }

    fn worker() {
        let old = actions();
        let mut guard = CancellationGuard::install().unwrap();
        assert!(
            CancellationGuard::install().is_err(),
            "nested process ownership must refuse"
        );
        println!("cooperative_inflight_ready");
        std::io::stdout().flush().unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while check().is_ok() {
            assert!(Instant::now() < deadline, "CPU parent never signalled");
            std::thread::sleep(Duration::from_millis(1));
        }
        println!("cooperative_stop_while_inflight_alive");
        std::io::stdout().flush().unwrap();
        let mut release = String::new();
        std::io::stdin().read_line(&mut release).unwrap();
        assert_eq!(release, "release_checked_completion\n");
        assert_eq!(check().unwrap_err().kind(), io::ErrorKind::Interrupted);
        guard.restore().unwrap();
        for (before, after) in old.iter().zip(actions()) {
            assert_eq!(before.sa_sigaction, after.sa_sigaction);
            assert_eq!(before.sa_flags, after.sa_flags);
        }
        println!("cooperative_checked_release_then_restored");
    }

    #[test]
    fn signals_wait_for_release_cpu() {
        if std::env::var_os(CHILD).is_some() {
            worker();
            return;
        }
        for signal in [libc::SIGINT, libc::SIGTERM] {
            let mut child = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", SELECTOR, "--nocapture", "--test-threads=1"])
                .env(CHILD, "1")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap();
            let pid = libc::pid_t::try_from(child.id()).unwrap();
            let stdout = child.stdout.take().unwrap();
            let (tx, rx) = std::sync::mpsc::channel();
            let reader = std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    if tx.send(line.unwrap()).is_err() {
                        break;
                    }
                }
            });
            let wait_line = |needle: &str| -> io::Result<()> {
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    let line = rx
                        .recv_timeout(deadline.saturating_duration_since(Instant::now()))
                        .map_err(|e| io::Error::other(e.to_string()))?;
                    if line.contains(needle) {
                        return Ok(());
                    }
                }
            };
            let result = (|| -> io::Result<()> {
                wait_line("cooperative_inflight_ready")?;
                // Exact PID is the CPU-only child spawned immediately above.
                if unsafe { libc::kill(pid, signal) } != 0 {
                    return Err(io::Error::last_os_error());
                }
                wait_line("cooperative_stop_while_inflight_alive")?;
                assert!(
                    child.try_wait()?.is_none(),
                    "signal must not reap simulated inflight child"
                );
                child
                    .stdin
                    .as_mut()
                    .unwrap()
                    .write_all(b"release_checked_completion\n")?;
                wait_line("cooperative_checked_release_then_restored")?;
                Ok(())
            })();
            if result.is_err() {
                // This exact child is CPU-only, never a GPU fixture. Bound a
                // broken self-test without suggesting this as GPU safe drain.
                if child.try_wait().unwrap().is_none() {
                    child.kill().unwrap();
                }
            }
            let status = child.wait().unwrap();
            drop(rx);
            reader.join().unwrap();
            result.unwrap();
            assert!(status.success());
        }
    }

    #[test]
    fn unwind_restores_handlers_cpu() {
        if std::env::var_os(CHILD).is_none() {
            // Signal-disposition mutation is confined to the CPU subprocess.
            let output = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "metal::test_cancellation::tests::unwind_restores_handlers_cpu",
                    "--test-threads=1",
                ])
                .env(CHILD, "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stdout)
            );
            return;
        }
        let old = actions();
        let context_saw_guard = std::cell::Cell::new(false);
        struct SimulatedContext<'a>(&'a std::cell::Cell<bool>);
        impl Drop for SimulatedContext<'_> {
            fn drop(&mut self) {
                self.0.set(ACTIVE.load(Ordering::Relaxed));
            }
        }
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = CancellationGuard::install().unwrap();
            let _context = SimulatedContext(&context_saw_guard);
            panic!("simulated CPU unwind after checked completion/context teardown");
        }));
        assert!(result.is_err());
        assert!(
            context_saw_guard.get(),
            "context teardown precedes handler restore on unwind"
        );
        for (before, after) in old.iter().zip(actions()) {
            assert_eq!(before.sa_sigaction, after.sa_sigaction);
            assert_eq!(before.sa_flags, after.sa_flags);
        }
        // A poisoned ownership mutex refuses new fixtures after an unwind.
        assert!(CancellationGuard::install().is_err());
    }
}
