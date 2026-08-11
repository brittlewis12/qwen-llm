use anyhow::{Result, anyhow};
use std::process::ExitCode;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicI32, Ordering};

static REQUESTED_SIGNAL: AtomicI32 = AtomicI32::new(0);
static INSTALLED: OnceLock<Result<(), i32>> = OnceLock::new();

extern "C" fn handle_signal(signal: libc::c_int) {
    REQUESTED_SIGNAL
        .compare_exchange(0, signal, Ordering::SeqCst, Ordering::SeqCst)
        .ok();
}

fn install_signal(signal: libc::c_int) -> Result<(), i32> {
    // SAFETY: sigaction is initialized before use, the handler has C ABI and
    // static lifetime, and the mask pointer remains valid for the call.
    unsafe {
        let mut action = std::mem::MaybeUninit::<libc::sigaction>::zeroed().assume_init();
        action.sa_sigaction = handle_signal as *const () as libc::sighandler_t;
        action.sa_flags = 0;
        if libc::sigemptyset(&mut action.sa_mask) != 0 {
            return Err(*libc::__error());
        }
        if libc::sigaction(signal, &action, std::ptr::null_mut()) != 0 {
            return Err(*libc::__error());
        }
    }
    Ok(())
}

pub fn install() -> Result<()> {
    let installed = INSTALLED.get_or_init(|| {
        install_signal(libc::SIGINT)?;
        install_signal(libc::SIGTERM)
    });
    installed
        .as_ref()
        .map(|_| ())
        .map_err(|errno| anyhow!("install graceful termination handlers: errno {errno}"))
}

pub fn checkpoint() -> Result<()> {
    let signal = REQUESTED_SIGNAL.load(Ordering::SeqCst);
    if signal == 0 {
        return Ok(());
    }
    Err(anyhow!(
        "termination signal {signal} received; unwinding for Metal teardown"
    ))
}

fn signal_exit_code(signal: libc::c_int) -> ExitCode {
    ExitCode::from((128 + signal).clamp(1, u8::MAX as libc::c_int) as u8)
}

pub fn finish(result: Result<()>) -> ExitCode {
    let signal = REQUESTED_SIGNAL.load(Ordering::SeqCst);
    match result {
        Ok(()) if signal == 0 => ExitCode::SUCCESS,
        Ok(()) => {
            eprintln!("qwen: graceful teardown completed after termination signal {signal}");
            signal_exit_code(signal)
        }
        Err(error) => {
            eprintln!("Error: {error:?}");
            if signal == 0 {
                ExitCode::FAILURE
            } else {
                signal_exit_code(signal)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_is_clear_before_any_signal() {
        assert!(checkpoint().is_ok());
    }

    #[test]
    fn signal_exit_codes_follow_shell_convention() {
        assert_eq!(signal_exit_code(libc::SIGINT), ExitCode::from(130));
        assert_eq!(signal_exit_code(libc::SIGTERM), ExitCode::from(143));
    }
}
