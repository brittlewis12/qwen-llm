//! Checked command-buffer completion.
//!
//! `waitUntilCompleted` also returns for command buffers the GPU did not run
//! to completion: faults, timeouts, and command buffers discarded as innocent
//! victims of a GPU recovery (`kIOGPUCommandBufferCallbackErrorInnocentVictim`)
//! triggered by any client on the machine. Their outputs are partial or never
//! written. Every host-side wait must therefore check the final status before
//! reading results; use [`wait_completed`] / [`commit_and_wait`] instead of a
//! bare `waitUntilCompleted()`.

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

use super::MetalError;

/// Block until `cmd` finishes, then fail unless it completed without error.
/// Does not commit; pair with an earlier `commit()`.
pub fn wait_completed(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    #[allow(clippy::disallowed_methods)]
    cmd.waitUntilCompleted();
    command_buffer_completed(cmd)
}

/// Block until `cmd` finishes WITHOUT checking how it finished. Only for a
/// caller that inspects `status()` and `error()` itself right after (to
/// poison a session or report a family-specific error) or reads nothing the
/// command wrote; everything else uses [`wait_completed`]. The bare
/// `waitUntilCompleted` is a disallowed method (clippy.toml), so every
/// unchecked wait is spelled here, where it can be found.
pub fn wait_unchecked(cmd: &ProtocolObject<dyn MTLCommandBuffer>) {
    #[allow(clippy::disallowed_methods)]
    cmd.waitUntilCompleted();
}

/// `commit()` then [`wait_completed`].
pub fn commit_and_wait(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    cmd.commit();
    wait_completed(cmd)
}

#[cfg(test)]
thread_local! {
    static FAIL_NEXT_CHECK: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Test seam: the next completion check on this thread reports a discarded
/// command buffer, so failure paths can be exercised without a GPU recovery.
/// Disarmed when the guard drops, so an early return cannot leak it into a
/// later healthy check.
#[cfg(test)]
#[must_use = "the injection is disarmed when the guard drops"]
pub(crate) fn fail_next_completion_check() -> FailNextCompletionCheck {
    FAIL_NEXT_CHECK.with(|flag| flag.set(true));
    FailNextCompletionCheck
}

#[cfg(test)]
pub(crate) struct FailNextCompletionCheck;

#[cfg(test)]
impl Drop for FailNextCompletionCheck {
    fn drop(&mut self) {
        FAIL_NEXT_CHECK.with(|flag| flag.set(false));
    }
}

/// Check an already-finished command buffer: `Ok` only for status `Completed`
/// with no attached `NSError`.
pub fn command_buffer_completed(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), MetalError> {
    #[cfg(test)]
    if FAIL_NEXT_CHECK.with(|flag| flag.replace(false)) {
        return Err(MetalError::CommandBufferFailed {
            status: "Error".to_string(),
            error: "injected discarded command buffer".to_string(),
        });
    }
    let status = cmd.status();
    let error = cmd.error();
    if status == MTLCommandBufferStatus::Completed && error.is_none() {
        return Ok(());
    }
    Err(MetalError::CommandBufferFailed {
        status: status_name(status).to_string(),
        error: match error {
            Some(error) => format!(
                "{} code={}: {}",
                error.domain(),
                error.code(),
                error.localizedDescription()
            ),
            None => "none".to_string(),
        },
    })
}

fn status_name(status: MTLCommandBufferStatus) -> &'static str {
    match status {
        MTLCommandBufferStatus::NotEnqueued => "NotEnqueued",
        MTLCommandBufferStatus::Enqueued => "Enqueued",
        MTLCommandBufferStatus::Committed => "Committed",
        MTLCommandBufferStatus::Scheduled => "Scheduled",
        MTLCommandBufferStatus::Completed => "Completed",
        MTLCommandBufferStatus::Error => "Error",
        _ => "Unknown",
    }
}
