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
    cmd.waitUntilCompleted();
    command_buffer_completed(cmd)
}

/// `commit()` then [`wait_completed`].
pub fn commit_and_wait(cmd: &ProtocolObject<dyn MTLCommandBuffer>) -> Result<(), MetalError> {
    cmd.commit();
    wait_completed(cmd)
}

/// Check an already-finished command buffer: `Ok` only for status `Completed`
/// with no attached `NSError`.
pub fn command_buffer_completed(
    cmd: &ProtocolObject<dyn MTLCommandBuffer>,
) -> Result<(), MetalError> {
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
