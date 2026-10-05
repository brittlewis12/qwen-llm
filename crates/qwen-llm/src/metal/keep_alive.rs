//! Idle residency keep-alive for no-copy weights.
//!
//! Metal wires the pages of a no-copy buffer while commands use it and, as
//! observed on macOS (not an API guarantee), unwires them about two seconds
//! after the GPU goes idle (PERF-LOG 2026-10-05, placement screen); the next
//! command then re-wires every page it touches (~1 s for GLM-5.3-Flash's
//! 109.5 GiB). A pulse is one ordinary command buffer that marks the buffers
//! used for reading and runs a single one-element dispatch, submitted inside
//! that window to keep the pages wired between requests.
//!
//! It is not a residency set: nothing is requested, committed or held beyond
//! the pulse's own command, so it adds no state that outlives the process's
//! in-flight commands. A pulse never waits; [`ResidencyKeepAlive::poll`]
//! surfaces a failed or stuck one.

use super::{Buffer, KernelEncoder, MetalContext, MetalError, MetalTensor, encode_fill_f32};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};
use std::time::{Duration, Instant};

/// A pulse still running after this long is reported as stuck.
pub const KEEP_ALIVE_STUCK_AFTER: Duration = Duration::from_secs(10);

pub struct ResidencyKeepAlive {
    marker: MetalTensor,
    in_flight: Option<(Retained<ProtocolObject<dyn MTLCommandBuffer>>, Instant)>,
    pulses: u64,
}

/// What [`ResidencyKeepAlive::pulse`] did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KeepAlivePulse {
    Submitted,
    /// The previous pulse has not completed; nothing was submitted.
    StillInFlight,
}

fn failure(status: &str, error: impl Into<String>) -> MetalError {
    MetalError::CommandBufferFailed {
        status: status.into(),
        error: error.into(),
    }
}

impl ResidencyKeepAlive {
    pub fn new(ctx: &MetalContext) -> Result<Self, MetalError> {
        Ok(Self {
            marker: MetalTensor::zeros_f32(ctx, vec![1])?,
            in_flight: None,
            pulses: 0,
        })
    }

    pub fn pulses(&self) -> u64 {
        self.pulses
    }

    /// Settle the previous pulse's state without submitting: `Ok(true)` when
    /// one is still running, an error when it failed or is stuck.
    pub fn poll(&mut self) -> Result<bool, MetalError> {
        let Some((command, submitted)) = &self.in_flight else {
            return Ok(false);
        };
        match command.status() {
            MTLCommandBufferStatus::Completed => {
                self.in_flight = None;
                Ok(false)
            }
            MTLCommandBufferStatus::Error => {
                let error = command
                    .error()
                    .map(|e| e.localizedDescription().to_string())
                    .unwrap_or_default();
                self.in_flight = None;
                Err(failure("keep-alive pulse error", error))
            }
            _ if submitted.elapsed() > KEEP_ALIVE_STUCK_AFTER => Err(failure(
                "keep-alive pulse stuck",
                format!("running for {:?}", submitted.elapsed()),
            )),
            _ => Ok(true),
        }
    }

    /// Wait (polling) up to `timeout` for the previous pulse to finish, so
    /// no pulse outlives the buffers it names.
    pub fn settle(&mut self, timeout: Duration) -> Result<(), MetalError> {
        let started = Instant::now();
        while self.poll()? {
            if started.elapsed() > timeout {
                return Err(failure(
                    "keep-alive pulse unsettled",
                    format!("still running after {timeout:?}"),
                ));
            }
            std::thread::sleep(Duration::from_millis(5));
        }
        Ok(())
    }

    /// Submit one pulse over `buffers` unless the previous one is still in
    /// flight; a previous pulse that failed or is stuck is reported instead.
    pub fn pulse(
        &mut self,
        ctx: &MetalContext,
        buffers: &[&Buffer],
    ) -> Result<KeepAlivePulse, MetalError> {
        if self.poll()? {
            return Ok(KeepAlivePulse::StillInFlight);
        }
        let command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| failure("keep-alive pulse", "no command buffer"))?;
        let enc = KernelEncoder::begin(&command);
        for buffer in buffers {
            enc.use_resource_read(buffer);
        }
        encode_fill_f32(ctx, &enc, &self.marker, 0.0)?;
        enc.end();
        command.commit();
        self.in_flight = Some((command, Instant::now()));
        self.pulses += 1;
        Ok(KeepAlivePulse::Submitted)
    }
}

/// `kern.memorystatus_vm_pressure_level` (1 normal, 2 warning, 4 critical),
/// or `None` when the host does not report it.
pub fn host_memory_pressure_level() -> Option<u32> {
    let mut level = 0u32;
    let mut size = std::mem::size_of::<u32>();
    // SAFETY: a fixed C string name and a correctly sized output buffer.
    let result = unsafe {
        libc::sysctlbyname(
            c"kern.memorystatus_vm_pressure_level".as_ptr(),
            (&mut level as *mut u32).cast(),
            &mut size,
            std::ptr::null_mut(),
            0,
        )
    };
    (result == 0 && size == std::mem::size_of::<u32>()).then_some(level)
}
