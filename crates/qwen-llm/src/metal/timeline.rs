//! Host/GPU timeline capture for the one-ahead token loops.
//!
//! `GPUStartTime`/`GPUEndTime` are reported on the host monotonic clock in
//! seconds. `host_now_s` returns the same clock so a probe can place commits,
//! encodes and waits on the GPU's axis and separate queue latency from
//! execution.
use std::sync::OnceLock;

/// Absolute seconds on the clock Metal reports for `GPUStartTime`.
pub fn host_now_s() -> f64 {
    static SCALE: OnceLock<f64> = OnceLock::new();
    let scale = *SCALE.get_or_init(|| {
        let mut info = libc::mach_timebase_info { numer: 0, denom: 0 };
        // SAFETY: `info` is a valid out-pointer.
        unsafe { libc::mach_timebase_info(&mut info) };
        f64::from(info.numer) / f64::from(info.denom) / 1e9
    });
    // SAFETY: no preconditions.
    unsafe { libc::mach_absolute_time() as f64 * scale }
}

/// One command buffer in a token-loop burst, on the host monotonic clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct TokenFrameTimeline {
    /// Burst ordinal. Frame 0 is the sampler-only seed frame when present.
    pub frame: usize,
    /// Whether this command ran a model forward (the seed frame does not).
    pub forward: bool,
    /// Host encode span for this command.
    pub encode_start_s: f64,
    pub encode_end_s: f64,
    /// Host time the command was committed.
    pub commit_s: f64,
    /// Host span of the blocking wait for this command.
    pub wait_start_s: f64,
    pub wait_end_s: f64,
    /// GPU execution span reported by Metal.
    pub gpu_start_s: f64,
    pub gpu_end_s: f64,
}

impl TokenFrameTimeline {
    pub fn encode_ms(&self) -> f64 {
        (self.encode_end_s - self.encode_start_s) * 1e3
    }
    pub fn commit_to_gpu_start_ms(&self) -> f64 {
        (self.gpu_start_s - self.commit_s) * 1e3
    }
    pub fn gpu_ms(&self) -> f64 {
        (self.gpu_end_s - self.gpu_start_s) * 1e3
    }
    pub fn gpu_end_to_wait_return_ms(&self) -> f64 {
        (self.wait_end_s - self.gpu_end_s) * 1e3
    }
    pub fn wait_ms(&self) -> f64 {
        (self.wait_end_s - self.wait_start_s) * 1e3
    }
}
