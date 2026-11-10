//! Host/GPU timeline capture for the one-ahead token loops.
//!
//! `GPUStartTime`/`GPUEndTime` are reported on the host monotonic clock in
//! seconds. `host_now_s` returns the same clock so a probe can place commits,
//! encodes and waits on the GPU's axis and separate queue latency from
//! execution.
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};
use std::sync::OnceLock;
use std::time::{Duration, Instant};

crate::env_flag!(
    /// Poll the token loop's command buffer instead of blocking on it. Keeps the
    /// thread runnable so it is scheduled promptly under host CPU load, where a
    /// blocked waiter can lose several milliseconds after the GPU has finished.
    /// Probe only; hot-spins one core for the wait.
    default_off spin_wait_requested, "QWEN_GPU_LOOP_SPIN"
);

crate::env_flag!(
    /// Poll with `yield_now` instead of a hot spin. Gentler on a loaded host but
    /// does not observe completion as promptly as `QWEN_GPU_LOOP_SPIN`.
    default_off yield_wait_requested, "QWEN_GPU_LOOP_YIELD"
);

#[derive(Clone, Copy, PartialEq, Eq)]
enum WaitMode {
    Block,
    Spin,
    Yield,
}

fn wait_mode() -> WaitMode {
    if yield_wait_requested() {
        WaitMode::Yield
    } else if spin_wait_requested() {
        WaitMode::Spin
    } else {
        WaitMode::Block
    }
}

pub fn spin_wait_enabled() -> bool {
    wait_mode() != WaitMode::Block
}

/// Poll until the command completes (or fails), bounded so a lost command
/// cannot spin forever. The caller still calls the ordinary wait to surface
/// errors.
pub fn spin_wait_until_completed(command: &Retained<ProtocolObject<dyn MTLCommandBuffer>>) {
    let mode = wait_mode();
    if mode == WaitMode::Block {
        return;
    }
    let started = Instant::now();
    loop {
        match command.status() {
            MTLCommandBufferStatus::Completed | MTLCommandBufferStatus::Error => return,
            _ if started.elapsed() > Duration::from_millis(250) => return,
            _ if mode == WaitMode::Yield => std::thread::yield_now(),
            _ => std::hint::spin_loop(),
        }
    }
}

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
