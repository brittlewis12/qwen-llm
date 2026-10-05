//! `--idle-residency-secs`: best-effort retention of a backend's no-copy
//! weights between requests through ordinary command buffers
//! (`metal::ResidencyKeepAlive`), so a request after a pause does not pay to
//! re-wire them. Family-neutral scheduling; each backend names its eligible
//! buffers (`retained_buffers()`). Every family whose serve weights are
//! no-copy GGUF windows defaults to a 60 s window ([`DEFAULT_WINDOW`]);
//! families whose weights are Metal-allocated copies are always wired and
//! refuse the flag.
//!
//! Window: closed until the first activity (a warm-up, or the first request
//! for backends without one, so a cold model is never faulted in by a
//! pulse); it opens again whenever a request finishes, whatever its outcome
//! (activity is activity); pulses stop once it lapses, so an idle server
//! returns its weights to pageable memory.
//! Pulsing is suspended while the host reports memory pressure (warning or
//! critical) and stops for the server's lifetime on any pulse failure;
//! requests are never affected.

use qwen_llm::metal::{
    Buffer, KeepAlivePulse, MetalContext, ResidencyKeepAlive, host_memory_pressure_level,
};
use std::time::{Duration, Instant};

pub(crate) const IDLE_RESIDENCY_ENV: &str = "QWEN_SERVE_IDLE_RESIDENCY_SECS";
/// The default for every no-copy family: a 1 GiB kill check of ordinary
/// command-buffer wiring recovered in every case (PERF-LOG 2026-10-05), and
/// 60 s covers an interactive turn without holding the host for minutes.
pub(crate) const DEFAULT_WINDOW: Duration = Duration::from_secs(60);
/// Well inside the ~2 s after which an idle GPU unwires no-copy weights.
const PULSE_INTERVAL: Duration = Duration::from_millis(500);
/// Bound on waiting for a final pulse at shutdown.
const SETTLE_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) struct IdleResidency {
    family: &'static str,
    window: Duration,
    keep_alive: Option<ResidencyKeepAlive>,
    /// `None` until the first activity: the window starts closed.
    last_activity: Option<Instant>,
    last_pulse: Option<Instant>,
    suspended_for_pressure: bool,
}

/// The configured window: the flag, else the environment, else the
/// family's default.
pub(crate) fn configured_window(flag: Option<u64>, family_default: Duration) -> Duration {
    flag.or_else(|| {
        std::env::var(IDLE_RESIDENCY_ENV)
            .ok()
            .and_then(|value| value.parse().ok())
    })
    .map_or(family_default, Duration::from_secs)
}

impl IdleResidency {
    pub(crate) fn new(family: &'static str, window: Duration) -> Self {
        if !window.is_zero() {
            tracing::info!(
                target: "qwen_diag",
                "serve idle residency: family={family} window_s={} pulse_ms={}",
                window.as_secs(),
                PULSE_INTERVAL.as_millis()
            );
        }
        Self {
            family,
            window,
            keep_alive: None,
            last_activity: None,
            last_pulse: None,
            suspended_for_pressure: false,
        }
    }

    fn disable(&mut self, why: &str) {
        tracing::warn!(
            "serve: {} idle residency disabled for this server: {why}",
            self.family
        );
        self.window = Duration::ZERO;
    }

    /// The backend warmed up or a request finished.
    pub(crate) fn note_activity(&mut self) {
        self.last_activity = Some(Instant::now());
    }

    /// Surface a failed or stuck pulse before a request uses the GPU.
    pub(crate) fn before_request(&mut self) {
        if let Some(keep_alive) = &mut self.keep_alive
            && let Err(error) = keep_alive.poll()
        {
            self.disable(&error.to_string());
        }
    }

    /// Called from the owner loop's idle tick.
    pub(crate) fn on_idle(&mut self, ctx: &MetalContext, buffers: &[&Buffer]) {
        if self.window.is_zero() {
            return;
        }
        if let Some(keep_alive) = &mut self.keep_alive
            && let Err(error) = keep_alive.poll()
        {
            return self.disable(&error.to_string());
        }
        if self
            .last_activity
            .is_none_or(|at| at.elapsed() > self.window)
            || self
                .last_pulse
                .is_some_and(|at| at.elapsed() < PULSE_INTERVAL)
        {
            return;
        }
        let pressured = host_memory_pressure_level().is_some_and(|level| level >= 2);
        if pressured != self.suspended_for_pressure {
            tracing::info!(
                target: "qwen_diag",
                "serve idle residency: family={} {}",
                self.family,
                if pressured { "suspended under host memory pressure" } else { "resumed" }
            );
            self.suspended_for_pressure = pressured;
        }
        if pressured {
            return;
        }
        if self.keep_alive.is_none() {
            match ResidencyKeepAlive::new(ctx) {
                Ok(keep_alive) => self.keep_alive = Some(keep_alive),
                Err(error) => return self.disable(&error.to_string()),
            }
        }
        match self
            .keep_alive
            .as_mut()
            .expect("created above")
            .pulse(ctx, buffers)
        {
            Ok(KeepAlivePulse::Submitted) => self.last_pulse = Some(Instant::now()),
            Ok(KeepAlivePulse::StillInFlight) => {}
            Err(error) => self.disable(&error.to_string()),
        }
    }

    /// Stop pulsing and wait (bounded) for the last pulse, so none outlives
    /// the buffers it names.
    pub(crate) fn shutdown(&mut self) {
        self.window = Duration::ZERO;
        if let Some(keep_alive) = &mut self.keep_alive
            && let Err(error) = keep_alive.settle(SETTLE_TIMEOUT)
        {
            tracing::warn!("serve: {} idle residency at shutdown: {error}", self.family);
        }
    }

    #[cfg(test)]
    pub(crate) fn window(&self) -> Duration {
        self.window
    }

    #[cfg(test)]
    pub(crate) fn is_open(&self) -> bool {
        !self.window.is_zero()
            && self
                .last_activity
                .is_some_and(|at| at.elapsed() <= self.window)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_flag_wins_then_the_environment_then_the_family_default() {
        assert_eq!(
            configured_window(Some(90), DEFAULT_WINDOW),
            Duration::from_secs(90)
        );
        assert_eq!(configured_window(Some(0), DEFAULT_WINDOW), Duration::ZERO);
        if std::env::var_os(IDLE_RESIDENCY_ENV).is_none() {
            assert_eq!(configured_window(None, DEFAULT_WINDOW), DEFAULT_WINDOW);
        }
        let residency = IdleResidency::new("test", Duration::ZERO);
        assert!(residency.window().is_zero());
    }

    #[test]
    fn the_window_stays_closed_until_the_first_activity() {
        let mut residency = IdleResidency::new("test", DEFAULT_WINDOW);
        assert!(!residency.is_open(), "a cold model is never pulsed");
        residency.note_activity();
        assert!(residency.is_open());
        let mut off = IdleResidency::new("test", Duration::ZERO);
        off.note_activity();
        assert!(!off.is_open());
    }
}
