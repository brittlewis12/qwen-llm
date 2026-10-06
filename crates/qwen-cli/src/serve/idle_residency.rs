//! `--idle-residency-secs`: best-effort retention of a backend's no-copy
//! weights between requests through ordinary command buffers
//! (`metal::ResidencyKeepAlive`), so a request after a pause does not pay to
//! re-wire them. Family-neutral scheduling; each backend names its eligible
//! buffers (`retained_buffers()`). Every family whose serve backend names its
//! no-copy GGUF windows defaults to a 60 s window ([`DEFAULT_WINDOW`]); Qwen's
//! backend does not yet and refuses the flag.
//!
//! Window: closed until the first activity (a warm-up, or the first request
//! that submitted GPU compute work, so a cold model is never faulted in by a
//! pulse). It opens again when such a request finishes, whatever its outcome
//! (a failure or abort after submission still used the weights). A request
//! counts by what it actually submitted: [`IdleResidency::before_request`]
//! snapshots the process's compute-encoder count and
//! [`IdleResidency::request_finished`] compares it, so model lists, refusals,
//! admission or allocation failures and cancellations before the first
//! command never open or renew it. Only compute encoders count (blit-only
//! work does not). A snapshot still unmatched when the owner goes idle is
//! dropped before any pulse, so the keep-alive's own work can never renew
//! the window. Each finish is logged at debug on `qwen_diag`
//! (`RUST_LOG=info,qwen_diag=debug`). Pulses stop once it lapses, so an idle
//! server returns its weights to pageable memory.
//! Pulsing stops for the server's lifetime on any pulse failure; requests are
//! never affected.
//!
//! Pressure: a pulse is skipped while the host reports memory pressure of
//! warning or worse; pulsing resumes on the next idle tick after it clears,
//! within the original window (no cooldown), and a pulse already submitted
//! is not cancelled. A host that cannot report pressure is treated as
//! unpressured; that is logged once. This is suspension, not protection
//! against re-pinning under oscillating pressure.

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
    /// Compute encoders begun before the current request.
    encoders_at_request: Option<u64>,
    /// The process's compute-encoder count (injectable for tests).
    encoders: fn() -> u64,
    /// The unavailable-pressure-reading notice was logged.
    pressure_unknown_logged: bool,
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
            encoders_at_request: None,
            encoders: qwen_llm::metal::compute_encoders_begun,
            pressure_unknown_logged: false,
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

    /// The backend warmed up (ran the model outside a request).
    pub(crate) fn note_activity(&mut self) {
        self.last_activity = Some(Instant::now());
    }

    /// A request is about to run: surface a failed or stuck pulse, and
    /// snapshot the compute-encoder count its submissions are measured from.
    pub(crate) fn before_request(&mut self) {
        self.encoders_at_request = Some((self.encoders)());
        if let Some(keep_alive) = &mut self.keep_alive
            && let Err(error) = keep_alive.poll()
        {
            self.disable(&error.to_string());
        }
    }

    /// A connection finished. Only one whose request submitted GPU compute
    /// work (encoders begun since [`IdleResidency::before_request`]) opens or
    /// renews the window.
    pub(crate) fn request_finished(&mut self) {
        let submitted = self
            .encoders_at_request
            .take()
            .map(|start| (self.encoders)().saturating_sub(start));
        let renews = submitted.is_some_and(|count| count > 0);
        if renews {
            self.note_activity();
        }
        if !self.window.is_zero() {
            tracing::debug!(
                target: "qwen_diag",
                "serve idle residency: family={} finished compute_encoders={} window={}",
                self.family,
                submitted.map_or_else(|| "none".to_owned(), |count| count.to_string()),
                if renews { "renewed" } else { "unchanged" }
            );
        }
    }

    /// The owner is idle, so no request is in flight: a snapshot without a
    /// matching finish is stale and must not let a pulse's encoders count.
    fn forget_unfinished_request(&mut self) {
        self.encoders_at_request = None;
    }

    /// Called from the owner loop's idle tick.
    /// `buffers` is called only when a pulse is due, so a closed or
    /// recently pulsed window costs no enumeration.
    pub(crate) fn on_idle<'b>(
        &mut self,
        ctx: &MetalContext,
        buffers: impl FnOnce() -> Vec<&'b Buffer>,
    ) {
        self.forget_unfinished_request();
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
        let level = host_memory_pressure_level();
        if level.is_none() && !self.pressure_unknown_logged {
            self.pressure_unknown_logged = true;
            tracing::info!(
                target: "qwen_diag",
                "serve idle residency: family={} host memory pressure unavailable; pulsing without suspension",
                self.family
            );
        }
        let pressured = level.is_some_and(|level| level >= 2);
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
            .pulse(ctx, &buffers())
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

    static FAKE_ENCODERS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    fn fake_encoders() -> u64 {
        FAKE_ENCODERS.load(std::sync::atomic::Ordering::SeqCst)
    }

    fn submit() {
        FAKE_ENCODERS.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }

    /// The window follows submitted GPU work, not request lifecycle: a
    /// request that submitted nothing (model list, refusal, admission
    /// failure, cancellation before the first command) neither opens nor
    /// renews it; one that submitted anything (even if it failed afterwards)
    /// does; a finish without a request start counts nothing.
    #[test]
    fn only_requests_that_submitted_gpu_work_open_or_renew_the_window() {
        let mut residency = IdleResidency::new("test", DEFAULT_WINDOW);
        residency.encoders = fake_encoders;
        residency.request_finished();
        assert!(!residency.is_open(), "a finish without a request");
        residency.before_request();
        residency.request_finished();
        assert!(!residency.is_open(), "nothing submitted opened the window");
        residency.before_request();
        submit();
        residency.request_finished();
        assert!(residency.is_open());
        let opened = residency.last_activity;
        std::thread::sleep(Duration::from_millis(5));
        residency.before_request();
        residency.request_finished();
        assert_eq!(
            residency.last_activity, opened,
            "no submission renewed the window"
        );
        residency.before_request();
        submit();
        residency.request_finished();
        assert!(residency.last_activity > opened);
    }

    /// A request start without a finish (a path that never settles through
    /// the owner) followed by an idle pulse: the pulse's encoders must not
    /// renew the window on a later finish.
    #[test]
    fn an_idle_pulse_never_counts_as_request_work() {
        let mut residency = IdleResidency::new("test", DEFAULT_WINDOW);
        residency.encoders = fake_encoders;
        residency.before_request();
        residency.forget_unfinished_request();
        submit();
        residency.request_finished();
        assert!(!residency.is_open(), "a pulse opened the window");
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
