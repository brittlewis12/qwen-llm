//! Output-proportional CPU memory admission in serve (map #14; cx 01a10cc
//! jam 2026-10-08). A buffer that grows with a response's output admits
//! each growth step before allocating it: the step's whole simultaneous
//! peak (new allocation plus the old one while it is copied), less what the
//! buffer already holds (already reflected in the process headroom), must
//! fit the process headroom, with Batch 3a's typed refusals (503
//! `memory_admission_denied` for pressure, 500 for unreadable telemetry). A
//! successful check is not a reservation; every step checks again.
//!
//! A refusal raised inside a [`super::http::GenerationSink`] travels through
//! its `io::Result` as an [`OutputRefusal`], so the request handler answers
//! with the typed error instead of treating it as a client disconnect.

use super::items::ServeError;
use std::io;

/// The smallest growth step of an admitted output buffer.
pub(crate) const OUTPUT_STEP_BYTES: usize = 64 << 10;

/// A typed refusal carried through a sink's `io::Error`.
#[derive(Debug)]
pub(crate) struct OutputRefusal(pub(crate) ServeError);

impl std::fmt::Display for OutputRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "output memory refused: {}", self.0.message)
    }
}

impl std::error::Error for OutputRefusal {}

/// Wrap a typed refusal for a sink's `io::Result`.
pub(crate) fn refusal_error(error: ServeError) -> io::Error {
    io::Error::other(OutputRefusal(error))
}

/// The typed refusal an `io::Error` carries, if any.
pub(crate) fn refusal_in(error: &io::Error) -> Option<&ServeError> {
    error
        .get_ref()?
        .downcast_ref::<OutputRefusal>()
        .map(|refusal| &refusal.0)
}

/// Admit one growth step whose whole simultaneous peak is `peak` bytes while
/// `held` bytes are already allocated.
pub(crate) fn admit_growth(peak: u64, held: u64, headroom: Option<u64>) -> Result<(), ServeError> {
    super::transport_memory::admit_resident_transport(peak.saturating_sub(held), headroom)
}

/// Capacity of a buffer of `capacity` elements grown to hold `needed`: at
/// least double (bounded copying), at least `step`, never below `needed`;
/// `None` if doubling overflows.
pub(crate) fn grown_capacity(capacity: usize, needed: usize, step: usize) -> Option<usize> {
    Some(needed.max(capacity.checked_mul(2)?).max(step))
}

/// An output size whose arithmetic overflows: Batch 3a's typed 500
/// `memory_size_overflow` (not pressure, not a client abort).
pub(crate) fn size_overflow(what: &str) -> ServeError {
    let (status, error_type, code) = super::transport_memory::refusal_kind(
        qwen_llm::metal::MetalMemoryAdmissionReason::RequiredBytesOverflow,
    );
    ServeError {
        status,
        error_type,
        code: Some(code),
        param: None,
        message: format!("{what}: output size arithmetic overflowed"),
    }
}

/// An admitted allocation that still failed: a server error (no pressure
/// decision was made). Unlike an admission refusal, it may leave a buffer's
/// capacity partially changed.
pub(crate) fn allocation_failure(what: &str, error: impl std::fmt::Display) -> ServeError {
    ServeError::server_error(format!("{what}: admitted allocation failed: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refusals_round_trip_through_io_errors() {
        let error =
            super::super::transport_memory::admit_resident_transport(10, Some(5)).unwrap_err();
        let carried = refusal_error(error.clone());
        assert_eq!(refusal_in(&carried), Some(&error));
        assert!(refusal_in(&io::Error::new(io::ErrorKind::BrokenPipe, "gone")).is_none());
        assert!(refusal_in(&io::Error::other("plain")).is_none());
    }

    #[test]
    fn growth_prices_the_outstanding_peak_and_doubles() {
        // 10 held, peak 30: 20 outstanding.
        assert!(admit_growth(30, 10, Some(20)).is_ok());
        let refused = admit_growth(30, 10, Some(19)).unwrap_err();
        assert_eq!(
            (refused.status, refused.code),
            (503, Some("memory_admission_denied"))
        );
        let unreadable = admit_growth(30, 10, None).unwrap_err();
        assert_eq!(
            (unreadable.status, unreadable.code),
            (500, Some("memory_signal_unavailable"))
        );
        // A reported zero is an omitted budget (Batch 3a convention).
        assert!(admit_growth(30, 10, Some(0)).is_ok());
        assert_eq!(grown_capacity(0, 1, 64), Some(64));
        assert_eq!(grown_capacity(64, 65, 64), Some(128));
        assert_eq!(grown_capacity(64, 1000, 64), Some(1000));
        assert_eq!(grown_capacity(usize::MAX / 2 + 1, 1, 64), None);
        let overflow = size_overflow("collection");
        assert_eq!(
            (overflow.status, overflow.code),
            (500, Some("memory_size_overflow"))
        );
        assert_eq!(allocation_failure("collection", "oom").status, 500);
    }
}
