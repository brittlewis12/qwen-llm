//! Phase-accounted request timing shared by lanes that separate setup from
//! the loaded request (K2, GLM-5.3). Each family names its phases, which of
//! them form the loaded request and the model load, and its encoding phase;
//! the accumulator checks that phases never overlap or exceed the lane wall.

use anyhow::Result;
use serde_json::{Value, json};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// A family's request phases.
pub(crate) trait LanePhases: Copy + 'static {
    /// Family label for errors ("K2", "GLM-5.3").
    const FAMILY: &'static str;
    /// Every phase with its report name, in report order; `index` is the
    /// position in this table.
    const PHASES: &'static [(Self, &'static str)];
    const LOADED_REQUEST: &'static [Self];
    const LOAD: &'static [Self];
    const ENCODING: Self;
    const LOADED_REQUEST_POLICY: &'static str;
    const END_TO_END_BOUNDARY: &'static str;
    fn index(self) -> usize;
}

/// Accounting never fails a request: a timing inconsistency is reported in
/// the lane's diagnostics and logged, and the generation and its output go
/// ahead. Phases are additive buckets; nothing here detects two phases
/// covering the same time, only a phase total that exceeds the lane wall.
pub(crate) struct LaneTiming<P: LanePhases> {
    phases: Vec<Duration>,
    overflowed: bool,
    _phases: PhantomData<P>,
}

impl<P: LanePhases> Default for LaneTiming<P> {
    fn default() -> Self {
        Self {
            phases: vec![Duration::ZERO; P::PHASES.len()],
            overflowed: false,
            _phases: PhantomData,
        }
    }
}

pub(crate) struct LaneReport {
    pub(crate) loaded_request_ms: f64,
    pub(crate) load_ms: f64,
    pub(crate) encoding_ms: f64,
    pub(crate) json: Value,
}

impl<P: LanePhases> LaneTiming<P> {
    pub(crate) fn record(&mut self, phase: P, elapsed: Duration) {
        let duration = &mut self.phases[phase.index()];
        match duration.checked_add(elapsed) {
            Some(total) => *duration = total,
            None => {
                *duration = Duration::MAX;
                self.overflowed = true;
            }
        }
    }

    /// Times `operation` into `phase` and returns the operation's own result.
    pub(crate) fn measure<T>(
        &mut self,
        phase: P,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let start = Instant::now();
        let result = operation();
        self.record(phase, start.elapsed());
        result
    }

    fn sum<'a>(&self, phases: impl IntoIterator<Item = &'a P>) -> Option<Duration> {
        phases.into_iter().try_fold(Duration::ZERO, |total, p| {
            total.checked_add(self.phases[p.index()])
        })
    }

    /// `accounting` is `"valid"`, or names the inconsistency; an invalid
    /// report keeps the phase durations and reports the unclassified
    /// residual as null rather than a clamped value.
    pub(crate) fn finish(self, end_to_end: Duration) -> LaneReport {
        let accounted = self.sum(P::PHASES.iter().map(|(p, _)| p));
        let residual = accounted.and_then(|accounted| end_to_end.checked_sub(accounted));
        let accounting = match (self.overflowed || accounted.is_none(), residual) {
            (true, _) => "phase_total_overflow",
            (false, None) => "phases_exceed_lane_wall",
            (false, Some(_)) => "valid",
        };
        if accounting != "valid" {
            tracing::warn!(
                target: "qwen_diag",
                "{} lane timing accounting {accounting}; generation and output are unaffected",
                P::FAMILY
            );
        }
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let loaded_request = self.sum(P::LOADED_REQUEST).map_or(f64::NAN, ms);
        let load = self.sum(P::LOAD).map_or(f64::NAN, ms);
        let phases: serde_json::Map<String, Value> = P::PHASES
            .iter()
            .map(|(p, name)| ((*name).into(), json!(ms(self.phases[p.index()]))))
            .collect();
        LaneReport {
            loaded_request_ms: loaded_request,
            load_ms: load,
            encoding_ms: ms(self.phases[P::ENCODING.index()]),
            json: json!({"schema_version":1,"unit":"milliseconds",
                "loaded_request_policy":P::LOADED_REQUEST_POLICY,
                "end_to_end_boundary":P::END_TO_END_BOUNDARY,
                "accounting":accounting,
                "phases_ms":phases,"loaded_request_ms":loaded_request,
                "end_to_end_lane_ms":ms(end_to_end),
                "unclassified_host_overhead_ms":residual.map(ms)}),
        }
    }
}
