//! Phase-accounted request timing shared by lanes that separate setup from
//! the loaded request (K2, GLM-5.3). Each family names its phases, which of
//! them form the loaded request and the model load, and its encoding phase;
//! the accumulator checks that phases never overlap or exceed the lane wall.

use anyhow::{Context, Result};
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

pub(crate) struct LaneTiming<P: LanePhases> {
    phases: Vec<Duration>,
    _phases: PhantomData<P>,
}

impl<P: LanePhases> Default for LaneTiming<P> {
    fn default() -> Self {
        Self {
            phases: vec![Duration::ZERO; P::PHASES.len()],
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
    pub(crate) fn record(&mut self, phase: P, elapsed: Duration) -> Result<()> {
        let duration = &mut self.phases[phase.index()];
        *duration = duration
            .checked_add(elapsed)
            .with_context(|| format!("{} timing phase overflow", P::FAMILY))?;
        Ok(())
    }

    pub(crate) fn measure<T>(
        &mut self,
        phase: P,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let start = Instant::now();
        let result = operation();
        self.record(phase, start.elapsed())?;
        result
    }

    fn sum<'a>(&self, phases: impl IntoIterator<Item = &'a P>) -> Result<Duration> {
        phases.into_iter().try_fold(Duration::ZERO, |total, p| {
            total
                .checked_add(self.phases[p.index()])
                .with_context(|| format!("{} timing total overflow", P::FAMILY))
        })
    }

    pub(crate) fn finish(self, end_to_end: Duration) -> Result<LaneReport> {
        let accounted = self.sum(P::PHASES.iter().map(|(p, _)| p))?;
        let residual = end_to_end
            .checked_sub(accounted)
            .with_context(|| format!("{} timing phases overlap or exceed lane wall", P::FAMILY))?;
        let loaded_request = self.sum(P::LOADED_REQUEST)?;
        let load = self.sum(P::LOAD)?;
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let phases: serde_json::Map<String, Value> = P::PHASES
            .iter()
            .map(|(p, name)| ((*name).into(), json!(ms(self.phases[p.index()]))))
            .collect();
        Ok(LaneReport {
            loaded_request_ms: ms(loaded_request),
            load_ms: ms(load),
            encoding_ms: ms(self.phases[P::ENCODING.index()]),
            json: json!({"schema_version":1,"unit":"milliseconds",
                "loaded_request_policy":P::LOADED_REQUEST_POLICY,
                "end_to_end_boundary":P::END_TO_END_BOUNDARY,
                "phases_ms":phases,"loaded_request_ms":ms(loaded_request),
                "end_to_end_lane_ms":ms(end_to_end),"unclassified_host_overhead_ms":ms(residual)}),
        })
    }
}
