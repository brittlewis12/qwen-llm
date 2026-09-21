use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(super) enum Phase {
    ArtifactLayout,
    TokenizerConstruction,
    ArtifactVerification,
    InputAcquisition,
    Rendering,
    RequestPreparation,
    Encoding,
    ModelLoad,
    SessionSetup,
    ResidentExecution,
}
const PHASES: [(Phase, &str); 10] = [
    (Phase::ArtifactLayout, "artifact_layout"),
    (Phase::TokenizerConstruction, "tokenizer_construction"),
    (Phase::ArtifactVerification, "artifact_verification"),
    (Phase::InputAcquisition, "input_acquisition"),
    (Phase::Rendering, "rendering"),
    (Phase::RequestPreparation, "request_preparation"),
    (Phase::Encoding, "encoding"),
    (Phase::ModelLoad, "model_load"),
    (Phase::SessionSetup, "session_setup"),
    (Phase::ResidentExecution, "resident_execution"),
];

#[derive(Default)]
pub(super) struct Timing {
    phases: [Duration; 10],
}
pub(super) struct Report {
    pub(super) loaded_request_ms: f64,
    pub(super) load_ms: f64,
    pub(super) encoding_ms: f64,
    pub(super) json: Value,
}

impl Timing {
    pub(super) fn record(&mut self, phase: Phase, elapsed: Duration) -> Result<()> {
        let duration = &mut self.phases[phase as usize];
        *duration = duration
            .checked_add(elapsed)
            .context("K2 timing phase overflow")?;
        Ok(())
    }
    pub(super) fn measure<T>(
        &mut self,
        phase: Phase,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let start = Instant::now();
        let result = operation();
        self.record(phase, start.elapsed())?;
        result
    }
    fn sum(&self, phases: impl IntoIterator<Item = Phase>) -> Result<Duration> {
        phases.into_iter().try_fold(Duration::ZERO, |total, p| {
            total
                .checked_add(self.phases[p as usize])
                .context("K2 timing total overflow")
        })
    }
    pub(super) fn finish(self, end_to_end: Duration) -> Result<Report> {
        let accounted = self.sum(PHASES.map(|(p, _)| p))?;
        let residual = end_to_end
            .checked_sub(accounted)
            .context("K2 timing phases overlap or exceed lane wall")?;
        let loaded_request = self.sum([
            Phase::Encoding,
            Phase::RequestPreparation,
            Phase::ResidentExecution,
        ])?;
        let load = self.sum([Phase::ModelLoad, Phase::SessionSetup])?;
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let phases: serde_json::Map<String, Value> = PHASES
            .into_iter()
            .map(|(p, name)| (name.into(), json!(ms(self.phases[p as usize]))))
            .collect();
        Ok(Report {
            loaded_request_ms: ms(loaded_request),
            load_ms: ms(load),
            encoding_ms: ms(self.phases[Phase::Encoding as usize]),
            json: json!({"schema_version":1,"unit":"milliseconds",
                "loaded_request_policy":"sum_encoding_request_preparation_resident_execution_not_continuous_wall",
                "end_to_end_boundary":"k2_run_entry_through_generator_return_excludes_initial_gguf_open_final_formatting_stats_serialization",
                "phases_ms":phases,"loaded_request_ms":ms(loaded_request),
                "end_to_end_lane_ms":ms(end_to_end),"unclassified_host_overhead_ms":ms(residual)}),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn example(extra: Option<Phase>) -> Report {
        let mut timing = Timing::default();
        for phase in [
            Phase::Encoding,
            Phase::RequestPreparation,
            Phase::ResidentExecution,
        ] {
            timing.record(phase, Duration::from_millis(10)).unwrap();
        }
        if let Some(phase) = extra {
            timing.record(phase, Duration::from_secs(2)).unwrap();
        }
        timing
            .finish(Duration::from_millis(
                31 + if extra.is_some() { 2000 } else { 0 },
            ))
            .unwrap()
    }
    #[test]
    fn k2_setup_wait_and_verification_do_not_contaminate_loaded_request_time() {
        let baseline = example(None);
        assert_eq!(baseline.loaded_request_ms, 30.);
        assert_eq!(example(Some(Phase::TokenizerConstruction)).load_ms, 0.);
        assert_eq!(example(Some(Phase::ArtifactLayout)).load_ms, 0.);
        assert_eq!(example(Some(Phase::ModelLoad)).load_ms, 2000.);
        assert_eq!(example(Some(Phase::SessionSetup)).load_ms, 2000.);
        for phase in [
            Phase::ArtifactVerification,
            Phase::InputAcquisition,
            Phase::ArtifactLayout,
            Phase::TokenizerConstruction,
            Phase::ModelLoad,
            Phase::SessionSetup,
            Phase::Rendering,
        ] {
            let added = example(Some(phase));
            assert_eq!(added.loaded_request_ms, baseline.loaded_request_ms);
            assert_eq!(added.encoding_ms, baseline.encoding_ms);
            assert!((added.json["end_to_end_lane_ms"].as_f64().unwrap() - 2031.).abs() < 1e-9);
            assert_eq!(added.json["unclassified_host_overhead_ms"], 1.);
        }
        assert_eq!(baseline.json["phases_ms"]["artifact_verification"], 0.);
        assert_eq!(baseline.json["phases_ms"]["rendering"], 0.);
        let chat = example(Some(Phase::Rendering));
        assert_eq!(chat.json["phases_ms"]["rendering"], 2000.);
    }
    #[test]
    fn k2_timing_rejects_overlap_and_duration_overflow() {
        let mut timing = Timing::default();
        timing
            .record(Phase::Encoding, Duration::from_secs(2))
            .unwrap();
        assert!(timing.finish(Duration::from_secs(1)).is_err());
        let mut timing = Timing::default();
        timing.record(Phase::ModelLoad, Duration::MAX).unwrap();
        assert!(
            timing
                .record(Phase::ModelLoad, Duration::from_nanos(1))
                .is_err()
        );
    }
}
