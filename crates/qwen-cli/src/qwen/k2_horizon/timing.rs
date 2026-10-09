//! K2 request phases for the shared [`crate::lane_timing`] accumulator.

use crate::lane_timing::{LanePhases, LaneReport, LaneTiming};

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

impl LanePhases for Phase {
    const FAMILY: &'static str = "K2";
    const PHASES: &'static [(Self, &'static str)] = &[
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
    const LOADED_REQUEST: &'static [Self] = &[
        Phase::Encoding,
        Phase::RequestPreparation,
        Phase::ResidentExecution,
    ];
    const LOAD: &'static [Self] = &[Phase::ModelLoad, Phase::SessionSetup];
    const ENCODING: Self = Phase::Encoding;
    const LOADED_REQUEST_POLICY: &'static str =
        "sum_encoding_request_preparation_resident_execution_not_continuous_wall";
    const END_TO_END_BOUNDARY: &'static str = "k2_run_entry_through_generator_return_excludes_initial_gguf_open_final_formatting_stats_serialization";
    fn index(self) -> usize {
        self as usize
    }
}

pub(super) type Timing = LaneTiming<Phase>;
#[allow(dead_code)]
pub(super) type Report = LaneReport;

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    fn example(extra: Option<Phase>) -> Report {
        let mut timing = Timing::default();
        for phase in [
            Phase::Encoding,
            Phase::RequestPreparation,
            Phase::ResidentExecution,
        ] {
            timing.record(phase, Duration::from_millis(10));
        }
        if let Some(phase) = extra {
            timing.record(phase, Duration::from_secs(2));
        }
        let report = timing.finish(Duration::from_millis(
            31 + if extra.is_some() { 2000 } else { 0 },
        ));
        assert_eq!(report.json["accounting"], "valid");
        report
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
    /// Inconsistent accounting is reported, never an error that could
    /// suppress the generated output.
    #[test]
    fn k2_timing_reports_excess_and_overflow_without_failing() {
        let mut timing = Timing::default();
        timing.record(Phase::Encoding, Duration::from_secs(2));
        let report = timing.finish(Duration::from_secs(1));
        assert_eq!(report.json["accounting"], "phases_exceed_lane_wall");
        assert!(report.json["unclassified_host_overhead_ms"].is_null());
        assert_eq!(report.encoding_ms, 2000.);
        let mut timing = Timing::default();
        timing.record(Phase::ModelLoad, Duration::MAX);
        timing.record(Phase::ModelLoad, Duration::from_nanos(1));
        let report = timing.finish(Duration::from_secs(1));
        assert_eq!(report.json["accounting"], "phase_total_overflow");
        assert!(report.json["unclassified_host_overhead_ms"].is_null());
    }

    /// A failed operation's own error is returned, not replaced by timing.
    #[test]
    fn measure_returns_the_operation_error() {
        let mut timing = Timing::default();
        let error = timing
            .measure(Phase::Encoding, || -> anyhow::Result<()> {
                anyhow::bail!("encode failed")
            })
            .unwrap_err();
        assert_eq!(error.to_string(), "encode failed");
    }
}
