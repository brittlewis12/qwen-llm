use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JobError {
    pub(crate) r#type: String,
    pub(crate) code: String,
    pub(crate) param: Option<String>,
    pub(crate) message: String,
}

impl JobError {
    pub(crate) fn restart() -> Self {
        Self {
            r#type: "server_error".into(),
            code: "server_restart".into(),
            param: None,
            message: "Server restarted before durable completion; inference was not retried."
                .into(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JobState {
    Queued,
    Running,
    Finalizing,
    Completed,
    Cancelled,
    Failed,
    Interrupted,
}

impl JobState {
    pub(crate) fn terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Cancelled | Self::Failed | Self::Interrupted
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum GenerationState {
    Pending,
    Running,
    Completed,
    Cancelled,
    Failed,
    Interrupted,
}

impl GenerationState {
    pub(crate) fn terminal(self) -> bool {
        !matches!(self, Self::Pending | Self::Running)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Phase {
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum StopReason {
    StopToken,
    TokenLimit,
    Cancelled,
    ExecutionError,
    ServerRestart,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Counters {
    pub(crate) prompt_tokens: u64,
    pub(crate) consumed_prompt_tokens: u64,
    pub(crate) sampled_tokens: u64,
    pub(crate) consumed_generated_tokens: u64,
}

impl Counters {
    pub(super) fn follows(&self, old: &Self) -> bool {
        self.prompt_tokens == old.prompt_tokens
            && self.consumed_prompt_tokens >= old.consumed_prompt_tokens
            && self.consumed_prompt_tokens <= self.prompt_tokens
            && self.sampled_tokens >= old.sampled_tokens
            && self.consumed_generated_tokens >= old.consumed_generated_tokens
            && self.consumed_generated_tokens <= self.sampled_tokens
            && (self.sampled_tokens == 0 || self.consumed_prompt_tokens == self.prompt_tokens)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Generation {
    pub(crate) state: GenerationState,
    pub(crate) phase: Option<Phase>,
    #[serde(flatten)]
    pub(crate) counters: Counters,
    pub(crate) stop_reason: Option<StopReason>,
    pub(crate) error: Option<JobError>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ObservationState {
    Pending,
    Writing,
    Complete,
    Partial,
    Failed,
    NotRequested,
}

impl ObservationState {
    pub(super) fn terminal(self) -> bool {
        !matches!(self, Self::Pending | Self::Writing)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct Observations {
    pub(crate) state: ObservationState,
    pub(crate) committed_records: u64,
    pub(crate) error: Option<JobError>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ResultStatus {
    pub(crate) available: bool,
    pub(crate) complete: bool,
    pub(crate) url: String,
    pub(crate) error: Option<JobError>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JobStatus {
    pub(crate) schema_version: u32,
    pub(crate) id: String,
    pub(crate) revision: u64,
    pub(crate) created_at_ms: u64,
    pub(crate) updated_at_ms: u64,
    pub(crate) state: JobState,
    pub(crate) cancel_requested: bool,
    pub(crate) generation: Generation,
    pub(crate) observations: Observations,
    pub(crate) result: ResultStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(crate) runtime: Option<RuntimeStatus>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeStatus {
    pub(crate) publication_error: Option<JobError>,
    pub(crate) execution_settled: bool,
    pub(crate) generation: Option<RuntimeGeneration>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RuntimeGeneration {
    pub(crate) stop_reason: StopReason,
    pub(crate) counters: Counters,
    pub(crate) error: Option<JobError>,
}

impl JobStatus {
    pub(super) fn queued(id: String, observations: bool, now: u64) -> Self {
        Self {
            schema_version: 1,
            result: ResultStatus {
                available: false,
                complete: false,
                url: format!("/v1/lens/jobs/{id}/result"),
                error: None,
            },
            id,
            revision: 0,
            runtime: None,
            created_at_ms: now,
            updated_at_ms: now,
            state: JobState::Queued,
            cancel_requested: false,
            generation: Generation {
                state: GenerationState::Pending,
                phase: None,
                counters: Counters::default(),
                stop_reason: None,
                error: None,
            },
            observations: Observations {
                state: if observations {
                    ObservationState::Pending
                } else {
                    ObservationState::NotRequested
                },
                committed_records: 0,
                error: None,
            },
        }
    }

    pub(super) fn request_cancel(&mut self) {
        self.cancel_requested = true;
        if self.state == JobState::Queued {
            self.state = JobState::Cancelled;
            self.generation.state = GenerationState::Cancelled;
            self.generation.stop_reason = Some(StopReason::Cancelled);
            if self.observations.state == ObservationState::Pending {
                self.observations.state = ObservationState::Partial;
            }
            self.result.complete = true;
        }
    }

    pub(super) fn interrupt(&mut self) {
        if self.state.terminal() {
            return;
        }
        self.state = JobState::Interrupted;
        if !self.generation.state.terminal() {
            self.generation.state = GenerationState::Interrupted;
            self.generation.phase = None;
            self.generation.stop_reason = Some(StopReason::ServerRestart);
            self.generation.error = Some(JobError::restart());
        }
        if !self.observations.state.terminal() {
            self.observations.state = ObservationState::Partial;
            self.observations.error = Some(JobError::restart());
        }
        self.result.complete = true;
        self.result.error.get_or_insert_with(JobError::restart);
    }
}
