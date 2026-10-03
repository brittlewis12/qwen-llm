//! Cache-isolated native baseline execution on the existing resident owner.

mod execute;
pub(crate) mod preconditions;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use execute::run_tokens;
#[cfg(test)]
pub(crate) use tests::Fixture as CpuFixture;
mod writer;
pub(crate) use execute::{Outcome, checkpoint, run_loaded};
pub(super) use writer::CPU_UPPER_BYTES;
pub(crate) use writer::Sink;

use super::{
    control,
    jobs::{state::Counters, store::JobStore},
    lens_http::{AcceptedJob, Admission, ApiError, ReservedSubmission, input::Request},
    owner_activity,
};
use crate::prompt_template::QwenPromptTemplate;
use anyhow::{Context, Result, ensure};
use qwen_llm::{sampling::SamplingConfig, tokenizer::Tokenizer};
use serde_json::{Value, json};
use std::sync::{Arc, mpsc::SyncSender};

pub(crate) const RECORD_BYTES: usize = 1024 * 1024;
// Leave room for worst-case JSON byte numbers, escaped/lossy text and framing.
pub(crate) const MAX_TOKEN_PIECE_BYTES: usize = (RECORD_BYTES - 1024) / 10;

pub(crate) struct Profile {
    pub(crate) model_id: String,
    pub(crate) identity: String,
    pub(crate) protocol: QwenPromptTemplate,
    pub(crate) tokenizer: Arc<Tokenizer>,
    pub(crate) layers: u32,
    pub(crate) hidden: usize,
    pub(crate) context: usize,
    pub(crate) max_tokens: usize,
    pub(crate) no_thinking_supported: bool,
}

pub(crate) struct Prepared {
    pub(crate) prompt: Vec<i32>,
    pub(crate) sampling: SamplingConfig,
    pub(crate) max_tokens: usize,
    pub(super) record: Vec<u8>,
}
impl Prepared {
    pub(crate) fn counters(&self) -> Counters {
        Counters {
            prompt_tokens: self.prompt.len() as u64,
            ..Counters::default()
        }
    }
    pub(crate) fn capacity(&self) -> Result<usize> {
        ensure!(
            !self.prompt.is_empty() && self.max_tokens > 0,
            "empty native generation"
        );
        self.prompt
            .len()
            .checked_add(self.max_tokens - 1)
            .context("native context overflow")
    }
}

impl Profile {
    pub(crate) fn prepare(&self, request: &Request) -> Result<Prepared, ApiError> {
        preconditions::check(request, &self.identity, |_| None)?;
        if matches!(
            request.input,
            super::lens_http::input::Input::Messages {
                generation_mode: crate::lens_input::LensMessageMode::NoThinking,
                ..
            }
        ) && !self.no_thinking_supported
        {
            return Err(ApiError::new(
                400,
                "unsupported_capability",
                "The resident deployment has no qualified no-thinking mode.",
            ));
        }
        if request.diagnostics.as_ref().is_some_and(|d| {
            !d.directions.is_empty()
                || !d.operations.is_empty()
                || !d.readouts.is_empty()
                || !d.residual_pairs.is_empty()
        }) {
            return Err(ApiError::new(
                400,
                "unsupported_capability",
                "Only baseline generation is recovered; diagnostic requests are not silently ignored.",
            ));
        }
        let prepared = request
            .prepare_generation(
                self.protocol,
                &self.tokenizer,
                self.context,
                self.max_tokens,
            )
            .map_err(|cause| ApiError::new(400, "invalid_request", cause.to_string()))?;
        // Bound serialization before materializing a Value tree of token bytes
        // and spans. A small authored request can otherwise amplify substantially.
        let bytes = writer::encode(&prepared.record(request.prefill()))
            .map_err(|cause| ApiError::new(413, "prepared_input_too_large", cause.to_string()))?;
        let mut record: Value = serde_json::from_slice(&bytes).expect("serialized prepared input");
        record["model_identity"] = self.identity.clone().into();
        record["model_identity_kind"] = "runtime_gguf_metadata_not_content_hash".into();
        record["requested_preconditions"] = json!(request.preconditions);
        record["asset_identities"] = json!({});
        record["resolved_scopes"] = json!([]);
        record["effective_operation_ids"] = json!([]);
        record["sampling"] = json!(request.generation.sampling);
        record["execution"] = json!({"prefill":"serial","decode":"serial","cache":"isolated_diagnostic","forward":"post_block_serial","sampler_version":qwen_llm::sampling::SAMPLER_ALGORITHM_VERSION});
        let record = writer::encode(&record)
            .map_err(|cause| ApiError::new(413, "prepared_input_too_large", cause.to_string()))?;
        Ok(Prepared {
            prompt: prepared.input.token_ids,
            sampling: request.generation.sampling.config(),
            max_tokens: request.generation.max_new_tokens,
            record,
        })
    }
}

pub(super) struct NativeAdmission {
    pub(super) profile: Arc<Profile>,
    pub(super) store: Arc<JobStore>,
    pub(super) sender: SyncSender<control::Event>,
    pub(super) gate: control::ExecutionGate,
    pub(super) activity: owner_activity::Admission,
}
impl Admission for NativeAdmission {
    fn capabilities(&self) -> Value {
        let mut generation_modes = if self.profile.protocol == QwenPromptTemplate::Qwen38 {
            vec![
                "auto",
                "thinking",
                "no_thinking",
                "low",
                "medium",
                "high",
                "xhigh",
            ]
        } else {
            vec!["auto", "thinking", "no_thinking"]
        };
        if !self.profile.no_thinking_supported {
            generation_modes.retain(|mode| *mode != "no_thinking");
        }
        json!({"schema_version":1,"available":true,"unavailable_reason":null,
            "model":{"id":self.profile.model_id,"identity":self.profile.identity,"template":self.profile.protocol.renderer_name(),"layers":self.profile.layers,"vocabulary_size":self.profile.tokenizer.n_vocab(),"hidden_size":self.profile.hidden},
            "model_identity_kind":"runtime_gguf_metadata_not_content_hash","input_kinds":["messages"],
            "generation_modes":generation_modes,
            "assistant_prefill_channels":["reasoning","final"],"operations":[],"readout_modes":[],"capture_stage":"post_block_after_operations","request_preconditions":true,
            "execution":{"prefill":"serial","decode":"serial","cache":"isolated_diagnostic","baseline_only":true},
            "limits":{"max_new_tokens":self.profile.max_tokens,"max_context_tokens":self.profile.context,"max_token_piece_bytes":MAX_TOKEN_PIECE_BYTES,"max_directions":0,"max_operations":0,"max_readouts":0,"max_top_k":0,"max_queued_jobs":1}})
    }
    fn assets(&self) -> Value {
        json!({"schema_version":1,"model_identity":self.profile.identity,"assets":[]})
    }
    fn reserve(&self, request: &Request) -> Result<Box<dyn ReservedSubmission>, ApiError> {
        preconditions::check(request, &self.profile.identity, |_| None)?;
        let slot = self
            .gate
            .reserve()
            .map_err(|message| ApiError::new(429, "queue_full", message))?;
        let activity = self.activity.try_admit().ok_or_else(|| {
            ApiError::new(503, "server_stopping", "Model owner admission is closed")
        })?;
        let prepared = self.profile.prepare(request)?;
        Ok(Box::new(Reservation {
            prepared,
            slot,
            activity,
            sender: self.sender.clone(),
            store: Arc::clone(&self.store),
        }))
    }
}

struct Reservation {
    prepared: Prepared,
    slot: control::ExecutionPermit,
    activity: owner_activity::ActivityGuard,
    sender: SyncSender<control::Event>,
    store: Arc<JobStore>,
}
impl ReservedSubmission for Reservation {
    fn enqueue(self: Box<Self>, job: AcceptedJob) -> Result<(), String> {
        let Self {
            prepared,
            slot,
            activity,
            sender,
            store,
        } = *self;
        let writer = writer::Writer::spawn(store, job.id, job.control, &prepared, slot.gate())
            .map_err(|cause| cause.to_string())?;
        let gate = slot.gate();
        gate.deliver(
            &sender,
            control::Event::Native(Queued {
                prepared,
                writer: Some(writer),
                _slot: slot,
                _activity: activity,
            }),
        )
        .map_err(str::to_owned)
    }
}

pub(super) struct Queued {
    prepared: Prepared,
    writer: Option<writer::Writer>,
    _slot: control::ExecutionPermit,
    _activity: owner_activity::ActivityGuard,
}
impl Queued {
    pub(super) fn run(mut self, backend: &mut dyn super::http::GenerationBackend) -> Result<()> {
        let writer = self.writer.take().expect("queued writer");
        let outcome = match writer.wait_ready() {
            Ok(writer::Readiness::Execute) => {
                backend.generate_native(&self.prepared, writer.sink())
            }
            Ok(writer::Readiness::Settled) => return writer.join_settled(),
            Err(cause) => {
                tracing::error!("native publication preparation: {cause:#}");
                Outcome::failed(
                    self.prepared.counters(),
                    "artifact_prepare_failed",
                    "Native publication could not be prepared; no model forward ran.",
                )
            }
        };
        writer.finish(outcome)
    }
}
