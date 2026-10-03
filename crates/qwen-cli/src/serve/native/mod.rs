//! Cache-isolated native baseline execution on the existing resident owner.

mod execute;
#[cfg(test)]
mod fitted_live;
#[cfg(test)]
mod intervention_live;
mod interventions;
mod measurements;
mod observe;
#[cfg(test)]
mod paired_live;
pub(crate) mod preconditions;
mod readouts;
pub(crate) mod registry;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) use execute::run_tokens;
#[cfg(test)]
pub(crate) use observe::run_cpu_readouts;
#[cfg(test)]
pub(crate) use tests::Fixture as CpuFixture;
mod staging;
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

pub(crate) fn with_staged(
    prepared: &Prepared,
    sink: &Sink,
    reserve: u64,
    execute: impl FnOnce(&registry::Staged) -> Outcome,
) -> Outcome {
    let staged = (|| {
        checkpoint(sink)?;
        let staged = if prepared.staging.keys.is_empty() {
            registry::Staged::default()
        } else {
            sink.stage(&prepared.staging, reserve)?
        };
        checkpoint(sink)?;
        Ok(staged)
    })();
    match staged {
        Ok(staged) => execute(&staged),
        Err(cause) => Outcome::diagnostic_preparation_failed(prepared.counters(), cause),
    }
}

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
    pub(crate) plain_readouts: bool,
    pub(crate) registry: Option<Arc<registry::Registry>>,
}

pub(crate) struct Prepared {
    pub(crate) prompt: Vec<i32>,
    pub(crate) sampling: SamplingConfig,
    pub(crate) max_tokens: usize,
    pub(super) record: Vec<u8>,
    pub(super) readouts: readouts::Plan,
    pub(super) interventions: interventions::Plan,
    pub(super) staging: staging::Plan,
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
    fn directions_supported(&self) -> bool {
        self.registry.as_ref().is_some_and(|r| {
            r.metadata().any(|a| {
                a["direction_rows"]
                    .as_array()
                    .is_some_and(|rows| !rows.is_empty())
            })
        })
    }
    fn check_preconditions(&self, request: &Request) -> Result<(), ApiError> {
        preconditions::check(request, &self.identity, |alias| {
            if self.plain_readouts && alias == "plain" {
                Some(self.identity.as_str())
            } else {
                self.registry.as_ref()?.asset(alias).ok()?["identity"].as_str()
            }
        })
    }
    pub(crate) fn prepare(&self, request: &Request) -> Result<Prepared, ApiError> {
        self.check_preconditions(request)?;
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
        if !self.plain_readouts
            && request
                .diagnostics
                .as_ref()
                .is_some_and(|d| !d.residual_pairs.is_empty())
        {
            return Err(ApiError::new(
                400,
                "unsupported_capability",
                "The resident model has no qualified residual pair capture.",
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
        let values = request
            .diagnostics
            .as_ref()
            .map(|d| d.readouts.as_slice())
            .unwrap_or_default();
        if !values.is_empty() && !self.plain_readouts {
            return Err(ApiError::new(
                400,
                "unsupported_capability",
                "The resident model has no qualified passive readout head.",
            ));
        }
        let readouts = readouts::Plan::compile_with_registry(
            values,
            self.layers,
            prepared.input.token_ids.len(),
            request.generation.max_new_tokens,
            self.tokenizer.n_vocab() as usize,
            self.registry.clone(),
            self.hidden,
        )
        .map_err(|cause| ApiError::new(400, "invalid_readout", cause.to_string()))?;
        let diagnostics = request.diagnostics.as_ref();
        let pairs = measurements::Plan::compile(
            diagnostics
                .map(|d| d.residual_pairs.as_slice())
                .unwrap_or_default(),
            self.layers,
            prepared.input.token_ids.len(),
            request.generation.max_new_tokens,
        )
        .map_err(|cause| ApiError::new(400, "invalid_residual_pair", cause.to_string()))?;
        let readouts = readouts
            .with_pairs(pairs, self.hidden, self.tokenizer.n_vocab() as usize)
            .map_err(|cause| ApiError::new(400, "invalid_retention", cause.to_string()))?;
        let interventions = interventions::Plan::compile(
            diagnostics
                .map(|d| d.directions.as_slice())
                .unwrap_or_default(),
            diagnostics
                .map(|d| d.operations.as_slice())
                .unwrap_or_default(),
            self.layers,
            prepared.input.token_ids.len(),
            request.generation.max_new_tokens,
            self.tokenizer.n_vocab() as usize,
            self.hidden,
            self.registry.as_deref(),
        )
        .map_err(|cause| ApiError::new(400, "invalid_intervention", cause.to_string()))?;
        let staging = staging::Plan::new(
            self.registry.clone(),
            readouts
                .matrices
                .union(&interventions.matrices)
                .cloned()
                .collect(),
        )
        .map_err(|cause| ApiError::new(400, "invalid_diagnostic_staging", cause.to_string()))?;
        // Bound serialization before materializing a Value tree of token bytes
        // and spans. A small authored request can otherwise amplify substantially.
        let bytes = writer::encode(&prepared.record(request.prefill()))
            .map_err(|cause| ApiError::new(413, "prepared_input_too_large", cause.to_string()))?;
        let mut record: Value = serde_json::from_slice(&bytes).expect("serialized prepared input");
        record["model_identity"] = self.identity.clone().into();
        record["model_identity_kind"] = "runtime_gguf_metadata_not_content_hash".into();
        record["requested_preconditions"] = json!(request.preconditions);
        record["asset_identities"] = json!({});
        for (alias, asset) in &interventions.assets {
            record["asset_identities"][alias] = asset["identity"].clone();
        }
        for readout in &readouts.readouts {
            record["asset_identities"][&readout.lens] = if readout.lens == "plain" {
                self.identity.clone().into()
            } else {
                self.registry
                    .as_ref()
                    .expect("compiled registry")
                    .asset(&readout.lens)
                    .expect("compiled asset")["identity"]
                    .clone()
            };
        }
        record["resolved_scopes"] = json!(
            readouts
                .readouts
                .iter()
                .map(|r| json!({"id":r.id,"kind":"readout","scope":r.scope}))
                .chain(
                    interventions
                        .operations
                        .iter()
                        .map(|op| json!({"id":op.id,"kind":"operation","scope":op.scope}))
                )
                .collect::<Vec<_>>()
        );
        record["readout_admission"] = json!({"head_evaluations_upper":readouts.head_evaluations,"output_rows_upper":readouts.output_rows,"output_scores_upper":readouts.output_scores});
        record["residual_pair_scopes"] = json!(readouts.pairs.requests);
        record["effective_operation_ids"] = json!(
            interventions
                .operations
                .iter()
                .filter(|op| crate::lens_intervention::operation_enabled(op))
                .map(|op| &op.id)
                .collect::<Vec<_>>()
        );
        record["retention_admission"] = json!({"hidden_size":self.hidden,"vocabulary_size":self.tokenizer.n_vocab(),
            "raw_bytes_upper":readouts.archive_bytes,"source_arrays_upper":readouts.retained_sources.len(),"score_arrays_upper":readouts.retained_heads.len(),
            "before_arrays_upper":readouts.pairs.sites(),"pair_rows_upper":readouts.pairs.rows});
        record["intervention_admission"] = json!({"applications_upper":interventions.applications,"direction_rows":interventions.direction_rows,"projected_rows":interventions.projected_rows,"matrix_bytes":staging.matrix_bytes});
        record["sampling"] = json!(request.generation.sampling);
        record["execution"] = json!({"prefill":"serial","decode":"serial","cache":"isolated_diagnostic","forward":"post_block_serial","sampler_version":qwen_llm::sampling::SAMPLER_ALGORITHM_VERSION});
        let record = writer::encode(&record)
            .map_err(|cause| ApiError::new(413, "prepared_input_too_large", cause.to_string()))?;
        Ok(Prepared {
            prompt: prepared.input.token_ids,
            sampling: request.generation.sampling.config(),
            max_tokens: request.generation.max_new_tokens,
            record,
            readouts,
            interventions,
            staging,
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
            "assistant_prefill_channels":["reasoning","final"],"operations":if self.profile.directions_supported() { interventions::OPERATORS.to_vec() } else {vec![]},"readout_modes":if self.profile.plain_readouts { vec!["full_vocabulary"] } else { vec![] },"capture_stage":"post_block_after_operations","request_preconditions":true,
            "execution":{"prefill":"serial","decode":"serial","cache":"isolated_diagnostic","baseline_only":!self.profile.plain_readouts},
            "readout_retention_modes":if self.profile.plain_readouts {vec!["scores_and_residual"]} else {vec![]},
            "residual_pair_capture":self.profile.plain_readouts,
            "limits":{"max_new_tokens":self.profile.max_tokens,"max_context_tokens":self.profile.context,"max_token_piece_bytes":MAX_TOKEN_PIECE_BYTES,
                "max_directions":if self.profile.directions_supported() {interventions::MAX_DIRECTIONS} else {0},
                "max_operations":if self.profile.directions_supported() {interventions::MAX_OPERATIONS} else {0},
                "max_operation_applications":interventions::MAX_APPLICATIONS,"max_direction_rows":interventions::MAX_DIRECTION_ROWS,
                "max_projection_products":interventions::MAX_PROJECTION_PRODUCTS,
                "max_archive_bytes":super::jobs::store::MAX_ARCHIVE_BYTES,"max_retained_array_bytes":super::jobs::store::MAX_ARRAY_BYTES,
                "max_residual_pairs":if self.profile.plain_readouts {measurements::MAX_REQUESTS} else {0},"max_residual_pair_rows":measurements::MAX_ROWS,
                "max_readouts":if self.profile.plain_readouts {readouts::MAX_READOUTS} else {0},"max_top_k":if self.profile.plain_readouts {readouts::MAX_TOP_K.min(self.profile.tokenizer.n_vocab() as usize)} else {0},"max_head_evaluations":readouts::MAX_HEAD_EVALUATIONS,"max_readout_rows":readouts::MAX_ROWS,"max_readout_scores":readouts::MAX_SCORES,"max_readout_label_bytes":readouts::MAX_LABEL_BYTES,"max_queued_jobs":1}})
    }
    fn assets(&self) -> Value {
        let mut assets = if self.profile.plain_readouts {
            vec![
                json!({"alias":"plain","kind":"plain_logit_lens","identity":self.profile.identity,"available":true,"unavailable_reason":null,
                "source_layers":(0..self.profile.layers).collect::<Vec<_>>(),"target_layer":null,"readout_modes":["full_vocabulary"],"direction_rows":[],"transfer":"identity"}),
            ]
        } else {
            vec![]
        };
        if let Some(registry) = &self.profile.registry {
            assets.extend(registry.metadata().cloned());
        }
        json!({"schema_version":1,"model_identity":self.profile.identity,"assets":assets})
    }
    fn reserve(&self, request: &Request) -> Result<Box<dyn ReservedSubmission>, ApiError> {
        self.profile.check_preconditions(request)?;
        let slot = self
            .gate
            .reserve()
            .map_err(|message| ApiError::new(429, "queue_full", message))?;
        let activity = self.activity.try_prepare().ok_or_else(|| {
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
    fn archive_bytes(&self) -> u64 {
        self.prepared.readouts.archive_bytes
    }
    fn enqueue(self: Box<Self>, job: AcceptedJob) -> Result<(), String> {
        let Self {
            prepared,
            slot,
            mut activity,
            sender,
            store,
        } = *self;
        activity.mark_work();
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
