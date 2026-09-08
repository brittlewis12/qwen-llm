//! Run output envelope and serialization.

use super::*;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RunExecutionScheduleBasis {
    EffectivePlan,
    SweepSourcePlan,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RunExecution {
    pub(super) requested_prefill: PrefillExecution,
    pub(super) effective_prefill: RunEffectivePrefill,
    pub(super) schedule_basis: RunExecutionScheduleBasis,
    pub(super) numerical_relationship: RunNumericalRelationship,
    pub(super) minimum_span_tokens: usize,
    pub(super) chunk_cap_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) block_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) attention_matrix_max_position: Option<u64>,
    pub(super) scratch_priced_upper_bytes: u64,
    pub(super) packed_spans: Vec<RunPackedPrefillSpan>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) serial_reason: Option<RunSerialReason>,
}

impl RunExecution {
    pub(crate) fn serial(
        requested_prefill: PrefillExecution,
        schedule_basis: RunExecutionScheduleBasis,
        automatic_reason: RunSerialReason,
    ) -> Self {
        let serial_reason = if requested_prefill == PrefillExecution::Serial {
            RunSerialReason::RequestedSerial
        } else {
            automatic_reason
        };
        Self {
            requested_prefill,
            effective_prefill: RunEffectivePrefill::Serial,
            schedule_basis,
            numerical_relationship: RunNumericalRelationship::SerialReference,
            minimum_span_tokens: PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS,
            chunk_cap_tokens: PACKED_PREFILL_CHUNK_CAP_TOKENS,
            block_tokens: None,
            attention_matrix_max_position: None,
            scratch_priced_upper_bytes: 0,
            packed_spans: Vec::new(),
            serial_reason: Some(serial_reason),
        }
    }

    pub(super) fn dense_packed(
        schedule_basis: RunExecutionScheduleBasis,
        block_tokens: u32,
        attention_matrix_max_position: u64,
        scratch_priced_upper_bytes: u64,
        packed_spans: Vec<RunPackedPrefillSpan>,
    ) -> Self {
        Self {
            requested_prefill: PrefillExecution::Auto,
            effective_prefill: RunEffectivePrefill::DensePackedPassiveSpans,
            schedule_basis,
            numerical_relationship:
                RunNumericalRelationship::PackedReductionTopologyDiffersFromSerial,
            minimum_span_tokens: PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS,
            chunk_cap_tokens: PACKED_PREFILL_CHUNK_CAP_TOKENS,
            block_tokens: Some(block_tokens),
            attention_matrix_max_position: (attention_matrix_max_position > 0)
                .then_some(attention_matrix_max_position),
            scratch_priced_upper_bytes,
            packed_spans,
            serial_reason: None,
        }
    }

    pub(crate) fn runtime_serial(
        requested_prefill: PrefillExecution,
        automatic_reason: RunSerialReason,
    ) -> Self {
        Self::serial(
            requested_prefill,
            RunExecutionScheduleBasis::EffectivePlan,
            automatic_reason,
        )
    }

    pub(crate) fn validate(&self, runtime_kind: &str, prompt_len: usize) -> Result<()> {
        ensure!(
            self.minimum_span_tokens == PACKED_PREFILL_MIN_PASSIVE_SPAN_TOKENS
                && self.chunk_cap_tokens == PACKED_PREFILL_CHUNK_CAP_TOKENS,
            "run execution policy constants are unsupported"
        );
        match self.effective_prefill {
            RunEffectivePrefill::Serial => {
                ensure!(
                    self.numerical_relationship == RunNumericalRelationship::SerialReference
                        && self.block_tokens.is_none()
                        && self.attention_matrix_max_position.is_none()
                        && self.scratch_priced_upper_bytes == 0
                        && self.packed_spans.is_empty()
                        && self.serial_reason.is_some(),
                    "serial run execution metadata contains packed state"
                );
                let reason_matches = match (self.requested_prefill, self.serial_reason) {
                    (PrefillExecution::Serial, Some(RunSerialReason::RequestedSerial)) => true,
                    (PrefillExecution::Auto, Some(reason)) => {
                        reason != RunSerialReason::RequestedSerial
                    }
                    _ => false,
                };
                ensure!(
                    reason_matches,
                    "serial run execution reason differs from the request"
                );
                let runtime_matches = match self.serial_reason {
                    Some(RunSerialReason::RequestedSerial) => true,
                    Some(
                        RunSerialReason::NoEligiblePassiveSpan
                        | RunSerialReason::DensePackedMemoryAdmissionDenied
                        | RunSerialReason::MoePackedNotQualified
                        | RunSerialReason::CohortSerialPolicy,
                    ) => runtime_kind == "ordinary_qwen",
                    Some(RunSerialReason::FlashNextPackedNotImplemented) => {
                        runtime_kind == "flash_next"
                    }
                    Some(RunSerialReason::MusePackedNotImplemented) => {
                        runtime_kind == "muse_glimmer"
                    }
                    None => false,
                };
                ensure!(
                    runtime_matches,
                    "serial run execution reason differs from the runtime"
                );
            }
            RunEffectivePrefill::DensePackedPassiveSpans => {
                ensure!(
                    runtime_kind == "ordinary_qwen"
                        && self.requested_prefill == PrefillExecution::Auto
                        && self.numerical_relationship
                            == RunNumericalRelationship::PackedReductionTopologyDiffersFromSerial
                        && self.serial_reason.is_none()
                        && self.scratch_priced_upper_bytes > 0
                        && !self.packed_spans.is_empty(),
                    "packed run execution metadata has an inconsistent runtime or policy"
                );
                let block_tokens = self
                    .block_tokens
                    .context("packed run execution requires block_tokens")?;
                ensure!(
                    self.attention_matrix_max_position
                        .is_none_or(|position| position >= u64::from(block_tokens)),
                    "packed run execution attention-matrix extent is smaller than its block"
                );
                let final_prompt_index = prompt_len
                    .checked_sub(1)
                    .context("packed run execution requires a nonempty prompt")?;
                let mut previous_end = 0usize;
                let mut longest = 0usize;
                for span in &self.packed_spans {
                    let length = span
                        .end
                        .checked_sub(span.start)
                        .context("packed run execution span is reversed")?;
                    ensure!(
                        span.start >= previous_end
                            && span.end <= final_prompt_index
                            && length >= self.minimum_span_tokens,
                        "packed run execution contains an invalid, overlapping, or final-token span"
                    );
                    previous_end = span.end;
                    longest = longest.max(length);
                }
                ensure!(
                    usize::try_from(block_tokens)? == longest.min(self.chunk_cap_tokens),
                    "packed run execution block size differs from its spans"
                );
            }
        }
        Ok(())
    }

    pub(crate) fn validate_against_plan(
        &self,
        runtime_kind: &str,
        plan: &LensPlan,
        prompt_len: usize,
    ) -> Result<()> {
        self.validate(runtime_kind, prompt_len)?;
        let expected = expected_packed_prefill_spans(plan, prompt_len)?;
        match (self.effective_prefill, self.serial_reason) {
            (RunEffectivePrefill::DensePackedPassiveSpans, None) => ensure!(
                self.packed_spans == expected,
                "packed run execution spans differ from passive plan spans"
            ),
            (RunEffectivePrefill::Serial, Some(RunSerialReason::NoEligiblePassiveSpan)) => {
                ensure!(
                    expected.is_empty(),
                    "serial run claims no eligible passive span, but the plan has one"
                )
            }
            (
                RunEffectivePrefill::Serial,
                Some(RunSerialReason::DensePackedMemoryAdmissionDenied),
            ) => ensure!(
                !expected.is_empty(),
                "serial run claims packed-memory denial without an eligible passive span"
            ),
            _ => {}
        }
        Ok(())
    }

    pub(super) fn effective_prefill(&self) -> RunEffectivePrefill {
        self.effective_prefill
    }

    pub(super) fn serial_reason(&self) -> Option<RunSerialReason> {
        self.serial_reason
    }

    pub(super) fn packed_spans(&self) -> &[RunPackedPrefillSpan] {
        &self.packed_spans
    }

    pub(crate) fn schedule_basis(&self) -> RunExecutionScheduleBasis {
        self.schedule_basis
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct RunOutput {
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(crate) linear_transports: Vec<serde_json::Value>,
    pub(super) schema: &'static str,
    pub(super) schema_version: u32,
    pub(super) runtime_kind: &'static str,
    pub(super) model_path: PathBuf,
    pub(super) canonical_plan_path: PathBuf,
    pub(super) authored_plan: LensPlan,
    pub(super) authored_plan_canonical_json_blake3: String,
    pub(super) plan: LensPlan,
    pub(super) position_bindings: Vec<PositionBinding>,
    pub(super) input_source: &'static str,
    pub(super) add_special_tokens: Option<bool>,
    pub(super) rendering: LensInputRendering,
    pub(super) prompt_token_ids: Vec<i32>,
    pub(super) generated_token_ids: Vec<i32>,
    pub(super) sampler: RunSampler,
    pub(super) max_new_tokens: usize,
    pub(super) decoded_text: String,
    pub(super) stop_reason: String,
    pub(super) execution: RunExecution,
    pub(super) operation_applications: Vec<OperationApplication>,
    pub(super) requested_live_readouts: Vec<ReadoutDefinition>,
    pub(super) live_readouts: Vec<LiveReadout>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub(super) native_hyper_captures: Vec<NativeHyperCapture>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(super) execution_binding: Option<RunExecutionBinding>,
}

pub(super) struct BundleOutputBudget {
    pub(super) consumed: u64,
    pub(super) reserved: u64,
}

impl BundleOutputBudget {
    pub(super) fn new(reserved: usize) -> Result<Self> {
        let reserved = u64::try_from(reserved).context("reserved sweep output bytes")?;
        ensure!(
            reserved <= MAX_SWEEP_BUNDLE_BYTES,
            "reserved bundle output bytes {reserved} exceed limit {MAX_SWEEP_BUNDLE_BYTES}"
        );
        Ok(Self {
            consumed: 0,
            reserved,
        })
    }

    pub(super) fn charge(&mut self, byte_length: usize) -> Result<()> {
        let consumed = self
            .consumed
            .checked_add(u64::try_from(byte_length).context("serialized child byte length")?)
            .context("cohort serialized child byte count overflow")?;
        let committed = consumed
            .checked_add(self.reserved)
            .context("cohort committed byte count overflow")?;
        ensure!(
            committed <= MAX_SWEEP_BUNDLE_BYTES,
            "committed bundle bytes {committed} exceed limit {}",
            MAX_SWEEP_BUNDLE_BYTES
        );
        self.consumed = consumed;
        Ok(())
    }

    pub(super) fn release_reservation(&mut self) {
        self.reserved = 0;
    }
}

pub(crate) struct RunResult {
    pub(crate) linear_transports: Vec<serde_json::Value>,
    pub(crate) prompt_token_ids: Vec<i32>,
    pub(crate) generated_token_ids: Vec<i32>,
    pub(crate) decoded_text: String,
    pub(crate) stop_reason: String,
    pub(crate) operation_applications: Vec<OperationApplication>,
    pub(crate) live_readouts: Vec<LiveReadout>,
    pub(crate) native_hyper_captures: Vec<NativeHyperCapture>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunExecutionBinding {
    pub(crate) deployed_model_content_blake3: String,
    pub(crate) content_identity_outcome: String,
    pub(crate) weight_bytes_hashed: u64,
    pub(crate) published_lenses: Vec<RunPublishedLensBinding>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunPublishedLensBinding {
    pub(crate) lens_id: String,
    pub(crate) manifest: PathBuf,
    pub(crate) manifest_canonical_json_blake3: String,
    pub(crate) profile: String,
    pub(crate) method: String,
    pub(crate) target_layer: u32,
    pub(crate) fitted_checkpoint: String,
    pub(crate) fitted_checkpoint_revision: String,
    pub(crate) source_repository: String,
    pub(crate) source_revision: String,
    pub(crate) source_sha256: String,
    pub(crate) payload_blake3: String,
    pub(crate) claims_basis: String,
    pub(crate) transfer_validation_status: String,
    pub(crate) selected_token_ids: Vec<u32>,
    pub(crate) selected_matrices: Vec<RunPublishedMatrixBinding>,
}

#[derive(Debug, Serialize)]
pub(crate) struct RunPublishedMatrixBinding {
    pub(crate) source_layer: u32,
    pub(crate) blake3: String,
}

pub(crate) fn emit_run_output(
    args: &LensRunArgs,
    runtime_kind: &'static str,
    plan_path: &Path,
    bound_plan: BoundLensPlan,
    prepared_input: &PreparedLensInput,
    result: RunResult,
    execution: RunExecution,
    execution_binding: Option<RunExecutionBinding>,
    output_path: Option<&Path>,
) -> Result<()> {
    let artifact = build_run_output(
        &args.model,
        run_sampler(args),
        args.max_new_tokens,
        runtime_kind,
        plan_path,
        bound_plan,
        prepared_input,
        result,
        execution,
        execution_binding,
    );
    let stdout_format = effective_run_stdout_format(args.format, output_path.is_some());
    let bytes = if output_path.is_some() || stdout_format == RunStdoutFormat::Json {
        Some(serialize_run_output(&artifact)?)
    } else {
        None
    };
    if let (Some(path), Some(bytes)) = (output_path, &bytes) {
        crate::write_atomic_replace(path, bytes)?;
    }
    match stdout_format {
        RunStdoutFormat::Summary => print_run_summary(&artifact, output_path),
        RunStdoutFormat::Json => {
            let stdout = std::io::stdout();
            let mut stdout = stdout.lock();
            stdout
                .write_all(bytes.as_deref().expect("JSON output was serialized"))
                .context("write Lens run JSON")?;
            stdout.write_all(b"\n").context("finish Lens run JSON")?;
        }
    }
    Ok(())
}

pub(super) fn build_run_output(
    model_path: &Path,
    sampler: RunSampler,
    max_new_tokens: usize,
    runtime_kind: &'static str,
    plan_path: &Path,
    bound_plan: BoundLensPlan,
    prepared_input: &PreparedLensInput,
    result: RunResult,
    execution: RunExecution,
    execution_binding: Option<RunExecutionBinding>,
) -> RunOutput {
    let requested_live_readouts = bound_plan.resolved.readouts.clone();
    RunOutput {
        linear_transports: result.linear_transports,
        schema: RUN_SCHEMA,
        schema_version: RUN_SCHEMA_VERSION,
        runtime_kind,
        model_path: model_path.to_path_buf(),
        canonical_plan_path: plan_path.to_path_buf(),
        authored_plan: bound_plan.authored,
        authored_plan_canonical_json_blake3: bound_plan.authored_plan_canonical_json_blake3,
        requested_live_readouts,
        plan: bound_plan.resolved,
        position_bindings: bound_plan.position_bindings,
        input_source: prepared_input.source,
        add_special_tokens: prepared_input.add_special_tokens,
        rendering: prepared_input.rendering.clone(),
        prompt_token_ids: result.prompt_token_ids,
        generated_token_ids: result.generated_token_ids,
        sampler,
        max_new_tokens,
        decoded_text: result.decoded_text,
        stop_reason: result.stop_reason,
        execution,
        operation_applications: result.operation_applications,
        live_readouts: result.live_readouts,
        native_hyper_captures: result.native_hyper_captures,
        execution_binding,
    }
}

pub(super) fn validate_run_output(artifact: &RunOutput) -> Result<()> {
    if artifact.execution.schedule_basis() == RunExecutionScheduleBasis::EffectivePlan {
        artifact.execution.validate_against_plan(
            artifact.runtime_kind,
            &artifact.plan,
            artifact.prompt_token_ids.len(),
        )?;
    } else {
        artifact
            .execution
            .validate(artifact.runtime_kind, artifact.prompt_token_ids.len())?;
    }
    Ok(())
}

pub(super) fn serialize_run_output(artifact: &RunOutput) -> Result<Vec<u8>> {
    validate_run_output(artifact)?;
    let bytes = serde_json::to_vec(artifact).context("serialize Lens run artifact")?;
    ensure!(
        bytes.len() <= MAX_RUN_ARTIFACT_BYTES,
        "serialized Lens run artifact is {} bytes; limit is {MAX_RUN_ARTIFACT_BYTES}",
        bytes.len()
    );
    Ok(bytes)
}

pub(super) fn write_new_run_output(
    path: &Path,
    artifact: &RunOutput,
    max_bytes: usize,
) -> Result<usize> {
    validate_run_output(artifact)?;
    ensure!(
        max_bytes > 0 && max_bytes <= MAX_RUN_ARTIFACT_BYTES,
        "run artifact writer requires a positive limit no larger than {MAX_RUN_ARTIFACT_BYTES}"
    );
    let serialized: Result<usize> = (|| {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_NOFOLLOW)
            .open(path)
            .with_context(|| format!("create run artifact {}", path.display()))?;
        let written = {
            let mut buffered = std::io::BufWriter::new(&mut file);
            let mut bounded = crate::ByteLimitedWriter::new(&mut buffered, max_bytes);
            serde_json::to_writer(&mut bounded, artifact)
                .with_context(|| format!("serialize run artifact {}", path.display()))?;
            bounded
                .flush()
                .with_context(|| format!("flush run artifact {}", path.display()))?;
            bounded.written()
        };
        file.sync_all()
            .with_context(|| format!("sync run artifact {}", path.display()))?;
        Ok(written)
    })();
    if serialized.is_err() {
        let _ = std::fs::remove_file(path);
    }
    serialized
}

pub(super) fn print_run_summary(artifact: &RunOutput, output_path: Option<&Path>) {
    print!("{}", run_summary(artifact, output_path));
}

pub(super) fn run_summary(artifact: &RunOutput, output_path: Option<&Path>) -> String {
    let serial_reason = artifact
        .execution
        .serial_reason()
        .map_or("none", RunSerialReason::as_str);
    let mut summary = format!(
        "runtime={} model={}\nprefill_execution={} serial_reason={}\ngenerated_text={}\nstop_reason={}\noperation_applications={} live_readouts={}\n",
        artifact.runtime_kind,
        artifact.model_path.display(),
        artifact.execution.effective_prefill().as_str(),
        serial_reason,
        serde_json::to_string(&artifact.decoded_text).expect("string serialization cannot fail"),
        artifact.stop_reason,
        artifact.operation_applications.len(),
        artifact.live_readouts.len()
    );
    if let Some(path) = output_path {
        summary.push_str(&format!("artifact={}\n", path.display()));
    }
    summary
}

pub(super) fn ensure_new_bundle_output(path: &Path) -> Result<()> {
    match std::fs::symlink_metadata(path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Ok(_) => bail!("bundle output {} already exists", path.display()),
        Err(error) => {
            Err(error).with_context(|| format!("inspect bundle output {}", path.display()))
        }
    }
}
