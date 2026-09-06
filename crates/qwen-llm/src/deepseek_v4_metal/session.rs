//! DeepSeekV4Session: lifecycle, prefill, decode, snapshots.

use super::*;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum DeepSeekV4SessionPhase {
    ReadyWithoutObservation { next_position: u32 },
    ReadyWithObservation { next_position: u32 },
    Poisoned { next_position: u32 },
}

impl DeepSeekV4SessionPhase {
    pub(super) fn fresh() -> Self {
        Self::ReadyWithoutObservation { next_position: 0 }
    }

    pub(super) fn next_position(self) -> u32 {
        match self {
            Self::ReadyWithoutObservation { next_position }
            | Self::ReadyWithObservation { next_position }
            | Self::Poisoned { next_position } => next_position,
        }
    }

    pub(super) fn ready_position(self) -> Result<u32, DeepSeekV4MetalError> {
        match self {
            Self::ReadyWithoutObservation { next_position }
            | Self::ReadyWithObservation { next_position } => Ok(next_position),
            Self::Poisoned { .. } => {
                invalid("DeepSeek V4 session is poisoned by an incomplete token")
            }
        }
    }

    pub(super) fn observation_valid(self) -> bool {
        matches!(self, Self::ReadyWithObservation { .. })
    }

    pub(super) fn begin_mutation(&mut self) -> Result<u32, DeepSeekV4MetalError> {
        let next_position = self.ready_position()?;
        *self = Self::Poisoned { next_position };
        Ok(next_position)
    }

    pub(super) fn complete_mutation(
        &mut self,
        start_position: u32,
        next_position: u32,
        publish_observation: bool,
    ) -> Result<(), DeepSeekV4MetalError> {
        if next_position <= start_position {
            return invalid("DeepSeek V4 mutation did not advance the session position");
        }
        match *self {
            Self::Poisoned {
                next_position: poisoned_position,
            } if poisoned_position == start_position => {
                *self = if publish_observation {
                    Self::ReadyWithObservation { next_position }
                } else {
                    Self::ReadyWithoutObservation { next_position }
                };
                Ok(())
            }
            _ => invalid("DeepSeek V4 mutation completed from an invalid session phase"),
        }
    }

    pub(super) fn begin_restore(&mut self) -> Result<u32, DeepSeekV4MetalError> {
        let next_position = self.ready_position()?;
        *self = Self::Poisoned { next_position };
        Ok(next_position)
    }

    pub(super) fn complete_restore(
        &mut self,
        replaced_position: u32,
        restored_position: u32,
    ) -> Result<(), DeepSeekV4MetalError> {
        match *self {
            Self::Poisoned { next_position } if next_position == replaced_position => {
                *self = Self::ReadyWithoutObservation {
                    next_position: restored_position,
                };
                Ok(())
            }
            _ => invalid("DeepSeek V4 restore completed from an invalid session phase"),
        }
    }
}

pub struct DeepSeekV4Session {
    pub(super) residency: Arc<DeepSeekV4MetalResidency>,
    pub(super) capacity: DeepSeekV4SessionCapacity,
    pub(super) token_id: MetalTensor,
    pub(super) embedding: MetalTensor,
    pub(super) residual_primary: MetalTensor,
    pub(super) residual_secondary: MetalTensor,
    pub(super) hyper_connection: DeepSeekV4HyperConnectionScratch,
    pub(super) attention: DeepSeekV4PositionZeroAttentionScratch,
    pub(super) sparse_csa: DeepSeekV4SparseCsaScratch,
    pub(super) layer_selections: DeepSeekV4LayerSelectionRecords,
    pub(super) raw_cache: MetalTensor,
    pub(super) compressor_frontiers: DeepSeekV4CompressorFrontiers,
    pub(super) moe: DeepSeekV4MoeScratch,
    pub(super) layer_routes: DeepSeekV4LayerRouteRecords,
    pub(super) final_hidden: MetalTensor,
    pub(super) final_normalized_hidden: MetalTensor,
    pub(super) logits: MetalTensor,
    pub(super) prefill: prefill::DeepSeekV4PrefillScratch,
    pub(super) phase: DeepSeekV4SessionPhase,
    pub(super) committed_tokens: Vec<u32>,
    pub(super) snapshot_model_content_id: Option<DeepSeekV4ModelContentId>,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) decision_diagnostics: diagnostics::DeepSeekV4DecisionCapture,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_shadow: DeepSeekV4Fp4ShadowScratch,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_collapsed_selections: DeepSeekV4LayerFp4SelectionRecords,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_shadow_diagnostics: diagnostics::DeepSeekV4Fp4ShadowCapture,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_selection_mode: DeepSeekV4Fp4SessionMode,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_counterfactual_trace: diagnostics::DeepSeekV4Fp4CounterfactualTrace,
    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fp4_score_dispatch_ledger: Option<DeepSeekV4Fp4ScoreDispatchLedger>,
}

pub struct DeepSeekV4SessionConstructionFailure {
    pub(super) residency: Arc<DeepSeekV4MetalResidency>,
    pub(super) error: DeepSeekV4MetalError,
}

impl DeepSeekV4SessionConstructionFailure {
    pub fn into_parts(self) -> (DeepSeekV4MetalResidency, DeepSeekV4MetalError) {
        let Self { residency, error } = self;
        let residency = Arc::try_unwrap(residency).unwrap_or_else(|_| {
            unreachable!("failed DeepSeek V4 session construction retained shared residency")
        });
        (residency, error)
    }

    pub(super) fn into_error(self) -> DeepSeekV4MetalError {
        self.error
    }
}

impl DeepSeekV4Session {
    pub fn new(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_inner(ctx, Arc::new(residency), None)
    }

    /// Construct one sequence-private session over shared immutable weights.
    /// Mutable cache, routing, scratch, logits, and transcript state remain
    /// owned by the returned session.
    #[doc(hidden)]
    pub fn new_shared(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_shared_inner(ctx, residency, None)
    }

    /// Construct a sequence-private shared session whose snapshots are scoped
    /// by a caller-provided identity. Transient in-process users may bind an
    /// ephemeral identity; durable users must bind the full model-content ID.
    #[doc(hidden)]
    pub fn new_shared_with_model_content_id(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
        model_content_id: DeepSeekV4ModelContentId,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_shared_inner(ctx, residency, Some(model_content_id))
    }

    pub(super) fn new_shared_inner(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
        model_content_id: Option<DeepSeekV4ModelContentId>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        if residency._residency_set.is_some() {
            return invalid(
                "shared DeepSeek V4 sessions require QWEN_DSV4_RESIDENCY_SET=0 because residency sets are command-queue scoped",
            );
        }
        Self::new_inner(ctx, residency, model_content_id)
    }

    pub fn new_with_model_content_id(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
    ) -> Result<Self, DeepSeekV4MetalError> {
        Self::new_with_model_content_id_recoverable(ctx, residency, model_content_id)
            .map_err(DeepSeekV4SessionConstructionFailure::into_error)
    }

    /// Construct an exclusively-owned session without losing residency when
    /// session scratch allocation or validation fails.
    pub fn new_with_model_content_id_recoverable(
        ctx: &MetalContext,
        residency: DeepSeekV4MetalResidency,
        model_content_id: DeepSeekV4ModelContentId,
    ) -> Result<Self, DeepSeekV4SessionConstructionFailure> {
        let residency = Arc::new(residency);
        match Self::new_inner(ctx, Arc::clone(&residency), Some(model_content_id)) {
            Ok(session) => {
                drop(residency);
                Ok(session)
            }
            Err(error) => Err(DeepSeekV4SessionConstructionFailure { residency, error }),
        }
    }

    pub(super) fn new_inner(
        ctx: &MetalContext,
        residency: Arc<DeepSeekV4MetalResidency>,
        snapshot_model_content_id: Option<DeepSeekV4ModelContentId>,
    ) -> Result<Self, DeepSeekV4MetalError> {
        residency.validate_context(ctx)?;
        validate_session_config(residency.config())?;
        for name in session_required_tensor_names(residency.config()) {
            residency.require_tensor(&name)?;
        }
        validate_session_lookup_dtypes(
            residency.require_tensor("token_embd.weight")?.dtype,
            residency.require_tensor("output.weight")?.dtype,
        )?;

        let token_id =
            MetalTensor::from_bytes(ctx, bytemuck::bytes_of(&0_i32), vec![1], GgmlType::I32)?;
        let residual_shape = vec![
            DEEPSEEK_V4_HIDDEN_SIZE as u64,
            DEEPSEEK_V4_CONNECTION_COUNT as u64,
        ];
        let attention_config = deepseek_v4_session_attention_config();
        let moe_config = deepseek_v4_session_moe_config(residency.config());
        let capacity = residency.session_capacity();
        let compressor_frontiers =
            DeepSeekV4CompressorFrontiers::new(ctx, residency.config(), capacity)?;
        let mut committed_tokens = Vec::new();
        committed_tokens
            .try_reserve_exact(capacity.forward_limit())
            .map_err(|error| {
                DeepSeekV4MetalError::Invalid(format!(
                    "reserve DeepSeek V4 committed-token transcript: {error}"
                ))
            })?;

        Ok(Self {
            residency,
            capacity,
            token_id,
            embedding: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64])?,
            residual_primary: MetalTensor::zeros_f32(ctx, residual_shape.clone())?,
            residual_secondary: MetalTensor::zeros_f32(ctx, residual_shape)?,
            hyper_connection: DeepSeekV4HyperConnectionScratch::new(ctx, DEEPSEEK_V4_HIDDEN_SIZE)?,
            attention: DeepSeekV4PositionZeroAttentionScratch::new(ctx, attention_config)?,
            sparse_csa: DeepSeekV4SparseCsaScratch::new(ctx, capacity.csa_physical_rows())?,
            layer_selections: DeepSeekV4LayerSelectionRecords::new(ctx)?,
            raw_cache: MetalTensor::zeros_f16(
                ctx,
                vec![
                    attention_config.head_dim as u64,
                    DEEPSEEK_V4_LOCAL_WINDOW as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
            compressor_frontiers,
            moe: DeepSeekV4MoeScratch::new(ctx, moe_config)?,
            layer_routes: DeepSeekV4LayerRouteRecords::new(ctx, moe_config)?,
            final_hidden: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_HIDDEN_SIZE as u64])?,
            final_normalized_hidden: MetalTensor::zeros_f32(
                ctx,
                vec![DEEPSEEK_V4_HIDDEN_SIZE as u64],
            )?,
            logits: MetalTensor::zeros_f32(ctx, vec![DEEPSEEK_V4_VOCAB_SIZE as u64])?,
            prefill: prefill::DeepSeekV4PrefillScratch::new(
                ctx,
                capacity.csa_physical_rows(),
                moe_config.expert_count,
            )?,
            phase: DeepSeekV4SessionPhase::fresh(),
            committed_tokens,
            snapshot_model_content_id,
            #[cfg(feature = "dsv4-diagnostics")]
            decision_diagnostics: diagnostics::DeepSeekV4DecisionCapture::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_shadow: DeepSeekV4Fp4ShadowScratch::new(ctx, capacity.csa_physical_rows())?,
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_collapsed_selections: DeepSeekV4LayerFp4SelectionRecords::new(ctx)?,
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_shadow_diagnostics: diagnostics::DeepSeekV4Fp4ShadowCapture::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_selection_mode: DeepSeekV4Fp4SessionMode::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_counterfactual_trace: diagnostics::DeepSeekV4Fp4CounterfactualTrace::default(),
            #[cfg(feature = "dsv4-diagnostics")]
            fp4_score_dispatch_ledger: None,
        })
    }

    pub fn residency(&self) -> &DeepSeekV4MetalResidency {
        self.residency.as_ref()
    }

    /// Consumes the session and returns its retained weight residency.
    ///
    /// This is the multi-request lifecycle primitive: sessions are rebuilt
    /// per request from one long-lived residency rather than reset in place.
    /// It is deliberately valid from **any** phase, including poisoned:
    /// residency tensors are immutable weight views that no session mutation
    /// path can touch, so recovering the residency from a failed session and
    /// constructing a fresh session is the sanctioned poison-recovery story.
    /// All session-owned scratch, cache, and transcript state is dropped.
    pub fn into_residency(self) -> Result<DeepSeekV4MetalResidency, DeepSeekV4MetalError> {
        match Arc::try_unwrap(self.residency) {
            Ok(residency) => Ok(residency),
            Err(_) => invalid(
                "cannot recover exclusive DeepSeek V4 residency while another shared owner exists",
            ),
        }
    }

    /// Consume a session while retaining shared immutable model ownership.
    #[doc(hidden)]
    pub fn into_shared_residency(self) -> Arc<DeepSeekV4MetalResidency> {
        self.residency
    }

    pub fn logits(&self) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        if !self.phase.observation_valid() {
            return invalid("DeepSeek V4 logits have not completed");
        }
        Ok(&self.logits)
    }

    pub fn final_normalized_hidden(&self) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        if !self.phase.observation_valid() {
            return invalid("DeepSeek V4 final normalized hidden state has not completed");
        }
        Ok(&self.final_normalized_hidden)
    }

    pub fn next_position(&self) -> u32 {
        self.phase.next_position()
    }

    pub fn committed_tokens(&self) -> &[u32] {
        &self.committed_tokens
    }

    pub fn capacity(&self) -> DeepSeekV4SessionCapacity {
        self.capacity
    }

    pub(crate) fn bound_model_content_id(&self) -> Option<DeepSeekV4ModelContentId> {
        self.snapshot_model_content_id
    }

    #[doc(hidden)]
    pub fn multigroup_selector_telemetry(&self) -> DeepSeekV4MultigroupSelectorTelemetry {
        self.sparse_csa.multigroup_selector_telemetry()
    }

    /// Explicitly seals the exact multi-group sparse selector for its measured
    /// far-context crossover band. Production already selects that band on the
    /// qualified device; this retains the diagnostics policy surface.
    #[doc(hidden)]
    pub fn enable_multigroup_selector_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let position = self.phase.ready_position()?;
        if position != 0 {
            return invalid(format!(
                "DeepSeek V4 multi-group selector must be enabled at position zero, got {position}"
            ));
        }
        self.sparse_csa.enable_multigroup_selector_experiment()
    }

    #[doc(hidden)]
    pub fn disable_multigroup_selector(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let position = self.phase.ready_position()?;
        if position != 0 {
            return invalid(format!(
                "DeepSeek V4 multi-group selector must be disabled at position zero, got {position}"
            ));
        }
        self.sparse_csa.disable_multigroup_selector()
    }

    /// Enables exact post-Hadamard FP4 lineage capture for a diagnostics-only
    /// observer. It must be armed before any token mutates the session so every
    /// subsequently visible row has one unambiguous pre-F16 source.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn enable_fp4_shadow_lineage(&mut self) -> Result<(), DeepSeekV4MetalError> {
        let position = self.phase.ready_position()?;
        if position != 0 {
            return invalid(format!(
                "DeepSeek V4 FP4 shadow lineage must be enabled at position zero, got {position}"
            ));
        }
        self.compressor_frontiers.enable_fp4_shadow_lineage()?;
        self.fp4_shadow_diagnostics.enable_lineage();
        Ok(())
    }

    /// Enables a diagnostics-only counterfactual that consumes FP4 selector
    /// IDs while retaining the authoritative F16 attention cache. Such a
    /// session is not compatible with snapshot-v1 export or restore.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn enable_fp4_shadow_selection_counterfactual(
        &mut self,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.fp4_selection_mode != DeepSeekV4Fp4SessionMode::F16Authoritative {
            return invalid("DeepSeek V4 FP4 score plan is already sealed");
        }
        self.enable_fp4_shadow_lineage()?;
        self.fp4_selection_mode = DeepSeekV4Fp4SessionMode::PairedCounterfactual;
        Ok(())
    }

    /// Enables lineage and seals a diagnostics-only FP4-only score plan at
    /// position zero. Use [`Self::seal_fp4_no_double_score_experiment`] after
    /// an ordinary lineage-preserving prefix instead.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn enable_fp4_no_double_score_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        self.enable_fp4_shadow_lineage()?;
        self.seal_fp4_no_double_score_experiment()
    }

    /// Seals a diagnostics-only FP4-only score plan for all later sparse
    /// positions. Lineage must already have been enabled at position zero.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn seal_fp4_no_double_score_experiment(&mut self) -> Result<(), DeepSeekV4MetalError> {
        self.phase.ready_position()?;
        if self.fp4_selection_mode != DeepSeekV4Fp4SessionMode::F16Authoritative {
            return invalid("DeepSeek V4 FP4 score plan is already sealed");
        }
        if !self.fp4_shadow_diagnostics.lineage_enabled() {
            return invalid(
                "DeepSeek V4 FP4 lineage must be enabled before sealing the no-double-score experiment",
            );
        }
        self.decision_diagnostics
            .ensure_no_active_capture("seal the FP4 score plan")?;
        self.fp4_shadow_diagnostics
            .ensure_no_active_capture("seal the FP4 score plan")?;
        self.fp4_selection_mode = DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental;
        Ok(())
    }

    /// Arms one diagnostics-only packed or singleton FP4 comparison. Packed
    /// capture intentionally admits only a final single sparse query.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn arm_fp4_shadow_report(&mut self, position: u32) -> Result<(), DeepSeekV4MetalError> {
        self.capacity.validate_position(position)?;
        let current = self.phase.ready_position()?;
        self.fp4_shadow_diagnostics.arm(current, position)?;
        Ok(())
    }

    /// Atomically arms a singleton paired-score audit while an FP4-only
    /// experiment remains the consumed selection authority.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn arm_fp4_paired_singleton_audit(
        &mut self,
        position: u32,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.fp4_selection_mode != DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental {
            return invalid("paired FP4 audit requires a sealed FP4-only experiment");
        }
        self.capacity.validate_position(position)?;
        let current = self.phase.ready_position()?;
        if position != current {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: current,
                actual: position,
            }
            .into());
        }
        self.decision_diagnostics.validate_arm(position)?;
        self.fp4_shadow_diagnostics
            .validate_arm(current, position)?;
        self.decision_diagnostics.arm(position)?;
        self.fp4_shadow_diagnostics.arm(current, position)?;
        Ok(())
    }

    /// Takes a completed owned FP4 observer report and returns the capture to
    /// idle so a subsequent position can be armed.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_fp4_shadow_report(
        &mut self,
    ) -> Result<DeepSeekV4Fp4ShadowReport, DeepSeekV4MetalError> {
        Ok(self.fp4_shadow_diagnostics.take()?)
    }

    /// Takes the last successfully encoded singleton or packed CSA score
    /// schedule. The ledger records operation invocations, not GPU duration.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_fp4_score_dispatch_ledger(
        &mut self,
    ) -> Result<DeepSeekV4Fp4ScoreDispatchLedger, DeepSeekV4MetalError> {
        self.fp4_score_dispatch_ledger.take().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "DeepSeek V4 FP4 score dispatch ledger is unavailable".into(),
            )
        })
    }

    /// Arms the feature-gated, single-use decision capture for the next sparse
    /// CSA token.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn arm_decision_transcript(&mut self, position: u32) -> Result<(), DeepSeekV4MetalError> {
        if self.fp4_selection_mode == DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental {
            return invalid(
                "FP4-only sessions must arm decisions through arm_fp4_paired_singleton_audit",
            );
        }
        self.capacity.validate_position(position)?;
        if position != self.phase.ready_position()? {
            return Err(DeepSeekV4DiagnosticsError::WrongPosition {
                expected: position,
                actual: self.phase.next_position(),
            }
            .into());
        }
        self.decision_diagnostics.arm(position)?;
        Ok(())
    }

    /// Takes a complete owned transcript. Partial or duplicate takes fail closed.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_decision_transcript(
        &mut self,
    ) -> Result<DeepSeekV4DecisionTranscript, DeepSeekV4MetalError> {
        Ok(self.decision_diagnostics.take()?)
    }

    /// Takes a complete transcript and returns the diagnostics capture to idle
    /// for the next consecutive-token observation window.
    #[cfg(feature = "dsv4-diagnostics")]
    pub fn take_decision_transcript_and_reset(
        &mut self,
    ) -> Result<DeepSeekV4DecisionTranscript, DeepSeekV4MetalError> {
        Ok(self.decision_diagnostics.take_and_reset()?)
    }

    pub fn cache_contract(&self) -> DeepSeekV4AttentionCacheContract {
        DeepSeekV4AttentionCacheContract::LlamaCppB10222F16
    }

    pub fn indexer_contract(&self) -> DeepSeekV4IndexerContract {
        DeepSeekV4IndexerContract::LlamaCppB10222F16HadamardV1
    }

    /// Copy completed logits out of shared Metal storage.
    pub fn copy_logits_f32(&self) -> Result<Vec<f32>, DeepSeekV4MetalError> {
        host_read_f32(self.logits()?, "completed DeepSeek V4 logits")
    }

    /// Copy the completed normalized hidden state out of shared Metal storage.
    #[cfg(feature = "dsv4-diagnostics")]
    #[doc(hidden)]
    pub fn copy_final_normalized_hidden_f32(&self) -> Result<Vec<f32>, DeepSeekV4MetalError> {
        host_read_f32(
            self.final_normalized_hidden()?,
            "completed DeepSeek V4 final normalized hidden",
        )
    }

    /// Compatibility entry point for the original position-zero differential.
    pub fn forward_token_zero(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.forward_token_zero_with_progress(ctx, token_id, |_| {})
    }

    pub fn forward_token_zero_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        let next_position = self.phase.next_position();
        if next_position != 0 {
            return invalid(format!(
                "position-zero entry point requires a fresh session, next position is {}",
                next_position
            ));
        }
        self.forward_token_with_progress(ctx, token_id, layer_completed)
    }

    pub fn forward_token(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.forward_token_with_progress(ctx, token_id, |_| {})
    }

    /// Execute one complete token and retain the raw cache and every compressor
    /// frontier needed by the next position. Ordinary decode encodes every
    /// layer into one ordered serial Metal pass, then validates immutable
    /// per-layer route and sparse-selection records before publishing progress.
    pub fn forward_token_with_progress(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        layer_completed: impl FnMut(usize),
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            layer_completed,
            None,
            None,
            None,
        )
    }

    /// Execute one singleton token while timing the retained, separately
    /// encoded one-command-per-layer comparator. The ordinary whole-token path
    /// does not allocate or sample clocks.
    #[doc(hidden)]
    pub fn forward_token_profiled(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<DeepSeekV4CommandProfile, DeepSeekV4MetalError> {
        let position = self.phase.next_position();
        let started = std::time::Instant::now();
        let mut layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            |_| {},
            Some(&mut layers),
            None,
            None,
        )?;
        debug_assert_eq!(layers.len(), DEEPSEEK_V4_LAYER_COUNT);
        Ok(DeepSeekV4CommandProfile {
            position,
            forward_wall_ms: started.elapsed().as_secs_f64() * 1e3,
            layers,
        })
    }

    /// Time the ordinary one-command, one-encoder path without adding GPU
    /// samples, command buffers, encoders, or dispatches.
    #[doc(hidden)]
    pub fn forward_token_whole_profiled(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
    ) -> Result<DeepSeekV4WholeTokenProfile, DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.decision_diagnostics
                .ensure_no_active_capture("profile a whole token")?;
            self.fp4_shadow_diagnostics
                .ensure_no_active_capture("profile a whole token")?;
        }
        let mut profile = DeepSeekV4WholeTokenProfile::default();
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            |_| {},
            None,
            None,
            Some(&mut profile),
        )?;
        Ok(profile)
    }

    /// Execute one singleton token while sampling ten encoder-delimited stages
    /// in only the requested layers. This intentionally retains the historical
    /// per-layer command schedule as an attribution comparator.
    #[doc(hidden)]
    pub fn forward_token_stage_profiled(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        sampled_layers: &[usize],
    ) -> Result<DeepSeekV4StageProfile, DeepSeekV4MetalError> {
        let mut recorder = DeepSeekV4StageRecorder::new(ctx, sampled_layers)?;
        let position = self.phase.next_position();
        let started = std::time::Instant::now();
        let mut layers = Vec::with_capacity(DEEPSEEK_V4_LAYER_COUNT);
        self.forward_token_with_progress_and_profile(
            ctx,
            token_id,
            |_| {},
            Some(&mut layers),
            Some(&mut recorder),
            None,
        )?;
        debug_assert_eq!(layers.len(), DEEPSEEK_V4_LAYER_COUNT);
        Ok(DeepSeekV4StageProfile {
            position,
            forward_wall_ms: started.elapsed().as_secs_f64() * 1e3,
            layers,
            sampled_layers: recorder.take_resolved()?,
        })
    }

    pub(super) fn forward_token_with_progress_and_profile(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        mut layer_completed: impl FnMut(usize),
        routing_profile: Option<&mut Vec<DeepSeekV4LayerCommandProfile>>,
        stage_recorder: Option<&mut DeepSeekV4StageRecorder>,
        mut whole_profile: Option<&mut DeepSeekV4WholeTokenProfile>,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        let forward_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let guards_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        self.residency.validate_context(ctx)?;
        let position = self.phase.ready_position()?;
        if token_id as usize >= DEEPSEEK_V4_VOCAB_SIZE {
            return invalid(format!(
                "token id {token_id} is outside vocabulary {DEEPSEEK_V4_VOCAB_SIZE}"
            ));
        }
        self.capacity.validate_position(position)?;
        let next_position = position
            .checked_add(1)
            .ok_or_else(|| DeepSeekV4MetalError::Invalid("position overflow".into()))?;
        self.validate_committed_token_append(position, 1)?;

        let begun_position = self.phase.begin_mutation()?;
        debug_assert_eq!(begun_position, position);
        host_write_i32(&self.token_id, &[token_id as i32], "DeepSeek V4 token ID")?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics.begin_forward(position)?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.fp4_shadow_diagnostics.begin_singleton(position)?;
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.position = position;
            profile.guards_phase_cpu_ms = guards_started
                .expect("whole-token profile requires a guards timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }
        let result = self.forward_token_inner(
            ctx,
            token_id,
            position,
            &mut layer_completed,
            routing_profile,
            stage_recorder,
            whole_profile.as_deref_mut(),
        );
        match result {
            Ok(()) => {
                let causal_commit_started =
                    whole_profile.as_ref().map(|_| std::time::Instant::now());
                self.commit_tokens(&[token_id]);
                self.phase
                    .complete_mutation(position, next_position, true)?;
                if let Some(profile) = whole_profile {
                    profile.causal_commit_cpu_ms = causal_commit_started
                        .expect("whole-token profile requires a causal-commit timer")
                        .elapsed()
                        .as_secs_f64()
                        * 1e3;
                    profile.forward_wall_ms = forward_started
                        .expect("whole-token profile requires a forward timer")
                        .elapsed()
                        .as_secs_f64()
                        * 1e3;
                }
                Ok(&self.logits)
            }
            Err(error) => Err(error),
        }
    }

    pub(super) fn validate_committed_token_append(
        &self,
        start_position: u32,
        token_count: usize,
    ) -> Result<(), DeepSeekV4MetalError> {
        if self.committed_tokens.len() != start_position as usize {
            return invalid(format!(
                "DeepSeek V4 committed-token transcript has length {}, expected {start_position}",
                self.committed_tokens.len()
            ));
        }
        let end = self
            .committed_tokens
            .len()
            .checked_add(token_count)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(
                    "DeepSeek V4 committed-token transcript length overflow".into(),
                )
            })?;
        if end > self.capacity.forward_limit() {
            return invalid(format!(
                "DeepSeek V4 committed-token transcript would reach {end}, beyond capacity {}",
                self.capacity.forward_limit()
            ));
        }
        if self.committed_tokens.capacity() < end {
            return invalid(format!(
                "DeepSeek V4 committed-token transcript capacity {} cannot record {end} tokens",
                self.committed_tokens.capacity()
            ));
        }
        Ok(())
    }

    pub(super) fn commit_tokens(&mut self, tokens: &[u32]) {
        debug_assert!(
            self.committed_tokens.len() + tokens.len() <= self.committed_tokens.capacity()
        );
        self.committed_tokens.extend_from_slice(tokens);
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_token_layer(
        &mut self,
        ctx: &MetalContext,
        layer_encoder: &mut DeepSeekV4LayerEncoder<'_, '_>,
        token_id: u32,
        position: u32,
        layer: usize,
        route_record: &DeepSeekV4RouteRecord,
        selection_record: &DeepSeekV4SelectionRecord,
    ) -> Result<DeepSeekV4EncodedLayer, DeepSeekV4MetalError> {
        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;
        let raw_cache = self.raw_cache_layer(layer)?;
        let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;

        if layer == 0 {
            encode_get_rows_f32(
                ctx,
                layer_encoder.current(),
                self.residency.require_tensor("token_embd.weight")?,
                &self.token_id,
                &self.embedding,
                1,
                DEEPSEEK_V4_HIDDEN_SIZE,
            )?;
            self.hyper_connection.encode_initial_repeat(
                ctx,
                layer_encoder.current(),
                &self.embedding,
                &self.residual_primary,
            )?;
        }

        self.hyper_connection.encode_pre(
            ctx,
            layer_encoder.current(),
            &self.residual_primary,
            self.layer_tensor(layer, "hc_attn_fn.weight")?,
            self.layer_tensor(layer, "hc_attn_scale.weight")?,
            self.layer_tensor(layer, "hc_attn_base.weight")?,
            rms_eps,
            hc_eps,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::AttentionPrepare)?;

        self.attention.encode_prepare_local_f16(
            ctx,
            layer_encoder.current(),
            self.hyper_connection.collapsed_input(),
            self.layer_tensor(layer, "attn_norm.weight")?,
            self.layer_tensor(layer, "attn_q_a.weight")?,
            self.layer_tensor(layer, "attn_q_a_norm.weight")?,
            self.layer_tensor(layer, "attn_q_b.weight")?,
            self.layer_tensor(layer, "attn_kv.weight")?,
            self.layer_tensor(layer, "attn_kv_a_norm.weight")?,
            &raw_cache,
            position,
            rope,
            rms_eps,
        )?;
        self.compressor_frontiers.encode_layer(
            ctx,
            layer_encoder.current(),
            &self.residency,
            layer,
            position,
            self.attention.normalized_input(),
            rope,
            rms_eps,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::AttentionCore)?;

        let csa_rows = self.compressor_frontiers.csa_rows(layer, position)?;
        let sparse_visible_count =
            if let Some(rows) = csa_rows.filter(|rows| rows.count > DEEPSEEK_V4_CSA_TOP_K) {
                #[cfg(not(feature = "dsv4-diagnostics"))]
                self.sparse_csa.encode(
                    ctx,
                    layer_encoder.current(),
                    self.attention.q_lora(),
                    self.attention.normalized_input(),
                    self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                    self.layer_tensor(layer, "indexer.proj.weight")?,
                    rows,
                    position,
                    rope,
                    selection_record,
                )?;
                #[cfg(feature = "dsv4-diagnostics")]
                let collapsed_fp4 =
                    self.fp4_selection_mode == DeepSeekV4Fp4SessionMode::Fp4OnlyExperimental;
                #[cfg(feature = "dsv4-diagnostics")]
                if collapsed_fp4 {
                    self.sparse_csa.encode_prepare(
                        ctx,
                        layer_encoder.current(),
                        self.attention.q_lora(),
                        self.attention.normalized_input(),
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        position,
                        rope,
                        selection_record,
                    )?;
                    let fp4_record = self.fp4_collapsed_selections.layer(layer)?;
                    self.fp4_shadow.encode_into(
                        ctx,
                        layer_encoder.current(),
                        &self.sparse_csa.index_queries,
                        &self.sparse_csa.head_weights,
                        rows,
                        &selection_record.visible_count,
                        fp4_record.output(selection_record),
                    )?;
                    self.attention.encode_selected_attention_f16(
                        ctx,
                        layer_encoder.current(),
                        &raw_cache,
                        rows,
                        fp4_record.selection_view(selection_record),
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        position,
                        rope,
                    )?;
                } else {
                    self.sparse_csa.encode(
                        ctx,
                        layer_encoder.current(),
                        self.attention.q_lora(),
                        self.attention.normalized_input(),
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        position,
                        rope,
                        selection_record,
                    )?;
                    self.attention.encode_selected_attention_f16(
                        ctx,
                        layer_encoder.current(),
                        &raw_cache,
                        rows,
                        self.sparse_csa.selection_view(selection_record),
                        self.layer_tensor(layer, "attn_sinks.weight")?,
                        position,
                        rope,
                    )?;
                }
                #[cfg(not(feature = "dsv4-diagnostics"))]
                self.attention.encode_selected_attention_f16(
                    ctx,
                    layer_encoder.current(),
                    &raw_cache,
                    rows,
                    self.sparse_csa.selection_view(selection_record),
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
                Some(rows.count)
            } else {
                let compressed = self.compressor_frontiers.attention_rows(layer, position)?;
                self.attention.encode_dense_attention_f16(
                    ctx,
                    layer_encoder.current(),
                    &raw_cache,
                    compressed,
                    self.residency.config().attention_kinds[layer],
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
                None
            };
        layer_encoder.boundary(DeepSeekV4StageKind::AttentionOutput)?;

        let attention_output = self.attention.encode_attention_output(
            ctx,
            layer_encoder.current(),
            self.layer_tensor(layer, "attn_output_a.weight")?,
            self.layer_tensor(layer, "attn_output_b.weight")?,
            position,
            rope,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::HyperConnectionBridge)?;

        self.hyper_connection.encode_post(
            ctx,
            layer_encoder.current(),
            attention_output,
            &self.residual_primary,
            &self.residual_secondary,
        )?;
        self.hyper_connection.encode_pre(
            ctx,
            layer_encoder.current(),
            &self.residual_secondary,
            self.layer_tensor(layer, "hc_ffn_fn.weight")?,
            self.layer_tensor(layer, "hc_ffn_scale.weight")?,
            self.layer_tensor(layer, "hc_ffn_base.weight")?,
            rms_eps,
            hc_eps,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeRouter)?;

        self.moe.encode_router(
            ctx,
            layer_encoder.current(),
            self.hyper_connection.collapsed_input(),
            self.layer_tensor(layer, "ffn_norm.weight")?,
            self.layer_tensor(layer, "ffn_gate_inp.weight")?,
            rms_eps,
        )?;
        if layer < self.residency.config().hash_layer_count as usize {
            self.moe.encode_route_hash_gpu_into(
                ctx,
                layer_encoder.current(),
                token_id as usize,
                self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?,
                route_record,
            )?;
        } else {
            self.moe.encode_route_learned_gpu_into(
                ctx,
                layer_encoder.current(),
                self.layer_tensor(layer, "exp_probs_b.bias")?,
                route_record,
            )?;
        }
        self.moe.validate_indexed_experts(
            self.layer_tensor(layer, "ffn_gate_exps.weight")?,
            self.layer_tensor(layer, "ffn_up_exps.weight")?,
            self.layer_tensor(layer, "ffn_down_exps.weight")?,
            self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
            self.layer_tensor(layer, "ffn_up_shexp.weight")?,
            self.layer_tensor(layer, "ffn_down_shexp.weight")?,
            self.residency.config().swiglu_clamp_experts[layer],
            self.residency.config().swiglu_clamp_shared[layer],
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeRoutedExperts)?;

        self.moe.encode_routed_experts_all_slots_from_record(
            ctx,
            layer_encoder.current(),
            self.layer_tensor(layer, "ffn_gate_exps.weight")?,
            self.layer_tensor(layer, "ffn_up_exps.weight")?,
            self.layer_tensor(layer, "ffn_down_exps.weight")?,
            self.residency.config().swiglu_clamp_experts[layer],
            route_record,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeSharedExpert)?;

        self.moe.encode_shared_expert(
            ctx,
            layer_encoder.current(),
            self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
            self.layer_tensor(layer, "ffn_up_shexp.weight")?,
            self.layer_tensor(layer, "ffn_down_shexp.weight")?,
            self.residency.config().swiglu_clamp_shared[layer],
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::MoeCombine)?;

        let moe_output = self.moe.encode_expert_combine_from_record(
            ctx,
            layer_encoder.current(),
            route_record,
        )?;
        layer_encoder.boundary(DeepSeekV4StageKind::LayerTail)?;

        self.hyper_connection.encode_post(
            ctx,
            layer_encoder.current(),
            moe_output,
            &self.residual_secondary,
            &self.residual_primary,
        )?;
        if layer + 1 == DEEPSEEK_V4_LAYER_COUNT {
            self.hyper_connection.encode_head(
                ctx,
                layer_encoder.current(),
                &self.residual_primary,
                self.residency.require_tensor("output_hc_fn.weight")?,
                self.residency.require_tensor("output_hc_scale.weight")?,
                self.residency.require_tensor("output_hc_base.weight")?,
                &self.final_hidden,
                rms_eps,
                hc_eps,
            )?;
            encode_rms_norm_mul_f32(
                ctx,
                layer_encoder.current(),
                &self.final_hidden,
                self.residency.require_tensor("output_norm.weight")?,
                &self.final_normalized_hidden,
                rms_eps,
            )?;
            encode_projection(
                ctx,
                layer_encoder.current(),
                self.residency.require_tensor("output.weight")?,
                &self.final_normalized_hidden,
                &self.logits,
                DEEPSEEK_V4_HIDDEN_SIZE,
                DEEPSEEK_V4_VOCAB_SIZE,
                "output logits",
            )?;
        }
        Ok(DeepSeekV4EncodedLayer {
            sparse_visible_count,
        })
    }

    pub(super) fn forward_token_inner(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        position: u32,
        layer_completed: &mut impl FnMut(usize),
        mut routing_profile: Option<&mut Vec<DeepSeekV4LayerCommandProfile>>,
        mut stage_recorder: Option<&mut DeepSeekV4StageRecorder>,
        whole_profile: Option<&mut DeepSeekV4WholeTokenProfile>,
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        let decision_capture_active = self.decision_diagnostics.is_capturing();
        #[cfg(not(feature = "dsv4-diagnostics"))]
        let decision_capture_active = false;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_shadow_capture_active = self.fp4_shadow_diagnostics.is_capturing();
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_score_plan = self
            .fp4_selection_mode
            .score_plan(fp4_shadow_capture_active);
        #[cfg(feature = "dsv4-diagnostics")]
        if decision_capture_active && !fp4_score_plan.runs_f16() {
            return invalid("decision capture requires an F16 or paired CSA score plan");
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_requires_instrumented_schedule =
            matches!(fp4_score_plan, DeepSeekV4Fp4ScorePlan::Paired { .. });
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_score_dispatch_ledger = DeepSeekV4Fp4ScoreDispatchLedger::new(
            DeepSeekV4Fp4ShadowExecution::Singleton,
            position,
            fp4_score_plan.kind(),
            fp4_score_plan.consumed_source(),
        );
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = None;
        }
        #[cfg(not(feature = "dsv4-diagnostics"))]
        let fp4_requires_instrumented_schedule = false;
        if routing_profile.is_none()
            && stage_recorder.is_none()
            && !decision_capture_active
            && !fp4_requires_instrumented_schedule
        {
            return self.forward_token_inner_collapsed(
                ctx,
                token_id,
                position,
                layer_completed,
                whole_profile,
            );
        }
        if whole_profile.is_some() {
            return invalid(
                "whole-token profiling is incompatible with layer, stage, or decision profiling",
            );
        }

        let rms_eps = self.residency.config().attention_rms_epsilon;
        let hc_eps = self.residency.config().hyper_connection_epsilon;

        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            let selection_record = self.sparse_csa.default_record();
            let route_record = self.moe.default_route_record();
            let raw_cache = self.raw_cache_layer(layer)?;
            let rope = deepseek_v4_layer_rope(self.residency.config(), layer)?;
            let encode_started = routing_profile.as_ref().map(|_| std::time::Instant::now());
            let command = ctx.queue.commandBuffer().ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "failed to allocate layer {layer} command buffer"
                ))
            })?;
            let stage_sampled = stage_recorder
                .as_ref()
                .is_some_and(|recorder| recorder.samples_layer(layer));
            let mut encoder = if stage_sampled {
                stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(
                        &command,
                        layer,
                        DeepSeekV4StageKind::AttentionHyperConnection,
                    )?
            } else {
                KernelEncoder::begin(&command)
            };
            if layer == 0 {
                encode_get_rows_f32(
                    ctx,
                    &encoder,
                    self.residency.require_tensor("token_embd.weight")?,
                    &self.token_id,
                    &self.embedding,
                    1,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                )?;
                self.hyper_connection.encode_initial_repeat(
                    ctx,
                    &encoder,
                    &self.embedding,
                    &self.residual_primary,
                )?;
            }

            self.hyper_connection.encode_pre(
                ctx,
                &encoder,
                &self.residual_primary,
                self.layer_tensor(layer, "hc_attn_fn.weight")?,
                self.layer_tensor(layer, "hc_attn_scale.weight")?,
                self.layer_tensor(layer, "hc_attn_base.weight")?,
                rms_eps,
                hc_eps,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::AttentionPrepare)?;
            }
            self.attention.encode_prepare_local_f16(
                ctx,
                &encoder,
                self.hyper_connection.collapsed_input(),
                self.layer_tensor(layer, "attn_norm.weight")?,
                self.layer_tensor(layer, "attn_q_a.weight")?,
                self.layer_tensor(layer, "attn_q_a_norm.weight")?,
                self.layer_tensor(layer, "attn_q_b.weight")?,
                self.layer_tensor(layer, "attn_kv.weight")?,
                self.layer_tensor(layer, "attn_kv_a_norm.weight")?,
                &raw_cache,
                position,
                rope,
                rms_eps,
            )?;
            self.compressor_frontiers.encode_layer(
                ctx,
                &encoder,
                &self.residency,
                layer,
                position,
                self.attention.normalized_input(),
                rope,
                rms_eps,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::AttentionCore)?;
            }
            let csa_rows = self.compressor_frontiers.csa_rows(layer, position)?;
            if let Some(rows) = csa_rows.filter(|rows| rows.count > DEEPSEEK_V4_CSA_TOP_K) {
                #[cfg(not(feature = "dsv4-diagnostics"))]
                self.sparse_csa.encode(
                    ctx,
                    &encoder,
                    self.attention.q_lora(),
                    self.attention.normalized_input(),
                    self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                    self.layer_tensor(layer, "indexer.proj.weight")?,
                    rows,
                    position,
                    rope,
                    &selection_record,
                )?;
                #[cfg(feature = "dsv4-diagnostics")]
                {
                    self.sparse_csa.encode_prepare(
                        ctx,
                        &encoder,
                        self.attention.q_lora(),
                        self.attention.normalized_input(),
                        self.layer_tensor(layer, "indexer.attn_q_b.weight")?,
                        self.layer_tensor(layer, "indexer.proj.weight")?,
                        rows,
                        position,
                        rope,
                        &selection_record,
                    )?;
                    fp4_score_dispatch_ledger.record_common_prepare()?;
                    if fp4_score_plan.runs_f16() {
                        self.sparse_csa.encode_f16_score_and_select(
                            ctx,
                            &encoder,
                            rows,
                            &selection_record,
                        )?;
                        fp4_score_dispatch_ledger.record_f16_score_and_selector()?;
                    }
                }
                #[cfg(feature = "dsv4-diagnostics")]
                if fp4_score_plan.runs_fp4() {
                    self.fp4_shadow.encode(
                        ctx,
                        &encoder,
                        &self.sparse_csa.index_queries,
                        &self.sparse_csa.head_weights,
                        rows,
                        &selection_record.visible_count,
                    )?;
                    fp4_score_dispatch_ledger.record_fp4_pipeline()?;
                }
                #[cfg(feature = "dsv4-diagnostics")]
                let selected = if fp4_score_plan.consumes_fp4() {
                    self.fp4_shadow.selection_view()
                } else {
                    self.sparse_csa.selection_view(&selection_record)
                };
                #[cfg(not(feature = "dsv4-diagnostics"))]
                let selected = self.sparse_csa.selection_view(&selection_record);
                self.attention.encode_selected_attention_f16(
                    ctx,
                    &encoder,
                    &raw_cache,
                    rows,
                    selected,
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
            } else {
                let compressed = self.compressor_frontiers.attention_rows(layer, position)?;
                self.attention.encode_dense_attention_f16(
                    ctx,
                    &encoder,
                    &raw_cache,
                    compressed,
                    self.residency.config().attention_kinds[layer],
                    self.layer_tensor(layer, "attn_sinks.weight")?,
                    position,
                    rope,
                )?;
            }
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::AttentionOutput)?;
            }
            let attention_output = self.attention.encode_attention_output(
                ctx,
                &encoder,
                self.layer_tensor(layer, "attn_output_a.weight")?,
                self.layer_tensor(layer, "attn_output_b.weight")?,
                position,
                rope,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::HyperConnectionBridge)?;
            }
            self.hyper_connection.encode_post(
                ctx,
                &encoder,
                attention_output,
                &self.residual_primary,
                &self.residual_secondary,
            )?;
            self.hyper_connection.encode_pre(
                ctx,
                &encoder,
                &self.residual_secondary,
                self.layer_tensor(layer, "hc_ffn_fn.weight")?,
                self.layer_tensor(layer, "hc_ffn_scale.weight")?,
                self.layer_tensor(layer, "hc_ffn_base.weight")?,
                rms_eps,
                hc_eps,
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeRouter)?;
            }
            self.moe.encode_router(
                ctx,
                &encoder,
                self.hyper_connection.collapsed_input(),
                self.layer_tensor(layer, "ffn_norm.weight")?,
                self.layer_tensor(layer, "ffn_gate_inp.weight")?,
                rms_eps,
            )?;
            if layer < self.residency.config().hash_layer_count as usize {
                self.moe.encode_route_hash_gpu(
                    ctx,
                    &encoder,
                    token_id as usize,
                    self.layer_tensor(layer, "ffn_gate_tid2eid.weight")?,
                )?;
            } else {
                self.moe.encode_route_learned_gpu(
                    ctx,
                    &encoder,
                    self.layer_tensor(layer, "exp_probs_b.bias")?,
                )?;
            }
            self.moe.validate_indexed_experts(
                self.layer_tensor(layer, "ffn_gate_exps.weight")?,
                self.layer_tensor(layer, "ffn_up_exps.weight")?,
                self.layer_tensor(layer, "ffn_down_exps.weight")?,
                self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                self.layer_tensor(layer, "ffn_down_shexp.weight")?,
                self.residency.config().swiglu_clamp_experts[layer],
                self.residency.config().swiglu_clamp_shared[layer],
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeRoutedExperts)?;
            }
            self.moe.encode_routed_experts_all_slots(
                ctx,
                &encoder,
                self.layer_tensor(layer, "ffn_gate_exps.weight")?,
                self.layer_tensor(layer, "ffn_up_exps.weight")?,
                self.layer_tensor(layer, "ffn_down_exps.weight")?,
                self.residency.config().swiglu_clamp_experts[layer],
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeSharedExpert)?;
            }
            self.moe.encode_shared_expert(
                ctx,
                &encoder,
                self.layer_tensor(layer, "ffn_gate_shexp.weight")?,
                self.layer_tensor(layer, "ffn_up_shexp.weight")?,
                self.layer_tensor(layer, "ffn_down_shexp.weight")?,
                self.residency.config().swiglu_clamp_shared[layer],
            )?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::MoeCombine)?;
            }
            let moe_output = self.moe.encode_expert_combine(ctx, &encoder)?;
            if stage_sampled {
                encoder.end();
                encoder = stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .begin(&command, layer, DeepSeekV4StageKind::LayerTail)?;
            }
            self.hyper_connection.encode_post(
                ctx,
                &encoder,
                moe_output,
                &self.residual_secondary,
                &self.residual_primary,
            )?;

            if layer + 1 == DEEPSEEK_V4_LAYER_COUNT {
                self.hyper_connection.encode_head(
                    ctx,
                    &encoder,
                    &self.residual_primary,
                    self.residency.require_tensor("output_hc_fn.weight")?,
                    self.residency.require_tensor("output_hc_scale.weight")?,
                    self.residency.require_tensor("output_hc_base.weight")?,
                    &self.final_hidden,
                    rms_eps,
                    hc_eps,
                )?;
                encode_rms_norm_mul_f32(
                    ctx,
                    &encoder,
                    &self.final_hidden,
                    self.residency.require_tensor("output_norm.weight")?,
                    &self.final_normalized_hidden,
                    rms_eps,
                )?;
                encode_projection(
                    ctx,
                    &encoder,
                    self.residency.require_tensor("output.weight")?,
                    &self.final_normalized_hidden,
                    &self.logits,
                    DEEPSEEK_V4_HIDDEN_SIZE,
                    DEEPSEEK_V4_VOCAB_SIZE,
                    "output logits",
                )?;
            }
            encoder.end();
            let encode_cpu_ms = encode_started
                .map(|started| started.elapsed().as_secs_f64() * 1e3)
                .unwrap_or_default();
            command.commit();
            command.waitUntilCompleted();
            if let Some(error) = command.error() {
                return invalid(format!("layer {layer} command failed: {error:?}"));
            }
            self.moe
                .validate_gpu_route_record_completed(&route_record)?;
            if self.residency.config().attention_kinds[layer] == AttentionKind::CompressedSparse
                && (position as usize + 1) / 4 > DEEPSEEK_V4_CSA_TOP_K
                && {
                    #[cfg(feature = "dsv4-diagnostics")]
                    {
                        fp4_score_plan.runs_f16()
                    }
                    #[cfg(not(feature = "dsv4-diagnostics"))]
                    {
                        true
                    }
                }
            {
                self.sparse_csa.validate_completed(&selection_record)?;
            }

            #[cfg(feature = "dsv4-diagnostics")]
            let (csa_decision, indexer_head_weights, indexer_query_norms, indexer_key_norms) =
                if self.decision_diagnostics.is_capturing()
                    && self.residency.config().attention_kinds[layer]
                        == AttentionKind::CompressedSparse
                {
                    let rows = self.compressor_frontiers.csa_rows(layer, position)?;
                    match rows.filter(|rows| rows.count > DEEPSEEK_V4_CSA_TOP_K) {
                        Some(rows) => {
                            let query_norms =
                                if self.sparse_csa.use_f16_matrix_score(ctx, rows.count) {
                                    host_l2_norms_f16_rows(
                                        &self.sparse_csa.matrix_queries_f16,
                                        64,
                                        128,
                                        "diagnostic sparse CSA F16 matrix queries",
                                    )?
                                } else {
                                    host_l2_norms_f32_rows(
                                        &self.sparse_csa.index_queries,
                                        64,
                                        128,
                                        "diagnostic sparse CSA index queries",
                                    )?
                                };
                            (
                                Some(
                                    self.sparse_csa
                                        .capture_decision(rows.count, &selection_record)?,
                                ),
                                host_read_f32(
                                    &self.sparse_csa.head_weights,
                                    "diagnostic sparse CSA head weights",
                                )?,
                                query_norms,
                                host_l2_norms_f16_rows(
                                    rows.indexer_cache,
                                    rows.count,
                                    128,
                                    "diagnostic sparse CSA index keys",
                                )?,
                            )
                        }
                        None => (None, Vec::new(), Vec::new(), Vec::new()),
                    }
                } else {
                    (None, Vec::new(), Vec::new(), Vec::new())
                };

            #[cfg(feature = "dsv4-diagnostics")]
            if self.decision_diagnostics.is_capturing() {
                let route = self.moe.capture_route_decision(&route_record)?;
                self.decision_diagnostics.capture_layer(
                    layer,
                    csa_decision,
                    indexer_head_weights,
                    indexer_query_norms,
                    indexer_key_norms,
                    route,
                )?;
            }

            #[cfg(feature = "dsv4-diagnostics")]
            if self.fp4_shadow_diagnostics.is_capturing()
                && self.residency.config().attention_kinds[layer] == AttentionKind::CompressedSparse
            {
                let rows = self
                    .compressor_frontiers
                    .csa_rows(layer, position)?
                    .ok_or_else(|| {
                        DeepSeekV4MetalError::Invalid(format!(
                            "FP4 shadow CSA layer {layer} has no published rows"
                        ))
                    })?;
                let report = self.fp4_shadow.capture_layer(
                    layer,
                    position,
                    rows,
                    &self.sparse_csa.scores,
                    &self.sparse_csa.selected_mask,
                    &self.sparse_csa.cache_order_ids,
                    &selection_record.selected_count,
                    &selection_record.status,
                )?;
                self.fp4_shadow_diagnostics.capture_layer(report)?;
            }
            #[cfg(feature = "dsv4-diagnostics")]
            if fp4_score_plan.consumes_fp4()
                && self.residency.config().attention_kinds[layer] == AttentionKind::CompressedSparse
                && (position as usize + 1) / 4 > DEEPSEEK_V4_CSA_TOP_K
            {
                self.fp4_shadow.validate_completed()?;
                self.fp4_shadow.record_counterfactual_selection(
                    &mut self.fp4_counterfactual_trace,
                    DeepSeekV4Fp4ShadowExecution::Singleton,
                    DeepSeekV4Fp4SelectionSource::Fp4,
                    position,
                    layer,
                )?;
            }

            let command_gpu_ms = if routing_profile.is_some() || stage_sampled {
                let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
                if !gpu_seconds.is_finite() || gpu_seconds <= 0.0 {
                    return invalid(format!(
                        "layer {layer} returned invalid Metal command timestamps: ({}, {})",
                        command.GPUStartTime(),
                        command.GPUEndTime()
                    ));
                }
                Some(gpu_seconds * 1e3)
            } else {
                None
            };
            if let Some(profile) = routing_profile.as_deref_mut() {
                profile.push(DeepSeekV4LayerCommandProfile {
                    layer,
                    attention_kind: self.residency.config().attention_kinds[layer],
                    routing_kind: if layer < self.residency.config().hash_layer_count as usize {
                        DeepSeekV4RoutingKind::Hash
                    } else {
                        DeepSeekV4RoutingKind::Learned
                    },
                    encode_cpu_ms,
                    command_gpu_ms: command_gpu_ms
                        .expect("profiled command requires a GPU duration"),
                });
            }
            if stage_sampled {
                stage_recorder
                    .as_deref_mut()
                    .expect("sampled layer requires a stage recorder")
                    .record_command_gpu_ms(
                        layer,
                        command_gpu_ms.expect("sampled command requires a GPU duration"),
                    );
            }
            layer_completed(layer);
        }

        if let Some(recorder) = stage_recorder {
            recorder.resolve(ctx)?;
        }

        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        self.fp4_shadow_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            fp4_score_dispatch_ledger.validate_completed()?;
            self.fp4_score_dispatch_ledger = Some(fp4_score_dispatch_ledger);
        }

        Ok(())
    }

    pub(super) fn forward_token_inner_collapsed(
        &mut self,
        ctx: &MetalContext,
        token_id: u32,
        position: u32,
        layer_completed: &mut impl FnMut(usize),
        mut whole_profile: Option<&mut DeepSeekV4WholeTokenProfile>,
    ) -> Result<(), DeepSeekV4MetalError> {
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_score_plan = self.fp4_selection_mode.score_plan(false);
        #[cfg(feature = "dsv4-diagnostics")]
        if matches!(fp4_score_plan, DeepSeekV4Fp4ScorePlan::Paired { .. }) {
            return invalid("paired FP4 scoring requires the instrumented singleton schedule");
        }
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_collapsed_active = fp4_score_plan == DeepSeekV4Fp4ScorePlan::Fp4Only;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_execution = DeepSeekV4Fp4ShadowExecution::SingletonCollapsed;
        #[cfg(feature = "dsv4-diagnostics")]
        let fp4_source = fp4_score_plan.consumed_source();
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_score_dispatch_ledger = fp4_collapsed_active.then(|| {
            DeepSeekV4Fp4ScoreDispatchLedger::new(
                fp4_execution,
                position,
                fp4_score_plan.kind(),
                fp4_source,
            )
        });
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = None;
        }
        let reset_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        self.layer_routes.reset_for_token()?;
        self.layer_selections.reset_for_token()?;
        #[cfg(feature = "dsv4-diagnostics")]
        if fp4_collapsed_active {
            self.fp4_collapsed_selections
                .reset_for_token(fp4_execution, fp4_source)?;
        }
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.record_reset_cpu_ms = reset_started
                .expect("whole-token profile requires a reset timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let create_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let command = ctx.queue.commandBuffer().ok_or_else(|| {
            DeepSeekV4MetalError::Invalid(
                "failed to allocate DeepSeek V4 whole-token command buffer".into(),
            )
        })?;
        let mut sparse_visible_counts = [None; DEEPSEEK_V4_LAYER_COUNT];
        let mut token_encoder = DeepSeekV4LayerEncoder::begin(&command, 0, None)?;
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.command_encoder_create_cpu_ms = create_started
                .expect("whole-token profile requires a command-creation timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let encode_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        for (layer, sparse_visible_count) in sparse_visible_counts.iter_mut().enumerate() {
            let route_record = self.layer_routes.layer(layer)?;
            let selection_record = self.layer_selections.layer(layer)?;
            let encoded = self.encode_token_layer(
                ctx,
                &mut token_encoder,
                token_id,
                position,
                layer,
                &route_record,
                &selection_record,
            )?;
            *sparse_visible_count = encoded.sparse_visible_count;
            #[cfg(feature = "dsv4-diagnostics")]
            if encoded.sparse_visible_count.is_some() && fp4_collapsed_active {
                let ledger = fp4_score_dispatch_ledger
                    .as_mut()
                    .expect("collapsed FP4 score plan requires a dispatch ledger");
                ledger.record_common_prepare()?;
                ledger.record_fp4_pipeline()?;
            }
        }
        token_encoder.end();
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.encode_cpu_ms = encode_started
                .expect("whole-token profile requires an encode timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let commit_wait_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let commit_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        command.commit();
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.commit_cpu_ms = commit_started
                .expect("whole-token profile requires a commit timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }
        command.waitUntilCompleted();
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.commit_wait_wall_ms = commit_wait_started
                .expect("whole-token profile requires a commit/wait timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let status_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        if let Some(error) = command.error() {
            return invalid(format!("whole-token Metal command failed: {error:?}"));
        }
        if let Some(profile) = whole_profile.as_deref_mut() {
            let gpu_seconds = command.GPUEndTime() - command.GPUStartTime();
            if !gpu_seconds.is_finite() || gpu_seconds <= 0.0 {
                return invalid(format!(
                    "whole-token command returned invalid Metal timestamps: ({}, {})",
                    command.GPUStartTime(),
                    command.GPUEndTime()
                ));
            }
            profile.command_gpu_start_seconds = command.GPUStartTime();
            profile.command_gpu_end_seconds = command.GPUEndTime();
            profile.command_gpu_ms = gpu_seconds * 1e3;
            if profile.wait_residual_ms() < -0.25 {
                return invalid(format!(
                    "whole-token commit/wait envelope {:.6} ms is shorter than GPU duration {:.6} ms",
                    profile.commit_wait_wall_ms, profile.command_gpu_ms
                ));
            }
            profile.command_status_cpu_ms = status_started
                .expect("whole-token profile requires a command-status timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let record_read_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        let completed_routes = self.layer_routes.read_completed()?;
        let completed_selections = self.layer_selections.read_completed()?;
        #[cfg(feature = "dsv4-diagnostics")]
        let completed_fp4_selections = fp4_collapsed_active
            .then(|| self.fp4_collapsed_selections.read_completed())
            .transpose()?;
        if let Some(profile) = whole_profile.as_deref_mut() {
            profile.record_read_cpu_ms = record_read_started
                .expect("whole-token profile requires a record-read timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        let validate_started = whole_profile.as_ref().map(|_| std::time::Instant::now());
        #[cfg(feature = "dsv4-diagnostics")]
        let mut fp4_ids = [None; DEEPSEEK_V4_LAYER_COUNT];
        for (layer, sparse_visible_count) in sparse_visible_counts.iter().copied().enumerate() {
            completed_routes.validate_layer(layer)?;
            if let Some(visible_count) = sparse_visible_count {
                completed_selections.validate_layer(layer, visible_count)?;
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(completed_fp4) = completed_fp4_selections.as_ref() {
                    fp4_ids[layer] = Some(completed_fp4.validate_layer(
                        layer,
                        visible_count,
                        fp4_execution,
                        fp4_source,
                    )?);
                }
            } else {
                #[cfg(feature = "dsv4-diagnostics")]
                if let Some(completed_fp4) = completed_fp4_selections.as_ref() {
                    completed_fp4.validate_inactive_layer(layer, fp4_execution, fp4_source)?;
                }
            }
        }
        #[cfg(feature = "dsv4-diagnostics")]
        if let Some(ledger) = fp4_score_dispatch_ledger.as_ref() {
            ledger.validate_completed()?;
        }
        #[cfg(feature = "dsv4-diagnostics")]
        if fp4_collapsed_active {
            for (layer, sparse_visible_count) in sparse_visible_counts.iter().copied().enumerate() {
                if let Some(visible_count) = sparse_visible_count {
                    self.fp4_counterfactual_trace.record(
                        fp4_execution,
                        fp4_source,
                        position,
                        layer,
                        visible_count as i32,
                        fp4_ids[layer]
                            .expect("validated collapsed FP4 layer requires retained IDs"),
                    )?;
                }
            }
        }
        for layer in 0..DEEPSEEK_V4_LAYER_COUNT {
            layer_completed(layer);
        }

        #[cfg(feature = "dsv4-diagnostics")]
        self.decision_diagnostics.finish()?;
        #[cfg(feature = "dsv4-diagnostics")]
        {
            self.fp4_score_dispatch_ledger = fp4_score_dispatch_ledger;
        }
        if let Some(profile) = whole_profile {
            profile.record_validate_callback_cpu_ms = validate_started
                .expect("whole-token profile requires a validation timer")
                .elapsed()
                .as_secs_f64()
                * 1e3;
        }

        Ok(())
    }

    pub(super) fn raw_cache_layer(
        &self,
        layer: usize,
    ) -> Result<MetalTensor, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("raw-cache layer {layer} is out of range"));
        }
        let layer_elements = checked_mul(
            self.attention.config().head_dim,
            DEEPSEEK_V4_LOCAL_WINDOW,
            "raw-cache layer elements",
        )?;
        Ok(self.raw_cache.view_subrange(
            checked_mul(layer, layer_elements, "raw-cache layer offset")? as u64,
            vec![
                self.attention.config().head_dim as u64,
                DEEPSEEK_V4_LOCAL_WINDOW as u64,
            ],
        ))
    }

    pub(super) fn layer_tensor(
        &self,
        layer: usize,
        suffix: &str,
    ) -> Result<&MetalTensor, DeepSeekV4MetalError> {
        self.residency
            .require_tensor(&format!("blk.{layer}.{suffix}"))
    }
}

pub(super) fn deepseek_v4_session_moe_config(config: &DeepSeekV4Config) -> DeepSeekV4MoeConfig {
    DeepSeekV4MoeConfig {
        hidden_size: DEEPSEEK_V4_HIDDEN_SIZE,
        ffn_size: 2_048,
        expert_count: config.expert_count as usize,
        top_k: 6,
        routed_scale: config.expert_weights_scale,
    }
}

pub(super) fn validate_session_config(
    config: &DeepSeekV4Config,
) -> Result<(), DeepSeekV4MetalError> {
    config.validate_flash_0731_profile()?;
    if config.hidden_size != 4_096
        || config.vocab_size != 129_280
        || config.layer_count != 43
        || config.hyper_connection_count != 4
        || config.attention_head_count != 64
        || config.key_length != 512
        || config.value_length != 512
        || config.q_lora_rank != 1_024
        || config.output_group_count != 8
        || config.output_lora_rank != 1_024
        || config.expert_used_count != 6
        || config.expert_feed_forward_length != 2_048
        || config.shared_expert_count != 1
        || config.sinkhorn_iterations != DEEPSEEK_V4_SINKHORN_ITERATIONS as u32
    {
        return invalid("native session requires the exact Flash-0731 dimensions");
    }
    if !flash_0731_expert_count_supported(config.expert_count) {
        return invalid(format!(
            "native session supports Flash-0731 expert counts 160, 216, and 256, got {}",
            config.expert_count
        ));
    }
    if !config.expert_weights_norm || config.expert_gating_func != 4 {
        return invalid(
            "native route helper requires normalized expert weights and sqrt-softplus gating function 4",
        );
    }
    Ok(())
}

pub(super) fn session_required_tensor_names(config: &DeepSeekV4Config) -> Vec<String> {
    let mut names = [
        "token_embd.weight",
        "output_hc_fn.weight",
        "output_hc_scale.weight",
        "output_hc_base.weight",
        "output_norm.weight",
        "output.weight",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect::<Vec<_>>();
    let common = [
        "hc_attn_fn.weight",
        "hc_attn_scale.weight",
        "hc_attn_base.weight",
        "attn_norm.weight",
        "attn_sinks.weight",
        "attn_q_a.weight",
        "attn_q_a_norm.weight",
        "attn_q_b.weight",
        "attn_kv.weight",
        "attn_kv_a_norm.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_scale.weight",
        "hc_ffn_base.weight",
        "ffn_norm.weight",
        "ffn_gate_inp.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
    ];
    for layer in 0..config.layer_count as usize {
        names.extend(common.iter().map(|suffix| format!("blk.{layer}.{suffix}")));
        let route = if layer < config.hash_layer_count as usize {
            "ffn_gate_tid2eid.weight"
        } else {
            "exp_probs_b.bias"
        };
        names.push(format!("blk.{layer}.{route}"));
        match config.attention_kinds[layer] {
            AttentionKind::SlidingWindow => {}
            AttentionKind::CompressedSparse => {
                names.extend(
                    [
                        "attn_compressor_kv.weight",
                        "attn_compressor_gate.weight",
                        "attn_compressor_ape.weight",
                        "attn_compressor_norm.weight",
                        "indexer.attn_q_b.weight",
                        "indexer.proj.weight",
                        "indexer_compressor_kv.weight",
                        "indexer_compressor_gate.weight",
                        "indexer_compressor_ape.weight",
                        "indexer_compressor_norm.weight",
                    ]
                    .into_iter()
                    .map(|suffix| format!("blk.{layer}.{suffix}")),
                );
            }
            AttentionKind::HeavilyCompressed => {
                names.extend(
                    [
                        "attn_compressor_kv.weight",
                        "attn_compressor_gate.weight",
                        "attn_compressor_ape.weight",
                        "attn_compressor_norm.weight",
                    ]
                    .into_iter()
                    .map(|suffix| format!("blk.{layer}.{suffix}")),
                );
            }
        }
    }
    names
}

pub(super) fn validate_session_lookup_storage(gguf: &GgufFile) -> Result<(), DeepSeekV4MetalError> {
    let dtype = |name: &str| {
        gguf.tensors
            .iter()
            .find(|desc| desc.name == name)
            .map(|desc| desc.dtype)
            .ok_or_else(|| {
                DeepSeekV4MetalError::Invalid(format!(
                    "DeepSeek V4 session requires tensor {name:?}"
                ))
            })
    };
    validate_session_lookup_dtypes(dtype("token_embd.weight")?, dtype("output.weight")?)
}

pub(super) fn validate_session_lookup_dtypes(
    embedding_dtype: GgmlType,
    output_dtype: GgmlType,
) -> Result<(), DeepSeekV4MetalError> {
    if !matches!(embedding_dtype, GgmlType::Q6_K | GgmlType::Q8_0) {
        return invalid(format!(
            "token_embd.weight must be Q6_K or Q8_0 for the native session get_rows path, got {embedding_dtype:?}"
        ));
    }
    if !matches!(output_dtype, GgmlType::Q6_K | GgmlType::Q8_0) {
        return invalid(format!(
            "output.weight must be Q6_K or Q8_0 for the native session logits path, got {output_dtype:?}"
        ));
    }
    Ok(())
}
