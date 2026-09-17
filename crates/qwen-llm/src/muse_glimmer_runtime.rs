//! Request-shaped Muse Glimmer text runtime.

use crate::gguf::GgufFile;
use crate::metal::{
    MetalContext, MetalMemoryAdmission, PostBlockIntervention, evaluate_metal_memory_admission,
};
use crate::muse_glimmer::{MuseGlimmerArtifactProfile, MuseGlimmerConfig};
use crate::muse_glimmer_lens::{
    MuseGlimmerLensCapture, MuseGlimmerLensCaptureBank, MuseGlimmerLensError, MuseGlimmerLensRule,
    MuseGlimmerSelectedTokenCovectors, muse_glimmer_selected_token_covectors,
    project_f16_transport_covectors,
};
use crate::muse_glimmer_lens_fit::{
    MuseGlimmerAdjacentRowSlab, MuseGlimmerAdjacentSelectedTokenFit,
    MuseGlimmerBatchedFullTransportRowFit, MuseGlimmerFullTransportRowFit,
    MuseGlimmerMultiSourceSelectedTokenFit, MuseGlimmerOneBlockVjp,
    MuseGlimmerQueryBatchComposedVjp, MuseGlimmerQueryBatchOneBlockVjp,
};
use crate::muse_glimmer_residency::{
    MuseGlimmerMetalWeightPlan, MuseGlimmerMetalWeights, MuseGlimmerResidencyError,
};
use crate::muse_glimmer_text_session::{
    MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES, MuseGlimmerBatchedFullReadout,
    MuseGlimmerFullReadoutWorkspace, MuseGlimmerFullReadoutWorkspacePlan,
    MuseGlimmerPostBlockForward, MuseGlimmerPreparedF16Transport, MuseGlimmerTextForward,
    MuseGlimmerTextGeometry, MuseGlimmerTextSession, MuseGlimmerTextSessionError,
    MuseGlimmerTextSessionMemoryPlan,
};
use objc2_metal::MTLDevice;

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerRuntimeError {
    #[error(transparent)]
    Residency(#[from] MuseGlimmerResidencyError),
    #[error(transparent)]
    Session(#[from] MuseGlimmerTextSessionError),
    #[error(transparent)]
    Lens(#[from] MuseGlimmerLensError),
    #[error("invalid Muse Glimmer runtime contract: {0}")]
    Invalid(String),
}

#[derive(Clone, Copy, Debug)]
pub struct MuseGlimmerRuntimeAdmission {
    pub aggregate: MetalMemoryAdmission,
    pub weights: MetalMemoryAdmission,
    pub session: MetalMemoryAdmission,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MuseGlimmerRuntimeOptions {
    /// Allow tolerance-qualified H128 attention for ordinary generated tokens at
    /// visible KV ranges >=1024 on the Q8 M4 Max lane, through admitted model context.
    pub split_decode: bool,
    /// Allow Q8 matrix prefill with tiled N128 attention and online packed remainders; scalar kernels unchanged.
    /// Restricted to Q8/M4 Max and the admitted model context, not a benchmark length.
    pub matrix_prefill: bool,
}

impl Default for MuseGlimmerRuntimeOptions {
    fn default() -> Self {
        Self {
            split_decode: true,
            matrix_prefill: true,
        }
    }
}

impl MuseGlimmerRuntimeOptions {
    pub const REFERENCE: Self = Self {
        split_decode: false,
        matrix_prefill: false,
    };

    fn resolve(self, profile: MuseGlimmerArtifactProfile, device: &str, unified: bool) -> Self {
        if profile == MuseGlimmerArtifactProfile::UnslothQ8_0 && device == "Apple M4 Max" && unified
        {
            self
        } else {
            Self::REFERENCE
        }
    }
}

pub struct MuseGlimmerLoadedModel {
    weights: MuseGlimmerMetalWeights,
    session: MuseGlimmerTextSession,
    capacity: usize,
    admission: MuseGlimmerRuntimeAdmission,
    observed_weight_bytes: u64,
    device_registry_id: u64,
    math_options: MuseGlimmerRuntimeOptions,
}

impl MuseGlimmerLoadedModel {
    pub fn load(
        ctx: &MetalContext,
        gguf: &GgufFile,
        capacity: usize,
    ) -> Result<Self, MuseGlimmerRuntimeError> {
        Self::load_with_options(ctx, gguf, capacity, MuseGlimmerRuntimeOptions::default())
    }

    /// Preserve the original arithmetic for analysis artifacts and reference benchmarks.
    pub fn load_reference(
        ctx: &MetalContext,
        gguf: &GgufFile,
        capacity: usize,
    ) -> Result<Self, MuseGlimmerRuntimeError> {
        Self::load_with_options(ctx, gguf, capacity, MuseGlimmerRuntimeOptions::REFERENCE)
    }

    pub fn load_with_options(
        ctx: &MetalContext,
        gguf: &GgufFile,
        capacity: usize,
        options: MuseGlimmerRuntimeOptions,
    ) -> Result<Self, MuseGlimmerRuntimeError> {
        let weight_plan = MuseGlimmerMetalWeightPlan::for_release(ctx, gguf)?;
        let options = options.resolve(
            weight_plan.artifact_profile(),
            &ctx.device.name().to_string(),
            ctx.device.hasUnifiedMemory(),
        );
        let geometry = MuseGlimmerTextGeometry::from_config(weight_plan.config(), capacity)?;
        let session_plan = MuseGlimmerTextSessionMemoryPlan::for_geometry_with_split_decode(
            ctx,
            &geometry,
            options.split_decode,
        )?;
        let aggregate_bytes = weight_plan
            .memory_plan()
            .priced_upper_bytes()
            .checked_add(session_plan.priced_upper_bytes())
            .ok_or_else(|| {
                MuseGlimmerRuntimeError::Invalid(
                    "weight and session priced byte total overflow".into(),
                )
            })?;

        let _allocation_transaction = ctx.begin_allocation_transaction();
        let aggregate = evaluate_metal_memory_admission(
            aggregate_bytes,
            MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES,
            ctx.memory_signals(),
            true,
        );
        if !aggregate.admitted {
            return invalid(format!(
                "combined weight and session admission denied: reason={} required={:?} working_set_headroom={:?} process_remaining={:?}",
                aggregate.reason.as_str(),
                aggregate.required_bytes,
                aggregate.working_set_headroom_bytes,
                aggregate.signals.process_limit_remaining_bytes
            ));
        }

        let admitted_weights = weight_plan.admit(ctx.memory_signals())?;
        let realized = MuseGlimmerMetalWeights::realize(ctx, gguf, admitted_weights)?;
        let weight_admission = realized.admission();
        let observed_weight_bytes = realized.observed_allocation_delta();
        let weights = realized.into_weights();
        let session = MuseGlimmerTextSession::new_with_split_decode(
            ctx,
            weights.config(),
            capacity,
            options.split_decode,
        )?;
        if session.memory_plan() != &session_plan {
            return invalid("realized session memory plan differs from aggregate admission");
        }
        let session_admission = session.admission();

        Ok(Self {
            weights,
            session,
            capacity,
            admission: MuseGlimmerRuntimeAdmission {
                aggregate,
                weights: weight_admission,
                session: session_admission,
            },
            observed_weight_bytes,
            device_registry_id: ctx.device.registryID(),
            math_options: options,
        })
    }

    pub fn config(&self) -> &MuseGlimmerConfig {
        self.weights.config()
    }

    /// Effective selection after artifact/device qualification, not just permission.
    pub fn math_options(&self) -> MuseGlimmerRuntimeOptions {
        self.math_options
    }

    pub fn artifact_profile(&self) -> MuseGlimmerArtifactProfile {
        self.weights.artifact_profile()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn admission(&self) -> MuseGlimmerRuntimeAdmission {
        self.admission
    }

    pub fn observed_weight_bytes(&self) -> u64 {
        self.observed_weight_bytes
    }

    pub fn observed_session_bytes(&self) -> u64 {
        self.session.observed_allocation_delta()
    }

    pub fn selected_token_lens_covectors(
        &self,
        ctx: &MetalContext,
        token_ids: &[u32],
    ) -> Result<MuseGlimmerSelectedTokenCovectors, MuseGlimmerRuntimeError> {
        Ok(muse_glimmer_selected_token_covectors(
            ctx,
            &self.weights,
            token_ids,
        )?)
    }

    pub fn project_f16_transport_lens_covectors(
        &self,
        ctx: &MetalContext,
        transport_bytes: &[u8],
        covectors: &MuseGlimmerSelectedTokenCovectors,
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "loaded model belongs to Metal device registry {}, projection context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        if covectors.hidden_size() != self.config().hidden_size as usize {
            return invalid("selected-token covectors have the wrong hidden size");
        }
        Ok(project_f16_transport_covectors(
            ctx,
            transport_bytes,
            covectors,
        )?)
    }

    pub fn create_runner<'ctx, 'model>(
        &'model mut self,
        ctx: &'ctx MetalContext,
    ) -> Result<MuseGlimmerTextRunner<'ctx, 'model>, MuseGlimmerRuntimeError> {
        if ctx.device.registryID() != self.device_registry_id {
            return invalid(format!(
                "loaded model belongs to Metal device registry {}, runner context is {}",
                self.device_registry_id,
                ctx.device.registryID()
            ));
        }
        let Self {
            weights,
            session,
            math_options,
            ..
        } = self;
        let forward = MuseGlimmerTextForward::new_with_tiled_prefill(
            ctx,
            weights,
            math_options.matrix_prefill,
        )?;
        Ok(MuseGlimmerTextRunner { forward, session })
    }
}

pub struct MuseGlimmerTextRunner<'ctx, 'model> {
    forward: MuseGlimmerTextForward<'ctx, 'model>,
    session: &'model mut MuseGlimmerTextSession,
}

impl MuseGlimmerTextRunner<'_, '_> {
    pub fn capacity(&self) -> usize {
        self.session.geometry().capacity()
    }

    pub fn next_position(&self) -> usize {
        self.session.next_position()
    }

    pub fn remaining_forwards(&self) -> usize {
        self.session.remaining_forwards()
    }

    pub fn forward_token(&mut self, token: u32) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self.forward.forward_generated_token(token, self.session)?)
    }

    /// Apply the resident deployed output norm, projection, scale, and softcap
    /// to one final post-block residual without advancing the text session.
    pub fn deployed_logits_from_post_block_residual(
        &mut self,
        residual: &[f32],
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .deployed_logits_from_post_block_residual(residual, self.session)?)
    }

    /// Apply one row-major F16 hidden-to-hidden transport to a post-block
    /// residual without advancing or otherwise mutating the text session.
    /// Artifact consumers remain responsible for binding the bytes to this
    /// model, fitting corpus, method, and selected source layer.
    pub fn apply_f16_post_block_transport(
        &self,
        transport_bytes: &[u8],
        source_residual: &[f32],
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .apply_f16_post_block_transport(transport_bytes, source_residual)?)
    }

    /// Upload and validate one F16 transport for repeated passive applications.
    pub fn prepare_f16_post_block_transport(
        &self,
        transport_bytes: &[u8],
    ) -> Result<MuseGlimmerPreparedF16Transport, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .prepare_f16_post_block_transport(transport_bytes)?)
    }

    /// Allocate reusable storage for batched passive full-vocabulary readout.
    pub fn full_readout_workspace_plan(
        &self,
        row_capacity: usize,
    ) -> Result<MuseGlimmerFullReadoutWorkspacePlan, MuseGlimmerRuntimeError> {
        Ok(self.forward.full_readout_workspace_plan(row_capacity)?)
    }

    /// Whether this resident head has an arithmetic-equivalent command-batched path.
    pub fn supports_command_batched_full_readout(&self) -> bool {
        self.forward.supports_command_batched_full_readout()
    }

    /// Allocate reusable storage after refreshing Metal memory admission.
    pub fn create_full_readout_workspace(
        &self,
        row_capacity: usize,
    ) -> Result<MuseGlimmerFullReadoutWorkspace, MuseGlimmerRuntimeError> {
        Ok(self.forward.create_full_readout_workspace(row_capacity)?)
    }

    /// Run independent source rows through a prepared transport and the
    /// deployed output tail, returning compact exact top-k results.
    pub fn apply_prepared_f16_transport_topk_rows(
        &self,
        workspace: &mut MuseGlimmerFullReadoutWorkspace,
        transport: &MuseGlimmerPreparedF16Transport,
        source_rows: &[f32],
        top_k: usize,
        transported_rows: &[usize],
    ) -> Result<MuseGlimmerBatchedFullReadout, MuseGlimmerRuntimeError> {
        Ok(self.forward.apply_prepared_f16_transport_topk_rows(
            workspace,
            transport,
            source_rows,
            top_k,
            transported_rows,
        )?)
    }

    /// Apply a prepared transport without advancing or mutating the text session.
    pub fn apply_prepared_f16_post_block_transport(
        &self,
        transport: &MuseGlimmerPreparedF16Transport,
        source_residual: &[f32],
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .apply_prepared_f16_post_block_transport(transport, source_residual)?)
    }

    pub fn forward_token_with_post_block_interventions(
        &mut self,
        token: u32,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self.forward.forward_token_with_post_block_interventions(
            token,
            interventions,
            self.session,
        )?)
    }

    /// Forward one scalar token normally while copying selected post-block
    /// residuals from the same command buffer. Layer IDs must be sorted unique.
    pub fn forward_token_capture_post_blocks(
        &mut self,
        token: u32,
        layer_ids: &[u32],
    ) -> Result<MuseGlimmerPostBlockForward, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .forward_token_capture_post_blocks(token, layer_ids, self.session)?)
    }

    pub fn forward_token_capture_post_blocks_with_interventions(
        &mut self,
        token: u32,
        layer_ids: &[u32],
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<MuseGlimmerPostBlockForward, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .forward_token_capture_post_blocks_with_interventions(
                token,
                layer_ids,
                interventions,
                self.session,
            )?)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self.forward.prefill(tokens, self.session)?)
    }

    /// Run a bounded scalar prompt from a fresh session and capture one
    /// nonzero block's three residual coordinates for every prompt token.
    pub fn capture_fresh_lens_prompt(
        &mut self,
        tokens: &[u32],
        target_block: u32,
    ) -> Result<MuseGlimmerLensCapture, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .capture_fresh_lens_prompt(tokens, target_block, self.session)?)
    }

    /// Capture block input, post-attention, and post-block coordinates for a
    /// sorted unique set of nonzero blocks in one fresh scalar prompt pass.
    pub fn capture_fresh_lens_prompt_blocks(
        &mut self,
        tokens: &[u32],
        target_blocks: &[u32],
    ) -> Result<MuseGlimmerLensCaptureBank, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .capture_fresh_lens_prompt_blocks(tokens, target_blocks, self.session)?)
    }

    /// Reverse one `[T,H]` cotangent through the selected full-attention block's
    /// smooth F32 model-level replay. Capture diagnostics report drift from
    /// production's F16 KV path; the VJP is intentionally not an STE through
    /// that conversion.
    pub fn lens_one_full_attention_block_vjp(
        &self,
        capture: &MuseGlimmerLensCapture,
        target_cotangent: &[f32],
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .lens_one_full_attention_block_vjp(capture, target_cotangent, rule)?)
    }

    /// Reverse a query-major `[Q,T,H]` cotangent bank through one attention
    /// block. The primal replay is shared across Q; Q must be in the bounded
    /// range advertised by `MUSE_GLIMMER_QUERY_BATCH_MAX`.
    pub fn lens_one_attention_block_vjp_query_batch(
        &self,
        capture: &MuseGlimmerLensCapture,
        target_cotangents: &[f32],
        query_count: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerRuntimeError> {
        Ok(self.forward.lens_one_attention_block_vjp_query_batch(
            capture,
            target_cotangents,
            query_count,
            rule,
        )?)
    }

    pub fn lens_one_full_attention_block_vjp_query_batch(
        &self,
        capture: &MuseGlimmerLensCapture,
        target_cotangents: &[f32],
        query_count: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerQueryBatchOneBlockVjp, MuseGlimmerRuntimeError> {
        Ok(self.forward.lens_one_full_attention_block_vjp_query_batch(
            capture,
            target_cotangents,
            query_count,
            rule,
        )?)
    }

    pub fn lens_composed_vjp_query_batch(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        target_cotangents: &[f32],
        query_count: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerQueryBatchComposedVjp, MuseGlimmerRuntimeError> {
        Ok(self.forward.lens_composed_vjp_query_batch(
            captures,
            target_block,
            source_layers,
            target_cotangents,
            query_count,
            rule,
        )?)
    }

    /// Fit one direction per selected token from `target_block` to exactly
    /// `target_block - 1`. Positions are `skip_first..T-1`; each VJP places
    /// one covector on every valid target row and means the matching source rows.
    pub fn fit_adjacent_full_attention_selected_tokens(
        &self,
        capture: &MuseGlimmerLensCapture,
        covectors: &MuseGlimmerSelectedTokenCovectors,
        skip_first: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerAdjacentSelectedTokenFit, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .fit_adjacent_full_attention_selected_tokens(capture, covectors, skip_first, rule)?)
    }

    /// Fit contiguous hidden-coordinate rows through one adjacent full-attention
    /// block with a fixed-size resident VJP bank. Values are row-major `[R,H]`.
    pub fn fit_adjacent_full_attention_rows_batched(
        &self,
        capture: &MuseGlimmerLensCapture,
        rows: std::ops::Range<u32>,
        skip_first: usize,
        dim_batch: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerAdjacentRowSlab, MuseGlimmerRuntimeError> {
        Ok(self
            .forward
            .fit_adjacent_full_attention_rows_batched(capture, rows, skip_first, dim_batch, rule)?)
    }

    /// Fit selected-token directions from one target to arbitrary strictly
    /// increasing post-block source layers below it.
    pub fn fit_selected_tokens_to_sources(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        covectors: &MuseGlimmerSelectedTokenCovectors,
        skip_first: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerMultiSourceSelectedTokenFit, MuseGlimmerRuntimeError> {
        Ok(self.forward.fit_selected_tokens_to_sources(
            captures,
            target_block,
            source_layers,
            covectors,
            skip_first,
            rule,
        )?)
    }

    /// Fit selected rows of the scalar full-transport oracle. Each output row
    /// is a hidden-space basis covector placed at every valid target position.
    pub fn fit_full_transport_rows_to_sources(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        output_row_ids: &[u32],
        skip_first: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerFullTransportRowFit, MuseGlimmerRuntimeError> {
        Ok(self.forward.fit_full_transport_rows_to_sources(
            captures,
            target_block,
            source_layers,
            output_row_ids,
            skip_first,
            rule,
        )?)
    }

    /// Fit full-transport rows with exact query batches, chunking the row IDs
    /// by `query_batch_size` while retaining scalar source/row orientation.
    #[allow(clippy::too_many_arguments)]
    pub fn fit_full_transport_rows_to_sources_batched(
        &self,
        captures: &MuseGlimmerLensCaptureBank,
        target_block: u32,
        source_layers: &[u32],
        output_row_ids: &[u32],
        skip_first: usize,
        query_batch_size: usize,
        rule: MuseGlimmerLensRule,
    ) -> Result<MuseGlimmerBatchedFullTransportRowFit, MuseGlimmerRuntimeError> {
        Ok(self.forward.fit_full_transport_rows_to_sources_batched(
            captures,
            target_block,
            source_layers,
            output_row_ids,
            skip_first,
            query_batch_size,
            rule,
        )?)
    }

    pub fn prefill_with_command_checkpoint<F>(
        &mut self,
        tokens: &[u32],
        mut checkpoint: F,
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError>
    where
        F: FnMut() -> Result<(), String>,
    {
        Ok(self
            .forward
            .prefill_with_command_checkpoint(tokens, self.session, || {
                checkpoint().map_err(MuseGlimmerTextSessionError::Checkpoint)
            })?)
    }

    pub fn reset(&mut self) -> Result<(), MuseGlimmerRuntimeError> {
        Ok(self.session.reset()?)
    }

    /// Retain an already-consumed causal prefix. The caller must establish token
    /// identity and forward a nonempty suffix to obtain current logits.
    pub fn rewind_prefix(&mut self, position: usize) -> Result<(), MuseGlimmerRuntimeError> {
        Ok(self.session.rewind_prefix(position)?)
    }
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerRuntimeError> {
    Err(MuseGlimmerRuntimeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::muse_glimmer::MuseGlimmerConfig;

    #[test]
    fn muse_math_defaults_resolve_only_qualified_lanes_and_preserve_rollbacks() {
        let defaults = MuseGlimmerRuntimeOptions::default();
        assert!(defaults.split_decode && defaults.matrix_prefill);
        for profile in [
            MuseGlimmerArtifactProfile::UnslothQ8_0,
            MuseGlimmerArtifactProfile::UnslothBf16,
        ] {
            for device in ["Apple M4 Max", "Apple M4 Pro", "Apple M5 Max", "Other"] {
                for unified in [false, true] {
                    for split_decode in [false, true] {
                        for matrix_prefill in [false, true] {
                            let requested = MuseGlimmerRuntimeOptions {
                                split_decode,
                                matrix_prefill,
                            };
                            let expected = if profile == MuseGlimmerArtifactProfile::UnslothQ8_0
                                && device == "Apple M4 Max"
                                && unified
                            {
                                requested
                            } else {
                                MuseGlimmerRuntimeOptions::REFERENCE
                            };
                            assert_eq!(requested.resolve(profile, device, unified), expected);
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[ignore = "serial Metal, production lease, Muse default/explicit and reference/rollback delivery"]
    fn muse_math_default_delivery() {
        use crate::tokenizer::LlamaCppTokenizer;
        let _lease = crate::metal::acquire_metal_benchmark_lease()
            .expect("production GPU lease and wired-memory gate required");
        let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
        let gguf = GgufFile::open(path).unwrap();
        let config = MuseGlimmerConfig::from_gguf(&gguf).unwrap();
        let tokenizer = LlamaCppTokenizer::open(path).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/bench/tokenizer-messages/current-reva-short-qwen36.json"
        ))
        .unwrap();
        let request = crate::muse_glimmer_request::MuseGlimmerRequest::single_turn(
            "begin",
            Some(
                fixture["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            ),
        );
        let rendered = request.render(config.chat_template_profile, None).unwrap();
        let tokens: Vec<u32> = tokenizer
            .encode(&rendered, false)
            .unwrap()
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        assert_eq!(tokens.len(), 6229);
        let ctx = MetalContext::new().unwrap();
        let enabled = MuseGlimmerRuntimeOptions {
            split_decode: true,
            matrix_prefill: true,
        };
        let mut expected = None;
        for lane in ["explicit", "default", "reference", "rollback"] {
            let optimized = matches!(lane, "explicit" | "default");
            let prompt = if optimized {
                &tokens[..]
            } else {
                &tokens[..31]
            };
            let capacity = prompt.len() + 16;
            let mut model = match lane {
                "default" => MuseGlimmerLoadedModel::load(&ctx, &gguf, capacity),
                "reference" => MuseGlimmerLoadedModel::load_reference(&ctx, &gguf, capacity),
                _ => MuseGlimmerLoadedModel::load_with_options(
                    &ctx,
                    &gguf,
                    capacity,
                    if optimized {
                        enabled
                    } else {
                        MuseGlimmerRuntimeOptions::REFERENCE
                    },
                ),
            }
            .unwrap();
            assert_eq!(
                model.math_options(),
                if optimized {
                    enabled
                } else {
                    MuseGlimmerRuntimeOptions::REFERENCE
                }
            );
            assert_eq!(
                model
                    .session
                    .memory_plan()
                    .allocations()
                    .iter()
                    .any(|allocation| allocation.name == "split_decode_partials"),
                optimized
            );
            let plan = model.session.memory_plan().clone();
            let mut runner = model.create_runner(&ctx).unwrap();
            crate::metal::dispatch_census_begin();
            let mut logits = runner
                .prefill_with_command_checkpoint(prompt, || Ok(()))
                .unwrap();
            let mut rows = Vec::new();
            let mut ids = Vec::new();
            for step in 0..=16 {
                assert!(logits.iter().all(|v| v.is_finite()));
                let token = logits
                    .iter()
                    .enumerate()
                    .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
                    .unwrap()
                    .0 as u32;
                rows.push(logits.iter().map(|v| v.to_bits()).collect::<Vec<_>>());
                ids.push(token);
                if step < 16 {
                    logits = runner.forward_token(token).unwrap();
                }
            }
            assert_eq!(runner.next_position(), capacity);
            assert!(runner.forward_token(ids[16]).is_err());
            assert_eq!(runner.next_position(), capacity);
            let census = crate::metal::dispatch_census_take();
            let kernels: Vec<_> = census.iter().map(|row| row.kernel.clone()).collect();
            assert_eq!(
                kernels
                    .iter()
                    .any(|k| k == "kernel_muse_prefill_tiled_f32_h128"),
                optimized
            );
            assert_eq!(
                kernels
                    .iter()
                    .any(|k| k == "kernel_muse_split_attention_h128"),
                optimized
            );
            let result = (rows, ids, kernels, plan);
            if matches!(lane, "explicit" | "reference") {
                expected = Some(result);
            } else {
                assert_eq!(result, expected.take().unwrap(), "{lane} delivery differs");
            }
            eprintln!(
                "MUSE_DEFAULT_DELIVERY lane={lane} prompt={} transitions=16 passed=true",
                prompt.len()
            );
        }
    }

    #[test]
    #[ignore = "serial Metal, delivered optimized Muse prefill option short-shape composition"]
    fn optimized_prefill_runner_short_composition() {
        use crate::tokenizer::LlamaCppTokenizer;
        let path = crate::test_fixtures::MUSE_GLIMMER_Q8_0.path();
        let gguf = GgufFile::open(path).unwrap();
        let ctx = MetalContext::new().unwrap();
        let options = MuseGlimmerRuntimeOptions {
            split_decode: true,
            matrix_prefill: true,
        };
        let before = ctx.current_allocated_size();
        assert!(MuseGlimmerLoadedModel::load_with_options(&ctx, &gguf, 131073, options).is_err());
        assert_eq!(ctx.current_allocated_size(), before);
        let mut model =
            MuseGlimmerLoadedModel::load_with_options(&ctx, &gguf, 144, options).unwrap();
        let mut reference = MuseGlimmerTextSession::new(&ctx, model.config(), 144).unwrap();
        let tokenizer = LlamaCppTokenizer::open(path).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../../docs/bench/tokenizer-messages/current-reva-short-qwen36.json"
        ))
        .unwrap();
        let request = crate::muse_glimmer_request::MuseGlimmerRequest::single_turn(
            "begin",
            Some(
                fixture["messages"][0]["content"]
                    .as_str()
                    .unwrap()
                    .to_owned(),
            ),
        );
        let rendered = request
            .render(model.config().chat_template_profile, None)
            .unwrap();
        let tokens: Vec<u32> = tokenizer
            .encode(&rendered, false)
            .unwrap()
            .into_iter()
            .map(|id| u32::try_from(id).unwrap())
            .collect();
        let argmax = |logits: &[f32]| -> u32 {
            logits
                .iter()
                .enumerate()
                .max_by(|(ai, a), (bi, b)| a.total_cmp(b).then_with(|| bi.cmp(ai)))
                .unwrap()
                .0 as u32
        };
        for count in [16, 31, 128] {
            reference.reset().unwrap();
            let expected = {
                let forward = MuseGlimmerTextForward::new(&ctx, &model.weights).unwrap();
                let mut expected = vec![forward.prefill(&tokens[..count], &mut reference).unwrap()];
                for step in 0..16 {
                    expected.push(
                        forward
                            .forward_token(argmax(&expected[step]), &mut reference)
                            .unwrap(),
                    );
                }
                expected
            };
            let mut runner = model.create_runner(&ctx).unwrap();
            runner.reset().unwrap();
            let mut checkpoints = 0;
            let mut actual = runner
                .prefill_with_command_checkpoint(&tokens[..count], || {
                    checkpoints += 1;
                    Ok(())
                })
                .unwrap();
            assert!(checkpoints > 0);
            for (step, reference) in expected.iter().enumerate() {
                let mut dot = 0.0_f64;
                let mut aa = 0.0_f64;
                let mut bb = 0.0_f64;
                let mut difference = 0.0_f64;
                let mut max_abs = 0.0_f64;
                assert_eq!(actual.len(), reference.len());
                for (&a, &b) in reference.iter().zip(&actual) {
                    let (a, b) = (a as f64, b as f64);
                    assert!(a.is_finite() && b.is_finite());
                    dot += a * b;
                    aa += a * a;
                    bb += b * b;
                    difference += (a - b).powi(2);
                    max_abs = max_abs.max((a - b).abs());
                }
                let cosine = dot / (aa * bb).sqrt();
                let relative_rms = (difference / aa).sqrt();
                let (rms_gate, abs_gate) = if step == 0 {
                    (0.002, 0.1)
                } else {
                    (0.006, 0.3)
                };
                assert!(
                    cosine > 0.999_99 && relative_rms < rms_gate && max_abs < abs_gate,
                    "short count={count} step={step} cos={cosine} RMS={relative_rms} abs={max_abs}"
                );
                assert_eq!(argmax(&actual), argmax(reference));
                if step < 16 {
                    actual = runner.forward_token(argmax(&actual)).unwrap();
                }
            }
            assert_eq!(runner.next_position(), count + 16);
            eprintln!(
                "MUSE_OPTIMIZED_RUNNER tokens={count} logits_and_17_greedy_pass=true checkpoints={checkpoints}"
            );
        }
    }

    #[test]
    fn request_shaped_session_plan_matches_capacity() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let geometry = MuseGlimmerTextGeometry::from_config(&config, 257).unwrap();
        let plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(&ctx, &geometry).unwrap();
        assert!(plan.priced_upper_bytes() >= plan.logical_bytes());
        assert_eq!(geometry.capacity(), 257);
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn deployed_output_tail_matches_forward_from_captured_final_residual_bitwise() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let mut model = MuseGlimmerLoadedModel::load(&ctx, &gguf, 1).expect("load Muse Q8 target");
        let final_layer = model.config().layer_count - 1;
        let token = model.config().bos_token_id;
        let mut runner = model.create_runner(&ctx).expect("create Muse runner");
        let captured = runner
            .forward_token_capture_post_blocks(token, &[final_layer])
            .expect("forward and capture final residual");
        let next_position = runner.next_position();
        let tail_logits = runner
            .deployed_logits_from_post_block_residual(
                captured.layer_values(0).expect("captured final residual"),
            )
            .expect("run deployed output tail");
        let hidden = captured.hidden_size;
        let mut identity = vec![0u8; hidden * hidden * 2];
        for coordinate in 0..hidden {
            let offset = (coordinate * hidden + coordinate) * 2;
            identity[offset..offset + 2].copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
        }
        let transported = runner
            .apply_f16_post_block_transport(
                &identity,
                captured.layer_values(0).expect("captured final residual"),
            )
            .expect("apply identity transport");

        assert_eq!(runner.next_position(), next_position);
        assert_eq!(
            transported
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            captured
                .layer_values(0)
                .unwrap()
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            tail_logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>(),
            captured
                .logits
                .iter()
                .map(|value| value.to_bits())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn batched_full_readout_matches_scalar_oracle_exactly() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF")
            .unwrap_or_else(|_| crate::test_fixtures::MUSE_GLIMMER_Q8_0.path().into());
        let gguf = GgufFile::open(&path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let mut model = MuseGlimmerLoadedModel::load(&ctx, &gguf, 2).expect("load Muse Q8 target");
        let final_layer = model.config().layer_count - 1;
        let token = model.config().bos_token_id;
        let mut runner = model.create_runner(&ctx).expect("create Muse runner");
        let mut source_rows = Vec::new();
        for _ in 0..2 {
            let captured = runner
                .forward_token_capture_post_blocks(token, &[final_layer])
                .expect("capture final residual");
            source_rows.extend_from_slice(captured.layer_values(0).unwrap());
        }
        let hidden = source_rows.len() / 2;
        let mut identity = vec![0u8; hidden * hidden * 2];
        for coordinate in 0..hidden {
            let offset = (coordinate * hidden + coordinate) * 2;
            identity[offset..offset + 2].copy_from_slice(&half::f16::ONE.to_bits().to_le_bytes());
        }
        let transport = runner
            .prepare_f16_post_block_transport(&identity)
            .expect("prepare identity transport");
        let mut scalar = Vec::new();
        for source in source_rows.chunks_exact(hidden) {
            let transported = runner
                .apply_prepared_f16_post_block_transport(&transport, source)
                .expect("scalar transport");
            let logits = runner
                .deployed_logits_from_post_block_residual(&transported)
                .expect("scalar output tail");
            let mut scores = logits.into_iter().enumerate().collect::<Vec<_>>();
            scores.sort_by(|left, right| {
                right
                    .1
                    .total_cmp(&left.1)
                    .then_with(|| left.0.cmp(&right.0))
            });
            scores.truncate(16);
            scalar.push((transported, scores));
        }
        let mut workspace = runner
            .create_full_readout_workspace(2)
            .expect("create batched workspace");
        let batched = runner
            .apply_prepared_f16_transport_topk_rows(
                &mut workspace,
                &transport,
                &source_rows,
                16,
                &[0, 1],
            )
            .expect("batched readout");
        for row in 0..2 {
            assert_eq!(
                batched.transported_rows[row]
                    .values
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
                scalar[row]
                    .0
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>()
            );
            let actual = batched.rows[row]
                .scores
                .iter()
                .map(|score| (score.token_id as usize, score.logit.to_bits()))
                .collect::<Vec<_>>();
            let expected = scalar[row]
                .1
                .iter()
                .map(|&(token_id, logit)| (token_id, logit.to_bits()))
                .collect::<Vec<_>>();
            assert_eq!(actual, expected);
        }
    }
}
