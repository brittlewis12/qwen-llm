//! Request-shaped Muse Glimmer text runtime.

use crate::gguf::GgufFile;
use crate::metal::{
    MetalContext, MetalMemoryAdmission, PostBlockIntervention, evaluate_metal_memory_admission,
};
use crate::muse_glimmer::MuseGlimmerConfig;
use crate::muse_glimmer_lens::{
    MuseGlimmerLensCapture, MuseGlimmerLensCaptureBank, MuseGlimmerLensError, MuseGlimmerLensRule,
    MuseGlimmerSelectedTokenCovectors, muse_glimmer_selected_token_covectors,
};
use crate::muse_glimmer_lens_fit::{
    MuseGlimmerAdjacentSelectedTokenFit, MuseGlimmerMultiSourceSelectedTokenFit,
    MuseGlimmerOneBlockVjp,
};
use crate::muse_glimmer_residency::{
    MuseGlimmerMetalWeightPlan, MuseGlimmerMetalWeights, MuseGlimmerResidencyError,
};
use crate::muse_glimmer_text_session::{
    MUSE_GLIMMER_TEXT_SESSION_RESERVE_BYTES, MuseGlimmerPostBlockForward, MuseGlimmerTextForward,
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

pub struct MuseGlimmerLoadedModel {
    weights: MuseGlimmerMetalWeights,
    session: Option<MuseGlimmerTextSession>,
    capacity: usize,
    admission: MuseGlimmerRuntimeAdmission,
    observed_weight_bytes: u64,
    device_registry_id: u64,
}

impl MuseGlimmerLoadedModel {
    pub fn load(
        ctx: &MetalContext,
        gguf: &GgufFile,
        capacity: usize,
    ) -> Result<Self, MuseGlimmerRuntimeError> {
        let weight_plan = MuseGlimmerMetalWeightPlan::for_release(ctx, gguf)?;
        let geometry = MuseGlimmerTextGeometry::from_config(weight_plan.config(), capacity)?;
        let session_plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(ctx, &geometry)?;
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
        let session = MuseGlimmerTextSession::new(ctx, weights.config(), capacity)?;
        if session.memory_plan() != &session_plan {
            return invalid("realized session memory plan differs from aggregate admission");
        }
        let session_admission = session.admission();

        Ok(Self {
            weights,
            session: Some(session),
            capacity,
            admission: MuseGlimmerRuntimeAdmission {
                aggregate,
                weights: weight_admission,
                session: session_admission,
            },
            observed_weight_bytes,
            device_registry_id: ctx.device.registryID(),
        })
    }

    pub fn config(&self) -> &MuseGlimmerConfig {
        self.weights.config()
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
        self.session
            .as_ref()
            .map_or(0, MuseGlimmerTextSession::observed_allocation_delta)
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
        let forward = MuseGlimmerTextForward::new(ctx, &self.weights)?;
        let session = self.session.take().ok_or_else(|| {
            MuseGlimmerRuntimeError::Invalid("loaded model session was consumed".into())
        })?;
        Ok(MuseGlimmerTextRunner { forward, session })
    }
}

pub struct MuseGlimmerTextRunner<'ctx, 'model> {
    forward: MuseGlimmerTextForward<'ctx, 'model>,
    session: MuseGlimmerTextSession,
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
        Ok(self.forward.forward_token(token, &mut self.session)?)
    }

    pub fn forward_token_with_post_block_interventions(
        &mut self,
        token: u32,
        interventions: &[PostBlockIntervention<'_>],
    ) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self.forward.forward_token_with_post_block_interventions(
            token,
            interventions,
            &mut self.session,
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
            .forward_token_capture_post_blocks(token, layer_ids, &mut self.session)?)
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
                &mut self.session,
            )?)
    }

    pub fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>, MuseGlimmerRuntimeError> {
        Ok(self.forward.prefill(tokens, &mut self.session)?)
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
            .capture_fresh_lens_prompt(tokens, target_block, &mut self.session)?)
    }

    /// Capture block input, post-attention, and post-block coordinates for a
    /// sorted unique set of nonzero blocks in one fresh scalar prompt pass.
    pub fn capture_fresh_lens_prompt_blocks(
        &mut self,
        tokens: &[u32],
        target_blocks: &[u32],
    ) -> Result<MuseGlimmerLensCaptureBank, MuseGlimmerRuntimeError> {
        Ok(self.forward.capture_fresh_lens_prompt_blocks(
            tokens,
            target_blocks,
            &mut self.session,
        )?)
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
            .prefill_with_command_checkpoint(tokens, &mut self.session, || {
                checkpoint().map_err(MuseGlimmerTextSessionError::Checkpoint)
            })?)
    }

    pub fn reset(&mut self) -> Result<(), MuseGlimmerRuntimeError> {
        Ok(self.session.reset()?)
    }
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerRuntimeError> {
    Err(MuseGlimmerRuntimeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::muse_glimmer::MuseGlimmerConfig;

    #[test]
    fn request_shaped_session_plan_matches_capacity() {
        let ctx = MetalContext::new().unwrap();
        let config = MuseGlimmerConfig::unsloth_release_reference();
        let geometry = MuseGlimmerTextGeometry::from_config(&config, 257).unwrap();
        let plan = MuseGlimmerTextSessionMemoryPlan::for_geometry(&ctx, &geometry).unwrap();
        assert!(plan.priced_upper_bytes() >= plan.logical_bytes());
        assert_eq!(geometry.capacity(), 257);
    }
}
