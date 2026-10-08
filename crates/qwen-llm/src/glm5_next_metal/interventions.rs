//! Residual-writer interventions and captures inside one GLM-5.3 decode step
//! (`qwen-lens run` raw directions).
//!
//! Each site is a single `hidden_size` F32 vector before GLM's four-stream
//! mHC post mixes it into the residual, so an operation there edits exactly
//! one module's contribution. A projection with coefficient `a` and a unit
//! direction `v` at a module's output is the activation form of the weight
//! edit `W' = W - a v v^T W` on that module's output projection. Every
//! operation and capture runs in caller order within its (site, block);
//! interventions apply on every forward they are given (prompt and
//! generated tokens alike), and only through the serial decode path.

use super::*;
use crate::metal::{PostBlockIntervention, encode_post_block_intervention_f32};

/// A residual writer's output inside one block.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Glm5NextSite {
    /// The token embedding row before the streams are repeated (block 0).
    Embedding,
    /// The KDA or MLA output projection, before the attention mHC post.
    MixerOutput,
    /// The routed experts' weighted sum (MoE blocks).
    RoutedExpertsOutput,
    /// The shared expert's down projection, before it joins the routed sum
    /// (MoE blocks).
    SharedExpertOutput,
    /// The FFN output before the FFN mHC post: the dense down projection
    /// (blocks 0-2) or routed plus shared.
    FfnOutput,
}

/// When a capture reads its site relative to that site's operations.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Glm5NextCapturePoint {
    BeforeOperations,
    AfterOperations,
}

/// One operation at a module site of `op`'s layer (the block index).
#[derive(Clone, Copy)]
pub struct Glm5NextModuleIntervention<'a> {
    pub site: Glm5NextSite,
    pub op: PostBlockIntervention<'a>,
}

/// One request to copy a site's vector to the host during the step.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub struct Glm5NextSiteCapture {
    pub site: Glm5NextSite,
    pub block: usize,
    pub point: Glm5NextCapturePoint,
}

/// Interventions and captures for one step, with the capture observer.
pub(super) struct Hooks<'h, 'a> {
    pub(super) module: &'h [Glm5NextModuleIntervention<'a>],
    pub(super) captures: &'h [Glm5NextSiteCapture],
    pub(super) observer: &'h mut dyn FnMut(Glm5NextSiteCapture, &[f32]),
}

impl Hooks<'_, '_> {
    pub(super) fn is_empty(&self) -> bool {
        self.module.is_empty() && self.captures.is_empty()
    }
}

pub(super) fn op_layer(op: &PostBlockIntervention<'_>) -> usize {
    match op {
        PostBlockIntervention::Fixed { layer, .. }
        | PostBlockIntervention::ResidualL2Relative { layer, .. }
        | PostBlockIntervention::Projection { layer, .. }
        | PostBlockIntervention::SourceToTarget { layer, .. } => *layer as usize,
    }
}

fn op_vectors<'t>(op: &PostBlockIntervention<'t>) -> Vec<&'t MetalTensor> {
    match op {
        PostBlockIntervention::Fixed { direction, .. }
        | PostBlockIntervention::ResidualL2Relative { direction, .. }
        | PostBlockIntervention::Projection { direction, .. } => vec![direction],
        PostBlockIntervention::SourceToTarget { source, target, .. } => vec![source, target],
    }
}

fn op_coefficient(op: &PostBlockIntervention<'_>) -> f32 {
    match op {
        PostBlockIntervention::Fixed { coefficient, .. }
        | PostBlockIntervention::ResidualL2Relative { coefficient, .. }
        | PostBlockIntervention::Projection { coefficient, .. }
        | PostBlockIntervention::SourceToTarget { coefficient, .. } => *coefficient,
    }
}

impl Glm5NextSession<'_> {
    /// Whether `site` exists in `block`.
    fn site_exists(&self, site: Glm5NextSite, block: usize) -> bool {
        let Some(kind) = self.weights.config.blocks.get(block) else {
            return false;
        };
        match site {
            Glm5NextSite::Embedding => block == 0,
            Glm5NextSite::MixerOutput | Glm5NextSite::FfnOutput => true,
            Glm5NextSite::RoutedExpertsOutput | Glm5NextSite::SharedExpertOutput => {
                kind.ffn == FfnKind::Moe
            }
        }
    }

    /// Refuses an operation or capture list before any work: every site
    /// exists in its block; directions are F32 `hidden_size` vectors in
    /// bounds; coefficients are finite and nonzero (a zero dose is an
    /// omitted operation). Directions cannot alias session buffers: those
    /// are private, and no public method returns one.
    pub(super) fn validate_hooks(&self, hooks: &Hooks<'_, '_>) -> Result<()> {
        let h = self.weights.config.hidden_size as u64;
        for (index, m) in hooks.module.iter().enumerate() {
            let block = op_layer(&m.op);
            if !self.site_exists(m.site, block) {
                return invalid(format!(
                    "intervention {index}: site {:?} does not exist in block {block}",
                    m.site
                ));
            }
            let coefficient = op_coefficient(&m.op);
            if !coefficient.is_finite() || coefficient == 0.0 {
                return invalid(format!(
                    "intervention {index}: coefficient {coefficient} must be finite and nonzero (omit a zero dose)"
                ));
            }
            for vector in op_vectors(&m.op) {
                let end = vector.offset.checked_add(h * 4);
                if vector.dtype != GgmlType::F32
                    || vector.n_elements() != h
                    || !vector.offset.is_multiple_of(4)
                    || end.is_none_or(|end| end > vector.buffer.length() as u64)
                {
                    return invalid(format!(
                        "intervention {index}: directions must be aligned F32 [{h}] in bounds"
                    ));
                }
            }
        }
        for (index, capture) in hooks.captures.iter().enumerate() {
            if !self.site_exists(capture.site, capture.block) {
                return invalid(format!(
                    "capture {index}: site {:?} does not exist in block {}",
                    capture.site, capture.block
                ));
            }
        }
        Ok(())
    }

    /// Captures due before, then operations, then captures due after, at
    /// one (site, block). A capture ends the command (so the GPU has written
    /// the vector), reads it, and begins the next command.
    pub(super) fn apply_site_hooks(
        &self,
        ctx: &MetalContext,
        command: &mut objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
        enc: &mut KernelEncoder,
        hooks: &mut Hooks<'_, '_>,
        site: Glm5NextSite,
        block: usize,
        tensor: &MetalTensor,
    ) -> Result<()> {
        if hooks.is_empty() {
            return Ok(());
        }
        self.capture_site(
            ctx,
            command,
            enc,
            hooks,
            site,
            block,
            tensor,
            Glm5NextCapturePoint::BeforeOperations,
        )?;
        for m in hooks
            .module
            .iter()
            .filter(|m| m.site == site && op_layer(&m.op) == block)
        {
            encode_post_block_intervention_f32(ctx, enc, tensor, &m.op)?;
        }
        self.capture_site(
            ctx,
            command,
            enc,
            hooks,
            site,
            block,
            tensor,
            Glm5NextCapturePoint::AfterOperations,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn capture_site(
        &self,
        ctx: &MetalContext,
        command: &mut objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn MTLCommandBuffer>>,
        enc: &mut KernelEncoder,
        hooks: &mut Hooks<'_, '_>,
        site: Glm5NextSite,
        block: usize,
        tensor: &MetalTensor,
        point: Glm5NextCapturePoint,
    ) -> Result<()> {
        let wanted: Vec<Glm5NextSiteCapture> = hooks
            .captures
            .iter()
            .copied()
            .filter(|c| c.site == site && c.block == block && c.point == point)
            .collect();
        if wanted.is_empty() {
            return Ok(());
        }
        let next = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let finished = std::mem::replace(command, next);
        // The ended encoder is swapped out for one on the next command.
        std::mem::replace(enc, KernelEncoder::begin(command)).end();
        finished.commit();
        wait_completed(&finished)?;
        let values = read_f32(tensor)?;
        for capture in wanted {
            (hooks.observer)(capture, &values);
        }
        Ok(())
    }

    /// Like [`Self::forward`], applying `module` operations at their sites
    /// and reporting `captures` (serial decode path). Everything is
    /// validated before the token executes.
    pub fn forward_with_interventions(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        module: &[Glm5NextModuleIntervention<'_>],
        captures: &[Glm5NextSiteCapture],
        observer: &mut dyn FnMut(Glm5NextSiteCapture, &[f32]),
    ) -> Result<Vec<f32>> {
        let mut hooks = Hooks {
            module,
            captures,
            observer,
        };
        self.validate_hooks(&hooks)?;
        Ok(self
            .step_hooked(ctx, token, true, &[], &mut |_, _, _| {}, &mut hooks)?
            .expect("logits requested"))
    }
}
