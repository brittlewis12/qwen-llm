//! Bounded coordinates, rules, and selected-token targets for Muse lenses.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, RmsNormVjpRule, SwiGluVjpRule,
    encode_get_rows_f32,
};
use crate::muse_glimmer_residency::{MuseGlimmerMetalWeights, MuseGlimmerResidencyError};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

pub const MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS: usize = 16;
pub const MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS: usize = 256;

/// Residual-stream coordinates relative to one selected Muse block.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MuseGlimmerLensCoordinate {
    /// Input to the block. For block `b > 0`, this is exactly block `b - 1`'s
    /// [`Self::PostBlockResidual`] coordinate.
    InputResidual,
    /// Residual after the normalized attention branch has been added.
    PostAttentionResidual,
    /// Residual after the normalized FFN branch has been added; this is the
    /// Hugging Face block-output coordinate used as a Muse lens layer.
    PostBlockResidual,
}

impl MuseGlimmerLensCoordinate {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InputResidual => "input_residual",
            Self::PostAttentionResidual => "post_attention_residual",
            Self::PostBlockResidual => "post_block_residual",
        }
    }
}

/// RMSNorm sites whose distinction defines Muse J/R transport.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MuseGlimmerRmsNormSite {
    ResidualPreAttention,
    AttentionBranchPostNorm,
    ResidualPreFeedForward,
    FeedForwardBranchPostNorm,
    AttentionQuery,
    AttentionKey,
}

impl MuseGlimmerRmsNormSite {
    const fn is_attention_internal(self) -> bool {
        matches!(self, Self::AttentionQuery | Self::AttentionKey)
    }
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MuseGlimmerAttentionVjpRule {
    OrdinaryJacobian,
}

/// Stable Muse lens rule identifiers.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MuseGlimmerLensRule {
    /// Ordinary activation Jacobian throughout the block.
    J,
    /// Detached RMS scale at residual/branch RMSNorm sites and the existing
    /// RelP identity/half SwiGLU rule. Attention internals stay ordinary.
    R,
}

impl MuseGlimmerLensRule {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::J => "J",
            Self::R => "R",
        }
    }

    pub const fn rms_norm_rule(self, site: MuseGlimmerRmsNormSite) -> RmsNormVjpRule {
        match (self, site.is_attention_internal()) {
            (Self::R, false) => RmsNormVjpRule::RelpDetachedScale,
            _ => RmsNormVjpRule::Jacobian,
        }
    }

    pub const fn swiglu_rule(self) -> SwiGluVjpRule {
        match self {
            Self::J => SwiGluVjpRule::Jacobian,
            Self::R => SwiGluVjpRule::RelpIdentityHalf,
        }
    }

    pub const fn attention_rule(self) -> MuseGlimmerAttentionVjpRule {
        MuseGlimmerAttentionVjpRule::OrdinaryJacobian
    }
}

/// The selected-token linear score used as a target before final softcapping.
/// Its covector is `logit_scale * output_norm_gamma * output_row`; the
/// residual-dependent RMS denominator and nonlinear final softcap are absent.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum MuseGlimmerLensScore {
    PreSoftcapSelectedToken,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerLensCapture {
    target_block: u32,
    token_ids: Vec<u32>,
    hidden_size: usize,
    input_residuals: Vec<f32>,
    post_attention_residuals: Vec<f32>,
    post_block_residuals: Vec<f32>,
}

impl MuseGlimmerLensCapture {
    pub(crate) fn new(
        target_block: u32,
        token_ids: Vec<u32>,
        hidden_size: usize,
        input_residuals: Vec<f32>,
        post_attention_residuals: Vec<f32>,
        post_block_residuals: Vec<f32>,
    ) -> Self {
        Self {
            target_block,
            token_ids,
            hidden_size,
            input_residuals,
            post_attention_residuals,
            post_block_residuals,
        }
    }

    pub fn target_block(&self) -> u32 {
        self.target_block
    }

    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn n_tokens(&self) -> usize {
        self.token_ids.len()
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn input_residuals(&self) -> &[f32] {
        &self.input_residuals
    }

    pub fn post_attention_residuals(&self) -> &[f32] {
        &self.post_attention_residuals
    }

    pub fn post_block_residuals(&self) -> &[f32] {
        &self.post_block_residuals
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerSelectedTokenCovectors {
    token_ids: Vec<u32>,
    hidden_size: usize,
    logit_scale: f32,
    values: Vec<f32>,
}

impl MuseGlimmerSelectedTokenCovectors {
    pub fn token_ids(&self) -> &[u32] {
        &self.token_ids
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn logit_scale(&self) -> f32 {
        self.logit_scale
    }

    /// Token-major `logit_scale * output_norm_gamma * output_row`, `[K,H]`.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    pub fn token_values(&self, slot: usize) -> Option<&[f32]> {
        let start = slot.checked_mul(self.hidden_size)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MuseGlimmerLensError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Residency(#[from] MuseGlimmerResidencyError),
    #[error("invalid Muse Glimmer lens contract: {0}")]
    Invalid(String),
    #[error("Muse Glimmer lens command buffer failed: {0}")]
    CommandBuffer(String),
}

/// Extract bounded selected-token covectors directly from resident Muse
/// output weights. This does not materialize the vocabulary-sized output.
pub fn muse_glimmer_selected_token_covectors(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalWeights,
    token_ids: &[u32],
) -> Result<MuseGlimmerSelectedTokenCovectors, MuseGlimmerLensError> {
    weights.validate_context(ctx)?;
    let config = weights.config();
    if token_ids.is_empty() || token_ids.len() > MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS {
        return invalid(format!(
            "selected-token count must be in 1..={MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS}, got {}",
            token_ids.len()
        ));
    }
    if let Some(&token) = token_ids.iter().find(|&&token| token >= config.vocab_size) {
        return invalid(format!(
            "token {token} is outside vocabulary {}",
            config.vocab_size
        ));
    }
    let hidden_size = config.hidden_size as usize;
    let output = weights.require_tensor("output.weight")?;
    if output.shape.as_slice() != [hidden_size as u64, config.vocab_size as u64] {
        return invalid(format!("unexpected output weight shape {:?}", output.shape));
    }
    let output_norm = weights.require_tensor("output_norm.weight")?;
    if output_norm.dtype != GgmlType::F32 || output_norm.shape.as_slice() != [hidden_size as u64] {
        return invalid(format!(
            "output norm must be F32 [{hidden_size}], got {:?} {:?}",
            output_norm.dtype, output_norm.shape
        ));
    }

    let ids = token_ids.iter().map(|&id| id as i32).collect::<Vec<_>>();
    let ids = MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&ids),
        vec![token_ids.len() as u64],
        GgmlType::I32,
    )?;
    let rows = MetalTensor::zeros_f32(ctx, vec![hidden_size as u64, token_ids.len() as u64])?;
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| MuseGlimmerLensError::CommandBuffer("allocation failed".into()))?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = encode_get_rows_f32(
        ctx,
        &encoder,
        output,
        &ids,
        &rows,
        token_ids.len(),
        hidden_size,
    );
    encoder.end();
    encode_result?;
    command.commit();
    command.waitUntilCompleted();
    let status = command.status();
    let command_error = command.error().map(|error| error.to_string());
    if status != MTLCommandBufferStatus::Completed || command_error.is_some() {
        return Err(MuseGlimmerLensError::CommandBuffer(format!(
            "status={status:?}, error={command_error:?}"
        )));
    }

    let mut values = read_f32(&rows);
    let gamma = read_f32(output_norm);
    fold_selected_token_covectors(&mut values, &gamma, config.logit_scale, hidden_size)?;
    Ok(MuseGlimmerSelectedTokenCovectors {
        token_ids: token_ids.to_vec(),
        hidden_size,
        logit_scale: config.logit_scale,
        values,
    })
}

fn fold_selected_token_covectors(
    rows: &mut [f32],
    gamma: &[f32],
    logit_scale: f32,
    hidden_size: usize,
) -> Result<(), MuseGlimmerLensError> {
    if hidden_size == 0
        || gamma.len() != hidden_size
        || !rows.len().is_multiple_of(hidden_size)
        || !logit_scale.is_finite()
    {
        return invalid("invalid selected-token covector dimensions or scale");
    }
    for (index, value) in rows.iter_mut().enumerate() {
        *value *= gamma[index % hidden_size] * logit_scale;
        if !value.is_finite() {
            return invalid(format!(
                "non-finite selected-token covector at index {index}"
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_capture_request(
    token_count: usize,
    target_block: u32,
    layer_count: usize,
    next_position: usize,
    capacity: usize,
) -> Result<(), MuseGlimmerLensError> {
    if next_position != 0 {
        return invalid(format!(
            "capture requires a fresh session at position zero, got {next_position}"
        ));
    }
    if token_count == 0 || token_count > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
        return invalid(format!(
            "capture prompt length must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got {token_count}"
        ));
    }
    if token_count > capacity {
        return invalid(format!(
            "capture of {token_count} tokens exceeds session capacity {capacity}"
        ));
    }
    if target_block == 0 || target_block as usize >= layer_count {
        return invalid(format!(
            "target block must be in 1..{layer_count}, got {target_block}"
        ));
    }
    Ok(())
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let mut values = vec![0.0_f32; tensor.n_elements() as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.as_mut_ptr(),
            values.len(),
        );
    }
    values
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerLensError> {
    Err(MuseGlimmerLensError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::muse_glimmer_residency::MuseGlimmerMetalWeightPlan;
    use crate::muse_glimmer_text_session::{MuseGlimmerTextForward, MuseGlimmerTextSession};

    #[test]
    fn muse_rule_identifiers_pin_rms_swiglu_and_attention_semantics() {
        assert_eq!(MuseGlimmerLensRule::J.as_str(), "J");
        assert_eq!(MuseGlimmerLensRule::R.as_str(), "R");
        assert_eq!(
            MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
            RmsNormVjpRule::RelpDetachedScale
        );
        assert_eq!(
            MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
            RmsNormVjpRule::Jacobian
        );
        assert_eq!(
            MuseGlimmerLensRule::R.swiglu_rule(),
            SwiGluVjpRule::RelpIdentityHalf
        );
        assert_eq!(
            MuseGlimmerLensRule::R.attention_rule(),
            MuseGlimmerAttentionVjpRule::OrdinaryJacobian
        );
    }

    #[test]
    fn coordinate_names_and_selected_score_formula_are_explicit() {
        assert_eq!(
            MuseGlimmerLensCoordinate::PostBlockResidual.as_str(),
            "post_block_residual"
        );
        let mut rows = vec![2.0, -3.0, 4.0, 5.0];
        fold_selected_token_covectors(&mut rows, &[0.5, 2.0], 0.25, 2).unwrap();
        assert_eq!(rows, [0.25, -1.5, 0.5, 2.5]);
    }

    #[test]
    fn capture_contract_is_fresh_scalar_and_bounded() {
        validate_capture_request(16, 1, 52, 0, 16).unwrap();
        assert!(validate_capture_request(17, 1, 52, 0, 17).is_err());
        assert!(validate_capture_request(1, 1, 52, 1, 16).is_err());
        assert!(validate_capture_request(1, 0, 52, 0, 16).is_err());
        assert!(validate_capture_request(1, 52, 52, 0, 16).is_err());
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn real_q8_capture_and_selected_covector_smoke() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf".into()
        });
        let gguf = GgufFile::open(path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan =
            MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 weights");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let covectors =
            muse_glimmer_selected_token_covectors(&ctx, &weights, &[weights.config().eos_token_id])
                .expect("extract selected Muse target");
        assert_eq!(
            covectors.values().len(),
            weights.config().hidden_size as usize
        );
        assert!(covectors.values().iter().all(|value| value.is_finite()));

        let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), 1)
            .expect("allocate bounded Muse session");
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let capture = forward
            .capture_fresh_lens_prompt(&[weights.config().bos_token_id], 1, &mut session)
            .expect("capture Muse residual coordinates");
        assert_eq!(capture.n_tokens(), 1);
        assert_eq!(
            capture.input_residuals().len(),
            weights.config().hidden_size as usize
        );
        assert!(
            capture
                .post_block_residuals()
                .iter()
                .all(|value| value.is_finite())
        );
    }
}
