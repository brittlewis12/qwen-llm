//! Reusable post-PLE Qwen3.8-Flash-Next block composition.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_copy_offset_f32,
};
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig, Qwen4ExpError};
use crate::qwen4exp_gdn::{
    GatedDeltaNetMetalGeometry, GatedDeltaNetMetalWeights, GatedDeltaNetMetalWorkspace,
    Qwen4ExpGdnError, encode_gated_delta_net,
};
use crate::qwen4exp_layer_zero::Qwen4ExpResidualMetalWeights;
use crate::qwen4exp_metal::{
    GatedResidualMetalReadWeights, GatedResidualMetalScratch, Qwen4ExpMetalError,
    encode_gated_residual_mix, validate_and_preflight_gated_residual_mix,
};
use crate::qwen4exp_moe::{
    Qwen4ExpMoeError, Qwen4ExpMoeMetalGeometry, Qwen4ExpMoeMetalWeights, Qwen4ExpMoeMetalWorkspace,
    encode_qwen4exp_moe,
};
use crate::qwen4exp_qsa::{
    Qwen4ExpQsaError, QwenSparseAttentionMetalGeometry, QwenSparseAttentionMetalWeights,
    QwenSparseAttentionMetalWorkspace, encode_qwen_sparse_attention_text,
};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice, MTLResource};

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpPostPleBlockError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Residual(#[from] Qwen4ExpMetalError),
    #[error(transparent)]
    GatedDeltaNet(#[from] Qwen4ExpGdnError),
    #[error(transparent)]
    QwenSparseAttention(#[from] Qwen4ExpQsaError),
    #[error(transparent)]
    Moe(#[from] Qwen4ExpMoeError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next post-PLE block contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next post-PLE block command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Qwen4ExpPostPleMixerMetalGeometry {
    GatedDeltaNet(GatedDeltaNetMetalGeometry),
    QwenSparseAttention(QwenSparseAttentionMetalGeometry),
}

impl Qwen4ExpPostPleMixerMetalGeometry {
    pub fn hidden_size(self) -> usize {
        match self {
            Self::GatedDeltaNet(geometry) => geometry.hidden_size(),
            Self::QwenSparseAttention(geometry) => geometry.hidden_size(),
        }
    }

    pub fn kind(self) -> MixerKind {
        match self {
            Self::GatedDeltaNet(_) => MixerKind::GatedDeltaNet,
            Self::QwenSparseAttention(_) => MixerKind::QwenSparseAttention,
        }
    }

    pub fn qsa(self) -> Option<QwenSparseAttentionMetalGeometry> {
        match self {
            Self::QwenSparseAttention(geometry) => Some(geometry),
            Self::GatedDeltaNet(_) => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpPostPleBlockMetalGeometry {
    layer: u32,
    branch_count: usize,
    hidden_size: usize,
    low_rank: usize,
    eps: f32,
    mixer: Qwen4ExpPostPleMixerMetalGeometry,
    moe: Qwen4ExpMoeMetalGeometry,
}

impl Qwen4ExpPostPleBlockMetalGeometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer: u32,
        branch_count: usize,
        hidden_size: usize,
        low_rank: usize,
        eps: f32,
        mixer: Qwen4ExpPostPleMixerMetalGeometry,
        moe: Qwen4ExpMoeMetalGeometry,
    ) -> Result<Self, Qwen4ExpPostPleBlockError> {
        let geometry = Self {
            layer,
            branch_count,
            hidden_size,
            low_rank,
            eps,
            mixer,
            moe,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(
        config: &Qwen4ExpConfig,
        layer: u32,
        qsa_capacity: Option<usize>,
    ) -> Result<Self, Qwen4ExpPostPleBlockError> {
        config.validate()?;
        if layer >= config.layer_count {
            return invalid(format!(
                "layer {layer} is outside {} layers",
                config.layer_count
            ));
        }
        if config
            .ple
            .as_ref()
            .is_some_and(|ple| ple.layers.contains(&layer))
        {
            return invalid(format!(
                "layer {layer} has a PLE transform and is not a post-PLE-only block"
            ));
        }
        let mixer = match config.mixer_kind(layer) {
            Some(MixerKind::GatedDeltaNet) => {
                if qsa_capacity.is_some() {
                    return invalid("GDN block must not receive QSA capacity");
                }
                Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(
                    GatedDeltaNetMetalGeometry::from_config(config)?,
                )
            }
            Some(MixerKind::QwenSparseAttention) => {
                let capacity = qsa_capacity.ok_or_else(|| {
                    Qwen4ExpPostPleBlockError::Invalid(
                        "QSA block requires an explicit cache capacity".into(),
                    )
                })?;
                Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(
                    QwenSparseAttentionMetalGeometry::from_config(config, layer, capacity)?,
                )
            }
            None => return invalid(format!("layer {layer} has no mixer schedule entry")),
        };
        Self::new(
            layer,
            config.hyper_connection.count as usize,
            config.hidden_size as usize,
            config.hyper_connection.low_rank as usize,
            config.rms_norm_eps,
            mixer,
            Qwen4ExpMoeMetalGeometry::from_config(config)?,
        )
    }

    pub fn layer(self) -> u32 {
        self.layer
    }

    pub fn branch_count(self) -> usize {
        self.branch_count
    }

    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub fn low_rank(self) -> usize {
        self.low_rank
    }

    pub fn eps(self) -> f32 {
        self.eps
    }

    pub fn mixer(self) -> Qwen4ExpPostPleMixerMetalGeometry {
        self.mixer
    }

    pub fn moe(self) -> Qwen4ExpMoeMetalGeometry {
        self.moe
    }

    pub fn hyper_width(self) -> usize {
        self.branch_count
            .checked_mul(self.hidden_size)
            .expect("validated post-PLE hyper width")
    }

    fn validate(self) -> Result<(), Qwen4ExpPostPleBlockError> {
        if self.branch_count == 0 || self.hidden_size == 0 || self.low_rank == 0 {
            return invalid("branch, hidden, and low-rank dimensions must be nonzero");
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return invalid("RMS epsilon must be finite and positive");
        }
        if self.mixer.hidden_size() != self.hidden_size
            || self.moe.hidden_size() != self.hidden_size
        {
            return invalid("residual, mixer, and MoE hidden dimensions differ");
        }
        let hyper = self
            .branch_count
            .checked_mul(self.hidden_size)
            .ok_or_else(|| {
                Qwen4ExpPostPleBlockError::Invalid("hyper-residual width overflow".into())
            })?;
        for (name, value) in [
            ("branch count", self.branch_count),
            ("hidden size", self.hidden_size),
            ("low rank", self.low_rank),
            ("hyper width", hyper),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("{name} {value} exceeds u32"));
            }
        }
        hyper.checked_mul(4).ok_or_else(|| {
            Qwen4ExpPostPleBlockError::Invalid("hyper-residual byte count overflow".into())
        })?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum Qwen4ExpPostPleMixerMetalWeights<'a> {
    GatedDeltaNet(GatedDeltaNetMetalWeights<'a>),
    QwenSparseAttention(QwenSparseAttentionMetalWeights<'a>),
}

impl Qwen4ExpPostPleMixerMetalWeights<'_> {
    pub fn geometry(self) -> Qwen4ExpPostPleMixerMetalGeometry {
        match self {
            Self::GatedDeltaNet(weights) => {
                Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(weights.geometry)
            }
            Self::QwenSparseAttention(weights) => {
                Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(weights.geometry)
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpPostPleBlockMetalWeights<'a> {
    pub geometry: Qwen4ExpPostPleBlockMetalGeometry,
    pub attention_residual: Qwen4ExpResidualMetalWeights<'a>,
    pub mixer: Qwen4ExpPostPleMixerMetalWeights<'a>,
    pub ffn_residual: Qwen4ExpResidualMetalWeights<'a>,
    pub moe: Qwen4ExpMoeMetalWeights<'a>,
}

impl<'a> Qwen4ExpPostPleBlockMetalWeights<'a> {
    pub fn bind(
        weights: &'a Qwen4ExpMetalWeights,
        layer: u32,
        qsa_capacity: Option<usize>,
    ) -> Result<Self, Qwen4ExpPostPleBlockError> {
        let geometry =
            Qwen4ExpPostPleBlockMetalGeometry::from_config(weights.config(), layer, qsa_capacity)?;
        let mixer = match geometry.mixer {
            Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(_) => {
                Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(GatedDeltaNetMetalWeights::bind(
                    weights, layer,
                )?)
            }
            Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(qsa_geometry) => {
                Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(
                    QwenSparseAttentionMetalWeights::bind(weights, layer, qsa_geometry.capacity())?,
                )
            }
        };
        Ok(Self {
            geometry,
            attention_residual: bind_residual(weights, layer, "attn")?,
            mixer,
            ffn_residual: bind_residual(weights, layer, "ffn")?,
            moe: Qwen4ExpMoeMetalWeights::bind(weights, layer)?,
        })
    }
}

fn bind_residual<'a>(
    weights: &'a Qwen4ExpMetalWeights,
    layer: u32,
    role: &str,
) -> Result<Qwen4ExpResidualMetalWeights<'a>, Qwen4ExpPostPleBlockError> {
    let prefix = format!("blk.{layer}.hc_{role}");
    Ok(Qwen4ExpResidualMetalWeights {
        read: GatedResidualMetalReadWeights {
            norm: weights.require_tensor(&format!("{prefix}_norm.weight"))?,
            down: weights.require_tensor(&format!("{prefix}_down.weight"))?,
            up: weights.require_tensor(&format!("{prefix}_up.weight"))?,
        },
        inject: weights.require_tensor(&format!("{prefix}_inject.weight"))?,
    })
}

pub enum Qwen4ExpPostPleMixerMetalWorkspace {
    GatedDeltaNet(Box<GatedDeltaNetMetalWorkspace>),
    QwenSparseAttention(Box<QwenSparseAttentionMetalWorkspace>),
}

impl Qwen4ExpPostPleMixerMetalWorkspace {
    fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpPostPleMixerMetalGeometry,
    ) -> Result<Self, Qwen4ExpPostPleBlockError> {
        Ok(match geometry {
            Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(geometry) => {
                Self::GatedDeltaNet(Box::new(GatedDeltaNetMetalWorkspace::new(ctx, geometry)?))
            }
            Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(geometry) => {
                Self::QwenSparseAttention(Box::new(QwenSparseAttentionMetalWorkspace::new(
                    ctx, geometry,
                )?))
            }
        })
    }

    pub fn geometry(&self) -> Qwen4ExpPostPleMixerMetalGeometry {
        match self {
            Self::GatedDeltaNet(workspace) => {
                Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(workspace.geometry())
            }
            Self::QwenSparseAttention(workspace) => {
                Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(workspace.geometry())
            }
        }
    }

    pub fn committed_length(&self) -> Option<usize> {
        match self {
            Self::GatedDeltaNet(_) => None,
            Self::QwenSparseAttention(workspace) => Some(workspace.committed_length()),
        }
    }

    fn reset(&mut self) -> Result<(), Qwen4ExpPostPleBlockError> {
        match self {
            Self::GatedDeltaNet(workspace) => workspace.reset()?,
            Self::QwenSparseAttention(workspace) => workspace.reset()?,
        }
        Ok(())
    }

    fn release_after(&mut self) -> Result<(), Qwen4ExpPostPleBlockError> {
        match self {
            Self::GatedDeltaNet(workspace) => workspace.release_after()?,
            Self::QwenSparseAttention(workspace) => workspace.release_after()?,
        }
        Ok(())
    }

    unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpPostPleBlockError> {
        match self {
            Self::GatedDeltaNet(workspace) => unsafe { workspace.abandon_uncommitted()? },
            Self::QwenSparseAttention(workspace) => unsafe { workspace.abandon_uncommitted()? },
        }
        Ok(())
    }
}

pub struct Qwen4ExpPostPleBlockMetalWorkspace {
    geometry: Qwen4ExpPostPleBlockMetalGeometry,
    mixer_output: MetalTensor,
    moe_output: MetalTensor,
    residual: GatedResidualMetalScratch,
    mixer: Qwen4ExpPostPleMixerMetalWorkspace,
    moe: Qwen4ExpMoeMetalWorkspace,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
    encode_failed: bool,
}

impl Qwen4ExpPostPleBlockMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpPostPleBlockMetalGeometry,
    ) -> Result<Self, Qwen4ExpPostPleBlockError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            mixer_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            moe_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            residual: GatedResidualMetalScratch::new(
                ctx,
                geometry.branch_count,
                geometry.hidden_size,
                geometry.low_rank,
            )?,
            mixer: Qwen4ExpPostPleMixerMetalWorkspace::new(ctx, geometry.mixer)?,
            moe: Qwen4ExpMoeMetalWorkspace::new(ctx, geometry.moe)?,
            active_command: None,
            state_poisoned: false,
            encode_failed: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpPostPleBlockMetalGeometry {
        self.geometry
    }

    pub fn mixer_committed_length(&self) -> Option<usize> {
        self.mixer.committed_length()
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpPostPleBlockError> {
        self.require_idle()?;
        self.mixer.reset()?;
        self.moe.reset()?;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpPostPleBlockError> {
        let Some(command) = self.active_command.clone() else {
            return Ok(());
        };
        let status = command.status();
        if matches!(
            status,
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return invalid(format!(
                "workspace owner is not committed (status {status:?}); commit it or abandon the uncommitted command"
            ));
        }
        command.waitUntilCompleted();
        let status = command.status();
        let command_error = command.error().map(|error| error.to_string());
        let mut child_errors = Vec::new();
        if let Err(error) = self.residual.release_after() {
            child_errors.push(format!("residual: {error}"));
        }
        if let Err(error) = self.mixer.release_after() {
            child_errors.push(format!("mixer: {error}"));
        }
        if let Err(error) = self.moe.release_after() {
            child_errors.push(format!("MoE: {error}"));
        }
        self.active_command = None;
        if status == MTLCommandBufferStatus::Completed
            && command_error.is_none()
            && child_errors.is_empty()
            && !self.encode_failed
        {
            self.encode_failed = false;
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpPostPleBlockError::CommandBuffer(format!(
                "status={status:?}, error={command_error:?}, encode_failed={}, children={child_errors:?}",
                self.encode_failed
            )))
        }
    }

    /// Release a block from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may mutate mixer state after another
    /// token acquires this workspace.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpPostPleBlockError> {
        let Some(command) = self.active_command.as_ref() else {
            return Ok(());
        };
        let status = command.status();
        if status != MTLCommandBufferStatus::NotEnqueued {
            return invalid(format!(
                "only a NotEnqueued workspace owner can be abandoned, got {status:?}"
            ));
        }
        let mut child_errors = Vec::new();
        if let Err(error) = unsafe { self.residual.abandon_uncommitted() } {
            child_errors.push(format!("residual: {error}"));
        }
        if let Err(error) = unsafe { self.mixer.abandon_uncommitted() } {
            child_errors.push(format!("mixer: {error}"));
        }
        if let Err(error) = unsafe { self.moe.abandon_uncommitted() } {
            child_errors.push(format!("MoE: {error}"));
        }
        if !child_errors.is_empty() {
            return invalid(format!(
                "could not abandon every post-PLE child: {child_errors:?}"
            ));
        }
        self.active_command = None;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    #[cfg(test)]
    pub(crate) fn mixer_output_tensor(&self) -> &MetalTensor {
        &self.mixer_output
    }

    #[cfg(test)]
    pub(crate) fn moe_output_tensor(&self) -> &MetalTensor {
        &self.moe_output
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpPostPleBlockError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "consume the block output in its owning command, then release the workspace"]
pub struct Qwen4ExpPostPleBlockMetalRead<'workspace, 'resources> {
    workspace: &'workspace mut Qwen4ExpPostPleBlockMetalWorkspace,
    hyper_residual: &'resources MetalTensor,
}

pub struct Qwen4ExpPostPleBlockMetalOutput<'a> {
    workspace: &'a Qwen4ExpPostPleBlockMetalWorkspace,
    hyper_residual: &'a MetalTensor,
}

impl Qwen4ExpPostPleBlockMetalRead<'_, '_> {
    pub fn output(&self) -> Qwen4ExpPostPleBlockMetalOutput<'_> {
        Qwen4ExpPostPleBlockMetalOutput {
            workspace: self.workspace,
            hyper_residual: self.hyper_residual,
        }
    }
}

impl Qwen4ExpPostPleBlockMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hyper_width() as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn layer(&self) -> u32 {
        self.workspace.geometry.layer
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpPostPleBlockError> {
        validate_encoder(ctx, enc)?;
        require_owner(self.workspace, enc)?;
        require_tensor(
            "post-PLE copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hyper_width() as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("post-PLE hyper residual", self.hyper_residual),
                ("post-PLE copied output destination", destination),
            ],
        )?;
        let mut tensors = top_level_tensors(self.workspace, self.hyper_residual);
        tensors.push(("post-PLE copied output destination", destination));
        require_disjoint(&tensors)?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(
            ctx,
            enc,
            self.hyper_residual,
            0,
            destination,
            self.workspace.geometry.hyper_width(),
        )?;
        Ok(())
    }
}

pub fn encode_qwen4exp_post_ple_block<'workspace, 'resources>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    position: usize,
    hyper_residual: &'resources MetalTensor,
    weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
    workspace: &'workspace mut Qwen4ExpPostPleBlockMetalWorkspace,
) -> Result<Qwen4ExpPostPleBlockMetalRead<'workspace, 'resources>, Qwen4ExpPostPleBlockError> {
    validate_encoder(ctx, enc)?;
    validate_and_preflight(ctx, position, hyper_residual, weights, workspace)?;
    reserve_command(workspace, enc)?;
    if let Err(error) = encode_step(ctx, enc, hyper_residual, weights, workspace) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpPostPleBlockMetalRead {
        workspace,
        hyper_residual,
    })
}

pub(crate) fn validate_and_preflight(
    ctx: &MetalContext,
    position: usize,
    hyper_residual: &MetalTensor,
    weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
    workspace: &Qwen4ExpPostPleBlockMetalWorkspace,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if weights.geometry != workspace.geometry {
        return invalid("post-PLE weight and workspace geometry differ");
    }
    if weights.mixer.geometry() != workspace.geometry.mixer
        || workspace.mixer.geometry() != workspace.geometry.mixer
        || weights.moe.geometry != workspace.geometry.moe
        || workspace.moe.geometry() != workspace.geometry.moe
    {
        return invalid("nested mixer or MoE geometry differs from the block");
    }
    match &workspace.mixer {
        Qwen4ExpPostPleMixerMetalWorkspace::QwenSparseAttention(qsa)
            if qsa.committed_length() != position =>
        {
            return invalid(format!(
                "QSA committed length {} differs from token position {position}",
                qsa.committed_length()
            ));
        }
        Qwen4ExpPostPleMixerMetalWorkspace::QwenSparseAttention(qsa)
            if qsa.committed_length() >= qsa.geometry().capacity() =>
        {
            return invalid(format!(
                "QSA token capacity {} is exhausted",
                qsa.geometry().capacity()
            ));
        }
        Qwen4ExpPostPleMixerMetalWorkspace::GatedDeltaNet(_)
        | Qwen4ExpPostPleMixerMetalWorkspace::QwenSparseAttention(_) => {}
    }
    validate_contract(ctx, hyper_residual, weights, workspace)
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_residual: &MetalTensor,
    weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
    workspace: &mut Qwen4ExpPostPleBlockMetalWorkspace,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    let g = workspace.geometry;
    let attention = encode_gated_residual_mix(
        ctx,
        enc,
        hyper_residual,
        &workspace.mixer_output,
        g.eps,
        weights.attention_residual.read,
        weights.attention_residual.inject,
        &mut workspace.residual,
    )?;
    match (weights.mixer, &mut workspace.mixer) {
        (
            Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(weights),
            Qwen4ExpPostPleMixerMetalWorkspace::GatedDeltaNet(mixer),
        ) => {
            let read = encode_gated_delta_net(ctx, enc, attention.mixed(), weights, mixer)?;
            read.output()
                .encode_copy_to(ctx, enc, &workspace.mixer_output)?;
            drop(read);
        }
        (
            Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(weights),
            Qwen4ExpPostPleMixerMetalWorkspace::QwenSparseAttention(mixer),
        ) => {
            let read =
                encode_qwen_sparse_attention_text(ctx, enc, attention.mixed(), weights, mixer)?;
            read.output()
                .encode_copy_to(ctx, enc, &workspace.mixer_output)?;
            drop(read);
        }
        _ => return invalid("mixer weight and workspace variants differ"),
    }
    attention.encode_combine()?;

    let ffn = encode_gated_residual_mix(
        ctx,
        enc,
        hyper_residual,
        &workspace.moe_output,
        g.eps,
        weights.ffn_residual.read,
        weights.ffn_residual.inject,
        &mut workspace.residual,
    )?;
    let moe = encode_qwen4exp_moe(ctx, enc, ffn.mixed(), weights.moe, &mut workspace.moe)?;
    moe.output()
        .encode_copy_to(ctx, enc, &workspace.moe_output)?;
    drop(moe);
    ffn.encode_combine()?;
    Ok(())
}

fn validate_contract(
    ctx: &MetalContext,
    hyper_residual: &MetalTensor,
    weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
    workspace: &Qwen4ExpPostPleBlockMetalWorkspace,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    let g = workspace.geometry;
    validate_and_preflight_gated_residual_mix(
        ctx,
        hyper_residual,
        &workspace.mixer_output,
        g.eps,
        weights.attention_residual.read,
        weights.attention_residual.inject,
        &workspace.residual,
    )?;
    match (weights.mixer, &workspace.mixer) {
        (
            Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(weights),
            Qwen4ExpPostPleMixerMetalWorkspace::GatedDeltaNet(mixer),
        ) => {
            crate::qwen4exp_gdn::validate_contract(
                ctx,
                workspace.residual.mixed_tensor(),
                weights,
                mixer,
            )?;
            crate::qwen4exp_gdn::preflight(ctx, weights)?;
        }
        (
            Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(weights),
            Qwen4ExpPostPleMixerMetalWorkspace::QwenSparseAttention(mixer),
        ) => {
            crate::qwen4exp_qsa::validate_contract(
                ctx,
                workspace.residual.mixed_tensor(),
                weights,
                mixer,
            )?;
            crate::qwen4exp_qsa::preflight(ctx, weights)?;
        }
        _ => return invalid("mixer weight and workspace variants differ"),
    }
    validate_and_preflight_gated_residual_mix(
        ctx,
        hyper_residual,
        &workspace.moe_output,
        g.eps,
        weights.ffn_residual.read,
        weights.ffn_residual.inject,
        &workspace.residual,
    )?;
    crate::qwen4exp_moe::validate_contract(
        ctx,
        workspace.residual.mixed_tensor(),
        weights.moe,
        &workspace.moe,
    )?;
    crate::qwen4exp_moe::preflight(ctx, weights.moe)?;

    for (name, tensor, shape) in [
        (
            "post-PLE hyper residual",
            hyper_residual,
            vec![g.hyper_width() as u64],
        ),
        (
            "post-PLE mixer output",
            &workspace.mixer_output,
            vec![g.hidden_size as u64],
        ),
        (
            "post-PLE MoE output",
            &workspace.moe_output,
            vec![g.hidden_size as u64],
        ),
    ] {
        require_tensor(name, tensor, GgmlType::F32, &shape, true)?;
    }
    let residual_weights = residual_weight_tensors(weights);
    require_read_only_weights(&residual_weights)?;
    let mut tensors = top_level_tensors(workspace, hyper_residual);
    tensors.extend(residual_weights);
    require_same_device(ctx, &tensors)?;
    require_disjoint(&top_level_tensors(workspace, hyper_residual))?;
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

fn validate_encoder(
    ctx: &MetalContext,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("post-PLE dependent dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "post-PLE encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn require_owner(
    workspace: &Qwen4ExpPostPleBlockMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    let command = enc.parent_command_buffer();
    let Some(owner) = workspace.active_command.as_ref() else {
        return invalid("post-PLE output has no owning command buffer");
    };
    if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
        return invalid("post-PLE output must be copied by its owning command buffer");
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpPostPleBlockMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.encode_failed = false;
    Ok(())
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpPostPleBlockError> {
    if tensor.dtype != dtype || tensor.shape != shape {
        return invalid(format!(
            "{name} must be {dtype:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    require_range(name, tensor)
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpPostPleBlockError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| {
            Qwen4ExpPostPleBlockError::Invalid("tensor element count overflow".into())
        })?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpPostPleBlockError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
    })?;
    if block == 0 || !elements.is_multiple_of(block) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    elements
        .checked_div(block)
        .and_then(|units| units.checked_mul(bytes))
        .ok_or_else(|| Qwen4ExpPostPleBlockError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpPostPleBlockError> {
    let alignment = match tensor.dtype {
        GgmlType::F32 | GgmlType::I32 => 4,
        _ => 2,
    };
    if !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    let bytes = storage_bytes(tensor)?;
    let end = tensor
        .offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpPostPleBlockError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_read_only_weights(
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpPostPleBlockError> {
    for (name, tensor) in tensors {
        if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
            return invalid(format!("{name} must have read-only weight provenance"));
        }
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpPostPleBlockError> {
    let expected = ctx.device.registryID();
    for (name, tensor) in tensors {
        let actual = tensor.buffer.device().registryID();
        if actual != expected {
            return invalid(format!(
                "{name} belongs to Metal device registry {actual}, expected {expected}"
            ));
        }
    }
    Ok(())
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpPostPleBlockError> {
    for left in 0..tensors.len() {
        let left_bytes = storage_bytes(tensors[left].1)?;
        for right in left + 1..tensors.len() {
            if Retained::as_ptr(&tensors[left].1.buffer)
                != Retained::as_ptr(&tensors[right].1.buffer)
            {
                continue;
            }
            let right_bytes = storage_bytes(tensors[right].1)?;
            let left_end = tensors[left].1.offset.saturating_add(left_bytes);
            let right_end = tensors[right].1.offset.saturating_add(right_bytes);
            if tensors[left].1.offset < right_end && tensors[right].1.offset < left_end {
                return invalid(format!("{} overlaps {}", tensors[left].0, tensors[right].0));
            }
        }
    }
    Ok(())
}

fn residual_weight_tensors(
    weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("attention HC norm", weights.attention_residual.read.norm),
        ("attention HC down", weights.attention_residual.read.down),
        ("attention HC up", weights.attention_residual.read.up),
        ("attention HC injection", weights.attention_residual.inject),
        ("FFN HC norm", weights.ffn_residual.read.norm),
        ("FFN HC down", weights.ffn_residual.read.down),
        ("FFN HC up", weights.ffn_residual.read.up),
        ("FFN HC injection", weights.ffn_residual.inject),
    ]
}

fn top_level_tensors<'a>(
    workspace: &'a Qwen4ExpPostPleBlockMetalWorkspace,
    hyper_residual: &'a MetalTensor,
) -> Vec<(&'static str, &'a MetalTensor)> {
    vec![
        ("post-PLE hyper residual", hyper_residual),
        ("post-PLE mixer output", &workspace.mixer_output),
        ("post-PLE MoE output", &workspace.moe_output),
        (
            "post-PLE residual mixed output",
            workspace.residual.mixed_tensor(),
        ),
    ]
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpPostPleBlockError> {
    Err(Qwen4ExpPostPleBlockError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn four_layer_config() -> Qwen4ExpConfig {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        config.context_length = 16;
        config.layer_count = 4;
        config.hidden_size = 256;
        config.vocab_size = 32;
        config.hyper_connection.low_rank = 32;
        config.attention.query_heads = 4;
        config.attention.kv_heads = 2;
        config.gated_delta_net.key_heads = 1;
        config.gated_delta_net.value_heads = 1;
        config.gated_delta_net.inner_size = 128;
        config.moe.expert_count = 16;
        config.moe.expert_intermediate_size = 32;
        config.moe.shared_expert_intermediate_size = 32;
        config.qsa.token_budget = 8;
        config.compress_ratios = vec![0, 0, 0, 4];
        let ple = config.ple.as_mut().unwrap();
        ple.token_vocab_size = 32;
        ple.heads_per_ngram = 1;
        ple.embedding_head_dim = 128;
        ple.eos_token_id = 31;
        ple.image_token_id = Some(30);
        ple.multipliers = vec![3, 5, 7];
        ple.head_offsets = vec![0, 17];
        ple.head_vocab_sizes = vec![17, 19];
        config.validate().unwrap();
        config
    }

    #[test]
    fn reference_schedule_binds_only_valid_post_ple_mixers_and_capacities() {
        let config = four_layer_config();
        let layer_two = Qwen4ExpPostPleBlockMetalGeometry::from_config(&config, 2, None).unwrap();
        assert_eq!(layer_two.layer(), 2);
        assert_eq!(layer_two.mixer().kind(), MixerKind::GatedDeltaNet);
        assert_eq!(layer_two.branch_count(), 4);
        assert_eq!(layer_two.hidden_size(), 256);
        assert_eq!(layer_two.low_rank(), 32);

        let layer_three =
            Qwen4ExpPostPleBlockMetalGeometry::from_config(&config, 3, Some(8)).unwrap();
        assert_eq!(layer_three.layer(), 3);
        assert_eq!(layer_three.mixer().kind(), MixerKind::QwenSparseAttention);
        let qsa = layer_three.mixer().qsa().unwrap();
        assert_eq!(qsa.compression_ratio(), 4);
        assert_eq!(qsa.capacity(), 8);
        assert_eq!(qsa.token_budget(), 8);

        let ple_error = Qwen4ExpPostPleBlockMetalGeometry::from_config(&config, 1, None)
            .unwrap_err()
            .to_string();
        assert!(ple_error.contains("PLE transform"));
        let gdn_capacity_error =
            Qwen4ExpPostPleBlockMetalGeometry::from_config(&config, 2, Some(8))
                .unwrap_err()
                .to_string();
        assert!(gdn_capacity_error.contains("must not receive QSA capacity"));
        let missing_capacity = Qwen4ExpPostPleBlockMetalGeometry::from_config(&config, 3, None)
            .unwrap_err()
            .to_string();
        assert!(missing_capacity.contains("requires an explicit cache capacity"));
        let unaligned_capacity =
            Qwen4ExpPostPleBlockMetalGeometry::from_config(&config, 3, Some(6))
                .unwrap_err()
                .to_string();
        assert!(unaligned_capacity.contains("ratio-aligned"));
    }
}
