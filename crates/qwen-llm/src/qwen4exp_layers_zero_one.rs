//! One-command composition through the first two Qwen3.8-Flash-Next blocks.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_copy_offset_f32,
};
use crate::qwen4exp::{MixerKind, PleConfig, PleHistory, Qwen4ExpConfig, Qwen4ExpError};
#[cfg(test)]
use crate::qwen4exp_composition_trace::{
    Qwen4ExpCompositionTracePhase, Qwen4ExpCompositionTraceStage, encode_qwen4exp_composition_trace,
};
use crate::qwen4exp_gdn::{
    GatedDeltaNetMetalWeights, GatedDeltaNetMetalWorkspace, GatedDeltaNetPackedScratch,
    Qwen4ExpGdnError, encode_gated_delta_net, encode_gated_delta_net_packed_into_workspace,
    validate_and_preflight_gated_delta_net_packed_workspace,
};
use crate::qwen4exp_layer_zero::{
    Qwen4ExpLayerZeroError, Qwen4ExpLayerZeroMetalGeometry, Qwen4ExpLayerZeroMetalWeights,
    Qwen4ExpLayerZeroMetalWorkspace, Qwen4ExpResidualMetalWeights, encode_layer_zero_gdn_packed,
    encode_qwen4exp_layer_zero, validate_and_preflight_layer_zero_gdn_packed,
};
use crate::qwen4exp_metal::{
    GatedResidualMetalReadWeights, GatedResidualMetalScratch, GatedResidualPackedScratch,
    Qwen4ExpMetalError, encode_gated_residual_mix, encode_gated_residual_packed_mix,
    validate_and_preflight_gated_residual_mix, validate_and_preflight_gated_residual_packed_mix,
};
use crate::qwen4exp_moe::{
    Qwen4ExpMoeError, Qwen4ExpMoeMetalWeights, Qwen4ExpMoeMetalWorkspace,
    Qwen4ExpMoePackedMotorScratch, encode_qwen4exp_moe, encode_qwen4exp_moe_packed_motor_for_layer,
    preflight_packed as preflight_moe_packed, validate_packed_contract as validate_moe_packed,
};
use crate::qwen4exp_ple::PleIq4NlTable;
use crate::qwen4exp_ple_metal::{
    Qwen4ExpPleMetalError, Qwen4ExpPleMetalGeometry, Qwen4ExpPleMetalWeights,
    Qwen4ExpPleMetalWorkspace, Qwen4ExpPlePackedMotorScratch, encode_qwen4exp_ple,
    encode_qwen4exp_ple_packed_into_workspace,
    validate_and_preflight_qwen4exp_ple_packed_workspace,
};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice, MTLResource};

const LAYER_ZERO: u32 = 0;
const LAYER_ONE: u32 = 1;

#[cfg(test)]
fn capture_composition_stage(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    layer: u32,
    phase: Qwen4ExpCompositionTracePhase,
    source: &MetalTensor,
    width: usize,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    encode_qwen4exp_composition_trace(
        ctx,
        enc,
        Qwen4ExpCompositionTraceStage { layer, phase },
        source,
        width,
    )?;
    Ok(())
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpLayersZeroOneError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Residual(#[from] Qwen4ExpMetalError),
    #[error(transparent)]
    LayerZero(#[from] Qwen4ExpLayerZeroError),
    #[error(transparent)]
    Ple(#[from] Qwen4ExpPleMetalError),
    #[error(transparent)]
    GatedDeltaNet(#[from] Qwen4ExpGdnError),
    #[error(transparent)]
    Moe(#[from] Qwen4ExpMoeError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next layers-zero-one contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next layers-zero-one command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpLayersZeroOneMetalGeometry {
    context_length: usize,
    layer_count: usize,
    layer_zero: Qwen4ExpLayerZeroMetalGeometry,
    ple: Qwen4ExpPleMetalGeometry,
}

impl Qwen4ExpLayersZeroOneMetalGeometry {
    pub fn new(
        context_length: usize,
        layer_count: usize,
        layer_zero: Qwen4ExpLayerZeroMetalGeometry,
        ple: Qwen4ExpPleMetalGeometry,
    ) -> Result<Self, Qwen4ExpLayersZeroOneError> {
        let geometry = Self {
            context_length,
            layer_count,
            layer_zero,
            ple,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(config: &Qwen4ExpConfig) -> Result<Self, Qwen4ExpLayersZeroOneError> {
        config.validate()?;
        if config.layer_count < 2 {
            return invalid("two-layer composition requires at least two layers");
        }
        for layer in [LAYER_ZERO, LAYER_ONE] {
            if config.mixer_kind(layer) != Some(MixerKind::GatedDeltaNet) {
                return invalid(format!("layer {layer} must use Gated DeltaNet"));
            }
        }
        Self::new(
            config.context_length as usize,
            config.layer_count as usize,
            Qwen4ExpLayerZeroMetalGeometry::from_config(config)?,
            Qwen4ExpPleMetalGeometry::from_config(config, LAYER_ONE)?,
        )
    }

    pub fn context_length(self) -> usize {
        self.context_length
    }

    pub fn layer_count(self) -> usize {
        self.layer_count
    }

    pub fn layer_zero(self) -> Qwen4ExpLayerZeroMetalGeometry {
        self.layer_zero
    }

    pub fn ple(self) -> Qwen4ExpPleMetalGeometry {
        self.ple
    }

    pub fn branch_count(self) -> usize {
        self.layer_zero.branch_count()
    }

    pub fn hidden_size(self) -> usize {
        self.layer_zero.hidden_size()
    }

    pub fn low_rank(self) -> usize {
        self.layer_zero.low_rank()
    }

    pub fn vocab_size(self) -> usize {
        self.layer_zero.vocab_size()
    }

    pub fn hyper_width(self) -> usize {
        self.layer_zero.hyper_width()
    }

    pub fn eps(self) -> f32 {
        self.layer_zero.eps()
    }

    fn validate(self) -> Result<(), Qwen4ExpLayersZeroOneError> {
        if self.context_length == 0 || self.layer_count < 2 {
            return invalid("context length must be nonzero and layer count at least two");
        }
        if self.ple.branch_count() != self.layer_zero.branch_count()
            || self.ple.hidden_size() != self.layer_zero.hidden_size()
            || self.ple.eps().to_bits() != self.layer_zero.eps().to_bits()
        {
            return invalid("layer-zero and PLE residual geometries differ");
        }
        for (name, value) in [
            ("context length", self.context_length),
            ("layer count", self.layer_count),
            ("hyper width", self.hyper_width()),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("{name} {value} exceeds u32"));
            }
        }
        self.hyper_width().checked_mul(4).ok_or_else(|| {
            Qwen4ExpLayersZeroOneError::Invalid("hyper-residual byte count overflow".into())
        })?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpLayersZeroOneMetalWeights<'a> {
    pub geometry: Qwen4ExpLayersZeroOneMetalGeometry,
    pub ple_config: &'a PleConfig,
    pub layer_zero: Qwen4ExpLayerZeroMetalWeights<'a>,
    pub ple: Qwen4ExpPleMetalWeights<'a>,
    pub layer_one_attention_residual: Qwen4ExpResidualMetalWeights<'a>,
    pub layer_one_gdn: GatedDeltaNetMetalWeights<'a>,
    pub layer_one_ffn_residual: Qwen4ExpResidualMetalWeights<'a>,
    pub layer_one_moe: Qwen4ExpMoeMetalWeights<'a>,
}

impl<'a> Qwen4ExpLayersZeroOneMetalWeights<'a> {
    pub fn bind(weights: &'a Qwen4ExpMetalWeights) -> Result<Self, Qwen4ExpLayersZeroOneError> {
        let geometry = Qwen4ExpLayersZeroOneMetalGeometry::from_config(weights.config())?;
        let ple_config =
            weights.config().ple.as_ref().ok_or_else(|| {
                Qwen4ExpLayersZeroOneError::Invalid("model has no PLE config".into())
            })?;
        Ok(Self {
            geometry,
            ple_config,
            layer_zero: Qwen4ExpLayerZeroMetalWeights::bind(weights)?,
            ple: Qwen4ExpPleMetalWeights::bind(weights, LAYER_ONE)?,
            layer_one_attention_residual: bind_residual(weights, LAYER_ONE, "attn")?,
            layer_one_gdn: GatedDeltaNetMetalWeights::bind(weights, LAYER_ONE)?,
            layer_one_ffn_residual: bind_residual(weights, LAYER_ONE, "ffn")?,
            layer_one_moe: Qwen4ExpMoeMetalWeights::bind(weights, LAYER_ONE)?,
        })
    }
}

fn bind_residual<'a>(
    weights: &'a Qwen4ExpMetalWeights,
    layer: u32,
    role: &str,
) -> Result<Qwen4ExpResidualMetalWeights<'a>, Qwen4ExpLayersZeroOneError> {
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

pub struct Qwen4ExpLayersZeroOneMetalWorkspace {
    geometry: Qwen4ExpLayersZeroOneMetalGeometry,
    layer_zero: Qwen4ExpLayerZeroMetalWorkspace,
    hyper_residual: MetalTensor,
    ple: Qwen4ExpPleMetalWorkspace,
    mixer_output: MetalTensor,
    moe_output: MetalTensor,
    residual: GatedResidualMetalScratch,
    layer_one_gdn: GatedDeltaNetMetalWorkspace,
    layer_one_moe: Qwen4ExpMoeMetalWorkspace,
    history: PleHistory,
    pending_history: Option<PleHistory>,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
    encode_failed: bool,
}

impl Qwen4ExpLayersZeroOneMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpLayersZeroOneMetalGeometry,
    ) -> Result<Self, Qwen4ExpLayersZeroOneError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            layer_zero: Qwen4ExpLayerZeroMetalWorkspace::new(ctx, geometry.layer_zero)?,
            hyper_residual: MetalTensor::zeros_f32(ctx, vec![geometry.hyper_width() as u64])?,
            ple: Qwen4ExpPleMetalWorkspace::new(ctx, geometry.ple)?,
            mixer_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size() as u64])?,
            moe_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size() as u64])?,
            residual: GatedResidualMetalScratch::new(
                ctx,
                geometry.branch_count(),
                geometry.hidden_size(),
                geometry.low_rank(),
            )?,
            layer_one_gdn: GatedDeltaNetMetalWorkspace::new(ctx, geometry.layer_zero.gdn())?,
            layer_one_moe: Qwen4ExpMoeMetalWorkspace::new(ctx, geometry.layer_zero.moe())?,
            history: PleHistory::default(),
            pending_history: None,
            active_command: None,
            state_poisoned: false,
            encode_failed: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpLayersZeroOneMetalGeometry {
        self.geometry
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub(crate) fn validate_hc_up_binding(
        &self,
        ctx: &MetalContext,
    ) -> Result<(), Qwen4ExpLayersZeroOneError> {
        self.require_idle()?;
        if self.state_poisoned || self.encode_failed || self.pending_history.is_some() {
            return invalid("HC configuration requires healthy released bootstrap layers");
        }
        self.layer_zero.validate_hc_up_binding(ctx)?;
        self.residual.validate_hc_up_binding(ctx)?;
        Ok(())
    }

    pub(crate) fn bind_hc_up_mix(&mut self, enabled: bool) {
        self.layer_zero.bind_hc_up_mix(enabled);
        self.residual.bind_hc_up_mix(enabled);
    }

    #[cfg(test)]
    pub(crate) fn hc_up_binding_states(&self) -> [bool; 2] {
        [
            self.layer_zero.hc_up_mix_enabled(),
            self.residual.hc_up_mix_enabled(),
        ]
    }

    pub fn next_position(&self) -> Option<u64> {
        self.history.next_position()
    }

    pub fn prior_tokens(&self) -> &[u32] {
        self.history.prior_tokens()
    }

    #[cfg(test)]
    pub(crate) fn checkpoint_history_for_tests(&self) -> PleHistory {
        self.require_idle().unwrap();
        assert!(self.pending_history.is_none() && !self.state_poisoned && !self.encode_failed);
        self.history.clone()
    }

    #[cfg(test)]
    pub(crate) fn restore_history_for_tests(&mut self, history: &PleHistory) {
        self.checkpoint_history_for_tests();
        assert!(history.next_position() <= self.history.next_position());
        self.history = history.clone();
    }

    #[cfg(test)]
    pub(crate) fn persistent_state_tensors(&self) -> Vec<MetalTensor> {
        let mut tensors = self.layer_zero.persistent_state_tensors();
        tensors.extend(self.ple.persistent_state_tensors());
        tensors.extend(self.layer_one_gdn.persistent_state_tensors());
        tensors
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpLayersZeroOneError> {
        self.require_idle()?;
        if self.pending_history.is_some() {
            return invalid("cannot reset while a PLE history update is pending");
        }
        self.layer_zero.reset()?;
        self.ple.reset()?;
        self.layer_one_gdn.reset()?;
        self.layer_one_moe.reset()?;
        self.history.reset();
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpLayersZeroOneError> {
        let Some(command) = self.active_command.clone() else {
            if self.pending_history.is_some() {
                return invalid("PLE history is pending without an owning command");
            }
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
        if let Err(error) = self.layer_zero.release_after() {
            child_errors.push(format!("layer zero: {error}"));
        }
        if let Err(error) = self.ple.release_after() {
            child_errors.push(format!("PLE: {error}"));
        }
        if let Err(error) = self.residual.release_after() {
            child_errors.push(format!("layer-one residual: {error}"));
        }
        if let Err(error) = self.layer_one_gdn.release_after() {
            child_errors.push(format!("layer-one GDN: {error}"));
        }
        if let Err(error) = self.layer_one_moe.release_after() {
            child_errors.push(format!("layer-one MoE: {error}"));
        }
        self.active_command = None;
        let pending = self.pending_history.take();
        let pending_present = pending.is_some();
        if status == MTLCommandBufferStatus::Completed
            && command_error.is_none()
            && child_errors.is_empty()
            && !self.encode_failed
            && let Some(pending) = pending
        {
            self.history = pending;
            self.encode_failed = false;
            return Ok(());
        }
        self.state_poisoned = true;
        Err(Qwen4ExpLayersZeroOneError::CommandBuffer(format!(
            "status={status:?}, error={command_error:?}, encode_failed={}, pending_history={pending_present}, children={child_errors:?}",
            self.encode_failed
        )))
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may mutate five child workspaces and
    /// their causal state after another token acquires this composition.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpLayersZeroOneError> {
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
        if let Err(error) = unsafe { self.layer_zero.abandon_uncommitted() } {
            child_errors.push(format!("layer zero: {error}"));
        }
        if let Err(error) = unsafe { self.ple.abandon_uncommitted() } {
            child_errors.push(format!("PLE: {error}"));
        }
        if let Err(error) = unsafe { self.residual.abandon_uncommitted() } {
            child_errors.push(format!("layer-one residual: {error}"));
        }
        if let Err(error) = unsafe { self.layer_one_gdn.abandon_uncommitted() } {
            child_errors.push(format!("layer-one GDN: {error}"));
        }
        if let Err(error) = unsafe { self.layer_one_moe.abandon_uncommitted() } {
            child_errors.push(format!("layer-one MoE: {error}"));
        }
        if !child_errors.is_empty() {
            return invalid(format!(
                "could not abandon every layers-zero-one child: {child_errors:?}"
            ));
        }
        self.active_command = None;
        self.pending_history = None;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpLayersZeroOneError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "copy the layer-one residual in its owning command, then release the workspace"]
pub struct Qwen4ExpLayersZeroOneMetalRead<'a> {
    workspace: &'a mut Qwen4ExpLayersZeroOneMetalWorkspace,
}

pub struct Qwen4ExpLayersZeroOneMetalOutput<'a> {
    workspace: &'a Qwen4ExpLayersZeroOneMetalWorkspace,
}

impl Qwen4ExpLayersZeroOneMetalRead<'_> {
    pub fn output(&self) -> Qwen4ExpLayersZeroOneMetalOutput<'_> {
        Qwen4ExpLayersZeroOneMetalOutput {
            workspace: self.workspace,
        }
    }
}

impl Qwen4ExpLayersZeroOneMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hyper_width() as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn branch_count(&self) -> usize {
        self.workspace.geometry.branch_count()
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpLayersZeroOneError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("layers-zero-one output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("layers-zero-one output must be copied by its owning command buffer");
        }
        require_tensor(
            "layers-zero-one copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hyper_width() as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                (
                    "layers-zero-one hyper residual",
                    &self.workspace.hyper_residual,
                ),
                ("layers-zero-one copied output destination", destination),
            ],
        )?;
        let mut tensors = top_level_tensors(self.workspace);
        tensors.push(("layers-zero-one copied output destination", destination));
        require_disjoint(&tensors)?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(
            ctx,
            enc,
            &self.workspace.hyper_residual,
            0,
            destination,
            self.workspace.geometry.hyper_width(),
        )?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_qwen4exp_layers_zero_one<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: u64,
    table: PleIq4NlTable<'_>,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<Qwen4ExpLayersZeroOneMetalRead<'a>, Qwen4ExpLayersZeroOneError> {
    validate_and_preflight(ctx, enc, token_id, position, weights, workspace)?;
    let next_history = stage_ple_rows(token_id, position, table, weights, workspace)?;
    encode_qwen4exp_layers_zero_one_staged(ctx, enc, token_id, weights, next_history, workspace)
}

pub(crate) fn encode_qwen4exp_layers_zero_one_staged<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    next_history: PleHistory,
    workspace: &'a mut Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<Qwen4ExpLayersZeroOneMetalRead<'a>, Qwen4ExpLayersZeroOneError> {
    if !workspace.ple.has_staged_rows() {
        return invalid("layers-zero-one PLE rows were not staged by the parent");
    }
    reserve_command(workspace, enc)?;
    workspace.pending_history = Some(next_history);
    if let Err(error) = encode_step(ctx, enc, token_id, weights, workspace) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpLayersZeroOneMetalRead { workspace })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_and_preflight_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: u64,
    ple_embedding: &MetalTensor,
    hyper_residual: &MetalTensor,
    bridge: &MetalTensor,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpLayersZeroOneMetalWorkspace,
    residual: &GatedResidualPackedScratch,
    gdn: &GatedDeltaNetPackedScratch,
    ple: &Qwen4ExpPlePackedMotorScratch,
    moe: &Qwen4ExpMoePackedMotorScratch,
) -> Result<(PleHistory, Vec<u32>), Qwen4ExpLayersZeroOneError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.pending_history.is_some() {
        return invalid("workspace has a PLE history update without a command owner");
    }
    if weights.geometry != workspace.geometry {
        return invalid("packed layers-zero-one weight and workspace geometry differ");
    }
    validate_nested_geometry(weights, workspace)?;
    let tokens = token_ids.len();
    if tokens <= 1 {
        return invalid("packed layers-zero-one requires at least two tokens");
    }
    if let Some((index, token_id)) = token_ids
        .iter()
        .copied()
        .enumerate()
        .find(|(_, token_id)| *token_id as usize >= workspace.geometry.vocab_size())
    {
        return invalid(format!(
            "token ID {token_id} at packed row {index} is outside vocabulary {}",
            workspace.geometry.vocab_size()
        ));
    }
    match workspace.history.next_position() {
        None if start_position != 0 => {
            return invalid("fresh packed causal state must begin at position zero");
        }
        Some(expected) if start_position != expected => {
            return invalid(format!(
                "packed start position {start_position} is discontinuous from {expected}; reset causal state first"
            ));
        }
        None | Some(_) => {}
    }
    let end_position = start_position.checked_add(tokens as u64).ok_or_else(|| {
        Qwen4ExpLayersZeroOneError::Invalid("packed position range overflow".into())
    })?;
    if end_position > workspace.geometry.context_length() as u64 {
        return invalid(format!(
            "packed position range {start_position}..{end_position} exceeds context length {}",
            workspace.geometry.context_length()
        ));
    }
    validate_ple_config(weights.ple_config, workspace.geometry)?;

    let g = workspace.geometry;
    require_tensor(
        "packed layers-zero-one PLE embedding",
        ple_embedding,
        GgmlType::F32,
        &[g.hidden_size() as u64, tokens as u64],
        true,
    )?;
    require_tensor(
        "packed layers-zero-one hyper residual",
        hyper_residual,
        GgmlType::F32,
        &[g.hyper_width() as u64, tokens as u64],
        true,
    )?;
    require_tensor(
        "packed layers-zero-one bridge",
        bridge,
        GgmlType::F32,
        &[g.hidden_size() as u64, tokens as u64],
        true,
    )?;
    require_same_device(
        ctx,
        &[
            ("packed layers-zero-one PLE embedding", ple_embedding),
            ("packed layers-zero-one hyper residual", hyper_residual),
            ("packed layers-zero-one bridge", bridge),
        ],
    )?;
    require_disjoint(&[
        ("packed layers-zero-one PLE embedding", ple_embedding),
        ("packed layers-zero-one hyper residual", hyper_residual),
        ("packed layers-zero-one bridge", bridge),
    ])?;

    let mixed = residual.mixed_view(tokens)?;
    for residual_weights in [
        weights.layer_zero.attention_residual,
        weights.layer_zero.ffn_residual,
        weights.layer_one_attention_residual,
        weights.layer_one_ffn_residual,
    ] {
        validate_and_preflight_gated_residual_packed_mix(
            ctx,
            hyper_residual,
            bridge,
            g.eps(),
            residual_weights.read,
            residual_weights.inject,
            residual,
            tokens,
        )?;
    }
    require_read_only_weights(&[
        (
            "layer-zero attention HC norm",
            weights.layer_zero.attention_residual.read.norm,
        ),
        (
            "layer-zero attention HC down",
            weights.layer_zero.attention_residual.read.down,
        ),
        (
            "layer-zero attention HC up",
            weights.layer_zero.attention_residual.read.up,
        ),
        (
            "layer-zero attention HC injection",
            weights.layer_zero.attention_residual.inject,
        ),
        (
            "layer-zero FFN HC norm",
            weights.layer_zero.ffn_residual.read.norm,
        ),
        (
            "layer-zero FFN HC down",
            weights.layer_zero.ffn_residual.read.down,
        ),
        (
            "layer-zero FFN HC up",
            weights.layer_zero.ffn_residual.read.up,
        ),
        (
            "layer-zero FFN HC injection",
            weights.layer_zero.ffn_residual.inject,
        ),
        (
            "layer-one attention HC norm",
            weights.layer_one_attention_residual.read.norm,
        ),
        (
            "layer-one attention HC down",
            weights.layer_one_attention_residual.read.down,
        ),
        (
            "layer-one attention HC up",
            weights.layer_one_attention_residual.read.up,
        ),
        (
            "layer-one attention HC injection",
            weights.layer_one_attention_residual.inject,
        ),
        (
            "layer-one FFN HC norm",
            weights.layer_one_ffn_residual.read.norm,
        ),
        (
            "layer-one FFN HC down",
            weights.layer_one_ffn_residual.read.down,
        ),
        (
            "layer-one FFN HC up",
            weights.layer_one_ffn_residual.read.up,
        ),
        (
            "layer-one FFN HC injection",
            weights.layer_one_ffn_residual.inject,
        ),
    ])?;
    validate_and_preflight_layer_zero_gdn_packed(
        ctx,
        enc,
        &mixed,
        weights.layer_zero.gdn,
        &workspace.layer_zero,
        gdn,
        tokens,
    )?;
    validate_and_preflight_gated_delta_net_packed_workspace(
        ctx,
        enc,
        &mixed,
        weights.layer_one_gdn,
        &workspace.layer_one_gdn,
        gdn,
        tokens,
    )?;
    validate_and_preflight_qwen4exp_ple_packed_workspace(
        ctx,
        enc,
        ple_embedding,
        hyper_residual,
        weights.ple,
        &workspace.ple,
        ple,
        tokens,
    )?;
    for moe_weights in [weights.layer_zero.moe, weights.layer_one_moe] {
        validate_moe_packed(ctx, &mixed, moe_weights, moe, tokens)?;
        preflight_moe_packed(ctx, moe_weights, tokens)?;
    }
    ctx.pipeline("kernel_copy_offset_f32")?;

    let head_count = g.ple().head_count();
    let row_capacity = tokens.checked_mul(head_count).ok_or_else(|| {
        Qwen4ExpLayersZeroOneError::Invalid("packed PLE row count overflow".into())
    })?;
    let mut next_history = workspace.history.clone();
    let mut row_ids = Vec::with_capacity(row_capacity);
    for (index, &token_id) in token_ids.iter().enumerate() {
        let position = start_position + index as u64;
        let (advanced, rows) = next_history.advanced(weights.ple_config, token_id, position)?;
        if rows.len() != head_count {
            return invalid(format!(
                "PLE produced {} rows at packed token {index}, expected {head_count}",
                rows.len()
            ));
        }
        row_ids.extend_from_slice(&rows);
        next_history = advanced;
    }
    Ok((next_history, row_ids))
}

/// Encode packed layers zero and one into their scalar causal-state owners.
///
/// # Safety
///
/// The caller must retain the owning command and all packed tensors and
/// scratch until completion or permanent abandonment. Any failure poisons the
/// layers-zero-one transaction.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn encode_qwen4exp_layers_zero_one_packed_staged(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    ple_embedding: &MetalTensor,
    hyper_residual: &MetalTensor,
    bridge: &MetalTensor,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    next_history: PleHistory,
    workspace: &mut Qwen4ExpLayersZeroOneMetalWorkspace,
    residual: &mut GatedResidualPackedScratch,
    gdn: &GatedDeltaNetPackedScratch,
    ple: &Qwen4ExpPlePackedMotorScratch,
    moe: &Qwen4ExpMoePackedMotorScratch,
    tokens: usize,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    reserve_command(workspace, enc)?;
    workspace.pending_history = Some(next_history);
    let encoded = (|| {
        let g = workspace.geometry;
        let hidden = g.hidden_size();

        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::LayerInput,
            hyper_residual,
            g.hyper_width(),
        )?;

        let attention = unsafe {
            encode_gated_residual_packed_mix(
                ctx,
                enc,
                hyper_residual,
                bridge,
                g.eps(),
                weights.layer_zero.attention_residual.read,
                weights.layer_zero.attention_residual.inject,
                residual,
                tokens,
            )
        }?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::AttentionInput,
            attention.mixed(),
            hidden,
        )?;
        let mixer_output = unsafe {
            encode_layer_zero_gdn_packed(
                ctx,
                enc,
                attention.mixed(),
                weights.layer_zero.gdn,
                &mut workspace.layer_zero,
                gdn,
                tokens,
            )
        }?;
        encode_copy_offset_f32(ctx, enc, &mixer_output, 0, bridge, hidden * tokens)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::MixerOutput,
            bridge,
            hidden,
        )?;
        attention.encode_combine(enc)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::AttentionOutput,
            hyper_residual,
            g.hyper_width(),
        )?;

        let ffn = unsafe {
            encode_gated_residual_packed_mix(
                ctx,
                enc,
                hyper_residual,
                bridge,
                g.eps(),
                weights.layer_zero.ffn_residual.read,
                weights.layer_zero.ffn_residual.inject,
                residual,
                tokens,
            )
        }?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::FfnInput,
            ffn.mixed(),
            hidden,
        )?;
        let moe_output = unsafe {
            encode_qwen4exp_moe_packed_motor_for_layer(
                ctx,
                enc,
                ffn.mixed(),
                weights.layer_zero.moe,
                moe,
                tokens,
                LAYER_ZERO,
                MixerKind::GatedDeltaNet,
            )
        }?;
        encode_copy_offset_f32(ctx, enc, &moe_output, 0, bridge, hidden * tokens)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::MoeOutput,
            bridge,
            hidden,
        )?;
        ffn.encode_combine(enc)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ZERO,
            Qwen4ExpCompositionTracePhase::LayerOutput,
            hyper_residual,
            g.hyper_width(),
        )?;

        let ple_output = unsafe {
            encode_qwen4exp_ple_packed_into_workspace(
                ctx,
                enc,
                ple_embedding,
                hyper_residual,
                weights.ple,
                &mut workspace.ple,
                ple,
                tokens,
            )
        }?;
        encode_copy_offset_f32(
            ctx,
            enc,
            &ple_output,
            0,
            hyper_residual,
            g.hyper_width() * tokens,
        )?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::PleOutput,
            hyper_residual,
            g.hyper_width(),
        )?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::LayerInput,
            hyper_residual,
            g.hyper_width(),
        )?;

        let attention = unsafe {
            encode_gated_residual_packed_mix(
                ctx,
                enc,
                hyper_residual,
                bridge,
                g.eps(),
                weights.layer_one_attention_residual.read,
                weights.layer_one_attention_residual.inject,
                residual,
                tokens,
            )
        }?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::AttentionInput,
            attention.mixed(),
            hidden,
        )?;
        let mixer_output = unsafe {
            encode_gated_delta_net_packed_into_workspace(
                ctx,
                enc,
                attention.mixed(),
                weights.layer_one_gdn,
                &mut workspace.layer_one_gdn,
                gdn,
                tokens,
            )
        }?;
        encode_copy_offset_f32(ctx, enc, &mixer_output, 0, bridge, hidden * tokens)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::MixerOutput,
            bridge,
            hidden,
        )?;
        attention.encode_combine(enc)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::AttentionOutput,
            hyper_residual,
            g.hyper_width(),
        )?;

        let ffn = unsafe {
            encode_gated_residual_packed_mix(
                ctx,
                enc,
                hyper_residual,
                bridge,
                g.eps(),
                weights.layer_one_ffn_residual.read,
                weights.layer_one_ffn_residual.inject,
                residual,
                tokens,
            )
        }?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::FfnInput,
            ffn.mixed(),
            hidden,
        )?;
        let moe_output = unsafe {
            encode_qwen4exp_moe_packed_motor_for_layer(
                ctx,
                enc,
                ffn.mixed(),
                weights.layer_one_moe,
                moe,
                tokens,
                LAYER_ONE,
                MixerKind::GatedDeltaNet,
            )
        }?;
        encode_copy_offset_f32(ctx, enc, &moe_output, 0, bridge, hidden * tokens)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::MoeOutput,
            bridge,
            hidden,
        )?;
        ffn.encode_combine(enc)?;
        #[cfg(test)]
        capture_composition_stage(
            ctx,
            enc,
            LAYER_ONE,
            Qwen4ExpCompositionTracePhase::LayerOutput,
            hyper_residual,
            g.hyper_width(),
        )?;
        Ok(())
    })();
    if encoded.is_err() {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
    }
    encoded
}

pub(crate) fn validate_and_preflight(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: u64,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.pending_history.is_some() {
        return invalid("workspace has a PLE history update without a command owner");
    }
    if weights.geometry != workspace.geometry {
        return invalid("layers-zero-one weight and workspace geometry differ");
    }
    validate_nested_geometry(weights, workspace)?;
    if token_id as usize >= workspace.geometry.vocab_size() {
        return invalid(format!(
            "token ID {token_id} is outside vocabulary {}",
            workspace.geometry.vocab_size()
        ));
    }
    if position >= workspace.geometry.context_length() as u64 {
        return invalid(format!(
            "position {position} is outside context length {}",
            workspace.geometry.context_length()
        ));
    }
    match workspace.history.next_position() {
        None if position != 0 => {
            return invalid("fresh causal state must begin at position zero");
        }
        Some(expected) if position != expected => {
            return invalid(format!(
                "position {position} is discontinuous from {expected}; reset causal state first"
            ));
        }
        None | Some(_) => {}
    }
    validate_ple_config(weights.ple_config, workspace.geometry)?;
    validate_contract(ctx, weights, workspace)
}

pub(crate) fn stage_ple_rows(
    token_id: u32,
    position: u64,
    table: PleIq4NlTable<'_>,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &mut Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<PleHistory, Qwen4ExpLayersZeroOneError> {
    let (next_history, row_ids) =
        workspace
            .history
            .advanced(weights.ple_config, token_id, position)?;
    workspace.ple.stage_rows(table, &row_ids)?;
    Ok(next_history)
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &mut Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    let layer_zero = encode_qwen4exp_layer_zero(
        ctx,
        enc,
        token_id,
        weights.layer_zero,
        &mut workspace.layer_zero,
    )?;
    layer_zero
        .output()
        .encode_copy_to(ctx, enc, &workspace.hyper_residual)?;
    drop(layer_zero);

    let ple = encode_qwen4exp_ple(
        ctx,
        enc,
        &workspace.hyper_residual,
        weights.ple,
        &mut workspace.ple,
    )?;
    ple.output()
        .encode_copy_to(ctx, enc, &workspace.hyper_residual)?;
    drop(ple);
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::PleOutput,
        &workspace.hyper_residual,
        workspace.geometry.hyper_width(),
    )?;

    let g = workspace.geometry;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::LayerInput,
        &workspace.hyper_residual,
        g.hyper_width(),
    )?;
    let attention = encode_gated_residual_mix(
        ctx,
        enc,
        &workspace.hyper_residual,
        &workspace.mixer_output,
        g.eps(),
        weights.layer_one_attention_residual.read,
        weights.layer_one_attention_residual.inject,
        &mut workspace.residual,
    )?;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::AttentionInput,
        attention.mixed(),
        g.hidden_size(),
    )?;
    let gdn = encode_gated_delta_net(
        ctx,
        enc,
        attention.mixed(),
        weights.layer_one_gdn,
        &mut workspace.layer_one_gdn,
    )?;
    gdn.output()
        .encode_copy_to(ctx, enc, &workspace.mixer_output)?;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::MixerOutput,
        &workspace.mixer_output,
        g.hidden_size(),
    )?;
    drop(gdn);
    attention.encode_combine()?;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::AttentionOutput,
        &workspace.hyper_residual,
        g.hyper_width(),
    )?;

    let ffn = encode_gated_residual_mix(
        ctx,
        enc,
        &workspace.hyper_residual,
        &workspace.moe_output,
        g.eps(),
        weights.layer_one_ffn_residual.read,
        weights.layer_one_ffn_residual.inject,
        &mut workspace.residual,
    )?;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::FfnInput,
        ffn.mixed(),
        g.hidden_size(),
    )?;
    let moe = encode_qwen4exp_moe(
        ctx,
        enc,
        ffn.mixed(),
        weights.layer_one_moe,
        &mut workspace.layer_one_moe,
    )?;
    moe.output()
        .encode_copy_to(ctx, enc, &workspace.moe_output)?;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::MoeOutput,
        &workspace.moe_output,
        g.hidden_size(),
    )?;
    drop(moe);
    ffn.encode_combine()?;
    #[cfg(test)]
    capture_composition_stage(
        ctx,
        enc,
        LAYER_ONE,
        Qwen4ExpCompositionTracePhase::LayerOutput,
        &workspace.hyper_residual,
        g.hyper_width(),
    )?;
    Ok(())
}

fn validate_nested_geometry(
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    let g = workspace.geometry;
    if weights.layer_zero.geometry != g.layer_zero
        || weights.ple.geometry != g.ple
        || weights.layer_one_gdn.geometry != g.layer_zero.gdn()
        || weights.layer_one_moe.geometry != g.layer_zero.moe()
    {
        return invalid("nested weight geometry differs from the composition");
    }
    if workspace.layer_zero.geometry() != g.layer_zero
        || workspace.ple.geometry() != g.ple
        || workspace.layer_one_gdn.geometry() != g.layer_zero.gdn()
        || workspace.layer_one_moe.geometry() != g.layer_zero.moe()
        || workspace.residual.branch_count() != g.branch_count()
        || workspace.residual.hidden_size() != g.hidden_size()
        || workspace.residual.low_rank() != g.low_rank()
    {
        return invalid("nested workspace geometry differs from the composition");
    }
    Ok(())
}

fn validate_ple_config(
    config: &PleConfig,
    geometry: Qwen4ExpLayersZeroOneMetalGeometry,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    config.validate(geometry.hidden_size() as u32, geometry.layer_count() as u32)?;
    if config.token_vocab_size as usize != geometry.vocab_size()
        || !config.layers.contains(&LAYER_ONE)
        || config.layers.contains(&LAYER_ZERO)
        || config.head_count()? as usize != geometry.ple.head_count()
        || config.embedding_head_dim as usize != geometry.ple.head_dim()
        || config.conv_kernel as usize != geometry.ple.kernel_size()
        || config.ngram_size as usize != geometry.ple.dilation()
    {
        return invalid("PLE hash and Metal geometries differ");
    }
    Ok(())
}

fn validate_contract(
    ctx: &MetalContext,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    let g = workspace.geometry;
    crate::qwen4exp_layer_zero::validate_contract(ctx, weights.layer_zero, &workspace.layer_zero)?;
    crate::qwen4exp_ple_metal::validate_contract(
        ctx,
        &workspace.hyper_residual,
        weights.ple,
        &workspace.ple,
    )?;
    crate::qwen4exp_ple_metal::preflight(ctx, weights.ple)?;
    validate_and_preflight_gated_residual_mix(
        ctx,
        &workspace.hyper_residual,
        &workspace.mixer_output,
        g.eps(),
        weights.layer_one_attention_residual.read,
        weights.layer_one_attention_residual.inject,
        &workspace.residual,
    )?;
    crate::qwen4exp_gdn::validate_contract(
        ctx,
        workspace.residual.mixed_tensor(),
        weights.layer_one_gdn,
        &workspace.layer_one_gdn,
    )?;
    crate::qwen4exp_gdn::preflight(ctx, weights.layer_one_gdn)?;
    validate_and_preflight_gated_residual_mix(
        ctx,
        &workspace.hyper_residual,
        &workspace.moe_output,
        g.eps(),
        weights.layer_one_ffn_residual.read,
        weights.layer_one_ffn_residual.inject,
        &workspace.residual,
    )?;
    crate::qwen4exp_moe::validate_contract(
        ctx,
        workspace.residual.mixed_tensor(),
        weights.layer_one_moe,
        &workspace.layer_one_moe,
    )?;
    crate::qwen4exp_moe::preflight(ctx, weights.layer_one_moe)?;

    for (name, tensor, shape) in [
        (
            "layers-zero-one hyper residual",
            &workspace.hyper_residual,
            vec![g.hyper_width() as u64],
        ),
        (
            "layers-zero-one mixer output",
            &workspace.mixer_output,
            vec![g.hidden_size() as u64],
        ),
        (
            "layers-zero-one MoE output",
            &workspace.moe_output,
            vec![g.hidden_size() as u64],
        ),
    ] {
        require_tensor(name, tensor, GgmlType::F32, &shape, true)?;
    }
    let residual_weights = layer_one_residual_weight_tensors(weights);
    require_read_only_weights(&residual_weights)?;
    let mut tensors = top_level_tensors(workspace);
    tensors.extend(residual_weights);
    require_same_device(ctx, &tensors)?;
    require_disjoint(&top_level_tensors(workspace))?;
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

fn validate_encoder(
    ctx: &MetalContext,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("layers-zero-one dependent dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "layers-zero-one encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpLayersZeroOneMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpLayersZeroOneError> {
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
) -> Result<(), Qwen4ExpLayersZeroOneError> {
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

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpLayersZeroOneError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| {
            Qwen4ExpLayersZeroOneError::Invalid("tensor element count overflow".into())
        })?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpLayersZeroOneError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
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
        .ok_or_else(|| Qwen4ExpLayersZeroOneError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpLayersZeroOneError> {
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
        .ok_or_else(|| Qwen4ExpLayersZeroOneError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpLayersZeroOneError> {
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

fn require_read_only_weights(
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpLayersZeroOneError> {
    for (name, tensor) in tensors {
        if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
            return invalid(format!("{name} must have read-only weight provenance"));
        }
    }
    Ok(())
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpLayersZeroOneError> {
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

fn top_level_tensors(
    workspace: &Qwen4ExpLayersZeroOneMetalWorkspace,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("layers-zero-one hyper residual", &workspace.hyper_residual),
        ("layers-zero-one mixer output", &workspace.mixer_output),
        ("layers-zero-one MoE output", &workspace.moe_output),
        (
            "layer-one residual mixed output",
            workspace.residual.mixed_tensor(),
        ),
    ]
}

fn layer_one_residual_weight_tensors(
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        (
            "layer-one attention HC norm",
            weights.layer_one_attention_residual.read.norm,
        ),
        (
            "layer-one attention HC down",
            weights.layer_one_attention_residual.read.down,
        ),
        (
            "layer-one attention HC up",
            weights.layer_one_attention_residual.read.up,
        ),
        (
            "layer-one attention HC injection",
            weights.layer_one_attention_residual.inject,
        ),
        (
            "layer-one FFN HC norm",
            weights.layer_one_ffn_residual.read.norm,
        ),
        (
            "layer-one FFN HC down",
            weights.layer_one_ffn_residual.read.down,
        ),
        (
            "layer-one FFN HC up",
            weights.layer_one_ffn_residual.read.up,
        ),
        (
            "layer-one FFN HC injection",
            weights.layer_one_ffn_residual.inject,
        ),
    ]
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpLayersZeroOneError> {
    Err(Qwen4ExpLayersZeroOneError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::qwen4exp_forward::{
        GatedResidualReadWeights, PleConvState, PleStepWeights, gated_residual_combine,
        gated_residual_mix, ple_step,
    };
    use crate::qwen4exp_gdn::GatedDeltaNetMetalGeometry;
    use crate::qwen4exp_moe::Qwen4ExpMoeMetalGeometry;
    use crate::qwen4exp_packed_prefill::{
        Qwen4ExpPackedPrefillWorkspace, encode_qwen4exp_packed_prefill,
    };
    use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
    use crate::tensor::TensorDesc;
    use objc2_metal::MTLCommandQueue;

    const BRANCHES: usize = 4;
    const HIDDEN: usize = 256;
    const RANK: usize = 32;
    const HYPER: usize = BRANCHES * HIDDEN;
    const VOCAB: usize = 32;
    const EXPERTS: usize = 16;
    const TOP_K: usize = 10;
    const FFN: usize = 32;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 128;
    const KERNEL: usize = 4;
    const DILATION: usize = 3;
    const CONTEXT: usize = 34;
    const TABLE_ROWS: usize = 36;

    struct ResidualFixture {
        norm_cpu: Vec<f32>,
        down_cpu: Vec<f32>,
        up_cpu: Vec<f32>,
        inject_cpu: Vec<f32>,
        norm: MetalTensor,
        down: MetalTensor,
        up: MetalTensor,
        inject: MetalTensor,
    }

    impl ResidualFixture {
        fn new(ctx: &MetalContext, seed: usize) -> Self {
            let norm_cpu = values(HYPER, seed, 0.002, 1.0);
            let down_bytes = quantize_rows(
                &values(HYPER * RANK, seed + 1, 0.0007, 0.0),
                GgmlType::Q8_0,
                HYPER,
            );
            let up_bytes = quantize_rows(
                &values(RANK * HYPER, seed + 2, 0.001, 0.0),
                GgmlType::Q8_0,
                RANK,
            );
            let down_cpu = dequant(&down_bytes, GgmlType::Q8_0, vec![HYPER, RANK]);
            let up_cpu = dequant(&up_bytes, GgmlType::Q8_0, vec![RANK, HYPER]);
            let inject_cpu = values(HYPER * BRANCHES, seed + 3, 0.0005, 0.0);
            Self {
                norm: weight_f32(ctx, &norm_cpu, vec![HYPER as u64]),
                down: weight_bytes(
                    ctx,
                    &down_bytes,
                    vec![HYPER as u64, RANK as u64],
                    GgmlType::Q8_0,
                ),
                up: weight_bytes(
                    ctx,
                    &up_bytes,
                    vec![RANK as u64, HYPER as u64],
                    GgmlType::Q8_0,
                ),
                inject: weight_f32(ctx, &inject_cpu, vec![HYPER as u64, BRANCHES as u64]),
                norm_cpu,
                down_cpu,
                up_cpu,
                inject_cpu,
            }
        }

        fn cpu_read(&self) -> GatedResidualReadWeights<'_> {
            GatedResidualReadWeights {
                norm: &self.norm_cpu,
                down: &self.down_cpu,
                up: &self.up_cpu,
            }
        }

        fn metal(&self) -> Qwen4ExpResidualMetalWeights<'_> {
            Qwen4ExpResidualMetalWeights {
                read: GatedResidualMetalReadWeights {
                    norm: &self.norm,
                    down: &self.down,
                    up: &self.up,
                },
                inject: &self.inject,
            }
        }
    }

    struct GdnFixture {
        qkv: MetalTensor,
        gate: MetalTensor,
        beta: MetalTensor,
        alpha: MetalTensor,
        a: MetalTensor,
        dt_bias: MetalTensor,
        conv: MetalTensor,
        norm: MetalTensor,
        output: MetalTensor,
    }

    impl GdnFixture {
        fn new(ctx: &MetalContext, geometry: GatedDeltaNetMetalGeometry, seed: usize) -> Self {
            let quant_projection = |n_in: usize, n_out: usize, offset: usize| {
                quantize_rows(
                    &values(n_in * n_out, seed + offset, 0.0012, 0.0),
                    GgmlType::Q8_0,
                    n_in,
                )
            };
            let qkv = quant_projection(HIDDEN, geometry.conv_width(), 1);
            let gate = quant_projection(HIDDEN, geometry.value_width(), 2);
            let output = quant_projection(geometry.value_width(), HIDDEN, 3);
            Self {
                qkv: weight_bytes(
                    ctx,
                    &qkv,
                    vec![HIDDEN as u64, geometry.conv_width() as u64],
                    GgmlType::Q8_0,
                ),
                gate: weight_bytes(
                    ctx,
                    &gate,
                    vec![HIDDEN as u64, geometry.value_width() as u64],
                    GgmlType::Q8_0,
                ),
                beta: weight_f32(
                    ctx,
                    &values(HIDDEN, seed + 4, 0.002, 0.0),
                    vec![HIDDEN as u64, 1],
                ),
                alpha: weight_f32(
                    ctx,
                    &values(HIDDEN, seed + 5, 0.002, 0.0),
                    vec![HIDDEN as u64, 1],
                ),
                a: weight_f32(ctx, &[-0.55], vec![1]),
                dt_bias: weight_f32(ctx, &[0.15], vec![1]),
                conv: weight_f32(
                    ctx,
                    &values(geometry.conv_width() * KERNEL, seed + 6, 0.004, 0.0),
                    vec![KERNEL as u64, geometry.conv_width() as u64],
                ),
                norm: weight_f32(
                    ctx,
                    &values(HEAD_DIM, seed + 7, 0.003, 1.0),
                    vec![HEAD_DIM as u64],
                ),
                output: weight_bytes(
                    ctx,
                    &output,
                    vec![geometry.value_width() as u64, HIDDEN as u64],
                    GgmlType::Q8_0,
                ),
            }
        }

        fn weights(&self, geometry: GatedDeltaNetMetalGeometry) -> GatedDeltaNetMetalWeights<'_> {
            GatedDeltaNetMetalWeights {
                geometry,
                qkv: &self.qkv,
                gate: &self.gate,
                beta: &self.beta,
                alpha: &self.alpha,
                a: &self.a,
                dt_bias: &self.dt_bias,
                conv: &self.conv,
                norm: &self.norm,
                output: &self.output,
            }
        }
    }

    struct MoeFixture {
        router: MetalTensor,
        routed_gate: MetalTensor,
        routed_up: MetalTensor,
        routed_down: MetalTensor,
        shared_router: MetalTensor,
        shared_gate: MetalTensor,
        shared_up: MetalTensor,
        shared_down: MetalTensor,
    }

    impl MoeFixture {
        fn new(ctx: &MetalContext, seed: usize) -> Self {
            let routed_gate = quantize_rows(
                &values(HIDDEN * FFN * EXPERTS, seed + 1, 0.003, 0.0),
                GgmlType::IQ4_XS,
                HIDDEN,
            );
            let routed_up = quantize_rows(
                &values(HIDDEN * FFN * EXPERTS, seed + 2, 0.003, 0.0),
                GgmlType::IQ4_XS,
                HIDDEN,
            );
            let routed_down = quantize_rows(
                &values(FFN * HIDDEN * EXPERTS, seed + 3, 0.004, 0.0),
                GgmlType::IQ4_NL,
                FFN,
            );
            let shared_gate = quantize_rows(
                &values(HIDDEN * FFN, seed + 4, 0.003, 0.0),
                GgmlType::Q8_0,
                HIDDEN,
            );
            let shared_up = quantize_rows(
                &values(HIDDEN * FFN, seed + 5, 0.003, 0.0),
                GgmlType::Q8_0,
                HIDDEN,
            );
            let shared_down = quantize_rows(
                &values(FFN * HIDDEN, seed + 6, 0.004, 0.0),
                GgmlType::Q8_0,
                FFN,
            );
            Self {
                router: weight_f32(
                    ctx,
                    &values(HIDDEN * EXPERTS, seed, 0.002, 0.0),
                    vec![HIDDEN as u64, EXPERTS as u64],
                ),
                routed_gate: weight_bytes(
                    ctx,
                    &routed_gate,
                    vec![HIDDEN as u64, FFN as u64, EXPERTS as u64],
                    GgmlType::IQ4_XS,
                ),
                routed_up: weight_bytes(
                    ctx,
                    &routed_up,
                    vec![HIDDEN as u64, FFN as u64, EXPERTS as u64],
                    GgmlType::IQ4_XS,
                ),
                routed_down: weight_bytes(
                    ctx,
                    &routed_down,
                    vec![FFN as u64, HIDDEN as u64, EXPERTS as u64],
                    GgmlType::IQ4_NL,
                ),
                shared_router: weight_f32(
                    ctx,
                    &values(HIDDEN, seed + 7, 0.003, 0.0),
                    vec![HIDDEN as u64],
                ),
                shared_gate: weight_bytes(
                    ctx,
                    &shared_gate,
                    vec![HIDDEN as u64, FFN as u64],
                    GgmlType::Q8_0,
                ),
                shared_up: weight_bytes(
                    ctx,
                    &shared_up,
                    vec![HIDDEN as u64, FFN as u64],
                    GgmlType::Q8_0,
                ),
                shared_down: weight_bytes(
                    ctx,
                    &shared_down,
                    vec![FFN as u64, HIDDEN as u64],
                    GgmlType::Q8_0,
                ),
            }
        }

        fn weights(&self, geometry: Qwen4ExpMoeMetalGeometry) -> Qwen4ExpMoeMetalWeights<'_> {
            Qwen4ExpMoeMetalWeights {
                geometry,
                router: &self.router,
                routed_gate: &self.routed_gate,
                routed_up: &self.routed_up,
                routed_down: &self.routed_down,
                shared_router: &self.shared_router,
                shared_gate: &self.shared_gate,
                shared_up: &self.shared_up,
                shared_down: &self.shared_down,
            }
        }
    }

    struct PleFixture {
        key_cpu: Vec<f32>,
        value_cpu: Vec<f32>,
        key_norm_cpu: Vec<f32>,
        query_norm_cpu: Vec<f32>,
        conv_norm_cpu: Vec<f32>,
        conv_cpu: Vec<f32>,
        key: MetalTensor,
        value: MetalTensor,
        key_norm: MetalTensor,
        query_norm: MetalTensor,
        conv_norm: MetalTensor,
        conv: MetalTensor,
    }

    impl PleFixture {
        fn new(ctx: &MetalContext) -> Self {
            let key_bytes = quantize_rows(
                &values(HIDDEN * HYPER, 81, 0.001, 0.0),
                GgmlType::Q8_0,
                HIDDEN,
            );
            let value_bytes = quantize_rows(
                &values(HIDDEN * HIDDEN, 82, 0.0012, 0.0),
                GgmlType::Q8_0,
                HIDDEN,
            );
            let key_cpu = dequant(&key_bytes, GgmlType::Q8_0, vec![HIDDEN, HYPER]);
            let value_cpu = dequant(&value_bytes, GgmlType::Q8_0, vec![HIDDEN, HIDDEN]);
            let key_norm_cpu = values(HYPER, 83, 0.002, 1.0);
            let query_norm_cpu = values(HYPER, 84, 0.002, 0.95);
            let conv_norm_cpu = values(HYPER, 85, 0.002, 1.05);
            let conv_cpu = values(HYPER * KERNEL, 86, 0.008, 0.025);
            Self {
                key: weight_bytes(
                    ctx,
                    &key_bytes,
                    vec![HIDDEN as u64, HYPER as u64],
                    GgmlType::Q8_0,
                ),
                value: weight_bytes(
                    ctx,
                    &value_bytes,
                    vec![HIDDEN as u64, HIDDEN as u64],
                    GgmlType::Q8_0,
                ),
                key_norm: weight_f32(ctx, &key_norm_cpu, vec![HYPER as u64]),
                query_norm: weight_f32(ctx, &query_norm_cpu, vec![HYPER as u64]),
                conv_norm: weight_f32(ctx, &conv_norm_cpu, vec![HYPER as u64]),
                conv: weight_f32(ctx, &conv_cpu, vec![KERNEL as u64, HYPER as u64]),
                key_cpu,
                value_cpu,
                key_norm_cpu,
                query_norm_cpu,
                conv_norm_cpu,
                conv_cpu,
            }
        }

        fn metal(&self, geometry: Qwen4ExpPleMetalGeometry) -> Qwen4ExpPleMetalWeights<'_> {
            Qwen4ExpPleMetalWeights {
                geometry,
                key: &self.key,
                value: &self.value,
                key_norm: &self.key_norm,
                query_norm: &self.query_norm,
                conv_norm: &self.conv_norm,
                conv: &self.conv,
            }
        }

        fn cpu(&self) -> PleStepWeights<'_> {
            PleStepWeights {
                key: &self.key_cpu,
                value: &self.value_cpu,
                key_norm: &self.key_norm_cpu,
                query_norm: &self.query_norm_cpu,
                conv_norm: &self.conv_norm_cpu,
                conv: &self.conv_cpu,
            }
        }
    }

    struct TableFixture {
        desc: TensorDesc,
        bytes: Vec<u8>,
    }

    impl TableFixture {
        fn new() -> Self {
            let source = values(TABLE_ROWS * HEAD_DIM, 91, 0.02, 0.0);
            let bytes = quantize_rows(&source, GgmlType::IQ4_NL, HEAD_DIM);
            Self {
                desc: TensorDesc {
                    name: "synthetic_ple_table".into(),
                    shape: vec![HEAD_DIM as u64, TABLE_ROWS as u64],
                    dtype: GgmlType::IQ4_NL,
                    shard_idx: 0,
                    data_offset: 0,
                    n_bytes: bytes.len() as u64,
                },
                bytes,
            }
        }

        fn table(&self) -> PleIq4NlTable<'_> {
            PleIq4NlTable::new(&self.desc, &self.bytes, TABLE_ROWS as u64).unwrap()
        }
    }

    struct SyntheticFixture {
        geometry: Qwen4ExpLayersZeroOneMetalGeometry,
        ple_config: PleConfig,
        token_cpu: Vec<f32>,
        token: MetalTensor,
        l0_attention: ResidualFixture,
        l0_gdn: GdnFixture,
        l0_ffn: ResidualFixture,
        l0_moe: MoeFixture,
        ple: PleFixture,
        l1_attention: ResidualFixture,
        l1_gdn: GdnFixture,
        l1_ffn: ResidualFixture,
        l1_moe: MoeFixture,
        table: TableFixture,
    }

    impl SyntheticFixture {
        fn new(ctx: &MetalContext) -> Self {
            let gdn =
                GatedDeltaNetMetalGeometry::new(HIDDEN, 1, 1, HEAD_DIM, KERNEL, 1e-6).unwrap();
            let moe = Qwen4ExpMoeMetalGeometry::new(HIDDEN, EXPERTS, TOP_K, FFN, FFN).unwrap();
            let layer_zero =
                Qwen4ExpLayerZeroMetalGeometry::new(BRANCHES, HIDDEN, RANK, VOCAB, 1e-6, gdn, moe)
                    .unwrap();
            let ple_geometry = Qwen4ExpPleMetalGeometry::new(
                BRANCHES, HIDDEN, HEADS, HEAD_DIM, KERNEL, DILATION, 1e-6,
            )
            .unwrap();
            let geometry =
                Qwen4ExpLayersZeroOneMetalGeometry::new(CONTEXT, 2, layer_zero, ple_geometry)
                    .unwrap();
            let token_bytes = quantize_rows(
                &values(HIDDEN * VOCAB, 3, 0.025, 0.0),
                GgmlType::Q8_0,
                HIDDEN,
            );
            let token_cpu = dequant(&token_bytes, GgmlType::Q8_0, vec![HIDDEN, VOCAB]);
            Self {
                geometry,
                ple_config: PleConfig {
                    token_vocab_size: VOCAB as u32,
                    layers: vec![1],
                    ngram_size: 3,
                    heads_per_ngram: 1,
                    embedding_head_dim: HEAD_DIM as u32,
                    conv_kernel: KERNEL as u32,
                    eos_token_id: 31,
                    image_token_id: Some(30),
                    multipliers: vec![3, 5, 7],
                    head_offsets: vec![0, 17],
                    head_vocab_sizes: vec![17, 19],
                },
                token: weight_bytes(
                    ctx,
                    &token_bytes,
                    vec![HIDDEN as u64, VOCAB as u64],
                    GgmlType::Q8_0,
                ),
                token_cpu,
                l0_attention: ResidualFixture::new(ctx, 10),
                l0_gdn: GdnFixture::new(ctx, gdn, 20),
                l0_ffn: ResidualFixture::new(ctx, 30),
                l0_moe: MoeFixture::new(ctx, 40),
                ple: PleFixture::new(ctx),
                l1_attention: ResidualFixture::new(ctx, 50),
                l1_gdn: GdnFixture::new(ctx, gdn, 60),
                l1_ffn: ResidualFixture::new(ctx, 70),
                l1_moe: MoeFixture::new(ctx, 80),
                table: TableFixture::new(),
            }
        }

        fn weights(&self) -> Qwen4ExpLayersZeroOneMetalWeights<'_> {
            let layer = self.geometry.layer_zero();
            Qwen4ExpLayersZeroOneMetalWeights {
                geometry: self.geometry,
                ple_config: &self.ple_config,
                layer_zero: Qwen4ExpLayerZeroMetalWeights {
                    geometry: layer,
                    token_embedding: &self.token,
                    attention_residual: self.l0_attention.metal(),
                    gdn: self.l0_gdn.weights(layer.gdn()),
                    ffn_residual: self.l0_ffn.metal(),
                    moe: self.l0_moe.weights(layer.moe()),
                },
                ple: self.ple.metal(self.geometry.ple()),
                layer_one_attention_residual: self.l1_attention.metal(),
                layer_one_gdn: self.l1_gdn.weights(layer.gdn()),
                layer_one_ffn_residual: self.l1_ffn.metal(),
                layer_one_moe: self.l1_moe.weights(layer.moe()),
            }
        }
    }

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal initialization failed: {error}"),
        }
    }

    fn values(count: usize, seed: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let raw = (index * 37 + index / 11 * 7 + seed * 13 + 5) % 127;
                bias + (raw as f32 - 63.0) * scale
            })
            .collect()
    }

    fn quantize_rows(values: &[f32], dtype: GgmlType, row_width: usize) -> Vec<u8> {
        assert!(!values.is_empty() && values.len().is_multiple_of(row_width));
        let (block, block_bytes) = dtype.storage_layout().unwrap();
        assert!((row_width as u64).is_multiple_of(block));
        let expected = values.len() / block as usize * block_bytes as usize;
        let mut bytes = vec![0_u8; expected];
        unsafe {
            llama_cpp_sys_2::ggml_quantize_init(dtype as u32);
            let written = llama_cpp_sys_2::ggml_quantize_chunk(
                dtype as u32,
                values.as_ptr(),
                bytes.as_mut_ptr().cast(),
                0,
                (values.len() / row_width) as i64,
                row_width as i64,
                std::ptr::null(),
            );
            assert_eq!(written, expected);
        }
        bytes
    }

    fn dequant(bytes: &[u8], dtype: GgmlType, shape: Vec<usize>) -> Vec<f32> {
        let desc = TensorDesc {
            name: format!("synthetic_{dtype:?}"),
            shape: shape.into_iter().map(|value| value as u64).collect(),
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        };
        crate::codec::dequant_to_f32(&desc, bytes).unwrap()
    }

    fn weight_bytes(
        ctx: &MetalContext,
        bytes: &[u8],
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> MetalTensor {
        let mut tensor = MetalTensor::from_bytes(ctx, bytes, shape, dtype).unwrap();
        tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
    }

    fn weight_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        weight_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32)
    }

    fn tensor_f32(ctx: &MetalContext, values: &[f32]) -> MetalTensor {
        MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(values),
            vec![values.len() as u64],
            GgmlType::F32,
        )
        .unwrap()
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        assert_eq!(tensor.dtype, GgmlType::F32);
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(source, tensor.n_elements() as usize).to_vec()
        }
    }

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = atol + rtol * expected.abs();
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "{label}[{index}]: expected {expected}, got {actual}, tolerance {tolerance}"
            );
        }
    }

    fn assert_similarity(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        maximum_absolute: f32,
        minimum_cosine: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(
            actual.iter().all(|value| value.is_finite()),
            "{label} finite"
        );
        let maximum = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let dot = actual
            .iter()
            .zip(expected)
            .map(|(&actual, &expected)| f64::from(actual) * f64::from(expected))
            .sum::<f64>();
        let actual_norm = actual
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let expected_norm = expected
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let cosine = dot / (actual_norm * expected_norm).sqrt();
        assert!(
            maximum <= maximum_absolute && cosine >= minimum_cosine,
            "{label}: max abs {maximum}, cosine {cosine}"
        );
    }

    fn max_delta(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max)
    }

    fn standalone_gdn(
        ctx: &MetalContext,
        input: &[f32],
        weights: GatedDeltaNetMetalWeights<'_>,
        workspace: &mut GatedDeltaNetMetalWorkspace,
    ) -> Vec<f32> {
        let input = tensor_f32(ctx, input);
        let output = MetalTensor::zeros_f32(ctx, vec![HIDDEN as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_gated_delta_net(ctx, &encoder, &input, weights, workspace).unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &output)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        read_f32(&output)
    }

    fn standalone_moe(
        ctx: &MetalContext,
        input: &[f32],
        weights: Qwen4ExpMoeMetalWeights<'_>,
        workspace: &mut Qwen4ExpMoeMetalWorkspace,
    ) -> Vec<f32> {
        let input = tensor_f32(ctx, input);
        let output = MetalTensor::zeros_f32(ctx, vec![HIDDEN as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_moe(ctx, &encoder, &input, weights, workspace).unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &output)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        read_f32(&output)
    }

    fn integrated_step(
        ctx: &MetalContext,
        fixture: &SyntheticFixture,
        workspace: &mut Qwen4ExpLayersZeroOneMetalWorkspace,
        token: u32,
        position: u64,
    ) -> Vec<f32> {
        let output = MetalTensor::zeros_f32(ctx, vec![HYPER as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_layers_zero_one(
            ctx,
            &encoder,
            token,
            position,
            fixture.table.table(),
            fixture.weights(),
            workspace,
        )
        .unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &output)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        read_f32(&output)
    }

    fn packed_steps(
        ctx: &MetalContext,
        fixture: &SyntheticFixture,
        workspace: &mut Qwen4ExpPackedPrefillWorkspace,
        tokens: &[u32],
        start_position: u64,
    ) -> (Vec<f32>, Vec<crate::metal::DispatchCensusRow>) {
        let output = MetalTensor::zeros_f32(ctx, vec![HYPER as u64, tokens.len() as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let read = encode_qwen4exp_packed_prefill(
            ctx,
            &encoder,
            tokens,
            start_position,
            fixture.table.table(),
            fixture.weights(),
            workspace,
        )
        .unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &output)
            .unwrap();
        let census = crate::metal::dispatch_census_take();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        (read_f32(&output), census)
    }

    #[test]
    fn packed_two_layer_prefix_matches_chronological_scalar_chunks() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let fixture = SyntheticFixture::new(&ctx);
        let tokens = (0..CONTEXT)
            .map(|index| ((index * 7 + index / 3 * 5 + 3) % 29) as u32)
            .collect::<Vec<_>>();
        let mut scalar_workspace =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, fixture.geometry).unwrap();
        let mut scalar = Vec::with_capacity(CONTEXT * HYPER);
        for (position, &token) in tokens.iter().enumerate() {
            scalar.extend(integrated_step(
                &ctx,
                &fixture,
                &mut scalar_workspace,
                token,
                position as u64,
            ));
        }

        for packed_tokens in [1_usize, 2, 8, 16, 33] {
            let mut workspace =
                Qwen4ExpPackedPrefillWorkspace::new(&ctx, fixture.geometry, packed_tokens).unwrap();
            let (actual, census) =
                packed_steps(&ctx, &fixture, &mut workspace, &tokens[..packed_tokens], 0);
            assert_eq!(
                census.len(),
                if packed_tokens == 1 { 84 } else { 87 },
                "N={packed_tokens} packed two-layer dispatch census: {census:#?}"
            );
            let count = |kernel: &str| census.iter().filter(|row| row.kernel == kernel).count();
            assert_eq!(
                count("kernel_qwen4exp_hc_repeat_packed_f32"),
                usize::from(packed_tokens > 1),
                "N={packed_tokens} repeat route: {census:#?}"
            );
            assert_eq!(
                count("kernel_qwen4exp_ple_conv_epilogue_packed_f32"),
                1,
                "N={packed_tokens} PLE chronology route: {census:#?}"
            );
            assert_eq!(
                count("kernel_gdn_step_decay_packed_f32")
                    + count("kernel_gdn_step_decay_packed_nsg4_f32"),
                2,
                "N={packed_tokens} GDN chronology route: {census:#?}"
            );
            assert_eq!(
                count("kernel_moe_route_bucket_slots_f32"),
                if packed_tokens == 1 { 0 } else { 2 },
                "N={packed_tokens} MoE route: {census:#?}"
            );
            for token in 0..packed_tokens {
                if packed_tokens == 1 {
                    assert!(
                        actual[token * HYPER..(token + 1) * HYPER]
                            .iter()
                            .zip(&scalar[token * HYPER..(token + 1) * HYPER])
                            .all(|(actual, expected)| actual.to_bits() == expected.to_bits()),
                        "packed N=1 must preserve the exact singleton route"
                    );
                }
                assert_similarity(
                    &format!("packed N={packed_tokens} token {token}"),
                    &actual[token * HYPER..(token + 1) * HYPER],
                    &scalar[token * HYPER..(token + 1) * HYPER],
                    6e-2,
                    0.999_999,
                );
            }
            assert_eq!(workspace.next_position(), Some(packed_tokens as u64));
            let retained_start = packed_tokens.saturating_sub(2);
            assert_eq!(
                workspace.prior_tokens(),
                &tokens[retained_start..packed_tokens]
            );

            let (continuation, _) = packed_steps(
                &ctx,
                &fixture,
                &mut workspace,
                &tokens[packed_tokens..packed_tokens + 1],
                packed_tokens as u64,
            );
            assert_similarity(
                &format!("packed N={packed_tokens} retained-state continuation"),
                &continuation,
                &scalar[packed_tokens * HYPER..(packed_tokens + 1) * HYPER],
                6e-2,
                0.999_999,
            );
            assert_eq!(workspace.next_position(), Some(packed_tokens as u64 + 1));
        }
    }

    #[test]
    fn packed_history_and_command_lifecycle_are_atomic_and_resettable() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let fixture = SyntheticFixture::new(&ctx);
        let mut workspace = Qwen4ExpPackedPrefillWorkspace::new(&ctx, fixture.geometry, 2).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64, 2]).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let error = encode_qwen4exp_packed_prefill(
            &ctx,
            &encoder,
            &[3, 7],
            1,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("fresh packed causal state"), "{error}");
        encoder.end();
        assert_eq!(workspace.next_position(), None);
        assert!(!workspace.is_poisoned());

        for block in 0..4 {
            for field in 0..4 {
                let source = match (block, field) {
                    (0, 0) => &fixture.l0_attention.norm,
                    (0, 1) => &fixture.l0_attention.down,
                    (0, 2) => &fixture.l0_attention.up,
                    (0, 3) => &fixture.l0_attention.inject,
                    (1, 0) => &fixture.l0_ffn.norm,
                    (1, 1) => &fixture.l0_ffn.down,
                    (1, 2) => &fixture.l0_ffn.up,
                    (1, 3) => &fixture.l0_ffn.inject,
                    (2, 0) => &fixture.l1_attention.norm,
                    (2, 1) => &fixture.l1_attention.down,
                    (2, 2) => &fixture.l1_attention.up,
                    (2, 3) => &fixture.l1_attention.inject,
                    (3, 0) => &fixture.l1_ffn.norm,
                    (3, 1) => &fixture.l1_ffn.down,
                    (3, 2) => &fixture.l1_ffn.up,
                    (3, 3) => &fixture.l1_ffn.inject,
                    _ => unreachable!(),
                };
                let mut writable = source.clone();
                writable.provenance = MetalTensorProvenance::OwnedWritable;
                let mut malformed = fixture.weights();
                let target = match block {
                    0 => &mut malformed.layer_zero.attention_residual,
                    1 => &mut malformed.layer_zero.ffn_residual,
                    2 => &mut malformed.layer_one_attention_residual,
                    3 => &mut malformed.layer_one_ffn_residual,
                    _ => unreachable!(),
                };
                match field {
                    0 => target.read.norm = &writable,
                    1 => target.read.down = &writable,
                    2 => target.read.up = &writable,
                    3 => target.inject = &writable,
                    _ => unreachable!(),
                }
                let command = ctx.queue.commandBuffer().unwrap();
                let encoder = KernelEncoder::begin(&command);
                let error = encode_qwen4exp_packed_prefill(
                    &ctx,
                    &encoder,
                    &[3, 7],
                    0,
                    fixture.table.table(),
                    malformed,
                    &mut workspace,
                )
                .err()
                .unwrap()
                .to_string();
                assert!(
                    error.contains("read-only"),
                    "block={block} field={field}: {error}"
                );
                encoder.end();
                workspace.release_after().unwrap();
                assert_eq!(workspace.next_position(), None);
                assert!(!workspace.is_poisoned());
            }
        }

        let mut writable_token = fixture.token.clone();
        writable_token.provenance = MetalTensorProvenance::OwnedWritable;
        let mut malformed = fixture.weights();
        malformed.layer_zero.token_embedding = &writable_token;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_qwen4exp_packed_prefill(
                &ctx,
                &encoder,
                &[3, 7],
                0,
                fixture.table.table(),
                malformed,
                &mut workspace,
            )
            .is_err()
        );
        encoder.end();
        assert_eq!(workspace.next_position(), None);
        assert!(!workspace.is_poisoned());

        let abandoned = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&abandoned);
        let read = encode_qwen4exp_packed_prefill(
            &ctx,
            &encoder,
            &[3, 7],
            0,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        let foreign = ctx.queue.commandBuffer().unwrap();
        let foreign_encoder = KernelEncoder::begin(&foreign);
        assert!(
            read.output()
                .encode_copy_to(&ctx, &foreign_encoder, &output)
                .is_err()
        );
        foreign_encoder.end();
        drop(read);
        assert!(workspace.release_after().is_err());
        assert!(workspace.reset().is_err());
        encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        drop(abandoned);
        assert_eq!(workspace.next_position(), None);
        assert!(workspace.prior_tokens().is_empty());

        let (first, _) = packed_steps(&ctx, &fixture, &mut workspace, &[3, 7], 0);
        assert_eq!(workspace.next_position(), Some(2));
        assert_eq!(workspace.prior_tokens(), &[3, 7]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let error = encode_qwen4exp_packed_prefill(
            &ctx,
            &encoder,
            &[11],
            3,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("discontinuous from 2"), "{error}");
        encoder.end();
        assert_eq!(workspace.next_position(), Some(2));

        workspace.reset().unwrap();
        assert_eq!(workspace.next_position(), None);
        assert!(workspace.prior_tokens().is_empty());
        let (reset, _) = packed_steps(&ctx, &fixture, &mut workspace, &[3, 7], 0);
        assert_close("reset packed causal state", &reset, &first, 1e-6, 1e-6);
    }

    #[test]
    fn sequential_two_layer_prefix_matches_independent_control_through_ple_lag_nine() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let fixture = SyntheticFixture::new(&ctx);
        fixture.ple_config.validate(HIDDEN as u32, 2).unwrap();
        let geometry = fixture.geometry;
        let layer = geometry.layer_zero();
        let mut history = PleHistory::default();
        let mut ple_state = PleConvState::fresh(HYPER, KERNEL, DILATION).unwrap();
        let mut l0_gdn = GatedDeltaNetMetalWorkspace::new(&ctx, layer.gdn()).unwrap();
        let mut l0_moe = Qwen4ExpMoeMetalWorkspace::new(&ctx, layer.moe()).unwrap();
        let mut l1_gdn = GatedDeltaNetMetalWorkspace::new(&ctx, layer.gdn()).unwrap();
        let mut l1_moe = Qwen4ExpMoeMetalWorkspace::new(&ctx, layer.moe()).unwrap();
        let mut integrated = Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, geometry).unwrap();
        let tokens = [3_u32, 7, 11, 5, 13, 2, 17, 19, 23, 29, 4, 9];
        let mut observed_lag_nine = false;
        let mut observed_retained_gdn = false;

        for (position, token) in tokens.into_iter().enumerate() {
            let embedding =
                fixture.token_cpu[token as usize * HIDDEN..(token as usize + 1) * HIDDEN].to_vec();
            let initial = (0..BRANCHES)
                .flat_map(|_| embedding.iter().copied())
                .collect::<Vec<_>>();
            let (mixed, state) = gated_residual_mix(
                &initial,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                fixture.l0_attention.cpu_read(),
            )
            .unwrap();
            let l0_gdn_output = standalone_gdn(
                &ctx,
                &mixed,
                fixture.l0_gdn.weights(layer.gdn()),
                &mut l0_gdn,
            );
            let after_attention =
                gated_residual_combine(&l0_gdn_output, &state, &fixture.l0_attention.inject_cpu)
                    .unwrap();
            let (mixed, state) = gated_residual_mix(
                &after_attention,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                fixture.l0_ffn.cpu_read(),
            )
            .unwrap();
            let l0_moe_output = standalone_moe(
                &ctx,
                &mixed,
                fixture.l0_moe.weights(layer.moe()),
                &mut l0_moe,
            );
            let layer_zero =
                gated_residual_combine(&l0_moe_output, &state, &fixture.l0_ffn.inject_cpu).unwrap();

            let (next_history, rows) = history
                .advanced(&fixture.ple_config, token, position as u64)
                .unwrap();
            let mut ple_embedding = vec![0.0_f32; HIDDEN];
            fixture
                .table
                .table()
                .gather_f32_into(&rows, &mut ple_embedding)
                .unwrap();
            let ple_output = ple_step(
                &ple_embedding,
                &layer_zero,
                BRANCHES,
                HIDDEN,
                KERNEL,
                DILATION,
                geometry.eps(),
                fixture.ple.cpu(),
                &mut ple_state,
            )
            .unwrap();
            if position >= 9 {
                let mut fresh = PleConvState::fresh(HYPER, KERNEL, DILATION).unwrap();
                let without_history = ple_step(
                    &ple_embedding,
                    &layer_zero,
                    BRANCHES,
                    HIDDEN,
                    KERNEL,
                    DILATION,
                    geometry.eps(),
                    fixture.ple.cpu(),
                    &mut fresh,
                )
                .unwrap();
                observed_lag_nine |= max_delta(&ple_output, &without_history) > 1e-4;
            }
            history = next_history;

            let (l1_mixed, state) = gated_residual_mix(
                &ple_output,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                fixture.l1_attention.cpu_read(),
            )
            .unwrap();
            let l1_gdn_output = standalone_gdn(
                &ctx,
                &l1_mixed,
                fixture.l1_gdn.weights(layer.gdn()),
                &mut l1_gdn,
            );
            if position == tokens.len() - 1 {
                let mut fresh = GatedDeltaNetMetalWorkspace::new(&ctx, layer.gdn()).unwrap();
                let fresh_output = standalone_gdn(
                    &ctx,
                    &l1_mixed,
                    fixture.l1_gdn.weights(layer.gdn()),
                    &mut fresh,
                );
                observed_retained_gdn = max_delta(&l1_gdn_output, &fresh_output) > 1e-5;
            }
            let after_attention =
                gated_residual_combine(&l1_gdn_output, &state, &fixture.l1_attention.inject_cpu)
                    .unwrap();
            let (l1_ffn_mixed, state) = gated_residual_mix(
                &after_attention,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                fixture.l1_ffn.cpu_read(),
            )
            .unwrap();
            let l1_moe_output = standalone_moe(
                &ctx,
                &l1_ffn_mixed,
                fixture.l1_moe.weights(layer.moe()),
                &mut l1_moe,
            );
            let expected =
                gated_residual_combine(&l1_moe_output, &state, &fixture.l1_ffn.inject_cpu).unwrap();
            assert!(l0_gdn_output.iter().any(|value| value.abs() > 1e-6));
            assert!(l0_moe_output.iter().any(|value| value.abs() > 1e-6));
            assert!(l1_gdn_output.iter().any(|value| value.abs() > 1e-6));
            assert!(l1_moe_output.iter().any(|value| value.abs() > 1e-6));

            let actual = integrated_step(&ctx, &fixture, &mut integrated, token, position as u64);
            assert_close(
                "layer-one GDN boundary",
                &read_f32(&integrated.mixer_output),
                &l1_gdn_output,
                1.5e-3,
                1.5e-3,
            );
            assert_close(
                "layer-one MoE boundary",
                &read_f32(&integrated.moe_output),
                &l1_moe_output,
                2e-3,
                2e-3,
            );
            assert_close("two-layer residual", &actual, &expected, 2.5e-3, 2e-3);
            assert_eq!(integrated.next_position(), Some(position as u64 + 1));
            let retained_start = (position + 1).saturating_sub(2);
            assert_eq!(
                integrated.prior_tokens(),
                &tokens[retained_start..=position]
            );
        }
        assert!(
            observed_lag_nine,
            "PLE lag-nine state had no measurable effect"
        );
        assert!(
            observed_retained_gdn,
            "layer-one GDN retained state had no measurable effect"
        );
    }

    #[test]
    fn history_and_command_lifecycle_are_atomic_and_resettable() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let fixture = SyntheticFixture::new(&ctx);
        let mut workspace =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, fixture.geometry).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();
        assert_eq!(workspace.next_position(), None);
        assert!(workspace.prior_tokens().is_empty());

        let discontinuous = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&discontinuous);
        let error = encode_qwen4exp_layers_zero_one(
            &ctx,
            &encoder,
            3,
            1,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("fresh causal state"));
        encoder.end();

        let mut writable_token = fixture.token.clone();
        writable_token.provenance = MetalTensorProvenance::OwnedWritable;
        let mut malformed = fixture.weights();
        malformed.layer_zero.token_embedding = &writable_token;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_qwen4exp_layers_zero_one(
                &ctx,
                &encoder,
                3,
                0,
                fixture.table.table(),
                malformed,
                &mut workspace,
            )
            .is_err()
        );
        encoder.end();
        assert!(workspace.active_command.is_none());
        assert!(workspace.pending_history.is_none());
        assert_eq!(workspace.next_position(), None);
        assert!(!workspace.is_poisoned());

        let mut writable_residual_norm = fixture.l1_attention.norm.clone();
        writable_residual_norm.provenance = MetalTensorProvenance::OwnedWritable;
        let mut malformed = fixture.weights();
        malformed.layer_one_attention_residual.read.norm = &writable_residual_norm;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_qwen4exp_layers_zero_one(
                &ctx,
                &encoder,
                3,
                0,
                fixture.table.table(),
                malformed,
                &mut workspace,
            )
            .is_err()
        );
        encoder.end();
        assert!(workspace.active_command.is_none());
        assert!(workspace.pending_history.is_none());
        assert_eq!(workspace.next_position(), None);
        assert!(!workspace.is_poisoned());

        let abandoned = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&abandoned);
        let read = encode_qwen4exp_layers_zero_one(
            &ctx,
            &encoder,
            3,
            0,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(read.workspace.next_position(), None);
        let foreign = ctx.queue.commandBuffer().unwrap();
        let foreign_encoder = KernelEncoder::begin(&foreign);
        assert!(
            read.output()
                .encode_copy_to(&ctx, &foreign_encoder, &output)
                .is_err()
        );
        foreign_encoder.end();
        drop(read);
        assert!(workspace.release_after().is_err());
        assert!(workspace.reset().is_err());
        let second = ctx.queue.commandBuffer().unwrap();
        let second_encoder = KernelEncoder::begin(&second);
        assert!(
            encode_qwen4exp_layers_zero_one(
                &ctx,
                &second_encoder,
                3,
                0,
                fixture.table.table(),
                fixture.weights(),
                &mut workspace,
            )
            .is_err()
        );
        second_encoder.end();
        encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        drop(abandoned);
        assert_eq!(workspace.next_position(), None);
        assert!(workspace.prior_tokens().is_empty());

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_layers_zero_one(
            &ctx,
            &encoder,
            3,
            0,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        read.output()
            .encode_copy_to(&ctx, &encoder, &output)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        assert_eq!(workspace.next_position(), None);
        assert!(unsafe { workspace.abandon_uncommitted() }.is_err());
        assert!(workspace.reset().is_err());
        workspace.release_after().unwrap();
        let first_output = read_f32(&output);
        assert_eq!(workspace.next_position(), Some(1));
        assert_eq!(workspace.prior_tokens(), &[3]);
        workspace.release_after().unwrap();
        assert_eq!(workspace.next_position(), Some(1));

        let discontinuous = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&discontinuous);
        let error = encode_qwen4exp_layers_zero_one(
            &ctx,
            &encoder,
            7,
            2,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("discontinuous from 1"));
        encoder.end();

        let abandoned = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&abandoned);
        let read = encode_qwen4exp_layers_zero_one(
            &ctx,
            &encoder,
            7,
            1,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        drop(read);
        encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        drop(abandoned);
        assert_eq!(workspace.next_position(), Some(1));
        assert_eq!(workspace.prior_tokens(), &[3]);

        let _ = integrated_step(&ctx, &fixture, &mut workspace, 7, 1);
        assert_eq!(workspace.next_position(), Some(2));
        assert_eq!(workspace.prior_tokens(), &[3, 7]);
        workspace.state_poisoned = true;
        workspace.reset().unwrap();
        assert!(!workspace.is_poisoned());
        assert_eq!(workspace.next_position(), None);
        assert!(workspace.prior_tokens().is_empty());
        let reset_output = integrated_step(&ctx, &fixture, &mut workspace, 3, 0);
        assert_close(
            "reset causal state",
            &reset_output,
            &first_output,
            1e-6,
            1e-6,
        );
    }

    fn control_layer_zero(
        ctx: &MetalContext,
        token: u32,
        weights: Qwen4ExpLayerZeroMetalWeights<'_>,
        workspace: &mut Qwen4ExpLayerZeroMetalWorkspace,
        output: &MetalTensor,
    ) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_layer_zero(ctx, &encoder, token, weights, workspace).unwrap();
        read.output().encode_copy_to(ctx, &encoder, output).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
    }

    fn control_ple(
        ctx: &MetalContext,
        input: &MetalTensor,
        rows: &[u32],
        table: PleIq4NlTable<'_>,
        weights: Qwen4ExpPleMetalWeights<'_>,
        workspace: &mut Qwen4ExpPleMetalWorkspace,
        output: &MetalTensor,
    ) {
        workspace.stage_rows(table, rows).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_ple(ctx, &encoder, input, weights, workspace).unwrap();
        read.output().encode_copy_to(ctx, &encoder, output).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    fn control_layer_one(
        ctx: &MetalContext,
        hyper: &MetalTensor,
        eps: f32,
        attention_weights: Qwen4ExpResidualMetalWeights<'_>,
        gdn_weights: GatedDeltaNetMetalWeights<'_>,
        ffn_weights: Qwen4ExpResidualMetalWeights<'_>,
        moe_weights: Qwen4ExpMoeMetalWeights<'_>,
        scratch: &mut GatedResidualMetalScratch,
        gdn_workspace: &mut GatedDeltaNetMetalWorkspace,
        moe_workspace: &mut Qwen4ExpMoeMetalWorkspace,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let hidden = gdn_weights.geometry.hidden_size();
        let gdn_output = MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap();
        let moe_output = MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let residual = encode_gated_residual_mix(
            ctx,
            &encoder,
            hyper,
            &gdn_output,
            eps,
            attention_weights.read,
            attention_weights.inject,
            scratch,
        )
        .unwrap();
        let gdn =
            encode_gated_delta_net(ctx, &encoder, residual.mixed(), gdn_weights, gdn_workspace)
                .unwrap();
        gdn.output()
            .encode_copy_to(ctx, &encoder, &gdn_output)
            .unwrap();
        drop(gdn);
        residual.encode_combine().unwrap();
        encoder.end();
        command.commit();
        gdn_workspace.release_after().unwrap();
        scratch.release_after().unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let residual = encode_gated_residual_mix(
            ctx,
            &encoder,
            hyper,
            &moe_output,
            eps,
            ffn_weights.read,
            ffn_weights.inject,
            scratch,
        )
        .unwrap();
        let moe = encode_qwen4exp_moe(ctx, &encoder, residual.mixed(), moe_weights, moe_workspace)
            .unwrap();
        moe.output()
            .encode_copy_to(ctx, &encoder, &moe_output)
            .unwrap();
        drop(moe);
        residual.encode_combine().unwrap();
        encoder.end();
        command.commit();
        moe_workspace.release_after().unwrap();
        scratch.release_after().unwrap();
        (
            read_f32(hyper),
            read_f32(&gdn_output),
            read_f32(&moe_output),
        )
    }

    fn assert_binding(actual: &MetalTensor, resident: &Qwen4ExpMetalWeights, name: &str) {
        let expected = resident
            .require_tensor(name)
            .unwrap_or_else(|_| panic!("missing released tensor {name}"));
        assert!(
            std::ptr::eq(actual, expected),
            "binding does not reference {name}"
        );
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_LAYERS_ZERO_ONE_GGUF to the pinned full release"]
    fn released_two_layer_prefix_matches_separate_command_primitives() {
        let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let resident = realized.weights();
        let weights = Qwen4ExpLayersZeroOneMetalWeights::bind(resident).unwrap();
        let table = resident.ple_source().bind(&gguf).unwrap();
        let geometry = weights.geometry;
        for (actual, name) in [
            (weights.ple.key, "blk.1.ple_key.weight"),
            (
                weights.layer_one_attention_residual.read.norm,
                "blk.1.hc_attn_norm.weight",
            ),
            (
                weights.layer_one_attention_residual.read.down,
                "blk.1.hc_attn_down.weight",
            ),
            (
                weights.layer_one_attention_residual.read.up,
                "blk.1.hc_attn_up.weight",
            ),
            (
                weights.layer_one_attention_residual.inject,
                "blk.1.hc_attn_inject.weight",
            ),
            (weights.layer_one_gdn.qkv, "blk.1.attn_qkv.weight"),
            (
                weights.layer_one_ffn_residual.read.norm,
                "blk.1.hc_ffn_norm.weight",
            ),
            (
                weights.layer_one_ffn_residual.read.down,
                "blk.1.hc_ffn_down.weight",
            ),
            (
                weights.layer_one_ffn_residual.read.up,
                "blk.1.hc_ffn_up.weight",
            ),
            (
                weights.layer_one_ffn_residual.inject,
                "blk.1.hc_ffn_inject.weight",
            ),
            (weights.layer_one_moe.router, "blk.1.ffn_gate_inp.weight"),
        ] {
            assert_binding(actual, resident, name);
        }
        assert_eq!(weights.layer_zero.token_embedding.dtype, GgmlType::Q8_0);
        assert_eq!(
            weights.layer_one_attention_residual.read.down.dtype,
            GgmlType::Q8_0
        );
        assert_eq!(
            weights.layer_one_attention_residual.read.up.dtype,
            GgmlType::Q8_0
        );
        assert_eq!(weights.ple.key.dtype, GgmlType::Q8_0);
        assert_eq!(weights.ple.value.dtype, GgmlType::Q8_0);
        assert_eq!(weights.layer_one_gdn.qkv.dtype, GgmlType::Q8_0);
        assert_eq!(weights.layer_one_gdn.gate.dtype, GgmlType::Q8_0);
        assert_eq!(weights.layer_one_gdn.output.dtype, GgmlType::Q8_0);
        assert_eq!(
            weights.layer_one_ffn_residual.read.down.dtype,
            GgmlType::Q8_0
        );
        assert_eq!(weights.layer_one_ffn_residual.read.up.dtype, GgmlType::Q8_0);
        assert_eq!(weights.layer_one_moe.routed_gate.dtype, GgmlType::IQ3_XXS);
        assert_eq!(weights.layer_one_moe.routed_up.dtype, GgmlType::IQ3_XXS);
        assert_eq!(weights.layer_one_moe.routed_down.dtype, GgmlType::IQ4_NL);
        assert_eq!(weights.layer_one_moe.shared_gate.dtype, GgmlType::Q8_0);
        assert_eq!(weights.layer_one_moe.shared_up.dtype, GgmlType::Q8_0);
        assert_eq!(weights.layer_one_moe.shared_down.dtype, GgmlType::Q8_0);

        let mut integrated = Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, geometry).unwrap();
        let mut layer_zero =
            Qwen4ExpLayerZeroMetalWorkspace::new(&ctx, geometry.layer_zero()).unwrap();
        let mut ple = Qwen4ExpPleMetalWorkspace::new(&ctx, geometry.ple()).unwrap();
        let mut residual = GatedResidualMetalScratch::new(
            &ctx,
            geometry.branch_count(),
            geometry.hidden_size(),
            geometry.low_rank(),
        )
        .unwrap();
        let mut gdn = GatedDeltaNetMetalWorkspace::new(&ctx, geometry.layer_zero().gdn()).unwrap();
        let mut moe = Qwen4ExpMoeMetalWorkspace::new(&ctx, geometry.layer_zero().moe()).unwrap();
        let layer_zero_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let ple_output = MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let integrated_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let mut history = PleHistory::default();

        for (position, token) in [35_u32, 201].into_iter().enumerate() {
            control_layer_zero(
                &ctx,
                token,
                weights.layer_zero,
                &mut layer_zero,
                &layer_zero_output,
            );
            let (next_history, rows) = history
                .advanced(weights.ple_config, token, position as u64)
                .unwrap();
            control_ple(
                &ctx,
                &layer_zero_output,
                &rows,
                table,
                weights.ple,
                &mut ple,
                &ple_output,
            );
            history = next_history;
            let (expected, expected_gdn, expected_moe) = control_layer_one(
                &ctx,
                &ple_output,
                geometry.eps(),
                weights.layer_one_attention_residual,
                weights.layer_one_gdn,
                weights.layer_one_ffn_residual,
                weights.layer_one_moe,
                &mut residual,
                &mut gdn,
                &mut moe,
            );

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_qwen4exp_layers_zero_one(
                &ctx,
                &encoder,
                token,
                position as u64,
                table,
                weights,
                &mut integrated,
            )
            .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &integrated_output)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            integrated.release_after().unwrap();

            assert_close(
                "released layer-one GDN boundary",
                &read_f32(&integrated.mixer_output),
                &expected_gdn,
                2e-5,
                2e-5,
            );
            assert_close(
                "released layer-one MoE boundary",
                &read_f32(&integrated.moe_output),
                &expected_moe,
                3e-5,
                3e-5,
            );
            assert_close(
                "released two-layer residual",
                &read_f32(&integrated_output),
                &expected,
                4e-5,
                4e-5,
            );
        }
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_LAYERS_ZERO_ONE_GGUF to the pinned full release"]
    fn released_packed_two_layer_prefix_matches_scalar_transaction() {
        let path = crate::test_fixtures::QWEN4EXP_Q3_K_XL.required();
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let resident = realized.weights();
        let weights = Qwen4ExpLayersZeroOneMetalWeights::bind(resident).unwrap();
        let table = resident.ple_source().bind(&gguf).unwrap();
        let geometry = weights.geometry;
        let tokens = [35_u32, 201, 17, 91, 5, 403, 29, 811];

        let scalar_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let mut scalar_workspace =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, geometry).unwrap();
        let mut scalar = Vec::with_capacity(tokens.len() * geometry.hyper_width());
        for (position, &token) in tokens.iter().enumerate() {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_qwen4exp_layers_zero_one(
                &ctx,
                &encoder,
                token,
                position as u64,
                table,
                weights,
                &mut scalar_workspace,
            )
            .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &scalar_output)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            scalar_workspace.release_after().unwrap();
            scalar.extend(read_f32(&scalar_output));
        }

        let packed_output = MetalTensor::zeros_f32(
            &ctx,
            vec![geometry.hyper_width() as u64, tokens.len() as u64],
        )
        .unwrap();
        let mut packed_workspace =
            Qwen4ExpPackedPrefillWorkspace::new(&ctx, geometry, tokens.len()).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let read = encode_qwen4exp_packed_prefill(
            &ctx,
            &encoder,
            &tokens,
            0,
            table,
            weights,
            &mut packed_workspace,
        )
        .unwrap();
        read.output()
            .encode_copy_to(&ctx, &encoder, &packed_output)
            .unwrap();
        let census = crate::metal::dispatch_census_take();
        drop(read);
        encoder.end();
        command.commit();
        packed_workspace.release_after().unwrap();
        assert_eq!(census.len(), 87, "released packed census: {census:#?}");

        let packed = read_f32(&packed_output);
        for token in 0..tokens.len() {
            let start = token * geometry.hyper_width();
            let end = start + geometry.hyper_width();
            assert_similarity(
                &format!("released packed token {token}"),
                &packed[start..end],
                &scalar[start..end],
                1e-4,
                0.999_999_9,
            );
        }
        assert_eq!(packed_workspace.next_position(), Some(tokens.len() as u64));
    }
}
