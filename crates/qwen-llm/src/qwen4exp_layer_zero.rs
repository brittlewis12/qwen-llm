//! One-token Metal composition for the first Qwen3.8-Flash-Next block.
//!
//! The released model applies PLE before zero-based layer 1, so layer 0 is the
//! largest prefix that can currently execute without the pending Metal PLE
//! path. This module owns the token embedding, four-stream residual, GDN state,
//! and MoE scratch needed to prove that the existing primitives compose in one
//! serial command.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_copy_offset_f32, encode_get_rows_f32,
};
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig, Qwen4ExpError};
use crate::qwen4exp_gdn::{
    GatedDeltaNetMetalGeometry, GatedDeltaNetMetalWeights, GatedDeltaNetMetalWorkspace,
    Qwen4ExpGdnError, encode_gated_delta_net,
};
use crate::qwen4exp_metal::{
    GatedResidualMetalReadWeights, GatedResidualMetalScratch, Qwen4ExpMetalError,
    encode_gated_residual_mix, validate_and_preflight_gated_residual_mix,
};
use crate::qwen4exp_moe::{
    Qwen4ExpMoeError, Qwen4ExpMoeMetalGeometry, Qwen4ExpMoeMetalWeights, Qwen4ExpMoeMetalWorkspace,
    encode_qwen4exp_moe,
};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource,
};

const LAYER: u32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpLayerZeroError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    GatedResidual(#[from] Qwen4ExpMetalError),
    #[error(transparent)]
    GatedDeltaNet(#[from] Qwen4ExpGdnError),
    #[error(transparent)]
    Moe(#[from] Qwen4ExpMoeError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next layer-zero contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next layer-zero command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpLayerZeroMetalGeometry {
    branch_count: usize,
    hidden_size: usize,
    low_rank: usize,
    vocab_size: usize,
    eps: f32,
    gdn: GatedDeltaNetMetalGeometry,
    moe: Qwen4ExpMoeMetalGeometry,
}

impl Qwen4ExpLayerZeroMetalGeometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        branch_count: usize,
        hidden_size: usize,
        low_rank: usize,
        vocab_size: usize,
        eps: f32,
        gdn: GatedDeltaNetMetalGeometry,
        moe: Qwen4ExpMoeMetalGeometry,
    ) -> Result<Self, Qwen4ExpLayerZeroError> {
        let geometry = Self {
            branch_count,
            hidden_size,
            low_rank,
            vocab_size,
            eps,
            gdn,
            moe,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(config: &Qwen4ExpConfig) -> Result<Self, Qwen4ExpLayerZeroError> {
        config.validate()?;
        if config.mixer_kind(LAYER) != Some(MixerKind::GatedDeltaNet) {
            return invalid("released layer 0 must use Gated DeltaNet");
        }
        if config
            .ple
            .as_ref()
            .is_some_and(|ple| ple.layers.contains(&LAYER))
        {
            return invalid("layer-zero composition cannot skip a configured PLE transform");
        }
        Self::new(
            config.hyper_connection.count as usize,
            config.hidden_size as usize,
            config.hyper_connection.low_rank as usize,
            config.vocab_size as usize,
            config.rms_norm_eps,
            GatedDeltaNetMetalGeometry::from_config(config)?,
            Qwen4ExpMoeMetalGeometry::from_config(config)?,
        )
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

    pub fn vocab_size(self) -> usize {
        self.vocab_size
    }

    pub fn hyper_width(self) -> usize {
        self.branch_count
            .checked_mul(self.hidden_size)
            .expect("validated layer-zero hyper width")
    }

    pub fn eps(self) -> f32 {
        self.eps
    }

    pub fn gdn(self) -> GatedDeltaNetMetalGeometry {
        self.gdn
    }

    pub fn moe(self) -> Qwen4ExpMoeMetalGeometry {
        self.moe
    }

    fn validate(self) -> Result<(), Qwen4ExpLayerZeroError> {
        if self.branch_count == 0
            || self.hidden_size == 0
            || self.low_rank == 0
            || self.vocab_size == 0
        {
            return invalid("branch, hidden, rank, and vocabulary dimensions must be nonzero");
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return invalid("RMS epsilon must be finite and positive");
        }
        if self.gdn.hidden_size() != self.hidden_size || self.moe.hidden_size() != self.hidden_size
        {
            return invalid("residual, GDN, and MoE hidden dimensions differ");
        }
        if self.vocab_size > i32::MAX as usize {
            return invalid("token IDs must fit signed i32");
        }
        let hyper_width = self
            .branch_count
            .checked_mul(self.hidden_size)
            .ok_or_else(|| {
                Qwen4ExpLayerZeroError::Invalid("hyper-residual width overflow".into())
            })?;
        for (name, value) in [
            ("branch count", self.branch_count),
            ("hidden size", self.hidden_size),
            ("low rank", self.low_rank),
            ("vocabulary size", self.vocab_size),
            ("hyper-residual width", hyper_width),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("{name} {value} exceeds u32"));
            }
        }
        hyper_width.checked_mul(4).ok_or_else(|| {
            Qwen4ExpLayerZeroError::Invalid("hyper-residual byte count overflow".into())
        })?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpResidualMetalWeights<'a> {
    pub read: GatedResidualMetalReadWeights<'a>,
    pub inject: &'a MetalTensor,
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpLayerZeroMetalWeights<'a> {
    pub geometry: Qwen4ExpLayerZeroMetalGeometry,
    pub token_embedding: &'a MetalTensor,
    pub attention_residual: Qwen4ExpResidualMetalWeights<'a>,
    pub gdn: GatedDeltaNetMetalWeights<'a>,
    pub ffn_residual: Qwen4ExpResidualMetalWeights<'a>,
    pub moe: Qwen4ExpMoeMetalWeights<'a>,
}

impl<'a> Qwen4ExpLayerZeroMetalWeights<'a> {
    pub fn bind(weights: &'a Qwen4ExpMetalWeights) -> Result<Self, Qwen4ExpLayerZeroError> {
        let geometry = Qwen4ExpLayerZeroMetalGeometry::from_config(weights.config())?;
        Ok(Self {
            geometry,
            token_embedding: weights.require_tensor("token_embd.weight")?,
            attention_residual: bind_residual(weights, "attn")?,
            gdn: GatedDeltaNetMetalWeights::bind(weights, LAYER)?,
            ffn_residual: bind_residual(weights, "ffn")?,
            moe: Qwen4ExpMoeMetalWeights::bind(weights, LAYER)?,
        })
    }
}

fn bind_residual<'a>(
    weights: &'a Qwen4ExpMetalWeights,
    role: &str,
) -> Result<Qwen4ExpResidualMetalWeights<'a>, Qwen4ExpLayerZeroError> {
    let prefix = format!("blk.{LAYER}.hc_{role}");
    Ok(Qwen4ExpResidualMetalWeights {
        read: GatedResidualMetalReadWeights {
            norm: weights.require_tensor(&format!("{prefix}_norm.weight"))?,
            down: weights.require_tensor(&format!("{prefix}_down.weight"))?,
            up: weights.require_tensor(&format!("{prefix}_up.weight"))?,
        },
        inject: weights.require_tensor(&format!("{prefix}_inject.weight"))?,
    })
}

pub struct Qwen4ExpLayerZeroMetalWorkspace {
    geometry: Qwen4ExpLayerZeroMetalGeometry,
    token_id: MetalTensor,
    embedding: MetalTensor,
    hyper_residual: MetalTensor,
    mixer_output: MetalTensor,
    moe_output: MetalTensor,
    residual: GatedResidualMetalScratch,
    gdn: GatedDeltaNetMetalWorkspace,
    moe: Qwen4ExpMoeMetalWorkspace,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
    encode_failed: bool,
}

impl Qwen4ExpLayerZeroMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpLayerZeroMetalGeometry,
    ) -> Result<Self, Qwen4ExpLayerZeroError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            token_id: MetalTensor::zeros_i32(ctx, vec![1])?,
            embedding: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            hyper_residual: MetalTensor::zeros_f32(ctx, vec![geometry.hyper_width() as u64])?,
            mixer_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            moe_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            residual: GatedResidualMetalScratch::new(
                ctx,
                geometry.branch_count,
                geometry.hidden_size,
                geometry.low_rank,
            )?,
            gdn: GatedDeltaNetMetalWorkspace::new(ctx, geometry.gdn)?,
            moe: Qwen4ExpMoeMetalWorkspace::new(ctx, geometry.moe)?,
            active_command: None,
            state_poisoned: false,
            encode_failed: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpLayerZeroMetalGeometry {
        self.geometry
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpLayerZeroError> {
        self.require_idle()?;
        self.gdn.reset()?;
        self.moe.reset()?;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpLayerZeroError> {
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
        if let Err(error) = self.gdn.release_after() {
            child_errors.push(format!("GDN: {error}"));
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
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpLayerZeroError::CommandBuffer(format!(
                "status={status:?}, error={command_error:?}, encode_failed={}, children={child_errors:?}",
                self.encode_failed
            )))
        }
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may mutate GDN state or scratch after
    /// another token has acquired the workspace.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpLayerZeroError> {
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
        if let Err(error) = unsafe { self.gdn.abandon_uncommitted() } {
            child_errors.push(format!("GDN: {error}"));
        }
        if let Err(error) = unsafe { self.moe.abandon_uncommitted() } {
            child_errors.push(format!("MoE: {error}"));
        }
        if !child_errors.is_empty() {
            return invalid(format!(
                "could not abandon every layer-zero child: {child_errors:?}"
            ));
        }
        self.active_command = None;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpLayerZeroError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "copy the layer-zero residual in its owning command, then release the workspace"]
pub struct Qwen4ExpLayerZeroMetalRead<'a> {
    workspace: &'a mut Qwen4ExpLayerZeroMetalWorkspace,
}

pub struct Qwen4ExpLayerZeroMetalOutput<'a> {
    workspace: &'a Qwen4ExpLayerZeroMetalWorkspace,
}

impl Qwen4ExpLayerZeroMetalRead<'_> {
    pub fn output(&self) -> Qwen4ExpLayerZeroMetalOutput<'_> {
        Qwen4ExpLayerZeroMetalOutput {
            workspace: self.workspace,
        }
    }
}

impl Qwen4ExpLayerZeroMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hyper_width() as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn branch_count(&self) -> usize {
        self.workspace.geometry.branch_count
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpLayerZeroError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("layer-zero output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("layer-zero output must be copied by its owning command buffer");
        }
        require_tensor(
            "layer-zero copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hyper_width() as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("layer-zero hyper residual", &self.workspace.hyper_residual),
                ("layer-zero copied output destination", destination),
            ],
        )?;
        let mut tensors = top_level_workspace_tensors(self.workspace);
        tensors.push(("layer-zero copied output destination", destination));
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

pub fn encode_qwen4exp_layer_zero<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    weights: Qwen4ExpLayerZeroMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpLayerZeroMetalWorkspace,
) -> Result<Qwen4ExpLayerZeroMetalRead<'a>, Qwen4ExpLayerZeroError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if weights.geometry != workspace.geometry {
        return invalid("layer-zero weight and workspace geometry differ");
    }
    if weights.gdn.geometry != workspace.geometry.gdn {
        return invalid("layer-zero GDN weight and workspace geometry differ");
    }
    if weights.moe.geometry != workspace.geometry.moe {
        return invalid("layer-zero MoE weight and workspace geometry differ");
    }
    if token_id as usize >= workspace.geometry.vocab_size {
        return invalid(format!(
            "token ID {token_id} is outside vocabulary {}",
            workspace.geometry.vocab_size
        ));
    }
    validate_contract(ctx, weights, workspace)?;
    write_token_id(&workspace.token_id, token_id as i32)?;
    reserve_command(workspace, enc)?;

    if let Err(error) = encode_step(ctx, enc, weights, workspace) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpLayerZeroMetalRead { workspace })
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: Qwen4ExpLayerZeroMetalWeights<'_>,
    workspace: &mut Qwen4ExpLayerZeroMetalWorkspace,
) -> Result<(), Qwen4ExpLayerZeroError> {
    let g = workspace.geometry;
    encode_get_rows_f32(
        ctx,
        enc,
        weights.token_embedding,
        &workspace.token_id,
        &workspace.embedding,
        1,
        g.hidden_size,
    )?;
    for branch in 0..g.branch_count {
        let destination = workspace
            .hyper_residual
            .view_subrange((branch * g.hidden_size) as u64, vec![g.hidden_size as u64]);
        encode_copy_offset_f32(
            ctx,
            enc,
            &workspace.embedding,
            0,
            &destination,
            g.hidden_size,
        )?;
    }

    let attention_read = encode_gated_residual_mix(
        ctx,
        enc,
        &workspace.hyper_residual,
        &workspace.mixer_output,
        g.eps,
        weights.attention_residual.read,
        weights.attention_residual.inject,
        &mut workspace.residual,
    )?;
    let gdn_read = encode_gated_delta_net(
        ctx,
        enc,
        attention_read.mixed(),
        weights.gdn,
        &mut workspace.gdn,
    )?;
    gdn_read
        .output()
        .encode_copy_to(ctx, enc, &workspace.mixer_output)?;
    drop(gdn_read);
    attention_read.encode_combine()?;

    let ffn_read = encode_gated_residual_mix(
        ctx,
        enc,
        &workspace.hyper_residual,
        &workspace.moe_output,
        g.eps,
        weights.ffn_residual.read,
        weights.ffn_residual.inject,
        &mut workspace.residual,
    )?;
    let moe_read =
        encode_qwen4exp_moe(ctx, enc, ffn_read.mixed(), weights.moe, &mut workspace.moe)?;
    moe_read
        .output()
        .encode_copy_to(ctx, enc, &workspace.moe_output)?;
    drop(moe_read);
    ffn_read.encode_combine()?;
    Ok(())
}

pub(crate) fn validate_contract(
    ctx: &MetalContext,
    weights: Qwen4ExpLayerZeroMetalWeights<'_>,
    workspace: &Qwen4ExpLayerZeroMetalWorkspace,
) -> Result<(), Qwen4ExpLayerZeroError> {
    let g = workspace.geometry;
    require_projection(
        "token embedding",
        weights.token_embedding,
        g.hidden_size,
        g.vocab_size,
        &[GgmlType::Q8_0],
    )?;
    for (name, tensor, dtype, shape) in [
        ("token ID", &workspace.token_id, GgmlType::I32, vec![1]),
        (
            "token embedding output",
            &workspace.embedding,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
        (
            "hyper residual",
            &workspace.hyper_residual,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "mixer output",
            &workspace.mixer_output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
        (
            "MoE output",
            &workspace.moe_output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, true)?;
    }
    validate_residual(
        ctx,
        "attention",
        &workspace.hyper_residual,
        &workspace.mixer_output,
        weights.attention_residual,
        &workspace.residual,
        g,
    )?;
    crate::qwen4exp_gdn::validate_contract(
        ctx,
        workspace.residual.mixed_tensor(),
        weights.gdn,
        &workspace.gdn,
    )?;
    crate::qwen4exp_gdn::preflight(ctx, weights.gdn)?;
    validate_residual(
        ctx,
        "FFN",
        &workspace.hyper_residual,
        &workspace.moe_output,
        weights.ffn_residual,
        &workspace.residual,
        g,
    )?;
    crate::qwen4exp_moe::validate_contract(
        ctx,
        workspace.residual.mixed_tensor(),
        weights.moe,
        &workspace.moe,
    )?;
    crate::qwen4exp_moe::preflight(ctx, weights.moe)?;

    let named_weights = named_weight_tensors(weights);
    require_read_only_weights(&named_weights)?;
    let mut tensors = named_weights;
    tensors.extend(top_level_workspace_tensors(workspace));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)?;

    let get_rows = ctx.pipeline("kernel_get_rows_q8_0_f32")?;
    if get_rows.threadExecutionWidth() != 32 || get_rows.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid(format!(
            "Q8_0 row pipeline requires SIMD width and capacity 32, got width={} capacity={}",
            get_rows.threadExecutionWidth(),
            get_rows.maxTotalThreadsPerThreadgroup()
        ));
    }
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_residual(
    ctx: &MetalContext,
    role: &str,
    hyper_residual: &MetalTensor,
    block_output: &MetalTensor,
    weights: Qwen4ExpResidualMetalWeights<'_>,
    scratch: &GatedResidualMetalScratch,
    geometry: Qwen4ExpLayerZeroMetalGeometry,
) -> Result<(), Qwen4ExpLayerZeroError> {
    let hyper = geometry.hyper_width();
    require_tensor(
        &format!("{role} residual norm"),
        weights.read.norm,
        GgmlType::F32,
        &[hyper as u64],
        false,
    )?;
    require_projection(
        &format!("{role} residual down"),
        weights.read.down,
        hyper,
        geometry.low_rank,
        &[GgmlType::F32, GgmlType::Q8_0],
    )?;
    require_projection(
        &format!("{role} residual up"),
        weights.read.up,
        geometry.low_rank,
        hyper,
        &[GgmlType::F32, GgmlType::Q8_0],
    )?;
    require_projection(
        &format!("{role} residual injection"),
        weights.inject,
        hyper,
        geometry.branch_count,
        &[GgmlType::F32],
    )?;
    validate_and_preflight_gated_residual_mix(
        ctx,
        hyper_residual,
        block_output,
        geometry.eps,
        weights.read,
        weights.inject,
        scratch,
    )?;
    Ok(())
}

fn named_weight_tensors(
    weights: Qwen4ExpLayerZeroMetalWeights<'_>,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("token embedding", weights.token_embedding),
        ("attention HC norm", weights.attention_residual.read.norm),
        ("attention HC down", weights.attention_residual.read.down),
        ("attention HC up", weights.attention_residual.read.up),
        ("attention HC injection", weights.attention_residual.inject),
        ("GDN QKV", weights.gdn.qkv),
        ("GDN gate", weights.gdn.gate),
        ("GDN beta", weights.gdn.beta),
        ("GDN alpha", weights.gdn.alpha),
        ("GDN A", weights.gdn.a),
        ("GDN dt bias", weights.gdn.dt_bias),
        ("GDN convolution", weights.gdn.conv),
        ("GDN norm", weights.gdn.norm),
        ("GDN output", weights.gdn.output),
        ("FFN HC norm", weights.ffn_residual.read.norm),
        ("FFN HC down", weights.ffn_residual.read.down),
        ("FFN HC up", weights.ffn_residual.read.up),
        ("FFN HC injection", weights.ffn_residual.inject),
        ("MoE router", weights.moe.router),
        ("MoE routed gate", weights.moe.routed_gate),
        ("MoE routed up", weights.moe.routed_up),
        ("MoE routed down", weights.moe.routed_down),
        ("MoE shared router", weights.moe.shared_router),
        ("MoE shared gate", weights.moe.shared_gate),
        ("MoE shared up", weights.moe.shared_up),
        ("MoE shared down", weights.moe.shared_down),
    ]
}

fn top_level_workspace_tensors(
    workspace: &Qwen4ExpLayerZeroMetalWorkspace,
) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("token ID", &workspace.token_id),
        ("token embedding output", &workspace.embedding),
        ("hyper residual", &workspace.hyper_residual),
        ("mixer output", &workspace.mixer_output),
        ("MoE output", &workspace.moe_output),
        ("residual mixed output", workspace.residual.mixed_tensor()),
    ]
}

fn validate_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), Qwen4ExpLayerZeroError> {
    let actual = enc.parent_command_buffer().device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("layer-zero dependent dispatches require a serial encoder");
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpLayerZeroMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpLayerZeroError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.encode_failed = false;
    Ok(())
}

fn write_token_id(tensor: &MetalTensor, token_id: i32) -> Result<(), Qwen4ExpLayerZeroError> {
    require_tensor("token ID", tensor, GgmlType::I32, &[1], true)?;
    let offset = usize::try_from(tensor.offset)
        .map_err(|_| Qwen4ExpLayerZeroError::Invalid("token ID offset exceeds usize".into()))?;
    if offset + size_of::<i32>() > tensor.buffer.length() {
        return invalid("token ID write exceeds its Metal buffer");
    }
    unsafe {
        tensor
            .buffer
            .contents()
            .as_ptr()
            .cast::<u8>()
            .add(offset)
            .cast::<i32>()
            .write(token_id);
    }
    Ok(())
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    allowed_dtypes: &[GgmlType],
) -> Result<(), Qwen4ExpLayerZeroError> {
    let shape = [n_in as u64, n_out as u64];
    if tensor.shape != shape || !allowed_dtypes.contains(&tensor.dtype) {
        return invalid(format!(
            "{name} must use {allowed_dtypes:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    let (block, _) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpLayerZeroError::Invalid(format!("{name} has unsupported dtype {:?}", tensor.dtype))
    })?;
    if !(n_in as u64).is_multiple_of(block) {
        return invalid(format!(
            "{name} row width {n_in} is not aligned to {block} elements"
        ));
    }
    require_range(name, tensor)
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpLayerZeroError> {
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

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpLayerZeroError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| Qwen4ExpLayerZeroError::Invalid("tensor element count overflow".into()))?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpLayerZeroError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
    })?;
    if block == 0 || !elements.is_multiple_of(block) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    (elements / block)
        .checked_mul(bytes)
        .ok_or_else(|| Qwen4ExpLayerZeroError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpLayerZeroError> {
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
        .ok_or_else(|| Qwen4ExpLayerZeroError::Invalid(format!("{name} range overflow")))?;
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
) -> Result<(), Qwen4ExpLayerZeroError> {
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
) -> Result<(), Qwen4ExpLayerZeroError> {
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

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpLayerZeroError> {
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

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpLayerZeroError> {
    Err(Qwen4ExpLayerZeroError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::qwen4exp_forward::{
        GatedResidualReadWeights, gated_residual_combine, gated_residual_mix,
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

    struct CpuResidualWeights {
        norm: Vec<f32>,
        down: Vec<f32>,
        up: Vec<f32>,
        inject: Vec<f32>,
    }

    impl CpuResidualWeights {
        fn read(&self) -> GatedResidualReadWeights<'_> {
            GatedResidualReadWeights {
                norm: &self.norm,
                down: &self.down,
                up: &self.up,
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

    fn geometry() -> Qwen4ExpLayerZeroMetalGeometry {
        Qwen4ExpLayerZeroMetalGeometry::new(
            BRANCHES,
            HIDDEN,
            RANK,
            VOCAB,
            1e-6,
            GatedDeltaNetMetalGeometry::new(HIDDEN, 1, 1, 128, 4, 1e-6).unwrap(),
            Qwen4ExpMoeMetalGeometry::new(HIDDEN, EXPERTS, TOP_K, FFN, FFN).unwrap(),
        )
        .unwrap()
    }

    fn values(count: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let raw = (index * 37 + index / 11 * 7 + seed * 13 + 5) % 127;
                (raw as f32 - 63.0) * scale
            })
            .collect()
    }

    fn tensor_f32(ctx: &MetalContext, data: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(data), shape, GgmlType::F32).unwrap()
    }

    fn weight_f32(ctx: &MetalContext, data: &[f32], shape: Vec<u64>) -> MetalTensor {
        let mut tensor = tensor_f32(ctx, data, shape);
        tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
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

    fn encode_q8_0_block(d: f32, seed: usize) -> [u8; 34] {
        let mut block = [0_u8; 34];
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for lane in 0..32 {
            let quant = ((seed * 13 + lane * 7 + 5) % 31) as i8 - 15;
            block[2 + lane] = quant as u8;
        }
        block
    }

    fn q8_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
        assert!(n_in.is_multiple_of(32));
        let blocks_per_row = n_in / 32;
        let mut bytes = Vec::with_capacity(experts * n_out * blocks_per_row * 34);
        for expert in 0..experts {
            for row in 0..n_out {
                for block in 0..blocks_per_row {
                    let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                    let sign = if ordinal.is_multiple_of(3) { -1.0 } else { 1.0 };
                    let d = sign * (ordinal % 5 + 1) as f32 / 2_048.0;
                    bytes.extend_from_slice(&encode_q8_0_block(d, ordinal));
                }
            }
        }
        bytes
    }

    fn quantize_rows(values: &[f32], dtype: GgmlType, n_per_row: usize) -> Vec<u8> {
        assert!(values.len().is_multiple_of(n_per_row));
        let (block, bytes_per_block) = dtype.storage_layout().unwrap();
        assert!((n_per_row as u64).is_multiple_of(block));
        let expected = values.len() / block as usize * bytes_per_block as usize;
        let mut bytes = vec![0_u8; expected];
        unsafe {
            llama_cpp_sys_2::ggml_quantize_init(dtype as u32);
            let written = llama_cpp_sys_2::ggml_quantize_chunk(
                dtype as u32,
                values.as_ptr(),
                bytes.as_mut_ptr().cast(),
                0,
                (values.len() / n_per_row) as i64,
                n_per_row as i64,
                std::ptr::null(),
            );
            assert_eq!(written, expected);
        }
        bytes
    }

    fn dequant_matrix(bytes: &[u8], dtype: GgmlType, n_in: usize, n_out: usize) -> Vec<f32> {
        let desc = TensorDesc {
            name: format!("synthetic_{dtype:?}"),
            shape: vec![n_in as u64, n_out as u64],
            dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        };
        crate::codec::dequant_to_f32(&desc, bytes).unwrap()
    }

    fn standalone_gdn(
        ctx: &MetalContext,
        input: &[f32],
        weights: GatedDeltaNetMetalWeights<'_>,
        workspace: &mut GatedDeltaNetMetalWorkspace,
    ) -> Vec<f32> {
        let hidden = weights.geometry.hidden_size();
        let input_gpu = tensor_f32(ctx, input, vec![hidden as u64]);
        let output_gpu = MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_gated_delta_net(ctx, &encoder, &input_gpu, weights, workspace).unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &output_gpu)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        read_f32(&output_gpu)
    }

    fn standalone_moe(
        ctx: &MetalContext,
        input: &[f32],
        weights: Qwen4ExpMoeMetalWeights<'_>,
        workspace: &mut Qwen4ExpMoeMetalWorkspace,
    ) -> Vec<f32> {
        let hidden = weights.geometry.hidden_size();
        let input_gpu = tensor_f32(ctx, input, vec![hidden as u64]);
        let output_gpu = MetalTensor::zeros_f32(ctx, vec![hidden as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_moe(ctx, &encoder, &input_gpu, weights, workspace).unwrap();
        read.output()
            .encode_copy_to(ctx, &encoder, &output_gpu)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        read_f32(&output_gpu)
    }

    #[test]
    fn released_geometry_stops_before_second_block_ple() {
        let config = Qwen4ExpConfig::flash_next_reference();
        assert_eq!(config.ple.as_ref().unwrap().layers, [1]);
        let geometry = Qwen4ExpLayerZeroMetalGeometry::from_config(&config).unwrap();
        assert_eq!(geometry.branch_count(), 4);
        assert_eq!(geometry.hidden_size(), 2_560);
        assert_eq!(geometry.low_rank(), 320);
        assert_eq!(geometry.vocab_size(), 248_320);
        assert_eq!(geometry.hyper_width(), 10_240);
    }

    #[test]
    fn two_tokens_compose_embedding_gdn_and_moe_in_one_command() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let geometry = geometry();
        let token_bytes = q8_bank(HIDDEN, VOCAB, 1, 11);
        let token_f32 = dequant_matrix(&token_bytes, GgmlType::Q8_0, HIDDEN, VOCAB);

        let attention_cpu = CpuResidualWeights {
            norm: values(HYPER, 1, 0.002)
                .into_iter()
                .map(|value| value + 1.0)
                .collect(),
            down: values(HYPER * RANK, 2, 0.0005),
            up: values(RANK * HYPER, 3, 0.001),
            inject: values(HYPER * BRANCHES, 4, 0.0005),
        };
        let ffn_cpu = CpuResidualWeights {
            norm: values(HYPER, 5, 0.002)
                .into_iter()
                .map(|value| value + 1.0)
                .collect(),
            down: values(HYPER * RANK, 6, 0.0005),
            up: values(RANK * HYPER, 7, 0.001),
            inject: values(HYPER * BRANCHES, 8, 0.0005),
        };

        let qkv = values(HIDDEN * geometry.gdn().conv_width(), 20, 0.0003);
        let gdn_gate = values(HIDDEN * geometry.gdn().value_width(), 21, 0.0004);
        let beta = values(HIDDEN, 22, 0.001);
        let alpha = values(HIDDEN, 23, 0.001);
        let transformed_a = vec![-0.55];
        let dt_bias = vec![0.15];
        let conv = values(geometry.gdn().conv_width() * 4, 24, 0.003);
        let gdn_norm = values(128, 25, 0.002)
            .into_iter()
            .map(|value| value + 1.0)
            .collect::<Vec<_>>();
        let gdn_output = values(geometry.gdn().value_width() * HIDDEN, 26, 0.0005);

        let router = values(HIDDEN * EXPERTS, 30, 0.001);
        let shared_router = values(HIDDEN, 31, 0.002);
        let routed_gate_values = values(HIDDEN * FFN * EXPERTS, 32, 0.002);
        let routed_up_values = values(HIDDEN * FFN * EXPERTS, 33, 0.002);
        let routed_gate_bytes = quantize_rows(&routed_gate_values, GgmlType::IQ4_XS, HIDDEN);
        let routed_up_bytes = quantize_rows(&routed_up_values, GgmlType::IQ4_XS, HIDDEN);
        let routed_down_bytes = q8_bank(FFN, HIDDEN, EXPERTS, 34);
        let shared_gate_bytes = q8_bank(HIDDEN, FFN, 1, 35);
        let shared_up_bytes = q8_bank(HIDDEN, FFN, 1, 36);
        let shared_down_bytes = q8_bank(FFN, HIDDEN, 1, 37);

        let token_gpu = weight_bytes(
            &ctx,
            &token_bytes,
            vec![HIDDEN as u64, VOCAB as u64],
            GgmlType::Q8_0,
        );
        let attention_norm_gpu = weight_f32(&ctx, &attention_cpu.norm, vec![HYPER as u64]);
        let attention_down_gpu =
            weight_f32(&ctx, &attention_cpu.down, vec![HYPER as u64, RANK as u64]);
        let attention_up_gpu = weight_f32(&ctx, &attention_cpu.up, vec![RANK as u64, HYPER as u64]);
        let attention_inject_gpu = weight_f32(
            &ctx,
            &attention_cpu.inject,
            vec![HYPER as u64, BRANCHES as u64],
        );
        let ffn_norm_gpu = weight_f32(&ctx, &ffn_cpu.norm, vec![HYPER as u64]);
        let ffn_down_gpu = weight_f32(&ctx, &ffn_cpu.down, vec![HYPER as u64, RANK as u64]);
        let ffn_up_gpu = weight_f32(&ctx, &ffn_cpu.up, vec![RANK as u64, HYPER as u64]);
        let ffn_inject_gpu = weight_f32(&ctx, &ffn_cpu.inject, vec![HYPER as u64, BRANCHES as u64]);

        let qkv_gpu = weight_f32(
            &ctx,
            &qkv,
            vec![HIDDEN as u64, geometry.gdn().conv_width() as u64],
        );
        let gdn_gate_gpu = weight_f32(
            &ctx,
            &gdn_gate,
            vec![HIDDEN as u64, geometry.gdn().value_width() as u64],
        );
        let beta_gpu = weight_f32(&ctx, &beta, vec![HIDDEN as u64, 1]);
        let alpha_gpu = weight_f32(&ctx, &alpha, vec![HIDDEN as u64, 1]);
        let a_gpu = weight_f32(&ctx, &transformed_a, vec![1]);
        let dt_gpu = weight_f32(&ctx, &dt_bias, vec![1]);
        let conv_gpu = weight_f32(&ctx, &conv, vec![4, geometry.gdn().conv_width() as u64]);
        let gdn_norm_gpu = weight_f32(&ctx, &gdn_norm, vec![128]);
        let gdn_output_gpu = weight_f32(
            &ctx,
            &gdn_output,
            vec![geometry.gdn().value_width() as u64, HIDDEN as u64],
        );
        let gdn_weights = GatedDeltaNetMetalWeights {
            geometry: geometry.gdn(),
            qkv: &qkv_gpu,
            gate: &gdn_gate_gpu,
            beta: &beta_gpu,
            alpha: &alpha_gpu,
            a: &a_gpu,
            dt_bias: &dt_gpu,
            conv: &conv_gpu,
            norm: &gdn_norm_gpu,
            output: &gdn_output_gpu,
        };

        let router_gpu = weight_f32(&ctx, &router, vec![HIDDEN as u64, EXPERTS as u64]);
        let routed_gate_gpu = weight_bytes(
            &ctx,
            &routed_gate_bytes,
            vec![HIDDEN as u64, FFN as u64, EXPERTS as u64],
            GgmlType::IQ4_XS,
        );
        let routed_up_gpu = weight_bytes(
            &ctx,
            &routed_up_bytes,
            vec![HIDDEN as u64, FFN as u64, EXPERTS as u64],
            GgmlType::IQ4_XS,
        );
        let routed_down_gpu = weight_bytes(
            &ctx,
            &routed_down_bytes,
            vec![FFN as u64, HIDDEN as u64, EXPERTS as u64],
            GgmlType::Q8_0,
        );
        let shared_router_gpu = weight_f32(&ctx, &shared_router, vec![HIDDEN as u64]);
        let shared_gate_gpu = weight_bytes(
            &ctx,
            &shared_gate_bytes,
            vec![HIDDEN as u64, FFN as u64],
            GgmlType::Q8_0,
        );
        let shared_up_gpu = weight_bytes(
            &ctx,
            &shared_up_bytes,
            vec![HIDDEN as u64, FFN as u64],
            GgmlType::Q8_0,
        );
        let shared_down_gpu = weight_bytes(
            &ctx,
            &shared_down_bytes,
            vec![FFN as u64, HIDDEN as u64],
            GgmlType::Q8_0,
        );
        let moe_weights = Qwen4ExpMoeMetalWeights {
            geometry: geometry.moe(),
            router: &router_gpu,
            routed_gate: &routed_gate_gpu,
            routed_up: &routed_up_gpu,
            routed_down: &routed_down_gpu,
            shared_router: &shared_router_gpu,
            shared_gate: &shared_gate_gpu,
            shared_up: &shared_up_gpu,
            shared_down: &shared_down_gpu,
        };
        let weights = Qwen4ExpLayerZeroMetalWeights {
            geometry,
            token_embedding: &token_gpu,
            attention_residual: Qwen4ExpResidualMetalWeights {
                read: GatedResidualMetalReadWeights {
                    norm: &attention_norm_gpu,
                    down: &attention_down_gpu,
                    up: &attention_up_gpu,
                },
                inject: &attention_inject_gpu,
            },
            gdn: gdn_weights,
            ffn_residual: Qwen4ExpResidualMetalWeights {
                read: GatedResidualMetalReadWeights {
                    norm: &ffn_norm_gpu,
                    down: &ffn_down_gpu,
                    up: &ffn_up_gpu,
                },
                inject: &ffn_inject_gpu,
            },
            moe: moe_weights,
        };

        let mut control_gdn = GatedDeltaNetMetalWorkspace::new(&ctx, geometry.gdn()).unwrap();
        let mut control_moe = Qwen4ExpMoeMetalWorkspace::new(&ctx, geometry.moe()).unwrap();
        let mut integrated = Qwen4ExpLayerZeroMetalWorkspace::new(&ctx, geometry).unwrap();
        let exported = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();

        let mismatched_gdn = GatedDeltaNetMetalWeights {
            geometry: GatedDeltaNetMetalGeometry::new(HIDDEN, 1, 2, 128, 4, 1e-6).unwrap(),
            ..gdn_weights
        };
        let mismatched_weights = Qwen4ExpLayerZeroMetalWeights {
            gdn: mismatched_gdn,
            ..weights
        };
        let mismatch_command = ctx.queue.commandBuffer().unwrap();
        let mismatch_encoder = KernelEncoder::begin(&mismatch_command);
        let error = encode_qwen4exp_layer_zero(
            &ctx,
            &mismatch_encoder,
            3,
            mismatched_weights,
            &mut integrated,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("GDN weight and workspace geometry differ"));
        mismatch_encoder.end();
        assert!(!integrated.is_poisoned());

        let mismatched_moe = Qwen4ExpMoeMetalWeights {
            geometry: Qwen4ExpMoeMetalGeometry::new(HIDDEN, 15, TOP_K, FFN, FFN).unwrap(),
            ..moe_weights
        };
        let mismatched_weights = Qwen4ExpLayerZeroMetalWeights {
            moe: mismatched_moe,
            ..weights
        };
        let mismatch_command = ctx.queue.commandBuffer().unwrap();
        let mismatch_encoder = KernelEncoder::begin(&mismatch_command);
        let error = encode_qwen4exp_layer_zero(
            &ctx,
            &mismatch_encoder,
            3,
            mismatched_weights,
            &mut integrated,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("MoE weight and workspace geometry differ"));
        mismatch_encoder.end();
        assert!(!integrated.is_poisoned());

        let invalid_command = ctx.queue.commandBuffer().unwrap();
        let invalid_encoder = KernelEncoder::begin(&invalid_command);
        let error = encode_qwen4exp_layer_zero(
            &ctx,
            &invalid_encoder,
            VOCAB as u32,
            weights,
            &mut integrated,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("outside vocabulary"));
        invalid_encoder.end();

        for (step, token) in [3_usize, 7].into_iter().enumerate() {
            let embedding = token_f32[token * HIDDEN..(token + 1) * HIDDEN].to_vec();
            let hyper = (0..BRANCHES)
                .flat_map(|_| embedding.iter().copied())
                .collect::<Vec<_>>();
            let (attention_mixed, attention_state) = gated_residual_mix(
                &hyper,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                attention_cpu.read(),
            )
            .unwrap();
            let mixer_output =
                standalone_gdn(&ctx, &attention_mixed, gdn_weights, &mut control_gdn);
            if step == 1 {
                let mut fresh_gdn = GatedDeltaNetMetalWorkspace::new(&ctx, geometry.gdn()).unwrap();
                let fresh_output =
                    standalone_gdn(&ctx, &attention_mixed, gdn_weights, &mut fresh_gdn);
                let causal_delta = mixer_output
                    .iter()
                    .zip(fresh_output)
                    .map(|(stateful, fresh)| (stateful - fresh).abs())
                    .fold(0.0_f32, f32::max);
                assert!(causal_delta > 1e-5, "GDN causal delta={causal_delta}");
            }
            let after_attention =
                gated_residual_combine(&mixer_output, &attention_state, &attention_cpu.inject)
                    .unwrap();
            let (ffn_mixed, ffn_state) = gated_residual_mix(
                &after_attention,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                ffn_cpu.read(),
            )
            .unwrap();
            let moe_output = standalone_moe(&ctx, &ffn_mixed, moe_weights, &mut control_moe);
            let expected =
                gated_residual_combine(&moe_output, &ffn_state, &ffn_cpu.inject).unwrap();
            assert!(mixer_output.iter().any(|value| value.abs() > 1e-4));
            assert!(moe_output.iter().any(|value| value.abs() > 1e-4));
            for branch in 0..BRANCHES {
                let start = branch * HIDDEN;
                assert!(
                    expected[start..start + HIDDEN]
                        .iter()
                        .zip(&hyper[start..start + HIDDEN])
                        .any(|(output, initial)| (output - initial).abs() > 1e-6),
                    "branch {branch} received no block contribution"
                );
            }

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read =
                encode_qwen4exp_layer_zero(&ctx, &encoder, token as u32, weights, &mut integrated)
                    .unwrap();
            assert_eq!(read.output().branch_count(), BRANCHES);
            read.output()
                .encode_copy_to(&ctx, &encoder, &exported)
                .unwrap();
            drop(read);
            encoder.end();
            if step == 0 {
                assert!(integrated.release_after().is_err());
            }
            command.commit();
            integrated.release_after().unwrap();

            assert_close(
                "embedding",
                &read_f32(&integrated.embedding),
                &embedding,
                1e-7,
                1e-7,
            );
            assert_close(
                "GDN output",
                &read_f32(&integrated.mixer_output),
                &mixer_output,
                5e-4,
                8e-4,
            );
            assert_close(
                "MoE output",
                &read_f32(&integrated.moe_output),
                &moe_output,
                5e-4,
                8e-4,
            );
            assert_close(
                "layer-zero residual",
                &read_f32(&integrated.hyper_residual),
                &expected,
                8e-4,
                1e-3,
            );
            assert_close(
                "copied layer-zero residual",
                &read_f32(&exported),
                &expected,
                8e-4,
                1e-3,
            );
        }

        let abandoned_command = ctx.queue.commandBuffer().unwrap();
        let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
        let read =
            encode_qwen4exp_layer_zero(&ctx, &abandoned_encoder, 9, weights, &mut integrated)
                .unwrap();
        drop(read);
        abandoned_encoder.end();
        unsafe { integrated.abandon_uncommitted() }.unwrap();
        drop(abandoned_command);
        integrated.reset().unwrap();
        assert!(!integrated.is_poisoned());
    }

    fn gguf_dequant(gguf: &GgufFile, name: &str) -> Vec<f32> {
        let desc = gguf.find(name).unwrap();
        crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
    }

    fn gguf_dequant_row(gguf: &GgufFile, name: &str, row: usize) -> Vec<f32> {
        let desc = gguf.find(name).unwrap();
        assert_eq!(desc.shape.len(), 2);
        let row_width = desc.shape[0] as usize;
        let row_count = desc.shape[1] as usize;
        assert!(row < row_count);
        let (block, block_bytes) = desc.dtype.storage_layout().unwrap();
        assert!((row_width as u64).is_multiple_of(block));
        let row_bytes = row_width / block as usize * block_bytes as usize;
        let start = row * row_bytes;
        let bytes = gguf.try_slice(desc).unwrap();
        let row_desc = TensorDesc {
            name: format!("{name}.row.{row}"),
            shape: vec![row_width as u64, 1],
            dtype: desc.dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: row_bytes as u64,
        };
        crate::codec::dequant_to_f32(&row_desc, &bytes[start..start + row_bytes]).unwrap()
    }

    fn real_residual_weights(gguf: &GgufFile, role: &str) -> CpuResidualWeights {
        let prefix = format!("blk.0.hc_{role}");
        CpuResidualWeights {
            norm: gguf_dequant(gguf, &format!("{prefix}_norm.weight")),
            down: gguf_dequant(gguf, &format!("{prefix}_down.weight")),
            up: gguf_dequant(gguf, &format!("{prefix}_up.weight")),
            inject: gguf_dequant(gguf, &format!("{prefix}_inject.weight")),
        }
    }

    fn assert_similarity(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        max_abs: f32,
        minimum_cosine: f64,
    ) {
        assert_eq!(actual.len(), expected.len());
        assert!(actual.iter().all(|value| value.is_finite()));
        let observed_max = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).abs())
            .fold(0.0_f32, f32::max);
        let dot = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| *actual as f64 * *expected as f64)
            .sum::<f64>();
        let actual_norm = actual
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let expected_norm = expected
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>()
            .sqrt();
        let cosine = dot / (actual_norm * expected_norm).max(1e-30);
        eprintln!("[{label}] max_abs={observed_max:.3e} cosine={cosine:.9}");
        assert!(observed_max <= max_abs, "{label} max_abs={observed_max}");
        assert!(cosine >= minimum_cosine, "{label} cosine={cosine}");
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_LAYER_ZERO_GGUF to the pinned first release shard"]
    fn released_layer_zero_matches_independent_composition_control() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_LAYER_ZERO_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_LAYER_ZERO_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let metal_weights = realized.weights();
        let weights = Qwen4ExpLayerZeroMetalWeights::bind(metal_weights).unwrap();
        let geometry = weights.geometry;
        assert_eq!(weights.token_embedding.dtype, GgmlType::Q8_0);
        assert_eq!(weights.attention_residual.read.down.dtype, GgmlType::Q8_0);
        assert_eq!(weights.gdn.qkv.dtype, GgmlType::Q8_0);
        assert_eq!(weights.gdn.output.dtype, GgmlType::Q8_0);
        assert_eq!(weights.ffn_residual.read.up.dtype, GgmlType::Q8_0);
        assert_eq!(weights.moe.routed_gate.dtype, GgmlType::IQ3_XXS);
        assert_eq!(weights.moe.routed_down.dtype, GgmlType::IQ4_NL);

        let attention_cpu = real_residual_weights(&gguf, "attn");
        let ffn_cpu = real_residual_weights(&gguf, "ffn");
        let mut control_gdn = GatedDeltaNetMetalWorkspace::new(&ctx, geometry.gdn()).unwrap();
        let mut control_moe = Qwen4ExpMoeMetalWorkspace::new(&ctx, geometry.moe()).unwrap();
        let mut integrated = Qwen4ExpLayerZeroMetalWorkspace::new(&ctx, geometry).unwrap();
        let exported = MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();

        for (step, token) in [35_usize, 201].into_iter().enumerate() {
            let embedding = gguf_dequant_row(&gguf, "token_embd.weight", token);
            let hyper = (0..geometry.branch_count())
                .flat_map(|_| embedding.iter().copied())
                .collect::<Vec<_>>();
            let (attention_mixed, attention_state) = gated_residual_mix(
                &hyper,
                geometry.branch_count(),
                geometry.hidden_size(),
                geometry.low_rank(),
                geometry.eps(),
                attention_cpu.read(),
            )
            .unwrap();
            let mixer_output =
                standalone_gdn(&ctx, &attention_mixed, weights.gdn, &mut control_gdn);
            if step == 1 {
                let mut fresh_gdn = GatedDeltaNetMetalWorkspace::new(&ctx, geometry.gdn()).unwrap();
                let fresh_output =
                    standalone_gdn(&ctx, &attention_mixed, weights.gdn, &mut fresh_gdn);
                let causal_delta = mixer_output
                    .iter()
                    .zip(fresh_output)
                    .map(|(stateful, fresh)| (stateful - fresh).abs())
                    .fold(0.0_f32, f32::max);
                assert!(causal_delta > 1e-5, "real GDN causal delta={causal_delta}");
            }
            let after_attention =
                gated_residual_combine(&mixer_output, &attention_state, &attention_cpu.inject)
                    .unwrap();
            let (ffn_mixed, ffn_state) = gated_residual_mix(
                &after_attention,
                geometry.branch_count(),
                geometry.hidden_size(),
                geometry.low_rank(),
                geometry.eps(),
                ffn_cpu.read(),
            )
            .unwrap();
            let moe_output = standalone_moe(&ctx, &ffn_mixed, weights.moe, &mut control_moe);
            let expected =
                gated_residual_combine(&moe_output, &ffn_state, &ffn_cpu.inject).unwrap();

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read =
                encode_qwen4exp_layer_zero(&ctx, &encoder, token as u32, weights, &mut integrated)
                    .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &exported)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            integrated.release_after().unwrap();

            assert_close(
                "real token embedding",
                &read_f32(&integrated.embedding),
                &embedding,
                1e-7,
                1e-7,
            );
            assert_similarity(
                "real GDN output",
                &read_f32(&integrated.mixer_output),
                &mixer_output,
                2e-2,
                0.999,
            );
            assert_similarity(
                "real FFN mixed input",
                &read_f32(integrated.residual.mixed_tensor()),
                &ffn_mixed,
                2e-2,
                0.999,
            );
            assert_similarity(
                "real MoE output",
                &read_f32(&integrated.moe_output),
                &moe_output,
                5e-2,
                0.998,
            );
            assert_similarity(
                "real layer-zero residual",
                &read_f32(&exported),
                &expected,
                8e-2,
                0.998,
            );
        }
    }
}
