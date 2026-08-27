//! Packed one-command execution through the first two Flash-Next blocks.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_copy_offset_f32, encode_get_rows_f32,
};
use crate::qwen4exp::{PleConfig, PleHistory, Qwen4ExpError};
use crate::qwen4exp_gdn::{
    GatedDeltaNetPackedScratch, Qwen4ExpGdnError, encode_gated_delta_net_packed,
    preflight_packed as preflight_gdn_packed, validate_packed_contract as validate_gdn_packed,
};
use crate::qwen4exp_layers_zero_one::{
    Qwen4ExpLayersZeroOneMetalGeometry, Qwen4ExpLayersZeroOneMetalWeights,
};
use crate::qwen4exp_metal::{
    GatedResidualPackedScratch, Qwen4ExpMetalError, encode_gated_residual_packed_mix,
    encode_hc_repeat_packed, validate_and_preflight_gated_residual_packed_mix,
    validate_and_preflight_hc_repeat_packed,
};
use crate::qwen4exp_moe::{
    Qwen4ExpMoeError, Qwen4ExpMoePackedMotorScratch, encode_qwen4exp_moe_packed_motor,
    preflight as preflight_moe_singleton, preflight_packed as preflight_moe_packed,
    validate_packed_contract as validate_moe_packed,
};
use crate::qwen4exp_ple::{PleGatherError, PleIq4NlTable};
use crate::qwen4exp_ple_metal::{
    Qwen4ExpPleMetalError, Qwen4ExpPlePackedMotorScratch, encode_qwen4exp_ple_packed_motor,
    preflight_packed_motor as preflight_ple_packed,
    validate_packed_motor_contract as validate_ple_packed,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource,
};

const MAX_PACKED_TOKENS: usize = 2_048;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpPackedPrefillError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Gather(#[from] PleGatherError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Residual(#[from] Qwen4ExpMetalError),
    #[error(transparent)]
    GatedDeltaNet(#[from] Qwen4ExpGdnError),
    #[error(transparent)]
    Moe(#[from] Qwen4ExpMoeError),
    #[error(transparent)]
    Ple(#[from] Qwen4ExpPleMetalError),
    #[error("invalid Qwen3.8-Flash-Next packed prefill contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next packed prefill command buffer failed: {0}")]
    CommandBuffer(String),
}

pub struct Qwen4ExpPackedPrefillWorkspace {
    geometry: Qwen4ExpLayersZeroOneMetalGeometry,
    capacity: usize,
    token_ids: MetalTensor,
    embedding: MetalTensor,
    ple_embedding: MetalTensor,
    hyper_residual: MetalTensor,
    bridge: MetalTensor,
    residual: GatedResidualPackedScratch,
    gdn: GatedDeltaNetPackedScratch,
    layer_zero_conv_state: MetalTensor,
    layer_zero_delta_state: MetalTensor,
    layer_one_conv_state: MetalTensor,
    layer_one_delta_state: MetalTensor,
    ple_state: MetalTensor,
    ple: Qwen4ExpPlePackedMotorScratch,
    moe: Qwen4ExpMoePackedMotorScratch,
    history: PleHistory,
    pending_history: Option<PleHistory>,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
    encode_failed: bool,
}

impl Qwen4ExpPackedPrefillWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpLayersZeroOneMetalGeometry,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpPackedPrefillError> {
        if capacity == 0 || capacity > MAX_PACKED_TOKENS {
            return invalid(format!(
                "packed prefill capacity must be in 1..={MAX_PACKED_TOKENS}, got {capacity}"
            ));
        }
        if capacity > geometry.context_length() {
            return invalid(format!(
                "packed prefill capacity {capacity} exceeds context length {}",
                geometry.context_length()
            ));
        }
        let hidden = geometry.hidden_size();
        let hyper = geometry.hyper_width();
        let gdn_geometry = geometry.layer_zero().gdn();
        let ple_geometry = geometry.ple();
        let shape = |width: usize| vec![width as u64, capacity as u64];

        Ok(Self {
            geometry,
            capacity,
            token_ids: MetalTensor::zeros_i32(ctx, vec![capacity as u64])?,
            embedding: MetalTensor::zeros_f32(ctx, shape(hidden))?,
            ple_embedding: MetalTensor::zeros_f32(ctx, shape(hidden))?,
            hyper_residual: MetalTensor::zeros_f32(ctx, shape(hyper))?,
            bridge: MetalTensor::zeros_f32(ctx, shape(hidden))?,
            residual: GatedResidualPackedScratch::new(
                ctx,
                geometry.branch_count(),
                hidden,
                geometry.low_rank(),
                capacity,
            )?,
            gdn: GatedDeltaNetPackedScratch::new(ctx, gdn_geometry, capacity)?,
            layer_zero_conv_state: zero_f32(ctx, vec![gdn_geometry.conv_state_elements() as u64])?,
            layer_zero_delta_state: zero_f32(
                ctx,
                vec![gdn_geometry.delta_state_elements() as u64],
            )?,
            layer_one_conv_state: zero_f32(ctx, vec![gdn_geometry.conv_state_elements() as u64])?,
            layer_one_delta_state: zero_f32(ctx, vec![gdn_geometry.delta_state_elements() as u64])?,
            ple_state: zero_f32(ctx, vec![ple_geometry.history_len() as u64, hyper as u64])?,
            ple: Qwen4ExpPlePackedMotorScratch::new(ctx, ple_geometry, capacity)?,
            moe: Qwen4ExpMoePackedMotorScratch::new(ctx, geometry.layer_zero().moe(), capacity)?,
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

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn next_position(&self) -> Option<u64> {
        self.history.next_position()
    }

    pub fn prior_tokens(&self) -> &[u32] {
        self.history.prior_tokens()
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpPackedPrefillError> {
        self.require_idle()?;
        if self.pending_history.is_some() {
            return invalid("cannot reset while a packed PLE history update is pending");
        }
        for state in [
            &self.layer_zero_conv_state,
            &self.layer_zero_delta_state,
            &self.layer_one_conv_state,
            &self.layer_one_delta_state,
            &self.ple_state,
        ] {
            zero_writable_f32(state)?;
        }
        self.history.reset();
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpPackedPrefillError> {
        let Some(command) = self.active_command.clone() else {
            if self.pending_history.is_some() {
                return invalid("packed PLE history is pending without an owning command");
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
        self.active_command = None;
        let pending = self.pending_history.take();
        let pending_present = pending.is_some();
        if status == MTLCommandBufferStatus::Completed
            && command_error.is_none()
            && !self.encode_failed
            && let Some(pending) = pending
        {
            self.history = pending;
            self.encode_failed = false;
            return Ok(());
        }
        self.state_poisoned = true;
        Err(Qwen4ExpPackedPrefillError::CommandBuffer(format!(
            "status={status:?}, error={command_error:?}, encode_failed={}, pending_history={pending_present}",
            self.encode_failed
        )))
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may mutate causal state after a
    /// subsequent chunk has acquired this workspace.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpPackedPrefillError> {
        let Some(command) = self.active_command.as_ref() else {
            return Ok(());
        };
        let status = command.status();
        if status != MTLCommandBufferStatus::NotEnqueued {
            return invalid(format!(
                "only a NotEnqueued workspace owner can be abandoned, got {status:?}"
            ));
        }
        self.active_command = None;
        self.pending_history = None;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpPackedPrefillError> {
        if self.active_command.is_some() {
            invalid("packed prefill workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "copy the packed residual in its owning command, then release the workspace"]
pub struct Qwen4ExpPackedPrefillRead<'a> {
    workspace: &'a mut Qwen4ExpPackedPrefillWorkspace,
    tokens: usize,
}

pub struct Qwen4ExpPackedPrefillOutput<'a> {
    workspace: &'a Qwen4ExpPackedPrefillWorkspace,
    tokens: usize,
}

impl Qwen4ExpPackedPrefillRead<'_> {
    pub fn output(&self) -> Qwen4ExpPackedPrefillOutput<'_> {
        Qwen4ExpPackedPrefillOutput {
            workspace: self.workspace,
            tokens: self.tokens,
        }
    }
}

impl Qwen4ExpPackedPrefillOutput<'_> {
    pub fn token_count(&self) -> usize {
        self.tokens
    }

    pub fn n_elements(&self) -> u64 {
        (self.workspace.geometry.hyper_width() * self.tokens) as u64
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
    ) -> Result<(), Qwen4ExpPackedPrefillError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("packed prefill output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("packed prefill output must be copied by its owning command buffer");
        }
        let hyper = self.workspace.geometry.hyper_width();
        require_tensor(
            "packed prefill copied output destination",
            destination,
            GgmlType::F32,
            &[hyper as u64, self.tokens as u64],
            true,
        )?;
        let output = prefix_view(
            "packed prefill output",
            &self.workspace.hyper_residual,
            hyper,
            self.tokens,
            self.workspace.capacity,
        )?;
        let tensors = [
            ("packed prefill output", &output),
            ("packed prefill copied output destination", destination),
        ];
        require_same_device(ctx, &tensors)?;
        require_disjoint(&tensors)?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(ctx, enc, &output, 0, destination, hyper * self.tokens)?;
        Ok(())
    }
}

#[allow(clippy::too_many_arguments)]
pub fn encode_qwen4exp_packed_prefill<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: u64,
    table: PleIq4NlTable<'_>,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpPackedPrefillWorkspace,
) -> Result<Qwen4ExpPackedPrefillRead<'a>, Qwen4ExpPackedPrefillError> {
    validate_and_preflight(
        ctx,
        enc,
        token_ids,
        start_position,
        table,
        weights,
        workspace,
    )?;
    let (next_history, ple_embedding) = prepare_ple_embedding(
        token_ids,
        start_position,
        table,
        weights.ple_config,
        &workspace.history,
        workspace.geometry,
    )?;
    write_i32_prefix(&workspace.token_ids, token_ids)?;
    write_f32_prefix(&workspace.ple_embedding, &ple_embedding)?;

    reserve_command(workspace, enc)?;
    workspace.pending_history = Some(next_history);
    if let Err(error) = encode_step(ctx, enc, weights, workspace, token_ids.len()) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpPackedPrefillRead {
        workspace,
        tokens: token_ids.len(),
    })
}

#[allow(clippy::too_many_arguments)]
fn validate_and_preflight(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_ids: &[u32],
    start_position: u64,
    table: PleIq4NlTable<'_>,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpPackedPrefillWorkspace,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.pending_history.is_some() {
        return invalid("workspace has a packed PLE history update without a command owner");
    }
    let tokens = token_ids.len();
    if tokens == 0 || tokens > workspace.capacity {
        return invalid(format!(
            "packed token count {tokens} is outside capacity {}",
            workspace.capacity
        ));
    }
    if weights.geometry != workspace.geometry {
        return invalid("packed prefill weight and workspace geometry differ");
    }
    validate_nested_geometry(weights, workspace)?;
    validate_positions(token_ids, start_position, workspace)?;
    validate_ple_config(weights.ple_config, workspace.geometry)?;
    if table.row_width() != workspace.geometry.ple().head_dim() {
        return invalid(format!(
            "PLE table row width {} differs from packed head width {}",
            table.row_width(),
            workspace.geometry.ple().head_dim()
        ));
    }
    validate_contract(ctx, weights, workspace, tokens)
}

fn validate_nested_geometry(
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpPackedPrefillWorkspace,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let geometry = workspace.geometry;
    if weights.layer_zero.geometry != geometry.layer_zero()
        || weights.ple.geometry != geometry.ple()
        || weights.layer_zero.gdn.geometry != geometry.layer_zero().gdn()
        || weights.layer_one_gdn.geometry != geometry.layer_zero().gdn()
        || weights.layer_zero.moe.geometry != geometry.layer_zero().moe()
        || weights.layer_one_moe.geometry != geometry.layer_zero().moe()
    {
        return invalid("nested packed weight geometry differs from the composition");
    }
    Ok(())
}

fn validate_positions(
    token_ids: &[u32],
    start_position: u64,
    workspace: &Qwen4ExpPackedPrefillWorkspace,
) -> Result<(), Qwen4ExpPackedPrefillError> {
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
    let end = start_position
        .checked_add(token_ids.len() as u64)
        .ok_or_else(|| {
            Qwen4ExpPackedPrefillError::Invalid("packed position range overflow".into())
        })?;
    if end > workspace.geometry.context_length() as u64 {
        return invalid(format!(
            "packed position range {start_position}..{end} exceeds context length {}",
            workspace.geometry.context_length()
        ));
    }
    Ok(())
}

fn validate_ple_config(
    config: &PleConfig,
    geometry: Qwen4ExpLayersZeroOneMetalGeometry,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    config.validate(geometry.hidden_size() as u32, geometry.layer_count() as u32)?;
    if config.token_vocab_size as usize != geometry.vocab_size()
        || !config.layers.contains(&1)
        || config.layers.contains(&0)
        || config.head_count()? as usize != geometry.ple().head_count()
        || config.embedding_head_dim as usize != geometry.ple().head_dim()
        || config.conv_kernel as usize != geometry.ple().kernel_size()
        || config.ngram_size as usize != geometry.ple().dilation()
    {
        return invalid("PLE hash and packed Metal geometries differ");
    }
    Ok(())
}

fn validate_contract(
    ctx: &MetalContext,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &Qwen4ExpPackedPrefillWorkspace,
    tokens: usize,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let geometry = workspace.geometry;
    let hidden = geometry.hidden_size();
    let hyper = geometry.hyper_width();
    let token_ids = prefix_view_i32(&workspace.token_ids, tokens, workspace.capacity)?;
    let embedding = prefix_view(
        "packed token embedding output",
        &workspace.embedding,
        hidden,
        tokens,
        workspace.capacity,
    )?;
    let ple_embedding = prefix_view(
        "packed PLE embedding",
        &workspace.ple_embedding,
        hidden,
        tokens,
        workspace.capacity,
    )?;
    let hyper_residual = prefix_view(
        "packed hyper residual",
        &workspace.hyper_residual,
        hyper,
        tokens,
        workspace.capacity,
    )?;
    let bridge = prefix_view(
        "packed block-output bridge",
        &workspace.bridge,
        hidden,
        tokens,
        workspace.capacity,
    )?;
    let residual_mixed = workspace.residual.mixed_view(tokens)?;

    require_projection(
        "packed token embedding",
        weights.layer_zero.token_embedding,
        hidden,
        geometry.vocab_size(),
        GgmlType::Q8_0,
    )?;
    require_read_only("packed token embedding", weights.layer_zero.token_embedding)?;
    require_tensor(
        "packed token IDs",
        &token_ids,
        GgmlType::I32,
        &[tokens as u64],
        true,
    )?;
    let get_rows = ctx.pipeline("kernel_get_rows_q8_0_f32")?;
    if get_rows.threadExecutionWidth() != 32 || get_rows.maxTotalThreadsPerThreadgroup() < 32 {
        return invalid(format!(
            "packed Q8_0 row pipeline requires SIMD width and capacity 32, got width={} capacity={}",
            get_rows.threadExecutionWidth(),
            get_rows.maxTotalThreadsPerThreadgroup()
        ));
    }

    if tokens > 1 {
        validate_and_preflight_hc_repeat_packed(
            ctx,
            &embedding,
            &hyper_residual,
            geometry.branch_count(),
            hidden,
            tokens,
        )?;
    } else {
        ctx.pipeline("kernel_copy_offset_f32")?;
    }

    for (role, residual) in [
        (
            "layer-zero attention",
            weights.layer_zero.attention_residual,
        ),
        ("layer-zero FFN", weights.layer_zero.ffn_residual),
        ("layer-one attention", weights.layer_one_attention_residual),
        ("layer-one FFN", weights.layer_one_ffn_residual),
    ] {
        for (name, tensor) in [
            ("norm", residual.read.norm),
            ("down", residual.read.down),
            ("up", residual.read.up),
            ("injection", residual.inject),
        ] {
            require_read_only(&format!("packed {role} HC {name}"), tensor)?;
        }
        validate_and_preflight_gated_residual_packed_mix(
            ctx,
            &hyper_residual,
            &bridge,
            geometry.eps(),
            residual.read,
            residual.inject,
            &workspace.residual,
            tokens,
        )?;
    }

    for (weights, conv_state, delta_state) in [
        (
            weights.layer_zero.gdn,
            &workspace.layer_zero_conv_state,
            &workspace.layer_zero_delta_state,
        ),
        (
            weights.layer_one_gdn,
            &workspace.layer_one_conv_state,
            &workspace.layer_one_delta_state,
        ),
    ] {
        validate_gdn_packed(
            ctx,
            &residual_mixed,
            weights,
            conv_state,
            delta_state,
            &workspace.gdn,
            tokens,
        )?;
        preflight_gdn_packed(ctx, weights)?;
    }

    for weights in [weights.layer_zero.moe, weights.layer_one_moe] {
        validate_moe_packed(ctx, &residual_mixed, weights, &workspace.moe, tokens)?;
        if tokens == 1 {
            preflight_moe_singleton(ctx, weights)?;
        } else {
            preflight_moe_packed(ctx, weights)?;
        }
    }

    validate_ple_packed(
        ctx,
        &ple_embedding,
        &hyper_residual,
        weights.ple,
        &workspace.ple_state,
        &workspace.ple,
        tokens,
    )?;
    preflight_ple_packed(ctx, weights.ple)?;

    let top_level = [
        ("packed token IDs", &token_ids),
        ("packed token embedding output", &embedding),
        ("packed PLE embedding", &ple_embedding),
        ("packed hyper residual", &hyper_residual),
        ("packed block-output bridge", &bridge),
        (
            "packed layer-zero GDN convolution state",
            &workspace.layer_zero_conv_state,
        ),
        (
            "packed layer-zero GDN delta state",
            &workspace.layer_zero_delta_state,
        ),
        (
            "packed layer-one GDN convolution state",
            &workspace.layer_one_conv_state,
        ),
        (
            "packed layer-one GDN delta state",
            &workspace.layer_one_delta_state,
        ),
        ("packed PLE convolution state", &workspace.ple_state),
        ("packed token embedding", weights.layer_zero.token_embedding),
    ];
    require_same_device(ctx, &top_level)?;
    require_disjoint(&top_level)?;
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

fn prepare_ple_embedding(
    token_ids: &[u32],
    start_position: u64,
    table: PleIq4NlTable<'_>,
    config: &PleConfig,
    history: &PleHistory,
    geometry: Qwen4ExpLayersZeroOneMetalGeometry,
) -> Result<(PleHistory, Vec<f32>), Qwen4ExpPackedPrefillError> {
    let mut next_history = history.clone();
    let head_count = geometry.ple().head_count();
    let mut row_ids = Vec::with_capacity(token_ids.len() * head_count);
    for (index, &token_id) in token_ids.iter().enumerate() {
        let position = start_position + index as u64;
        let (advanced, rows) = next_history.advanced(config, token_id, position)?;
        if rows.len() != head_count {
            return invalid(format!(
                "PLE produced {} rows at packed token {index}, expected {head_count}",
                rows.len()
            ));
        }
        row_ids.extend_from_slice(&rows);
        next_history = advanced;
    }
    let elements = geometry
        .hidden_size()
        .checked_mul(token_ids.len())
        .ok_or_else(|| {
            Qwen4ExpPackedPrefillError::Invalid("packed PLE staging size overflow".into())
        })?;
    let mut embedding = vec![0.0_f32; elements];
    table.gather_f32_into(&row_ids, &mut embedding)?;
    Ok((next_history, embedding))
}

fn reserve_command(
    workspace: &mut Qwen4ExpPackedPrefillWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.encode_failed = false;
    Ok(())
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: Qwen4ExpLayersZeroOneMetalWeights<'_>,
    workspace: &mut Qwen4ExpPackedPrefillWorkspace,
    tokens: usize,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let geometry = workspace.geometry;
    let hidden = geometry.hidden_size();
    let hyper = geometry.hyper_width();
    let token_ids = prefix_view_i32(&workspace.token_ids, tokens, workspace.capacity)?;
    let embedding = prefix_view(
        "packed token embedding output",
        &workspace.embedding,
        hidden,
        tokens,
        workspace.capacity,
    )?;
    let ple_embedding = prefix_view(
        "packed PLE embedding",
        &workspace.ple_embedding,
        hidden,
        tokens,
        workspace.capacity,
    )?;
    let hyper_residual = prefix_view(
        "packed hyper residual",
        &workspace.hyper_residual,
        hyper,
        tokens,
        workspace.capacity,
    )?;
    let bridge = prefix_view(
        "packed block-output bridge",
        &workspace.bridge,
        hidden,
        tokens,
        workspace.capacity,
    )?;

    encode_get_rows_f32(
        ctx,
        enc,
        weights.layer_zero.token_embedding,
        &token_ids,
        &embedding,
        tokens,
        hidden,
    )?;
    if tokens == 1 {
        for branch in 0..geometry.branch_count() {
            let destination =
                hyper_residual.view_subrange((branch * hidden) as u64, vec![hidden as u64]);
            encode_copy_offset_f32(ctx, enc, &embedding, 0, &destination, hidden)?;
        }
    } else {
        encode_hc_repeat_packed(
            ctx,
            enc,
            &embedding,
            &hyper_residual,
            geometry.branch_count(),
            hidden,
            tokens,
        )?;
    }

    encode_layer(
        ctx,
        enc,
        &hyper_residual,
        &bridge,
        weights.layer_zero.attention_residual,
        weights.layer_zero.gdn,
        &workspace.layer_zero_conv_state,
        &workspace.layer_zero_delta_state,
        weights.layer_zero.ffn_residual,
        weights.layer_zero.moe,
        &mut workspace.residual,
        &workspace.gdn,
        &workspace.moe,
        geometry.eps(),
        tokens,
    )?;

    let ple_output = unsafe {
        encode_qwen4exp_ple_packed_motor(
            ctx,
            enc,
            &ple_embedding,
            &hyper_residual,
            weights.ple,
            &workspace.ple_state,
            &workspace.ple,
            tokens,
        )
    }?;
    encode_copy_offset_f32(ctx, enc, &ple_output, 0, &hyper_residual, hyper * tokens)?;

    encode_layer(
        ctx,
        enc,
        &hyper_residual,
        &bridge,
        weights.layer_one_attention_residual,
        weights.layer_one_gdn,
        &workspace.layer_one_conv_state,
        &workspace.layer_one_delta_state,
        weights.layer_one_ffn_residual,
        weights.layer_one_moe,
        &mut workspace.residual,
        &workspace.gdn,
        &workspace.moe,
        geometry.eps(),
        tokens,
    )
}

#[allow(clippy::too_many_arguments)]
fn encode_layer(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_residual: &MetalTensor,
    bridge: &MetalTensor,
    attention_residual: crate::qwen4exp_layer_zero::Qwen4ExpResidualMetalWeights<'_>,
    gdn_weights: crate::qwen4exp_gdn::GatedDeltaNetMetalWeights<'_>,
    conv_state: &MetalTensor,
    delta_state: &MetalTensor,
    ffn_residual: crate::qwen4exp_layer_zero::Qwen4ExpResidualMetalWeights<'_>,
    moe_weights: crate::qwen4exp_moe::Qwen4ExpMoeMetalWeights<'_>,
    residual: &mut GatedResidualPackedScratch,
    gdn: &GatedDeltaNetPackedScratch,
    moe: &Qwen4ExpMoePackedMotorScratch,
    eps: f32,
    tokens: usize,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let hidden = gdn_weights.geometry.hidden_size();
    let attention = unsafe {
        encode_gated_residual_packed_mix(
            ctx,
            enc,
            hyper_residual,
            bridge,
            eps,
            attention_residual.read,
            attention_residual.inject,
            residual,
            tokens,
        )
    }?;
    let gdn_output = unsafe {
        encode_gated_delta_net_packed(
            ctx,
            enc,
            attention.mixed(),
            gdn_weights,
            conv_state,
            delta_state,
            gdn,
            tokens,
        )
    }?;
    encode_copy_offset_f32(ctx, enc, &gdn_output, 0, bridge, hidden * tokens)?;
    attention.encode_combine(enc)?;

    let ffn = unsafe {
        encode_gated_residual_packed_mix(
            ctx,
            enc,
            hyper_residual,
            bridge,
            eps,
            ffn_residual.read,
            ffn_residual.inject,
            residual,
            tokens,
        )
    }?;
    let moe_output = unsafe {
        encode_qwen4exp_moe_packed_motor(ctx, enc, ffn.mixed(), moe_weights, moe, tokens)
    }?;
    encode_copy_offset_f32(ctx, enc, &moe_output, 0, bridge, hidden * tokens)?;
    ffn.encode_combine(enc)?;
    Ok(())
}

fn validate_encoder(
    ctx: &MetalContext,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("packed prefill dependent dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "packed prefill encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn prefix_view(
    name: &str,
    tensor: &MetalTensor,
    width: usize,
    tokens: usize,
    capacity: usize,
) -> Result<MetalTensor, Qwen4ExpPackedPrefillError> {
    if tokens == 0 || tokens > capacity {
        return invalid(format!(
            "{name} token count {tokens} is outside capacity {capacity}"
        ));
    }
    let elements = width.checked_mul(tokens).ok_or_else(|| {
        Qwen4ExpPackedPrefillError::Invalid(format!("{name} element count overflow"))
    })?;
    let view = tensor.view_subrange(0, vec![width as u64, tokens as u64]);
    if view.n_elements() as usize != elements {
        return invalid(format!("{name} prefix view has the wrong element count"));
    }
    Ok(view)
}

fn prefix_view_i32(
    tensor: &MetalTensor,
    tokens: usize,
    capacity: usize,
) -> Result<MetalTensor, Qwen4ExpPackedPrefillError> {
    if tokens == 0 || tokens > capacity {
        return invalid(format!(
            "packed token count {tokens} is outside capacity {capacity}"
        ));
    }
    Ok(tensor.view_subrange(0, vec![tokens as u64]))
}

fn write_i32_prefix(
    tensor: &MetalTensor,
    values: &[u32],
) -> Result<(), Qwen4ExpPackedPrefillError> {
    if tensor.dtype != GgmlType::I32 || !tensor.is_writable() {
        return invalid("packed token staging requires writable I32 storage");
    }
    let signed = values
        .iter()
        .copied()
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                Qwen4ExpPackedPrefillError::Invalid(format!(
                    "packed token ID {value} exceeds signed i32"
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    write_prefix_bytes(tensor, bytemuck::cast_slice(&signed))
}

fn write_f32_prefix(
    tensor: &MetalTensor,
    values: &[f32],
) -> Result<(), Qwen4ExpPackedPrefillError> {
    if tensor.dtype != GgmlType::F32 || !tensor.is_writable() {
        return invalid("packed PLE staging requires writable F32 storage");
    }
    write_prefix_bytes(tensor, bytemuck::cast_slice(values))
}

fn write_prefix_bytes(
    tensor: &MetalTensor,
    bytes: &[u8],
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let offset = usize::try_from(tensor.offset).map_err(|_| {
        Qwen4ExpPackedPrefillError::Invalid("packed staging offset exceeds usize".into())
    })?;
    let end = offset.checked_add(bytes.len()).ok_or_else(|| {
        Qwen4ExpPackedPrefillError::Invalid("packed staging range overflow".into())
    })?;
    if end > tensor.buffer.length() {
        return invalid("packed staging write exceeds its Metal buffer");
    }
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            tensor.buffer.contents().as_ptr().cast::<u8>().add(offset),
            bytes.len(),
        );
    }
    Ok(())
}

fn zero_f32(
    ctx: &MetalContext,
    shape: Vec<u64>,
) -> Result<MetalTensor, Qwen4ExpPackedPrefillError> {
    let elements = shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Qwen4ExpPackedPrefillError::Invalid("zero tensor shape overflow".into()))?;
    Ok(MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&vec![0.0_f32; elements]),
        shape,
        GgmlType::F32,
    )?)
}

fn zero_writable_f32(tensor: &MetalTensor) -> Result<(), Qwen4ExpPackedPrefillError> {
    if tensor.dtype != GgmlType::F32 || !tensor.is_writable() {
        return invalid("packed state reset requires writable F32 storage");
    }
    let offset = usize::try_from(tensor.offset).map_err(|_| {
        Qwen4ExpPackedPrefillError::Invalid("packed state offset exceeds usize".into())
    })?;
    let bytes = usize::try_from(storage_bytes(tensor)?).map_err(|_| {
        Qwen4ExpPackedPrefillError::Invalid("packed state byte length exceeds usize".into())
    })?;
    let end = offset.checked_add(bytes).ok_or_else(|| {
        Qwen4ExpPackedPrefillError::Invalid("packed state reset range overflow".into())
    })?;
    if end > tensor.buffer.length() {
        return invalid("packed state reset range exceeds its Metal buffer");
    }
    unsafe {
        std::ptr::write_bytes(
            tensor.buffer.contents().as_ptr().cast::<u8>().add(offset),
            0,
            bytes,
        );
    }
    Ok(())
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    dtype: GgmlType,
) -> Result<(), Qwen4ExpPackedPrefillError> {
    let shape = [n_in as u64, n_out as u64];
    if tensor.dtype != dtype || tensor.shape != shape {
        return invalid(format!(
            "{name} must be {dtype:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    let (block, _) = dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpPackedPrefillError::Invalid(format!("{name} has unsupported dtype {dtype:?}"))
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
) -> Result<(), Qwen4ExpPackedPrefillError> {
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

fn require_read_only(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpPackedPrefillError> {
    if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
        invalid(format!("{name} must have read-only weight provenance"))
    } else {
        Ok(())
    }
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpPackedPrefillError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| {
            Qwen4ExpPackedPrefillError::Invalid("tensor element count overflow".into())
        })?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpPackedPrefillError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
    })?;
    if block == 0 || !elements.is_multiple_of(block) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    (elements / block)
        .checked_mul(bytes)
        .ok_or_else(|| Qwen4ExpPackedPrefillError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpPackedPrefillError> {
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
        .ok_or_else(|| Qwen4ExpPackedPrefillError::Invalid(format!("{name} range overflow")))?;
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
) -> Result<(), Qwen4ExpPackedPrefillError> {
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

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpPackedPrefillError> {
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

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpPackedPrefillError> {
    Err(Qwen4ExpPackedPrefillError::Invalid(detail.into()))
}
