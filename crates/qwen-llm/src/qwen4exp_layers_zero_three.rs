//! One-command composition through the first Qwen3.8-Flash-Next QSA cycle.

use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor, encode_copy_offset_f32};
use crate::qwen4exp::MixerKind;
use crate::qwen4exp_layers_zero_one::{
    Qwen4ExpLayersZeroOneError, Qwen4ExpLayersZeroOneMetalGeometry,
    Qwen4ExpLayersZeroOneMetalWeights, Qwen4ExpLayersZeroOneMetalWorkspace,
    encode_qwen4exp_layers_zero_one,
};
use crate::qwen4exp_ple::PleIq4NlTable;
use crate::qwen4exp_post_ple_block::{
    Qwen4ExpPostPleBlockError, Qwen4ExpPostPleBlockMetalGeometry, Qwen4ExpPostPleBlockMetalWeights,
    Qwen4ExpPostPleBlockMetalWorkspace, encode_qwen4exp_post_ple_block,
};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice, MTLResource};

const LAYER_TWO: u32 = 2;
const LAYER_THREE: u32 = 3;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpLayersZeroThreeError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    LayersZeroOne(#[from] Qwen4ExpLayersZeroOneError),
    #[error(transparent)]
    PostPleBlock(#[from] Qwen4ExpPostPleBlockError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next layers-zero-three contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next layers-zero-three command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpLayersZeroThreeMetalGeometry {
    zero_one: Qwen4ExpLayersZeroOneMetalGeometry,
    layer_two: Qwen4ExpPostPleBlockMetalGeometry,
    layer_three: Qwen4ExpPostPleBlockMetalGeometry,
}

impl Qwen4ExpLayersZeroThreeMetalGeometry {
    pub fn new(
        zero_one: Qwen4ExpLayersZeroOneMetalGeometry,
        layer_two: Qwen4ExpPostPleBlockMetalGeometry,
        layer_three: Qwen4ExpPostPleBlockMetalGeometry,
    ) -> Result<Self, Qwen4ExpLayersZeroThreeError> {
        let geometry = Self {
            zero_one,
            layer_two,
            layer_three,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(
        config: &crate::qwen4exp::Qwen4ExpConfig,
        qsa_capacity: usize,
    ) -> Result<Self, Qwen4ExpLayersZeroThreeError> {
        if config.layer_count < 4 {
            return invalid("first-cycle composition requires at least four layers");
        }
        Self::new(
            Qwen4ExpLayersZeroOneMetalGeometry::from_config(config)?,
            Qwen4ExpPostPleBlockMetalGeometry::from_config(config, LAYER_TWO, None)?,
            Qwen4ExpPostPleBlockMetalGeometry::from_config(
                config,
                LAYER_THREE,
                Some(qsa_capacity),
            )?,
        )
    }

    pub fn zero_one(self) -> Qwen4ExpLayersZeroOneMetalGeometry {
        self.zero_one
    }

    pub fn layer_two(self) -> Qwen4ExpPostPleBlockMetalGeometry {
        self.layer_two
    }

    pub fn layer_three(self) -> Qwen4ExpPostPleBlockMetalGeometry {
        self.layer_three
    }

    pub fn context_length(self) -> usize {
        self.zero_one.context_length()
    }

    pub fn capacity(self) -> usize {
        self.layer_three
            .mixer()
            .qsa()
            .expect("validated layer-three QSA geometry")
            .capacity()
    }

    pub fn branch_count(self) -> usize {
        self.zero_one.branch_count()
    }

    pub fn hidden_size(self) -> usize {
        self.zero_one.hidden_size()
    }

    pub fn hyper_width(self) -> usize {
        self.zero_one.hyper_width()
    }

    fn validate(self) -> Result<(), Qwen4ExpLayersZeroThreeError> {
        if self.layer_two.layer() != LAYER_TWO
            || self.layer_two.mixer().kind() != MixerKind::GatedDeltaNet
            || self.layer_three.layer() != LAYER_THREE
            || self.layer_three.mixer().kind() != MixerKind::QwenSparseAttention
        {
            return invalid("layers 2 and 3 must be GDN and QSA respectively");
        }
        for block in [self.layer_two, self.layer_three] {
            if block.branch_count() != self.zero_one.branch_count()
                || block.hidden_size() != self.zero_one.hidden_size()
                || block.low_rank() != self.zero_one.low_rank()
                || block.eps().to_bits() != self.zero_one.eps().to_bits()
            {
                return invalid("post-PLE block geometry differs from layers zero-one");
            }
        }
        let qsa =
            self.layer_three.mixer().qsa().ok_or_else(|| {
                Qwen4ExpLayersZeroThreeError::Invalid("layer 3 is not QSA".into())
            })?;
        if qsa.capacity() > self.context_length() {
            return invalid("layer-three QSA capacity exceeds model context");
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpLayersZeroThreeMetalWeights<'a> {
    pub geometry: Qwen4ExpLayersZeroThreeMetalGeometry,
    pub zero_one: Qwen4ExpLayersZeroOneMetalWeights<'a>,
    pub layer_two: Qwen4ExpPostPleBlockMetalWeights<'a>,
    pub layer_three: Qwen4ExpPostPleBlockMetalWeights<'a>,
}

impl<'a> Qwen4ExpLayersZeroThreeMetalWeights<'a> {
    pub fn bind(
        weights: &'a Qwen4ExpMetalWeights,
        qsa_capacity: usize,
    ) -> Result<Self, Qwen4ExpLayersZeroThreeError> {
        let geometry =
            Qwen4ExpLayersZeroThreeMetalGeometry::from_config(weights.config(), qsa_capacity)?;
        Ok(Self {
            geometry,
            zero_one: Qwen4ExpLayersZeroOneMetalWeights::bind(weights)?,
            layer_two: Qwen4ExpPostPleBlockMetalWeights::bind(weights, LAYER_TWO, None)?,
            layer_three: Qwen4ExpPostPleBlockMetalWeights::bind(
                weights,
                LAYER_THREE,
                Some(qsa_capacity),
            )?,
        })
    }
}

pub struct Qwen4ExpLayersZeroThreeMetalWorkspace {
    geometry: Qwen4ExpLayersZeroThreeMetalGeometry,
    zero_one: Qwen4ExpLayersZeroOneMetalWorkspace,
    hyper_residual: MetalTensor,
    layer_two: Qwen4ExpPostPleBlockMetalWorkspace,
    layer_three: Qwen4ExpPostPleBlockMetalWorkspace,
    committed_length: usize,
    pending_length: Option<usize>,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
    encode_failed: bool,
}

impl Qwen4ExpLayersZeroThreeMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpLayersZeroThreeMetalGeometry,
    ) -> Result<Self, Qwen4ExpLayersZeroThreeError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            zero_one: Qwen4ExpLayersZeroOneMetalWorkspace::new(ctx, geometry.zero_one)?,
            hyper_residual: MetalTensor::zeros_f32(ctx, vec![geometry.hyper_width() as u64])?,
            layer_two: Qwen4ExpPostPleBlockMetalWorkspace::new(ctx, geometry.layer_two)?,
            layer_three: Qwen4ExpPostPleBlockMetalWorkspace::new(ctx, geometry.layer_three)?,
            committed_length: 0,
            pending_length: None,
            active_command: None,
            state_poisoned: false,
            encode_failed: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpLayersZeroThreeMetalGeometry {
        self.geometry
    }

    pub fn committed_length(&self) -> usize {
        self.committed_length
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpLayersZeroThreeError> {
        self.require_idle()?;
        if self.pending_length.is_some() {
            return invalid("cannot reset while a sequence update is pending");
        }
        self.zero_one.reset()?;
        self.layer_two.reset()?;
        self.layer_three.reset()?;
        self.committed_length = 0;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpLayersZeroThreeError> {
        let Some(command) = self.active_command.clone() else {
            if self.pending_length.is_some() {
                return invalid("sequence length is pending without an owning command");
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
        if let Err(error) = self.zero_one.release_after() {
            child_errors.push(format!("layers zero-one: {error}"));
        }
        if let Err(error) = self.layer_two.release_after() {
            child_errors.push(format!("layer two: {error}"));
        }
        if let Err(error) = self.layer_three.release_after() {
            child_errors.push(format!("layer three: {error}"));
        }
        self.active_command = None;
        let pending = self.pending_length.take();
        let pending_present = pending.is_some();
        if let Some(expected) = pending {
            if self.zero_one.next_position() != Some(expected as u64) {
                child_errors.push(format!(
                    "layers zero-one committed position {:?}, expected {expected}",
                    self.zero_one.next_position()
                ));
            }
            if self.layer_three.mixer_committed_length() != Some(expected) {
                child_errors.push(format!(
                    "layer-three QSA committed length {:?}, expected {expected}",
                    self.layer_three.mixer_committed_length()
                ));
            }
            if status == MTLCommandBufferStatus::Completed
                && command_error.is_none()
                && child_errors.is_empty()
                && !self.encode_failed
            {
                self.committed_length = expected;
                self.encode_failed = false;
                return Ok(());
            }
        }
        self.state_poisoned = true;
        Err(Qwen4ExpLayersZeroThreeError::CommandBuffer(format!(
            "status={status:?}, error={command_error:?}, encode_failed={}, pending_length={pending_present}, children={child_errors:?}",
            self.encode_failed
        )))
    }

    /// Release a first-cycle workspace from a command that will not be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may mutate four layers of causal
    /// state after another token acquires this workspace.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpLayersZeroThreeError> {
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
        if let Err(error) = unsafe { self.zero_one.abandon_uncommitted() } {
            child_errors.push(format!("layers zero-one: {error}"));
        }
        if let Err(error) = unsafe { self.layer_two.abandon_uncommitted() } {
            child_errors.push(format!("layer two: {error}"));
        }
        if let Err(error) = unsafe { self.layer_three.abandon_uncommitted() } {
            child_errors.push(format!("layer three: {error}"));
        }
        if !child_errors.is_empty() {
            return invalid(format!(
                "could not abandon every layers-zero-three child: {child_errors:?}"
            ));
        }
        self.active_command = None;
        self.pending_length = None;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpLayersZeroThreeError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "copy the first-cycle residual in its owning command, then release the workspace"]
pub struct Qwen4ExpLayersZeroThreeMetalRead<'a> {
    workspace: &'a mut Qwen4ExpLayersZeroThreeMetalWorkspace,
}

pub struct Qwen4ExpLayersZeroThreeMetalOutput<'a> {
    workspace: &'a Qwen4ExpLayersZeroThreeMetalWorkspace,
}

impl Qwen4ExpLayersZeroThreeMetalRead<'_> {
    pub fn output(&self) -> Qwen4ExpLayersZeroThreeMetalOutput<'_> {
        Qwen4ExpLayersZeroThreeMetalOutput {
            workspace: self.workspace,
        }
    }
}

impl Qwen4ExpLayersZeroThreeMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hyper_width() as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn branch_count(&self) -> usize {
        self.workspace.geometry.branch_count()
    }

    pub fn position(&self) -> usize {
        self.workspace.committed_length
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpLayersZeroThreeError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("layers-zero-three output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("layers-zero-three output must be copied by its owning command buffer");
        }
        require_tensor(
            "layers-zero-three copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hyper_width() as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                (
                    "layers-zero-three hyper residual",
                    &self.workspace.hyper_residual,
                ),
                ("layers-zero-three copied output destination", destination),
            ],
        )?;
        require_disjoint(&[
            (
                "layers-zero-three hyper residual",
                &self.workspace.hyper_residual,
            ),
            ("layers-zero-three copied output destination", destination),
        ])?;
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
pub fn encode_qwen4exp_layers_zero_three<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: Qwen4ExpLayersZeroThreeMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpLayersZeroThreeMetalWorkspace,
) -> Result<Qwen4ExpLayersZeroThreeMetalRead<'a>, Qwen4ExpLayersZeroThreeError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if workspace.pending_length.is_some() {
        return invalid("workspace has a sequence update without a command owner");
    }
    if weights.geometry != workspace.geometry {
        return invalid("layers-zero-three weight and workspace geometry differ");
    }
    if position != workspace.committed_length {
        return invalid(format!(
            "position {position} differs from committed length {}",
            workspace.committed_length
        ));
    }
    if position >= workspace.geometry.capacity() {
        return invalid(format!(
            "first-cycle QSA capacity {} is exhausted",
            workspace.geometry.capacity()
        ));
    }
    validate_nested_geometry(weights, workspace)?;
    crate::qwen4exp_layers_zero_one::validate_and_preflight(
        ctx,
        enc,
        token_id,
        position as u64,
        weights.zero_one,
        &workspace.zero_one,
    )?;
    crate::qwen4exp_post_ple_block::validate_and_preflight(
        ctx,
        position,
        &workspace.hyper_residual,
        weights.layer_two,
        &workspace.layer_two,
    )?;
    crate::qwen4exp_post_ple_block::validate_and_preflight(
        ctx,
        position,
        &workspace.hyper_residual,
        weights.layer_three,
        &workspace.layer_three,
    )?;
    validate_bridge(ctx, workspace)?;
    crate::qwen4exp_layers_zero_one::stage_ple_rows(
        token_id,
        position as u64,
        table,
        weights.zero_one,
        &mut workspace.zero_one,
    )?;
    reserve_command(workspace, enc, position + 1)?;
    if let Err(error) = encode_step(ctx, enc, token_id, position, table, weights, workspace) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpLayersZeroThreeMetalRead { workspace })
}

#[allow(clippy::too_many_arguments)]
fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    token_id: u32,
    position: usize,
    table: PleIq4NlTable<'_>,
    weights: Qwen4ExpLayersZeroThreeMetalWeights<'_>,
    workspace: &mut Qwen4ExpLayersZeroThreeMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
    let zero_one = encode_qwen4exp_layers_zero_one(
        ctx,
        enc,
        token_id,
        position as u64,
        table,
        weights.zero_one,
        &mut workspace.zero_one,
    )?;
    zero_one
        .output()
        .encode_copy_to(ctx, enc, &workspace.hyper_residual)?;
    drop(zero_one);

    let layer_two = encode_qwen4exp_post_ple_block(
        ctx,
        enc,
        position,
        &workspace.hyper_residual,
        weights.layer_two,
        &mut workspace.layer_two,
    )?;
    drop(layer_two);
    let layer_three = encode_qwen4exp_post_ple_block(
        ctx,
        enc,
        position,
        &workspace.hyper_residual,
        weights.layer_three,
        &mut workspace.layer_three,
    )?;
    drop(layer_three);
    Ok(())
}

fn validate_nested_geometry(
    weights: Qwen4ExpLayersZeroThreeMetalWeights<'_>,
    workspace: &Qwen4ExpLayersZeroThreeMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
    let g = workspace.geometry;
    if weights.zero_one.geometry != g.zero_one
        || weights.layer_two.geometry != g.layer_two
        || weights.layer_three.geometry != g.layer_three
        || workspace.zero_one.geometry() != g.zero_one
        || workspace.layer_two.geometry() != g.layer_two
        || workspace.layer_three.geometry() != g.layer_three
    {
        return invalid("nested weight or workspace geometry differs from the first cycle");
    }
    if workspace
        .zero_one
        .next_position()
        .map(|value| value as usize)
        != (workspace.committed_length > 0).then_some(workspace.committed_length)
    {
        return invalid("layers-zero-one history differs from the outer committed length");
    }
    if workspace.layer_three.mixer_committed_length() != Some(workspace.committed_length) {
        return invalid("layer-three QSA length differs from the outer committed length");
    }
    Ok(())
}

fn validate_bridge(
    ctx: &MetalContext,
    workspace: &Qwen4ExpLayersZeroThreeMetalWorkspace,
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
    require_tensor(
        "layers-zero-three hyper residual",
        &workspace.hyper_residual,
        GgmlType::F32,
        &[workspace.geometry.hyper_width() as u64],
        true,
    )?;
    require_same_device(
        ctx,
        &[(
            "layers-zero-three hyper residual",
            &workspace.hyper_residual,
        )],
    )?;
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

fn validate_encoder(
    ctx: &MetalContext,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("layers-zero-three dependent dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "layers-zero-three encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpLayersZeroThreeMetalWorkspace,
    enc: &KernelEncoder,
    pending_length: usize,
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.pending_length = Some(pending_length);
    workspace.encode_failed = false;
    Ok(())
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
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

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpLayersZeroThreeError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| {
            Qwen4ExpLayersZeroThreeError::Invalid("tensor element count overflow".into())
        })?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpLayersZeroThreeError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
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
        .ok_or_else(|| Qwen4ExpLayersZeroThreeError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpLayersZeroThreeError> {
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
        .ok_or_else(|| Qwen4ExpLayersZeroThreeError::Invalid(format!("{name} range overflow")))?;
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
) -> Result<(), Qwen4ExpLayersZeroThreeError> {
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

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpLayersZeroThreeError> {
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

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpLayersZeroThreeError> {
    Err(Qwen4ExpLayersZeroThreeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::metal::{allocation_census_begin, allocation_census_take};
    use crate::metal_forward::encode_mat_vec_dispatch;
    use crate::qwen4exp::{PleConfig, Qwen4ExpConfig};
    use crate::qwen4exp_forward::{
        GatedResidualReadWeights, gated_residual_combine, gated_residual_mix,
    };
    use crate::qwen4exp_gdn::{
        GatedDeltaNetMetalGeometry, GatedDeltaNetMetalWeights, GatedDeltaNetMetalWorkspace,
        encode_gated_delta_net,
    };
    use crate::qwen4exp_layer_zero::{Qwen4ExpLayerZeroMetalWeights, Qwen4ExpResidualMetalWeights};
    use crate::qwen4exp_metal::GatedResidualMetalReadWeights;
    use crate::qwen4exp_metal::{GatedResidualMetalScratch, encode_final_gated_residual_mix};
    use crate::qwen4exp_moe::{
        Qwen4ExpMoeMetalGeometry, Qwen4ExpMoeMetalWeights, Qwen4ExpMoeMetalWorkspace,
        encode_qwen4exp_moe,
    };
    use crate::qwen4exp_ple_metal::{Qwen4ExpPleMetalGeometry, Qwen4ExpPleMetalWeights};
    use crate::qwen4exp_post_ple_block::{
        Qwen4ExpPostPleMixerMetalGeometry, Qwen4ExpPostPleMixerMetalWeights,
    };
    use crate::qwen4exp_profile::packed_stage_sample_count;
    use crate::qwen4exp_qsa::{QwenSparseAttentionMetalGeometry, QwenSparseAttentionMetalWeights};
    use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
    use crate::qwen4exp_runtime::forward_qwen4exp_text_token_sync;
    use crate::qwen4exp_text_session::{
        Qwen4ExpPackedEncodeCpuTiming, Qwen4ExpTextSessionMetalGeometry,
        Qwen4ExpTextSessionMetalWeights, Qwen4ExpTextSessionMetalWorkspace,
        encode_qwen4exp_text_packed, encode_qwen4exp_text_packed_layer_sampled,
        encode_qwen4exp_text_token,
    };
    use crate::tensor::TensorDesc;
    use half::{bf16, f16};
    use objc2_metal::MTLCommandQueue;

    const BRANCHES: usize = 4;
    const HIDDEN: usize = 256;
    const RANK: usize = 32;
    const HYPER: usize = BRANCHES * HIDDEN;
    const VOCAB: usize = 32;
    const EXPERTS: usize = 16;
    const TOP_K: usize = 10;
    const FFN: usize = 32;
    const GDN_HEAD_DIM: usize = 128;
    const PLE_HEADS: usize = 2;
    const PLE_KERNEL: usize = 4;
    const PLE_DILATION: usize = 3;
    const CONTEXT: usize = 16;
    const CAPACITY: usize = 8;
    const TABLE_ROWS: usize = 36;

    struct AllocationCensusGuard {
        active: bool,
    }

    impl AllocationCensusGuard {
        fn begin() -> Self {
            allocation_census_begin();
            Self { active: true }
        }

        fn take(mut self) -> Vec<crate::metal::MetalAllocationCensusRow> {
            self.active = false;
            allocation_census_take()
        }
    }

    impl Drop for AllocationCensusGuard {
        fn drop(&mut self) {
            if self.active {
                let _ = allocation_census_take();
            }
        }
    }

    fn synthetic_config() -> Qwen4ExpConfig {
        let mut config = Qwen4ExpConfig::flash_next_reference();
        config.context_length = CONTEXT as u32;
        config.layer_count = 4;
        config.hidden_size = HIDDEN as u32;
        config.vocab_size = VOCAB as u32;
        config.hyper_connection.low_rank = RANK as u32;
        config.attention.query_heads = 4;
        config.attention.kv_heads = 2;
        config.gated_delta_net.key_heads = 1;
        config.gated_delta_net.value_heads = 1;
        config.gated_delta_net.inner_size = GDN_HEAD_DIM as u32;
        config.moe.expert_count = EXPERTS as u32;
        config.moe.experts_per_token = TOP_K as u32;
        config.moe.expert_intermediate_size = FFN as u32;
        config.moe.shared_expert_intermediate_size = FFN as u32;
        config.qsa.token_budget = 8;
        config.compress_ratios = vec![0, 0, 0, 4];
        config.ple = Some(PleConfig {
            token_vocab_size: VOCAB as u32,
            layers: vec![1],
            ngram_size: PLE_DILATION as u32,
            heads_per_ngram: 1,
            embedding_head_dim: GDN_HEAD_DIM as u32,
            conv_kernel: PLE_KERNEL as u32,
            eos_token_id: 31,
            image_token_id: Some(30),
            multipliers: vec![3, 5, 7],
            head_offsets: vec![0, 17],
            head_vocab_sizes: vec![17, 19],
        });
        config.validate().unwrap();
        config
    }

    fn context() -> Option<MetalContext> {
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
        tensor.provenance = crate::metal::MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
    }

    fn weight_f32(
        ctx: &MetalContext,
        count: usize,
        shape: Vec<u64>,
        seed: usize,
        bias: f32,
    ) -> MetalTensor {
        let data = values(count, seed, 0.002, bias);
        weight_bytes(ctx, bytemuck::cast_slice(&data), shape, GgmlType::F32)
    }

    fn weight_quant(
        ctx: &MetalContext,
        count: usize,
        shape: Vec<u64>,
        dtype: GgmlType,
        row_width: usize,
        seed: usize,
    ) -> MetalTensor {
        let bytes = quantize_rows(&values(count, seed, 0.002, 0.0), dtype, row_width);
        weight_bytes(ctx, &bytes, shape, dtype)
    }

    fn weight_bf16(ctx: &MetalContext, count: usize, shape: Vec<u64>, seed: usize) -> MetalTensor {
        let data = values(count, seed, 0.002, 0.0)
            .into_iter()
            .map(|value| bf16::from_f32(value).to_bits())
            .collect::<Vec<_>>();
        weight_bytes(ctx, bytemuck::cast_slice(&data), shape, GgmlType::BF16)
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

    fn read_numeric_tensor(tensor: &MetalTensor) -> Vec<f32> {
        match tensor.dtype {
            GgmlType::F32 => read_f32(tensor),
            GgmlType::F16 => unsafe {
                let source = tensor
                    .buffer
                    .contents()
                    .as_ptr()
                    .cast::<u8>()
                    .add(tensor.offset as usize)
                    .cast::<u16>();
                std::slice::from_raw_parts(source, tensor.n_elements() as usize)
                    .iter()
                    .map(|&bits| f16::from_bits(bits).to_f32())
                    .collect()
            },
            dtype => panic!("unsupported numeric test tensor type {dtype:?}"),
        }
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

    fn assert_close(label: &str, actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                actual.is_finite() && (actual - expected).abs() <= tolerance,
                "{label}[{index}]: expected {expected}, got {actual}"
            );
        }
    }

    fn max_delta(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0, f32::max)
    }

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
                &values(HYPER * RANK, seed + 1, 0.002, 0.0),
                GgmlType::Q8_0,
                HYPER,
            );
            let up_bytes = quantize_rows(
                &values(RANK * HYPER, seed + 2, 0.002, 0.0),
                GgmlType::Q8_0,
                RANK,
            );
            let down_cpu = dequant(&down_bytes, GgmlType::Q8_0, vec![HYPER, RANK]);
            let up_cpu = dequant(&up_bytes, GgmlType::Q8_0, vec![RANK, HYPER]);
            let inject_cpu = values(HYPER * BRANCHES, seed + 3, 0.002, 0.0);
            Self {
                norm: weight_bytes(
                    ctx,
                    bytemuck::cast_slice(&norm_cpu),
                    vec![HYPER as u64],
                    GgmlType::F32,
                ),
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
                inject: weight_bytes(
                    ctx,
                    bytemuck::cast_slice(&inject_cpu),
                    vec![HYPER as u64, BRANCHES as u64],
                    GgmlType::F32,
                ),
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

        fn weights(&self) -> Qwen4ExpResidualMetalWeights<'_> {
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
        tensors: [MetalTensor; 9],
    }

    impl GdnFixture {
        fn new(ctx: &MetalContext, geometry: GatedDeltaNetMetalGeometry, seed: usize) -> Self {
            Self {
                tensors: [
                    weight_quant(
                        ctx,
                        HIDDEN * geometry.conv_width(),
                        vec![HIDDEN as u64, geometry.conv_width() as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * geometry.value_width(),
                        vec![HIDDEN as u64, geometry.value_width() as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed + 1,
                    ),
                    weight_f32(ctx, HIDDEN, vec![HIDDEN as u64, 1], seed + 2, 0.0),
                    weight_f32(ctx, HIDDEN, vec![HIDDEN as u64, 1], seed + 3, 0.0),
                    weight_f32(ctx, 1, vec![1], seed + 4, -0.55),
                    weight_f32(ctx, 1, vec![1], seed + 5, 0.15),
                    weight_f32(
                        ctx,
                        geometry.conv_width() * PLE_KERNEL,
                        vec![PLE_KERNEL as u64, geometry.conv_width() as u64],
                        seed + 6,
                        0.0,
                    ),
                    weight_f32(ctx, GDN_HEAD_DIM, vec![GDN_HEAD_DIM as u64], seed + 7, 1.0),
                    weight_quant(
                        ctx,
                        geometry.value_width() * HIDDEN,
                        vec![geometry.value_width() as u64, HIDDEN as u64],
                        GgmlType::Q8_0,
                        geometry.value_width(),
                        seed + 8,
                    ),
                ],
            }
        }

        fn weights(&self, geometry: GatedDeltaNetMetalGeometry) -> GatedDeltaNetMetalWeights<'_> {
            let t = &self.tensors;
            GatedDeltaNetMetalWeights {
                geometry,
                qkv: &t[0],
                gate: &t[1],
                beta: &t[2],
                alpha: &t[3],
                a: &t[4],
                dt_bias: &t[5],
                conv: &t[6],
                norm: &t[7],
                output: &t[8],
            }
        }
    }

    struct MoeFixture {
        tensors: [MetalTensor; 8],
    }

    impl MoeFixture {
        fn new(ctx: &MetalContext, seed: usize) -> Self {
            Self {
                tensors: [
                    weight_f32(
                        ctx,
                        HIDDEN * EXPERTS,
                        vec![HIDDEN as u64, EXPERTS as u64],
                        seed,
                        0.0,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * FFN * EXPERTS,
                        vec![HIDDEN as u64, FFN as u64, EXPERTS as u64],
                        GgmlType::IQ4_XS,
                        HIDDEN,
                        seed + 1,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * FFN * EXPERTS,
                        vec![HIDDEN as u64, FFN as u64, EXPERTS as u64],
                        GgmlType::IQ4_XS,
                        HIDDEN,
                        seed + 2,
                    ),
                    weight_quant(
                        ctx,
                        FFN * HIDDEN * EXPERTS,
                        vec![FFN as u64, HIDDEN as u64, EXPERTS as u64],
                        GgmlType::IQ4_NL,
                        FFN,
                        seed + 3,
                    ),
                    weight_f32(ctx, HIDDEN, vec![HIDDEN as u64], seed + 4, 0.0),
                    weight_quant(
                        ctx,
                        HIDDEN * FFN,
                        vec![HIDDEN as u64, FFN as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed + 5,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * FFN,
                        vec![HIDDEN as u64, FFN as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed + 6,
                    ),
                    weight_quant(
                        ctx,
                        FFN * HIDDEN,
                        vec![FFN as u64, HIDDEN as u64],
                        GgmlType::Q8_0,
                        FFN,
                        seed + 7,
                    ),
                ],
            }
        }

        fn weights(&self, geometry: Qwen4ExpMoeMetalGeometry) -> Qwen4ExpMoeMetalWeights<'_> {
            let t = &self.tensors;
            Qwen4ExpMoeMetalWeights {
                geometry,
                router: &t[0],
                routed_gate: &t[1],
                routed_up: &t[2],
                routed_down: &t[3],
                shared_router: &t[4],
                shared_gate: &t[5],
                shared_up: &t[6],
                shared_down: &t[7],
            }
        }
    }

    struct GdnLayerFixture {
        attention: ResidualFixture,
        gdn: GdnFixture,
        ffn: ResidualFixture,
        moe: MoeFixture,
    }

    impl GdnLayerFixture {
        fn new(ctx: &MetalContext, gdn: GatedDeltaNetMetalGeometry, seed: usize) -> Self {
            Self {
                attention: ResidualFixture::new(ctx, seed),
                gdn: GdnFixture::new(ctx, gdn, seed + 20),
                ffn: ResidualFixture::new(ctx, seed + 40),
                moe: MoeFixture::new(ctx, seed + 60),
            }
        }
    }

    struct QsaFixture {
        tensors: [MetalTensor; 10],
    }

    impl QsaFixture {
        fn new(ctx: &MetalContext, query_heads: usize, kv_heads: usize, seed: usize) -> Self {
            let query_width = query_heads * 256;
            let query_projection_width = query_width * 2;
            let kv_width = kv_heads * 256;
            Self {
                tensors: [
                    weight_quant(
                        ctx,
                        HIDDEN * query_projection_width,
                        vec![HIDDEN as u64, query_projection_width as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * kv_width,
                        vec![HIDDEN as u64, kv_width as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed + 1,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * kv_width,
                        vec![HIDDEN as u64, kv_width as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        seed + 2,
                    ),
                    weight_quant(
                        ctx,
                        query_width * HIDDEN,
                        vec![query_width as u64, HIDDEN as u64],
                        GgmlType::Q8_0,
                        query_width,
                        seed + 3,
                    ),
                    weight_f32(ctx, 256, vec![256], seed + 4, 1.0),
                    weight_f32(ctx, 256, vec![256], seed + 5, 1.0),
                    weight_bf16(ctx, HIDDEN * 512, vec![HIDDEN as u64, 512], seed + 6),
                    weight_bf16(ctx, HIDDEN * 128, vec![HIDDEN as u64, 128], seed + 7),
                    weight_f32(ctx, 128, vec![128], seed + 8, 1.0),
                    weight_f32(ctx, 128, vec![128], seed + 9, 1.0),
                ],
            }
        }

        fn weights(
            &self,
            geometry: QwenSparseAttentionMetalGeometry,
        ) -> QwenSparseAttentionMetalWeights<'_> {
            let t = &self.tensors;
            QwenSparseAttentionMetalWeights {
                geometry,
                query: &t[0],
                key: &t[1],
                value: &t[2],
                output: &t[3],
                query_norm: &t[4],
                key_norm: &t[5],
                index_query: &t[6],
                index_key: &t[7],
                index_query_norm: &t[8],
                index_key_norm: &t[9],
            }
        }
    }

    struct PleFixture {
        tensors: [MetalTensor; 6],
    }

    impl PleFixture {
        fn new(ctx: &MetalContext) -> Self {
            Self {
                tensors: [
                    weight_quant(
                        ctx,
                        HIDDEN * HYPER,
                        vec![HIDDEN as u64, HYPER as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        501,
                    ),
                    weight_quant(
                        ctx,
                        HIDDEN * HIDDEN,
                        vec![HIDDEN as u64, HIDDEN as u64],
                        GgmlType::Q8_0,
                        HIDDEN,
                        502,
                    ),
                    weight_f32(ctx, HYPER, vec![HYPER as u64], 503, 1.0),
                    weight_f32(ctx, HYPER, vec![HYPER as u64], 504, 1.0),
                    weight_f32(ctx, HYPER, vec![HYPER as u64], 505, 1.0),
                    weight_f32(
                        ctx,
                        HYPER * PLE_KERNEL,
                        vec![PLE_KERNEL as u64, HYPER as u64],
                        506,
                        0.02,
                    ),
                ],
            }
        }

        fn weights(&self, geometry: Qwen4ExpPleMetalGeometry) -> Qwen4ExpPleMetalWeights<'_> {
            let t = &self.tensors;
            Qwen4ExpPleMetalWeights {
                geometry,
                key: &t[0],
                value: &t[1],
                key_norm: &t[2],
                query_norm: &t[3],
                conv_norm: &t[4],
                conv: &t[5],
            }
        }
    }

    struct TableFixture {
        desc: TensorDesc,
        bytes: Vec<u8>,
    }

    impl TableFixture {
        fn new() -> Self {
            let bytes = quantize_rows(
                &values(TABLE_ROWS * GDN_HEAD_DIM, 601, 0.02, 0.0),
                GgmlType::IQ4_NL,
                GDN_HEAD_DIM,
            );
            Self {
                desc: TensorDesc {
                    name: "synthetic_ple_table".into(),
                    shape: vec![GDN_HEAD_DIM as u64, TABLE_ROWS as u64],
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
        config: Qwen4ExpConfig,
        geometry: Qwen4ExpLayersZeroThreeMetalGeometry,
        token: MetalTensor,
        layers: [GdnLayerFixture; 3],
        layer_three_attention: ResidualFixture,
        layer_three_qsa: QsaFixture,
        layer_three_ffn: ResidualFixture,
        layer_three_moe: MoeFixture,
        ple: PleFixture,
        table: TableFixture,
    }

    impl SyntheticFixture {
        fn new(ctx: &MetalContext) -> Self {
            Self::new_with_qsa_kv_heads(ctx, 2)
        }

        fn new_with_qsa_kv_heads(ctx: &MetalContext, qsa_kv_heads: u32) -> Self {
            let mut config = synthetic_config();
            config.attention.kv_heads = qsa_kv_heads;
            Self::new_with_config(ctx, config)
        }

        fn new_with_two_qsa_layers(ctx: &MetalContext) -> Self {
            let mut config = synthetic_config();
            config.layer_count = 8;
            config.attention.kv_heads = 1;
            config.compress_ratios = (0..config.layer_count)
                .map(|layer| if layer % 4 == 3 { 4 } else { 0 })
                .collect();
            Self::new_with_config(ctx, config)
        }

        fn new_with_config(ctx: &MetalContext, config: Qwen4ExpConfig) -> Self {
            config.validate().unwrap();
            let geometry =
                Qwen4ExpLayersZeroThreeMetalGeometry::from_config(&config, CAPACITY).unwrap();
            let gdn = geometry.zero_one().layer_zero().gdn();
            let qsa_query_heads = config.attention.query_heads as usize;
            let qsa_kv_heads = config.attention.kv_heads as usize;
            Self {
                token: weight_quant(
                    ctx,
                    HIDDEN * VOCAB,
                    vec![HIDDEN as u64, VOCAB as u64],
                    GgmlType::Q8_0,
                    HIDDEN,
                    1,
                ),
                layers: [
                    GdnLayerFixture::new(ctx, gdn, 10),
                    GdnLayerFixture::new(ctx, gdn, 110),
                    GdnLayerFixture::new(ctx, gdn, 210),
                ],
                layer_three_attention: ResidualFixture::new(ctx, 310),
                layer_three_qsa: QsaFixture::new(ctx, qsa_query_heads, qsa_kv_heads, 330),
                layer_three_ffn: ResidualFixture::new(ctx, 350),
                layer_three_moe: MoeFixture::new(ctx, 370),
                ple: PleFixture::new(ctx),
                table: TableFixture::new(),
                geometry,
                config,
            }
        }

        fn zero_one_weights(&self) -> Qwen4ExpLayersZeroOneMetalWeights<'_> {
            let zero = self.geometry.zero_one();
            let layer = zero.layer_zero();
            Qwen4ExpLayersZeroOneMetalWeights {
                geometry: zero,
                ple_config: self.config.ple.as_ref().unwrap(),
                layer_zero: Qwen4ExpLayerZeroMetalWeights {
                    geometry: layer,
                    token_embedding: &self.token,
                    attention_residual: self.layers[0].attention.weights(),
                    gdn: self.layers[0].gdn.weights(layer.gdn()),
                    ffn_residual: self.layers[0].ffn.weights(),
                    moe: self.layers[0].moe.weights(layer.moe()),
                },
                ple: self.ple.weights(zero.ple()),
                layer_one_attention_residual: self.layers[1].attention.weights(),
                layer_one_gdn: self.layers[1].gdn.weights(layer.gdn()),
                layer_one_ffn_residual: self.layers[1].ffn.weights(),
                layer_one_moe: self.layers[1].moe.weights(layer.moe()),
            }
        }

        fn layer_two_weights(&self) -> Qwen4ExpPostPleBlockMetalWeights<'_> {
            let geometry = self.geometry.layer_two();
            let gdn = match geometry.mixer() {
                crate::qwen4exp_post_ple_block::Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(
                    geometry,
                ) => geometry,
                _ => unreachable!(),
            };
            Qwen4ExpPostPleBlockMetalWeights {
                geometry,
                attention_residual: self.layers[2].attention.weights(),
                mixer: Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(
                    self.layers[2].gdn.weights(gdn),
                ),
                ffn_residual: self.layers[2].ffn.weights(),
                moe: self.layers[2].moe.weights(geometry.moe()),
            }
        }

        fn layer_three_weights(&self) -> Qwen4ExpPostPleBlockMetalWeights<'_> {
            let geometry = self.geometry.layer_three();
            Qwen4ExpPostPleBlockMetalWeights {
                geometry,
                attention_residual: self.layer_three_attention.weights(),
                mixer: Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(
                    self.layer_three_qsa
                        .weights(geometry.mixer().qsa().unwrap()),
                ),
                ffn_residual: self.layer_three_ffn.weights(),
                moe: self.layer_three_moe.weights(geometry.moe()),
            }
        }

        fn weights(&self) -> Qwen4ExpLayersZeroThreeMetalWeights<'_> {
            Qwen4ExpLayersZeroThreeMetalWeights {
                geometry: self.geometry,
                zero_one: self.zero_one_weights(),
                layer_two: self.layer_two_weights(),
                layer_three: self.layer_three_weights(),
            }
        }

        fn text_weights<'a>(
            &'a self,
            tail: &'a TextSessionTailFixture,
            geometry: &Qwen4ExpTextSessionMetalGeometry,
        ) -> Qwen4ExpTextSessionMetalWeights<'a> {
            let post_ple = geometry
                .post_ple()
                .iter()
                .copied()
                .map(|block| match block.mixer() {
                    Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(gdn) => {
                        Qwen4ExpPostPleBlockMetalWeights {
                            geometry: block,
                            attention_residual: self.layers[2].attention.weights(),
                            mixer: Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(
                                self.layers[2].gdn.weights(gdn),
                            ),
                            ffn_residual: self.layers[2].ffn.weights(),
                            moe: self.layers[2].moe.weights(block.moe()),
                        }
                    }
                    Qwen4ExpPostPleMixerMetalGeometry::QwenSparseAttention(qsa) => {
                        Qwen4ExpPostPleBlockMetalWeights {
                            geometry: block,
                            attention_residual: self.layer_three_attention.weights(),
                            mixer: Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(
                                self.layer_three_qsa.weights(qsa),
                            ),
                            ffn_residual: self.layer_three_ffn.weights(),
                            moe: self.layer_three_moe.weights(block.moe()),
                        }
                    }
                })
                .collect();
            Qwen4ExpTextSessionMetalWeights {
                geometry: geometry.clone(),
                zero_one: self.zero_one_weights(),
                post_ple,
                final_read: tail.final_residual.weights().read,
                output: &tail.output,
            }
        }
    }

    struct TextSessionTailFixture {
        final_residual: ResidualFixture,
        output: MetalTensor,
        output_cpu: Vec<f32>,
    }

    impl TextSessionTailFixture {
        fn new(ctx: &MetalContext) -> Self {
            let output_bytes = quantize_rows(
                &values(HIDDEN * VOCAB, 901, 0.002, 0.0),
                GgmlType::Q6_K,
                HIDDEN,
            );
            Self {
                final_residual: ResidualFixture::new(ctx, 801),
                output: weight_bytes(
                    ctx,
                    &output_bytes,
                    vec![HIDDEN as u64, VOCAB as u64],
                    GgmlType::Q6_K,
                ),
                output_cpu: dequant(&output_bytes, GgmlType::Q6_K, vec![HIDDEN, VOCAB]),
            }
        }

        fn cpu_logits(&self, hidden: &[f32]) -> Vec<f32> {
            self.output_cpu
                .chunks_exact(HIDDEN)
                .map(|row| {
                    row.iter()
                        .zip(hidden)
                        .map(|(weight, value)| weight * value)
                        .sum()
                })
                .collect()
        }
    }

    fn run_zero_one(
        ctx: &MetalContext,
        fixture: &SyntheticFixture,
        workspace: &mut Qwen4ExpLayersZeroOneMetalWorkspace,
        token: u32,
        position: usize,
        output: &MetalTensor,
    ) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_layers_zero_one(
            ctx,
            &encoder,
            token,
            position as u64,
            fixture.table.table(),
            fixture.zero_one_weights(),
            workspace,
        )
        .unwrap();
        read.output().encode_copy_to(ctx, &encoder, output).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
    }

    fn run_post(
        ctx: &MetalContext,
        position: usize,
        hyper: &MetalTensor,
        weights: Qwen4ExpPostPleBlockMetalWeights<'_>,
        workspace: &mut Qwen4ExpPostPleBlockMetalWorkspace,
    ) {
        let expected_elements = weights.geometry.hyper_width() as u64;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_qwen4exp_post_ple_block(ctx, &encoder, position, hyper, weights, workspace)
                .unwrap();
        assert_eq!(read.output().n_elements(), expected_elements);
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
    }

    fn run_final_control(
        ctx: &MetalContext,
        hyper: &MetalTensor,
        tail: &TextSessionTailFixture,
        scratch: &mut GatedResidualMetalScratch,
        logits: &MetalTensor,
        eps: f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let final_read = encode_final_gated_residual_mix(
            ctx,
            &encoder,
            hyper,
            eps,
            tail.final_residual.weights().read,
            scratch,
        )
        .unwrap();
        encode_mat_vec_dispatch(
            ctx,
            &encoder,
            &tail.output,
            final_read.mixed(),
            logits,
            HIDDEN,
            VOCAB,
        )
        .unwrap();
        drop(final_read);
        encoder.end();
        command.commit();
        scratch.release_after().unwrap();
        (read_f32(scratch.mixed_tensor()), read_f32(logits))
    }

    fn run_text_session(
        ctx: &MetalContext,
        fixture: &SyntheticFixture,
        tail: &TextSessionTailFixture,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
        token: u32,
        position: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let weights = fixture.text_weights(tail, geometry);
        let logits = forward_qwen4exp_text_token_sync(
            ctx,
            token,
            fixture.table.table(),
            &weights,
            workspace,
        )
        .unwrap();
        assert_eq!(logits.position(), position);
        assert_eq!(logits.n_elements(), VOCAB as u64);
        assert_eq!(logits.dtype(), GgmlType::F32);
        let logits = logits.to_vec();
        (read_f32(workspace.final_hidden_tensor()), logits)
    }

    fn run_packed_text_session(
        ctx: &MetalContext,
        fixture: &SyntheticFixture,
        tail: &TextSessionTailFixture,
        geometry: &Qwen4ExpTextSessionMetalGeometry,
        workspace: &mut Qwen4ExpTextSessionMetalWorkspace,
        tokens: &[u32],
        start_position: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let weights = fixture.text_weights(tail, geometry);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            ctx,
            &encoder,
            tokens,
            start_position,
            fixture.table.table(),
            &weights,
            workspace,
        )
        .unwrap();
        assert_eq!(pending.position(), start_position + tokens.len() - 1);
        drop(pending);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        (
            read_f32(workspace.final_hidden_tensor()),
            workspace.logits().unwrap().to_vec(),
        )
    }

    fn assert_selected_band_reset_sequences(
        census: &[crate::metal::DispatchCensusRow],
        expected_qsa_layers: usize,
    ) {
        let expected = [
            "kernel_qwen4exp_qsa_reset_selected_controls_i32",
            "kernel_qwen4exp_qsa_norm_rope_packed_f32",
            "kernel_qwen4exp_qsa_index_scores_packed_4x128_f16",
            "kernel_deepseek_v4_select_top_k_radix4_ids_f32",
            "kernel_qwen4exp_qsa_expand_ids_packed_i32",
            "kernel_qwen4exp_qsa_attention_logits_packed_f16",
            "kernel_qwen4exp_qsa_attention_softmax_value_packed_f16",
            "kernel_qwen4exp_qsa_audit_selected_i32",
        ];
        let names = census
            .iter()
            .filter(|row| row.tag.as_deref() == Some("qwen4exp.qsa.selected_band.0"))
            .map(|row| row.kernel.as_str())
            .collect::<Vec<_>>();
        assert_eq!(names.len(), expected.len() * expected_qsa_layers);
        for layer in 0..expected_qsa_layers {
            let start = layer * expected.len();
            assert_eq!(&names[start..start + expected.len()], expected);
        }
    }

    fn argmax(values: &[f32]) -> usize {
        values
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.total_cmp(right))
            .map(|(index, _)| index)
            .unwrap()
    }

    fn run_integrated(
        ctx: &MetalContext,
        fixture: &SyntheticFixture,
        workspace: &mut Qwen4ExpLayersZeroThreeMetalWorkspace,
        token: u32,
        position: usize,
    ) -> Vec<f32> {
        let output = MetalTensor::zeros_f32(ctx, vec![HYPER as u64]).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_layers_zero_three(
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

    #[test]
    fn text_session_logits_match_separate_commands_across_qsa_publication() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, CAPACITY).unwrap();
        let allocation_census = AllocationCensusGuard::begin();
        let mut integrated =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, geometry.clone()).unwrap();
        let allocation_census = allocation_census.take();
        let mut observed = allocation_census
            .iter()
            .map(|row| row.requested_bytes)
            .collect::<Vec<_>>();
        let mut planned = integrated
            .memory_plan()
            .allocations()
            .iter()
            .map(|allocation| allocation.logical_bytes)
            .collect::<Vec<_>>();
        observed.sort_unstable();
        planned.sort_unstable();
        assert_eq!(observed, planned);
        let mut zero_one =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, geometry.zero_one()).unwrap();
        let mut layer_two =
            Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, geometry.post_ple()[0]).unwrap();
        let mut layer_three =
            Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, geometry.post_ple()[1]).unwrap();
        let mut final_read = GatedResidualMetalScratch::new(&ctx, BRANCHES, HIDDEN, RANK).unwrap();
        let hyper = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();
        let control_logits = MetalTensor::zeros_f32(&ctx, vec![VOCAB as u64]).unwrap();

        for (position, token) in [1_u32, 7, 3, 11, 5].into_iter().enumerate() {
            let weights = fixture.text_weights(&tail, &geometry);
            run_zero_one(&ctx, &fixture, &mut zero_one, token, position, &hyper);
            run_post(&ctx, position, &hyper, weights.post_ple[0], &mut layer_two);
            run_post(
                &ctx,
                position,
                &hyper,
                weights.post_ple[1],
                &mut layer_three,
            );
            let (expected_hidden, expected_logits) = run_final_control(
                &ctx,
                &hyper,
                &tail,
                &mut final_read,
                &control_logits,
                geometry.eps(),
            );
            let (actual_hidden, actual_logits) = run_text_session(
                &ctx,
                &fixture,
                &tail,
                &geometry,
                &mut integrated,
                token,
                position,
            );
            assert_close(
                "text-session final hidden",
                &actual_hidden,
                &expected_hidden,
                8e-5,
            );
            assert_close(
                "text-session logits",
                &actual_logits,
                &expected_logits,
                1e-4,
            );
            assert_close(
                "text-session Q6_K logits",
                &actual_logits,
                &tail.cpu_logits(&actual_hidden),
                2e-4,
            );
            assert_eq!(argmax(&actual_logits), argmax(&expected_logits));
            assert_eq!(integrated.committed_length(), position + 1);
            assert_eq!(integrated.qsa_committed_lengths(), vec![(3, position + 1)]);
        }
        assert_eq!(layer_three.mixer_committed_length(), Some(5));
    }

    #[test]
    fn packed_text_session_matches_scalar_and_hands_off_without_state_migration() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, CAPACITY).unwrap();
        let mut scalar =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, geometry.clone()).unwrap();
        let mut packed =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        let tokens = [1_u32, 7, 3, 11, 5];
        let persistent_before = packed.persistent_state_tensors();

        let mut scalar_hidden = Vec::new();
        let mut scalar_logits = Vec::new();
        for (position, &token) in tokens.iter().enumerate() {
            (scalar_hidden, scalar_logits) = run_text_session(
                &ctx,
                &fixture,
                &tail,
                &geometry,
                &mut scalar,
                token,
                position,
            );
        }

        let weights = fixture.text_weights(&tail, &geometry);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &tokens,
            0,
            fixture.table.table(),
            &weights,
            &mut packed,
        )
        .unwrap();
        assert_eq!(pending.position(), tokens.len() - 1);
        drop(pending);
        encoder.end();
        command.commit();
        packed.release_after().unwrap();
        let persistent_after = packed.persistent_state_tensors();
        assert_eq!(persistent_before.len(), persistent_after.len());
        for (before, after) in persistent_before.iter().zip(&persistent_after) {
            assert_eq!(
                Retained::as_ptr(&after.buffer),
                Retained::as_ptr(&before.buffer)
            );
            assert_eq!(after.offset, before.offset);
            assert_eq!(after.shape, before.shape);
            assert_eq!(after.dtype, before.dtype);
            assert_eq!(after.provenance(), before.provenance());
        }

        let packed_hidden = read_f32(packed.final_hidden_tensor());
        let packed_logits = packed.logits().unwrap().to_vec();
        assert_close(
            "packed text-session final hidden",
            &packed_hidden,
            &scalar_hidden,
            2e-3,
        );
        assert_close(
            "packed text-session logits",
            &packed_logits,
            &scalar_logits,
            5e-3,
        );
        assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
        assert_eq!(packed.committed_length(), tokens.len());
        assert_eq!(packed.qsa_committed_lengths(), vec![(3, tokens.len())]);
        assert_eq!(packed.ple_prior_tokens(), scalar.ple_prior_tokens());

        let continuation = 9;
        let (scalar_hidden, scalar_logits) = run_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut scalar,
            continuation,
            tokens.len(),
        );
        let (packed_hidden, packed_logits) = run_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut packed,
            continuation,
            tokens.len(),
        );
        assert_close(
            "packed-to-scalar handoff hidden",
            &packed_hidden,
            &scalar_hidden,
            2e-3,
        );
        assert_close(
            "packed-to-scalar handoff logits",
            &packed_logits,
            &scalar_logits,
            5e-3,
        );
        assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
        assert_eq!(packed.committed_length(), tokens.len() + 1);
        assert_eq!(packed.qsa_committed_lengths(), vec![(3, tokens.len() + 1)]);
    }

    #[test]
    fn packed_text_session_composes_selected_qsa_and_returns_to_scalar() {
        const SELECTED_CAPACITY: usize = 16;

        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new_with_qsa_kv_heads(&ctx, 1);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, SELECTED_CAPACITY)
                .unwrap();
        let mut scalar =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, geometry.clone()).unwrap();
        let mut packed =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        assert_eq!(packed.packed_prefill_capacity(), Some(8));
        assert!(packed.packed_selected_capable());

        let tokens = [1_u32, 7, 3, 11, 5, 9, 2, 13, 4, 6, 8, 10, 12];
        let persistent_before = packed.persistent_state_tensors();
        let weights = fixture.text_weights(&tail, &geometry);
        let mut start = 0;
        for chunk_tokens in [8, 4] {
            let end = start + chunk_tokens;
            let mut scalar_hidden = Vec::new();
            let mut scalar_logits = Vec::new();
            for (position, &token) in tokens.iter().enumerate().take(end).skip(start) {
                (scalar_hidden, scalar_logits) = run_text_session(
                    &ctx,
                    &fixture,
                    &tail,
                    &geometry,
                    &mut scalar,
                    token,
                    position,
                );
            }

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let pending = encode_qwen4exp_text_packed(
                &ctx,
                &encoder,
                &tokens[start..end],
                start,
                fixture.table.table(),
                &weights,
                &mut packed,
            )
            .unwrap();
            assert_eq!(pending.position(), end - 1);
            drop(pending);
            encoder.end();
            command.commit();
            packed.release_after().unwrap();

            let packed_hidden = read_f32(packed.final_hidden_tensor());
            let packed_logits = packed.logits().unwrap().to_vec();
            assert_close(
                &format!("selected packed boundary {end} hidden"),
                &packed_hidden,
                &scalar_hidden,
                2e-3,
            );
            assert_close(
                &format!("selected packed boundary {end} logits"),
                &packed_logits,
                &scalar_logits,
                5e-3,
            );
            assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
            assert_eq!(packed.committed_length(), end);
            assert_eq!(packed.qsa_committed_lengths(), vec![(3, end)]);
            assert_eq!(packed.ple_prior_tokens(), scalar.ple_prior_tokens());

            let scalar_qsa = scalar.qsa_persistent_state_tensors();
            let packed_qsa = packed.qsa_persistent_state_tensors();
            assert_eq!(packed_qsa.len(), scalar_qsa.len());
            for ((packed_layer, packed_state), (scalar_layer, scalar_state)) in
                packed_qsa.iter().zip(&scalar_qsa)
            {
                assert_eq!(packed_layer, scalar_layer);
                assert_eq!(packed_state.len(), scalar_state.len());
                for (state_index, (packed_tensor, scalar_tensor)) in
                    packed_state.iter().zip(scalar_state).enumerate()
                {
                    assert_eq!(packed_tensor.dtype, scalar_tensor.dtype);
                    assert_eq!(packed_tensor.shape, scalar_tensor.shape);
                    assert_close(
                        &format!("QSA layer {packed_layer} state {state_index} at boundary {end}"),
                        &read_numeric_tensor(packed_tensor),
                        &read_numeric_tensor(scalar_tensor),
                        5e-3,
                    );
                }
            }
            start = end;
        }

        let persistent_after = packed.persistent_state_tensors();
        assert_eq!(persistent_before.len(), persistent_after.len());
        for (before, after) in persistent_before.iter().zip(&persistent_after) {
            assert_eq!(
                Retained::as_ptr(&after.buffer),
                Retained::as_ptr(&before.buffer)
            );
            assert_eq!(after.offset, before.offset);
            assert_eq!(after.shape, before.shape);
            assert_eq!(after.dtype, before.dtype);
            assert_eq!(after.provenance(), before.provenance());
        }

        let continuation = tokens[start];
        let (scalar_hidden, scalar_logits) = run_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut scalar,
            continuation,
            start,
        );
        let (packed_hidden, packed_logits) = run_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut packed,
            continuation,
            start,
        );
        assert_close(
            "selected packed-to-scalar handoff hidden",
            &packed_hidden,
            &scalar_hidden,
            2e-3,
        );
        assert_close(
            "selected packed-to-scalar handoff logits",
            &packed_logits,
            &scalar_logits,
            5e-3,
        );
        assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
        assert_eq!(packed.committed_length(), start + 1);
        assert_eq!(packed.qsa_committed_lengths(), vec![(3, start + 1)]);
    }

    #[test]
    fn selected_packed_session_preflight_and_abandon_preserve_boundaries() {
        const SELECTED_CAPACITY: usize = 16;

        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new_with_qsa_kv_heads(&ctx, 1);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, SELECTED_CAPACITY)
                .unwrap();
        let tokens = [1_u32, 7, 3, 11, 5, 9, 2, 13, 4, 6, 8, 10];
        let weights = fixture.text_weights(&tail, &geometry);

        let mut dense_only = Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_dense_packed(
            &ctx,
            geometry.clone(),
        )
        .unwrap();
        assert!(!dense_only.packed_selected_capable());
        run_packed_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut dense_only,
            &tokens[..8],
            0,
        );
        let dense_logits = dense_only.logits().unwrap().to_vec();
        let dense_history = dense_only.ple_prior_tokens().to_vec();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let error = match encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &tokens[8..],
            8,
            fixture.table.table(),
            &weights,
            &mut dense_only,
        ) {
            Ok(_) => panic!("dense-only scratch unexpectedly admitted selected QSA"),
            Err(error) => error,
        };
        let census = crate::metal::dispatch_census_take();
        encoder.end();
        drop(command);
        assert!(error.to_string().contains("selected-capable scratch"));
        assert!(census.is_empty());
        dense_only.release_after().unwrap();
        assert!(!dense_only.is_poisoned());
        assert_eq!(dense_only.committed_length(), 8);
        assert_eq!(dense_only.qsa_committed_lengths(), vec![(3, 8)]);
        assert_eq!(dense_only.ple_prior_tokens(), dense_history);
        assert_eq!(dense_only.logits().unwrap().as_slice(), dense_logits);

        let mut selected =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        run_packed_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut selected,
            &tokens[..8],
            0,
        );
        let selected_logits = selected.logits().unwrap().to_vec();
        let selected_history = selected.ple_prior_tokens().to_vec();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &tokens[8..],
            8,
            fixture.table.table(),
            &weights,
            &mut selected,
        )
        .unwrap();
        drop(pending);
        encoder.end();
        unsafe { selected.abandon_uncommitted().unwrap() };
        drop(command);
        assert!(!selected.is_poisoned());
        assert_eq!(selected.committed_length(), 8);
        assert_eq!(selected.qsa_committed_lengths(), vec![(3, 8)]);
        assert_eq!(selected.ple_prior_tokens(), selected_history);
        assert_eq!(selected.logits().unwrap().as_slice(), selected_logits);

        run_packed_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut selected,
            &tokens[8..],
            8,
        );
        assert_eq!(selected.committed_length(), tokens.len());
        assert_eq!(selected.qsa_committed_lengths(), vec![(3, tokens.len())]);
    }

    #[test]
    fn shared_selected_controls_reset_for_each_qsa_layer_and_profile_route() {
        const SELECTED_CAPACITY: usize = 16;

        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new_with_two_qsa_layers(&ctx);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, SELECTED_CAPACITY)
                .unwrap();
        assert_eq!(
            geometry
                .post_ple()
                .iter()
                .filter(|block| block.mixer().kind() == MixerKind::QwenSparseAttention)
                .count(),
            2
        );
        let tokens = [1_u32, 7, 3, 11, 5, 9, 2, 13, 4, 6, 8, 10];
        let weights = fixture.text_weights(&tail, &geometry);

        let mut ordinary =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        run_packed_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut ordinary,
            &tokens[..8],
            0,
        );
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &tokens[8..],
            8,
            fixture.table.table(),
            &weights,
            &mut ordinary,
        )
        .unwrap();
        let ordinary_census = crate::metal::dispatch_census_take();
        drop(pending);
        encoder.end();
        command.commit();
        ordinary.release_after().unwrap();
        assert_selected_band_reset_sequences(&ordinary_census, 2);
        assert_eq!(ordinary.qsa_committed_lengths(), vec![(3, 12), (7, 12)]);
        let ordinary_hidden = read_f32(ordinary.final_hidden_tensor());
        let ordinary_logits = ordinary.logits().unwrap().to_vec();

        let mut sampled =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        run_packed_text_session(
            &ctx,
            &fixture,
            &tail,
            &geometry,
            &mut sampled,
            &tokens[..8],
            0,
        );
        let sample_count = packed_stage_sample_count(weights.post_ple.len()).unwrap();
        let samples = ctx.timestamp_sample_buffer(sample_count).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let mut cpu_timing = Qwen4ExpPackedEncodeCpuTiming::default();
        crate::metal::dispatch_census_begin();
        let (pending, spans) = encode_qwen4exp_text_packed_layer_sampled(
            &ctx,
            &command,
            &samples,
            &tokens[8..],
            8,
            fixture.table.table(),
            &weights,
            &mut sampled,
            &mut cpu_timing,
        )
        .unwrap();
        let sampled_census = crate::metal::dispatch_census_take();
        drop(pending);
        assert!(!spans.is_empty());
        command.commit();
        sampled.release_after().unwrap();
        assert_selected_band_reset_sequences(&sampled_census, 2);
        assert_eq!(sampled.qsa_committed_lengths(), vec![(3, 12), (7, 12)]);
        assert_close(
            "ordinary/stage-sampled selected hidden",
            &read_f32(sampled.final_hidden_tensor()),
            &ordinary_hidden,
            1e-6,
        );
        assert_close(
            "ordinary/stage-sampled selected logits",
            &sampled.logits().unwrap().to_vec(),
            &ordinary_logits,
            1e-6,
        );
    }

    #[test]
    fn packed_text_session_matches_scalar_at_qsa_residues_and_capacity() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, CAPACITY).unwrap();
        let stream = [1_u32, 7, 3, 11, 5, 9, 2, 13];

        for (start, tokens) in [(0, 2), (0, 8), (1, 2), (2, 2), (3, 2)] {
            let mut scalar =
                Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, geometry.clone()).unwrap();
            let mut packed = Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(
                &ctx,
                geometry.clone(),
            )
            .unwrap();
            for (position, &token) in stream.iter().take(start).enumerate() {
                run_text_session(
                    &ctx,
                    &fixture,
                    &tail,
                    &geometry,
                    &mut scalar,
                    token,
                    position,
                );
                run_text_session(
                    &ctx,
                    &fixture,
                    &tail,
                    &geometry,
                    &mut packed,
                    token,
                    position,
                );
            }
            let mut scalar_hidden = Vec::new();
            let mut scalar_logits = Vec::new();
            for (position, &token) in stream.iter().enumerate().skip(start).take(tokens) {
                (scalar_hidden, scalar_logits) = run_text_session(
                    &ctx,
                    &fixture,
                    &tail,
                    &geometry,
                    &mut scalar,
                    token,
                    position,
                );
            }

            let weights = fixture.text_weights(&tail, &geometry);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let pending = encode_qwen4exp_text_packed(
                &ctx,
                &encoder,
                &stream[start..start + tokens],
                start,
                fixture.table.table(),
                &weights,
                &mut packed,
            )
            .unwrap();
            drop(pending);
            encoder.end();
            command.commit();
            packed.release_after().unwrap();

            let packed_hidden = read_f32(packed.final_hidden_tensor());
            let packed_logits = packed.logits().unwrap().to_vec();
            assert_close(
                &format!("packed start={start} tokens={tokens} hidden"),
                &packed_hidden,
                &scalar_hidden,
                2e-3,
            );
            assert_close(
                &format!("packed start={start} tokens={tokens} logits"),
                &packed_logits,
                &scalar_logits,
                5e-3,
            );
            assert_eq!(argmax(&packed_logits), argmax(&scalar_logits));
            assert_eq!(packed.committed_length(), start + tokens);
            assert_eq!(packed.qsa_committed_lengths(), vec![(3, start + tokens)]);
            assert_eq!(packed.ple_prior_tokens(), scalar.ple_prior_tokens());
        }
    }

    #[test]
    fn packed_text_session_preflight_and_abandon_are_transactional() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, CAPACITY).unwrap();
        let mut workspace =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests_with_packed(&ctx, geometry.clone())
                .unwrap();
        let writable_injection =
            MetalTensor::zeros_f32(&ctx, vec![HYPER as u64, BRANCHES as u64]).unwrap();
        let mut malformed = fixture.text_weights(&tail, &geometry);
        malformed.post_ple.last_mut().unwrap().ffn_residual.inject = &writable_injection;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        assert!(
            encode_qwen4exp_text_packed(
                &ctx,
                &encoder,
                &[1, 7],
                0,
                fixture.table.table(),
                &malformed,
                &mut workspace,
            )
            .is_err()
        );
        assert!(crate::metal::dispatch_census_take().is_empty());
        encoder.end();
        drop(command);
        assert!(!workspace.is_poisoned());
        assert_eq!(workspace.committed_length(), 0);
        workspace.release_after().unwrap();

        let weights = fixture.text_weights(&tail, &geometry);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &[1, 7],
            0,
            fixture.table.table(),
            &weights,
            &mut workspace,
        )
        .unwrap();
        drop(pending);
        encoder.end();
        unsafe { workspace.abandon_uncommitted().unwrap() };
        drop(command);
        assert!(!workspace.is_poisoned());
        assert_eq!(workspace.committed_length(), 0);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 0)]);
        assert!(workspace.ple_prior_tokens().is_empty());
        assert!(workspace.logits().is_err());

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &[1, 7],
            0,
            fixture.table.table(),
            &weights,
            &mut workspace,
        )
        .unwrap();
        drop(pending);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        assert_eq!(workspace.committed_length(), 2);
        workspace.reset().unwrap();
        assert_eq!(workspace.committed_length(), 0);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 0)]);
        assert!(workspace.ple_prior_tokens().is_empty());

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_packed(
            &ctx,
            &encoder,
            &[1, 7],
            0,
            fixture.table.table(),
            &weights,
            &mut workspace,
        )
        .unwrap();
        drop(pending);
        workspace.mark_pending_encode_failed_for_tests();
        encoder.end();
        command.commit();
        assert!(workspace.release_after().is_err());
        assert!(workspace.is_poisoned());
        assert!(workspace.logits().is_err());
        workspace.reset().unwrap();
        assert_eq!(workspace.committed_length(), 0);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 0)]);
        assert!(workspace.ple_prior_tokens().is_empty());
    }

    #[test]
    fn text_session_lifecycle_is_transactional_resettable_and_capacity_strict() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let tail = TextSessionTailFixture::new(&ctx);
        let geometry =
            Qwen4ExpTextSessionMetalGeometry::from_config(&fixture.config, CAPACITY).unwrap();
        let mut workspace =
            Qwen4ExpTextSessionMetalWorkspace::new_for_tests(&ctx, geometry.clone()).unwrap();
        assert!(workspace.logits().is_err());

        let weights = fixture.text_weights(&tail, &geometry);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_token(
            &ctx,
            &encoder,
            1,
            0,
            fixture.table.table(),
            &weights,
            &mut workspace,
        )
        .unwrap();
        drop(pending);
        let foreign_command = ctx.queue.commandBuffer().unwrap();
        let foreign_encoder = KernelEncoder::begin(&foreign_command);
        assert!(
            encode_qwen4exp_text_token(
                &ctx,
                &foreign_encoder,
                2,
                1,
                fixture.table.table(),
                &weights,
                &mut workspace,
            )
            .is_err()
        );
        foreign_encoder.end();
        assert!(workspace.release_after().is_err());
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        let first_logits = workspace.logits().unwrap().to_vec();
        assert_eq!(workspace.committed_length(), 1);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 1)]);

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let pending = encode_qwen4exp_text_token(
            &ctx,
            &encoder,
            2,
            1,
            fixture.table.table(),
            &weights,
            &mut workspace,
        )
        .unwrap();
        drop(pending);
        encoder.end();
        unsafe { workspace.abandon_uncommitted().unwrap() };
        drop(command);
        assert_eq!(workspace.committed_length(), 1);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 1)]);
        assert_eq!(workspace.logits().unwrap().as_slice(), first_logits);

        workspace.reset().unwrap();
        assert_eq!(workspace.committed_length(), 0);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 0)]);
        assert!(workspace.ple_prior_tokens().is_empty());
        assert!(workspace.logits().is_err());
        let (_, replayed_logits) =
            run_text_session(&ctx, &fixture, &tail, &geometry, &mut workspace, 1, 0);
        assert_eq!(replayed_logits, first_logits);

        let malformed_bytes = quantize_rows(
            &values(TABLE_ROWS * HIDDEN, 991, 0.01, 0.0),
            GgmlType::IQ4_NL,
            HIDDEN,
        );
        let malformed_desc = TensorDesc {
            name: "malformed_text_session_ple".into(),
            shape: vec![HIDDEN as u64, TABLE_ROWS as u64],
            dtype: GgmlType::IQ4_NL,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: malformed_bytes.len() as u64,
        };
        let malformed =
            PleIq4NlTable::new(&malformed_desc, &malformed_bytes, TABLE_ROWS as u64).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_qwen4exp_text_token(&ctx, &encoder, 2, 1, malformed, &weights, &mut workspace,)
                .is_err()
        );
        encoder.end();
        assert!(!workspace.is_poisoned());
        assert_eq!(workspace.committed_length(), 1);
        assert_eq!(workspace.logits().unwrap().as_slice(), first_logits);

        for position in 1..CAPACITY {
            run_text_session(
                &ctx,
                &fixture,
                &tail,
                &geometry,
                &mut workspace,
                (position as u32 * 3 + 1) % VOCAB as u32,
                position,
            );
        }
        assert_eq!(workspace.committed_length(), CAPACITY);
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, CAPACITY)]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let error = match encode_qwen4exp_text_token(
            &ctx,
            &encoder,
            1,
            CAPACITY,
            fixture.table.table(),
            &weights,
            &mut workspace,
        ) {
            Ok(_) => panic!("capacity exhaustion unexpectedly encoded a token"),
            Err(error) => error,
        };
        encoder.end();
        assert!(error.to_string().contains("capacity"));
        assert!(!workspace.is_poisoned());
        workspace.reset().unwrap();
        assert_eq!(workspace.qsa_committed_lengths(), vec![(3, 0)]);
    }

    #[test]
    fn generic_layer_two_matches_independent_cpu_residual_control() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let geometry = fixture.geometry.layer_two();
        let gdn_geometry = match geometry.mixer() {
            crate::qwen4exp_post_ple_block::Qwen4ExpPostPleMixerMetalGeometry::GatedDeltaNet(
                geometry,
            ) => geometry,
            _ => unreachable!(),
        };
        let mut block = Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, geometry).unwrap();
        let mut gdn = GatedDeltaNetMetalWorkspace::new(&ctx, gdn_geometry).unwrap();
        let mut moe = Qwen4ExpMoeMetalWorkspace::new(&ctx, geometry.moe()).unwrap();
        let layer = &fixture.layers[2];
        let mut recurrent_state_observed = false;

        for position in 0..2 {
            let input = values(HYPER, 700 + position, 0.006, 0.03);
            let (attention_input, attention_state) = gated_residual_mix(
                &input,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                layer.attention.cpu_read(),
            )
            .unwrap();
            let expected_mixer = standalone_gdn(
                &ctx,
                &attention_input,
                layer.gdn.weights(gdn_geometry),
                &mut gdn,
            );
            if position == 1 {
                let mut fresh = GatedDeltaNetMetalWorkspace::new(&ctx, gdn_geometry).unwrap();
                let fresh_output = standalone_gdn(
                    &ctx,
                    &attention_input,
                    layer.gdn.weights(gdn_geometry),
                    &mut fresh,
                );
                recurrent_state_observed = max_delta(&expected_mixer, &fresh_output) > 1e-6;
            }
            let after_attention = gated_residual_combine(
                &expected_mixer,
                &attention_state,
                &layer.attention.inject_cpu,
            )
            .unwrap();
            let (ffn_input, ffn_state) = gated_residual_mix(
                &after_attention,
                BRANCHES,
                HIDDEN,
                RANK,
                geometry.eps(),
                layer.ffn.cpu_read(),
            )
            .unwrap();
            let expected_moe = standalone_moe(
                &ctx,
                &ffn_input,
                layer.moe.weights(geometry.moe()),
                &mut moe,
            );
            let expected =
                gated_residual_combine(&expected_moe, &ffn_state, &layer.ffn.inject_cpu).unwrap();

            let actual = tensor_f32(&ctx, &input);
            run_post(
                &ctx,
                position,
                &actual,
                fixture.layer_two_weights(),
                &mut block,
            );
            assert_close(
                "independent layer-two mixer",
                &read_f32(block.mixer_output_tensor()),
                &expected_mixer,
                2e-3,
            );
            assert_close(
                "independent layer-two MoE",
                &read_f32(block.moe_output_tensor()),
                &expected_moe,
                3e-3,
            );
            assert_close(
                "independent layer-two residual",
                &read_f32(&actual),
                &expected,
                4e-3,
            );
            assert!(expected_mixer.iter().any(|value| value.abs() > 1e-6));
            assert!(expected_moe.iter().any(|value| value.abs() > 1e-6));
        }
        assert!(
            recurrent_state_observed,
            "independent control did not observe retained GDN state"
        );
    }

    #[test]
    fn five_tokens_match_separate_commands_across_first_qsa_block() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        assert_eq!(fixture.geometry.branch_count(), BRANCHES);
        assert_eq!(fixture.geometry.hidden_size(), HIDDEN);
        assert_eq!(fixture.geometry.capacity(), CAPACITY);
        assert_eq!(fixture.geometry.zero_one().ple().head_count(), PLE_HEADS);
        assert_eq!(fixture.geometry.zero_one().ple().dilation(), PLE_DILATION);
        assert_eq!(
            fixture
                .geometry
                .layer_three()
                .mixer()
                .qsa()
                .unwrap()
                .compression_ratio(),
            4
        );
        let mut integrated =
            Qwen4ExpLayersZeroThreeMetalWorkspace::new(&ctx, fixture.geometry).unwrap();
        let mut zero_one =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, fixture.geometry.zero_one()).unwrap();
        let mut layer_two =
            Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, fixture.geometry.layer_two()).unwrap();
        let mut layer_three =
            Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, fixture.geometry.layer_three()).unwrap();
        let control = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();
        let tokens = [3, 7, 11, 5, 13];
        let mut retained_gdn = false;

        for (position, token) in tokens.into_iter().enumerate() {
            run_zero_one(&ctx, &fixture, &mut zero_one, token, position, &control);
            let layer_two_input = read_f32(&control);
            run_post(
                &ctx,
                position,
                &control,
                fixture.layer_two_weights(),
                &mut layer_two,
            );
            let expected_l2_mixer = read_f32(layer_two.mixer_output_tensor());
            let expected_l2_moe = read_f32(layer_two.moe_output_tensor());
            run_post(
                &ctx,
                position,
                &control,
                fixture.layer_three_weights(),
                &mut layer_three,
            );
            let expected_l3_mixer = read_f32(layer_three.mixer_output_tensor());
            let expected_l3_moe = read_f32(layer_three.moe_output_tensor());
            let expected = read_f32(&control);

            let actual = run_integrated(&ctx, &fixture, &mut integrated, token, position);
            assert_close(
                "layer-two mixer boundary",
                &read_f32(integrated.layer_two.mixer_output_tensor()),
                &expected_l2_mixer,
                2e-5,
            );
            assert_close(
                "layer-two MoE boundary",
                &read_f32(integrated.layer_two.moe_output_tensor()),
                &expected_l2_moe,
                3e-5,
            );
            assert_close(
                "layer-three mixer boundary",
                &read_f32(integrated.layer_three.mixer_output_tensor()),
                &expected_l3_mixer,
                3e-5,
            );
            assert_close(
                "layer-three MoE boundary",
                &read_f32(integrated.layer_three.moe_output_tensor()),
                &expected_l3_moe,
                4e-5,
            );
            assert_close("four-layer residual", &actual, &expected, 5e-5);
            assert!(actual.iter().any(|value| value.abs() > 1e-6));
            assert_eq!(integrated.committed_length(), position + 1);
            assert_eq!(
                integrated.zero_one.next_position(),
                Some(position as u64 + 1)
            );
            assert_eq!(
                integrated.layer_three.mixer_committed_length(),
                Some(position + 1)
            );
            assert_eq!(layer_three.mixer_committed_length(), Some(position + 1));
            if position == 4 {
                let fresh_input = MetalTensor::from_bytes(
                    &ctx,
                    bytemuck::cast_slice(&layer_two_input),
                    vec![HYPER as u64],
                    GgmlType::F32,
                )
                .unwrap();
                let mut fresh =
                    Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, fixture.geometry.layer_two())
                        .unwrap();
                run_post(
                    &ctx,
                    position,
                    &fresh_input,
                    fixture.layer_two_weights(),
                    &mut fresh,
                );
                retained_gdn =
                    max_delta(&expected_l2_mixer, &read_f32(fresh.mixer_output_tensor())) > 1e-6;
            }
        }
        assert!(
            retained_gdn,
            "layer-two GDN retained state had no measurable effect"
        );
    }

    #[test]
    fn generic_qsa_block_capacity_failure_is_nonmutating_and_resettable() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let geometry = fixture.geometry.layer_three();
        let mut workspace = Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, geometry).unwrap();
        for position in 0..CAPACITY {
            let input = tensor_f32(&ctx, &values(HYPER, 900 + position, 0.004, 0.02));
            run_post(
                &ctx,
                position,
                &input,
                fixture.layer_three_weights(),
                &mut workspace,
            );
        }
        assert_eq!(workspace.mixer_committed_length(), Some(CAPACITY));
        assert!(!workspace.is_poisoned());

        let input = tensor_f32(&ctx, &values(HYPER, 999, 0.004, 0.02));
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let error = encode_qwen4exp_post_ple_block(
            &ctx,
            &encoder,
            CAPACITY,
            &input,
            fixture.layer_three_weights(),
            &mut workspace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("capacity"));
        assert_eq!(workspace.mixer_committed_length(), Some(CAPACITY));
        assert!(!workspace.is_poisoned());
        encoder.end();
        workspace.release_after().unwrap();
        assert_eq!(workspace.mixer_committed_length(), Some(CAPACITY));
        workspace.reset().unwrap();
        assert_eq!(workspace.mixer_committed_length(), Some(0));
        assert!(!workspace.is_poisoned());
    }

    #[test]
    fn outer_lifecycle_is_atomic_resettable_and_capacity_strict() {
        let Some(ctx) = context() else { return };
        let fixture = SyntheticFixture::new(&ctx);
        let mut workspace =
            Qwen4ExpLayersZeroThreeMetalWorkspace::new(&ctx, fixture.geometry).unwrap();
        let output = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();

        let undersized_table =
            PleIq4NlTable::new(&fixture.table.desc, &fixture.table.bytes, 1).unwrap();
        let malformed = ctx.queue.commandBuffer().unwrap();
        let malformed_encoder = KernelEncoder::begin(&malformed);
        assert!(
            encode_qwen4exp_layers_zero_three(
                &ctx,
                &malformed_encoder,
                3,
                0,
                undersized_table,
                fixture.weights(),
                &mut workspace,
            )
            .is_err()
        );
        malformed_encoder.end();
        assert!(workspace.active_command.is_none());
        assert!(workspace.pending_length.is_none());
        assert_eq!(workspace.committed_length(), 0);
        assert!(!workspace.is_poisoned());

        let discontinuous = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&discontinuous);
        assert!(
            encode_qwen4exp_layers_zero_three(
                &ctx,
                &encoder,
                3,
                1,
                fixture.table.table(),
                fixture.weights(),
                &mut workspace
            )
            .err()
            .unwrap()
            .to_string()
            .contains("committed length")
        );
        encoder.end();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_layers_zero_three(
            &ctx,
            &encoder,
            3,
            0,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        assert_eq!(read.workspace.committed_length(), 0);
        assert_eq!(read.workspace.zero_one.next_position(), None);
        assert_eq!(read.workspace.layer_three.mixer_committed_length(), Some(0));
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
            encode_qwen4exp_layers_zero_three(
                &ctx,
                &second_encoder,
                3,
                0,
                fixture.table.table(),
                fixture.weights(),
                &mut workspace
            )
            .is_err()
        );
        second_encoder.end();
        encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        assert_eq!(workspace.committed_length(), 0);
        assert_eq!(workspace.zero_one.next_position(), None);
        assert_eq!(workspace.layer_three.mixer_committed_length(), Some(0));

        let first = run_integrated(&ctx, &fixture, &mut workspace, 3, 0);
        assert_eq!(workspace.committed_length(), 1);
        let abandoned = ctx.queue.commandBuffer().unwrap();
        let abandoned_encoder = KernelEncoder::begin(&abandoned);
        let read = encode_qwen4exp_layers_zero_three(
            &ctx,
            &abandoned_encoder,
            7,
            1,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        drop(read);
        abandoned_encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        assert_eq!(workspace.committed_length(), 1);
        assert_eq!(workspace.zero_one.next_position(), Some(1));
        assert_eq!(workspace.layer_three.mixer_committed_length(), Some(1));
        let committed = ctx.queue.commandBuffer().unwrap();
        let committed_encoder = KernelEncoder::begin(&committed);
        let read = encode_qwen4exp_layers_zero_three(
            &ctx,
            &committed_encoder,
            7,
            1,
            fixture.table.table(),
            fixture.weights(),
            &mut workspace,
        )
        .unwrap();
        drop(read);
        committed_encoder.end();
        committed.commit();
        assert!(unsafe { workspace.abandon_uncommitted() }.is_err());
        assert!(workspace.reset().is_err());
        assert_eq!(workspace.committed_length(), 1);
        workspace.release_after().unwrap();
        assert_eq!(workspace.committed_length(), 2);
        assert_eq!(workspace.zero_one.next_position(), Some(2));
        assert_eq!(workspace.layer_three.mixer_committed_length(), Some(2));

        workspace.reset().unwrap();
        let reset = run_integrated(&ctx, &fixture, &mut workspace, 3, 0);
        assert_close("reset token zero", &reset, &first, 1e-6);
        for (position, token) in [7, 11, 5, 13, 2, 17, 19].into_iter().enumerate() {
            run_integrated(&ctx, &fixture, &mut workspace, token, position + 1);
        }
        assert_eq!(workspace.committed_length(), CAPACITY);
        let before_nested = workspace.zero_one.next_position();
        let before_qsa = workspace.layer_three.mixer_committed_length();
        let exhausted = ctx.queue.commandBuffer().unwrap();
        let exhausted_encoder = KernelEncoder::begin(&exhausted);
        assert!(
            encode_qwen4exp_layers_zero_three(
                &ctx,
                &exhausted_encoder,
                23,
                CAPACITY,
                fixture.table.table(),
                fixture.weights(),
                &mut workspace
            )
            .err()
            .unwrap()
            .to_string()
            .contains("capacity")
        );
        exhausted_encoder.end();
        assert_eq!(workspace.committed_length(), CAPACITY);
        assert_eq!(workspace.zero_one.next_position(), before_nested);
        assert_eq!(workspace.layer_three.mixer_committed_length(), before_qsa);
        assert!(!workspace.is_poisoned());
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

    fn assert_residual_bindings(
        weights: Qwen4ExpResidualMetalWeights<'_>,
        resident: &Qwen4ExpMetalWeights,
        layer: u32,
        role: &str,
    ) {
        for (actual, suffix) in [
            (weights.read.norm, "norm"),
            (weights.read.down, "down"),
            (weights.read.up, "up"),
            (weights.inject, "inject"),
        ] {
            assert_binding(
                actual,
                resident,
                &format!("blk.{layer}.hc_{role}_{suffix}.weight"),
            );
        }
    }

    fn assert_moe_bindings(
        weights: Qwen4ExpMoeMetalWeights<'_>,
        resident: &Qwen4ExpMetalWeights,
        layer: u32,
    ) {
        for (actual, suffix) in [
            (weights.router, "ffn_gate_inp.weight"),
            (weights.routed_gate, "ffn_gate_exps.weight"),
            (weights.routed_up, "ffn_up_exps.weight"),
            (weights.routed_down, "ffn_down_exps.weight"),
            (weights.shared_router, "ffn_gate_inp_shexp.weight"),
            (weights.shared_gate, "ffn_gate_shexp.weight"),
            (weights.shared_up, "ffn_up_shexp.weight"),
            (weights.shared_down, "ffn_down_shexp.weight"),
        ] {
            assert_binding(actual, resident, &format!("blk.{layer}.{suffix}"));
        }
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_LAYERS_ZERO_THREE_GGUF to the pinned full release"]
    fn released_first_cycle_matches_separate_command_compositions() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_LAYERS_ZERO_THREE_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_LAYERS_ZERO_THREE_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let resident = realized.weights();
        let weights = Qwen4ExpLayersZeroThreeMetalWeights::bind(resident, 4).unwrap();
        let table = resident.ple_source().bind(&gguf).unwrap();
        let l2 = weights.layer_two;
        let l3 = weights.layer_three;
        assert_residual_bindings(l2.attention_residual, resident, 2, "attn");
        assert_residual_bindings(l2.ffn_residual, resident, 2, "ffn");
        let Qwen4ExpPostPleMixerMetalWeights::GatedDeltaNet(l2_gdn) = l2.mixer else {
            panic!("layer 2 was not GDN")
        };
        for (actual, suffix) in [
            (l2_gdn.qkv, "attn_qkv.weight"),
            (l2_gdn.gate, "attn_gate.weight"),
            (l2_gdn.beta, "ssm_beta.weight"),
            (l2_gdn.alpha, "ssm_alpha.weight"),
            (l2_gdn.a, "ssm_a"),
            (l2_gdn.dt_bias, "ssm_dt.bias"),
            (l2_gdn.conv, "ssm_conv1d.weight"),
            (l2_gdn.norm, "ssm_norm.weight"),
            (l2_gdn.output, "ssm_out.weight"),
        ] {
            assert_binding(actual, resident, &format!("blk.2.{suffix}"));
        }
        assert_moe_bindings(l2.moe, resident, 2);
        assert_eq!(l2.moe.routed_gate.dtype, GgmlType::IQ4_XS);
        assert_eq!(l2.moe.routed_up.dtype, GgmlType::IQ4_XS);
        assert_eq!(l2.moe.routed_down.dtype, GgmlType::Q8_0);
        let Qwen4ExpPostPleMixerMetalWeights::QwenSparseAttention(qsa) = l3.mixer else {
            panic!("layer 3 was not QSA")
        };
        assert_residual_bindings(l3.attention_residual, resident, 3, "attn");
        assert_residual_bindings(l3.ffn_residual, resident, 3, "ffn");
        for (actual, name) in [
            (qsa.query, "blk.3.attn_q.weight"),
            (qsa.key, "blk.3.attn_k.weight"),
            (qsa.value, "blk.3.attn_v.weight"),
            (qsa.output, "blk.3.attn_output.weight"),
            (qsa.query_norm, "blk.3.attn_q_norm.weight"),
            (qsa.key_norm, "blk.3.attn_k_norm.weight"),
            (qsa.index_query, "blk.3.indexer.q_proj.weight"),
            (qsa.index_key, "blk.3.indexer.k_proj.weight"),
            (qsa.index_query_norm, "blk.3.indexer.q_norm.weight"),
            (qsa.index_key_norm, "blk.3.indexer.k_norm.weight"),
        ] {
            assert_binding(actual, resident, name);
        }
        assert_moe_bindings(l3.moe, resident, 3);
        assert_eq!(qsa.index_query.dtype, GgmlType::BF16);
        assert_eq!(qsa.index_key.dtype, GgmlType::BF16);
        assert_eq!(qsa.query.dtype, GgmlType::Q8_0);
        assert_eq!(qsa.key.dtype, GgmlType::Q8_0);
        assert_eq!(qsa.value.dtype, GgmlType::Q8_0);
        assert_eq!(qsa.output.dtype, GgmlType::Q8_0);

        let geometry = weights.geometry;
        let mut integrated = Qwen4ExpLayersZeroThreeMetalWorkspace::new(&ctx, geometry).unwrap();
        let mut zero_one =
            Qwen4ExpLayersZeroOneMetalWorkspace::new(&ctx, geometry.zero_one()).unwrap();
        let mut layer_two =
            Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, geometry.layer_two()).unwrap();
        let mut layer_three =
            Qwen4ExpPostPleBlockMetalWorkspace::new(&ctx, geometry.layer_three()).unwrap();
        let control = MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let actual = MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        for (position, token) in [35_u32, 201, 17, 89].into_iter().enumerate() {
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_qwen4exp_layers_zero_one(
                &ctx,
                &encoder,
                token,
                position as u64,
                table,
                weights.zero_one,
                &mut zero_one,
            )
            .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &control)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            zero_one.release_after().unwrap();
            run_post(&ctx, position, &control, weights.layer_two, &mut layer_two);
            run_post(
                &ctx,
                position,
                &control,
                weights.layer_three,
                &mut layer_three,
            );

            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_qwen4exp_layers_zero_three(
                &ctx,
                &encoder,
                token,
                position,
                table,
                weights,
                &mut integrated,
            )
            .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &actual)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            integrated.release_after().unwrap();
            assert_close(
                "released layer-two mixer",
                &read_f32(integrated.layer_two.mixer_output_tensor()),
                &read_f32(layer_two.mixer_output_tensor()),
                3e-5,
            );
            assert_close(
                "released layer-two MoE",
                &read_f32(integrated.layer_two.moe_output_tensor()),
                &read_f32(layer_two.moe_output_tensor()),
                4e-5,
            );
            assert_close(
                "released layer-three mixer",
                &read_f32(integrated.layer_three.mixer_output_tensor()),
                &read_f32(layer_three.mixer_output_tensor()),
                4e-5,
            );
            assert_close(
                "released layer-three MoE",
                &read_f32(integrated.layer_three.moe_output_tensor()),
                &read_f32(layer_three.moe_output_tensor()),
                5e-5,
            );
            assert_close(
                "released first-cycle residual",
                &read_f32(&actual),
                &read_f32(&control),
                6e-5,
            );
        }
        assert_eq!(integrated.committed_length(), 4);
        assert_eq!(layer_three.mixer_committed_length(), Some(4));
    }
}
