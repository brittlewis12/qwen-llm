//! Metal primitives for Qwen3.8-Flash-Next gated residuals.

use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::metal_forward::{
    MfError, encode_mat_mat_dispatch, encode_mat_vec_dispatch, validate_f32_q8_mat_mat_addressing,
};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource, MTLSize,
};

const SIMD_WIDTH: usize = 32;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Qwen4ExpHcPackedProjectionArm {
    WideF32Down,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Qwen4ExpHcPackedProjectionRecord {
    pub arm: Qwen4ExpHcPackedProjectionArm,
    pub start_position: usize,
    pub tokens: usize,
    pub n_in: usize,
    pub n_out: usize,
}

#[cfg(test)]
#[derive(Clone)]
struct Qwen4ExpHcPackedProjectionBinding {
    arm: Qwen4ExpHcPackedProjectionArm,
    start_position: usize,
    tokens: usize,
    records: std::rc::Rc<std::cell::RefCell<Vec<Qwen4ExpHcPackedProjectionRecord>>>,
}

#[cfg(test)]
thread_local! {
    static QWEN4EXP_HC_PACKED_PROJECTION_OVERRIDE: std::cell::RefCell<Option<Qwen4ExpHcPackedProjectionBinding>> = const {
        std::cell::RefCell::new(None)
    };
}

#[cfg(test)]
pub(crate) fn with_qwen4exp_hc_packed_projection_override<R>(
    arm: Qwen4ExpHcPackedProjectionArm,
    start_position: usize,
    tokens: usize,
    f: impl FnOnce() -> R,
) -> (R, Vec<Qwen4ExpHcPackedProjectionRecord>) {
    assert!(tokens > 0);
    start_position
        .checked_add(tokens)
        .expect("HC projection override range overflow");

    struct RestoreOverride(Option<Qwen4ExpHcPackedProjectionBinding>);

    impl Drop for RestoreOverride {
        fn drop(&mut self) {
            QWEN4EXP_HC_PACKED_PROJECTION_OVERRIDE.with(|slot| {
                *slot.borrow_mut() = self.0.take();
            });
        }
    }

    QWEN4EXP_HC_PACKED_PROJECTION_OVERRIDE.with(|slot| {
        assert!(
            slot.borrow().is_none(),
            "HC projection overrides cannot nest"
        );
    });
    let records = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
    let binding = Qwen4ExpHcPackedProjectionBinding {
        arm,
        start_position,
        tokens,
        records: records.clone(),
    };
    let previous =
        QWEN4EXP_HC_PACKED_PROJECTION_OVERRIDE.with(|slot| slot.borrow_mut().replace(binding));
    debug_assert!(previous.is_none());
    let _restore = RestoreOverride(previous);
    let result = f();
    let records = records.borrow().clone();
    (result, records)
}

#[cfg(test)]
pub(crate) fn qwen4exp_hc_packed_projection_override_active() -> bool {
    QWEN4EXP_HC_PACKED_PROJECTION_OVERRIDE.with(|slot| slot.borrow().is_some())
}

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpMetalError {
    #[error("invalid gated-residual contract: {0}")]
    InvalidContract(String),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error("gated-residual command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy)]
pub struct GatedResidualMetalReadWeights<'a> {
    pub norm: &'a MetalTensor,
    pub down: &'a MetalTensor,
    pub up: &'a MetalTensor,
}

pub struct GatedResidualMetalScratch {
    branch_count: usize,
    hidden_size: usize,
    low_rank: usize,
    normalized: MetalTensor,
    low: MetalTensor,
    raw_gate: MetalTensor,
    mixed: MetalTensor,
    injection: MetalTensor,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

impl GatedResidualMetalScratch {
    pub fn new(
        ctx: &MetalContext,
        branch_count: usize,
        hidden_size: usize,
        low_rank: usize,
    ) -> Result<Self, Qwen4ExpMetalError> {
        let hyper_hidden = validate_geometry(branch_count, hidden_size, low_rank)?;
        Ok(Self {
            branch_count,
            hidden_size,
            low_rank,
            normalized: MetalTensor::zeros_f32(ctx, vec![hyper_hidden as u64])?,
            low: MetalTensor::zeros_f32(ctx, vec![low_rank as u64])?,
            raw_gate: MetalTensor::zeros_f32(ctx, vec![hyper_hidden as u64])?,
            mixed: MetalTensor::zeros_f32(ctx, vec![hidden_size as u64])?,
            injection: MetalTensor::zeros_f32(ctx, vec![branch_count as u64])?,
            active_command: None,
        })
    }

    pub fn branch_count(&self) -> usize {
        self.branch_count
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn low_rank(&self) -> usize {
        self.low_rank
    }

    pub(crate) fn mixed_tensor(&self) -> &MetalTensor {
        &self.mixed
    }

    /// Wait for the owning command buffer and permit reuse from another one.
    /// The command must already be ended and committed by the caller.
    pub fn release_after(&mut self) -> Result<(), Qwen4ExpMetalError> {
        let Some(command) = self.active_command.clone() else {
            return Ok(());
        };

        let status = command.status();
        if matches!(
            status,
            MTLCommandBufferStatus::NotEnqueued | MTLCommandBufferStatus::Enqueued
        ) {
            return Err(invalid(format!(
                "scratch owner is not committed (status {status:?}); commit it or abandon the uncommitted command"
            )));
        }

        command.waitUntilCompleted();
        let status = command.status();
        let error = command.error().map(|error| error.to_string());
        self.active_command = None;
        if status == MTLCommandBufferStatus::Completed && error.is_none() {
            Ok(())
        } else {
            Err(Qwen4ExpMetalError::CommandBuffer(format!(
                "status={status:?}, error={error:?}"
            )))
        }
    }

    /// Release scratch from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command buffer. Committing it after this call may race a later
    /// scratch owner.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpMetalError> {
        let Some(command) = self.active_command.as_ref() else {
            return Ok(());
        };
        let status = command.status();
        if status != MTLCommandBufferStatus::NotEnqueued {
            return Err(invalid(format!(
                "only a NotEnqueued scratch owner can be abandoned, got {status:?}"
            )));
        }
        self.active_command = None;
        Ok(())
    }
}

pub(crate) struct GatedResidualPackedScratch {
    branch_count: usize,
    hidden_size: usize,
    low_rank: usize,
    capacity: usize,
    normalized: MetalTensor,
    low: MetalTensor,
    raw_gate: MetalTensor,
    mixed: MetalTensor,
    injection: MetalTensor,
}

impl GatedResidualPackedScratch {
    pub(crate) fn new(
        ctx: &MetalContext,
        branch_count: usize,
        hidden_size: usize,
        low_rank: usize,
        capacity: usize,
    ) -> Result<Self, Qwen4ExpMetalError> {
        let hyper_hidden = validate_geometry(branch_count, hidden_size, low_rank)?;
        if capacity == 0 || u32::try_from(capacity).is_err() {
            return Err(invalid(format!(
                "packed HC capacity must be in 1..=u32::MAX, got {capacity}"
            )));
        }
        for (name, n_in, n_out) in [
            ("down", hyper_hidden, low_rank),
            ("up", low_rank, hyper_hidden),
            ("injection", hyper_hidden, branch_count),
        ] {
            let elements = n_in
                .checked_mul(n_out)
                .ok_or_else(|| invalid(format!("packed HC {name} projection overflow")))?;
            if u32::try_from(elements).is_err() {
                return Err(invalid(format!(
                    "packed HC {name} projection has {elements} elements, exceeding u32 shader addressing"
                )));
            }
        }
        for (name, width) in [("hyper", hyper_hidden), ("low-rank", low_rank)] {
            let q8_row_bytes = width
                .checked_div(32)
                .and_then(|blocks| blocks.checked_mul(34))
                .ok_or_else(|| invalid(format!("packed HC {name} Q8 row-byte overflow")))?;
            if u32::try_from(q8_row_bytes).is_err() {
                return Err(invalid(format!(
                    "packed HC {name} Q8 row-byte stride {q8_row_bytes} exceeds u32 shader addressing"
                )));
            }
        }
        for (name, width) in [
            ("normalized", hyper_hidden),
            ("low-rank", low_rank),
            ("raw gate", hyper_hidden),
            ("mixed", hidden_size),
            ("injection", branch_count),
        ] {
            let elements = width
                .checked_mul(capacity)
                .ok_or_else(|| invalid(format!("packed HC {name} element count overflow")))?;
            if u32::try_from(elements).is_err() {
                return Err(invalid(format!(
                    "packed HC {name} element count {elements} exceeds u32"
                )));
            }
            elements
                .checked_mul(size_of::<f32>())
                .ok_or_else(|| invalid(format!("packed HC {name} byte count overflow")))?;
        }
        Ok(Self {
            branch_count,
            hidden_size,
            low_rank,
            capacity,
            normalized: MetalTensor::zeros_f32(ctx, vec![hyper_hidden as u64, capacity as u64])?,
            low: MetalTensor::zeros_f32(ctx, vec![low_rank as u64, capacity as u64])?,
            raw_gate: MetalTensor::zeros_f32(ctx, vec![hyper_hidden as u64, capacity as u64])?,
            mixed: MetalTensor::zeros_f32(ctx, vec![hidden_size as u64, capacity as u64])?,
            injection: MetalTensor::zeros_f32(ctx, vec![branch_count as u64, capacity as u64])?,
        })
    }

    fn prefix_view(
        &self,
        name: &str,
        tensor: &MetalTensor,
        width: usize,
        tokens: usize,
    ) -> Result<MetalTensor, Qwen4ExpMetalError> {
        if tokens == 0 || tokens > self.capacity {
            return Err(invalid(format!(
                "{name} token count {tokens} is outside capacity {}",
                self.capacity
            )));
        }
        let elements = width
            .checked_mul(tokens)
            .ok_or_else(|| invalid(format!("{name} element count overflow")))?;
        let view = tensor.view_subrange(0, vec![width as u64, tokens as u64]);
        if view.n_elements() as usize != elements {
            return Err(invalid(format!(
                "{name} prefix view has the wrong element count"
            )));
        }
        Ok(view)
    }

    pub(crate) fn mixed_view(&self, tokens: usize) -> Result<MetalTensor, Qwen4ExpMetalError> {
        self.prefix_view(
            "packed HC mixed output",
            &self.mixed,
            self.hidden_size,
            tokens,
        )
    }
}

#[must_use = "encode the residual block, then call encode_combine"]
pub struct GatedResidualMetalRead<'scratch, 'resources, 'pass> {
    ctx: &'pass MetalContext,
    encoder: &'pass KernelEncoder,
    hyper_input: &'resources MetalTensor,
    block_output: &'resources MetalTensor,
    inject: &'resources MetalTensor,
    scratch: &'scratch mut GatedResidualMetalScratch,
}

impl GatedResidualMetalRead<'_, '_, '_> {
    pub fn mixed(&self) -> &MetalTensor {
        &self.scratch.mixed
    }

    pub fn encode_combine(self) -> Result<(), Qwen4ExpMetalError> {
        let hyper_hidden = self.scratch.branch_count * self.scratch.hidden_size;
        encode_mat_vec_dispatch(
            self.ctx,
            self.encoder,
            self.inject,
            &self.scratch.normalized,
            &self.scratch.injection,
            hyper_hidden,
            self.scratch.branch_count,
        )?;
        encode_hc_injection(
            self.ctx,
            self.encoder,
            self.block_output,
            &self.scratch.injection,
            self.hyper_input,
            self.scratch.branch_count,
            self.scratch.hidden_size,
        )?;
        Ok(())
    }
}

#[must_use = "encode the packed residual block, then call encode_combine"]
pub(crate) struct GatedResidualPackedRead<'scratch, 'resources, 'pass> {
    ctx: &'pass MetalContext,
    command: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    hyper_input: &'resources MetalTensor,
    block_output: &'resources MetalTensor,
    inject: &'resources MetalTensor,
    scratch: &'scratch mut GatedResidualPackedScratch,
    normalized: MetalTensor,
    mixed: MetalTensor,
    tokens: usize,
}

impl GatedResidualPackedRead<'_, '_, '_> {
    pub(crate) fn mixed(&self) -> &MetalTensor {
        &self.mixed
    }

    pub(crate) fn encode_combine(self, enc: &KernelEncoder) -> Result<(), Qwen4ExpMetalError> {
        validate_encoder(self.ctx, enc)?;
        let command = enc.parent_command_buffer();
        if !std::ptr::addr_eq(Retained::as_ptr(&command), Retained::as_ptr(&self.command)) {
            return Err(invalid(
                "packed HC combine encoder belongs to a different command",
            ));
        }
        let hyper_hidden = self.scratch.branch_count * self.scratch.hidden_size;
        let injection = self.scratch.prefix_view(
            "packed HC injection",
            &self.scratch.injection,
            self.scratch.branch_count,
            self.tokens,
        )?;
        encode_mat_mat_dispatch(
            self.ctx,
            enc,
            self.inject,
            &self.normalized,
            &injection,
            hyper_hidden,
            self.scratch.branch_count,
            self.tokens,
        )?;
        encode_hc_injection_packed(
            self.ctx,
            enc,
            self.block_output,
            &injection,
            self.hyper_input,
            self.scratch.branch_count,
            self.scratch.hidden_size,
            self.tokens,
        )?;
        Ok(())
    }
}

#[must_use = "consume the final mixed activation before releasing scratch"]
pub struct GatedResidualMetalFinalRead<'scratch> {
    scratch: &'scratch mut GatedResidualMetalScratch,
}

impl GatedResidualMetalFinalRead<'_> {
    pub fn mixed(&self) -> &MetalTensor {
        &self.scratch.mixed
    }
}

pub fn encode_gated_residual_mix<'scratch, 'resources, 'pass>(
    ctx: &'pass MetalContext,
    enc: &'pass KernelEncoder,
    hyper_input: &'resources MetalTensor,
    block_output: &'resources MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    inject: &'resources MetalTensor,
    scratch: &'scratch mut GatedResidualMetalScratch,
) -> Result<GatedResidualMetalRead<'scratch, 'resources, 'pass>, Qwen4ExpMetalError> {
    validate_encoder(ctx, enc)?;
    validate_and_preflight_gated_residual_mix(
        ctx,
        hyper_input,
        block_output,
        eps,
        weights,
        inject,
        scratch,
    )?;
    reserve_command(scratch, enc)?;
    let hyper_hidden = scratch.branch_count * scratch.hidden_size;
    encode_read(ctx, enc, hyper_input, eps, weights, scratch, hyper_hidden)?;
    Ok(GatedResidualMetalRead {
        ctx,
        encoder: enc,
        hyper_input,
        block_output,
        inject,
        scratch,
    })
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Qwen4ExpHcPackedProjectionRole {
    Down,
    Up,
}

#[cfg(test)]
fn active_hc_packed_projection_binding() -> Option<Qwen4ExpHcPackedProjectionBinding> {
    let range = crate::qwen4exp_composition_trace::qwen4exp_diagnostic_execution_range()?;
    QWEN4EXP_HC_PACKED_PROJECTION_OVERRIDE.with(|slot| {
        slot.borrow()
            .clone()
            .filter(|binding| range == (binding.start_position, binding.tokens))
    })
}

#[cfg(test)]
fn preflight_hc_packed_projection_override(
    ctx: &MetalContext,
    weights: GatedResidualMetalReadWeights<'_>,
    hyper_hidden: usize,
    low_rank: usize,
    tokens: usize,
) -> Result<(), Qwen4ExpMetalError> {
    if active_hc_packed_projection_binding().is_none() {
        return Ok(());
    }
    let projections = [(weights.down, hyper_hidden, low_rank)];
    for &(weight, n_in, n_out) in &projections {
        if weight.dtype != GgmlType::Q8_0
            || !n_in.is_multiple_of(64)
            || !n_out.is_multiple_of(16)
            || !tokens.is_multiple_of(128)
        {
            return Err(invalid(format!(
                "HC F32 projection override rejects [{tokens},{n_in}] -> [{tokens},{n_out}] {:?}",
                weight.dtype
            )));
        }
    }
    let pipeline = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c16k64")?;
    if pipeline.threadExecutionWidth() != SIMD_WIDTH
        || pipeline.maxTotalThreadsPerThreadgroup() < 128
        || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return Err(invalid(
            "HC F32 projection override requires four SIMDgroups and 4 KiB threadgroup memory",
        ));
    }
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_q8_f32_mma_r2c16k64(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    tokens: usize,
) -> Result<(), Qwen4ExpMetalError> {
    if weight.dtype != GgmlType::Q8_0
        || input.dtype != GgmlType::F32
        || output.dtype != GgmlType::F32
        || weight.n_elements() as usize != n_in * n_out
        || input.n_elements() as usize != n_in * tokens
        || output.n_elements() as usize != n_out * tokens
        || !n_in.is_multiple_of(64)
        || !n_out.is_multiple_of(16)
        || !tokens.is_multiple_of(128)
    {
        return Err(invalid(format!(
            "HC Q8 F32 R2C16K64 requires aligned [{tokens},{n_in}] -> [{tokens},{n_out}]"
        )));
    }
    let pipeline = ctx.pipeline("kernel_mat_mat_q8_0_f32_r2c16k64")?;
    if pipeline.threadExecutionWidth() != SIMD_WIDTH
        || pipeline.maxTotalThreadsPerThreadgroup() < 128
        || ctx.device.maxThreadgroupMemoryLength() < 4_096
    {
        return Err(invalid(
            "HC Q8 F32 R2C16K64 requires four SIMDgroups and 4 KiB threadgroup memory",
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        m: u32,
        n: u32,
        k: u32,
        nb01: u32,
        stride_b: u32,
    }
    let row_bytes = n_in
        .checked_div(32)
        .and_then(|blocks| blocks.checked_mul(34))
        .ok_or_else(|| invalid("HC Q8 F32 R2C16K64 row-byte overflow"))?;
    let args = Args {
        m: u32::try_from(n_out).map_err(|_| invalid("HC F32 output exceeds u32"))?,
        n: u32::try_from(tokens).map_err(|_| invalid("HC F32 token count exceeds u32"))?,
        k: u32::try_from(n_in).map_err(|_| invalid("HC F32 input exceeds u32"))?,
        nb01: u32::try_from(row_bytes).map_err(|_| invalid("HC F32 row bytes exceed u32"))?,
        stride_b: u32::try_from(n_in).map_err(|_| invalid("HC F32 stride exceeds u32"))?,
    };
    enc.set_pipeline(&pipeline);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, weight);
    enc.set_tensor(2, input);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, 4_096);
    enc.dispatch(
        MTLSize {
            width: tokens.div_ceil(128),
            height: n_out / 16,
            depth: 1,
        },
        MTLSize {
            width: 128,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn encode_hc_packed_projection_override(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    role: Qwen4ExpHcPackedProjectionRole,
    weight: &MetalTensor,
    input: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    tokens: usize,
) -> Result<bool, Qwen4ExpMetalError> {
    let Some(binding) = active_hc_packed_projection_binding() else {
        return Ok(false);
    };
    let enabled = matches!(
        (binding.arm, role),
        (
            Qwen4ExpHcPackedProjectionArm::WideF32Down,
            Qwen4ExpHcPackedProjectionRole::Down
        )
    );
    if !enabled {
        return Ok(false);
    }
    if tokens != binding.tokens {
        return Err(invalid(format!(
            "HC projection override expected {} rows, got {tokens}",
            binding.tokens
        )));
    }
    let _tag =
        crate::metal::dispatch_census_tag_scope(|| "qwen4exp.hc_precision.wide_f32_down".into());
    encode_q8_f32_mma_r2c16k64(ctx, enc, weight, input, output, n_in, n_out, tokens)?;
    binding
        .records
        .borrow_mut()
        .push(Qwen4ExpHcPackedProjectionRecord {
            arm: binding.arm,
            start_position: binding.start_position,
            tokens,
            n_in,
            n_out,
        });
    Ok(true)
}

/// Encode a packed HC read into transaction-owned scratch.
///
/// # Safety
///
/// The caller must hold every tensor and exclusive logical ownership of
/// `hyper_input` and `scratch` until the command completes successfully or is
/// permanently abandoned. If `block_output` is GPU-produced, it must be
/// encoded on the same command after `mixed()` is consumed and before
/// `encode_combine()`. No other command may access these resources, and
/// abandonment requires permanently discarding every command reference. Any
/// encode or command failure makes mutable contents indeterminate; the caller
/// must poison the enclosing transaction rather than expose or reuse them.
pub(crate) unsafe fn encode_gated_residual_packed_mix<'scratch, 'resources, 'ctx>(
    ctx: &'ctx MetalContext,
    enc: &KernelEncoder,
    hyper_input: &'resources MetalTensor,
    block_output: &'resources MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    inject: &'resources MetalTensor,
    scratch: &'scratch mut GatedResidualPackedScratch,
    tokens: usize,
) -> Result<GatedResidualPackedRead<'scratch, 'resources, 'ctx>, Qwen4ExpMetalError> {
    validate_encoder(ctx, enc)?;
    validate_and_preflight_gated_residual_packed_mix(
        ctx,
        hyper_input,
        block_output,
        eps,
        weights,
        inject,
        scratch,
        tokens,
    )?;

    let hyper_hidden = scratch.branch_count * scratch.hidden_size;
    let normalized = scratch.prefix_view(
        "packed HC normalized scratch",
        &scratch.normalized,
        hyper_hidden,
        tokens,
    )?;
    let low = scratch.prefix_view(
        "packed HC low-rank scratch",
        &scratch.low,
        scratch.low_rank,
        tokens,
    )?;
    let raw_gate = scratch.prefix_view(
        "packed HC raw gate scratch",
        &scratch.raw_gate,
        hyper_hidden,
        tokens,
    )?;
    let mixed = scratch.prefix_view(
        "packed HC mixed output",
        &scratch.mixed,
        scratch.hidden_size,
        tokens,
    )?;

    encode_hc_norm_packed(
        ctx,
        enc,
        hyper_input,
        weights.norm,
        &normalized,
        scratch.branch_count,
        scratch.hidden_size,
        tokens,
        eps,
    )?;
    #[cfg(test)]
    let down_overridden = encode_hc_packed_projection_override(
        ctx,
        enc,
        Qwen4ExpHcPackedProjectionRole::Down,
        weights.down,
        &normalized,
        &low,
        hyper_hidden,
        scratch.low_rank,
        tokens,
    )?;
    #[cfg(not(test))]
    let down_overridden = false;
    if !down_overridden {
        encode_mat_mat_dispatch(
            ctx,
            enc,
            weights.down,
            &normalized,
            &low,
            hyper_hidden,
            scratch.low_rank,
            tokens,
        )?;
    }
    encode_hc_low_activation(ctx, enc, &low, scratch.branch_count)?;
    #[cfg(test)]
    let up_overridden = encode_hc_packed_projection_override(
        ctx,
        enc,
        Qwen4ExpHcPackedProjectionRole::Up,
        weights.up,
        &low,
        &raw_gate,
        scratch.low_rank,
        hyper_hidden,
        tokens,
    )?;
    #[cfg(not(test))]
    let up_overridden = false;
    if !up_overridden {
        encode_mat_mat_dispatch(
            ctx,
            enc,
            weights.up,
            &low,
            &raw_gate,
            scratch.low_rank,
            hyper_hidden,
            tokens,
        )?;
    }
    encode_hc_gated_mean_packed(
        ctx,
        enc,
        &normalized,
        &raw_gate,
        &mixed,
        scratch.branch_count,
        scratch.hidden_size,
        tokens,
    )?;
    Ok(GatedResidualPackedRead {
        ctx,
        command: enc.parent_command_buffer(),
        hyper_input,
        block_output,
        inject,
        scratch,
        normalized,
        mixed,
        tokens,
    })
}

pub(crate) fn validate_and_preflight_gated_residual_mix(
    ctx: &MetalContext,
    hyper_input: &MetalTensor,
    block_output: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    inject: &MetalTensor,
    scratch: &GatedResidualMetalScratch,
) -> Result<(), Qwen4ExpMetalError> {
    let hyper_hidden = validate_read_contract(hyper_input, eps, weights, scratch, true)?;
    validate_combine_contract(hyper_input, block_output, inject, scratch, hyper_hidden)?;
    require_disjoint(&[
        ("hyper input", hyper_input),
        ("block output", block_output),
        ("HC norm", weights.norm),
        ("HC down", weights.down),
        ("HC up", weights.up),
        ("HC injection", inject),
        ("normalized scratch", &scratch.normalized),
        ("low-rank scratch", &scratch.low),
        ("raw gate scratch", &scratch.raw_gate),
        ("mixed output", &scratch.mixed),
        ("injection scratch", &scratch.injection),
    ])?;
    require_same_device(
        ctx,
        &[
            ("hyper input", hyper_input),
            ("block output", block_output),
            ("HC norm", weights.norm),
            ("HC down", weights.down),
            ("HC up", weights.up),
            ("HC injection", inject),
            ("normalized scratch", &scratch.normalized),
            ("low-rank scratch", &scratch.low),
            ("raw gate scratch", &scratch.raw_gate),
            ("mixed output", &scratch.mixed),
            ("injection scratch", &scratch.injection),
        ],
    )?;
    preflight_mix(ctx, weights.down.dtype, weights.up.dtype)?;
    preflight_combine(ctx)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_and_preflight_gated_residual_packed_mix(
    ctx: &MetalContext,
    hyper_input: &MetalTensor,
    block_output: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    inject: &MetalTensor,
    scratch: &GatedResidualPackedScratch,
    tokens: usize,
) -> Result<(), Qwen4ExpMetalError> {
    if tokens == 0 || tokens > scratch.capacity {
        return Err(invalid(format!(
            "packed HC token count {tokens} is outside capacity {}",
            scratch.capacity
        )));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("RMS epsilon must be finite and positive"));
    }
    let hyper_hidden =
        validate_geometry(scratch.branch_count, scratch.hidden_size, scratch.low_rank)?;
    require_f32_shape(
        "packed HC hyper input",
        hyper_input,
        &[hyper_hidden as u64, tokens as u64],
        true,
    )?;
    require_f32_shape(
        "packed HC block output",
        block_output,
        &[scratch.hidden_size as u64, tokens as u64],
        false,
    )?;
    require_f32_shape(
        "packed HC norm",
        weights.norm,
        &[hyper_hidden as u64],
        false,
    )?;
    require_projection(
        "packed HC down",
        weights.down,
        hyper_hidden,
        scratch.low_rank,
    )?;
    require_projection("packed HC up", weights.up, scratch.low_rank, hyper_hidden)?;
    require_projection(
        "packed HC injection",
        inject,
        hyper_hidden,
        scratch.branch_count,
    )?;
    if inject.dtype != GgmlType::F32 {
        return Err(invalid(format!(
            "packed HC injection must use F32 storage, got {:?}",
            inject.dtype
        )));
    }
    validate_f32_q8_mat_mat_addressing(weights.down.dtype, hyper_hidden, scratch.low_rank, tokens)?;
    validate_f32_q8_mat_mat_addressing(weights.up.dtype, scratch.low_rank, hyper_hidden, tokens)?;
    validate_f32_q8_mat_mat_addressing(inject.dtype, hyper_hidden, scratch.branch_count, tokens)?;
    #[cfg(test)]
    preflight_hc_packed_projection_override(ctx, weights, hyper_hidden, scratch.low_rank, tokens)?;
    for (name, tensor, width) in [
        (
            "packed HC normalized scratch",
            &scratch.normalized,
            hyper_hidden,
        ),
        ("packed HC low-rank scratch", &scratch.low, scratch.low_rank),
        (
            "packed HC raw gate scratch",
            &scratch.raw_gate,
            hyper_hidden,
        ),
        (
            "packed HC mixed output",
            &scratch.mixed,
            scratch.hidden_size,
        ),
        (
            "packed HC injection scratch",
            &scratch.injection,
            scratch.branch_count,
        ),
    ] {
        require_f32_shape(name, tensor, &[width as u64, scratch.capacity as u64], true)?;
    }
    let tensors = [
        ("packed HC hyper input", hyper_input),
        ("packed HC block output", block_output),
        ("packed HC norm", weights.norm),
        ("packed HC down", weights.down),
        ("packed HC up", weights.up),
        ("packed HC injection", inject),
        ("packed HC normalized scratch", &scratch.normalized),
        ("packed HC low-rank scratch", &scratch.low),
        ("packed HC raw gate scratch", &scratch.raw_gate),
        ("packed HC mixed output", &scratch.mixed),
        ("packed HC injection scratch", &scratch.injection),
    ];
    require_disjoint(&tensors)?;
    require_same_device(ctx, &tensors)?;
    preflight_packed_mix(ctx, weights.down.dtype, weights.up.dtype)?;
    preflight_packed_combine(ctx)
}

pub fn encode_final_gated_residual_mix<'scratch>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_input: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    scratch: &'scratch mut GatedResidualMetalScratch,
) -> Result<GatedResidualMetalFinalRead<'scratch>, Qwen4ExpMetalError> {
    validate_encoder(ctx, enc)?;
    validate_and_preflight_final_gated_residual_mix(ctx, hyper_input, eps, weights, scratch)?;
    let hyper_hidden = scratch.branch_count * scratch.hidden_size;
    reserve_command(scratch, enc)?;
    encode_read(ctx, enc, hyper_input, eps, weights, scratch, hyper_hidden)?;
    Ok(GatedResidualMetalFinalRead { scratch })
}

pub(crate) fn validate_and_preflight_final_gated_residual_mix(
    ctx: &MetalContext,
    hyper_input: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    scratch: &GatedResidualMetalScratch,
) -> Result<(), Qwen4ExpMetalError> {
    validate_read_contract(hyper_input, eps, weights, scratch, false)?;
    require_disjoint(&[
        ("hyper input", hyper_input),
        ("HC norm", weights.norm),
        ("HC down", weights.down),
        ("HC up", weights.up),
        ("normalized scratch", &scratch.normalized),
        ("low-rank scratch", &scratch.low),
        ("raw gate scratch", &scratch.raw_gate),
        ("mixed output", &scratch.mixed),
    ])?;
    require_same_device(
        ctx,
        &[
            ("hyper input", hyper_input),
            ("HC norm", weights.norm),
            ("HC down", weights.down),
            ("HC up", weights.up),
            ("normalized scratch", &scratch.normalized),
            ("low-rank scratch", &scratch.low),
            ("raw gate scratch", &scratch.raw_gate),
            ("mixed output", &scratch.mixed),
        ],
    )?;
    preflight_mix(ctx, weights.down.dtype, weights.up.dtype)
}

fn validate_read_contract(
    hyper_input: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    scratch: &GatedResidualMetalScratch,
    writable_input: bool,
) -> Result<usize, Qwen4ExpMetalError> {
    if !eps.is_finite() || eps <= 0.0 {
        return Err(invalid("RMS epsilon must be finite and positive"));
    }
    let hyper_hidden =
        validate_geometry(scratch.branch_count, scratch.hidden_size, scratch.low_rank)?;
    require_f32("hyper input", hyper_input, hyper_hidden, writable_input)?;
    require_f32("HC norm", weights.norm, hyper_hidden, false)?;
    require_projection("HC down", weights.down, hyper_hidden, scratch.low_rank)?;
    require_projection("HC up", weights.up, scratch.low_rank, hyper_hidden)?;
    require_f32(
        "normalized scratch",
        &scratch.normalized,
        hyper_hidden,
        true,
    )?;
    require_f32("low-rank scratch", &scratch.low, scratch.low_rank, true)?;
    require_f32("raw gate scratch", &scratch.raw_gate, hyper_hidden, true)?;
    require_f32("mixed output", &scratch.mixed, scratch.hidden_size, true)?;
    Ok(hyper_hidden)
}

fn validate_combine_contract(
    hyper_input: &MetalTensor,
    block_output: &MetalTensor,
    inject: &MetalTensor,
    scratch: &GatedResidualMetalScratch,
    hyper_hidden: usize,
) -> Result<(), Qwen4ExpMetalError> {
    require_f32("hyper input", hyper_input, hyper_hidden, true)?;
    require_f32("block output", block_output, scratch.hidden_size, false)?;
    require_projection("HC injection", inject, hyper_hidden, scratch.branch_count)?;
    if inject.dtype != GgmlType::F32 {
        return Err(invalid(format!(
            "HC injection must use F32 storage, got {:?}",
            inject.dtype
        )));
    }
    require_f32(
        "injection scratch",
        &scratch.injection,
        scratch.branch_count,
        true,
    )
}

fn encode_read(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_input: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    scratch: &GatedResidualMetalScratch,
    hyper_hidden: usize,
) -> Result<(), Qwen4ExpMetalError> {
    encode_hc_norm(
        ctx,
        enc,
        hyper_input,
        weights.norm,
        &scratch.normalized,
        scratch.branch_count,
        scratch.hidden_size,
        eps,
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.down,
        &scratch.normalized,
        &scratch.low,
        hyper_hidden,
        scratch.low_rank,
    )?;
    encode_hc_low_activation(ctx, enc, &scratch.low, scratch.branch_count)?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.up,
        &scratch.low,
        &scratch.raw_gate,
        scratch.low_rank,
        hyper_hidden,
    )?;
    encode_hc_gated_mean(
        ctx,
        enc,
        &scratch.normalized,
        &scratch.raw_gate,
        &scratch.mixed,
        scratch.branch_count,
        scratch.hidden_size,
    )?;
    Ok(())
}

fn validate_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), Qwen4ExpMetalError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return Err(invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        )));
    }
    if enc.is_concurrent() {
        return Err(invalid(
            "gated-residual dependent dispatches require a serial encoder",
        ));
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return Err(invalid(format!(
            "gated-residual encoding requires a NotEnqueued command buffer, got {status:?}"
        )));
    }
    Ok(())
}

fn reserve_command(
    scratch: &mut GatedResidualMetalScratch,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpMetalError> {
    let command = enc.parent_command_buffer();
    match scratch.active_command.as_ref() {
        None => {
            scratch.active_command = Some(command);
            Ok(())
        }
        Some(owner) if std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) => {
            Ok(())
        }
        Some(_) => Err(invalid(
            "scratch is still owned by another command buffer; call release_after once it completes",
        )),
    }
}

fn preflight_mix(
    ctx: &MetalContext,
    down_dtype: GgmlType,
    up_dtype: GgmlType,
) -> Result<(), Qwen4ExpMetalError> {
    ctx.pipeline("kernel_qwen4exp_hc_rms_norm_f32")?;
    ctx.pipeline("kernel_qwen4exp_hc_low_silu_f32")?;
    ctx.pipeline("kernel_qwen4exp_hc_gated_mean_f32")?;
    preflight_projection(ctx, down_dtype)?;
    preflight_projection(ctx, up_dtype)
}

fn preflight_combine(ctx: &MetalContext) -> Result<(), Qwen4ExpMetalError> {
    preflight_projection(ctx, GgmlType::F32)?;
    ctx.pipeline("kernel_qwen4exp_hc_inject_f32")?;
    Ok(())
}

fn preflight_packed_mix(
    ctx: &MetalContext,
    down_dtype: GgmlType,
    up_dtype: GgmlType,
) -> Result<(), Qwen4ExpMetalError> {
    let norm = ctx.pipeline("kernel_qwen4exp_ple_grouped_rms_norm_packed_f32")?;
    if norm.threadExecutionWidth() != SIMD_WIDTH
        || norm.maxTotalThreadsPerThreadgroup() < SIMD_WIDTH
    {
        return Err(invalid(format!(
            "packed HC norm needs execution width {SIMD_WIDTH}, got width={} capacity={}",
            norm.threadExecutionWidth(),
            norm.maxTotalThreadsPerThreadgroup()
        )));
    }
    ctx.pipeline("kernel_qwen4exp_hc_low_silu_f32")?;
    ctx.pipeline("kernel_qwen4exp_hc_gated_mean_packed_f32")?;
    preflight_packed_projection(ctx, down_dtype)?;
    preflight_packed_projection(ctx, up_dtype)
}

fn preflight_packed_combine(ctx: &MetalContext) -> Result<(), Qwen4ExpMetalError> {
    preflight_packed_projection(ctx, GgmlType::F32)?;
    ctx.pipeline("kernel_qwen4exp_hc_inject_packed_f32")?;
    Ok(())
}

/// Kernel names every Flash-Next projection dtype must be able to build.
/// Returns `None` for unsupported dtypes so callers can report their own
/// family-specific error.
pub(crate) fn projection_kernel_names(
    dtype: GgmlType,
    packed: bool,
    allow_bf16: bool,
) -> Option<&'static [&'static str]> {
    Some(match (dtype, packed) {
        (GgmlType::F32, false) => &["kernel_mat_vec_f32_f32", "kernel_mat_vec_f32_f32_lcpp_r2"],
        (GgmlType::Q8_0, false) => &["kernel_mat_vec_q8_0_f32", "kernel_mat_vec_q8_0_f32_lcpp"],
        (GgmlType::BF16, false) if allow_bf16 => &["kernel_mat_vec_bf16_f32"],
        (GgmlType::F32, true) => &["kernel_mat_mat_f32_f32"],
        (GgmlType::Q8_0, true) => &[
            "kernel_mat_mat_q8_0_f32",
            "kernel_mat_mat_q8_0_f32_n16",
            "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
        ],
        _ => return None,
    })
}

/// Build the pipelines one projection shape (scalar or packed) needs.
/// `Ok(false)` means the dtype is unsupported; pipeline failures surface as
/// `MetalError`. Callers that require both shapes call this twice.
pub(crate) fn preflight_projection_pipelines(
    ctx: &MetalContext,
    dtype: GgmlType,
    packed: bool,
    allow_bf16: bool,
) -> Result<bool, MetalError> {
    let Some(kernels) = projection_kernel_names(dtype, packed, allow_bf16) else {
        return Ok(false);
    };
    for kernel in kernels {
        ctx.pipeline(kernel)?;
    }
    Ok(true)
}

fn preflight_packed_projection(
    ctx: &MetalContext,
    dtype: GgmlType,
) -> Result<(), Qwen4ExpMetalError> {
    preflight_projection(ctx, dtype)?;
    if !preflight_projection_pipelines(ctx, dtype, true, false)? {
        return Err(invalid(format!(
            "unsupported packed gated-residual projection dtype {dtype:?}"
        )));
    }
    Ok(())
}

fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpMetalError> {
    if !preflight_projection_pipelines(ctx, dtype, false, false)? {
        return Err(invalid(format!(
            "unsupported gated-residual projection dtype {dtype:?}"
        )));
    }
    Ok(())
}

fn validate_geometry(
    branch_count: usize,
    hidden_size: usize,
    low_rank: usize,
) -> Result<usize, Qwen4ExpMetalError> {
    if branch_count == 0 || hidden_size == 0 || low_rank == 0 {
        return Err(invalid(
            "branch count, hidden size, and low rank must be nonzero",
        ));
    }
    let hyper_hidden = branch_count
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid("hyper-connection width overflow"))?;
    for (name, value) in [
        ("branch count", branch_count),
        ("hidden size", hidden_size),
        ("low rank", low_rank),
        ("hyper-connection width", hyper_hidden),
    ] {
        if u32::try_from(value).is_err() {
            return Err(invalid(format!("{name} {value} exceeds u32")));
        }
    }
    Ok(hyper_hidden)
}

fn require_f32(
    name: &str,
    tensor: &MetalTensor,
    expected_elements: usize,
    writable: bool,
) -> Result<(), Qwen4ExpMetalError> {
    if tensor.dtype != GgmlType::F32 {
        return Err(invalid(format!(
            "{name} must use F32 storage, got {:?}",
            tensor.dtype
        )));
    }
    let elements = checked_elements(&tensor.shape)?;
    if elements != expected_elements {
        return Err(invalid(format!(
            "{name} has {elements} elements, expected {expected_elements}"
        )));
    }
    if writable && !tensor.is_writable() {
        return Err(invalid(format!("{name} must be writable")));
    }
    require_physical_range(name, tensor, expected_elements, 4, 4)
}

fn require_f32_shape(
    name: &str,
    tensor: &MetalTensor,
    expected_shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpMetalError> {
    if tensor.dtype != GgmlType::F32 || tensor.shape != expected_shape {
        return Err(invalid(format!(
            "{name} must be F32 with shape {expected_shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        )));
    }
    let elements = checked_elements(expected_shape)?;
    if writable && !tensor.is_writable() {
        return Err(invalid(format!("{name} must be writable")));
    }
    require_physical_range(name, tensor, elements, 4, 4)
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), Qwen4ExpMetalError> {
    let expected_shape = [n_in as u64, n_out as u64];
    if tensor.shape != expected_shape {
        return Err(invalid(format!(
            "{name} shape {:?} does not match {:?}",
            tensor.shape, expected_shape
        )));
    }
    let elements = n_in
        .checked_mul(n_out)
        .ok_or_else(|| invalid(format!("{name} element count overflow")))?;
    match tensor.dtype {
        GgmlType::F32 => require_physical_range(name, tensor, elements, 4, 4),
        GgmlType::Q8_0 => {
            if !n_in.is_multiple_of(32) {
                return Err(invalid(format!(
                    "{name} input width {n_in} is not Q8_0 block aligned"
                )));
            }
            require_physical_range(name, tensor, elements / 32, 34, 2)
        }
        dtype => Err(invalid(format!(
            "{name} must use F32 or Q8_0 storage, got {dtype:?}"
        ))),
    }
}

fn checked_elements(shape: &[u64]) -> Result<usize, Qwen4ExpMetalError> {
    let elements = shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension));
    elements
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid("tensor element count overflows host addressing"))
}

fn require_physical_range(
    name: &str,
    tensor: &MetalTensor,
    units: usize,
    bytes_per_unit: usize,
    alignment: u64,
) -> Result<(), Qwen4ExpMetalError> {
    let bytes = units
        .checked_mul(bytes_per_unit)
        .ok_or_else(|| invalid(format!("{name} byte length overflow")))?;
    if !tensor.offset.is_multiple_of(alignment) {
        return Err(invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        )));
    }
    let end = tensor
        .offset
        .checked_add(bytes as u64)
        .ok_or_else(|| invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return Err(invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        )));
    }
    Ok(())
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpMetalError> {
    let elements = checked_elements(&tensor.shape)?;
    let bytes = match tensor.dtype {
        GgmlType::F32 => elements.checked_mul(4),
        GgmlType::Q8_0 if elements.is_multiple_of(32) => elements
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(34)),
        dtype => {
            return Err(invalid(format!(
                "cannot derive gated-residual storage bytes for {dtype:?}"
            )));
        }
    }
    .ok_or_else(|| invalid("tensor storage byte length overflow"))?;
    u64::try_from(bytes).map_err(|_| invalid("tensor storage byte length exceeds u64"))
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpMetalError> {
    for left in 0..tensors.len() {
        let left_bytes = storage_bytes(tensors[left].1)?;
        for right in left + 1..tensors.len() {
            if tensor_ranges_overlap(
                tensors[left].1,
                left_bytes,
                tensors[right].1,
                storage_bytes(tensors[right].1)?,
            ) {
                return Err(invalid(format!(
                    "{} overlaps {}",
                    tensors[left].0, tensors[right].0
                )));
            }
        }
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpMetalError> {
    let expected = ctx.device.registryID();
    for (name, tensor) in tensors {
        let actual = tensor.buffer.device().registryID();
        if actual != expected {
            return Err(invalid(format!(
                "{name} belongs to Metal device registry {actual}, expected {expected}"
            )));
        }
    }
    Ok(())
}

fn tensor_ranges_overlap(
    left: &MetalTensor,
    left_bytes: u64,
    right: &MetalTensor,
    right_bytes: u64,
) -> bool {
    if Retained::as_ptr(&left.buffer) != Retained::as_ptr(&right.buffer) {
        return false;
    }
    let left_end = left.offset.saturating_add(left_bytes);
    let right_end = right.offset.saturating_add(right_bytes);
    left.offset < right_end && right.offset < left_end
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcNormArgs {
    branch_count: u32,
    hidden_size: u32,
    eps: f32,
}

fn encode_hc_norm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weight: &MetalTensor,
    normalized: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_rms_norm_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &HcNormArgs {
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
            eps,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, normalized);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    let simdgroups = threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (simdgroups * size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: branch_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedHcNormArgs {
    tokens: u32,
    branch_count: u32,
    hidden_size: u32,
    eps: f32,
}

#[allow(clippy::too_many_arguments)]
fn encode_hc_norm_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weight: &MetalTensor,
    normalized: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
    tokens: usize,
    eps: f32,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_ple_grouped_rms_norm_packed_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PackedHcNormArgs {
            tokens: tokens as u32,
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
            eps,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, normalized);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    let simdgroups = threads.div_ceil(32);
    enc.set_threadgroup_memory(0, (simdgroups * size_of::<f32>()).max(32));
    enc.dispatch(
        MTLSize {
            width: branch_count * tokens,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcLowArgs {
    count: u32,
    inverse_branches: f32,
}

fn encode_hc_low_activation(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    low: &MetalTensor,
    branch_count: usize,
) -> Result<(), MetalError> {
    let count = low.n_elements() as usize;
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_low_silu_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &HcLowArgs {
            count: count as u32,
            inverse_branches: 1.0 / branch_count as f32,
        },
    );
    enc.set_tensor(1, low);
    dispatch_1d(enc, &pipeline, count);
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct HcBranchArgs {
    branch_count: u32,
    hidden_size: u32,
}

fn encode_hc_gated_mean(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    normalized: &MetalTensor,
    raw_gate: &MetalTensor,
    mixed: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_gated_mean_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &HcBranchArgs {
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
        },
    );
    enc.set_tensor(1, normalized);
    enc.set_tensor(2, raw_gate);
    enc.set_tensor(3, mixed);
    dispatch_1d(enc, &pipeline, hidden_size);
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PackedHcBranchArgs {
    tokens: u32,
    branch_count: u32,
    hidden_size: u32,
}

pub(crate) fn encode_hc_repeat_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    embedding: &MetalTensor,
    hyper_residual: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
    tokens: usize,
) -> Result<(), Qwen4ExpMetalError> {
    validate_encoder(ctx, enc)?;
    let count = validate_and_preflight_hc_repeat_packed(
        ctx,
        embedding,
        hyper_residual,
        branch_count,
        hidden_size,
        tokens,
    )?;
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_repeat_packed_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PackedHcBranchArgs {
            tokens: tokens as u32,
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
        },
    );
    enc.set_tensor(1, embedding);
    enc.set_tensor(2, hyper_residual);
    dispatch_1d(enc, &pipeline, count);
    Ok(())
}

pub(crate) fn validate_and_preflight_hc_repeat_packed(
    ctx: &MetalContext,
    embedding: &MetalTensor,
    hyper_residual: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
    tokens: usize,
) -> Result<usize, Qwen4ExpMetalError> {
    if branch_count == 0 || hidden_size == 0 || tokens == 0 {
        return Err(invalid(
            "packed HC repeat branch count, hidden size, and token count must be nonzero",
        ));
    }
    for (name, value) in [
        ("branch count", branch_count),
        ("hidden size", hidden_size),
        ("token count", tokens),
    ] {
        if u32::try_from(value).is_err() {
            return Err(invalid(format!(
                "packed HC repeat {name} {value} exceeds u32"
            )));
        }
    }
    let hyper_hidden = branch_count
        .checked_mul(hidden_size)
        .ok_or_else(|| invalid("packed HC repeat hyper width overflow"))?;
    let count = hyper_hidden
        .checked_mul(tokens)
        .ok_or_else(|| invalid("packed HC repeat element count overflow"))?;
    if u32::try_from(count).is_err() {
        return Err(invalid(format!(
            "packed HC repeat element count {count} exceeds u32 shader addressing"
        )));
    }
    require_f32_shape(
        "packed HC repeat embedding",
        embedding,
        &[hidden_size as u64, tokens as u64],
        false,
    )?;
    require_f32_shape(
        "packed HC repeat residual",
        hyper_residual,
        &[hyper_hidden as u64, tokens as u64],
        true,
    )?;
    let tensors = [
        ("packed HC repeat embedding", embedding),
        ("packed HC repeat residual", hyper_residual),
    ];
    require_disjoint(&tensors)?;
    require_same_device(ctx, &tensors)?;
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_repeat_packed_f32")?;
    if pipeline.maxTotalThreadsPerThreadgroup() == 0 {
        return Err(invalid(
            "packed HC repeat pipeline reports zero threadgroup capacity",
        ));
    }
    Ok(count)
}

#[allow(clippy::too_many_arguments)]
fn encode_hc_gated_mean_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    normalized: &MetalTensor,
    raw_gate: &MetalTensor,
    mixed: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
    tokens: usize,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_gated_mean_packed_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PackedHcBranchArgs {
            tokens: tokens as u32,
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
        },
    );
    enc.set_tensor(1, normalized);
    enc.set_tensor(2, raw_gate);
    enc.set_tensor(3, mixed);
    dispatch_1d(enc, &pipeline, hidden_size * tokens);
    Ok(())
}

fn encode_hc_injection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    block_output: &MetalTensor,
    raw_injection: &MetalTensor,
    residual: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_inject_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &HcBranchArgs {
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
        },
    );
    enc.set_tensor(1, block_output);
    enc.set_tensor(2, raw_injection);
    enc.set_tensor(3, residual);
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: hidden_size.div_ceil(threads),
            height: branch_count,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn encode_hc_injection_packed(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    block_output: &MetalTensor,
    raw_injection: &MetalTensor,
    residual: &MetalTensor,
    branch_count: usize,
    hidden_size: usize,
    tokens: usize,
) -> Result<(), MetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_inject_packed_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PackedHcBranchArgs {
            tokens: tokens as u32,
            branch_count: branch_count as u32,
            hidden_size: hidden_size as u32,
        },
    );
    enc.set_tensor(1, block_output);
    enc.set_tensor(2, raw_injection);
    enc.set_tensor(3, residual);
    dispatch_1d(enc, &pipeline, branch_count * hidden_size * tokens);
    Ok(())
}

fn dispatch_1d(
    enc: &KernelEncoder,
    pipeline: &objc2::runtime::ProtocolObject<dyn objc2_metal::MTLComputePipelineState>,
    count: usize,
) {
    let threads = pipeline.maxTotalThreadsPerThreadgroup().min(1024);
    enc.dispatch(
        MTLSize {
            width: count.div_ceil(threads),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads,
            height: 1,
            depth: 1,
        },
    );
}

fn invalid(detail: impl Into<String>) -> Qwen4ExpMetalError {
    Qwen4ExpMetalError::InvalidContract(detail.into())
}

#[cfg(test)]
mod tests {

    #[test]
    fn projection_kernel_names_pin_each_caller_list() {
        use super::projection_kernel_names as names;
        use crate::tensor::GgmlType;
        assert_eq!(
            names(GgmlType::F32, false, false),
            Some(&["kernel_mat_vec_f32_f32", "kernel_mat_vec_f32_f32_lcpp_r2"][..])
        );
        assert_eq!(
            names(GgmlType::Q8_0, false, false),
            Some(&["kernel_mat_vec_q8_0_f32", "kernel_mat_vec_q8_0_f32_lcpp"][..])
        );
        assert_eq!(names(GgmlType::BF16, false, false), None);
        assert_eq!(
            names(GgmlType::BF16, false, true),
            Some(&["kernel_mat_vec_bf16_f32"][..])
        );
        assert_eq!(
            names(GgmlType::F32, true, false),
            Some(&["kernel_mat_mat_f32_f32"][..])
        );
        assert_eq!(
            names(GgmlType::Q8_0, true, false),
            Some(
                &[
                    "kernel_mat_mat_q8_0_f32",
                    "kernel_mat_mat_q8_0_f32_n16",
                    "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
                ][..]
            )
        );
        assert_eq!(names(GgmlType::BF16, true, true), None);
        assert_eq!(names(GgmlType::Q4_K, false, true), None);
    }
    use super::*;
    use crate::metal::MetalTensorProvenance;
    use crate::qwen4exp_forward::{
        GatedResidualReadWeights, gated_residual_combine, gated_residual_mix,
    };
    use objc2_metal::MTLCommandQueue;

    fn packed_test_context() -> Option<MetalContext> {
        crate::test_fixtures::metal_context_or_skip()
    }

    #[test]
    fn hc_projection_override_rejects_nesting_without_losing_outer_binding() {
        assert!(!qwen4exp_hc_packed_projection_override_active());
        let outer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_qwen4exp_hc_packed_projection_override(
                Qwen4ExpHcPackedProjectionArm::WideF32Down,
                2_051,
                2_048,
                || {
                    let nested = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        with_qwen4exp_hc_packed_projection_override(
                            Qwen4ExpHcPackedProjectionArm::WideF32Down,
                            2_051,
                            2_048,
                            || (),
                        );
                    }));
                    assert!(nested.is_err());
                    assert!(qwen4exp_hc_packed_projection_override_active());
                    panic!("exercise outer HC override restoration");
                },
            );
        }));
        assert!(outer.is_err());
        assert!(!qwen4exp_hc_packed_projection_override_active());
    }

    fn tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
    }

    fn offset_tensor(
        ctx: &MetalContext,
        prefix: usize,
        data: &[u8],
        suffix: usize,
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> MetalTensor {
        let mut bytes = vec![0xa5; prefix];
        bytes.extend_from_slice(data);
        bytes.resize(bytes.len() + suffix, 0x5a);
        MetalTensor {
            buffer: ctx.buffer_from(&bytes).unwrap(),
            offset: prefix as u64,
            shape,
            dtype,
            provenance: MetalTensorProvenance::OwnedWritable,
        }
    }

    fn assert_guards(tensor: &MetalTensor, prefix: usize, suffix: usize) {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                tensor.buffer.contents().as_ptr().cast::<u8>(),
                tensor.buffer.length(),
            )
        };
        assert!(bytes[..prefix].iter().all(|&byte| byte == 0xa5));
        assert!(
            bytes[bytes.len() - suffix..]
                .iter()
                .all(|&byte| byte == 0x5a)
        );
    }

    fn q8_bank(n_in: usize, n_out: usize, seed: usize) -> Vec<u8> {
        assert!(n_in.is_multiple_of(32));
        let mut bytes = Vec::with_capacity(n_in / 32 * n_out * 34);
        for row in 0..n_out {
            for block in 0..n_in / 32 {
                let ordinal = row * (n_in / 32) + block + seed;
                let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                let scale = sign * (ordinal % 7 + 1) as f32 / 4096.0;
                bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                for lane in 0..32 {
                    let quant = ((ordinal * 11 + lane * 7 + 3) % 31) as i8 - 15;
                    bytes.push(quant as u8);
                }
            }
        }
        bytes
    }

    fn dequant_q8(bytes: &[u8], n_in: usize, n_out: usize) -> Vec<f32> {
        let desc = crate::tensor::TensorDesc {
            name: "synthetic_hc_q8".into(),
            shape: vec![n_in as u64, n_out as u64],
            dtype: GgmlType::Q8_0,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: bytes.len() as u64,
        };
        crate::codec::dequant_to_f32(&desc, bytes).unwrap()
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
        let count = tensor.n_elements() as usize;
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>();
            std::slice::from_raw_parts(source, count).to_vec()
        }
    }

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}"
            );
        }
    }

    fn assert_bits_eq(label: &str, actual: &[f32], expected: &[f32]) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "{label}[{index}]: expected {expected}, got {actual}"
            );
        }
    }

    fn assert_similarity(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        maximum_relative_rms: f64,
        minimum_cosine: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(actual.iter().all(|value| value.is_finite()), "{label}");
        let dot = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| *actual as f64 * *expected as f64)
            .sum::<f64>();
        let actual_square = actual
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let expected_square = expected
            .iter()
            .map(|value| (*value as f64).powi(2))
            .sum::<f64>();
        let difference_square = actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (*actual as f64 - *expected as f64).powi(2))
            .sum::<f64>();
        let relative_rms = (difference_square / expected_square.max(1e-30)).sqrt();
        let cosine = dot / (actual_square * expected_square).sqrt().max(1e-30);
        eprintln!("[{label}] relative_rms={relative_rms:.3e} cosine={cosine:.9}");
        assert!(
            relative_rms <= maximum_relative_rms,
            "{label} relative_rms={relative_rms}"
        );
        assert!(cosine >= minimum_cosine, "{label} cosine={cosine}");
    }

    #[allow(clippy::too_many_arguments)]
    fn assert_tokenwise_similarity(
        label: &str,
        actual: &[f32],
        expected: &[f32],
        width: usize,
        tokens: usize,
        maximum_relative_rms: f64,
        minimum_cosine: f64,
        maximum_absolute: f32,
    ) {
        assert_eq!(actual.len(), width * tokens, "{label} actual shape");
        assert_eq!(expected.len(), width * tokens, "{label} expected shape");
        for token in 0..tokens {
            let start = token * width;
            let actual_row = &actual[start..start + width];
            let expected_row = &expected[start..start + width];
            assert_similarity(
                &format!("{label} token={token}"),
                actual_row,
                expected_row,
                maximum_relative_rms,
                minimum_cosine,
            );
            let observed_max = actual_row
                .iter()
                .zip(expected_row)
                .map(|(actual, expected)| (actual - expected).abs())
                .fold(0.0_f32, f32::max);
            assert!(
                observed_max <= maximum_absolute,
                "{label} token={token} max_abs={observed_max}"
            );
        }
    }

    fn assert_only_token_changed(
        label: &str,
        baseline: &[f32],
        perturbed: &[f32],
        width: usize,
        tokens: usize,
        changed_token: usize,
    ) {
        assert_eq!(baseline.len(), width * tokens, "{label} baseline shape");
        assert_eq!(perturbed.len(), width * tokens, "{label} perturbed shape");
        for token in 0..tokens {
            let start = token * width;
            let baseline_row = &baseline[start..start + width];
            let perturbed_row = &perturbed[start..start + width];
            if token == changed_token {
                assert!(
                    baseline_row
                        .iter()
                        .zip(perturbed_row)
                        .any(|(left, right)| left.to_bits() != right.to_bits()),
                    "{label} changed token was unaffected"
                );
            } else {
                assert_bits_eq(
                    &format!("{label} isolated token={token}"),
                    perturbed_row,
                    baseline_row,
                );
            }
        }
    }

    struct SerialPackedHcTrace {
        normalized: Vec<f32>,
        low: Vec<f32>,
        raw_gate: Vec<f32>,
        mixed: Vec<f32>,
        injection: Vec<f32>,
        residual: Vec<f32>,
    }

    #[allow(clippy::too_many_arguments)]
    fn serial_hc_trace(
        ctx: &MetalContext,
        hyper_rows: &[f32],
        block_rows: &[f32],
        tokens: usize,
        branch_count: usize,
        hidden_size: usize,
        low_rank: usize,
        weights: GatedResidualMetalReadWeights<'_>,
        inject: &MetalTensor,
    ) -> SerialPackedHcTrace {
        let hyper_hidden = branch_count * hidden_size;
        let mut trace = SerialPackedHcTrace {
            normalized: Vec::with_capacity(tokens * hyper_hidden),
            low: Vec::with_capacity(tokens * low_rank),
            raw_gate: Vec::with_capacity(tokens * hyper_hidden),
            mixed: Vec::with_capacity(tokens * hidden_size),
            injection: Vec::with_capacity(tokens * branch_count),
            residual: Vec::with_capacity(tokens * hyper_hidden),
        };
        let mut scratch =
            GatedResidualMetalScratch::new(ctx, branch_count, hidden_size, low_rank).unwrap();
        for token in 0..tokens {
            let hyper_start = token * hyper_hidden;
            let block_start = token * hidden_size;
            let hyper = tensor(
                ctx,
                &hyper_rows[hyper_start..hyper_start + hyper_hidden],
                vec![hyper_hidden as u64],
            );
            let block = tensor(
                ctx,
                &block_rows[block_start..block_start + hidden_size],
                vec![hidden_size as u64],
            );
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            encode_gated_residual_mix(
                ctx,
                &encoder,
                &hyper,
                &block,
                1e-6,
                weights,
                inject,
                &mut scratch,
            )
            .unwrap()
            .encode_combine()
            .unwrap();
            encoder.end();
            command.commit();
            scratch.release_after().unwrap();
            trace.normalized.extend(read_f32(&scratch.normalized));
            trace.low.extend(read_f32(&scratch.low));
            trace.raw_gate.extend(read_f32(&scratch.raw_gate));
            trace.mixed.extend(read_f32(&scratch.mixed));
            trace.injection.extend(read_f32(&scratch.injection));
            trace.residual.extend(read_f32(&hyper));
        }
        trace
    }

    #[test]
    fn gated_residual_metal_matches_nonzero_cpu_golden() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        let input = [1.0, -2.0, 0.5, 3.0];
        let norm = [1.1, 0.9, 1.2, 0.8];
        let down = [
            0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 1.1, -1.2,
        ];
        let up = [
            0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8, -0.9, -1.0, 1.1, 1.2,
        ];
        let inject = [0.1, 0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
        let block_output = [0.25, -0.75];
        let (expected_mixed, state) = gated_residual_mix(
            &input,
            2,
            2,
            3,
            1e-6,
            GatedResidualReadWeights {
                norm: &norm,
                down: &down,
                up: &up,
            },
        )
        .unwrap();
        let expected_output = gated_residual_combine(&block_output, &state, &inject).unwrap();

        let input_gpu = tensor(&ctx, &input, vec![4]);
        let norm_gpu = tensor(&ctx, &norm, vec![4]);
        let down_gpu = tensor(&ctx, &down, vec![4, 3]);
        let up_gpu = tensor(&ctx, &up, vec![3, 4]);
        let inject_gpu = tensor(&ctx, &inject, vec![4, 2]);
        let block_gpu = tensor(&ctx, &block_output, vec![2]);
        let mut scratch = GatedResidualMetalScratch::new(&ctx, 2, 2, 3).unwrap();
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        let read = encode_gated_residual_mix(
            &ctx,
            &encoder,
            &input_gpu,
            &block_gpu,
            1e-6,
            GatedResidualMetalReadWeights {
                norm: &norm_gpu,
                down: &down_gpu,
                up: &up_gpu,
            },
            &inject_gpu,
            &mut scratch,
        )
        .unwrap();
        assert_eq!(read.mixed().n_elements(), 2);
        read.encode_combine().unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );
        scratch.release_after().unwrap();

        assert_close(&read_f32(&scratch.mixed), &expected_mixed, 3e-5);
        assert_close(&read_f32(&input_gpu), &expected_output, 3e-5);

        let final_input = tensor(&ctx, &input, vec![4]);
        let mut final_scratch = GatedResidualMetalScratch::new(&ctx, 2, 2, 3).unwrap();
        let final_command = ctx.queue.commandBuffer().expect("final command buffer");
        let final_encoder = KernelEncoder::begin(&final_command);
        let final_read = encode_final_gated_residual_mix(
            &ctx,
            &final_encoder,
            &final_input,
            1e-6,
            GatedResidualMetalReadWeights {
                norm: &norm_gpu,
                down: &down_gpu,
                up: &up_gpu,
            },
            &mut final_scratch,
        )
        .unwrap();
        final_encoder.end();
        final_command.commit();
        final_command.waitUntilCompleted();
        assert!(
            final_command.error().is_none(),
            "final command failed: {:?}",
            final_command.error()
        );
        let final_mixed = read_f32(final_read.mixed());
        drop(final_read);
        final_scratch.release_after().unwrap();
        assert_close(&final_mixed, &expected_mixed, 3e-5);
    }

    #[test]
    fn nonzero_q8_projections_and_offset_views_match_cpu() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        const BRANCHES: usize = 4;
        const HIDDEN: usize = 32;
        const RANK: usize = 32;
        const HYPER: usize = BRANCHES * HIDDEN;
        let input = (0..HYPER)
            .map(|index| ((index * 13 + 5) % 61) as f32 * 0.004 - 0.12)
            .collect::<Vec<_>>();
        let norm = (0..HYPER)
            .map(|index| 0.8 + (index % 9) as f32 * 0.05)
            .collect::<Vec<_>>();
        let inject = (0..HYPER * BRANCHES)
            .map(|index| ((index * 5 + 1) % 17) as f32 * 0.002 - 0.016)
            .collect::<Vec<_>>();
        let block_output = (0..HIDDEN)
            .map(|index| ((index * 7 + 2) % 23) as f32 * 0.006 - 0.066)
            .collect::<Vec<_>>();
        let down_bytes = q8_bank(HYPER, RANK, 17);
        let up_bytes = q8_bank(RANK, HYPER, 10_003);
        let down = dequant_q8(&down_bytes, HYPER, RANK);
        let up = dequant_q8(&up_bytes, RANK, HYPER);
        let (expected_mixed, state) = gated_residual_mix(
            &input,
            BRANCHES,
            HIDDEN,
            RANK,
            1e-6,
            GatedResidualReadWeights {
                norm: &norm,
                down: &down,
                up: &up,
            },
        )
        .unwrap();
        let expected_output = gated_residual_combine(&block_output, &state, &inject).unwrap();

        let input_gpu = offset_tensor(
            &ctx,
            16,
            bytemuck::cast_slice(&input),
            20,
            vec![HYPER as u64],
            GgmlType::F32,
        );
        let norm_gpu = offset_tensor(
            &ctx,
            20,
            bytemuck::cast_slice(&norm),
            16,
            vec![HYPER as u64],
            GgmlType::F32,
        );
        let down_gpu = offset_tensor(
            &ctx,
            18,
            &down_bytes,
            22,
            vec![HYPER as u64, RANK as u64],
            GgmlType::Q8_0,
        );
        let up_gpu = offset_tensor(
            &ctx,
            22,
            &up_bytes,
            18,
            vec![RANK as u64, HYPER as u64],
            GgmlType::Q8_0,
        );
        let inject_gpu = offset_tensor(
            &ctx,
            24,
            bytemuck::cast_slice(&inject),
            12,
            vec![HYPER as u64, BRANCHES as u64],
            GgmlType::F32,
        );
        let block_gpu = offset_tensor(
            &ctx,
            12,
            bytemuck::cast_slice(&block_output),
            24,
            vec![HIDDEN as u64],
            GgmlType::F32,
        );
        let mut scratch = GatedResidualMetalScratch::new(&ctx, BRANCHES, HIDDEN, RANK).unwrap();
        scratch.normalized = offset_tensor(
            &ctx,
            16,
            &vec![0; HYPER * 4],
            20,
            vec![HYPER as u64],
            GgmlType::F32,
        );
        scratch.low = offset_tensor(
            &ctx,
            20,
            &[0; RANK * 4],
            16,
            vec![RANK as u64],
            GgmlType::F32,
        );
        scratch.raw_gate = offset_tensor(
            &ctx,
            24,
            &vec![0; HYPER * 4],
            12,
            vec![HYPER as u64],
            GgmlType::F32,
        );
        scratch.mixed = offset_tensor(
            &ctx,
            12,
            &[0; HIDDEN * 4],
            24,
            vec![HIDDEN as u64],
            GgmlType::F32,
        );
        scratch.injection = offset_tensor(
            &ctx,
            28,
            &[0; BRANCHES * 4],
            8,
            vec![BRANCHES as u64],
            GgmlType::F32,
        );

        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode_gated_residual_mix(
            &ctx,
            &encoder,
            &input_gpu,
            &block_gpu,
            1e-6,
            GatedResidualMetalReadWeights {
                norm: &norm_gpu,
                down: &down_gpu,
                up: &up_gpu,
            },
            &inject_gpu,
            &mut scratch,
        )
        .unwrap()
        .encode_combine()
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );
        scratch.release_after().unwrap();

        assert_close(&read_f32(&scratch.mixed), &expected_mixed, 4e-5);
        assert_close(&read_f32(&input_gpu), &expected_output, 4e-5);
        for (tensor, prefix, suffix) in [
            (&input_gpu, 16, 20),
            (&norm_gpu, 20, 16),
            (&down_gpu, 18, 22),
            (&up_gpu, 22, 18),
            (&inject_gpu, 24, 12),
            (&block_gpu, 12, 24),
            (&scratch.normalized, 16, 20),
            (&scratch.low, 20, 16),
            (&scratch.raw_gate, 24, 12),
            (&scratch.mixed, 12, 24),
            (&scratch.injection, 28, 8),
        ] {
            assert_guards(tensor, prefix, suffix);
        }
    }

    #[test]
    fn released_geometry_q8_zero_gates_preserves_exact_branch_contract() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        const BRANCHES: usize = 4;
        const HIDDEN: usize = 2_560;
        const RANK: usize = 320;
        const HYPER: usize = BRANCHES * HIDDEN;
        let input = (0..HYPER)
            .map(|index| ((index * 17 + 5) % 127) as f32 * 0.002 - 0.126)
            .collect::<Vec<_>>();
        let norm = (0..HYPER)
            .map(|index| 0.75 + (index % 13) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let block_output = (0..HIDDEN)
            .map(|index| ((index * 11 + 3) % 97) as f32 * 0.001 - 0.048)
            .collect::<Vec<_>>();
        let mut expected_mixed = vec![0.0f32; HIDDEN];
        for branch in 0..BRANCHES {
            let offset = branch * HIDDEN;
            let mean_square = input[offset..offset + HIDDEN]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / HIDDEN as f32;
            let scale = 1.0 / (mean_square + 1e-6).sqrt();
            for hidden in 0..HIDDEN {
                expected_mixed[hidden] +=
                    input[offset + hidden] * scale * norm[offset + hidden] * 0.125;
            }
        }
        let expected_output = input
            .iter()
            .enumerate()
            .map(|(index, value)| value + block_output[index % HIDDEN])
            .collect::<Vec<_>>();

        let q8_bytes = |elements: usize| vec![0u8; elements / 32 * 34];
        let input_gpu = tensor(&ctx, &input, vec![HYPER as u64]);
        let norm_gpu = tensor(&ctx, &norm, vec![HYPER as u64]);
        let down_gpu = MetalTensor::from_bytes(
            &ctx,
            &q8_bytes(HYPER * RANK),
            vec![HYPER as u64, RANK as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let up_gpu = MetalTensor::from_bytes(
            &ctx,
            &q8_bytes(RANK * HYPER),
            vec![RANK as u64, HYPER as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let inject_gpu = tensor(&ctx, &vec![0.0; HYPER * BRANCHES], vec![HYPER as u64, 4]);
        let block_gpu = tensor(&ctx, &block_output, vec![HIDDEN as u64]);
        let mut scratch = GatedResidualMetalScratch::new(&ctx, BRANCHES, HIDDEN, RANK).unwrap();
        let command = ctx.queue.commandBuffer().expect("command buffer");
        let encoder = KernelEncoder::begin(&command);
        encode_gated_residual_mix(
            &ctx,
            &encoder,
            &input_gpu,
            &block_gpu,
            1e-6,
            GatedResidualMetalReadWeights {
                norm: &norm_gpu,
                down: &down_gpu,
                up: &up_gpu,
            },
            &inject_gpu,
            &mut scratch,
        )
        .unwrap()
        .encode_combine()
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );
        scratch.release_after().unwrap();

        assert_close(&read_f32(&scratch.mixed), &expected_mixed, 2e-5);
        assert_close(&read_f32(&input_gpu), &expected_output, 2e-6);
    }

    #[test]
    fn packed_hc_matches_serial_stages_residual_and_projection_routes() {
        let Some(ctx) = packed_test_context() else {
            return;
        };
        const BRANCHES: usize = 4;
        const HIDDEN: usize = 2_560;
        const RANK: usize = 320;
        const HYPER: usize = BRANCHES * HIDDEN;
        const TOKENS: usize = 33;

        let hyper_rows = (0..TOKENS * HYPER)
            .map(|index| {
                let token = index / HYPER;
                let centered = ((index * 37 + token * 19 + 11) % 251) as f32 - 125.0;
                centered * 0.001_37 + token as f32 * 0.000_3
            })
            .collect::<Vec<_>>();
        let block_rows = (0..TOKENS * HIDDEN)
            .map(|index| {
                let token = index / HIDDEN;
                let centered = ((index * 29 + token * 13 + 7) % 191) as f32 - 95.0;
                centered * 0.001_11 - token as f32 * 0.000_2
            })
            .collect::<Vec<_>>();
        let norm = (0..HYPER)
            .map(|index| 0.72 + (index * 7 % 29) as f32 * 0.021)
            .collect::<Vec<_>>();
        let injection = (0..HYPER * BRANCHES)
            .map(|index| ((index * 17 + 3) % 101) as f32 * 0.000_9 - 0.045)
            .collect::<Vec<_>>();
        let down_bytes = q8_bank(HYPER, RANK, 43);
        let up_bytes = q8_bank(RANK, HYPER, 10_043);
        let norm_gpu = tensor(&ctx, &norm, vec![HYPER as u64]);
        let down_gpu = MetalTensor::from_bytes(
            &ctx,
            &down_bytes,
            vec![HYPER as u64, RANK as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let up_gpu = MetalTensor::from_bytes(
            &ctx,
            &up_bytes,
            vec![RANK as u64, HYPER as u64],
            GgmlType::Q8_0,
        )
        .unwrap();
        let inject_gpu = tensor(&ctx, &injection, vec![HYPER as u64, BRANCHES as u64]);
        let weights = GatedResidualMetalReadWeights {
            norm: &norm_gpu,
            down: &down_gpu,
            up: &up_gpu,
        };
        let serial = serial_hc_trace(
            &ctx,
            &hyper_rows,
            &block_rows,
            TOKENS,
            BRANCHES,
            HIDDEN,
            RANK,
            weights,
            &inject_gpu,
        );

        let mut scratch =
            GatedResidualPackedScratch::new(&ctx, BRANCHES, HIDDEN, RANK, TOKENS).unwrap();
        let mut packed_n33 = None;
        for tokens in [1_usize, 2, 8, 16, 33] {
            let hyper = tensor(
                &ctx,
                &hyper_rows[..tokens * HYPER],
                vec![HYPER as u64, tokens as u64],
            );
            let block = tensor(
                &ctx,
                &block_rows[..tokens * HIDDEN],
                vec![HIDDEN as u64, tokens as u64],
            );
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            crate::metal::dispatch_census_begin();
            let read = unsafe {
                encode_gated_residual_packed_mix(
                    &ctx,
                    &encoder,
                    &hyper,
                    &block,
                    1e-6,
                    weights,
                    &inject_gpu,
                    &mut scratch,
                    tokens,
                )
            }
            .unwrap();
            let mixed = read.mixed().clone();
            if tokens == 8 {
                encoder.end();
                let combine_encoder = KernelEncoder::begin(&command);
                read.encode_combine(&combine_encoder).unwrap();
                combine_encoder.end();
            } else {
                read.encode_combine(&encoder).unwrap();
                encoder.end();
            }
            let census = crate::metal::dispatch_census_take();
            let q8_matvec = if crate::metal::mat_vec_q8_0_lcpp_enabled() {
                "kernel_mat_vec_q8_0_f32_lcpp"
            } else {
                "kernel_mat_vec_q8_0_f32"
            };
            let f32_matvec = if census
                .iter()
                .any(|row| row.kernel == "kernel_mat_vec_f32_f32_lcpp_r2")
            {
                "kernel_mat_vec_f32_f32_lcpp_r2"
            } else {
                "kernel_mat_vec_f32_f32"
            };
            let expected_projection_kernels = match tokens {
                1 => vec![q8_matvec, q8_matvec, f32_matvec],
                2 => vec![
                    "kernel_mat_mat_q8_0_f32",
                    "kernel_mat_mat_q8_0_f32",
                    "kernel_mat_mat_f32_f32",
                ],
                8 => vec![
                    "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32",
                    "kernel_mat_mat_q8_0_f32",
                    "kernel_mat_mat_f32_f32",
                ],
                16 => vec![
                    "kernel_mat_mat_q8_0_f32_n16",
                    "kernel_mat_mat_q8_0_f32_n16",
                    "kernel_mat_mat_f32_f32",
                ],
                33 => vec![
                    "kernel_mat_mat_q8_0_f32",
                    "kernel_mat_mat_q8_0_f32",
                    "kernel_mat_mat_f32_f32",
                ],
                _ => unreachable!(),
            };
            let projection_kernels = census
                .iter()
                .filter_map(|row| {
                    matches!(
                        row.kernel.as_str(),
                        "kernel_mat_vec_q8_0_f32_lcpp"
                            | "kernel_mat_vec_q8_0_f32"
                            | "kernel_mat_vec_f32_f32_lcpp_r2"
                            | "kernel_mat_vec_f32_f32"
                            | "kernel_mat_mat_q8_0_mma8v_r1c1k128_f32"
                            | "kernel_mat_mat_q8_0_f32_n16"
                            | "kernel_mat_mat_q8_0_f32"
                            | "kernel_mat_mat_f32_f32"
                    )
                    .then_some(row.kernel.as_str())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                projection_kernels, expected_projection_kernels,
                "N={tokens} packed HC projection sequence: {census:#?}"
            );
            command.commit();
            command.waitUntilCompleted();
            assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
            assert!(command.error().is_none());

            let normalized = read_f32(
                &scratch
                    .prefix_view("packed HC normalized", &scratch.normalized, HYPER, tokens)
                    .unwrap(),
            );
            let low = read_f32(
                &scratch
                    .prefix_view("packed HC low", &scratch.low, RANK, tokens)
                    .unwrap(),
            );
            let raw_gate = read_f32(
                &scratch
                    .prefix_view("packed HC raw gate", &scratch.raw_gate, HYPER, tokens)
                    .unwrap(),
            );
            let injection = read_f32(
                &scratch
                    .prefix_view("packed HC injection", &scratch.injection, BRANCHES, tokens)
                    .unwrap(),
            );
            let expected_normalized = &serial.normalized[..tokens * HYPER];
            let expected_low = &serial.low[..tokens * RANK];
            let expected_raw_gate = &serial.raw_gate[..tokens * HYPER];
            let expected_mixed = &serial.mixed[..tokens * HIDDEN];
            let expected_injection = &serial.injection[..tokens * BRANCHES];
            let expected_residual = &serial.residual[..tokens * HYPER];
            let packed_mixed = read_f32(&mixed);
            let packed_residual = read_f32(&hyper);
            if tokens == TOKENS {
                packed_n33 = Some(SerialPackedHcTrace {
                    normalized: normalized.clone(),
                    low: low.clone(),
                    raw_gate: raw_gate.clone(),
                    mixed: packed_mixed.clone(),
                    injection: injection.clone(),
                    residual: packed_residual.clone(),
                });
            }
            assert_bits_eq(
                &format!("packed HC N={tokens} normalized"),
                &normalized,
                expected_normalized,
            );
            if tokens == 1 {
                for (stage, actual, expected) in [
                    ("low", low.as_slice(), expected_low),
                    ("raw gate", raw_gate.as_slice(), expected_raw_gate),
                    ("mixed", packed_mixed.as_slice(), expected_mixed),
                    ("injection", injection.as_slice(), expected_injection),
                    ("residual", packed_residual.as_slice(), expected_residual),
                ] {
                    assert_bits_eq(&format!("packed HC N=1 {stage}"), actual, expected);
                }
            } else {
                assert!(
                    low.iter()
                        .zip(expected_low)
                        .any(|(actual, expected)| actual.to_bits() != expected.to_bits()),
                    "packed HC N={tokens} must exercise matrix reduction"
                );
                for (stage, actual, expected, width, maximum_absolute) in [
                    ("low", low.as_slice(), expected_low, RANK, 2e-2),
                    (
                        "raw gate",
                        raw_gate.as_slice(),
                        expected_raw_gate,
                        HYPER,
                        1e-2,
                    ),
                    (
                        "mixed",
                        packed_mixed.as_slice(),
                        expected_mixed,
                        HIDDEN,
                        1e-3,
                    ),
                    (
                        "injection",
                        injection.as_slice(),
                        expected_injection,
                        BRANCHES,
                        1e-3,
                    ),
                    (
                        "residual",
                        packed_residual.as_slice(),
                        expected_residual,
                        HYPER,
                        1e-4,
                    ),
                ] {
                    assert_tokenwise_similarity(
                        &format!("packed HC N={tokens} {stage}"),
                        actual,
                        expected,
                        width,
                        tokens,
                        2e-3,
                        if stage == "raw gate" {
                            0.999_998
                        } else {
                            0.999_999_8
                        },
                        maximum_absolute,
                    );
                }
            }
        }

        let changed_token = 17;
        let mut perturbed_hyper = hyper_rows.clone();
        let mut perturbed_block = block_rows.clone();
        for (index, value) in perturbed_hyper[changed_token * HYPER..(changed_token + 1) * HYPER]
            .iter_mut()
            .enumerate()
        {
            *value += 1.5 + (index % 17) as f32 * 0.031;
        }
        for (index, value) in perturbed_block[changed_token * HIDDEN..(changed_token + 1) * HIDDEN]
            .iter_mut()
            .enumerate()
        {
            *value -= 1.25 + (index % 13) as f32 * 0.027;
        }
        let hyper = tensor(&ctx, &perturbed_hyper, vec![HYPER as u64, TOKENS as u64]);
        let block = tensor(&ctx, &perturbed_block, vec![HIDDEN as u64, TOKENS as u64]);
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = unsafe {
            encode_gated_residual_packed_mix(
                &ctx,
                &encoder,
                &hyper,
                &block,
                1e-6,
                weights,
                &inject_gpu,
                &mut scratch,
                TOKENS,
            )
        }
        .unwrap();
        let mixed = read.mixed().clone();
        read.encode_combine(&encoder).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());
        let perturbed = SerialPackedHcTrace {
            normalized: read_f32(
                &scratch
                    .prefix_view(
                        "perturbed packed HC normalized",
                        &scratch.normalized,
                        HYPER,
                        TOKENS,
                    )
                    .unwrap(),
            ),
            low: read_f32(
                &scratch
                    .prefix_view("perturbed packed HC low", &scratch.low, RANK, TOKENS)
                    .unwrap(),
            ),
            raw_gate: read_f32(
                &scratch
                    .prefix_view(
                        "perturbed packed HC raw gate",
                        &scratch.raw_gate,
                        HYPER,
                        TOKENS,
                    )
                    .unwrap(),
            ),
            mixed: read_f32(&mixed),
            injection: read_f32(
                &scratch
                    .prefix_view(
                        "perturbed packed HC injection",
                        &scratch.injection,
                        BRANCHES,
                        TOKENS,
                    )
                    .unwrap(),
            ),
            residual: read_f32(&hyper),
        };
        let baseline = packed_n33.expect("capture packed HC N=33 baseline");
        for (stage, baseline, perturbed, width) in [
            (
                "normalized",
                baseline.normalized.as_slice(),
                perturbed.normalized.as_slice(),
                HYPER,
            ),
            (
                "low",
                baseline.low.as_slice(),
                perturbed.low.as_slice(),
                RANK,
            ),
            (
                "raw gate",
                baseline.raw_gate.as_slice(),
                perturbed.raw_gate.as_slice(),
                HYPER,
            ),
            (
                "mixed",
                baseline.mixed.as_slice(),
                perturbed.mixed.as_slice(),
                HIDDEN,
            ),
            (
                "injection",
                baseline.injection.as_slice(),
                perturbed.injection.as_slice(),
                BRANCHES,
            ),
            (
                "residual",
                baseline.residual.as_slice(),
                perturbed.residual.as_slice(),
                HYPER,
            ),
        ] {
            assert_only_token_changed(
                &format!("packed HC {stage}"),
                baseline,
                perturbed,
                width,
                TOKENS,
                changed_token,
            );
        }
    }

    #[test]
    fn packed_hc_repeat_preserves_token_major_branch_layout_and_guards() {
        let Some(ctx) = packed_test_context() else {
            return;
        };
        const BRANCHES: usize = 4;
        const HIDDEN: usize = 37;
        const TOKENS: usize = 33;
        const PREFIX: usize = 16;
        const SUFFIX: usize = 20;

        let values = (0..TOKENS * HIDDEN)
            .map(|index| {
                let token = index / HIDDEN;
                let hidden = index % HIDDEN;
                token as f32 * 10.0 + hidden as f32 * 0.03125 - 3.0
            })
            .collect::<Vec<_>>();
        let embedding = tensor(&ctx, &values, vec![HIDDEN as u64, TOKENS as u64]);
        let output_bytes = vec![0_u8; TOKENS * BRANCHES * HIDDEN * size_of::<f32>()];
        let output = offset_tensor(
            &ctx,
            PREFIX,
            &output_bytes,
            SUFFIX,
            vec![(BRANCHES * HIDDEN) as u64, TOKENS as u64],
            GgmlType::F32,
        );

        let command = ctx.queue.commandBuffer().expect("repeat command");
        let encoder = KernelEncoder::begin(&command);
        crate::metal::dispatch_census_begin();
        encode_hc_repeat_packed(
            &ctx, &encoder, &embedding, &output, BRANCHES, HIDDEN, TOKENS,
        )
        .unwrap();
        let census = crate::metal::dispatch_census_take();
        assert_eq!(census.len(), 1, "packed repeat census: {census:#?}");
        assert_eq!(census[0].kernel, "kernel_qwen4exp_hc_repeat_packed_f32");
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(
            command.error().is_none(),
            "command failed: {:?}",
            command.error()
        );

        let actual = read_f32(&output);
        for token in 0..TOKENS {
            for branch in 0..BRANCHES {
                let start = (token * BRANCHES + branch) * HIDDEN;
                assert_bits_eq(
                    &format!("packed HC repeat token {token} branch {branch}"),
                    &actual[start..start + HIDDEN],
                    &values[token * HIDDEN..(token + 1) * HIDDEN],
                );
            }
        }
        assert_guards(&output, PREFIX, SUFFIX);

        let shared =
            MetalTensor::zeros_f32(&ctx, vec![(BRANCHES * HIDDEN) as u64, TOKENS as u64]).unwrap();
        let mut overlapping_embedding = shared.clone();
        overlapping_embedding.shape = vec![HIDDEN as u64, TOKENS as u64];
        let error = validate_and_preflight_hc_repeat_packed(
            &ctx,
            &overlapping_embedding,
            &shared,
            BRANCHES,
            HIDDEN,
            TOKENS,
        )
        .unwrap_err();
        assert!(error.to_string().contains("overlaps"), "{error}");
    }

    #[test]
    fn packed_hc_contract_rejects_aliases_before_encoding() {
        let Some(ctx) = packed_test_context() else {
            return;
        };
        const BRANCHES: usize = 2;
        const HIDDEN: usize = 32;
        const RANK: usize = 32;
        const HYPER: usize = BRANCHES * HIDDEN;
        let oversized = match GatedResidualPackedScratch::new(&ctx, 1, 65_536, 65_536, 1) {
            Ok(_) => panic!("packed HC accepted projection addressing above u32"),
            Err(error) => error,
        };
        assert!(
            oversized.to_string().contains("shader addressing"),
            "{oversized}"
        );
        let hyper = tensor(&ctx, &[0.25; HYPER], vec![HYPER as u64, 1]);
        let block = tensor(&ctx, &[0.5; HIDDEN], vec![HIDDEN as u64, 1]);
        let norm = tensor(&ctx, &[1.0; HYPER], vec![HYPER as u64]);
        let down = tensor(&ctx, &[0.0; HYPER * RANK], vec![HYPER as u64, RANK as u64]);
        let up = tensor(&ctx, &[0.0; RANK * HYPER], vec![RANK as u64, HYPER as u64]);
        let inject = tensor(
            &ctx,
            &[0.0; HYPER * BRANCHES],
            vec![HYPER as u64, BRANCHES as u64],
        );
        let mut scratch = GatedResidualPackedScratch::new(&ctx, BRANCHES, HIDDEN, RANK, 1).unwrap();
        scratch.normalized = hyper.clone();
        let error = validate_and_preflight_gated_residual_packed_mix(
            &ctx,
            &hyper,
            &block,
            1e-6,
            GatedResidualMetalReadWeights {
                norm: &norm,
                down: &down,
                up: &up,
            },
            &inject,
            &scratch,
            1,
        )
        .unwrap_err();
        assert!(error.to_string().contains("overlaps"), "{error}");
        assert_bits_eq(
            "rejected packed HC residual",
            &read_f32(&hyper),
            &[0.25; HYPER],
        );
    }

    #[test]
    fn scratch_cannot_cross_command_buffers_before_completion() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        let input = tensor(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let norm = tensor(&ctx, &[1.0; 4], vec![4]);
        let down = tensor(&ctx, &[0.0; 8], vec![4, 2]);
        let up = tensor(&ctx, &[0.0; 8], vec![2, 4]);
        let inject = tensor(&ctx, &[0.0; 8], vec![4, 2]);
        let block = tensor(&ctx, &[1.0, 2.0], vec![2]);
        let weights = GatedResidualMetalReadWeights {
            norm: &norm,
            down: &down,
            up: &up,
        };
        let mut scratch = GatedResidualMetalScratch::new(&ctx, 2, 2, 2).unwrap();

        let first_command = ctx.queue.commandBuffer().expect("first command");
        let first_encoder = KernelEncoder::begin(&first_command);
        encode_gated_residual_mix(
            &ctx,
            &first_encoder,
            &input,
            &block,
            1e-6,
            weights,
            &inject,
            &mut scratch,
        )
        .unwrap()
        .encode_combine()
        .unwrap();
        first_encoder.end();

        let uncommitted_error = scratch.release_after().unwrap_err().to_string();
        assert!(uncommitted_error.contains("not committed"));

        let blocked_command = ctx.queue.commandBuffer().expect("blocked command");
        let blocked_encoder = KernelEncoder::begin(&blocked_command);
        assert!(
            encode_final_gated_residual_mix(
                &ctx,
                &blocked_encoder,
                &input,
                1e-6,
                weights,
                &mut scratch,
            )
            .is_err()
        );
        blocked_encoder.end();

        first_command.commit();
        scratch.release_after().unwrap();

        let resumed_command = ctx.queue.commandBuffer().expect("resumed command");
        let resumed_encoder = KernelEncoder::begin(&resumed_command);
        let final_read = encode_final_gated_residual_mix(
            &ctx,
            &resumed_encoder,
            &input,
            1e-6,
            weights,
            &mut scratch,
        )
        .unwrap();
        resumed_encoder.end();
        resumed_command.commit();
        resumed_command.waitUntilCompleted();
        drop(final_read);
        scratch.release_after().unwrap();

        let abandoned_command = ctx.queue.commandBuffer().expect("abandoned command");
        let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
        let abandoned_read = encode_final_gated_residual_mix(
            &ctx,
            &abandoned_encoder,
            &input,
            1e-6,
            weights,
            &mut scratch,
        )
        .unwrap();
        drop(abandoned_read);
        abandoned_encoder.end();
        // SAFETY: The encoder has ended and the sole caller-held command
        // reference is dropped immediately without being committed.
        unsafe { scratch.abandon_uncommitted() }.unwrap();
        drop(abandoned_command);

        let recovered_command = ctx.queue.commandBuffer().expect("recovered command");
        let recovered_encoder = KernelEncoder::begin(&recovered_command);
        let recovered_read = encode_final_gated_residual_mix(
            &ctx,
            &recovered_encoder,
            &input,
            1e-6,
            weights,
            &mut scratch,
        )
        .unwrap();
        recovered_encoder.end();
        recovered_command.commit();
        drop(recovered_read);
        scratch.release_after().unwrap();
    }

    #[test]
    fn gated_residual_rejects_aliases_and_read_only_residuals() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        let input = tensor(&ctx, &[1.0, 2.0, 3.0, 4.0], vec![4]);
        let norm = tensor(&ctx, &[1.0; 4], vec![4]);
        let down = tensor(&ctx, &[0.0; 8], vec![4, 2]);
        let up = tensor(&ctx, &[0.0; 8], vec![2, 4]);
        let inject = tensor(&ctx, &[0.0; 8], vec![4, 2]);
        let block = tensor(&ctx, &[1.0, 2.0], vec![2]);
        let mut scratch = GatedResidualMetalScratch::new(&ctx, 2, 2, 2).unwrap();
        scratch.normalized = input.clone();
        let command = ctx.queue.commandBuffer().expect("validation command");
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_gated_residual_mix(
                &ctx,
                &encoder,
                &input,
                &block,
                1e-6,
                GatedResidualMetalReadWeights {
                    norm: &norm,
                    down: &down,
                    up: &up,
                },
                &inject,
                &mut scratch,
            )
            .is_err()
        );

        scratch.normalized = MetalTensor::zeros_f32(&ctx, vec![4]).unwrap();
        let mut read_only = input.clone();
        read_only.provenance = crate::metal::MetalTensorProvenance::OwnedWeightReadOnly;
        assert!(
            encode_gated_residual_mix(
                &ctx,
                &encoder,
                &read_only,
                &block,
                1e-6,
                GatedResidualMetalReadWeights {
                    norm: &norm,
                    down: &down,
                    up: &up,
                },
                &inject,
                &mut scratch,
            )
            .is_err()
        );
        encoder.end();

        let concurrent_command = ctx.queue.commandBuffer().expect("concurrent command");
        let concurrent_encoder = KernelEncoder::begin_concurrent(&concurrent_command);
        let mut concurrent_scratch = GatedResidualMetalScratch::new(&ctx, 2, 2, 2).unwrap();
        assert!(
            encode_gated_residual_mix(
                &ctx,
                &concurrent_encoder,
                &input,
                &block,
                1e-6,
                GatedResidualMetalReadWeights {
                    norm: &norm,
                    down: &down,
                    up: &up,
                },
                &inject,
                &mut concurrent_scratch,
            )
            .is_err()
        );
        concurrent_encoder.end();
    }
}
