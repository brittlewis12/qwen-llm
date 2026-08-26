//! Metal primitives for Qwen3.8-Flash-Next gated residuals.

use crate::metal::{KernelEncoder, MetalContext, MetalError, MetalTensor};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLSize,
};

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
    require_serial(enc)?;
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
    preflight_mix(ctx, weights.down.dtype, weights.up.dtype)?;
    preflight_combine(ctx)
}

pub fn encode_final_gated_residual_mix<'scratch>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_input: &MetalTensor,
    eps: f32,
    weights: GatedResidualMetalReadWeights<'_>,
    scratch: &'scratch mut GatedResidualMetalScratch,
) -> Result<GatedResidualMetalFinalRead<'scratch>, Qwen4ExpMetalError> {
    require_serial(enc)?;
    let hyper_hidden = validate_read_contract(hyper_input, eps, weights, scratch, false)?;
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
    preflight_mix(ctx, weights.down.dtype, weights.up.dtype)?;
    reserve_command(scratch, enc)?;
    encode_read(ctx, enc, hyper_input, eps, weights, scratch, hyper_hidden)?;
    Ok(GatedResidualMetalFinalRead { scratch })
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

fn require_serial(enc: &KernelEncoder) -> Result<(), Qwen4ExpMetalError> {
    if enc.is_concurrent() {
        Err(invalid(
            "gated-residual dependent dispatches require a serial encoder",
        ))
    } else {
        Ok(())
    }
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

fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpMetalError> {
    let kernels: &[&str] = match dtype {
        GgmlType::F32 => &["kernel_mat_vec_f32_f32", "kernel_mat_vec_f32_f32_lcpp_r2"],
        GgmlType::Q8_0 => &["kernel_mat_vec_q8_0_f32", "kernel_mat_vec_q8_0_f32_lcpp"],
        _ => {
            return Err(invalid(format!(
                "unsupported gated-residual projection dtype {dtype:?}"
            )));
        }
    };
    for kernel in kernels {
        ctx.pipeline(kernel)?;
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
    use super::*;
    use crate::metal::MetalTensorProvenance;
    use crate::qwen4exp_forward::{
        GatedResidualReadWeights, gated_residual_combine, gated_residual_mix,
    };
    use objc2_metal::MTLCommandQueue;

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
