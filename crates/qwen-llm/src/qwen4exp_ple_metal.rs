//! One-token Metal PLE execution for Qwen3.8-Flash-Next.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_copy_offset_f32, encode_get_rows_f32,
};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::qwen4exp::{Qwen4ExpConfig, Qwen4ExpError};
use crate::qwen4exp_ple::{PleGatherError, PleIq4NlTable};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource, MTLSize,
};
use std::mem::size_of;

const SIMD_WIDTH: usize = 32;
const PLE_THREADS: usize = 256;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpPleMetalError {
    #[error(transparent)]
    Config(#[from] Qwen4ExpError),
    #[error(transparent)]
    Gather(#[from] PleGatherError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next PLE contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next PLE command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen4ExpPleMetalGeometry {
    branch_count: usize,
    hidden_size: usize,
    head_count: usize,
    head_dim: usize,
    kernel_size: usize,
    dilation: usize,
    eps: f32,
}

impl Qwen4ExpPleMetalGeometry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        branch_count: usize,
        hidden_size: usize,
        head_count: usize,
        head_dim: usize,
        kernel_size: usize,
        dilation: usize,
        eps: f32,
    ) -> Result<Self, Qwen4ExpPleMetalError> {
        let geometry = Self {
            branch_count,
            hidden_size,
            head_count,
            head_dim,
            kernel_size,
            dilation,
            eps,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(config: &Qwen4ExpConfig, layer: u32) -> Result<Self, Qwen4ExpPleMetalError> {
        config.validate()?;
        let ple = config.ple.as_ref().ok_or_else(|| {
            Qwen4ExpPleMetalError::Invalid("model has no PLE configuration".into())
        })?;
        if !ple.layers.contains(&layer) {
            return invalid(format!("layer {layer} is not configured for PLE"));
        }
        Self::new(
            config.hyper_connection.count as usize,
            config.hidden_size as usize,
            ple.head_count()? as usize,
            ple.embedding_head_dim as usize,
            ple.conv_kernel as usize,
            ple.ngram_size as usize,
            config.rms_norm_eps,
        )
    }

    pub fn branch_count(self) -> usize {
        self.branch_count
    }

    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub fn head_count(self) -> usize {
        self.head_count
    }

    pub fn head_dim(self) -> usize {
        self.head_dim
    }

    pub fn kernel_size(self) -> usize {
        self.kernel_size
    }

    pub fn dilation(self) -> usize {
        self.dilation
    }

    pub fn eps(self) -> f32 {
        self.eps
    }

    pub fn hyper_width(self) -> usize {
        self.branch_count
            .checked_mul(self.hidden_size)
            .expect("validated PLE hyper width")
    }

    pub fn history_len(self) -> usize {
        (self.kernel_size - 1)
            .checked_mul(self.dilation)
            .expect("validated PLE history length")
    }

    pub fn packed_row_bytes(self) -> usize {
        self.head_dim
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(18))
            .expect("validated PLE packed row size")
    }

    pub fn packed_staging_bytes(self) -> usize {
        self.packed_row_bytes()
            .checked_mul(self.head_count)
            .expect("validated PLE packed staging size")
    }

    fn validate(self) -> Result<(), Qwen4ExpPleMetalError> {
        if self.branch_count == 0
            || self.hidden_size == 0
            || self.head_count == 0
            || self.head_dim == 0
            || self.kernel_size == 0
            || self.dilation == 0
        {
            return invalid("PLE dimensions must be nonzero");
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return invalid("PLE RMS epsilon must be finite and positive");
        }
        if !self.head_dim.is_multiple_of(32) {
            return invalid(format!(
                "PLE embedding head width {} is not IQ4_NL block aligned",
                self.head_dim
            ));
        }
        if self.head_count.checked_mul(self.head_dim) != Some(self.hidden_size) {
            return invalid("PLE head count times head width must equal hidden size");
        }
        if self.head_count > i32::MAX as usize {
            return invalid("PLE staged row IDs must fit signed i32");
        }
        let hyper_width = self
            .branch_count
            .checked_mul(self.hidden_size)
            .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("PLE hyper width overflow".into()))?;
        let history_len = (self.kernel_size - 1)
            .checked_mul(self.dilation)
            .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("PLE history length overflow".into()))?;
        let state_elements = hyper_width.checked_mul(history_len).ok_or_else(|| {
            Qwen4ExpPleMetalError::Invalid("PLE convolution state size overflow".into())
        })?;
        let packed_bytes = self
            .head_dim
            .checked_div(32)
            .and_then(|blocks| blocks.checked_mul(18))
            .and_then(|row_bytes| row_bytes.checked_mul(self.head_count))
            .ok_or_else(|| {
                Qwen4ExpPleMetalError::Invalid("PLE packed staging size overflow".into())
            })?;
        for (name, value) in [
            ("branch count", self.branch_count),
            ("hidden size", self.hidden_size),
            ("head count", self.head_count),
            ("head width", self.head_dim),
            ("kernel size", self.kernel_size),
            ("dilation", self.dilation),
            ("hyper width", hyper_width),
            ("history length", history_len),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("PLE {name} {value} exceeds u32"));
            }
        }
        for (name, elements) in [
            ("key projection", self.hidden_size.checked_mul(hyper_width)),
            (
                "value projection",
                self.hidden_size.checked_mul(self.hidden_size),
            ),
            (
                "convolution weights",
                hyper_width.checked_mul(self.kernel_size),
            ),
            ("convolution state", Some(state_elements)),
        ] {
            let elements = elements.ok_or_else(|| {
                Qwen4ExpPleMetalError::Invalid(format!("PLE {name} element count overflow"))
            })?;
            elements.checked_mul(size_of::<f32>()).ok_or_else(|| {
                Qwen4ExpPleMetalError::Invalid(format!("PLE {name} byte count overflow"))
            })?;
        }
        if packed_bytes == 0 {
            return invalid("PLE packed staging must be nonempty");
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpPleMetalWeights<'a> {
    pub geometry: Qwen4ExpPleMetalGeometry,
    pub key: &'a MetalTensor,
    pub value: &'a MetalTensor,
    pub key_norm: &'a MetalTensor,
    pub query_norm: &'a MetalTensor,
    pub conv_norm: &'a MetalTensor,
    pub conv: &'a MetalTensor,
}

impl<'a> Qwen4ExpPleMetalWeights<'a> {
    pub fn bind(
        weights: &'a Qwen4ExpMetalWeights,
        layer: u32,
    ) -> Result<Self, Qwen4ExpPleMetalError> {
        let geometry = Qwen4ExpPleMetalGeometry::from_config(weights.config(), layer)?;
        let prefix = format!("blk.{layer}.ple");
        Ok(Self {
            geometry,
            key: weights.require_tensor(&format!("{prefix}_key.weight"))?,
            value: weights.require_tensor(&format!("{prefix}_value.weight"))?,
            key_norm: weights.require_tensor(&format!("{prefix}_norm_key.weight"))?,
            query_norm: weights.require_tensor(&format!("{prefix}_norm_query.weight"))?,
            conv_norm: weights.require_tensor(&format!("{prefix}_norm_conv.weight"))?,
            conv: weights.require_tensor(&format!("{prefix}_conv1d.weight"))?,
        })
    }
}

pub struct Qwen4ExpPleMetalWorkspace {
    geometry: Qwen4ExpPleMetalGeometry,
    packed_rows: MetalTensor,
    local_row_ids: MetalTensor,
    embedding: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    key_norm: MetalTensor,
    query_norm: MetalTensor,
    gated: MetalTensor,
    conv_input: MetalTensor,
    conv_state: MetalTensor,
    conv_raw: MetalTensor,
    output: MetalTensor,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    rows_staged: bool,
    state_poisoned: bool,
    encode_failed: bool,
}

impl Qwen4ExpPleMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpPleMetalGeometry,
    ) -> Result<Self, Qwen4ExpPleMetalError> {
        geometry.validate()?;
        let packed_rows = MetalTensor::from_bytes(
            ctx,
            &vec![0_u8; geometry.packed_staging_bytes()],
            vec![geometry.head_dim as u64, geometry.head_count as u64],
            GgmlType::IQ4_NL,
        )?;
        let local_ids = (0..geometry.head_count)
            .map(|row| row as i32)
            .collect::<Vec<_>>();
        let hyper = geometry.hyper_width();
        Ok(Self {
            geometry,
            packed_rows,
            local_row_ids: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&local_ids),
                vec![geometry.head_count as u64],
                GgmlType::I32,
            )?,
            embedding: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            key: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            value: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            key_norm: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            query_norm: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            gated: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            conv_input: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            conv_state: zero_f32(ctx, vec![geometry.history_len() as u64, hyper as u64])?,
            conv_raw: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![hyper as u64])?,
            active_command: None,
            rows_staged: false,
            state_poisoned: false,
            encode_failed: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpPleMetalGeometry {
        self.geometry
    }

    pub fn has_staged_rows(&self) -> bool {
        self.rows_staged
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn stage_rows(
        &mut self,
        table: PleIq4NlTable<'_>,
        row_ids: &[u32],
    ) -> Result<(), Qwen4ExpPleMetalError> {
        self.require_idle()?;
        if self.state_poisoned {
            return invalid("workspace causal state is indeterminate; reset it before staging");
        }
        self.rows_staged = false;
        if table.row_width() != self.geometry.head_dim {
            return invalid(format!(
                "PLE table row width {} differs from staged head width {}",
                table.row_width(),
                self.geometry.head_dim
            ));
        }
        if row_ids.len() != self.geometry.head_count {
            return invalid(format!(
                "PLE row ID count {} differs from head count {}",
                row_ids.len(),
                self.geometry.head_count
            ));
        }
        let mut packed = vec![0_u8; self.geometry.packed_staging_bytes()];
        table.gather_packed_into(row_ids, &mut packed)?;
        require_tensor(
            "PLE packed row staging",
            &self.packed_rows,
            GgmlType::IQ4_NL,
            &[
                self.geometry.head_dim as u64,
                self.geometry.head_count as u64,
            ],
            true,
        )?;
        write_tensor_bytes(&self.packed_rows, &packed)?;
        self.rows_staged = true;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpPleMetalError> {
        self.require_idle()?;
        zero_writable_f32(&self.conv_state)?;
        self.rows_staged = false;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpPleMetalError> {
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
        let error = command.error().map(|error| error.to_string());
        self.active_command = None;
        if status == MTLCommandBufferStatus::Completed && error.is_none() && !self.encode_failed {
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpPleMetalError::CommandBuffer(format!(
                "status={status:?}, error={error:?}, encode_failed={}",
                self.encode_failed
            )))
        }
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command. Committing it later may race a later owner or mutate
    /// causal convolution state out of order.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpPleMetalError> {
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
        self.rows_staged = false;
        self.state_poisoned = false;
        self.encode_failed = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpPleMetalError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "copy or consume the PLE output in its owning command, then release the workspace"]
pub struct Qwen4ExpPleMetalRead<'a> {
    workspace: &'a mut Qwen4ExpPleMetalWorkspace,
}

pub struct Qwen4ExpPleMetalOutput<'a> {
    workspace: &'a Qwen4ExpPleMetalWorkspace,
}

impl Qwen4ExpPleMetalRead<'_> {
    pub fn output(&self) -> Qwen4ExpPleMetalOutput<'_> {
        Qwen4ExpPleMetalOutput {
            workspace: self.workspace,
        }
    }
}

impl Qwen4ExpPleMetalOutput<'_> {
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
    ) -> Result<(), Qwen4ExpPleMetalError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("PLE output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("PLE output must be copied by its owning command buffer");
        }
        require_tensor(
            "PLE copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hyper_width() as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("PLE output", &self.workspace.output),
                ("PLE copied output destination", destination),
            ],
        )?;
        let mut tensors = workspace_tensors(self.workspace);
        tensors.push(("PLE copied output destination", destination));
        require_disjoint(&tensors)?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(
            ctx,
            enc,
            &self.workspace.output,
            0,
            destination,
            self.workspace.geometry.hyper_width(),
        )?;
        Ok(())
    }
}

pub fn encode_qwen4exp_ple<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_input: &MetalTensor,
    weights: Qwen4ExpPleMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpPleMetalWorkspace,
) -> Result<Qwen4ExpPleMetalRead<'a>, Qwen4ExpPleMetalError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if !workspace.rows_staged {
        return invalid("PLE rows must be staged before encoding a token");
    }
    if weights.geometry != workspace.geometry {
        return invalid("PLE weight and workspace geometry differ");
    }
    validate_contract(ctx, hyper_input, weights, workspace)?;
    preflight(ctx, weights)?;
    reserve_command(workspace, enc)?;
    workspace.rows_staged = false;
    if let Err(error) = encode_step(ctx, enc, hyper_input, weights, workspace) {
        workspace.encode_failed = true;
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpPleMetalRead { workspace })
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    hyper_input: &MetalTensor,
    weights: Qwen4ExpPleMetalWeights<'_>,
    workspace: &Qwen4ExpPleMetalWorkspace,
) -> Result<(), Qwen4ExpPleMetalError> {
    let g = workspace.geometry;
    encode_get_rows_f32(
        ctx,
        enc,
        &workspace.packed_rows,
        &workspace.local_row_ids,
        &workspace.embedding,
        g.head_count,
        g.head_dim,
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.key,
        &workspace.embedding,
        &workspace.key,
        g.hidden_size,
        g.hyper_width(),
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.value,
        &workspace.embedding,
        &workspace.value,
        g.hidden_size,
        g.hidden_size,
    )?;
    encode_grouped_rms_norm(
        ctx,
        enc,
        &workspace.key,
        weights.key_norm,
        &workspace.key_norm,
        g,
    )?;
    encode_grouped_rms_norm(
        ctx,
        enc,
        hyper_input,
        weights.query_norm,
        &workspace.query_norm,
        g,
    )?;
    encode_gate(ctx, enc, workspace, g)?;
    encode_grouped_rms_norm(
        ctx,
        enc,
        &workspace.gated,
        weights.conv_norm,
        &workspace.conv_input,
        g,
    )?;
    encode_conv_epilogue(ctx, enc, hyper_input, weights.conv, workspace, g)?;
    Ok(())
}

pub(crate) fn validate_contract(
    ctx: &MetalContext,
    hyper_input: &MetalTensor,
    weights: Qwen4ExpPleMetalWeights<'_>,
    workspace: &Qwen4ExpPleMetalWorkspace,
) -> Result<(), Qwen4ExpPleMetalError> {
    let g = workspace.geometry;
    require_tensor(
        "PLE hyper input",
        hyper_input,
        GgmlType::F32,
        &[g.hyper_width() as u64],
        false,
    )?;
    require_projection(
        "PLE key projection",
        weights.key,
        g.hidden_size,
        g.hyper_width(),
    )?;
    require_projection(
        "PLE value projection",
        weights.value,
        g.hidden_size,
        g.hidden_size,
    )?;
    for (name, tensor) in [
        ("PLE key norm", weights.key_norm),
        ("PLE query norm", weights.query_norm),
        ("PLE convolution norm", weights.conv_norm),
    ] {
        require_tensor(
            name,
            tensor,
            GgmlType::F32,
            &[g.hyper_width() as u64],
            false,
        )?;
    }
    require_tensor(
        "PLE convolution",
        weights.conv,
        GgmlType::F32,
        &[g.kernel_size as u64, g.hyper_width() as u64],
        false,
    )?;
    for (name, tensor, dtype, shape) in [
        (
            "PLE packed row staging",
            &workspace.packed_rows,
            GgmlType::IQ4_NL,
            vec![g.head_dim as u64, g.head_count as u64],
        ),
        (
            "PLE local row IDs",
            &workspace.local_row_ids,
            GgmlType::I32,
            vec![g.head_count as u64],
        ),
        (
            "PLE embedding",
            &workspace.embedding,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
        (
            "PLE key scratch",
            &workspace.key,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "PLE value scratch",
            &workspace.value,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
        (
            "PLE normalized key",
            &workspace.key_norm,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "PLE normalized query",
            &workspace.query_norm,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "PLE gated value",
            &workspace.gated,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "PLE convolution input",
            &workspace.conv_input,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "PLE convolution state",
            &workspace.conv_state,
            GgmlType::F32,
            vec![g.history_len() as u64, g.hyper_width() as u64],
        ),
        (
            "PLE raw convolution output",
            &workspace.conv_raw,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
        (
            "PLE output",
            &workspace.output,
            GgmlType::F32,
            vec![g.hyper_width() as u64],
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, true)?;
    }
    let named_weights = named_weight_tensors(weights);
    require_read_only_weights(&named_weights)?;
    let mut tensors = named_weights;
    tensors.push(("PLE hyper input", hyper_input));
    tensors.extend(workspace_tensors(workspace));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)
}

pub(crate) fn preflight(
    ctx: &MetalContext,
    weights: Qwen4ExpPleMetalWeights<'_>,
) -> Result<(), Qwen4ExpPleMetalError> {
    for dtype in [weights.key.dtype, weights.value.dtype] {
        preflight_projection(ctx, dtype)?;
    }
    let get_rows = ctx.pipeline("kernel_get_rows_iq4_nl_f32")?;
    require_pipeline_geometry("PLE IQ4_NL row gather", &get_rows, SIMD_WIDTH, SIMD_WIDTH)?;
    let grouped = ctx.pipeline("kernel_qwen4exp_hc_rms_norm_f32")?;
    require_pipeline_geometry("PLE grouped RMSNorm", &grouped, SIMD_WIDTH, PLE_THREADS)?;
    let gate = ctx.pipeline("kernel_qwen4exp_ple_gate_f32")?;
    require_pipeline_geometry("PLE gate", &gate, SIMD_WIDTH, PLE_THREADS)?;
    let conv = ctx.pipeline("kernel_qwen4exp_ple_conv_epilogue_f32")?;
    if conv.maxTotalThreadsPerThreadgroup() < PLE_THREADS {
        return invalid(format!(
            "PLE convolution pipeline supports {} threads, needs {PLE_THREADS}",
            conv.maxTotalThreadsPerThreadgroup()
        ));
    }
    let scratch_bytes = (PLE_THREADS / SIMD_WIDTH) * size_of::<f32>();
    if ctx.device.maxThreadgroupMemoryLength() < scratch_bytes {
        return invalid(format!(
            "PLE reductions need {scratch_bytes} threadgroup bytes, device has {}",
            ctx.device.maxThreadgroupMemoryLength()
        ));
    }
    ctx.pipeline("kernel_copy_offset_f32")?;
    Ok(())
}

fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpPleMetalError> {
    let kernels: &[&str] = match dtype {
        GgmlType::F32 => &["kernel_mat_vec_f32_f32", "kernel_mat_vec_f32_f32_lcpp_r2"],
        GgmlType::Q8_0 => &["kernel_mat_vec_q8_0_f32", "kernel_mat_vec_q8_0_f32_lcpp"],
        _ => return invalid(format!("unsupported PLE projection dtype {dtype:?}")),
    };
    for kernel in kernels {
        ctx.pipeline(kernel)?;
    }
    Ok(())
}

fn require_pipeline_geometry(
    name: &str,
    pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
    execution_width: usize,
    threads: usize,
) -> Result<(), Qwen4ExpPleMetalError> {
    if pipeline.threadExecutionWidth() != execution_width
        || pipeline.maxTotalThreadsPerThreadgroup() < threads
    {
        return invalid(format!(
            "{name} needs execution width {execution_width} and {threads} threads, got width={} capacity={}",
            pipeline.threadExecutionWidth(),
            pipeline.maxTotalThreadsPerThreadgroup()
        ));
    }
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GroupedNormArgs {
    branch_count: u32,
    hidden_size: u32,
    eps: f32,
}

fn encode_grouped_rms_norm(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weight: &MetalTensor,
    output: &MetalTensor,
    geometry: Qwen4ExpPleMetalGeometry,
) -> Result<(), Qwen4ExpPleMetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_hc_rms_norm_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &GroupedNormArgs {
            branch_count: geometry.branch_count as u32,
            hidden_size: geometry.hidden_size as u32,
            eps: geometry.eps,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, output);
    enc.set_threadgroup_memory(0, (PLE_THREADS / SIMD_WIDTH) * size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: geometry.branch_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: PLE_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PleGateArgs {
    branch_count: u32,
    hidden_size: u32,
}

fn encode_gate(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    workspace: &Qwen4ExpPleMetalWorkspace,
    geometry: Qwen4ExpPleMetalGeometry,
) -> Result<(), Qwen4ExpPleMetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_ple_gate_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PleGateArgs {
            branch_count: geometry.branch_count as u32,
            hidden_size: geometry.hidden_size as u32,
        },
    );
    enc.set_tensor(1, &workspace.key_norm);
    enc.set_tensor(2, &workspace.query_norm);
    enc.set_tensor(3, &workspace.value);
    enc.set_tensor(4, &workspace.gated);
    enc.set_threadgroup_memory(0, (PLE_THREADS / SIMD_WIDTH) * size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: geometry.branch_count,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: PLE_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PleConvArgs {
    channels: u32,
    history_len: u32,
    kernel_size: u32,
    dilation: u32,
}

fn encode_conv_epilogue(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    residual: &MetalTensor,
    weight: &MetalTensor,
    workspace: &Qwen4ExpPleMetalWorkspace,
    geometry: Qwen4ExpPleMetalGeometry,
) -> Result<(), Qwen4ExpPleMetalError> {
    let pipeline = ctx.pipeline("kernel_qwen4exp_ple_conv_epilogue_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &PleConvArgs {
            channels: geometry.hyper_width() as u32,
            history_len: geometry.history_len() as u32,
            kernel_size: geometry.kernel_size as u32,
            dilation: geometry.dilation as u32,
        },
    );
    enc.set_tensor(1, &workspace.conv_input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, &workspace.conv_state);
    enc.set_tensor(4, residual);
    enc.set_tensor(5, &workspace.gated);
    enc.set_tensor(6, &workspace.conv_raw);
    enc.set_tensor(7, &workspace.output);
    enc.dispatch(
        MTLSize {
            width: geometry.hyper_width().div_ceil(PLE_THREADS),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: PLE_THREADS,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

fn validate_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), Qwen4ExpPleMetalError> {
    let command = enc.parent_command_buffer();
    let actual = command.device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("PLE dependent dispatches require a serial encoder");
    }
    let status = command.status();
    if status != MTLCommandBufferStatus::NotEnqueued {
        return invalid(format!(
            "PLE encoding requires a NotEnqueued command buffer, got {status:?}"
        ));
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpPleMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpPleMetalError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    workspace.encode_failed = false;
    Ok(())
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<(), Qwen4ExpPleMetalError> {
    let shape = [n_in as u64, n_out as u64];
    if tensor.shape != shape || !matches!(tensor.dtype, GgmlType::F32 | GgmlType::Q8_0) {
        return invalid(format!(
            "{name} must use F32 or Q8_0 with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    let (block, _) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpPleMetalError::Invalid(format!("{name} has unsupported dtype {:?}", tensor.dtype))
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
) -> Result<(), Qwen4ExpPleMetalError> {
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

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpPleMetalError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("tensor element count overflow".into()))?;
    let (block, bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpPleMetalError::Invalid(format!("unsupported dtype {:?}", tensor.dtype))
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
        .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpPleMetalError> {
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
        .ok_or_else(|| Qwen4ExpPleMetalError::Invalid(format!("{name} range overflow")))?;
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
) -> Result<(), Qwen4ExpPleMetalError> {
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
) -> Result<(), Qwen4ExpPleMetalError> {
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

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpPleMetalError> {
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

fn named_weight_tensors(weights: Qwen4ExpPleMetalWeights<'_>) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("PLE key projection", weights.key),
        ("PLE value projection", weights.value),
        ("PLE key norm", weights.key_norm),
        ("PLE query norm", weights.query_norm),
        ("PLE convolution norm", weights.conv_norm),
        ("PLE convolution", weights.conv),
    ]
}

fn workspace_tensors(workspace: &Qwen4ExpPleMetalWorkspace) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("PLE packed row staging", &workspace.packed_rows),
        ("PLE local row IDs", &workspace.local_row_ids),
        ("PLE embedding", &workspace.embedding),
        ("PLE key scratch", &workspace.key),
        ("PLE value scratch", &workspace.value),
        ("PLE normalized key", &workspace.key_norm),
        ("PLE normalized query", &workspace.query_norm),
        ("PLE gated value", &workspace.gated),
        ("PLE convolution input", &workspace.conv_input),
        ("PLE convolution state", &workspace.conv_state),
        ("PLE raw convolution output", &workspace.conv_raw),
        ("PLE output", &workspace.output),
    ]
}

fn write_tensor_bytes(tensor: &MetalTensor, bytes: &[u8]) -> Result<(), Qwen4ExpPleMetalError> {
    if !tensor.is_writable() {
        return invalid("PLE staging destination must be writable");
    }
    let expected = usize::try_from(storage_bytes(tensor)?).map_err(|_| {
        Qwen4ExpPleMetalError::Invalid("PLE staging byte count exceeds usize".into())
    })?;
    if bytes.len() != expected {
        return invalid(format!(
            "PLE staging source has {} bytes, expected {expected}",
            bytes.len()
        ));
    }
    let offset = usize::try_from(tensor.offset)
        .map_err(|_| Qwen4ExpPleMetalError::Invalid("PLE staging offset exceeds usize".into()))?;
    let end = offset
        .checked_add(expected)
        .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("PLE staging range overflow".into()))?;
    if end > tensor.buffer.length() {
        return invalid("PLE staging write exceeds its Metal buffer");
    }
    // SAFETY: the complete destination range is validated writable and idle,
    // and the source slice has exactly the destination byte count.
    unsafe {
        std::ptr::copy_nonoverlapping(
            bytes.as_ptr(),
            tensor.buffer.contents().as_ptr().cast::<u8>().add(offset),
            expected,
        );
    }
    Ok(())
}

fn zero_f32(ctx: &MetalContext, shape: Vec<u64>) -> Result<MetalTensor, Qwen4ExpPleMetalError> {
    let elements = shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("zero tensor shape overflow".into()))?;
    Ok(MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&vec![0.0_f32; elements]),
        shape,
        GgmlType::F32,
    )?)
}

fn zero_writable_f32(tensor: &MetalTensor) -> Result<(), Qwen4ExpPleMetalError> {
    if tensor.dtype != GgmlType::F32 || !tensor.is_writable() {
        return invalid("PLE state reset requires writable F32 storage");
    }
    let offset = usize::try_from(tensor.offset)
        .map_err(|_| Qwen4ExpPleMetalError::Invalid("state offset exceeds usize".into()))?;
    let bytes = usize::try_from(storage_bytes(tensor)?)
        .map_err(|_| Qwen4ExpPleMetalError::Invalid("state byte length exceeds usize".into()))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpPleMetalError::Invalid("state reset range overflow".into()))?;
    if end > tensor.buffer.length() {
        return invalid("state reset range exceeds its Metal buffer");
    }
    // SAFETY: validation proves the writable range is in bounds, and idle
    // ownership excludes in-flight GPU access.
    unsafe {
        std::ptr::write_bytes(
            tensor.buffer.contents().as_ptr().cast::<u8>().add(offset),
            0,
            bytes,
        );
    }
    Ok(())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpPleMetalError> {
    Err(Qwen4ExpPleMetalError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::qwen4exp::{PleHistory, Qwen4ExpConfig};
    use crate::qwen4exp_forward::{PleConvState, PleStepWeights, ple_step};
    use crate::tensor::TensorDesc;
    use objc2_metal::MTLCommandQueue;

    const BRANCHES: usize = 4;
    const HIDDEN: usize = 320;
    const HEADS: usize = 2;
    const HEAD_DIM: usize = 160;
    const HYPER: usize = BRANCHES * HIDDEN;
    const KERNEL: usize = 4;
    const DILATION: usize = 3;
    const HISTORY: usize = (KERNEL - 1) * DILATION;
    const TABLE_ROWS: usize = 23;

    struct SyntheticFixture {
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

    impl SyntheticFixture {
        fn new(ctx: &MetalContext) -> Self {
            let key_bytes = q8_matrix(HIDDEN, HYPER, 3);
            let value_bytes = q8_matrix(HIDDEN, HIDDEN, 17);
            let key_cpu = dequant_matrix(&key_bytes, GgmlType::Q8_0, HIDDEN, HYPER);
            let value_cpu = dequant_matrix(&value_bytes, GgmlType::Q8_0, HIDDEN, HIDDEN);
            let key_norm_cpu = values(HYPER, 5, 0.007, 1.0);
            let query_norm_cpu = values(HYPER, 11, 0.006, 0.95);
            let conv_norm_cpu = values(HYPER, 19, 0.005, 1.05);
            let conv_cpu = (0..HYPER)
                .flat_map(|channel| {
                    (0..KERNEL).map(move |tap| {
                        let magnitude = 0.015 + ((channel * 5 + tap * 7) % 9) as f32 * 0.002;
                        if (channel + tap).is_multiple_of(3) {
                            -magnitude
                        } else {
                            magnitude
                        }
                    })
                })
                .collect::<Vec<_>>();
            Self {
                key_cpu,
                value_cpu,
                key_norm_cpu: key_norm_cpu.clone(),
                query_norm_cpu: query_norm_cpu.clone(),
                conv_norm_cpu: conv_norm_cpu.clone(),
                conv_cpu: conv_cpu.clone(),
                key: quant_weight(
                    ctx,
                    &key_bytes,
                    vec![HIDDEN as u64, HYPER as u64],
                    GgmlType::Q8_0,
                ),
                value: quant_weight(
                    ctx,
                    &value_bytes,
                    vec![HIDDEN as u64, HIDDEN as u64],
                    GgmlType::Q8_0,
                ),
                key_norm: f32_weight(ctx, &key_norm_cpu, vec![HYPER as u64]),
                query_norm: f32_weight(ctx, &query_norm_cpu, vec![HYPER as u64]),
                conv_norm: f32_weight(ctx, &conv_norm_cpu, vec![HYPER as u64]),
                conv: f32_weight(ctx, &conv_cpu, vec![KERNEL as u64, HYPER as u64]),
            }
        }

        fn metal_weights(&self, geometry: Qwen4ExpPleMetalGeometry) -> Qwen4ExpPleMetalWeights<'_> {
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

        fn cpu_weights(&self) -> PleStepWeights<'_> {
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
        fn new(rows: usize, row_width: usize) -> Self {
            let blocks_per_row = row_width / 32;
            let mut bytes = Vec::with_capacity(rows * blocks_per_row * 18);
            for row in 0..rows {
                for block in 0..blocks_per_row {
                    let ordinal = row * blocks_per_row + block;
                    let sign = if ordinal.is_multiple_of(4) { -1.0 } else { 1.0 };
                    let scale = sign * (ordinal % 7 + 1) as f32 / 512.0;
                    bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                    for lane in 0..16 {
                        let low = (ordinal * 3 + lane * 5 + 1) % 16;
                        let high = (ordinal * 7 + lane * 11 + 2) % 16;
                        bytes.push(low as u8 | (high as u8) << 4);
                    }
                }
            }
            Self {
                desc: TensorDesc {
                    name: "synthetic_ple_table".into(),
                    shape: vec![row_width as u64, rows as u64],
                    dtype: GgmlType::IQ4_NL,
                    shard_idx: 0,
                    data_offset: 0,
                    n_bytes: bytes.len() as u64,
                },
                bytes,
            }
        }

        fn table(&self) -> PleIq4NlTable<'_> {
            PleIq4NlTable::new(&self.desc, &self.bytes, self.desc.shape[1]).unwrap()
        }
    }

    struct CpuIntermediates {
        key: Vec<f32>,
        value: Vec<f32>,
        key_norm: Vec<f32>,
        query_norm: Vec<f32>,
        gated: Vec<f32>,
        conv_input: Vec<f32>,
        conv_raw: Vec<f32>,
    }

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal initialization failed: {error}"),
        }
    }

    fn geometry() -> Qwen4ExpPleMetalGeometry {
        Qwen4ExpPleMetalGeometry::new(BRANCHES, HIDDEN, HEADS, HEAD_DIM, KERNEL, DILATION, 1e-6)
            .unwrap()
    }

    fn values(count: usize, seed: usize, scale: f32, bias: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let centered = ((index * 17 + seed * 13 + 5) % 31) as f32 - 15.0;
                bias + centered * scale
            })
            .collect()
    }

    fn q8_matrix(n_in: usize, n_out: usize, seed: usize) -> Vec<u8> {
        assert!(n_in.is_multiple_of(32));
        let blocks_per_row = n_in / 32;
        let mut bytes = Vec::with_capacity(n_out * blocks_per_row * 34);
        for row in 0..n_out {
            for block in 0..blocks_per_row {
                let ordinal = row * blocks_per_row + block + seed;
                let sign = if ordinal.is_multiple_of(5) { -1.0 } else { 1.0 };
                let scale = sign * (ordinal % 7 + 1) as f32 / 2_048.0;
                bytes.extend_from_slice(&half::f16::from_f32(scale).to_bits().to_le_bytes());
                for lane in 0..32 {
                    let quant = ((ordinal * 11 + lane * 7 + 3) % 31) as i8 - 15;
                    bytes.push(quant as u8);
                }
            }
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

    fn quant_weight(
        ctx: &MetalContext,
        bytes: &[u8],
        shape: Vec<u64>,
        dtype: GgmlType,
    ) -> MetalTensor {
        let mut tensor = MetalTensor::from_bytes(ctx, bytes, shape, dtype).unwrap();
        tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
    }

    fn f32_weight(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        quant_weight(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32)
    }

    fn f32_tensor(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
    }

    fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
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

    fn read_bytes(tensor: &MetalTensor) -> Vec<u8> {
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize);
            std::slice::from_raw_parts(source, tensor.n_bytes() as usize).to_vec()
        }
    }

    fn mat_vec(weight: &[f32], input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        assert_eq!(weight.len(), n_in * n_out);
        weight
            .chunks_exact(n_in)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(left, right)| left * right)
                    .sum()
            })
            .collect()
    }

    fn grouped_norm(
        input: &[f32],
        weight: &[f32],
        branches: usize,
        hidden: usize,
        eps: f32,
    ) -> Vec<f32> {
        let mut output = vec![0.0_f32; branches * hidden];
        for branch in 0..branches {
            let base = branch * hidden;
            let sum_square = input[base..base + hidden]
                .iter()
                .map(|value| value * value)
                .sum::<f32>();
            let scale = 1.0 / (sum_square / hidden as f32 + eps).sqrt();
            for index in 0..hidden {
                output[base + index] = input[base + index] * scale * weight[base + index];
            }
        }
        output
    }

    fn cpu_intermediates(
        embedding: &[f32],
        hyper: &[f32],
        weights: PleStepWeights<'_>,
        state: &PleConvState,
    ) -> CpuIntermediates {
        let key = mat_vec(weights.key, embedding, HIDDEN, HYPER);
        let value = mat_vec(weights.value, embedding, HIDDEN, HIDDEN);
        let key_norm = grouped_norm(&key, weights.key_norm, BRANCHES, HIDDEN, 1e-6);
        let query_norm = grouped_norm(hyper, weights.query_norm, BRANCHES, HIDDEN, 1e-6);
        let mut gated = vec![0.0_f32; HYPER];
        for branch in 0..BRANCHES {
            let base = branch * HIDDEN;
            let score = key_norm[base..base + HIDDEN]
                .iter()
                .zip(&query_norm[base..base + HIDDEN])
                .map(|(left, right)| left * right)
                .sum::<f32>()
                / (HIDDEN as f32).sqrt();
            let sign = if score > 0.0 {
                1.0
            } else if score < 0.0 {
                -1.0
            } else {
                0.0
            };
            let gate = 1.0 / (1.0 + (-(sign * score.abs().max(1e-6).sqrt())).exp());
            for hidden in 0..HIDDEN {
                gated[base + hidden] = value[hidden] * gate;
            }
        }
        let conv_input = grouped_norm(&gated, weights.conv_norm, BRANCHES, HIDDEN, 1e-6);
        let state_values = state.values();
        let mut conv_raw = vec![0.0_f32; HYPER];
        for channel in 0..HYPER {
            let state_base = channel * HISTORY;
            for tap in 0..KERNEL {
                let lag = (KERNEL - 1 - tap) * DILATION;
                let value = if lag == 0 {
                    conv_input[channel]
                } else {
                    state_values[state_base + HISTORY - lag]
                };
                conv_raw[channel] += weights.conv[channel * KERNEL + tap] * value;
            }
        }
        CpuIntermediates {
            key,
            value,
            key_norm,
            query_norm,
            gated,
            conv_input,
            conv_raw,
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
        max_abs: f32,
        minimum_cosine: f64,
    ) {
        assert_eq!(actual.len(), expected.len(), "{label} length");
        assert!(actual.iter().all(|value| value.is_finite()), "{label}");
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
    fn release_geometry_derives_dilation_from_ngram_order() {
        let config = Qwen4ExpConfig::flash_next_reference();
        let geometry = Qwen4ExpPleMetalGeometry::from_config(&config, 1).unwrap();
        assert_eq!(geometry.branch_count(), 4);
        assert_eq!(geometry.hidden_size(), 2_560);
        assert_eq!(geometry.head_count(), 16);
        assert_eq!(geometry.head_dim(), 160);
        assert_eq!(geometry.kernel_size(), 4);
        assert_eq!(geometry.dilation(), 3);
        assert_eq!(geometry.history_len(), 9);
        assert!(Qwen4ExpPleMetalGeometry::from_config(&config, 0).is_err());
    }

    #[test]
    fn metal_gate_preserves_positive_negative_and_zero_scores() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let geometry = geometry();
        let workspace = Qwen4ExpPleMetalWorkspace::new(&ctx, geometry).unwrap();
        let mut key = vec![0.0_f32; HYPER];
        let mut query = vec![0.0_f32; HYPER];
        let value = (0..HIDDEN)
            .map(|index| (index + 1) as f32 / 512.0)
            .collect::<Vec<_>>();
        for (branch, query_value) in [3.0_f32, -3.0, 0.0, -3.0].into_iter().enumerate() {
            key[branch * HIDDEN] = if branch == 3 { -2.0 } else { 2.0 };
            query[branch * HIDDEN] = query_value;
        }
        write_tensor_bytes(&workspace.key_norm, bytemuck::cast_slice(&key)).unwrap();
        write_tensor_bytes(&workspace.query_norm, bytemuck::cast_slice(&query)).unwrap();
        write_tensor_bytes(&workspace.value, bytemuck::cast_slice(&value)).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_gate(&ctx, &encoder, &workspace, geometry).unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert_eq!(command.status(), MTLCommandBufferStatus::Completed);
        assert!(command.error().is_none());

        let actual = read_f32(&workspace.gated);
        let positive_score = 6.0 / (HIDDEN as f32).sqrt();
        let positive_gate = 1.0 / (1.0 + (-positive_score.sqrt()).exp());
        let negative_gate = 1.0 / (1.0 + positive_score.sqrt().exp());
        for hidden in 0..HIDDEN {
            assert_close(
                "positive PLE gate",
                &[actual[hidden]],
                &[value[hidden] * positive_gate],
                2e-7,
                2e-7,
            );
            assert_close(
                "negative PLE gate",
                &[actual[HIDDEN + hidden]],
                &[value[hidden] * negative_gate],
                2e-7,
                2e-7,
            );
            assert_eq!(
                actual[2 * HIDDEN + hidden].to_bits(),
                (value[hidden] * 0.5).to_bits(),
                "zero PLE gate at hidden {hidden}"
            );
            assert_close(
                "double-negative PLE gate",
                &[actual[3 * HIDDEN + hidden]],
                &[value[hidden] * positive_gate],
                2e-7,
                2e-7,
            );
        }
    }

    #[test]
    fn malformed_contract_does_not_reserve_or_consume_staging() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let geometry = geometry();
        let fixture = SyntheticFixture::new(&ctx);
        let table_fixture = TableFixture::new(TABLE_ROWS, HEAD_DIM);
        let table = table_fixture.table();
        let mut workspace = Qwen4ExpPleMetalWorkspace::new(&ctx, geometry).unwrap();
        let input = f32_tensor(&ctx, &vec![0.25; HYPER], vec![HYPER as u64]);
        workspace.stage_rows(table, &[2, 7]).unwrap();

        let mut writable_key = fixture.key.clone();
        writable_key.provenance = MetalTensorProvenance::OwnedWritable;
        let mut malformed_weights = fixture.metal_weights(geometry);
        malformed_weights.key = &writable_key;
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_qwen4exp_ple(&ctx, &encoder, &input, malformed_weights, &mut workspace,)
                .is_err()
        );
        encoder.end();
        drop(command);
        assert!(workspace.active_command.is_none());
        assert!(workspace.has_staged_rows());
        assert!(!workspace.is_poisoned());

        let original_output = std::mem::replace(&mut workspace.output, input.clone());
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        assert!(
            encode_qwen4exp_ple(
                &ctx,
                &encoder,
                &input,
                fixture.metal_weights(geometry),
                &mut workspace,
            )
            .is_err()
        );
        encoder.end();
        workspace.output = original_output;
        assert!(workspace.active_command.is_none());
        assert!(workspace.has_staged_rows());
        assert!(!workspace.is_poisoned());
    }

    #[test]
    fn staged_q8_ple_matches_cpu_through_all_dilated_lags() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let geometry = geometry();
        let fixture = SyntheticFixture::new(&ctx);
        let table_fixture = TableFixture::new(TABLE_ROWS, HEAD_DIM);
        let table = table_fixture.table();
        let mut cpu_state = PleConvState::fresh(HYPER, KERNEL, DILATION).unwrap();
        let mut workspace = Qwen4ExpPleMetalWorkspace::new(&ctx, geometry).unwrap();
        let exported = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();

        for step in 0..12 {
            let row_ids = [step % TABLE_ROWS, (step * 7 + 5) % TABLE_ROWS].map(|row| row as u32);
            let mut embedding = vec![0.0_f32; HIDDEN];
            table.gather_f32_into(&row_ids, &mut embedding).unwrap();
            let hyper = values(HYPER, step * 23 + 7, 0.013, -0.04 + step as f32 * 0.002);
            let intermediates =
                cpu_intermediates(&embedding, &hyper, fixture.cpu_weights(), &cpu_state);
            let expected = ple_step(
                &embedding,
                &hyper,
                BRANCHES,
                HIDDEN,
                KERNEL,
                DILATION,
                1e-6,
                fixture.cpu_weights(),
                &mut cpu_state,
            )
            .unwrap();

            workspace.stage_rows(table, &row_ids).unwrap();
            let input = f32_tensor(&ctx, &hyper, vec![HYPER as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_qwen4exp_ple(
                &ctx,
                &encoder,
                &input,
                fixture.metal_weights(geometry),
                &mut workspace,
            )
            .unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &exported)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            workspace.release_after().unwrap();

            assert_close(
                "embedding",
                &read_f32(&workspace.embedding),
                &embedding,
                0.0,
                0.0,
            );
            assert_close(
                "key projection",
                &read_f32(&workspace.key),
                &intermediates.key,
                2e-5,
                2e-5,
            );
            assert_close(
                "value projection",
                &read_f32(&workspace.value),
                &intermediates.value,
                2e-5,
                2e-5,
            );
            assert_close(
                "normalized key",
                &read_f32(&workspace.key_norm),
                &intermediates.key_norm,
                8e-5,
                5e-5,
            );
            assert_close(
                "normalized query",
                &read_f32(&workspace.query_norm),
                &intermediates.query_norm,
                8e-5,
                5e-5,
            );
            assert_close(
                "gated value",
                &read_f32(&workspace.gated),
                &intermediates.gated,
                8e-5,
                5e-5,
            );
            assert_close(
                "convolution input",
                &read_f32(&workspace.conv_input),
                &intermediates.conv_input,
                1e-4,
                8e-5,
            );
            assert_close(
                "raw convolution",
                &read_f32(&workspace.conv_raw),
                &intermediates.conv_raw,
                1e-4,
                8e-5,
            );
            assert_close(
                "convolution state",
                &read_f32(&workspace.conv_state),
                &cpu_state.values(),
                1e-4,
                8e-5,
            );
            assert_close("PLE output", &read_f32(&exported), &expected, 2e-4, 1e-4);
        }
        assert!(
            cpu_state
                .values()
                .chunks_exact(HISTORY)
                .all(|history| { history[0] != 0.0 && history[HISTORY - DILATION] != 0.0 })
        );
    }

    #[test]
    fn staging_and_causal_state_obey_command_ownership() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let geometry = geometry();
        let fixture = SyntheticFixture::new(&ctx);
        let table_fixture = TableFixture::new(TABLE_ROWS, HEAD_DIM);
        let table = table_fixture.table();
        let mut workspace = Qwen4ExpPleMetalWorkspace::new(&ctx, geometry).unwrap();
        let hyper = values(HYPER, 41, 0.01, 0.03);
        let input = f32_tensor(&ctx, &hyper, vec![HYPER as u64]);
        let exported = MetalTensor::zeros_f32(&ctx, vec![HYPER as u64]).unwrap();
        let rows = [3_u32, 11];

        workspace.stage_rows(table, &rows).unwrap();
        let staged = read_bytes(&workspace.packed_rows);
        assert!(matches!(
            workspace.stage_rows(table, &[3, TABLE_ROWS as u32]),
            Err(Qwen4ExpPleMetalError::Gather(
                PleGatherError::RowOutOfRange { .. }
            ))
        ));
        assert_eq!(read_bytes(&workspace.packed_rows), staged);
        assert!(!workspace.has_staged_rows());
        workspace.stage_rows(table, &rows).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_ple(
            &ctx,
            &encoder,
            &input,
            fixture.metal_weights(geometry),
            &mut workspace,
        )
        .unwrap();
        let foreign_command = ctx.queue.commandBuffer().unwrap();
        let foreign_encoder = KernelEncoder::begin(&foreign_command);
        assert!(
            read.output()
                .encode_copy_to(&ctx, &foreign_encoder, &exported)
                .is_err()
        );
        foreign_encoder.end();
        drop(read);
        assert!(workspace.release_after().is_err());
        assert!(workspace.stage_rows(table, &rows).is_err());
        let second_command = ctx.queue.commandBuffer().unwrap();
        let second_encoder = KernelEncoder::begin(&second_command);
        assert!(
            encode_qwen4exp_ple(
                &ctx,
                &second_encoder,
                &input,
                fixture.metal_weights(geometry),
                &mut workspace,
            )
            .is_err()
        );
        second_encoder.end();
        encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        drop(command);
        assert!(
            read_f32(&workspace.conv_state)
                .iter()
                .all(|&value| value == 0.0)
        );

        workspace.stage_rows(table, &rows).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read = encode_qwen4exp_ple(
            &ctx,
            &encoder,
            &input,
            fixture.metal_weights(geometry),
            &mut workspace,
        )
        .unwrap();
        read.output()
            .encode_copy_to(&ctx, &encoder, &exported)
            .unwrap();
        drop(read);
        encoder.end();
        command.commit();
        assert!(workspace.stage_rows(table, &rows).is_err());
        assert!(unsafe { workspace.abandon_uncommitted() }.is_err());
        workspace.release_after().unwrap();
        assert!(
            read_f32(&workspace.conv_state)
                .iter()
                .any(|&value| value != 0.0)
        );

        workspace.state_poisoned = true;
        assert!(workspace.stage_rows(table, &rows).is_err());
        workspace.reset().unwrap();
        assert!(!workspace.is_poisoned());
        assert!(!workspace.has_staged_rows());
        assert!(
            read_f32(&workspace.conv_state)
                .iter()
                .all(|&value| value == 0.0)
        );
    }

    fn gguf_dequant(gguf: &GgufFile, name: &str) -> Vec<f32> {
        let desc = gguf.find(name).unwrap_or_else(|| panic!("missing {name}"));
        crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
    }

    fn gguf_weight(ctx: &MetalContext, gguf: &GgufFile, name: &str) -> MetalTensor {
        let desc = gguf.find(name).unwrap_or_else(|| panic!("missing {name}"));
        let mut tensor =
            MetalTensor::from_gguf_tensor(ctx, desc, gguf.try_slice(desc).unwrap()).unwrap();
        tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_PLE_GGUF to the pinned first release shard"]
    fn released_ple_matches_cpu_across_complete_dilated_history() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_PLE_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_PLE_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let config = Qwen4ExpConfig::from_gguf(&gguf).unwrap();
        let ple = config.ple.as_ref().unwrap();
        let geometry = Qwen4ExpPleMetalGeometry::from_config(&config, 1).unwrap();
        let table_desc = gguf.find("per_layer_token_embd.weight").unwrap();
        let table = PleIq4NlTable::new(
            table_desc,
            gguf.try_slice(table_desc).unwrap(),
            ple.logical_row_count().unwrap(),
        )
        .unwrap();
        let ctx = MetalContext::new().expect("initialize Metal");

        let key = gguf_weight(&ctx, &gguf, "blk.1.ple_key.weight");
        let value = gguf_weight(&ctx, &gguf, "blk.1.ple_value.weight");
        let key_norm = gguf_weight(&ctx, &gguf, "blk.1.ple_norm_key.weight");
        let query_norm = gguf_weight(&ctx, &gguf, "blk.1.ple_norm_query.weight");
        let conv_norm = gguf_weight(&ctx, &gguf, "blk.1.ple_norm_conv.weight");
        let conv = gguf_weight(&ctx, &gguf, "blk.1.ple_conv1d.weight");
        assert_eq!(key.dtype, GgmlType::Q8_0);
        assert_eq!(value.dtype, GgmlType::Q8_0);
        let metal_weights = Qwen4ExpPleMetalWeights {
            geometry,
            key: &key,
            value: &value,
            key_norm: &key_norm,
            query_norm: &query_norm,
            conv_norm: &conv_norm,
            conv: &conv,
        };
        let key_cpu = gguf_dequant(&gguf, "blk.1.ple_key.weight");
        let value_cpu = gguf_dequant(&gguf, "blk.1.ple_value.weight");
        let key_norm_cpu = gguf_dequant(&gguf, "blk.1.ple_norm_key.weight");
        let query_norm_cpu = gguf_dequant(&gguf, "blk.1.ple_norm_query.weight");
        let conv_norm_cpu = gguf_dequant(&gguf, "blk.1.ple_norm_conv.weight");
        let conv_cpu = gguf_dequant(&gguf, "blk.1.ple_conv1d.weight");
        let cpu_weights = PleStepWeights {
            key: &key_cpu,
            value: &value_cpu,
            key_norm: &key_norm_cpu,
            query_norm: &query_norm_cpu,
            conv_norm: &conv_norm_cpu,
            conv: &conv_cpu,
        };

        let mut history = PleHistory::default();
        let mut cpu_state = PleConvState::fresh(
            geometry.hyper_width(),
            geometry.kernel_size(),
            geometry.dilation(),
        )
        .unwrap();
        let mut workspace = Qwen4ExpPleMetalWorkspace::new(&ctx, geometry).unwrap();
        let exported = MetalTensor::zeros_f32(&ctx, vec![geometry.hyper_width() as u64]).unwrap();
        let tokens = [35_u32, 201, 42, 17, 91, 7, 333, 19, 88, 1_024];

        for (position, token) in tokens.into_iter().enumerate() {
            let (next_history, rows) = history.advanced(ple, token, position as u64).unwrap();
            let mut embedding = vec![0.0_f32; geometry.hidden_size()];
            table.gather_f32_into(&rows, &mut embedding).unwrap();
            let hyper = values(
                geometry.hyper_width(),
                position * 29 + 13,
                0.009,
                position as f32 * 0.001 - 0.03,
            );
            let expected = ple_step(
                &embedding,
                &hyper,
                geometry.branch_count(),
                geometry.hidden_size(),
                geometry.kernel_size(),
                geometry.dilation(),
                geometry.eps(),
                cpu_weights,
                &mut cpu_state,
            )
            .unwrap();

            workspace.stage_rows(table, &rows).unwrap();
            let input = f32_tensor(&ctx, &hyper, vec![geometry.hyper_width() as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read =
                encode_qwen4exp_ple(&ctx, &encoder, &input, metal_weights, &mut workspace).unwrap();
            read.output()
                .encode_copy_to(&ctx, &encoder, &exported)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            workspace.release_after().unwrap();
            history = next_history;

            assert_close(
                "real staged embedding",
                &read_f32(&workspace.embedding),
                &embedding,
                0.0,
                0.0,
            );
            assert_similarity(
                "real PLE output",
                &read_f32(&exported),
                &expected,
                1e-5,
                0.999_999_9,
            );
            assert_similarity(
                "real PLE convolution state",
                &read_f32(&workspace.conv_state),
                &cpu_state.values(),
                5e-5,
                0.999_999_9,
            );
        }
    }
}
