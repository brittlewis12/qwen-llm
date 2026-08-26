//! One-token Metal Gated DeltaNet execution for Qwen3.8-Flash-Next.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_copy_offset_f32, encode_gdn_decay_chain_f32, encode_gdn_step_decay_f32,
    encode_l2_norm_pair_batched_f32, encode_mat_vec_f32_sigmoid, encode_ssm_conv_silu_f32,
};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::qwen4exp::{MixerKind, Qwen4ExpConfig};
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLDevice, MTLResource, MTLSize,
};

const REQUIRED_HEAD_DIM: usize = 128;
const REQUIRED_CONV_KERNEL: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpGdnError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next GDN contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next GDN command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GatedDeltaNetMetalGeometry {
    hidden_size: usize,
    key_heads: usize,
    value_heads: usize,
    head_dim: usize,
    conv_kernel: usize,
    eps: f32,
}

impl GatedDeltaNetMetalGeometry {
    pub fn new(
        hidden_size: usize,
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        conv_kernel: usize,
        eps: f32,
    ) -> Result<Self, Qwen4ExpGdnError> {
        let geometry = Self {
            hidden_size,
            key_heads,
            value_heads,
            head_dim,
            conv_kernel,
            eps,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(config: &Qwen4ExpConfig) -> Result<Self, Qwen4ExpGdnError> {
        if config.gated_delta_net.key_head_dim != config.gated_delta_net.value_head_dim {
            return invalid("GDN key and value head dimensions must match");
        }
        Self::new(
            config.hidden_size as usize,
            config.gated_delta_net.key_heads as usize,
            config.gated_delta_net.value_heads as usize,
            config.gated_delta_net.key_head_dim as usize,
            config.gated_delta_net.conv_kernel as usize,
            config.rms_norm_eps,
        )
    }

    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub fn key_heads(self) -> usize {
        self.key_heads
    }

    pub fn value_heads(self) -> usize {
        self.value_heads
    }

    pub fn head_dim(self) -> usize {
        self.head_dim
    }

    pub fn conv_kernel(self) -> usize {
        self.conv_kernel
    }

    pub fn eps(self) -> f32 {
        self.eps
    }

    pub fn key_width(self) -> usize {
        self.key_heads
            .checked_mul(self.head_dim)
            .expect("validated GDN key width")
    }

    pub fn value_width(self) -> usize {
        self.value_heads
            .checked_mul(self.head_dim)
            .expect("validated GDN value width")
    }

    pub fn conv_width(self) -> usize {
        self.key_width()
            .checked_mul(2)
            .and_then(|width| width.checked_add(self.value_width()))
            .expect("validated GDN convolution width")
    }

    pub fn conv_state_elements(self) -> usize {
        (self.conv_kernel - 1)
            .checked_mul(self.conv_width())
            .expect("validated GDN convolution state")
    }

    pub fn delta_state_elements(self) -> usize {
        self.value_heads
            .checked_mul(self.head_dim)
            .and_then(|elements| elements.checked_mul(self.head_dim))
            .expect("validated GDN delta state")
    }

    fn validate(self) -> Result<(), Qwen4ExpGdnError> {
        if self.hidden_size == 0 || self.key_heads == 0 || self.value_heads == 0 {
            return invalid("hidden size and head counts must be nonzero");
        }
        if self.head_dim != REQUIRED_HEAD_DIM {
            return invalid(format!(
                "GDN head dimension must be {REQUIRED_HEAD_DIM}, got {}",
                self.head_dim
            ));
        }
        if self.conv_kernel != REQUIRED_CONV_KERNEL {
            return invalid(format!(
                "GDN convolution width must be {REQUIRED_CONV_KERNEL}, got {}",
                self.conv_kernel
            ));
        }
        if !self.value_heads.is_multiple_of(self.key_heads) {
            return invalid("value-head count must be a multiple of key-head count");
        }
        if !self.eps.is_finite() || self.eps <= 0.0 {
            return invalid("GDN epsilon must be finite and positive");
        }
        let key_width = self
            .key_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Qwen4ExpGdnError::Invalid("GDN key width overflow".into()))?;
        let value_width = self
            .value_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Qwen4ExpGdnError::Invalid("GDN value width overflow".into()))?;
        let conv_width = key_width
            .checked_mul(2)
            .and_then(|width| width.checked_add(value_width))
            .ok_or_else(|| Qwen4ExpGdnError::Invalid("GDN convolution width overflow".into()))?;
        let conv_state = (self.conv_kernel - 1)
            .checked_mul(conv_width)
            .ok_or_else(|| Qwen4ExpGdnError::Invalid("GDN convolution state overflow".into()))?;
        let delta_state = value_width
            .checked_mul(self.head_dim)
            .ok_or_else(|| Qwen4ExpGdnError::Invalid("GDN delta state overflow".into()))?;
        for (name, value) in [
            ("hidden size", self.hidden_size),
            ("key heads", self.key_heads),
            ("value heads", self.value_heads),
            ("key width", key_width),
            ("value width", value_width),
            ("convolution width", conv_width),
            ("convolution state", conv_state),
            ("delta state", delta_state),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("{name} {value} exceeds u32"));
            }
        }
        for (name, elements) in [
            ("QKV projection", self.hidden_size.checked_mul(conv_width)),
            ("gate projection", self.hidden_size.checked_mul(value_width)),
            (
                "head projection",
                self.hidden_size.checked_mul(self.value_heads),
            ),
            (
                "output projection",
                value_width.checked_mul(self.hidden_size),
            ),
            (
                "convolution weights",
                conv_width.checked_mul(self.conv_kernel),
            ),
        ] {
            let elements = elements.ok_or_else(|| {
                Qwen4ExpGdnError::Invalid(format!("GDN {name} element count overflow"))
            })?;
            elements.checked_mul(4).ok_or_else(|| {
                Qwen4ExpGdnError::Invalid(format!("GDN {name} F32 byte count overflow"))
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct GatedDeltaNetMetalWeights<'a> {
    pub geometry: GatedDeltaNetMetalGeometry,
    pub qkv: &'a MetalTensor,
    pub gate: &'a MetalTensor,
    pub beta: &'a MetalTensor,
    pub alpha: &'a MetalTensor,
    pub a: &'a MetalTensor,
    pub dt_bias: &'a MetalTensor,
    pub conv: &'a MetalTensor,
    pub norm: &'a MetalTensor,
    pub output: &'a MetalTensor,
}

impl<'a> GatedDeltaNetMetalWeights<'a> {
    pub fn bind(weights: &'a Qwen4ExpMetalWeights, layer: u32) -> Result<Self, Qwen4ExpGdnError> {
        if weights.config().mixer_kind(layer) != Some(MixerKind::GatedDeltaNet) {
            return invalid(format!("layer {layer} is not a Gated DeltaNet layer"));
        }
        let prefix = format!("blk.{layer}");
        Ok(Self {
            geometry: GatedDeltaNetMetalGeometry::from_config(weights.config())?,
            qkv: weights.require_tensor(&format!("{prefix}.attn_qkv.weight"))?,
            gate: weights.require_tensor(&format!("{prefix}.attn_gate.weight"))?,
            beta: weights.require_tensor(&format!("{prefix}.ssm_beta.weight"))?,
            alpha: weights.require_tensor(&format!("{prefix}.ssm_alpha.weight"))?,
            a: weights.require_tensor(&format!("{prefix}.ssm_a"))?,
            dt_bias: weights.require_tensor(&format!("{prefix}.ssm_dt.bias"))?,
            conv: weights.require_tensor(&format!("{prefix}.ssm_conv1d.weight"))?,
            norm: weights.require_tensor(&format!("{prefix}.ssm_norm.weight"))?,
            output: weights.require_tensor(&format!("{prefix}.ssm_out.weight"))?,
        })
    }
}

pub struct GatedDeltaNetMetalWorkspace {
    geometry: GatedDeltaNetMetalGeometry,
    conv_state: MetalTensor,
    delta_state: MetalTensor,
    qkv: MetalTensor,
    qkv_conv: MetalTensor,
    gate: MetalTensor,
    beta: MetalTensor,
    alpha: MetalTensor,
    decay: MetalTensor,
    query_norm: MetalTensor,
    key_norm: MetalTensor,
    recurrent: MetalTensor,
    normalized: MetalTensor,
    output: MetalTensor,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
}

impl GatedDeltaNetMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: GatedDeltaNetMetalGeometry,
    ) -> Result<Self, Qwen4ExpGdnError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            conv_state: zero_f32(ctx, vec![geometry.conv_state_elements() as u64])?,
            delta_state: zero_f32(ctx, vec![geometry.delta_state_elements() as u64])?,
            qkv: MetalTensor::zeros_f32(ctx, vec![geometry.conv_width() as u64])?,
            qkv_conv: MetalTensor::zeros_f32(ctx, vec![geometry.conv_width() as u64])?,
            gate: MetalTensor::zeros_f32(ctx, vec![geometry.value_width() as u64])?,
            beta: MetalTensor::zeros_f32(ctx, vec![geometry.value_heads as u64])?,
            alpha: MetalTensor::zeros_f32(ctx, vec![geometry.value_heads as u64])?,
            decay: MetalTensor::zeros_f32(ctx, vec![geometry.value_heads as u64])?,
            query_norm: MetalTensor::zeros_f32(ctx, vec![geometry.key_width() as u64])?,
            key_norm: MetalTensor::zeros_f32(ctx, vec![geometry.key_width() as u64])?,
            recurrent: MetalTensor::zeros_f32(ctx, vec![geometry.value_width() as u64])?,
            normalized: MetalTensor::zeros_f32(ctx, vec![geometry.value_width() as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            active_command: None,
            state_poisoned: false,
        })
    }

    pub fn geometry(&self) -> GatedDeltaNetMetalGeometry {
        self.geometry
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpGdnError> {
        self.require_idle()?;
        zero_writable_tensor(&self.conv_state)?;
        zero_writable_tensor(&self.delta_state)?;
        self.state_poisoned = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpGdnError> {
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
        if status == MTLCommandBufferStatus::Completed && error.is_none() {
            Ok(())
        } else {
            self.state_poisoned = true;
            Err(Qwen4ExpGdnError::CommandBuffer(format!(
                "status={status:?}, error={error:?}"
            )))
        }
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command buffer. Committing it after this call may race a later
    /// workspace owner or mutate causal state out of order.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpGdnError> {
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
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpGdnError> {
        if self.active_command.is_some() {
            invalid("workspace state is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "consume the GDN output in the same command, then release its workspace"]
pub struct GatedDeltaNetMetalRead<'a> {
    workspace: &'a mut GatedDeltaNetMetalWorkspace,
}

pub struct GatedDeltaNetMetalOutput<'a> {
    workspace: &'a GatedDeltaNetMetalWorkspace,
}

impl GatedDeltaNetMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.output.n_elements()
    }

    pub fn dtype(&self) -> GgmlType {
        self.workspace.output.dtype
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpGdnError> {
        let encoder_device = enc.parent_command_buffer().device().registryID();
        if encoder_device != ctx.device.registryID() {
            return invalid(format!(
                "encoder belongs to Metal device registry {encoder_device}, context is {}",
                ctx.device.registryID()
            ));
        }
        if enc.is_concurrent() {
            return invalid("dependent GDN output copies require a serial encoder");
        }
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("GDN output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("GDN output must be copied by its owning command buffer");
        }
        require_f32(
            "GDN copied output destination",
            destination,
            self.workspace.geometry.hidden_size,
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("GDN output", &self.workspace.output),
                ("GDN copied output destination", destination),
            ],
        )?;
        require_disjoint(&[
            ("GDN copied output destination", destination),
            ("GDN convolution state", &self.workspace.conv_state),
            ("GDN delta state", &self.workspace.delta_state),
            ("GDN QKV scratch", &self.workspace.qkv),
            ("GDN convolved QKV", &self.workspace.qkv_conv),
            ("GDN gate scratch", &self.workspace.gate),
            ("GDN beta scratch", &self.workspace.beta),
            ("GDN alpha scratch", &self.workspace.alpha),
            ("GDN decay scratch", &self.workspace.decay),
            ("GDN query norm", &self.workspace.query_norm),
            ("GDN key norm", &self.workspace.key_norm),
            ("GDN recurrent output", &self.workspace.recurrent),
            ("GDN normalized output", &self.workspace.normalized),
            ("GDN output", &self.workspace.output),
        ])?;
        ctx.pipeline("kernel_copy_offset_f32")?;
        encode_copy_offset_f32(
            ctx,
            enc,
            &self.workspace.output,
            0,
            destination,
            self.workspace.geometry.hidden_size,
        )?;
        Ok(())
    }
}

impl GatedDeltaNetMetalRead<'_> {
    pub fn output(&self) -> GatedDeltaNetMetalOutput<'_> {
        GatedDeltaNetMetalOutput {
            workspace: self.workspace,
        }
    }
}

pub fn encode_gated_delta_net<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: GatedDeltaNetMetalWeights<'_>,
    workspace: &'a mut GatedDeltaNetMetalWorkspace,
) -> Result<GatedDeltaNetMetalRead<'a>, Qwen4ExpGdnError> {
    let encoder_device = enc.parent_command_buffer().device().registryID();
    if encoder_device != ctx.device.registryID() {
        return invalid(format!(
            "encoder belongs to Metal device registry {encoder_device}, context is {}",
            ctx.device.registryID()
        ));
    }
    if enc.is_concurrent() {
        return invalid("dependent GDN dispatches require a serial encoder");
    }
    if workspace.state_poisoned {
        return invalid("workspace causal state is indeterminate; reset it before reuse");
    }
    if weights.geometry != workspace.geometry {
        return invalid("GDN weight and workspace geometry differ");
    }
    validate_contract(ctx, input, weights, workspace)?;
    preflight(ctx, weights)?;
    reserve_command(workspace, enc)?;

    let geometry = workspace.geometry;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.qkv,
        input,
        &workspace.qkv,
        geometry.hidden_size,
        geometry.conv_width(),
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.gate,
        input,
        &workspace.gate,
        geometry.hidden_size,
        geometry.value_width(),
    )?;
    encode_mat_vec_f32_sigmoid(
        ctx,
        enc,
        weights.beta,
        input,
        &workspace.beta,
        geometry.hidden_size,
        geometry.value_heads,
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.alpha,
        input,
        &workspace.alpha,
        geometry.hidden_size,
        geometry.value_heads,
    )?;
    encode_gdn_decay_chain_f32(
        ctx,
        enc,
        &workspace.alpha,
        weights.dt_bias,
        weights.a,
        &workspace.decay,
    )?;
    encode_ssm_conv_silu_f32(
        ctx,
        enc,
        &workspace.qkv,
        &workspace.conv_state,
        weights.conv,
        &workspace.qkv_conv,
        geometry.conv_width(),
    )?;

    let query = workspace
        .qkv_conv
        .view_subrange(0, vec![geometry.key_width() as u64]);
    let key = workspace.qkv_conv.view_subrange(
        geometry.key_width() as u64,
        vec![geometry.key_width() as u64],
    );
    let value = workspace.qkv_conv.view_subrange(
        (2 * geometry.key_width()) as u64,
        vec![geometry.value_width() as u64],
    );
    encode_l2_norm_pair_batched_f32(
        ctx,
        enc,
        &query,
        &workspace.query_norm,
        &key,
        &workspace.key_norm,
        geometry.key_heads,
        geometry.head_dim,
        geometry.eps,
    )?;
    encode_gdn_step_decay_f32(
        ctx,
        enc,
        &workspace.query_norm,
        &workspace.key_norm,
        &value,
        &workspace.decay,
        &workspace.beta,
        &workspace.delta_state,
        &workspace.recurrent,
        geometry.value_heads,
        geometry.key_heads,
        geometry.head_dim,
    )?;
    encode_rmsnorm_sigmoid_gated(
        ctx,
        enc,
        &workspace.recurrent,
        weights.norm,
        &workspace.gate,
        &workspace.normalized,
        geometry,
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.output,
        &workspace.normalized,
        &workspace.output,
        geometry.value_width(),
        geometry.hidden_size,
    )?;

    Ok(GatedDeltaNetMetalRead { workspace })
}

fn validate_contract(
    ctx: &MetalContext,
    input: &MetalTensor,
    weights: GatedDeltaNetMetalWeights<'_>,
    workspace: &GatedDeltaNetMetalWorkspace,
) -> Result<(), Qwen4ExpGdnError> {
    let geometry = workspace.geometry;
    require_f32("GDN input", input, geometry.hidden_size, false)?;
    require_projection(
        "GDN QKV projection",
        weights.qkv,
        geometry.hidden_size,
        geometry.conv_width(),
        true,
    )?;
    require_projection(
        "GDN gate projection",
        weights.gate,
        geometry.hidden_size,
        geometry.value_width(),
        true,
    )?;
    require_projection(
        "GDN beta projection",
        weights.beta,
        geometry.hidden_size,
        geometry.value_heads,
        false,
    )?;
    require_projection(
        "GDN alpha projection",
        weights.alpha,
        geometry.hidden_size,
        geometry.value_heads,
        false,
    )?;
    require_f32("GDN transformed A", weights.a, geometry.value_heads, false)?;
    require_f32("GDN dt bias", weights.dt_bias, geometry.value_heads, false)?;
    require_projection(
        "GDN convolution",
        weights.conv,
        geometry.conv_kernel,
        geometry.conv_width(),
        false,
    )?;
    require_f32("GDN norm", weights.norm, geometry.head_dim, false)?;
    require_projection(
        "GDN output projection",
        weights.output,
        geometry.value_width(),
        geometry.hidden_size,
        true,
    )?;
    for (name, tensor, elements) in [
        (
            "GDN convolution state",
            &workspace.conv_state,
            geometry.conv_state_elements(),
        ),
        (
            "GDN delta state",
            &workspace.delta_state,
            geometry.delta_state_elements(),
        ),
        ("GDN QKV scratch", &workspace.qkv, geometry.conv_width()),
        (
            "GDN convolved QKV",
            &workspace.qkv_conv,
            geometry.conv_width(),
        ),
        ("GDN gate scratch", &workspace.gate, geometry.value_width()),
        ("GDN beta scratch", &workspace.beta, geometry.value_heads),
        ("GDN alpha scratch", &workspace.alpha, geometry.value_heads),
        ("GDN decay scratch", &workspace.decay, geometry.value_heads),
        (
            "GDN query norm",
            &workspace.query_norm,
            geometry.key_width(),
        ),
        ("GDN key norm", &workspace.key_norm, geometry.key_width()),
        (
            "GDN recurrent output",
            &workspace.recurrent,
            geometry.value_width(),
        ),
        (
            "GDN normalized output",
            &workspace.normalized,
            geometry.value_width(),
        ),
        ("GDN output", &workspace.output, geometry.hidden_size),
    ] {
        require_f32(name, tensor, elements, true)?;
    }
    require_read_only_weights(&[
        ("GDN QKV projection", weights.qkv),
        ("GDN gate projection", weights.gate),
        ("GDN beta projection", weights.beta),
        ("GDN alpha projection", weights.alpha),
        ("GDN transformed A", weights.a),
        ("GDN dt bias", weights.dt_bias),
        ("GDN convolution", weights.conv),
        ("GDN norm", weights.norm),
        ("GDN output projection", weights.output),
    ])?;
    let tensors = [
        ("GDN input", input),
        ("GDN QKV projection", weights.qkv),
        ("GDN gate projection", weights.gate),
        ("GDN beta projection", weights.beta),
        ("GDN alpha projection", weights.alpha),
        ("GDN transformed A", weights.a),
        ("GDN dt bias", weights.dt_bias),
        ("GDN convolution", weights.conv),
        ("GDN norm", weights.norm),
        ("GDN output projection", weights.output),
        ("GDN convolution state", &workspace.conv_state),
        ("GDN delta state", &workspace.delta_state),
        ("GDN QKV scratch", &workspace.qkv),
        ("GDN convolved QKV", &workspace.qkv_conv),
        ("GDN gate scratch", &workspace.gate),
        ("GDN beta scratch", &workspace.beta),
        ("GDN alpha scratch", &workspace.alpha),
        ("GDN decay scratch", &workspace.decay),
        ("GDN query norm", &workspace.query_norm),
        ("GDN key norm", &workspace.key_norm),
        ("GDN recurrent output", &workspace.recurrent),
        ("GDN normalized output", &workspace.normalized),
        ("GDN output", &workspace.output),
    ];
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)
}

fn require_f32(
    name: &str,
    tensor: &MetalTensor,
    expected_elements: usize,
    writable: bool,
) -> Result<(), Qwen4ExpGdnError> {
    if tensor.dtype != GgmlType::F32 || tensor.n_elements() != expected_elements as u64 {
        return invalid(format!(
            "{name} must be F32 with {expected_elements} elements, got {:?} with {}",
            tensor.dtype,
            tensor.n_elements()
        ));
    }
    if writable && !tensor.is_writable() {
        return invalid(format!("{name} must be writable"));
    }
    require_range(name, tensor, 4)
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    quantized_allowed: bool,
) -> Result<(), Qwen4ExpGdnError> {
    if tensor.shape != [n_in as u64, n_out as u64] {
        return invalid(format!(
            "{name} shape {:?} does not match [{n_in}, {n_out}]",
            tensor.shape
        ));
    }
    match tensor.dtype {
        GgmlType::F32 => require_range(name, tensor, 4),
        GgmlType::Q8_0 if quantized_allowed && n_in.is_multiple_of(32) => {
            require_range(name, tensor, 2)
        }
        dtype => invalid(format!("{name} has unsupported dtype {dtype:?}")),
    }
}

fn require_range(name: &str, tensor: &MetalTensor, alignment: u64) -> Result<(), Qwen4ExpGdnError> {
    if !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    let bytes = tensor.n_bytes();
    let end = tensor
        .offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpGdnError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_read_only_weights(weights: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpGdnError> {
    for (name, tensor) in weights {
        if tensor.provenance() == MetalTensorProvenance::OwnedWritable {
            return invalid(format!("{name} must have read-only weight provenance"));
        }
    }
    Ok(())
}

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpGdnError> {
    for left in 0..tensors.len() {
        for right in left + 1..tensors.len() {
            if Retained::as_ptr(&tensors[left].1.buffer)
                != Retained::as_ptr(&tensors[right].1.buffer)
            {
                continue;
            }
            let left_end = tensors[left]
                .1
                .offset
                .saturating_add(tensors[left].1.n_bytes());
            let right_end = tensors[right]
                .1
                .offset
                .saturating_add(tensors[right].1.n_bytes());
            if tensors[left].1.offset < right_end && tensors[right].1.offset < left_end {
                return invalid(format!("{} overlaps {}", tensors[left].0, tensors[right].0));
            }
        }
    }
    Ok(())
}

fn require_same_device(
    ctx: &MetalContext,
    tensors: &[(&str, &MetalTensor)],
) -> Result<(), Qwen4ExpGdnError> {
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

fn preflight(
    ctx: &MetalContext,
    weights: GatedDeltaNetMetalWeights<'_>,
) -> Result<(), Qwen4ExpGdnError> {
    for dtype in [weights.qkv.dtype, weights.gate.dtype, weights.output.dtype] {
        preflight_projection(ctx, dtype)?;
    }
    preflight_projection(ctx, GgmlType::F32)?;
    for kernel in [
        "kernel_mat_vec_f32_f32_sigmoid",
        "kernel_gdn_decay_chain_f32",
        "kernel_ssm_conv_silu_f32",
        "kernel_l2_norm_pair_batched_f32",
        "kernel_l2_norm_pair_hd128_r4_f32",
        "kernel_gdn_step_decay_f32",
        "kernel_qwen4exp_gdn_rmsnorm_sigmoid_hd128_r4_f32",
    ] {
        ctx.pipeline(kernel)?;
    }
    Ok(())
}

fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpGdnError> {
    let kernels: &[&str] = match dtype {
        GgmlType::F32 => &["kernel_mat_vec_f32_f32", "kernel_mat_vec_f32_f32_lcpp_r2"],
        GgmlType::Q8_0 => &["kernel_mat_vec_q8_0_f32", "kernel_mat_vec_q8_0_f32_lcpp"],
        _ => return invalid(format!("unsupported GDN projection dtype {dtype:?}")),
    };
    for kernel in kernels {
        ctx.pipeline(kernel)?;
    }
    Ok(())
}

fn encode_rmsnorm_sigmoid_gated(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weight: &MetalTensor,
    gate: &MetalTensor,
    output: &MetalTensor,
    geometry: GatedDeltaNetMetalGeometry,
) -> Result<(), Qwen4ExpGdnError> {
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_heads: u32,
        eps: f32,
    }
    let pipeline = ctx.pipeline("kernel_qwen4exp_gdn_rmsnorm_sigmoid_hd128_r4_f32")?;
    enc.set_pipeline(&pipeline);
    enc.set_bytes(
        0,
        &Args {
            n_heads: geometry.value_heads as u32,
            eps: geometry.eps * geometry.head_dim as f32,
        },
    );
    enc.set_tensor(1, input);
    enc.set_tensor(2, weight);
    enc.set_tensor(3, gate);
    enc.set_tensor(4, output);
    enc.dispatch(
        MTLSize {
            width: geometry.value_heads.div_ceil(4),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 32,
            height: 4,
            depth: 1,
        },
    );
    Ok(())
}

fn reserve_command(
    workspace: &mut GatedDeltaNetMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpGdnError> {
    let command = enc.parent_command_buffer();
    match workspace.active_command.as_ref() {
        None => {
            workspace.active_command = Some(command);
            Ok(())
        }
        Some(owner) if std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) => {
            Ok(())
        }
        Some(_) => invalid(
            "workspace is still owned by another command buffer; release it after completion",
        ),
    }
}

fn zero_f32(ctx: &MetalContext, shape: Vec<u64>) -> Result<MetalTensor, Qwen4ExpGdnError> {
    let elements = shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| Qwen4ExpGdnError::Invalid("zero tensor shape overflow".into()))?;
    let values = vec![0.0_f32; elements];
    Ok(MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&values),
        shape,
        GgmlType::F32,
    )?)
}

fn zero_writable_tensor(tensor: &MetalTensor) -> Result<(), Qwen4ExpGdnError> {
    if tensor.dtype != GgmlType::F32 || !tensor.is_writable() {
        return invalid("GDN state reset requires writable F32 storage");
    }
    let offset = usize::try_from(tensor.offset)
        .map_err(|_| Qwen4ExpGdnError::Invalid("state offset exceeds usize".into()))?;
    let bytes = usize::try_from(tensor.n_bytes())
        .map_err(|_| Qwen4ExpGdnError::Invalid("state byte length exceeds usize".into()))?;
    let end = offset
        .checked_add(bytes)
        .ok_or_else(|| Qwen4ExpGdnError::Invalid("state reset range overflow".into()))?;
    if end > tensor.buffer.length() {
        return invalid("state reset range exceeds its Metal buffer");
    }
    // SAFETY: validation proves the complete writable tensor range lies in a
    // shared Metal buffer, and idle ownership excludes in-flight GPU access.
    unsafe {
        std::ptr::write_bytes(
            tensor.buffer.contents().as_ptr().cast::<u8>().add(offset),
            0,
            bytes,
        );
    }
    Ok(())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpGdnError> {
    Err(Qwen4ExpGdnError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
    use objc2_metal::MTLCommandQueue;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    const GROUPED_TILED_ORACLE_JSON: &str =
        include_str!("../tests/fixtures/qwen4exp_gdn_grouped_tiled_v1.json");
    const GROUPED_TILED_ORACLE_F32: &[u8] =
        include_bytes!("../tests/fixtures/qwen4exp_gdn_grouped_tiled_v1.f32");

    struct CpuWeights {
        qkv: Vec<f32>,
        gate: Vec<f32>,
        beta: Vec<f32>,
        alpha: Vec<f32>,
        a: Vec<f32>,
        dt_bias: Vec<f32>,
        conv: Vec<f32>,
        norm: Vec<f32>,
        output: Vec<f32>,
    }

    struct CpuState {
        conv: Vec<f32>,
        delta: Vec<f32>,
    }

    fn values(count: usize, seed: usize, scale: f32) -> Vec<f32> {
        (0..count)
            .map(|index| {
                let raw = ((index * 17 + seed * 11 + 3) % 37) as f32 - 18.0;
                raw * scale
            })
            .collect()
    }

    fn tensor(ctx: &MetalContext, data: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(data), shape, GgmlType::F32).unwrap()
    }

    fn weight(ctx: &MetalContext, data: &[f32], shape: Vec<u64>) -> MetalTensor {
        let mut tensor = tensor(ctx, data, shape);
        tensor.provenance = MetalTensorProvenance::OwnedWeightReadOnly;
        tensor
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

    fn mat_vec(weight: &[f32], input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        assert_eq!(weight.len(), n_in * n_out);
        weight
            .chunks_exact(n_in)
            .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
            .collect()
    }

    fn sigmoid(value: f32) -> f32 {
        1.0 / (1.0 + (-value).exp())
    }

    fn softplus(value: f32) -> f32 {
        if value > 20.0 {
            value
        } else if value < -20.0 {
            value.exp()
        } else {
            (1.0 + value.exp()).ln()
        }
    }

    fn cpu_step(
        geometry: GatedDeltaNetMetalGeometry,
        weights: &CpuWeights,
        input: &[f32],
        state: &mut CpuState,
    ) -> Vec<f32> {
        let qkv = mat_vec(
            &weights.qkv,
            input,
            geometry.hidden_size,
            geometry.conv_width(),
        );
        let gate = mat_vec(
            &weights.gate,
            input,
            geometry.hidden_size,
            geometry.value_width(),
        );
        let beta = mat_vec(
            &weights.beta,
            input,
            geometry.hidden_size,
            geometry.value_heads,
        )
        .into_iter()
        .map(sigmoid)
        .collect::<Vec<_>>();
        let alpha = mat_vec(
            &weights.alpha,
            input,
            geometry.hidden_size,
            geometry.value_heads,
        );
        let decay = alpha
            .iter()
            .zip(&weights.dt_bias)
            .zip(&weights.a)
            .map(|((&alpha, &bias), &a)| (softplus(alpha + bias) * a).exp())
            .collect::<Vec<_>>();

        let mut convolved = vec![0.0; geometry.conv_width()];
        for channel in 0..geometry.conv_width() {
            let mut sum = weights.conv[channel * geometry.conv_kernel + 3] * qkv[channel];
            for tap in 0..3 {
                sum += weights.conv[channel * geometry.conv_kernel + tap]
                    * state.conv[tap * geometry.conv_width() + channel];
            }
            convolved[channel] = sum * sigmoid(sum);
        }
        for tap in 0..2 {
            let source = (tap + 1) * geometry.conv_width();
            let destination = tap * geometry.conv_width();
            state
                .conv
                .copy_within(source..source + geometry.conv_width(), destination);
        }
        state.conv[2 * geometry.conv_width()..].copy_from_slice(&qkv);

        let mut query = convolved[..geometry.key_width()].to_vec();
        let mut key = convolved[geometry.key_width()..2 * geometry.key_width()].to_vec();
        let value = &convolved[2 * geometry.key_width()..];
        for head in 0..geometry.key_heads {
            let range = head * geometry.head_dim..(head + 1) * geometry.head_dim;
            for vector in [&mut query, &mut key] {
                let norm = vector[range.clone()]
                    .iter()
                    .map(|value| value * value)
                    .sum::<f32>()
                    .sqrt()
                    .max(geometry.eps);
                for value in &mut vector[range.clone()] {
                    *value /= norm;
                }
            }
        }

        let mut recurrent = vec![0.0; geometry.value_width()];
        for value_head in 0..geometry.value_heads {
            let key_head = value_head % geometry.key_heads;
            let q = &query[key_head * geometry.head_dim..(key_head + 1) * geometry.head_dim];
            let k = &key[key_head * geometry.head_dim..(key_head + 1) * geometry.head_dim];
            for value_lane in 0..geometry.head_dim {
                let state_start = (value_head * geometry.head_dim + value_lane) * geometry.head_dim;
                let state_row = &mut state.delta[state_start..state_start + geometry.head_dim];
                for state_value in state_row.iter_mut() {
                    *state_value *= decay[value_head];
                }
                let prediction = state_row
                    .iter()
                    .zip(k)
                    .map(|(state, key)| state * key)
                    .sum::<f32>();
                let correction = (value[value_head * geometry.head_dim + value_lane] - prediction)
                    * beta[value_head];
                for (state_value, key) in state_row.iter_mut().zip(k) {
                    *state_value += correction * key;
                }
                recurrent[value_head * geometry.head_dim + value_lane] = state_row
                    .iter()
                    .zip(q)
                    .map(|(state, query)| state * query)
                    .sum();
            }
        }

        let scaled_eps = geometry.eps * geometry.head_dim as f32;
        let mut normalized = vec![0.0; geometry.value_width()];
        for head in 0..geometry.value_heads {
            let start = head * geometry.head_dim;
            let row = &recurrent[start..start + geometry.head_dim];
            let mean_square =
                row.iter().map(|value| value * value).sum::<f32>() / geometry.head_dim as f32;
            let scale = 1.0 / (mean_square + scaled_eps).sqrt();
            for lane in 0..geometry.head_dim {
                normalized[start + lane] =
                    row[lane] * scale * weights.norm[lane] * sigmoid(gate[start + lane]);
            }
        }
        mat_vec(
            &weights.output,
            &normalized,
            geometry.value_width(),
            geometry.hidden_size,
        )
    }

    fn assert_close(actual: &[f32], expected: &[f32], atol: f32, rtol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            let tolerance = atol + rtol * expected.abs();
            assert!(
                (actual - expected).abs() <= tolerance,
                "index {index}: expected {expected}, got {actual}, tolerance {tolerance}"
            );
        }
    }

    #[derive(Deserialize)]
    struct GroupedTiledOracle {
        schema_version: u32,
        generator_version: u32,
        geometry: OracleGeometry,
        layout: OracleLayout,
        recipe: OracleRecipe,
        binary: OracleBinary,
        l2_sentinel: OracleL2Sentinel,
        fault_sensitivity_max_abs_output: BTreeMap<String, f32>,
    }

    #[derive(Deserialize)]
    struct OracleGeometry {
        hidden_size: usize,
        key_heads: usize,
        value_heads: usize,
        head_dim: usize,
        conv_kernel: usize,
        eps: f32,
        tokens: usize,
    }

    #[derive(Deserialize)]
    struct OracleLayout {
        gguf_from_checkpoint: Vec<usize>,
    }

    #[derive(Deserialize)]
    struct OracleRecipe {
        qkv: Formula2,
        gate: Formula2,
        beta: Formula2,
        alpha: Formula2,
        conv: Formula2,
        output: Formula2,
        conv_state: Formula2,
        delta_state: Formula3,
        norm: NormRecipe,
        transformed_a: Vec<f32>,
        dt_bias: Vec<f32>,
        inputs: Vec<Vec<f32>>,
    }

    #[derive(Deserialize)]
    struct Formula2 {
        multipliers: [i64; 2],
        add: i64,
        modulus: i64,
        center: i64,
        scale: f32,
    }

    #[derive(Deserialize)]
    struct Formula3 {
        multipliers: [i64; 3],
        add: i64,
        modulus: i64,
        center: i64,
        scale: f32,
    }

    #[derive(Deserialize)]
    struct NormRecipe {
        base: f32,
        step: f32,
        modulus: usize,
    }

    #[derive(Deserialize)]
    struct OracleBinary {
        file: String,
        dtype: String,
        byte_order: String,
        sha256: String,
        sections: BTreeMap<String, OracleSection>,
    }

    #[derive(Deserialize)]
    struct OracleSection {
        offset_f32: usize,
        count_f32: usize,
        shape: Vec<usize>,
    }

    #[derive(Deserialize)]
    struct OracleL2Sentinel {
        max_abs_additive_delta: f32,
    }

    fn formula2(rows: usize, columns: usize, formula: &Formula2) -> Vec<f32> {
        let mut output = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                let raw = (row as i64 * formula.multipliers[0]
                    + column as i64 * formula.multipliers[1]
                    + formula.add)
                    .rem_euclid(formula.modulus)
                    - formula.center;
                output.push(raw as f32 * formula.scale);
            }
        }
        output
    }

    fn formula3(first: usize, second: usize, third: usize, formula: &Formula3) -> Vec<f32> {
        let mut output = Vec::with_capacity(first * second * third);
        for first_index in 0..first {
            for second_index in 0..second {
                for third_index in 0..third {
                    let raw = (first_index as i64 * formula.multipliers[0]
                        + second_index as i64 * formula.multipliers[1]
                        + third_index as i64 * formula.multipliers[2]
                        + formula.add)
                        .rem_euclid(formula.modulus)
                        - formula.center;
                    output.push(raw as f32 * formula.scale);
                }
            }
        }
        output
    }

    fn permute_blocks(source: &[f32], block_elements: usize, permutation: &[usize]) -> Vec<f32> {
        assert_eq!(source.len(), block_elements * permutation.len());
        let mut output = Vec::with_capacity(source.len());
        for &source_head in permutation {
            let start = source_head * block_elements;
            output.extend_from_slice(&source[start..start + block_elements]);
        }
        output
    }

    fn permute_value_tail(
        source: &[f32],
        prefix_elements: usize,
        block_elements: usize,
        permutation: &[usize],
    ) -> Vec<f32> {
        let mut output = source[..prefix_elements].to_vec();
        output.extend(permute_blocks(
            &source[prefix_elements..],
            block_elements,
            permutation,
        ));
        output
    }

    fn permute_value_columns(
        source: &[f32],
        rows: usize,
        value_heads: usize,
        head_dim: usize,
        permutation: &[usize],
    ) -> Vec<f32> {
        let row_width = value_heads * head_dim;
        assert_eq!(source.len(), rows * row_width);
        let mut output = Vec::with_capacity(source.len());
        for row in source.chunks_exact(row_width) {
            output.extend(permute_blocks(row, head_dim, permutation));
        }
        output
    }

    fn permute_conv_state(
        source: &[f32],
        rows: usize,
        key_width: usize,
        head_dim: usize,
        permutation: &[usize],
    ) -> Vec<f32> {
        let row_width = 2 * key_width + permutation.len() * head_dim;
        assert_eq!(source.len(), rows * row_width);
        let mut output = Vec::with_capacity(source.len());
        for row in source.chunks_exact(row_width) {
            output.extend_from_slice(&row[..2 * key_width]);
            output.extend(permute_blocks(&row[2 * key_width..], head_dim, permutation));
        }
        output
    }

    fn write_f32(tensor: &MetalTensor, values: &[f32]) {
        assert!(tensor.is_writable());
        assert_eq!(tensor.dtype, GgmlType::F32);
        assert_eq!(tensor.n_elements() as usize, values.len());
        let offset = tensor.offset as usize;
        assert!(offset + std::mem::size_of_val(values) <= tensor.buffer.length());
        unsafe {
            let destination = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(offset)
                .cast::<f32>();
            std::ptr::copy_nonoverlapping(values.as_ptr(), destination, values.len());
        }
    }

    fn oracle_f32() -> Vec<f32> {
        assert!(GROUPED_TILED_ORACLE_F32.len().is_multiple_of(4));
        GROUPED_TILED_ORACLE_F32
            .chunks_exact(4)
            .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
            .collect()
    }

    fn oracle_section<'a>(
        values: &'a [f32],
        sections: &BTreeMap<String, OracleSection>,
        name: &str,
    ) -> &'a [f32] {
        let section = sections.get(name).unwrap();
        assert_eq!(section.shape.iter().product::<usize>(), section.count_f32);
        let end = section.offset_f32 + section.count_f32;
        &values[section.offset_f32..end]
    }

    #[test]
    fn sigmoid_gated_rmsnorm_matches_cpu_and_differs_from_silu() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        let geometry = GatedDeltaNetMetalGeometry::new(32, 1, 2, 128, 4, 1e-6).unwrap();
        let input = values(256, 1, 0.02);
        let norm = (0..128)
            .map(|index| 0.75 + (index % 11) as f32 * 0.03125)
            .collect::<Vec<_>>();
        let gate = (0_usize..256)
            .map(|index| if index.is_multiple_of(2) { -2.0 } else { 1.5 })
            .collect::<Vec<_>>();
        let input_gpu = tensor(&ctx, &input, vec![256]);
        let norm_gpu = weight(&ctx, &norm, vec![128]);
        let gate_gpu = tensor(&ctx, &gate, vec![256]);
        let output_gpu = MetalTensor::zeros_f32(&ctx, vec![256]).unwrap();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        encode_rmsnorm_sigmoid_gated(
            &ctx,
            &encoder,
            &input_gpu,
            &norm_gpu,
            &gate_gpu,
            &output_gpu,
            geometry,
        )
        .unwrap();
        encoder.end();
        command.commit();
        command.waitUntilCompleted();
        assert!(command.error().is_none());

        let mut expected = vec![0.0; 256];
        let mut silu = vec![0.0; 256];
        for head in 0..2 {
            let start = head * 128;
            let mean_square = input[start..start + 128]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / 128.0;
            let scale = 1.0 / (mean_square + geometry.eps * 128.0).sqrt();
            for (lane, &norm_weight) in norm.iter().enumerate().take(128) {
                let index = start + lane;
                let normalized = input[index] * scale * norm_weight;
                expected[index] = normalized * sigmoid(gate[index]);
                silu[index] = normalized * gate[index] * sigmoid(gate[index]);
            }
        }
        let actual = read_f32(&output_gpu);
        assert_close(&actual, &expected, 2e-5, 2e-5);
        assert!(
            actual
                .iter()
                .zip(silu)
                .any(|(&sigmoid_value, silu_value)| (sigmoid_value - silu_value).abs() > 0.1)
        );
    }

    #[test]
    fn two_token_gdn_composition_matches_cpu_state_and_output() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        let geometry = GatedDeltaNetMetalGeometry::new(32, 1, 1, 128, 4, 1e-6).unwrap();
        let cpu_weights = CpuWeights {
            qkv: values(geometry.hidden_size * geometry.conv_width(), 2, 0.0005),
            gate: values(geometry.hidden_size * geometry.value_width(), 3, 0.001),
            beta: values(geometry.hidden_size * geometry.value_heads, 4, 0.002),
            alpha: values(geometry.hidden_size * geometry.value_heads, 5, 0.002),
            a: vec![-0.55],
            dt_bias: vec![0.15],
            conv: values(geometry.conv_width() * 4, 6, 0.01),
            norm: (0..128)
                .map(|index| 0.8 + (index % 7) as f32 * 0.04)
                .collect(),
            output: values(geometry.value_width() * geometry.hidden_size, 7, 0.0008),
        };
        let qkv_gpu = weight(
            &ctx,
            &cpu_weights.qkv,
            vec![geometry.hidden_size as u64, geometry.conv_width() as u64],
        );
        let gate_gpu = weight(
            &ctx,
            &cpu_weights.gate,
            vec![geometry.hidden_size as u64, geometry.value_width() as u64],
        );
        let beta_gpu = weight(
            &ctx,
            &cpu_weights.beta,
            vec![geometry.hidden_size as u64, geometry.value_heads as u64],
        );
        let alpha_gpu = weight(
            &ctx,
            &cpu_weights.alpha,
            vec![geometry.hidden_size as u64, geometry.value_heads as u64],
        );
        let a_gpu = weight(&ctx, &cpu_weights.a, vec![1]);
        let dt_gpu = weight(&ctx, &cpu_weights.dt_bias, vec![1]);
        let conv_gpu = weight(
            &ctx,
            &cpu_weights.conv,
            vec![4, geometry.conv_width() as u64],
        );
        let norm_gpu = weight(&ctx, &cpu_weights.norm, vec![128]);
        let output_gpu = weight(
            &ctx,
            &cpu_weights.output,
            vec![geometry.value_width() as u64, geometry.hidden_size as u64],
        );
        let weights = GatedDeltaNetMetalWeights {
            geometry,
            qkv: &qkv_gpu,
            gate: &gate_gpu,
            beta: &beta_gpu,
            alpha: &alpha_gpu,
            a: &a_gpu,
            dt_bias: &dt_gpu,
            conv: &conv_gpu,
            norm: &norm_gpu,
            output: &output_gpu,
        };
        let mut workspace = GatedDeltaNetMetalWorkspace::new(&ctx, geometry).unwrap();
        let mut cpu_state = CpuState {
            conv: vec![0.0; geometry.conv_state_elements()],
            delta: vec![0.0; geometry.delta_state_elements()],
        };

        for token in 0..2 {
            let input = values(geometry.hidden_size, 20 + token, 0.01);
            let expected = cpu_step(geometry, &cpu_weights, &input, &mut cpu_state);
            let input_gpu = tensor(&ctx, &input, vec![geometry.hidden_size as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_gated_delta_net(&ctx, &encoder, &input_gpu, weights, &mut workspace)
                .unwrap();
            assert_eq!(read.output().n_elements(), geometry.hidden_size as u64);
            drop(read);
            encoder.end();
            if token == 0 {
                let error = workspace.release_after().unwrap_err().to_string();
                assert!(error.contains("not committed"));
                let blocked_command = ctx.queue.commandBuffer().unwrap();
                let blocked_encoder = KernelEncoder::begin(&blocked_command);
                assert!(
                    encode_gated_delta_net(
                        &ctx,
                        &blocked_encoder,
                        &input_gpu,
                        weights,
                        &mut workspace,
                    )
                    .is_err()
                );
                blocked_encoder.end();
            }
            command.commit();
            workspace.release_after().unwrap();

            assert_close(&read_f32(&workspace.output), &expected, 3e-4, 5e-4);
            assert_close(
                &read_f32(&workspace.conv_state),
                &cpu_state.conv,
                2e-5,
                2e-5,
            );
            assert_close(
                &read_f32(&workspace.delta_state),
                &cpu_state.delta,
                3e-4,
                5e-4,
            );
        }

        workspace.reset().unwrap();
        assert!(
            read_f32(&workspace.conv_state)
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            read_f32(&workspace.delta_state)
                .iter()
                .all(|&value| value == 0.0)
        );

        let concurrent_command = ctx.queue.commandBuffer().unwrap();
        let concurrent_encoder = KernelEncoder::begin_concurrent(&concurrent_command);
        let input_gpu = tensor(&ctx, &values(32, 44, 0.01), vec![32]);
        assert!(
            encode_gated_delta_net(
                &ctx,
                &concurrent_encoder,
                &input_gpu,
                weights,
                &mut workspace,
            )
            .is_err()
        );
        concurrent_encoder.end();

        let abandoned_command = ctx.queue.commandBuffer().unwrap();
        let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
        let read = encode_gated_delta_net(
            &ctx,
            &abandoned_encoder,
            &input_gpu,
            weights,
            &mut workspace,
        )
        .unwrap();
        drop(read);
        abandoned_encoder.end();
        // SAFETY: the encoder ended and the only caller-held command reference
        // is discarded immediately without being committed.
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        drop(abandoned_command);
        workspace.reset().unwrap();

        workspace.state_poisoned = true;
        let poisoned_command = ctx.queue.commandBuffer().unwrap();
        let poisoned_encoder = KernelEncoder::begin(&poisoned_command);
        assert!(
            encode_gated_delta_net(&ctx, &poisoned_encoder, &input_gpu, weights, &mut workspace,)
                .is_err()
        );
        poisoned_encoder.end();
        workspace.reset().unwrap();
        assert!(!workspace.state_poisoned);
    }

    #[test]
    fn grouped_checkpoint_oracle_matches_tiled_multi_head_decode() {
        let ctx = match MetalContext::new() {
            Ok(ctx) => ctx,
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => return,
            Err(error) => panic!("Metal initialization failed: {error}"),
        };
        let fixture: GroupedTiledOracle = serde_json::from_str(GROUPED_TILED_ORACLE_JSON).unwrap();
        assert_eq!(fixture.schema_version, 1);
        assert_eq!(fixture.generator_version, 1);
        assert_eq!(fixture.binary.file, "qwen4exp_gdn_grouped_tiled_v1.f32");
        assert_eq!(fixture.binary.dtype, "f32");
        assert_eq!(fixture.binary.byte_order, "little");
        assert!(fixture.l2_sentinel.max_abs_additive_delta > 0.5);
        let digest = format!("{:x}", Sha256::digest(GROUPED_TILED_ORACLE_F32));
        assert_eq!(digest, fixture.binary.sha256);
        for sentinel in [
            "grouped_modulo_mapping",
            "silu_output_gate",
            "zero_initial_convolution_state",
            "zero_initial_recurrent_state",
        ] {
            assert!(fixture.fault_sensitivity_max_abs_output[sentinel] > 0.1);
        }

        let geometry = GatedDeltaNetMetalGeometry::new(
            fixture.geometry.hidden_size,
            fixture.geometry.key_heads,
            fixture.geometry.value_heads,
            fixture.geometry.head_dim,
            fixture.geometry.conv_kernel,
            fixture.geometry.eps,
        )
        .unwrap();
        let permutation = &fixture.layout.gguf_from_checkpoint;
        assert_eq!(permutation, &[0, 2, 1, 3]);
        assert_eq!(fixture.geometry.tokens, fixture.recipe.inputs.len());

        let grouped_qkv = formula2(
            geometry.conv_width(),
            geometry.hidden_size,
            &fixture.recipe.qkv,
        );
        let qkv = permute_value_tail(
            &grouped_qkv,
            2 * geometry.key_width() * geometry.hidden_size,
            geometry.head_dim * geometry.hidden_size,
            permutation,
        );
        let gate = permute_blocks(
            &formula2(
                geometry.value_width(),
                geometry.hidden_size,
                &fixture.recipe.gate,
            ),
            geometry.head_dim * geometry.hidden_size,
            permutation,
        );
        let beta = permute_blocks(
            &formula2(
                geometry.value_heads,
                geometry.hidden_size,
                &fixture.recipe.beta,
            ),
            geometry.hidden_size,
            permutation,
        );
        let alpha = permute_blocks(
            &formula2(
                geometry.value_heads,
                geometry.hidden_size,
                &fixture.recipe.alpha,
            ),
            geometry.hidden_size,
            permutation,
        );
        let transformed_a = permute_blocks(&fixture.recipe.transformed_a, 1, permutation);
        let dt_bias = permute_blocks(&fixture.recipe.dt_bias, 1, permutation);
        let grouped_conv = formula2(
            geometry.conv_width(),
            geometry.conv_kernel,
            &fixture.recipe.conv,
        );
        let conv = permute_value_tail(
            &grouped_conv,
            2 * geometry.key_width() * geometry.conv_kernel,
            geometry.head_dim * geometry.conv_kernel,
            permutation,
        );
        let norm = (0..geometry.head_dim)
            .map(|index| {
                fixture.recipe.norm.base
                    + (index % fixture.recipe.norm.modulus) as f32 * fixture.recipe.norm.step
            })
            .collect::<Vec<_>>();
        let output = permute_value_columns(
            &formula2(
                geometry.hidden_size,
                geometry.value_width(),
                &fixture.recipe.output,
            ),
            geometry.hidden_size,
            geometry.value_heads,
            geometry.head_dim,
            permutation,
        );
        let initial_conv = permute_conv_state(
            &formula2(
                geometry.conv_kernel - 1,
                geometry.conv_width(),
                &fixture.recipe.conv_state,
            ),
            geometry.conv_kernel - 1,
            geometry.key_width(),
            geometry.head_dim,
            permutation,
        );
        let initial_delta = permute_blocks(
            &formula3(
                geometry.value_heads,
                geometry.head_dim,
                geometry.head_dim,
                &fixture.recipe.delta_state,
            ),
            geometry.head_dim * geometry.head_dim,
            permutation,
        );

        let qkv_gpu = weight(
            &ctx,
            &qkv,
            vec![geometry.hidden_size as u64, geometry.conv_width() as u64],
        );
        let gate_gpu = weight(
            &ctx,
            &gate,
            vec![geometry.hidden_size as u64, geometry.value_width() as u64],
        );
        let beta_gpu = weight(
            &ctx,
            &beta,
            vec![geometry.hidden_size as u64, geometry.value_heads as u64],
        );
        let alpha_gpu = weight(
            &ctx,
            &alpha,
            vec![geometry.hidden_size as u64, geometry.value_heads as u64],
        );
        let a_gpu = weight(&ctx, &transformed_a, vec![geometry.value_heads as u64]);
        let dt_gpu = weight(&ctx, &dt_bias, vec![geometry.value_heads as u64]);
        let conv_gpu = weight(
            &ctx,
            &conv,
            vec![geometry.conv_kernel as u64, geometry.conv_width() as u64],
        );
        let norm_gpu = weight(&ctx, &norm, vec![geometry.head_dim as u64]);
        let output_gpu = weight(
            &ctx,
            &output,
            vec![geometry.value_width() as u64, geometry.hidden_size as u64],
        );
        let weights = GatedDeltaNetMetalWeights {
            geometry,
            qkv: &qkv_gpu,
            gate: &gate_gpu,
            beta: &beta_gpu,
            alpha: &alpha_gpu,
            a: &a_gpu,
            dt_bias: &dt_gpu,
            conv: &conv_gpu,
            norm: &norm_gpu,
            output: &output_gpu,
        };
        let mut workspace = GatedDeltaNetMetalWorkspace::new(&ctx, geometry).unwrap();
        write_f32(&workspace.conv_state, &initial_conv);
        write_f32(&workspace.delta_state, &initial_delta);
        let exported_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.hidden_size as u64]).unwrap();

        let oracle = oracle_f32();
        let expected_output = oracle_section(&oracle, &fixture.binary.sections, "output");
        let expected_recurrent = oracle_section(
            &oracle,
            &fixture.binary.sections,
            "recurrent_unscaled_tiled",
        );
        let expected_delta = oracle_section(&oracle, &fixture.binary.sections, "delta_state_tiled");
        let expected_conv = oracle_section(&oracle, &fixture.binary.sections, "conv_state_tiled");
        let l2_query = oracle_section(&oracle, &fixture.binary.sections, "l2_query");
        let l2_key = oracle_section(&oracle, &fixture.binary.sections, "l2_key");
        let expected_l2_query = oracle_section(&oracle, &fixture.binary.sections, "l2_query_ggml");
        let expected_l2_key = oracle_section(&oracle, &fixture.binary.sections, "l2_key_ggml");

        let l2_query_gpu = tensor(
            &ctx,
            l2_query,
            vec![geometry.head_dim as u64, geometry.key_heads as u64],
        );
        let l2_key_gpu = tensor(
            &ctx,
            l2_key,
            vec![geometry.head_dim as u64, geometry.key_heads as u64],
        );
        let l2_query_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.key_width() as u64]).unwrap();
        let l2_key_output =
            MetalTensor::zeros_f32(&ctx, vec![geometry.key_width() as u64]).unwrap();
        let l2_command = ctx.queue.commandBuffer().unwrap();
        let l2_encoder = KernelEncoder::begin(&l2_command);
        encode_l2_norm_pair_batched_f32(
            &ctx,
            &l2_encoder,
            &l2_query_gpu,
            &l2_query_output,
            &l2_key_gpu,
            &l2_key_output,
            geometry.key_heads,
            geometry.head_dim,
            geometry.eps,
        )
        .unwrap();
        l2_encoder.end();
        l2_command.commit();
        l2_command.waitUntilCompleted();
        assert!(l2_command.error().is_none());
        assert_close(&read_f32(&l2_query_output), expected_l2_query, 2e-6, 2e-6);
        assert_close(&read_f32(&l2_key_output), expected_l2_key, 2e-6, 2e-6);

        for (token, input) in fixture.recipe.inputs.iter().enumerate() {
            let input_gpu = tensor(&ctx, input, vec![geometry.hidden_size as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read = encode_gated_delta_net(&ctx, &encoder, &input_gpu, weights, &mut workspace)
                .unwrap();
            assert_eq!(read.output().dtype(), GgmlType::F32);
            read.output()
                .encode_copy_to(&ctx, &encoder, &exported_output)
                .unwrap();
            drop(read);
            encoder.end();
            command.commit();
            workspace.release_after().unwrap();

            let output_start = token * geometry.hidden_size;
            assert_close(
                &read_f32(&exported_output),
                &expected_output[output_start..output_start + geometry.hidden_size],
                4e-4,
                8e-4,
            );
            let recurrent_start = token * geometry.value_width();
            assert_close(
                &read_f32(&workspace.recurrent),
                &expected_recurrent[recurrent_start..recurrent_start + geometry.value_width()],
                4e-4,
                8e-4,
            );
        }
        assert_close(
            &read_f32(&workspace.delta_state),
            expected_delta,
            4e-4,
            8e-4,
        );
        assert_close(&read_f32(&workspace.conv_state), expected_conv, 2e-5, 2e-5);
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_GDN_GGUF to the pinned full release"]
    fn released_layer_zero_matches_cpu_quantized_oracle() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_GDN_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_GDN_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let metal_weights = realized.weights();

        let mut gdn_layers = 0;
        for layer in 0..48 {
            let binding = GatedDeltaNetMetalWeights::bind(metal_weights, layer);
            if layer % 4 == 3 {
                assert!(binding.is_err(), "QSA layer {layer} accepted as GDN");
            } else {
                binding.unwrap();
                gdn_layers += 1;
            }
        }
        assert_eq!(gdn_layers, 36);

        let geometry = GatedDeltaNetMetalGeometry::from_config(metal_weights.config()).unwrap();
        let dequant = |name: &str| {
            let desc = gguf.find(name).unwrap();
            crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
        };
        let cpu_weights = CpuWeights {
            qkv: dequant("blk.0.attn_qkv.weight"),
            gate: dequant("blk.0.attn_gate.weight"),
            beta: dequant("blk.0.ssm_beta.weight"),
            alpha: dequant("blk.0.ssm_alpha.weight"),
            a: dequant("blk.0.ssm_a"),
            dt_bias: dequant("blk.0.ssm_dt.bias"),
            conv: dequant("blk.0.ssm_conv1d.weight"),
            norm: dequant("blk.0.ssm_norm.weight"),
            output: dequant("blk.0.ssm_out.weight"),
        };
        let input = values(geometry.hidden_size, 91, 0.01);
        let mut cpu_state = CpuState {
            conv: vec![0.0; geometry.conv_state_elements()],
            delta: vec![0.0; geometry.delta_state_elements()],
        };
        let expected = cpu_step(geometry, &cpu_weights, &input, &mut cpu_state);

        let input_gpu = tensor(&ctx, &input, vec![geometry.hidden_size as u64]);
        let weights = GatedDeltaNetMetalWeights::bind(metal_weights, 0).unwrap();
        let mut workspace = GatedDeltaNetMetalWorkspace::new(&ctx, geometry).unwrap();
        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_gated_delta_net(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();

        assert_close(&read_f32(&workspace.output), &expected, 2e-2, 2e-3);
        assert_close(
            &read_f32(&workspace.conv_state),
            &cpu_state.conv,
            1e-2,
            2e-3,
        );
        assert_close(
            &read_f32(&workspace.delta_state),
            &cpu_state.delta,
            2e-2,
            2e-3,
        );
    }
}
