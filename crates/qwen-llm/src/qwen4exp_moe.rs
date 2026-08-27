//! One-token Metal MoE execution for Qwen3.8-Flash-Next.

use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, MetalTensorProvenance,
    encode_axpy_scalar_f32, encode_copy_offset_f32, encode_dot_sigmoid_f32,
    encode_moe_down_iq4_nl_f32, encode_moe_down_weighted_sum_q8_0_f32,
    encode_moe_swiglu_iq3_xxs_f32, encode_moe_swiglu_iq3_xxs_f32_fast,
    encode_moe_swiglu_iq4_xs_f32, encode_moe_weighted_sum_f32, encode_shared_swiglu_q8_0_f32,
    encode_topk_logits_softmax_f32,
};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::qwen4exp::Qwen4ExpConfig;
use crate::qwen4exp_residency::{Qwen4ExpMetalWeights, Qwen4ExpResidencyError};
use crate::tensor::GgmlType;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLComputePipelineState, MTLDevice,
    MTLResource,
};

const MAX_TOP_K: usize = 16;

crate::env_flag!(
    default_on qwen4exp_moe_iq3_fast_enabled,
    "QWEN4EXP_MOE_IQ3_FAST"
);

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpMoeError {
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Forward(#[from] MfError),
    #[error(transparent)]
    Residency(#[from] Qwen4ExpResidencyError),
    #[error("invalid Qwen3.8-Flash-Next MoE contract: {0}")]
    Invalid(String),
    #[error("Qwen3.8-Flash-Next MoE command buffer failed: {0}")]
    CommandBuffer(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Qwen4ExpMoeMetalGeometry {
    hidden_size: usize,
    expert_count: usize,
    experts_per_token: usize,
    routed_intermediate_size: usize,
    shared_intermediate_size: usize,
}

impl Qwen4ExpMoeMetalGeometry {
    pub fn new(
        hidden_size: usize,
        expert_count: usize,
        experts_per_token: usize,
        routed_intermediate_size: usize,
        shared_intermediate_size: usize,
    ) -> Result<Self, Qwen4ExpMoeError> {
        let geometry = Self {
            hidden_size,
            expert_count,
            experts_per_token,
            routed_intermediate_size,
            shared_intermediate_size,
        };
        geometry.validate()?;
        Ok(geometry)
    }

    pub fn from_config(config: &Qwen4ExpConfig) -> Result<Self, Qwen4ExpMoeError> {
        Self::new(
            config.hidden_size as usize,
            config.moe.expert_count as usize,
            config.moe.experts_per_token as usize,
            config.moe.expert_intermediate_size as usize,
            config.moe.shared_expert_intermediate_size as usize,
        )
    }

    pub fn hidden_size(self) -> usize {
        self.hidden_size
    }

    pub fn expert_count(self) -> usize {
        self.expert_count
    }

    pub fn experts_per_token(self) -> usize {
        self.experts_per_token
    }

    pub fn routed_intermediate_size(self) -> usize {
        self.routed_intermediate_size
    }

    pub fn shared_intermediate_size(self) -> usize {
        self.shared_intermediate_size
    }

    fn validate(self) -> Result<(), Qwen4ExpMoeError> {
        if self.hidden_size == 0
            || self.expert_count == 0
            || self.experts_per_token == 0
            || self.routed_intermediate_size == 0
            || self.shared_intermediate_size == 0
        {
            return invalid("MoE dimensions and top-k must be nonzero");
        }
        if self.experts_per_token > self.expert_count || self.experts_per_token > MAX_TOP_K {
            return invalid(format!(
                "MoE top-k {} must be no larger than expert count {} or {MAX_TOP_K}",
                self.experts_per_token, self.expert_count
            ));
        }
        if self.expert_count > i32::MAX as usize {
            return invalid("MoE expert IDs must fit signed i32");
        }
        if !self.hidden_size.is_multiple_of(256) {
            return invalid(format!(
                "MoE hidden size {} must be divisible by 256 for released routed weights",
                self.hidden_size
            ));
        }
        if !self.routed_intermediate_size.is_multiple_of(32) {
            return invalid(format!(
                "MoE routed intermediate size {} must be divisible by 32",
                self.routed_intermediate_size
            ));
        }
        if !self.shared_intermediate_size.is_multiple_of(32) {
            return invalid(format!(
                "MoE shared intermediate size {} must be divisible by 32",
                self.shared_intermediate_size
            ));
        }

        for (name, value) in [
            ("hidden size", self.hidden_size),
            ("expert count", self.expert_count),
            ("top-k", self.experts_per_token),
            ("routed intermediate size", self.routed_intermediate_size),
            ("shared intermediate size", self.shared_intermediate_size),
        ] {
            if u32::try_from(value).is_err() {
                return invalid(format!("MoE {name} {value} exceeds u32"));
            }
        }

        let products = [
            ("router", &[self.hidden_size, self.expert_count][..]),
            (
                "routed expert bank",
                &[
                    self.hidden_size,
                    self.routed_intermediate_size,
                    self.expert_count,
                ][..],
            ),
            (
                "routed down bank",
                &[
                    self.routed_intermediate_size,
                    self.hidden_size,
                    self.expert_count,
                ][..],
            ),
            (
                "routed inner scratch",
                &[self.experts_per_token, self.routed_intermediate_size][..],
            ),
            (
                "routed output scratch",
                &[self.experts_per_token, self.hidden_size][..],
            ),
            (
                "shared gate/up",
                &[self.hidden_size, self.shared_intermediate_size][..],
            ),
            (
                "shared down",
                &[self.shared_intermediate_size, self.hidden_size][..],
            ),
        ];
        for (name, factors) in products {
            let elements = checked_product(factors, name)?;
            if u32::try_from(elements).is_err() {
                return invalid(format!("MoE {name} offsets exceed u32"));
            }
            elements.checked_mul(4).ok_or_else(|| {
                Qwen4ExpMoeError::Invalid(format!("MoE {name} byte count overflow"))
            })?;
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub struct Qwen4ExpMoeMetalWeights<'a> {
    pub geometry: Qwen4ExpMoeMetalGeometry,
    pub router: &'a MetalTensor,
    pub routed_gate: &'a MetalTensor,
    pub routed_up: &'a MetalTensor,
    pub routed_down: &'a MetalTensor,
    pub shared_router: &'a MetalTensor,
    pub shared_gate: &'a MetalTensor,
    pub shared_up: &'a MetalTensor,
    pub shared_down: &'a MetalTensor,
}

impl<'a> Qwen4ExpMoeMetalWeights<'a> {
    pub fn bind(weights: &'a Qwen4ExpMetalWeights, layer: u32) -> Result<Self, Qwen4ExpMoeError> {
        if layer >= weights.config().layer_count {
            return invalid(format!(
                "MoE layer {layer} is outside {} layers",
                weights.config().layer_count
            ));
        }
        let prefix = format!("blk.{layer}");
        Ok(Self {
            geometry: Qwen4ExpMoeMetalGeometry::from_config(weights.config())?,
            router: weights.require_tensor(&format!("{prefix}.ffn_gate_inp.weight"))?,
            routed_gate: weights.require_tensor(&format!("{prefix}.ffn_gate_exps.weight"))?,
            routed_up: weights.require_tensor(&format!("{prefix}.ffn_up_exps.weight"))?,
            routed_down: weights.require_tensor(&format!("{prefix}.ffn_down_exps.weight"))?,
            shared_router: weights
                .require_tensor(&format!("{prefix}.ffn_gate_inp_shexp.weight"))?,
            shared_gate: weights.require_tensor(&format!("{prefix}.ffn_gate_shexp.weight"))?,
            shared_up: weights.require_tensor(&format!("{prefix}.ffn_up_shexp.weight"))?,
            shared_down: weights.require_tensor(&format!("{prefix}.ffn_down_shexp.weight"))?,
        })
    }
}

pub struct Qwen4ExpMoeMetalWorkspace {
    geometry: Qwen4ExpMoeMetalGeometry,
    router_logits: MetalTensor,
    topk_ids: MetalTensor,
    topk_weights: MetalTensor,
    shared_gate: MetalTensor,
    routed_inner: MetalTensor,
    routed_expert_output: MetalTensor,
    shared_inner: MetalTensor,
    shared_output: MetalTensor,
    output: MetalTensor,
    active_command: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
    state_poisoned: bool,
}

impl Qwen4ExpMoeMetalWorkspace {
    pub fn new(
        ctx: &MetalContext,
        geometry: Qwen4ExpMoeMetalGeometry,
    ) -> Result<Self, Qwen4ExpMoeError> {
        geometry.validate()?;
        Ok(Self {
            geometry,
            router_logits: MetalTensor::zeros_f32(ctx, vec![geometry.expert_count as u64])?,
            topk_ids: MetalTensor::zeros_i32(ctx, vec![geometry.experts_per_token as u64])?,
            topk_weights: MetalTensor::zeros_f32(ctx, vec![geometry.experts_per_token as u64])?,
            shared_gate: MetalTensor::zeros_f32(ctx, vec![1])?,
            routed_inner: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.routed_intermediate_size as u64,
                    geometry.experts_per_token as u64,
                ],
            )?,
            routed_expert_output: MetalTensor::zeros_f32(
                ctx,
                vec![
                    geometry.hidden_size as u64,
                    geometry.experts_per_token as u64,
                ],
            )?,
            shared_inner: MetalTensor::zeros_f32(
                ctx,
                vec![geometry.shared_intermediate_size as u64],
            )?,
            shared_output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            output: MetalTensor::zeros_f32(ctx, vec![geometry.hidden_size as u64])?,
            active_command: None,
            state_poisoned: false,
        })
    }

    pub fn geometry(&self) -> Qwen4ExpMoeMetalGeometry {
        self.geometry
    }

    pub fn is_poisoned(&self) -> bool {
        self.state_poisoned
    }

    pub fn reset(&mut self) -> Result<(), Qwen4ExpMoeError> {
        self.require_idle()?;
        self.state_poisoned = false;
        Ok(())
    }

    pub fn release_after(&mut self) -> Result<(), Qwen4ExpMoeError> {
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
            Err(Qwen4ExpMoeError::CommandBuffer(format!(
                "status={status:?}, error={error:?}"
            )))
        }
    }

    /// Release a workspace from a command buffer that will never be committed.
    ///
    /// # Safety
    ///
    /// The caller must end and permanently discard every reference to the
    /// owning command buffer. Committing it later may race a subsequent owner.
    pub unsafe fn abandon_uncommitted(&mut self) -> Result<(), Qwen4ExpMoeError> {
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
        self.state_poisoned = false;
        Ok(())
    }

    fn require_idle(&self) -> Result<(), Qwen4ExpMoeError> {
        if self.active_command.is_some() {
            invalid("workspace is still owned by a command buffer")
        } else {
            Ok(())
        }
    }
}

#[must_use = "consume or copy the MoE output in its owning command, then release the workspace"]
pub struct Qwen4ExpMoeMetalRead<'a> {
    workspace: &'a mut Qwen4ExpMoeMetalWorkspace,
}

pub struct Qwen4ExpMoeMetalOutput<'a> {
    workspace: &'a Qwen4ExpMoeMetalWorkspace,
}

impl Qwen4ExpMoeMetalRead<'_> {
    pub fn output(&self) -> Qwen4ExpMoeMetalOutput<'_> {
        Qwen4ExpMoeMetalOutput {
            workspace: self.workspace,
        }
    }
}

impl Qwen4ExpMoeMetalOutput<'_> {
    pub fn n_elements(&self) -> u64 {
        self.workspace.geometry.hidden_size as u64
    }

    pub fn dtype(&self) -> GgmlType {
        GgmlType::F32
    }

    pub fn encode_copy_to(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        destination: &MetalTensor,
    ) -> Result<(), Qwen4ExpMoeError> {
        validate_encoder(ctx, enc)?;
        let command = enc.parent_command_buffer();
        let Some(owner) = self.workspace.active_command.as_ref() else {
            return invalid("MoE output has no owning command buffer");
        };
        if !std::ptr::addr_eq(Retained::as_ptr(owner), Retained::as_ptr(&command)) {
            return invalid("MoE output must be copied by its owning command buffer");
        }
        require_tensor(
            "MoE copied output destination",
            destination,
            GgmlType::F32,
            &[self.workspace.geometry.hidden_size as u64],
            true,
        )?;
        require_same_device(
            ctx,
            &[
                ("MoE output", &self.workspace.output),
                ("MoE copied output destination", destination),
            ],
        )?;
        let mut tensors = workspace_tensors(self.workspace);
        tensors.push(("MoE copied output destination", destination));
        require_disjoint(&tensors)?;
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

pub fn encode_qwen4exp_moe<'a>(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    workspace: &'a mut Qwen4ExpMoeMetalWorkspace,
) -> Result<Qwen4ExpMoeMetalRead<'a>, Qwen4ExpMoeError> {
    validate_encoder(ctx, enc)?;
    if workspace.state_poisoned {
        return invalid("workspace state is indeterminate; reset it before reuse");
    }
    workspace.require_idle()?;
    if weights.geometry != workspace.geometry {
        return invalid("MoE weight and workspace geometry differ");
    }
    validate_contract(ctx, input, weights, workspace)?;
    preflight(ctx, weights)?;
    reserve_command(workspace, enc)?;

    if let Err(error) = encode_step(ctx, enc, input, weights, workspace) {
        workspace.state_poisoned = true;
        return Err(error);
    }
    Ok(Qwen4ExpMoeMetalRead { workspace })
}

fn encode_step(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    workspace: &Qwen4ExpMoeMetalWorkspace,
) -> Result<(), Qwen4ExpMoeError> {
    let g = workspace.geometry;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.router,
        input,
        &workspace.router_logits,
        g.hidden_size,
        g.expert_count,
    )?;
    encode_topk_logits_softmax_f32(
        ctx,
        enc,
        &workspace.router_logits,
        &workspace.topk_ids,
        &workspace.topk_weights,
        g.expert_count,
        g.experts_per_token,
    )?;
    encode_dot_sigmoid_f32(
        ctx,
        enc,
        weights.shared_router,
        input,
        &workspace.shared_gate,
        g.hidden_size,
    )?;

    match weights.routed_gate.dtype {
        GgmlType::IQ3_XXS => {
            let encode = if qwen4exp_moe_iq3_fast_enabled() {
                encode_moe_swiglu_iq3_xxs_f32_fast
            } else {
                encode_moe_swiglu_iq3_xxs_f32
            };
            encode(
                ctx,
                enc,
                weights.routed_gate,
                weights.routed_up,
                input,
                &workspace.topk_ids,
                &workspace.routed_inner,
                g.hidden_size,
                g.routed_intermediate_size,
                g.expert_count,
                g.experts_per_token,
            )?
        }
        GgmlType::IQ4_XS => encode_moe_swiglu_iq4_xs_f32(
            ctx,
            enc,
            weights.routed_gate,
            weights.routed_up,
            input,
            &workspace.topk_ids,
            &workspace.routed_inner,
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            g.experts_per_token,
        )?,
        dtype => return invalid(format!("unsupported routed gate/up dtype {dtype:?}")),
    }

    match weights.routed_down.dtype {
        GgmlType::IQ4_NL => {
            encode_moe_down_iq4_nl_f32(
                ctx,
                enc,
                weights.routed_down,
                &workspace.routed_inner,
                &workspace.topk_ids,
                &workspace.routed_expert_output,
                g.routed_intermediate_size,
                g.hidden_size,
                g.expert_count,
                g.experts_per_token,
            )?;
            encode_moe_weighted_sum_f32(
                ctx,
                enc,
                &workspace.routed_expert_output,
                &workspace.topk_weights,
                &workspace.output,
                g.hidden_size,
                g.experts_per_token,
            )?;
        }
        GgmlType::Q8_0 => encode_moe_down_weighted_sum_q8_0_f32(
            ctx,
            enc,
            weights.routed_down,
            &workspace.routed_inner,
            &workspace.topk_ids,
            &workspace.topk_weights,
            &workspace.output,
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            g.experts_per_token,
        )?,
        dtype => return invalid(format!("unsupported routed down dtype {dtype:?}")),
    }

    encode_shared_swiglu_q8_0_f32(
        ctx,
        enc,
        weights.shared_gate,
        weights.shared_up,
        input,
        &workspace.shared_inner,
        g.hidden_size,
        g.shared_intermediate_size,
    )?;
    encode_mat_vec_dispatch(
        ctx,
        enc,
        weights.shared_down,
        &workspace.shared_inner,
        &workspace.shared_output,
        g.shared_intermediate_size,
        g.hidden_size,
    )?;
    encode_axpy_scalar_f32(
        ctx,
        enc,
        &workspace.shared_output,
        &workspace.shared_gate,
        &workspace.output,
    )?;
    Ok(())
}

pub(crate) fn validate_contract(
    ctx: &MetalContext,
    input: &MetalTensor,
    weights: Qwen4ExpMoeMetalWeights<'_>,
    workspace: &Qwen4ExpMoeMetalWorkspace,
) -> Result<(), Qwen4ExpMoeError> {
    let g = workspace.geometry;
    require_tensor(
        "MoE input",
        input,
        GgmlType::F32,
        &[g.hidden_size as u64],
        false,
    )?;
    require_offset_alignment("MoE input", input, 16)?;
    require_projection(
        "MoE router",
        weights.router,
        g.hidden_size,
        g.expert_count,
        &[GgmlType::F32],
    )?;
    require_offset_alignment("MoE router", weights.router, 16)?;
    require_expert_bank(
        "MoE routed gate bank",
        weights.routed_gate,
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        &[GgmlType::IQ3_XXS, GgmlType::IQ4_XS],
    )?;
    require_expert_bank(
        "MoE routed up bank",
        weights.routed_up,
        g.hidden_size,
        g.routed_intermediate_size,
        g.expert_count,
        &[GgmlType::IQ3_XXS, GgmlType::IQ4_XS],
    )?;
    if weights.routed_gate.dtype != weights.routed_up.dtype {
        return invalid(format!(
            "MoE routed gate/up dtypes differ: {:?}/{:?}",
            weights.routed_gate.dtype, weights.routed_up.dtype
        ));
    }
    require_expert_bank(
        "MoE routed down bank",
        weights.routed_down,
        g.routed_intermediate_size,
        g.hidden_size,
        g.expert_count,
        &[GgmlType::IQ4_NL, GgmlType::Q8_0],
    )?;
    require_tensor(
        "MoE shared router",
        weights.shared_router,
        GgmlType::F32,
        &[g.hidden_size as u64],
        false,
    )?;
    require_projection(
        "MoE shared gate",
        weights.shared_gate,
        g.hidden_size,
        g.shared_intermediate_size,
        &[GgmlType::Q8_0],
    )?;
    require_projection(
        "MoE shared up",
        weights.shared_up,
        g.hidden_size,
        g.shared_intermediate_size,
        &[GgmlType::Q8_0],
    )?;
    require_projection(
        "MoE shared down",
        weights.shared_down,
        g.shared_intermediate_size,
        g.hidden_size,
        &[GgmlType::Q8_0],
    )?;

    for (name, tensor, dtype, shape) in [
        (
            "MoE router logits",
            &workspace.router_logits,
            GgmlType::F32,
            vec![g.expert_count as u64],
        ),
        (
            "MoE top-k IDs",
            &workspace.topk_ids,
            GgmlType::I32,
            vec![g.experts_per_token as u64],
        ),
        (
            "MoE top-k weights",
            &workspace.topk_weights,
            GgmlType::F32,
            vec![g.experts_per_token as u64],
        ),
        (
            "MoE shared gate scalar",
            &workspace.shared_gate,
            GgmlType::F32,
            vec![1],
        ),
        (
            "MoE routed inner",
            &workspace.routed_inner,
            GgmlType::F32,
            vec![
                g.routed_intermediate_size as u64,
                g.experts_per_token as u64,
            ],
        ),
        (
            "MoE routed expert output",
            &workspace.routed_expert_output,
            GgmlType::F32,
            vec![g.hidden_size as u64, g.experts_per_token as u64],
        ),
        (
            "MoE shared inner",
            &workspace.shared_inner,
            GgmlType::F32,
            vec![g.shared_intermediate_size as u64],
        ),
        (
            "MoE shared output",
            &workspace.shared_output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
        (
            "MoE output",
            &workspace.output,
            GgmlType::F32,
            vec![g.hidden_size as u64],
        ),
    ] {
        require_tensor(name, tensor, dtype, &shape, true)?;
    }

    let named_weights = [
        ("MoE router", weights.router),
        ("MoE routed gate bank", weights.routed_gate),
        ("MoE routed up bank", weights.routed_up),
        ("MoE routed down bank", weights.routed_down),
        ("MoE shared router", weights.shared_router),
        ("MoE shared gate", weights.shared_gate),
        ("MoE shared up", weights.shared_up),
        ("MoE shared down", weights.shared_down),
    ];
    require_read_only_weights(&named_weights)?;

    let mut tensors = vec![("MoE input", input)];
    tensors.extend(named_weights);
    tensors.extend(workspace_tensors(workspace));
    require_same_device(ctx, &tensors)?;
    require_disjoint(&tensors)
}

fn workspace_tensors(workspace: &Qwen4ExpMoeMetalWorkspace) -> Vec<(&'static str, &MetalTensor)> {
    vec![
        ("MoE router logits", &workspace.router_logits),
        ("MoE top-k IDs", &workspace.topk_ids),
        ("MoE top-k weights", &workspace.topk_weights),
        ("MoE shared gate scalar", &workspace.shared_gate),
        ("MoE routed inner", &workspace.routed_inner),
        ("MoE routed expert output", &workspace.routed_expert_output),
        ("MoE shared inner", &workspace.shared_inner),
        ("MoE shared output", &workspace.shared_output),
        ("MoE output", &workspace.output),
    ]
}

pub(crate) fn preflight(
    ctx: &MetalContext,
    weights: Qwen4ExpMoeMetalWeights<'_>,
) -> Result<(), Qwen4ExpMoeError> {
    preflight_projection(ctx, weights.router.dtype)?;
    preflight_projection(ctx, weights.shared_down.dtype)?;
    let (routed_gate_kernel, routed_gate_threads) = match weights.routed_gate.dtype {
        GgmlType::IQ3_XXS if qwen4exp_moe_iq3_fast_enabled() => {
            ("kernel_moe_swiglu_iq3_xxs_f32_fast", 64)
        }
        GgmlType::IQ3_XXS => ("kernel_moe_swiglu_iq3_xxs_f32", 64),
        GgmlType::IQ4_XS => ("kernel_moe_swiglu_iq4_xs_f32", 128),
        dtype => return invalid(format!("unsupported routed gate/up dtype {dtype:?}")),
    };
    require_pipeline_capacity(ctx, "kernel_topk_logits_softmax_f32", 1, 0)?;
    require_pipeline_capacity(ctx, "kernel_dot_sigmoid_f32", 32, 0)?;
    require_pipeline_capacity(ctx, routed_gate_kernel, routed_gate_threads, 0)?;
    match weights.routed_down.dtype {
        GgmlType::IQ4_NL => {
            require_pipeline_capacity(ctx, "kernel_moe_down_iq4_nl_f32", 128, 0)?;
            ctx.pipeline("kernel_moe_weighted_sum_f32")?;
        }
        GgmlType::Q8_0 => require_pipeline_capacity(
            ctx,
            "kernel_moe_down_weighted_sum_q8_0_f32",
            128,
            32 * 2 * size_of::<f32>(),
        )?,
        dtype => return invalid(format!("unsupported routed down dtype {dtype:?}")),
    }
    require_pipeline_capacity(
        ctx,
        "kernel_shared_swiglu_q8_0_f32_lcpp",
        128,
        32 * 2 * 2 * size_of::<f32>(),
    )?;
    ctx.pipeline("kernel_axpy_scalar_f32")?;
    Ok(())
}

fn preflight_projection(ctx: &MetalContext, dtype: GgmlType) -> Result<(), Qwen4ExpMoeError> {
    let kernels: &[(&str, usize, usize)] = match dtype {
        GgmlType::F32 => &[
            ("kernel_mat_vec_f32_f32", 128, 0),
            (
                "kernel_mat_vec_f32_f32_lcpp_r2",
                128,
                32 * 2 * size_of::<f32>(),
            ),
        ],
        GgmlType::Q8_0 => &[
            ("kernel_mat_vec_q8_0_f32", 64, 0),
            (
                "kernel_mat_vec_q8_0_f32_lcpp",
                128,
                32 * 2 * size_of::<f32>(),
            ),
        ],
        _ => return invalid(format!("unsupported MoE projection dtype {dtype:?}")),
    };
    for &(kernel, threads, dynamic_memory) in kernels {
        require_pipeline_capacity(ctx, kernel, threads, dynamic_memory)?;
    }
    Ok(())
}

fn require_pipeline_capacity(
    ctx: &MetalContext,
    name: &str,
    threads: usize,
    dynamic_memory: usize,
) -> Result<(), Qwen4ExpMoeError> {
    let pipeline = ctx.pipeline(name)?;
    let execution_width = pipeline.threadExecutionWidth();
    if execution_width != 32 {
        return invalid(format!(
            "MoE pipeline {name} requires SIMD width 32, got {execution_width}"
        ));
    }
    let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
    if max_threads < threads {
        return invalid(format!(
            "MoE pipeline {name} requires {threads} threads per threadgroup, got {max_threads}"
        ));
    }
    let static_memory = pipeline.staticThreadgroupMemoryLength();
    let required_memory = static_memory
        .checked_add(dynamic_memory)
        .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("MoE pipeline {name} memory overflow")))?;
    let available_memory = ctx.device.maxThreadgroupMemoryLength();
    if required_memory > available_memory {
        return invalid(format!(
            "MoE pipeline {name} requires {required_memory} threadgroup bytes ({static_memory} static plus {dynamic_memory} dynamic), device exposes {available_memory}"
        ));
    }
    Ok(())
}

fn validate_encoder(ctx: &MetalContext, enc: &KernelEncoder) -> Result<(), Qwen4ExpMoeError> {
    let actual = enc.parent_command_buffer().device().registryID();
    let expected = ctx.device.registryID();
    if actual != expected {
        return invalid(format!(
            "encoder belongs to Metal device registry {actual}, context is {expected}"
        ));
    }
    if enc.is_concurrent() {
        return invalid("dependent MoE dispatches require a serial encoder");
    }
    Ok(())
}

fn reserve_command(
    workspace: &mut Qwen4ExpMoeMetalWorkspace,
    enc: &KernelEncoder,
) -> Result<(), Qwen4ExpMoeError> {
    workspace.require_idle()?;
    workspace.active_command = Some(enc.parent_command_buffer());
    Ok(())
}

fn require_projection(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    allowed_dtypes: &[GgmlType],
) -> Result<(), Qwen4ExpMoeError> {
    let shape = [n_in as u64, n_out as u64];
    if tensor.shape != shape || !allowed_dtypes.contains(&tensor.dtype) {
        return invalid(format!(
            "{name} must use {allowed_dtypes:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    require_row_alignment(name, tensor, n_in)?;
    require_range(name, tensor)
}

fn require_expert_bank(
    name: &str,
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    allowed_dtypes: &[GgmlType],
) -> Result<(), Qwen4ExpMoeError> {
    let shape = [n_in as u64, n_out as u64, expert_count as u64];
    if tensor.shape != shape || !allowed_dtypes.contains(&tensor.dtype) {
        return invalid(format!(
            "{name} must use {allowed_dtypes:?} with shape {shape:?}, got {:?} {:?}",
            tensor.dtype, tensor.shape
        ));
    }
    require_row_alignment(name, tensor, n_in)?;
    require_range(name, tensor)
}

fn require_tensor(
    name: &str,
    tensor: &MetalTensor,
    dtype: GgmlType,
    shape: &[u64],
    writable: bool,
) -> Result<(), Qwen4ExpMoeError> {
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

fn require_row_alignment(
    name: &str,
    tensor: &MetalTensor,
    row_elements: usize,
) -> Result<(), Qwen4ExpMoeError> {
    let (block_elements, _) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpMoeError::Invalid(format!("{name} has unsupported dtype {:?}", tensor.dtype))
    })?;
    if !(row_elements as u64).is_multiple_of(block_elements) {
        return invalid(format!(
            "{name} row width {row_elements} is not aligned to {block_elements} elements for {:?}",
            tensor.dtype
        ));
    }
    Ok(())
}

fn storage_bytes(tensor: &MetalTensor) -> Result<u64, Qwen4ExpMoeError> {
    let elements = tensor
        .shape
        .iter()
        .try_fold(1_u64, |product, &dimension| product.checked_mul(dimension))
        .ok_or_else(|| Qwen4ExpMoeError::Invalid("tensor element count overflow".into()))?;
    let (block_elements, block_bytes) = tensor.dtype.storage_layout().ok_or_else(|| {
        Qwen4ExpMoeError::Invalid(format!("unsupported tensor dtype {:?}", tensor.dtype))
    })?;
    if block_elements == 0 || !elements.is_multiple_of(block_elements) {
        return invalid(format!(
            "tensor shape {:?} is not block-aligned for {:?}",
            tensor.shape, tensor.dtype
        ));
    }
    (elements / block_elements)
        .checked_mul(block_bytes)
        .ok_or_else(|| Qwen4ExpMoeError::Invalid("tensor byte count overflow".into()))
}

fn require_range(name: &str, tensor: &MetalTensor) -> Result<(), Qwen4ExpMoeError> {
    let alignment = match tensor.dtype {
        GgmlType::F32 | GgmlType::I32 => 4,
        GgmlType::IQ3_XXS | GgmlType::IQ4_XS | GgmlType::IQ4_NL | GgmlType::Q8_0 => 2,
        dtype => return invalid(format!("{name} has unsupported dtype {dtype:?}")),
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
        .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range offset={} bytes={bytes} exceeds buffer={}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    Ok(())
}

fn require_offset_alignment(
    name: &str,
    tensor: &MetalTensor,
    alignment: u64,
) -> Result<(), Qwen4ExpMoeError> {
    if alignment == 0 || !tensor.offset.is_multiple_of(alignment) {
        return invalid(format!(
            "{name} offset {} is not {alignment}-byte aligned",
            tensor.offset
        ));
    }
    Ok(())
}

fn require_read_only_weights(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpMoeError> {
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
) -> Result<(), Qwen4ExpMoeError> {
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

fn require_disjoint(tensors: &[(&str, &MetalTensor)]) -> Result<(), Qwen4ExpMoeError> {
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

fn checked_product(factors: &[usize], name: &str) -> Result<usize, Qwen4ExpMoeError> {
    factors.iter().try_fold(1_usize, |product, &factor| {
        product
            .checked_mul(factor)
            .ok_or_else(|| Qwen4ExpMoeError::Invalid(format!("MoE {name} element count overflow")))
    })
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, Qwen4ExpMoeError> {
    Err(Qwen4ExpMoeError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::qwen4exp_residency::Qwen4ExpMetalWeightPlan;
    use crate::tensor::TensorDesc;
    use objc2_metal::MTLCommandQueue;
    use serde::Deserialize;
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    const ROUTING_ORACLE_JSON: &str =
        include_str!("../tests/fixtures/qwen4exp_moe_routing_v1.json");
    const ROUTING_ORACLE_F32: &[u8] =
        include_bytes!("../tests/fixtures/qwen4exp_moe_routing_v1.f32");

    #[derive(Deserialize)]
    struct RoutingOracle {
        schema_version: u32,
        generator_version: u32,
        geometry: OracleGeometry,
        semantics: OracleSemantics,
        topk_ids: Vec<i32>,
        sentinels: OracleSentinels,
        binary: OracleBinary,
    }

    #[derive(Deserialize)]
    struct OracleGeometry {
        hidden_size: usize,
        expert_count: usize,
        experts_per_token: usize,
        routed_intermediate_size: usize,
        shared_intermediate_size: usize,
    }

    #[derive(Deserialize)]
    struct OracleSentinels {
        exact_tie_ids: Vec<i32>,
        selected_weight_sum: f32,
        full_softmax_selected_sum_without_renormalization: f32,
    }

    #[derive(Deserialize)]
    struct OracleSemantics {
        router: String,
        tie_break: String,
        normalization: String,
        shared_gate: String,
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

    struct CpuMoeResult {
        routed_inner: Vec<f32>,
        routed_expert_output: Vec<f32>,
        shared_inner: Vec<f32>,
        shared_output: Vec<f32>,
        output: Vec<f32>,
    }

    fn metal_context() -> Option<MetalContext> {
        match MetalContext::new() {
            Ok(ctx) => Some(ctx),
            Err(MetalError::EmptyLibrary) | Err(MetalError::NoDevice) => None,
            Err(error) => panic!("Metal initialization failed: {error}"),
        }
    }

    fn parse_oracle() -> (RoutingOracle, Vec<f32>) {
        let oracle: RoutingOracle = serde_json::from_str(ROUTING_ORACLE_JSON).unwrap();
        assert_eq!(oracle.schema_version, 1);
        assert_eq!(oracle.generator_version, 1);
        assert_eq!(oracle.binary.file, "qwen4exp_moe_routing_v1.f32");
        assert_eq!(oracle.binary.dtype, "f32");
        assert_eq!(oracle.binary.byte_order, "little");
        assert_eq!(oracle.semantics.router, "F32 matrix-vector product");
        assert_eq!(oracle.semantics.tie_break, "lower expert ID first");
        assert_eq!(
            oracle.semantics.normalization,
            "softmax over selected logits"
        );
        assert_eq!(
            oracle.semantics.shared_gate,
            "sigmoid(dot(shared_router, input))"
        );
        assert_eq!(oracle.sentinels.exact_tie_ids, [3, 14]);
        assert_eq!(oracle.sentinels.selected_weight_sum, 1.0);
        assert!(
            oracle
                .sentinels
                .full_softmax_selected_sum_without_renormalization
                < 0.9
        );
        assert_eq!(
            format!("{:x}", Sha256::digest(ROUTING_ORACLE_F32)),
            oracle.binary.sha256
        );
        assert!(ROUTING_ORACLE_F32.len().is_multiple_of(4));
        let values = ROUTING_ORACLE_F32
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_le_bytes(*bytes))
            .collect::<Vec<_>>();
        let required_sections = [
            "input",
            "router",
            "shared_router",
            "router_logits",
            "topk_weights",
            "shared_gate",
        ];
        assert_eq!(oracle.binary.sections.len(), required_sections.len());
        for name in required_sections {
            assert!(oracle.binary.sections.contains_key(name));
        }
        let mut sections = oracle.binary.sections.values().collect::<Vec<_>>();
        sections.sort_by_key(|section| section.offset_f32);
        let mut cursor = 0_usize;
        for section in sections {
            assert_eq!(section.offset_f32, cursor);
            assert_eq!(section.shape.iter().product::<usize>(), section.count_f32);
            cursor = cursor.checked_add(section.count_f32).unwrap();
            assert!(cursor <= values.len());
        }
        assert_eq!(cursor, values.len());
        (oracle, values)
    }

    fn oracle_section<'a>(oracle: &RoutingOracle, values: &'a [f32], name: &str) -> &'a [f32] {
        let section = &oracle.binary.sections[name];
        assert_eq!(section.shape.iter().product::<usize>(), section.count_f32);
        &values[section.offset_f32..section.offset_f32 + section.count_f32]
    }

    fn geometry(oracle: &RoutingOracle) -> Qwen4ExpMoeMetalGeometry {
        Qwen4ExpMoeMetalGeometry::new(
            oracle.geometry.hidden_size,
            oracle.geometry.expert_count,
            oracle.geometry.experts_per_token,
            oracle.geometry.routed_intermediate_size,
            oracle.geometry.shared_intermediate_size,
        )
        .unwrap()
    }

    fn tensor_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32).unwrap()
    }

    fn weight_f32(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> MetalTensor {
        let mut tensor = tensor_f32(ctx, values, shape);
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

    fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
        assert_eq!(tensor.dtype, GgmlType::I32);
        unsafe {
            let source = tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>();
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

    fn mat_vec(weight: &[f32], input: &[f32], n_in: usize, n_out: usize) -> Vec<f32> {
        assert_eq!(weight.len(), n_in * n_out);
        assert_eq!(input.len(), n_in);
        weight
            .chunks_exact(n_in)
            .take(n_out)
            .map(|row| {
                row.iter()
                    .zip(input)
                    .map(|(weight, value)| weight * value)
                    .sum()
            })
            .collect()
    }

    fn silu(value: f32) -> f32 {
        value / (1.0 + (-value).exp())
    }

    fn encode_iq4_xs_block(d: f32, seed: usize) -> [u8; 136] {
        let mut block = [0_u8; 136];
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        let mut scales_h = 0_u16;
        for subblock in 0..8 {
            let scale = (seed * 11 + subblock * 7 + 3) % 64;
            block[4 + subblock / 2] |= ((scale & 0x0f) as u8) << (4 * (subblock % 2));
            scales_h |= ((scale >> 4) as u16) << (2 * subblock);
            for lane in 0..16 {
                let low = (seed + subblock * 5 + lane * 3) % 16;
                let high = (seed * 7 + subblock * 3 + lane * 5 + 1) % 16;
                block[8 + subblock * 16 + lane] = low as u8 | ((high as u8) << 4);
            }
        }
        block[2..4].copy_from_slice(&scales_h.to_le_bytes());
        block
    }

    fn synthetic_iq4_xs_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
        assert!(n_in.is_multiple_of(256));
        let blocks_per_row = n_in / 256;
        let mut bytes = Vec::with_capacity(experts * n_out * blocks_per_row * 136);
        for expert in 0..experts {
            for row in 0..n_out {
                for block in 0..blocks_per_row {
                    let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                    let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                    let d = sign * (ordinal % 5 + 1) as f32 / 65_536.0;
                    bytes.extend_from_slice(&encode_iq4_xs_block(d, ordinal));
                }
            }
        }
        bytes
    }

    fn encode_iq4_nl_block(d: f32, seed: usize) -> [u8; 18] {
        let mut block = [0_u8; 18];
        block[..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for lane in 0..16 {
            let low = (seed + lane * 3) % 16;
            let high = (seed * 5 + lane * 7 + 1) % 16;
            block[2 + lane] = low as u8 | ((high as u8) << 4);
        }
        block
    }

    fn synthetic_iq4_nl_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
        assert!(n_in.is_multiple_of(32));
        let blocks_per_row = n_in / 32;
        let mut bytes = Vec::with_capacity(experts * n_out * blocks_per_row * 18);
        for expert in 0..experts {
            for row in 0..n_out {
                for block in 0..blocks_per_row {
                    let ordinal = (expert * n_out + row) * blocks_per_row + block + seed;
                    let sign = if ordinal.is_multiple_of(2) { 1.0 } else { -1.0 };
                    let d = sign * (ordinal % 7 + 1) as f32 / 16_384.0;
                    bytes.extend_from_slice(&encode_iq4_nl_block(d, ordinal));
                }
            }
        }
        bytes
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

    fn synthetic_q8_0_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<u8> {
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

    fn synthetic_f32_bank(n_in: usize, n_out: usize, experts: usize, seed: usize) -> Vec<f32> {
        (0..n_in * n_out * experts)
            .map(|index| {
                let row = index / n_in;
                let column = index % n_in;
                let raw = (row * 29 + column * 17 + index / 13 * 7 + seed * 11 + 5) % 127;
                (raw as f32 - 63.0) * 0.0015
            })
            .collect()
    }

    fn quantize_rows(values: &[f32], dtype: GgmlType, n_per_row: usize) -> Vec<u8> {
        assert!(!values.is_empty());
        assert!(values.len().is_multiple_of(n_per_row));
        let (block_elements, block_bytes) = dtype.storage_layout().unwrap();
        assert!((n_per_row as u64).is_multiple_of(block_elements));
        let expected_bytes = values.len() / block_elements as usize * block_bytes as usize;
        let mut bytes = vec![0_u8; expected_bytes];
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
            assert_eq!(written, expected_bytes);
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

    fn dequant_expert(
        bytes: &[u8],
        dtype: GgmlType,
        n_in: usize,
        n_out: usize,
        expert: usize,
    ) -> Vec<f32> {
        let (block_elements, block_bytes) = dtype.storage_layout().unwrap();
        let row_bytes = n_in / block_elements as usize * block_bytes as usize;
        let expert_bytes = n_out * row_bytes;
        let start = expert * expert_bytes;
        dequant_matrix(&bytes[start..start + expert_bytes], dtype, n_in, n_out)
    }

    struct CpuQuantizedInputs<'a> {
        geometry: Qwen4ExpMoeMetalGeometry,
        input: &'a [f32],
        topk_ids: &'a [i32],
        topk_weights: &'a [f32],
        shared_gate_scalar: f32,
        routed_gate: &'a [u8],
        routed_up: &'a [u8],
        routed_dtype: GgmlType,
        routed_down: &'a [u8],
        routed_down_dtype: GgmlType,
        shared_gate: &'a [u8],
        shared_up: &'a [u8],
        shared_down: &'a [u8],
    }

    fn cpu_quantized_moe(inputs: CpuQuantizedInputs<'_>) -> CpuMoeResult {
        let g = inputs.geometry;
        let mut routed_inner = vec![0.0; g.experts_per_token * g.routed_intermediate_size];
        for (slot, &expert) in inputs.topk_ids.iter().enumerate() {
            let expert = expert as usize;
            let gate = dequant_expert(
                inputs.routed_gate,
                inputs.routed_dtype,
                g.hidden_size,
                g.routed_intermediate_size,
                expert,
            );
            let up = dequant_expert(
                inputs.routed_up,
                inputs.routed_dtype,
                g.hidden_size,
                g.routed_intermediate_size,
                expert,
            );
            let gate = mat_vec(
                &gate,
                inputs.input,
                g.hidden_size,
                g.routed_intermediate_size,
            );
            let up = mat_vec(&up, inputs.input, g.hidden_size, g.routed_intermediate_size);
            for row in 0..g.routed_intermediate_size {
                routed_inner[slot * g.routed_intermediate_size + row] = silu(gate[row]) * up[row];
            }
        }

        let mut routed_expert_output = vec![0.0; g.experts_per_token * g.hidden_size];
        let mut output = vec![0.0; g.hidden_size];
        for (slot, (&expert, &route_weight)) in
            inputs.topk_ids.iter().zip(inputs.topk_weights).enumerate()
        {
            let down = dequant_expert(
                inputs.routed_down,
                inputs.routed_down_dtype,
                g.routed_intermediate_size,
                g.hidden_size,
                expert as usize,
            );
            let expert_output = mat_vec(
                &down,
                &routed_inner
                    [slot * g.routed_intermediate_size..(slot + 1) * g.routed_intermediate_size],
                g.routed_intermediate_size,
                g.hidden_size,
            );
            routed_expert_output[slot * g.hidden_size..(slot + 1) * g.hidden_size]
                .copy_from_slice(&expert_output);
            for (output, expert_value) in output.iter_mut().zip(expert_output) {
                *output += route_weight * expert_value;
            }
        }

        let shared_gate_weight = dequant_matrix(
            inputs.shared_gate,
            GgmlType::Q8_0,
            g.hidden_size,
            g.shared_intermediate_size,
        );
        let shared_up_weight = dequant_matrix(
            inputs.shared_up,
            GgmlType::Q8_0,
            g.hidden_size,
            g.shared_intermediate_size,
        );
        let shared_down_weight = dequant_matrix(
            inputs.shared_down,
            GgmlType::Q8_0,
            g.shared_intermediate_size,
            g.hidden_size,
        );
        let shared_gate = mat_vec(
            &shared_gate_weight,
            inputs.input,
            g.hidden_size,
            g.shared_intermediate_size,
        );
        let shared_up = mat_vec(
            &shared_up_weight,
            inputs.input,
            g.hidden_size,
            g.shared_intermediate_size,
        );
        let shared_inner = shared_gate
            .into_iter()
            .zip(shared_up)
            .map(|(gate, up)| silu(gate) * up)
            .collect::<Vec<_>>();
        let shared_output = mat_vec(
            &shared_down_weight,
            &shared_inner,
            g.shared_intermediate_size,
            g.hidden_size,
        );
        for (output, &shared) in output.iter_mut().zip(&shared_output) {
            *output += inputs.shared_gate_scalar * shared;
        }
        CpuMoeResult {
            routed_inner,
            routed_expert_output,
            shared_inner,
            shared_output,
            output,
        }
    }

    #[test]
    fn released_geometry_and_kernel_constraints_are_explicit() {
        let reference = Qwen4ExpConfig::flash_next_reference();
        let geometry = Qwen4ExpMoeMetalGeometry::from_config(&reference).unwrap();
        assert_eq!(geometry.hidden_size(), 2_560);
        assert_eq!(geometry.expert_count(), 512);
        assert_eq!(geometry.experts_per_token(), 10);
        assert_eq!(geometry.routed_intermediate_size(), 640);
        assert_eq!(geometry.shared_intermediate_size(), 640);
        assert!(Qwen4ExpMoeMetalGeometry::new(255, 16, 10, 32, 32).is_err());
        assert!(Qwen4ExpMoeMetalGeometry::new(256, 16, 17, 32, 32).is_err());
        assert!(Qwen4ExpMoeMetalGeometry::new(256, 16, 10, 31, 32).is_err());
        assert!(Qwen4ExpMoeMetalGeometry::new(256, 16, 10, 32, 31).is_err());
    }

    #[test]
    fn iq4_xs_q8_composition_matches_routing_fixture_and_cpu_dequant() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let (oracle, values) = parse_oracle();
        let g = geometry(&oracle);
        let input = oracle_section(&oracle, &values, "input");
        let router = oracle_section(&oracle, &values, "router");
        let shared_router = oracle_section(&oracle, &values, "shared_router");
        let expected_logits = oracle_section(&oracle, &values, "router_logits");
        let expected_weights = oracle_section(&oracle, &values, "topk_weights");
        let expected_shared_gate = oracle_section(&oracle, &values, "shared_gate")[0];

        let routed_gate_bytes = synthetic_iq4_xs_bank(
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            17,
        );
        let routed_up_bytes = synthetic_iq4_xs_bank(
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            10_003,
        );
        let routed_down_bytes = synthetic_q8_0_bank(
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            20_011,
        );
        let shared_gate_bytes =
            synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 30_007);
        let shared_up_bytes =
            synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 40_009);
        let shared_down_bytes =
            synthetic_q8_0_bank(g.shared_intermediate_size, g.hidden_size, 1, 50_021);
        let expected = cpu_quantized_moe(CpuQuantizedInputs {
            geometry: g,
            input,
            topk_ids: &oracle.topk_ids,
            topk_weights: expected_weights,
            shared_gate_scalar: expected_shared_gate,
            routed_gate: &routed_gate_bytes,
            routed_up: &routed_up_bytes,
            routed_dtype: GgmlType::IQ4_XS,
            routed_down: &routed_down_bytes,
            routed_down_dtype: GgmlType::Q8_0,
            shared_gate: &shared_gate_bytes,
            shared_up: &shared_up_bytes,
            shared_down: &shared_down_bytes,
        });

        let input_gpu = tensor_f32(&ctx, input, vec![g.hidden_size as u64]);
        let router_gpu = weight_f32(
            &ctx,
            router,
            vec![g.hidden_size as u64, g.expert_count as u64],
        );
        let routed_gate_gpu = weight_bytes(
            &ctx,
            &routed_gate_bytes,
            vec![
                g.hidden_size as u64,
                g.routed_intermediate_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ4_XS,
        );
        let routed_up_gpu = weight_bytes(
            &ctx,
            &routed_up_bytes,
            vec![
                g.hidden_size as u64,
                g.routed_intermediate_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ4_XS,
        );
        let routed_down_gpu = weight_bytes(
            &ctx,
            &routed_down_bytes,
            vec![
                g.routed_intermediate_size as u64,
                g.hidden_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::Q8_0,
        );
        let shared_router_gpu = weight_f32(&ctx, shared_router, vec![g.hidden_size as u64]);
        let shared_gate_gpu = weight_bytes(
            &ctx,
            &shared_gate_bytes,
            vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
            GgmlType::Q8_0,
        );
        let shared_up_gpu = weight_bytes(
            &ctx,
            &shared_up_bytes,
            vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
            GgmlType::Q8_0,
        );
        let shared_down_gpu = weight_bytes(
            &ctx,
            &shared_down_bytes,
            vec![g.shared_intermediate_size as u64, g.hidden_size as u64],
            GgmlType::Q8_0,
        );
        let weights = Qwen4ExpMoeMetalWeights {
            geometry: g,
            router: &router_gpu,
            routed_gate: &routed_gate_gpu,
            routed_up: &routed_up_gpu,
            routed_down: &routed_down_gpu,
            shared_router: &shared_router_gpu,
            shared_gate: &shared_gate_gpu,
            shared_up: &shared_up_gpu,
            shared_down: &shared_down_gpu,
        };
        let mut workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();
        let exported = MetalTensor::zeros_f32(&ctx, vec![g.hidden_size as u64]).unwrap();

        let mutable_router = tensor_f32(
            &ctx,
            router,
            vec![g.hidden_size as u64, g.expert_count as u64],
        );
        let bad_weights = Qwen4ExpMoeMetalWeights {
            router: &mutable_router,
            ..weights
        };
        let bad_command = ctx.queue.commandBuffer().unwrap();
        let bad_encoder = KernelEncoder::begin(&bad_command);
        let error =
            encode_qwen4exp_moe(&ctx, &bad_encoder, &input_gpu, bad_weights, &mut workspace)
                .err()
                .unwrap()
                .to_string();
        assert!(error.contains("read-only weight provenance"));
        bad_encoder.end();

        let mut padded_input = vec![0.0_f32; g.hidden_size + 1];
        padded_input[1..].copy_from_slice(input);
        let padded_input_gpu = tensor_f32(&ctx, &padded_input, vec![padded_input.len() as u64]);
        let misaligned_input = padded_input_gpu.view_subrange(1, vec![g.hidden_size as u64]);
        let misaligned_command = ctx.queue.commandBuffer().unwrap();
        let misaligned_encoder = KernelEncoder::begin(&misaligned_command);
        let error = encode_qwen4exp_moe(
            &ctx,
            &misaligned_encoder,
            &misaligned_input,
            weights,
            &mut workspace,
        )
        .err()
        .unwrap()
        .to_string();
        assert!(error.contains("not 16-byte aligned"));
        misaligned_encoder.end();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_qwen4exp_moe(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
        assert_eq!(read.output().dtype(), GgmlType::F32);
        assert_eq!(read.output().n_elements(), g.hidden_size as u64);

        let wrong_command = ctx.queue.commandBuffer().unwrap();
        let wrong_encoder = KernelEncoder::begin(&wrong_command);
        assert!(
            read.output()
                .encode_copy_to(&ctx, &wrong_encoder, &exported)
                .is_err()
        );
        wrong_encoder.end();
        read.output()
            .encode_copy_to(&ctx, &encoder, &exported)
            .unwrap();
        drop(read);
        encoder.end();
        assert!(workspace.release_after().is_err());
        command.commit();
        workspace.release_after().unwrap();

        assert_close(
            "router logits",
            &read_f32(&workspace.router_logits),
            expected_logits,
            2e-5,
            2e-5,
        );
        assert_eq!(read_i32(&workspace.topk_ids), oracle.topk_ids);
        assert_close(
            "top-k weights",
            &read_f32(&workspace.topk_weights),
            expected_weights,
            2e-6,
            2e-6,
        );
        assert_close(
            "shared gate",
            &read_f32(&workspace.shared_gate),
            &[expected_shared_gate],
            2e-6,
            2e-6,
        );
        assert_close(
            "routed inner",
            &read_f32(&workspace.routed_inner),
            &expected.routed_inner,
            2e-4,
            2e-4,
        );
        assert_close(
            "shared inner",
            &read_f32(&workspace.shared_inner),
            &expected.shared_inner,
            2e-5,
            2e-4,
        );
        assert_close(
            "shared output",
            &read_f32(&workspace.shared_output),
            &expected.shared_output,
            2e-5,
            2e-4,
        );
        assert_close(
            "MoE output",
            &read_f32(&workspace.output),
            &expected.output,
            3e-4,
            3e-4,
        );
        assert_close(
            "copied MoE output",
            &read_f32(&exported),
            &expected.output,
            3e-4,
            3e-4,
        );
        assert!(!workspace.is_poisoned());
    }

    #[test]
    fn iq3_xxs_iq4_nl_path_is_zero_safe_and_abandonable() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let (oracle, values) = parse_oracle();
        let g = geometry(&oracle);
        let input = oracle_section(&oracle, &values, "input");
        let router = oracle_section(&oracle, &values, "router");
        let shared_router = oracle_section(&oracle, &values, "shared_router");
        let expected_weights = oracle_section(&oracle, &values, "topk_weights");
        let expected_shared_gate = oracle_section(&oracle, &values, "shared_gate")[0];
        let routed_elements = g.hidden_size * g.routed_intermediate_size * g.expert_count;
        let routed_bytes = vec![0_u8; routed_elements / 256 * 98];
        let routed_down_bytes = synthetic_iq4_nl_bank(
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            60_013,
        );
        let shared_gate_bytes =
            synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 70_001);
        let shared_up_bytes =
            synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 80_021);
        let shared_down_bytes =
            synthetic_q8_0_bank(g.shared_intermediate_size, g.hidden_size, 1, 90_007);
        let expected = cpu_quantized_moe(CpuQuantizedInputs {
            geometry: g,
            input,
            topk_ids: &oracle.topk_ids,
            topk_weights: expected_weights,
            shared_gate_scalar: expected_shared_gate,
            routed_gate: &routed_bytes,
            routed_up: &routed_bytes,
            routed_dtype: GgmlType::IQ3_XXS,
            routed_down: &routed_down_bytes,
            routed_down_dtype: GgmlType::IQ4_NL,
            shared_gate: &shared_gate_bytes,
            shared_up: &shared_up_bytes,
            shared_down: &shared_down_bytes,
        });
        assert!(expected.routed_inner.iter().all(|&value| value == 0.0));
        assert!(
            expected
                .routed_expert_output
                .iter()
                .all(|&value| value == 0.0)
        );

        let input_gpu = tensor_f32(&ctx, input, vec![g.hidden_size as u64]);
        let router_gpu = weight_f32(
            &ctx,
            router,
            vec![g.hidden_size as u64, g.expert_count as u64],
        );
        let routed_gate_gpu = weight_bytes(
            &ctx,
            &routed_bytes,
            vec![
                g.hidden_size as u64,
                g.routed_intermediate_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ3_XXS,
        );
        let routed_up_gpu = weight_bytes(
            &ctx,
            &routed_bytes,
            vec![
                g.hidden_size as u64,
                g.routed_intermediate_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ3_XXS,
        );
        let routed_down_gpu = weight_bytes(
            &ctx,
            &routed_down_bytes,
            vec![
                g.routed_intermediate_size as u64,
                g.hidden_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ4_NL,
        );
        let shared_router_gpu = weight_f32(&ctx, shared_router, vec![g.hidden_size as u64]);
        let shared_gate_gpu = weight_bytes(
            &ctx,
            &shared_gate_bytes,
            vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
            GgmlType::Q8_0,
        );
        let shared_up_gpu = weight_bytes(
            &ctx,
            &shared_up_bytes,
            vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
            GgmlType::Q8_0,
        );
        let shared_down_gpu = weight_bytes(
            &ctx,
            &shared_down_bytes,
            vec![g.shared_intermediate_size as u64, g.hidden_size as u64],
            GgmlType::Q8_0,
        );
        let weights = Qwen4ExpMoeMetalWeights {
            geometry: g,
            router: &router_gpu,
            routed_gate: &routed_gate_gpu,
            routed_up: &routed_up_gpu,
            routed_down: &routed_down_gpu,
            shared_router: &shared_router_gpu,
            shared_gate: &shared_gate_gpu,
            shared_up: &shared_up_gpu,
            shared_down: &shared_down_gpu,
        };
        let mut workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();

        let concurrent_command = ctx.queue.commandBuffer().unwrap();
        let concurrent_encoder = KernelEncoder::begin_concurrent(&concurrent_command);
        assert!(
            encode_qwen4exp_moe(
                &ctx,
                &concurrent_encoder,
                &input_gpu,
                weights,
                &mut workspace,
            )
            .is_err()
        );
        concurrent_encoder.end();

        let command = ctx.queue.commandBuffer().unwrap();
        let encoder = KernelEncoder::begin(&command);
        let read =
            encode_qwen4exp_moe(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
        drop(read);
        encoder.end();
        command.commit();
        workspace.release_after().unwrap();
        assert!(
            read_f32(&workspace.routed_inner)
                .iter()
                .all(|&value| value == 0.0)
        );
        assert!(
            read_f32(&workspace.routed_expert_output)
                .iter()
                .all(|&value| value == 0.0)
        );
        assert_close(
            "IQ3/IQ4 MoE output",
            &read_f32(&workspace.output),
            &expected.output,
            2e-5,
            2e-4,
        );

        let abandoned_command = ctx.queue.commandBuffer().unwrap();
        let abandoned_encoder = KernelEncoder::begin(&abandoned_command);
        let read = encode_qwen4exp_moe(
            &ctx,
            &abandoned_encoder,
            &input_gpu,
            weights,
            &mut workspace,
        )
        .unwrap();
        drop(read);
        abandoned_encoder.end();
        unsafe { workspace.abandon_uncommitted() }.unwrap();
        drop(abandoned_command);
        workspace.reset().unwrap();

        let reuse_command = ctx.queue.commandBuffer().unwrap();
        let reuse_encoder = KernelEncoder::begin(&reuse_command);
        let read =
            encode_qwen4exp_moe(&ctx, &reuse_encoder, &input_gpu, weights, &mut workspace).unwrap();
        drop(read);
        reuse_encoder.end();
        reuse_command.commit();
        workspace.release_after().unwrap();
        assert_close(
            "reused IQ3/IQ4 MoE output",
            &read_f32(&workspace.output),
            &expected.output,
            2e-5,
            2e-4,
        );

        workspace.state_poisoned = true;
        let poisoned_command = ctx.queue.commandBuffer().unwrap();
        let poisoned_encoder = KernelEncoder::begin(&poisoned_command);
        assert!(
            encode_qwen4exp_moe(&ctx, &poisoned_encoder, &input_gpu, weights, &mut workspace,)
                .is_err()
        );
        poisoned_encoder.end();
        workspace.reset().unwrap();
        assert!(!workspace.is_poisoned());
    }

    #[test]
    fn iq3_xxs_iq4_nl_and_q8_paths_match_nonzero_cpu_dequant() {
        let Some(ctx) = metal_context() else {
            return;
        };
        let (oracle, values) = parse_oracle();
        let g = geometry(&oracle);
        let input = oracle_section(&oracle, &values, "input");
        let router = oracle_section(&oracle, &values, "router");
        let shared_router = oracle_section(&oracle, &values, "shared_router");
        let expected_weights = oracle_section(&oracle, &values, "topk_weights");
        let expected_shared_gate = oracle_section(&oracle, &values, "shared_gate")[0];

        let gate_values = synthetic_f32_bank(
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            101,
        );
        let up_values = synthetic_f32_bank(
            g.hidden_size,
            g.routed_intermediate_size,
            g.expert_count,
            211,
        );
        let gate_bytes = quantize_rows(&gate_values, GgmlType::IQ3_XXS, g.hidden_size);
        let up_bytes = quantize_rows(&up_values, GgmlType::IQ3_XXS, g.hidden_size);
        let down_iq4_bytes = synthetic_iq4_nl_bank(
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            60_013,
        );
        let down_q8_bytes = synthetic_q8_0_bank(
            g.routed_intermediate_size,
            g.hidden_size,
            g.expert_count,
            65_017,
        );
        let shared_gate_bytes =
            synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 70_001);
        let shared_up_bytes =
            synthetic_q8_0_bank(g.hidden_size, g.shared_intermediate_size, 1, 80_021);
        let shared_down_bytes =
            synthetic_q8_0_bank(g.shared_intermediate_size, g.hidden_size, 1, 90_007);

        let expected_iq4 = cpu_quantized_moe(CpuQuantizedInputs {
            geometry: g,
            input,
            topk_ids: &oracle.topk_ids,
            topk_weights: expected_weights,
            shared_gate_scalar: expected_shared_gate,
            routed_gate: &gate_bytes,
            routed_up: &up_bytes,
            routed_dtype: GgmlType::IQ3_XXS,
            routed_down: &down_iq4_bytes,
            routed_down_dtype: GgmlType::IQ4_NL,
            shared_gate: &shared_gate_bytes,
            shared_up: &shared_up_bytes,
            shared_down: &shared_down_bytes,
        });
        let expected_q8 = cpu_quantized_moe(CpuQuantizedInputs {
            geometry: g,
            input,
            topk_ids: &oracle.topk_ids,
            topk_weights: expected_weights,
            shared_gate_scalar: expected_shared_gate,
            routed_gate: &gate_bytes,
            routed_up: &up_bytes,
            routed_dtype: GgmlType::IQ3_XXS,
            routed_down: &down_q8_bytes,
            routed_down_dtype: GgmlType::Q8_0,
            shared_gate: &shared_gate_bytes,
            shared_up: &shared_up_bytes,
            shared_down: &shared_down_bytes,
        });
        assert!(
            expected_iq4
                .routed_inner
                .iter()
                .any(|value| value.abs() > 1e-5)
        );
        assert!(
            expected_iq4
                .routed_expert_output
                .iter()
                .any(|value| value.abs() > 1e-5)
        );
        assert!(
            expected_q8
                .routed_expert_output
                .iter()
                .any(|value| value.abs() > 1e-5)
        );

        let input_gpu = tensor_f32(&ctx, input, vec![g.hidden_size as u64]);
        let router_gpu = weight_f32(
            &ctx,
            router,
            vec![g.hidden_size as u64, g.expert_count as u64],
        );
        let gate_gpu = weight_bytes(
            &ctx,
            &gate_bytes,
            vec![
                g.hidden_size as u64,
                g.routed_intermediate_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ3_XXS,
        );
        let up_gpu = weight_bytes(
            &ctx,
            &up_bytes,
            vec![
                g.hidden_size as u64,
                g.routed_intermediate_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ3_XXS,
        );
        let down_iq4_gpu = weight_bytes(
            &ctx,
            &down_iq4_bytes,
            vec![
                g.routed_intermediate_size as u64,
                g.hidden_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::IQ4_NL,
        );
        let down_q8_gpu = weight_bytes(
            &ctx,
            &down_q8_bytes,
            vec![
                g.routed_intermediate_size as u64,
                g.hidden_size as u64,
                g.expert_count as u64,
            ],
            GgmlType::Q8_0,
        );
        let shared_router_gpu = weight_f32(&ctx, shared_router, vec![g.hidden_size as u64]);
        let shared_gate_gpu = weight_bytes(
            &ctx,
            &shared_gate_bytes,
            vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
            GgmlType::Q8_0,
        );
        let shared_up_gpu = weight_bytes(
            &ctx,
            &shared_up_bytes,
            vec![g.hidden_size as u64, g.shared_intermediate_size as u64],
            GgmlType::Q8_0,
        );
        let shared_down_gpu = weight_bytes(
            &ctx,
            &shared_down_bytes,
            vec![g.shared_intermediate_size as u64, g.hidden_size as u64],
            GgmlType::Q8_0,
        );
        let iq4_weights = Qwen4ExpMoeMetalWeights {
            geometry: g,
            router: &router_gpu,
            routed_gate: &gate_gpu,
            routed_up: &up_gpu,
            routed_down: &down_iq4_gpu,
            shared_router: &shared_router_gpu,
            shared_gate: &shared_gate_gpu,
            shared_up: &shared_up_gpu,
            shared_down: &shared_down_gpu,
        };
        let q8_weights = Qwen4ExpMoeMetalWeights {
            routed_down: &down_q8_gpu,
            ..iq4_weights
        };

        let mut iq4_workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();
        let iq4_command = ctx.queue.commandBuffer().unwrap();
        let iq4_encoder = KernelEncoder::begin(&iq4_command);
        let read = encode_qwen4exp_moe(
            &ctx,
            &iq4_encoder,
            &input_gpu,
            iq4_weights,
            &mut iq4_workspace,
        )
        .unwrap();
        drop(read);
        iq4_encoder.end();
        iq4_command.commit();
        iq4_workspace.release_after().unwrap();
        assert_close(
            "nonzero IQ3 routed inner",
            &read_f32(&iq4_workspace.routed_inner),
            &expected_iq4.routed_inner,
            3e-4,
            3e-4,
        );
        assert_close(
            "nonzero IQ4_NL expert output",
            &read_f32(&iq4_workspace.routed_expert_output),
            &expected_iq4.routed_expert_output,
            5e-4,
            5e-4,
        );
        assert_close(
            "nonzero IQ3/IQ4 MoE output",
            &read_f32(&iq4_workspace.output),
            &expected_iq4.output,
            7e-4,
            7e-4,
        );

        let mut q8_workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();
        let q8_command = ctx.queue.commandBuffer().unwrap();
        let q8_encoder = KernelEncoder::begin(&q8_command);
        let read =
            encode_qwen4exp_moe(&ctx, &q8_encoder, &input_gpu, q8_weights, &mut q8_workspace)
                .unwrap();
        drop(read);
        q8_encoder.end();
        q8_command.commit();
        q8_workspace.release_after().unwrap();
        assert_close(
            "nonzero IQ3/Q8 MoE output",
            &read_f32(&q8_workspace.output),
            &expected_q8.output,
            7e-4,
            7e-4,
        );
    }

    fn stable_topk(logits: &[f32], top_k: usize) -> (Vec<i32>, Vec<f32>) {
        let mut ids = (0..logits.len()).collect::<Vec<_>>();
        ids.sort_by(|&left, &right| {
            logits[right]
                .total_cmp(&logits[left])
                .then_with(|| left.cmp(&right))
        });
        ids.truncate(top_k);
        let maximum = logits[ids[0]];
        let exponentials = ids
            .iter()
            .map(|&expert| (logits[expert] - maximum).exp())
            .collect::<Vec<_>>();
        let sum = exponentials.iter().sum::<f32>();
        (
            ids.into_iter().map(|expert| expert as i32).collect(),
            exponentials.into_iter().map(|value| value / sum).collect(),
        )
    }

    fn gguf_dequant(gguf: &GgufFile, name: &str) -> Vec<f32> {
        let desc = gguf.find(name).unwrap();
        crate::codec::dequant_to_f32(desc, gguf.try_slice(desc).unwrap()).unwrap()
    }

    fn gguf_dequant_expert(gguf: &GgufFile, name: &str, expert: usize) -> Vec<f32> {
        let desc = gguf.find(name).unwrap();
        assert_eq!(desc.shape.len(), 3);
        let n_in = desc.shape[0] as usize;
        let n_out = desc.shape[1] as usize;
        let (block_elements, block_bytes) = desc.dtype.storage_layout().unwrap();
        assert!((n_in as u64).is_multiple_of(block_elements));
        let row_bytes = n_in / block_elements as usize * block_bytes as usize;
        let expert_bytes = n_out * row_bytes;
        let start = expert * expert_bytes;
        let bytes = gguf.try_slice(desc).unwrap();
        let expert_desc = TensorDesc {
            name: format!("{name}.expert.{expert}"),
            shape: vec![n_in as u64, n_out as u64],
            dtype: desc.dtype,
            shard_idx: 0,
            data_offset: 0,
            n_bytes: expert_bytes as u64,
        };
        crate::codec::dequant_to_f32(&expert_desc, &bytes[start..start + expert_bytes]).unwrap()
    }

    fn real_input(router: &[f32], g: Qwen4ExpMoeMetalGeometry) -> (Vec<f32>, Vec<f32>) {
        for seed in 0..64 {
            let input = (0..g.hidden_size)
                .map(|index| {
                    let raw = ((index * 37 + index / 11 * 5 + seed * 17 + 3) % 257) as f32;
                    (raw - 128.0) * 0.00075
                })
                .collect::<Vec<_>>();
            let logits = mat_vec(router, &input, g.hidden_size, g.expert_count);
            let (ids, _) = stable_topk(&logits, g.experts_per_token);
            let mut sorted = logits.clone();
            sorted.sort_by(|left, right| right.total_cmp(left));
            let boundary_margin = sorted[g.experts_per_token - 1] - sorted[g.experts_per_token];
            if ids.iter().any(|&expert| expert >= 256) && boundary_margin > 1e-4 {
                return (input, logits);
            }
        }
        panic!("could not find a stable real-weight routing input with a high expert ID")
    }

    fn real_cpu_oracle(
        gguf: &GgufFile,
        layer: u32,
        g: Qwen4ExpMoeMetalGeometry,
    ) -> (Vec<f32>, Vec<f32>, Vec<i32>, Vec<f32>, f32, CpuMoeResult) {
        let prefix = format!("blk.{layer}");
        let router = gguf_dequant(gguf, &format!("{prefix}.ffn_gate_inp.weight"));
        let (input, logits) = real_input(&router, g);
        let (ids, weights) = stable_topk(&logits, g.experts_per_token);
        let shared_router = gguf_dequant(gguf, &format!("{prefix}.ffn_gate_inp_shexp.weight"));
        let shared_gate_scalar = 1.0
            / (1.0
                + (-shared_router
                    .iter()
                    .zip(&input)
                    .map(|(weight, input)| weight * input)
                    .sum::<f32>())
                .exp());

        let gate_desc = gguf
            .find(&format!("{prefix}.ffn_gate_exps.weight"))
            .unwrap();
        let down_desc = gguf
            .find(&format!("{prefix}.ffn_down_exps.weight"))
            .unwrap();
        let mut routed_inner = vec![0.0; g.experts_per_token * g.routed_intermediate_size];
        for (slot, &expert) in ids.iter().enumerate() {
            let gate = gguf_dequant_expert(
                gguf,
                &format!("{prefix}.ffn_gate_exps.weight"),
                expert as usize,
            );
            let up = gguf_dequant_expert(
                gguf,
                &format!("{prefix}.ffn_up_exps.weight"),
                expert as usize,
            );
            let gate = mat_vec(&gate, &input, g.hidden_size, g.routed_intermediate_size);
            let up = mat_vec(&up, &input, g.hidden_size, g.routed_intermediate_size);
            for row in 0..g.routed_intermediate_size {
                routed_inner[slot * g.routed_intermediate_size + row] = silu(gate[row]) * up[row];
            }
        }
        let mut routed_expert_output = vec![0.0; g.experts_per_token * g.hidden_size];
        let mut output = vec![0.0; g.hidden_size];
        for (slot, (&expert, &route_weight)) in ids.iter().zip(&weights).enumerate() {
            let down = gguf_dequant_expert(
                gguf,
                &format!("{prefix}.ffn_down_exps.weight"),
                expert as usize,
            );
            let expert_output = mat_vec(
                &down,
                &routed_inner
                    [slot * g.routed_intermediate_size..(slot + 1) * g.routed_intermediate_size],
                g.routed_intermediate_size,
                g.hidden_size,
            );
            routed_expert_output[slot * g.hidden_size..(slot + 1) * g.hidden_size]
                .copy_from_slice(&expert_output);
            for (output, expert_value) in output.iter_mut().zip(expert_output) {
                *output += route_weight * expert_value;
            }
        }

        let shared_gate_weight = gguf_dequant(gguf, &format!("{prefix}.ffn_gate_shexp.weight"));
        let shared_up_weight = gguf_dequant(gguf, &format!("{prefix}.ffn_up_shexp.weight"));
        let shared_down_weight = gguf_dequant(gguf, &format!("{prefix}.ffn_down_shexp.weight"));
        let shared_gate = mat_vec(
            &shared_gate_weight,
            &input,
            g.hidden_size,
            g.shared_intermediate_size,
        );
        let shared_up = mat_vec(
            &shared_up_weight,
            &input,
            g.hidden_size,
            g.shared_intermediate_size,
        );
        let shared_inner = shared_gate
            .into_iter()
            .zip(shared_up)
            .map(|(gate, up)| silu(gate) * up)
            .collect::<Vec<_>>();
        let shared_output = mat_vec(
            &shared_down_weight,
            &shared_inner,
            g.shared_intermediate_size,
            g.hidden_size,
        );
        for (output, &shared) in output.iter_mut().zip(&shared_output) {
            *output += shared_gate_scalar * shared;
        }
        assert!(matches!(
            gate_desc.dtype,
            GgmlType::IQ3_XXS | GgmlType::IQ4_XS
        ));
        assert!(matches!(down_desc.dtype, GgmlType::IQ4_NL | GgmlType::Q8_0));
        (
            input,
            logits,
            ids,
            weights,
            shared_gate_scalar,
            CpuMoeResult {
                routed_inner,
                routed_expert_output,
                shared_inner,
                shared_output,
                output,
            },
        )
    }

    fn assert_similarity(label: &str, actual: &[f32], expected: &[f32], max_abs: f32, cosine: f64) {
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
        let observed_cosine = dot / (actual_norm * expected_norm).max(1e-30);
        eprintln!("[{label}] max_abs={observed_max:.3e} cosine={observed_cosine:.9}");
        assert!(observed_max <= max_abs, "{label} max_abs={observed_max}");
        assert!(
            observed_cosine >= cosine,
            "{label} cosine={observed_cosine}"
        );
    }

    #[test]
    #[ignore = "set QWEN4EXP_Q3_K_XL_MOE_GGUF to the pinned first release shard"]
    fn released_layers_two_through_four_match_selected_expert_cpu_oracles() {
        let path = std::env::var_os("QWEN4EXP_Q3_K_XL_MOE_GGUF")
            .expect("QWEN4EXP_Q3_K_XL_MOE_GGUF must point to the first Q3 shard");
        let gguf = GgufFile::open(path).expect("open released UD-Q3_K_XL GGUF");
        let ctx = MetalContext::new().expect("initialize Metal");
        let plan = Qwen4ExpMetalWeightPlan::for_ud_q3_k_xl(&ctx, &gguf).unwrap();
        let admitted = plan.admit(ctx.memory_signals()).unwrap();
        let realized = Qwen4ExpMetalWeights::realize(&ctx, &gguf, admitted).unwrap();
        let metal_weights = realized.weights();
        let g = Qwen4ExpMoeMetalGeometry::from_config(metal_weights.config()).unwrap();
        let mut workspace = Qwen4ExpMoeMetalWorkspace::new(&ctx, g).unwrap();

        for layer in [2_u32, 3, 4] {
            let weights = Qwen4ExpMoeMetalWeights::bind(metal_weights, layer).unwrap();
            if layer == 2 {
                assert_eq!(weights.routed_gate.dtype, GgmlType::IQ4_XS);
                assert_eq!(weights.routed_down.dtype, GgmlType::Q8_0);
            } else if layer == 3 {
                assert_eq!(weights.routed_gate.dtype, GgmlType::IQ3_XXS);
                assert_eq!(weights.routed_down.dtype, GgmlType::IQ4_NL);
            } else {
                assert_eq!(weights.routed_gate.dtype, GgmlType::IQ3_XXS);
                assert_eq!(weights.routed_down.dtype, GgmlType::Q8_0);
            }
            let (input, logits, ids, route_weights, shared_gate, expected) =
                real_cpu_oracle(&gguf, layer, g);
            assert!(ids.iter().any(|&expert| expert >= 256));
            let input_gpu = tensor_f32(&ctx, &input, vec![g.hidden_size as u64]);
            let command = ctx.queue.commandBuffer().unwrap();
            let encoder = KernelEncoder::begin(&command);
            let read =
                encode_qwen4exp_moe(&ctx, &encoder, &input_gpu, weights, &mut workspace).unwrap();
            drop(read);
            encoder.end();
            command.commit();
            workspace.release_after().unwrap();

            assert_close(
                &format!("layer {layer} router logits"),
                &read_f32(&workspace.router_logits),
                &logits,
                2e-4,
                2e-4,
            );
            assert_eq!(read_i32(&workspace.topk_ids), ids);
            assert_close(
                &format!("layer {layer} route weights"),
                &read_f32(&workspace.topk_weights),
                &route_weights,
                2e-5,
                2e-5,
            );
            assert_close(
                &format!("layer {layer} shared gate"),
                &read_f32(&workspace.shared_gate),
                &[shared_gate],
                2e-5,
                2e-5,
            );
            assert_similarity(
                &format!("layer {layer} routed inner"),
                &read_f32(&workspace.routed_inner),
                &expected.routed_inner,
                if layer == 2 { 2e-3 } else { 3e-2 },
                0.999,
            );
            if layer == 3 {
                assert_similarity(
                    "layer 3 routed expert output",
                    &read_f32(&workspace.routed_expert_output),
                    &expected.routed_expert_output,
                    5e-2,
                    0.998,
                );
            }
            assert_similarity(
                &format!("layer {layer} shared inner"),
                &read_f32(&workspace.shared_inner),
                &expected.shared_inner,
                2e-3,
                0.999_99,
            );
            assert_similarity(
                &format!("layer {layer} shared output"),
                &read_f32(&workspace.shared_output),
                &expected.shared_output,
                5e-3,
                0.999_99,
            );
            assert_similarity(
                &format!("layer {layer} final output"),
                &read_f32(&workspace.output),
                &expected.output,
                if layer == 2 { 2e-2 } else { 8e-2 },
                0.998,
            );
        }
    }
}
