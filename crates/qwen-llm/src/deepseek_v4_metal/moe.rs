//! DeepSeek V4 MoE scratch, routing, and expert projections.

use super::*;

pub(super) const DEEPSEEK_V4_ROUTE_STATUS_READY: i32 = 1;

pub(super) const DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT: i32 = -1;

pub(super) const DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS: i32 = -2;

pub(super) const DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN: i32 = -3;

pub(super) const DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT: i32 = -4;

pub(super) const DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_WEIGHT: i32 = -6;

pub(super) const DEEPSEEK_V4_ROUTE_MAX_EXPERTS: usize = 256;

pub(super) const DEEPSEEK_V4_ROUTE_MAX_TOP_K: usize = 6;

/// Reusable session-owned storage for one native DS4 single-token MoE body.
///
/// Production routing publishes a transient GPU record consumed by indexed
/// expert projections in the same serial command. The host `route_*` and static
/// `encode_experts` methods remain independent differential oracles.
pub struct DeepSeekV4MoeScratch {
    pub(super) config: DeepSeekV4MoeConfig,
    pub(super) normalized_input: MetalTensor,
    pub(super) logits: MetalTensor,
    pub(super) expert_ids: MetalTensor,
    pub(super) weights: MetalTensor,
    pub(super) route_status: MetalTensor,
    pub(super) gate: MetalTensor,
    pub(super) up: MetalTensor,
    pub(super) inner: MetalTensor,
    pub(super) routed_inner: MetalTensor,
    pub(super) expert_outputs: MetalTensor,
    pub(super) routed_output: MetalTensor,
    pub(super) shared_output: MetalTensor,
    pub(super) final_output: MetalTensor,
}

pub(super) const DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH: usize = DEEPSEEK_V4_ROUTE_MAX_TOP_K + 1;

#[derive(Clone)]
pub(super) struct DeepSeekV4RouteRecord {
    pub(super) expert_ids: MetalTensor,
    pub(super) weights: MetalTensor,
    pub(super) status: MetalTensor,
}

impl DeepSeekV4RouteRecord {
    pub(super) fn validate(&self, config: DeepSeekV4MoeConfig) -> Result<(), DeepSeekV4MetalError> {
        validate_i32(
            &self.expert_ids,
            &[config.top_k as u64],
            true,
            "MoE route-record expert IDs",
        )?;
        validate_f32(
            &self.weights,
            &[config.top_k as u64],
            true,
            "MoE route-record weights",
        )?;
        validate_i32(&self.status, &[1], true, "MoE route-record status")
    }
}

pub(super) struct DeepSeekV4LayerRouteRecords {
    pub(super) integers: MetalTensor,
    pub(super) weights: MetalTensor,
    pub(super) config: DeepSeekV4MoeConfig,
}

pub(super) struct DeepSeekV4CompletedLayerRouteRecords {
    pub(super) integers: Vec<i32>,
    pub(super) weights: Vec<f32>,
    pub(super) config: DeepSeekV4MoeConfig,
}

impl DeepSeekV4LayerRouteRecords {
    pub(super) fn new(
        ctx: &MetalContext,
        config: DeepSeekV4MoeConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        config.checked()?;
        if config.top_k != DEEPSEEK_V4_ROUTE_MAX_TOP_K {
            return invalid(format!(
                "layer route records require top-k {DEEPSEEK_V4_ROUTE_MAX_TOP_K}, got {}",
                config.top_k
            ));
        }
        Ok(Self {
            integers: MetalTensor::zeros_i32(
                ctx,
                vec![
                    DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH as u64,
                    DEEPSEEK_V4_LAYER_COUNT as u64,
                ],
            )?,
            weights: MetalTensor::zeros_f32(
                ctx,
                vec![config.top_k as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            )?,
            config,
        })
    }

    pub(super) fn layer(
        &self,
        layer: usize,
    ) -> Result<DeepSeekV4RouteRecord, DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("MoE route-record layer {layer} is out of range"));
        }
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            true,
            "MoE layer-route integer records",
        )?;
        validate_f32(
            &self.weights,
            &[self.config.top_k as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            true,
            "MoE layer-route weight records",
        )?;
        let integer_base = checked_mul(
            layer,
            DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH,
            "MoE route-record integer layer offset",
        )? as u64;
        let weight_base = checked_mul(
            layer,
            self.config.top_k,
            "MoE route-record weight layer offset",
        )? as u64;
        let record = DeepSeekV4RouteRecord {
            expert_ids: self
                .integers
                .view_subrange(integer_base, vec![self.config.top_k as u64]),
            weights: self
                .weights
                .view_subrange(weight_base, vec![self.config.top_k as u64]),
            status: self
                .integers
                .view_subrange(integer_base + self.config.top_k as u64, vec![1]),
        };
        record.validate(self.config)?;
        Ok(record)
    }

    pub(super) fn reset_for_token(&self) -> Result<(), DeepSeekV4MetalError> {
        host_write_i32(
            &self.integers,
            &[0; DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH * DEEPSEEK_V4_LAYER_COUNT],
            "reset MoE layer-route records",
        )
    }

    pub(super) fn read_completed(
        &self,
    ) -> Result<DeepSeekV4CompletedLayerRouteRecords, DeepSeekV4MetalError> {
        validate_i32(
            &self.integers,
            &[
                DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH as u64,
                DEEPSEEK_V4_LAYER_COUNT as u64,
            ],
            false,
            "completed MoE layer-route integer records",
        )?;
        validate_f32(
            &self.weights,
            &[self.config.top_k as u64, DEEPSEEK_V4_LAYER_COUNT as u64],
            false,
            "completed MoE layer-route weight records",
        )?;
        Ok(DeepSeekV4CompletedLayerRouteRecords {
            integers: host_read_i32(&self.integers, "completed MoE layer-route integers")?,
            weights: host_read_f32(&self.weights, "completed MoE layer-route weights")?,
            config: self.config,
        })
    }
}

impl DeepSeekV4CompletedLayerRouteRecords {
    pub(super) fn validate_layer(&self, layer: usize) -> Result<(), DeepSeekV4MetalError> {
        if layer >= DEEPSEEK_V4_LAYER_COUNT {
            return invalid(format!("completed MoE route layer {layer} is out of range"));
        }
        let integer_base = checked_mul(
            layer,
            DEEPSEEK_V4_ROUTE_RECORD_I32_WIDTH,
            "completed MoE route integer layer offset",
        )?;
        let weight_base = checked_mul(
            layer,
            self.config.top_k,
            "completed MoE route weight layer offset",
        )?;
        let status = self.integers[integer_base + self.config.top_k];
        if status != DEEPSEEK_V4_ROUTE_STATUS_READY {
            return invalid(format!(
                "layer {layer} GPU route failed with status {status} ({})",
                deepseek_v4_route_status_name(status)
            ));
        }
        for slot in 0..self.config.top_k {
            let expert = self.integers[integer_base + slot];
            let Ok(expert_index) = usize::try_from(expert) else {
                return invalid(format!(
                    "layer {layer} GPU route slot {slot} returned negative expert {expert}"
                ));
            };
            if expert_index >= self.config.expert_count {
                return invalid(format!(
                    "layer {layer} GPU route slot {slot} returned expert {expert_index} outside {}",
                    self.config.expert_count
                ));
            }
            let weight = self.weights[weight_base + slot];
            if !weight.is_finite() || weight < 0.0 {
                return invalid(format!(
                    "layer {layer} GPU route slot {slot} returned invalid weight {weight}"
                ));
            }
        }
        Ok(())
    }
}

impl DeepSeekV4MoeScratch {
    pub fn new(
        ctx: &MetalContext,
        config: DeepSeekV4MoeConfig,
    ) -> Result<Self, DeepSeekV4MetalError> {
        config.checked()?;
        let c = config;
        let fused_q6_scratch = checked_mul(c.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        let ids = vec![0i32; c.top_k];
        Ok(Self {
            config,
            normalized_input: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
            logits: MetalTensor::zeros_f32(ctx, vec![c.expert_count as u64])?,
            expert_ids: MetalTensor::from_bytes(
                ctx,
                bytemuck::cast_slice(&ids),
                vec![c.top_k as u64],
                GgmlType::I32,
            )?,
            weights: MetalTensor::zeros_f32(ctx, vec![c.top_k as u64])?,
            route_status: MetalTensor::zeros_i32(ctx, vec![1])?,
            gate: MetalTensor::zeros_f32(ctx, vec![fused_q6_scratch as u64])?,
            up: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
            inner: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64])?,
            routed_inner: MetalTensor::zeros_f32(ctx, vec![c.ffn_size as u64, c.top_k as u64])?,
            expert_outputs: MetalTensor::zeros_f32(
                ctx,
                vec![c.hidden_size as u64, c.top_k as u64],
            )?,
            routed_output: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
            shared_output: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
            final_output: MetalTensor::zeros_f32(ctx, vec![c.hidden_size as u64])?,
        })
    }

    pub fn config(&self) -> DeepSeekV4MoeConfig {
        self.config
    }

    pub fn normalized_input(&self) -> &MetalTensor {
        &self.normalized_input
    }

    pub fn logits(&self) -> &MetalTensor {
        &self.logits
    }

    pub fn expert_ids(&self) -> &MetalTensor {
        &self.expert_ids
    }

    pub fn weights(&self) -> &MetalTensor {
        &self.weights
    }

    pub fn expert_outputs(&self) -> &MetalTensor {
        &self.expert_outputs
    }

    #[cfg(test)]
    pub(super) fn routed_inner(&self) -> &MetalTensor {
        &self.routed_inner
    }

    pub fn routed_output(&self) -> &MetalTensor {
        &self.routed_output
    }

    pub fn shared_output(&self) -> &MetalTensor {
        &self.shared_output
    }

    pub fn final_output(&self) -> &MetalTensor {
        &self.final_output
    }

    /// Encode input RMSNorm and the `[H,E]` router projection. The returned
    /// buffers are not host-readable until the caller completes the command.
    pub fn encode_router<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        input: &MetalTensor,
        ffn_norm: &MetalTensor,
        gate_inp: &MetalTensor,
        rms_eps: f32,
    ) -> Result<(&'a MetalTensor, &'a MetalTensor), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_router")?;
        validate_eps(rms_eps, "MoE RMSNorm epsilon")?;
        let c = self.config;
        self.validate_scratch()?;
        validate_f32(input, &[c.hidden_size as u64], false, "MoE input")?;
        validate_f32(ffn_norm, &[c.hidden_size as u64], false, "MoE norm weight")?;
        validate_matvec_weight(gate_inp, c.hidden_size, c.expert_count, "MoE router weight")?;
        encode_rms_norm_mul_f32(ctx, enc, input, ffn_norm, &self.normalized_input, rms_eps)?;
        encode_projection(
            ctx,
            enc,
            gate_inp,
            &self.normalized_input,
            &self.logits,
            c.hidden_size,
            c.expert_count,
            "MoE router",
        )?;
        Ok((&self.normalized_input, &self.logits))
    }

    pub(super) fn encode_route_learned_gpu(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        correction_bias: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_route_learned_gpu_into(ctx, enc, correction_bias, &record)
    }

    pub(super) fn encode_route_learned_gpu_into(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        correction_bias: &MetalTensor,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_f32(
            correction_bias,
            &[c.expert_count as u64],
            false,
            "GPU router correction bias",
        )?;
        self.encode_route_gpu(
            ctx,
            enc,
            correction_bias,
            0,
            c.expert_count,
            "kernel_deepseek_v4_route_learned",
            DEEPSEEK_V4_ROUTE_MAX_EXPERTS,
            record,
        )
    }

    pub(super) fn encode_route_hash_gpu(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_id: usize,
        token_to_expert: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_route_hash_gpu_into(ctx, enc, token_id, token_to_expert, &record)
    }

    pub(super) fn encode_route_hash_gpu_into(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        token_id: usize,
        token_to_expert: &MetalTensor,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_i32_bank(token_to_expert, c.top_k, "GPU token-to-expert map")?;
        let vocab_size = usize::try_from(token_to_expert.shape[1]).map_err(|_| {
            DeepSeekV4MetalError::Invalid("GPU hash vocabulary exceeds usize".into())
        })?;
        self.encode_route_gpu(
            ctx,
            enc,
            token_to_expert,
            token_id,
            vocab_size,
            "kernel_deepseek_v4_route_hash",
            1,
            record,
        )
    }

    pub(super) fn encode_route_gpu(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        auxiliary: &MetalTensor,
        token_id: usize,
        vocab_size: usize,
        kernel: &'static str,
        threads_per_group: usize,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, kernel)?;
        self.validate_scratch()?;
        let c = self.config;
        record.validate(c)?;
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Args {
            expert_count: u32,
            top_k: u32,
            token_id: u32,
            vocab_size: u32,
            routed_scale: f32,
        }
        let args = Args {
            expert_count: u32::try_from(c.expert_count).map_err(|_| {
                DeepSeekV4MetalError::Invalid("GPU route expert count exceeds u32".into())
            })?,
            top_k: u32::try_from(c.top_k)
                .map_err(|_| DeepSeekV4MetalError::Invalid("GPU route top-k exceeds u32".into()))?,
            token_id: u32::try_from(token_id).map_err(|_| {
                DeepSeekV4MetalError::Invalid("GPU route token ID exceeds u32".into())
            })?,
            vocab_size: u32::try_from(vocab_size).map_err(|_| {
                DeepSeekV4MetalError::Invalid("GPU route vocabulary exceeds u32".into())
            })?,
            routed_scale: c.routed_scale,
        };
        let pso = ctx.pipeline(kernel)?;
        validate_deepseek_v4_route_pipeline_geometry(
            kernel,
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup(),
            threads_per_group,
        )?;
        enc.set_pipeline(&pso);
        enc.set_bytes(0, &args);
        enc.set_tensor(1, &self.logits);
        enc.set_tensor(2, auxiliary);
        enc.set_tensor(3, &record.expert_ids);
        enc.set_tensor(4, &record.weights);
        enc.set_tensor(5, &record.status);
        enc.dispatch(
            MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width: threads_per_group,
                height: 1,
                depth: 1,
            },
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn validate_gpu_route_completed(&self) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.validate_gpu_route_record_completed(&record)
    }

    pub(super) fn validate_gpu_route_record_completed(
        &self,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        record.validate(self.config)?;
        let status = host_read_i32(&record.status, "GPU route status")?;
        if status.as_slice() != [DEEPSEEK_V4_ROUTE_STATUS_READY] {
            return invalid(format!(
                "GPU route failed with status {} ({})",
                status[0],
                deepseek_v4_route_status_name(status[0])
            ));
        }
        Ok(())
    }

    pub(super) fn default_route_record(&self) -> DeepSeekV4RouteRecord {
        DeepSeekV4RouteRecord {
            expert_ids: self.expert_ids.clone(),
            weights: self.weights.clone(),
            status: self.route_status.clone(),
        }
    }

    #[cfg(test)]
    pub(super) fn capture_gpu_route_record(
        &self,
    ) -> Result<DeepSeekV4GpuRouteRecord, DeepSeekV4MetalError> {
        let status = host_read_i32(&self.route_status, "GPU route status")?;
        Ok(DeepSeekV4GpuRouteRecord {
            status: status[0],
            expert_ids: host_read_i32(&self.expert_ids, "GPU selected expert IDs")?,
            weights: host_read_f32(&self.weights, "GPU selected expert weights")?,
        })
    }

    /// Host differential for token-major contiguous `[K,V]` I32 storage.
    pub fn route_hash(
        &self,
        token_id: usize,
        token_to_expert: &MetalTensor,
    ) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_i32_bank(token_to_expert, c.top_k, "token-to-expert map")?;
        let vocab = usize::try_from(token_to_expert.shape[1])
            .map_err(|_| DeepSeekV4MetalError::Invalid("hash vocabulary exceeds usize".into()))?;
        if token_id >= vocab {
            return invalid(format!(
                "hash token id {token_id} is outside vocabulary {vocab}"
            ));
        }
        let logits = host_read_f32(&self.logits, "MoE logits")?;
        let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits)
            .map_err(|error| DeepSeekV4MetalError::Invalid(format!("router scores: {error}")))?;
        let map = host_read_i32(token_to_expert, "token-to-expert map")?;
        let start = checked_mul(token_id, c.top_k, "hash route row offset")?;
        let mut selected = Vec::with_capacity(c.top_k);
        for &expert in &map[start..start + c.top_k] {
            let expert = usize::try_from(expert).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("hash route contains negative ID {expert}"))
            })?;
            if expert >= c.expert_count {
                return invalid(format!(
                    "hash route expert {expert} exceeds expert count {}",
                    c.expert_count
                ));
            }
            selected.push(expert);
        }
        let decision = crate::deepseek_v4_oracle::hash_route(&scores, &selected, c.routed_scale)
            .map_err(|error| DeepSeekV4MetalError::Invalid(format!("hash route: {error}")))?;
        self.store_route(&decision.expert_ids, &decision.weights)
    }

    /// Host differential for tie-stable learned routing. Selected weights use
    /// the unbiased `sqrt(softplus(logit))` score.
    pub fn route_learned(&self, correction_bias: &MetalTensor) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        validate_f32(
            correction_bias,
            &[c.expert_count as u64],
            false,
            "router correction bias",
        )?;
        let logits = host_read_f32(&self.logits, "MoE logits")?;
        let bias = host_read_f32(correction_bias, "router correction bias")?;
        let scores = crate::deepseek_v4_oracle::sqrt_softplus_scores(&logits)
            .map_err(|error| DeepSeekV4MetalError::Invalid(format!("router scores: {error}")))?;
        let decision =
            crate::deepseek_v4_oracle::learned_route(&scores, &bias, c.top_k, c.routed_scale)
                .map_err(|error| {
                    DeepSeekV4MetalError::Invalid(format!("learned route: {error}"))
                })?;
        self.store_route(&decision.expert_ids, &decision.weights)
    }

    /// Static-view differential for selected and shared expert execution.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_experts<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_clamp: f32,
        shared_clamp: f32,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_experts")?;
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("MoE expert clamp must be finite and positive");
        }
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("MoE shared clamp must be finite and positive");
        }
        let c = self.config;
        self.validate_scratch()?;
        validate_expert_bank(
            gate_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
            "routed down bank",
        )?;
        validate_matvec_weight(shared_gate, c.hidden_size, c.ffn_size, "shared gate")?;
        validate_matvec_weight(shared_up, c.hidden_size, c.ffn_size, "shared up")?;
        validate_matvec_weight(shared_down, c.ffn_size, c.hidden_size, "shared down")?;

        let ids = host_read_i32(&self.expert_ids, "selected expert IDs")?;
        for (slot, &id) in ids.iter().enumerate() {
            let expert = usize::try_from(id).map_err(|_| {
                DeepSeekV4MetalError::Invalid(format!("selected expert ID {id} is negative"))
            })?;
            if expert >= c.expert_count {
                return invalid(format!(
                    "selected expert ID {expert} exceeds expert count {}",
                    c.expert_count
                ));
            }
            let gate = expert_weight_view(
                gate_bank,
                c.hidden_size,
                c.ffn_size,
                expert,
                "routed gate slice",
            )?;
            let up = expert_weight_view(
                up_bank,
                c.hidden_size,
                c.ffn_size,
                expert,
                "routed up slice",
            )?;
            let down = expert_weight_view(
                down_bank,
                c.ffn_size,
                c.hidden_size,
                expert,
                "routed down slice",
            )?;
            let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
            encode_projection(
                ctx,
                enc,
                &gate,
                &self.normalized_input,
                &gate_scratch,
                c.hidden_size,
                c.ffn_size,
                "routed gate",
            )?;
            encode_projection(
                ctx,
                enc,
                &up,
                &self.normalized_input,
                &self.up,
                c.hidden_size,
                c.ffn_size,
                "routed up",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate_scratch,
                &self.up,
                &self.inner,
                expert_clamp,
            )?;
            let output = self
                .expert_outputs
                .view_subrange((slot * c.hidden_size) as u64, vec![c.hidden_size as u64]);
            encode_projection(
                ctx,
                enc,
                &down,
                &self.inner,
                &output,
                c.ffn_size,
                c.hidden_size,
                "routed down",
            )?;
        }

        let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
        encode_projection(
            ctx,
            enc,
            shared_gate,
            &self.normalized_input,
            &gate_scratch,
            c.hidden_size,
            c.ffn_size,
            "shared gate",
        )?;
        encode_projection(
            ctx,
            enc,
            shared_up,
            &self.normalized_input,
            &self.up,
            c.hidden_size,
            c.ffn_size,
            "shared up",
        )?;
        encode_ds4_clamped_swiglu(ctx, enc, &gate_scratch, &self.up, &self.inner, shared_clamp)?;
        encode_projection(
            ctx,
            enc,
            shared_down,
            &self.inner,
            &self.shared_output,
            c.ffn_size,
            c.hidden_size,
            "shared down",
        )?;
        crate::metal::encode_moe_weighted_sum_f32(
            ctx,
            enc,
            &self.expert_outputs,
            &self.weights,
            &self.routed_output,
            c.hidden_size,
            c.top_k,
        )?;
        crate::metal::encode_add_f32(
            ctx,
            enc,
            &self.routed_output,
            &self.shared_output,
            &self.final_output,
        )?;
        Ok(&self.final_output)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn validate_indexed_experts(
        &self,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_clamp: f32,
        shared_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("indexed MoE expert clamp must be finite and positive");
        }
        if !shared_clamp.is_finite() || shared_clamp <= 0.0 {
            return invalid("indexed MoE shared clamp must be finite and positive");
        }
        let c = self.config;
        self.validate_scratch()?;
        validate_expert_bank(
            gate_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "indexed routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "indexed routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
            "indexed routed down bank",
        )?;
        validate_matvec_weight(shared_gate, c.hidden_size, c.ffn_size, "shared gate")?;
        validate_matvec_weight(shared_up, c.hidden_size, c.ffn_size, "shared up")?;
        validate_matvec_weight(shared_down, c.ffn_size, c.hidden_size, "shared down")?;
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn encode_routed_experts_indexed(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_routed_experts_indexed_from_record(
            ctx,
            enc,
            gate_bank,
            up_bank,
            down_bank,
            expert_clamp,
            &record,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_routed_experts_indexed_from_record(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_routed_experts_indexed")?;
        let c = self.config;
        record.validate(c)?;
        let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
        for slot in 0..c.top_k {
            encode_ds4_indexed_expert_projection(
                ctx,
                enc,
                gate_bank,
                &self.normalized_input,
                &record.expert_ids,
                &record.status,
                &gate_scratch,
                c.hidden_size,
                c.ffn_size,
                c.expert_count,
                slot,
                "indexed routed gate",
            )?;
            encode_ds4_indexed_expert_projection(
                ctx,
                enc,
                up_bank,
                &self.normalized_input,
                &record.expert_ids,
                &record.status,
                &self.up,
                c.hidden_size,
                c.ffn_size,
                c.expert_count,
                slot,
                "indexed routed up",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate_scratch,
                &self.up,
                &self.inner,
                expert_clamp,
            )?;
            let output = self
                .expert_outputs
                .view_subrange((slot * c.hidden_size) as u64, vec![c.hidden_size as u64]);
            encode_ds4_indexed_expert_projection(
                ctx,
                enc,
                down_bank,
                &self.inner,
                &record.expert_ids,
                &record.status,
                &output,
                c.ffn_size,
                c.hidden_size,
                c.expert_count,
                slot,
                "indexed routed down",
            )?;
        }
        Ok(())
    }

    pub(super) fn encode_routed_experts_all_slots(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_routed_experts_all_slots_from_record(
            ctx,
            enc,
            gate_bank,
            up_bank,
            down_bank,
            expert_clamp,
            &record,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_routed_experts_all_slots_from_record(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_routed_experts_all_slots")?;
        let c = self.config;
        if c.top_k != DEEPSEEK_V4_ROUTE_MAX_TOP_K {
            return invalid(format!(
                "all-slot routed experts require top-k {DEEPSEEK_V4_ROUTE_MAX_TOP_K}, got {}",
                c.top_k
            ));
        }
        if !expert_clamp.is_finite() || expert_clamp <= 0.0 {
            return invalid("all-slot routed expert clamp must be finite and positive");
        }
        self.validate_scratch()?;
        record.validate(c)?;
        validate_expert_bank(
            gate_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "all-slot routed gate bank",
        )?;
        validate_expert_bank(
            up_bank,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            "all-slot routed up bank",
        )?;
        validate_expert_bank(
            down_bank,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
            "all-slot routed down bank",
        )?;
        if deepseek_v4_all_slots_q3q4_scope_qualified(
            &ctx.device.name().to_string(),
            c,
            gate_bank.dtype,
            up_bank.dtype,
            down_bank.dtype,
        ) {
            if !deepseek_v4_all_slots_q3q4_enabled() {
                return self.encode_routed_experts_indexed_from_record(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    down_bank,
                    expert_clamp,
                    record,
                );
            }
            if deepseek_v4_all_slots_q3q4_fast_enabled() {
                static REPORTED_FAST: std::sync::Once = std::sync::Once::new();
                REPORTED_FAST.call_once(|| {
                    eprintln!(
                        "deepseek_v4: fast all-slot Q3_K/Q4_K routed experts active for K160 REAP; arithmetic rollback=QWEN_DSV4_ALL_SLOTS_Q3Q4_FAST=0 serial rollback=QWEN_DSV4_ALL_SLOTS_Q3Q4=0"
                    );
                });
                return self.encode_routed_experts_all_slots_q3q4_fast_from_record(
                    ctx,
                    enc,
                    gate_bank,
                    up_bank,
                    down_bank,
                    expert_clamp,
                    record,
                );
            }
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: all-slot Q3_K/Q4_K routed experts active for K160 REAP; rollback=QWEN_DSV4_ALL_SLOTS_Q3Q4=0"
                );
            });
        }
        if gate_bank.dtype != up_bank.dtype
            || !matches!(
                gate_bank.dtype,
                GgmlType::IQ2_XS
                    | GgmlType::IQ2_S
                    | GgmlType::IQ3_XXS
                    | GgmlType::IQ3_S
                    | GgmlType::Q3_K
            )
        {
            return invalid(format!(
                "all-slot routed gate/up require matching IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, or Q3_K storage, got {:?}/{:?}",
                gate_bank.dtype, up_bank.dtype
            ));
        }
        if !matches!(
            down_bank.dtype,
            GgmlType::IQ3_XXS | GgmlType::MXFP4 | GgmlType::Q4_K
        ) {
            return invalid(format!(
                "all-slot routed down requires IQ3_XXS, MXFP4, or Q4_K storage, got {:?}",
                down_bank.dtype
            ));
        }
        let gate_kernel = match gate_bank.dtype {
            GgmlType::IQ2_XS => "kernel_deepseek_v4_all_slots_swiglu_iq2_xs_f32_fast",
            GgmlType::IQ2_S => "kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast",
            GgmlType::IQ3_XXS => "kernel_deepseek_v4_all_slots_swiglu_iq3_xxs_f32_fast",
            GgmlType::IQ3_S => "kernel_deepseek_v4_all_slots_swiglu_iq3_s_f32_fast",
            GgmlType::Q3_K => "kernel_deepseek_v4_all_slots_swiglu_q3_K_f32",
            _ => unreachable!("gate/up dtype was validated above"),
        };
        let (down_kernel, down_threads) = match down_bank.dtype {
            GgmlType::IQ3_XXS => ("kernel_deepseek_v4_all_slots_down_iq3_xxs_f32_fast", 64),
            GgmlType::MXFP4 => ("kernel_deepseek_v4_all_slots_down_mxfp4_f32", 128),
            GgmlType::Q4_K => ("kernel_deepseek_v4_all_slots_down_q4_K_f32", 64),
            _ => unreachable!("down dtype was validated above"),
        };
        validate_deepseek_v4_all_slot_pipeline(ctx, gate_kernel, 64, 64)?;
        validate_deepseek_v4_all_slot_pipeline(ctx, down_kernel, down_threads, 0)?;
        encode_ds4_all_slots_gate_up_swiglu(
            ctx,
            enc,
            gate_bank,
            up_bank,
            &self.normalized_input,
            &record.expert_ids,
            &record.status,
            &self.routed_inner,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
            expert_clamp,
        )?;
        encode_ds4_all_slots_down(
            ctx,
            enc,
            down_bank,
            &self.routed_inner,
            &record.expert_ids,
            &record.status,
            &self.expert_outputs,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_routed_experts_all_slots_q3q4_fast_from_record(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        expert_clamp: f32,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "DeepSeek V4 fast all-slot Q3_K/Q4_K routed experts")?;
        let c = self.config;
        record.validate(c)?;
        if c.ffn_size > c.hidden_size {
            return invalid(format!(
                "fast all-slot Q3_K/Q4_K requires FFN width {} <= hidden width {} for the up-projection scratch alias",
                c.ffn_size, c.hidden_size
            ));
        }
        let up_scratch = self.expert_outputs.view_subrange(
            0,
            vec![c.ffn_size as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        );
        encode_ds4_all_slots_q3_k_fast(
            ctx,
            enc,
            gate_bank,
            &self.normalized_input,
            &record.expert_ids,
            &record.status,
            &self.routed_inner,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
        )?;
        encode_ds4_all_slots_q3_k_fast(
            ctx,
            enc,
            up_bank,
            &self.normalized_input,
            &record.expert_ids,
            &record.status,
            &up_scratch,
            c.hidden_size,
            c.ffn_size,
            c.expert_count,
        )?;
        let routed_elements = c.ffn_size.checked_mul(c.top_k).ok_or_else(|| {
            DeepSeekV4MetalError::Invalid("fast all-slot routed width overflow".into())
        })?;
        let gate = self
            .routed_inner
            .view_subrange(0, vec![routed_elements as u64]);
        let up = up_scratch.view_subrange(0, vec![routed_elements as u64]);
        encode_ds4_clamped_swiglu(ctx, enc, &gate, &up, &gate, expert_clamp)?;
        encode_ds4_all_slots_q4_k_fast(
            ctx,
            enc,
            down_bank,
            &self.routed_inner,
            &record.expert_ids,
            &record.status,
            &self.expert_outputs,
            c.ffn_size,
            c.hidden_size,
            c.expert_count,
        )
    }

    pub(super) fn encode_shared_expert(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        shared_clamp: f32,
    ) -> Result<(), DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_shared_expert")?;
        let c = self.config;
        let fused_q6_scratch = checked_mul(c.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        let fused = deepseek_v4_decode_shared_swiglu_enabled()
            && shared_gate.dtype == shared_up.dtype
            && matches!(shared_gate.dtype, GgmlType::Q6_K | GgmlType::Q8_0);
        let shared_inner = if fused && shared_gate.dtype == GgmlType::Q6_K {
            self.gate.view_subrange(0, vec![c.ffn_size as u64])
        } else {
            self.inner.clone()
        };
        if fused {
            static REPORTED: std::sync::Once = std::sync::Once::new();
            REPORTED.call_once(|| {
                eprintln!(
                    "deepseek_v4: decode shared gate/up/clamped-SwiGLU runs one fused dispatch; rollback=QWEN_DSV4_DECODE_SHARED_SWIGLU=0"
                );
            });
            match shared_gate.dtype {
                GgmlType::Q6_K => {
                    let scratch = self.gate.view_subrange(0, vec![fused_q6_scratch as u64]);
                    crate::metal::encode_ds4_shared_swiglu_q6_k_f32(
                        ctx,
                        enc,
                        shared_gate,
                        shared_up,
                        &self.normalized_input,
                        &scratch,
                        c.hidden_size,
                        c.ffn_size,
                        shared_clamp,
                    )
                }
                GgmlType::Q8_0 => crate::metal::encode_ds4_shared_swiglu_q8_0_f32(
                    ctx,
                    enc,
                    shared_gate,
                    shared_up,
                    &self.normalized_input,
                    &shared_inner,
                    c.hidden_size,
                    c.ffn_size,
                    shared_clamp,
                ),
                _ => unreachable!("fused shared-expert dtype was qualified"),
            }
            .map_err(DeepSeekV4MetalError::Metal)?;
        } else {
            let gate_scratch = self.gate.view_subrange(0, vec![c.ffn_size as u64]);
            encode_projection(
                ctx,
                enc,
                shared_gate,
                &self.normalized_input,
                &gate_scratch,
                c.hidden_size,
                c.ffn_size,
                "shared gate",
            )?;
            encode_projection(
                ctx,
                enc,
                shared_up,
                &self.normalized_input,
                &self.up,
                c.hidden_size,
                c.ffn_size,
                "shared up",
            )?;
            encode_ds4_clamped_swiglu(
                ctx,
                enc,
                &gate_scratch,
                &self.up,
                &self.inner,
                shared_clamp,
            )?;
        }
        encode_projection(
            ctx,
            enc,
            shared_down,
            &shared_inner,
            &self.shared_output,
            c.ffn_size,
            c.hidden_size,
            "shared down",
        )?;
        Ok(())
    }

    pub(super) fn encode_expert_combine<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        let record = self.default_route_record();
        self.encode_expert_combine_from_record(ctx, enc, &record)
    }

    pub(super) fn encode_expert_combine_from_record<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        require_serial(enc, "deepseek_v4_moe_expert_combine")?;
        let c = self.config;
        record.validate(c)?;
        crate::metal::encode_moe_weighted_sum_f32(
            ctx,
            enc,
            &self.expert_outputs,
            &record.weights,
            &self.routed_output,
            c.hidden_size,
            c.top_k,
        )?;
        crate::metal::encode_add_f32(
            ctx,
            enc,
            &self.routed_output,
            &self.shared_output,
            &self.final_output,
        )?;
        Ok(&self.final_output)
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    pub(super) fn encode_experts_indexed<'a>(
        &'a self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        gate_bank: &MetalTensor,
        up_bank: &MetalTensor,
        down_bank: &MetalTensor,
        shared_gate: &MetalTensor,
        shared_up: &MetalTensor,
        shared_down: &MetalTensor,
        expert_clamp: f32,
        shared_clamp: f32,
    ) -> Result<&'a MetalTensor, DeepSeekV4MetalError> {
        self.validate_indexed_experts(
            gate_bank,
            up_bank,
            down_bank,
            shared_gate,
            shared_up,
            shared_down,
            expert_clamp,
            shared_clamp,
        )?;
        self.encode_routed_experts_indexed(ctx, enc, gate_bank, up_bank, down_bank, expert_clamp)?;
        self.encode_shared_expert(ctx, enc, shared_gate, shared_up, shared_down, shared_clamp)?;
        self.encode_expert_combine(ctx, enc)
    }

    pub(super) fn store_route(
        &self,
        expert_ids: &[usize],
        weights: &[f32],
    ) -> Result<(), DeepSeekV4MetalError> {
        if expert_ids.len() != self.config.top_k || weights.len() != self.config.top_k {
            return invalid("router returned an unexpected top-k length");
        }
        let ids = expert_ids
            .iter()
            .map(|&id| {
                i32::try_from(id).map_err(|_| {
                    DeepSeekV4MetalError::Invalid(format!("expert ID {id} exceeds i32"))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        host_write_i32(&self.expert_ids, &ids, "selected expert IDs")?;
        host_write_f32(&self.weights, weights, "selected expert weights")
    }

    #[cfg(feature = "dsv4-diagnostics")]
    pub(super) fn capture_route_decision(
        &self,
        record: &DeepSeekV4RouteRecord,
    ) -> Result<DeepSeekV4RouteDecision, DeepSeekV4MetalError> {
        record.validate(self.config)?;
        Ok(diagnostics::build_route_decision(
            host_read_i32(&record.expert_ids, "diagnostic routed expert IDs")?,
            host_read_f32(&record.weights, "diagnostic routed expert weights")?,
            self.config.expert_count,
            self.config.routed_scale,
        )?)
    }

    pub(super) fn validate_scratch(&self) -> Result<(), DeepSeekV4MetalError> {
        let c = self.config;
        c.checked()?;
        let fused_q6_scratch = checked_mul(c.ffn_size, 3, "MoE gate and fused Q6 scratch")?;
        validate_f32(
            &self.normalized_input,
            &[c.hidden_size as u64],
            true,
            "MoE normalized input scratch",
        )?;
        validate_f32(
            &self.logits,
            &[c.expert_count as u64],
            true,
            "MoE logits scratch",
        )?;
        validate_i32(&self.expert_ids, &[c.top_k as u64], true, "MoE ID scratch")?;
        validate_f32(&self.weights, &[c.top_k as u64], true, "MoE weight scratch")?;
        validate_i32(&self.route_status, &[1], true, "MoE route status")?;
        validate_f32(
            &self.gate,
            &[fused_q6_scratch as u64],
            true,
            "MoE gate and fused Q6 scratch",
        )?;
        for (tensor, name) in [
            (&self.up, "MoE up scratch"),
            (&self.inner, "MoE inner scratch"),
        ] {
            validate_f32(tensor, &[c.ffn_size as u64], true, name)?;
        }
        validate_f32(
            &self.routed_inner,
            &[c.ffn_size as u64, c.top_k as u64],
            true,
            "MoE all-slot inner scratch",
        )?;
        validate_f32(
            &self.expert_outputs,
            &[c.hidden_size as u64, c.top_k as u64],
            true,
            "MoE expert output scratch",
        )?;
        for (tensor, name) in [
            (&self.routed_output, "MoE routed output scratch"),
            (&self.shared_output, "MoE shared output scratch"),
            (&self.final_output, "MoE final output scratch"),
        ] {
            validate_f32(tensor, &[c.hidden_size as u64], true, name)?;
        }
        Ok(())
    }
}

pub(super) const DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE: &str = "Apple M4 Max";

pub(super) const DEEPSEEK_V4_ALL_SLOTS_Q3Q4_EXPERT_COUNT: usize = 160;

pub(super) const DEEPSEEK_V4_ALL_SLOTS_Q3Q4_FFN_SIZE: usize = 2_048;

pub(super) fn deepseek_v4_all_slots_q3q4_scope_qualified(
    device_name: &str,
    config: DeepSeekV4MoeConfig,
    gate_dtype: GgmlType,
    up_dtype: GgmlType,
    down_dtype: GgmlType,
) -> bool {
    device_name == DEEPSEEK_V4_ALL_SLOTS_Q3Q4_QUALIFIED_DEVICE
        && config.hidden_size == DEEPSEEK_V4_HIDDEN_SIZE
        && config.ffn_size == DEEPSEEK_V4_ALL_SLOTS_Q3Q4_FFN_SIZE
        && config.expert_count == DEEPSEEK_V4_ALL_SLOTS_Q3Q4_EXPERT_COUNT
        && config.top_k == DEEPSEEK_V4_ROUTE_MAX_TOP_K
        && gate_dtype == GgmlType::Q3_K
        && up_dtype == GgmlType::Q3_K
        && down_dtype == GgmlType::Q4_K
}

pub(super) fn validate_deepseek_v4_route_pipeline_geometry(
    kernel: &str,
    thread_execution_width: usize,
    max_threads_per_group: usize,
    requested_threads: usize,
) -> Result<(), DeepSeekV4MetalError> {
    if max_threads_per_group < requested_threads {
        return invalid(format!(
            "{kernel} supports only {max_threads_per_group} threads, need {requested_threads}"
        ));
    }
    if requested_threads == DEEPSEEK_V4_ROUTE_MAX_EXPERTS && thread_execution_width != 32 {
        return invalid(format!(
            "{kernel} requires 32-lane simdgroups for eight-group reduction, got {thread_execution_width}"
        ));
    }
    Ok(())
}

#[cfg(test)]
#[derive(Clone, Debug)]
pub(super) struct DeepSeekV4GpuRouteRecord {
    pub(super) status: i32,
    pub(super) expert_ids: Vec<i32>,
    pub(super) weights: Vec<f32>,
}

pub(super) fn deepseek_v4_route_status_name(status: i32) -> &'static str {
    match status {
        0 => "pending",
        DEEPSEEK_V4_ROUTE_STATUS_READY => "ready",
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_LOGIT => "non-finite logit",
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_BIAS => "non-finite bias or corrected score",
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_TOKEN => "invalid hash token",
        DEEPSEEK_V4_ROUTE_STATUS_INVALID_EXPERT => "invalid hash expert",
        DEEPSEEK_V4_ROUTE_STATUS_NONFINITE_WEIGHT => "non-finite normalized weight",
        _ => "unknown",
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_indexed_expert_projection(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    slot: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    require_serial(enc, name)?;
    if expert_ids.shape.len() != 1 || expert_ids.shape[0] == 0 {
        return invalid(format!(
            "{name} expert IDs must be a nonempty I32 vector, got {:?}",
            expert_ids.shape
        ));
    }
    validate_i32(
        expert_ids,
        &expert_ids.shape,
        false,
        &format!("{name} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{name} route status"))?;
    let top_k = usize::try_from(expert_ids.shape[0])
        .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} top-k exceeds usize")))?;
    if slot >= top_k {
        return invalid(format!("{name} slot {slot} exceeds top-k {top_k}"));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, name)?;
    validate_f32(input, &[n_in as u64], false, &format!("{name} input"))?;
    validate_f32(output, &[n_out as u64], true, &format!("{name} output"))?;

    let (kernel, rows_per_group, threads_per_group, block_size) = match bank.dtype {
        GgmlType::IQ2_XS => (
            "kernel_deepseek_v4_indexed_mat_vec_iq2_xs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ2_S => (
            "kernel_deepseek_v4_indexed_mat_vec_iq2_s_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ3_XXS => (
            "kernel_deepseek_v4_indexed_mat_vec_iq3_xxs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::IQ3_S => (
            "kernel_deepseek_v4_indexed_mat_vec_iq3_s_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::Q3_K => ("kernel_deepseek_v4_indexed_mat_vec_q3_K_f32", 8, 64, 256),
        GgmlType::Q4_K => ("kernel_deepseek_v4_indexed_mat_vec_q4_K_f32", 4, 64, 256),
        GgmlType::MXFP4 => ("kernel_deepseek_v4_indexed_mat_vec_mxfp4_f32", 4, 128, 32),
        dtype => {
            return invalid(format!(
                "{name} has unsupported indexed expert dtype {dtype:?}"
            ));
        }
    };
    if !n_in.is_multiple_of(block_size) {
        return invalid(format!(
            "{name} input width {n_in} is not divisible by block size {block_size}"
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        slot: u32,
    }
    let args = Args {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{name} expert count exceeds u32"))
        })?,
        slot: u32::try_from(slot)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} slot exceeds u32")))?,
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads_per_group {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {threads_per_group} threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_group),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_all_slots_gate_up_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate_bank: &MetalTensor,
    up_bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 all-slot gate/up SwiGLU";
    require_serial(enc, NAME)?;
    if !clamp.is_finite() || clamp <= 0.0 {
        return invalid(format!("{NAME} clamp must be finite and positive"));
    }
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_expert_bank(gate_bank, n_in, n_out, expert_count, "all-slot gate bank")?;
    validate_expert_bank(up_bank, n_in, n_out, expert_count, "all-slot up bank")?;
    validate_f32(input, &[n_in as u64], false, &format!("{NAME} input"))?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    if gate_bank.dtype != up_bank.dtype {
        return invalid(format!(
            "{NAME} requires matching gate/up dtypes, got {:?} and {:?}",
            gate_bank.dtype, up_bank.dtype
        ));
    }
    let kernel = match gate_bank.dtype {
        GgmlType::IQ2_XS => "kernel_deepseek_v4_all_slots_swiglu_iq2_xs_f32_fast",
        GgmlType::IQ2_S => "kernel_deepseek_v4_all_slots_swiglu_iq2_s_f32_fast",
        GgmlType::IQ3_XXS => "kernel_deepseek_v4_all_slots_swiglu_iq3_xxs_f32_fast",
        GgmlType::IQ3_S => "kernel_deepseek_v4_all_slots_swiglu_iq3_s_f32_fast",
        GgmlType::Q3_K => "kernel_deepseek_v4_all_slots_swiglu_q3_K_f32",
        dtype => return invalid(format!("{NAME} does not support {dtype:?}")),
    };
    if !n_in.is_multiple_of(256) {
        return invalid(format!("{NAME} input width {n_in} is not divisible by 256"));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        top_k: u32,
        clamp: f32,
    }
    let args = Args {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{NAME} expert count exceeds u32"))
        })?,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K as u32,
        clamp,
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < 64 {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and 64 threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, gate_bank);
    enc.set_tensor(2, up_bank);
    enc.set_tensor(3, input);
    enc.set_tensor(4, expert_ids);
    enc.set_tensor(5, route_status);
    enc.set_tensor(6, output);
    enc.set_threadgroup_memory(0, 16 * std::mem::size_of::<f32>());
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(8),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_all_slots_down(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 all-slot down";
    require_serial(enc, NAME)?;
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_expert_bank(bank, n_in, n_out, expert_count, "all-slot down bank")?;
    validate_f32(
        input,
        &[n_in as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} input"),
    )?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    let (kernel, rows_per_group, threads_per_group, block_size) = match bank.dtype {
        GgmlType::IQ3_XXS => (
            "kernel_deepseek_v4_all_slots_down_iq3_xxs_f32_fast",
            8,
            64,
            256,
        ),
        GgmlType::MXFP4 => ("kernel_deepseek_v4_all_slots_down_mxfp4_f32", 4, 128, 32),
        GgmlType::Q4_K => ("kernel_deepseek_v4_all_slots_down_q4_K_f32", 4, 64, 256),
        dtype => return invalid(format!("{NAME} does not support {dtype:?}")),
    };
    if !n_in.is_multiple_of(block_size) {
        return invalid(format!(
            "{NAME} input width {n_in} is not divisible by {block_size}"
        ));
    }
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n_in: u32,
        n_out: u32,
        n_expert: u32,
        top_k: u32,
        clamp: f32,
    }
    let args = Args {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{NAME} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{NAME} expert count exceeds u32"))
        })?,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K as u32,
        clamp: 0.0,
    };
    let pso = ctx.pipeline(kernel)?;
    if pso.threadExecutionWidth() != 32 || pso.maxTotalThreadsPerThreadgroup() < threads_per_group {
        return invalid(format!(
            "{kernel} requires SIMD width 32 and {threads_per_group} threads, got width {} max {}",
            pso.threadExecutionWidth(),
            pso.maxTotalThreadsPerThreadgroup()
        ));
    }
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(rows_per_group),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: threads_per_group,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_all_slots_q3_k_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 fast all-slot Q3_K projection";
    require_serial(enc, NAME)?;
    if bank.dtype != GgmlType::Q3_K {
        return invalid(format!("{NAME} requires Q3_K, got {:?}", bank.dtype));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, NAME)?;
    validate_f32(input, &[n_in as u64], false, &format!("{NAME} input"))?;
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    if !n_in.is_multiple_of(256) {
        return invalid(format!("{NAME} input width {n_in} is not divisible by 256"));
    }
    let args = deepseek_v4_all_slots_args(n_in, n_out, expert_count, 0.0, NAME)?;
    let kernel = "kernel_deepseek_v4_all_slots_mat_vec_q3_K_f32_fast";
    let pso = ctx.pipeline(kernel)?;
    validate_deepseek_v4_all_slot_pipeline(ctx, kernel, 64, 0)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn encode_ds4_all_slots_q4_k_fast(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    bank: &MetalTensor,
    input: &MetalTensor,
    expert_ids: &MetalTensor,
    route_status: &MetalTensor,
    output: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
) -> Result<(), DeepSeekV4MetalError> {
    const NAME: &str = "DeepSeek V4 fast all-slot Q4_K projection";
    require_serial(enc, NAME)?;
    if bank.dtype != GgmlType::Q4_K {
        return invalid(format!("{NAME} requires Q4_K, got {:?}", bank.dtype));
    }
    validate_expert_bank(bank, n_in, n_out, expert_count, NAME)?;
    validate_f32(
        input,
        &[n_in as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} input"),
    )?;
    validate_i32(
        expert_ids,
        &[DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        false,
        &format!("{NAME} expert IDs"),
    )?;
    validate_i32(route_status, &[1], false, &format!("{NAME} route status"))?;
    validate_f32(
        output,
        &[n_out as u64, DEEPSEEK_V4_ROUTE_MAX_TOP_K as u64],
        true,
        &format!("{NAME} output"),
    )?;
    if !n_in.is_multiple_of(256) {
        return invalid(format!("{NAME} input width {n_in} is not divisible by 256"));
    }
    let args = deepseek_v4_all_slots_args(n_in, n_out, expert_count, 0.0, NAME)?;
    let kernel = "kernel_deepseek_v4_all_slots_mat_vec_q4_K_f32_fast";
    let pso = ctx.pipeline(kernel)?;
    validate_deepseek_v4_all_slot_pipeline(ctx, kernel, 64, 0)?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &args);
    enc.set_tensor(1, bank);
    enc.set_tensor(2, input);
    enc.set_tensor(3, expert_ids);
    enc.set_tensor(4, route_status);
    enc.set_tensor(5, output);
    enc.dispatch(
        MTLSize {
            width: n_out.div_ceil(4),
            height: DEEPSEEK_V4_ROUTE_MAX_TOP_K,
            depth: 1,
        },
        MTLSize {
            width: 64,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) fn deepseek_v4_all_slots_args(
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    clamp: f32,
    name: &str,
) -> Result<DeepSeekV4AllSlotsArgs, DeepSeekV4MetalError> {
    Ok(DeepSeekV4AllSlotsArgs {
        n_in: u32::try_from(n_in)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_in exceeds u32")))?,
        n_out: u32::try_from(n_out)
            .map_err(|_| DeepSeekV4MetalError::Invalid(format!("{name} n_out exceeds u32")))?,
        n_expert: u32::try_from(expert_count).map_err(|_| {
            DeepSeekV4MetalError::Invalid(format!("{name} expert count exceeds u32"))
        })?,
        top_k: DEEPSEEK_V4_ROUTE_MAX_TOP_K as u32,
        clamp,
    })
}

pub(super) fn encode_ds4_clamped_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    gate: &MetalTensor,
    up: &MetalTensor,
    output: &MetalTensor,
    clamp: f32,
) -> Result<(), DeepSeekV4MetalError> {
    let n = gate.n_elements() as usize;
    validate_f32(gate, &[n as u64], false, "DS4 SwiGLU gate")?;
    validate_f32(up, &[n as u64], false, "DS4 SwiGLU up")?;
    validate_f32(output, &[n as u64], true, "DS4 SwiGLU output")?;
    let n = u32::try_from(n)
        .map_err(|_| DeepSeekV4MetalError::Invalid("DS4 SwiGLU width exceeds u32".into()))?;
    #[repr(C)]
    #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
    struct Args {
        n: u32,
        clamp: f32,
    }
    let pso = ctx.pipeline("kernel_deepseek_v4_clamped_swiglu")?;
    enc.set_pipeline(&pso);
    enc.set_bytes(0, &Args { n, clamp });
    enc.set_tensor(1, gate);
    enc.set_tensor(2, up);
    enc.set_tensor(3, output);
    enc.dispatch(
        MTLSize {
            width: (n as usize).div_ceil(256),
            height: 1,
            depth: 1,
        },
        MTLSize {
            width: 256,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}

pub(super) fn validate_expert_bank(
    tensor: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert_count: usize,
    name: &str,
) -> Result<(), DeepSeekV4MetalError> {
    let shape = [n_in as u64, n_out as u64, expert_count as u64];
    if tensor.shape != shape {
        return invalid(format!(
            "{name} must have exact GGUF shape {shape:?}, got {:?}",
            tensor.shape
        ));
    }
    let (_, expert_bytes) = matvec_weight_bytes(tensor.dtype, n_in, n_out, name)?;
    let bank_bytes = checked_mul(expert_bytes, expert_count, &format!("{name} bank bytes"))?;
    if tensor.n_bytes() != bank_bytes as u64 {
        return invalid(format!(
            "{name} byte layout is not {expert_count} contiguous {expert_bytes}-byte slices"
        ));
    }
    let end = tensor
        .offset
        .checked_add(bank_bytes as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} range overflow")))?;
    if end > tensor.buffer.length() as u64 {
        return invalid(format!(
            "{name} range [{}, {end}) exceeds buffer length {}",
            tensor.offset,
            tensor.buffer.length()
        ));
    }
    let alignment = weight_offset_alignment(tensor.dtype);
    for expert in 0..expert_count {
        let offset = tensor.offset + (expert * expert_bytes) as u64;
        if !offset.is_multiple_of(alignment) {
            return invalid(format!(
                "{name} expert {expert} offset {offset} is not aligned to {alignment} bytes"
            ));
        }
    }
    Ok(())
}

pub(super) fn expert_weight_view(
    bank: &MetalTensor,
    n_in: usize,
    n_out: usize,
    expert: usize,
    name: &str,
) -> Result<MetalTensor, DeepSeekV4MetalError> {
    let expert_count = bank
        .shape
        .get(2)
        .copied()
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} has no expert axis")))?;
    validate_expert_bank(bank, n_in, n_out, expert_count, name)?;
    if expert >= expert_count {
        return invalid(format!("{name} expert {expert} exceeds {expert_count}"));
    }
    let (_, expert_bytes) = matvec_weight_bytes(bank.dtype, n_in, n_out, name)?;
    let byte_offset = checked_mul(expert, expert_bytes, &format!("{name} byte offset"))?;
    let offset = bank
        .offset
        .checked_add(byte_offset as u64)
        .ok_or_else(|| DeepSeekV4MetalError::Invalid(format!("{name} offset overflow")))?;
    let view = MetalTensor {
        buffer: bank.buffer.clone(),
        offset,
        shape: vec![n_in as u64, n_out as u64],
        dtype: bank.dtype,
        provenance: bank.provenance(),
    };
    validate_matvec_weight(&view, n_in, n_out, name)?;
    Ok(view)
}
