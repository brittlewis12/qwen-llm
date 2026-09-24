//! Gated-DeltaNet mixer capture and VJP readbacks.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct GdnGeometry {
    pub(super) hidden_size: usize,
    pub(super) n_v_heads: usize,
    pub(super) n_k_heads: usize,
    pub(super) head_dim: usize,
    pub(super) qk_elements: usize,
    pub(super) v_elements: usize,
    pub(super) conv_dim: usize,
    pub(super) state_elements: usize,
    pub(super) conv_state_elements: usize,
}

impl GdnGeometry {
    pub(super) fn new(layer: u32, arch: Arch) -> Result<Self, WorkspaceLensError> {
        let hidden_size = arch.hidden_size as usize;
        let n_v_heads = arch.gdn_n_v_heads as usize;
        let n_k_heads = arch.gdn_n_k_heads as usize;
        let head_dim = arch.gdn_head_dim as usize;
        if hidden_size == 0
            || head_dim != 128
            || n_v_heads == 0
            || n_k_heads == 0
            || !n_v_heads.is_multiple_of(n_k_heads)
        {
            return Err(WorkspaceLensError::UnsupportedGdnGeometry {
                layer,
                n_v: n_v_heads,
                n_k: n_k_heads,
                head_dim,
            });
        }
        let qk_elements = n_k_heads
            .checked_mul(head_dim)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let v_elements = n_v_heads
            .checked_mul(head_dim)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let conv_dim = qk_elements
            .checked_mul(2)
            .and_then(|value| value.checked_add(v_elements))
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let state_elements = v_elements
            .checked_mul(head_dim)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let conv_state_elements = conv_dim
            .checked_mul(3)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        Ok(Self {
            hidden_size,
            n_v_heads,
            n_k_heads,
            head_dim,
            qk_elements,
            v_elements,
            conv_dim,
            state_elements,
            conv_state_elements,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct GdnMixerWeights<'a> {
    pub(super) attn_norm: &'a MetalTensor,
    pub(super) in_proj_qkv: &'a MetalTensor,
    pub(super) in_proj_z: &'a MetalTensor,
    pub(super) beta_proj: &'a MetalTensor,
    pub(super) alpha_proj: &'a MetalTensor,
    pub(super) a_log: &'a MetalTensor,
    pub(super) dt_bias: &'a MetalTensor,
    pub(super) conv1d: &'a MetalTensor,
    pub(super) norm: &'a MetalTensor,
    pub(super) out_proj: &'a MetalTensor,
}

impl<'a> From<&'a MetalGdnBlock> for GdnMixerWeights<'a> {
    fn from(block: &'a MetalGdnBlock) -> Self {
        Self {
            attn_norm: &block.attn_norm,
            in_proj_qkv: &block.in_proj_qkv,
            in_proj_z: &block.in_proj_z,
            beta_proj: &block.beta_proj,
            alpha_proj: &block.alpha_proj,
            a_log: &block.a_log,
            dt_bias: &block.dt_bias,
            conv1d: &block.conv1d,
            norm: &block.norm,
            out_proj: &block.out_proj,
        }
    }
}

pub(super) fn validate_gdn_weights(
    layer: u32,
    block: &MetalGdnBlock,
    geometry: GdnGeometry,
) -> Result<(), WorkspaceLensError> {
    let weights = GdnMixerWeights::from(block);
    for (role, weight, expected) in [
        (
            LinearRole::GdnQkv,
            weights.in_proj_qkv,
            [geometry.hidden_size, geometry.conv_dim],
        ),
        (
            LinearRole::GdnZ,
            weights.in_proj_z,
            [geometry.hidden_size, geometry.v_elements],
        ),
        (
            LinearRole::GdnBeta,
            weights.beta_proj,
            [geometry.hidden_size, geometry.n_v_heads],
        ),
        (
            LinearRole::GdnAlpha,
            weights.alpha_proj,
            [geometry.hidden_size, geometry.n_v_heads],
        ),
        (
            LinearRole::GdnOut,
            weights.out_proj,
            [geometry.v_elements, geometry.hidden_size],
        ),
    ] {
        let id = WorkspaceLensLinear::Layer { index: layer, role };
        let got = linear_shape(id, weight)?;
        if got != expected {
            return Err(WorkspaceLensError::InvalidGdnLinearShape {
                layer,
                role,
                got,
                expected,
            });
        }
        validate_vjp_dtype(id, weight)?;
    }
    for (name, tensor, expected_elements) in [
        ("pre-mixer norm", weights.attn_norm, geometry.hidden_size),
        ("A log", weights.a_log, geometry.n_v_heads),
        ("timestep bias", weights.dt_bias, geometry.n_v_heads),
        (
            "conv1d",
            weights.conv1d,
            geometry
                .conv_dim
                .checked_mul(4)
                .ok_or(WorkspaceLensError::SizeOverflow)?,
        ),
        ("internal norm", weights.norm, geometry.head_dim),
    ] {
        if tensor.dtype != GgmlType::F32 || tensor.n_elements() as usize != expected_elements {
            return Err(WorkspaceLensError::InvalidGdnTensor {
                layer,
                name,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                expected_elements,
            });
        }
    }
    Ok(())
}

pub(super) struct GdnReplayTensors {
    pub(super) input: MetalTensor,
    pub(super) normalized: MetalTensor,
    pub(super) qkv_source: MetalTensor,
    pub(super) z: MetalTensor,
    pub(super) beta_source: MetalTensor,
    pub(super) beta: MetalTensor,
    pub(super) alpha_source: MetalTensor,
    pub(super) decay: MetalTensor,
    pub(super) initial_conv_state: MetalTensor,
    pub(super) conv_state: MetalTensor,
    pub(super) conv_checkpoints: MetalTensor,
    pub(super) q_raw: MetalTensor,
    pub(super) k_raw: MetalTensor,
    pub(super) v: MetalTensor,
    pub(super) q: MetalTensor,
    pub(super) k: MetalTensor,
    pub(super) initial_recurrence_state: MetalTensor,
    pub(super) recurrence_state: MetalTensor,
    pub(super) recurrence_checkpoints: MetalTensor,
    pub(super) recurrence_output: MetalTensor,
    pub(super) normed: MetalTensor,
    pub(super) mixer_output: MetalTensor,
}

impl GdnReplayTensors {
    pub(super) fn new(
        context: &MetalContext,
        geometry: GdnGeometry,
        input: &[f32],
        initial_conv_state: &[f32],
        initial_recurrence_state: &[f32],
        n_tokens: usize,
    ) -> Result<Self, WorkspaceLensError> {
        let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
        let qkv_elements = checked_product(n_tokens, geometry.conv_dim)?;
        let qk_elements = checked_product(n_tokens, geometry.qk_elements)?;
        let v_elements = checked_product(n_tokens, geometry.v_elements)?;
        let scalar_elements = checked_product(n_tokens, geometry.n_v_heads)?;
        let conv_checkpoint_elements = checked_product(n_tokens, geometry.conv_state_elements)?;
        let recurrence_checkpoint_elements = checked_product(n_tokens, geometry.state_elements)?;
        for (name, values, expected) in [
            ("GDN replay input", input, hidden_elements),
            (
                "GDN initial conv state",
                initial_conv_state,
                geometry.conv_state_elements,
            ),
            (
                "GDN initial recurrence state",
                initial_recurrence_state,
                geometry.state_elements,
            ),
        ] {
            if values.len() != expected {
                return Err(WorkspaceLensError::ActivationSize {
                    name,
                    got: values.len(),
                    expected,
                });
            }
        }
        let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
        Ok(Self {
            input: MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(input),
                hidden_shape.clone(),
                GgmlType::F32,
            )?,
            normalized: MetalTensor::zeros_f32(context, hidden_shape.clone())?,
            qkv_source: flat_f32(context, qkv_elements)?,
            z: flat_f32(context, v_elements)?,
            beta_source: flat_f32(context, scalar_elements)?,
            beta: flat_f32(context, scalar_elements)?,
            alpha_source: flat_f32(context, scalar_elements)?,
            decay: flat_f32(context, scalar_elements)?,
            initial_conv_state: f32_from_slice(context, initial_conv_state)?,
            conv_state: f32_from_slice(context, initial_conv_state)?,
            conv_checkpoints: flat_f32(context, conv_checkpoint_elements)?,
            q_raw: flat_f32(context, qk_elements)?,
            k_raw: flat_f32(context, qk_elements)?,
            v: flat_f32(context, v_elements)?,
            q: flat_f32(context, qk_elements)?,
            k: flat_f32(context, qk_elements)?,
            initial_recurrence_state: f32_from_slice(context, initial_recurrence_state)?,
            recurrence_state: f32_from_slice(context, initial_recurrence_state)?,
            recurrence_checkpoints: flat_f32(context, recurrence_checkpoint_elements)?,
            recurrence_output: flat_f32(context, v_elements)?,
            normed: flat_f32(context, v_elements)?,
            mixer_output: flat_f32(context, hidden_elements)?,
        })
    }

    pub(super) fn encode_forward(
        &self,
        context: &MetalContext,
        encoder: &KernelEncoder,
        geometry: GdnGeometry,
        weights: GdnMixerWeights<'_>,
        n_tokens: usize,
    ) -> Result<(), WorkspaceLensError> {
        encode_rms_norm_mul_rows_f32(
            context,
            encoder,
            &self.input,
            weights.attn_norm,
            &self.normalized,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
        )?;
        for token in 0..n_tokens {
            let hidden = row_view(&self.normalized, token, geometry.hidden_size);
            let qkv = row_view(&self.qkv_source, token, geometry.conv_dim);
            let z = row_view(&self.z, token, geometry.v_elements);
            let beta = row_view(&self.beta_source, token, geometry.n_v_heads);
            let alpha = row_view(&self.alpha_source, token, geometry.n_v_heads);
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.in_proj_qkv,
                &hidden,
                &qkv,
                geometry.hidden_size,
                geometry.conv_dim,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.in_proj_z,
                &hidden,
                &z,
                geometry.hidden_size,
                geometry.v_elements,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.beta_proj,
                &hidden,
                &beta,
                geometry.hidden_size,
                geometry.n_v_heads,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.alpha_proj,
                &hidden,
                &alpha,
                geometry.hidden_size,
                geometry.n_v_heads,
            )?;
        }
        encode_sigmoid_f32(context, encoder, &self.beta_source, &self.beta)?;
        encode_gdn_decay_chain_batched_f32(
            context,
            encoder,
            &self.alpha_source,
            weights.dt_bias,
            weights.a_log,
            &self.decay,
            n_tokens,
            geometry.n_v_heads,
        )?;
        encode_gdn_prep_packed_ckpt_f32(
            context,
            encoder,
            &self.qkv_source,
            &self.conv_state,
            weights.conv1d,
            &self.q_raw,
            &self.k_raw,
            &self.v,
            &self.conv_checkpoints,
            n_tokens,
            n_tokens,
            geometry.n_k_heads,
            geometry.n_v_heads,
            geometry.head_dim,
        )?;
        encode_l2_norm_batched_f32(
            context,
            encoder,
            &self.q_raw,
            &self.q,
            checked_product(n_tokens, geometry.n_k_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_batched_f32(
            context,
            encoder,
            &self.k_raw,
            &self.k,
            checked_product(n_tokens, geometry.n_k_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_gdn_step_decay_packed_ckpt_f32(
            context,
            encoder,
            &self.q,
            &self.k,
            &self.v,
            &self.decay,
            &self.beta,
            &self.recurrence_state,
            &self.recurrence_output,
            &self.recurrence_checkpoints,
            n_tokens,
            n_tokens,
            geometry.n_v_heads,
            geometry.n_k_heads,
            geometry.head_dim,
        )?;
        encode_rmsnorm_gated_f32(
            context,
            encoder,
            &self.recurrence_output,
            weights.norm,
            &self.z,
            &self.normed,
            checked_product(n_tokens, geometry.n_v_heads)?,
            geometry.head_dim,
            RMS_EPS * geometry.head_dim as f32,
        )?;
        for token in 0..n_tokens {
            let normed = row_view(&self.normed, token, geometry.v_elements);
            let mixer = row_view(&self.mixer_output, token, geometry.hidden_size);
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.out_proj,
                &normed,
                &mixer,
                geometry.v_elements,
                geometry.hidden_size,
            )?;
        }
        Ok(())
    }
}

pub(super) struct GdnReplayVjpReadback {
    pub(super) mixer_outputs: Vec<f32>,
    pub(super) final_conv_state: Vec<f32>,
    pub(super) final_recurrence_state: Vec<f32>,
    pub(super) grad_input: Vec<f32>,
    pub(super) grad_initial_conv_state: Vec<f32>,
    pub(super) grad_initial_recurrence_state: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_mixer_replay_vjp_readback(
    context: &MetalContext,
    geometry: GdnGeometry,
    weights: GdnMixerWeights<'_>,
    input: &[f32],
    initial_conv_state: &[f32],
    initial_recurrence_state: &[f32],
    grad_mixer_output: &[f32],
    n_tokens: usize,
    rule: GdnMixerVjpRule,
    read_state_diagnostics: bool,
) -> Result<GdnReplayVjpReadback, WorkspaceLensError> {
    if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_GDN_TOKENS {
        return Err(WorkspaceLensError::GdnPromptTooLong {
            got: n_tokens,
            max: MAX_WORKSPACE_LENS_GDN_TOKENS,
        });
    }
    let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
    let qkv_elements = checked_product(n_tokens, geometry.conv_dim)?;
    let qk_elements = checked_product(n_tokens, geometry.qk_elements)?;
    let v_elements = checked_product(n_tokens, geometry.v_elements)?;
    let scalar_elements = checked_product(n_tokens, geometry.n_v_heads)?;
    if grad_mixer_output.len() != hidden_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "GDN mixer cotangent",
            got: grad_mixer_output.len(),
            expected: hidden_elements,
        });
    }
    let replay = GdnReplayTensors::new(
        context,
        geometry,
        input,
        initial_conv_state,
        initial_recurrence_state,
        n_tokens,
    )?;
    let grad_mixer = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_mixer_output),
        row_shape(geometry.hidden_size, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_normed = MetalTensor::zeros_f32(context, row_shape(geometry.v_elements, n_tokens)?)?;
    let grad_recurrence_output = flat_f32(context, v_elements)?;
    let grad_z = flat_f32(context, v_elements)?;
    let grad_q = flat_f32(context, qk_elements)?;
    let grad_k = flat_f32(context, qk_elements)?;
    let grad_v = flat_f32(context, v_elements)?;
    let grad_decay = flat_f32(context, scalar_elements)?;
    let grad_beta = flat_f32(context, scalar_elements)?;
    let zero_final_recurrence_state = flat_f32(context, geometry.state_elements)?;
    let grad_initial_recurrence_state = flat_f32(context, geometry.state_elements)?;
    let recurrence_state_scratch_a = flat_f32(context, geometry.state_elements)?;
    let recurrence_state_scratch_b = flat_f32(context, geometry.state_elements)?;
    let correction_scratch = flat_f32(context, geometry.v_elements)?;
    let residual_scratch = flat_f32(context, geometry.v_elements)?;
    let grad_q_raw = flat_f32(context, qk_elements)?;
    let grad_k_raw = flat_f32(context, qk_elements)?;
    let grad_qkv = flat_f32(context, qkv_elements)?;
    let zero_final_conv_state = flat_f32(context, geometry.conv_state_elements)?;
    let grad_initial_conv_state = flat_f32(context, geometry.conv_state_elements)?;
    let conv_state_scratch_a = flat_f32(context, geometry.conv_state_elements)?;
    let conv_state_scratch_b = flat_f32(context, geometry.conv_state_elements)?;
    let grad_alpha_source = flat_f32(context, scalar_elements)?;
    let grad_beta_source = flat_f32(context, scalar_elements)?;
    let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
    let grad_hidden_qkv = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_z = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_alpha = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_beta = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_sum_a = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_sum_b = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        replay.encode_forward(context, &encoder, geometry, weights, n_tokens)?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            weights.out_proj,
            &grad_mixer,
            &grad_normed,
            geometry.v_elements,
            geometry.hidden_size,
            n_tokens,
        )?;
        let grad_normed_flat = grad_normed.view_subrange(0, vec![v_elements as u64]);
        encode_rmsnorm_gated_vjp_f32(
            context,
            &encoder,
            &replay.recurrence_output,
            weights.norm,
            &replay.z,
            &grad_normed_flat,
            &grad_recurrence_output,
            &grad_z,
            checked_product(n_tokens, geometry.n_v_heads)?,
            geometry.head_dim,
            RMS_EPS * geometry.head_dim as f32,
        )?;
        encode_fill_f32(context, &encoder, &zero_final_recurrence_state, 0.0)?;
        encode_gdn_step_decay_packed_vjp_f32(
            context,
            &encoder,
            &replay.q,
            &replay.k,
            &replay.v,
            &replay.decay,
            &replay.beta,
            &replay.initial_recurrence_state,
            &replay.recurrence_checkpoints,
            n_tokens,
            &grad_recurrence_output,
            &zero_final_recurrence_state,
            &grad_q,
            &grad_k,
            &grad_v,
            &grad_decay,
            &grad_beta,
            &grad_initial_recurrence_state,
            &recurrence_state_scratch_a,
            &recurrence_state_scratch_b,
            &correction_scratch,
            &residual_scratch,
            n_tokens,
            geometry.n_v_heads,
            geometry.n_k_heads,
            geometry.head_dim,
        )?;
        let packed_k_heads = checked_product(n_tokens, geometry.n_k_heads)?;
        encode_l2_norm_vjp_batched_f32(
            context,
            &encoder,
            &replay.q_raw,
            &grad_q,
            &grad_q_raw,
            packed_k_heads,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_l2_norm_vjp_batched_f32(
            context,
            &encoder,
            &replay.k_raw,
            &grad_k,
            &grad_k_raw,
            packed_k_heads,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_fill_f32(context, &encoder, &zero_final_conv_state, 0.0)?;
        let conv_weight = weights
            .conv1d
            .view_subrange(0, vec![(geometry.conv_dim * 4) as u64]);
        encode_ssm_conv_silu_split_packed_vjp_f32(
            context,
            &encoder,
            &replay.qkv_source,
            &replay.initial_conv_state,
            &replay.conv_checkpoints,
            n_tokens,
            &conv_weight,
            &grad_q_raw,
            &grad_k_raw,
            &grad_v,
            &zero_final_conv_state,
            &grad_qkv,
            &grad_initial_conv_state,
            &conv_state_scratch_a,
            &conv_state_scratch_b,
            n_tokens,
            geometry.n_k_heads,
            geometry.n_v_heads,
            geometry.head_dim,
        )?;
        for token in 0..n_tokens {
            let alpha_source = row_view(&replay.alpha_source, token, geometry.n_v_heads);
            let decay = row_view(&replay.decay, token, geometry.n_v_heads);
            let grad_decay_row = row_view(&grad_decay, token, geometry.n_v_heads);
            let grad_alpha_row = row_view(&grad_alpha_source, token, geometry.n_v_heads);
            encode_gdn_decay_chain_vjp_f32(
                context,
                &encoder,
                &alpha_source,
                weights.dt_bias,
                weights.a_log,
                &decay,
                &grad_decay_row,
                &grad_alpha_row,
            )?;
        }
        encode_sigmoid_output_vjp_f32(
            context,
            &encoder,
            &replay.beta,
            &grad_beta,
            &grad_beta_source,
        )?;
        let grad_qkv_rows = grad_qkv.view_subrange(0, row_shape(geometry.conv_dim, n_tokens)?);
        let grad_z_rows = grad_z.view_subrange(0, row_shape(geometry.v_elements, n_tokens)?);
        let grad_alpha_rows =
            grad_alpha_source.view_subrange(0, row_shape(geometry.n_v_heads, n_tokens)?);
        let grad_beta_rows =
            grad_beta_source.view_subrange(0, row_shape(geometry.n_v_heads, n_tokens)?);
        for (weight, grad_output, grad_hidden_output, n_out) in [
            (
                weights.in_proj_qkv,
                &grad_qkv_rows,
                &grad_hidden_qkv,
                geometry.conv_dim,
            ),
            (
                weights.in_proj_z,
                &grad_z_rows,
                &grad_hidden_z,
                geometry.v_elements,
            ),
            (
                weights.alpha_proj,
                &grad_alpha_rows,
                &grad_hidden_alpha,
                geometry.n_v_heads,
            ),
            (
                weights.beta_proj,
                &grad_beta_rows,
                &grad_hidden_beta,
                geometry.n_v_heads,
            ),
        ] {
            encode_frozen_linear_vjp_f32(
                context,
                &encoder,
                weight,
                grad_output,
                grad_hidden_output,
                geometry.hidden_size,
                n_out,
                n_tokens,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_qkv,
            &grad_hidden_z,
            &grad_hidden_sum_a,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_alpha,
            &grad_hidden_beta,
            &grad_hidden_sum_b,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_sum_a,
            &grad_hidden_sum_b,
            &grad_hidden,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            context,
            &encoder,
            &replay.input,
            weights.attn_norm,
            &grad_hidden,
            &grad_input,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
            match rule {
                GdnMixerVjpRule::Jacobian => RmsNormVjpRule::Jacobian,
                GdnMixerVjpRule::Relp => RmsNormVjpRule::RelpDetachedScale,
            },
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;
    let (
        final_conv_state,
        final_recurrence_state,
        grad_initial_conv_state,
        grad_initial_recurrence_state,
    ) = if read_state_diagnostics {
        (
            read_f32(&replay.conv_state, geometry.conv_state_elements),
            read_f32(&replay.recurrence_state, geometry.state_elements),
            read_f32(&grad_initial_conv_state, geometry.conv_state_elements),
            read_f32(&grad_initial_recurrence_state, geometry.state_elements),
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
    Ok(GdnReplayVjpReadback {
        mixer_outputs: read_f32(&replay.mixer_output, hidden_elements),
        final_conv_state,
        final_recurrence_state,
        grad_input: read_f32(&grad_input, hidden_elements),
        grad_initial_conv_state,
        grad_initial_recurrence_state,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_mixer_replay_vjp_batch_readback(
    context: &MetalContext,
    geometry: GdnGeometry,
    weights: GdnMixerWeights<'_>,
    input: &[f32],
    initial_conv_state: &[f32],
    initial_recurrence_state: &[f32],
    grad_mixer_outputs: &[f32],
    n_tokens: usize,
    n_query: usize,
    rule: GdnMixerVjpRule,
    read_state_diagnostics: bool,
) -> Result<GdnReplayVjpReadback, WorkspaceLensError> {
    if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_GDN_TOKENS {
        return Err(WorkspaceLensError::GdnPromptTooLong {
            got: n_tokens,
            max: MAX_WORKSPACE_LENS_GDN_TOKENS,
        });
    }
    if n_query == 0 {
        return Err(WorkspaceLensError::EmptyQueryBatch);
    }
    if n_query > MAX_WORKSPACE_LENS_DIM_BATCH {
        return Err(WorkspaceLensError::WorkspaceQueryBatchTooLarge {
            got: n_query,
            max: MAX_WORKSPACE_LENS_DIM_BATCH,
        });
    }
    let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
    let qkv_elements = checked_product(n_tokens, geometry.conv_dim)?;
    let qk_elements = checked_product(n_tokens, geometry.qk_elements)?;
    let v_elements = checked_product(n_tokens, geometry.v_elements)?;
    let scalar_elements = checked_product(n_tokens, geometry.n_v_heads)?;
    let query_rows = checked_product(n_query, n_tokens)?;
    let hidden_query_elements = checked_product(n_query, hidden_elements)?;
    let qkv_query_elements = checked_product(n_query, qkv_elements)?;
    let qk_query_elements = checked_product(n_query, qk_elements)?;
    let v_query_elements = checked_product(n_query, v_elements)?;
    let scalar_query_elements = checked_product(n_query, scalar_elements)?;
    let state_query_elements = checked_product(n_query, geometry.state_elements)?;
    let conv_state_query_elements = checked_product(n_query, geometry.conv_state_elements)?;
    if grad_mixer_outputs.len() != hidden_query_elements {
        return Err(WorkspaceLensError::ActivationSize {
            name: "GDN mixer cotangent query bank",
            got: grad_mixer_outputs.len(),
            expected: hidden_query_elements,
        });
    }
    let replay = GdnReplayTensors::new(
        context,
        geometry,
        input,
        initial_conv_state,
        initial_recurrence_state,
        n_tokens,
    )?;
    let grad_mixer = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_mixer_outputs),
        row_shape(geometry.hidden_size, query_rows)?,
        GgmlType::F32,
    )?;
    let grad_normed = MetalTensor::zeros_f32(context, row_shape(geometry.v_elements, query_rows)?)?;
    let grad_recurrence_output = flat_f32(context, v_query_elements)?;
    let grad_z = flat_f32(context, v_query_elements)?;
    let grad_q = flat_f32(context, qk_query_elements)?;
    let grad_k = flat_f32(context, qk_query_elements)?;
    let grad_v = flat_f32(context, v_query_elements)?;
    let grad_decay = flat_f32(context, scalar_query_elements)?;
    let grad_beta = flat_f32(context, scalar_query_elements)?;
    let zero_final_recurrence_state = flat_f32(context, geometry.state_elements)?;
    let grad_initial_recurrence_state = flat_f32(context, state_query_elements)?;
    let recurrence_state_scratch_a = flat_f32(context, state_query_elements)?;
    let recurrence_state_scratch_b = flat_f32(context, state_query_elements)?;
    let correction_scratch = flat_f32(context, v_query_elements)?;
    let residual_scratch = flat_f32(context, v_query_elements)?;
    let grad_q_raw = flat_f32(context, qk_query_elements)?;
    let grad_k_raw = flat_f32(context, qk_query_elements)?;
    let grad_qkv = flat_f32(context, qkv_query_elements)?;
    let zero_final_conv_state = flat_f32(context, geometry.conv_state_elements)?;
    let grad_initial_conv_state = flat_f32(context, conv_state_query_elements)?;
    let conv_state_scratch_a = flat_f32(context, conv_state_query_elements)?;
    let conv_state_scratch_b = flat_f32(context, conv_state_query_elements)?;
    let grad_alpha_source = flat_f32(context, scalar_query_elements)?;
    let grad_beta_source = flat_f32(context, scalar_query_elements)?;
    let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
    let hidden_query_shape = row_shape(geometry.hidden_size, query_rows)?;
    let grad_hidden_qkv = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_z = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_alpha = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_beta = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_sum_a = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_sum_b = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_query_shape)?;

    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        replay.encode_forward(context, &encoder, geometry, weights, n_tokens)?;
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            weights.out_proj,
            &grad_mixer,
            &grad_normed,
            geometry.v_elements,
            geometry.hidden_size,
            query_rows,
        )?;
        encode_fill_f32(context, &encoder, &zero_final_recurrence_state, 0.0)?;
        encode_fill_f32(context, &encoder, &zero_final_conv_state, 0.0)?;
        let conv_weight = weights
            .conv1d
            .view_subrange(0, vec![(geometry.conv_dim * 4) as u64]);
        let packed_k_heads = checked_product(n_tokens, geometry.n_k_heads)?;
        for query in 0..n_query {
            let grad_normed_query = flat_query_view(&grad_normed, query, v_elements)?;
            let grad_recurrence_output_query =
                flat_query_view(&grad_recurrence_output, query, v_elements)?;
            let grad_z_query = flat_query_view(&grad_z, query, v_elements)?;
            let grad_q_query = flat_query_view(&grad_q, query, qk_elements)?;
            let grad_k_query = flat_query_view(&grad_k, query, qk_elements)?;
            let grad_v_query = flat_query_view(&grad_v, query, v_elements)?;
            let grad_decay_query = flat_query_view(&grad_decay, query, scalar_elements)?;
            let grad_beta_query = flat_query_view(&grad_beta, query, scalar_elements)?;
            let grad_initial_recurrence_state_query = flat_query_view(
                &grad_initial_recurrence_state,
                query,
                geometry.state_elements,
            )?;
            let recurrence_state_scratch_a_query =
                flat_query_view(&recurrence_state_scratch_a, query, geometry.state_elements)?;
            let recurrence_state_scratch_b_query =
                flat_query_view(&recurrence_state_scratch_b, query, geometry.state_elements)?;
            let correction_scratch_query =
                flat_query_view(&correction_scratch, query, geometry.v_elements)?;
            let residual_scratch_query =
                flat_query_view(&residual_scratch, query, geometry.v_elements)?;
            let grad_q_raw_query = flat_query_view(&grad_q_raw, query, qk_elements)?;
            let grad_k_raw_query = flat_query_view(&grad_k_raw, query, qk_elements)?;
            let grad_qkv_query = flat_query_view(&grad_qkv, query, qkv_elements)?;
            let grad_initial_conv_state_query = flat_query_view(
                &grad_initial_conv_state,
                query,
                geometry.conv_state_elements,
            )?;
            let conv_state_scratch_a_query =
                flat_query_view(&conv_state_scratch_a, query, geometry.conv_state_elements)?;
            let conv_state_scratch_b_query =
                flat_query_view(&conv_state_scratch_b, query, geometry.conv_state_elements)?;
            let grad_alpha_source_query =
                flat_query_view(&grad_alpha_source, query, scalar_elements)?;
            let grad_beta_source_query =
                flat_query_view(&grad_beta_source, query, scalar_elements)?;

            encode_rmsnorm_gated_vjp_f32(
                context,
                &encoder,
                &replay.recurrence_output,
                weights.norm,
                &replay.z,
                &grad_normed_query,
                &grad_recurrence_output_query,
                &grad_z_query,
                checked_product(n_tokens, geometry.n_v_heads)?,
                geometry.head_dim,
                RMS_EPS * geometry.head_dim as f32,
            )?;
            encode_gdn_step_decay_packed_vjp_f32(
                context,
                &encoder,
                &replay.q,
                &replay.k,
                &replay.v,
                &replay.decay,
                &replay.beta,
                &replay.initial_recurrence_state,
                &replay.recurrence_checkpoints,
                n_tokens,
                &grad_recurrence_output_query,
                &zero_final_recurrence_state,
                &grad_q_query,
                &grad_k_query,
                &grad_v_query,
                &grad_decay_query,
                &grad_beta_query,
                &grad_initial_recurrence_state_query,
                &recurrence_state_scratch_a_query,
                &recurrence_state_scratch_b_query,
                &correction_scratch_query,
                &residual_scratch_query,
                n_tokens,
                geometry.n_v_heads,
                geometry.n_k_heads,
                geometry.head_dim,
            )?;
            encode_l2_norm_vjp_batched_f32(
                context,
                &encoder,
                &replay.q_raw,
                &grad_q_query,
                &grad_q_raw_query,
                packed_k_heads,
                geometry.head_dim,
                RMS_EPS,
            )?;
            encode_l2_norm_vjp_batched_f32(
                context,
                &encoder,
                &replay.k_raw,
                &grad_k_query,
                &grad_k_raw_query,
                packed_k_heads,
                geometry.head_dim,
                RMS_EPS,
            )?;
            encode_ssm_conv_silu_split_packed_vjp_f32(
                context,
                &encoder,
                &replay.qkv_source,
                &replay.initial_conv_state,
                &replay.conv_checkpoints,
                n_tokens,
                &conv_weight,
                &grad_q_raw_query,
                &grad_k_raw_query,
                &grad_v_query,
                &zero_final_conv_state,
                &grad_qkv_query,
                &grad_initial_conv_state_query,
                &conv_state_scratch_a_query,
                &conv_state_scratch_b_query,
                n_tokens,
                geometry.n_k_heads,
                geometry.n_v_heads,
                geometry.head_dim,
            )?;
            for token in 0..n_tokens {
                let alpha_source = row_view(&replay.alpha_source, token, geometry.n_v_heads);
                let decay = row_view(&replay.decay, token, geometry.n_v_heads);
                let grad_decay_row = row_view(&grad_decay_query, token, geometry.n_v_heads);
                let grad_alpha_row = row_view(&grad_alpha_source_query, token, geometry.n_v_heads);
                encode_gdn_decay_chain_vjp_f32(
                    context,
                    &encoder,
                    &alpha_source,
                    weights.dt_bias,
                    weights.a_log,
                    &decay,
                    &grad_decay_row,
                    &grad_alpha_row,
                )?;
            }
            encode_sigmoid_output_vjp_f32(
                context,
                &encoder,
                &replay.beta,
                &grad_beta_query,
                &grad_beta_source_query,
            )?;
        }
        let grad_qkv_rows = grad_qkv.view_subrange(0, row_shape(geometry.conv_dim, query_rows)?);
        let grad_z_rows = grad_z.view_subrange(0, row_shape(geometry.v_elements, query_rows)?);
        let grad_alpha_rows =
            grad_alpha_source.view_subrange(0, row_shape(geometry.n_v_heads, query_rows)?);
        let grad_beta_rows =
            grad_beta_source.view_subrange(0, row_shape(geometry.n_v_heads, query_rows)?);
        for (weight, grad_output, grad_hidden_output, n_out) in [
            (
                weights.in_proj_qkv,
                &grad_qkv_rows,
                &grad_hidden_qkv,
                geometry.conv_dim,
            ),
            (
                weights.in_proj_z,
                &grad_z_rows,
                &grad_hidden_z,
                geometry.v_elements,
            ),
            (
                weights.alpha_proj,
                &grad_alpha_rows,
                &grad_hidden_alpha,
                geometry.n_v_heads,
            ),
            (
                weights.beta_proj,
                &grad_beta_rows,
                &grad_hidden_beta,
                geometry.n_v_heads,
            ),
        ] {
            encode_frozen_linear_vjp_f32(
                context,
                &encoder,
                weight,
                grad_output,
                grad_hidden_output,
                geometry.hidden_size,
                n_out,
                query_rows,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_qkv,
            &grad_hidden_z,
            &grad_hidden_sum_a,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_alpha,
            &grad_hidden_beta,
            &grad_hidden_sum_b,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_sum_a,
            &grad_hidden_sum_b,
            &grad_hidden,
        )?;
        for query in 0..n_query {
            let offset = u64::try_from(checked_product(query, hidden_elements)?)
                .map_err(|_| WorkspaceLensError::SizeOverflow)?;
            let grad_hidden_query = grad_hidden.view_subrange(offset, hidden_shape.clone());
            let grad_input_query = grad_input.view_subrange(offset, hidden_shape.clone());
            encode_rms_norm_mul_vjp_rows_f32(
                context,
                &encoder,
                &replay.input,
                weights.attn_norm,
                &grad_hidden_query,
                &grad_input_query,
                n_tokens,
                geometry.hidden_size,
                RMS_EPS,
                match rule {
                    GdnMixerVjpRule::Jacobian => RmsNormVjpRule::Jacobian,
                    GdnMixerVjpRule::Relp => RmsNormVjpRule::RelpDetachedScale,
                },
            )?;
        }
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;
    let (
        final_conv_state,
        final_recurrence_state,
        grad_initial_conv_state,
        grad_initial_recurrence_state,
    ) = if read_state_diagnostics {
        (
            read_f32(&replay.conv_state, geometry.conv_state_elements),
            read_f32(&replay.recurrence_state, geometry.state_elements),
            read_f32(&grad_initial_conv_state, conv_state_query_elements),
            read_f32(&grad_initial_recurrence_state, state_query_elements),
        )
    } else {
        (Vec::new(), Vec::new(), Vec::new(), Vec::new())
    };
    Ok(GdnReplayVjpReadback {
        mixer_outputs: read_f32(&replay.mixer_output, hidden_elements),
        final_conv_state,
        final_recurrence_state,
        grad_input: read_f32(&grad_input, hidden_query_elements),
        grad_initial_conv_state,
        grad_initial_recurrence_state,
    })
}

pub(super) struct GdnBlockComposition {
    pub(super) values: Vec<f32>,
    pub(super) grad_post_mixer_residuals: Vec<f32>,
    pub(super) mixer: WorkspaceLensGdnVjp,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compose_gdn_block_vjp(
    context: &MetalContext,
    layer: u32,
    hidden_size: usize,
    intermediate_size: usize,
    post_mixer_residuals: &[f32],
    post_norm: &MetalTensor,
    gate_weight: &MetalTensor,
    up_weight: &MetalTensor,
    down_weight: &MetalTensor,
    grad_block_output: &[f32],
    n_rows: usize,
    rule: GdnBlockVjpRule,
    mixer_vjp: impl FnOnce(&[f32], GdnMixerVjpRule) -> Result<WorkspaceLensGdnVjp, WorkspaceLensError>,
) -> Result<GdnBlockComposition, WorkspaceLensError> {
    let grad_post_mixer_residuals = dense_ffn_vjp_rows_readback(
        context,
        layer,
        hidden_size,
        intermediate_size,
        post_mixer_residuals,
        post_norm,
        gate_weight,
        up_weight,
        down_weight,
        grad_block_output,
        n_rows,
        match rule {
            GdnBlockVjpRule::Jacobian => DenseFfnVjpRule::Jacobian,
            GdnBlockVjpRule::Relp => DenseFfnVjpRule::Relp,
        },
    )?;
    let mixer = mixer_vjp(
        &grad_post_mixer_residuals,
        match rule {
            GdnBlockVjpRule::Jacobian => GdnMixerVjpRule::Jacobian,
            GdnBlockVjpRule::Relp => GdnMixerVjpRule::Relp,
        },
    )?;
    if mixer.values.len() != grad_post_mixer_residuals.len() {
        return Err(WorkspaceLensError::ActivationSize {
            name: "GDN mixer branch cotangent",
            got: mixer.values.len(),
            expected: grad_post_mixer_residuals.len(),
        });
    }
    let values = grad_post_mixer_residuals
        .iter()
        .zip(&mixer.values)
        .map(|(&identity, &branch)| identity + branch)
        .collect();
    Ok(GdnBlockComposition {
        values,
        grad_post_mixer_residuals,
        mixer,
    })
}

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    /// Advance a bounded prompt while capturing the real input trajectory and
    /// recurrent boundary states for one GDN layer.
    ///
    /// The selected layer must be nonzero. A command failure after an earlier
    /// token succeeds can leave that successful prefix consumed, matching the
    /// existing token-at-a-time workspace-lens forward contract.
    pub fn forward_prompt_with_gdn_capture(
        &mut self,
        token_ids: &[i32],
        layer: u32,
    ) -> Result<WorkspaceLensGdnForward, WorkspaceLensError> {
        if token_ids.is_empty() {
            return Err(WorkspaceLensError::EmptyGdnPrompt);
        }
        if token_ids.len() > MAX_WORKSPACE_LENS_GDN_TOKENS {
            return Err(WorkspaceLensError::GdnPromptTooLong {
                got: token_ids.len(),
                max: MAX_WORKSPACE_LENS_GDN_TOKENS,
            });
        }
        if layer == 0 {
            return Err(WorkspaceLensError::GdnCaptureRequiresPreviousLayer);
        }
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= self.arch().vocab_size {
                return Err(MfError::BadToken(token_id, self.arch().vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let start_position = self.sequence.position();
        let last_position = start_position
            .checked_add(token_ids.len() - 1)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        u32::try_from(last_position)
            .map_err(|_| WorkspaceLensError::PositionOverflow(last_position))?;
        let (gdn_index, geometry) = {
            let (block, gdn_index, geometry) = self.resolve_gdn(layer)?;
            validate_gdn_weights(layer, block, geometry)?;
            (gdn_index, geometry)
        };
        let (initial_conv_state, initial_recurrence_state) = {
            let state = unsafe { self.sequence.metal_session_mut() };
            state.ensure_usable()?;
            let conv = state
                .gdn_conv
                .get(gdn_index)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            let recurrence = state
                .gdn_state
                .get(gdn_index)
                .ok_or(WorkspaceLensError::SizeOverflow)?;
            (
                read_f32(conv, geometry.conv_state_elements),
                read_f32(recurrence, geometry.state_elements),
            )
        };

        let hidden_elements = token_ids
            .len()
            .checked_mul(geometry.hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        let mut input_residuals = Vec::with_capacity(hidden_elements);
        let mut post_mixer_residuals = Vec::with_capacity(hidden_elements);
        let mut post_block_residuals = Vec::with_capacity(hidden_elements);
        let mut final_logits = Vec::new();
        let capture_layers = [layer - 1, layer];
        for &token_id in token_ids {
            let forward = match self.forward_token_with_dense_ffn_capture(token_id, &capture_layers)
            {
                Ok(forward) => forward,
                Err(error) => {
                    let state = unsafe { self.sequence.metal_session_mut() };
                    state.poison("GDN prompt capture forward failed");
                    return Err(error);
                }
            };
            let hidden = geometry.hidden_size;
            input_residuals.extend_from_slice(&forward.capture.post_block_residuals[..hidden]);
            post_mixer_residuals
                .extend_from_slice(&forward.capture.pre_ffn_residuals[hidden..2 * hidden]);
            post_block_residuals
                .extend_from_slice(&forward.capture.post_block_residuals[hidden..2 * hidden]);
            final_logits = forward.logits;
        }
        let (final_conv_state, final_recurrence_state) = {
            let state = unsafe { self.sequence.metal_session_mut() };
            state.ensure_usable()?;
            (
                read_f32(&state.gdn_conv[gdn_index], geometry.conv_state_elements),
                read_f32(&state.gdn_state[gdn_index], geometry.state_elements),
            )
        };
        Ok(WorkspaceLensGdnForward {
            identity: self.identity(),
            owner_token_id: self.model.owner_token_id(),
            layer,
            start_position,
            token_ids: token_ids.to_vec(),
            hidden_size: geometry.hidden_size,
            final_logits,
            input_residuals,
            post_mixer_residuals,
            post_block_residuals,
            initial_conv_state,
            initial_recurrence_state,
            final_conv_state,
            final_recurrence_state,
        })
    }

    /// Reverse the isolated mixer branch represented by a matching GDN prompt
    /// capture. The result stops at the selected layer's input residual. If the
    /// incoming cotangent is on `input + mixer(input)`, callers add that same
    /// cotangent as the residual identity branch. Terminal convolution and
    /// recurrence-state cotangents are fixed to zero, so captures are isolated
    /// sequences and cannot be stitched into a longer reverse pass.
    pub fn gdn_mixer_vjp(
        &self,
        forward: &WorkspaceLensGdnForward,
        grad_mixer_output: &[f32],
        rule: GdnMixerVjpRule,
    ) -> Result<WorkspaceLensGdnVjp, WorkspaceLensError> {
        if forward.identity != self.identity() {
            return Err(WorkspaceLensError::GdnCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(WorkspaceLensError::GdnCaptureOwnerMismatch);
        }
        let (block, _, geometry) = self.resolve_gdn(forward.layer)?;
        validate_gdn_weights(forward.layer, block, geometry)?;
        let n_tokens = forward.n_tokens();
        if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_GDN_TOKENS {
            return Err(WorkspaceLensError::ActivationSize {
                name: "GDN capture token count",
                got: n_tokens,
                expected: 1,
            });
        }
        let hidden_elements = n_tokens
            .checked_mul(geometry.hidden_size)
            .ok_or(WorkspaceLensError::SizeOverflow)?;
        for (name, values, expected) in [
            (
                "GDN captured input residuals",
                forward.input_residuals.as_slice(),
                hidden_elements,
            ),
            (
                "GDN captured post-mixer residuals",
                forward.post_mixer_residuals.as_slice(),
                hidden_elements,
            ),
            (
                "GDN captured post-block residuals",
                forward.post_block_residuals.as_slice(),
                hidden_elements,
            ),
            (
                "GDN captured initial conv state",
                forward.initial_conv_state.as_slice(),
                geometry.conv_state_elements,
            ),
            (
                "GDN captured initial recurrence state",
                forward.initial_recurrence_state.as_slice(),
                geometry.state_elements,
            ),
            (
                "GDN captured final conv state",
                forward.final_conv_state.as_slice(),
                geometry.conv_state_elements,
            ),
            (
                "GDN captured final recurrence state",
                forward.final_recurrence_state.as_slice(),
                geometry.state_elements,
            ),
        ] {
            if values.len() != expected {
                return Err(WorkspaceLensError::ActivationSize {
                    name,
                    got: values.len(),
                    expected,
                });
            }
        }
        if forward.hidden_size != geometry.hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "GDN capture hidden size",
                got: forward.hidden_size,
                expected: geometry.hidden_size,
            });
        }
        if grad_mixer_output.len() != hidden_elements {
            return Err(WorkspaceLensError::ActivationSize {
                name: "GDN mixer cotangent",
                got: grad_mixer_output.len(),
                expected: hidden_elements,
            });
        }
        let replay = gdn_mixer_replay_vjp_readback(
            self.model.context(),
            geometry,
            GdnMixerWeights::from(block),
            &forward.input_residuals,
            &forward.initial_conv_state,
            &forward.initial_recurrence_state,
            grad_mixer_output,
            n_tokens,
            rule,
            true,
        )?;
        let residual_replay_max_abs_error = forward
            .input_residuals
            .iter()
            .zip(&replay.mixer_outputs)
            .zip(&forward.post_mixer_residuals)
            .map(|((&input, &mixer), &observed)| finite_abs_difference(input + mixer, observed))
            .fold(0.0f32, f32::max);
        Ok(WorkspaceLensGdnVjp {
            layer: forward.layer,
            n_tokens,
            hidden_size: geometry.hidden_size,
            values: replay.grad_input,
            grad_initial_conv_state: replay.grad_initial_conv_state,
            grad_initial_recurrence_state: replay.grad_initial_recurrence_state,
            replay_mixer_outputs: replay.mixer_outputs,
            residual_replay_max_abs_error,
            final_conv_state_max_abs_error: max_abs_difference(
                &replay.final_conv_state,
                &forward.final_conv_state,
            ),
            final_recurrence_state_max_abs_error: max_abs_difference(
                &replay.final_recurrence_state,
                &forward.final_recurrence_state,
            ),
        })
    }

    /// Reverse a complete GDN block over the captured prompt trajectory.
    /// This composes the rowwise dense FFN VJP, its residual identity, the
    /// temporal GDN mixer VJP, and the mixer residual identity.
    pub fn gdn_block_vjp(
        &self,
        forward: &WorkspaceLensGdnForward,
        grad_block_output: &[f32],
        rule: GdnBlockVjpRule,
    ) -> Result<WorkspaceLensGdnBlockVjp, WorkspaceLensError> {
        if forward.identity != self.identity() {
            return Err(WorkspaceLensError::GdnCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(WorkspaceLensError::GdnCaptureOwnerMismatch);
        }
        let n_tokens = forward.n_tokens();
        let hidden_size = self.arch().hidden_size as usize;
        let expected = checked_product(n_tokens, hidden_size)?;
        if grad_block_output.len() != expected {
            return Err(WorkspaceLensError::ActivationSize {
                name: "GDN block cotangent",
                got: grad_block_output.len(),
                expected,
            });
        }
        if forward.post_mixer_residuals.len() != expected {
            return Err(WorkspaceLensError::ActivationSize {
                name: "GDN captured post-mixer residuals",
                got: forward.post_mixer_residuals.len(),
                expected,
            });
        }
        let (post_norm, gate, up, down) = self.resolve_dense_ffn(forward.layer)?;
        let composition = compose_gdn_block_vjp(
            self.model.context(),
            forward.layer,
            hidden_size,
            self.arch().intermediate_size as usize,
            &forward.post_mixer_residuals,
            post_norm,
            gate,
            up,
            down,
            grad_block_output,
            n_tokens,
            rule,
            |grad_post_mixer, mixer_rule| self.gdn_mixer_vjp(forward, grad_post_mixer, mixer_rule),
        )?;
        Ok(WorkspaceLensGdnBlockVjp {
            layer: forward.layer,
            n_tokens,
            hidden_size,
            values: composition.values,
            grad_post_mixer_residuals: composition.grad_post_mixer_residuals,
            mixer: composition.mixer,
        })
    }

    pub(super) fn resolve_gdn(
        &self,
        layer: u32,
    ) -> Result<(&MetalGdnBlock, usize, GdnGeometry), WorkspaceLensError> {
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            WorkspaceLensError::InvalidLayer {
                layer,
                n_layers: self.arch().n_layer,
            },
        )?;
        let MetalBlock::Gdn(block) = block else {
            return Err(WorkspaceLensError::NotGdnLayer { layer });
        };
        let gdn_index = self.model.metal_model().blocks[..layer as usize]
            .iter()
            .filter(|block| matches!(block, MetalBlock::Gdn(_)))
            .count();
        Ok((block, gdn_index, GdnGeometry::new(layer, self.arch())?))
    }
}
