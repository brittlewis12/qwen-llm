//! Attention mixer capture and VJP readbacks.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct AttnGeometry {
    pub(super) hidden_size: usize,
    pub(super) n_q_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) n_rot: usize,
    pub(super) q_elements: usize,
    pub(super) q_full_elements: usize,
    pub(super) kv_elements: usize,
    pub(super) rope_theta: f32,
}

impl AttnGeometry {
    pub(super) fn new(arch: Arch) -> Result<Self, WorkspaceLensError> {
        let hidden_size = arch.hidden_size as usize;
        let n_q_heads = arch.n_q_heads as usize;
        let n_kv_heads = arch.n_kv_heads as usize;
        let head_dim = arch.attn_head_dim as usize;
        let n_rot = (head_dim as f32 * arch.partial_rotary_factor) as usize;
        if hidden_size == 0
            || n_q_heads == 0
            || n_kv_heads == 0
            || !n_q_heads.is_multiple_of(n_kv_heads)
            || head_dim == 0
            || n_rot == 0
            || n_rot > head_dim
            || !n_rot.is_multiple_of(2)
            || !arch.rope_theta.is_finite()
            || arch.rope_theta <= 0.0
        {
            return Err(WorkspaceLensError::SizeOverflow);
        }
        let q_elements = checked_product(n_q_heads, head_dim)?;
        Ok(Self {
            hidden_size,
            n_q_heads,
            n_kv_heads,
            head_dim,
            n_rot,
            q_elements,
            q_full_elements: checked_product(q_elements, 2)?,
            kv_elements: checked_product(n_kv_heads, head_dim)?,
            rope_theta: arch.rope_theta,
        })
    }
}

pub(super) struct CpuCausalAttentionForward {
    pub(super) attention_output: Vec<f32>,
    pub(super) gated_output: Vec<f32>,
}

pub(super) struct CpuCausalAttentionVjp {
    pub(super) grad_q: Vec<f32>,
    pub(super) grad_k: Vec<f32>,
    pub(super) grad_v: Vec<f32>,
    pub(super) grad_gate: Vec<f32>,
}

#[allow(clippy::needless_range_loop)]
pub(super) fn cpu_causal_gated_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    n_tokens: usize,
    geometry: AttnGeometry,
) -> Result<CpuCausalAttentionForward, WorkspaceLensError> {
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    for (name, values, expected) in [
        ("attention Q", q, q_total),
        ("attention K", k, kv_total),
        ("attention V", v, kv_total),
        ("attention gate", gate, q_total),
    ] {
        if values.len() != expected {
            return Err(WorkspaceLensError::ActivationSize {
                name,
                got: values.len(),
                expected,
            });
        }
    }
    let group = geometry.n_q_heads / geometry.n_kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut attention_output = vec![0.0f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.n_q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.n_q_heads + q_head) * geometry.head_dim;
            let mut scores = vec![0.0f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0f32;
                for index in 0..geometry.head_dim {
                    score += q[q_base + index] * k[k_base + index];
                }
                scores[key_token] = score * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                denominator += *score;
            }
            for score in &mut scores {
                *score /= denominator;
            }
            for key_token in 0..=token {
                let v_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                let probability = scores[key_token];
                for index in 0..geometry.head_dim {
                    attention_output[q_base + index] += probability * v[v_base + index];
                }
            }
        }
    }
    let gated_output = attention_output
        .iter()
        .zip(gate)
        .map(|(&attention, &gate)| attention * (1.0 + (-gate).exp()).recip())
        .collect();
    Ok(CpuCausalAttentionForward {
        attention_output,
        gated_output,
    })
}

#[allow(clippy::needless_range_loop)]
pub(super) fn cpu_causal_gated_attention_vjp(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    grad_gated_output: &[f32],
    n_tokens: usize,
    geometry: AttnGeometry,
) -> Result<CpuCausalAttentionVjp, WorkspaceLensError> {
    let forward = cpu_causal_gated_attention_forward(q, k, v, gate, n_tokens, geometry)?;
    cpu_causal_gated_attention_vjp_with_forward(
        q,
        k,
        v,
        gate,
        grad_gated_output,
        n_tokens,
        geometry,
        &forward,
    )
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub(super) fn cpu_causal_gated_attention_vjp_with_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    grad_gated_output: &[f32],
    n_tokens: usize,
    geometry: AttnGeometry,
    forward: &CpuCausalAttentionForward,
) -> Result<CpuCausalAttentionVjp, WorkspaceLensError> {
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    for (name, values, expected) in [
        ("attention VJP Q", q, q_total),
        ("attention VJP K", k, kv_total),
        ("attention VJP V", v, kv_total),
        ("attention VJP gate", gate, q_total),
    ] {
        if values.len() != expected {
            return Err(WorkspaceLensError::ActivationSize {
                name,
                got: values.len(),
                expected,
            });
        }
    }
    if grad_gated_output.len() != q_total || forward.attention_output.len() != q_total {
        return Err(WorkspaceLensError::ActivationSize {
            name: "gated attention cotangent/shared forward",
            got: grad_gated_output.len().min(forward.attention_output.len()),
            expected: q_total,
        });
    }
    let group = geometry.n_q_heads / geometry.n_kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut grad_q = vec![0.0f32; q_total];
    let mut grad_k = vec![0.0f32; kv_total];
    let mut grad_v = vec![0.0f32; kv_total];
    let mut grad_gate = vec![0.0f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.n_q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.n_q_heads + q_head) * geometry.head_dim;
            let mut grad_attention = vec![0.0f32; geometry.head_dim];
            for index in 0..geometry.head_dim {
                let sigmoid = (1.0 + (-gate[q_base + index]).exp()).recip();
                grad_attention[index] = grad_gated_output[q_base + index] * sigmoid;
                grad_gate[q_base + index] = grad_gated_output[q_base + index]
                    * forward.attention_output[q_base + index]
                    * sigmoid
                    * (1.0 - sigmoid);
            }
            let mut scores = vec![0.0f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0f32;
                for index in 0..geometry.head_dim {
                    score += q[q_base + index] * k[k_base + index];
                }
                scores[key_token] = score * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denominator = 0.0f32;
            for score in &mut scores {
                *score = (*score - max).exp();
                denominator += *score;
            }
            for score in &mut scores {
                *score /= denominator;
            }
            let mut grad_probability = vec![0.0f32; token + 1];
            for key_token in 0..=token {
                let v_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                for index in 0..geometry.head_dim {
                    grad_probability[key_token] += grad_attention[index] * v[v_base + index];
                    grad_v[v_base + index] += scores[key_token] * grad_attention[index];
                }
            }
            let probability_dot = scores
                .iter()
                .zip(&grad_probability)
                .map(|(&probability, &gradient)| probability * gradient)
                .sum::<f32>();
            for key_token in 0..=token {
                let grad_score =
                    scores[key_token] * (grad_probability[key_token] - probability_dot);
                let k_base = (key_token * geometry.n_kv_heads + kv_head) * geometry.head_dim;
                for index in 0..geometry.head_dim {
                    grad_q[q_base + index] += scale * grad_score * k[k_base + index];
                    grad_k[k_base + index] += scale * grad_score * q[q_base + index];
                }
            }
        }
    }
    Ok(CpuCausalAttentionVjp {
        grad_q,
        grad_k,
        grad_v,
        grad_gate,
    })
}

#[derive(Clone, Copy)]
pub(super) struct AttnMixerWeights<'a> {
    pub(super) attn_norm: &'a MetalTensor,
    pub(super) q: &'a MetalTensor,
    pub(super) k: &'a MetalTensor,
    pub(super) v: &'a MetalTensor,
    pub(super) o: &'a MetalTensor,
    pub(super) q_norm: &'a MetalTensor,
    pub(super) k_norm: &'a MetalTensor,
}

impl<'a> From<&'a MetalAttnBlock> for AttnMixerWeights<'a> {
    fn from(block: &'a MetalAttnBlock) -> Self {
        Self {
            attn_norm: &block.attn_norm,
            q: &block.q,
            k: &block.k,
            v: &block.v,
            o: &block.o,
            q_norm: &block.q_norm,
            k_norm: &block.k_norm,
        }
    }
}

pub(super) fn validate_attn_weights(
    layer: u32,
    block: &MetalAttnBlock,
    geometry: AttnGeometry,
) -> Result<(), WorkspaceLensError> {
    let weights = AttnMixerWeights::from(block);
    for (role, weight, expected) in [
        (
            LinearRole::AttentionQAndGate,
            weights.q,
            [geometry.hidden_size, geometry.q_full_elements],
        ),
        (
            LinearRole::AttentionK,
            weights.k,
            [geometry.hidden_size, geometry.kv_elements],
        ),
        (
            LinearRole::AttentionV,
            weights.v,
            [geometry.hidden_size, geometry.kv_elements],
        ),
        (
            LinearRole::AttentionOut,
            weights.o,
            [geometry.q_elements, geometry.hidden_size],
        ),
    ] {
        let id = WorkspaceLensLinear::Layer { index: layer, role };
        let got = linear_shape(id, weight)?;
        if got != expected {
            return Err(WorkspaceLensError::InvalidAttnLinearShape {
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
        ("Q norm", weights.q_norm, geometry.head_dim),
        ("K norm", weights.k_norm, geometry.head_dim),
    ] {
        if tensor.dtype != GgmlType::F32 || tensor.n_elements() as usize != expected_elements {
            return Err(WorkspaceLensError::InvalidAttnTensor {
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

pub(super) struct AttnReplayFrontTensors {
    pub(super) input: MetalTensor,
    pub(super) normalized: MetalTensor,
    pub(super) q_full: MetalTensor,
    pub(super) q_raw: MetalTensor,
    pub(super) gate: MetalTensor,
    pub(super) k_raw: MetalTensor,
    pub(super) v: MetalTensor,
    pub(super) q_normed: MetalTensor,
    pub(super) k_normed: MetalTensor,
}

impl AttnReplayFrontTensors {
    pub(super) fn new(
        context: &MetalContext,
        geometry: AttnGeometry,
        input: &[f32],
        n_tokens: usize,
    ) -> Result<Self, WorkspaceLensError> {
        let hidden_total = checked_product(n_tokens, geometry.hidden_size)?;
        if input.len() != hidden_total {
            return Err(WorkspaceLensError::ActivationSize {
                name: "attention replay input",
                got: input.len(),
                expected: hidden_total,
            });
        }
        let q_total = checked_product(n_tokens, geometry.q_elements)?;
        let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
        let q_full_total = checked_product(n_tokens, geometry.q_full_elements)?;
        let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
        Ok(Self {
            input: MetalTensor::from_bytes(
                context,
                bytemuck::cast_slice(input),
                hidden_shape.clone(),
                GgmlType::F32,
            )?,
            normalized: MetalTensor::zeros_f32(context, hidden_shape)?,
            q_full: flat_f32(context, q_full_total)?,
            q_raw: flat_f32(context, q_total)?,
            gate: flat_f32(context, q_total)?,
            k_raw: flat_f32(context, kv_total)?,
            v: flat_f32(context, kv_total)?,
            q_normed: flat_f32(context, q_total)?,
            k_normed: flat_f32(context, kv_total)?,
        })
    }

    pub(super) fn encode(
        &self,
        context: &MetalContext,
        encoder: &KernelEncoder,
        geometry: AttnGeometry,
        weights: AttnMixerWeights<'_>,
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
            let q_full = row_view(&self.q_full, token, geometry.q_full_elements);
            let q = row_view(&self.q_raw, token, geometry.q_elements);
            let gate = row_view(&self.gate, token, geometry.q_elements);
            let k = row_view(&self.k_raw, token, geometry.kv_elements);
            let v = row_view(&self.v, token, geometry.kv_elements);
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.q,
                &hidden,
                &q_full,
                geometry.hidden_size,
                geometry.q_full_elements,
            )?;
            encode_split_q_gate_f32(
                context,
                encoder,
                &q_full,
                &q,
                &gate,
                geometry.n_q_heads,
                geometry.head_dim,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.k,
                &hidden,
                &k,
                geometry.hidden_size,
                geometry.kv_elements,
            )?;
            encode_mat_vec_dispatch(
                context,
                encoder,
                weights.v,
                &hidden,
                &v,
                geometry.hidden_size,
                geometry.kv_elements,
            )?;
        }
        encode_rms_norm_batched_f32(
            context,
            encoder,
            &self.q_raw,
            weights.q_norm,
            &self.q_normed,
            checked_product(n_tokens, geometry.n_q_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        encode_rms_norm_batched_f32(
            context,
            encoder,
            &self.k_raw,
            weights.k_norm,
            &self.k_normed,
            checked_product(n_tokens, geometry.n_kv_heads)?,
            geometry.head_dim,
            RMS_EPS,
        )?;
        Ok(())
    }
}

pub(super) struct AttnMixerVjpReadback {
    pub(super) mixer_outputs: Vec<f32>,
    pub(super) grad_input: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attn_mixer_replay_vjp_readback(
    context: &MetalContext,
    geometry: AttnGeometry,
    weights: AttnMixerWeights<'_>,
    input: &[f32],
    grad_mixer_output: &[f32],
    n_tokens: usize,
    rule: AttnBlockVjpRule,
) -> Result<AttnMixerVjpReadback, WorkspaceLensError> {
    if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_ATTN_TOKENS {
        return Err(WorkspaceLensError::AttnPromptTooLong {
            got: n_tokens,
            max: MAX_WORKSPACE_LENS_ATTN_TOKENS,
        });
    }
    let hidden_total = checked_product(n_tokens, geometry.hidden_size)?;
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    if input.len() != hidden_total || grad_mixer_output.len() != hidden_total {
        return Err(WorkspaceLensError::ActivationSize {
            name: "attention mixer input/cotangent",
            got: input.len().min(grad_mixer_output.len()),
            expected: hidden_total,
        });
    }
    let front = AttnReplayFrontTensors::new(context, geometry, input, n_tokens)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = front.encode(context, &encoder, geometry, weights, n_tokens);
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;

    let q_raw = read_f32(&front.q_raw, q_total);
    let k_raw = read_f32(&front.k_raw, kv_total);
    let gate = read_f32(&front.gate, q_total);
    let v = read_f32(&front.v, kv_total);
    let mut q = read_f32(&front.q_normed, q_total);
    let mut k = read_f32(&front.k_normed, kv_total);
    rope_neox_rows_in_place(
        &mut q,
        n_tokens,
        geometry.n_q_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        false,
    )?;
    rope_neox_rows_in_place(
        &mut k,
        n_tokens,
        geometry.n_kv_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        false,
    )?;
    let attention = cpu_causal_gated_attention_forward(&q, &k, &v, &gate, n_tokens, geometry)?;

    let gated_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&attention.gated_output),
        row_shape(geometry.q_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let mixer_output = MetalTensor::zeros_f32(context, row_shape(geometry.hidden_size, n_tokens)?)?;
    let grad_mixer = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_mixer_output),
        row_shape(geometry.hidden_size, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_gated = MetalTensor::zeros_f32(context, row_shape(geometry.q_elements, n_tokens)?)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        for token in 0..n_tokens {
            let gated = row_view(&gated_output, token, geometry.q_elements);
            let mixer = row_view(&mixer_output, token, geometry.hidden_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                weights.o,
                &gated,
                &mixer,
                geometry.q_elements,
                geometry.hidden_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            weights.o,
            &grad_mixer,
            &grad_gated,
            geometry.q_elements,
            geometry.hidden_size,
            n_tokens,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;
    let mixer_outputs = read_f32(&mixer_output, hidden_total);
    let grad_gated = read_f32(&grad_gated, q_total);

    let mut attention_vjp =
        cpu_causal_gated_attention_vjp(&q, &k, &v, &gate, &grad_gated, n_tokens, geometry)?;
    rope_neox_rows_in_place(
        &mut attention_vjp.grad_q,
        n_tokens,
        geometry.n_q_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        true,
    )?;
    rope_neox_rows_in_place(
        &mut attention_vjp.grad_k,
        n_tokens,
        geometry.n_kv_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        true,
    )?;
    let q_norm_weight = read_f32(weights.q_norm, geometry.head_dim);
    let k_norm_weight = read_f32(weights.k_norm, geometry.head_dim);
    let grad_q_raw = cpu_weighted_rms_vjp_rows(
        &q_raw,
        &q_norm_weight,
        &attention_vjp.grad_q,
        checked_product(n_tokens, geometry.n_q_heads)?,
        geometry.head_dim,
        false,
    )?;
    let grad_k_raw = cpu_weighted_rms_vjp_rows(
        &k_raw,
        &k_norm_weight,
        &attention_vjp.grad_k,
        checked_product(n_tokens, geometry.n_kv_heads)?,
        geometry.head_dim,
        false,
    )?;
    let mut grad_q_full = vec![0.0f32; checked_product(n_tokens, geometry.q_full_elements)?];
    for token in 0..n_tokens {
        for head in 0..geometry.n_q_heads {
            let source = (token * geometry.n_q_heads + head) * geometry.head_dim;
            let destination = (token * geometry.n_q_heads + head) * 2 * geometry.head_dim;
            grad_q_full[destination..destination + geometry.head_dim]
                .copy_from_slice(&grad_q_raw[source..source + geometry.head_dim]);
            grad_q_full[destination + geometry.head_dim..destination + 2 * geometry.head_dim]
                .copy_from_slice(&attention_vjp.grad_gate[source..source + geometry.head_dim]);
        }
    }

    let grad_q_full = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_q_full),
        row_shape(geometry.q_full_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_k = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_k_raw),
        row_shape(geometry.kv_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let grad_v = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&attention_vjp.grad_v),
        row_shape(geometry.kv_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
    let grad_hidden_q = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_k = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_v = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden_qk = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_hidden = MetalTensor::zeros_f32(context, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_shape)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        for (weight, grad_output, grad_hidden, n_out) in [
            (
                weights.q,
                &grad_q_full,
                &grad_hidden_q,
                geometry.q_full_elements,
            ),
            (weights.k, &grad_k, &grad_hidden_k, geometry.kv_elements),
            (weights.v, &grad_v, &grad_hidden_v, geometry.kv_elements),
        ] {
            encode_frozen_linear_vjp_f32(
                context,
                &encoder,
                weight,
                grad_output,
                grad_hidden,
                geometry.hidden_size,
                n_out,
                n_tokens,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_q,
            &grad_hidden_k,
            &grad_hidden_qk,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_qk,
            &grad_hidden_v,
            &grad_hidden,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            context,
            &encoder,
            &front.input,
            weights.attn_norm,
            &grad_hidden,
            &grad_input,
            n_tokens,
            geometry.hidden_size,
            RMS_EPS,
            match rule {
                AttnBlockVjpRule::Jacobian => RmsNormVjpRule::Jacobian,
                AttnBlockVjpRule::Relp => RmsNormVjpRule::RelpDetachedScale,
            },
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;
    Ok(AttnMixerVjpReadback {
        mixer_outputs,
        grad_input: read_f32(&grad_input, hidden_total),
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn attn_mixer_replay_vjp_batch_readback(
    context: &MetalContext,
    geometry: AttnGeometry,
    weights: AttnMixerWeights<'_>,
    input: &[f32],
    grad_mixer_outputs: &[f32],
    n_tokens: usize,
    n_query: usize,
    rule: AttnBlockVjpRule,
) -> Result<AttnMixerVjpReadback, WorkspaceLensError> {
    if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_ATTN_TOKENS {
        return Err(WorkspaceLensError::AttnPromptTooLong {
            got: n_tokens,
            max: MAX_WORKSPACE_LENS_ATTN_TOKENS,
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
    let hidden_total = checked_product(n_tokens, geometry.hidden_size)?;
    let q_total = checked_product(n_tokens, geometry.q_elements)?;
    let kv_total = checked_product(n_tokens, geometry.kv_elements)?;
    let q_full_total = checked_product(n_tokens, geometry.q_full_elements)?;
    let query_rows = checked_product(n_query, n_tokens)?;
    let hidden_query_total = checked_product(n_query, hidden_total)?;
    let q_query_total = checked_product(n_query, q_total)?;
    let kv_query_total = checked_product(n_query, kv_total)?;
    let q_full_query_total = checked_product(n_query, q_full_total)?;
    if input.len() != hidden_total {
        return Err(WorkspaceLensError::ActivationSize {
            name: "attention mixer input",
            got: input.len(),
            expected: hidden_total,
        });
    }
    if grad_mixer_outputs.len() != hidden_query_total {
        return Err(WorkspaceLensError::ActivationSize {
            name: "attention mixer cotangent query bank",
            got: grad_mixer_outputs.len(),
            expected: hidden_query_total,
        });
    }

    let front = AttnReplayFrontTensors::new(context, geometry, input, n_tokens)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = front.encode(context, &encoder, geometry, weights, n_tokens);
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;

    let q_raw = read_f32(&front.q_raw, q_total);
    let k_raw = read_f32(&front.k_raw, kv_total);
    let gate = read_f32(&front.gate, q_total);
    let v = read_f32(&front.v, kv_total);
    let mut q = read_f32(&front.q_normed, q_total);
    let mut k = read_f32(&front.k_normed, kv_total);
    rope_neox_rows_in_place(
        &mut q,
        n_tokens,
        geometry.n_q_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        false,
    )?;
    rope_neox_rows_in_place(
        &mut k,
        n_tokens,
        geometry.n_kv_heads,
        geometry.head_dim,
        geometry.n_rot,
        0,
        geometry.rope_theta,
        false,
    )?;
    let attention = cpu_causal_gated_attention_forward(&q, &k, &v, &gate, n_tokens, geometry)?;

    let gated_output = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&attention.gated_output),
        row_shape(geometry.q_elements, n_tokens)?,
        GgmlType::F32,
    )?;
    let mixer_output = MetalTensor::zeros_f32(context, row_shape(geometry.hidden_size, n_tokens)?)?;
    let grad_mixer = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(grad_mixer_outputs),
        row_shape(geometry.hidden_size, query_rows)?,
        GgmlType::F32,
    )?;
    let grad_gated = MetalTensor::zeros_f32(context, row_shape(geometry.q_elements, query_rows)?)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        for token in 0..n_tokens {
            let gated = row_view(&gated_output, token, geometry.q_elements);
            let mixer = row_view(&mixer_output, token, geometry.hidden_size);
            encode_mat_vec_dispatch(
                context,
                &encoder,
                weights.o,
                &gated,
                &mixer,
                geometry.q_elements,
                geometry.hidden_size,
            )?;
        }
        encode_frozen_linear_vjp_f32(
            context,
            &encoder,
            weights.o,
            &grad_mixer,
            &grad_gated,
            geometry.q_elements,
            geometry.hidden_size,
            query_rows,
        )?;
        Ok(())
    })();
    encoder.end();
    encode_result?;
    command.commit();
    crate::metal::wait_unchecked(&command);
    validate_completed_command(&command)?;
    let mixer_outputs = read_f32(&mixer_output, hidden_total);
    let grad_gated = read_f32(&grad_gated, q_query_total);

    let q_norm_weight = read_f32(weights.q_norm, geometry.head_dim);
    let k_norm_weight = read_f32(weights.k_norm, geometry.head_dim);
    let mut grad_q_full_values = Vec::with_capacity(q_full_query_total);
    let mut grad_k_values = Vec::with_capacity(kv_query_total);
    let mut grad_v_values = Vec::with_capacity(kv_query_total);
    for grad_gated_query in grad_gated.chunks_exact(q_total) {
        let mut attention_vjp = cpu_causal_gated_attention_vjp_with_forward(
            &q,
            &k,
            &v,
            &gate,
            grad_gated_query,
            n_tokens,
            geometry,
            &attention,
        )?;
        rope_neox_rows_in_place(
            &mut attention_vjp.grad_q,
            n_tokens,
            geometry.n_q_heads,
            geometry.head_dim,
            geometry.n_rot,
            0,
            geometry.rope_theta,
            true,
        )?;
        rope_neox_rows_in_place(
            &mut attention_vjp.grad_k,
            n_tokens,
            geometry.n_kv_heads,
            geometry.head_dim,
            geometry.n_rot,
            0,
            geometry.rope_theta,
            true,
        )?;
        let grad_q_raw = cpu_weighted_rms_vjp_rows(
            &q_raw,
            &q_norm_weight,
            &attention_vjp.grad_q,
            checked_product(n_tokens, geometry.n_q_heads)?,
            geometry.head_dim,
            false,
        )?;
        let grad_k_raw = cpu_weighted_rms_vjp_rows(
            &k_raw,
            &k_norm_weight,
            &attention_vjp.grad_k,
            checked_product(n_tokens, geometry.n_kv_heads)?,
            geometry.head_dim,
            false,
        )?;
        let mut grad_q_full = vec![0.0f32; q_full_total];
        for token in 0..n_tokens {
            for head in 0..geometry.n_q_heads {
                let source = (token * geometry.n_q_heads + head) * geometry.head_dim;
                let destination = (token * geometry.n_q_heads + head) * 2 * geometry.head_dim;
                grad_q_full[destination..destination + geometry.head_dim]
                    .copy_from_slice(&grad_q_raw[source..source + geometry.head_dim]);
                grad_q_full[destination + geometry.head_dim..destination + 2 * geometry.head_dim]
                    .copy_from_slice(&attention_vjp.grad_gate[source..source + geometry.head_dim]);
            }
        }
        grad_q_full_values.extend(grad_q_full);
        grad_k_values.extend(grad_k_raw);
        grad_v_values.extend(attention_vjp.grad_v);
    }
    for (name, got, expected) in [
        (
            "attention packed Q/gate cotangent query bank",
            grad_q_full_values.len(),
            q_full_query_total,
        ),
        (
            "attention K cotangent query bank",
            grad_k_values.len(),
            kv_query_total,
        ),
        (
            "attention V cotangent query bank",
            grad_v_values.len(),
            kv_query_total,
        ),
    ] {
        if got != expected {
            return Err(WorkspaceLensError::ActivationSize {
                name,
                got,
                expected,
            });
        }
    }

    let grad_q_full = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_q_full_values),
        row_shape(geometry.q_full_elements, query_rows)?,
        GgmlType::F32,
    )?;
    let grad_k = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_k_values),
        row_shape(geometry.kv_elements, query_rows)?,
        GgmlType::F32,
    )?;
    let grad_v = MetalTensor::from_bytes(
        context,
        bytemuck::cast_slice(&grad_v_values),
        row_shape(geometry.kv_elements, query_rows)?,
        GgmlType::F32,
    )?;
    let hidden_shape = row_shape(geometry.hidden_size, n_tokens)?;
    let hidden_query_shape = row_shape(geometry.hidden_size, query_rows)?;
    let grad_hidden_q = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_k = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_v = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden_qk = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_hidden = MetalTensor::zeros_f32(context, hidden_query_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(context, hidden_query_shape)?;
    let command = context
        .queue
        .commandBuffer()
        .ok_or(WorkspaceLensError::MissingCommandBuffer)?;
    let encoder = KernelEncoder::begin(&command);
    let encode_result = (|| -> Result<(), WorkspaceLensError> {
        for (weight, grad_output, grad_hidden, n_out) in [
            (
                weights.q,
                &grad_q_full,
                &grad_hidden_q,
                geometry.q_full_elements,
            ),
            (weights.k, &grad_k, &grad_hidden_k, geometry.kv_elements),
            (weights.v, &grad_v, &grad_hidden_v, geometry.kv_elements),
        ] {
            encode_frozen_linear_vjp_f32(
                context,
                &encoder,
                weight,
                grad_output,
                grad_hidden,
                geometry.hidden_size,
                n_out,
                query_rows,
            )?;
        }
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_q,
            &grad_hidden_k,
            &grad_hidden_qk,
        )?;
        encode_add_f32(
            context,
            &encoder,
            &grad_hidden_qk,
            &grad_hidden_v,
            &grad_hidden,
        )?;
        for query in 0..n_query {
            let offset = u64::try_from(checked_product(query, hidden_total)?)
                .map_err(|_| WorkspaceLensError::SizeOverflow)?;
            let grad_hidden_query = grad_hidden.view_subrange(offset, hidden_shape.clone());
            let grad_input_query = grad_input.view_subrange(offset, hidden_shape.clone());
            encode_rms_norm_mul_vjp_rows_f32(
                context,
                &encoder,
                &front.input,
                weights.attn_norm,
                &grad_hidden_query,
                &grad_input_query,
                n_tokens,
                geometry.hidden_size,
                RMS_EPS,
                match rule {
                    AttnBlockVjpRule::Jacobian => RmsNormVjpRule::Jacobian,
                    AttnBlockVjpRule::Relp => RmsNormVjpRule::RelpDetachedScale,
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
    Ok(AttnMixerVjpReadback {
        mixer_outputs,
        grad_input: read_f32(&grad_input, hidden_query_total),
    })
}

impl<'model, 'sequence> WorkspaceLensSession<'model, 'sequence> {
    /// Advance a fresh bounded prompt while capturing one full-attention block.
    /// The replay derivative is the model-level F32 attention Jacobian; it does
    /// not differentiate the production F16/Q8 KV storage conversion.
    pub fn forward_prompt_with_attn_capture(
        &mut self,
        token_ids: &[i32],
        layer: u32,
    ) -> Result<WorkspaceLensAttnForward, WorkspaceLensError> {
        if token_ids.is_empty() {
            return Err(WorkspaceLensError::EmptyAttnPrompt);
        }
        if token_ids.len() > MAX_WORKSPACE_LENS_ATTN_TOKENS {
            return Err(WorkspaceLensError::AttnPromptTooLong {
                got: token_ids.len(),
                max: MAX_WORKSPACE_LENS_ATTN_TOKENS,
            });
        }
        if layer == 0 {
            return Err(WorkspaceLensError::AttnCaptureRequiresPreviousLayer);
        }
        if self.sequence.position() != 0 {
            return Err(WorkspaceLensError::AttnCaptureRequiresFreshSequence(
                self.sequence.position(),
            ));
        }
        for &token_id in token_ids {
            if token_id < 0 || token_id as u32 >= self.arch().vocab_size {
                return Err(MfError::BadToken(token_id, self.arch().vocab_size).into());
            }
        }
        self.sequence.ensure_can_append(token_ids.len())?;
        let geometry = {
            let (block, geometry) = self.resolve_attn(layer)?;
            validate_attn_weights(layer, block, geometry)?;
            geometry
        };
        let hidden_elements = checked_product(token_ids.len(), geometry.hidden_size)?;
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
                    state.poison("attention prompt capture forward failed");
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
        Ok(WorkspaceLensAttnForward {
            identity: self.identity(),
            owner_token_id: self.model.owner_token_id(),
            layer,
            token_ids: token_ids.to_vec(),
            hidden_size: geometry.hidden_size,
            final_logits,
            input_residuals,
            post_mixer_residuals,
            post_block_residuals,
        })
    }

    /// Reverse one captured full-attention block using an ordinary F32 causal
    /// attention Jacobian and the selected residual-stream/FFN lens rule.
    pub fn attn_block_vjp(
        &self,
        forward: &WorkspaceLensAttnForward,
        grad_block_output: &[f32],
        rule: AttnBlockVjpRule,
    ) -> Result<WorkspaceLensAttnBlockVjp, WorkspaceLensError> {
        if forward.identity != self.identity() {
            return Err(WorkspaceLensError::AttnCaptureModelMismatch);
        }
        if forward.owner_token_id != self.model.owner_token_id() {
            return Err(WorkspaceLensError::AttnCaptureOwnerMismatch);
        }
        let (block, geometry) = self.resolve_attn(forward.layer)?;
        validate_attn_weights(forward.layer, block, geometry)?;
        let n_tokens = forward.n_tokens();
        if n_tokens == 0 || n_tokens > MAX_WORKSPACE_LENS_ATTN_TOKENS {
            return Err(WorkspaceLensError::AttnPromptTooLong {
                got: n_tokens,
                max: MAX_WORKSPACE_LENS_ATTN_TOKENS,
            });
        }
        let hidden_elements = checked_product(n_tokens, geometry.hidden_size)?;
        for (name, values) in [
            (
                "attention captured inputs",
                forward.input_residuals.as_slice(),
            ),
            (
                "attention captured post-mixer residuals",
                forward.post_mixer_residuals.as_slice(),
            ),
            (
                "attention captured post-block residuals",
                forward.post_block_residuals.as_slice(),
            ),
            ("attention block cotangent", grad_block_output),
        ] {
            if values.len() != hidden_elements {
                return Err(WorkspaceLensError::ActivationSize {
                    name,
                    got: values.len(),
                    expected: hidden_elements,
                });
            }
        }
        if forward.hidden_size != geometry.hidden_size {
            return Err(WorkspaceLensError::ActivationSize {
                name: "attention capture hidden size",
                got: forward.hidden_size,
                expected: geometry.hidden_size,
            });
        }
        let grad_post_mixer_residuals = dense_ffn_vjp_rows_readback(
            self.model.context(),
            forward.layer,
            geometry.hidden_size,
            self.arch().intermediate_size as usize,
            &forward.post_mixer_residuals,
            &block.post_attn_norm,
            &block.ffn_gate,
            &block.ffn_up,
            &block.ffn_down,
            grad_block_output,
            n_tokens,
            match rule {
                AttnBlockVjpRule::Jacobian => DenseFfnVjpRule::Jacobian,
                AttnBlockVjpRule::Relp => DenseFfnVjpRule::Relp,
            },
        )?;
        let mixer = attn_mixer_replay_vjp_readback(
            self.model.context(),
            geometry,
            AttnMixerWeights::from(block),
            &forward.input_residuals,
            &grad_post_mixer_residuals,
            n_tokens,
            rule,
        )?;
        let values = grad_post_mixer_residuals
            .iter()
            .zip(&mixer.grad_input)
            .map(|(&identity, &branch)| identity + branch)
            .collect();
        let residual_replay_max_abs_error = forward
            .input_residuals
            .iter()
            .zip(&mixer.mixer_outputs)
            .zip(&forward.post_mixer_residuals)
            .map(|((&input, &mixer), &observed)| finite_abs_difference(input + mixer, observed))
            .fold(0.0f32, f32::max);
        Ok(WorkspaceLensAttnBlockVjp {
            layer: forward.layer,
            n_tokens,
            hidden_size: geometry.hidden_size,
            values,
            grad_post_mixer_residuals,
            replay_mixer_outputs: mixer.mixer_outputs,
            residual_replay_max_abs_error,
        })
    }

    pub(super) fn resolve_attn(
        &self,
        layer: u32,
    ) -> Result<(&MetalAttnBlock, AttnGeometry), WorkspaceLensError> {
        let block = self.model.metal_model().blocks.get(layer as usize).ok_or(
            WorkspaceLensError::InvalidLayer {
                layer,
                n_layers: self.arch().n_layer,
            },
        )?;
        let MetalBlock::Attn(block) = block else {
            return Err(WorkspaceLensError::NotAttentionLayer { layer });
        };
        Ok((block, AttnGeometry::new(self.arch())?))
    }
}
