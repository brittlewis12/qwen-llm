//! Bounded one-block replay and VJP for Muse attention blocks.
//!
//! Production capture uses the scalar text path's F16 KV cache. This module
//! intentionally replays and differentiates the smooth F32 model-level graph,
//! matching the ordinary-Qwen research semantics rather than applying an STE
//! through F16 KV conversion. Replay diagnostics quantify that deployment
//! difference.

use crate::metal::{
    KernelEncoder, MetalContext, MetalTensor, encode_add_f32, encode_frozen_linear_vjp_f32,
    encode_rms_norm_batched_f32, encode_rms_norm_mul_rows_f32, encode_rms_norm_mul_vjp_rows_f32,
    encode_silu_mul_f32, encode_silu_mul_vjp_f32,
};
use crate::metal_forward::encode_mat_vec_dispatch;
use crate::muse_glimmer_lens::{
    MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS, MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS,
    MuseGlimmerLensCapture, MuseGlimmerLensError, MuseGlimmerLensRule, MuseGlimmerRmsNormSite,
    MuseGlimmerSelectedTokenCovectors,
};
use crate::muse_glimmer_residency::{MuseGlimmerMetalLayerWeights, MuseGlimmerMetalModelWeights};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue};

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerOneBlockVjp {
    pub target_block: u32,
    pub rule: MuseGlimmerLensRule,
    pub n_tokens: usize,
    pub hidden_size: usize,
    /// Full cotangent at the selected block input, flattened `[T,H]`.
    pub input_cotangent: Vec<f32>,
    /// F32 replay versus production capture, whose attention KV path is F16.
    pub post_attention_replay_max_abs_error: f32,
    /// F32 replay versus production capture after the complete block.
    pub post_block_replay_max_abs_error: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MuseGlimmerAdjacentSelectedTokenFit {
    pub source_block: u32,
    pub target_block: u32,
    pub rule: MuseGlimmerLensRule,
    pub token_ids: Vec<u32>,
    pub n_valid_positions: usize,
    pub hidden_size: usize,
    /// Token-major fitted directions, flattened `[K,H]`.
    pub values: Vec<f32>,
    pub post_attention_replay_max_abs_error: f32,
    pub post_block_replay_max_abs_error: f32,
}

impl MuseGlimmerAdjacentSelectedTokenFit {
    pub fn token_values(&self, slot: usize) -> Option<&[f32]> {
        let start = slot.checked_mul(self.hidden_size)?;
        self.values.get(start..start.checked_add(self.hidden_size)?)
    }
}

pub(crate) fn muse_glimmer_fit_adjacent_full_attention_selected_tokens(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    covectors: &MuseGlimmerSelectedTokenCovectors,
    skip_first: usize,
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerAdjacentSelectedTokenFit, MuseGlimmerLensError> {
    let target_block = capture.target_block();
    let source_block = target_block.checked_sub(1).ok_or_else(|| {
        MuseGlimmerLensError::Invalid("adjacent fit requires a nonzero target block".into())
    })?;
    if covectors.token_ids().is_empty()
        || covectors.token_ids().len() > MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS
    {
        return invalid(format!(
            "adjacent fit selected-token count must be in 1..={MUSE_GLIMMER_LENS_MAX_SELECTED_TOKENS}, got {}",
            covectors.token_ids().len()
        ));
    }
    if covectors.hidden_size() != capture.hidden_size() {
        return invalid(format!(
            "covector hidden size {} differs from capture hidden size {}",
            covectors.hidden_size(),
            capture.hidden_size()
        ));
    }
    let valid_positions = adjacent_fit_position_range(capture.n_tokens(), skip_first)?;
    let n_valid_positions = valid_positions.len();
    let positions = valid_positions.clone().collect::<Vec<_>>();
    let mut values = Vec::with_capacity(covectors.token_ids().len() * capture.hidden_size());
    let mut post_attention_replay_max_abs_error = 0.0_f32;
    let mut post_block_replay_max_abs_error = 0.0_f32;
    for slot in 0..covectors.token_ids().len() {
        let covector = covectors.token_values(slot).ok_or_else(|| {
            MuseGlimmerLensError::Invalid(format!("selected-token covector slot {slot} is missing"))
        })?;
        let target_cotangent = muse_glimmer_positioned_target_cotangent(
            covector,
            capture.n_tokens(),
            capture.hidden_size(),
            &positions,
        )?;
        let vjp = muse_glimmer_one_full_attention_block_vjp(
            ctx,
            weights,
            capture,
            &target_cotangent,
            rule,
        )?;
        values.extend(mean_reduce_position_rows(
            &vjp.input_cotangent,
            capture.n_tokens(),
            capture.hidden_size(),
            valid_positions.clone(),
        )?);
        post_attention_replay_max_abs_error =
            post_attention_replay_max_abs_error.max(vjp.post_attention_replay_max_abs_error);
        post_block_replay_max_abs_error =
            post_block_replay_max_abs_error.max(vjp.post_block_replay_max_abs_error);
    }
    require_finite("adjacent selected-token fit", &values)?;
    Ok(MuseGlimmerAdjacentSelectedTokenFit {
        source_block,
        target_block,
        rule,
        token_ids: covectors.token_ids().to_vec(),
        n_valid_positions,
        hidden_size: capture.hidden_size(),
        values,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
    })
}

fn adjacent_fit_position_range(
    n_tokens: usize,
    skip_first: usize,
) -> Result<std::ops::Range<usize>, MuseGlimmerLensError> {
    let final_position = n_tokens.checked_sub(1).ok_or_else(|| {
        MuseGlimmerLensError::Invalid("adjacent fit requires at least two prompt tokens".into())
    })?;
    if skip_first >= final_position {
        return invalid(format!(
            "skip_first {skip_first} leaves no positions before final prompt position {final_position}"
        ));
    }
    Ok(skip_first..final_position)
}

fn mean_reduce_position_rows(
    values: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    positions: std::ops::Range<usize>,
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    validate_len(
        "source cotangent bank",
        values,
        checked_mul(n_tokens, hidden_size, "source cotangent elements")?,
    )?;
    if positions.is_empty() || positions.end > n_tokens {
        return invalid("source reduction positions are empty or out of range");
    }
    let count = positions.len();
    let mut reduced = vec![0.0_f32; hidden_size];
    for position in positions {
        let row = &values[position * hidden_size..(position + 1) * hidden_size];
        for (destination, &value) in reduced.iter_mut().zip(row) {
            *destination += value;
        }
    }
    let scale = (count as f32).recip();
    for value in &mut reduced {
        *value *= scale;
    }
    require_finite("mean-reduced source cotangent", &reduced)?;
    Ok(reduced)
}

/// Place one `[H]` score covector at selected rows of a zeroed `[T,H]` bank.
pub fn muse_glimmer_positioned_target_cotangent(
    covector: &[f32],
    n_tokens: usize,
    hidden_size: usize,
    positions: &[usize],
) -> Result<Vec<f32>, MuseGlimmerLensError> {
    if hidden_size == 0 || covector.len() != hidden_size {
        return invalid(format!(
            "target covector length {} differs from nonzero hidden size {hidden_size}",
            covector.len()
        ));
    }
    if n_tokens == 0 || n_tokens > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS || positions.is_empty() {
        return invalid(format!(
            "target bank requires 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS} tokens and at least one position"
        ));
    }
    require_finite("target covector", covector)?;
    let mut values = vec![0.0_f32; checked_mul(n_tokens, hidden_size, "target bank elements")?];
    for (slot, &position) in positions.iter().enumerate() {
        if position >= n_tokens {
            return invalid(format!(
                "target position {position} is outside token count {n_tokens}"
            ));
        }
        if positions[..slot].contains(&position) {
            return invalid(format!("target position {position} occurs more than once"));
        }
        let start = position * hidden_size;
        values[start..start + hidden_size].copy_from_slice(covector);
    }
    Ok(values)
}

#[derive(Clone, Copy)]
struct MuseAttentionGeometry {
    hidden: usize,
    feed_forward: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    query: usize,
    kv: usize,
    rope_theta: f32,
}

impl MuseAttentionGeometry {
    fn from_weights(
        weights: &MuseGlimmerMetalModelWeights<'_>,
    ) -> Result<Self, MuseGlimmerLensError> {
        let config = weights.config;
        let geometry = Self {
            hidden: config.hidden_size as usize,
            feed_forward: config.feed_forward_size as usize,
            q_heads: config.query_head_count as usize,
            kv_heads: config.kv_head_count as usize,
            head_dim: config.key_head_dim as usize,
            query: usize::try_from(config.query_width()?)
                .map_err(|_| MuseGlimmerLensError::Invalid("query width exceeds usize".into()))?,
            kv: usize::try_from(config.kv_width()?)
                .map_err(|_| MuseGlimmerLensError::Invalid("KV width exceeds usize".into()))?,
            rope_theta: config.rope_theta,
        };
        if geometry.hidden == 0
            || geometry.feed_forward == 0
            || geometry.q_heads == 0
            || geometry.kv_heads == 0
            || !geometry.q_heads.is_multiple_of(geometry.kv_heads)
            || geometry.query != geometry.q_heads * geometry.head_dim
            || geometry.kv != geometry.kv_heads * geometry.head_dim
            || !geometry.rope_theta.is_finite()
            || geometry.rope_theta <= 0.0
        {
            return invalid("inconsistent Muse attention geometry");
        }
        Ok(geometry)
    }
}

struct MuseAttentionForward {
    attention_output: Vec<f32>,
    gated_output: Vec<f32>,
}

struct MuseAttentionVjp {
    grad_q: Vec<f32>,
    grad_k: Vec<f32>,
    grad_v: Vec<f32>,
    grad_gate: Vec<f32>,
}

fn adjacent_pair_rope_rows_in_place(
    values: &mut [f32],
    n_tokens: usize,
    n_heads: usize,
    head_dim: usize,
    theta: f32,
    inverse: bool,
) -> Result<(), MuseGlimmerLensError> {
    if n_tokens == 0
        || n_heads == 0
        || head_dim == 0
        || !head_dim.is_multiple_of(2)
        || !theta.is_finite()
        || theta <= 0.0
    {
        return invalid(
            "adjacent-pair RoPE requires nonzero rows/heads, even head_dim, and positive finite theta",
        );
    }
    validate_len(
        "adjacent-pair RoPE rows",
        values,
        checked_mul(
            checked_mul(n_tokens, n_heads, "RoPE token-head count")?,
            head_dim,
            "RoPE element count",
        )?,
    )?;
    for token in 0..n_tokens {
        for head in 0..n_heads {
            let base = (token * n_heads + head) * head_dim;
            for pair in 0..head_dim / 2 {
                let relative = 2 * pair;
                let angle = token as f32 * theta.powf(-(relative as f32) / head_dim as f32);
                let (mut sine, cosine) = angle.sin_cos();
                if inverse {
                    sine = -sine;
                }
                let first = values[base + relative];
                let second = values[base + relative + 1];
                values[base + relative] = first * cosine - second * sine;
                values[base + relative + 1] = first * sine + second * cosine;
            }
        }
    }
    require_finite("adjacent-pair RoPE output", values)
}

#[allow(clippy::needless_range_loop)]
fn cpu_causal_gqa_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
) -> Result<MuseAttentionForward, MuseGlimmerLensError> {
    let q_total = checked_mul(n_tokens, geometry.query, "Q elements")?;
    let kv_total = checked_mul(n_tokens, geometry.kv, "KV elements")?;
    validate_len("attention Q", q, q_total)?;
    validate_len("attention K", k, kv_total)?;
    validate_len("attention V", v, kv_total)?;
    validate_len("attention gate", gate, q_total)?;
    require_finite("attention Q", q)?;
    require_finite("attention K", k)?;
    require_finite("attention V", v)?;
    require_finite("attention gate", gate)?;

    let group = geometry.q_heads / geometry.kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut attention_output = vec![0.0_f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.q_heads + q_head) * geometry.head_dim;
            let mut probabilities = vec![0.0_f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0_f32;
                for dim in 0..geometry.head_dim {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                probabilities[key_token] = score * scale;
            }
            softmax_in_place(&mut probabilities);
            for key_token in 0..=token {
                let v_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    attention_output[q_base + dim] += probabilities[key_token] * v[v_base + dim];
                }
            }
        }
    }
    let gated_output = attention_output
        .iter()
        .zip(gate)
        .map(|(&attention, &gate)| attention * sigmoid(gate))
        .collect::<Vec<_>>();
    require_finite("attention output", &attention_output)?;
    require_finite("gated attention output", &gated_output)?;
    Ok(MuseAttentionForward {
        attention_output,
        gated_output,
    })
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
fn cpu_causal_gqa_vjp(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    grad_gated: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
    forward: &MuseAttentionForward,
) -> Result<MuseAttentionVjp, MuseGlimmerLensError> {
    let q_total = checked_mul(n_tokens, geometry.query, "Q elements")?;
    let kv_total = checked_mul(n_tokens, geometry.kv, "KV elements")?;
    validate_len("attention cotangent", grad_gated, q_total)?;
    validate_len("attention forward", &forward.attention_output, q_total)?;
    let group = geometry.q_heads / geometry.kv_heads;
    let scale = (geometry.head_dim as f32).sqrt().recip();
    let mut grad_q = vec![0.0_f32; q_total];
    let mut grad_k = vec![0.0_f32; kv_total];
    let mut grad_v = vec![0.0_f32; kv_total];
    let mut grad_gate = vec![0.0_f32; q_total];
    for token in 0..n_tokens {
        for q_head in 0..geometry.q_heads {
            let kv_head = q_head / group;
            let q_base = (token * geometry.q_heads + q_head) * geometry.head_dim;
            let mut grad_attention = vec![0.0_f32; geometry.head_dim];
            for dim in 0..geometry.head_dim {
                let gate_value = sigmoid(gate[q_base + dim]);
                grad_attention[dim] = grad_gated[q_base + dim] * gate_value;
                grad_gate[q_base + dim] = grad_gated[q_base + dim]
                    * forward.attention_output[q_base + dim]
                    * gate_value
                    * (1.0 - gate_value);
            }
            let mut probabilities = vec![0.0_f32; token + 1];
            for key_token in 0..=token {
                let k_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                let mut score = 0.0_f32;
                for dim in 0..geometry.head_dim {
                    score += q[q_base + dim] * k[k_base + dim];
                }
                probabilities[key_token] = score * scale;
            }
            softmax_in_place(&mut probabilities);
            let mut grad_probability = vec![0.0_f32; token + 1];
            for key_token in 0..=token {
                let v_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    grad_probability[key_token] += grad_attention[dim] * v[v_base + dim];
                    grad_v[v_base + dim] += probabilities[key_token] * grad_attention[dim];
                }
            }
            let probability_dot = probabilities
                .iter()
                .zip(&grad_probability)
                .map(|(&probability, &gradient)| probability * gradient)
                .sum::<f32>();
            for key_token in 0..=token {
                let grad_score =
                    probabilities[key_token] * (grad_probability[key_token] - probability_dot);
                let k_base = (key_token * geometry.kv_heads + kv_head) * geometry.head_dim;
                for dim in 0..geometry.head_dim {
                    grad_q[q_base + dim] += scale * grad_score * k[k_base + dim];
                    grad_k[k_base + dim] += scale * grad_score * q[q_base + dim];
                }
            }
        }
    }
    for (name, values) in [
        ("attention grad Q", grad_q.as_slice()),
        ("attention grad K", grad_k.as_slice()),
        ("attention grad V", grad_v.as_slice()),
        ("attention grad gate", grad_gate.as_slice()),
    ] {
        require_finite(name, values)?;
    }
    Ok(MuseAttentionVjp {
        grad_q,
        grad_k,
        grad_v,
        grad_gate,
    })
}

struct ReplayTensors {
    input: MetalTensor,
    attention_normed: MetalTensor,
    q_raw: MetalTensor,
    q: MetalTensor,
    k_raw: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    attention_gate: MetalTensor,
    gated_attention: MetalTensor,
    attention_branch_raw: MetalTensor,
    attention_branch: MetalTensor,
    post_attention: MetalTensor,
    ffn_normed: MetalTensor,
    ffn_gate_raw: MetalTensor,
    ffn_up: MetalTensor,
    ffn_inner: MetalTensor,
    ffn_branch_raw: MetalTensor,
    ffn_branch: MetalTensor,
    post_block: MetalTensor,
}

impl ReplayTensors {
    fn new(
        ctx: &MetalContext,
        input: &[f32],
        n_tokens: usize,
        geometry: MuseAttentionGeometry,
    ) -> Result<Self, MuseGlimmerLensError> {
        let hidden_shape = row_shape(geometry.hidden, n_tokens)?;
        let query_shape = row_shape(geometry.query, n_tokens)?;
        let kv_shape = row_shape(geometry.kv, n_tokens)?;
        let ffn_shape = row_shape(geometry.feed_forward, n_tokens)?;
        Ok(Self {
            input: from_f32(ctx, input, hidden_shape.clone())?,
            attention_normed: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            q_raw: MetalTensor::zeros_f32(ctx, query_shape.clone())?,
            q: MetalTensor::zeros_f32(ctx, query_shape.clone())?,
            k_raw: MetalTensor::zeros_f32(ctx, kv_shape.clone())?,
            k: MetalTensor::zeros_f32(ctx, kv_shape.clone())?,
            v: MetalTensor::zeros_f32(ctx, kv_shape)?,
            attention_gate: MetalTensor::zeros_f32(ctx, query_shape.clone())?,
            gated_attention: MetalTensor::zeros_f32(ctx, query_shape)?,
            attention_branch_raw: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            attention_branch: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            post_attention: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            ffn_normed: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            ffn_gate_raw: MetalTensor::zeros_f32(ctx, ffn_shape.clone())?,
            ffn_up: MetalTensor::zeros_f32(ctx, ffn_shape.clone())?,
            ffn_inner: MetalTensor::zeros_f32(ctx, ffn_shape)?,
            ffn_branch_raw: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            ffn_branch: MetalTensor::zeros_f32(ctx, hidden_shape.clone())?,
            post_block: MetalTensor::zeros_f32(ctx, hidden_shape)?,
        })
    }
}

struct ReplayState {
    tensors: ReplayTensors,
    q: Vec<f32>,
    k: Vec<f32>,
    v: Vec<f32>,
    attention_gate: Vec<f32>,
    attention: MuseAttentionForward,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct MuseGlimmerOneBlockReplay {
    pub post_attention_residuals: Vec<f32>,
    pub post_block_residuals: Vec<f32>,
}

/// Test seam for evaluating the smooth F32 model-level block graph on an
/// arbitrary `[T,H]` input. It deliberately does not round K/V through the
/// production F16 cache.
#[cfg(test)]
pub(crate) fn muse_glimmer_replay_one_full_attention_block(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
    input: &[f32],
    n_tokens: usize,
) -> Result<MuseGlimmerOneBlockReplay, MuseGlimmerLensError> {
    require_full_attention_block(weights, target_block)?;
    muse_glimmer_replay_one_attention_block(ctx, weights, target_block, input, n_tokens)
}

#[cfg(test)]
pub(crate) fn muse_glimmer_replay_one_attention_block(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
    input: &[f32],
    n_tokens: usize,
) -> Result<MuseGlimmerOneBlockReplay, MuseGlimmerLensError> {
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    let layer = validate_replay_request(weights, target_block, input, n_tokens, geometry)?;
    let state = replay_state(ctx, weights, layer, input, n_tokens, geometry)?;
    replay_readback(&state)
}

fn replay_readback(state: &ReplayState) -> Result<MuseGlimmerOneBlockReplay, MuseGlimmerLensError> {
    let post_attention_residuals = read_f32(&state.tensors.post_attention);
    let post_block_residuals = read_f32(&state.tensors.post_block);
    require_finite(
        "replayed post-attention residual",
        &post_attention_residuals,
    )?;
    require_finite("replayed post-block residual", &post_block_residuals)?;
    Ok(MuseGlimmerOneBlockReplay {
        post_attention_residuals,
        post_block_residuals,
    })
}

fn replay_state(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    input: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
) -> Result<ReplayState, MuseGlimmerLensError> {
    let tensors = ReplayTensors::new(ctx, input, n_tokens, geometry)?;
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.input,
            layer.attention_norm,
            &tensors.attention_normed,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
        )?;
        for token in 0..n_tokens {
            let input = row_view(&tensors.attention_normed, token, geometry.hidden);
            for (weight, output, width) in [
                (layer.attention_query, &tensors.q_raw, geometry.query),
                (layer.attention_key, &tensors.k_raw, geometry.kv),
                (layer.attention_value, &tensors.v, geometry.kv),
                (
                    layer.attention_gate,
                    &tensors.attention_gate,
                    geometry.query,
                ),
            ] {
                encode_mat_vec_dispatch(
                    ctx,
                    encoder,
                    weight,
                    &input,
                    &row_view(output, token, width),
                    geometry.hidden,
                    width,
                )?;
            }
        }
        encode_rms_norm_batched_f32(
            ctx,
            encoder,
            &tensors.q_raw,
            layer.query_norm,
            &tensors.q,
            n_tokens * geometry.q_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
        )?;
        encode_rms_norm_batched_f32(
            ctx,
            encoder,
            &tensors.k_raw,
            layer.key_norm,
            &tensors.k,
            n_tokens * geometry.kv_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
        )?;
        Ok(())
    })?;

    let mut q = read_f32(&tensors.q);
    let mut k = read_f32(&tensors.k);
    let v = read_f32(&tensors.v);
    let attention_gate = read_f32(&tensors.attention_gate);
    if layer.sliding_attention {
        adjacent_pair_rope_rows_in_place(
            &mut q,
            n_tokens,
            geometry.q_heads,
            geometry.head_dim,
            geometry.rope_theta,
            false,
        )?;
        adjacent_pair_rope_rows_in_place(
            &mut k,
            n_tokens,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.rope_theta,
            false,
        )?;
    }
    let attention = cpu_causal_gqa_forward(&q, &k, &v, &attention_gate, n_tokens, geometry)?;
    write_f32(&tensors.gated_attention, &attention.gated_output)?;

    run_command(ctx, |encoder| {
        for token in 0..n_tokens {
            encode_mat_vec_dispatch(
                ctx,
                encoder,
                layer.attention_output,
                &row_view(&tensors.gated_attention, token, geometry.query),
                &row_view(&tensors.attention_branch_raw, token, geometry.hidden),
                geometry.query,
                geometry.hidden,
            )?;
        }
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.attention_branch_raw,
            layer.post_attention_norm,
            &tensors.attention_branch,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &tensors.input,
            &tensors.attention_branch,
            &tensors.post_attention,
        )?;
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.post_attention,
            layer.feed_forward_norm,
            &tensors.ffn_normed,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
        )?;
        for token in 0..n_tokens {
            let input = row_view(&tensors.ffn_normed, token, geometry.hidden);
            for (weight, output) in [
                (layer.feed_forward_gate, &tensors.ffn_gate_raw),
                (layer.feed_forward_up, &tensors.ffn_up),
            ] {
                encode_mat_vec_dispatch(
                    ctx,
                    encoder,
                    weight,
                    &input,
                    &row_view(output, token, geometry.feed_forward),
                    geometry.hidden,
                    geometry.feed_forward,
                )?;
            }
        }
        encode_silu_mul_f32(
            ctx,
            encoder,
            &tensors.ffn_gate_raw,
            &tensors.ffn_up,
            &tensors.ffn_inner,
        )?;
        for token in 0..n_tokens {
            encode_mat_vec_dispatch(
                ctx,
                encoder,
                layer.feed_forward_down,
                &row_view(&tensors.ffn_inner, token, geometry.feed_forward),
                &row_view(&tensors.ffn_branch_raw, token, geometry.hidden),
                geometry.feed_forward,
                geometry.hidden,
            )?;
        }
        encode_rms_norm_mul_rows_f32(
            ctx,
            encoder,
            &tensors.ffn_branch_raw,
            layer.post_feed_forward_norm,
            &tensors.ffn_branch,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &tensors.post_attention,
            &tensors.ffn_branch,
            &tensors.post_block,
        )?;
        Ok(())
    })?;
    Ok(ReplayState {
        tensors,
        q,
        k,
        v,
        attention_gate,
        attention,
    })
}

pub(crate) fn muse_glimmer_one_full_attention_block_vjp(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangent: &[f32],
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerLensError> {
    require_full_attention_block(weights, capture.target_block())?;
    muse_glimmer_one_attention_block_vjp(ctx, weights, capture, target_cotangent, rule)
}

pub(crate) fn muse_glimmer_one_attention_block_vjp(
    ctx: &MetalContext,
    weights: &MuseGlimmerMetalModelWeights<'_>,
    capture: &MuseGlimmerLensCapture,
    target_cotangent: &[f32],
    rule: MuseGlimmerLensRule,
) -> Result<MuseGlimmerOneBlockVjp, MuseGlimmerLensError> {
    let geometry = MuseAttentionGeometry::from_weights(weights)?;
    let target_block = capture.target_block();
    let layer = validate_replay_request(
        weights,
        target_block,
        capture.input_residuals(),
        capture.n_tokens(),
        geometry,
    )?;
    validate_request(capture, target_cotangent, geometry.hidden)?;
    let n_tokens = capture.n_tokens();
    let state = replay_state(
        ctx,
        weights,
        layer,
        capture.input_residuals(),
        n_tokens,
        geometry,
    )?;
    let replay = replay_readback(&state)?;
    let ReplayState {
        tensors,
        q,
        k,
        v,
        attention_gate,
        attention,
    } = state;
    let post_attention_replay_max_abs_error = max_abs_difference(
        &replay.post_attention_residuals,
        capture.post_attention_residuals(),
    )?;
    let post_block_replay_max_abs_error =
        max_abs_difference(&replay.post_block_residuals, capture.post_block_residuals())?;

    let hidden_shape = row_shape(geometry.hidden, n_tokens)?;
    let ffn_shape = row_shape(geometry.feed_forward, n_tokens)?;
    let query_shape = row_shape(geometry.query, n_tokens)?;
    let kv_shape = row_shape(geometry.kv, n_tokens)?;
    let grad_post_block = from_f32(ctx, target_cotangent, hidden_shape.clone())?;
    let grad_ffn_raw = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_ffn_inner = MetalTensor::zeros_f32(ctx, ffn_shape.clone())?;
    let grad_ffn_gate = MetalTensor::zeros_f32(ctx, ffn_shape.clone())?;
    let grad_ffn_up = MetalTensor::zeros_f32(ctx, ffn_shape)?;
    let grad_ffn_norm_gate = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_ffn_norm_up = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_ffn_norm = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_post_attention_branch = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_post_attention = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.ffn_branch_raw,
            layer.post_feed_forward_norm,
            &grad_post_block,
            &grad_ffn_raw,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::FeedForwardBranchPostNorm),
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.feed_forward_down,
            &grad_ffn_raw,
            &grad_ffn_inner,
            geometry.feed_forward,
            geometry.hidden,
            n_tokens,
        )?;
        encode_silu_mul_vjp_f32(
            ctx,
            encoder,
            &tensors.ffn_gate_raw,
            &tensors.ffn_up,
            &grad_ffn_inner,
            &grad_ffn_gate,
            &grad_ffn_up,
            n_tokens,
            geometry.feed_forward,
            rule.swiglu_rule(),
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.feed_forward_gate,
            &grad_ffn_gate,
            &grad_ffn_norm_gate,
            geometry.hidden,
            geometry.feed_forward,
            n_tokens,
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.feed_forward_up,
            &grad_ffn_up,
            &grad_ffn_norm_up,
            geometry.hidden,
            geometry.feed_forward,
            n_tokens,
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_ffn_norm_gate,
            &grad_ffn_norm_up,
            &grad_ffn_norm,
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.post_attention,
            layer.feed_forward_norm,
            &grad_ffn_norm,
            &grad_post_attention_branch,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreFeedForward),
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_post_block,
            &grad_post_attention_branch,
            &grad_post_attention,
        )?;
        Ok(())
    })?;

    let grad_attention_raw = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_gated = MetalTensor::zeros_f32(ctx, query_shape.clone())?;
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.attention_branch_raw,
            layer.post_attention_norm,
            &grad_post_attention,
            &grad_attention_raw,
            n_tokens,
            geometry.hidden,
            weights.config.post_norm_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionBranchPostNorm),
        )?;
        encode_frozen_linear_vjp_f32(
            ctx,
            encoder,
            layer.attention_output,
            &grad_attention_raw,
            &grad_gated,
            geometry.query,
            geometry.hidden,
            n_tokens,
        )?;
        Ok(())
    })?;
    let mut attention_vjp = cpu_causal_gqa_vjp(
        &q,
        &k,
        &v,
        &attention_gate,
        &read_f32(&grad_gated),
        n_tokens,
        geometry,
        &attention,
    )?;
    if layer.sliding_attention {
        adjacent_pair_rope_rows_in_place(
            &mut attention_vjp.grad_q,
            n_tokens,
            geometry.q_heads,
            geometry.head_dim,
            geometry.rope_theta,
            true,
        )?;
        adjacent_pair_rope_rows_in_place(
            &mut attention_vjp.grad_k,
            n_tokens,
            geometry.kv_heads,
            geometry.head_dim,
            geometry.rope_theta,
            true,
        )?;
    }

    let grad_q = from_f32(ctx, &attention_vjp.grad_q, query_shape.clone())?;
    let grad_k = from_f32(ctx, &attention_vjp.grad_k, kv_shape.clone())?;
    let grad_v = from_f32(ctx, &attention_vjp.grad_v, kv_shape)?;
    let grad_gate = from_f32(ctx, &attention_vjp.grad_gate, query_shape)?;
    let grad_q_raw = MetalTensor::zeros_f32(ctx, row_shape(geometry.query, n_tokens)?)?;
    let grad_k_raw = MetalTensor::zeros_f32(ctx, row_shape(geometry.kv, n_tokens)?)?;
    let grad_norm_q = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_k = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_v = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_gate = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_qk = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm_qkv = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_norm = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_input_branch = MetalTensor::zeros_f32(ctx, hidden_shape.clone())?;
    let grad_input = MetalTensor::zeros_f32(ctx, hidden_shape)?;
    let q_raw_heads = tensors.q_raw.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.q_heads)?,
    );
    let grad_q_heads = grad_q.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.q_heads)?,
    );
    let grad_q_raw_heads = grad_q_raw.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.q_heads)?,
    );
    let k_raw_heads = tensors.k_raw.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.kv_heads)?,
    );
    let grad_k_heads = grad_k.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.kv_heads)?,
    );
    let grad_k_raw_heads = grad_k_raw.view_subrange(
        0,
        row_shape(geometry.head_dim, n_tokens * geometry.kv_heads)?,
    );
    run_command(ctx, |encoder| {
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &q_raw_heads,
            layer.query_norm,
            &grad_q_heads,
            &grad_q_raw_heads,
            n_tokens * geometry.q_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
        )?;
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &k_raw_heads,
            layer.key_norm,
            &grad_k_heads,
            &grad_k_raw_heads,
            n_tokens * geometry.kv_heads,
            geometry.head_dim,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionKey),
        )?;
        for (weight, grad_output, grad_hidden, n_out) in [
            (
                layer.attention_query,
                &grad_q_raw,
                &grad_norm_q,
                geometry.query,
            ),
            (layer.attention_key, &grad_k_raw, &grad_norm_k, geometry.kv),
            (layer.attention_value, &grad_v, &grad_norm_v, geometry.kv),
            (
                layer.attention_gate,
                &grad_gate,
                &grad_norm_gate,
                geometry.query,
            ),
        ] {
            encode_frozen_linear_vjp_f32(
                ctx,
                encoder,
                weight,
                grad_output,
                grad_hidden,
                geometry.hidden,
                n_out,
                n_tokens,
            )?;
        }
        encode_add_f32(ctx, encoder, &grad_norm_q, &grad_norm_k, &grad_norm_qk)?;
        encode_add_f32(ctx, encoder, &grad_norm_qk, &grad_norm_v, &grad_norm_qkv)?;
        encode_add_f32(ctx, encoder, &grad_norm_qkv, &grad_norm_gate, &grad_norm)?;
        encode_rms_norm_mul_vjp_rows_f32(
            ctx,
            encoder,
            &tensors.input,
            layer.attention_norm,
            &grad_norm,
            &grad_input_branch,
            n_tokens,
            geometry.hidden,
            weights.config.rms_epsilon,
            rule.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
        )?;
        encode_add_f32(
            ctx,
            encoder,
            &grad_post_attention,
            &grad_input_branch,
            &grad_input,
        )?;
        Ok(())
    })?;
    let input_cotangent = read_f32(&grad_input);
    require_finite("input cotangent", &input_cotangent)?;
    Ok(MuseGlimmerOneBlockVjp {
        target_block,
        rule,
        n_tokens,
        hidden_size: geometry.hidden,
        input_cotangent,
        post_attention_replay_max_abs_error,
        post_block_replay_max_abs_error,
    })
}

fn validate_replay_request<'a>(
    weights: &'a MuseGlimmerMetalModelWeights<'a>,
    target_block: u32,
    input: &[f32],
    n_tokens: usize,
    geometry: MuseAttentionGeometry,
) -> Result<&'a MuseGlimmerMetalLayerWeights<'a>, MuseGlimmerLensError> {
    if target_block == 0 {
        return invalid("one-block replay requires a nonzero target block");
    }
    let layer = weights.layers.get(target_block as usize).ok_or_else(|| {
        MuseGlimmerLensError::Invalid(format!("target block {target_block} is out of range"))
    })?;
    if n_tokens == 0 || n_tokens > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
        return invalid(format!(
            "replay token count must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got {n_tokens}"
        ));
    }
    validate_len(
        "replay input",
        input,
        checked_mul(n_tokens, geometry.hidden, "replay input elements")?,
    )?;
    require_finite("replay input", input)?;
    validate_layer_shapes(layer, geometry)?;
    Ok(layer)
}

fn validate_request(
    capture: &MuseGlimmerLensCapture,
    target_cotangent: &[f32],
    hidden_size: usize,
) -> Result<(), MuseGlimmerLensError> {
    if capture.target_block() == 0 {
        return invalid("one-block VJP requires a nonzero target block");
    }
    if capture.n_tokens() == 0 || capture.n_tokens() > MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS {
        return invalid(format!(
            "capture token count must be in 1..={MUSE_GLIMMER_LENS_MAX_PROMPT_TOKENS}, got {}",
            capture.n_tokens()
        ));
    }
    if capture.hidden_size() != hidden_size {
        return invalid(format!(
            "capture hidden size {} differs from model hidden size {hidden_size}",
            capture.hidden_size()
        ));
    }
    let expected = checked_mul(capture.n_tokens(), hidden_size, "capture elements")?;
    for (name, values) in [
        ("input residuals", capture.input_residuals()),
        (
            "post-attention residuals",
            capture.post_attention_residuals(),
        ),
        ("post-block residuals", capture.post_block_residuals()),
        ("target cotangent", target_cotangent),
    ] {
        validate_len(name, values, expected)?;
        require_finite(name, values)?;
    }
    Ok(())
}

fn require_full_attention_block(
    weights: &MuseGlimmerMetalModelWeights<'_>,
    target_block: u32,
) -> Result<(), MuseGlimmerLensError> {
    let layer = weights.layers.get(target_block as usize).ok_or_else(|| {
        MuseGlimmerLensError::Invalid(format!("target block {target_block} is out of range"))
    })?;
    if layer.sliding_attention {
        return invalid(format!(
            "target block {target_block} uses sliding attention; this API requires full attention"
        ));
    }
    Ok(())
}

fn validate_layer_shapes(
    layer: &MuseGlimmerMetalLayerWeights<'_>,
    geometry: MuseAttentionGeometry,
) -> Result<(), MuseGlimmerLensError> {
    for (name, tensor, shape) in [
        (
            "attention query",
            layer.attention_query,
            [geometry.hidden, geometry.query],
        ),
        (
            "attention key",
            layer.attention_key,
            [geometry.hidden, geometry.kv],
        ),
        (
            "attention value",
            layer.attention_value,
            [geometry.hidden, geometry.kv],
        ),
        (
            "attention gate",
            layer.attention_gate,
            [geometry.hidden, geometry.query],
        ),
        (
            "attention output",
            layer.attention_output,
            [geometry.query, geometry.hidden],
        ),
        (
            "FFN gate",
            layer.feed_forward_gate,
            [geometry.hidden, geometry.feed_forward],
        ),
        (
            "FFN up",
            layer.feed_forward_up,
            [geometry.hidden, geometry.feed_forward],
        ),
        (
            "FFN down",
            layer.feed_forward_down,
            [geometry.feed_forward, geometry.hidden],
        ),
    ] {
        if tensor.shape.as_slice() != [shape[0] as u64, shape[1] as u64] {
            return invalid(format!(
                "{name} shape {:?} differs from {shape:?}",
                tensor.shape
            ));
        }
    }
    Ok(())
}

fn run_command<F>(ctx: &MetalContext, encode: F) -> Result<(), MuseGlimmerLensError>
where
    F: FnOnce(&KernelEncoder) -> Result<(), MuseGlimmerLensError>,
{
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or_else(|| MuseGlimmerLensError::CommandBuffer("allocation failed".into()))?;
    let encoder = KernelEncoder::begin(&command);
    let result = encode(&encoder);
    encoder.end();
    result?;
    command.commit();
    command.waitUntilCompleted();
    let status = command.status();
    let error = command.error().map(|error| error.to_string());
    if status != MTLCommandBufferStatus::Completed || error.is_some() {
        return Err(MuseGlimmerLensError::CommandBuffer(format!(
            "status={status:?}, error={error:?}"
        )));
    }
    Ok(())
}

fn row_shape(width: usize, rows: usize) -> Result<Vec<u64>, MuseGlimmerLensError> {
    checked_mul(width, rows, "row elements")?;
    Ok(vec![width as u64, rows as u64])
}

fn row_view(tensor: &MetalTensor, row: usize, width: usize) -> MetalTensor {
    tensor.view_subrange((row * width) as u64, vec![width as u64])
}

fn from_f32(
    ctx: &MetalContext,
    values: &[f32],
    shape: Vec<u64>,
) -> Result<MetalTensor, MuseGlimmerLensError> {
    Ok(MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(values),
        shape,
        GgmlType::F32,
    )?)
}

fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let mut values = vec![0.0_f32; tensor.n_elements() as usize];
    unsafe {
        std::ptr::copy_nonoverlapping(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.as_mut_ptr(),
            values.len(),
        );
    }
    values
}

fn write_f32(tensor: &MetalTensor, values: &[f32]) -> Result<(), MuseGlimmerLensError> {
    validate_len("Metal tensor write", values, tensor.n_elements() as usize)?;
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            values.len(),
        );
    }
    Ok(())
}

fn max_abs_difference(left: &[f32], right: &[f32]) -> Result<f32, MuseGlimmerLensError> {
    validate_len("replay comparison", left, right.len())?;
    let value = left
        .iter()
        .zip(right)
        .map(|(&left, &right)| (left - right).abs())
        .fold(0.0_f32, f32::max);
    if !value.is_finite() {
        return invalid("non-finite replay error");
    }
    Ok(value)
}

fn softmax_in_place(values: &mut [f32]) {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0_f32;
    for value in values.iter_mut() {
        *value = (*value - max).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

fn sigmoid(value: f32) -> f32 {
    (1.0 + (-value).exp()).recip()
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, MuseGlimmerLensError> {
    left.checked_mul(right)
        .ok_or_else(|| MuseGlimmerLensError::Invalid(format!("{name} overflow")))
}

fn validate_len(name: &str, values: &[f32], expected: usize) -> Result<(), MuseGlimmerLensError> {
    if values.len() != expected {
        return invalid(format!(
            "{name} length {} differs from expected {expected}",
            values.len()
        ));
    }
    Ok(())
}

fn require_finite(name: &str, values: &[f32]) -> Result<(), MuseGlimmerLensError> {
    if let Some(index) = values.iter().position(|value| !value.is_finite()) {
        return invalid(format!(
            "{name} contains a non-finite value at index {index}"
        ));
    }
    Ok(())
}

fn invalid<T>(detail: impl Into<String>) -> Result<T, MuseGlimmerLensError> {
    Err(MuseGlimmerLensError::Invalid(detail.into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::GgufFile;
    use crate::metal::{
        rms_norm_mul_vjp_rows_f32_readback_for_test, silu_mul_vjp_f32_readback_for_test,
    };
    use crate::muse_glimmer_lens::muse_glimmer_selected_token_covectors;
    use crate::muse_glimmer_metal::encode_muse_glimmer_rope_adjacent_pair_in_place_f32;
    use crate::muse_glimmer_residency::{MuseGlimmerMetalWeightPlan, MuseGlimmerMetalWeights};
    use crate::muse_glimmer_text_session::{MuseGlimmerTextForward, MuseGlimmerTextSession};

    fn tiny_geometry() -> MuseAttentionGeometry {
        MuseAttentionGeometry {
            hidden: 4,
            feed_forward: 6,
            q_heads: 2,
            kv_heads: 1,
            head_dim: 2,
            query: 4,
            kv: 2,
            rope_theta: 500_000.0,
        }
    }

    fn dot(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right)
            .map(|(&left, &right)| left * right)
            .sum()
    }

    #[test]
    fn cpu_causal_gqa_vjp_matches_directional_finite_difference() {
        let geometry = tiny_geometry();
        let tokens = 3;
        let q = (0..tokens * geometry.query)
            .map(|index| index as f32 * 0.07 - 0.3)
            .collect::<Vec<_>>();
        let k = (0..tokens * geometry.kv)
            .map(|index| index as f32 * -0.09 + 0.2)
            .collect::<Vec<_>>();
        let v = (0..tokens * geometry.kv)
            .map(|index| index as f32 * 0.11 - 0.1)
            .collect::<Vec<_>>();
        let gate = (0..tokens * geometry.query)
            .map(|index| index as f32 * 0.05 - 0.2)
            .collect::<Vec<_>>();
        let cotangent = (0..tokens * geometry.query)
            .map(|index| (index as f32 * 0.31).sin())
            .collect::<Vec<_>>();
        let forward = cpu_causal_gqa_forward(&q, &k, &v, &gate, tokens, geometry).unwrap();
        let vjp =
            cpu_causal_gqa_vjp(&q, &k, &v, &gate, &cotangent, tokens, geometry, &forward).unwrap();
        let epsilon = 1e-3_f32;
        for (primal, gradient, label) in [
            (q.as_slice(), vjp.grad_q.as_slice(), "Q"),
            (k.as_slice(), vjp.grad_k.as_slice(), "K"),
            (v.as_slice(), vjp.grad_v.as_slice(), "V"),
            (gate.as_slice(), vjp.grad_gate.as_slice(), "gate"),
        ] {
            let direction = (0..primal.len())
                .map(|index| (index as f32 * 0.73 + 0.2).cos())
                .collect::<Vec<_>>();
            let plus = primal
                .iter()
                .zip(&direction)
                .map(|(&value, &direction)| value + epsilon * direction)
                .collect::<Vec<_>>();
            let minus = primal
                .iter()
                .zip(&direction)
                .map(|(&value, &direction)| value - epsilon * direction)
                .collect::<Vec<_>>();
            let evaluate = |replacement: &[f32]| {
                let (q_value, k_value, v_value, gate_value) = match label {
                    "Q" => (replacement, k.as_slice(), v.as_slice(), gate.as_slice()),
                    "K" => (q.as_slice(), replacement, v.as_slice(), gate.as_slice()),
                    "V" => (q.as_slice(), k.as_slice(), replacement, gate.as_slice()),
                    _ => (q.as_slice(), k.as_slice(), v.as_slice(), replacement),
                };
                dot(
                    &cpu_causal_gqa_forward(
                        q_value, k_value, v_value, gate_value, tokens, geometry,
                    )
                    .unwrap()
                    .gated_output,
                    &cotangent,
                )
            };
            let finite_difference = (evaluate(&plus) - evaluate(&minus)) / (2.0 * epsilon);
            let reverse = dot(gradient, &direction);
            assert!(
                (finite_difference - reverse).abs() < 3e-4,
                "{label}: finite difference {finite_difference}, reverse {reverse}"
            );
        }
    }

    #[test]
    fn adjacent_pair_rope_matches_production_kernel_and_inverse() {
        const TOKENS: usize = 3;
        const Q_HEADS: usize = 2;
        const KV_HEADS: usize = 1;
        const HEAD_DIM: usize = 4;
        const THETA: f32 = 10_000.0;
        let q_original = (0..TOKENS * Q_HEADS * HEAD_DIM)
            .map(|index| index as f32 * 0.07 - 0.5)
            .collect::<Vec<_>>();
        let k_original = (0..TOKENS * KV_HEADS * HEAD_DIM)
            .map(|index| index as f32 * -0.09 + 0.4)
            .collect::<Vec<_>>();
        let mut q_cpu = q_original.clone();
        let mut k_cpu = k_original.clone();
        adjacent_pair_rope_rows_in_place(&mut q_cpu, TOKENS, Q_HEADS, HEAD_DIM, THETA, false)
            .unwrap();
        adjacent_pair_rope_rows_in_place(&mut k_cpu, TOKENS, KV_HEADS, HEAD_DIM, THETA, false)
            .unwrap();

        let ctx = MetalContext::new().unwrap();
        let q_gpu = from_f32(
            &ctx,
            &q_original,
            row_shape(Q_HEADS * HEAD_DIM, TOKENS).unwrap(),
        )
        .unwrap();
        let k_gpu = from_f32(
            &ctx,
            &k_original,
            row_shape(KV_HEADS * HEAD_DIM, TOKENS).unwrap(),
        )
        .unwrap();
        run_command(&ctx, |encoder| {
            for token in 0..TOKENS {
                encode_muse_glimmer_rope_adjacent_pair_in_place_f32(
                    &ctx,
                    encoder,
                    &row_view(&q_gpu, token, Q_HEADS * HEAD_DIM),
                    &row_view(&k_gpu, token, KV_HEADS * HEAD_DIM),
                    Q_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    token as u32,
                    THETA,
                )?;
            }
            Ok(())
        })
        .unwrap();
        for (&cpu, gpu) in q_cpu.iter().zip(read_f32(&q_gpu)) {
            assert!((cpu - gpu).abs() < 2e-6);
        }
        for (&cpu, gpu) in k_cpu.iter().zip(read_f32(&k_gpu)) {
            assert!((cpu - gpu).abs() < 2e-6);
        }
        adjacent_pair_rope_rows_in_place(&mut q_cpu, TOKENS, Q_HEADS, HEAD_DIM, THETA, true)
            .unwrap();
        adjacent_pair_rope_rows_in_place(&mut k_cpu, TOKENS, KV_HEADS, HEAD_DIM, THETA, true)
            .unwrap();
        for (&actual, &expected) in q_cpu.iter().zip(&q_original) {
            assert!((actual - expected).abs() < 2e-6);
        }
        for (&actual, &expected) in k_cpu.iter().zip(&k_original) {
            assert!((actual - expected).abs() < 2e-6);
        }
    }

    #[test]
    fn r_rules_differ_from_j_at_residual_norm_and_swiglu_only() {
        assert_ne!(
            MuseGlimmerLensRule::J.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
            MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention)
        );
        assert_ne!(
            MuseGlimmerLensRule::J.swiglu_rule(),
            MuseGlimmerLensRule::R.swiglu_rule()
        );
        assert_eq!(
            MuseGlimmerLensRule::J.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery),
            MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::AttentionQuery)
        );
        assert_eq!(
            MuseGlimmerLensRule::J.attention_rule(),
            MuseGlimmerLensRule::R.attention_rule()
        );
    }

    #[test]
    fn r_kernels_match_detached_rms_and_identity_half_references() {
        let ctx = MetalContext::new().unwrap();
        let x = [1.5_f32, -0.5];
        let weight = [0.75_f32, 1.25];
        let incoming = [0.4_f32, -0.8];
        let epsilon = 1e-5_f32;
        let j_rms = rms_norm_mul_vjp_rows_f32_readback_for_test(
            &ctx,
            &x,
            &weight,
            &incoming,
            1,
            2,
            epsilon,
            MuseGlimmerLensRule::J.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
        )
        .unwrap();
        let r_rms = rms_norm_mul_vjp_rows_f32_readback_for_test(
            &ctx,
            &x,
            &weight,
            &incoming,
            1,
            2,
            epsilon,
            MuseGlimmerLensRule::R.rms_norm_rule(MuseGlimmerRmsNormSite::ResidualPreAttention),
        )
        .unwrap();
        let scale = ((x[0] * x[0] + x[1] * x[1]) / 2.0 + epsilon).sqrt().recip();
        let r_rms_reference = [
            incoming[0] * weight[0] * scale,
            incoming[1] * weight[1] * scale,
        ];
        for (&actual, &reference) in r_rms.iter().zip(&r_rms_reference) {
            assert!((actual - reference).abs() < 1e-6);
        }
        assert!(
            j_rms
                .iter()
                .zip(&r_rms)
                .any(|(&j, &r)| (j - r).abs() > 1e-3)
        );

        let gate = [0.7_f32, -1.1];
        let up = [1.3_f32, -0.4];
        let grad = [0.6_f32, 0.9];
        let (j_gate, j_up) = silu_mul_vjp_f32_readback_for_test(
            &ctx,
            &gate,
            &up,
            &grad,
            1,
            2,
            MuseGlimmerLensRule::J.swiglu_rule(),
        )
        .unwrap();
        let (r_gate, r_up) = silu_mul_vjp_f32_readback_for_test(
            &ctx,
            &gate,
            &up,
            &grad,
            1,
            2,
            MuseGlimmerLensRule::R.swiglu_rule(),
        )
        .unwrap();
        for index in 0..gate.len() {
            let sigmoid_gate = sigmoid(gate[index]);
            assert!((r_gate[index] - 0.5 * grad[index] * up[index] * sigmoid_gate).abs() < 1e-6);
            assert!((r_up[index] - 0.5 * grad[index] * gate[index] * sigmoid_gate).abs() < 1e-6);
        }
        assert!(
            j_gate
                .iter()
                .zip(&r_gate)
                .any(|(&j, &r)| (j - r).abs() > 1e-3)
        );
        assert!(j_up.iter().zip(&r_up).any(|(&j, &r)| (j - r).abs() > 1e-3));
    }

    #[test]
    fn request_validation_rejects_bad_shapes_and_non_finite_values() {
        let capture =
            MuseGlimmerLensCapture::new(3, vec![1], 2, vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
        validate_request(&capture, &[1.0, 2.0], 2).unwrap();
        assert!(validate_request(&capture, &[1.0], 2).is_err());
        assert!(validate_request(&capture, &[1.0, f32::NAN], 2).is_err());
        let block_zero =
            MuseGlimmerLensCapture::new(0, vec![1], 2, vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]);
        assert!(validate_request(&block_zero, &[1.0, 2.0], 2).is_err());
    }

    #[test]
    fn positioned_target_cotangent_places_rows_and_rejects_bad_requests() {
        let values = muse_glimmer_positioned_target_cotangent(&[1.0, -2.0], 3, 2, &[1, 2]).unwrap();
        assert_eq!(values, [0.0, 0.0, 1.0, -2.0, 1.0, -2.0]);
        assert!(muse_glimmer_positioned_target_cotangent(&[1.0], 3, 2, &[1]).is_err());
        assert!(muse_glimmer_positioned_target_cotangent(&[1.0, 2.0], 3, 2, &[3]).is_err());
        assert!(muse_glimmer_positioned_target_cotangent(&[1.0, 2.0], 3, 2, &[1, 1]).is_err());
        assert!(muse_glimmer_positioned_target_cotangent(&[1.0, f32::NAN], 3, 2, &[1]).is_err());
    }

    #[test]
    fn adjacent_fit_positions_exclude_final_and_honor_skip_first() {
        assert_eq!(adjacent_fit_position_range(5, 0).unwrap(), 0..4);
        assert_eq!(adjacent_fit_position_range(5, 2).unwrap(), 2..4);
        assert!(adjacent_fit_position_range(1, 0).is_err());
        assert!(adjacent_fit_position_range(5, 4).is_err());
    }

    #[test]
    fn adjacent_fit_reduction_means_only_valid_source_rows() {
        let rows = [1.0, 10.0, 3.0, 30.0, 5.0, 50.0, 100.0, 1000.0];
        assert_eq!(
            mean_reduce_position_rows(&rows, 4, 2, 1..3).unwrap(),
            [4.0, 40.0]
        );
        assert!(mean_reduce_position_rows(&rows, 4, 2, 3..3).is_err());
        assert!(mean_reduce_position_rows(&rows[..6], 4, 2, 1..3).is_err());
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn real_q8_block_51_replay_and_vjp_smoke() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf".into()
        });
        let gguf = GgufFile::open(path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan =
            MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 weights");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let tokens = [weights.config().bos_token_id, 19_873, 24];
        let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
            .expect("allocate bounded Muse session");
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let capture = forward
            .capture_fresh_lens_prompt(&tokens, 51, &mut session)
            .expect("capture block 51");
        let covectors =
            muse_glimmer_selected_token_covectors(&ctx, &weights, &[weights.config().eos_token_id])
                .expect("extract EOS covector");
        let target_cotangent = muse_glimmer_positioned_target_cotangent(
            covectors.token_values(0).unwrap(),
            tokens.len(),
            weights.config().hidden_size as usize,
            &[2],
        )
        .expect("place EOS covector on final prompt row");
        let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
        let fit = muse_glimmer_fit_adjacent_full_attention_selected_tokens(
            &ctx,
            &bound,
            &capture,
            &covectors,
            0,
            MuseGlimmerLensRule::J,
        )
        .expect("fit adjacent selected-token direction");
        assert_eq!(fit.source_block, 50);
        assert_eq!(fit.target_block, 51);
        assert_eq!(fit.n_valid_positions, 2);
        assert_eq!(fit.token_ids, [weights.config().eos_token_id]);
        assert_eq!(fit.values.len(), weights.config().hidden_size as usize);
        assert!(fit.values.iter().all(|value| value.is_finite()));
        let j = muse_glimmer_one_full_attention_block_vjp(
            &ctx,
            &bound,
            &capture,
            &target_cotangent,
            MuseGlimmerLensRule::J,
        )
        .expect("J replay and reverse block 51");
        let r = muse_glimmer_one_full_attention_block_vjp(
            &ctx,
            &bound,
            &capture,
            &target_cotangent,
            MuseGlimmerLensRule::R,
        )
        .expect("R replay and reverse block 51");
        let direction = (0..capture.input_residuals().len())
            .map(|index| (index as f32 * 0.013_37 + 0.41).sin())
            .collect::<Vec<_>>();
        let epsilon = 0.04_f32;
        let plus = capture
            .input_residuals()
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value + epsilon * direction)
            .collect::<Vec<_>>();
        let minus = capture
            .input_residuals()
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value - epsilon * direction)
            .collect::<Vec<_>>();
        let plus_replay =
            muse_glimmer_replay_one_full_attention_block(&ctx, &bound, 51, &plus, tokens.len())
                .expect("positive directional replay");
        let minus_replay =
            muse_glimmer_replay_one_full_attention_block(&ctx, &bound, 51, &minus, tokens.len())
                .expect("negative directional replay");
        let finite_difference = (dot(&plus_replay.post_block_residuals, &target_cotangent)
            - dot(&minus_replay.post_block_residuals, &target_cotangent))
            / (2.0 * epsilon);
        let reverse_directional = dot(&j.input_cotangent, &direction);
        let absolute_error = (finite_difference - reverse_directional).abs();
        let relative_error = absolute_error
            / finite_difference
                .abs()
                .max(reverse_directional.abs())
                .max(1e-6);
        let jr_max_abs_difference = j
            .input_cotangent
            .iter()
            .zip(&r.input_cotangent)
            .map(|(&j, &r)| (j - r).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "Muse Q8 block 51 T=3: post_attention_max_abs={} post_block_max_abs={} epsilon={} finite_difference={} reverse={} absolute_error={} relative_error={} jr_grad_max_abs_difference={} fit_max_abs={}",
            j.post_attention_replay_max_abs_error,
            j.post_block_replay_max_abs_error,
            epsilon,
            finite_difference,
            reverse_directional,
            absolute_error,
            relative_error,
            jr_max_abs_difference,
            fit.values
                .iter()
                .map(|value| value.abs())
                .fold(0.0_f32, f32::max),
        );
        assert!(j.input_cotangent.iter().all(|value| value.is_finite()));
        assert!(r.input_cotangent.iter().all(|value| value.is_finite()));
        assert!(j.post_attention_replay_max_abs_error < 0.022);
        assert!(j.post_block_replay_max_abs_error < 0.022);
        assert!(relative_error < 0.005);
        assert!(jr_max_abs_difference > 1e-4);
    }

    #[test]
    #[ignore = "requires the authenticated local Unsloth Muse Glimmer Q8_0 target"]
    fn real_q8_sliding_block_50_replay_and_vjp_smoke() {
        let path = std::env::var("MUSE_GLIMMER_Q8_GGUF").unwrap_or_else(|_| {
            "/Users/tito/models/muse-glimmer/Muse-Glimmer-30B-Q8_0.gguf".into()
        });
        let gguf = GgufFile::open(path).expect("open Muse Q8 target");
        let ctx = MetalContext::new().expect("open Metal context");
        let plan =
            MuseGlimmerMetalWeightPlan::for_release(&ctx, &gguf).expect("qualify Muse Q8 target");
        let admitted = plan
            .admit(ctx.memory_signals())
            .expect("admit Muse Q8 weights");
        let realized = MuseGlimmerMetalWeights::realize(&ctx, &gguf, admitted)
            .expect("realize Muse Q8 weights");
        let weights = realized.into_weights();
        let tokens = [weights.config().bos_token_id, 19_873, 24];
        let mut session = MuseGlimmerTextSession::new(&ctx, weights.config(), tokens.len())
            .expect("allocate bounded Muse session");
        let forward = MuseGlimmerTextForward::new(&ctx, &weights).expect("bind Muse forward");
        let capture = forward
            .capture_fresh_lens_prompt(&tokens, 50, &mut session)
            .expect("capture sliding block 50");
        assert_eq!(capture.target_block(), 50);
        let covectors =
            muse_glimmer_selected_token_covectors(&ctx, &weights, &[weights.config().eos_token_id])
                .expect("extract EOS covector");
        let target_cotangent = muse_glimmer_positioned_target_cotangent(
            covectors.token_values(0).unwrap(),
            tokens.len(),
            weights.config().hidden_size as usize,
            &[2],
        )
        .expect("place EOS covector on final prompt row");
        let bound = MuseGlimmerMetalModelWeights::bind(&weights).expect("bind resident weights");
        assert!(
            muse_glimmer_one_full_attention_block_vjp(
                &ctx,
                &bound,
                &capture,
                &target_cotangent,
                MuseGlimmerLensRule::J,
            )
            .is_err()
        );
        let j = muse_glimmer_one_attention_block_vjp(
            &ctx,
            &bound,
            &capture,
            &target_cotangent,
            MuseGlimmerLensRule::J,
        )
        .expect("J replay and reverse sliding block 50");
        let r = muse_glimmer_one_attention_block_vjp(
            &ctx,
            &bound,
            &capture,
            &target_cotangent,
            MuseGlimmerLensRule::R,
        )
        .expect("R replay and reverse sliding block 50");
        let direction = (0..capture.input_residuals().len())
            .map(|index| (index as f32 * 0.017_11 + 0.29).cos())
            .collect::<Vec<_>>();
        let epsilon = 0.04_f32;
        let plus = capture
            .input_residuals()
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value + epsilon * direction)
            .collect::<Vec<_>>();
        let minus = capture
            .input_residuals()
            .iter()
            .zip(&direction)
            .map(|(&value, &direction)| value - epsilon * direction)
            .collect::<Vec<_>>();
        let plus_replay =
            muse_glimmer_replay_one_attention_block(&ctx, &bound, 50, &plus, tokens.len())
                .expect("positive sliding directional replay");
        let minus_replay =
            muse_glimmer_replay_one_attention_block(&ctx, &bound, 50, &minus, tokens.len())
                .expect("negative sliding directional replay");
        let finite_difference = (dot(&plus_replay.post_block_residuals, &target_cotangent)
            - dot(&minus_replay.post_block_residuals, &target_cotangent))
            / (2.0 * epsilon);
        let reverse_directional = dot(&j.input_cotangent, &direction);
        let absolute_error = (finite_difference - reverse_directional).abs();
        let relative_error = absolute_error
            / finite_difference
                .abs()
                .max(reverse_directional.abs())
                .max(1e-6);
        let jr_max_abs_difference = j
            .input_cotangent
            .iter()
            .zip(&r.input_cotangent)
            .map(|(&j, &r)| (j - r).abs())
            .fold(0.0_f32, f32::max);
        eprintln!(
            "Muse Q8 sliding block 50/source 49 T=3: post_attention_max_abs={} post_block_max_abs={} epsilon={} finite_difference={} reverse={} absolute_error={} relative_error={} jr_grad_max_abs_difference={}",
            j.post_attention_replay_max_abs_error,
            j.post_block_replay_max_abs_error,
            epsilon,
            finite_difference,
            reverse_directional,
            absolute_error,
            relative_error,
            jr_max_abs_difference,
        );
        assert!(j.input_cotangent.iter().all(|value| value.is_finite()));
        assert!(r.input_cotangent.iter().all(|value| value.is_finite()));
        assert!(j.post_attention_replay_max_abs_error < 0.015);
        assert!(j.post_block_replay_max_abs_error < 0.015);
        assert!(relative_error < 0.002);
        assert!(jr_max_abs_difference > 1e-4);
    }
}
