//! CPU reference primitives for Qwen3.8-Flash-Next gated residuals and PLE.

use crate::forward::mat_vec_pub;

#[derive(Debug, thiserror::Error)]
pub enum Qwen4ExpForwardError {
    #[error("invalid reference geometry: {0}")]
    InvalidGeometry(&'static str),
    #[error("shape mismatch for {name}: expected {expected}, got {got}")]
    Shape {
        name: &'static str,
        expected: usize,
        got: usize,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct GatedResidualReadState {
    residual: Vec<f32>,
    normalized: Vec<f32>,
    branch_count: usize,
    hidden_size: usize,
}

#[derive(Clone, Copy)]
pub struct GatedResidualReadWeights<'a> {
    pub norm: &'a [f32],
    pub down: &'a [f32],
    pub up: &'a [f32],
}

pub fn gated_residual_mix(
    hyper_input: &[f32],
    branch_count: usize,
    hidden_size: usize,
    low_rank: usize,
    eps: f32,
    weights: GatedResidualReadWeights<'_>,
) -> Result<(Vec<f32>, GatedResidualReadState), Qwen4ExpForwardError> {
    validate_geometry(branch_count, hidden_size, eps)?;
    if low_rank == 0 {
        return Err(Qwen4ExpForwardError::InvalidGeometry(
            "hyper-connection rank must be nonzero",
        ));
    }
    let hyper_hidden =
        branch_count
            .checked_mul(hidden_size)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "hyper-connection width overflow",
            ))?;
    require_len("hyper_input", hyper_input, hyper_hidden)?;
    require_len("hc_norm", weights.norm, hyper_hidden)?;
    require_len(
        "hc_down",
        weights.down,
        hyper_hidden
            .checked_mul(low_rank)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "down projection size overflow",
            ))?,
    )?;
    require_len(
        "hc_up",
        weights.up,
        low_rank
            .checked_mul(hyper_hidden)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "up projection size overflow",
            ))?,
    )?;

    let normalized = grouped_rms_norm(hyper_input, weights.norm, branch_count, hidden_size, eps);
    let mut low = mat_vec_pub(weights.down, hyper_hidden, low_rank, &normalized);
    for value in &mut low {
        *value = silu(*value / branch_count as f32);
    }
    let mut gate = mat_vec_pub(weights.up, low_rank, hyper_hidden, &low);
    for value in &mut gate {
        *value = sigmoid(*value);
    }

    let mut mixed = vec![0.0_f32; hidden_size];
    for branch in 0..branch_count {
        let offset = branch * hidden_size;
        for hidden in 0..hidden_size {
            mixed[hidden] += gate[offset + hidden] * normalized[offset + hidden];
        }
    }
    let inverse_branches = 1.0 / branch_count as f32;
    for value in &mut mixed {
        *value *= inverse_branches;
    }

    Ok((
        mixed,
        GatedResidualReadState {
            residual: hyper_input.to_vec(),
            normalized,
            branch_count,
            hidden_size,
        },
    ))
}

pub fn gated_residual_combine(
    block_output: &[f32],
    state: &GatedResidualReadState,
    inject: &[f32],
) -> Result<Vec<f32>, Qwen4ExpForwardError> {
    require_len("block_output", block_output, state.hidden_size)?;
    let hyper_hidden = state.branch_count.checked_mul(state.hidden_size).ok_or(
        Qwen4ExpForwardError::InvalidGeometry("hyper-connection width overflow"),
    )?;
    require_len(
        "hc_inject",
        inject,
        hyper_hidden.checked_mul(state.branch_count).ok_or(
            Qwen4ExpForwardError::InvalidGeometry("injection projection size overflow"),
        )?,
    )?;

    let mut injection = mat_vec_pub(inject, hyper_hidden, state.branch_count, &state.normalized);
    for value in &mut injection {
        *value = 2.0 * sigmoid(*value / state.branch_count as f32);
    }

    let mut output = state.residual.clone();
    for (branch, &injection_weight) in injection.iter().enumerate() {
        let offset = branch * state.hidden_size;
        for hidden in 0..state.hidden_size {
            output[offset + hidden] += block_output[hidden] * injection_weight;
        }
    }
    Ok(output)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PleConvState {
    channels: usize,
    history_len: usize,
    values: Vec<u32>,
}

impl PleConvState {
    pub fn fresh(
        channels: usize,
        kernel_size: usize,
        dilation: usize,
    ) -> Result<Self, Qwen4ExpForwardError> {
        if channels == 0 || kernel_size == 0 || dilation == 0 {
            return Err(Qwen4ExpForwardError::InvalidGeometry(
                "PLE convolution dimensions must be nonzero",
            ));
        }
        let history_len = kernel_size
            .checked_sub(1)
            .and_then(|span| span.checked_mul(dilation))
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "PLE convolution history overflow",
            ))?;
        let len =
            channels
                .checked_mul(history_len)
                .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                    "PLE convolution state overflow",
                ))?;
        Ok(Self {
            channels,
            history_len,
            values: vec![0_u32; len],
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn history_len(&self) -> usize {
        self.history_len
    }

    pub fn values(&self) -> Vec<f32> {
        self.values
            .iter()
            .map(|&bits| f32::from_bits(bits))
            .collect()
    }
}

#[derive(Clone, Copy)]
pub struct PleStepWeights<'a> {
    pub key: &'a [f32],
    pub value: &'a [f32],
    pub key_norm: &'a [f32],
    pub query_norm: &'a [f32],
    pub conv_norm: &'a [f32],
    pub conv: &'a [f32],
}

pub fn ple_step(
    embedding: &[f32],
    hyper_input: &[f32],
    branch_count: usize,
    hidden_size: usize,
    kernel_size: usize,
    dilation: usize,
    eps: f32,
    weights: PleStepWeights<'_>,
    state: &mut PleConvState,
) -> Result<Vec<f32>, Qwen4ExpForwardError> {
    validate_geometry(branch_count, hidden_size, eps)?;
    if kernel_size == 0 || dilation == 0 {
        return Err(Qwen4ExpForwardError::InvalidGeometry(
            "PLE convolution dimensions must be nonzero",
        ));
    }
    let hyper_hidden = branch_count
        .checked_mul(hidden_size)
        .ok_or(Qwen4ExpForwardError::InvalidGeometry("PLE width overflow"))?;
    require_len("ple_embedding", embedding, hidden_size)?;
    require_len("hyper_input", hyper_input, hyper_hidden)?;
    require_len(
        "ple_key",
        weights.key,
        hidden_size
            .checked_mul(hyper_hidden)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "PLE key projection size overflow",
            ))?,
    )?;
    require_len(
        "ple_value",
        weights.value,
        hidden_size
            .checked_mul(hidden_size)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "PLE value projection size overflow",
            ))?,
    )?;
    require_len("ple_key_norm", weights.key_norm, hyper_hidden)?;
    require_len("ple_query_norm", weights.query_norm, hyper_hidden)?;
    require_len("ple_conv_norm", weights.conv_norm, hyper_hidden)?;
    require_len(
        "ple_conv",
        weights.conv,
        kernel_size
            .checked_mul(hyper_hidden)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "PLE convolution size overflow",
            ))?,
    )?;
    let expected_history =
        (kernel_size - 1)
            .checked_mul(dilation)
            .ok_or(Qwen4ExpForwardError::InvalidGeometry(
                "PLE convolution history overflow",
            ))?;
    if state.channels != hyper_hidden || state.history_len != expected_history {
        return Err(Qwen4ExpForwardError::InvalidGeometry(
            "PLE convolution state geometry mismatch",
        ));
    }

    let key = mat_vec_pub(weights.key, hidden_size, hyper_hidden, embedding);
    let value = mat_vec_pub(weights.value, hidden_size, hidden_size, embedding);
    let key = grouped_rms_norm(&key, weights.key_norm, branch_count, hidden_size, eps);
    let query = grouped_rms_norm(
        hyper_input,
        weights.query_norm,
        branch_count,
        hidden_size,
        eps,
    );

    let mut gated_value = vec![0.0_f32; hyper_hidden];
    let scale = 1.0 / (hidden_size as f32).sqrt();
    for branch in 0..branch_count {
        let offset = branch * hidden_size;
        let score = key[offset..offset + hidden_size]
            .iter()
            .zip(&query[offset..offset + hidden_size])
            .map(|(&left, &right)| left * right)
            .sum::<f32>()
            * scale;
        let sign = if score > 0.0 {
            1.0
        } else if score < 0.0 {
            -1.0
        } else {
            0.0
        };
        let gate = sigmoid(sign * score.abs().max(1e-6).sqrt());
        for hidden in 0..hidden_size {
            gated_value[offset + hidden] = value[hidden] * gate;
        }
    }

    let conv_input = grouped_rms_norm(
        &gated_value,
        weights.conv_norm,
        branch_count,
        hidden_size,
        eps,
    );
    let conv_output =
        depthwise_dilated_conv_step(&conv_input, weights.conv, kernel_size, dilation, state);

    Ok(hyper_input
        .iter()
        .zip(gated_value)
        .zip(conv_output)
        .map(|((&residual, gated), conv)| residual + gated + silu(conv))
        .collect())
}

fn grouped_rms_norm(
    input: &[f32],
    weights: &[f32],
    branch_count: usize,
    hidden_size: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; input.len()];
    for branch in 0..branch_count {
        let offset = branch * hidden_size;
        let input_branch = &input[offset..offset + hidden_size];
        let mean_square =
            input_branch.iter().map(|value| value * value).sum::<f32>() / hidden_size as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        for hidden in 0..hidden_size {
            output[offset + hidden] = input_branch[hidden] * scale * weights[offset + hidden];
        }
    }
    output
}

fn depthwise_dilated_conv_step(
    input: &[f32],
    weights: &[f32],
    kernel_size: usize,
    dilation: usize,
    state: &mut PleConvState,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; state.channels];
    for channel in 0..state.channels {
        let state_offset = channel * state.history_len;
        for kernel in 0..kernel_size {
            let lag = (kernel_size - 1 - kernel) * dilation;
            let value = if lag == 0 {
                input[channel]
            } else {
                f32::from_bits(state.values[state_offset + state.history_len - lag])
            };
            output[channel] += weights[channel * kernel_size + kernel] * value;
        }
    }
    if state.history_len > 0 {
        for (channel, &value) in input.iter().enumerate().take(state.channels) {
            let offset = channel * state.history_len;
            state
                .values
                .copy_within(offset + 1..offset + state.history_len, offset);
            state.values[offset + state.history_len - 1] = value.to_bits();
        }
    }
    output
}

fn validate_geometry(
    branch_count: usize,
    hidden_size: usize,
    eps: f32,
) -> Result<(), Qwen4ExpForwardError> {
    if branch_count == 0 || hidden_size == 0 {
        return Err(Qwen4ExpForwardError::InvalidGeometry(
            "branch count and hidden size must be nonzero",
        ));
    }
    if !eps.is_finite() || eps <= 0.0 {
        return Err(Qwen4ExpForwardError::InvalidGeometry(
            "RMS epsilon must be finite and positive",
        ));
    }
    Ok(())
}

fn require_len(
    name: &'static str,
    values: &[f32],
    expected: usize,
) -> Result<(), Qwen4ExpForwardError> {
    if values.len() != expected {
        Err(Qwen4ExpForwardError::Shape {
            name,
            expected,
            got: values.len(),
        })
    } else {
        Ok(())
    }
}

#[inline]
fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[inline]
fn silu(value: f32) -> f32 {
    value * sigmoid(value)
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn gated_residual_zero_projections_reduce_to_half_norm_read_and_plain_write() {
        let input = [1.0, 2.0, 3.0, 4.0];
        let norm = [1.0; 4];
        let down = [0.0; 8];
        let up = [0.0; 8];
        let inject = [0.0; 8];
        let (mixed, state) = gated_residual_mix(
            &input,
            2,
            2,
            2,
            1e-6,
            GatedResidualReadWeights {
                norm: &norm,
                down: &down,
                up: &up,
            },
        )
        .unwrap();
        let first_scale = (2.5_f32 + 1e-6).sqrt();
        let second_scale = (12.5_f32 + 1e-6).sqrt();
        assert!((mixed[0] - 0.25 * (1.0 / first_scale + 3.0 / second_scale)).abs() < 1e-6);
        assert!((mixed[1] - 0.25 * (2.0 / first_scale + 4.0 / second_scale)).abs() < 1e-6);
        assert_eq!(
            gated_residual_combine(&[10.0, 20.0], &state, &inject).unwrap(),
            vec![11.0, 22.0, 13.0, 24.0]
        );
    }

    #[test]
    fn gated_residual_nonzero_projection_matches_golden_vector() {
        let input = [1.0, -2.0, 0.5, 3.0];
        let norm = [1.1, 0.9, 1.2, 0.8];
        let down = [
            0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8, 0.9, -1.0, 1.1, -1.2,
        ];
        let up = [
            0.2, -0.1, 0.3, -0.4, 0.5, -0.6, 0.7, 0.8, -0.9, -1.0, 1.1, 1.2,
        ];
        let inject = [0.1, 0.2, 0.3, 0.4, -0.5, 0.6, -0.7, 0.8];
        let (mixed, state) = gated_residual_mix(
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
        assert_close(&mixed, &[0.25145912, 0.021970138], 2e-6);
        let output = gated_residual_combine(&[0.25, -0.75], &state, &inject).unwrap();
        assert_close(&output, &[1.2731817, -2.819545, 0.7292096, 2.3123713], 2e-6);
    }

    #[test]
    fn gated_residual_rejects_zero_rank() {
        assert!(
            gated_residual_mix(
                &[1.0, 2.0],
                1,
                2,
                0,
                1e-6,
                GatedResidualReadWeights {
                    norm: &[1.0, 1.0],
                    down: &[],
                    up: &[],
                },
            )
            .is_err()
        );
    }

    #[test]
    fn ple_step_matches_positive_negative_and_zero_gate_golden_vector() {
        let key = [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = [0.25, 0.0, -0.5, 0.0];
        let unit_norm = [1.0; 6];
        let conv_norm = [1.0, 1.0, 0.7, 1.3, 1.5, 0.5];
        let conv = [
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0,
        ];
        let mut state = PleConvState::fresh(6, 3, 2).unwrap();
        let output = ple_step(
            &[1.0, 0.0],
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            3,
            2,
            3,
            2,
            1e-6,
            PleStepWeights {
                key: &key,
                value: &value,
                key_norm: &unit_norm,
                query_norm: &unit_norm,
                conv_norm: &conv_norm,
                conv: &conv,
            },
            &mut state,
        )
        .unwrap();
        assert_close(
            &output,
            &[
                2.8415756,
                -0.38713607,
                4.9698067,
                -0.116700545,
                15.355058,
                -0.2501295,
            ],
            3e-5,
        );
    }

    #[test]
    fn ple_convolution_state_is_forkable_and_dilated() {
        let mut state = PleConvState::fresh(2, 3, 2).unwrap();
        let weights = [1.0, 10.0, 100.0, -1.0, -10.0, -100.0];
        for step in 1..=4 {
            depthwise_dilated_conv_step(
                &[step as f32, (step + 10) as f32],
                &weights,
                3,
                2,
                &mut state,
            );
        }
        let mut fork = state.clone();
        assert_eq!(
            depthwise_dilated_conv_step(&[5.0, 15.0], &weights, 3, 2, &mut state),
            [531.0, -1641.0]
        );
        assert_eq!(
            depthwise_dilated_conv_step(&[50.0, 60.0], &weights, 3, 2, &mut fork),
            [5031.0, -6141.0]
        );
        assert_ne!(state, fork);
    }

    #[test]
    fn ple_state_preserves_signed_zero_bits() {
        let mut state = PleConvState::fresh(1, 2, 1).unwrap();
        depthwise_dilated_conv_step(&[-0.0], &[0.0, 1.0], 2, 1, &mut state);
        assert_eq!(state.values()[0].to_bits(), (-0.0_f32).to_bits());
    }

    #[test]
    fn malformed_reference_shapes_fail_without_mutating_ple_state() {
        let mut state = PleConvState::fresh(4, 2, 2).unwrap();
        let before = state.clone();
        let result = ple_step(
            &[1.0, 2.0],
            &[1.0; 4],
            2,
            2,
            2,
            2,
            1e-6,
            PleStepWeights {
                key: &[0.0; 7],
                value: &[0.0; 4],
                key_norm: &[1.0; 4],
                query_norm: &[1.0; 4],
                conv_norm: &[1.0; 4],
                conv: &[0.0; 8],
            },
            &mut state,
        );
        assert!(result.is_err());
        assert_eq!(state, before);
    }
}
