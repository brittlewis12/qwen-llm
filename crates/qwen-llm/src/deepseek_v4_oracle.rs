//! Operation-level CPU semantics for DeepSeek V4.
//!
//! These routines are deliberately independent of the Qwen forward path and do
//! not constitute a DeepSeek generation runtime. They provide small, explicit
//! numerical contracts for later Metal implementation and differential tests.

use half::bf16;

const ROUTER_NORM_FLOOR: f32 = 6.103_515_6e-5;

pub type OracleResult<T> = Result<T, DeepSeekV4OracleError>;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum DeepSeekV4OracleError {
    #[error("{name} length mismatch: expected {expected}, got {actual}")]
    Length {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("invalid {name}: {detail}")]
    Invalid { name: &'static str, detail: String },
    #[error("dimension calculation overflowed for {0}")]
    DimensionOverflow(&'static str),
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperConnectionControls {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    /// Sinkhorn output in row-major `[sinkhorn_row, sinkhorn_column]` order.
    pub combination: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HyperConnectionPre {
    pub input: Vec<f32>,
    pub mixes: Vec<f32>,
    pub controls: HyperConnectionControls,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FusedHyperConnection {
    pub residual: Vec<f32>,
    pub next: HyperConnectionPre,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedKvProjection {
    pub q_lora_raw: Vec<f32>,
    pub q_lora: Vec<f32>,
    pub queries: Vec<f32>,
    pub kv_raw: Vec<f32>,
    pub kv: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeParameters {
    pub rotary_dim: usize,
    pub theta: f32,
    pub scaling_factor: f32,
    pub original_context_length: u32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}

impl RopeParameters {
    pub const fn local(rotary_dim: usize, theta: f32) -> Self {
        Self {
            rotary_dim,
            theta,
            scaling_factor: 1.0,
            original_context_length: 0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }

    pub const fn yarn(
        rotary_dim: usize,
        theta: f32,
        scaling_factor: f32,
        original_context_length: u32,
        beta_fast: f32,
        beta_slow: f32,
    ) -> Self {
        Self {
            rotary_dim,
            theta,
            scaling_factor,
            original_context_length,
            beta_fast,
            beta_slow,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RopeDirection {
    Forward,
    Inverse,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompressedRow {
    pub start_position: u32,
    /// F32 output after pooling, RMSNorm, and RoPE but before cache-format
    /// quantization or BF16 storage round trips.
    pub value: Vec<f32>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AttentionVisibility {
    pub raw_start: u64,
    pub raw_end: u64,
    pub completed_compressed_rows: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CompressorState {
    ratio: usize,
    head_dim: usize,
    width: usize,
    next_position: u64,
    kv: Vec<f32>,
    scores: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct RoutingDecision {
    pub expert_ids: Vec<usize>,
    pub weights: Vec<f32>,
}

pub fn rms_norm(x: &[f32], weight: Option<&[f32]>, eps: f32) -> OracleResult<Vec<f32>> {
    if x.is_empty() {
        return invalid("rms_norm input", "must be nonempty");
    }
    require_positive("rms_norm epsilon", eps)?;
    require_finite("rms_norm input", x)?;
    if let Some(weight) = weight {
        require_len("rms_norm weight", weight, x.len())?;
        require_finite("rms_norm weight", weight)?;
    }

    let sum_squares = x
        .iter()
        .map(|&value| f64::from(value) * f64::from(value))
        .sum::<f64>();
    let mean_squares = (sum_squares / x.len() as f64) as f32;
    if !mean_squares.is_finite() {
        return invalid("rms_norm input", "sum of squares exceeds f32");
    }
    let scale = 1.0 / (mean_squares + eps).sqrt();
    let output: Vec<f32> = match weight {
        Some(weight) => x
            .iter()
            .zip(weight)
            .map(|(&value, &weight)| value * scale * weight)
            .collect(),
        None => x.iter().map(|&value| value * scale).collect(),
    };
    require_finite("rms_norm output", &output)?;
    Ok(output)
}

pub fn head_rms_norm_in_place(
    x: &mut [f32],
    head_count: usize,
    head_dim: usize,
    eps: f32,
) -> OracleResult<()> {
    if head_count == 0 || head_dim == 0 {
        return invalid("head RMSNorm shape", "dimensions must be nonzero");
    }
    let expected = checked_mul(head_count, head_dim, "head_count * head_dim")?;
    require_len("head RMSNorm input", x, expected)?;
    require_positive("head RMSNorm epsilon", eps)?;
    require_finite("head RMSNorm input", x)?;
    let mut output = Vec::with_capacity(x.len());
    for head in x.chunks_exact(head_dim) {
        let normalized = rms_norm(head, None, eps)?;
        output.extend_from_slice(&normalized);
    }
    x.copy_from_slice(&output);
    Ok(())
}

pub fn mat_vec(
    weights: &[f32],
    input_dim: usize,
    output_dim: usize,
    input: &[f32],
) -> OracleResult<Vec<f32>> {
    if input_dim == 0 || output_dim == 0 {
        return invalid("matvec shape", "dimensions must be nonzero");
    }
    require_len("matvec input", input, input_dim)?;
    let expected = checked_mul(input_dim, output_dim, "matvec weight shape")?;
    require_len("matvec weights", weights, expected)?;
    require_finite("matvec input", input)?;
    require_finite("matvec weights", weights)?;

    let mut output = vec![0.0; output_dim];
    for (out, row) in output.iter_mut().zip(weights.chunks_exact(input_dim)) {
        let mut sum = 0.0f32;
        for (&weight, &value) in row.iter().zip(input) {
            sum += weight * value;
        }
        if !sum.is_finite() {
            return invalid("matvec output", "contains a non-finite accumulation");
        }
        *out = sum;
    }
    Ok(output)
}

pub fn split_sinkhorn(
    mixes: &[f32],
    scale: &[f32],
    base: &[f32],
    connection_count: usize,
    iterations: usize,
    eps: f32,
) -> OracleResult<HyperConnectionControls> {
    if connection_count == 0 {
        return invalid("hyper-connection count", "must be nonzero");
    }
    if iterations == 0 {
        return invalid("Sinkhorn iterations", "must be nonzero");
    }
    require_positive("hyper-connection epsilon", eps)?;
    require_len("hyper-connection scale", scale, 3)?;
    require_finite("hyper-connection scale", scale)?;
    let matrix_len = checked_mul(
        connection_count,
        connection_count,
        "hyper-connection matrix",
    )?;
    let parameter_count = checked_add(
        checked_mul(connection_count, 2, "hyper-connection gates")?,
        matrix_len,
        "hyper-connection parameters",
    )?;
    require_len("hyper-connection mixes", mixes, parameter_count)?;
    require_len("hyper-connection base", base, parameter_count)?;
    require_finite("hyper-connection mixes", mixes)?;
    require_finite("hyper-connection base", base)?;

    let mut pre = Vec::with_capacity(connection_count);
    let mut post = Vec::with_capacity(connection_count);
    for index in 0..connection_count {
        pre.push(sigmoid(mixes[index] * scale[0] + base[index]) + eps);
    }
    for index in 0..connection_count {
        let offset = connection_count + index;
        post.push(2.0 * sigmoid(mixes[offset] * scale[1] + base[offset]));
    }

    let matrix_offset = connection_count * 2;
    let mut combination = vec![0.0; matrix_len];
    for row in 0..connection_count {
        let row_start = row * connection_count;
        let mut row_max = f32::NEG_INFINITY;
        for column in 0..connection_count {
            let index = row_start + column;
            let value = mixes[matrix_offset + index] * scale[2] + base[matrix_offset + index];
            combination[index] = value;
            row_max = row_max.max(value);
        }
        let mut row_sum = 0.0f32;
        for value in &mut combination[row_start..row_start + connection_count] {
            *value = (*value - row_max).exp();
            row_sum += *value;
        }
        let inverse = 1.0 / row_sum;
        for value in &mut combination[row_start..row_start + connection_count] {
            *value = *value * inverse + eps;
        }
    }

    normalize_sinkhorn_columns(&mut combination, connection_count, eps);
    for _ in 1..iterations {
        normalize_sinkhorn_rows(&mut combination, connection_count, eps);
        normalize_sinkhorn_columns(&mut combination, connection_count, eps);
    }
    require_finite("hyper-connection pre gates", &pre)?;
    require_finite("hyper-connection post gates", &post)?;
    require_finite("hyper-connection combination", &combination)?;

    Ok(HyperConnectionControls {
        pre,
        post,
        combination,
    })
}

pub fn hyper_connection_collapse(
    residual: &[f32],
    weights: &[f32],
    hidden_size: usize,
    connection_count: usize,
) -> OracleResult<Vec<f32>> {
    if hidden_size == 0 || connection_count == 0 {
        return invalid("hyper-connection pre shape", "dimensions must be nonzero");
    }
    let residual_len = checked_mul(hidden_size, connection_count, "hyper-connection residual")?;
    require_len("hyper-connection residual", residual, residual_len)?;
    require_len("hyper-connection pre gates", weights, connection_count)?;
    require_finite("hyper-connection residual", residual)?;
    require_finite("hyper-connection pre gates", weights)?;

    let mut output = vec![0.0; hidden_size];
    for stream in 0..connection_count {
        let stream_values = &residual[stream * hidden_size..(stream + 1) * hidden_size];
        for (output, &value) in output.iter_mut().zip(stream_values) {
            *output += value * weights[stream];
        }
    }
    require_finite("hyper-connection pre output", &output)?;
    Ok(output)
}

pub fn hyper_connection_pre(
    residual: &[f32],
    hidden_size: usize,
    connection_count: usize,
    function: &[f32],
    scale: &[f32],
    base: &[f32],
    rms_eps: f32,
    sinkhorn_iterations: usize,
    hc_eps: f32,
) -> OracleResult<HyperConnectionPre> {
    let residual_len = checked_mul(hidden_size, connection_count, "hyper-connection residual")?;
    require_len("hyper-connection residual", residual, residual_len)?;
    let parameter_width = checked_add(connection_count, 2, "hyper-connection parameter width")?;
    let parameter_count = checked_mul(
        connection_count,
        parameter_width,
        "hyper-connection parameter count",
    )?;
    let normalized = rms_norm(residual, None, rms_eps)?;
    let mixes = mat_vec(function, residual_len, parameter_count, &normalized)?;
    let controls = split_sinkhorn(
        &mixes,
        scale,
        base,
        connection_count,
        sinkhorn_iterations,
        hc_eps,
    )?;

    let input = hyper_connection_collapse(residual, &controls.pre, hidden_size, connection_count)?;

    Ok(HyperConnectionPre {
        input,
        mixes,
        controls,
    })
}

pub fn hyper_connection_post(
    block_output: &[f32],
    residual: &[f32],
    controls: &HyperConnectionControls,
    hidden_size: usize,
    connection_count: usize,
) -> OracleResult<Vec<f32>> {
    if hidden_size == 0 || connection_count == 0 {
        return invalid("hyper-connection post shape", "dimensions must be nonzero");
    }
    require_len("hyper-connection block output", block_output, hidden_size)?;
    let residual_len = checked_mul(hidden_size, connection_count, "hyper-connection residual")?;
    require_len("hyper-connection residual", residual, residual_len)?;
    require_finite("hyper-connection block output", block_output)?;
    require_finite("hyper-connection residual", residual)?;
    require_len(
        "hyper-connection post gates",
        &controls.post,
        connection_count,
    )?;
    require_finite("hyper-connection post gates", &controls.post)?;
    require_finite("hyper-connection combination", &controls.combination)?;
    require_len(
        "hyper-connection combination",
        &controls.combination,
        checked_mul(
            connection_count,
            connection_count,
            "hyper-connection combination",
        )?,
    )?;

    let mut output = vec![0.0; residual_len];
    for destination in 0..connection_count {
        for dimension in 0..hidden_size {
            let mut value = block_output[dimension] * controls.post[destination];
            for source in 0..connection_count {
                let combination = controls.combination[source * connection_count + destination];
                value += combination * residual[source * hidden_size + dimension];
            }
            output[destination * hidden_size + dimension] = value;
        }
    }
    require_finite("hyper-connection post output", &output)?;
    Ok(output)
}

pub fn hyper_connection_post_then_pre(
    block_output: &[f32],
    residual: &[f32],
    controls: &HyperConnectionControls,
    hidden_size: usize,
    connection_count: usize,
    next_function: &[f32],
    next_scale: &[f32],
    next_base: &[f32],
    rms_eps: f32,
    sinkhorn_iterations: usize,
    hc_eps: f32,
) -> OracleResult<FusedHyperConnection> {
    let residual = hyper_connection_post(
        block_output,
        residual,
        controls,
        hidden_size,
        connection_count,
    )?;
    let next = hyper_connection_pre(
        &residual,
        hidden_size,
        connection_count,
        next_function,
        next_scale,
        next_base,
        rms_eps,
        sinkhorn_iterations,
        hc_eps,
    )?;
    Ok(FusedHyperConnection { residual, next })
}

pub fn hyper_connection_head(
    residual: &[f32],
    hidden_size: usize,
    connection_count: usize,
    function: &[f32],
    scale: f32,
    base: &[f32],
    rms_eps: f32,
    hc_eps: f32,
) -> OracleResult<Vec<f32>> {
    let residual_len = checked_mul(
        hidden_size,
        connection_count,
        "hyper-connection head residual",
    )?;
    require_len("hyper-connection head residual", residual, residual_len)?;
    require_len("hyper-connection head base", base, connection_count)?;
    require_positive("hyper-connection head epsilon", hc_eps)?;
    if !scale.is_finite() {
        return invalid("hyper-connection head scale", "must be finite");
    }
    require_finite("hyper-connection head base", base)?;

    let normalized = rms_norm(residual, None, rms_eps)?;
    let mixes = mat_vec(function, residual_len, connection_count, &normalized)?;
    let pre = mixes
        .iter()
        .zip(base)
        .map(|(&mix, &base)| sigmoid(mix * scale + base) + hc_eps)
        .collect::<Vec<_>>();
    let mut output = vec![0.0; hidden_size];
    for stream in 0..connection_count {
        for dimension in 0..hidden_size {
            output[dimension] += residual[stream * hidden_size + dimension] * pre[stream];
        }
    }
    require_finite("hyper-connection head output", &output)?;
    Ok(output)
}

pub fn shared_kv_projection(
    input: &[f32],
    q_a: &[f32],
    q_a_norm: &[f32],
    q_b: &[f32],
    kv_weight: &[f32],
    kv_norm: &[f32],
    q_lora_rank: usize,
    head_count: usize,
    head_dim: usize,
    rms_eps: f32,
) -> OracleResult<SharedKvProjection> {
    let q_width = checked_mul(head_count, head_dim, "query projection width")?;
    let q_lora_raw = mat_vec(q_a, input.len(), q_lora_rank, input)?;
    let q_lora = rms_norm(&q_lora_raw, Some(q_a_norm), rms_eps)?;
    let mut queries = mat_vec(q_b, q_lora_rank, q_width, &q_lora)?;
    head_rms_norm_in_place(&mut queries, head_count, head_dim, rms_eps)?;
    let kv_raw = mat_vec(kv_weight, input.len(), head_dim, input)?;
    let kv = rms_norm(&kv_raw, Some(kv_norm), rms_eps)?;
    Ok(SharedKvProjection {
        q_lora_raw,
        q_lora,
        queries,
        kv_raw,
        kv,
    })
}

pub fn rope_tail_in_place(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    position: u32,
    parameters: RopeParameters,
    direction: RopeDirection,
) -> OracleResult<()> {
    if head_count == 0 || head_dim == 0 {
        return invalid("RoPE shape", "dimensions must be nonzero");
    }
    let expected = checked_mul(head_count, head_dim, "RoPE input shape")?;
    require_len("RoPE input", values, expected)?;
    require_finite("RoPE input", values)?;
    validate_rope(parameters, head_dim)?;
    let mut output = values.to_vec();

    let frequency_scale = 1.0 / parameters.scaling_factor;
    let yarn = parameters.scaling_factor > 1.0;
    let theta_scale = parameters.theta.powf(-2.0 / parameters.rotary_dim as f32);
    let (correction_low, correction_high) = if yarn {
        yarn_correction_range(parameters)
    } else {
        (0.0, 0.0)
    };
    let sine_sign = match direction {
        RopeDirection::Forward => 1.0,
        RopeDirection::Inverse => -1.0,
    };
    let tail_offset = head_dim - parameters.rotary_dim;

    for head in 0..head_count {
        let tail_start = head * head_dim + tail_offset;
        let tail = &mut output[tail_start..tail_start + parameters.rotary_dim];
        let mut extrapolated = position as f32;
        for pair_offset in (0..parameters.rotary_dim).step_by(2) {
            let interpolated = frequency_scale * extrapolated;
            let angle = if yarn {
                let ramp = yarn_ramp(correction_low, correction_high, pair_offset / 2);
                interpolated * (1.0 - ramp) + extrapolated * ramp
            } else {
                extrapolated
            };
            let (sine, cosine) = angle.sin_cos();
            let sine = sine * sine_sign;
            let first = tail[pair_offset];
            let second = tail[pair_offset + 1];
            tail[pair_offset] = first * cosine - second * sine;
            tail[pair_offset + 1] = first * sine + second * cosine;
            extrapolated *= theta_scale;
        }
    }
    require_finite("RoPE output", &output)?;
    values.copy_from_slice(&output);
    Ok(())
}

impl CompressorState {
    pub fn new(ratio: usize, head_dim: usize) -> OracleResult<Self> {
        if !matches!(ratio, 4 | 128) {
            return invalid("compressor ratio", "must be 4 or 128");
        }
        if head_dim == 0 {
            return invalid("compressor head dimension", "must be nonzero");
        }
        let coefficient = if ratio == 4 { 2 } else { 1 };
        let width = checked_mul(coefficient, head_dim, "compressor width")?;
        let rows = checked_mul(coefficient, ratio, "compressor rows")?;
        let state_len = checked_mul(width, rows, "compressor state")?;
        Ok(Self {
            ratio,
            head_dim,
            width,
            next_position: 0,
            kv: vec![0.0; state_len],
            scores: vec![f32::NEG_INFINITY; state_len],
        })
    }

    pub fn from_snapshot(
        ratio: usize,
        head_dim: usize,
        next_position: u64,
        kv: Vec<f32>,
        scores: Vec<f32>,
    ) -> OracleResult<Self> {
        let mut state = Self::new(ratio, head_dim)?;
        require_len("compressor KV snapshot", &kv, state.kv.len())?;
        require_len("compressor score snapshot", &scores, state.scores.len())?;
        if next_position > u64::from(u32::MAX) {
            return invalid(
                "compressor snapshot position",
                "must remain representable by the u32 position API",
            );
        }
        require_finite("compressor KV snapshot", &kv)?;
        if scores
            .iter()
            .any(|score| !score.is_finite() && *score != f32::NEG_INFINITY)
        {
            return invalid(
                "compressor score snapshot",
                "scores must be finite or negative infinity",
            );
        }
        state.next_position = next_position;
        state.kv = kv;
        state.scores = scores;
        state.validate_snapshot_phase()?;
        Ok(state)
    }

    pub fn ratio(&self) -> usize {
        self.ratio
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn projection_width(&self) -> usize {
        self.width
    }

    pub fn next_position(&self) -> u64 {
        self.next_position
    }

    pub fn kv_state(&self) -> &[f32] {
        &self.kv
    }

    pub fn score_state(&self) -> &[f32] {
        &self.scores
    }

    fn validate_snapshot_phase(&self) -> OracleResult<()> {
        let position = self.next_position as usize;
        let row_count = self.scores.len() / self.width;
        for row in 0..row_count {
            let start = row * self.width;
            let end = start + self.width;
            let scores = &self.scores[start..end];
            let kv = &self.kv[start..end];
            let expected_initialized = if self.ratio == 4 {
                position >= self.ratio
                    || (row >= self.ratio && row - self.ratio < position % self.ratio)
            } else {
                position >= self.ratio || row < position % self.ratio
            };
            let initialized = scores.iter().all(|score| score.is_finite());
            let uninitialized = scores.iter().all(|score| *score == f32::NEG_INFINITY)
                && kv.iter().all(|value| *value == 0.0);
            if (expected_initialized && !initialized) || (!expected_initialized && !uninitialized) {
                return invalid(
                    "compressor snapshot phase",
                    &format!(
                        "row {row} does not match next position {} for ratio {}",
                        self.next_position, self.ratio
                    ),
                );
            }
        }
        if self.ratio == 4 && position >= self.ratio {
            let overwritten_rows = position % self.ratio;
            for row in overwritten_rows..self.ratio {
                let previous = row * self.width;
                let current = (self.ratio + row) * self.width;
                if self.kv[previous..previous + self.width]
                    != self.kv[current..current + self.width]
                    || self.scores[previous..previous + self.width]
                        != self.scores[current..current + self.width]
                {
                    return invalid(
                        "compressor snapshot phase",
                        &format!(
                            "ratio-4 untouched overlap row {row} differs between lanes at next position {}",
                            self.next_position
                        ),
                    );
                }
            }
        }
        Ok(())
    }

    pub fn push_projected(
        &mut self,
        position: u32,
        projected_kv: &[f32],
        projected_scores: &[f32],
        ape: &[f32],
        norm_weight: &[f32],
        rms_eps: f32,
        rope: RopeParameters,
    ) -> OracleResult<Option<CompressedRow>> {
        let (candidate, emitted) = self.prepare_projected_push(
            position,
            projected_kv,
            projected_scores,
            ape,
            norm_weight,
            rms_eps,
            rope,
        )?;
        *self = candidate;
        Ok(emitted)
    }

    /// Builds the post-push state without mutating the current frontier.
    ///
    /// Cache transactions use this to stage one bounded candidate instead of
    /// cloning once in the caller and again for the transactional oracle API.
    pub fn prepare_projected_push(
        &self,
        position: u32,
        projected_kv: &[f32],
        projected_scores: &[f32],
        ape: &[f32],
        norm_weight: &[f32],
        rms_eps: f32,
        rope: RopeParameters,
    ) -> OracleResult<(Self, Option<CompressedRow>)> {
        if u64::from(position) != self.next_position {
            return invalid(
                "compressor position",
                &format!(
                    "expected sequential position {}, got {position}",
                    self.next_position
                ),
            );
        }
        require_len("compressor projected KV", projected_kv, self.width)?;
        require_len("compressor projected scores", projected_scores, self.width)?;
        require_len(
            "compressor APE",
            ape,
            checked_mul(self.ratio, self.width, "compressor APE")?,
        )?;
        require_len("compressor norm weight", norm_weight, self.head_dim)?;
        require_positive("compressor RMSNorm epsilon", rms_eps)?;
        require_finite("compressor projected KV", projected_kv)?;
        require_finite("compressor projected scores", projected_scores)?;
        require_finite("compressor APE", ape)?;
        require_finite("compressor norm weight", norm_weight)?;
        validate_rope(rope, self.head_dim)?;
        let following_position =
            position
                .checked_add(1)
                .ok_or_else(|| DeepSeekV4OracleError::Invalid {
                    name: "compressor position",
                    detail: "cannot advance beyond u32::MAX".into(),
                })?;

        let position_in_group = position as usize % self.ratio;
        let row = if self.ratio == 4 {
            self.ratio + position_in_group
        } else {
            position_in_group
        };
        let mut candidate = self.clone();
        let state_offset = row * candidate.width;
        let ape_offset = position_in_group * self.width;
        candidate.kv[state_offset..state_offset + candidate.width].copy_from_slice(projected_kv);
        for index in 0..candidate.width {
            let score = projected_scores[index] + ape[ape_offset + index];
            if !score.is_finite() {
                return invalid("compressor score", "projection plus APE is non-finite");
            }
            candidate.scores[state_offset + index] = score;
        }
        candidate.next_position = u64::from(following_position);

        if !(following_position as usize).is_multiple_of(candidate.ratio) {
            return Ok((candidate, None));
        }

        let pooled = candidate.pool()?;
        let mut value = rms_norm(&pooled, Some(norm_weight), rms_eps)?;
        let start_position = following_position
            .checked_sub(candidate.ratio as u32)
            .ok_or_else(|| DeepSeekV4OracleError::Invalid {
                name: "compressor position",
                detail: "compression boundary precedes the first complete group".into(),
            })?;
        rope_tail_in_place(
            &mut value,
            1,
            candidate.head_dim,
            start_position,
            rope,
            RopeDirection::Forward,
        )?;
        if candidate.ratio == 4 {
            candidate.roll_overlap_state();
        }
        Ok((
            candidate,
            Some(CompressedRow {
                start_position,
                value,
            }),
        ))
    }

    pub fn pool(&self) -> OracleResult<Vec<f32>> {
        compressor_pool(&self.kv, &self.scores, self.head_dim, self.ratio)
    }

    fn roll_overlap_state(&mut self) {
        let group_len = self.ratio * self.width;
        let current = self.kv[group_len..2 * group_len].to_vec();
        self.kv[..group_len].copy_from_slice(&current);
        self.kv[group_len..2 * group_len].copy_from_slice(&current);
        let current = self.scores[group_len..2 * group_len].to_vec();
        self.scores[..group_len].copy_from_slice(&current);
        self.scores[group_len..2 * group_len].copy_from_slice(&current);
    }
}

pub fn compressor_pool(
    kv: &[f32],
    scores: &[f32],
    head_dim: usize,
    ratio: usize,
) -> OracleResult<Vec<f32>> {
    if head_dim == 0 || !matches!(ratio, 4 | 128) {
        return invalid(
            "compressor pool shape",
            "head dimension must be nonzero and ratio must be 4 or 128",
        );
    }
    let coefficient = if ratio == 4 { 2 } else { 1 };
    let width = checked_mul(coefficient, head_dim, "compressor pool width")?;
    let rows = checked_mul(coefficient, ratio, "compressor pool rows")?;
    let state_len = checked_mul(width, rows, "compressor pool state")?;
    require_len("compressor pool KV", kv, state_len)?;
    require_len("compressor pool scores", scores, state_len)?;
    require_finite("compressor pool KV", kv)?;
    if scores
        .iter()
        .any(|score| !score.is_finite() && *score != f32::NEG_INFINITY)
    {
        return invalid(
            "compressor pool scores",
            "scores must be finite or negative infinity",
        );
    }

    let mut output = vec![0.0; head_dim];
    for dimension in 0..head_dim {
        let mut maximum = f32::NEG_INFINITY;
        if ratio == 4 {
            for row in 0..ratio {
                maximum = maximum.max(scores[row * width + dimension]);
                maximum = maximum.max(scores[(ratio + row) * width + head_dim + dimension]);
            }
        } else {
            for row in 0..ratio {
                maximum = maximum.max(scores[row * width + dimension]);
            }
        }
        if maximum == f32::NEG_INFINITY {
            continue;
        }

        let mut denominator = 0.0f32;
        let mut numerator = 0.0f32;
        if ratio == 4 {
            for row in 0..ratio {
                let previous_offset = row * width + dimension;
                let current_offset = (ratio + row) * width + head_dim + dimension;
                let previous_weight = (scores[previous_offset] - maximum).exp();
                let current_weight = (scores[current_offset] - maximum).exp();
                denominator += previous_weight + current_weight;
                numerator += previous_weight * kv[previous_offset];
                numerator += current_weight * kv[current_offset];
            }
        } else {
            for row in 0..ratio {
                let offset = row * width + dimension;
                let weight = (scores[offset] - maximum).exp();
                denominator += weight;
                numerator += weight * kv[offset];
            }
        }
        output[dimension] = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };
    }
    require_finite("compressor pool output", &output)?;
    Ok(output)
}

pub fn attention_visibility(
    position: u32,
    local_window: usize,
    compression_ratio: usize,
) -> OracleResult<AttentionVisibility> {
    if local_window == 0 {
        return invalid("local attention window", "must be nonzero");
    }
    if !matches!(compression_ratio, 0 | 4 | 128) {
        return invalid("attention compression ratio", "must be 0, 4, or 128");
    }
    let raw_end = u64::from(position) + 1;
    let local_window = u64::try_from(local_window)
        .map_err(|_| DeepSeekV4OracleError::DimensionOverflow("local attention window"))?;
    let raw_start = raw_end.saturating_sub(local_window);
    let completed_compressed_rows = if compression_ratio == 0 {
        0
    } else {
        usize::try_from(raw_end / compression_ratio as u64)
            .map_err(|_| DeepSeekV4OracleError::DimensionOverflow("completed compressed rows"))?
    };
    Ok(AttentionVisibility {
        raw_start,
        raw_end,
        completed_compressed_rows,
    })
}

pub fn shared_kv_attention(
    queries: &[f32],
    head_count: usize,
    head_dim: usize,
    raw_kv: &[f32],
    compressed_kv: &[f32],
    compressed_allowed: Option<&[bool]>,
    sinks: &[f32],
) -> OracleResult<Vec<f32>> {
    if head_count == 0 || head_dim == 0 {
        return invalid("attention shape", "dimensions must be nonzero");
    }
    require_len(
        "attention queries",
        queries,
        checked_mul(head_count, head_dim, "attention queries")?,
    )?;
    require_len("attention sinks", sinks, head_count)?;
    let raw_rows = matrix_rows("raw KV", raw_kv, head_dim)?;
    let compressed_rows = matrix_rows("compressed KV", compressed_kv, head_dim)?;
    if let Some(allowed) = compressed_allowed {
        require_len("compressed attention mask", allowed, compressed_rows)?;
    }
    if sinks.iter().any(|value| !value.is_finite()) {
        return invalid("attention sinks", "must be finite");
    }
    require_finite("attention queries", queries)?;
    require_finite("raw KV", raw_kv)?;
    require_finite("compressed KV", compressed_kv)?;

    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut output = vec![0.0; head_count * head_dim];
    let mut logits = Vec::with_capacity(raw_rows + compressed_rows);
    for head in 0..head_count {
        logits.clear();
        let query = &queries[head * head_dim..(head + 1) * head_dim];
        let mut maximum = sinks[head];
        for row in raw_kv.chunks_exact(head_dim) {
            let logit = dot(query, row) * scale;
            logits.push(logit);
            maximum = maximum.max(logit);
        }
        for (row_index, row) in compressed_kv.chunks_exact(head_dim).enumerate() {
            if compressed_allowed.is_some_and(|mask| !mask[row_index]) {
                logits.push(f32::NEG_INFINITY);
            } else {
                let logit = dot(query, row) * scale;
                logits.push(logit);
                maximum = maximum.max(logit);
            }
        }

        let head_output = &mut output[head * head_dim..(head + 1) * head_dim];
        let mut denominator = (sinks[head] - maximum).exp();
        for (row_index, row) in raw_kv.chunks_exact(head_dim).enumerate() {
            let weight = (logits[row_index] - maximum).exp();
            denominator += weight;
            add_scaled(head_output, row, weight);
        }
        for (compressed_index, row) in compressed_kv.chunks_exact(head_dim).enumerate() {
            let logit = logits[raw_rows + compressed_index];
            if logit == f32::NEG_INFINITY {
                continue;
            }
            let weight = (logit - maximum).exp();
            denominator += weight;
            add_scaled(head_output, row, weight);
        }
        for value in head_output {
            *value /= denominator;
        }
    }
    require_finite("attention output", &output)?;
    Ok(output)
}

pub fn grouped_low_rank_projection(
    heads: &[f32],
    head_count: usize,
    head_dim: usize,
    group_count: usize,
    rank: usize,
    output_a: &[f32],
) -> OracleResult<Vec<f32>> {
    if head_count == 0 || head_dim == 0 || rank == 0 {
        return invalid(
            "grouped low-rank projection shape",
            "dimensions must be nonzero",
        );
    }
    if group_count == 0 || !head_count.is_multiple_of(group_count) {
        return invalid(
            "attention output groups",
            "must be nonzero and divide the head count",
        );
    }
    let full_width = checked_mul(head_count, head_dim, "attention head width")?;
    require_len("attention head output", heads, full_width)?;
    let group_width = full_width / group_count;
    let low_rank_width = checked_mul(group_count, rank, "grouped low-rank width")?;
    require_len(
        "grouped output A",
        output_a,
        checked_mul(group_width, low_rank_width, "grouped output A")?,
    )?;
    require_finite("attention head output", heads)?;
    require_finite("grouped output A", output_a)?;

    let mut low_rank = vec![0.0; low_rank_width];
    for group in 0..group_count {
        let group_input = &heads[group * group_width..(group + 1) * group_width];
        for rank_index in 0..rank {
            let output_index = group * rank + rank_index;
            let row = &output_a[output_index * group_width..(output_index + 1) * group_width];
            low_rank[output_index] = dot(row, group_input);
        }
    }
    require_finite("grouped low-rank projection", &low_rank)?;
    Ok(low_rank)
}

pub fn grouped_low_rank_output(
    heads: &[f32],
    head_count: usize,
    head_dim: usize,
    group_count: usize,
    rank: usize,
    output_a: &[f32],
    output_b: &[f32],
    hidden_size: usize,
) -> OracleResult<Vec<f32>> {
    if hidden_size == 0 {
        return invalid(
            "grouped attention output shape",
            "hidden size must be nonzero",
        );
    }
    let low_rank =
        grouped_low_rank_projection(heads, head_count, head_dim, group_count, rank, output_a)?;
    require_finite("grouped output B", output_b)?;
    require_len(
        "grouped output B",
        output_b,
        checked_mul(low_rank.len(), hidden_size, "grouped output B")?,
    )?;
    mat_vec(output_b, low_rank.len(), hidden_size, &low_rank)
}

pub fn bf16_roundtrip_in_place(values: &mut [f32]) -> OracleResult<()> {
    require_finite("BF16 roundtrip input", values)?;
    let mut output = values.to_vec();
    for value in &mut output {
        *value = bf16::from_f32(*value).to_f32();
    }
    require_finite("BF16 roundtrip output", &output)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn attention_fp8_nope_bf16_rope_roundtrip_in_place(
    values: &mut [f32],
    rotary_dim: usize,
) -> OracleResult<()> {
    if values.is_empty() || rotary_dim == 0 || rotary_dim > values.len() {
        return invalid(
            "attention cache rotary dimension",
            "must be nonzero and no larger than the row width",
        );
    }
    let nope_dim = values.len() - rotary_dim;
    if !nope_dim.is_multiple_of(64) {
        return invalid("attention cache NoPE dimension", "must be divisible by 64");
    }
    require_finite("attention cache row", values)?;
    let mut output = values.to_vec();
    let (nope, rope) = output.split_at_mut(nope_dim);
    for block in nope.chunks_exact_mut(64) {
        bf16_roundtrip_in_place(block)?;
        let mut maximum = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
        maximum = maximum.max(1.0e-4);
        let scale = power_of_two_ceiling(maximum / 448.0);
        for value in block {
            *value = e4m3fn_roundtrip((*value / scale).clamp(-448.0, 448.0)) * scale;
        }
    }
    bf16_roundtrip_in_place(rope)?;
    require_finite("attention cache output", &output)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn hadamard_128_in_place(values: &mut [f32]) -> OracleResult<()> {
    require_len("Hadamard input", values, 128)?;
    require_finite("Hadamard input", values)?;
    let mut output = values.to_vec();
    crate::trellis_offline::fwht128_blocks(&mut output);
    require_finite("Hadamard output", &output)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn fp4_activation_roundtrip_in_place(values: &mut [f32]) -> OracleResult<()> {
    if values.is_empty() || !values.len().is_multiple_of(32) {
        return invalid(
            "FP4 activation row",
            "length must be nonzero and divisible by 32",
        );
    }
    require_finite("FP4 activation row", values)?;
    for block in values.chunks_exact_mut(32) {
        let mut maximum = block.iter().map(|value| value.abs()).fold(0.0, f32::max);
        maximum = maximum.max(7.052_966e-38);
        let scale = power_of_two_ceiling(maximum / 6.0);
        for value in block {
            *value = e2m1_roundtrip((*value / scale).clamp(-6.0, 6.0)) * scale;
        }
    }
    Ok(())
}

pub fn indexer_qat_roundtrip_in_place(values: &mut [f32]) -> OracleResult<()> {
    let mut output = values.to_vec();
    hadamard_128_in_place(&mut output)?;
    fp4_activation_roundtrip_in_place(&mut output)?;
    values.copy_from_slice(&output);
    Ok(())
}

pub fn indexer_scores(
    queries: &[f32],
    head_weights: &[f32],
    compressed_keys: &[f32],
    head_count: usize,
    head_dim: usize,
) -> OracleResult<Vec<f32>> {
    if head_count == 0 || head_dim == 0 {
        return invalid("indexer shape", "dimensions must be nonzero");
    }
    require_len(
        "indexer queries",
        queries,
        checked_mul(head_count, head_dim, "indexer queries")?,
    )?;
    require_len("indexer head weights", head_weights, head_count)?;
    let row_count = matrix_rows("indexer compressed keys", compressed_keys, head_dim)?;
    require_finite("indexer queries", queries)?;
    require_finite("indexer head weights", head_weights)?;
    require_finite("indexer compressed keys", compressed_keys)?;
    let scale = 1.0 / checked_mul(head_count, head_dim, "indexer scale")? as f32;
    let scale = scale.sqrt();

    let mut scores = Vec::with_capacity(row_count);
    for key in compressed_keys.chunks_exact(head_dim) {
        let mut score = 0.0f32;
        for (head, &weight) in head_weights.iter().enumerate() {
            let query = &queries[head * head_dim..(head + 1) * head_dim];
            score += dot(query, key).max(0.0) * weight * scale;
        }
        scores.push(score);
    }
    require_finite("indexer scores", &scores)?;
    Ok(scores)
}

pub fn top_k_indices(scores: &[f32], top_k: usize) -> OracleResult<Vec<usize>> {
    if scores.iter().any(|score| !score.is_finite()) {
        return invalid("top-k scores", "must be finite");
    }
    let mut indices = (0..scores.len()).collect::<Vec<_>>();
    indices.sort_unstable_by(|&left, &right| {
        if scores[left] > scores[right] {
            std::cmp::Ordering::Less
        } else if scores[left] < scores[right] {
            std::cmp::Ordering::Greater
        } else {
            left.cmp(&right)
        }
    });
    indices.truncate(top_k.min(indices.len()));
    Ok(indices)
}

pub fn top_k_mask(scores: &[f32], top_k: usize) -> OracleResult<Vec<bool>> {
    let mut mask = vec![false; scores.len()];
    for index in top_k_indices(scores, top_k)? {
        mask[index] = true;
    }
    Ok(mask)
}

pub fn sqrt_softplus_scores(logits: &[f32]) -> OracleResult<Vec<f32>> {
    if logits.iter().any(|logit| !logit.is_finite()) {
        return invalid("router logits", "must be finite");
    }
    Ok(logits.iter().map(|&logit| softplus(logit).sqrt()).collect())
}

pub fn hash_route(
    scores: &[f32],
    selected_experts: &[usize],
    routed_scale: f32,
) -> OracleResult<RoutingDecision> {
    if selected_experts.is_empty() {
        return invalid("hash-selected experts", "must be nonempty");
    }
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return invalid("routed expert scale", "must be finite and positive");
    }
    require_router_scores(scores)?;
    let mut seen = vec![false; scores.len()];
    let mut weights = Vec::with_capacity(selected_experts.len());
    for &expert in selected_experts {
        if expert >= scores.len() {
            return invalid("hash-selected experts", "contains an out-of-range expert");
        }
        if seen[expert] {
            return invalid("hash-selected experts", "contains a duplicate expert");
        }
        seen[expert] = true;
        weights.push(scores[expert]);
    }
    normalize_router_weights(&mut weights, routed_scale)?;
    Ok(RoutingDecision {
        expert_ids: selected_experts.to_vec(),
        weights,
    })
}

pub fn learned_route(
    scores: &[f32],
    correction_bias: &[f32],
    top_k: usize,
    routed_scale: f32,
) -> OracleResult<RoutingDecision> {
    require_len("router correction bias", correction_bias, scores.len())?;
    require_router_scores(scores)?;
    require_finite("router correction bias", correction_bias)?;
    if top_k == 0 || top_k > scores.len() {
        return invalid("router top-k", "must be in 1..=expert_count");
    }
    if !routed_scale.is_finite() || routed_scale <= 0.0 {
        return invalid("routed expert scale", "must be finite and positive");
    }
    let selection = scores
        .iter()
        .zip(correction_bias)
        .map(|(&score, &bias)| score + bias)
        .collect::<Vec<_>>();
    let expert_ids = top_k_indices(&selection, top_k)?;
    let mut weights = expert_ids
        .iter()
        .map(|&expert| scores[expert])
        .collect::<Vec<_>>();
    normalize_router_weights(&mut weights, routed_scale)?;
    Ok(RoutingDecision {
        expert_ids,
        weights,
    })
}

pub fn clamped_swiglu(gate: &[f32], up: &[f32], clamp: f32) -> OracleResult<Vec<f32>> {
    if gate.is_empty() {
        return invalid("SwiGLU input", "must be nonempty");
    }
    require_len("SwiGLU up projection", up, gate.len())?;
    require_positive("SwiGLU clamp", clamp)?;
    require_finite("SwiGLU gate projection", gate)?;
    require_finite("SwiGLU up projection", up)?;
    let output = gate
        .iter()
        .zip(up)
        .map(|(&gate, &up)| {
            let gate = gate.min(clamp);
            let up = up.clamp(-clamp, clamp);
            gate * sigmoid(gate) * up
        })
        .collect::<Vec<_>>();
    require_finite("SwiGLU output", &output)?;
    Ok(output)
}

fn normalize_sinkhorn_rows(matrix: &mut [f32], size: usize, eps: f32) {
    for row in matrix.chunks_exact_mut(size) {
        let sum = row.iter().sum::<f32>();
        let inverse = 1.0 / (sum + eps);
        for value in row {
            *value *= inverse;
        }
    }
}

fn normalize_sinkhorn_columns(matrix: &mut [f32], size: usize, eps: f32) {
    for column in 0..size {
        let mut sum = 0.0f32;
        for row in 0..size {
            sum += matrix[row * size + column];
        }
        let inverse = 1.0 / (sum + eps);
        for row in 0..size {
            matrix[row * size + column] *= inverse;
        }
    }
}

fn normalize_router_weights(weights: &mut [f32], routed_scale: f32) -> OracleResult<()> {
    let sum = weights.iter().sum::<f32>();
    if !sum.is_finite() {
        return invalid("router weights", "normalization sum is non-finite");
    }
    let denominator = sum.max(ROUTER_NORM_FLOOR);
    for weight in weights {
        *weight = *weight / denominator * routed_scale;
    }
    Ok(())
}

fn validate_rope(parameters: RopeParameters, head_dim: usize) -> OracleResult<()> {
    if parameters.rotary_dim == 0
        || parameters.rotary_dim > head_dim
        || !parameters.rotary_dim.is_multiple_of(2)
    {
        return invalid(
            "RoPE rotary dimension",
            "must be even, nonzero, and no larger than the head dimension",
        );
    }
    if !parameters.theta.is_finite() || parameters.theta <= 1.0 {
        return invalid("RoPE theta", "must be finite and greater than one");
    }
    require_positive("RoPE scaling factor", parameters.scaling_factor)?;
    if parameters.scaling_factor < 1.0 {
        return invalid("RoPE scaling factor", "values below one are unsupported");
    }
    if parameters.scaling_factor > 1.0 {
        if parameters.original_context_length == 0 {
            return invalid("RoPE original context", "must be nonzero for YaRN");
        }
        require_positive("RoPE beta fast", parameters.beta_fast)?;
        require_positive("RoPE beta slow", parameters.beta_slow)?;
    }
    Ok(())
}

fn yarn_correction_range(parameters: RopeParameters) -> (f32, f32) {
    let correction = |rotations: f32| {
        parameters.rotary_dim as f32
            * (parameters.original_context_length as f32 / (rotations * 2.0 * std::f32::consts::PI))
                .ln()
            / (2.0 * parameters.theta.ln())
    };
    let low = correction(parameters.beta_fast).floor().max(0.0);
    let high = correction(parameters.beta_slow)
        .ceil()
        .min((parameters.rotary_dim - 1) as f32);
    (low, high)
}

fn yarn_ramp(low: f32, high: f32, pair_index: usize) -> f32 {
    1.0 - (((pair_index as f32 - low) / (high - low).max(0.001)).clamp(0.0, 1.0))
}

fn e4m3fn_value(index: usize) -> f32 {
    const EXPONENT_SCALE: [f32; 16] = [
        0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        128.0, 256.0,
    ];
    let exponent = (index >> 3) & 0x0f;
    let mantissa = index & 0x07;
    if exponent == 0 {
        mantissa as f32 * 0.001953125
    } else {
        (1.0 + mantissa as f32 * 0.125) * EXPONENT_SCALE[exponent]
    }
}

fn e4m3fn_roundtrip(value: f32) -> f32 {
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let absolute = value.abs().min(448.0);
    let mut low = 0usize;
    let mut high = 126usize;
    while low < high {
        let middle = (low + high).div_ceil(2);
        if e4m3fn_value(middle) <= absolute {
            low = middle;
        } else {
            high = middle - 1;
        }
    }
    let mut best = low;
    if best < 126 {
        let current_difference = (absolute - e4m3fn_value(best)).abs();
        let next_difference = (absolute - e4m3fn_value(best + 1)).abs();
        if next_difference < current_difference
            || (next_difference == current_difference
                && (best + 1).is_multiple_of(2)
                && !best.is_multiple_of(2))
        {
            best += 1;
        }
    }
    sign * e4m3fn_value(best)
}

fn e2m1_roundtrip(value: f32) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = if value < 0.0 { -1.0 } else { 1.0 };
    let absolute = value.abs().min(6.0);
    let mut best = 0usize;
    let mut best_difference = absolute;
    for (index, &candidate) in VALUES.iter().enumerate().skip(1) {
        let difference = (absolute - candidate).abs();
        if difference < best_difference
            || (difference == best_difference && index.is_multiple_of(2) && !best.is_multiple_of(2))
        {
            best = index;
            best_difference = difference;
        }
    }
    sign * VALUES[best]
}

fn power_of_two_ceiling(value: f32) -> f32 {
    2.0f32.powi(value.log2().ceil() as i32)
}

fn softplus(value: f32) -> f32 {
    if value > 20.0 {
        value
    } else if value < -20.0 {
        value.exp()
    } else {
        value.exp().ln_1p()
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .fold(0.0, |sum, (&left, &right)| sum + left * right)
}

fn add_scaled(output: &mut [f32], input: &[f32], scale: f32) {
    for (output, &input) in output.iter_mut().zip(input) {
        *output += input * scale;
    }
}

fn matrix_rows(name: &'static str, values: &[f32], width: usize) -> OracleResult<usize> {
    if width == 0 {
        return invalid(name, "row width must be nonzero");
    }
    if !values.len().is_multiple_of(width) {
        return invalid(
            name,
            &format!("length {} is not divisible by {width}", values.len()),
        );
    }
    Ok(values.len() / width)
}

fn require_finite(name: &'static str, values: &[f32]) -> OracleResult<()> {
    if values.iter().any(|value| !value.is_finite()) {
        return invalid(name, "must contain only finite values");
    }
    Ok(())
}

fn require_router_scores(scores: &[f32]) -> OracleResult<()> {
    if scores.is_empty() {
        return invalid("router scores", "must be nonempty");
    }
    if scores
        .iter()
        .any(|score| !score.is_finite() || *score < 0.0)
    {
        return invalid("router scores", "must be finite and nonnegative");
    }
    Ok(())
}

fn require_len<T>(name: &'static str, values: &[T], expected: usize) -> OracleResult<()> {
    if values.len() != expected {
        return Err(DeepSeekV4OracleError::Length {
            name,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

fn require_positive(name: &'static str, value: f32) -> OracleResult<()> {
    if !value.is_finite() || value <= 0.0 {
        return invalid(name, "must be finite and positive");
    }
    Ok(())
}

fn checked_mul(left: usize, right: usize, name: &'static str) -> OracleResult<usize> {
    left.checked_mul(right)
        .ok_or(DeepSeekV4OracleError::DimensionOverflow(name))
}

fn checked_add(left: usize, right: usize, name: &'static str) -> OracleResult<usize> {
    left.checked_add(right)
        .ok_or(DeepSeekV4OracleError::DimensionOverflow(name))
}

fn invalid<T>(name: &'static str, detail: &str) -> OracleResult<T> {
    Err(DeepSeekV4OracleError::Invalid {
        name,
        detail: detail.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "value {index}: expected {expected}, got {actual}"
            );
        }
    }

    #[test]
    fn sinkhorn_follows_twenty_cycle_contract() {
        let mixes = (0..24)
            .map(|index| (index as f32 - 11.5) * 0.13)
            .collect::<Vec<_>>();
        let base = (0..24)
            .map(|index| ((index * 7 % 13) as f32 - 6.0) * 0.04)
            .collect::<Vec<_>>();
        let controls = split_sinkhorn(&mixes, &[0.7, -0.4, 1.2], &base, 4, 20, 1e-6).unwrap();
        assert!(
            controls
                .pre
                .iter()
                .all(|value| *value > 0.0 && *value < 1.1)
        );
        assert!(
            controls
                .post
                .iter()
                .all(|value| *value > 0.0 && *value < 2.0)
        );
        for row in controls.combination.chunks_exact(4) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 5e-6);
        }
        for column in 0..4 {
            let sum = (0..4)
                .map(|row| controls.combination[row * 4 + column])
                .sum::<f32>();
            assert!((sum - 1.0).abs() < 5e-6);
        }
    }

    #[test]
    fn fused_hyper_connection_matches_unfused_order() {
        let hidden_size = 3;
        let connection_count = 4;
        let residual = (0..12)
            .map(|index| (index as f32 - 5.0) * 0.2)
            .collect::<Vec<_>>();
        let function = (0..12 * 24)
            .map(|index| ((index * 17 % 29) as f32 - 14.0) * 0.006)
            .collect::<Vec<_>>();
        let scale = [0.5, -0.25, 0.8];
        let base = (0..24)
            .map(|index| ((index * 5 % 11) as f32 - 5.0) * 0.03)
            .collect::<Vec<_>>();
        let first = hyper_connection_pre(
            &residual,
            hidden_size,
            connection_count,
            &function,
            &scale,
            &base,
            1e-6,
            20,
            1e-6,
        )
        .unwrap();
        let block_output = [0.4, -0.7, 1.1];
        let post = hyper_connection_post(
            &block_output,
            &residual,
            &first.controls,
            hidden_size,
            connection_count,
        )
        .unwrap();
        let next = hyper_connection_pre(
            &post,
            hidden_size,
            connection_count,
            &function,
            &scale,
            &base,
            1e-6,
            20,
            1e-6,
        )
        .unwrap();
        let fused = hyper_connection_post_then_pre(
            &block_output,
            &residual,
            &first.controls,
            hidden_size,
            connection_count,
            &function,
            &scale,
            &base,
            1e-6,
            20,
            1e-6,
        )
        .unwrap();
        assert_eq!(fused.residual, post);
        assert_eq!(fused.next, next);
    }

    #[test]
    fn rope_rotates_tail_and_inverse_restores_input() {
        let parameters = RopeParameters::yarn(4, 160_000.0, 16.0, 65_536, 32.0, 1.0);
        let mut values = vec![9.0, 8.0, 1.0, -2.0, 3.0, -4.0];
        let original = values.clone();
        rope_tail_in_place(
            &mut values,
            1,
            6,
            65_536,
            parameters,
            RopeDirection::Forward,
        )
        .unwrap();
        assert_eq!(&values[..2], &original[..2]);
        rope_tail_in_place(
            &mut values,
            1,
            6,
            65_536,
            parameters,
            RopeDirection::Inverse,
        )
        .unwrap();
        assert_close(&values, &original, 2e-5);
    }

    #[test]
    fn overlap_compressor_uses_previous_a_and_current_b() {
        let mut state = CompressorState::new(4, 2).unwrap();
        let ape = vec![0.0; 4 * 4];
        let norm = [1.0, 1.0];
        let rope = RopeParameters::local(2, 10_000.0);
        let mut first = None;
        for position in 0..4 {
            first = state
                .push_projected(
                    position,
                    &[10.0 + position as f32, 20.0, 100.0 + position as f32, 200.0],
                    &[0.0; 4],
                    &ape,
                    &norm,
                    1e-6,
                    rope,
                )
                .unwrap();
        }
        assert!(first.is_some());
        let first_unrotated = rms_norm(&[101.5, 200.0], Some(&norm), 1e-6).unwrap();
        assert_close(&first.unwrap().value, &first_unrotated, 2e-6);

        let mut second = None;
        for position in 4..8 {
            second = state
                .push_projected(
                    position,
                    &[30.0 + position as f32, 40.0, 300.0 + position as f32, 400.0],
                    &[0.0; 4],
                    &ape,
                    &norm,
                    1e-6,
                    rope,
                )
                .unwrap();
        }
        let expected_pool = [(11.5 + 305.5) / 2.0, (20.0 + 400.0) / 2.0];
        let mut expected = rms_norm(&expected_pool, Some(&norm), 1e-6).unwrap();
        rope_tail_in_place(&mut expected, 1, 2, 4, rope, RopeDirection::Forward).unwrap();
        assert_close(&second.unwrap().value, &expected, 2e-6);
    }

    #[test]
    fn hca_emits_on_boundary_and_snapshot_restores() {
        let mut state = CompressorState::new(128, 2).unwrap();
        let ape = vec![0.0; 128 * 2];
        let norm = [1.0, 0.5];
        let rope = RopeParameters::yarn(2, 160_000.0, 16.0, 65_536, 32.0, 1.0);
        for position in 0..64 {
            assert!(
                state
                    .push_projected(
                        position,
                        &[position as f32, 1.0],
                        &[0.0, 0.0],
                        &ape,
                        &norm,
                        1e-6,
                        rope,
                    )
                    .unwrap()
                    .is_none()
            );
        }
        let mut restored = CompressorState::from_snapshot(
            128,
            2,
            state.next_position(),
            state.kv_state().to_vec(),
            state.score_state().to_vec(),
        )
        .unwrap();
        let mut emitted = None;
        for position in 64..128 {
            let arguments = (
                &[position as f32, 1.0][..],
                &[0.0, 0.0][..],
                &ape[..],
                &norm[..],
            );
            let left = state
                .push_projected(
                    position,
                    arguments.0,
                    arguments.1,
                    arguments.2,
                    arguments.3,
                    1e-6,
                    rope,
                )
                .unwrap();
            let right = restored
                .push_projected(
                    position,
                    arguments.0,
                    arguments.1,
                    arguments.2,
                    arguments.3,
                    1e-6,
                    rope,
                )
                .unwrap();
            assert_eq!(left, right);
            emitted = left;
        }
        let emitted = emitted.unwrap();
        assert_eq!(emitted.start_position, 0);
        assert_eq!(state.next_position(), 128);
    }

    #[test]
    fn attention_uses_joint_sink_normalization() {
        let output =
            shared_kv_attention(&[1.0, 0.0], 1, 2, &[1.0, 0.0], &[0.0, 2.0], None, &[0.0]).unwrap();
        let first_logit = 1.0 / 2.0f32.sqrt();
        let denominator = first_logit.exp() + 1.0 + 1.0;
        let expected = [first_logit.exp() / denominator, 2.0 / denominator];
        assert_close(&output, &expected, 1e-6);
    }

    #[test]
    fn indexer_topk_and_routing_ties_prefer_lower_id() {
        assert_eq!(top_k_indices(&[1.0, 2.0, 2.0, -1.0], 3).unwrap(), [1, 2, 0]);
        let scores = sqrt_softplus_scores(&[0.0, 0.0, -2.0, 1.0]).unwrap();
        let decision = learned_route(&scores, &[0.0, 0.0, 10.0, 0.0], 2, 1.5).unwrap();
        assert_eq!(decision.expert_ids, [2, 3]);
        assert!((decision.weights.iter().sum::<f32>() - 1.5).abs() < 1e-6);
    }

    #[test]
    fn indexer_qat_is_deterministic_and_block_scaled() {
        let mut values = (0..128)
            .map(|index| (index as f32 - 63.5) * 0.03125)
            .collect::<Vec<_>>();
        let mut repeated = values.clone();
        indexer_qat_roundtrip_in_place(&mut values).unwrap();
        indexer_qat_roundtrip_in_place(&mut repeated).unwrap();
        assert_eq!(values, repeated);
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn swiglu_clamps_gate_only_above_and_up_both_directions() {
        let result = clamped_swiglu(&[-20.0, 20.0], &[-20.0, 20.0], 10.0).unwrap();
        assert!((result[0] - (-20.0 * sigmoid(-20.0) * -10.0)).abs() < 1e-6);
        assert!((result[1] - (10.0 * sigmoid(10.0) * 10.0)).abs() < 1e-5);
    }

    #[test]
    fn compressor_errors_are_transactional() {
        let mut state = CompressorState::new(4, 2).unwrap();
        let original = state.clone();
        let invalid_rope = RopeParameters::yarn(2, 1.0, 16.0, 65_536, 32.0, 1.0);
        assert!(
            state
                .push_projected(
                    0,
                    &[1.0, 2.0, 3.0, 4.0],
                    &[0.0; 4],
                    &[0.0; 16],
                    &[1.0, 1.0],
                    1e-6,
                    invalid_rope,
                )
                .is_err()
        );
        assert_eq!(state, original);
    }

    #[test]
    fn visibility_includes_boundary_rows_on_the_same_token() {
        let c4 = [(2, 0), (3, 1), (7, 2), (8, 2)];
        for (position, expected_rows) in c4 {
            let visibility = attention_visibility(position, 128, 4).unwrap();
            assert_eq!(visibility.completed_compressed_rows, expected_rows);
        }
        let before = attention_visibility(126, 128, 128).unwrap();
        assert_eq!(before.completed_compressed_rows, 0);
        assert_eq!((before.raw_start, before.raw_end), (0, 127));
        let boundary = attention_visibility(127, 128, 128).unwrap();
        assert_eq!(boundary.completed_compressed_rows, 1);
        assert_eq!((boundary.raw_start, boundary.raw_end), (0, 128));
        let after = attention_visibility(128, 128, 128).unwrap();
        assert_eq!(after.completed_compressed_rows, 1);
        assert_eq!((after.raw_start, after.raw_end), (1, 129));
    }

    #[test]
    fn mxfp4_midpoints_use_round_to_nearest_even() {
        assert_eq!(e2m1_roundtrip(0.25), 0.0);
        assert_eq!(e2m1_roundtrip(0.75), 1.0);
        assert_eq!(e2m1_roundtrip(1.25), 1.0);
        assert_eq!(e2m1_roundtrip(1.75), 2.0);
        assert_eq!(e2m1_roundtrip(2.5), 2.0);
        assert_eq!(e2m1_roundtrip(3.5), 4.0);
        assert_eq!(e2m1_roundtrip(5.0), 4.0);
    }

    #[test]
    fn zero_width_operations_fail_instead_of_panicking() {
        assert!(mat_vec(&[], 0, 1, &[]).is_err());
        assert!(head_rms_norm_in_place(&mut [], 1, 0, 1e-6).is_err());
        assert!(indexer_scores(&[], &[], &[], 0, 128).is_err());
    }

    #[test]
    fn indexer_selects_exact_top_512_with_stable_ties() {
        let scores = (0..600)
            .map(|index| ((index * 37) % 101) as f32)
            .collect::<Vec<_>>();
        let selected = top_k_indices(&scores, 512).unwrap();
        assert_eq!(selected.len(), 512);
        for pair in selected.windows(2) {
            assert!(
                scores[pair[0]] > scores[pair[1]]
                    || (scores[pair[0]] == scores[pair[1]] && pair[0] < pair[1])
            );
        }
        let all = top_k_indices(&scores[..500], 512).unwrap();
        assert_eq!(all.len(), 500);
    }

    #[test]
    fn snapshot_restore_rejects_invalid_phase_rows() {
        let state = CompressorState::new(4, 2).unwrap();
        let mut kv = state.kv_state().to_vec();
        kv[0] = 1.0;
        assert!(CompressorState::from_snapshot(4, 2, 0, kv, state.score_state().to_vec()).is_err());

        let terminal = CompressorState::from_snapshot(
            128,
            2,
            u64::from(u32::MAX),
            vec![0.0; 256],
            vec![0.0; 256],
        )
        .unwrap();
        assert_eq!(terminal.next_position(), u64::from(u32::MAX));

        let mut overlap = CompressorState::new(4, 2).unwrap();
        let ape = vec![0.0; 16];
        for position in 0..5 {
            overlap
                .push_projected(
                    position,
                    &[position as f32, 1.0, 2.0, 3.0],
                    &[0.0; 4],
                    &ape,
                    &[1.0; 2],
                    1e-6,
                    RopeParameters::local(2, 10_000.0),
                )
                .unwrap();
        }
        let mut corrupted = overlap.kv_state().to_vec();
        let width = overlap.projection_width();
        let untouched_current_row = (4 + 1) * width;
        corrupted[untouched_current_row] += 1.0;
        assert!(
            CompressorState::from_snapshot(
                4,
                2,
                overlap.next_position(),
                corrupted,
                overlap.score_state().to_vec(),
            )
            .is_err()
        );
    }

    #[test]
    fn fallible_in_place_operations_preserve_input_on_error() {
        let mut bf16 = [f32::MAX];
        let original_bf16 = bf16;
        assert!(bf16_roundtrip_in_place(&mut bf16).is_err());
        assert_eq!(bf16, original_bf16);

        let mut heads = [1.0, 2.0, f32::MAX, f32::MAX];
        let original_heads = heads;
        assert!(head_rms_norm_in_place(&mut heads, 2, 2, 1e-6).is_err());
        assert_eq!(heads, original_heads);
    }
}
