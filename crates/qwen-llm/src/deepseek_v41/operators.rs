//! Old REAP NativeF32 reference execution: no FP8/FP4 activation rounding.
//! Includes the index-owner publication correction. This is not a 1017
//! ReferenceFp8 implementation (notably, old grouped wo_a has no act_quant).
//! Matrices are
//! row-major [output, input]; streams are [stream, hidden]. State is per sequence.
//! Batched decode uses one Session per sequence, preserving independent positions.

mod attention;
mod mhc;
pub use attention::*;
pub use mhc::*;

pub type Result<T> = std::result::Result<T, String>;

pub trait Projection {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>>;
}

pub(crate) fn require(ok: bool, message: &str) -> Result<()> {
    if ok { Ok(()) } else { Err(message.into()) }
}

#[derive(Clone, Debug)]
pub struct Linear {
    pub input: usize,
    pub output: usize,
    pub weight: Vec<f32>,
}
impl Linear {
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        require(
            self.input > 0
                && self.output > 0
                && x.len() == self.input
                && self.weight.len() == self.input.checked_mul(self.output).unwrap_or(0),
            "linear shape",
        )?;
        Ok(self
            .weight
            .chunks_exact(self.input)
            .map(|w| dot(w, x))
            .collect())
    }
}
impl Projection for Linear {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        Linear::forward(self, x)
    }
}
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Pinned reference YaRN interpolation. Use original_context=0 for SWA.
pub fn rotary_frequencies(
    dim: usize,
    theta: f32,
    original_context: usize,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
) -> Result<Vec<f32>> {
    require(
        dim > 0
            && dim % 2 == 0
            && theta > 1.0
            && factor > 0.0
            && beta_fast > 0.0
            && beta_slow > 0.0,
        "rotary frequency geometry",
    )?;
    let corrected = |beta: f32| {
        dim as f32 * (original_context as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * theta.ln())
    };
    let low = corrected(beta_fast).floor().max(0.0);
    let high = corrected(beta_slow).ceil().min((dim - 1) as f32);
    Ok((0..dim / 2)
        .map(|i| {
            let frequency = theta.powf(-((2 * i) as f32) / dim as f32);
            if original_context == 0 {
                return frequency;
            }
            let ramp = ((i as f32 - low) / (high - low).max(1e-3)).clamp(0.0, 1.0);
            frequency / factor * ramp + frequency * (1.0 - ramp)
        })
        .collect())
}
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Result<Vec<f32>> {
    require(
        !x.is_empty() && x.len() == weight.len() && eps > 0.0,
        "norm shape/epsilon",
    )?;
    let scale = (dot(x, x) / x.len() as f32 + eps).sqrt().recip();
    Ok(x.iter().zip(weight).map(|(x, w)| (x * scale) * w).collect())
}

/// Old Expert.forward: upper-only gate clamp, two-sided up clamp, then SiLU.
/// A zero limit disables both clamps; mHC itself has no activation clamp.
pub fn clamped_swiglu(gate: &[f32], up: &[f32], limit: f32) -> Result<Vec<f32>> {
    require(gate.len() == up.len() && limit >= 0.0, "SwiGLU shape/limit")?;
    Ok(gate
        .iter()
        .zip(up)
        .map(|(&g, &u)| {
            let (g, u) = if limit > 0.0 {
                (g.min(limit), u.clamp(-limit, limit))
            } else {
                (g, u)
            };
            (g / (1.0 + (-g).exp())) * u
        })
        .collect())
}

/// Frequencies may be supplied directly from the pinned YaRN frequency table.
/// Adjacent pairs, not NeoX split halves; only the tail rotates.
pub fn rotary(x: &mut [f32], frequencies: &[f32], position: usize, inverse: bool) -> Result<()> {
    require(frequencies.len() * 2 <= x.len(), "rotary dimension")?;
    let start = x.len() - frequencies.len() * 2;
    for (pair, frequency) in x[start..].chunks_exact_mut(2).zip(frequencies) {
        let angle = position as f32 * frequency * if inverse { -1.0 } else { 1.0 };
        let (s, c) = angle.sin_cos();
        let (a, b) = (pair[0], pair[1]);
        pair[0] = a * c - b * s;
        pair[1] = a * s + b * c;
    }
    Ok(())
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod oracle_tests;
