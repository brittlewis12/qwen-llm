use super::{Linear, Result, dot, require, rms_norm};

#[derive(Clone, Debug)]
pub struct Mixes {
    pub pre: Vec<f32>,
    pub post: Vec<f32>,
    pub comb: Vec<f32>,
}

pub fn split_sinkhorn(
    mixes: &[f32],
    scale: [f32; 3],
    base: &[f32],
    streams: usize,
    iterations: usize,
    eps: f32,
) -> Result<Mixes> {
    let n = streams;
    require(
        n > 0
            && iterations > 0
            && eps > 0.0
            && mixes.len() == n * (n + 2)
            && base.len() == mixes.len(),
        "mHC geometry",
    )?;
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let pre = (0..n)
        .map(|i| sigmoid(mixes[i] * scale[0] + base[i]) + eps)
        .collect();
    let post = (n..2 * n)
        .map(|i| 2.0 * sigmoid(mixes[i] * scale[1] + base[i]))
        .collect();
    let mut comb: Vec<_> = (2 * n..mixes.len())
        .map(|i| mixes[i] * scale[2] + base[i])
        .collect();
    for row in comb.chunks_exact_mut(n) {
        let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        for x in row.iter_mut() {
            *x = (*x - max).exp();
        }
        let sum: f32 = row.iter().sum();
        for x in row {
            *x = *x / sum + eps;
        }
    }
    for iteration in 0..iterations {
        if iteration > 0 {
            for row in comb.chunks_exact_mut(n) {
                let sum: f32 = row.iter().sum();
                for x in row {
                    *x /= sum + eps;
                }
            }
        }
        for col in 0..n {
            let sum: f32 = (0..n).map(|row| comb[row * n + col]).sum();
            for row in 0..n {
                comb[row * n + col] /= sum + eps;
            }
        }
    }
    Ok(Mixes { pre, post, comb })
}

pub fn collapse(streams: &[f32], pre: &[f32]) -> Result<Vec<f32>> {
    require(
        !pre.is_empty() && !streams.is_empty() && streams.len() % pre.len() == 0,
        "collapse geometry",
    )?;
    let d = streams.len() / pre.len();
    Ok((0..d)
        .map(|j| {
            pre.iter()
                .enumerate()
                .map(|(s, p)| p * streams[s * d + j])
                .sum()
        })
        .collect())
}
pub fn merge(output: &[f32], residual: &[f32], mixes: &Mixes) -> Result<Vec<f32>> {
    let n = mixes.post.len();
    let d = output.len();
    require(
        n > 0 && d > 0 && residual.len() == n * d && mixes.comb.len() == n * n,
        "merge geometry",
    )?;
    // Reference comb is [source, destination], not [destination, source].
    Ok((0..n * d)
        .map(|i| {
            let dst = i / d;
            let j = i % d;
            mixes.post[dst] * output[j]
                + (0..n)
                    .map(|src| mixes.comb[src * n + dst] * residual[src * d + j])
                    .sum::<f32>()
        })
        .collect())
}

#[derive(Clone, Debug)]
pub struct HcWeights {
    pub projection: Linear,
    pub scale: [f32; 3],
    pub base: Vec<f32>,
    pub streams: usize,
    pub iterations: usize,
    pub norm_eps: f32,
    pub hc_eps: f32,
}
impl HcWeights {
    pub fn mixes(&self, x: &[f32]) -> Result<Mixes> {
        require(!x.is_empty() && self.norm_eps > 0.0, "mHC norm")?;
        let inv = (dot(x, x) / x.len() as f32 + self.norm_eps).sqrt().recip();
        let mut projection = self.projection.forward(x)?;
        for p in &mut projection {
            *p *= inv;
        }
        split_sinkhorn(
            &projection,
            self.scale,
            &self.base,
            self.streams,
            self.iterations,
            self.hc_eps,
        )
    }
}

/// Call for attention then FFN. Initialize pre to [1,0,...] at EACH token,
/// carry returned pre across layers, and collapse with the final pre at the head.
pub fn lagged_sublayer(
    residual: &[f32],
    incoming_pre: &[f32],
    weights: &HcWeights,
    norm: &[f32],
    sublayer: impl FnOnce(&[f32]) -> Result<Vec<f32>>,
) -> Result<(Vec<f32>, Vec<f32>)> {
    require(incoming_pre.len() == weights.streams, "lagged pre geometry")?;
    let current = weights.mixes(residual)?;
    let input = rms_norm(&collapse(residual, incoming_pre)?, norm, weights.norm_eps)?;
    let output = sublayer(&input)?;
    let merged = merge(&output, residual, &current)?;
    Ok((merged, current.pre))
}
