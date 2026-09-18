//! Bounded synthetic K2 equation reference, not a full-model CPU runtime.
//!
//! Matrices are output-major rows with contiguous input columns (GGUF [in, out]).
//! Activations are f32; dot products, RMS reductions, RoPE, and stable softmax
//! use f64 intermediates before rounding outputs to f32. This accuracy-oriented
//! oracle is not a promise of bitwise equivalence to CUDA BF16 or Metal kernels.
//! Every attention read, including the current token, uses stored/rounded K/V.

use half::f16;
use serde::Deserialize;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum ReferenceError {
    #[error("invalid synthetic K2 reference geometry or request: {0}")]
    Invalid(&'static str),
    #[error("invalid synthetic K2 weight shape or nonfinite value: {0}")]
    Weight(&'static str),
    #[error("nonfinite synthetic K2 activation: {0}")]
    Nonfinite(&'static str),
}

type Result<T> = std::result::Result<T, ReferenceError>;

#[derive(Clone, Debug, Deserialize)]
pub struct Geometry {
    pub hidden: usize,
    pub intermediate: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub norm_groups: usize,
    pub vocab: usize,
    pub theta: f64,
    pub epsilon: f64,
}

impl Geometry {
    fn validate(&self) -> Result<()> {
        if !(1..=64).contains(&self.hidden)
            || !(1..=128).contains(&self.intermediate)
            || !(1..=16).contains(&self.query_heads)
            || !(1..=16).contains(&self.kv_heads)
            || !(2..=16).contains(&self.head_dim)
            || !(1..=128).contains(&self.vocab)
            || self.norm_groups != 4
            || !self.hidden.is_multiple_of(self.norm_groups)
            || !self.query_heads.is_multiple_of(self.kv_heads)
            || !self.head_dim.is_multiple_of(2)
            || !self.theta.is_finite()
            || self.theta <= 0.0
            || self.epsilon != 1e-6
        {
            return Err(ReferenceError::Invalid(
                "geometry outside tiny-reference limits",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct LayerWeights {
    pub attention_norm: Vec<f32>,
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    pub attention_output: Vec<f32>,
    pub feed_forward_norm: Vec<f32>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub down: Vec<f32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Weights {
    pub embedding: Vec<f32>,
    pub layers: Vec<LayerWeights>,
    pub output_norm: Vec<f32>,
    pub output: Vec<f32>,
}

pub struct ReferenceModel {
    geometry: Geometry,
    weights: Weights,
}

impl ReferenceModel {
    pub fn new(geometry: Geometry, weights: Weights) -> Result<Self> {
        geometry.validate()?;
        if !(1..=4).contains(&weights.layers.len()) {
            return Err(ReferenceError::Invalid("expected 1..=4 synthetic layers"));
        }
        let g = &geometry;
        let h = g.hidden;
        let q = g.query_heads * g.head_dim;
        let k = g.kv_heads * g.head_dim;
        let f = g.intermediate;
        check_weight(&weights.embedding, g.vocab * h, "embedding")?;
        check_weight(&weights.output_norm, h, "output norm")?;
        check_weight(&weights.output, g.vocab * h, "untied output")?;
        for layer in &weights.layers {
            for (values, len, name) in [
                (&layer.attention_norm, h, "attention norm"),
                (&layer.query, q * h, "query"),
                (&layer.key, k * h, "key"),
                (&layer.value, k * h, "value"),
                (&layer.attention_output, h * q, "attention output"),
                (&layer.feed_forward_norm, h, "feed-forward norm"),
                (&layer.gate, f * h, "gate"),
                (&layer.up, f * h, "up"),
                (&layer.down, h * f, "down"),
            ] {
                check_weight(values, len, name)?;
            }
        }
        Ok(Self { geometry, weights })
    }

    /// Nonzero bases represent an isolated window, not an invented cached prefix.
    pub fn session(
        &self,
        storage: CacheStorage,
        start_position: usize,
        capacity: usize,
    ) -> Result<ReferenceSession<'_>> {
        if !(1..=32).contains(&capacity)
            || start_position
                .checked_add(capacity)
                .is_none_or(|end| end > 524_288)
        {
            return Err(ReferenceError::Invalid("reference capacity/position limit"));
        }
        Ok(ReferenceSession {
            model: self,
            storage,
            start_position,
            capacity,
            length: 0,
            cache: vec![LayerCache::default(); self.weights.layers.len()],
        })
    }
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub enum CacheStorage {
    F32,
    F16,
}

impl CacheStorage {
    fn round(self, values: &[f32]) -> Result<Vec<f32>> {
        let rounded = values
            .iter()
            .map(|&value| match self {
                Self::F32 => value,
                Self::F16 => f16::from_f32(value).to_f32(),
            })
            .collect::<Vec<_>>();
        finite(&rounded, "stored K/V")?;
        Ok(rounded)
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
struct LayerCache {
    // Token-major, then KV head, then channel. F16 uses exact round-tripped
    // values in f32 containers; this is a numerical oracle, not a memory layout.
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
}

pub struct ReferenceSession<'a> {
    model: &'a ReferenceModel,
    storage: CacheStorage,
    start_position: usize,
    capacity: usize,
    length: usize,
    cache: Vec<LayerCache>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct LayerTrace {
    pub attention_norm: Vec<f32>,
    pub query_rotated: Vec<f32>,
    pub key_stored: Vec<f32>,
    pub value_stored: Vec<f32>,
    pub attention: Vec<f32>,
    pub post_attention: Vec<f32>,
    pub feed_forward_norm: Vec<f32>,
    pub gated: Vec<f32>,
    pub residual: Vec<f32>,
}

#[derive(Debug, Deserialize, PartialEq)]
pub struct TokenTrace {
    pub position: usize,
    pub layers: Vec<LayerTrace>,
    pub output_norm: Vec<f32>,
    pub logits: Vec<f32>,
}

impl ReferenceSession<'_> {
    pub fn committed_len(&self) -> usize {
        self.length
    }

    /// Append is atomic across all tokens and layers, including output readout.
    /// The session borrows immutable weights, so a cache cannot change models.
    pub fn append(&mut self, position: usize, tokens: &[u32]) -> Result<Vec<TokenTrace>> {
        if position != self.start_position + self.length {
            return Err(ReferenceError::Invalid("noncontiguous absolute position"));
        }
        if tokens.is_empty() || tokens.len() > self.capacity - self.length {
            return Err(ReferenceError::Invalid("empty append or capacity exceeded"));
        }
        if tokens
            .iter()
            .any(|&id| id as usize >= self.model.geometry.vocab)
        {
            return Err(ReferenceError::Invalid("token outside vocabulary"));
        }
        let mut staged = self.cache.clone();
        let mut traces = Vec::with_capacity(tokens.len());
        for (offset, &token) in tokens.iter().enumerate() {
            traces.push(self.forward_token(token, position + offset, &mut staged)?);
        }
        self.cache = staged;
        self.length += tokens.len();
        Ok(traces)
    }

    fn forward_token(
        &self,
        token: u32,
        position: usize,
        caches: &mut [LayerCache],
    ) -> Result<TokenTrace> {
        let g = &self.model.geometry;
        let w = &self.model.weights;
        let offset = token as usize * g.hidden;
        let mut x = w.embedding[offset..offset + g.hidden].to_vec();
        let mut layers = Vec::with_capacity(w.layers.len());
        for (layer, cache) in w.layers.iter().zip(caches) {
            let attention_norm = group_rms(&x, &layer.attention_norm, g)?;
            let mut query_rotated = matvec(&layer.query, &attention_norm)?;
            let mut key = matvec(&layer.key, &attention_norm)?;
            let value = matvec(&layer.value, &attention_norm)?;
            rope(&mut query_rotated, position, g)?;
            rope(&mut key, position, g)?;
            let key_stored = self.storage.round(&key)?;
            let value_stored = self.storage.round(&value)?;
            cache.keys.push(key_stored.clone());
            cache.values.push(value_stored.clone());
            // Streaming order is the causal mask: future keys do not exist yet.
            let attention = attend(&query_rotated, cache, g)?;
            let post_attention = add(&x, &matvec(&layer.attention_output, &attention)?)?;
            let feed_forward_norm = group_rms(&post_attention, &layer.feed_forward_norm, g)?;
            let gate = matvec(&layer.gate, &feed_forward_norm)?;
            let up = matvec(&layer.up, &feed_forward_norm)?;
            let gated = gate
                .iter()
                .zip(up)
                .map(|(&gate, up)| {
                    let gate = f64::from(gate);
                    let silu = (gate / (1.0 + (-gate).exp())) as f32;
                    silu * up
                })
                .collect::<Vec<_>>();
            finite(&gated, "SwiGLU")?;
            x = add(&post_attention, &matvec(&layer.down, &gated)?)?;
            layers.push(LayerTrace {
                attention_norm,
                query_rotated,
                key_stored,
                value_stored,
                attention,
                post_attention,
                feed_forward_norm,
                gated,
                residual: x.clone(),
            });
        }
        let output_norm = group_rms(&x, &w.output_norm, g)?;
        let logits = matvec(&w.output, &output_norm)?;
        Ok(TokenTrace {
            position,
            layers,
            output_norm,
            logits,
        })
    }
}

fn check_weight(values: &[f32], len: usize, name: &'static str) -> Result<()> {
    if values.len() != len || values.iter().any(|v| !v.is_finite()) {
        return Err(ReferenceError::Weight(name));
    }
    Ok(())
}

fn finite(values: &[f32], name: &'static str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(ReferenceError::Nonfinite(name));
    }
    Ok(())
}

fn matvec(matrix: &[f32], x: &[f32]) -> Result<Vec<f32>> {
    let result = matrix
        .chunks_exact(x.len())
        .map(|row| {
            row.iter()
                .zip(x)
                .map(|(&a, &b)| f64::from(a) * f64::from(b))
                .sum::<f64>() as f32
        })
        .collect::<Vec<_>>();
    finite(&result, "matvec")?;
    Ok(result)
}

fn add(x: &[f32], y: &[f32]) -> Result<Vec<f32>> {
    let result = x.iter().zip(y).map(|(&a, &b)| a + b).collect::<Vec<_>>();
    finite(&result, "residual")?;
    Ok(result)
}

fn group_rms(x: &[f32], gamma: &[f32], g: &Geometry) -> Result<Vec<f32>> {
    let width = g.hidden / g.norm_groups;
    let mut result = Vec::with_capacity(x.len());
    for (group, weights) in x.chunks_exact(width).zip(gamma.chunks_exact(width)) {
        let sum = group.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>();
        let inverse = (sum / width as f64 + g.epsilon).sqrt().recip();
        result.extend(
            group
                .iter()
                .zip(weights)
                .map(|(&v, &w)| (f64::from(v) * inverse * f64::from(w)) as f32),
        );
    }
    finite(&result, "group RMS")?;
    Ok(result)
}

fn rope(values: &mut [f32], position: usize, g: &Geometry) -> Result<()> {
    let half = g.head_dim / 2;
    for head in values.chunks_exact_mut(g.head_dim) {
        for i in 0..half {
            let angle = position as f64 * g.theta.powf(-2.0 * i as f64 / g.head_dim as f64);
            let (sin, cos) = angle.sin_cos();
            let a = f64::from(head[i]);
            let b = f64::from(head[i + half]);
            head[i] = (a * cos - b * sin) as f32;
            head[i + half] = (b * cos + a * sin) as f32;
        }
    }
    finite(values, "RoPE")
}

fn attend(query: &[f32], cache: &LayerCache, g: &Geometry) -> Result<Vec<f32>> {
    let mut result = Vec::with_capacity(query.len());
    let group = g.query_heads / g.kv_heads;
    let scale = (g.head_dim as f64).sqrt().recip();
    for (head, q) in query.chunks_exact(g.head_dim).enumerate() {
        let start = (head / group) * g.head_dim;
        let scores = cache
            .keys
            .iter()
            .map(|key| {
                q.iter()
                    .zip(&key[start..start + g.head_dim])
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum::<f64>()
                    * scale
            })
            .collect::<Vec<_>>();
        let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exponentials = scores
            .iter()
            .map(|score| (score - max).exp())
            .collect::<Vec<_>>();
        let denominator = exponentials.iter().sum::<f64>();
        for channel in 0..g.head_dim {
            let value = exponentials
                .iter()
                .zip(&cache.values)
                .map(|(exp, values)| (exp / denominator) * f64::from(values[start + channel]))
                .sum::<f64>();
            result.push(value as f32);
        }
    }
    finite(&result, "attention")?;
    Ok(result)
}

#[cfg(test)]
mod tests;
