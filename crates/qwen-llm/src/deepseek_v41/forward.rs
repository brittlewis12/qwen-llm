//! Old REAP-272E single-token NativeF32 execution. Dense weights remain resident;
//! only activations and small normalization/control vectors are copied to CPU.
mod attention;
mod bind;
mod memory;
pub mod profile;
mod token_map;
pub use memory::RequestMemoryEstimate;
pub use token_map::build_compressed_token_map;

use super::{engram, loader::ResidentWeights, operators as op};
use crate::metal::{self, KernelEncoder, MetalContext, MetalTensor};
use crate::tensor::GgmlType;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLResource};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

pub type Result<T> = std::result::Result<T, String>;

/// Post-FFN block output, before the next layer's Engram injection. Hidden is
/// stream-major [4, hidden_size]; next_pre_mix is [4]. Position is zero-based.
/// Records are provisional until step_with_trace returns Ok: a later failure
/// poisons the sequence and does not commit its token/address position.
pub struct LayerTrace<'a> {
    pub position: usize,
    pub layer: usize,
    pub hidden: &'a [f32],
    pub next_pre_mix: &'a [f32],
}

enum EngramRows<'a> {
    Disk(&'a super::loader::DiskEngram),
    #[cfg(test)]
    Toy(fn(usize, &[u64]) -> Result<Vec<f32>>),
}
impl EngramRows<'_> {
    fn decode(&self, layer: usize, rows: &[u64]) -> Result<Vec<f32>> {
        match self {
            Self::Disk(disk) => disk.decode_rows(layer, rows),
            #[cfg(test)]
            Self::Toy(read) => read(layer, rows),
        }
    }
}

/// Dimensions can be reduced for fixtures; layer ownership remains the old
/// forty-layer schedule. Production callers should use Default.
#[derive(Clone, Debug)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub vocab: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub rope_dim: usize,
    pub query_rank: usize,
    pub output_rank: usize,
    pub groups: usize,
    pub experts: usize,
    pub topk: usize,
    pub index_heads: usize,
    pub index_dim: usize,
    pub index_topk: usize,
    pub window: usize,
    pub candidate_block: usize,
    pub candidate_topk: usize,
    pub norm_eps: f32,
    pub hc_eps: f32,
    pub hc_iterations: usize,
    pub clamp: f32,
    pub route_scale: f32,
    pub gate_temperature: f32,
    pub rope_theta: f32,
    pub compress_theta: f32,
    pub original_context: usize,
    pub rope_factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            hidden: 5120,
            intermediate: 2304,
            vocab: 129280,
            heads: 64,
            head_dim: 512,
            rope_dim: 64,
            query_rank: 1280,
            output_rank: 1024,
            groups: 8,
            experts: 272,
            topk: 6,
            index_heads: 32,
            index_dim: 128,
            index_topk: 512,
            window: 128,
            candidate_block: 8,
            candidate_topk: 2048,
            norm_eps: 1e-20,
            hc_eps: 1e-6,
            hc_iterations: 20,
            clamp: 10.0,
            route_scale: 1.5,
            gate_temperature: 1.0,
            rope_theta: 10000.0,
            compress_theta: 160000.0,
            original_context: 65536,
            rope_factor: 16.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

#[derive(Clone)]
struct Linear<'a> {
    ctx: &'a MetalContext,
    tensor: MetalTensor,
    input: usize,
    output: usize,
}
impl Linear<'_> {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let dims = [self.input, self.output, 0, 1];
        let _wall = profile::Wall("linear", dims, profile::start());
        op::require(x.len() == self.input, "resident linear input shape")?;
        let input = upload(self.ctx, x, vec![self.input as u64])?;
        let output = MetalTensor::zeros_f32(self.ctx, vec![self.output as u64])
            .map_err(|e| e.to_string())?;
        submit(self.ctx, "linear", dims, |enc| {
            metal::encode_mat_vec_f32(
                self.ctx,
                enc,
                &self.tensor,
                &input,
                &output,
                self.input,
                self.output,
            )
        })?;
        Ok(f32_slice(&output)?.to_vec())
    }
}
impl op::Projection for Linear<'_> {
    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        Linear::forward(self, x)
    }

    fn forward_grouped(projections: &[Self], x: &[f32]) -> Result<Vec<f32>> {
        let first = projections.first().ok_or("empty grouped projection")?;
        let groups = projections.len();
        op::require(
            first.input > 0
                && first.output > 0
                && first.input.checked_mul(groups) == Some(x.len())
                && projections.iter().all(|p| {
                    p.input == first.input
                        && p.output == first.output
                        && std::ptr::eq(p.ctx, first.ctx)
                }),
            "resident grouped linear shape/context",
        )?;
        let count = first
            .output
            .checked_mul(groups)
            .ok_or("grouped output overflow")?;
        // Dimensions: per-group input, per-group output, group count, tokens.
        let dims = [first.input, first.output, groups, 1];
        let _wall = profile::Wall("linear_grouped", dims, profile::start());
        let input = upload(first.ctx, x, vec![x.len() as u64])?;
        let output =
            MetalTensor::zeros_f32(first.ctx, vec![count as u64]).map_err(|e| e.to_string())?;
        submit(first.ctx, "linear_grouped", dims, |enc| {
            for (i, p) in projections.iter().enumerate() {
                let input = input.view_subrange((i * p.input) as u64, vec![p.input as u64]);
                let output = output.view_subrange((i * p.output) as u64, vec![p.output as u64]);
                metal::encode_mat_vec_f32(
                    p.ctx, enc, &p.tensor, &input, &output, p.input, p.output,
                )?;
            }
            Ok(())
        })?;
        Ok(f32_slice(&output)?.to_vec())
    }
}

fn submit(
    ctx: &MetalContext,
    role: &'static str,
    dimensions: [usize; 4],
    encode: impl FnOnce(&KernelEncoder) -> std::result::Result<(), metal::MetalError>,
) -> Result<()> {
    let started = profile::start();
    let command = ctx
        .queue
        .commandBuffer()
        .ok_or("Metal command buffer allocation")?;
    let encoder = KernelEncoder::begin(&command);
    let result = encode(&encoder);
    encoder.end();
    result.map_err(|e| e.to_string())?;
    let encoded = started.map(|s| s.elapsed().as_secs_f64() * 1000.0);
    let wait = started.map(|_| std::time::Instant::now());
    let result = metal::commit_and_wait(&command).map_err(|e| e.to_string());
    if let Some(wait) = wait {
        profile::submitted(
            role,
            dimensions,
            encoded.unwrap(),
            wait.elapsed().as_secs_f64() * 1000.0,
            result.is_ok(),
            command.GPUStartTime(),
            command.GPUEndTime(),
        );
    }
    result
}

fn upload(ctx: &MetalContext, values: &[f32], shape: Vec<u64>) -> Result<MetalTensor> {
    MetalTensor::from_bytes(ctx, bytemuck::cast_slice(values), shape, GgmlType::F32)
        .map_err(|e| e.to_string())
}

fn f32_slice(t: &MetalTensor) -> Result<&[f32]> {
    let count = t
        .shape
        .iter()
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or("dense extent overflow")?;
    let end = count
        .checked_mul(4)
        .and_then(|n| n.checked_add(t.offset))
        .ok_or("dense byte extent overflow")?;
    op::require(
        t.dtype == GgmlType::F32
            && t.offset % 4 == 0
            && t.buffer.storageMode() == objc2_metal::MTLStorageMode::Shared
            && end <= t.buffer.length() as u64,
        "invalid shared F32 view",
    )?;
    // Resident weights are immutable; transient GPU outputs are read only after
    // successful command completion. The returned borrow cannot outlive t.
    Ok(unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<f32>()
                .add((t.offset / 4) as usize),
            count as usize,
        )
    })
}

struct Hc<'a> {
    projection: Linear<'a>,
    scale: [f32; 3],
    base: Vec<f32>,
    norm: Vec<f32>,
}
impl Hc<'_> {
    fn prepare(&self, x: &[f32], pre: &[f32], c: &Config) -> Result<(op::Mixes, Vec<f32>)> {
        let inv = (op::dot(x, x) / x.len() as f32 + c.norm_eps).sqrt().recip();
        let mut raw = self.projection.forward(x)?;
        raw.iter_mut().for_each(|v| *v *= inv);
        let mixes = op::split_sinkhorn(&raw, self.scale, &self.base, 4, c.hc_iterations, c.hc_eps)?;
        let input = op::rms_norm(&op::collapse(x, pre)?, &self.norm, c.norm_eps)?;
        Ok((mixes, input))
    }
}
struct Engram<'a> {
    projection: Linear<'a>,
    q: Vec<f32>,
    k: Vec<f32>,
}
struct Layer<'a> {
    attention: attention::AttentionWeights<'a>,
    attn_hc: Hc<'a>,
    ffn_hc: Hc<'a>,
    router: Linear<'a>,
    bias: Vec<f32>,
    shared: [Linear<'a>; 3],
    banks: [&'a MetalTensor; 3],
    engram: Option<Engram<'a>>,
}

/// Bound once, without copying dense matrices or packed expert banks. Do not
/// mutate ResidentWeights buffers while this model is alive.
pub struct Forward<'a> {
    ctx: &'a MetalContext,
    engram_rows: EngramRows<'a>,
    config: Config,
    token_map: Vec<u32>,
    pad: u32,
    layers: Vec<Layer<'a>>,
    embedding: &'a MetalTensor,
    norm: Vec<f32>,
    head: Linear<'a>,
    identity: Arc<()>,
}

/// A failed/cancelled partial token poisons the sequence, rather than silently
/// reusing partially advanced KV caches. Start a fresh State to retry a prompt.
pub struct State {
    attention: attention::Session,
    addresses: engram::AddressState,
    max_context: usize,
    poisoned: bool,
    identity: Arc<()>,
}
impl State {
    pub fn position(&self) -> usize {
        self.addresses.position() as usize
    }
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }
}

impl<'a> Forward<'a> {
    pub fn new_state(&self, max_context: usize) -> Result<State> {
        op::require(
            max_context > 0 && max_context as u64 <= engram::MAX_POSITION,
            "invalid maximum context",
        )?;
        Ok(State {
            attention: attention::Session::default(),
            addresses: engram::AddressState::new(self.pad).map_err(|e| format!("{e:?}"))?,
            max_context,
            poisoned: false,
            identity: Arc::clone(&self.identity),
        })
    }

    /// Consume one raw tokenizer ID and return the full next-token logits.
    /// Cancellation is checked before each layer and after the final head.
    pub fn step(
        &self,
        state: &mut State,
        token: u32,
        cancel: Option<&AtomicBool>,
    ) -> Result<Vec<f32>> {
        self.step_with_trace(state, token, cancel, None)
    }

    /// Synchronously borrow each completed block's CPU F32 outputs. None adds
    /// no readbacks, copies, or allocations. Callback errors poison a partially
    /// executed sequence just like operator errors; callback panics also leave
    /// it poisoned. The callback may set cancel to stop before the next layer.
    pub fn step_with_trace(
        &self,
        state: &mut State,
        token: u32,
        cancel: Option<&AtomicBool>,
        trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        op::require(
            !state.poisoned,
            "sequence poisoned by a failed partial token",
        )?;
        op::require(
            Arc::ptr_eq(&state.identity, &self.identity),
            "state belongs to another model",
        )?;
        op::require(
            state.position() < state.max_context,
            "maximum context reached",
        )?;
        let compressed = *self
            .token_map
            .get(token as usize)
            .ok_or("token outside vocabulary")?;
        cancelled(cancel)?;
        let (rows, next) = state
            .addresses
            .step(engram::Token::Compressed(compressed))
            .map_err(|e| format!("{e:?}"))?;
        state.poisoned = true;
        let result = self.step_inner(state, token as usize, &rows, cancel, trace)?;
        state.addresses = next;
        state.poisoned = false;
        Ok(result)
    }

    fn step_inner(
        &self,
        state: &mut State,
        token: usize,
        rows: &engram::Rows,
        cancel: Option<&AtomicBool>,
        mut trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        let c = &self.config;
        let position = state.position();
        let embedding = &f32_slice(self.embedding)?[token * c.hidden..(token + 1) * c.hidden];
        let mut x = embedding.repeat(4);
        let mut pre = vec![1.0, 0.0, 0.0, 0.0];
        for (id, layer) in self.layers.iter().enumerate() {
            cancelled(cancel)?;
            if let Some(e) = &layer.engram {
                let hashes = &rows[if id == 1 { 0 } else { 1 }];
                let values = self.engram_rows.decode(id, hashes)?;
                let kv = e.projection.forward(&values)?;
                inject_engram(&mut x, &kv, &e.q, &e.k, c.norm_eps)?;
            }
            let (mixes, input) = layer.attn_hc.prepare(&x, &pre, c)?;
            let output = state
                .attention
                .attention(id, position, &input, &layer.attention)?;
            x = op::merge(&output, &x, &mixes)?;
            pre = mixes.pre;
            let (mixes, input) = layer.ffn_hc.prepare(&x, &pre, c)?;
            let output = Self::moe(self.ctx, &self.config, layer, &input)?;
            x = op::merge(&output, &x, &mixes)?;
            pre = mixes.pre;
            if let Some(callback) = trace.as_mut() {
                callback(LayerTrace {
                    position,
                    layer: id,
                    hidden: &x,
                    next_pre_mix: &pre,
                })?;
            }
        }
        let hidden = op::rms_norm(&op::collapse(&x, &pre)?, &self.norm, c.norm_eps)?;
        let logits = self.head.forward(&hidden)?;
        cancelled(cancel)?;
        op::require(logits.iter().all(|v| v.is_finite()), "non-finite logits")?;
        Ok(logits)
    }

    fn moe(ctx: &MetalContext, c: &Config, layer: &Layer<'_>, x: &[f32]) -> Result<Vec<f32>> {
        let (ids, weights) = route(
            &layer.router.forward(x)?,
            &layer.bias,
            c.topk,
            c.gate_temperature,
            c.route_scale,
        )?;
        let input = upload(ctx, x, vec![c.hidden as u64])?;
        let ids_tensor = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&ids),
            vec![c.topk as u64],
            GgmlType::I32,
        )
        .map_err(|e| e.to_string())?;
        let status = MetalTensor::from_bytes(
            ctx,
            bytemuck::cast_slice(&[metal::ROUTE_STATUS_READY]),
            vec![1],
            GgmlType::I32,
        )
        .map_err(|e| e.to_string())?;
        let activation = MetalTensor::zeros_f32(ctx, vec![c.intermediate as u64, c.topk as u64])
            .map_err(|e| e.to_string())?;
        let dims = [c.hidden, c.intermediate, c.experts, c.topk];
        let wall = profile::Wall("native_gate_up", dims, profile::start());
        submit(ctx, "native_gate_up", dims, |enc| {
            metal::encode_all_slots_gate_up_swiglu_native(
                ctx,
                enc,
                layer.banks[0],
                layer.banks[1],
                &input,
                &ids_tensor,
                &status,
                &activation,
                c.hidden,
                c.intermediate,
                c.experts,
                c.topk,
                c.clamp,
            )
        })?;
        drop(wall);
        // Old Expert.forward applies route weight BEFORE down, not after it.
        let mut weighted = f32_slice(&activation)?.to_vec();
        for (slot, &weight) in weighted.chunks_exact_mut(c.intermediate).zip(&weights) {
            slot.iter_mut().for_each(|v| *v *= weight);
        }
        let weighted = upload(ctx, &weighted, vec![c.intermediate as u64, c.topk as u64])?;
        let down = MetalTensor::zeros_f32(ctx, vec![c.hidden as u64, c.topk as u64])
            .map_err(|e| e.to_string())?;
        let dims = [c.intermediate, c.hidden, c.experts, c.topk];
        let wall = profile::Wall("native_down", dims, profile::start());
        submit(ctx, "native_down", dims, |enc| {
            metal::encode_all_slots_down_native(
                ctx,
                enc,
                layer.banks[2],
                &weighted,
                &ids_tensor,
                &status,
                &down,
                c.intermediate,
                c.hidden,
                c.experts,
                c.topk,
            )
        })?;
        drop(wall);
        let mut output = vec![0.0; c.hidden];
        for slot in f32_slice(&down)?.chunks_exact(c.hidden) {
            for (out, value) in output.iter_mut().zip(slot) {
                *out += value;
            }
        }
        let shared = op::clamped_swiglu(
            &layer.shared[0].forward(x)?,
            &layer.shared[1].forward(x)?,
            c.clamp,
        )?;
        for (out, value) in output.iter_mut().zip(layer.shared[2].forward(&shared)?) {
            *out += value;
        }
        Ok(output)
    }
}

fn cancelled(cancel: Option<&AtomicBool>) -> Result<()> {
    op::require(
        !cancel.is_some_and(|c| c.load(Ordering::Relaxed)),
        "cancelled",
    )
}

fn route(
    logits: &[f32],
    bias: &[f32],
    topk: usize,
    temperature: f32,
    scale: f32,
) -> Result<(Vec<i32>, Vec<f32>)> {
    op::require(
        !logits.is_empty()
            && bias.len() == logits.len()
            && topk > 0
            && topk <= logits.len()
            && temperature.is_finite()
            && temperature > 0.0
            && scale.is_finite()
            && logits.iter().chain(bias).all(|v| v.is_finite()),
        "router geometry/values",
    )?;
    let scores: Vec<_> = logits
        .iter()
        .map(|v| {
            let v = v / temperature;
            (v.max(0.0) + (-v.abs()).exp().ln_1p()).sqrt()
        })
        .collect();
    let selection: Vec<_> = scores.iter().zip(bias).map(|(s, b)| s + b).collect();
    // Sorted by expert ID to match the old source accumulation order.
    let ids = op::topk(&selection, topk);
    let denom = if topk > 1 {
        ids.iter().map(|&i| scores[i]).sum::<f32>() + 1e-20
    } else {
        1.0
    };
    let weights = ids.iter().map(|&i| scores[i] / denom * scale).collect();
    Ok((ids.into_iter().map(|i| i as i32).collect(), weights))
}

fn inject_engram(x: &mut [f32], kv: &[f32], q: &[f32], k: &[f32], eps: f32) -> Result<()> {
    op::require(
        x.len() % 4 == 0
            && !x.is_empty()
            && q.len() == x.len()
            && k.len() == x.len()
            && kv.len() == x.len() / 4 * 5,
        "Engram injection shape",
    )?;
    let d = x.len() / 4;
    let value = &kv[4 * d..];
    for s in 0..4 {
        let h = &mut x[s * d..(s + 1) * d];
        let key = &kv[s * d..(s + 1) * d];
        let rstd = (op::dot(h, h) / d as f32 + eps).sqrt().recip()
            * (op::dot(key, key) / d as f32 + eps).sqrt().recip();
        let dot: f32 = (0..d)
            .map(|j| h[j] * (q[s * d + j] * k[s * d + j]) * key[j])
            .sum();
        let dot = dot * rstd * (d as f32).sqrt().recip();
        let signed = dot.abs().max(1e-6).sqrt().copysign(dot);
        let gate = 1.0 / (1.0 + (-signed).exp());
        for (h, v) in h.iter_mut().zip(value) {
            *h += gate * v;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod toy;
