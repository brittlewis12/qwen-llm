//! GGUF V4.1 single-token NativeF32 execution. Dense weights remain resident;
//! only activations and small normalization/control vectors are copied to CPU.
mod attention;
mod attention_gpu;
mod bind;
mod ced;
mod ced_plan;
pub use ced::{CedCounters, CedLayerCounters, CedMemoryEstimate, PrefillExecution, PrefillReport};
mod engram_gpu;
mod prefill;
pub use prefill::DEFAULT_CHUNK_ROWS as PREFILL_CHUNK_ROWS;
pub use prefill::MAX_CHUNK_ROWS as MAX_PREFILL_CHUNK_ROWS;
mod snapshot;
pub use snapshot::Snapshot;
pub mod error;
pub use error::StateError;
pub mod capture;
mod ffn_gpu;
mod gpu_step;
mod hc_gpu;
mod memory;
pub(super) mod preflight;
pub use preflight::{ColdPrefillPreflight, preflight_cold_prefill};
pub mod profile;
mod token_map;
pub use memory::RequestMemoryEstimate;
pub use token_map::build_compressed_token_map;

use super::{engram, gguf::ResidentWeights, operators as op};
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
    Disk(&'a super::gguf::DiskEngram),
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

/// Runtime geometry comes from GGUF metadata. Default is the base 384-expert
/// architecture and is useful for estimates and reduced test fixtures.
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
            experts: 384,
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
    native_control: Option<MetalTensor>,
    input: usize,
    output: usize,
}
fn native_control(ctx: &MetalContext, dtype: GgmlType) -> Result<Option<MetalTensor>> {
    op::require(
        dense_linear_supported(dtype),
        &format!("unsupported NativeF32 linear storage: {dtype:?}"),
    )?;
    if matches!(
        dtype,
        GgmlType::F32
            | GgmlType::F16
            | GgmlType::BF16
            | GgmlType::IQ2_XXS
            | GgmlType::IQ1_S
            | GgmlType::IQ1_M
    ) {
        return Ok(None);
    }
    MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&[0i32, metal::ROUTE_STATUS_READY]),
        vec![2],
        GgmlType::I32,
    )
    .map(Some)
    .map_err(|e| e.to_string())
}

pub(crate) fn dense_linear_supported(dtype: GgmlType) -> bool {
    metal::all_slots_down_native_supported(dtype)
        || matches!(dtype, GgmlType::IQ2_XXS | GgmlType::IQ1_S | GgmlType::IQ1_M)
}

impl Linear<'_> {
    fn encode(
        &self,
        enc: &KernelEncoder,
        x: &MetalTensor,
        y: &MetalTensor,
    ) -> std::result::Result<(), metal::MetalError> {
        if x.dtype != GgmlType::F32
            || y.dtype != GgmlType::F32
            || x.n_elements() != self.input as u64
            || y.n_elements() != self.output as u64
        {
            return Err(metal::MetalError::BadShape {
                kernel: "v41_linear",
                detail: "linear requires matching F32 activation/output extents".into(),
            });
        }
        let x = x.view_bytes(0, vec![self.input as u64]);
        let y = y.view_bytes(0, vec![self.output as u64]);
        let direct: Option<
            fn(
                &MetalContext,
                &KernelEncoder,
                &MetalTensor,
                &MetalTensor,
                &MetalTensor,
                usize,
                usize,
            ) -> std::result::Result<(), metal::MetalError>,
        > = match self.tensor.dtype {
            GgmlType::F32 => Some(metal::encode_mat_vec_f32),
            GgmlType::F16 => Some(metal::encode_mat_vec_f16_f32),
            GgmlType::BF16 => Some(metal::encode_mat_vec_bf16_f32),
            GgmlType::IQ2_XXS => Some(metal::encode_mat_vec_iq2_xxs_f32),
            GgmlType::IQ1_S | GgmlType::IQ1_M => Some(metal::encode_mat_vec_iq1_f32),
            _ => None,
        };
        if let Some(encode) = direct {
            return encode(self.ctx, enc, &self.tensor, &x, &y, self.input, self.output);
        }
        if !metal::all_slots_down_native_supported(self.tensor.dtype) {
            return Err(metal::MetalError::BadShape {
                kernel: "v41_linear",
                detail: format!(
                    "unsupported NativeF32 linear storage: {:?}",
                    self.tensor.dtype
                ),
            });
        }
        // Native expert projection avoids fast matvec paths that narrow operands
        // to half. This is a single-bank projection, not matrix materialization.
        let control = self
            .native_control
            .as_ref()
            .expect("bound native projection control");
        let ids = control.view_subrange(0, vec![1]);
        let status = control.view_subrange(1, vec![1]);
        let bank = self
            .tensor
            .view_bytes(0, vec![self.input as u64, self.output as u64, 1]);
        let input = x.view_bytes(0, vec![self.input as u64, 1]);
        let output = y.view_bytes(0, vec![self.output as u64, 1]);
        metal::encode_all_slots_down_native(
            self.ctx,
            enc,
            &bank,
            &input,
            &ids,
            &status,
            &output,
            self.input,
            self.output,
            1,
            1,
        )
    }

    fn forward(&self, x: &[f32]) -> Result<Vec<f32>> {
        let dims = [self.input, self.output, 0, 1];
        let _wall = profile::Wall("linear", dims, profile::start());
        op::require(x.len() == self.input, "resident linear input shape")?;
        let input = upload(self.ctx, x, vec![self.input as u64])?;
        let output = MetalTensor::zeros_f32(self.ctx, vec![self.output as u64])
            .map_err(|e| e.to_string())?;
        submit(self.ctx, "linear", dims, |enc| {
            self.encode(enc, &input, &output)
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
                p.encode(enc, &input, &output)?;
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

fn row_bytes(t: &MetalTensor) -> Result<u64> {
    let (block, bytes) = t
        .dtype
        .storage_layout()
        .ok_or("unsupported tensor layout")?;
    let width = *t.shape.first().ok_or("empty tensor shape")?;
    op::require(width > 0 && width % block == 0, "unaligned tensor row")?;
    (width / block)
        .checked_mul(bytes)
        .ok_or_else(|| "row extent overflow".into())
}

fn decode_rows(t: &MetalTensor, first: usize, rows: usize) -> Result<Vec<f32>> {
    let stride = row_bytes(t)?;
    let available = t
        .shape
        .iter()
        .skip(1)
        .try_fold(1u64, |a, &b| a.checked_mul(b))
        .ok_or("row count overflow")?;
    op::require(
        (first as u64)
            .checked_add(rows as u64)
            .is_some_and(|end| end <= available),
        "row outside tensor view",
    )?;
    let size = stride
        .checked_mul(rows as u64)
        .ok_or("row extent overflow")?;
    let start = stride
        .checked_mul(first as u64)
        .and_then(|n| n.checked_add(t.offset))
        .ok_or("row offset overflow")?;
    let end = start.checked_add(size).ok_or("row extent overflow")?;
    op::require(
        t.buffer.storageMode() == objc2_metal::MTLStorageMode::Shared
            && end <= t.buffer.length() as u64,
        "invalid shared row extent",
    )?;
    let bytes = unsafe {
        std::slice::from_raw_parts(
            t.buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(start as usize),
            size as usize,
        )
    };
    let desc = crate::tensor::TensorDesc {
        name: "bounded forward rows".into(),
        shape: vec![t.shape[0], rows as u64],
        dtype: t.dtype,
        shard_idx: 0,
        data_offset: 0,
        n_bytes: size,
    };
    crate::codec::dequant_to_f32(&desc, bytes).map_err(|e| e.to_string())
}

struct Hc<'a> {
    projection: Linear<'a>,
    scale: [f32; 3],
    base: Vec<f32>,
    norm: Vec<f32>,
}
#[cfg(test)]
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
    gate: EngramGate,
}

enum EngramGate {
    Separate { q: Vec<f32>, k: Vec<f32> },
    Folded(Vec<f32>),
}

#[cfg(test)]
impl EngramGate {
    fn inject(&self, x: &mut [f32], kv: &[f32], eps: f32) -> Result<()> {
        match self {
            Self::Separate { q, k } => inject_engram(x, kv, q, k, eps),
            // Folded storage supplies the product directly. It is not claimed
            // to be bit-identical to separately rounded source q/k operands.
            Self::Folded(product) => inject_engram(x, kv, product, &vec![1.0; product.len()], eps),
        }
    }
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

/// Packed+dense resident bytes above this stay on the 2048-row chunk. Q2
/// (~186 GiB) and Maya-S (~166 GiB) fit the 4096-row envelope; IQ3 (~217 GiB)
/// does not have the stability margin measured for the larger chunk.
const FULL_CHUNK_RESIDENT_LIMIT: u64 = 200 * (1u64 << 30);

pub(super) fn prefill_rows_for_resident(resident_bytes: u64) -> usize {
    if resident_bytes <= FULL_CHUNK_RESIDENT_LIMIT {
        prefill::MAX_CHUNK_ROWS
    } else {
        prefill::DEFAULT_CHUNK_ROWS
    }
}

/// Bound once, without copying dense matrices or packed expert banks. Do not
/// mutate ResidentWeights buffers while this model is alive.
pub struct Forward<'a> {
    ctx: &'a MetalContext,
    engram_rows: EngramRows<'a>,
    engram_hash: engram::HashConfig,
    schedule: super::gguf::Schedule,
    config: Config,
    token_map: Vec<u32>,
    pad: u32,
    layers: Vec<Layer<'a>>,
    embedding: &'a MetalTensor,
    norm: Vec<f32>,
    head: Linear<'a>,
    prefill_rows: usize,
    identity: Arc<()>,
}

/// A failed/cancelled partial token poisons the sequence, rather than silently
/// reusing partially advanced KV caches. Start a fresh State to retry a prompt.
pub struct State {
    gpu_step: Option<gpu_step::Workspace>,
    #[cfg(test)]
    attention: attention::Session,
    addresses: engram::AddressState,
    max_context: usize,
    phase: StatePhase,
    prefill_origin: PrefillExecution,
    identity: Arc<()>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StatePhase {
    Ready,
    Provisional,
    Poisoned,
}
impl State {
    pub fn capacity(&self) -> usize {
        self.max_context
    }

    pub fn position(&self) -> usize {
        self.addresses.position() as usize
    }
    pub fn is_poisoned(&self) -> bool {
        self.phase == StatePhase::Poisoned
    }

    pub fn is_ready(&self) -> bool {
        self.phase == StatePhase::Ready
    }

    pub fn prefill_origin(&self) -> PrefillExecution {
        self.prefill_origin
    }
}

impl<'a> Forward<'a> {
    pub fn new_state(&self, max_context: usize) -> std::result::Result<State, StateError> {
        self.new_state_with_gpu_step(max_context)
    }

    #[cfg(test)]
    fn new_cpu_state(&self, max_context: usize) -> Result<State> {
        op::require(
            self.schedule == super::gguf::Schedule::default(),
            "CPU test oracle requires base attention schedule",
        )?;
        self.empty_state(max_context)
    }

    fn empty_state(&self, max_context: usize) -> Result<State> {
        op::require(
            max_context > 0 && max_context as u64 <= self.engram_hash.max_position,
            "invalid maximum context",
        )?;
        Ok(State {
            gpu_step: None,
            #[cfg(test)]
            attention: attention::Session::default(),
            addresses: engram::AddressState::new_with_config(self.pad, &self.engram_hash)
                .map_err(|e| format!("{e:?}"))?,
            max_context,
            phase: StatePhase::Ready,
            prefill_origin: PrefillExecution::FullForward,
            identity: Arc::clone(&self.identity),
        })
    }

    /// GPU token graph used by the production default.
    /// Capacity is the actual requested token capacity, not model metadata.
    pub fn new_state_with_gpu_step(
        &self,
        capacity: usize,
    ) -> std::result::Result<State, StateError> {
        let mut state = self.empty_state(capacity)?;
        state.gpu_step = Some(gpu_step::Workspace::new(self, capacity)?);
        Ok(state)
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

    /// Consume a token without computing unused vocabulary logits.
    pub fn advance(
        &self,
        state: &mut State,
        token: u32,
        cancel: Option<&AtomicBool>,
    ) -> Result<()> {
        self.execute_token(state, token, cancel, None, None, false)
            .map(|_| ())
    }

    /// Consume a nonempty suffix and return logits only for its final token.
    pub fn prefill(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> std::result::Result<Vec<f32>, StateError> {
        prefill::run(
            self,
            state,
            tokens,
            cancel,
            self.prefill_rows,
            true,
        )
    }

    /// Capture actual batched attention inputs and checkpoint outputs after GPU
    /// completion. Diagnostic copies do not change the prefill arithmetic path.
    pub fn prefill_with_capture(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
        capture: &mut dyn capture::Capture,
    ) -> std::result::Result<Vec<f32>, StateError> {
        prefill::run_captured(
            self,
            state,
            tokens,
            cancel,
            self.prefill_rows,
            true,
            Some(capture),
        )
    }

    /// Consume a nonempty suffix without computing vocabulary logits.
    pub fn advance_prefill(
        &self,
        state: &mut State,
        tokens: &[u32],
        cancel: Option<&AtomicBool>,
    ) -> std::result::Result<(), StateError> {
        prefill::run(
            self,
            state,
            tokens,
            cancel,
            self.prefill_rows,
            false,
        )
        .map(|_| ())
    }

    /// Synchronously borrow each completed block's CPU F32 outputs. None adds
    /// no readbacks, copies, or allocations. Callback errors poison a partially
    /// executed sequence just like operator errors; callback panics also leave
    /// it poisoned. The callback may set cancel to stop before the next layer.
    /// GPU-step mode captures on device and delivers callbacks in layer order
    /// after each segment completes; cancellation cannot undo that segment.
    pub fn step_with_trace(
        &self,
        state: &mut State,
        token: u32,
        cancel: Option<&AtomicBool>,
        trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
    ) -> Result<Vec<f32>> {
        self.step_with_capture(state, token, cancel, trace, None)
    }

    /// Capture normalized attention inputs and selected checkpoint outputs.
    /// Requires GPU-step state. None performs no capture allocation or copies.
    pub fn step_with_capture(
        &self,
        state: &mut State,
        token: u32,
        cancel: Option<&AtomicBool>,
        trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
        capture: Option<&mut dyn capture::Capture>,
    ) -> Result<Vec<f32>> {
        self.execute_token(state, token, cancel, trace, capture, true)
    }

    fn execute_token(
        &self,
        state: &mut State,
        token: u32,
        cancel: Option<&AtomicBool>,
        trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
        capture: Option<&mut dyn capture::Capture>,
        with_logits: bool,
    ) -> Result<Vec<f32>> {
        op::require(
            capture.is_none() || state.gpu_step.is_some(),
            "attention capture requires GPU step",
        )?;
        op::require(
            state.is_ready(),
            "sequence is provisional or poisoned by partial execution",
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
            .step_with_config(engram::Token::Compressed(compressed), &self.engram_hash)
            .map_err(|e| format!("{e:?}"))?;
        state.phase = StatePhase::Poisoned;
        // The synchronous step returns CPU values; persistent device state owns
        // Retained buffers. Completed commands must not outlive the priced step.
        let result = objc2::rc::autoreleasepool(|_| {
            self.step_inner(
                state,
                token as usize,
                &rows,
                cancel,
                trace,
                capture,
                with_logits,
            )
        })?;
        state.addresses = next;
        state.phase = StatePhase::Ready;
        Ok(result)
    }

    fn step_inner(
        &self,
        state: &mut State,
        token: usize,
        rows: &engram::LayerRows,
        cancel: Option<&AtomicBool>,
        trace: Option<&mut dyn FnMut(LayerTrace<'_>) -> Result<()>>,
        capture: Option<&mut dyn capture::Capture>,
        with_logits: bool,
    ) -> Result<Vec<f32>> {
        #[cfg(test)]
        let c = &self.config;
        let position = state.position();
        if let Some(workspace) = state.gpu_step.as_mut() {
            return workspace.run(
                self,
                position,
                token,
                rows,
                cancel,
                trace,
                capture,
                with_logits,
            );
        }
        #[cfg(not(test))]
        {
            Err("missing GPU step workspace".into())
        }
        #[cfg(test)]
        {
            let mut trace = trace;
            let embedding = decode_rows(self.embedding, token, 1)?;
            let mut x = embedding.repeat(4);
            let mut pre = vec![1.0, 0.0, 0.0, 0.0];
            for (id, layer) in self.layers.iter().enumerate() {
                cancelled(cancel)?;
                if let Some(e) = &layer.engram {
                    let hashes = rows.get(&id).ok_or("missing Engram hashes")?;
                    let values = self.engram_rows.decode(id, hashes)?;
                    let kv = e.projection.forward(&values)?;
                    e.gate.inject(&mut x, &kv, c.norm_eps)?;
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
            if !with_logits {
                return Ok(Vec::new());
            }
            let hidden = op::rms_norm(&op::collapse(&x, &pre)?, &self.norm, c.norm_eps)?;
            let logits = self.head.forward(&hidden)?;
            cancelled(cancel)?;
            op::require(logits.iter().all(|v| v.is_finite()), "non-finite logits")?;
            Ok(logits)
        }
    }

    #[cfg(test)]
    fn moe(ctx: &MetalContext, c: &Config, layer: &Layer<'_>, x: &[f32]) -> Result<Vec<f32>> {
        let (ids, weights) = route(
            &layer.router.forward(x)?,
            &layer.bias,
            c.topk,
            c.gate_temperature,
            c.route_scale,
        )?;
        let (banks, experts) = (layer.banks, c.experts);
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
        let dims = [c.hidden, c.intermediate, experts, c.topk];
        let wall = profile::Wall("native_gate_up", dims, profile::start());
        submit(ctx, "native_gate_up", dims, |enc| {
            metal::encode_all_slots_gate_up_swiglu_native(
                ctx,
                enc,
                banks[0],
                banks[1],
                &input,
                &ids_tensor,
                &status,
                &activation,
                c.hidden,
                c.intermediate,
                experts,
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
        let dims = [c.intermediate, c.hidden, experts, c.topk];
        let wall = profile::Wall("native_down", dims, profile::start());
        submit(ctx, "native_down", dims, |enc| {
            metal::encode_all_slots_down_native(
                ctx,
                enc,
                banks[2],
                &weighted,
                &ids_tensor,
                &status,
                &down,
                c.intermediate,
                c.hidden,
                experts,
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

#[cfg(test)]
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

#[cfg(test)]
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
