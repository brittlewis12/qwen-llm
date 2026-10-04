//! Native Metal decode for GLM-5.3-Flash: serial tokens with dense attention
//! (visible length below the sparse frontier, 2052).
//!
//! Composition only: every operation is a family-neutral, allocation-free
//! encoder in [`crate::metal`]. One command buffer per token; per-layer route
//! records are checked after completion, and any failure poisons the session
//! (KDA state may already have advanced, so the position cannot be retried).
//! Reference semantics: llama.cpp `src/models/glm5-next.cpp`.

use crate::gguf::GgufFile;
use crate::glm5_next::{
    ExecutionMode, FfnTensors, Glm5NextBlock, Glm5NextConfig, Glm5NextError, Glm5NextMemoryLedger,
    Glm5NextModel, MixerKind, MixerTensors,
};
use crate::metal::{
    KdaDecode, KernelEncoder, LATENT_NO_SINK, LearnedRoute, MetalContext, MetalError,
    MetalGgufBacking, MetalTensor, ROUTE_STATUS_READY, RetainedStorageDisposition, RouteScore,
    encode_add_f32, encode_all_slots_down, encode_all_slots_gate_up_swiglu, encode_clamped_swiglu,
    encode_get_rows_f32, encode_indexer_append, encode_kda_decode, encode_latent_attention,
    encode_mat_vec_q8_0_grouped_f32, encode_mhc4_collapse, encode_mhc4_controls, encode_mhc4_post,
    encode_mhc4_repeat, encode_moe_weighted_sum_f32, encode_rms_norm_mul_f32, encode_route_learned,
    encode_scatter_offset_f32_to_f16, evaluate_metal_memory_admission, wait_completed,
};
use crate::tensor::{GgmlType, TensorDesc};
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLDevice};
use std::collections::HashMap;

#[derive(Debug, thiserror::Error)]
pub enum Glm5NextMetalError {
    #[error(transparent)]
    Model(#[from] Glm5NextError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error("GLM-5.3 Metal: {0}")]
    Invalid(String),
}

impl From<crate::metal_forward::MfError> for Glm5NextMetalError {
    fn from(error: crate::metal_forward::MfError) -> Self {
        match error {
            crate::metal_forward::MfError::Metal(error) => Self::Metal(error),
            other => Self::Invalid(other.to_string()),
        }
    }
}

pub type Result<T> = std::result::Result<T, Glm5NextMetalError>;

fn invalid<T>(detail: impl Into<String>) -> Result<T> {
    Err(Glm5NextMetalError::Invalid(detail.into()))
}

/// Page size and per-buffer cap for retained windows on this device.
fn retained_geometry(ctx: &MetalContext) -> Result<(usize, usize)> {
    Ok((
        crate::metal::host_page_size_bytes()?,
        ctx.device.maxBufferLength(),
    ))
}

/// Executed weights as read-only views of retained no-copy GGUF windows.
pub struct Glm5NextWeights {
    pub config: Glm5NextConfig,
    pub embedding: MetalTensor,
    pub output_norm: MetalTensor,
    pub output: MetalTensor,
    pub blocks: Vec<Glm5NextBlock<MetalTensor>>,
    pub retained_bytes: u64,
    _backings: Vec<MetalGgufBacking>,
}

impl Glm5NextWeights {
    /// Admits serial decode and the retained bytes, then maps every executed
    /// tensor (NextN never) without copying.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile) -> Result<Self> {
        let model = Glm5NextModel::from_gguf(gguf)?;
        model.validate_execution(ExecutionMode::SerialDecode)?;
        let (page, max_buffer) = retained_geometry(ctx)?;
        let (plan, retained_bytes) = model.plan_retained(gguf, page, max_buffer)?;
        let admission =
            evaluate_metal_memory_admission(retained_bytes, 0, ctx.memory_signals(), true);
        if !admission.admitted {
            return invalid(format!(
                "weights do not fit: reason={:?} required={retained_bytes}",
                admission.reason
            ));
        }
        let mut backings = Vec::with_capacity(plan.windows.len());
        for window in &plan.windows {
            let mmap = gguf
                .retained_shard_mmap(window.shard_idx)
                .ok_or_else(|| Glm5NextMetalError::Invalid("missing retained shard".into()))?;
            backings.push(ctx.gguf_no_copy_window(
                mmap,
                window.shard_idx,
                window.mmap_offset as usize,
                window.length,
                32,
            )?);
        }
        let requests = model.retained_tensors();
        let mut views = HashMap::with_capacity(requests.len());
        for (entry, desc) in plan.entries.iter().zip(&requests) {
            if entry.name != desc.name {
                return invalid("retained entry order drift");
            }
            let tensor = match entry.disposition {
                RetainedStorageDisposition::View { window_index, .. } => {
                    let (eligibility, tensor) = backings[window_index].tensor(desc)?;
                    tensor.ok_or_else(|| {
                        Glm5NextMetalError::Invalid(format!(
                            "{}: retained view rejected: {eligibility:?}",
                            desc.name
                        ))
                    })?
                }
                RetainedStorageDisposition::CopyFallback { .. } => MetalTensor::copied_gguf_weight(
                    ctx,
                    desc,
                    gguf.try_slice(desc).map_err(Glm5NextError::from)?,
                )?,
                _ => return invalid(format!("{}: alias dispositions are not used", desc.name)),
            };
            views.insert(desc.name.clone(), tensor);
        }
        let mut take = |desc: &&TensorDesc| -> Result<MetalTensor> {
            views.remove(&desc.name).ok_or_else(|| {
                Glm5NextMetalError::Invalid(format!("missing realized {}", desc.name))
            })
        };
        let embedding = take(&model.token_embedding)?;
        let output_norm = take(&model.output_norm)?;
        let output = take(&model.output)?;
        let blocks = model
            .blocks
            .iter()
            .map(|block| block.try_map(&mut take))
            .collect::<Result<Vec<_>>>()?;
        if !views.is_empty() {
            return invalid(format!("{} realized tensors unused", views.len()));
        }
        Ok(Self {
            config: model.config.clone(),
            embedding,
            output_norm,
            output,
            blocks,
            retained_bytes,
            _backings: backings,
        })
    }
}

/// Per-layer persistent state.
enum LayerState {
    Kda {
        conv: MetalTensor,
        state: MetalTensor,
    },
    Mla {
        latent: MetalTensor,
        pending: MetalTensor,
        pooled: MetalTensor,
    },
}

/// Route record views for one MoE layer, kept per layer so a failure in any
/// layer survives until the post-token check.
struct RouteRecord {
    ids: MetalTensor,
    weights: MetalTensor,
    status: MetalTensor,
}

/// Activations shared by every block (one region, reused per sub-block).
struct Scratch {
    token: MetalTensor,
    embedding: MetalTensor,
    residual: [MetalTensor; 2],
    ones: MetalTensor,
    quarter: MetalTensor,
    no_sink: MetalTensor,
    normalized: MetalTensor,
    mixes: MetalTensor,
    pre: MetalTensor,
    post: MetalTensor,
    comb: MetalTensor,
    collapsed: MetalTensor,
    normed: MetalTensor,
    block_out: MetalTensor,
    // KDA
    q: MetalTensor,
    k: MetalTensor,
    v: MetalTensor,
    rank_a: MetalTensor,
    raw_gate: MetalTensor,
    raw_beta: MetalTensor,
    rank_b: MetalTensor,
    output_gate: MetalTensor,
    kda_out: MetalTensor,
    // MLA
    query_a: MetalTensor,
    query_r: MetalTensor,
    query: MetalTensor,
    latent_raw: MetalTensor,
    latent: MetalTensor,
    query_latent: MetalTensor,
    output_latent: MetalTensor,
    heads_out: MetalTensor,
    index_key: MetalTensor,
    index_gate: MetalTensor,
    // FFN
    dense_gate: MetalTensor,
    dense_up: MetalTensor,
    router: MetalTensor,
    expert_inner: MetalTensor,
    expert_out: MetalTensor,
    routed: MetalTensor,
    shared_gate: MetalTensor,
    shared_up: MetalTensor,
    shared: MetalTensor,
    // head
    final_hidden: MetalTensor,
    final_normed: MetalTensor,
    logits: MetalTensor,
}

fn zeros(ctx: &MetalContext, shape: &[u64]) -> Result<MetalTensor> {
    Ok(MetalTensor::zeros_f32(ctx, shape.to_vec())?)
}

fn filled(ctx: &MetalContext, value: f32, shape: &[u64]) -> Result<MetalTensor> {
    let n = shape.iter().product::<u64>() as usize;
    Ok(MetalTensor::from_bytes(
        ctx,
        bytemuck::cast_slice(&vec![value; n]),
        shape.to_vec(),
        GgmlType::F32,
    )?)
}

fn zeros_typed(
    ctx: &MetalContext,
    dtype: GgmlType,
    shape: &[u64],
    element_bytes: usize,
) -> Result<MetalTensor> {
    let n = shape.iter().product::<u64>() as usize;
    Ok(MetalTensor::from_bytes(
        ctx,
        &vec![0u8; n * element_bytes],
        shape.to_vec(),
        dtype,
    )?)
}

/// Host copy of an F32 tensor after its command completed (shared storage).
pub fn read_f32(tensor: &MetalTensor) -> Vec<f32> {
    let n = tensor.n_elements() as usize;
    // SAFETY: shared-storage buffer, offset and length validated at creation,
    // and no command writing it is in flight when callers read.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            n,
        )
        .to_vec()
    }
}

fn read_i32(tensor: &MetalTensor) -> Vec<i32> {
    let n = tensor.n_elements() as usize;
    // SAFETY: as in `read_f32`.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>(),
            n,
        )
        .to_vec()
    }
}

/// Intermediate a [`Glm5NextSession::forward_observed`] caller may inspect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm5NextProbe {
    /// Residual streams [4096, 4] after the attention sub-block (`hc_attn_post`).
    AttentionResidual,
    /// Residual streams [4096, 4] after the block (`l_out`).
    BlockResidual,
}

pub struct Glm5NextSession<'w> {
    weights: &'w Glm5NextWeights,
    capacity: usize,
    position: usize,
    poisoned: bool,
    layers: Vec<LayerState>,
    routes: Vec<Option<RouteRecord>>,
    s: Scratch,
    ledger: Glm5NextMemoryLedger,
}

impl<'w> Glm5NextSession<'w> {
    /// Allocates all session state for `capacity` positions (dense range only)
    /// after admitting the ledger's session and decode terms.
    pub fn new(ctx: &MetalContext, weights: &'w Glm5NextWeights, capacity: usize) -> Result<Self> {
        let c = &weights.config;
        let frontier = c.sparse_frontier() as usize;
        if capacity == 0 || capacity >= frontier {
            return invalid(format!(
                "capacity {capacity} must be within the dense range 1..{frontier} until sparse selection lands"
            ));
        }
        let ledger = Glm5NextMemoryLedger::new(c, weights.retained_bytes, capacity as u64, 1)?;
        let session_bytes = ledger.phase_peaks().decode - weights.retained_bytes;
        let admission =
            evaluate_metal_memory_admission(session_bytes, 0, ctx.memory_signals(), true);
        if !admission.admitted {
            return invalid(format!(
                "session does not fit: reason={:?} required={session_bytes}",
                admission.reason
            ));
        }
        let h = c.hidden_size as u64;
        let width = c.kda_width() as u64;
        let heads = c.head_count as u64;
        let d = c.kda_head_dim as u64;
        let pools = capacity.div_ceil(c.indexer_pool as usize) as u64;
        let mut layers = Vec::with_capacity(c.blocks.len());
        let mut routes = Vec::with_capacity(c.blocks.len());
        for block in &c.blocks {
            layers.push(match block.mixer {
                MixerKind::Kda => LayerState::Kda {
                    conv: zeros(ctx, &[width, 3, 3])?,
                    state: zeros(ctx, &[d, d, heads])?,
                },
                MixerKind::Mla => LayerState::Mla {
                    latent: zeros_typed(
                        ctx,
                        GgmlType::F16,
                        &[c.kv_lora_rank as u64, capacity as u64],
                        2,
                    )?,
                    pending: zeros_typed(
                        ctx,
                        GgmlType::F16,
                        &[c.indexer_head_dim as u64, 2, c.indexer_pool as u64],
                        2,
                    )?,
                    pooled: zeros_typed(
                        ctx,
                        GgmlType::F16,
                        &[c.indexer_head_dim as u64, pools.max(1)],
                        2,
                    )?,
                },
            });
            routes.push(match block.ffn {
                crate::glm5_next::FfnKind::Dense => None,
                crate::glm5_next::FfnKind::Moe => Some(RouteRecord {
                    ids: zeros_typed(ctx, GgmlType::I32, &[c.expert_used_count as u64], 4)?,
                    weights: zeros(ctx, &[c.expert_used_count as u64])?,
                    status: zeros_typed(ctx, GgmlType::I32, &[1], 4)?,
                }),
            });
        }
        let k = c.expert_used_count as u64;
        let s = Scratch {
            token: zeros_typed(ctx, GgmlType::I32, &[1], 4)?,
            embedding: zeros(ctx, &[h])?,
            residual: [zeros(ctx, &[h, 4])?, zeros(ctx, &[h, 4])?],
            ones: filled(ctx, 1.0, &[c.hc_width() as u64])?,
            quarter: filled(ctx, 0.25, &[4])?,
            no_sink: filled(ctx, LATENT_NO_SINK, &[heads])?,
            normalized: zeros(ctx, &[c.hc_width() as u64])?,
            mixes: zeros(ctx, &[c.hc_mix_count() as u64])?,
            pre: zeros(ctx, &[4])?,
            post: zeros(ctx, &[4])?,
            comb: zeros(ctx, &[4, 4])?,
            collapsed: zeros(ctx, &[h])?,
            normed: zeros(ctx, &[h])?,
            block_out: zeros(ctx, &[h])?,
            q: zeros(ctx, &[width])?,
            k: zeros(ctx, &[width])?,
            v: zeros(ctx, &[width])?,
            rank_a: zeros(ctx, &[d])?,
            raw_gate: zeros(ctx, &[width])?,
            raw_beta: zeros(ctx, &[heads])?,
            rank_b: zeros(ctx, &[d])?,
            output_gate: zeros(ctx, &[width])?,
            kda_out: zeros(ctx, &[width])?,
            query_a: zeros(ctx, &[c.q_lora_rank as u64])?,
            query_r: zeros(ctx, &[c.q_lora_rank as u64])?,
            query: zeros(ctx, &[c.mla_width() as u64])?,
            latent_raw: zeros(ctx, &[c.kv_lora_rank as u64])?,
            latent: zeros(ctx, &[c.kv_lora_rank as u64])?,
            query_latent: zeros(ctx, &[c.kv_lora_rank as u64, heads, 1])?,
            output_latent: zeros(ctx, &[c.kv_lora_rank as u64, heads, 1])?,
            heads_out: zeros(ctx, &[c.mla_width() as u64])?,
            index_key: zeros(ctx, &[c.indexer_head_dim as u64])?,
            index_gate: zeros(ctx, &[c.indexer_head_dim as u64])?,
            dense_gate: zeros(ctx, &[c.dense_ffn_size as u64])?,
            dense_up: zeros(ctx, &[c.dense_ffn_size as u64])?,
            router: zeros(ctx, &[c.expert_count as u64])?,
            expert_inner: zeros(ctx, &[c.expert_ffn_size as u64, k])?,
            expert_out: zeros(ctx, &[h, k])?,
            routed: zeros(ctx, &[h])?,
            shared_gate: zeros(ctx, &[c.shared_expert_ffn_size as u64])?,
            shared_up: zeros(ctx, &[c.shared_expert_ffn_size as u64])?,
            shared: zeros(ctx, &[h])?,
            final_hidden: zeros(ctx, &[h])?,
            final_normed: zeros(ctx, &[h])?,
            logits: zeros(ctx, &[c.vocab_size as u64])?,
        };
        Ok(Self {
            weights,
            capacity,
            position: 0,
            poisoned: false,
            layers,
            routes,
            s,
            ledger,
        })
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn ledger(&self) -> &Glm5NextMemoryLedger {
        &self.ledger
    }

    /// Decode one token at the next position; returns full-vocabulary logits.
    pub fn forward(&mut self, ctx: &MetalContext, token: u32) -> Result<Vec<f32>> {
        self.forward_observed(ctx, token, &[], &mut |_, _, _| {})
    }

    /// Like [`Self::forward`], committing once per block and reporting the
    /// requested probes after each block (diagnostics; same arithmetic).
    pub fn forward_observed(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        probes: &[Glm5NextProbe],
        observer: &mut dyn FnMut(Glm5NextProbe, usize, &[f32]),
    ) -> Result<Vec<f32>> {
        if self.poisoned {
            return invalid("session is poisoned by an earlier failed token");
        }
        let c = &self.weights.config;
        if token >= c.vocab_size {
            return invalid(format!("token {token} outside vocabulary"));
        }
        if self.position >= self.capacity {
            return invalid(format!(
                "position {} reaches capacity {}",
                self.position, self.capacity
            ));
        }
        let result = self.encode_token(ctx, token, probes, observer);
        match result {
            Ok(()) => {
                for (layer, route) in self.routes.iter().enumerate() {
                    if let Some(route) = route {
                        let status = read_i32(&route.status)[0];
                        if status != ROUTE_STATUS_READY {
                            self.poisoned = true;
                            return invalid(format!(
                                "block {layer} route failed with status {status}"
                            ));
                        }
                    }
                }
                self.position += 1;
                Ok(read_f32(&self.s.logits))
            }
            Err(error) => {
                self.poisoned = true;
                Err(error)
            }
        }
    }

    fn encode_token(
        &self,
        ctx: &MetalContext,
        token: u32,
        probes: &[Glm5NextProbe],
        observer: &mut dyn FnMut(Glm5NextProbe, usize, &[f32]),
    ) -> Result<()> {
        let w = self.weights;
        let c = &w.config;
        let s = &self.s;
        let h = c.hidden_size as usize;
        // SAFETY: shared-storage I32 [1]; no command is in flight.
        unsafe {
            *s.token
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(s.token.offset as usize)
                .cast::<i32>() = token as i32;
        }
        let observed = !probes.is_empty();
        let mut command = ctx
            .queue
            .commandBuffer()
            .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
        let mut enc = KernelEncoder::begin(&command);
        encode_get_rows_f32(ctx, &enc, &w.embedding, &s.token, &s.embedding, 1, h)?;
        encode_mhc4_repeat(ctx, &enc, h, &s.embedding, &s.residual[0])?;
        for (index, block) in w.blocks.iter().enumerate() {
            let (a, b) = (&s.residual[0], &s.residual[1]);
            // Attention sub-block: a -> b.
            self.encode_hc_pre(
                ctx,
                &enc,
                a,
                &block.attention_hc.mix,
                &block.attention_hc.scale,
                &block.attention_hc.base,
            )?;
            encode_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.collapsed,
                &block.attention_norm,
                &s.normed,
                c.rms_epsilon,
            )?;
            match (&block.mixer, &self.layers[index]) {
                (MixerTensors::Kda(kda), LayerState::Kda { conv, state }) => {
                    self.encode_kda(ctx, &enc, kda, conv, state)?
                }
                (
                    MixerTensors::Mla(mla),
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    },
                ) => self.encode_mla(ctx, &enc, mla, latent, pending, pooled)?,
                _ => return invalid(format!("block {index} state does not match its mixer")),
            }
            encode_mhc4_post(ctx, &enc, h, &s.block_out, a, &s.post, &s.comb, b)?;
            if observed && probes.contains(&Glm5NextProbe::AttentionResidual) {
                enc.end();
                command.commit();
                wait_completed(&command)?;
                observer(Glm5NextProbe::AttentionResidual, index, &read_f32(b));
                command = ctx
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
                enc = KernelEncoder::begin(&command);
            }
            // FFN sub-block: b -> a.
            self.encode_hc_pre(
                ctx,
                &enc,
                b,
                &block.ffn_hc.mix,
                &block.ffn_hc.scale,
                &block.ffn_hc.base,
            )?;
            encode_rms_norm_mul_f32(
                ctx,
                &enc,
                &s.collapsed,
                &block.ffn_norm,
                &s.normed,
                c.rms_epsilon,
            )?;
            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    let f = c.dense_ffn_size as usize;
                    matvec(ctx, &enc, &dense.gate, &s.normed, &s.dense_gate, h, f)?;
                    matvec(ctx, &enc, &dense.up, &s.normed, &s.dense_up, h, f)?;
                    encode_clamped_swiglu(
                        ctx,
                        &enc,
                        &s.dense_gate,
                        &s.dense_up,
                        &s.dense_gate,
                        c.swiglu_clamp,
                    )?;
                    matvec(ctx, &enc, &dense.down, &s.dense_gate, &s.block_out, f, h)?;
                }
                FfnTensors::Moe(moe) => {
                    let route = self.routes[index].as_ref().ok_or_else(|| {
                        Glm5NextMetalError::Invalid(format!("block {index} has no route record"))
                    })?;
                    let (e, f, k) = (
                        c.expert_count as usize,
                        c.expert_ffn_size as usize,
                        c.expert_used_count as usize,
                    );
                    matvec(ctx, &enc, &moe.router, &s.normed, &s.router, h, e)?;
                    let spec = LearnedRoute {
                        experts: e,
                        top_k: k,
                        score: RouteScore::Sigmoid,
                        routed_scale: c.expert_weights_scale,
                    };
                    encode_route_learned(
                        ctx,
                        &enc,
                        &spec,
                        &s.router,
                        &moe.selection_bias,
                        &route.ids,
                        &route.weights,
                        &route.status,
                    )?;
                    encode_all_slots_gate_up_swiglu(
                        ctx,
                        &enc,
                        &moe.gate_experts,
                        &moe.up_experts,
                        &s.normed,
                        &route.ids,
                        &route.status,
                        &s.expert_inner,
                        h,
                        f,
                        e,
                        k,
                        c.swiglu_clamp,
                    )?;
                    encode_all_slots_down(
                        ctx,
                        &enc,
                        &moe.down_experts,
                        &s.expert_inner,
                        &route.ids,
                        &route.status,
                        &s.expert_out,
                        f,
                        h,
                        e,
                        k,
                    )?;
                    encode_moe_weighted_sum_f32(
                        ctx,
                        &enc,
                        &s.expert_out,
                        &route.weights,
                        &s.routed,
                        h,
                        k,
                    )?;
                    let sf = c.shared_expert_ffn_size as usize;
                    matvec(
                        ctx,
                        &enc,
                        &moe.shared.gate,
                        &s.normed,
                        &s.shared_gate,
                        h,
                        sf,
                    )?;
                    matvec(ctx, &enc, &moe.shared.up, &s.normed, &s.shared_up, h, sf)?;
                    encode_clamped_swiglu(
                        ctx,
                        &enc,
                        &s.shared_gate,
                        &s.shared_up,
                        &s.shared_gate,
                        c.swiglu_clamp,
                    )?;
                    matvec(
                        ctx,
                        &enc,
                        &moe.shared.down,
                        &s.shared_gate,
                        &s.shared,
                        sf,
                        h,
                    )?;
                    encode_add_f32(ctx, &enc, &s.routed, &s.shared, &s.block_out)?;
                }
            }
            encode_mhc4_post(ctx, &enc, h, &s.block_out, b, &s.post, &s.comb, a)?;
            if observed && probes.contains(&Glm5NextProbe::BlockResidual) {
                enc.end();
                command.commit();
                wait_completed(&command)?;
                observer(Glm5NextProbe::BlockResidual, index, &read_f32(a));
                command = ctx
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
                enc = KernelEncoder::begin(&command);
            }
        }
        // Head: mean of the four streams, output norm, logits.
        encode_mhc4_collapse(ctx, &enc, h, &s.residual[0], &s.quarter, &s.final_hidden)?;
        encode_rms_norm_mul_f32(
            ctx,
            &enc,
            &s.final_hidden,
            &w.output_norm,
            &s.final_normed,
            c.rms_epsilon,
        )?;
        matvec(
            ctx,
            &enc,
            &w.output,
            &s.final_normed,
            &s.logits,
            h,
            c.vocab_size as usize,
        )?;
        enc.end();
        command.commit();
        wait_completed(&command)?;
        Ok(())
    }

    /// Flattened unweighted RMSNorm, mix projection, controls and collapse into
    /// `collapsed`; `post`/`comb` stay set for the matching post.
    fn encode_hc_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        mix: &MetalTensor,
        scale: &MetalTensor,
        base: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let s = &self.s;
        encode_rms_norm_mul_f32(ctx, enc, residual, &s.ones, &s.normalized, c.rms_epsilon)?;
        matvec(
            ctx,
            enc,
            mix,
            &s.normalized,
            &s.mixes,
            c.hc_width() as usize,
            c.hc_mix_count() as usize,
        )?;
        encode_mhc4_controls(
            ctx,
            enc,
            c.hc_epsilon,
            &s.mixes,
            scale,
            base,
            &s.pre,
            &s.post,
            &s.comb,
        )?;
        encode_mhc4_collapse(
            ctx,
            enc,
            c.hidden_size as usize,
            residual,
            &s.pre,
            &s.collapsed,
        )?;
        Ok(())
    }

    fn encode_kda(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        kda: &crate::glm5_next::KdaTensors<MetalTensor>,
        conv: &MetalTensor,
        state: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let s = &self.s;
        let (h, width, rank) = (
            c.hidden_size as usize,
            c.kda_width() as usize,
            c.kda_head_dim as usize,
        );
        matvec(ctx, enc, &kda.query, &s.normed, &s.q, h, width)?;
        matvec(ctx, enc, &kda.key, &s.normed, &s.k, h, width)?;
        matvec(ctx, enc, &kda.value, &s.normed, &s.v, h, width)?;
        matvec(ctx, enc, &kda.decay_a, &s.normed, &s.rank_a, h, rank)?;
        matvec(ctx, enc, &kda.decay_b, &s.rank_a, &s.raw_gate, rank, width)?;
        matvec(
            ctx,
            enc,
            &kda.beta,
            &s.normed,
            &s.raw_beta,
            h,
            c.head_count as usize,
        )?;
        matvec(ctx, enc, &kda.gate_a, &s.normed, &s.rank_b, h, rank)?;
        matvec(
            ctx,
            enc,
            &kda.gate_b,
            &s.rank_b,
            &s.output_gate,
            rank,
            width,
        )?;
        encode_kda_decode(
            ctx,
            enc,
            c.head_count as usize,
            &KdaDecode {
                q: &s.q,
                k: &s.k,
                v: &s.v,
                raw_gate: &s.raw_gate,
                raw_beta: &s.raw_beta,
                output_gate: &s.output_gate,
                q_conv: &kda.query_conv,
                k_conv: &kda.key_conv,
                v_conv: &kda.value_conv,
                neg_exp_a_log: &kda.neg_exp_a_log,
                dt_bias: &kda.decay_bias,
                output_norm: &kda.output_norm,
                conv_state: conv,
                state,
                out: &s.kda_out,
            },
            c.kda_gate_lower_bound,
            c.rms_epsilon,
        )?;
        matvec(ctx, enc, &kda.output, &s.kda_out, &s.block_out, width, h)?;
        Ok(())
    }

    fn encode_mla(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pending: &MetalTensor,
        pooled: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let s = &self.s;
        let h = c.hidden_size as usize;
        let (q_rank, kv) = (c.q_lora_rank as usize, c.kv_lora_rank as usize);
        let heads = c.head_count as usize;
        let head_dim = c.mla_key_head_dim as usize;
        matvec(ctx, enc, &mla.query_a, &s.normed, &s.query_a, h, q_rank)?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &s.query_a,
            &mla.query_a_norm,
            &s.query_r,
            c.rms_epsilon,
        )?;
        matvec(
            ctx,
            enc,
            &mla.query_b,
            &s.query_r,
            &s.query,
            q_rank,
            heads * head_dim,
        )?;
        matvec(ctx, enc, &mla.latent, &s.normed, &s.latent_raw, h, kv)?;
        encode_rms_norm_mul_f32(
            ctx,
            enc,
            &s.latent_raw,
            &mla.latent_norm,
            &s.latent,
            c.rms_epsilon,
        )?;
        encode_scatter_offset_f32_to_f16(ctx, enc, &s.latent, latent, self.position * kv, kv)?;
        encode_mat_vec_q8_0_grouped_f32(
            ctx,
            enc,
            &mla.key_absorb,
            &s.query,
            &s.query_latent,
            head_dim,
            kv,
            heads,
        )?;
        encode_latent_attention(
            ctx,
            enc,
            &s.query_latent,
            latent,
            &s.no_sink,
            &s.output_latent,
            self.position,
            1,
            1.0 / (head_dim as f32).sqrt(),
        )?;
        encode_mat_vec_q8_0_grouped_f32(
            ctx,
            enc,
            &mla.value_expand,
            &s.output_latent,
            &s.heads_out,
            kv,
            head_dim,
            heads,
        )?;
        matvec(
            ctx,
            enc,
            &mla.output,
            &s.heads_out,
            &s.block_out,
            heads * head_dim,
            h,
        )?;
        // Indexer cache maintenance (selection starts at the sparse frontier).
        let index_dim = c.indexer_head_dim as usize;
        matvec(
            ctx,
            enc,
            &mla.indexer.key,
            &s.normed,
            &s.index_key,
            h,
            index_dim,
        )?;
        matvec(
            ctx,
            enc,
            &mla.indexer.pool_gate,
            &s.normed,
            &s.index_gate,
            h,
            index_dim,
        )?;
        encode_indexer_append(
            ctx,
            enc,
            &s.index_key,
            &s.index_gate,
            &mla.indexer.key_norm,
            &mla.indexer.key_norm_bias,
            &mla.indexer.pool_position,
            pending,
            pooled,
            self.position,
            c.layer_norm_epsilon,
        )?;
        Ok(())
    }
}

fn matvec(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
) -> Result<()> {
    crate::metal_forward::encode_mat_vec_dispatch(ctx, enc, weight, x, y, n_in, n_out)?;
    Ok(())
}

#[cfg(test)]
mod tests;
