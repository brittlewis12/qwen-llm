//! Native Metal execution for GLM-5.3-Flash: serial decode and packed
//! prefill (`packed`), with dense latent attention below visible length 2052
//! and sparse DSA selection (indexer scores, top-512 pools plus the
//! incomplete pool's tail, selected attention) from there on.
//!
//! Composition only: every operation is a family-neutral, allocation-free
//! encoder in [`crate::metal`]. One command buffer per token; per-layer route
//! records are checked after completion, and any failure poisons the session
//! (KDA state may already have advanced, so the position cannot be retried).
//! Reference semantics: llama.cpp `src/models/glm5-next.cpp`.

use crate::gguf::GgufFile;
use crate::glm5_next::memory::{self, BufferSpec, BufferType};
use crate::glm5_next::{
    ExecutionMode, FfnKind, FfnTensors, Glm5NextBlock, Glm5NextConfig, Glm5NextError,
    Glm5NextMemoryLedger, Glm5NextModel, MixerKind, MixerTensors,
};
use crate::metal::{
    KdaDecode, KernelEncoder, LearnedRoute, MetalContext, MetalError, MetalGgufBacking,
    MetalTensor, ROUTE_STATUS_READY, RetainedStorageDisposition, RouteScore, encode_add_f32,
    encode_all_slots_down, encode_all_slots_gate_up_swiglu, encode_clamped_swiglu,
    encode_get_rows_f32, encode_indexer_append, encode_kda_decode, encode_latent_attention,
    encode_mat_vec_q8_0_grouped_f32, encode_mhc4_collapse, encode_mhc4_post, encode_mhc4_repeat,
    encode_moe_weighted_sum_f32, encode_rms_norm_mul_f32, encode_route_learned,
    encode_scatter_offset_f32_to_f16, evaluate_metal_memory_admission,
    evaluate_metal_memory_admission_with_cpu_bytes, wait_completed,
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
    /// The caller's checkpoint stopped a multi-step operation at a boundary
    /// where the session (if any) is consistent and unpoisoned.
    #[error("GLM-5.3 cancelled: {0}")]
    Cancelled(String),
    /// The requested session was refused by the device memory admission.
    /// `denied` (the error source) keeps the typed decision: pressure,
    /// telemetry or size. `advice` is evaluated for pressure refusals only.
    #[error(
        "GLM-5.3 session refused by memory admission ({}); budget {budget_bytes} bytes; {advice}",
        denied.reason.as_str()
    )]
    MemoryAdmission {
        #[source]
        denied: crate::metal::MemoryAdmissionDenied,
        budget_bytes: u64,
        advice: CapacityAdvice,
    },
    /// A shared encoder or command failed (GPU command status, a poisoned
    /// shared session, dispatch geometry), kept typed.
    #[error(transparent)]
    Forward(crate::metal_forward::MfError),
    /// An earlier token failed after its state began to change; the session
    /// can only be dropped.
    #[error("GLM-5.3 session is poisoned by an earlier failed token")]
    Poisoned,
    /// A snapshot cannot be restored into this session (another weights
    /// instance, lineage or policy version, no room, or a region mismatch).
    /// Nothing was written.
    #[error("GLM-5.3 snapshot refused: {0}")]
    SnapshotMismatch(String),
    /// A kernel reported a validation status for one block (and row, in a
    /// packed chunk): routing or sparse selection did not complete.
    #[error(
        "GLM-5.3 {stage} kernel validation failed in block {block}{} with status {status}",
        row.map(|r| format!(" row {r}")).unwrap_or_default()
    )]
    KernelValidation {
        stage: &'static str,
        /// Model block index (MLA selector slots are mapped to blocks).
        block: usize,
        row: Option<usize>,
        status: i32,
    },
}

/// What a refused session's budget would fit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CapacityAdvice {
    /// No smaller capacity was evaluated (a telemetry or size refusal, or a
    /// per-request session whose capacity is fixed).
    NotEvaluated,
    /// Evaluated: no capacity fits the budget.
    NoneFits,
    /// Evaluated: the largest capacity (positions) that fits.
    Fits(u64),
}

impl std::fmt::Display for CapacityAdvice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotEvaluated => f.write_str("no smaller capacity was evaluated"),
            Self::NoneFits => f.write_str("no capacity fits"),
            Self::Fits(n) => write!(f, "the largest capacity that fits is {n} positions"),
        }
    }
}

impl From<crate::metal_forward::MfError> for Glm5NextMetalError {
    fn from(error: crate::metal_forward::MfError) -> Self {
        match error {
            crate::metal_forward::MfError::Metal(error) => Self::Metal(error),
            other => Self::Forward(other),
        }
    }
}

impl Glm5NextMetalError {
    /// A device memory refusal caused by memory pressure (retryable once
    /// memory frees), as opposed to telemetry, size or any other failure.
    pub fn is_memory_pressure(&self) -> bool {
        matches!(self, Self::MemoryAdmission { denied, .. } if denied.reason.is_pressure())
    }
}

/// Budget a refused admission saw: working-set headroom, capped by the
/// process limit when the process reports one.
fn admission_budget(admission: &crate::metal::MetalMemoryAdmission) -> u64 {
    let headroom = admission.working_set_headroom_bytes.unwrap_or(0);
    match admission.signals.process_limit_remaining_bytes {
        Some(process) if process > 0 => headroom.min(process),
        _ => headroom,
    }
}

/// Model block index of the `index`th MLA block.
fn mla_block(config: &crate::glm5_next::Glm5NextConfig, index: usize) -> usize {
    config
        .blocks
        .iter()
        .enumerate()
        .filter(|(_, block)| block.mixer == MixerKind::Mla)
        .nth(index)
        .map_or(usize::MAX, |(block, _)| block)
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

/// Default packed-prefill chunk (rows per command buffer).
pub const DEFAULT_PREFILL_ROWS: usize = 512;

/// The device-priced plan of an admitted session (see [`preflight_session`]).
#[derive(Clone, Copy, Debug)]
pub struct Glm5NextPreflight {
    pub ledger: Glm5NextMemoryLedger,
}

/// Admits a session of `capacity` positions with `prefill_rows`-row packed
/// chunks against current device memory before any weight is mapped or
/// prefetched: retained windows, session buffers and reserve, priced with the
/// device. On refusal, reports the largest capacity that fits the same
/// budget (the smaller of working-set headroom and the process limit).
pub fn preflight_session(
    ctx: &MetalContext,
    gguf: &GgufFile,
    model: &Glm5NextModel<'_>,
    capacity: usize,
    prefill_rows: usize,
) -> Result<Glm5NextPreflight> {
    let (page, max_buffer) = retained_geometry(ctx)?;
    let (_, retained) = model.plan_retained(gguf, page, max_buffer)?;
    let price = device_price(ctx);
    let ledger = Glm5NextMemoryLedger::new(
        &model.config,
        retained,
        capacity as u64,
        prefill_rows as u64,
        &price,
    )?;
    let admission =
        evaluate_metal_memory_admission(ledger.peak_bytes(), 0, ctx.memory_signals(), true);
    let Some(denied) = admission.refusal() else {
        return Ok(Glm5NextPreflight { ledger });
    };
    let budget = admission_budget(&admission);
    // A smaller capacity is advice only when memory, not telemetry, refused.
    let advice = if denied.reason.is_pressure() {
        match Glm5NextMemoryLedger::max_capacity(
            &model.config,
            retained,
            prefill_rows as u64,
            budget,
            &price,
        ) {
            Ok(Some(n)) => CapacityAdvice::Fits(n),
            Ok(None) => CapacityAdvice::NoneFits,
            Err(_) => CapacityAdvice::NotEvaluated,
        }
    } else {
        CapacityAdvice::NotEvaluated
    };
    Err(Glm5NextMetalError::MemoryAdmission {
        denied,
        budget_bytes: budget,
        advice,
    })
}

/// What [`prefetch_retained`] did.
#[derive(Clone, Copy, Debug, Default)]
pub struct RetainedPrefetch {
    pub windows: usize,
    pub cold_windows: usize,
    pub bytes_read: u64,
    pub wall: std::time::Duration,
}

/// Warms the page cache for retained windows whose sampled residency is below
/// `threshold`, with parallel reads, before the zero-copy weights are first
/// touched. Only executed tensors' windows are considered, so the never-read
/// NextN tail neither triggers nor receives prefetch.
pub fn prefetch_retained(
    ctx: &MetalContext,
    gguf: &GgufFile,
    threshold: f64,
) -> Result<RetainedPrefetch> {
    prefetch_retained_with_cancel(ctx, gguf, threshold, &|| false)
}

/// [`prefetch_retained`] that stops with [`Glm5NextMetalError::Cancelled`]
/// once `should_cancel` returns true (polled per window and per read chunk).
/// Prefetch only warms the page cache, so stopping leaves nothing to undo.
pub fn prefetch_retained_with_cancel(
    ctx: &MetalContext,
    gguf: &GgufFile,
    threshold: f64,
    should_cancel: &(dyn Fn() -> bool + Sync),
) -> Result<RetainedPrefetch> {
    let started = std::time::Instant::now();
    let model = Glm5NextModel::from_gguf(gguf)?;
    let (page, max_buffer) = retained_geometry(ctx)?;
    let (plan, _) = model.plan_retained(gguf, page, max_buffer)?;
    let mut report = RetainedPrefetch {
        windows: plan.windows.len(),
        ..Default::default()
    };
    for window in &plan.windows {
        if should_cancel() {
            return Err(Glm5NextMetalError::Cancelled("retained prefetch".into()));
        }
        let shard = gguf
            .shards
            .get(window.shard_idx)
            .ok_or_else(|| Glm5NextMetalError::Invalid("missing retained shard".into()))?;
        let start = window.mmap_offset as usize;
        let end = start
            .checked_add(window.length)
            .filter(|&end| end <= shard.mmap.len())
            .ok_or_else(|| Glm5NextMetalError::Invalid("window outside its shard".into()))?;
        let resident =
            crate::cache_probe::probe_mapped_range_residency_sampled(&shard.mmap[start..end])
                .map(|r| r.resident_fraction())
                .unwrap_or(0.0);
        if resident >= threshold {
            continue;
        }
        let read = crate::prefetch::prefetch_fd_range_with_cancel(
            &shard.file,
            start as u64,
            end as u64,
            crate::prefetch::DEFAULT_WORKERS,
            crate::prefetch::DEFAULT_CHUNK_BYTES,
            should_cancel,
        )
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::Interrupted => {
                Glm5NextMetalError::Cancelled("retained prefetch".into())
            }
            _ => Glm5NextMetalError::Invalid(format!("prefetch: {e}")),
        })?;
        report.cold_windows += 1;
        report.bytes_read += read.bytes;
    }
    report.wall = started.elapsed();
    Ok(report)
}

/// Executed weights as read-only views of retained no-copy GGUF windows.
pub struct Glm5NextWeights {
    pub config: Glm5NextConfig,
    pub embedding: MetalTensor,
    pub output_norm: MetalTensor,
    pub output: MetalTensor,
    pub blocks: Vec<Glm5NextBlock<MetalTensor>>,
    pub retained_bytes: u64,
    /// Whether every executed weight cell has a packed-prefill path.
    pub packed_prefill_admitted: bool,
    /// Process-unique id of this load (snapshot identity).
    instance: u64,
    _backings: Vec<MetalGgufBacking>,
}

/// Source of [`Glm5NextWeights::instance`] ids.
static WEIGHTS_INSTANCES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

impl Glm5NextWeights {
    /// The retained no-copy weight buffers (for idle residency keep-alive).
    pub fn retained_buffers(&self) -> Vec<&crate::metal::Buffer> {
        self._backings
            .iter()
            .map(|backing| &backing.buffer)
            .collect()
    }

    /// Admits serial decode and the retained bytes, then maps every executed
    /// tensor (NextN never) without copying.
    pub fn load(ctx: &MetalContext, gguf: &GgufFile) -> Result<Self> {
        let model = Glm5NextModel::from_gguf(gguf)?;
        model.validate_execution(ExecutionMode::SerialDecode)?;
        let packed_prefill_admitted = model
            .validate_execution(ExecutionMode::PackedPrefill)
            .is_ok();
        let (page, max_buffer) = retained_geometry(ctx)?;
        let (plan, retained_bytes) = model.plan_retained(gguf, page, max_buffer)?;
        // Admission and the allocations it prices form one transaction.
        let _allocation = ctx.begin_allocation_transaction();
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
            packed_prefill_admitted,
            instance: WEIGHTS_INSTANCES.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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
    quarter: MetalTensor,
    no_sink: MetalTensor,
    hc_partial_dots: MetalTensor,
    hc_partial_sumsq: MetalTensor,
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
    /// `[3 * shared_ffn]`: SwiGLU output, then gate and up.
    shared_swiglu: MetalTensor,
    shared: MetalTensor,
    // head
    final_hidden: MetalTensor,
    final_normed: MetalTensor,
    logits: MetalTensor,
}

/// Ledger pricer for live sessions: the device's shared-buffer allocation
/// upper bound.
fn device_price(ctx: &MetalContext) -> impl Fn(u64) -> Option<u64> + '_ {
    |bytes| {
        ctx.price_shared_buffer_upper(bytes)
            .ok()
            .map(|p| p.priced_upper_bytes)
    }
}

/// Session buffers realized from one ledger spec list. Each buffer must be
/// taken by name exactly once, so the ledger and the session cannot drift:
/// a missing name, a duplicate or an untaken buffer is an error.
struct SpecBuffers {
    owner: &'static str,
    buffers: HashMap<&'static str, MetalTensor>,
}

impl SpecBuffers {
    /// Allocates every spec byte-zeroed (or filled). The caller has admitted
    /// the ledger, which priced these same specs with [`device_price`].
    fn allocate(ctx: &MetalContext, owner: &'static str, specs: &[BufferSpec]) -> Result<Self> {
        let mut buffers = HashMap::with_capacity(specs.len());
        for spec in specs {
            let dtype = match spec.dtype {
                BufferType::F32 => GgmlType::F32,
                BufferType::F16 => GgmlType::F16,
                BufferType::I32 => GgmlType::I32,
            };
            let tensor = if spec.fill == 0.0 {
                MetalTensor::zeros_dtype_unstaged(ctx, spec.shape.clone(), dtype)?
            } else if dtype == GgmlType::F32 {
                let n = spec.shape.iter().product::<u64>() as usize;
                MetalTensor::from_bytes(
                    ctx,
                    bytemuck::cast_slice(&vec![spec.fill; n]),
                    spec.shape.clone(),
                    dtype,
                )?
            } else {
                return invalid(format!("{owner}.{}: fill requires F32", spec.name));
            };
            if buffers.insert(spec.name, tensor).is_some() {
                return invalid(format!("{owner}.{}: duplicate buffer spec", spec.name));
            }
        }
        Ok(Self { owner, buffers })
    }

    fn take(&mut self, name: &str) -> Result<MetalTensor> {
        self.buffers.remove(name).ok_or_else(|| {
            Glm5NextMetalError::Invalid(format!("{}.{name}: no such buffer spec", self.owner))
        })
    }

    fn finish(self) -> Result<()> {
        if self.buffers.is_empty() {
            return Ok(());
        }
        let mut unused: Vec<_> = self.buffers.into_keys().collect();
        unused.sort_unstable();
        invalid(format!("{}: unused buffer specs {unused:?}", self.owner))
    }
}

impl Scratch {
    fn new(ctx: &MetalContext, c: &Glm5NextConfig) -> Result<Self> {
        let mut b = SpecBuffers::allocate(ctx, "decode_scratch", &memory::decode_scratch_specs(c))?;
        let s = Self {
            token: b.take("token")?,
            embedding: b.take("embedding")?,
            residual: [b.take("residual_a")?, b.take("residual_b")?],
            quarter: b.take("quarter")?,
            no_sink: b.take("no_sink")?,
            hc_partial_dots: b.take("hc_partial_dots")?,
            hc_partial_sumsq: b.take("hc_partial_sumsq")?,
            mixes: b.take("mixes")?,
            pre: b.take("pre")?,
            post: b.take("post")?,
            comb: b.take("comb")?,
            collapsed: b.take("collapsed")?,
            normed: b.take("normed")?,
            block_out: b.take("block_out")?,
            q: b.take("q")?,
            k: b.take("k")?,
            v: b.take("v")?,
            rank_a: b.take("rank_a")?,
            raw_gate: b.take("raw_gate")?,
            raw_beta: b.take("raw_beta")?,
            rank_b: b.take("rank_b")?,
            output_gate: b.take("output_gate")?,
            kda_out: b.take("kda_out")?,
            query_a: b.take("query_a")?,
            query_r: b.take("query_r")?,
            query: b.take("query")?,
            latent_raw: b.take("latent_raw")?,
            latent: b.take("latent")?,
            query_latent: b.take("query_latent")?,
            output_latent: b.take("output_latent")?,
            heads_out: b.take("heads_out")?,
            index_key: b.take("index_key")?,
            index_gate: b.take("index_gate")?,
            dense_gate: b.take("dense_gate")?,
            dense_up: b.take("dense_up")?,
            router: b.take("router")?,
            expert_inner: b.take("expert_inner")?,
            expert_out: b.take("expert_out")?,
            routed: b.take("routed")?,
            shared_swiglu: b.take("shared_swiglu")?,
            shared: b.take("shared")?,
            final_hidden: b.take("final_hidden")?,
            final_normed: b.take("final_normed")?,
            logits: b.take("logits")?,
        };
        b.finish()?;
        Ok(s)
    }
}

impl LayerState {
    fn new(
        ctx: &MetalContext,
        c: &Glm5NextConfig,
        mixer: MixerKind,
        capacity: u64,
    ) -> Result<Self> {
        let state = match mixer {
            MixerKind::Kda => {
                let mut b = SpecBuffers::allocate(ctx, "kda_state", &memory::kda_state_specs(c))?;
                let state = Self::Kda {
                    conv: b.take("conv")?,
                    state: b.take("state")?,
                };
                b.finish()?;
                state
            }
            MixerKind::Mla => {
                let specs = memory::mla_state_specs(c, capacity);
                let mut b = SpecBuffers::allocate(ctx, "mla_state", &specs)?;
                let state = Self::Mla {
                    latent: b.take("latent")?,
                    pending: b.take("pending")?,
                    pooled: b.take("pooled")?,
                };
                b.finish()?;
                state
            }
        };
        Ok(state)
    }
}

/// Decode sparse-selection scratch (sessions that reach the sparse
/// frontier), shared by every MLA block except `select_status`, which holds
/// one sticky slot per MLA block. Selection reuses the other buffers block
/// after block on the serial encoder: each block's attention consumes them
/// before the next block's selection overwrites them.
struct SparseScratch {
    index_query: MetalTensor,
    index_query_f16: MetalTensor,
    index_weights: MetalTensor,
    scores: MetalTensor,
    pool_ids: MetalTensor,
    pool_counts: MetalTensor,
    visible_pools: MetalTensor,
    visible_rows: MetalTensor,
    row_ids: MetalTensor,
    row_counts: MetalTensor,
    select_status: MetalTensor,
    attention_partials: MetalTensor,
    attention_partial_stats: MetalTensor,
}

impl SparseScratch {
    fn new(ctx: &MetalContext, c: &Glm5NextConfig, capacity: u64) -> Result<Self> {
        let specs = memory::sparse_decode_specs(c, capacity);
        let mut b = SpecBuffers::allocate(ctx, "sparse_decode", &specs)?;
        let s = Self {
            index_query: b.take("index_query")?,
            index_query_f16: b.take("index_query_f16")?,
            index_weights: b.take("index_weights")?,
            scores: b.take("scores")?,
            pool_ids: b.take("pool_ids")?,
            pool_counts: b.take("pool_counts")?,
            visible_pools: b.take("visible_pools")?,
            visible_rows: b.take("visible_rows")?,
            row_ids: b.take("row_ids")?,
            row_counts: b.take("row_counts")?,
            select_status: b.take("select_status")?,
            attention_partials: b.take("attention_partials")?,
            attention_partial_stats: b.take("attention_partial_stats")?,
        };
        b.finish()?;
        Ok(s)
    }

    /// Status slot of the `mla_index`-th MLA block.
    fn status(&self, mla_index: usize) -> MetalTensor {
        self.select_status.view_subrange(mla_index as u64, vec![1])
    }
}

/// Writes 4-byte elements into a shared-storage tensor of `dtype` holding
/// exactly `values.len()` elements. Callers are synchronized session code:
/// no command touching it may be in flight.
fn write_elements<T: bytemuck::Pod>(
    tensor: &MetalTensor,
    dtype: GgmlType,
    values: &[T],
) -> Result<()> {
    let end = tensor.offset.checked_add(4 * values.len() as u64);
    if tensor.dtype != dtype
        || std::mem::size_of::<T>() != 4
        || tensor.n_elements() != values.len() as u64
        || !tensor.offset.is_multiple_of(4)
        || end.is_none_or(|end| end > tensor.buffer.length() as u64)
    {
        return invalid(format!("cannot write {dtype:?} values into this tensor"));
    }
    // SAFETY: shared storage; dtype, element size, length, alignment and
    // range checked; the caller guarantees no in-flight command uses it.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<T>(),
            values.len(),
        );
    }
    Ok(())
}

fn write_i32(tensor: &MetalTensor, values: &[i32]) -> Result<()> {
    write_elements(tensor, GgmlType::I32, values)
}

fn write_f32(tensor: &MetalTensor, values: &[f32]) -> Result<()> {
    write_elements(tensor, GgmlType::F32, values)
}

impl RouteRecord {
    /// One record per MoE block (`None` for dense FFN blocks); `rows` is
    /// `None` for decode and the chunk size for packed prefill.
    fn for_blocks(
        ctx: &MetalContext,
        c: &Glm5NextConfig,
        rows: Option<u64>,
    ) -> Result<Vec<Option<Self>>> {
        let specs = memory::route_specs(c, rows);
        c.blocks
            .iter()
            .map(|block| {
                if block.ffn != FfnKind::Moe {
                    return Ok(None);
                }
                let mut b = SpecBuffers::allocate(ctx, "route", &specs)?;
                let record = Self {
                    ids: b.take("ids")?,
                    weights: b.take("weights")?,
                    status: b.take("status")?,
                };
                b.finish()?;
                Ok(Some(record))
            })
            .collect()
    }
}

/// Host copy of a session-owned shared-storage tensor of `dtype` with 4-byte
/// elements. Callers are synchronized session code: no command writing the
/// tensor may be in flight.
fn read_elements<T: bytemuck::Pod>(tensor: &MetalTensor, dtype: GgmlType) -> Result<Vec<T>> {
    let n = tensor.n_elements() as usize;
    let bytes = n
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Glm5NextMetalError::Invalid("read size overflow".into()))?;
    let end = tensor.offset.checked_add(bytes as u64);
    if tensor.dtype != dtype
        || std::mem::size_of::<T>() != 4
        || !tensor.offset.is_multiple_of(4)
        || end.is_none_or(|end| end > tensor.buffer.length() as u64)
    {
        return invalid(format!(
            "cannot read {:?} {:?} at offset {} as {dtype:?}",
            tensor.dtype, tensor.shape, tensor.offset
        ));
    }
    // SAFETY: shared-storage buffer; dtype, alignment and range checked above;
    // the caller guarantees no in-flight writer.
    let slice = unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize),
            bytes,
        )
    };
    Ok(bytemuck::cast_slice(slice).to_vec())
}

fn read_f32(tensor: &MetalTensor) -> Result<Vec<f32>> {
    read_elements(tensor, GgmlType::F32)
}

fn read_i32(tensor: &MetalTensor) -> Result<Vec<i32>> {
    read_elements(tensor, GgmlType::I32)
}

/// F16 session tensor promoted to F32 (diagnostics and tests).
#[cfg(test)]
fn read_f16(tensor: &MetalTensor) -> Result<Vec<f32>> {
    let n = tensor.n_elements() as usize;
    let end = tensor.offset.checked_add(2 * n as u64);
    if tensor.dtype != GgmlType::F16
        || !tensor.offset.is_multiple_of(2)
        || end.is_none_or(|end| end > tensor.buffer.length() as u64)
    {
        return invalid("cannot read tensor as F16");
    }
    // SAFETY: as in `read_elements`.
    let slice = unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize),
            2 * n,
        )
    };
    Ok(bytemuck::cast_slice::<u8, u16>(slice)
        .iter()
        .map(|&bits| half::f16::from_bits(bits).to_f32())
        .collect())
}

/// Intermediate a [`Glm5NextSession::forward_observed`] caller may inspect.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Glm5NextProbe {
    /// Residual streams [4096, 4] after the attention sub-block (`hc_attn_post`).
    AttentionResidual,
    /// Residual streams [4096, 4] after the block (`l_out`).
    BlockResidual,
}

/// Decode stages timed by [`Glm5NextSession::forward_stage_profiled`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Glm5NextStage {
    Embed,
    AttentionPre,
    Kda,
    MlaProjection,
    MlaIndexer,
    DenseAttention,
    SparseQuery,
    SparseScores,
    SparseSelect,
    SparseAttention,
    MlaOutput,
    AttentionPost,
    FfnPre,
    DenseFfn,
    Router,
    RoutedExperts,
    SharedExpert,
    FfnPost,
    Head,
}

impl Glm5NextStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Embed => "embed",
            Self::AttentionPre => "attention_pre",
            Self::Kda => "kda",
            Self::MlaProjection => "mla_projection",
            Self::MlaIndexer => "mla_indexer",
            Self::DenseAttention => "dense_attention",
            Self::SparseQuery => "sparse_query",
            Self::SparseScores => "sparse_scores",
            Self::SparseSelect => "sparse_select",
            Self::SparseAttention => "sparse_attention",
            Self::MlaOutput => "mla_output",
            Self::AttentionPost => "attention_post",
            Self::FfnPre => "ffn_pre",
            Self::DenseFfn => "dense_ffn",
            Self::Router => "router",
            Self::RoutedExperts => "routed_experts",
            Self::SharedExpert => "shared_expert",
            Self::FfnPost => "ffn_post",
            Self::Head => "head",
        }
    }
}

/// One timed encoder of a profiled decode step.
#[derive(Clone, Copy, Debug)]
pub struct Glm5NextStageSpan {
    pub stage: Glm5NextStage,
    /// Executed block, or `None` for the embedding and the head.
    pub block: Option<u32>,
    /// Stage-boundary timestamps scaled to the command's GPU time.
    pub gpu_ms: f64,
}

/// Attribution of one decode step: the step is encoded as one command with
/// one sampled encoder per stage (diagnostic; encoder boundaries add their
/// own cost, reported as the gap between the span sum and the command).
#[derive(Clone, Debug)]
pub struct Glm5NextStageReport {
    pub spans: Vec<Glm5NextStageSpan>,
    pub command_gpu_ms: f64,
    pub span_sum_ms: f64,
}

/// Samples per profiled step: two per stage span, with headroom.
const STAGE_SAMPLE_CAPACITY: usize = 2048;

struct StageRecorder {
    samples: crate::metal::MetalTimestampSampleBuffer,
    spans: Vec<(Glm5NextStage, Option<u32>)>,
    command_gpu_ms: f64,
}

pub struct Glm5NextSession<'w> {
    weights: &'w Glm5NextWeights,
    capacity: usize,
    position: usize,
    poisoned: bool,
    layers: Vec<LayerState>,
    routes: Vec<Option<RouteRecord>>,
    s: Scratch,
    /// Present when `capacity` reaches the sparse frontier.
    sparse: Option<SparseScratch>,
    packed: Option<packed::PackedScratch>,
    ledger: Glm5NextMemoryLedger,
    /// Net device-counter change across buffer construction (diagnostic).
    observed_allocation_delta: u64,
    /// Set only by [`Self::forward_stage_profiled`] for one step.
    stage_recorder: std::cell::RefCell<Option<StageRecorder>>,
    /// Test hook: the next packed chunk sees zero visible pools for this
    /// chunk row, so its sparse selection fails in every MLA block.
    #[cfg(test)]
    corrupt_sparse_row: Option<usize>,
}

impl<'w> Glm5NextSession<'w> {
    /// Allocates all session state for `capacity` positions after admitting
    /// the ledger's session terms (decode-only). From the sparse frontier
    /// (visible length 2052) on, attention runs over the indexer's selection.
    pub fn new(ctx: &MetalContext, weights: &'w Glm5NextWeights, capacity: usize) -> Result<Self> {
        Self::with_prefill_rows(ctx, weights, capacity, 0)
    }

    /// Like [`Self::new`], also allocating packed-prefill scratch for chunks of
    /// up to `prefill_rows` tokens (0: serial prefill only; at most
    /// `capacity`). Every buffer is built from the ledger's spec lists, which
    /// the ledger prices with the device before admission.
    pub fn with_prefill_rows(
        ctx: &MetalContext,
        weights: &'w Glm5NextWeights,
        capacity: usize,
        prefill_rows: usize,
    ) -> Result<Self> {
        Self::with_prefill_rows_and_cpu_reserve(ctx, weights, capacity, prefill_rows, 0)
    }

    /// Like [`Self::with_prefill_rows`], admitting the session buffers
    /// together with caller-owned future CPU storage (serve transport and
    /// control allowances) in one decision, so neither spends headroom the
    /// other already counted.
    pub fn with_prefill_rows_and_cpu_reserve(
        ctx: &MetalContext,
        weights: &'w Glm5NextWeights,
        capacity: usize,
        prefill_rows: usize,
        cpu_reserve_bytes: u64,
    ) -> Result<Self> {
        let c = &weights.config;
        if prefill_rows > 0 && !weights.packed_prefill_admitted {
            return invalid("packed prefill is not admitted for these weights' dtypes");
        }
        if capacity == 0 {
            return invalid("capacity must be positive");
        }
        let sparse_capacity = capacity >= c.sparse_frontier() as usize;
        let ledger = Glm5NextMemoryLedger::new(
            c,
            weights.retained_bytes,
            capacity as u64,
            prefill_rows as u64,
            &device_price(ctx),
        )?;
        let session_bytes = ledger.peak_bytes() - weights.retained_bytes;
        // Admission and the allocations it prices form one transaction.
        let _allocation = ctx.begin_allocation_transaction();
        let admission = evaluate_metal_memory_admission_with_cpu_bytes(
            session_bytes,
            cpu_reserve_bytes,
            0,
            ctx.memory_signals(),
            true,
        );
        if let Some(denied) = admission.refusal() {
            return Err(Glm5NextMetalError::MemoryAdmission {
                denied,
                budget_bytes: admission_budget(&admission),
                advice: CapacityAdvice::NotEvaluated,
            });
        }
        // Every buffer below comes from the spec lists the ledger priced.
        let before = ctx.current_allocated_size();
        let layers = c
            .blocks
            .iter()
            .map(|block| LayerState::new(ctx, c, block.mixer, capacity as u64))
            .collect::<Result<Vec<_>>>()?;
        let routes = RouteRecord::for_blocks(ctx, c, None)?;
        let s = Scratch::new(ctx, c)?;
        let sparse = sparse_capacity
            .then(|| SparseScratch::new(ctx, c, capacity as u64))
            .transpose()?;
        let packed = (prefill_rows > 0)
            .then(|| packed::PackedScratch::new(ctx, c, prefill_rows, capacity))
            .transpose()?;
        // Diagnostic only: the device counter also moves with unrelated
        // allocations and frees elsewhere in the process.
        let observed_allocation_delta = ctx.current_allocated_size().saturating_sub(before);
        Ok(Self {
            weights,
            capacity,
            position: 0,
            poisoned: false,
            layers,
            routes,
            s,
            sparse,
            packed,
            ledger,
            observed_allocation_delta,
            stage_recorder: std::cell::RefCell::new(None),
            #[cfg(test)]
            corrupt_sparse_row: None,
        })
    }

    /// Selects the packed-prefill arithmetic before the session commits work
    /// (no effect without packed scratch). Re-selecting the current lineage
    /// is always accepted; changing it once tokens are committed is refused
    /// without mutation, so the state (and any snapshot of it) never carries
    /// one lineage's arithmetic under the other's label.
    pub fn set_packed_lineage(&mut self, lineage: PackedLineage) -> Result<()> {
        let Some(packed) = self.packed.as_mut() else {
            return Ok(());
        };
        if packed.lineage == lineage {
            return Ok(());
        }
        if self.position > 0 {
            return Err(Glm5NextMetalError::Invalid(format!(
                "packed lineage cannot change from {:?} to {lineage:?} after {} committed tokens; start a fresh session",
                packed.lineage, self.position
            )));
        }
        packed.lineage = lineage;
        Ok(())
    }

    /// The packed-prefill lineage, or `None` for a serial-only session.
    pub fn packed_lineage(&self) -> Option<PackedLineage> {
        self.packed.as_ref().map(|packed| packed.lineage)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether a failed token left the session unusable.
    pub fn is_poisoned(&self) -> bool {
        self.poisoned
    }

    pub fn ledger(&self) -> &Glm5NextMemoryLedger {
        &self.ledger
    }

    /// Net change of the device's allocated-size counter while the session
    /// buffers were built. A diagnostic, not owned bytes: unrelated
    /// allocations or frees in the process move the same counter. In an
    /// isolated process it is bounded by
    /// [`Glm5NextMemoryLedger::session_buffer_bytes`].
    pub fn observed_allocation_delta(&self) -> u64 {
        self.observed_allocation_delta
    }

    /// Decode one token at the next position; returns full-vocabulary logits.
    pub fn forward(&mut self, ctx: &MetalContext, token: u32) -> Result<Vec<f32>> {
        self.forward_observed(ctx, token, &[], &mut |_, _, _| {})
    }

    /// Like [`Self::forward`] (same kernels, same state), encoding the step
    /// as one command with a timestamp-sampled encoder per stage. The
    /// encoder boundaries cost GPU time of their own, so the spans attribute
    /// a slightly slower step; compare shares, not the sum, with `forward`.
    pub fn forward_stage_profiled(
        &mut self,
        ctx: &MetalContext,
        token: u32,
    ) -> Result<(Vec<f32>, Glm5NextStageReport)> {
        *self.stage_recorder.borrow_mut() = Some(StageRecorder {
            samples: ctx.timestamp_sample_buffer(STAGE_SAMPLE_CAPACITY)?,
            spans: Vec::new(),
            command_gpu_ms: 0.0,
        });
        let result = self.forward(ctx, token);
        let recorder = self.stage_recorder.borrow_mut().take();
        let logits = result?;
        let recorder =
            recorder.ok_or_else(|| Glm5NextMetalError::Invalid("stage recorder lost".into()))?;
        let n = recorder.spans.len();
        if n == 0 {
            return invalid("profiled step recorded no stages");
        }
        let ticks = ctx.resolve_timestamp_samples(&recorder.samples, 2 * n)?;
        let total_ticks = ticks[2 * n - 1].saturating_sub(ticks[0]);
        if total_ticks == 0 || recorder.command_gpu_ms <= 0.0 {
            return invalid("profiled step has no measurable GPU time");
        }
        let scale = recorder.command_gpu_ms / total_ticks as f64;
        let spans: Vec<Glm5NextStageSpan> = recorder
            .spans
            .iter()
            .enumerate()
            .map(|(i, &(stage, block))| Glm5NextStageSpan {
                stage,
                block,
                gpu_ms: ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * scale,
            })
            .collect();
        let span_sum_ms = spans.iter().map(|span| span.gpu_ms).sum();
        Ok((
            logits,
            Glm5NextStageReport {
                spans,
                command_gpu_ms: recorder.command_gpu_ms,
                span_sum_ms,
            },
        ))
    }

    /// Starts a sampled encoder for `stage` when a profiled step is
    /// recording; otherwise does nothing.
    fn stage(
        &self,
        enc: &mut KernelEncoder,
        stage: Glm5NextStage,
        block: Option<usize>,
    ) -> Result<()> {
        let mut recorder = self.stage_recorder.borrow_mut();
        let Some(recorder) = recorder.as_mut() else {
            return Ok(());
        };
        let index = recorder.spans.len();
        if 2 * index + 1 >= recorder.samples.sample_count() {
            return invalid("stage profile sample capacity exceeded");
        }
        let command = enc.parent.clone();
        enc.finish();
        *enc = KernelEncoder::try_begin_sampled(
            &command,
            &recorder.samples,
            2 * index,
            2 * index + 1,
            false,
        )?;
        recorder.spans.push((stage, block.map(|b| b as u32)));
        Ok(())
    }

    /// Advance one token without the output head (prompt positions whose
    /// logits are unused; skips the head's 0.5 GB weight read).
    pub fn advance(&mut self, ctx: &MetalContext, token: u32) -> Result<()> {
        self.step(ctx, token, false, &[], &mut |_, _, _| {})?;
        Ok(())
    }

    /// Serial prefill: advance every token but the last, then return the last
    /// token's logits.
    pub fn prefill(&mut self, ctx: &MetalContext, tokens: &[u32]) -> Result<Vec<f32>> {
        self.validate_request(tokens)?;
        let Some((&last, head)) = tokens.split_last() else {
            return invalid("prefill requires at least one token");
        };
        for &token in head {
            self.advance(ctx, token)?;
        }
        self.forward(ctx, last)
    }

    /// Refuses a whole multi-token request before any token executes, so a
    /// bad token or capacity overrun never leaves a partially advanced prefix.
    fn validate_request(&self, tokens: &[u32]) -> Result<()> {
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
        }
        if tokens.is_empty() {
            return invalid("prefill requires at least one token");
        }
        let vocab = self.weights.config.vocab_size;
        if let Some((index, &token)) = tokens.iter().enumerate().find(|(_, t)| **t >= vocab) {
            return invalid(format!(
                "token[{index}] = {token} outside vocabulary {vocab}"
            ));
        }
        let end = self.position.checked_add(tokens.len());
        if end.is_none_or(|end| end > self.capacity) {
            return invalid(format!(
                "{} tokens from position {} exceed capacity {}",
                tokens.len(),
                self.position,
                self.capacity
            ));
        }
        Ok(())
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
        Ok(self
            .step(ctx, token, true, probes, observer)?
            .expect("logits requested"))
    }

    fn step(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        logits: bool,
        probes: &[Glm5NextProbe],
        observer: &mut dyn FnMut(Glm5NextProbe, usize, &[f32]),
    ) -> Result<Option<Vec<f32>>> {
        let mut no_captures = |_: Glm5NextSiteCapture, _: &[f32]| {};
        let mut hooks = interventions::Hooks {
            module: &[],
            captures: &[],
            observer: &mut no_captures,
        };
        self.step_hooked(ctx, token, logits, probes, observer, &mut hooks)
    }

    /// [`Self::step`] with module-site interventions and captures.
    fn step_hooked(
        &mut self,
        ctx: &MetalContext,
        token: u32,
        logits: bool,
        probes: &[Glm5NextProbe],
        observer: &mut dyn FnMut(Glm5NextProbe, usize, &[f32]),
        hooks: &mut interventions::Hooks<'_, '_>,
    ) -> Result<Option<Vec<f32>>> {
        if self.poisoned {
            return Err(Glm5NextMetalError::Poisoned);
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
        // Poisoned until the token completes and validates: recurrent state
        // mutates on the GPU, so any failure or unwind past this point leaves
        // the session unusable rather than silently at the old position.
        self.poisoned = true;
        let visible = self.position + 1;
        let sparse = visible >= c.sparse_frontier() as usize;
        if sparse {
            let s = self.sparse.as_ref().ok_or_else(|| {
                Glm5NextMetalError::Invalid("sparse position without sparse scratch".into())
            })?;
            write_i32(
                &s.visible_pools,
                &[(visible / c.indexer_pool as usize) as i32],
            )?;
            write_i32(&s.visible_rows, &[visible as i32])?;
            // Unwritten slots fail the check below.
            let mla = c.block_count(MixerKind::Mla);
            write_i32(&s.select_status, &vec![-1; mla])?;
        }
        self.encode_token(ctx, token, logits, probes, observer, hooks)?;
        for (layer, route) in self.routes.iter().enumerate() {
            if let Some(route) = route {
                let status = read_i32(&route.status)?[0];
                if status != ROUTE_STATUS_READY {
                    return Err(Glm5NextMetalError::KernelValidation {
                        stage: "route",
                        block: layer,
                        row: None,
                        status,
                    });
                }
            }
        }
        if sparse {
            let statuses = read_i32(&self.sparse.as_ref().expect("checked").select_status)?;
            if let Some((index, status)) = statuses
                .iter()
                .enumerate()
                .find(|(_, s)| **s != crate::metal::SELECT_STATUS_OK)
            {
                return Err(Glm5NextMetalError::KernelValidation {
                    stage: "sparse selection",
                    block: mla_block(c, index),
                    row: None,
                    status: *status,
                });
            }
        }
        let logits = logits.then(|| read_f32(&self.s.logits)).transpose()?;
        self.position += 1;
        self.poisoned = false;
        Ok(logits)
    }

    fn encode_token(
        &self,
        ctx: &MetalContext,
        token: u32,
        logits: bool,
        probes: &[Glm5NextProbe],
        observer: &mut dyn FnMut(Glm5NextProbe, usize, &[f32]),
        hooks: &mut interventions::Hooks<'_, '_>,
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
        self.stage(&mut enc, Glm5NextStage::Embed, None)?;
        encode_get_rows_f32(ctx, &enc, &w.embedding, &s.token, &s.embedding, 1, h)?;
        self.apply_site_hooks(
            ctx,
            &mut command,
            &mut enc,
            hooks,
            Glm5NextSite::Embedding,
            0,
            &s.embedding,
        )?;
        encode_mhc4_repeat(ctx, &enc, h, &s.embedding, &s.residual[0])?;
        let mut mla_index = 0;
        for (index, block) in w.blocks.iter().enumerate() {
            let (a, b) = (&s.residual[0], &s.residual[1]);
            // Attention sub-block: a -> b.
            self.stage(&mut enc, Glm5NextStage::AttentionPre, Some(index))?;
            self.encode_hc_pre(ctx, &enc, a, &block.attention_hc, &block.attention_norm)?;
            match (&block.mixer, &self.layers[index]) {
                (MixerTensors::Kda(kda), LayerState::Kda { conv, state }) => {
                    self.stage(&mut enc, Glm5NextStage::Kda, Some(index))?;
                    self.encode_kda(ctx, &enc, kda, conv, state)?
                }
                (
                    MixerTensors::Mla(mla),
                    LayerState::Mla {
                        latent,
                        pending,
                        pooled,
                    },
                ) => {
                    self.encode_mla(
                        ctx, &mut enc, mla, latent, pending, pooled, mla_index, index,
                    )?;
                    mla_index += 1;
                }
                _ => return invalid(format!("block {index} state does not match its mixer")),
            }
            self.apply_site_hooks(
                ctx,
                &mut command,
                &mut enc,
                hooks,
                Glm5NextSite::MixerOutput,
                index,
                &s.block_out,
            )?;
            self.stage(&mut enc, Glm5NextStage::AttentionPost, Some(index))?;
            encode_mhc4_post(ctx, &enc, h, &s.block_out, a, &s.post, &s.comb, b)?;
            if observed && probes.contains(&Glm5NextProbe::AttentionResidual) {
                enc.end();
                command.commit();
                wait_completed(&command)?;
                observer(Glm5NextProbe::AttentionResidual, index, &read_f32(b)?);
                command = ctx
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
                enc = KernelEncoder::begin(&command);
            }
            // FFN sub-block: b -> a.
            self.stage(&mut enc, Glm5NextStage::FfnPre, Some(index))?;
            self.encode_hc_pre(ctx, &enc, b, &block.ffn_hc, &block.ffn_norm)?;
            match &block.ffn {
                FfnTensors::Dense(dense) => {
                    self.stage(&mut enc, Glm5NextStage::DenseFfn, Some(index))?;
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
                    self.stage(&mut enc, Glm5NextStage::Router, Some(index))?;
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
                    self.stage(&mut enc, Glm5NextStage::RoutedExperts, Some(index))?;
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
                    self.apply_site_hooks(
                        ctx,
                        &mut command,
                        &mut enc,
                        hooks,
                        Glm5NextSite::RoutedExpertsOutput,
                        index,
                        &s.routed,
                    )?;
                    self.stage(&mut enc, Glm5NextStage::SharedExpert, Some(index))?;
                    let sf = c.shared_expert_ffn_size as usize;
                    shared_gate_up_swiglu(
                        ctx,
                        &enc,
                        &moe.shared,
                        &s.normed,
                        &s.shared_swiglu,
                        h,
                        sf,
                        c.swiglu_clamp,
                    )?;
                    matvec(
                        ctx,
                        &enc,
                        &moe.shared.down,
                        &s.shared_swiglu.view_subrange(0, vec![sf as u64]),
                        &s.shared,
                        sf,
                        h,
                    )?;
                    self.apply_site_hooks(
                        ctx,
                        &mut command,
                        &mut enc,
                        hooks,
                        Glm5NextSite::SharedExpertOutput,
                        index,
                        &s.shared,
                    )?;
                    encode_add_f32(ctx, &enc, &s.routed, &s.shared, &s.block_out)?;
                }
            }
            self.apply_site_hooks(
                ctx,
                &mut command,
                &mut enc,
                hooks,
                Glm5NextSite::FfnOutput,
                index,
                &s.block_out,
            )?;
            self.stage(&mut enc, Glm5NextStage::FfnPost, Some(index))?;
            encode_mhc4_post(ctx, &enc, h, &s.block_out, b, &s.post, &s.comb, a)?;
            if observed && probes.contains(&Glm5NextProbe::BlockResidual) {
                enc.end();
                command.commit();
                wait_completed(&command)?;
                observer(Glm5NextProbe::BlockResidual, index, &read_f32(a)?);
                command = ctx
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
                enc = KernelEncoder::begin(&command);
            }
        }
        if logits {
            self.stage(&mut enc, Glm5NextStage::Head, None)?;
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
        }
        enc.end();
        command.commit();
        wait_completed(&command)?;
        if let Some(recorder) = self.stage_recorder.borrow_mut().as_mut() {
            recorder.command_gpu_ms = (command.GPUEndTime() - command.GPUStartTime()) * 1e3;
        }
        Ok(())
    }

    /// The fused mHC pre ([`crate::metal::encode_mhc4_pre_q8_0`]) of one
    /// sub-block: mixes, controls and collapse into `collapsed`, then the
    /// block norm into `normed`; `post`/`comb` stay set for the matching
    /// post. GLM admits only Q8_0 mixes (coverage), and packed rows run the
    /// same kernels, so packed prefill keeps decode's mHC lineage.
    fn encode_hc_pre(
        &self,
        ctx: &MetalContext,
        enc: &KernelEncoder,
        residual: &MetalTensor,
        hc: &crate::glm5_next::HyperConnectionTensors<MetalTensor>,
        norm_weight: &MetalTensor,
    ) -> Result<()> {
        let c = &self.weights.config;
        let s = &self.s;
        let one = |t: &MetalTensor, shape: Vec<u64>| t.view_subrange(0, shape);
        let (h, mixes) = (c.hidden_size as u64, c.hc_mix_count() as u64);
        crate::metal::encode_mhc4_pre_q8_0(
            ctx,
            enc,
            c.hidden_size as usize,
            1,
            hc_pre_eps(c),
            &crate::metal::Mhc4PreInputs {
                residual: &one(residual, vec![h, 4, 1]),
                mix: &hc.mix,
                scale: &hc.scale,
                base: &hc.base,
                norm_weight,
            },
            &crate::metal::Mhc4PrePartials {
                dots: &s.hc_partial_dots,
                sumsq: &s.hc_partial_sumsq,
            },
            &crate::metal::Mhc4PreOutputs {
                mixes: &one(&s.mixes, vec![mixes, 1]),
                pre: &one(&s.pre, vec![4, 1]),
                post: &one(&s.post, vec![4, 1]),
                comb: &one(&s.comb, vec![4, 4, 1]),
                collapsed: &one(&s.collapsed, vec![h, 1]),
                normed: &one(&s.normed, vec![h, 1]),
            },
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
        // q, k and v share the input: one dispatch when all three are Q6_K
        // (each row bitwise equal to its own mat-vec, so packed rows keep
        // the lineage with separate dispatches).
        let qkv = [&kda.query, &kda.key, &kda.value];
        if qkv.iter().all(|w| w.dtype == GgmlType::Q6_K) && width.is_multiple_of(2) {
            crate::metal::encode_mat_vec_q6_k_x3_f32(
                ctx,
                enc,
                qkv,
                &s.normed,
                [&s.q, &s.k, &s.v],
                h,
                width,
            )?;
        } else {
            matvec(ctx, enc, &kda.query, &s.normed, &s.q, h, width)?;
            matvec(ctx, enc, &kda.key, &s.normed, &s.k, h, width)?;
            matvec(ctx, enc, &kda.value, &s.normed, &s.v, h, width)?;
        }
        matvec(ctx, enc, &kda.decay_a, &s.normed, &s.rank_a, h, rank)?;
        low_rank_expand(
            ctx,
            enc,
            &kda.decay_b,
            &s.rank_a,
            &s.raw_gate,
            rank,
            width,
            1,
        )?;
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
        low_rank_expand(
            ctx,
            enc,
            &kda.gate_b,
            &s.rank_b,
            &s.output_gate,
            rank,
            width,
            1,
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

    #[allow(clippy::too_many_arguments)]
    fn encode_mla(
        &self,
        ctx: &MetalContext,
        enc: &mut KernelEncoder,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pending: &MetalTensor,
        pooled: &MetalTensor,
        mla_index: usize,
        block: usize,
    ) -> Result<()> {
        self.stage(enc, Glm5NextStage::MlaProjection, Some(block))?;
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
        // Indexer cache maintenance from token zero. The pool this token
        // completes is published before attention: a query at visible length
        // L scores the first L / 4 pools, including its own.
        let index_dim = c.indexer_head_dim as usize;
        self.stage(enc, Glm5NextStage::MlaIndexer, Some(block))?;
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
        let scale = 1.0 / (head_dim as f32).sqrt();
        let visible = self.position + 1;
        if visible >= c.sparse_frontier() as usize {
            self.encode_sparse_attention(
                ctx, enc, mla, latent, pooled, mla_index, visible, scale, block,
            )?;
        } else {
            self.stage(enc, Glm5NextStage::DenseAttention, Some(block))?;
            encode_latent_attention(
                ctx,
                enc,
                &s.query_latent,
                latent,
                &s.no_sink,
                &s.output_latent,
                self.position,
                1,
                scale,
            )?;
        }
        self.stage(enc, Glm5NextStage::MlaOutput, Some(block))?;
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
        Ok(())
    }

    /// Sparse attention for one token at visible length `visible` (at or past
    /// the frontier): indexer query (F16-rounded) and head weights scaled by
    /// 1 / sqrt(heads * dim) (exact: 1/64), lightning scores over the visible
    /// pools, exact top-512 into this block's status slot, expansion to 2048
    /// rows plus the `visible % 4` tail, and online attention over exactly
    /// those latent rows (no sink).
    #[allow(clippy::too_many_arguments)]
    fn encode_sparse_attention(
        &self,
        ctx: &MetalContext,
        enc: &mut KernelEncoder,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pooled: &MetalTensor,
        mla_index: usize,
        visible: usize,
        scale: f32,
        block: usize,
    ) -> Result<()> {
        self.stage(enc, Glm5NextStage::SparseQuery, Some(block))?;
        let c = &self.weights.config;
        let s = &self.s;
        let sp = self.sparse.as_ref().ok_or_else(|| {
            Glm5NextMetalError::Invalid("sparse position without sparse scratch".into())
        })?;
        let (h, q_rank) = (c.hidden_size as usize, c.q_lora_rank as usize);
        let (ih, id) = (c.indexer_head_count as usize, c.indexer_head_dim as usize);
        let (heads, kv) = (c.head_count as usize, c.kv_lora_rank as usize);
        let query_width = ih * id;
        matvec(
            ctx,
            enc,
            &mla.indexer.query,
            &s.query_r,
            &sp.index_query,
            q_rank,
            query_width,
        )?;
        encode_scatter_offset_f32_to_f16(
            ctx,
            enc,
            &sp.index_query,
            &sp.index_query_f16,
            0,
            query_width,
        )?;
        let weights = sp.index_weights.view_subrange(0, vec![ih as u64]);
        matvec(
            ctx,
            enc,
            &mla.indexer.head_weights,
            &s.normed,
            &weights,
            h,
            ih,
        )?;
        crate::metal::encode_scale_f32_in_place(
            ctx,
            enc,
            &weights,
            1.0 / (query_width as f32).sqrt(),
        )?;
        let pool_capacity = pooled.shape[1] as usize;
        let visible_pools = visible / c.indexer_pool as usize;
        let top_pools = c.selected_pool_count() as usize;
        let row_slots = c.selection_width() as usize;
        self.stage(enc, Glm5NextStage::SparseScores, Some(block))?;
        crate::metal::encode_lightning_scores_f16_matrix(
            ctx,
            enc,
            &crate::metal::LightningScores {
                queries: &sp.index_query_f16,
                head_weights: &sp.index_weights,
                keys: pooled,
                visible_counts: &sp.visible_pools,
                scores: &sp.scores,
            },
            ih,
            id,
            pool_capacity,
            visible_pools,
            1,
        )?;
        self.stage(enc, Glm5NextStage::SparseSelect, Some(block))?;
        crate::metal::encode_select_top_k_ids(
            ctx,
            enc,
            &crate::metal::TopKSelection {
                scores: &sp.scores,
                visible_counts: &sp.visible_pools,
                ids: &sp.pool_ids,
                counts: &sp.pool_counts,
                status: &sp.status(mla_index),
            },
            pool_capacity,
            visible_pools,
            top_pools,
            1,
        )?;
        crate::metal::encode_indexer_expand_selection(
            ctx,
            enc,
            &crate::metal::IndexerSelection {
                pool_ids: &sp.pool_ids,
                pool_counts: &sp.pool_counts,
                visible_rows: &sp.visible_rows,
                row_ids: &sp.row_ids,
                row_counts: &sp.row_counts,
            },
            top_pools,
            row_slots,
            1,
        )?;
        let flat = |t: &MetalTensor| t.view_subrange(0, vec![(kv * heads) as u64, 1]);
        self.stage(enc, Glm5NextStage::SparseAttention, Some(block))?;
        crate::metal::encode_online_selected_attention_split_f16(
            ctx,
            enc,
            &crate::metal::SelectedAttention {
                queries: &flat(&s.query_latent),
                raw_cache: latent,
                raw_cache_before_chunk: latent,
                compressed_cache: latent,
                selected_ids: &sp.row_ids,
                selected_counts: &sp.row_counts,
                visible_counts: &sp.visible_rows,
                sinks: &s.no_sink,
                output: &flat(&s.output_latent),
            },
            crate::metal::SelectedAttentionShape {
                head_count: heads,
                query_count: 1,
                query_token_offset: 0,
                token_count: 1,
                chunk_start_position: self.position,
                window: 0,
                raw_cache_is_chunk: false,
                selected_slots: row_slots,
                compressed_capacity: latent.shape[1] as usize,
                scale,
                direct: true,
            },
            &crate::metal::SelectedAttentionPartials {
                values: &sp.attention_partials,
                stats: &sp.attention_partial_stats,
            },
        )?;
        Ok(())
    }
}

/// Epsilons of the fused mHC pre: the flattened residual's RMS and the block
/// norm both use the model's RMS epsilon.
fn hc_pre_eps(c: &crate::glm5_next::Glm5NextConfig) -> crate::metal::Mhc4PreEps {
    crate::metal::Mhc4PreEps {
        hc_rms: c.rms_epsilon,
        hc: c.hc_epsilon,
        norm: c.rms_epsilon,
    }
}

/// The shared expert's clamped SwiGLU into `scratch[..sf]` (`scratch` is
/// `[3 * sf]`: output, gate, up). Q6_K gate and up run DS4's fused kernel,
/// which equals two Q6_K mat-vecs plus [`encode_clamped_swiglu`] bitwise
/// (`metal::mat_vec` test `fused_q6_k_shared_swiglu_equals_separate_path_bitwise`),
/// so packed rows keep the separate path and the same lineage. Other dtypes
/// run the separate path here too.
#[allow(clippy::too_many_arguments)]
fn shared_gate_up_swiglu(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    shared: &crate::glm5_next::DenseFfnTensors<MetalTensor>,
    x: &MetalTensor,
    scratch: &MetalTensor,
    h: usize,
    sf: usize,
    clamp: f32,
) -> Result<()> {
    if shared.gate.dtype == GgmlType::Q6_K && shared.up.dtype == GgmlType::Q6_K {
        crate::metal::encode_ds4_shared_swiglu_q6_k_f32(
            ctx,
            enc,
            &shared.gate,
            &shared.up,
            x,
            scratch,
            h,
            sf,
            clamp,
        )?;
        return Ok(());
    }
    let part = |i: u64| scratch.view_subrange(i * sf as u64, vec![sf as u64]);
    let (out, gate, up) = (part(0), part(1), part(2));
    matvec(ctx, enc, &shared.gate, x, &gate, h, sf)?;
    matvec(ctx, enc, &shared.up, x, &up, h, sf)?;
    encode_clamped_swiglu(ctx, enc, &gate, &up, &out, clamp)?;
    Ok(())
}

/// KDA's low-rank expansions (`ssm_f_b`, `ssm_g_b`: rank 128 -> width):
/// the short-K Q8_0 kernel over `rows` independent inputs when the weight is
/// Q8_0 with a short row, else the generic mat-vec per row. Decode and
/// Exact packed rows both come here, so they keep one lineage.
#[allow(clippy::too_many_arguments)]
fn low_rank_expand(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weight: &MetalTensor,
    x: &MetalTensor,
    y: &MetalTensor,
    n_in: usize,
    n_out: usize,
    rows: usize,
) -> Result<()> {
    if weight.dtype == GgmlType::Q8_0 && matches!(n_in, 32 | 64 | 128 | 256) {
        crate::metal::encode_mat_vec_q8_0_short_k_f32(ctx, enc, weight, x, y, n_in, n_out, rows)?;
        return Ok(());
    }
    for row in 0..rows {
        let xr = x.view_subrange((row * n_in) as u64, vec![n_in as u64]);
        let yr = y.view_subrange((row * n_out) as u64, vec![n_out as u64]);
        matvec(ctx, enc, weight, &xr, &yr, n_in, n_out)?;
    }
    Ok(())
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

mod interventions;
mod lens;
mod packed;
pub use interventions::{
    Glm5NextCapturePoint, Glm5NextModuleIntervention, Glm5NextSite, Glm5NextSiteCapture,
};
mod snapshot;
pub use lens::Glm5NextCapture;
pub use packed::PackedLineage;
pub use snapshot::{Glm5NextSnapshot, SNAPSHOT_POLICY_VERSION, snapshot_bytes};

#[cfg(test)]
mod tests;
