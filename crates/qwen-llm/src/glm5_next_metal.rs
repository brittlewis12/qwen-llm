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
    /// The caller's checkpoint stopped a multi-step operation at a boundary
    /// where the session (if any) is consistent and unpoisoned.
    #[error("GLM-5.3 cancelled: {0}")]
    Cancelled(String),
    /// The requested session does not fit device memory.
    #[error(
        "GLM-5.3 session needs {required_bytes} bytes but {budget_bytes} are available ({reason}); {}",
        match fitting_capacity {
            Some(n) => format!("the largest capacity that fits is {n} positions"),
            None => "no capacity fits".to_string(),
        }
    )]
    MemoryAdmission {
        required_bytes: u64,
        budget_bytes: u64,
        reason: String,
        fitting_capacity: Option<u64>,
    },
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
    if admission.admitted {
        return Ok(Glm5NextPreflight { ledger });
    }
    let headroom = admission.working_set_headroom_bytes.unwrap_or(0);
    let budget = match admission.signals.process_limit_remaining_bytes {
        Some(process) if process > 0 => headroom.min(process),
        _ => headroom,
    };
    let fitting_capacity = Glm5NextMemoryLedger::max_capacity(
        &model.config,
        retained,
        prefill_rows as u64,
        budget,
        &price,
    )
    .ok()
    .flatten();
    Err(Glm5NextMetalError::MemoryAdmission {
        required_bytes: ledger.peak_bytes(),
        budget_bytes: budget,
        reason: format!("{:?}", admission.reason),
        fitting_capacity,
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
    _backings: Vec<MetalGgufBacking>,
}

impl Glm5NextWeights {
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
            ones: b.take("ones")?,
            quarter: b.take("quarter")?,
            no_sink: b.take("no_sink")?,
            normalized: b.take("normalized")?,
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
            shared_gate: b.take("shared_gate")?,
            shared_up: b.take("shared_up")?,
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
        };
        b.finish()?;
        Ok(s)
    }

    /// Status slot of the `mla_index`-th MLA block.
    fn status(&self, mla_index: usize) -> MetalTensor {
        self.select_status.view_subrange(mla_index as u64, vec![1])
    }
}

/// Writes I32 values into a shared-storage I32 tensor. Callers are
/// synchronized session code: no command touching it may be in flight.
fn write_i32(tensor: &MetalTensor, values: &[i32]) -> Result<()> {
    let end = tensor.offset.checked_add(4 * values.len() as u64);
    if tensor.dtype != GgmlType::I32
        || tensor.n_elements() != values.len() as u64
        || !tensor.offset.is_multiple_of(4)
        || end.is_none_or(|end| end > tensor.buffer.length() as u64)
    {
        return invalid("cannot write I32 values into this tensor");
    }
    // SAFETY: shared storage; dtype, length, alignment and range checked;
    // the caller guarantees no in-flight command uses the tensor.
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr(),
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<i32>(),
            values.len(),
        );
    }
    Ok(())
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
        let admission =
            evaluate_metal_memory_admission(session_bytes, 0, ctx.memory_signals(), true);
        if !admission.admitted {
            return invalid(format!(
                "session does not fit: reason={:?} required={session_bytes}",
                admission.reason
            ));
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
            #[cfg(test)]
            corrupt_sparse_row: None,
        })
    }

    /// Selects the packed-prefill arithmetic (no effect without packed scratch).
    pub fn set_packed_lineage(&mut self, lineage: PackedLineage) {
        if let Some(packed) = self.packed.as_mut() {
            packed.lineage = lineage;
        }
    }

    pub fn position(&self) -> usize {
        self.position
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
            return invalid("session is poisoned by an earlier failed token");
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
        self.encode_token(ctx, token, logits, probes, observer)?;
        for (layer, route) in self.routes.iter().enumerate() {
            if let Some(route) = route {
                let status = read_i32(&route.status)?[0];
                if status != ROUTE_STATUS_READY {
                    return invalid(format!("block {layer} route failed with status {status}"));
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
                return invalid(format!(
                    "MLA block {index} sparse selection failed with status {status}"
                ));
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
        let mut mla_index = 0;
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
                ) => {
                    self.encode_mla(ctx, &enc, mla, latent, pending, pooled, mla_index)?;
                    mla_index += 1;
                }
                _ => return invalid(format!("block {index} state does not match its mixer")),
            }
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
                observer(Glm5NextProbe::BlockResidual, index, &read_f32(a)?);
                command = ctx
                    .queue
                    .commandBuffer()
                    .ok_or_else(|| Glm5NextMetalError::Invalid("no command buffer".into()))?;
                enc = KernelEncoder::begin(&command);
            }
        }
        if logits {
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
        mla_index: usize,
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
        // Indexer cache maintenance from token zero. The pool this token
        // completes is published before attention: a query at visible length
        // L scores the first L / 4 pools, including its own.
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
        let scale = 1.0 / (head_dim as f32).sqrt();
        let visible = self.position + 1;
        if visible >= c.sparse_frontier() as usize {
            self.encode_sparse_attention(ctx, enc, mla, latent, pooled, mla_index, visible, scale)?;
        } else {
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
        enc: &KernelEncoder,
        mla: &crate::glm5_next::MlaTensors<MetalTensor>,
        latent: &MetalTensor,
        pooled: &MetalTensor,
        mla_index: usize,
        visible: usize,
        scale: f32,
    ) -> Result<()> {
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
        crate::metal::encode_online_selected_attention_f16(
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

mod packed;
pub use packed::PackedLineage;

#[cfg(test)]
mod tests;
