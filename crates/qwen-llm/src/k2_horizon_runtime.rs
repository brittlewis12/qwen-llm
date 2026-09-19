//! Research serial dense K2 forward path, shared by raw run and forward-only lens.
//!
//! Weights remain native and read-only; sessions borrow the exact loaded model
//! and context. One token command is in flight at a time. A whole append commits
//! once; any failure after submission poisons the session rather than exposing
//! partial state. No snapshots, fitting, eviction, or packed-prefill claims.

use crate::gguf::{GgufError, GgufFile};
use crate::k2_horizon::{K2HorizonConfig, K2HorizonError};
use crate::k2_horizon_metal::{
    encode_full_rope, encode_grouped_norm, encode_online_attention, encode_store_kv,
};
use crate::k2_horizon_plan::{K2ShortContextPlan, PlanError, TokenPlan};
use crate::metal::{
    KernelEncoder, MetalContext, MetalError, MetalTensor, encode_add_inplace_f32,
    encode_get_rows_f32, encode_silu_mul_f32, evaluate_metal_memory_admission,
    host_page_size_bytes,
};
use crate::metal_forward::{MfError, encode_mat_vec_dispatch};
use crate::tensor::GgmlType;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandQueue, MTLResource,
    MTLStorageMode,
};
use std::cell::Cell;

mod intervention;
mod lens;
mod residency;
mod state;
mod transport;
use intervention::InterventionArena;
pub use intervention::{K2Intervention, K2InterventionKind};
use lens::CaptureArena;
pub use lens::K2CapturedForward;
pub use residency::K2RuntimePlan;
use residency::ResidentWeights;
use state::Ledger;
pub use transport::{K2LinearF16, K2LinearReadout};

const RESERVE_BYTES: u64 = 256 * 1024 * 1024;
/// Shared application budget, not the kernel ceiling or an artifact whitelist.
/// Numerical evidence is scoped to the pinned final Q8_0 weights with F16 KV.
pub const GUARDED_APPLICATION_FORWARD_CEILING: usize = 256;
type Result<T> = std::result::Result<T, K2RuntimeError>;

#[derive(Debug, thiserror::Error)]
pub enum K2RuntimeError {
    #[error(transparent)]
    Profile(#[from] K2HorizonError),
    #[error(transparent)]
    Plan(#[from] PlanError),
    #[error(transparent)]
    Gguf(#[from] GgufError),
    #[error(transparent)]
    Metal(#[from] MetalError),
    #[error(transparent)]
    Dispatch(#[from] MfError),
    #[error("invalid K2 runtime contract: {0}")]
    Invalid(String),
    #[error("K2 session is poisoned; discard it and create a fresh session")]
    Poisoned,
}

fn invalid(message: impl Into<String>) -> K2RuntimeError {
    K2RuntimeError::Invalid(message.into())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttentionBackend {
    #[cfg(test)]
    Materialized,
    Online,
}

const DEFAULT_ATTENTION_BACKEND: AttentionBackend = AttentionBackend::Online;

/// Borrowing the source prevents descriptor mutation in safe Rust. File stamps
/// are rechecked, but are NOT a cryptographic checkpoint identity or file lock.
pub struct K2LoadedModel<'a> {
    ctx: &'a MetalContext,
    plan: K2RuntimePlan<'a>,
    weights: ResidentWeights,
    attention: AttentionBackend,
    session_active: Cell<bool>,
}

impl<'a> K2LoadedModel<'a> {
    /// Explicit research entry point; no claim of checkpoint numerical parity.
    /// Prices weights AND one session before allocating any weight buffers.
    pub fn load_unqualified(
        ctx: &'a MetalContext,
        source: &'a GgufFile,
        capacity: u32,
    ) -> Result<Self> {
        Self::load_with_attention_unqualified(ctx, source, capacity, DEFAULT_ATTENTION_BACKEND)
    }

    fn load_with_attention_unqualified(
        ctx: &'a MetalContext,
        source: &'a GgufFile,
        capacity: u32,
        attention: AttentionBackend,
    ) -> Result<Self> {
        let plan = K2RuntimePlan::inspect(
            source,
            capacity,
            host_page_size_bytes()?,
            ctx.max_buffer_length().min(8 * 1024 * 1024 * 1024),
        )?;
        let weight_price = price_buffers(ctx, &plan.weight_buffer_bytes()?)?;
        let session_price = price_buffers(ctx, &plan.session.buffer_bytes())?;
        let aggregate = weight_price
            .checked_add(session_price)
            .ok_or_else(|| invalid("aggregate price overflow"))?;
        let _transaction = ctx.begin_allocation_transaction();
        admit(ctx, aggregate)?;
        plan.revalidate_source()?;
        let before = ctx.current_allocated_size();
        let weights = ResidentWeights::realize(ctx, &plan)?;
        plan.revalidate_source()?;
        reconcile(ctx, before, weight_price)?;
        Ok(Self {
            ctx,
            plan,
            weights,
            attention,
            session_active: Cell::new(false),
        })
    }

    pub fn config(&self) -> &K2HorizonConfig {
        self.plan.config()
    }

    /// Each session receives separate KV/scratch and fresh admission. The capacity
    /// is the request extent priced at load, not the checkpoint's declared max.
    pub fn create_session(&self, start_position: u32) -> Result<K2Session<'_, 'a>> {
        if self.session_active.get() {
            return Err(invalid(
                "initial K2 runtime allows one live session per loaded model",
            ));
        }
        self.plan.revalidate_source()?;
        let request =
            K2ShortContextPlan::new(self.config().clone(), start_position, self.plan.capacity())?;
        let price = price_buffers(self.ctx, &self.plan.session.buffer_bytes())?;
        let _transaction = self.ctx.begin_allocation_transaction();
        admit(self.ctx, price)?;
        let before = self.ctx.current_allocated_size();
        let buffers = SessionBuffers::new(self.ctx, &self.plan.session)?;
        reconcile(self.ctx, before, price)?;
        self.plan.revalidate_source()?;
        let permit = SessionPermit::acquire(&self.session_active)?;
        Ok(K2Session {
            model: self,
            request,
            buffers,
            ledger: Ledger::default(),
            _permit: permit,
        })
    }
}

pub struct K2Session<'model, 'ctx> {
    model: &'model K2LoadedModel<'ctx>,
    request: K2ShortContextPlan,
    buffers: SessionBuffers,
    ledger: Ledger,
    _permit: SessionPermit<'model>,
}

struct SessionPermit<'a>(&'a Cell<bool>);

impl<'a> SessionPermit<'a> {
    fn acquire(active: &'a Cell<bool>) -> Result<Self> {
        if active.replace(true) {
            return Err(invalid("a session is already live"));
        }
        Ok(Self(active))
    }
}

impl Drop for SessionPermit<'_> {
    fn drop(&mut self) {
        self.0.set(false);
    }
}

impl K2Session<'_, '_> {
    pub fn committed_len(&self) -> u32 {
        self.ledger.prefix()
    }
    pub fn is_poisoned(&self) -> bool {
        self.ledger.is_poisoned()
    }

    /// Serial prefill/continuation, returning only the final token's logits.
    /// Validates ALL IDs before I32 upload. Successful earlier token commands
    /// remain staged until the entire append completes; errors expose no prefix.
    pub fn append(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        Ok(self.append_impl(tokens, &[], &[])?.logits)
    }

    fn append_impl(
        &mut self,
        tokens: &[u32],
        layers: &[u32],
        interventions: &[K2Intervention<'_>],
    ) -> Result<K2CapturedForward> {
        if let Err(error) = self.model.plan.revalidate_source() {
            self.ledger.poison();
            return Err(error);
        }
        let mut transaction = self.ledger.begin(
            tokens,
            self.model.config().vocab_size,
            self.request.capacity(),
        )?;
        let prefix = transaction.old_prefix();
        let absolute = self.request.append(
            prefix,
            self.request.start_position() + prefix,
            tokens.len() as u32,
        )?;
        let ctx = self.model.ctx;
        let captures = CaptureArena::allocate(ctx, layers)?;
        let interventions = InterventionArena::allocate(ctx, interventions)?;
        self.model.plan.revalidate_source()?;
        for (index, &id) in tokens.iter().enumerate() {
            let token = absolute.token(index as u32)?;
            let last = index + 1 == tokens.len();
            self.buffers.upload_id(id);
            let command = ctx
                .queue
                .commandBuffer()
                .ok_or_else(|| invalid("cannot create command buffer"))?;
            let encoder = KernelEncoder::begin(&command);
            let encoded = encode_token(
                ctx,
                &encoder,
                &self.model.weights,
                self.model.attention,
                &self.request,
                &token,
                &self.buffers,
                last,
                captures.as_ref().filter(|_| last),
                interventions.as_ref().filter(|_| last),
            );
            encoder.end();
            encoded?;
            transaction.submitting()?;
            command.commit();
            command.waitUntilCompleted();
            if command.status() != MTLCommandBufferStatus::Completed {
                return Err(invalid(format!(
                    "token command failed: {:?}",
                    command.error()
                )));
            }
            self.buffers.check_completed(&token, last)?;
            transaction.checked()?;
        }
        self.model.plan.revalidate_source()?;
        let logits = read_f32(&self.buffers.logits).to_vec();
        let residuals = captures
            .as_ref()
            .map(CaptureArena::read)
            .transpose()?
            .unwrap_or_default();
        let result = K2CapturedForward {
            absolute_position: self.request.start_position() + prefix + tokens.len() as u32 - 1,
            post_block_layers: layers.to_vec(),
            residuals,
            logits,
        };
        transaction.commit()?;
        Ok(result)
    }
}

fn encode_token(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &ResidentWeights,
    attention: AttentionBackend,
    request: &K2ShortContextPlan,
    token: &TokenPlan<'_>,
    b: &SessionBuffers,
    logits: bool,
    captures: Option<&CaptureArena>,
    interventions: Option<&InterventionArena>,
) -> Result<()> {
    encode_get_rows_f32(ctx, enc, &weights.embedding, &b.id, &b.residual, 1, 4096)?;
    for (index, layer) in weights.layers.iter().enumerate() {
        encode_grouped_norm(
            ctx,
            enc,
            request,
            &b.residual,
            &layer.attention_norm,
            &b.norm,
        )?;
        encode_mat_vec_dispatch(ctx, enc, &layer.query, &b.norm, &b.query, 4096, 4096)?;
        encode_mat_vec_dispatch(ctx, enc, &layer.key, &b.norm, &b.key, 4096, 1024)?;
        encode_mat_vec_dispatch(ctx, enc, &layer.value, &b.norm, &b.value, 4096, 1024)?;
        encode_full_rope(ctx, enc, token, &b.query, &b.key)?;
        encode_store_kv(ctx, enc, token, index as u32, &b.cache, &b.key, &b.value)?;
        let encode_attention = match attention {
            #[cfg(test)]
            AttentionBackend::Materialized => crate::k2_horizon_metal::encode_short_attention,
            AttentionBackend::Online => encode_online_attention,
        };
        encode_attention(
            ctx,
            enc,
            token,
            index as u32,
            &b.cache,
            &b.query,
            &b.attention,
        )?;
        encode_mat_vec_dispatch(
            ctx,
            enc,
            &layer.attention_output,
            &b.attention,
            &b.projection,
            4096,
            4096,
        )?;
        encode_add_inplace_f32(ctx, enc, &b.residual, &b.projection)?;
        encode_grouped_norm(
            ctx,
            enc,
            request,
            &b.residual,
            &layer.feed_forward_norm,
            &b.norm,
        )?;
        encode_mat_vec_dispatch(ctx, enc, &layer.gate, &b.norm, &b.gate, 4096, 12288)?;
        encode_mat_vec_dispatch(ctx, enc, &layer.up, &b.norm, &b.up, 4096, 12288)?;
        encode_silu_mul_f32(ctx, enc, &b.gate, &b.up, &b.gated)?;
        encode_mat_vec_dispatch(ctx, enc, &layer.down, &b.gated, &b.projection, 12288, 4096)?;
        encode_add_inplace_f32(ctx, enc, &b.residual, &b.projection)?;
        if let Some(interventions) = interventions {
            interventions.encode(ctx, enc, index as u32, &b.residual)?;
        }
        if let Some(captures) = captures {
            captures.encode(ctx, enc, index as u32, &b.residual)?;
        }
    }
    if logits {
        encode_readout(ctx, enc, weights, request, b)?;
    }
    Ok(())
}

fn encode_readout(
    ctx: &MetalContext,
    enc: &KernelEncoder,
    weights: &ResidentWeights,
    request: &K2ShortContextPlan,
    b: &SessionBuffers,
) -> Result<()> {
    encode_grouped_norm(
        ctx,
        enc,
        request,
        &b.residual,
        &weights.output_norm,
        &b.norm,
    )?;
    encode_mat_vec_dispatch(ctx, enc, &weights.output, &b.norm, &b.logits, 4096, 250624)?;
    Ok(())
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct SessionMemoryPlan {
    cache_bytes: u64,
}

impl SessionMemoryPlan {
    fn new(config: &K2HorizonConfig, capacity: u32) -> Result<Self> {
        let request = K2ShortContextPlan::new(config.clone(), 0, capacity)?;
        Ok(Self {
            cache_bytes: request.arena_bytes(),
        })
    }

    fn specs(&self) -> [(GgmlType, Vec<u64>); 13] {
        [
            (GgmlType::I32, vec![1]),
            (GgmlType::F32, vec![4096]),
            (GgmlType::F32, vec![4096]),
            (GgmlType::F32, vec![128, 32]),
            (GgmlType::F32, vec![128, 8]),
            (GgmlType::F32, vec![128, 8]),
            (GgmlType::F32, vec![128, 32]),
            (GgmlType::F32, vec![4096]),
            (GgmlType::F32, vec![12288]),
            (GgmlType::F32, vec![12288]),
            (GgmlType::F32, vec![12288]),
            (GgmlType::F32, vec![250624]),
            (GgmlType::F16, vec![self.cache_bytes / 2]),
        ]
    }

    fn buffer_bytes(&self) -> Vec<u64> {
        self.specs()
            .iter()
            .map(|(dtype, shape)| {
                shape.iter().product::<u64>() * if *dtype == GgmlType::F16 { 2 } else { 4 }
            })
            .collect()
    }
}

struct SessionBuffers {
    id: MetalTensor,
    residual: MetalTensor,
    norm: MetalTensor,
    query: MetalTensor,
    key: MetalTensor,
    value: MetalTensor,
    attention: MetalTensor,
    projection: MetalTensor,
    gate: MetalTensor,
    up: MetalTensor,
    gated: MetalTensor,
    logits: MetalTensor,
    cache: MetalTensor,
}

impl SessionBuffers {
    fn new(ctx: &MetalContext, plan: &SessionMemoryPlan) -> Result<Self> {
        let mut tensors = plan
            .specs()
            .into_iter()
            .map(|(dtype, shape)| -> Result<MetalTensor> {
                let tensor = MetalTensor::zeros_dtype(ctx, shape.clone(), dtype)?;
                if tensor.dtype != dtype || tensor.shape != shape || !tensor.is_writable() {
                    return Err(invalid("session allocation descriptor drift"));
                }
                validate_cpu_layout(
                    tensor.buffer.storageMode() == MTLStorageMode::Shared,
                    tensor.offset,
                    tensor.n_bytes(),
                    tensor.buffer.length() as u64,
                )?;
                Ok(tensor)
            });
        Ok(Self {
            id: tensors.next().unwrap()?,
            residual: tensors.next().unwrap()?,
            norm: tensors.next().unwrap()?,
            query: tensors.next().unwrap()?,
            key: tensors.next().unwrap()?,
            value: tensors.next().unwrap()?,
            attention: tensors.next().unwrap()?,
            projection: tensors.next().unwrap()?,
            gate: tensors.next().unwrap()?,
            up: tensors.next().unwrap()?,
            gated: tensors.next().unwrap()?,
            logits: tensors.next().unwrap()?,
            cache: tensors.next().unwrap()?,
        })
    }

    fn upload_id(&self, token: u32) {
        // All IDs are checked before entry; this private owned I32[1] buffer is
        // reused only after the prior synchronous command has finished.
        unsafe {
            self.id
                .buffer
                .contents()
                .as_ptr()
                .cast::<i32>()
                .write(token as i32);
        }
    }

    fn check_completed(&self, token: &TokenPlan<'_>, logits: bool) -> Result<()> {
        if read_f32(&self.residual)
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(invalid("nonfinite final residual"));
        }
        if logits
            && read_f32(&self.logits)
                .iter()
                .any(|value| !value.is_finite())
        {
            return Err(invalid("nonfinite logits"));
        }
        // The cache arena is privately allocated as shared F16 and every range
        // derives from the same validated plan. Only newly written rows are read.
        let cache = unsafe {
            std::slice::from_raw_parts(
                self.cache.buffer.contents().as_ptr().cast::<half::f16>(),
                self.cache.n_elements() as usize,
            )
        };
        for layer in 0..36 {
            let ranges = token.write_ranges(layer)?;
            for range in [ranges.key, ranges.value] {
                if cache[(range.start / 2) as usize..(range.end / 2) as usize]
                    .iter()
                    .any(|value| !value.is_finite())
                {
                    return Err(invalid(format!("nonfinite stored K/V in layer {layer}")));
                }
            }
        }
        Ok(())
    }
}

fn read_f32(tensor: &MetalTensor) -> &[f32] {
    // Private owned F32 tensors, after command completion. Construction checks
    // shared storage, zero offset, writability, exact shape, and physical extent.
    unsafe {
        std::slice::from_raw_parts(
            tensor
                .buffer
                .contents()
                .as_ptr()
                .cast::<u8>()
                .add(tensor.offset as usize)
                .cast::<f32>(),
            tensor.n_elements() as usize,
        )
    }
}

fn validate_cpu_layout(shared: bool, offset: u64, bytes: u64, buffer_bytes: u64) -> Result<()> {
    if !shared || offset != 0 || bytes == 0 || bytes > buffer_bytes || bytes > isize::MAX as u64 {
        return Err(invalid(
            "session CPU access requires bounded zero-offset shared storage",
        ));
    }
    Ok(())
}

fn price_buffers(ctx: &MetalContext, buffers: &[u64]) -> Result<u64> {
    buffers.iter().try_fold(0u64, |total, &bytes| {
        let price = ctx
            .price_shared_buffer_upper(bytes)
            .map_err(|error| invalid(error.to_string()))?
            .priced_upper_bytes;
        total
            .checked_add(price)
            .ok_or_else(|| invalid("buffer price overflow"))
    })
}

fn admit(ctx: &MetalContext, bytes: u64) -> Result<()> {
    let admission =
        evaluate_metal_memory_admission(bytes, RESERVE_BYTES, ctx.memory_signals(), true);
    if !admission.admitted {
        return Err(invalid(format!(
            "memory admission denied: {}",
            admission.reason.as_str()
        )));
    }
    Ok(())
}

fn reconcile(ctx: &MetalContext, before: u64, price: u64) -> Result<()> {
    let observed = ctx.current_allocated_size().saturating_sub(before);
    if observed > price {
        return Err(invalid(format!(
            "allocation delta {observed} exceeds price {price}"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod oracle_tests;
#[cfg(test)]
mod tests;
